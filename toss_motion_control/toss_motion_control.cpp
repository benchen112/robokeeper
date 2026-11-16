// ADD PWM FREQUENCY TO MOTOR SPEED FUNCTION/LIBRARY
// ADD CALIBRATION FUNCTION TO TURN GOALIE TO EITHER SIDE
// AND SENSE When we hit the ground to set that as our limit
// then return to center and go to other set limit and return to center


#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#include "pico/stdlib.h"
#include "hardware/sync.h"
#include "hardware/uart.h"
#include "hardware/timer.h"
#include "hardware/gpio.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// ==========================================
// 1. HARDWARE CONSTANTS & PARAMETERS
// ==========================================
#define UART_ID uart0
#define BAUD_RATE 115200
#define UART_TX_PIN 0
#define UART_RX_PIN 1

// Step & Direction pins routed through 3.3V-to-5V Level Shifter
#define STEP_PIN 27  // To DSY-1200RS Pin 9 (PULSE+)
#define DIR_PIN  28  // To DSY-1200RS Pin 11 (SIGH+)

const float DT = 0.001f;                // 1 kHz loop (1 ms)
const float GEAR_RATIO = 1.0f;         // CHANGED TO 1 FOR TEST, 10:1 Gearbox reduction, 3.14 radians of goalie equals 31.4 radians of motor
const float MAX_ACCEL = 251.328f;       // Motor acceleration (rad/s^2), THIS WILL BE HOW WE CHANGE DIFFICULTY LEVELS
const float MAX_MOTOR_SPEED = 62.832f;  // Hardware speed limit (rad/s -> 600 RPM)

// Hard Software Limits: +/- 90 degrees (pi/2 rad) at the goalie arm
const float MAX_ARM_ANGLE_RAD = (float)M_PI * 2.0f / 5.0f; // 72 degrees for test! --->CHANGE BACK TO 1.5708 rad (90 deg) LATER

// DSY-1200RS Electronic Gear Denominator (P04.09 default = 10,000 pulses/rev)
const float PULSES_PER_REV = 10000.0f;
const float RAD_TO_PULSES  = PULSES_PER_REV / (2.0f * (float)M_PI); // ~1591.55 pulses/rad
const uint64_t PAUSE_US = 5000000;
// Bounds requested pulse delays to 640 us/tick, including final corrections.
// Normal triangular-profile output is <= 142 pulses/tick even across both limits.
const int32_t MAX_STEP_PULSES_PER_TICK = 160;

// ==========================================
// 2. GLOBAL STATE VARIABLES
// ==========================================
// Absolute position tracking (0 = Center Goal, +25000 = +90 deg, -25000 = -90 deg)
volatile int32_t current_motor_pulses = 0; 

enum class MotionState { Idle, CommandPending, MovingOut, Pausing, MovingBack };
// Shared on the main core between the UART loop and the timer IRQ. Only the
// command handoff below masks interrupts; trajectory work belongs to the timer.
volatile MotionState motion_state = MotionState::Idle;
volatile float pending_arm_angle = 0.0f;

float move_start_motor_rad = 0.0f; // Motor angle at start of current move
float target_motor_rad = 0.0f;     // Absolute target motor angle (rad)
int32_t target_motor_pulses = 0;   // Exact final pulse coordinate
float total_move_dist = 0.0f;      // Magnitude of current move (rad)
float dist_traveled_rad = 0.0f;    // Distance covered in current move (rad)
int32_t move_sign = 1;            // +1 for positive, -1 for negative
float peak_vel = 0.0f;            // Peak velocity for current move (rad/s)
float t_accel = 0.0f;             // Time to accelerate to peak velocity (s)
float time_in_move = 0.0f;        // Elapsed time in current move (s)
uint64_t pause_deadline_us = 0;   // Timer-owned, so no shared 64-bit access

char rx_buffer[16];
int rx_index = 0;
bool discard_rx_line = false;

// ==========================================
// 3. HARDWARE STEP OUTPUT
// ==========================================

// Sends a specified number of step pulses to the motor driver
// Waits 2 microseconds between each step pulse so the driver doesn't
// intepret it as noise
inline void send_step_pulses(int32_t count) {
    for (int32_t i = 0; i < count; i++) {
        gpio_put(STEP_PIN, 1);
        busy_wait_us_32(2);
        gpio_put(STEP_PIN, 0);
        busy_wait_us_32(2);
    }
}

// ==========================================
// 4. MOTION CONTROL LOGIC (Absolute Target)
// ==========================================
// Timer-only. False means the target already equals the tracked pulse position.
bool start_new_move(float requested_arm_rad) {
    // 1. Clamp requested arm angle strictly within [-90 deg, +90 deg]
    if (requested_arm_rad > MAX_ARM_ANGLE_RAD) {
        requested_arm_rad = MAX_ARM_ANGLE_RAD;
    } else if (requested_arm_rad < -MAX_ARM_ANGLE_RAD) {
        requested_arm_rad = -MAX_ARM_ANGLE_RAD;
    }

    // 2. Convert to absolute motor radians, 3.14 radians of arm motion equals 31.4 radians of motor motion
    target_motor_rad = requested_arm_rad * GEAR_RATIO;
    target_motor_pulses = (int32_t)lroundf(target_motor_rad * RAD_TO_PULSES);
    move_start_motor_rad = (float)current_motor_pulses / RAD_TO_PULSES;

    float delta_rad = target_motor_rad - move_start_motor_rad;
    total_move_dist = fabsf(delta_rad);

    // Compare quantized positions so zero and sub-pulse commands complete too.
    if (target_motor_pulses == current_motor_pulses) {
        return false;
    }

    // 3. Set Direction Pin (P04.21 = 0: Positive Logic)
    move_sign = (delta_rad >= 0.0f) ? 1 : -1; // If delta_rad is positive, move_sign is 1; otherwise, it's -1
    gpio_put(DIR_PIN, (move_sign > 0) ? 1 : 0); // If delta_rad is positive, set direction pin high; otherwise, set it low
    busy_wait_us_32(5); // Setup time for direction optocoupler (SIGH+), so it doesn't misinterpret the direction change as noise

    // 4. Calculate triangular peak velocity: v = sqrt(theta * a)
    peak_vel = sqrtf(total_move_dist * MAX_ACCEL);

    // Any move from center (0) up to +/- 90 deg is a pure triangle
    // time to accelerate to peak velocity (s) = peak velocity / max acceleration
    t_accel = peak_vel / MAX_ACCEL;

    dist_traveled_rad = 0.0f;
    time_in_move = 0.0f;
    return true;
}

// ==========================================
// 5. INTERRUPT-DRIVEN TRAJECTORY GENERATOR (1 kHz)
// ==========================================
bool control_loop_callback(struct repeating_timer *t) {
    switch (motion_state) {
        case MotionState::Idle:
            return true;
        case MotionState::CommandPending:
            if (start_new_move(pending_arm_angle)) {
                motion_state = MotionState::MovingOut;
            } else {
                pause_deadline_us = time_us_64() + PAUSE_US;
                motion_state = MotionState::Pausing;
            }
            return true;
        case MotionState::Pausing:
            if (time_us_64() >= pause_deadline_us) {
                motion_state = start_new_move(0.0f)
                    ? MotionState::MovingBack : MotionState::Idle;
            }
            return true;
        case MotionState::MovingOut:
        case MotionState::MovingBack:
            break;
    }

    float target_velocity = 0.0f;
    // Total time for the move is 2 * t_accel since it's a triangular profile
    float total_move_time = (2.0f * t_accel);
    bool profile_complete = time_in_move >= total_move_time;

    // Calculate target velocity based on elapsed time in the move
    if (time_in_move < t_accel) {
        // Accelerating
        target_velocity = MAX_ACCEL * time_in_move;
    } 
    else if (time_in_move < total_move_time) {
        // Decelerating
        float t_decel = time_in_move - t_accel;
        target_velocity = peak_vel - (MAX_ACCEL * t_decel);
    } 
    else {
        // Move complete -> Snap directly to exact absolute target
        dist_traveled_rad = total_move_dist;
    }

    // Update distance traveled based on target velocity and time step
    if (!profile_complete) {
        dist_traveled_rad += target_velocity * DT;
        if (dist_traveled_rad > total_move_dist) {
            dist_traveled_rad = total_move_dist;
        }
        time_in_move += DT;
    }

    // Calculate desired absolute motor angle and convert to absolute pulse coordinate
    float desired_motor_rad = move_start_motor_rad + ((float)move_sign * dist_traveled_rad);
    int32_t desired_total_pulses = profile_complete ? target_motor_pulses
        : (int32_t)lroundf(desired_motor_rad * RAD_TO_PULSES);
    // Floating-point reconstruction must never round a step past the endpoint.
    if ((desired_total_pulses - target_motor_pulses) * move_sign > 0) {
        desired_total_pulses = target_motor_pulses;
    }

    // Fire the exact number of steps needed to reach this millisecond's coordinate
    int32_t steps_to_fire = (desired_total_pulses - current_motor_pulses) * move_sign;
    if (steps_to_fire > MAX_STEP_PULSES_PER_TICK) {
        steps_to_fire = MAX_STEP_PULSES_PER_TICK;
    }

    if (steps_to_fire > 0) {
        send_step_pulses(steps_to_fire);
        current_motor_pulses += (steps_to_fire * move_sign);
    }

    // Finish only AFTER the last physical pulses. A capped correction continues
    // next tick; switching state here makes each completed move transition once.
    if (profile_complete && current_motor_pulses == target_motor_pulses) {
        if (motion_state == MotionState::MovingOut) {
            pause_deadline_us = time_us_64() + PAUSE_US;
            motion_state = MotionState::Pausing;
        } else {
            motion_state = MotionState::Idle;
        }
    }

    return true;
}

// ==========================================
// 6. UART COMMUNICATION (Non-Blocking)
// ==========================================
void check_for_new_angle() {
    while (uart_is_readable(UART_ID)) {
        char c = uart_getc(UART_ID);

        // Discard (never queue) bytes observed while busy, including any partial
        // line. Keep discarding through its delimiter even if we become idle.
        if (motion_state != MotionState::Idle) {
            rx_index = 0;
            discard_rx_line = true;
        }

        if (c == '\n' || c == '\r') {
            if (!discard_rx_line && rx_index > 0) {
                rx_buffer[rx_index] = '\0';

                // Pi sends desired absolute arm angle in radians (-1.5708 to +1.5708, 0 = center)
                char *end;
                float requested_arm_angle = strtof(rx_buffer, &end);
                bool has_number = end != rx_buffer;
                while (isspace((unsigned char)*end)) {
                    ++end;
                }
                if (has_number && *end == '\0' && isfinite(requested_arm_angle)) {
                    // Short atomic handoff: the ISR never sees a partial command.
                    // Both run on the same core; the callback owns every move.
                    uint32_t irq_state = save_and_disable_interrupts();
                    if (motion_state == MotionState::Idle) {
                        pending_arm_angle = requested_arm_angle;
                        motion_state = MotionState::CommandPending;
                    }
                    restore_interrupts(irq_state);
                }
            }
            rx_index = 0;
            discard_rx_line = false;
        } 
        else if (!discard_rx_line) {
            if (rx_index < (int)sizeof(rx_buffer) - 1) {
                rx_buffer[rx_index++] = c;
            } else {
                rx_index = 0;
                discard_rx_line = true;
            }
        }
    }
}

void setup_hardware() {
    uart_init(UART_ID, BAUD_RATE);
    gpio_set_function(UART_TX_PIN, GPIO_FUNC_UART);
    gpio_set_function(UART_RX_PIN, GPIO_FUNC_UART);

    gpio_init(STEP_PIN);
    gpio_set_dir(STEP_PIN, GPIO_OUT);
    gpio_put(STEP_PIN, 0);

    gpio_init(DIR_PIN);
    gpio_set_dir(DIR_PIN, GPIO_OUT);
    gpio_put(DIR_PIN, 0);
}

// ==========================================
// 7. MAIN EXECUTION
// ==========================================
int main() {
    stdio_init_all();
    setup_hardware();

    struct repeating_timer timer;
    // Negative means 1 ms start-to-start; positive adds a delay after completion.
    add_repeating_timer_us(-1000, control_loop_callback, NULL, &timer);

    while (true) {
        check_for_new_angle();
    }

    return 0;
}
