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
#define BAUD_RATE 9600 // Matches picoSerial.begin(9600) on Arduino
#define UART_TX_PIN 0
#define UART_RX_PIN 1

// Match the working test_motor wiring through the 3.3V-to-5V level shifter.
#define STEP_PIN 26  // Pico physical pin 31 -> DSY-1200RS pin 9 (PULSE+)
#define DIR_PIN  27  // Pico physical pin 32 -> DSY-1200RS pin 11 (SIGH+)
#define LED_PIN  25  // Onboard LED toggles whenever a valid UART angle arrives

const float DT = 0.001f;                // 1 kHz loop (1 ms)
const float GEAR_RATIO = 1.0f;          // 1.0 for test (change to 10.0f with gearbox)
const float MAX_ACCEL = 251.328f;       // Motor acceleration (rad/s^2)
const float MAX_MOTOR_SPEED = 62.832f;  // Hardware speed limit (rad/s -> 600 RPM)

// Hard Software Limits: 72 degrees for test
const float MAX_ARM_ANGLE_RAD = (float)M_PI * 2.0f / 5.0f;

// DSY-1200RS Electronic Gear Scaling (10,000 pulses/rev)
const float PULSES_PER_REV = 10000.0f;
const float RAD_TO_PULSES  = PULSES_PER_REV / (2.0f * (float)M_PI); // ~1591.55 pulses/rad
const uint64_t PAUSE_US = 5000000;
const int32_t MAX_STEP_PULSES_PER_TICK = 80; // Safe bound for 5 us high + 5 us low

// ==========================================
// 2. GLOBAL STATE VARIABLES
// ==========================================
volatile int32_t current_motor_pulses = 0; 

enum class MotionState { Idle, CommandPending, MovingOut, Pausing, MovingBack };
volatile MotionState motion_state = MotionState::Idle;
volatile float pending_arm_angle = 0.0f;

float move_start_motor_rad = 0.0f;
float target_motor_rad = 0.0f;
int32_t target_motor_pulses = 0;
float total_move_dist = 0.0f;
float dist_traveled_rad = 0.0f;
int32_t move_sign = 1;
float peak_vel = 0.0f;
float t_accel = 0.0f;
float time_in_move = 0.0f;
uint64_t pause_deadline_us = 0;

char rx_buffer[32];
int rx_index = 0;
bool discard_rx_line = false;
bool led_state = true;

// ==========================================
// 3. HARDWARE STEP OUTPUT
// ==========================================
inline void send_step_pulses(int32_t count) {
    for (int32_t i = 0; i < count; i++) {
        gpio_put(STEP_PIN, 1);
        busy_wait_us_32(5); // 5 us ensures clean 5V edges through level shifter
        gpio_put(STEP_PIN, 0);
        busy_wait_us_32(5);
    }
}

// ==========================================
// 4. MOTION CONTROL LOGIC (Absolute Target)
// ==========================================
bool start_new_move(float requested_arm_rad) {
    if (requested_arm_rad > MAX_ARM_ANGLE_RAD) {
        requested_arm_rad = MAX_ARM_ANGLE_RAD;
    } else if (requested_arm_rad < -MAX_ARM_ANGLE_RAD) {
        requested_arm_rad = -MAX_ARM_ANGLE_RAD;
    }

    target_motor_rad = requested_arm_rad * GEAR_RATIO;
    target_motor_pulses = (int32_t)lroundf(target_motor_rad * RAD_TO_PULSES);
    move_start_motor_rad = (float)current_motor_pulses / RAD_TO_PULSES;

    float delta_rad = target_motor_rad - move_start_motor_rad;
    total_move_dist = fabsf(delta_rad);

    if (target_motor_pulses == current_motor_pulses) {
        return false;
    }

    move_sign = (delta_rad >= 0.0f) ? 1 : -1;
    gpio_put(DIR_PIN, (move_sign > 0) ? 1 : 0);
    busy_wait_us_32(10); // Setup time for direction optocoupler (SIGH+)

    peak_vel = sqrtf(total_move_dist * MAX_ACCEL);
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
    float total_move_time = (2.0f * t_accel);
    bool profile_complete = time_in_move >= total_move_time;

    if (time_in_move < t_accel) {
        target_velocity = MAX_ACCEL * time_in_move;
    } 
    else if (time_in_move < total_move_time) {
        float t_decel = time_in_move - t_accel;
        target_velocity = peak_vel - (MAX_ACCEL * t_decel);
    } 
    else {
        dist_traveled_rad = total_move_dist;
    }

    if (!profile_complete) {
        dist_traveled_rad += target_velocity * DT;
        if (dist_traveled_rad > total_move_dist) {
            dist_traveled_rad = total_move_dist;
        }
        time_in_move += DT;
    }

    float desired_motor_rad = move_start_motor_rad + ((float)move_sign * dist_traveled_rad);
    int32_t desired_total_pulses = profile_complete ? target_motor_pulses
        : (int32_t)lroundf(desired_motor_rad * RAD_TO_PULSES);

    if ((desired_total_pulses - target_motor_pulses) * move_sign > 0) {
        desired_total_pulses = target_motor_pulses;
    }

    int32_t steps_to_fire = (desired_total_pulses - current_motor_pulses) * move_sign;
    if (steps_to_fire > MAX_STEP_PULSES_PER_TICK) {
        steps_to_fire = MAX_STEP_PULSES_PER_TICK;
    }

    if (steps_to_fire > 0) {
        send_step_pulses(steps_to_fire);
        current_motor_pulses += (steps_to_fire * move_sign);
    }

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

        if (motion_state != MotionState::Idle) {
            rx_index = 0;
            discard_rx_line = true;
        }

        if (c == '\n' || c == '\r') {
            if (!discard_rx_line && rx_index > 0) {
                rx_buffer[rx_index] = '\0';

                char *end;
                float requested_arm_angle = strtof(rx_buffer, &end);
                bool has_number = end != rx_buffer;
                while (isspace((unsigned char)*end)) {
                    ++end;
                }
                if (has_number && *end == '\0' && isfinite(requested_arm_angle)) {
                    uint32_t irq_state = save_and_disable_interrupts();
                    if (motion_state == MotionState::Idle) {
                        pending_arm_angle = requested_arm_angle;
                        motion_state = MotionState::CommandPending;
                        led_state = !led_state;
                        gpio_put(LED_PIN, led_state); // Blinks when angle is accepted
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

    gpio_init(LED_PIN);
    gpio_set_dir(LED_PIN, GPIO_OUT);
    gpio_put(LED_PIN, 1); // Turns solid green on boot
}

int main() {
    stdio_init_all();
    setup_hardware();

    struct repeating_timer timer;
    add_repeating_timer_us(-1000, control_loop_callback, NULL, &timer);

    while (true) {
        check_for_new_angle();
    }

    return 0;
}
