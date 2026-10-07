"""Aim a hobby servo arm at the predicted camera-plane crossing.

Geometry: the servo shaft points at the kicker (perpendicular to the goal
plane), so its arm sweeps within the plane the ball crosses. Arm angle 0 is
straight down; positive turns toward +X (right as seen from the camera).
The pivot is given in meters: sideways from the camera midpoint and height
above the ground.

Heights from the provisional calibration are relative to a rolling ball's
center, i.e. one ball radius above the ground. A rolling ball is aimed at
that height exactly, since measured heights depend on the rig's tilt.
"""

import logging
import math
import threading


def arm_angle_deg(crossing_x_m, crossing_height_m, model, *, pivot_x_m=0.0,
                  pivot_height_m=0.3, ball_radius_m=0.11, limit_deg=90.0):
    """Arm angle (0 = straight down, + toward +X) to point at the ball center."""
    target_height = ball_radius_m if model == "rolling" else crossing_height_m + ball_radius_m
    angle = math.degrees(math.atan2(crossing_x_m - pivot_x_m, pivot_height_m - target_height))
    return max(-limit_deg, min(limit_deg, angle))


class ServoArm:
    """One servo on a GPIO pin (gpiozero). ``dry_run`` only logs, for laptops.

    Pulse widths map -90..+90 degrees; SG90-class servos are ~500-2500 us.
    ``invert`` flips the direction if positive angles turn the wrong way.
    """

    def __init__(self, pin=18, *, min_pulse_us=500, max_pulse_us=2500, invert=False,
                 limit_deg=90.0, dry_run=False):
        self.invert = invert
        self.limit_deg = limit_deg
        self.dry_run = dry_run
        self.angle = None
        self.lock = threading.Lock()
        self.servo = None
        if not dry_run:
            try:
                from gpiozero import AngularServo
            except ImportError as exc:
                raise RuntimeError("gpiozero is missing: sudo apt install python3-gpiozero "
                                   "python3-lgpio") from exc
            self.servo = AngularServo(pin, initial_angle=None, min_angle=-90, max_angle=90,
                                      min_pulse_width=min_pulse_us / 1e6,
                                      max_pulse_width=max_pulse_us / 1e6)

    def move(self, angle_deg):
        angle = max(-self.limit_deg, min(self.limit_deg, float(angle_deg)))
        with self.lock:
            self.angle = angle
            if self.servo is not None:
                self.servo.angle = -angle if self.invert else angle
            else:
                logging.info("Servo (dry run) -> %+.1f deg", angle)

    def center(self):
        self.move(0.0)

    def relax(self):
        """Stop sending pulses so an idle servo does not hum or jitter."""
        with self.lock:
            if self.servo is not None:
                self.servo.detach()

    def close(self):
        with self.lock:
            if self.servo is not None:
                self.servo.detach()
                self.servo.close()
                self.servo = None
