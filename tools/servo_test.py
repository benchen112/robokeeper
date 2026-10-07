#!/usr/bin/env python3
"""Check the servo's wiring, direction and travel before using it live.

Wiring (SG90): brown -> GND (pin 6), red -> 5 V (pin 2 or 4),
orange -> GPIO18 (pin 12).

    python3 tools/servo_test.py            # center, -45, +45, sweep, center
    python3 tools/servo_test.py 30 -60 0   # go to these angles in turn

Angles use live_keeper.py's convention: 0 = arm straight down, positive =
toward the camera's right, i.e. your right when standing behind the cameras
looking at the kicker. If +45 turns the arm to your left, pass --invert here
and --servo-invert to live_keeper.py. If the ends of travel buzz or stall,
narrow --min-us/--max-us (and use the same values there).
"""

import argparse
import logging
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robokeeper.servo import ServoArm  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("angles", nargs="*", type=float, help="Angles to visit (degrees)")
    parser.add_argument("--pin", type=int, default=18, help="BCM GPIO number (18 = pin 12)")
    parser.add_argument("--min-us", type=int, default=500)
    parser.add_argument("--max-us", type=int, default=2500)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--hold-s", type=float, default=1.5, help="Pause at each angle")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arm = ServoArm(args.pin, min_pulse_us=args.min_us, max_pulse_us=args.max_us,
                   invert=args.invert)
    steps = [(a, f"{a:+.0f} deg") for a in args.angles] or [
        (0, "0 deg: arm should hang straight down"),
        (-45, "-45 deg: arm should swing to your LEFT (standing behind the cameras)"),
        (45, "+45 deg: arm should swing to your RIGHT"),
        *((a, None) for a in range(-90, 91, 15)),
        (0, "back to 0 deg"),
    ]
    try:
        for angle, label in steps:
            if label:
                print(label)
            arm.move(angle)
            time.sleep(args.hold_s if label else 0.25)
    except KeyboardInterrupt:
        pass
    finally:
        arm.close()


if __name__ == "__main__":
    main()
