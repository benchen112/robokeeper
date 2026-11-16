#!/usr/bin/env python3
"""Type goalie angles in degrees on the Pi 5; the Pico turns the stepper there.

Replaces the Arduino tester (arduino_UART_motor_test/capstone_test.ino): flash
pi5_UART_motor_test/ to the Pico, wire it as in robokeeper/pico_motor.py, then

    python3 tools/pico_motor_test.py          # prompt for angles
    python3 tools/pico_motor_test.py 30 -30   # visit these angles in turn
    python3 tools/pico_motor_test.py --dry-run  # laptop: just print

Each move holds 5 s at the target and returns to 0 by itself (test firmware);
the Pico ignores new angles until it prints DONE. Positive should turn the arm
to your right standing behind the cameras; if not, pass --invert. The firmware
clamps to +/-72 deg.
"""

import argparse
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robokeeper.pico_motor import PicoMotor  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("angles", nargs="*", type=float, help="Angles to visit (degrees)")
    parser.add_argument("--port", default="/dev/ttyAMA0",
                        help="Pi 5 GPIO14/15 UART (needs dtparam=uart0=on)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--limit-deg", type=float, default=72.0)
    parser.add_argument("--dry-run", action="store_true", help="No serial port; only log")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    motor = PicoMotor(args.port, baud=args.baud, invert=args.invert, limit_deg=args.limit_deg,
                      dry_run=args.dry_run, on_reply=lambda text: print(f"  Pico: {text}"))
    try:
        if args.angles:
            for angle in args.angles:
                motor.state = "sent"
                print(f"-> {motor.move(angle):+.1f} deg")
                if args.dry_run:
                    continue
                if not motor.wait_for("OK", "BUSY", "ERR", timeout_s=1.0):
                    print("  no reply: check TX/RX/GND wiring, --port, and that the Pico is flashed")
                    break
                if motor.state != "OK" or not motor.wait_for("DONE", timeout_s=20.0):
                    break
            return
        print("Angle in degrees (+ = right, 0 = center), q to quit.")
        while True:
            text = input("> ").strip()
            if text.lower() in ("q", "quit", "exit"):
                break
            if not text:
                continue
            try:
                angle = float(text)
            except ValueError:
                print("  not a number")
                continue
            print(f"-> {motor.move(angle):+.1f} deg")
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        motor.close()


if __name__ == "__main__":
    main()
