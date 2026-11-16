"""Drive the goalie stepper through a Pico 2 over the Pi 5's hardware UART.

The Pico runs pi5_UART_motor_test/ firmware. Each command is one line holding
an absolute arm angle in RADIANS (0 = center); this class takes degrees, like
``ServoArm``, and converts them. Wiring (3.3 V on both sides, no level shifter):

    Pi 5 GPIO14 TXD (pin 8)  -> Pico GP1 RX (pin 2)
    Pi 5 GPIO15 RXD (pin 10) <- Pico GP0 TX (pin 1)   (replies)
    Pi 5 GND        (pin 6)  -- Pico GND    (pin 3)

Enable the UART once on the Pi 5: add ``dtparam=uart0=on`` to
/boot/firmware/config.txt and reboot; it appears as /dev/ttyAMA0.

The Pico answers each line with OK <rad>, BUSY or ERR, then AT when it reaches
the target and DONE when it is back at 0. The current test firmware holds
5 s at the target, returns to 0 by itself and ignores commands until DONE.
"""

import logging
import math
import threading
import time


class PicoMotor:
    """Angle commands to the Pico. ``dry_run`` only logs, for laptops.

    ``invert`` flips the direction if positive angles turn the wrong way.
    ``on_reply`` is called with each reply line from the Pico (reader thread).
    """

    def __init__(self, port="/dev/ttyAMA0", *, baud=115200, invert=False, limit_deg=72.0,
                 dry_run=False, on_reply=None):
        self.invert = invert
        self.limit_deg = limit_deg
        self.dry_run = dry_run
        self.on_reply = on_reply
        self.angle = None
        self.state = "unknown"  # last reply: OK, BUSY, ERR, AT or DONE
        self.lock = threading.Lock()
        self.serial = None
        self._reader = None
        if not dry_run:
            try:
                import serial
            except ImportError as exc:
                raise RuntimeError("pyserial is missing: sudo apt install python3-serial") from exc
            # timeout lets the reader thread notice close(); writes never block long
            # at 115200 baud (a command is ~10 bytes, ~1 ms).
            self.serial = serial.Serial(port, baud, timeout=0.2, write_timeout=0.1)
            self.serial.reset_input_buffer()
            self._reader = threading.Thread(target=self._read_replies, daemon=True)
            self._reader.start()

    def _read_replies(self):
        while self.serial is not None:
            try:
                line = self.serial.readline()
            except Exception:  # port closed under us
                return
            text = line.decode("ascii", "replace").strip()
            if not text:
                continue
            self.state = text.split()[0]
            logging.debug("Pico <- %s", text)
            if self.on_reply:
                self.on_reply(text)

    def move(self, angle_deg):
        """Send an absolute arm angle in degrees (+ toward +X). Returns it, clamped."""
        angle = max(-self.limit_deg, min(self.limit_deg, float(angle_deg)))
        radians = math.radians(-angle if self.invert else angle)
        with self.lock:
            self.angle = angle
            if self.serial is not None:
                self.serial.write(f"{radians:.4f}\n".encode("ascii"))
            else:
                logging.info("Pico motor (dry run) -> %+.1f deg (%.4f rad)", angle, radians)
        return angle

    def center(self):
        self.move(0.0)

    def wait_for(self, *states, timeout_s=10.0):
        """Wait until the last reply is one of ``states``; True if it arrived."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.state in states:
                return True
            time.sleep(0.005)
        return False

    def relax(self):
        """No-op: the stepper driver holds position; kept for ServoArm parity."""

    def close(self):
        with self.lock:
            port, self.serial = self.serial, None
        if port is not None:
            port.close()
        if self._reader is not None:
            self._reader.join(timeout=1.0)
