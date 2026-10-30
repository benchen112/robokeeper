"""Run the Pi 5 fan-header cooler at full speed while tracking.

The kernel normally drives the fan from CPU temperature (off below ~50 C), so
it would only spin up after the Pi is already warm. ``FanBoost`` hands the CPU
thermal zone to the ``user_space`` governor, which stops the kernel from
overriding the speed set here, and sets the fan to its maximum. ``close()``
restores the previous governor, which resumes temperature control.

If the process is killed before ``close()``, the fan stays at full speed until
reboot: the safe direction. sysfs writes need root, so they fall back to
``sudo -n tee`` (Raspberry Pi OS gives its default user passwordless sudo).
Failures are logged, never raised: tracking runs with or without the fan.
"""

import logging
from pathlib import Path
import subprocess

THERMAL = Path("/sys/class/thermal")


def _read(path):
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _write(path, value):
    try:
        path.write_text(f"{value}\n")
    except PermissionError:
        subprocess.run(["sudo", "-n", "tee", str(path)], input=f"{value}\n", text=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)


def cpu_temp_c(root=THERMAL):
    value = _read(root / "thermal_zone0" / "temp")
    return round(int(value) / 1000, 1) if value and value.lstrip("-").isdigit() else None


class FanBoost:
    def __init__(self, root=THERMAL):
        self.zone = root / "thermal_zone0"
        self.fan = next((d for d in sorted(root.glob("cooling_device*"))
                         if _read(d / "type") == "pwm-fan"), None)
        self._policy = None
        self.active = False

    def start(self):
        if self.fan is None:
            logging.warning("No fan-header cooler found; tracking without forced cooling")
            return False
        try:
            if "user_space" in (_read(self.zone / "available_policies") or "").split():
                self._policy = _read(self.zone / "policy")
                if self._policy != "user_space":
                    _write(self.zone / "policy", "user_space")
            else:
                logging.warning("Kernel thermal control cannot be paused; it may slow the fan again")
            _write(self.fan / "cur_state", _read(self.fan / "max_state"))
        except (OSError, subprocess.CalledProcessError) as exc:
            logging.warning("Could not take over the fan (%s); run with sudo or allow "
                            "passwordless sudo. Tracking continues without forced cooling", exc)
            self.close()
            return False
        self.active = True
        logging.info("Cooler at full speed (CPU %s C)", cpu_temp_c(self.zone.parent))
        return True

    def close(self):
        policy, self._policy = self._policy, None
        if policy and policy != "user_space":
            try:
                _write(self.zone / "policy", policy)
                logging.info("Cooler back under temperature control")
            except (OSError, subprocess.CalledProcessError) as exc:
                logging.warning("Could not restore fan control (%s); it stays at full speed "
                                "until reboot", exc)
        self.active = False
