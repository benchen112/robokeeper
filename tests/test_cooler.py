import tempfile
import unittest
from pathlib import Path

from robokeeper.cooler import FanBoost, cpu_temp_c


class FanBoostTests(unittest.TestCase):
    def thermal(self, root, policies="step_wise user_space"):
        zone = root / "thermal_zone0"
        zone.mkdir()
        (zone / "temp").write_text("61250\n")
        (zone / "policy").write_text("step_wise\n")
        (zone / "available_policies").write_text(policies + "\n")
        other = root / "cooling_device0"
        other.mkdir()
        (other / "type").write_text("cpufreq-cpu0\n")
        fan = root / "cooling_device1"
        fan.mkdir()
        (fan / "type").write_text("pwm-fan\n")
        (fan / "max_state").write_text("4\n")
        (fan / "cur_state").write_text("0\n")
        return zone, fan

    def test_full_speed_while_running_then_temperature_control(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            zone, fan = self.thermal(root)
            boost = FanBoost(root)
            self.assertTrue(boost.start())
            self.assertEqual((fan / "cur_state").read_text().strip(), "4")
            self.assertEqual((zone / "policy").read_text().strip(), "user_space")
            boost.close()
            boost.close()  # idempotent (atexit and finally)
            self.assertEqual((zone / "policy").read_text().strip(), "step_wise")
            self.assertEqual(cpu_temp_c(root), 61.2)

    def test_no_fan_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(FanBoost(Path(tmp)).start())
            self.assertIsNone(cpu_temp_c(Path(tmp)))


if __name__ == "__main__":
    unittest.main()
