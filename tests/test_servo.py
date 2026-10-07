import unittest

from robokeeper.servo import ServoArm, arm_angle_deg


class ArmAngleTests(unittest.TestCase):
    def test_rolling_ball_below_the_pivot(self):
        # Shaft 0.3 m up, ball center 0.11 m up: a ball crossing right under it hangs straight down.
        self.assertAlmostEqual(arm_angle_deg(0.0, -0.2, "rolling"), 0.0)
        self.assertAlmostEqual(arm_angle_deg(0.19, 0.5, "rolling"), 45.0)  # height ignored
        self.assertAlmostEqual(arm_angle_deg(-0.19, 0.0, "rolling"), -45.0)

    def test_pivot_offset_and_clamp(self):
        self.assertAlmostEqual(arm_angle_deg(0.5, 0.0, "rolling", pivot_x_m=0.5), 0.0)
        # An airborne ball above the shaft is out of reach: clamp to the side.
        self.assertEqual(arm_angle_deg(0.3, 0.6, "airborne"), 90.0)
        self.assertEqual(arm_angle_deg(2.0, 0.0, "rolling", limit_deg=60), 60.0)

    def test_dry_run_servo_tracks_requested_angle(self):
        arm = ServoArm(dry_run=True, limit_deg=80)
        arm.move(120)
        self.assertEqual(arm.angle, 80)
        arm.center()
        self.assertEqual(arm.angle, 0)
        arm.close()


if __name__ == "__main__":
    unittest.main()
