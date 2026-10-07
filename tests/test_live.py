import unittest

import numpy as np

from robokeeper.live import KickSession
from robokeeper.stereo3d import StereoCalibration
from robokeeper.vision import TrackResult

CALIBRATION = StereoCalibration(focal_px=1300, cx=640, cy=330, baseline_m=0.30,
                                vertical_offset_px=-33)


def eyes(t, point, observed=True):
    """Confirmed left/right track results for a world point, or an absent pair."""
    if not observed:
        absent = TrackResult(t, "absent", False, None, None, None, None, 0., ())
        return absent, absent
    left, right = CALIBRATION.project(point)
    radius = CALIBRATION.focal_px * 0.11 / point[2]
    return tuple(TrackResult(t, "confirmed", True, center, center, (0., 0.), radius, .9, ())
                 for center in (left, right))


class KickSessionTests(unittest.TestCase):
    def test_rolling_kick_finishes_at_predicted_crossing(self):
        session = KickSession(CALIBRATION)
        start, velocity = np.array([-0.4, -0.25, 7.0]), np.array([0.3, 0.0, -5.0])
        t = 0.0
        # Idle frames before the kick, then the ball in view until z = 2 m.
        for _ in range(20):
            session.update(t, *eyes(t, None, observed=False))
            t += 1 / 60
        self.assertEqual(session.state, "armed")
        kick = t
        while (point := start + velocity * (t - kick))[2] > 2.0:
            self.assertFalse(session.update(t, *eyes(t, point)))
            t += 1 / 30  # the Pi processes fewer pairs than the camera delivers
        self.assertEqual(session.state, "tracking")
        # Out of view: no measurements until the predicted crossing time passes.
        crossing_time = kick + 7.0 / 5.0
        while not session.update(t, *eyes(t, None, observed=False)):
            t += 1 / 30
            self.assertLess(t, crossing_time + 0.1)
        summary = session.summary()
        self.assertEqual(summary["reason"], "crossed")
        self.assertAlmostEqual(summary["crossing"]["x_m"], -0.4 + 0.3 * 7.0 / 5.0, delta=0.01)
        self.assertAlmostEqual(summary["crossing"]["height_m"], -0.25, delta=0.01)
        self.assertGreater(summary["first_prediction"]["made_at"]["z_m"], 6.0)
        last = summary["last_measurement"]
        self.assertAlmostEqual(last["range_m"], np.linalg.norm([last["x_m"], last["height_m"], last["z_m"]]),
                               delta=0.002)

    def test_no_ball_times_out_without_a_crossing(self):
        session = KickSession(CALIBRATION, max_wait_s=1.0)
        t = 0.0
        while not session.update(t, *eyes(t, None, observed=False)):
            t += 1 / 30
        self.assertEqual(session.summary()["reason"], "timeout")
        self.assertIsNone(session.summary()["crossing"])

    def test_ball_rolling_away_gives_no_crossing(self):
        session = KickSession(CALIBRATION, max_wait_s=1.5)
        t, point = 0.0, np.array([0., -0.25, 3.0])
        while not session.update(t, *eyes(t, point)):
            t += 1 / 30
            point = point + [0, 0, 4 / 30]
        self.assertEqual(session.summary()["reason"], "timeout")
        self.assertIsNone(session.summary()["crossing"])


if __name__ == "__main__":
    unittest.main()
