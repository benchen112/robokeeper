import json
from pathlib import Path
import sys
import tempfile
import unittest

import cv2
import numpy as np

from robokeeper.stereo3d import StereoCalibration, fit_kick_calibration, read_stereo_track
from robokeeper.trajectory import TrajectoryPredictor, servo_angle_deg

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

CALIBRATION = StereoCalibration(focal_px=1300, cx=640, cy=330, baseline_m=0.30,
                                disparity_offset_px=4, vertical_offset_px=-33)


def rolling_kick(start, velocity, times, rng=None, pixel_noise=0.0):
    """Stereo pairs (left_xy, r, right_xy, r) for a ball rolling at constant height."""
    pairs = []
    for t in times:
        point = np.add(start, np.multiply(velocity, t))
        left, right = CALIBRATION.project(point)
        if rng is not None:
            left = tuple(np.add(left, rng.normal(0, pixel_noise, 2)))
            right = tuple(np.add(right, rng.normal(0, pixel_noise, 2)))
        radius = CALIBRATION.focal_px * 0.11 / point[2]
        pairs.append((left, radius, right, radius))
    return pairs


class TriangulationTests(unittest.TestCase):
    def test_project_and_triangulate_round_trip_with_offsets(self):
        for point in ((0.0, 0.0, 5.0), (-0.8, -0.05, 2.3), (1.2, 0.4, 9.0)):
            left, right = CALIBRATION.project(point)
            measured = CALIBRATION.triangulate(left, right)
            np.testing.assert_allclose((measured.x_m, measured.height_m, measured.z_m), point,
                                       atol=1e-9)

    def test_tiny_or_negative_disparity_is_rejected(self):
        self.assertIsNone(CALIBRATION.triangulate((640, 300), (640, 333)))

    def test_calibration_json_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.json"
            CALIBRATION.save(path)
            self.assertEqual(StereoCalibration.load(path), CALIBRATION)


class KickCalibrationTests(unittest.TestCase):
    def test_recovers_focal_offset_and_horizon_from_rolling_balls(self):
        truth = StereoCalibration(focal_px=1320, cx=640, cy=315, baseline_m=0.30,
                                  vertical_offset_px=-33)
        clips = []
        for start_z, vx in ((7.0, 1.0), (4.0, 1.3)):
            pairs = []
            for t in np.arange(0, 0.8, 1 / 60):
                point = (-0.4 + vx * t, -0.03, start_z - 4.0 * t)
                left, right = truth.project(point)
                pairs.append((left, 20, right, 20))
            clips.append((pairs, start_z))
        fitted, diagnostics = fit_kick_calibration(clips, 0.30)
        self.assertAlmostEqual(fitted.focal_px, 1320, delta=15)
        self.assertAlmostEqual(fitted.vertical_offset_px, -33, delta=0.5)
        self.assertAlmostEqual(fitted.cy, 315, delta=1)
        self.assertAlmostEqual(diagnostics["ball_center_below_camera_m"], 0.03, delta=0.005)

    def test_needs_a_known_distance(self):
        pairs = rolling_kick((0, 0, 5), (0, 0, -3), np.arange(0, 0.2, 1 / 60))
        with self.assertRaises(ValueError):
            fit_kick_calibration([(pairs, None)], 0.30)


class TrajectoryTests(unittest.TestCase):
    def feed(self, predictor, points_by_time):
        crossing = None
        for t, point in points_by_time:
            predictor.add(t, point)
            crossing = predictor.predict()
        return crossing

    def test_rolling_ball_crossing_from_noisy_stereo(self):
        rng = np.random.default_rng(1)
        times = np.arange(0, 0.5, 1 / 60)
        start, velocity = (-0.4, -0.03, 7.0), (1.5, 0, -10.0)
        samples = []
        for t, (left, _, right, _) in zip(times, rolling_kick(start, velocity, times, rng, 1.0)):
            point = CALIBRATION.triangulate(left, right)
            samples.append((t, (point.x_m, point.height_m, point.z_m)))
        crossing = self.feed(TrajectoryPredictor(), samples)
        # True crossing: t = 0.7 s, X = -0.4 + 1.5 * 0.7 = 0.65 m.
        self.assertEqual(crossing.model, "rolling")
        self.assertAlmostEqual(crossing.time_s, 0.7, delta=0.03)
        self.assertAlmostEqual(crossing.x_m, 0.65, delta=0.08)
        self.assertAlmostEqual(crossing.height_m, -0.03, delta=0.03)
        self.assertLess(crossing.x_std_m, 0.1)

    def test_lofted_ball_uses_gravity(self):
        times = np.arange(0, 0.4, 1 / 60)
        samples = [(t, (0.2 * t, 0.1 + 4.0 * t - 4.905 * t**2, 8.0 - 12.0 * t)) for t in times]
        crossing = self.feed(TrajectoryPredictor(), samples)
        t_cross = 8.0 / 12.0
        self.assertEqual(crossing.model, "airborne")
        self.assertAlmostEqual(crossing.height_m, 0.1 + 4.0 * t_cross - 4.905 * t_cross**2, delta=0.02)
        self.assertAlmostEqual(crossing.x_m, 0.2 * t_cross, delta=0.01)

    def test_one_bad_depth_frame_is_rejected(self):
        times = np.arange(0, 0.3, 1 / 60)
        samples = [(t, (0.5 * t, 0.0, 5.0 - 6.0 * t)) for t in times]
        samples[8] = (samples[8][0], (samples[8][1][0], 0.0, 9.0))
        crossing = self.feed(TrajectoryPredictor(), samples)
        self.assertAlmostEqual(crossing.time_s, 5.0 / 6.0, delta=0.01)
        self.assertEqual(crossing.samples_used, len(samples) - 1)

    def test_no_prediction_until_enough_samples_or_when_receding(self):
        predictor = TrajectoryPredictor(min_samples=5)
        for i in range(4):
            predictor.add(i / 60, (0, 0, 5 - i * 0.1))
            self.assertIsNone(predictor.predict())
        receding = [(i / 60, (0, 0, 3 + i * 0.1)) for i in range(10)]
        self.assertIsNone(self.feed(TrajectoryPredictor(), receding))
        with self.assertRaises(ValueError):
            predictor.add(0, (0, 0, 5))

    def test_servo_angle_points_arm_in_camera_plane(self):
        self.assertAlmostEqual(servo_angle_deg(0, 1), 0)
        self.assertAlmostEqual(servo_angle_deg(1, 1), 45)
        self.assertAlmostEqual(servo_angle_deg(-1, 0), -90)
        self.assertEqual(servo_angle_deg(1, -0.5), 90)  # clamped below the pivot
        self.assertAlmostEqual(servo_angle_deg(0.5, 0.2, pivot_x_m=0.5, pivot_height_m=-0.3), 0)


class TrackReaderTests(unittest.TestCase):
    def test_keeps_only_unclipped_pairs_seen_by_both_eyes(self):
        def eye(center, radius, state="confirmed"):
            return {"state": state, "observed": True, "centroid_px": center, "radius_px": radius}

        rows = [
            {"frame_index": 0, "timestamp_s": 0.0, "both_confirmed_observed": True,
             "left": eye([600, 300], 20), "right": eye([540, 330], 20)},
            {"frame_index": 1, "timestamp_s": 0.1, "both_confirmed_observed": True,
             "left": eye([1270, 300], 20), "right": eye([1200, 330], 20)},
            {"frame_index": 2, "timestamp_s": 0.2, "both_confirmed_observed": False,
             "left": eye([600, 300], 20), "right": eye([540, 330], 20, "absent")},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "track.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            pairs = read_stereo_track(path)
        self.assertEqual([frame for frame, _, _ in pairs], [0])


class CheckerboardTests(unittest.TestCase):
    def test_rendered_board_views_recover_baseline_and_focal(self):
        from calibrate_stereo import calibrate_board, collect_board_pairs

        eye_size, pattern, square = (640, 400), (9, 6), 0.03
        k = np.array([[600, 0, 320], [0, 600, 200], [0, 0, 1]], np.float64)
        baseline = 0.30
        # Texture: one pixel per mm with a white margin of one square.
        scale = 1000
        margin = int(square * scale)
        cols, rows = pattern[0] + 1, pattern[1] + 1
        texture = np.full((rows * margin + 2 * margin, cols * margin + 2 * margin), 255, np.uint8)
        for r in range(rows):
            for c in range(cols):
                if (r + c) % 2 == 0:
                    texture[margin + r * margin: margin + (r + 1) * margin,
                            margin + c * margin: margin + (c + 1) * margin] = 0
        # Texture pixel -> board meters, with the first inner corner at the origin.
        to_board = np.array([[1 / scale, 0, -2 * square], [0, 1 / scale, -2 * square], [0, 0, 1]])
        rng = np.random.default_rng(3)
        frames = []
        # Like a good real capture: the board close enough to fill much of the
        # frame and moved across it, so lens distortion is constrained.
        for _ in range(20):
            rotation, _ = cv2.Rodrigues(rng.uniform(-0.4, 0.4, 3))
            translation = np.array([rng.uniform(-0.05, 0.22), rng.uniform(-0.3, 0.1),
                                    rng.uniform(0.9, 1.5)])
            packed = []
            for eye_x in (-baseline / 2, baseline / 2):
                t_eye = translation - np.array([eye_x + baseline / 2, 0, 0])
                homography = k @ np.column_stack([rotation[:, 0], rotation[:, 1], t_eye]) @ to_board
                packed.append(cv2.warpPerspective(texture, homography, eye_size,
                                                  borderValue=255))
            frames.append(np.hstack(packed))
        left, right = collect_board_pairs(frames, pattern, eye_size, min_move=0)
        calibration, diagnostics = calibrate_board(left, right, pattern, square, eye_size)
        self.assertGreaterEqual(diagnostics["pairs"], 10)
        self.assertAlmostEqual(calibration.baseline_m, baseline, delta=0.01)
        self.assertAlmostEqual(calibration.focal_px, 600, delta=30)
        self.assertLess(diagnostics["stereo_rms_px"], 1.0)
        point = (0.1, 0.0, 2.0)
        left_px = (320 + 600 * (point[0] + baseline / 2) / point[2], 200)
        right_px = (320 + 600 * (point[0] - baseline / 2) / point[2], 200)
        measured = calibration.triangulate(left_px, right_px)
        self.assertAlmostEqual(measured.z_m, 2.0, delta=0.05)
        # The principal point is the least constrained parameter; a few pixels
        # of error shift X by a few cm at 2 m.
        self.assertAlmostEqual(measured.x_m, 0.1, delta=0.04)


if __name__ == "__main__":
    unittest.main()
