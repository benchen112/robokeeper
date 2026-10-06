import csv
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import cv2
import numpy as np

from robokeeper import AppearanceBallSegmenter, BallCandidate, BallTracker
from track_ball import ReviewWriter, load_timestamps
from test_vision import SequenceSegmenter, candidate


class AppearanceTrackingTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(2)

    def ball(self, center=(120, 110)):
        frame = np.full((240, 320), 90, np.uint8)
        cv2.circle(frame, center, 25, 220, -1)
        cv2.circle(frame, center, 10, 30, -1)
        return frame

    def test_stationary_ball_acquires_without_background_motion(self):
        tracker = BallTracker(AppearanceBallSegmenter(min_radius=18, max_radius=35))
        results = [tracker.process(self.ball(), i / 60) for i in range(5)]
        self.assertEqual(results[2].state, 'confirmed')
        self.assertLess(np.linalg.norm(np.array(results[2].center) - (120, 110)), 4)

    def test_floor_preference_does_not_exclude_airborne_initial_ball(self):
        detector = AppearanceBallSegmenter(min_radius=18, max_radius=35,
                                           floor_region=(0, .75, 1, 1))
        found = detector.find_candidates(self.ball((140, 60)))
        self.assertTrue(any(np.linalg.norm(np.array(c.center) - (140, 60)) < 4 for c in found))

    def test_floor_preference_is_released_when_ball_rises(self):
        tracker = BallTracker(AppearanceBallSegmenter(min_radius=18, max_radius=35,
                                                     floor_region=(0, .5, 1, 1)))
        results = [tracker.process(self.ball((120, 150 - i * 5)), i / 60)
                   for i in range(20)]
        self.assertEqual(results[-1].state, 'confirmed')
        self.assertLess(abs(results[-1].center[1] - 55), 5)

    def test_unrelated_size_does_not_steal_track(self):
        oversized = BallCandidate((46., 50.), 30., .99, (16, 20, 60, 60))
        tracker = BallTracker(SequenceSegmenter([
            (candidate(40),), (candidate(42),), (candidate(44),), (oversized,),
        ]))
        results = [tracker.process(np.zeros((1, 1), np.uint8), i / 60) for i in range(4)]
        self.assertEqual(results[-1].state, 'predicted')
        self.assertFalse(results[-1].observed)

    def test_blank_image_does_not_become_an_edge_measurement(self):
        tracker = BallTracker(AppearanceBallSegmenter(min_radius=18, max_radius=35))
        for i in range(4):
            tracker.process(self.ball(), i / 60)
        result = tracker.process(np.full((240, 320), 90, np.uint8), 4 / 60)
        self.assertEqual(result.state, 'predicted')
        self.assertIsNone(result.center)

    def test_long_capture_gap_reacquires_instead_of_clipping_dt(self):
        tracker = BallTracker(SequenceSegmenter([
            (candidate(40),), (candidate(42),), (candidate(44),), (candidate(150),),
        ]))
        for i in range(3):
            tracker.process(np.zeros((1, 1), np.uint8), i / 60)
        result = tracker.process(np.zeros((1, 1), np.uint8), 1.0)
        self.assertEqual(result.state, 'tentative')
        self.assertEqual(result.velocity_px_s, (0., 0.))


class ReviewTests(unittest.TestCase):
    def test_timestamp_csv_preserves_irregular_spacing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'times.csv'
            path.write_text('frame_index,timestamp_s\n0,0\n1,0.02\n2,0.03\n')
            self.assertEqual(load_timestamps(path, 3), [0., .02, .03])
            with self.assertRaises(ValueError):
                load_timestamps(path, 4)
            path.write_text('frame_index,timestamp_s\n0,0\n1,0\n')
            with self.assertRaises(ValueError):
                load_timestamps(path, 2)

    def test_failed_encode_preserves_existing_review_and_saved_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'review.mp4'
            previous = b'previous completed review'
            path.write_bytes(previous)
            writer = ReviewWriter(path, 60, (128, 96))
            writer.write(np.full((96, 128, 3), 80, np.uint8))
            with patch('track_ball.subprocess.run', side_effect=RuntimeError('encoding interrupted')):
                with self.assertRaises(RuntimeError):
                    writer.close()
            self.assertEqual(path.read_bytes(), previous)
            self.assertTrue(writer.raw_path.exists())
            self.assertFalse(list(Path(folder).glob('review_encode_*.mp4')))

    def test_review_export_keeps_every_frame(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'review.mp4'
            writer = ReviewWriter(path, 60, (128, 96))
            for i in range(12):
                writer.write(np.full((96, 128, 3), i * 15, np.uint8))
            writer.close()
            capture = cv2.VideoCapture(str(path))
            count = 0
            while capture.read()[0]:
                count += 1
            capture.release()
            self.assertEqual(count, 12)
            self.assertFalse(list(Path(folder).glob('*.avi')))


if __name__ == '__main__':
    unittest.main()
