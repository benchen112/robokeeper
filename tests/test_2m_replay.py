"""Sparse manually reviewed checks for the real 2 m clip; not a recall benchmark."""
import unittest
from pathlib import Path

import cv2
import numpy as np

from robokeeper import AppearanceBallSegmenter, BallTracker, HybridBallSegmenter
from track_ball import load_timestamps


class TwoMetreReplayTests(unittest.TestCase):
    def test_hybrid_follows_kick_without_distance_profile(self):
        video = Path(__file__).resolve().parents[1] / 'recordings/test_2m_onground_20261002_104834_049431.mp4'
        if not video.exists():
            self.skipTest('Real 2 m recording is not installed')
        cv2.setNumThreads(2)
        capture = cv2.VideoCapture(str(video))
        timestamps = load_timestamps(video.with_suffix('.csv'), int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        tracker = BallTracker(HybridBallSegmenter(), confirmation_hits=1, adaptive_gate=True)
        labels = {110: (675, 402), 120: (764, 386), 130: (898, 366)}
        try:
            for index, timestamp in enumerate(timestamps[:206]):
                ok, frame = capture.read()
                self.assertTrue(ok)
                result = tracker.process(frame, timestamp)
                if index in labels:
                    self.assertTrue(result.observed, f'Missed ball at {index}')
                    self.assertLess(np.linalg.norm(np.array(result.center)-labels[index]), 20)
                if 155 <= index <= 205:
                    self.assertFalse(result.observed, f'False acquisition with ball out of view at {index}')
        finally:
            capture.release()

    def test_ball_identity_before_kick_and_after_edge_reentry(self):
        video = Path(__file__).resolve().parents[1] / 'recordings/test_2m_onground_20261002_104834_049431.mp4'
        if not video.exists():
            self.skipTest('Real 2 m recording is not installed')
        # Approximate centers checked visually in original-resolution frames.
        labels = {0: (550, 357), 20: (570, 353), 50: (567, 365),
                  80: (646, 419), 100: (644, 409), 120: (764, 386), 130: (898, 366)}
        cv2.setNumThreads(2)
        capture = cv2.VideoCapture(str(video))
        timestamps = load_timestamps(video.with_suffix('.csv'), int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        tracker = BallTracker(AppearanceBallSegmenter(min_radius=30, max_radius=140,
                                                     floor_region=(0, .45, 1, 1)))
        results = []
        try:
            for index, timestamp in enumerate(timestamps):
                ok, frame = capture.read()
                self.assertTrue(ok)
                result = tracker.process(frame, timestamp)
                results.append(result)
                if index in labels:
                    self.assertTrue(result.observed, f'No measurement at labeled frame {index}')
                    self.assertLess(np.linalg.norm(np.array(result.center) - labels[index]), 20,
                                    f'Wrong ball center at frame {index}')
                if 160 <= index <= 200:
                    self.assertFalse(result.observed, f'False detection while ball is out of frame: {index}')
                if index > 150 and result.observed:
                    self.assertGreater(result.center[0], 1140, f'Furniture stole the track: {index}')
            self.assertEqual(results[2].state, 'confirmed')
            self.assertTrue(any(r.state == 'confirmed' for r in results[230:245]))
            self.assertGreater(sum(r.observed for r in results[240:301]), 40)
            self.assertEqual(len(results), 321)
        finally:
            capture.release()
