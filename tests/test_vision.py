import unittest

import cv2
import numpy as np

from robokeeper import BallCandidate, BallTracker, MotionBallSegmenter


class SequenceSegmenter:
    ready = True

    def __init__(self, frames):
        self.frames = iter(frames)

    def find_candidates(self, frame):
        return next(self.frames)


def candidate(x, y=50, confidence=0.9):
    return BallCandidate((float(x), float(y)), 6.0, confidence, (x - 6, y - 6, 12, 12))


class BallVisionTests(unittest.TestCase):
    def test_moving_patterned_ball_and_static_lookalike(self):
        tracker = BallTracker(MotionBallSegmenter(warmup_frames=8, max_area=500))
        confirmed = []
        for index in range(75):
            frame = np.full((140, 240), 80, dtype=np.uint8)
            # A stationary ball-shaped object is part of the background.
            cv2.circle(frame, (190, 110), 8, 220, -1)
            if index >= 10:
                x = 20 + 2 * (index - 10)
                cv2.circle(frame, (x, 55), 7, 220, -1)
                cv2.circle(frame, (x, 55), 3, 30, -1)
            result = tracker.process(frame, index / 100)
            if result.state == "confirmed" and result.observed:
                confirmed.append((index, result.center))
        self.assertGreater(len(confirmed), 50)
        self.assertTrue(all(abs(center[0] - (20 + 2 * (index - 10))) <= 2
                            and abs(center[1] - 55) <= 2
                            for index, center in confirmed))

    def test_two_independent_1280x800_60fps_streams(self):
        trackers = [BallTracker(MotionBallSegmenter(warmup_frames=5, min_area=3))
                    for _ in range(2)]
        counts = [0, 0]
        for index in range(35):
            for camera_id, tracker in enumerate(trackers):
                frame = np.full((800, 1280), 80, dtype=np.uint8)
                if index >= 7:
                    x = 100 + 12 * (index - 7) + camera_id * 20
                    cv2.circle(frame, (x, 350), 3, 230, -1)
                    cv2.circle(frame, (x, 350), 1, 25, -1)
                result = tracker.process(frame, index / 60)
                if result.state == "confirmed" and result.observed:
                    counts[camera_id] += 1
                    self.assertLessEqual(abs(result.center[0] - x), 2)
        self.assertGreater(min(counts), 20)

    def test_single_frame_lookalike_does_not_confirm(self):
        tracker = BallTracker(SequenceSegmenter([
            (candidate(40),), (), (candidate(42),), (),
        ]))
        states = [tracker.process(np.zeros((1, 1), np.uint8), i / 100).state
                  for i in range(4)]
        self.assertEqual(states, ["tentative", "absent", "tentative", "absent"])

    def test_confirmed_track_survives_short_occlusion(self):
        tracker = BallTracker(SequenceSegmenter([
            (candidate(40),), (candidate(42),), (candidate(44),),
            (), (), (candidate(50),),
        ]))
        results = [tracker.process(np.zeros((1, 1), np.uint8), i / 100)
                   for i in range(6)]
        self.assertEqual([result.state for result in results], [
            "tentative", "tentative", "confirmed", "predicted", "predicted", "confirmed"
        ])
        self.assertIsNone(results[3].center)
        self.assertFalse(results[3].observed)
        self.assertAlmostEqual(results[5].center[0], 50)

    def test_large_scene_change_suppresses_candidates(self):
        segmenter = MotionBallSegmenter(warmup_frames=3)
        dark = np.full((100, 100), 60, dtype=np.uint8)
        for _ in range(4):
            segmenter.find_candidates(dark)
        bright = np.full((100, 100), 150, dtype=np.uint8)
        self.assertEqual(segmenter.find_candidates(bright), ())

    def test_timestamps_must_increase(self):
        tracker = BallTracker(SequenceSegmenter([(), ()]))
        tracker.process(np.zeros((1, 1), np.uint8), 1.0)
        with self.assertRaises(ValueError):
            tracker.process(np.zeros((1, 1), np.uint8), 1.0)


if __name__ == "__main__":
    unittest.main()
