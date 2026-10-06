"""Appearance regressions on reviewed crops; these are not an accuracy benchmark."""
import unittest
from pathlib import Path

import cv2

from robokeeper.appearance import BallAppearanceVerifier


class AppearanceVerifierTests(unittest.TestCase):
    def setUp(self):
        cv2.setNumThreads(2)
        self.root = Path(__file__).resolve().parents[1]
        self.verifier = BallAppearanceVerifier()

    def score(self, distance, index, x, y, radius):
        paths = list((self.root / 'recordings').glob(f'test_{distance}_onground*.mp4'))
        if not paths:
            self.skipTest('Reviewed recording is not installed')
        capture = cv2.VideoCapture(str(paths[0]))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            self.assertTrue(ok)
            gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (3, 3), 0)
            return self.verifier.score(gray, x, y, radius)
        finally:
            capture.release()

    def test_reviewed_window_and_shoes_are_rejected(self):
        for args in [('2m', 216, 755, 125, 21), ('2m', 263, 733, 409, 33),
                     ('5m', 91, 602, 417, 15), ('5m', 335, 580, 406, 14)]:
            with self.subTest(crop=args):
                self.assertLess(self.score(*args), -.6)

    def test_partly_visible_ball_and_separate_11m_clip(self):
        self.assertGreater(self.score('2m', 240, 1275, 355, 94), .6)
        # 11 m images were excluded from model training, but share this environment.
        self.assertGreater(self.score('11m', 180, 695, 378, 37), .6)
        self.assertGreater(self.score('11m', 200, 943, 343, 75), .6)

    @unittest.expectedFailure
    def test_small_11m_ball_is_still_a_known_classifier_miss(self):
        self.assertGreater(self.score('11m', 130, 575, 394, 15), -.6)
