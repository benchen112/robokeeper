import unittest

import cv2
import numpy as np

from robokeeper.focus import FocusMeter, bar, sharpness


class FocusTests(unittest.TestCase):
    def scene(self):
        rng = np.random.default_rng(0)
        image = np.full((200, 320), 100, np.uint8)
        for _ in range(40):
            x, y = rng.integers(0, 320), rng.integers(0, 200)
            cv2.rectangle(image, (int(x), int(y)), (int(x) + 12, int(y) + 8), int(rng.integers(0, 255)), -1)
        return image

    def test_blur_lowers_sharpness(self):
        sharp = self.scene()
        values = [sharpness(cv2.GaussianBlur(sharp, (0, 0), sigma)) if sigma else sharpness(sharp)
                  for sigma in (0, 1, 2, 4)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertGreater(values[0], 4 * values[-1])

    def test_trend_follows_focus_and_ignores_noise(self):
        rng = np.random.default_rng(1)
        meter = FocusMeter()
        for _ in range(40):
            meter.update(100 + rng.normal(0, 2))
        self.assertEqual(meter.trend, "steady")
        for value in np.linspace(100, 160, 30):
            meter.update(value + rng.normal(0, 2))
        self.assertEqual(meter.trend, "sharper")
        self.assertGreater(meter.fraction_of_best, .95)
        for value in np.linspace(160, 110, 30):
            meter.update(value + rng.normal(0, 2))
        self.assertEqual(meter.trend, "blurrier")
        self.assertLess(meter.fraction_of_best, .8)
        meter.reset_best()
        self.assertAlmostEqual(meter.fraction_of_best, 1.)

    def test_bar(self):
        self.assertEqual(bar(.5, 4), "##..")
        self.assertEqual(bar(1.4, 4), "####")


if __name__ == "__main__":
    unittest.main()
