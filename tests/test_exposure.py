import argparse
import unittest
from unittest import mock

import numpy as np

from robokeeper.exposure import (add_exposure_arguments, calibrate_exposure,
                                 open_calibrated_camera)


class FakeCamera:
    """Linear sensor: level = scene * exposure * gain; new settings apply 2 frames late."""

    def __init__(self, scene, bright_spot=0., white=235):
        self.scene, self.bright_spot, self.white = scene, bright_spot, white
        self.pending, self.applied, self.sequence = [], (2000, 8.), 0

    def set_exposure(self, exposure_us, gain):
        self.pending.append((self.sequence + 2, (exposure_us, gain)))

    def read(self, sequence):
        self.sequence += 1
        while self.pending and self.pending[0][0] <= self.sequence:
            self.applied = self.pending.pop(0)[1]
        exposure_us, gain = self.applied
        frame = np.full((80, 160), self.scene * exposure_us * gain)
        frame[:, :int(160 * self.bright_spot)] *= 4  # e.g. a window
        frame = np.clip(frame, 16, self.white).astype(np.uint8)  # limited range
        return self.sequence, frame, 0., 0., {"exposure_us": exposure_us, "analogue_gain": gain}


class ExposureTests(unittest.TestCase):
    def test_reaches_target_with_the_preferred_gain(self):
        camera = FakeCamera(scene=209 / (4000 * 8))  # the Pi reading: 209 at 4000 us, gain 8
        exposure_us, gain, level, clipped, converged = calibrate_exposure(camera, target=120)
        self.assertTrue(converged)
        self.assertAlmostEqual(level, 120, delta=120 * .12)
        self.assertEqual(gain, 8.)
        self.assertLess(exposure_us, 3000)

    def test_overexposed_limited_range_frame_goes_darker(self):
        # The Pi's failed run: mean 225 at 2000 us, gain 8, white at 235.
        camera = FakeCamera(scene=300 / (2000 * 8))
        exposure_us, gain, level, clipped, converged = calibrate_exposure(camera, target=120)
        self.assertTrue(converged)
        self.assertAlmostEqual(level, 120, delta=120 * .12)
        self.assertLess(exposure_us, 1000)
        self.assertLessEqual(clipped, .02)

    def test_too_bright_without_saturation_goes_darker(self):
        camera = FakeCamera(scene=200 / (2000 * 8))
        exposure_us, _, level, _, converged = calibrate_exposure(camera, target=120)
        self.assertTrue(converged)
        self.assertAlmostEqual(level, 120, delta=120 * .12)
        self.assertLess(exposure_us, 2000)

    def test_dark_scene_caps_exposure_then_raises_gain(self):
        camera = FakeCamera(scene=80 / (3000 * 8))  # needs gain ~12
        exposure_us, gain, level, _, converged = calibrate_exposure(camera, target=120)
        self.assertTrue(converged)
        self.assertEqual(exposure_us, 3000)
        self.assertGreater(gain, 8.)

    def test_too_dark_for_the_caps_reports_not_converged(self):
        camera = FakeCamera(scene=5 / (3000 * 8))
        exposure_us, gain, _, _, converged = calibrate_exposure(camera, target=120)
        self.assertFalse(converged)
        self.assertEqual((exposure_us, gain), (3000, 16.))
        self.assertEqual(camera.pending[-1][1], (3000, 16.))  # camera left at the result

    def test_saturated_area_forces_darker(self):
        camera = FakeCamera(scene=120 / (2000 * 8), bright_spot=.2)
        exposure_us, _, level, clipped, converged = calibrate_exposure(camera, target=120)
        # The target needs the window saturated: stop at the brightest unclipped setting.
        self.assertTrue(converged)
        self.assertLessEqual(clipped, .02)
        self.assertLess(exposure_us, 2000)
        self.assertGreater(level, 80)


class OpenCalibratedCameraTests(unittest.TestCase):
    def open(self, *argv):
        parser = argparse.ArgumentParser()
        add_exposure_arguments(parser)
        args = parser.parse_args(argv)
        args.eye_width, args.eye_height, args.fps, args.camera_num, args.tuning_file = 1280, 800, 60, 0, None
        opened = []
        def fake_camera(width, height, fps, camera_num, tuning_file, exposure_us, gain):
            opened.append((exposure_us, gain))
            camera = FakeCamera(scene=225 / (2000 * 8))
            camera.close = lambda: None
            return camera
        with mock.patch("robokeeper.stereo.LatestPicameraStereo", fake_camera):
            open_calibrated_camera(args)
        return args, opened

    def test_defaults_calibrate_and_record_the_result(self):
        args, opened = self.open()
        self.assertEqual(opened, [(2000, 8.0)])
        self.assertLess(args.exposure_us, 2000)  # Calibrated darker, and kept for the clip
        self.assertEqual(args.gain, 8.0)

    def test_given_exposure_is_used_as_is(self):
        args, opened = self.open("--exposure-us", "500", "--gain", "4")
        self.assertEqual(opened, [(500, 4.0)])
        self.assertEqual((args.exposure_us, args.gain), (500, 4.0))


if __name__ == "__main__":
    unittest.main()
