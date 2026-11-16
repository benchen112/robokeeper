import argparse
import unittest
from unittest import mock

import numpy as np

from robokeeper.exposure import (add_exposure_arguments, calibrate_exposure,
                                 open_calibrated_camera)


class FakeCamera:
    """Linear sensor: level = scene * exposure * gain; new settings apply 2 frames late."""

    def __init__(self, scene, bright_spot=0., white=235, spot_gain=4., bright_rows=0.):
        self.scene, self.bright_spot, self.white, self.spot_gain = scene, bright_spot, white, spot_gain
        self.bright_rows = bright_rows  # top rows this much brighter, e.g. ceiling lights
        self.pending, self.applied, self.sequence = [], (2000, 8.), 0

    def set_exposure(self, exposure_us, gain):
        self.pending.append((self.sequence + 2, (exposure_us, gain)))

    def read(self, sequence):
        self.sequence += 1
        while self.pending and self.pending[0][0] <= self.sequence:
            self.applied = self.pending.pop(0)[1]
        exposure_us, gain = self.applied
        frame = np.full((80, 160), self.scene * exposure_us * gain)
        frame[:, :int(160 * self.bright_spot)] *= self.spot_gain  # e.g. a window
        frame[:int(80 * self.bright_rows)] *= self.spot_gain
        frame = np.clip(frame, 16, self.white).astype(np.uint8)  # limited range
        return self.sequence, frame, 0., 0., {"exposure_us": exposure_us, "analogue_gain": gain}


class ExposureTests(unittest.TestCase):
    def test_reaches_target_with_the_preferred_gain(self):
        camera = FakeCamera(scene=120 / (600 * 8))  # bright enough at 600 us
        exposure_us, gain, level, clipped, converged = calibrate_exposure(camera, target=120)
        self.assertTrue(converged)
        self.assertAlmostEqual(level, 120, delta=120 * .12)
        self.assertEqual(gain, 8.)
        self.assertLessEqual(exposure_us, 1000)

    def test_dim_scene_raises_gain_at_the_blur_limit(self):
        camera = FakeCamera(scene=209 / (4000 * 8))  # the Pi reading: 209 at 4000 us, gain 8
        exposure_us, gain, level, clipped, converged = calibrate_exposure(camera, target=120)
        self.assertTrue(converged)
        self.assertAlmostEqual(level, 120, delta=120 * .12)
        self.assertGreater(gain, 8.)
        self.assertLessEqual(exposure_us, 1300)  # past 1000 us only once gain is at 16

    def test_ceiling_lights_above_the_metered_rows_do_not_stop_brightening(self):
        # The 2026-11-12 room: lights in the top of the frame clipped early, so the
        # whole-frame calibration stopped with the floor (and ball) at ~61.
        def run(rows):
            camera = FakeCamera(scene=60 / (1800 * 8), bright_rows=.2, spot_gain=3.)
            return calibrate_exposure(camera, target=120, meter_rows=rows)
        whole = run((0., 1.))
        lower = run((.4, 1.))
        self.assertTrue(lower[4])
        self.assertAlmostEqual(lower[2], 120, delta=120 * .12)
        self.assertLess(whole[2], 100)

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

    def test_sunlit_window_is_left_out_instead_of_darkening_the_room(self):
        # The 2026-11-04 hallway: glass doors far brighter than the room stay
        # saturated at any exposure; the room should still reach the target.
        camera = FakeCamera(scene=70 / (2000 * 8), bright_spot=.15, spot_gain=30.)
        info = {}
        exposure_us, gain, level, clipped, converged = calibrate_exposure(camera, target=120, info=info)
        self.assertTrue(converged)
        self.assertAlmostEqual(level, 120, delta=120 * .12)
        self.assertGreater(exposure_us * gain, 2000 * 8)  # brighter, not darker
        self.assertAlmostEqual(info["light_source_fraction"], .15, delta=.02)

    def test_dark_scene_maxes_gain_then_lengthens_exposure(self):
        camera = FakeCamera(scene=80 / (3000 * 8))  # needs 3000 us at gain 12, or 2250 at 16
        exposure_us, gain, level, _, converged = calibrate_exposure(camera, target=120)
        self.assertTrue(converged)
        self.assertEqual(gain, 16.)
        self.assertGreater(exposure_us, 1000)
        self.assertLess(exposure_us, 3000)

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
        self.assertLessEqual(args.exposure_us, 1000)  # Calibrated darker, and kept for the clip
        self.assertLess(args.gain, 9.)

    def test_given_exposure_is_used_as_is(self):
        args, opened = self.open("--exposure-us", "500", "--gain", "4")
        self.assertEqual(opened, [(500, 4.0)])
        self.assertEqual((args.exposure_us, args.gain), (500, 4.0))


if __name__ == "__main__":
    unittest.main()
