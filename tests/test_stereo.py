import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from robokeeper.stereo import LatestPicameraStereo, pair_diagnostics, split_stereo
from robokeeper.vision import TrackResult


class StereoTests(unittest.TestCase):
    def test_native_mode_and_manual_exposure_do_not_enumerate_sensor_modes(self):
        driver = Mock()
        driver.camera_configuration.return_value = {
            "main": {"size": (2560, 800)}, "raw": {"size": (2560, 800)}}
        driver.camera_controls = {"FrameDurationLimits": (16655, 1000000, 16655)}
        constructor = Mock(return_value=driver)
        constructor.global_camera_info.return_value = [{"Model": "arducam-pivariety"}]
        with patch.dict(sys.modules, {"picamera2": types.SimpleNamespace(Picamera2=constructor)}), \
                patch("robokeeper.stereo.threading.Thread"):
            camera = LatestPicameraStereo(tuning_file="ov9281_mono.json",
                                          exposure_us=3000, gain=2)
            config = driver.create_video_configuration.call_args.kwargs
            self.assertEqual(config["sensor"], {"output_size": (2560, 800), "bit_depth": 8})
            self.assertEqual(config["raw"]["format"], "R8")
            self.assertEqual(config["controls"], {"FrameRate": 60, "AeEnable": False,
                                                   "ExposureTime": 3000, "AnalogueGain": 2})
            self.assertFalse(config["queue"])
            constructor.load_tuning_file.assert_called_once_with("ov9281_mono.json")
            constructor.global_camera_info.assert_not_called()
            camera.close()
            driver.stop.assert_called_once()
            driver.close.assert_called_once()

    def test_eye_coordinates_and_swap_preserve_pixels(self):
        frame = np.zeros((80, 256), np.uint8)
        frame[30, 20] = 110
        frame[40, 128+25] = 220
        left, right = split_stereo(frame, (128, 80))
        self.assertEqual(left[30, 20], 110)
        self.assertEqual(right[40, 25], 220)
        swapped, _ = split_stereo(frame, (128, 80), swap=True)
        np.testing.assert_array_equal(swapped, right)
        with self.assertRaises(ValueError):
            split_stereo(frame[:, :128], (128, 80))

    def test_predictions_are_not_stereo_measurements(self):
        def result(state, observed, center):
            return TrackResult(1, state, observed, center, (12, 20),
                               (0, 0), 5, .9, ())
        left = result("confirmed", True, (15, 20))
        right = result("confirmed", True, (10, 22))
        self.assertEqual(pair_diagnostics(left, right)["raw_disparity_px"], 5)
        self.assertEqual(pair_diagnostics(left, right)["raw_vertical_offset_px"], -2)
        gap = pair_diagnostics(left, result("predicted", False, None))
        self.assertFalse(gap["both_confirmed_observed"])
        self.assertIsNone(gap["raw_disparity_px"])

    def test_picamera_copies_luma_and_releases_request(self):
        camera = LatestPicameraStereo.__new__(LatestPicameraStereo)
        camera.condition = threading.Condition()
        camera.sequence = 0
        camera.latest = None
        camera.running = True
        camera.error = None
        camera.timestamp_source = None
        camera.width, camera.height = 8, 4
        yuv = np.arange(48, dtype=np.uint8).reshape(6, 8)
        request = Mock()
        request.get_metadata.return_value = {"SensorTimestamp": 5_000_000_000,
                                            "ExposureTime": 1000}
        request.make_array.return_value = yuv
        request.release.side_effect = lambda: setattr(camera, "running", False)
        camera.camera = Mock()
        camera.camera.capture_request.return_value = request
        camera._read_loop()
        request.release.assert_called_once()
        sequence, frame, timestamp, received, metadata = camera.read(0)
        self.assertEqual(sequence, 1)
        self.assertEqual(timestamp, 5)
        self.assertEqual(metadata["timestamp_source"], "sensor")
        self.assertEqual(frame.shape, (4, 8))
        yuv.fill(0)
        self.assertNotEqual(frame.sum(), 0)
        self.assertGreater(received, 0)

    def test_picamera_releases_request_on_conversion_failure(self):
        camera = LatestPicameraStereo.__new__(LatestPicameraStereo)
        camera.condition = threading.Condition()
        camera.running = True
        request = Mock()
        request.make_array.side_effect = RuntimeError("Bad array")
        camera.camera = Mock()
        camera.camera.capture_request.return_value = request
        camera._read_loop()
        request.release.assert_called_once()
        self.assertEqual(camera.error, "Bad array")
        self.assertFalse(camera.running)

    def test_runner_tracks_both_eyes_and_preserves_replay_timing(self):
        with tempfile.TemporaryDirectory() as folder:
            video = Path(folder) / "stereo.avi"
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"),
                                     60, (256, 96))
            self.assertTrue(writer.isOpened())
            try:
                for i in range(16):
                    image = np.full((96, 256, 3), 50, np.uint8)
                    if i >= 5:
                        cv2.circle(image, (35+(i-5)*3, 60), 9, (230,)*3, -1)
                        cv2.circle(image, (128+25+(i-5)*3, 60), 9, (230,)*3, -1)
                    writer.write(image)
            finally:
                writer.release()
            video.with_suffix(".csv").write_text(
                "frame_index,timestamp_s\n" + "".join(f"{i},{i/50}\n" for i in range(16)))
            output = Path(folder) / "pairs.jsonl"
            completed = subprocess.run([
                sys.executable, str(Path(__file__).resolve().parents[1] / "track_stereo.py"),
                "--video", str(video), "--eye-width", "128", "--eye-height", "96",
                "--detector", "motion", "--jsonl", str(output),
            ], capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 16)
            self.assertEqual(rows[-1]["timestamp_s"], .3)
            self.assertEqual(rows[-1]["left"]["timestamp_s"], rows[-1]["right"]["timestamp_s"])
            self.assertIsNone(rows[-1]["pair_frame_age_ms"])
            self.assertEqual(rows[-1]["skipped_capture_frames"], 0)
            measured = [row for row in rows if row["both_confirmed_observed"]]
            self.assertTrue(measured)
            self.assertAlmostEqual(measured[-1]["raw_disparity_px"], 10, delta=1)


if __name__ == "__main__":
    unittest.main()
