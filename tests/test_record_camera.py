"""Regression checks for recording playback timing."""

import csv
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from record_camera import Clip, JpegFrames, Recorder, repair_video_timing


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools required")
class RecordingTimingTests(unittest.TestCase):
    def make_clip(self, directory, name):
        clip = Clip(Path(directory), name, {"video_fps": 60}, keep_raw=True)
        # The camera actually delivered 20 fps, although the driver reported 60.
        for index in range(31):
            frame = np.full((48, 64, 3), index * 7, dtype=np.uint8)
            ok, jpeg = cv2.imencode(".jpg", frame)
            self.assertTrue(ok)
            clip.write(jpeg.tobytes(), 1_000_000_000 + index * 50_000_000)
        clip.close()
        return clip

    def test_jpeg_frames_split_across_pipe_reads(self):
        parser = JpegFrames()
        frames = [b"\xff\xd8first\xff\xd9", b"\xff\xd8second\xff\xd9"]
        self.assertEqual(list(parser.feed(b"noise\xff")), [])
        self.assertEqual(list(parser.feed(b"\xd8first\xff")), [])
        self.assertEqual(list(parser.feed(b"\xd9" + frames[1] + b"partial")), frames)

    def packet_times(self, path):
        output = subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "packet=pts_time", "-of", "json", str(path)
        ])
        return [float(packet["pts_time"]) for packet in json.loads(output)["packets"]]

    def test_new_mp4_matches_capture_span(self):
        with TemporaryDirectory() as directory:
            clip = self.make_clip(directory, "new")
            path = Path(directory) / clip.convert_to_mp4()
            times = self.packet_times(path)
            self.assertEqual(len(times), 31)
            self.assertAlmostEqual(times[-1], 1.5, delta=0.03)
            self.assertTrue(clip.video_path.exists())

    def test_selects_60_of_each_100_frames_and_removes_raw_after_verified_mp4(self):
        with TemporaryDirectory() as directory:
            clip = Clip(Path(directory), "downsample", {"camera_fps": 100, "video_fps": 60})
            ok, jpeg = cv2.imencode(".jpg", np.zeros((48, 64, 3), dtype=np.uint8))
            self.assertTrue(ok)
            for index in range(150):
                clip.write(jpeg.tobytes(), 1_000_000_000 + index * 10_000_000)
            clip.close()
            self.assertEqual(clip.source_frame_count, 150)
            self.assertEqual(clip.frame_count, 90)
            with clip.timestamps_path.open() as stream:
                timestamps = [float(row["timestamp_s"]) for row in csv.DictReader(stream)]
            gaps = [round(b - a, 2) for a, b in zip(timestamps, timestamps[1:])]
            self.assertEqual(set(gaps), {0.01, 0.02})
            path = Path(directory) / clip.convert_to_mp4()
            self.assertEqual(len(self.packet_times(path)), 90)
            probe = json.loads(subprocess.check_output([
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=r_frame_rate,duration", "-of", "json", str(path)
            ]))["streams"][0]
            self.assertEqual(probe["r_frame_rate"], "60/1")
            self.assertAlmostEqual(float(probe["duration"]), 1.5, delta=0.03)
            self.assertFalse(clip.video_path.exists())
            metadata = json.loads(clip.metadata_path.read_text())
            self.assertEqual(metadata["frames_dropped_for_record_fps"], 60)
            self.assertTrue(metadata["raw_deleted_after_conversion"])

    def test_existing_mp4_can_be_retimed_without_losing_frames(self):
        with TemporaryDirectory() as directory:
            clip = self.make_clip(directory, "old")
            old_mp4 = clip.video_path.with_suffix(".mp4")
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "mjpeg", "-framerate", "60",
                "-i", str(clip.video_path),
                "-c:v", "libx264", "-preset", "ultrafast", str(old_mp4)
            ], check=True)
            old_times = self.packet_times(old_mp4)
            fixed = repair_video_timing(old_mp4)
            fixed_times = self.packet_times(fixed)
            self.assertAlmostEqual(old_times[-1], 0.5, delta=0.03)
            self.assertEqual(len(fixed_times), len(old_times))
            self.assertAlmostEqual(fixed_times[-1], 1.5, delta=0.03)
            self.assertTrue(old_mp4.exists())

    def test_live_compressed_stream_records_without_reencoding(self):
        with TemporaryDirectory() as directory:
            sample = self.make_clip(directory, "source")
            output_dir = Path(directory) / "output"
            args = SimpleNamespace(device="synthetic", width=64, height=48,
                                   fps=60, record_fps=30, keep_raw=False,
                                   output_dir=output_dir, preview_fps=10)
            command = ["ffmpeg", "-v", "error", "-stream_loop", "-1", "-re",
                       "-f", "mjpeg", "-framerate", "60", "-i", str(sample.video_path),
                       "-c:v", "copy", "-f", "mjpeg", "pipe:1"]
            recorder = Recorder(args, source_command=command)
            try:
                with self.assertRaises(ValueError):
                    recorder.start("invalid", 90)
                started = recorder.start("pass through", 24)
                self.assertEqual(started["current_record_fps"], 24)
                time.sleep(0.25)
                sequence, jpeg, running = recorder.next_preview(0)
                self.assertTrue(running)
                self.assertGreater(sequence, 0)
                self.assertTrue(jpeg.startswith(b"\xff\xd8"))
                status = recorder.stop()
                self.assertGreater(status["last_frame_count"], 5)
                self.assertGreater(status["last_source_frame_count"], status["last_frame_count"])
                saved = output_dir / status["last_saved"]
                self.assertEqual(len(self.packet_times(saved)), status["last_frame_count"])
                self.assertFalse(saved.with_suffix(".mjpg").exists())
            finally:
                recorder.close()


if __name__ == "__main__":
    unittest.main()
