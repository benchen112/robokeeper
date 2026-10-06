#!/usr/bin/env python3
"""Preview and record packed stereo clips from the Camarray CSI HAT.

By default a browser page shows the live stereo preview; name the clip there
and start/stop recording. ``--duration`` instead records one clip and exits.

Every captured pair is JPEG-encoded on worker threads and remuxed (no
re-encode) into an MJPEG AVI that ``track_stereo.py --video`` can replay. A
same-name CSV holds sensor timestamps so replay uses real capture times. After
recording stops, an H.264 MP4 copy is made for viewing in a browser or VS Code.
"""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
from http.server import ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from record_camera import Handler, safe_name


def encode(frame, quality):
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return jpeg.tobytes()


def remux(mjpg_path, avi_path, fps, frame_count):
    """Wrap the concatenated JPEGs in AVI without re-encoding, then verify."""
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "mjpeg",
         "-framerate", f"{fps:g}", "-i", str(mjpg_path), "-c:v", "copy",
         # Without an output rate the AVI gets a 1/600 time base and OpenCV
         # reports 10x the frames, which breaks replay against the CSV.
         "-r", f"{fps:g}", str(avi_path)],
        check=True, capture_output=True, text=True)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(avi_path)],
        check=True, capture_output=True, text=True)
    if int(probe.stdout.strip()) != frame_count:
        raise RuntimeError(f"AVI has {probe.stdout.strip()} frames, expected {frame_count}")
    # Replay reads through OpenCV, so check the frame count it will see.
    video = cv2.VideoCapture(str(avi_path))
    opencv_count = round(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    if opencv_count != frame_count:
        raise RuntimeError(f"OpenCV sees {opencv_count} AVI frames, expected {frame_count}")


def make_preview_mp4(avi_path, mp4_path, fps, frame_count, span_s):
    """Viewing copy only; tracking should keep using the AVI and CSV."""
    expected_span = (frame_count - 1) / fps if frame_count > 1 else 0
    # Stretch playback to the real sensor span if frames were lost.
    scale = span_s / expected_span if span_s > 0 and expected_span > 0 else 1.0
    temporary_path = mp4_path.with_suffix(".part.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(avi_path),
             "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
             # passthrough keeps every frame; the default CFR mode drops one
             # whenever the scaled span is slightly shorter than nominal.
             "-vf", f"setpts=(PTS-STARTPTS)*{scale:.12g}", "-fps_mode", "passthrough",
             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary_path)],
            check=True, capture_output=True, text=True)
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
             "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(temporary_path)],
            check=True, capture_output=True, text=True)
        if int(probe.stdout.strip()) != frame_count:
            raise RuntimeError(f"MP4 has {probe.stdout.strip()} frames, expected {frame_count}")
        temporary_path.replace(mp4_path)
    finally:
        temporary_path.unlink(missing_ok=True)


class StereoClip:
    """Files and per-frame rows for one recording; used only by its owner."""

    def __init__(self, output_dir, name, settings, pool):
        stem = f"{safe_name(name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.mjpg_path = output_dir / f"{stem}.mjpg"
        self.avi_path = output_dir / f"{stem}.avi"
        self.mp4_path = output_dir / f"{stem}.mp4"
        self.csv_path = output_dir / f"{stem}.csv"
        self.json_path = output_dir / f"{stem}.json"
        self.settings = settings
        self.pool = pool
        self.video_file = open(self.mjpg_path, "xb")
        try:
            self.csv_file = open(self.csv_path, "x", newline="", encoding="utf-8")
        except OSError:
            self.video_file.close()
            self.mjpg_path.unlink(missing_ok=True)
            raise
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(("frame_index", "timestamp_s", "sensor_timestamp_ns",
                              "capture_sequence", "exposure_us", "analogue_gain"))
        # Bounded so a slow SD card shows up as skipped frames, not RAM growth.
        self.pending = deque()
        self.max_pending = settings["encoder_threads"] * 4
        self.frame_count = self.skipped = 0
        self.last_sequence = None
        self.started = time.monotonic()
        self.sensor_times, self.exposures, self.gains = [], [], []

    def add(self, sequence, frame, metadata):
        if metadata["sensor_timestamp_ns"] is None:
            raise RuntimeError("Camera did not report SensorTimestamp")
        if self.last_sequence is not None:
            self.skipped += max(0, sequence - self.last_sequence - 1)
        self.last_sequence = sequence
        self.sensor_times.append(metadata["sensor_timestamp_ns"])
        self.exposures.append(metadata["exposure_us"])
        self.gains.append(metadata["analogue_gain"])
        self.pending.append((self.pool.submit(encode, frame, self.settings["jpeg_quality"]), (
            self.frame_count,
            f"{(metadata['sensor_timestamp_ns']-self.sensor_times[0])/1e9:.9f}",
            metadata["sensor_timestamp_ns"], sequence,
            metadata["exposure_us"], metadata["analogue_gain"])))
        self.frame_count += 1
        self.flush(self.max_pending)

    def flush(self, limit):
        while len(self.pending) > limit:
            future, row = self.pending.popleft()
            self.video_file.write(future.result())
            self.writer.writerow(row)

    def discard(self):
        for future, _ in self.pending:
            future.cancel()
        self.video_file.close()
        self.csv_file.close()
        for path in (self.mjpg_path, self.csv_path):
            path.unlink(missing_ok=True)

    def finish(self, stop_reason):
        """Write the AVI, MP4 and JSON summary; returns the summary."""
        try:
            self.flush(0)
        finally:
            self.video_file.close()
            self.csv_file.close()
        if self.frame_count == 0:
            for path in (self.mjpg_path, self.csv_path):
                path.unlink(missing_ok=True)
            raise ValueError("No frames were recorded; please try again")
        fps = self.settings["requested_fps"]
        gaps_ms = np.diff(self.sensor_times) / 1e6 if self.frame_count > 1 else np.array([])
        span_s = (self.sensor_times[-1] - self.sensor_times[0]) / 1e9
        summary = {
            "video": self.avi_path.name, "timestamps": self.csv_path.name,
            "preview_mp4": None, **self.settings,
            "frame_count": self.frame_count, "sensor_span_s": span_s,
            "average_fps": (self.frame_count - 1) / span_s if span_s > 0 else None,
            "skipped_by_recorder": self.skipped,
            # Sensor gaps over 1.5 frame periods mean frames never reached the recorder.
            "sensor_gaps_over_1_5_periods": int((gaps_ms > 1.5 * 1000 / fps).sum()),
            "max_sensor_gap_ms": float(gaps_ms.max()) if gaps_ms.size else None,
            "exposure_us_range": [min(self.exposures), max(self.exposures)],
            "analogue_gain_range": [min(self.gains), max(self.gains)],
            "stop_reason": stop_reason,
            "timestamp_note": "timestamp_s is SensorTimestamp relative to the first recorded frame.",
        }
        logging.info("Remuxing %d frames to %s", self.frame_count, self.avi_path)
        remux(self.mjpg_path, self.avi_path, fps, self.frame_count)
        if not self.settings["keep_mjpg"]:
            self.mjpg_path.unlink()
        self.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        logging.info("Saved %s: %d frames, %.1f fps, %d skipped, %d sensor gaps, "
                     "exposure %s us, gain %s", self.avi_path, self.frame_count,
                     summary["average_fps"] or 0, self.skipped,
                     summary["sensor_gaps_over_1_5_periods"],
                     summary["exposure_us_range"], summary["analogue_gain_range"])
        if self.skipped or summary["sensor_gaps_over_1_5_periods"]:
            logging.warning("Frames were lost; try --encoder-threads 4 or a lower --quality")
        # A failed viewing copy must not lose the AVI and CSV used for tracking.
        try:
            make_preview_mp4(self.avi_path, self.mp4_path, fps, self.frame_count, span_s)
            summary["preview_mp4"] = self.mp4_path.name
            self.json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            logging.info("Saved viewing copy %s", self.mp4_path)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            logging.warning("MP4 preview failed; the AVI is still valid: %s",
                            getattr(exc, "stderr", None) or exc)
        return summary


class StereoRecorder:
    """Run the camera continuously, feeding the preview and an optional clip.

    Provides the start/stop/status/next_preview interface record_camera's
    browser Handler expects.
    """

    def __init__(self, camera, args):
        self.camera = camera
        self.output_dir = args.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preview_interval = 1 / args.preview_fps
        self.settings = {
            "packed_size": [args.eye_width * 2, args.eye_height],
            "eye_size": [args.eye_width, args.eye_height],
            "requested_fps": args.fps, "jpeg_quality": args.quality,
            "requested_exposure_us": args.exposure_us,
            "requested_gain": args.gain if args.exposure_us else None,
            "encoder_threads": args.encoder_threads, "keep_mjpg": args.keep_mjpg,
        }
        self.pool = ThreadPoolExecutor(args.encoder_threads)
        self.condition = threading.Condition()
        self.running = True
        self.error = None
        self.clip = None
        # Startup stalls (encoder threads, auto exposure) drop a burst of early
        # frames, so the first second is preview only.
        self.ready_at = time.monotonic() + 1.0
        self.finalizing = None
        self.finalizing_error = None
        self.last_summary = None
        self.jpeg = None
        self.jpeg_sequence = 0
        self.last_preview = 0
        self.recent = deque(maxlen=61)
        list(self.pool.map(encode, [np.zeros((16, 16), np.uint8)] * args.encoder_threads,
                           [args.quality] * args.encoder_threads))
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        sequence = 0
        try:
            while self.running:
                sample = self.camera.read(sequence)
                if sample is None:
                    if not self.running:
                        break
                    raise RuntimeError(getattr(self.camera, "error", None)
                                       or "No new stereo frame within 2 seconds")
                sequence, frame, _, received, metadata = sample
                preview = None
                if received - self.last_preview >= self.preview_interval:
                    width = min(1280, frame.shape[1])
                    preview = encode(cv2.resize(
                        frame, (width, round(frame.shape[0] * width / frame.shape[1]))), 80)
                    self.last_preview = received
                with self.condition:
                    self.recent.append(received)
                    if self.clip:
                        self.clip.add(sequence, frame, metadata)
                    if preview:
                        self.jpeg = preview
                        self.jpeg_sequence += 1
                        self.condition.notify_all()
        except Exception as exc:
            logging.error("Recorder error: %s", exc)
            with self.condition:
                self.error = str(exc)
                self.running = False
                self.condition.notify_all()

    def start(self, name, record_fps=None):
        # record_fps is part of the shared Handler's interface; stereo keeps every frame.
        with self.condition:
            if not self.running or self.error:
                raise RuntimeError(self.error or "Camera is stopped")
            if self.clip:
                raise ValueError("Already recording")
            if self.finalizing:
                raise ValueError("Wait for the previous clip to finish saving")
            if time.monotonic() < self.ready_at:
                raise ValueError("Camera is still starting; try again in a second")
            self.clip = StereoClip(self.output_dir, name or "stereo", self.settings, self.pool)
            self.finalizing_error = None
            logging.info("Recording %s", self.clip.avi_path)
            return self.status_locked()

    def stop(self, stop_reason="stopped"):
        with self.condition:
            if not self.clip:
                raise ValueError("No recording is in progress")
            clip, self.clip = self.clip, None
            self.finalizing = clip.avi_path.name
        try:
            summary = clip.finish(stop_reason)
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
            with self.condition:
                self.finalizing_error = str(getattr(exc, "stderr", None) or exc)
            raise RuntimeError(self.finalizing_error) from exc
        finally:
            with self.condition:
                self.finalizing = None
        with self.condition:
            self.last_summary = summary
            return self.status_locked()

    def status_locked(self):
        recent = self.recent
        span = recent[-1] - recent[0] if len(recent) > 1 else 0
        clip = self.clip
        last = self.last_summary
        return {
            "recording": clip is not None,
            "current_file": clip.avi_path.name if clip else None,
            "frames_recorded": clip.frame_count if clip else 0,
            "recording_s": time.monotonic() - clip.started if clip else 0,
            "skipped": clip.skipped if clip else 0,
            "live_fps": (len(recent) - 1) / span if len(recent) >= 10 and span > 0 else None,
            "finalizing": self.finalizing is not None,
            "finalizing_file": self.finalizing,
            "finalizing_error": self.finalizing_error,
            "last_saved": last and {key: last[key] for key in (
                "video", "preview_mp4", "frame_count", "sensor_span_s", "average_fps",
                "skipped_by_recorder", "sensor_gaps_over_1_5_periods")},
            "camera_mode": self.settings,
            "error": self.error,
        }

    def status(self):
        with self.condition:
            return self.status_locked()

    def next_preview(self, previous):
        with self.condition:
            self.condition.wait_for(
                lambda: self.jpeg_sequence != previous or not self.running, timeout=5)
            return self.jpeg_sequence, self.jpeg, self.running

    def close(self):
        try:
            if self.clip:
                try:
                    self.stop("shutdown")
                except RuntimeError as exc:
                    logging.error("Could not save recording: %s", exc)
        finally:
            with self.condition:
                self.running = False
                self.condition.notify_all()
            self.thread.join(timeout=3)
            self.pool.shutdown(wait=True)


PAGE = b"""<!doctype html>
<html lang="en"><head><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robokeeper stereo recorder</title>
<style>
body{background:#111;color:#eee;font:16px system-ui,sans-serif;max-width:1300px;margin:2rem auto;padding:0 1rem}
h1{font-size:1.4rem}img{display:block;width:100%;height:auto;background:#222}
button{font:inherit;padding:.65rem 1.2rem;margin:1rem .5rem 0 0;cursor:pointer}
#start{background:#c33;color:white;border:0}#stop{background:#eee;color:#111;border:0}
button:disabled{opacity:.5;cursor:default}#status{min-height:1.5em}
#status.rec{color:#f66;font-weight:bold}
</style></head><body>
<h1>Robokeeper stereo recorder</h1>
<img src="/stream.mjpg" alt="Live stereo preview, left eye then right eye">
<label for="name">Video name</label> <input id="name" maxlength="60" placeholder="e.g. kick_11m">
<button id="start" disabled>Start recording</button><button id="stop" disabled>Stop recording</button>
<p id="status" role="status">Connecting...</p><p id="mode"></p><p id="details"></p>
<script>
const $ = id => document.getElementById(id);
const start = $('start'), stop = $('stop'), status = $('status');
async function refresh() {
  try {
    const response = await fetch('/api/status', {cache:'no-store'});
    if (!response.ok) throw new Error('Status request failed');
    const data = await response.json();
    start.disabled = data.recording || data.finalizing || !!data.error;
    stop.disabled = !data.recording;
    $('name').disabled = data.recording || data.finalizing;
    const m = data.camera_mode;
    $('mode').textContent = `Camera: ${m.packed_size.join('x')} (two ${m.eye_size.join('x')} eyes) at ${m.requested_fps} fps; ` +
      (m.requested_exposure_us ? `exposure ${m.requested_exposure_us} us, gain ${m.requested_gain}.` : 'auto exposure.');
    const fps = data.live_fps ? ` Camera ${data.live_fps.toFixed(1)} fps.` : '';
    status.className = data.recording ? 'rec' : '';
    status.textContent = data.error || data.finalizing_error || (data.recording
      ? `REC ${data.current_file}: ${data.recording_s.toFixed(1)} s, ${data.frames_recorded} frames, ${data.skipped} skipped.${fps}`
      : data.finalizing ? `Saving ${data.finalizing_file} and making MP4...` : `Ready to record.${fps}`);
    const s = data.last_saved;
    $('details').textContent = s
      ? `Last saved on Pi: ${s.video}` + (s.preview_mp4 ? ` + ${s.preview_mp4}` : ' (MP4 failed)') +
        ` - ${s.frame_count} frames over ${s.sensor_span_s.toFixed(2)} s, ${s.skipped_by_recorder} skipped.`
      : '';
  } catch (error) {
    start.disabled = true; stop.disabled = true;
    status.textContent = error.message;
  }
}
async function control(action) {
  start.disabled = true; stop.disabled = true;
  try {
    const response = await fetch(`/api/${action}`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body:JSON.stringify(action === 'start' ? {name:$('name').value} : {})
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Request failed');
    await refresh();
  } catch (error) {
    status.textContent = error.message;
    setTimeout(refresh, 2000);
  }
}
start.onclick = () => control('start');
stop.onclick = () => control('stop');
refresh(); setInterval(refresh, 1000);
</script></body></html>"""


class StereoHandler(Handler):
    def log_request(self, code="-", size="-"):
        # The page polls status every second; only log errors and controls.
        if self.path == "/api/status" and code == 200:
            return
        super().log_request(code, size)

    def do_GET(self):
        if self.path == "/":
            self.send_bytes(200, "text/html; charset=utf-8", PAGE)
        else:
            super().do_GET()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", default="stereo",
                        help="Clip name for --duration mode (the browser page asks for one)")
    parser.add_argument("--output-dir", type=Path, default=Path("recordings/stereo"))
    parser.add_argument("--duration", type=float,
                        help="Record one clip of this many seconds without the browser, then exit")
    parser.add_argument("--countdown", type=float, default=0,
                        help="With --duration: seconds of preview before recording starts")
    parser.add_argument("--camera-num", type=int, default=0)
    parser.add_argument("--tuning-file", help="Optional Picamera2 tuning JSON filename/path")
    parser.add_argument("--exposure-us", type=int, help="Lock exposure to reduce motion blur")
    parser.add_argument("--gain", type=float, default=1.0, help="Analogue gain with --exposure-us")
    parser.add_argument("--eye-width", type=int, default=1280)
    parser.add_argument("--eye-height", type=int, default=800)
    parser.add_argument("--fps", type=float, default=60)
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100")
    parser.add_argument("--encoder-threads", type=int, default=3)
    parser.add_argument("--host", default="0.0.0.0", help="Web server bind address")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preview-fps", type=float, default=10)
    parser.add_argument("--keep-mjpg", action="store_true", help="Keep the raw JPEG stream")
    args = parser.parse_args()
    for name in ("eye_width", "eye_height", "fps", "preview_fps", "encoder_threads",
                 "duration", "exposure_us", "gain"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if not 1 <= args.quality <= 100 or args.countdown < 0:
        parser.error("--quality must be 1-100 and --countdown non-negative")
    if args.countdown and not args.duration:
        parser.error("--countdown requires --duration")
    if args.gain != 1 and args.exposure_us is None:
        parser.error("--gain requires --exposure-us")
    if args.exposure_us is not None and args.exposure_us > 1e6/args.fps:
        parser.error("--exposure-us must fit within the requested frame period")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("ffmpeg and ffprobe are required (sudo apt install ffmpeg)")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    from robokeeper.stereo import LatestPicameraStereo

    camera = LatestPicameraStereo(args.eye_width * 2, args.eye_height, args.fps,
                                  args.camera_num, args.tuning_file,
                                  args.exposure_us, args.gain)
    try:
        recorder = StereoRecorder(camera, args)
        try:
            handler = type("Handler", (StereoHandler,), {"recorder": recorder})
            server = ThreadingHTTPServer((args.host, args.port), handler)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            logging.info("Preview: http://<pi-ip>:%d/", args.port)
            try:
                if args.duration:
                    time.sleep(max(args.countdown, 1.0))
                    recorder.start(args.name)
                    time.sleep(args.duration)
                    recorder.stop("duration")
                else:
                    logging.info("Name and start recordings from the browser; Ctrl+C quits")
                    while recorder.running:
                        time.sleep(0.5)
            except KeyboardInterrupt:
                pass
            finally:
                server.shutdown()
                server.server_close()
        finally:
            recorder.close()
    finally:
        camera.close()
    if recorder.error:
        raise RuntimeError(recorder.error)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, ImportError, subprocess.CalledProcessError) as exc:
        logging.error("%s", getattr(exc, "stderr", None) or exc)
        sys.exit(1)
