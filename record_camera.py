#!/usr/bin/env python3
"""Preview and record a V4L2/USB camera from a browser on a Raspberry Pi."""

import argparse
from collections import deque
import csv
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import re
import select
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class JpegFrames:
    """Split FFmpeg's concatenated JPEG byte stream across arbitrary pipe reads."""

    def __init__(self):
        self.buffer = bytearray()

    def feed(self, chunk):
        self.buffer.extend(chunk)
        while True:
            start = self.buffer.find(b"\xff\xd8")
            if start < 0:
                self.buffer[:] = self.buffer[-1:] if self.buffer.endswith(b"\xff") else b""
                return
            if start:
                del self.buffer[:start]
            end = self.buffer.find(b"\xff\xd9", 2)
            if end < 0:
                if len(self.buffer) > 16 * 1024 * 1024:
                    raise RuntimeError("Camera JPEG frame exceeded 16 MiB")
                return
            jpeg = bytes(self.buffer[:end + 2])
            del self.buffer[:end + 2]
            yield jpeg


class Clip:
    def __init__(self, output_dir, name, settings, keep_raw=False):
        self.video_path = output_dir / f"{name}.mjpg"
        self.timestamps_path = output_dir / f"{name}.csv"
        self.metadata_path = output_dir / f"{name}.json"
        self.video_file = self.video_path.open("wb")
        try:
            self.timestamps_file = self.timestamps_path.open("w", newline="", encoding="utf-8")
        except OSError:
            self.video_file.close()
            self.video_path.unlink(missing_ok=True)
            raise
        self.csv = csv.writer(self.timestamps_file)
        self.csv.writerow(("frame_index", "timestamp_s", "read_monotonic_ns"))
        self.settings = settings
        self.keep_raw = keep_raw
        source_rate = Fraction(str(settings.get("camera_fps", settings["video_fps"])))
        record_rate = Fraction(str(settings["video_fps"]))
        self.keep_numerator = record_rate.numerator * source_rate.denominator
        self.keep_denominator = source_rate.numerator * record_rate.denominator
        self.source_frame_count = 0
        self.first_source_ns = None
        self.last_source_ns = None
        self.frame_count = 0
        self.first_ns = None
        self.last_ns = None
        self.first_utc = None

    def write(self, jpeg, read_ns):
        source_index = self.source_frame_count
        self.source_frame_count += 1
        if self.first_source_ns is None:
            self.first_source_ns = read_ns
        self.last_source_ns = read_ns
        # Pick source indices nearest the desired uniform output cadence.
        # At 100 -> 60 fps, this keeps exactly 60 of every 100 source frames.
        if source_index * self.keep_numerator < self.frame_count * self.keep_denominator:
            return False
        if self.first_ns is None:
            self.first_ns = read_ns
            self.first_utc = datetime.now(timezone.utc).isoformat()
        self.video_file.write(jpeg)
        self.csv.writerow((self.frame_count, f"{(read_ns - self.first_ns) / 1e9:.9f}", read_ns))
        self.frame_count += 1
        self.last_ns = read_ns
        return True

    def close(self):
        self.video_file.close()
        self.timestamps_file.close()
        duration = (self.last_ns - self.first_ns) / 1e9 if self.frame_count > 1 else 0.0
        source_duration = (
            (self.last_source_ns - self.first_source_ns) / 1e9
            if self.source_frame_count > 1 else 0.0
        )
        metadata = {
            **self.settings,
            "video": self.video_path.name,
            "timestamps": self.timestamps_path.name,
            "frame_count": self.frame_count,
            "source_frame_count": self.source_frame_count,
            "frames_dropped_for_record_fps": self.source_frame_count - self.frame_count,
            "first_frame_utc": self.first_utc,
            "captured_span_s": duration,
            "source_span_s": source_duration,
            "average_capture_fps": ((self.source_frame_count - 1) / source_duration
                                    if source_duration > 0 else None),
            "average_record_fps": (self.frame_count - 1) / duration if duration > 0 else None,
            "timestamp_note": "Times are recorded when complete JPEG frames arrive from FFmpeg's pipe; they are not exposure times.",
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    def convert_to_mp4(self):
        mp4_path = self.video_path.with_suffix(".mp4")
        temporary_path = self.video_path.with_suffix(".part.mp4")
        # The raw MJPEG stream has no per-frame timestamps. FFmpeg initially
        # assigns the selected FPS. Keep exact CFR when camera delivery is near
        # the requested rate, and preserve real duration if capture fell behind.
        capture_span = (self.last_ns - self.first_ns) / 1e9 if self.frame_count > 1 else 0.0
        expected_span = (self.frame_count - 1) / self.settings["video_fps"] if self.frame_count > 1 else 0.0
        use_exact_fps = (self.frame_count >= 10 and expected_span > 0
                         and abs(capture_span - expected_span) <= expected_span * 0.05)
        playback_scale = (1.0 if use_exact_fps else
                          capture_span / expected_span if capture_span > 0 and expected_span > 0 else 1.0)
        playback_timing = ("Constant at selected record FPS" if use_exact_fps else
                           "Scaled to measured first-to-last-frame capture span")
        video_filter = (
            f"settb=AVTB,setpts=(PTS-STARTPTS)*{playback_scale:.12g},"
            "pad=ceil(iw/2)*2:ceil(ih/2)*2"
        )
        try:
            subprocess.run(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "mjpeg",
                 "-framerate", str(self.settings["video_fps"]), "-i", str(self.video_path),
                 "-an", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                 "-vf", video_filter, "-fps_mode", "passthrough", "-pix_fmt", "yuv420p",
                 "-movflags", "+faststart", str(temporary_path)],
                check=True, capture_output=True, text=True,
            )
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                 "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0",
                 str(temporary_path)],
                check=True, capture_output=True, text=True,
            )
            if int(probe.stdout.strip()) != self.frame_count:
                raise RuntimeError("MP4 frame count does not match the selected camera frames")
            temporary_path.replace(mp4_path)
            raw_deleted = False
            if not self.keep_raw:
                try:
                    self.video_path.unlink()
                    raw_deleted = True
                except OSError as exc:
                    print(f"Could not remove raw MJPEG {self.video_path}: {exc}", flush=True)
            metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            metadata["video"] = mp4_path.name
            metadata["raw_video"] = None if raw_deleted else self.video_path.name
            metadata["raw_deleted_after_conversion"] = raw_deleted
            metadata["video_codec"] = "H.264"
            metadata["playback_timing"] = playback_timing
            self.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
            print(f"Saved {mp4_path} ({self.frame_count} of {self.source_frame_count} frames); "
                  f"timestamps: {self.timestamps_path}", flush=True)
            return mp4_path.name
        except (OSError, subprocess.CalledProcessError, ValueError, RuntimeError) as exc:
            temporary_path.unlink(missing_ok=True)
            detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
            raise RuntimeError(f"MP4 conversion failed; raw MJPEG is at {self.video_path}: {detail}") from exc


def safe_name(value):
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", value.strip()).strip("_-")[:60]
    return name or "shot"


def repair_video_timing(video_path):
    """Retimestamp a clip made by the older fixed-FPS recorder without re-encoding."""
    video_path = video_path.resolve()
    metadata_path = video_path.with_suffix(".json")
    if video_path.suffix.lower() != ".mp4" or not video_path.is_file() or not metadata_path.is_file():
        raise ValueError("Give an existing recorder MP4 with its matching JSON sidecar")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("playback_timing"):
        raise ValueError("This recording already has corrected playback timing")
    count = metadata.get("frame_count", 0)
    span = metadata.get("captured_span_s", 0)
    fps = metadata.get("video_fps", 0)
    if count < 2 or span <= 0 or fps <= 0:
        raise ValueError("Metadata lacks valid frame count, capture span, or video FPS")
    scale = span * fps / (count - 1)
    fixed_path = video_path.with_name(f"{video_path.stem}_timing_fixed.mp4")
    fixed_metadata_path = fixed_path.with_suffix(".json")
    original_csv = video_path.with_suffix(".csv")
    fixed_csv = fixed_path.with_suffix(".csv")
    if fixed_path.exists() or fixed_metadata_path.exists() or fixed_csv.exists():
        raise ValueError(f"Output already exists: {fixed_path}")
    temporary_path = fixed_path.with_suffix(".part.mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-itsscale",
             f"{scale:.12g}", "-i", str(video_path), "-map", "0:v:0", "-an",
             "-c:v", "copy", "-movflags", "+faststart", str(temporary_path)],
            check=True, capture_output=True, text=True,
        )
        temporary_path.replace(fixed_path)
        if original_csv.exists():
            shutil.copy2(original_csv, fixed_csv)
            metadata["timestamps"] = fixed_csv.name
        metadata["video"] = fixed_path.name
        metadata["playback_timing"] = "Scaled to measured first-to-last-frame capture span"
        metadata["timing_repaired_from"] = video_path.name
        fixed_metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return fixed_path
    except (OSError, subprocess.CalledProcessError) as exc:
        temporary_path.unlink(missing_ok=True)
        detail = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise RuntimeError(f"Could not repair playback timing: {detail}") from exc


class Recorder:
    """Route one compressed camera stream to preview and optional clip storage."""

    def __init__(self, args, source_command=None):
        self.default_record_fps = args.record_fps
        self.keep_raw = args.keep_raw
        self.settings = {
            "device": args.device,
            "requested_width": args.width,
            "requested_height": args.height,
            "requested_fps": args.fps,
            "requested_fourcc": "MJPG",
            "width": args.width,
            "height": args.height,
            "camera_fps": args.fps,
            "camera_fourcc": "MJPG",
        }
        print(f"Requesting {args.width}x{args.height} MJPG at {args.fps:g} fps", flush=True)
        self.output_dir = args.output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.preview_interval = 1 / args.preview_fps
        self.condition = threading.Condition()
        self.running = True
        self.error = None
        self.clip = None
        self.finalizing = False
        self.finalizing_file = None
        self.finalizing_error = None
        self.last_saved = None
        self.last_capture_fps = None
        self.last_frame_count = 0
        self.last_source_frame_count = 0
        self.last_capture_span_s = None
        self.last_record_fps = None
        self.jpeg = None
        self.jpeg_sequence = 0
        self.last_preview_s = 0.0
        self.recent_frame_ns = deque(maxlen=101)
        self.stderr_lines = []
        command = source_command or [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin",
            "-f", "v4l2", "-input_format", "mjpeg",
            "-video_size", f"{args.width}x{args.height}", "-framerate", str(args.fps),
            "-i", args.device, "-map", "0:v:0", "-c:v", "copy",
            "-bsf:v", "mjpeg2jpeg", "-flush_packets", "1", "-f", "mjpeg", "pipe:1",
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, bufsize=0,
        )
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.stderr_thread.start()
        self.capture_thread.start()

    def _read_stderr(self):
        for line in self.process.stderr:
            self.stderr_lines.append(line.decode("utf-8", "replace").strip())
            self.stderr_lines = self.stderr_lines[-10:]

    def _capture_loop(self):
        frames = JpegFrames()
        try:
            while self.running:
                ready, _, _ = select.select([self.process.stdout], [], [], 5)
                if not ready:
                    raise RuntimeError("No camera frames arrived for 5 seconds")
                chunk = os.read(self.process.stdout.fileno(), 65536)
                if not chunk:
                    if self.running:
                        raise RuntimeError("Camera stream ended: " +
                                           ("; ".join(self.stderr_lines[-3:]) or "FFmpeg exited"))
                    break
                for jpeg in frames.feed(chunk):
                    read_ns = time.monotonic_ns()
                    with self.condition:
                        self.recent_frame_ns.append(read_ns)
                        if self.clip:
                            self.clip.write(jpeg, read_ns)
                        now_s = read_ns / 1e9
                        if now_s - self.last_preview_s >= self.preview_interval:
                            self.jpeg = jpeg
                            self.jpeg_sequence += 1
                            self.last_preview_s = now_s
                            self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.running = False
                self.condition.notify_all()
            print(f"Recorder error: {exc}", flush=True)

    def start(self, name, record_fps=None):
        with self.condition:
            if not self.running or self.error:
                raise RuntimeError(self.error or "Camera is stopped")
            if self.clip:
                raise ValueError("Already recording")
            if self.finalizing:
                raise ValueError("Wait for the previous MP4 to finish")
            if record_fps is None:
                record_fps = self.default_record_fps
            try:
                record_fps = float(record_fps)
            except (TypeError, ValueError) as exc:
                raise ValueError("Recording FPS must be a number") from exc
            if not math.isfinite(record_fps) or not 0 < record_fps <= self.settings["camera_fps"]:
                raise ValueError(f"Recording FPS must be above 0 and no higher than "
                                 f"the {self.settings['camera_fps']:g} fps camera rate")
            name = f"{safe_name(name)}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            clip_settings = {**self.settings, "video_fps": record_fps, "record_fps": record_fps}
            self.clip = Clip(self.output_dir, name, clip_settings, keep_raw=self.keep_raw)
            self.finalizing_error = None
            return self.status_locked()

    def stop(self):
        with self.condition:
            if not self.clip:
                raise ValueError("No recording is in progress")
            clip = self.clip
            self.clip = None
            clip.close()
            span_s = (clip.last_ns - clip.first_ns) / 1e9 if clip.frame_count > 1 else 0.0
            self.last_capture_fps = (clip.frame_count - 1) / span_s if span_s > 0 else None
            self.last_frame_count = clip.frame_count
            self.last_source_frame_count = clip.source_frame_count
            self.last_capture_span_s = span_s
            self.last_record_fps = clip.settings["record_fps"]
            if clip.frame_count == 0:
                clip.video_path.unlink(missing_ok=True)
                clip.timestamps_path.unlink(missing_ok=True)
                clip.metadata_path.unlink(missing_ok=True)
                raise ValueError("No frames were captured; please try again")
            self.finalizing = True
            self.finalizing_file = clip.video_path.with_suffix(".mp4").name
        try:
            saved = clip.convert_to_mp4()
        except RuntimeError as exc:
            with self.condition:
                self.finalizing_error = str(exc)
            raise
        else:
            with self.condition:
                self.last_saved = saved
                return self.status_locked()
        finally:
            with self.condition:
                self.finalizing = False
                self.finalizing_file = None

    def status_locked(self):
        clip_span = (
            (self.clip.last_ns - self.clip.first_ns) / 1e9
            if self.clip and self.clip.frame_count > 1 else 0.0
        )
        recent = self.recent_frame_ns
        live_span = (recent[-1] - recent[0]) / 1e9 if len(recent) > 1 else 0.0
        return {
            "recording": self.clip is not None,
            "current_file": self.clip.video_path.with_suffix(".mp4").name if self.clip else None,
            "frames_recorded": self.clip.frame_count if self.clip else 0,
            "source_frames_seen": self.clip.source_frame_count if self.clip else 0,
            "capture_fps": (self.clip.frame_count - 1) / clip_span if clip_span > 0 else None,
            "live_fps": (len(recent) - 1) / live_span if len(recent) >= 10 and live_span >= 0.1 else None,
            "last_capture_fps": self.last_capture_fps,
            "last_frame_count": self.last_frame_count,
            "last_source_frame_count": self.last_source_frame_count,
            "last_capture_span_s": self.last_capture_span_s,
            "default_record_fps": self.default_record_fps,
            "current_record_fps": self.clip.settings["record_fps"] if self.clip else None,
            "last_record_fps": self.last_record_fps,
            "finalizing": self.finalizing,
            "finalizing_file": self.finalizing_file,
            "finalizing_error": self.finalizing_error,
            "last_saved": self.last_saved,
            "camera_mode": self.settings,
            "error": self.error,
        }

    def status(self):
        with self.condition:
            return self.status_locked()

    def next_preview(self, previous):
        with self.condition:
            self.condition.wait_for(
                lambda: self.jpeg_sequence != previous or not self.running, timeout=5
            )
            return self.jpeg_sequence, self.jpeg, self.running

    def close(self):
        with self.condition:
            self.running = False
            self.condition.notify_all()
        if self.process.poll() is None:
            self.process.terminate()
        self.capture_thread.join(timeout=2)
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait()
        self.stderr_thread.join(timeout=2)
        try:
            if self.clip:
                try:
                    self.stop()
                except (ValueError, RuntimeError) as exc:
                    print(f"Could not finalize recording: {exc}", flush=True)
        finally:
            self.process.stdout.close()
            self.process.stderr.close()


PAGE = b"""<!doctype html>
<html lang="en"><head><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robokeeper camera recorder</title>
<style>
body{background:#111;color:#eee;font:16px system-ui,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem}
h1{font-size:1.4rem}img{display:block;width:100%;height:auto;background:#222}
button{font:inherit;padding:.65rem 1.2rem;margin:1rem .5rem 0 0;cursor:pointer}
#start{background:#c33;color:white;border:0}#stop{background:#eee;color:#111;border:0}
button:disabled{opacity:.5;cursor:default}#status{min-height:1.5em}
</style></head><body>
<h1>Robokeeper camera recorder</h1>
<img src="/stream.mjpg" alt="Live camera preview">
<label for="name">Video name</label> <input id="name" maxlength="60" placeholder="e.g. 11m_left_corner">
<label for="record-fps">Saved FPS</label> <input id="record-fps" type="number" min="0.01" step="any" value="60" style="width:5rem">
<button id="start" disabled>Start recording</button><button id="stop" disabled>Stop recording</button>
<p id="status" role="status">Connecting...</p><p id="mode"></p><p id="details"></p>
<script>
const start = document.getElementById('start');
const stop = document.getElementById('stop');
const status = document.getElementById('status');
const mode = document.getElementById('mode');
const details = document.getElementById('details');
const name = document.getElementById('name');
const recordFps = document.getElementById('record-fps');
let recordFpsInitialized = false;
async function refresh() {
  try {
    const response = await fetch('/api/status', {cache:'no-store'});
    if (!response.ok) throw new Error('Status request failed');
    const data = await response.json();
    start.disabled = !!data.recording || !!data.finalizing || !!data.error;
    stop.disabled = !data.recording;
    recordFps.disabled = !!data.recording || !!data.finalizing;
    const requested = data.camera_mode.requested_fps;
    if (!recordFpsInitialized) {
      recordFps.value = data.default_record_fps;
      recordFps.max = requested;
      recordFpsInitialized = true;
    }
    mode.textContent = `Camera: ${data.camera_mode.width}x${data.camera_mode.height} MJPG at ${requested} fps; saved FPS is selectable below.`;
    const measured = data.recording ? data.capture_fps : data.live_fps;
    const target = data.recording ? data.current_record_fps : requested;
    const fpsText = measured ? ` Measured ${measured.toFixed(1)} fps.` +
      (measured < target * 0.9 ? ' Capture is below the requested rate.' : '') : '';
    status.textContent = data.error || data.finalizing_error || (data.recording
      ? `Recording ${data.current_file} (${data.frames_recorded} saved of ${data.source_frames_seen} received frames).${fpsText}`
      : data.finalizing ? `Preparing ${data.finalizing_file}...${fpsText}` : `Ready to record.${fpsText}`);
    details.textContent = data.last_saved
      ? `Last saved on Pi: ${data.last_saved} (${data.last_frame_count} of ${data.last_source_frame_count} frames over ${data.last_capture_span_s.toFixed(2)} s).`
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
      body:JSON.stringify(action === 'start' ? {name:name.value, record_fps:Number(recordFps.value)} : {})
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


class Handler(BaseHTTPRequestHandler):
    recorder = None

    def send_bytes(self, code, content_type, body):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, code, data):
        self.send_bytes(code, "application/json; charset=utf-8", json.dumps(data).encode("utf-8"))

    def do_GET(self):
        if self.path == "/":
            self.send_bytes(200, "text/html; charset=utf-8", PAGE)
        elif self.path == "/api/status":
            self.send_json(200, self.recorder.status())
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            sequence = 0
            try:
                while True:
                    sequence, jpeg, running = self.recorder.next_preview(sequence)
                    if not running:
                        break
                    if jpeg is None:
                        continue
                    self.wfile.write(
                        b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                        + str(len(jpeg)).encode("ascii") + b"\r\n\r\n" + jpeg + b"\r\n"
                    )
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                pass
        else:
            self.send_error(404)

    def do_POST(self):
        try:
            if self.path == "/api/start":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    self.send_json(400, {"error": "Invalid Content-Length"})
                    return
                if length < 0:
                    self.send_json(400, {"error": "Invalid Content-Length"})
                    return
                if length > 1024:
                    self.send_json(413, {"error": "Name is too long"})
                    return
                try:
                    body = json.loads(self.rfile.read(length) or b"{}")
                    if not isinstance(body, dict) or not isinstance(body.get("name", ""), str):
                        raise ValueError("Name must be text")
                except (ValueError, UnicodeDecodeError):
                    self.send_json(400, {"error": "Invalid request body"})
                    return
                result = self.recorder.start(body.get("name", ""), body.get("record_fps"))
            elif self.path == "/api/stop":
                result = self.recorder.stop()
            else:
                self.send_error(404)
                return
            self.send_json(200, result)
        except ValueError as exc:
            self.send_json(409, {"error": str(exc)})
        except RuntimeError as exc:
            self.send_json(503, {"error": str(exc)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/video0", help="V4L2 camera device")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=float, default=100,
                        help="Requested camera FPS (100 or 120 at 1280x800 MJPG on this USB OV9281)")
    parser.add_argument("--record-fps", type=float, default=60,
                        help="Default saved FPS; evenly select frames from the camera stream")
    parser.add_argument("--fourcc", choices=("MJPG",), default="MJPG",
                        help="Compressed MJPG camera mode is required for direct recording")
    parser.add_argument("--output-dir", type=Path, default=Path("recordings"))
    parser.add_argument("--keep-raw", action="store_true",
                        help="Keep the large .mjpg file after the MP4 is verified")
    parser.add_argument("--repair-video", type=Path, metavar="MP4",
                        help="Correct playback speed of an older recording using its JSON sidecar")
    parser.add_argument("--preview-fps", type=float, default=10,
                        help="Maximum browser preview frame rate")
    parser.add_argument("--host", default="0.0.0.0", help="Web server bind address")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if (args.width <= 0 or args.height <= 0 or not math.isfinite(args.fps) or args.fps <= 0
            or not math.isfinite(args.record_fps) or not 0 < args.record_fps <= args.fps
            or not math.isfinite(args.preview_fps) or args.preview_fps <= 0):
        parser.error("width, height, and FPS values must be positive; record-fps cannot exceed camera fps")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("ffmpeg and ffprobe are required (sudo apt install ffmpeg)")
    if args.repair_video:
        print(f"Saved corrected video: {repair_video_timing(args.repair_video)}")
        return
    recorder = Recorder(args)
    Handler.recorder = recorder
    try:
        server = ThreadingHTTPServer((args.host, args.port), Handler)
        print(f"Open http://<pi-ip>:{args.port}/ in a browser; Ctrl+C stops the server.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
