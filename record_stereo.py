#!/usr/bin/env python3
"""Record packed stereo clips from the Camarray CSI HAT for offline replay.

Every captured pair is JPEG-encoded on worker threads and remuxed (no
re-encode) into an MJPEG AVI that ``track_stereo.py --video`` can replay. A
same-name CSV holds sensor timestamps so replay uses real capture times.
"""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import csv
from datetime import datetime
from http.server import ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
import subprocess
import sys
import threading
import time

import cv2
import numpy as np

from camera_stream import Handler
from record_camera import safe_name
from robokeeper.stereo import LatestPicameraStereo, PreviewBuffer


def encode(frame, quality):
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return jpeg.tobytes()


def remux(mjpg_path, avi_path, fps, frame_count):
    """Wrap the concatenated JPEGs in AVI without re-encoding, then verify."""
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "mjpeg",
         "-framerate", f"{fps:g}", "-i", str(mjpg_path), "-c:v", "copy", str(avi_path)],
        check=True, capture_output=True, text=True)
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(avi_path)],
        check=True, capture_output=True, text=True)
    if int(probe.stdout.strip()) != frame_count:
        raise RuntimeError(f"AVI has {probe.stdout.strip()} frames, expected {frame_count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", nargs="?", default="stereo", help="Clip name prefix")
    parser.add_argument("--output-dir", type=Path, default=Path("recordings/stereo"))
    parser.add_argument("--duration", type=float, help="Stop after this many seconds")
    parser.add_argument("--countdown", type=float, default=0,
                        help="Seconds of preview before recording starts (minimum 1)")
    parser.add_argument("--camera-num", type=int, default=0)
    parser.add_argument("--tuning-file", help="Optional Picamera2 tuning JSON filename/path")
    parser.add_argument("--exposure-us", type=int, help="Lock exposure to reduce motion blur")
    parser.add_argument("--gain", type=float, default=1.0, help="Analogue gain with --exposure-us")
    parser.add_argument("--eye-width", type=int, default=1280)
    parser.add_argument("--eye-height", type=int, default=800)
    parser.add_argument("--fps", type=float, default=60)
    parser.add_argument("--quality", type=int, default=95, help="JPEG quality 1-100")
    parser.add_argument("--encoder-threads", type=int, default=2)
    parser.add_argument("--serve", action="store_true", help="Serve a browser preview")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preview-fps", type=float, default=5)
    parser.add_argument("--keep-mjpg", action="store_true", help="Keep the raw JPEG stream")
    args = parser.parse_args()
    for name in ("eye_width", "eye_height", "fps", "preview_fps", "encoder_threads",
                 "duration", "exposure_us", "gain"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if not 1 <= args.quality <= 100 or args.countdown < 0:
        parser.error("--quality must be 1-100 and --countdown non-negative")
    if args.gain != 1 and args.exposure_us is None:
        parser.error("--gain requires --exposure-us")
    if args.exposure_us is not None and args.exposure_us > 1e6/args.fps:
        parser.error("--exposure-us must fit within the requested frame period")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{safe_name(args.name)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mjpg_path = args.output_dir / f"{stem}.mjpg"
    avi_path = args.output_dir / f"{stem}.avi"
    csv_path = args.output_dir / f"{stem}.csv"
    json_path = args.output_dir / f"{stem}.json"

    frame_count = skipped = 0
    first_sensor = None
    sensor_times, exposures, gains = [], [], []
    stop_reason = "ctrl+c"
    with ExitStack() as stack:
        camera = LatestPicameraStereo(args.eye_width * 2, args.eye_height, args.fps,
                                      args.camera_num, args.tuning_file,
                                      args.exposure_us, args.gain)
        stack.callback(camera.close)
        preview = None
        if args.serve:
            preview = PreviewBuffer()
            handler = type("StereoHandler", (Handler,), {"camera": preview})
            server = ThreadingHTTPServer((args.host, args.port), handler)
            stack.callback(server.server_close)
            stack.callback(server.shutdown)
            stack.callback(preview.close)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            logging.info("Preview: http://<pi-ip>:%d/", args.port)
        pool = stack.enter_context(ThreadPoolExecutor(args.encoder_threads))
        video_file = stack.enter_context(open(mjpg_path, "xb"))
        csv_file = stack.enter_context(open(csv_path, "x", newline="", encoding="utf-8"))
        writer = csv.writer(csv_file)
        writer.writerow(("frame_index", "timestamp_s", "sensor_timestamp_ns",
                         "capture_sequence", "exposure_us", "analogue_gain"))
        # Bounded so a slow SD card shows up as skipped frames, not RAM growth.
        pending = deque()
        max_pending = args.encoder_threads * 4

        def flush(limit):
            while len(pending) > limit:
                future, row = pending.popleft()
                video_file.write(future.result())
                writer.writerow(row)

        sequence = 0
        last_preview = 0
        started = None
        # Frames in the first second are only previewed: startup stalls (encoder
        # threads, auto exposure) otherwise drop a burst of early frames.
        armed_at = time.monotonic() + max(args.countdown, 1.0)
        warmed = False
        if args.countdown:
            logging.info("Recording starts in %.1f s", args.countdown)
        try:
            while True:
                sample = camera.read(sequence)
                if sample is None:
                    raise RuntimeError(camera.error or "No new stereo frame within 2 seconds")
                previous = sequence
                sequence, frame, timestamp, _, metadata = sample
                now = time.monotonic()
                if preview and now - last_preview >= 1/args.preview_fps:
                    shown = cv2.resize(frame, (1280, round(frame.shape[0]*1280/frame.shape[1])))
                    label = "REC" if started else f"starting in {max(0, armed_at-now):.0f}s"
                    cv2.putText(shown, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, 255, 2)
                    preview.publish(shown)
                    last_preview = now
                if now < armed_at:
                    if not warmed:
                        list(pool.map(encode, [frame] * args.encoder_threads,
                                      [args.quality] * args.encoder_threads))
                        warmed = True
                    continue
                if started is None:
                    started = now
                    logging.info("Recording %s (Ctrl+C to stop)", avi_path)
                elif args.duration and now - started >= args.duration:
                    stop_reason = "duration"
                    break
                if metadata["sensor_timestamp_ns"] is None:
                    raise RuntimeError("Camera did not report SensorTimestamp")
                if frame_count:
                    skipped += max(0, sequence - previous - 1)
                if first_sensor is None:
                    first_sensor = metadata["sensor_timestamp_ns"]
                sensor_times.append(metadata["sensor_timestamp_ns"])
                exposures.append(metadata["exposure_us"])
                gains.append(metadata["analogue_gain"])
                pending.append((pool.submit(encode, frame, args.quality), (
                    frame_count, f"{(metadata['sensor_timestamp_ns']-first_sensor)/1e9:.9f}",
                    metadata["sensor_timestamp_ns"], sequence,
                    metadata["exposure_us"], metadata["analogue_gain"])))
                frame_count += 1
                flush(max_pending)
                if frame_count % (5 * round(args.fps)) == 0:
                    logging.info("%d frames | %.1f fps | skipped %d", frame_count,
                                 frame_count / (now - started), skipped)
        except KeyboardInterrupt:
            pass
        flush(0)

    if frame_count == 0:
        for path in (mjpg_path, csv_path):
            path.unlink(missing_ok=True)
        raise RuntimeError("No frames were recorded")
    gaps_ms = np.diff(sensor_times) / 1e6 if frame_count > 1 else np.array([])
    period_ms = 1000 / args.fps
    span_s = (sensor_times[-1] - sensor_times[0]) / 1e9
    summary = {
        "video": avi_path.name, "timestamps": csv_path.name,
        "packed_size": [args.eye_width * 2, args.eye_height],
        "eye_size": [args.eye_width, args.eye_height],
        "requested_fps": args.fps, "jpeg_quality": args.quality,
        "requested_exposure_us": args.exposure_us,
        "requested_gain": args.gain if args.exposure_us else None,
        "frame_count": frame_count, "sensor_span_s": span_s,
        "average_fps": (frame_count - 1) / span_s if span_s > 0 else None,
        "skipped_by_recorder": skipped,
        # Sensor gaps over 1.5 frame periods mean frames never reached the recorder.
        "sensor_gaps_over_1_5_periods": int((gaps_ms > 1.5 * period_ms).sum()),
        "max_sensor_gap_ms": float(gaps_ms.max()) if gaps_ms.size else None,
        "exposure_us_range": [min(exposures), max(exposures)],
        "analogue_gain_range": [min(gains), max(gains)],
        "stop_reason": stop_reason,
        "timestamp_note": "timestamp_s is SensorTimestamp relative to the first recorded frame.",
    }
    logging.info("Remuxing %d frames to %s", frame_count, avi_path)
    remux(mjpg_path, avi_path, args.fps, frame_count)
    if not args.keep_mjpg:
        mjpg_path.unlink()
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    logging.info("Saved %s: %d frames, %.1f fps, %d skipped, %d sensor gaps, "
                 "exposure %s us, gain %s", avi_path, frame_count,
                 summary["average_fps"] or 0, skipped, summary["sensor_gaps_over_1_5_periods"],
                 summary["exposure_us_range"], summary["analogue_gain_range"])
    if skipped or summary["sensor_gaps_over_1_5_periods"]:
        logging.warning("Frames were lost; try --encoder-threads 3 or a lower --quality")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, ImportError, subprocess.CalledProcessError) as exc:
        logging.error("%s", getattr(exc, "stderr", None) or exc)
        sys.exit(1)
