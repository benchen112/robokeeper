#!/usr/bin/env python3
"""Test the existing ball pipeline on the side-by-side Camarray OV9281 feed."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from http.server import ThreadingHTTPServer
import json
import logging
import math
from pathlib import Path
import sys
import threading
import time

import cv2
import numpy as np

from camera_stream import Handler
from robokeeper import AppearanceBallSegmenter, BallTracker, HybridBallSegmenter, MotionBallSegmenter
from robokeeper.stereo import LatestPicameraStereo, PreviewBuffer, pair_diagnostics, split_stereo
from track_ball import (LatestCamera, ReviewWriter, annotate, as_json, load_timestamps,
                        scene_diagnostics)


def make_tracker(args):
    if args.detector == "hybrid":
        segmenter = HybridBallSegmenter(
            min_radius=args.min_radius, max_radius=args.max_radius,
            auto_floor=not args.no_auto_floor,
            verify_appearance=not args.no_appearance_verifier,
            fixed_camera=not args.moving_camera,
            acquisition_min_row=getattr(args, "min_ball_row", None))
    elif args.detector == "appearance":
        segmenter = AppearanceBallSegmenter(min_radius=args.min_radius,
                                            max_radius=args.max_radius or 140)
    else:
        segmenter = MotionBallSegmenter()
    return BallTracker(segmenter, confirmation_hits=1 if args.detector == "hybrid" else 3,
                       adaptive_gate=args.detector == "hybrid",
                       max_gap_s=getattr(args, "max_gap_s", 0.2)), segmenter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--device", help="Use a packed V4L2 feed instead of the CSI HAT")
    source.add_argument("--video", help="Replay a packed stereo video")
    parser.add_argument("--camera-num", type=int, default=0, help="Picamera2 camera index")
    parser.add_argument("--tuning-file", help="Optional Picamera2 tuning JSON filename/path")
    parser.add_argument("--exposure-us", type=int, help="Lock CSI exposure to reduce motion blur")
    parser.add_argument("--gain", type=float, default=1.0, help="Analogue gain with --exposure-us")
    parser.add_argument("--eye-width", type=int, default=1280)
    parser.add_argument("--eye-height", type=int, default=800)
    parser.add_argument("--fps", type=float, default=60)
    parser.add_argument("--fourcc", choices=("MJPG", "YUYV", "auto"), default="auto")
    parser.add_argument("--swap-eyes", action="store_true", help="Label the right half as left")
    parser.add_argument("--detector", choices=("hybrid", "appearance", "motion"), default="hybrid")
    parser.add_argument("--min-radius", type=int, default=3)
    parser.add_argument("--max-radius", type=int)
    parser.add_argument("--no-auto-floor", action="store_true")
    parser.add_argument("--no-appearance-verifier", action="store_true")
    parser.add_argument("--min-ball-row", type=int,
                        help="Ignore new balls centered above this pixel row (e.g. heads)")
    parser.add_argument("--max-gap-s", type=float, default=0.2,
                        help="Drop a track after a gap between processed frames this long")
    parser.add_argument("--moving-camera", action="store_true",
                        help="Compensate camera motion (slower); default assumes a rigid mount")
    parser.add_argument("--show-candidates", action="store_true")
    parser.add_argument("--opencv-threads", type=int, default=2)
    parser.add_argument("--display", action="store_true", help="Show a local OpenCV window")
    parser.add_argument("--serve", action="store_true", help="Serve annotated preview in a browser")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preview-fps", type=float, default=10)
    parser.add_argument("--jsonl", help="Pair results and timing; '-' writes to stdout")
    parser.add_argument("--timestamps", help="Replay timestamp CSV; auto-loads same-name sidecar")
    parser.add_argument("--review-video", help="Save a full-resolution annotated MP4 (requires --video)")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--duration", type=float, help="Limit live test wall time in seconds")
    args = parser.parse_args()
    for name in ("eye_width", "eye_height", "fps", "preview_fps", "opencv_threads",
                 "max_frames", "duration", "exposure_us", "gain"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0):
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.camera_num < 0 or not 1 <= args.port <= 65535:
        parser.error("Invalid camera index or port")
    if args.timestamps and not args.video:
        parser.error("--timestamps requires --video")
    if (args.device or args.video) and (args.tuning_file or args.exposure_us is not None or args.gain != 1):
        parser.error("Tuning, exposure and gain options require the Picamera2 CSI input")
    if args.gain != 1 and args.exposure_us is None:
        parser.error("--gain requires --exposure-us")
    if args.exposure_us is not None and args.exposure_us > 1e6/args.fps:
        parser.error("--exposure-us must fit within the requested frame period")
    if args.review_video and not args.video:
        parser.error("--review-video requires --video")
    if args.review_video and Path(args.review_video).suffix.lower() != ".mp4":
        parser.error("--review-video must end in .mp4")
    if args.video and args.duration:
        parser.error("--duration is for live tests; use --max-frames for replay")
    if args.jsonl and args.jsonl != "-" and args.video:
        inputs = [args.video, args.timestamps or str(Path(args.video).with_suffix(".csv"))]
        if Path(args.jsonl).resolve() in [Path(p).resolve() for p in inputs]:
            parser.error("JSONL output must differ from input files")
    if args.review_video and Path(args.review_video).resolve() == Path(args.video).resolve():
        parser.error("Review video must differ from the input video")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cv2.setNumThreads(args.opencv_threads)
    trackers = [make_tracker(args), make_tracker(args)]
    trails = [deque(maxlen=30), deque(maxlen=30)]
    timings = deque(maxlen=1000)
    counts = [0, 0]
    both_count = 0
    processed = skipped = sequence = 0
    started_run = last_report = time.monotonic()
    last_preview = 0
    first_timestamp = last_timestamp = None
    camera = video = preview = review = None

    def track_eye(tracker, view, timestamp):
        # OpenCV and large numpy ops release the GIL, so the eyes overlap in threads.
        start = time.monotonic()
        result = tracker.process(view, timestamp)
        return result, start, time.monotonic()

    try:
        with ExitStack() as stack:
            eye_pool = stack.enter_context(ThreadPoolExecutor(2))
            if args.video:
                video = cv2.VideoCapture(args.video)
                stack.callback(video.release)
                if not video.isOpened():
                    raise RuntimeError(f"Cannot open video {args.video}")
                fps = video.get(cv2.CAP_PROP_FPS)
                if not math.isfinite(fps) or fps <= 0:
                    raise RuntimeError("Replay video has no valid FPS")
                path = Path(args.timestamps) if args.timestamps else Path(args.video).with_suffix(".csv")
                timestamps = (load_timestamps(path, round(video.get(cv2.CAP_PROP_FRAME_COUNT)))
                              if path.exists() or args.timestamps else None)
                if args.review_video:
                    review = ReviewWriter(args.review_video, fps, (args.eye_width * 2, args.eye_height))
            else:
                camera = (LatestCamera(args.device, args.eye_width * 2, args.eye_height,
                                       args.fps, args.fourcc) if args.device else
                          LatestPicameraStereo(args.eye_width * 2, args.eye_height,
                                               args.fps, args.camera_num, args.tuning_file,
                                               args.exposure_us, args.gain))
                stack.callback(camera.close)
            output = None
            if args.jsonl == "-":
                output = sys.stdout
            elif args.jsonl:
                Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
                output = stack.enter_context(open(args.jsonl, "x", encoding="utf-8"))
            if args.serve:
                preview = PreviewBuffer()
                handler = type("StereoHandler", (Handler,), {"camera": preview})
                server = ThreadingHTTPServer((args.host, args.port), handler)
                stack.callback(server.server_close)
                stack.callback(server.shutdown)
                stack.callback(preview.close)
                threading.Thread(target=server.serve_forever, daemon=True).start()
                logging.info("Stereo preview: http://<pi-ip>:%d/ (Ctrl+C to stop)", args.port)
            while args.max_frames is None or processed < args.max_frames:
                if args.duration and time.monotonic() - started_run >= args.duration:
                    break
                metadata = {}
                received = None
                if video is not None:
                    ok, frame = video.read()
                    if not ok:
                        if timestamps is not None and processed != len(timestamps):
                            raise RuntimeError("Decoded video length differs from timestamp CSV")
                        break
                    timestamp = timestamps[processed] if timestamps is not None else processed / fps
                    sequence = processed + 1
                    metadata["timestamp_source"] = "recorded_arrival" if timestamps is not None else "video_fps"
                else:
                    sample = camera.read(sequence)
                    if sample is None:
                        raise RuntimeError(getattr(camera, "error", None) or
                                           "No new stereo frame within 2 seconds")
                    previous = sequence
                    if args.device:
                        sequence, frame, timestamp = sample
                        received = timestamp
                        metadata["timestamp_source"] = "receive_monotonic"
                    else:
                        sequence, frame, timestamp, received, metadata = sample
                    skipped += max(0, sequence - previous - 1)
                if last_timestamp is not None and timestamp <= last_timestamp:
                    raise RuntimeError("Stereo frame timestamps must strictly increase")
                if first_timestamp is None:
                    first_timestamp = timestamp
                last_timestamp = timestamp
                views = split_stereo(frame, (args.eye_width, args.eye_height), args.swap_eyes)
                if processed == 0:
                    logging.info("Input %dx%d; each eye %dx%d; timestamp source: %s",
                                 frame.shape[1], frame.shape[0], args.eye_width,
                                 args.eye_height, metadata["timestamp_source"])
                pair_start = time.monotonic()
                results, rows = [], []
                outcomes = list(eye_pool.map(track_eye, [t for t, _ in trackers], views,
                                             (timestamp, timestamp)))
                for i, ((_, segmenter), (result, start, end)) in enumerate(zip(trackers, outcomes)):
                    results.append(result)
                    rows.append(json.loads(as_json(
                        result, (end-start)*1000,
                        (end-received)*1000 if received is not None else None,
                        ("left", "right")[i], scene_diagnostics(segmenter))))
                    counts[i] += int(result.state == "confirmed" and result.observed)
                    if result.observed and result.filtered_center is not None:
                        trails[i].append(tuple(round(v) for v in result.filtered_center))
                    elif result.state == "absent":
                        trails[i].clear()
                end = time.monotonic()
                pair_ms = (end-pair_start)*1000
                timings.append(pair_ms)
                diagnostics = pair_diagnostics(*results)
                both_count += int(diagnostics["both_confirmed_observed"])
                if output:
                    print(json.dumps({
                        "frame_index": processed, "capture_sequence": sequence,
                        "timestamp_s": timestamp, "elapsed_s": timestamp-first_timestamp,
                        "packed_size": [frame.shape[1], frame.shape[0]],
                        "eye_size": [args.eye_width, args.eye_height],
                        "swapped_eyes": args.swap_eyes, "left": rows[0], "right": rows[1],
                        "pair_processing_ms": round(pair_ms, 2),
                        "pair_frame_age_ms": round((end-received)*1000, 2) if received is not None else None,
                        "skipped_capture_frames": skipped,
                        "eye_mean_luma": [round(float(view.mean()), 1) for view in views],
                        **metadata, **diagnostics,
                    }, separators=(",", ":")), file=output, flush=True)
                processed += 1
                now = time.monotonic()
                if args.display or review or (preview and now-last_preview >= 1/args.preview_fps):
                    rendered = []
                    for i, view in enumerate(views):
                        image = view.copy() if view.ndim == 3 else cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
                        annotate(image, results[i], frame_index=processed-1, trail=trails[i],
                                 show_candidates=args.show_candidates,
                                 diagnostics=scene_diagnostics(trackers[i][1]))
                        cv2.putText(image, ("LEFT", "RIGHT")[i], (12, 108),
                                    cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
                        rendered.append(image)
                    shown = np.hstack(rendered)
                    cv2.putText(shown, f"Pair {pair_ms:.1f} ms | skipped {skipped}",
                                (12, 135), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
                    if review:
                        review.write(shown)
                    # Preview downsampling only; trackers always receive full-resolution eyes.
                    shown = cv2.resize(shown, (1280, max(1, round(shown.shape[0]*1280/shown.shape[1]))))
                    if preview:
                        preview.publish(shown)
                        last_preview = now
                    if args.display:
                        cv2.imshow("Robokeeper stereo tracking", shown)
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break
                if now-last_report >= 5:
                    logging.info("%d pairs | %.1f pairs/s | p50/p95 %.1f/%.1f ms | "
                                 "L/R %s/%s | skipped %d", processed,
                                 processed/(now-started_run), *np.percentile(timings, [50, 95]),
                                 results[0].state, results[1].state, skipped)
                    last_report = now
    except KeyboardInterrupt:
        pass
    finally:
        if args.display:
            cv2.destroyAllWindows()
        if review:
            # Free tracker state first so it and ffmpeg are not both resident.
            trackers.clear()
            review.close()
            logging.info("Saved review video: %s", Path(args.review_video).resolve())
        if processed:
            logging.info("Finished: %d pairs; L/R confirmed observations %d/%d; "
                         "both %d; skipped %d; last %d pair processing p50/p95 %.1f/%.1f ms",
                         processed, *counts, both_count, skipped, len(timings),
                         *np.percentile(timings, [50, 95]))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        logging.error("%s", exc)
        sys.exit(1)
