#!/usr/bin/env python3
"""Track a ball from the OV9281 camera or a recorded video.

The JSONL output reports camera pixel coordinates. Only rows with state
"confirmed" and observed=true contain a new, temporally confirmed measurement.
"""

import argparse
import csv
from collections import deque
from pathlib import Path
import subprocess
import tempfile
import json
import logging
import math
import threading
import time

import cv2

from robokeeper import HybridBallSegmenter, AppearanceBallSegmenter, BallTracker, MotionBallSegmenter


class LatestCamera:
    """Read continuously so slow processing never builds a queue of stale frames."""

    def __init__(self, device: str, width: int, height: int, fps: int, fourcc: str):
        self.capture = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if not self.capture.isOpened():
            raise RuntimeError(f"Cannot open camera {device}")
        if fourcc != "auto":
            self.capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FPS, fps)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        logging.info(
            "Camera mode: %dx%d at %.1f fps (reported by driver)",
            self.capture.get(cv2.CAP_PROP_FRAME_WIDTH),
            self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT),
            self.capture.get(cv2.CAP_PROP_FPS),
        )
        self.condition = threading.Condition()
        self.sequence = 0
        self.latest = None
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        failures = 0
        while self.running:
            ok, frame = self.capture.read()
            if not ok:
                failures += 1
                if failures >= 10:
                    self.running = False
                    with self.condition:
                        self.condition.notify_all()
                    return
                time.sleep(0.01)
                continue
            failures = 0
            with self.condition:
                self.sequence += 1
                self.latest = (frame, time.monotonic())
                self.condition.notify_all()

    def read(self, previous_sequence: int):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence != previous_sequence or not self.running, timeout=2
            )
            if self.sequence == previous_sequence:
                return None
            return self.sequence, *self.latest

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
        self.capture.release()


def scene_diagnostics(segmenter):
    if not isinstance(segmenter, HybridBallSegmenter):
        return {}
    floor = segmenter.floor_estimator.estimate
    return {"floor_boundary_y": floor.boundary_y, "floor_confidence": round(floor.confidence, 3),
            "camera_motion_reliable": segmenter.camera_motion_reliable}


def annotate(frame, result, *, frame_index=None, trail=(), show_candidates=False,
             diagnostics=None):
    if show_candidates:
        for candidate in result.candidates:
            x, y, w, h = candidate.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 150, 255), 1)
    for first, second in zip(trail, tuple(trail)[1:]):
        cv2.line(frame, first, second, (0, 180, 0), 2)
    if result.filtered_center is not None:
        x, y = (round(v) for v in result.filtered_center)
        color = (0, 220, 0) if result.state == "confirmed" else (0, 220, 255)
        cv2.circle(frame, (x, y), max(3, round(result.radius or 0)), color, 2)
        cv2.drawMarker(frame, (x, y), color, cv2.MARKER_CROSS, 16, 2)
    if result.center is not None:
        cv2.circle(frame, tuple(round(v) for v in result.center), 3, (255, 255, 255), -1)
    if diagnostics and diagnostics.get("floor_boundary_y") is not None:
        floor_y = round(diagnostics["floor_boundary_y"] * frame.shape[0])
        for left in range(0, frame.shape[1], 24):
            cv2.line(frame, (left, floor_y), (left + 12, floor_y), (255, 160, 40), 1)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 85), (0, 0, 0), -1)
    label = f"{result.timestamp:.3f}s  {result.state}  observed={result.observed}"
    if frame_index is not None:
        label = f"frame {frame_index}  " + label
    cv2.putText(frame, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "Green: measured track | Yellow: tentative / gap prediction | White: detection",
                (12, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
    if diagnostics:
        confidence = diagnostics.get("floor_confidence", 0)
        floor_status = f"Floor-like boundary confidence: {confidence:.2f}" if diagnostics.get("floor_boundary_y") is not None else "Floor hint: unknown"
        cv2.putText(frame, floor_status, (12, 76), cv2.FONT_HERSHEY_SIMPLEX,
                    .5, (255, 160, 40), 1, cv2.LINE_AA)
    return frame


def load_timestamps(path, expected_frames):
    with open(path, newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_frames:
        raise ValueError("Timestamp CSV frame count does not match video")
    timestamps = []
    for index, row in enumerate(rows):
        value = float(row["timestamp_s"])
        if (int(row["frame_index"]) != index or not math.isfinite(value)
                or value < 0 or (timestamps and value <= timestamps[-1])):
            raise ValueError("Timestamp CSV must have sequential indices and increasing finite times")
        timestamps.append(value)
    return timestamps


class ReviewWriter:
    """Write each processed frame, then encode a browser-playable H.264 MP4."""

    def __init__(self, path, fps, size):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".avi", prefix="review_",
                                         dir=self.path.parent, delete=False) as stream:
            self.raw_path = Path(stream.name)
        self.frame_count = 0
        self.writer = cv2.VideoWriter(str(self.raw_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size)
        if not self.writer.isOpened():
            self.raw_path.unlink(missing_ok=True)
            raise RuntimeError("Cannot open review video writer")

    def write(self, frame):
        self.writer.write(frame)
        self.frame_count += 1

    def close(self):
        self.writer.release()
        # Encode separately so interruption cannot corrupt an existing review.
        # Keep the AVI if encoding or verification fails, for recovery.
        with tempfile.NamedTemporaryFile(suffix=".mp4", prefix="review_encode_",
                                         dir=self.path.parent, delete=False) as stream:
            encoded = Path(stream.name)
        try:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(self.raw_path),
                            "-c:v", "libx264", "-threads", "2", "-preset", "fast", "-crf", "18",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(encoded)], check=True)
            capture = cv2.VideoCapture(str(encoded))
            try:
                saved = round(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
            finally:
                capture.release()
            if saved != self.frame_count:
                raise RuntimeError(f"Review frame count mismatch: {saved} != {self.frame_count}; retained {self.raw_path}")
            encoded.replace(self.path)
            self.raw_path.unlink()
        finally:
            encoded.unlink(missing_ok=True)


def as_json(result, processing_ms: float, frame_age_ms: float | None, camera_id: str, diagnostics=None):
    return json.dumps({
        "camera_id": camera_id,
        "timestamp_s": result.timestamp,
        "state": result.state,
        "observed": result.observed,
        "centroid_px": result.center,
        "filtered_centroid_px": result.filtered_center,
        "velocity_px_s": result.velocity_px_s,
        "radius_px": result.radius,
        "confidence": round(result.confidence, 3),
        "candidate_count": len(result.candidates),
        "processing_ms": round(processing_ms, 2),
        "frame_age_ms": round(frame_age_ms, 2) if frame_age_ms is not None else None,
        **(diagnostics or {}),
    }, separators=(",", ":"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--device", default="/dev/video0", help="V4L2 camera path")
    source.add_argument("--video", help="Recorded video file for repeatable tuning")
    parser.add_argument("--camera-id", default="left", help="ID included in JSONL output")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--fourcc", choices=("MJPG", "YUYV", "auto"), default="MJPG")
    parser.add_argument("--detector", choices=("hybrid", "appearance", "motion"), default="hybrid")
    parser.add_argument("--min-radius", type=int, default=3, help="Minimum ball radius in pixels")
    parser.add_argument("--max-radius", type=int, help="Maximum radius; hybrid default derives from image size")
    parser.add_argument("--floor-region", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                        help="Optional normalized floor rectangle; soft acquisition preference only")
    parser.add_argument("--no-appearance-verifier", action="store_true",
                        help="Disable the learned ball appearance check in hybrid mode")
    parser.add_argument("--no-auto-floor", action="store_true", help="Disable the floor-like scene hint")
    parser.add_argument("--timestamps", help="Replay timestamp CSV; defaults to same-name sidecar if present")
    parser.add_argument("--review-video", help="Save annotated MP4 (requires --video and FFmpeg)")
    parser.add_argument("--show-candidates", action="store_true", help="Draw all proposal boxes in review/display")
    parser.add_argument("--opencv-threads", type=int, default=2, help="Bound OpenCV worker threads")
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--max-area", type=int, default=4000)
    parser.add_argument("--difference-threshold", type=int, default=18)
    parser.add_argument("--display", action="store_true", help="Show candidates and track")
    parser.add_argument("--jsonl", metavar="PATH", help="Write results; '-' means stdout")
    parser.add_argument("--max-frames", type=int, help="Stop after this many processed frames")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.review_video and not args.video:
        parser.error("--review-video requires --video")
    if args.review_video and Path(args.review_video).suffix.lower() != ".mp4":
        parser.error("--review-video must end in .mp4")
    if args.timestamps and not args.video:
        parser.error("--timestamps requires --video")
    if args.opencv_threads < 1:
        parser.error("--opencv-threads must be positive")
    if args.video:
        inputs = {Path(args.video).resolve(), Path(args.video).with_suffix(".csv").resolve()}
        if args.timestamps:
            inputs.add(Path(args.timestamps).resolve())
        outputs = [Path(p).resolve() for p in (args.review_video, args.jsonl) if p and p != "-"]
        if any(p in inputs for p in outputs) or len(set(outputs)) != len(outputs):
            parser.error("Output paths must differ from input files and each other")
    cv2.setNumThreads(args.opencv_threads)
    segmenter = MotionBallSegmenter(
        warmup_frames=args.warmup_frames,
        min_area=args.min_area,
        max_area=args.max_area,
        difference_threshold=args.difference_threshold,
    )
    if args.detector == "appearance":
        segmenter = AppearanceBallSegmenter(min_radius=args.min_radius, max_radius=args.max_radius or 140,
                                            floor_region=args.floor_region)
    if args.detector == "hybrid":
        segmenter = HybridBallSegmenter(min_radius=args.min_radius, max_radius=args.max_radius,
                                        floor_region=args.floor_region, auto_floor=not args.no_auto_floor,
                                        verify_appearance=not args.no_appearance_verifier)
    tracker = BallTracker(segmenter, confirmation_hits=1 if args.detector == "hybrid" else 3,
                          adaptive_gate=args.detector == "hybrid")
    output = None
    if args.jsonl and args.jsonl != "-":
        Path(args.jsonl).parent.mkdir(parents=True, exist_ok=True)
        output = open(args.jsonl, "w", encoding="utf-8")

    camera = None
    video = None
    review = None
    timestamps = None
    trail = deque(maxlen=30)
    try:
        if args.video:
            video = cv2.VideoCapture(args.video)
            if not video.isOpened():
                raise RuntimeError(f"Cannot open video {args.video}")
            video_fps = video.get(cv2.CAP_PROP_FPS) or 30.0
            timing_path = Path(args.timestamps) if args.timestamps else Path(args.video).with_suffix(".csv")
            if timing_path.exists() or args.timestamps:
                timestamps = load_timestamps(timing_path, round(video.get(cv2.CAP_PROP_FRAME_COUNT)))
                logging.info("Using recorded frame timestamps from %s", timing_path)
            if args.review_video:
                review = ReviewWriter(args.review_video, video_fps,
                                      (round(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
                                       round(video.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        else:
            camera = LatestCamera(args.device, args.width, args.height, args.fps, args.fourcc)
        sequence = 0
        processed = 0
        last_report = time.monotonic()
        while args.max_frames is None or processed < args.max_frames:
            if video:
                ok, frame = video.read()
                if not ok:
                    if timestamps is not None and processed != len(timestamps):
                        logging.warning("Decoded %d frames but timestamp CSV has %d rows; source decoding/timing needs review", processed, len(timestamps))
                    break
                timestamp = timestamps[processed] if timestamps is not None else processed / video_fps
            else:
                sample = camera.read(sequence)
                if sample is None:
                    if not camera.running:
                        raise RuntimeError("Camera stopped returning frames")
                    continue
                sequence, frame, timestamp = sample
            started = time.monotonic()
            result = tracker.process(frame, timestamp)
            finished = time.monotonic()
            processed += 1
            if args.jsonl:
                print(as_json(result, (finished - started) * 1000,
                              (finished - timestamp) * 1000 if camera else None,
                              args.camera_id, diagnostics=scene_diagnostics(segmenter)),
                      file=output, flush=True)
            if result.observed and result.filtered_center is not None:
                trail.append(tuple(round(v) for v in result.filtered_center))
            elif result.state == "absent":
                trail.clear()
            if args.display or review:
                shown = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                annotate(shown, result, frame_index=processed - 1, trail=trail,
                         show_candidates=args.show_candidates, diagnostics=scene_diagnostics(segmenter))
                if review:
                    review.write(shown)
            if args.display:
                cv2.imshow("Robokeeper 2D ball tracker", shown)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
            now = time.monotonic()
            if now - last_report >= 5:
                logging.info("Processed %d frames; latest state: %s", processed, result.state)
                last_report = now
    except KeyboardInterrupt:
        pass
    finally:
        if camera:
            camera.close()
        if video:
            video.release()
        if output:
            output.close()
        if args.display:
            cv2.destroyAllWindows()
        if review:
            review.close()
            logging.info("Saved review video: %s", Path(args.review_video).resolve())


if __name__ == "__main__":
    main()
