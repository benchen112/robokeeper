#!/usr/bin/env python3
"""Track a moving ball from the OV9281 camera or a recorded video.

The JSONL output reports camera pixel coordinates. Only rows with state
"confirmed" and observed=true contain a new, validated ball measurement.
"""

import argparse
import json
import logging
import threading
import time

import cv2

from robokeeper import BallTracker, MotionBallSegmenter


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


def annotate(frame, result):
    for candidate in result.candidates:
        x, y, w, h = candidate.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (80, 150, 255), 1)
    if result.filtered_center is not None:
        x, y = (round(v) for v in result.filtered_center)
        color = (0, 220, 0) if result.state == "confirmed" else (0, 220, 255)
        cv2.circle(frame, (x, y), max(3, round(result.radius or 0)), color, 2)
    cv2.putText(frame, result.state, (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.75, (0, 220, 0), 2, cv2.LINE_AA)
    return frame


def as_json(result, processing_ms: float, frame_age_ms: float | None, camera_id: str):
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
    parser.add_argument("--warmup-frames", type=int, default=5)
    parser.add_argument("--min-area", type=int, default=5)
    parser.add_argument("--max-area", type=int, default=4000)
    parser.add_argument("--difference-threshold", type=int, default=18)
    parser.add_argument("--display", action="store_true", help="Show candidates and track")
    parser.add_argument("--jsonl", metavar="PATH", help="Write results; '-' means stdout")
    parser.add_argument("--max-frames", type=int, help="Stop after this many processed frames")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    segmenter = MotionBallSegmenter(
        warmup_frames=args.warmup_frames,
        min_area=args.min_area,
        max_area=args.max_area,
        difference_threshold=args.difference_threshold,
    )
    tracker = BallTracker(segmenter)
    output = None
    if args.jsonl and args.jsonl != "-":
        output = open(args.jsonl, "w", encoding="utf-8")

    camera = None
    video = None
    try:
        if args.video:
            video = cv2.VideoCapture(args.video)
            if not video.isOpened():
                raise RuntimeError(f"Cannot open video {args.video}")
            video_fps = video.get(cv2.CAP_PROP_FPS) or 30.0
        else:
            camera = LatestCamera(args.device, args.width, args.height, args.fps, args.fourcc)
        sequence = 0
        processed = 0
        last_report = time.monotonic()
        while args.max_frames is None or processed < args.max_frames:
            if video:
                ok, frame = video.read()
                if not ok:
                    break
                timestamp = processed / video_fps
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
                              args.camera_id),
                      file=output, flush=True)
            if args.display:
                shown = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                cv2.imshow("Robokeeper 2D ball tracker", annotate(shown, result))
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


if __name__ == "__main__":
    main()
