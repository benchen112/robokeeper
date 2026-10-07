#!/usr/bin/env python3
"""Per-stage processing time of the stereo ball pipeline, by ball distance.

Replays recorded stereo clips through the same trackers track_stereo.py uses,
with timers wrapped around each pipeline stage. Frames are converted to
grayscale before timing, as the live CSI camera delivers them, and video
decoding is excluded. Each pair is assigned a ball distance (triangulated
when both eyes see it, otherwise from the ball's pixel radius).

To spare an uncooled Pi, processing pauses when the CPU passes --temp-limit
and resumes once it cools; pauses are not timed. CPU frequency is recorded
per frame so throttled frames can be spotted.

    python3 tools/profile_stereo.py recordings/stereo/kick_*m_*.avi \
        --output tracking_review/2026-10-06/profile.json
"""

import argparse
from collections import defaultdict
import functools
import json
import logging
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robokeeper import hybrid, vision  # noqa: E402
from robokeeper.appearance import BallAppearanceVerifier  # noqa: E402
from robokeeper.stereo import split_stereo  # noqa: E402
from robokeeper.stereo3d import StereoCalibration  # noqa: E402
from robokeeper.trajectory import TrajectoryPredictor  # noqa: E402
from track_ball import load_timestamps  # noqa: E402
from track_stereo import make_tracker  # noqa: E402

OTHER = "tracking logic, contours, misc"
MOTION = "motion mask: pixel math"
CHECKS = "candidate checks (edges, motion, patches)"
BUCKETS = (("no ball tracked", None, None), ("over 6 m", 6.0, 99.0), ("4-6 m", 4.0, 6.0),
           ("2.5-4 m", 2.5, 4.0), ("1.5-2.5 m", 1.5, 2.5), ("under 1.5 m", 0.0, 1.5))
BALL_RADIUS_M = 0.105


class StageTimer:
    """Self time per label for nested wrapped calls (children are subtracted)."""

    def __init__(self):
        self.stack = []
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)

    def reset(self):
        self.totals.clear()
        self.counts.clear()

    def wrap(self, function, label):
        """``label`` is a name, or callable(parent_label, args) returning one or None."""
        timer = self

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            parent = timer.stack[-1][0] if timer.stack else None
            name = label(parent, args) if callable(label) else label
            if name is None:
                return function(*args, **kwargs)
            entry = [name, 0.0]
            timer.stack.append(entry)
            start = time.perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - start
                timer.stack.pop()
                timer.totals[name] += elapsed - entry[1]
                timer.counts[name] += 1
                if timer.stack:
                    timer.stack[-1][1] += elapsed
        return wrapper


def install_timers(timer):
    def by_parent(mapping):
        return lambda parent, args: mapping.get(parent)

    def template_label(parent, args):
        image, template = args[0], args[1]
        large = image.shape[0] > template.shape[0] + 2 or image.shape[1] > template.shape[1] + 2
        return "template search near ball" if large else CHECKS

    patches = [
        (vision.BallTracker, "process", OTHER),
        (hybrid.HybridBallSegmenter, "find_candidates", OTHER),
        (hybrid.HybridBallSegmenter, "_coherent_acquisition", OTHER),
        (hybrid.HybridBallSegmenter, "_motion", MOTION),
        (hybrid.HybridBallSegmenter, "_camera_transform", "camera-motion check (features, flow)"),
        (hybrid.HybridBallSegmenter, "_motion_support", CHECKS),
        (hybrid.FloorEstimator, "update", "floor estimate (every 6th frame)"),
        (vision.AppearanceBallSegmenter, "_edge_support", CHECKS),
        (vision.AppearanceBallSegmenter, "_patch", CHECKS),
        (BallAppearanceVerifier, "score", "appearance classifier (HOG + trees)"),
        (cv2, "HoughCircles", "circle detection (Hough)"),
        (cv2, "matchTemplate", template_label),
        (cv2, "GaussianBlur", by_parent({OTHER: "pre-blur"})),
        (cv2, "Sobel", by_parent({OTHER: "edge gradients (full-frame Sobel)",
                                  MOTION: "motion mask: edge allowance (Sobel)"})),
        (cv2, "warpPerspective", by_parent({MOTION: "motion mask: align 4 frames (warp)"})),
        (cv2, "morphologyEx", by_parent({MOTION: "motion mask: morphology"})),
        (cv2, "accumulateWeighted", by_parent({MOTION: "motion mask: background update"})),
    ]
    for owner, name, label in patches:
        setattr(owner, name, timer.wrap(getattr(owner, name), label))


def read_sysfs(path, scale=1.0):
    try:
        return float(Path(path).read_text().strip()) / scale
    except OSError:
        return None


def cpu_temp():
    return read_sysfs("/sys/class/thermal/thermal_zone0/temp", 1000)


def cpu_mhz():
    return read_sysfs("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq", 1000)


def mem_available_mb():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 1024
    return None


def bucket_for(distance):
    if distance is None:
        return BUCKETS[0][0]
    return next(name for name, low, high in BUCKETS[1:] if low <= distance < high)


def profile_clip(path, calibration, timer, args):
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    count = round(video.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamps = load_timestamps(Path(path).with_suffix(".csv"), count)
    options = SimpleNamespace(detector="hybrid", min_radius=3, max_radius=None,
                              no_auto_floor=False, no_appearance_verifier=False,
                              moving_camera=args.moving_camera)
    trackers = [make_tracker(options)[0], make_tracker(options)[0]]
    predictor = TrajectoryPredictor()
    rows, paused = [], 0.0
    try:
        for index in range(count if args.max_frames is None else min(count, args.max_frames)):
            ok, frame = video.read()
            if not ok:
                break
            if index % 10 == 0 and mem_available_mb() < args.min_free_mb:
                raise MemoryError(f"Only {mem_available_mb():.0f} MB free; stopping to protect the Pi")
            temperature = cpu_temp()
            if temperature and temperature > args.temp_limit:
                wait_start = time.monotonic()
                while (cpu_temp() or 0) > args.temp_limit - 10:
                    time.sleep(1)
                paused += time.monotonic() - wait_start
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            views = split_stereo(gray, calibration.image_size)
            eyes = []
            for tracker, view in zip(trackers, views):
                timer.reset()
                start = time.perf_counter()
                result = tracker.process(view, timestamps[index])
                total = time.perf_counter() - start
                eyes.append({"ms": total * 1000, "result": result,
                             "stages": {k: v * 1000 for k, v in timer.totals.items()},
                             "calls": dict(timer.counts)})
            left, right = (eye["result"] for eye in eyes)
            distance = None
            start = time.perf_counter()
            if all(r.state == "confirmed" and r.observed for r in (left, right)):
                point = calibration.triangulate(left.center, right.center,
                                                radius_px=(left.radius + right.radius) / 2)
                if point:
                    distance = point.z_m
                    predictor.add(timestamps[index], (point.x_m, point.height_m, point.z_m))
                    predictor.predict()
            prediction_ms = (time.perf_counter() - start) * 1000
            if distance is None:
                radii = [r.radius for r in (left, right) if r.observed and r.radius]
                if radii:
                    distance = calibration.focal_px * BALL_RADIUS_M / max(radii)
            rows.append({
                "clip": Path(path).stem, "frame": index, "distance_m": distance,
                "bucket": bucket_for(distance),
                "radius_px": max([r.radius for r in (left, right) if r.observed and r.radius],
                                 default=None),
                "pair_ms": sum(eye["ms"] for eye in eyes), "eye_ms": [eye["ms"] for eye in eyes],
                "prediction_ms": prediction_ms,
                "stages": {k: sum(eye["stages"].get(k, 0) for eye in eyes)
                           for k in set().union(*(eye["stages"] for eye in eyes))},
                "calls": {k: sum(eye["calls"].get(k, 0) for eye in eyes)
                          for k in set().union(*(eye["calls"] for eye in eyes))},
                "cpu_mhz": cpu_mhz(), "cpu_c": temperature,
            })
    finally:
        video.release()
    return rows, paused


def summarize(rows, max_mhz):
    summary = []
    for name, _, _ in BUCKETS:
        group = [row for row in rows if row["bucket"] == name]
        if not group:
            continue
        pair = np.array([row["pair_ms"] for row in group])
        slowest_eye = np.array([max(row["eye_ms"]) for row in group])
        stages = defaultdict(float)
        calls = defaultdict(float)
        for row in group:
            for stage, value in row["stages"].items():
                stages[stage] += value / len(group)
            for stage, value in row["calls"].items():
                calls[stage] += value / len(group)
        radii = [row["radius_px"] for row in group if row["radius_px"]]
        summary.append({
            "bucket": name, "pairs": len(group),
            "ball_radius_px": [round(min(radii)), round(max(radii))] if radii else None,
            "pair_ms_median": round(float(np.median(pair)), 1),
            "pair_ms_p90": round(float(np.percentile(pair, 90)), 1),
            "slowest_eye_ms_median": round(float(np.median(slowest_eye)), 1),
            "prediction_ms_mean": round(float(np.mean([r["prediction_ms"] for r in group])), 3),
            "stage_ms_mean": {k: round(v, 2) for k, v in sorted(stages.items(), key=lambda kv: -kv[1])},
            "calls_per_pair": {k: round(v, 1) for k, v in calls.items()},
            "throttled_frames": sum(1 for r in group if r["cpu_mhz"] and r["cpu_mhz"] < max_mhz - 1),
        })
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("videos", nargs="+", type=Path)
    parser.add_argument("--calibration", type=Path, default=Path("calibration/stereo.json"))
    parser.add_argument("--opencv-threads", type=int, default=2, help="Same default as track_stereo.py")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--moving-camera", action="store_true", help="Same flag as track_stereo.py")
    parser.add_argument("--temp-limit", type=float, default=78.0)
    parser.add_argument("--min-free-mb", type=float, default=150.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    cv2.setNumThreads(args.opencv_threads)
    calibration = StereoCalibration.load(args.calibration)
    timer = StageTimer()
    install_timers(timer)
    max_mhz = read_sysfs("/sys/devices/system/cpu/cpu0/cpufreq/scaling_max_freq", 1000) or 0
    rows, paused = [], 0.0
    try:
        for path in args.videos:
            logging.info("Profiling %s", path)
            clip_rows, clip_paused = profile_clip(path, calibration, timer, args)
            rows += clip_rows
            paused += clip_paused
    except MemoryError as exc:
        logging.error("%s (keeping %d profiled pairs)", exc, len(rows))
    summary = summarize(rows, max_mhz)
    result = {"opencv_threads": args.opencv_threads, "cpu_max_mhz": max_mhz,
              "cooling_pauses_s": round(paused, 1), "summary": summary, "frames": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "frames"}, indent=1))


if __name__ == "__main__":
    main()
