#!/usr/bin/env python3
"""Replay stereo tracks through the trajectory predictor, as if live.

For each ``track_stereo.py --jsonl`` file, triangulates full-ball stereo
pairs with a calibration, feeds them to TrajectoryPredictor one frame at a
time, and reports how the predicted camera-plane crossing and servo angle
evolve. Accuracy is checked against the clip's own later frames: each early
prediction is extrapolated to the depth of the last observed frame and
compared with where the ball actually was there.

    python3 tools/predict_kicks.py tracking_review/2026-10-06/stereo_*.jsonl \
        --plot tracking_review/2026-10-06/topdown.png
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robokeeper.stereo3d import StereoCalibration, read_stereo_track  # noqa: E402
from robokeeper.trajectory import TrajectoryPredictor, servo_angle_deg  # noqa: E402

SERIES = ((214, 120, 42), (52, 104, 235), (122, 175, 27))  # BGR of #2a78d6, #eb6834, #1baf7a


def replay(track, calibration, args):
    predictor = TrajectoryPredictor(window_s=args.window_s)
    points, steps = [], []
    for frame_index, timestamp, (left_xy, _, right_xy, _) in track:
        point = calibration.triangulate(left_xy, right_xy)
        if point is None:
            continue
        predictor.add(timestamp, (point.x_m, point.height_m, point.z_m))
        points.append((timestamp, point.x_m, point.height_m, point.z_m))
        crossing = predictor.predict()
        if crossing:
            steps.append((frame_index, timestamp, crossing))
    return np.array(points), steps


def summarize(name, points, steps, args):
    if len(points) < 2 or not steps:
        return {"clip": name, "pairs": len(points), "prediction": None}
    end_t, end_x, _, end_z = points[-3:].mean(axis=0)
    start = points[0, 0]
    timeline, errors = [], []
    for frame_index, timestamp, crossing in steps:
        vx, _, vz = crossing.velocity_mps
        # Where this prediction says the ball would be at the last observed depth.
        t_at_end = crossing.time_s - (0 - end_z) / vz
        x_at_end = crossing.x_m - vx * (crossing.time_s - t_at_end)
        error = x_at_end - end_x
        errors.append((timestamp - start, error))
        timeline.append({
            "frame": frame_index, "elapsed_s": round(timestamp - start, 3),
            "crossing_x_m": round(crossing.x_m, 3), "x_std_m": round(crossing.x_std_m, 3),
            "time_to_cross_s": round(crossing.time_to_cross_s, 3),
            "servo_deg": round(servo_angle_deg(crossing.x_m, crossing.height_m,
                                               args.pivot_x_m, args.pivot_height_m), 1),
            "model": crossing.model, "check_error_m": round(error, 3),
        })
    final = steps[-1][2]
    return {
        "clip": name, "pairs": len(points),
        "observed_span_s": round(points[-1, 0] - start, 3),
        "start_m": [round(v, 2) for v in points[0, [1, 3]]],
        "last_seen_m": [round(float(end_x), 2), round(float(end_z), 2)],
        "first_prediction_after_s": timeline[0]["elapsed_s"],
        "prediction": {
            "crossing_x_m": round(final.x_m, 3), "x_std_m": round(final.x_std_m, 3),
            "height_m": round(final.height_m, 3), "speed_mps": round(final.speed_mps, 2),
            "time_to_cross_after_last_frame_s": round(final.time_to_cross_s, 3),
            "model": final.model,
            "servo_deg": round(servo_angle_deg(final.x_m, final.height_m,
                                               args.pivot_x_m, args.pivot_height_m), 1),
        },
        "check_error_m": {
            label: round(float(np.median([abs(e) for t, e in errors if lo <= t < hi])), 3)
            for label, lo, hi in (("first_0.2s", 0, 0.2), ("0.2-0.5s", 0.2, 0.5),
                                  ("after_0.5s", 0.5, 99))
            if any(lo <= t < hi for t, _ in errors)
        },
        "timeline": timeline,
    }


def draw_topdown(results, path, x_range=(-1.5, 2.5), z_max=8.0):
    """Top view: measured ball centers, final predicted path, plane crossing."""
    width, height, margin = 900, 1100, 70
    image = np.full((height, width, 3), 255, np.uint8)
    ink, muted, grid = (20, 20, 20), (110, 110, 110), (225, 225, 225)
    sx = (width - 2 * margin) / (x_range[1] - x_range[0])
    sz = (height - 2 * margin) / z_max
    px = lambda x, z: (int(margin + (x - x_range[0]) * sx), int(height - margin - z * sz))
    for z in range(0, int(z_max) + 1):
        cv2.line(image, px(x_range[0], z), px(x_range[1], z), grid, 1)
        cv2.putText(image, f"{z} m", (8, px(0, z)[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, .5, muted, 1, cv2.LINE_AA)
    for x in np.arange(x_range[0], x_range[1] + 0.01, 0.5):
        cv2.line(image, px(x, 0), px(x, z_max), grid, 1)
        cv2.putText(image, f"{x:+.1f}", (px(x, 0)[0] - 18, height - margin + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, muted, 1, cv2.LINE_AA)
    cv2.line(image, px(x_range[0], 0), px(x_range[1], 0), ink, 2)
    cv2.putText(image, "camera plane (Z = 0)", (px(x_range[0], 0)[0] + 6, px(0, 0)[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, .5, ink, 1, cv2.LINE_AA)
    cv2.circle(image, px(0, 0), 6, ink, -1)
    legend_y = 70
    for result, color in zip(results, SERIES):
        points = result.get("points")
        if points is None or not len(points) or result["prediction"] is None:
            continue
        crossing = result["prediction"]["crossing_x_m"]
        cv2.circle(image, (width - 330, legend_y - 5), 6, color, -1, cv2.LINE_AA)
        cv2.putText(image, f"{result['clip']}: crosses at X {crossing:+.2f} m",
                    (width - 315, legend_y), cv2.FONT_HERSHEY_SIMPLEX, .55, ink, 1, cv2.LINE_AA)
        legend_y += 26
        for _, x, _, z in points:
            cv2.circle(image, px(x, z), 5, color, -1, cv2.LINE_AA)
            cv2.circle(image, px(x, z), 6, (255, 255, 255), 1, cv2.LINE_AA)
        start, end = px(*result["last_seen_m"]), px(crossing, 0)
        for k in range(0, 20, 2):  # dashed predicted path
            a = np.add(start, np.subtract(end, start) * k / 20).astype(int)
            b = np.add(start, np.subtract(end, start) * (k + 1) / 20).astype(int)
            cv2.line(image, tuple(a), tuple(b), color, 2, cv2.LINE_AA)
        cv2.drawMarker(image, end, color, cv2.MARKER_TILTED_CROSS, 18, 3)
    cv2.putText(image, "Top view: ball centers (dots) and predicted path to the camera plane (x)",
                (margin, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, ink, 1, cv2.LINE_AA)
    cv2.putText(image, "X lateral (m, + = right as seen from camera)", (margin, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX, .5, muted, 1, cv2.LINE_AA)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tracks", nargs="+", type=Path)
    parser.add_argument("--calibration", type=Path, default=Path("calibration/stereo.json"))
    parser.add_argument("--window-s", type=float, default=0.6)
    parser.add_argument("--pivot-x-m", type=float, default=0.0, help="Servo pivot X")
    parser.add_argument("--pivot-height-m", type=float, default=0.0,
                        help="Servo pivot height relative to the cameras")
    parser.add_argument("--output", type=Path, help="Write full results JSON here")
    parser.add_argument("--plot", type=Path, help="Write a top-view PNG here")
    args = parser.parse_args()
    calibration = StereoCalibration.load(args.calibration)
    results = []
    for path in args.tracks:
        points, steps = replay(read_stereo_track(path, calibration.image_size), calibration, args)
        result = summarize(path.stem, points, steps, args)
        result["points"] = points
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("timeline", "points")}))
    if args.plot:
        draw_topdown(results, args.plot)
    if args.output:
        args.output.write_text(json.dumps(
            [{k: v for k, v in r.items() if k != "points"} for r in results], indent=2) + "\n",
            encoding="utf-8")


if __name__ == "__main__":
    main()
