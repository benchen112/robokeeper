#!/usr/bin/env python3
"""Calibrate the stereo pair for 3D ball positions.

``kicks``: provisional calibration from ``track_stereo.py --jsonl`` runs of a
ball rolled or kicked from known distances, e.g.

    python3 tools/calibrate_stereo.py kicks \
        tracking_review/2026-10-06/stereo_4m.jsonl:4 \
        tracking_review/2026-10-06/stereo_7m.jsonl:7

``checkerboard``: accurate calibration from a stereo clip (record_stereo.py)
of a printed checkerboard held at many angles and distances, e.g.

    python3 tools/calibrate_stereo.py checkerboard recordings/stereo/board_*.avi \
        --inner-corners 9x6 --square-mm 25

Both write a StereoCalibration JSON (default calibration/stereo.json).
"""

import argparse
import json
import logging
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robokeeper.stereo import split_stereo  # noqa: E402
from robokeeper.stereo3d import StereoCalibration, fit_kick_calibration, read_stereo_track  # noqa: E402


def find_board(gray, pattern):
    found, corners = cv2.findChessboardCornersSB(gray, pattern, flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
    return corners.reshape(-1, 2).astype(np.float32) if found else None


def collect_board_pairs(frames, pattern, eye_size, min_move=0.05, max_pairs=40):
    """Corner pairs from packed stereo frames where both eyes see the board.

    Skips views whose board center moved less than ``min_move`` of the image
    width since the last accepted pair, so a held-still board is not counted
    many times.
    """
    left_points, right_points = [], []
    last_center = None
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        left, right = split_stereo(gray, eye_size)
        corners_left = find_board(left, pattern)
        if corners_left is None:
            continue
        center = corners_left.mean(axis=0)
        if last_center is not None and np.linalg.norm(center - last_center) < min_move * eye_size[0]:
            continue
        corners_right = find_board(right, pattern)
        if corners_right is None:
            continue
        left_points.append(corners_left)
        right_points.append(corners_right)
        last_center = center
        if len(left_points) >= max_pairs:
            break
    return left_points, right_points


def calibrate_board(left_points, right_points, pattern, square_m, eye_size):
    if len(left_points) < 8:
        raise ValueError(f"Only {len(left_points)} usable board pairs; need at least 8. "
                         "Move the board through more positions and angles.")
    columns, rows = pattern
    grid = np.zeros((columns * rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * square_m
    objects = [grid] * len(left_points)
    # The OV9281 M12 lenses are low distortion; a free k3 overfits wildly and
    # then makes rectification zoom far in.
    flags = cv2.CALIB_FIX_K3
    rms_left, k1, d1, *_ = cv2.calibrateCamera(objects, left_points, eye_size, None, None,
                                               flags=flags)
    rms_right, k2, d2, *_ = cv2.calibrateCamera(objects, right_points, eye_size, None, None,
                                                flags=flags)
    rms, k1, d1, k2, d2, rotation, translation, *_ = cv2.stereoCalibrate(
        objects, left_points, right_points, k1, d1, k2, d2, eye_size,
        flags=cv2.CALIB_FIX_INTRINSIC)
    r1, r2, p1, p2, *_ = cv2.stereoRectify(k1, d1, k2, d2, eye_size, rotation, translation,
                                           flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    baseline = float(-p2[0, 3] / p2[0, 0])
    if baseline <= 0:
        raise ValueError("Right eye is left of the left eye; record with --swap-eyes or check wiring")
    as_lists = lambda **arrays: {key: np.asarray(value).tolist() for key, value in arrays.items()}
    calibration = StereoCalibration(
        focal_px=float(p1[0, 0]), cx=float(p1[0, 2]), cy=float(p1[1, 2]),
        baseline_m=baseline, disparity_offset_px=float(p1[0, 2] - p2[0, 2]),
        image_size=tuple(eye_size), method="checkerboard",
        notes=f"{len(left_points)} board pairs; stereo RMS {rms:.3f} px. Heights are "
              "relative to the rectified optical axis, not the ground.",
        rectification={"left": as_lists(K=k1, D=d1, R=r1, P=p1),
                       "right": as_lists(K=k2, D=d2, R=r2, P=p2)})
    diagnostics = {
        "pairs": len(left_points), "rms_left_px": round(rms_left, 3),
        "rms_right_px": round(rms_right, 3), "stereo_rms_px": round(rms, 3),
        "baseline_m": round(baseline, 4),
        "convergence_deg": round(float(np.degrees(np.arccos(
            np.clip((np.trace(rotation) - 1) / 2, -1, 1)))), 2),
    }
    return calibration, diagnostics


def video_frames(path, every):
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise RuntimeError(f"Cannot open video {path}")
    try:
        index = 0
        while True:
            ok, frame = video.read()
            if not ok:
                return
            if index % every == 0:
                yield frame
            index += 1
    finally:
        video.release()


def parse_clip(value):
    path, _, distance = value.rpartition(":")
    if not path:
        raise argparse.ArgumentTypeError("Use TRACK.jsonl:START_DISTANCE_M (or :- if unknown)")
    return Path(path), None if distance == "-" else float(distance)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("calibration/stereo.json"))
    parser.add_argument("--baseline-m", type=float, default=0.30,
                        help="Lens center spacing (measured 0.30 m)")
    parser.add_argument("--eye-size", default="1280x800")
    commands = parser.add_subparsers(dest="command", required=True)
    kicks = commands.add_parser("kicks", help="Provisional fit from tracked kick clips")
    kicks.add_argument("clips", nargs="+", type=parse_clip, metavar="TRACK.jsonl:DISTANCE_M")
    board = commands.add_parser("checkerboard", help="Checkerboard calibration from a stereo clip")
    board.add_argument("video", type=Path)
    board.add_argument("--inner-corners", default="9x6",
                       help="Inner corners per row x per column, e.g. 9x6 for a 10x7-square board")
    board.add_argument("--square-mm", type=float, required=True, help="Printed square size")
    board.add_argument("--every", type=int, default=6, help="Check every Nth frame")
    args = parser.parse_args()
    eye_size = tuple(int(v) for v in args.eye_size.split("x"))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "kicks":
        clips = [([pair for _, _, pair in read_stereo_track(path, eye_size)], distance)
                 for path, distance in args.clips]
        calibration, diagnostics = fit_kick_calibration(clips, args.baseline_m, eye_size)
    else:
        pattern = tuple(int(v) for v in args.inner_corners.split("x"))
        left, right = collect_board_pairs(video_frames(args.video, args.every), pattern, eye_size)
        calibration, diagnostics = calibrate_board(left, right, pattern, args.square_mm / 1000, eye_size)
        if abs(diagnostics["baseline_m"] - args.baseline_m) > 0.1 * args.baseline_m:
            logging.warning("Calibrated baseline %.3f m differs from the measured %.3f m; "
                            "check --square-mm", diagnostics["baseline_m"], args.baseline_m)
    calibration.save(args.output)
    logging.info("Saved %s calibration to %s", calibration.method, args.output)
    print(json.dumps({"focal_px": round(calibration.focal_px, 1), "cx": calibration.cx,
                      "cy": round(calibration.cy, 1), "baseline_m": calibration.baseline_m,
                      "vertical_offset_px": round(calibration.vertical_offset_px, 1),
                      **diagnostics}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, OSError) as exc:
        logging.error("%s", exc)
        sys.exit(1)
