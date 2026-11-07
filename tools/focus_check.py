#!/usr/bin/env python3
"""Focus the stereo lenses by the numbers instead of by eye.

On the Pi, point the cameras at a still, detailed target (printed text, a
checkerboard, a poster) about as far away as the ball will be (5-8 m), keep
everything in front of the cameras still (people walking through change the
numbers), and run:

    python3 tools/focus_check.py

Exposure is calibrated once and then locked, as live_keeper.py does it. Several
times a second it prints each camera's sharpness (physical LEFT/RIGHT, seen
from behind the cameras, using swap_eyes from the calibration), how close it
is to the best seen so far, and whether it is getting sharper or blurrier:

    LEFT   212 ######.... 61% of best  sharper   | RIGHT   598 ########## 99%  steady   | L/R 0.35

Turn one lens's focus ring slowly, a little at a time. If it says "blurrier",
turn the other way. Go just past the peak (it starts falling from 100%), then
come back until it reads ~100% of best, and lock the ring. A well focused left
camera should come close to the right one (L/R near 1; the two views differ a
little, so not exactly 1).

Press Enter to forget the best values (after moving the target or the
cameras); q then Enter, or Ctrl-C, to stop. At the end the crops it measured
are saved to runs/focus_check.jpg. Focusing moves the lens, so rerun
``tools/calibrate_stereo.py roll`` afterwards.

``--video CLIP.avi`` measures a recorded stereo clip instead (no camera).
"""

import argparse
import logging
from pathlib import Path
import queue
import sys
import threading
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robokeeper.exposure import add_exposure_arguments, check_exposure_arguments, open_calibrated_camera  # noqa: E402
from robokeeper.focus import FocusMeter, bar, sharpness  # noqa: E402
from robokeeper.stereo import clip_swaps_eyes, split_stereo  # noqa: E402
from robokeeper.stereo3d import StereoCalibration  # noqa: E402


def parse_roi(value):
    roi = tuple(float(v) for v in value.split(","))
    if len(roi) != 4 or not (0 <= roi[0] < roi[2] <= 1 and 0 <= roi[1] < roi[3] <= 1):
        raise argparse.ArgumentTypeError("ROI must be x0,y0,x1,y1 fractions of the image, e.g. .3,.3,.7,.7")
    return roi


def camera_frames(args):
    camera = open_calibrated_camera(args)
    try:
        sequence = 0
        while True:
            sample = camera.read(sequence)
            if sample is None:
                raise RuntimeError(camera.error or "No camera frames")
            sequence, frame, _, _, _ = sample
            yield frame
    finally:
        camera.close()


def video_frames(path):
    video = cv2.VideoCapture(str(path))
    if not video.isOpened():
        raise RuntimeError(f"Cannot open video {path}")
    try:
        while True:
            ok, frame = video.read()
            if not ok:
                return
            yield cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    finally:
        video.release()


def keyboard(commands):
    for line in sys.stdin:
        commands.put(line.strip().lower())
    commands.put("q")


def status_line(meters):
    parts = []
    for name, meter in meters.items():
        trend = meter.trend or "..."
        value = f"{meter.current:6.1f}" if meter.current < 100 else f"{meter.current:6.0f}"
        parts.append(f"{name:5s} {value} {bar(meter.fraction_of_best)} "
                     f"{meter.fraction_of_best:4.0%} of best  {trend:8s}")
    left, right = meters["LEFT"].current, meters["RIGHT"].current
    ratio = f"L/R {left / right:.2f}" if right else "L/R -"
    return " | ".join(parts + [ratio])


def save_crops(views, roi, path):
    crops = []
    for name, view in views.items():
        h, w = view.shape
        crop = view[int(roi[1] * h):int(roi[3] * h), int(roi[0] * w):int(roi[2] * w)].copy()
        cv2.putText(crop, name, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, .9, 255, 2)
        crops.append(crop)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.hstack(crops))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, help="Measure a recorded stereo clip instead of the camera")
    parser.add_argument("--roi", type=parse_roi, default=(.3, .3, .7, .7),
                        help="Region measured, x0,y0,x1,y1 fractions of each image (default: center)")
    parser.add_argument("--calibration", type=Path, default=Path("calibration/stereo.json"),
                        help="Where swap_eyes (which half is the physical left camera) comes from")
    parser.add_argument("--interval", type=float, default=.25, help="Seconds between status lines")
    parser.add_argument("--duration", type=float, help="Stop after this many seconds")
    parser.add_argument("--output", type=Path, default=Path("runs/focus_check.jpg"))
    parser.add_argument("--eye-width", type=int, default=1280)
    parser.add_argument("--eye-height", type=int, default=800)
    parser.add_argument("--fps", type=float, default=60)
    parser.add_argument("--camera-num", type=int, default=0)
    parser.add_argument("--tuning-file")
    add_exposure_arguments(parser)
    args = parser.parse_args()
    check_exposure_arguments(parser, args)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.video:
        swap = clip_swaps_eyes(args.video)
        frames = video_frames(args.video)
    else:
        swap = StereoCalibration.load(args.calibration).swap_eyes if args.calibration.exists() else False
        frames = camera_frames(args)
    logging.info("Physical LEFT camera is the %s half of the frames (swap_eyes %s)",
                 "right" if swap else "left", swap)

    commands = queue.Queue()
    if not args.video:
        threading.Thread(target=keyboard, args=(commands,), daemon=True).start()
        print("Turn a focus ring slowly. Enter = forget best values, q + Enter = stop.", flush=True)
    meters = {"LEFT": FocusMeter(), "RIGHT": FocusMeter()}
    live = sys.stdout.isatty() and not args.video
    start = next_report = time.monotonic()
    views = None
    try:
        for count, frame in enumerate(frames):
            left, right = split_stereo(frame, (args.eye_width, args.eye_height), swap)
            views = {"LEFT": left, "RIGHT": right}
            for name, view in views.items():
                meters[name].update(sharpness(view, args.roi))
            while not commands.empty():
                command = commands.get()
                if command == "q":
                    raise KeyboardInterrupt
                for meter in meters.values():
                    meter.reset_best()
                print("\nBest values forgotten", flush=True)
            now = time.monotonic()
            if args.video:
                report = count % 30 == 29
            else:
                report = now >= next_report
            if report:
                print(("\r" if live else "") + status_line(meters), end="" if live else "\n", flush=True)
                next_report = now + args.interval
            if args.duration and now - start > args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if live:
            print()
    if views is None:
        logging.error("No frames")
        return 1
    print(status_line(meters))
    save_crops(views, args.roi, args.output)
    logging.info("Saved the measured crops to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
