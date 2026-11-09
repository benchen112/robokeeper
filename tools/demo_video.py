#!/usr/bin/env python3
"""Presentation video of one recorded kick: what the cameras see and what the
keeper predicts, slowed down.

Replays a stereo clip (record_stereo.py) through the same trackers and kick
session as live_keeper.py and renders a 1920x1080 H.264 MP4 (plays in Google
Slides) with:

  top     both camera views, the tracked ball circled with its recent trail
  bottom  a top-down view (3D points and the predicted path to the camera line)
          and the goal face (predicted crossing; the measured one too when the
          clip's name carries it, e.g. x-1.0ft_height3.5ft__chip_L_...avi)

    python3 tools/demo_video.py recordings/stereo/2026-11-06/2_chips/x-1.0ft_height3.5ft__chip_L_20261106_115524.avi

By default it is trimmed to the kick (0.6 s before the first tracked frame to
0.6 s after the crossing) and slowed to 0.25x. Use the calibration that was
current when the clip was recorded (--calibration).
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robokeeper.live import KickSession  # noqa: E402
from robokeeper.stereo import clip_swaps_eyes, split_stereo  # noqa: E402
from robokeeper.stereo3d import StereoCalibration  # noqa: E402
from track_ball import load_timestamps  # noqa: E402
from track_stereo import make_tracker  # noqa: E402

FT = 0.3048
BALL_RADIUS_M = 0.11
W, H = 1920, 1080
VIEW_H = 600
# BGR. Same roles as the goal-face chart: blue = predicted, orange = measured.
SURFACE = (251, 252, 252)
TEXT = (11, 11, 11)
TEXT_2 = (78, 81, 82)
GRID = (223, 227, 228)
PREDICTED = (214, 120, 42)
MEASURED = (52, 104, 235)
TRACKED = (0, 230, 255)   # ball circle on the camera views (yellow on gray video)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def measured_from_name(path):
    """(x_ft, height_ft) from a 'x-1.0ft_height3.5ft__' prefix; height 0.36 ft
    (rolling ball center) when only x is given; None without x."""
    label = Path(path).name.split("__")[0] if "__" in Path(path).name else ""
    x = re.search(r"x([+-]?\d+(?:\.\d+)?)ft", label)
    h = re.search(r"height(\d+(?:\.\d+)?)ft", label)
    if not x:
        return None
    return float(x.group(1)), float(h.group(1)) if h else BALL_RADIUS_M / FT


def text(img, s, org, scale=.8, color=TEXT, thick=2):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


class Panel:
    """A plot area mapping (u, v) data coordinates into a rectangle of the frame."""

    def __init__(self, img, rect, u_range, v_range, title, u_label, u_grid=1, v_grid=1):
        self.img, (self.x0, self.y0, self.x1, self.y1) = img, rect
        self.u_range, self.v_range = u_range, v_range
        cv2.rectangle(img, (self.x0, self.y0), (self.x1, self.y1), SURFACE, -1)
        text(img, title, (self.x0, self.y0 - 14), .75, TEXT, 2)
        for u in range(int(np.ceil(u_range[0] / u_grid)) * u_grid, int(u_range[1]) + 1, u_grid):
            x = self.px(u, v_range[0])[0]
            cv2.line(img, (x, self.y0), (x, self.y1), GRID, 1)
            if u % (2 * u_grid) == 0:
                text(img, f"{u}", (x - 8, self.y1 + 24), .5, TEXT_2, 1)
        for v in range(int(np.ceil(v_range[0] / v_grid)) * v_grid, int(v_range[1]) + 1, v_grid):
            y = self.px(u_range[0], v)[1]
            cv2.line(img, (self.x0, y), (self.x1, y), GRID, 1)
            if v % (2 * v_grid) == 0:
                text(img, f"{v}", (self.x0 - 34, y + 6), .5, TEXT_2, 1)
        text(img, u_label, (self.x0 + (self.x1 - self.x0) // 2 - 120, self.y1 + 50), .55, TEXT_2, 1)

    def px(self, u, v):
        (u0, u1), (v0, v1) = self.u_range, self.v_range
        return (int(round(self.x0 + (u - u0) / (u1 - u0) * (self.x1 - self.x0))),
                int(round(self.y1 - (v - v0) / (v1 - v0) * (self.y1 - self.y0))))


def draw_eye(frame, result, trail, label):
    view = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for a, b in zip(list(trail), list(trail)[1:]):
        cv2.line(view, tuple(map(int, a)), tuple(map(int, b)), TRACKED, 3, cv2.LINE_AA)
    if result is not None and result.center is not None and result.observed:
        center = tuple(int(round(v)) for v in result.center)
        cv2.circle(view, center, int(round(result.radius)) + 6, TRACKED, 4, cv2.LINE_AA)
    view = cv2.resize(view, (W // 2, VIEW_H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(view, (0, 0), (300, 44), (0, 0, 0), -1)
    text(view, label, (12, 31), .9, (255, 255, 255), 2)
    return view


def render(views, results, trails, session, measured, now, clock_s, speed):
    img = np.full((H, W, 3), SURFACE, np.uint8)
    img[:VIEW_H, :W // 2] = draw_eye(views[0], results[0], trails[0], "LEFT camera")
    img[:VIEW_H, W // 2:] = draw_eye(views[1], results[1], trails[1], "RIGHT camera")
    cv2.line(img, (W // 2, 0), (W // 2, VIEW_H), SURFACE, 4)

    points = session.points
    prediction = session.prediction[0] if session.prediction else None
    lateral = max(7., max((abs(p["x_m"]) / FT + 1 for p in points), default=0),
                  abs(measured[0]) + 1 if measured else 0)

    # Top-down: x across, z (distance from the camera line) up.
    far = max(20., max((p["z_m"] / FT + 3 for p in points), default=0.))
    top = Panel(img, (110, 690, 760, 1000), (-lateral, lateral), (0, far),
                "Seen from above: distance from the cameras (ft)", "across the camera line (ft)", v_grid=5)
    cv2.line(img, top.px(-lateral, 0), top.px(lateral, 0), TEXT_2, 3)
    text(img, "camera line", (top.x1 - 150, top.px(0, 0)[1] - 8), .55, TEXT_2, 1)
    for p in points:
        cv2.circle(img, top.px(p["x_m"] / FT, p["z_m"] / FT), 5, PREDICTED, -1, cv2.LINE_AA)
    if prediction is not None and points:
        last = points[-1]
        cv2.line(img, top.px(last["x_m"] / FT, last["z_m"] / FT), top.px(prediction.x_m / FT, 0),
                 PREDICTED, 2, cv2.LINE_AA)
        cv2.circle(img, top.px(prediction.x_m / FT, 0), 10, PREDICTED, -1, cv2.LINE_AA)
    if measured:
        cv2.circle(img, top.px(measured[0], 0), 15, MEASURED, 3, cv2.LINE_AA)

    # Goal face: x across, height up.
    face = Panel(img, (900, 690, 1500, 1000), (-lateral, lateral), (0, 9),
                 "Goal face: height above the floor (ft)", "across the camera line (ft)")
    cv2.line(img, face.px(-lateral, 0), face.px(lateral, 0), TEXT_2, 3)
    if measured:
        cv2.circle(img, face.px(*measured), 16, MEASURED, 3, cv2.LINE_AA)
    if prediction is not None:
        x, h = prediction.x_m / FT, (prediction.height_m + BALL_RADIUS_M) / FT
        spread = prediction.x_std_m / FT
        cv2.line(img, face.px(x - spread, h), face.px(x + spread, h), PREDICTED, 2, cv2.LINE_AA)
        cv2.circle(img, face.px(x, h), 10, PREDICTED, -1, cv2.LINE_AA)

    # Readout.
    rx = 1560
    text(img, "Prediction", (rx, 715), .9, TEXT, 2)
    if prediction is not None:
        x, h = prediction.x_m / FT, (prediction.height_m + BALL_RADIUS_M) / FT
        ahead = prediction.time_s - now
        text(img, f"x {x:+.1f} ft", (rx, 765), 1.1, PREDICTED, 3)
        text(img, f"height {h:.1f} ft", (rx, 815), 1.1, PREDICTED, 3)
        text(img, "in the air" if prediction.model == "airborne" else "rolling", (rx, 855), .7, TEXT_2, 2)
        text(img, f"crosses in {ahead:.2f} s" if ahead > 0 else "crossed", (rx, 895), .8, TEXT, 2)
    else:
        text(img, "waiting for the ball" if not points else "tracking...", (rx, 765), .8, TEXT_2, 2)
    if measured:
        text(img, f"measured x {measured[0]:+.1f}, h {measured[1]:.1f} ft", (rx, 950), .6, MEASURED, 2)
    text(img, f"t = {clock_s:.2f} s   ({speed:g}x speed)", (rx, 1000), .6, TEXT_2, 1)

    return img


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path, help="Default: tracking_review/videos/<clip>.mp4")
    parser.add_argument("--calibration", type=Path, default=Path("calibration/stereo.json"))
    parser.add_argument("--speed", type=float, default=.25, help="Playback speed (0.25 = 4x slow motion)")
    parser.add_argument("--full", action="store_true", help="Whole clip instead of trimming to the kick")
    parser.add_argument("--fps", type=float, default=30, help="Output frame rate")
    args = parser.parse_args()

    calibration = StereoCalibration.load(args.calibration)
    options = SimpleNamespace(detector="hybrid", min_radius=3, max_radius=None, no_auto_floor=False,
                              no_appearance_verifier=False, moving_camera=False,
                              min_ball_row=round(calibration.cy - 100), max_gap_s=0.4)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        parser.error(f"Cannot open {args.video}")
    timestamps = load_timestamps(args.video.with_suffix(".csv"), round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    swap = clip_swaps_eyes(args.video)
    measured = measured_from_name(args.video)

    # Pass 1: track every frame (as live_keeper.py, armed from the first frame).
    trackers = [make_tracker(options)[0] for _ in range(2)]
    session = KickSession(calibration, max_wait_s=1e9)
    frames = []
    for timestamp in timestamps:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        views = split_stereo(gray, calibration.image_size, swap)
        results = [t.process(v, timestamp) for t, v in zip(trackers, views)]
        if not session.finished:
            session.update(timestamp, *results)
        snapshot = SimpleNamespace(points=list(session.points), prediction=session.prediction)
        frames.append((timestamp, views, results, snapshot))
    capture.release()
    if not frames:
        parser.error("No frames")

    start, end = frames[0][0], frames[-1][0]
    if not args.full:
        tracked = [t for t, _, results, _ in frames if any(r.observed for r in results)]
        crossing = session.prediction[0].time_s if session.prediction else None
        if tracked:
            start = max(start, tracked[0] - .6)
            end = min(end, (crossing if crossing else tracked[-1]) + .6)

    output = args.output or Path("tracking_review/videos") / f"{args.video.stem}.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    trails = [deque(maxlen=12), deque(maxlen=12)]
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.avi"
        writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"MJPG"), args.fps, (W, H))
        next_out = start
        step = args.speed / args.fps  # clip seconds per output frame
        for index, (timestamp, views, results, snapshot) in enumerate(frames):
            for eye in range(2):
                if results[eye].observed and results[eye].center is not None:
                    trails[eye].append(results[eye].center)
            if timestamp < start or timestamp > end:
                continue
            following = frames[index + 1][0] if index + 1 < len(frames) else timestamp + 1 / 60
            if next_out >= following:
                continue
            image = render(views, results, trails, snapshot, measured, timestamp, timestamp - start, args.speed)
            while next_out < following and next_out <= end:
                writer.write(image)
                next_out += step
        writer.release()
        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", str(output)],
                           check=True)
        else:
            shutil.copy(raw, output.with_suffix(".avi"))
            output = output.with_suffix(".avi")
    final = session.prediction[0] if session.prediction else None
    print(f"Saved {output} ({(end - start) / args.speed:.1f} s at {args.speed:g}x)")
    if final:
        print(f"Final prediction: x {final.x_m / FT:+.2f} ft, height {(final.height_m + BALL_RADIUS_M) / FT:.2f} ft"
              + (f"; measured x {measured[0]:+.2f}, height {measured[1]:.2f} ft" if measured else ""))
    else:
        print("No prediction for this clip (check --calibration matches the day it was recorded)")


if __name__ == "__main__":
    main()
