#!/usr/bin/env python3
"""Manually record test clips from a V4L2/USB camera on a Raspberry Pi."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import sys
import threading
import time

import cv2


def camera_fourcc(capture):
    value = int(capture.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00")


def read_commands(commands):
    print("Press Enter to start/stop a clip, or type q then Enter to quit.", flush=True)
    while True:
        line = sys.stdin.readline()
        if not line:
            commands.put("quit")
            return
        commands.put("quit" if line.strip().lower() == "q" else "toggle")


class Clip:
    def __init__(self, output_dir, name, size, fps, settings):
        self.video_path = output_dir / f"{name}.avi"
        self.timestamps_path = output_dir / f"{name}.csv"
        self.metadata_path = output_dir / f"{name}.json"
        self.writer = cv2.VideoWriter(
            str(self.video_path), cv2.VideoWriter_fourcc(*"MJPG"), fps, size
        )
        if not self.writer.isOpened():
            raise RuntimeError("Could not create MJPG AVI video; check OpenCV codecs and output path")
        try:
            self.timestamps_file = self.timestamps_path.open("w", newline="", encoding="utf-8")
        except OSError:
            self.writer.release()
            self.video_path.unlink(missing_ok=True)
            raise
        self.csv = csv.writer(self.timestamps_file)
        self.csv.writerow(("frame_index", "timestamp_s", "read_monotonic_ns"))
        self.settings = settings
        self.frame_count = 0
        self.first_ns = None
        self.last_ns = None
        self.first_utc = None

    def write(self, frame, read_ns):
        if self.first_ns is None:
            self.first_ns = read_ns
            self.first_utc = datetime.now(timezone.utc).isoformat()
        self.writer.write(frame)
        self.csv.writerow((self.frame_count, f"{(read_ns - self.first_ns) / 1e9:.9f}", read_ns))
        self.frame_count += 1
        self.last_ns = read_ns

    def close(self):
        self.writer.release()
        self.timestamps_file.close()
        duration = (self.last_ns - self.first_ns) / 1e9 if self.frame_count > 1 else 0.0
        metadata = {
            **self.settings,
            "video": self.video_path.name,
            "timestamps": self.timestamps_path.name,
            "frame_count": self.frame_count,
            "first_frame_utc": self.first_utc,
            "captured_span_s": duration,
            "average_capture_fps": (self.frame_count - 1) / duration if duration > 0 else None,
            "timestamp_note": "Times are recorded after each camera read returns; they are not exposure times.",
        }
        self.metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Saved {self.video_path} ({self.frame_count} frames); timestamps: {self.timestamps_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/video0", help="V4L2 camera device")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--fps", type=float, default=60)
    parser.add_argument("--fourcc", choices=("MJPG", "YUYV", "auto"), default="MJPG")
    parser.add_argument("--output-dir", type=Path, default=Path("recordings"))
    parser.add_argument("--preview", action="store_true", help="Show camera; press r to start/stop, q to quit")
    args = parser.parse_args()
    if args.width <= 0 or args.height <= 0 or args.fps <= 0:
        parser.error("width, height, and fps must be positive")
    if not sys.stdin.isatty() and not args.preview:
        parser.error("manual recording needs a terminal, or use --preview")

    capture = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {args.device}; check its path and close other camera programs")
    clip = None
    try:
        if args.fourcc != "auto":
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        capture.set(cv2.CAP_PROP_FPS, args.fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        settings = {
            "device": args.device,
            "requested_width": args.width,
            "requested_height": args.height,
            "requested_fps": args.fps,
            "requested_fourcc": args.fourcc,
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "video_fps": capture.get(cv2.CAP_PROP_FPS) or args.fps,
            "camera_fourcc": camera_fourcc(capture),
        }
        print(f"Camera: {settings['width']}x{settings['height']} at {settings['video_fps']:.2f} fps, "
              f"{settings['camera_fourcc']}", flush=True)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        commands = queue.SimpleQueue()
        if sys.stdin.isatty():
            threading.Thread(target=read_commands, args=(commands,), daemon=True).start()
        elif args.preview:
            print("Press r in the preview to start/stop; q to quit.", flush=True)
        session = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        clip_number = 0
        failures = 0
        while True:
            ok, frame = capture.read()
            read_ns = time.monotonic_ns()
            if not ok:
                failures += 1
                if failures >= 10:
                    raise RuntimeError("Camera stopped returning frames")
                time.sleep(0.01)
                continue
            failures = 0
            if args.preview:
                shown = frame.copy()
                cv2.putText(shown, "REC" if clip else "READY", (12, 32),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255) if clip else (0, 255, 0), 2)
                cv2.imshow("Camera recorder (r: record/stop, q: quit)", shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("r"):
                    commands.put("toggle")
                elif key in (ord("q"), 27):
                    commands.put("quit")
            quit_requested = False
            while not commands.empty():
                command = commands.get_nowait()
                if command == "quit":
                    quit_requested = True
                    break
                if clip:
                    clip.close()
                    clip = None
                else:
                    clip_number += 1
                    name = f"shot_{session}_{clip_number:03d}"
                    clip = Clip(args.output_dir, name, (frame.shape[1], frame.shape[0]),
                                settings["video_fps"], settings)
                    print(f"Recording {clip.video_path}...", flush=True)
            if quit_requested:
                break
            if clip:
                clip.write(frame, read_ns)
    except KeyboardInterrupt:
        pass
    finally:
        if clip:
            clip.close()
        capture.release()
        if args.preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
