# Robokeeper ball tracking

This project currently tracks the **2D pixel centroid** of one moving ball. The
prototype runs on the single OV9281 USB camera. Its default capture request is
**1280 × 800 at 60 fps**, matching the planned two-camera OV9281 stereo kit.
The driver may select a different mode; check the mode printed at startup.

The pipeline is camera independent: a frame and its timestamp go into
`BallTracker.process`, and a measured or predicted 2D track comes out. Each
camera in the stereo kit should have its own `BallTracker` instance. A later
stereo stage can match time-aligned left and right measurements and triangulate
3D position. This code does not estimate depth or the goal intersection yet.

## Run on the Pi

Install OpenCV and NumPy:

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy v4l-utils
```

From the project directory, check available camera modes, then run:

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
python3 track_ball.py --device /dev/video0 --display --jsonl detections.jsonl
```

Press **q** or **Esc** in the display to stop. If the camera does not support
the default mode, pass its exact advertised `--width`, `--height`, and `--fps`.
Use `--fourcc MJPG`, `YUYV`, or `auto` to match an advertised pixel format.
Close `camera_stream.py` before opening the same camera here.

For a headless run, omit `--display`. To replay a recording with repeatable
frame timestamps:

```bash
python3 track_ball.py --video shot.mp4 --display --jsonl replay.jsonl
```

`--camera-id` labels output rows (default `left`). `--difference-threshold`,
`--min-area`, and `--max-area` control candidate segmentation. The defaults
are starting values, **not calibrated thresholds for an 11 m shot**.

## Output and latency

Each JSONL row includes `camera_id`, `timestamp_s`, `state`, `observed`,
`centroid_px`, `filtered_centroid_px`, `velocity_px_s`, `radius_px`,
`confidence`, `candidate_count`, `processing_ms`, and `frame_age_ms`.
Coordinates use the original camera frame, with the origin at its top left.

Use a row as a new ball measurement only when `state` is `confirmed` and
`observed` is `true`. `tentative` means fewer than three consecutive matches.
`predicted` is a short gap prediction and has no measured centroid.
`frame_age_ms` measures time from the end of OpenCV's frame read to the end of
tracking. It does **not** include camera exposure, USB transfer, or decoding
before that read returns. Recorded video rows have `frame_age_ms: null`.

The live camera reader retains only the newest frame. If processing falls
behind capture, frames are dropped to keep the reported position current.
This matters especially when both stereo streams run at 60 fps; benchmark the
full capture and inference path on the intended computer.

## Current detector and next data collection step

`MotionBallSegmenter` uses a slowly updated grayscale background. It proposes
moving regions and scores their compactness and local visual contrast. The
tracker associates candidates over time, requires three matches to confirm,
and coasts briefly through occlusion. It suppresses detections when much of
the frame changes at once. It accepts both grayscale and BGR frames, including
the planned monochrome cameras.

This is a baseline segmentation pipeline, **not a trained soccer ball model**.
It cannot reliably distinguish every moving round object from a ball, detect a
stationary ball, or work during substantial camera movement. Camera-specific
footage is needed to measure accuracy and false positives at 11 m and 16.5 m.
When a ball is available, record representative shots at both distances, plus
empty-field and distractor clips, at the intended resolution, frame rate,
exposure, and lighting. Keep the raw frames and timestamps. Those clips will
support threshold tuning and, if needed, a learned segmenter that implements
the same candidate interface without changing the tracker or stereo layer.

Run the synthetic regression checks with:

```bash
python3 -m unittest discover -s tests -v
```

The existing [camera stream instructions](camera_stream_instructions.md) cover
browser viewing of the USB camera.
