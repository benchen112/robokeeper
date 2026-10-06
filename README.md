# Robokeeper ball tracking

This project currently tracks the **2D pixel centroid** of one ball. The default detector requires motion to acquire it. The
single-camera runner uses the OV9281 USB camera. Its default capture request is
**1280 × 800 at 60 fps**, matching the planned two-camera OV9281 stereo kit.
The driver may select a different mode; check the mode printed at startup.

The pipeline is camera independent: a frame and its timestamp go into
`BallTracker.process`, and a measured or predicted 2D track comes out. Each
camera in the stereo kit should have its own `BallTracker` instance. A later
stereo stage can match time-aligned left and right measurements and triangulate
3D position. This code does not estimate depth or the goal intersection yet.

## Test the connected Camarray stereo HAT

`track_stereo.py` runs the existing pipeline independently on both OV9281
views. The default input is the **CSI Camarray HAT via Picamera2**, with one
packed **2560 × 800 at 60 fps** stream split into two **1280 × 800** eyes.
This is the mode advertised for the kit in
[Arducam's Camarray documentation](https://docs.arducam.com/Raspberry-Pi-Camera/Multi-Camera-CamArray/Multi-Camera-CamArray/).
It does not open two camera indices for this HAT. Physical left/right labeling
can be reversed with `--swap-eyes`; confirm it by covering one lens.

Close other camera programs first. On the Pi, with the Arducam driver already
configured, check discovery and start the test:

```bash
rpicam-hello --list-cameras
python3 track_stereo.py --serve --jsonl stereo_runs/first_test.jsonl
```

Open `http://<pi-ip>:8000/` to see both annotated views. Move or roll the ball
to acquire it: the default hybrid pipeline requires motion. Start nearby,
then test at 5 m and 11 m with the actual ball and lighting. Green markers
are confirmed observations; yellow markers are tentative tracks or predictions.
Press Ctrl+C in the terminal to stop. For a local desktop window, replace
`--serve` with `--display` (q or Esc stops it). Omit both to benchmark headlessly:

```bash
python3 track_stereo.py --duration 30 --jsonl stereo_runs/headless_test.jsonl
```

Install `python3-picamera2`, `python3-opencv`, and `python3-numpy` from the Pi's
OS packages if needed. Driver/overlay setup remains specific to your Arducam
HAT and OS; use the
[Arducam quick start](https://docs.arducam.com/Raspberry-Pi-Camera/Multi-Camera-CamArray/quick-start/)
if `rpicam-hello --list-cameras` does not list it. `--camera-num` selects the
Picamera2 device index when another CSI camera is also present.

If libcamera reports a missing `arducam-pivariety_mono.json`, you can explicitly
use the installed OV9281 monochrome tuning with
`--tuning-file ov9281_mono.json`. The runner does not change OS tuning files.
For fast-ball tests, use `--exposure-us 3000 --gain 2` as an initial manual
exposure experiment, then adjust for your lighting. A shorter exposure reduces
blur but needs more light. These flags disable automatic exposure for that run;
omit them to restore automatic exposure. JSONL includes actual exposure, gain
and each eye's mean luma to help diagnose dark input.

`--eye-width` and `--eye-height` describe **each eye**, not the packed frame.
For example, `--eye-width 640 --eye-height 400 --fps 150` requests the kit's
1280 × 400 packed mode. Use full resolution first for small distant balls.
The code requests native 8-bit capture directly to avoid an invalid advertised
640 × 200 mode that can break Picamera2's full mode enumeration on this driver.
It checks the configured raw and processed sizes and refuses unexpected sizes
rather than splitting a scaled image. Tracking always uses full eye resolution;
only the preview is downsampled. Preview defaults to 10 fps to limit JPEG work.
The server listens on all interfaces; pass `--host 127.0.0.1` for local use.

Each JSONL line contains `left` and `right` results in **eye-local pixel
coordinates**, the same frame timestamp for both eyes, `capture_sequence`,
`pair_processing_ms`, `pair_frame_age_ms`, and cumulative
`skipped_capture_frames`. Existing output paths are refused; choose a new name
for each run. Console reports include pair throughput, recent processing
p50/p95, states, and confirmed-observation counts. These counts measure tracker
availability, not accuracy; visually check the markers on the real ball.

The CSI timestamp uses `SensorTimestamp` when provided; the JSONL records its
source, exposure and frame duration metadata. The reader retains only the
newest **whole stereo frame**, dropping old pairs when inference falls behind.
`pair_frame_age_ms` measures receive-to-tracking completion; it excludes sensor
exposure and transfer/ISP work before capture returns. Processing time excludes
JSONL writes and preview work, but those still affect throughput and skipped
frames. Left and right inference run sequentially on a shared input pair.

`both_confirmed_observed` requires current confirmed observations in both
eyes. `raw_disparity_px` (left x minus right x) and `raw_vertical_offset_px`
are inspection diagnostics only. They are unrectified, do not prove that both
trackers found the same object, and must not be used as depth or motor commands.
Stereo calibration, correspondence validation, triangulation and goal-plane
prediction are still required for the robot goalkeeper.

The same runner can replay a packed stereo recording, preserving a same-name
CSV sidecar or one specified with `--timestamps`, or use a packed USB feed:

```bash
python3 track_stereo.py --video stereo_clip.mp4 --display --jsonl stereo_runs/replay.jsonl
python3 track_stereo.py --device /dev/video0 --fourcc MJPG --serve
```

Use `--detector appearance` for stationary-acquisition comparison or
`--detector motion` for the original motion baseline. `--show-candidates`,
`--min-radius`, `--max-radius`, `--no-auto-floor`, and
`--no-appearance-verifier` support the same hybrid diagnostics as the
single-camera runner.

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

## Record test clips on the Pi

With the USB camera plugged in and no other program using it, run:

```bash
python3 record_camera.py --device /dev/video0 --width 1280 --height 800 --fps 100 --record-fps 60
```

Open `http://<pi-ip>:8000/` in a browser on the same network. The page shows a
live preview before and during recording. Enter a video name, then click
**Start recording** and **Stop recording** for each clip. Spaces and punctuation
in names become underscores; a timestamp is appended to prevent overwrites.
The **Saved FPS** field defaults to 60 and can be changed for each clip to any
positive rate no higher than the camera rate. The recorder evenly selects
frames from the 100 fps input; at 60 fps it keeps 60 of every 100 frames.
The preview shows at most 10 fps by default to limit browser traffic; recording
uses the selected saved rate. Use `--preview-fps` to change the preview.
Press Ctrl+C in the Pi terminal to shut down. Use `--host 127.0.0.1` for access
only from the Pi, or `--port` to change the web port. The requested mode is
printed at startup; choose a mode advertised by
`v4l2-ctl -d /dev/video0 --list-formats-ext` if it differs from the request.
The page shows the requested camera rate and measured saved FPS.
This USB OV9281 advertises 100 and 120 fps at 1280×800 MJPG, so the recorder
requests 100 fps by default. It does not advertise 60 fps at that mode.
For example, 100 fps over 1.5 seconds should deliver about 150 camera frames;
the default 60 fps recording saves about 90 of them. This recorder copies the
selected compressed MJPEG packets, avoiding the former decode and re-encode
bottleneck. The browser also reports the live camera rate, so you can check
whether the input itself keeps up.

Install FFmpeg for compressed MJPEG camera capture and the final H.264 MP4 files:

```bash
sudo apt install -y ffmpeg
```

Each clip goes into `recordings/` as a viewable H.264 `.mp4`, a `.csv` with one
timestamp per saved frame, and a `.json` with the camera and recording rates.
The camera's compressed MJPEG frames are selected without decoding; H.264
conversion happens after recording stops. The temporary `.mjpg` is removed
after the MP4's frame count is verified. Use `--keep-raw` to retain it for
analysis. You can also delete `.mjpg` files from older successful recordings
after checking their MP4s; keep the CSV and JSON sidecars. If conversion fails,
the `.mjpg` is preserved. The CSV timestamps mark arrival from FFmpeg's pipe,
not exposure. Playback uses the selected rate when capture keeps up and the
measured duration if it falls behind.
Replay a clip with
`python3 track_ball.py --video recordings/<clip>.mp4 --display`.

To correct a clip made by an earlier version of the recorder, keep its `.json`
sidecar beside the MP4 and run:

```bash
python3 record_camera.py --repair-video recordings/<clip>.mp4
```

This saves a new `<clip>_timing_fixed.mp4` without changing the original.

`--camera-id` labels output rows (default `left`). The same detector can be
used independently for both stereo cameras; geometry and timing calibration
will still be needed for triangulation.

## Output and latency

Each JSONL row includes `camera_id`, `timestamp_s`, `state`, `observed`,
`centroid_px`, `filtered_centroid_px`, `velocity_px_s`, `radius_px`,
`confidence`, `candidate_count`, `processing_ms`, and `frame_age_ms`.
Coordinates use the original camera frame, with the origin at its top left.

Use a row as a new ball measurement only when `state` is `confirmed` and
`observed` is `true`. `tentative` means the tracker has not yet accumulated its configured matches.
The hybrid detector verifies coherent proposals over at least three frames
before allowing acquisition; the other modes confirm in the tracker.
`predicted` is a short gap prediction and has no measured centroid.
`frame_age_ms` measures time from the end of OpenCV's frame read to the end of
tracking. It does **not** include camera exposure, USB transfer, or decoding
before that read returns. Recorded video rows have `frame_age_ms: null`.

The live camera reader retains only the newest frame. If processing falls
behind capture, frames are dropped to keep the reported position current.
This matters especially when both stereo streams run at 60 fps; benchmark the
full capture and inference path on the intended computer.

## Motion tracking and video review

The default `--detector hybrid` combines motion regions, circle boundaries,
local appearance matching, and a small grayscale HOG appearance verifier.
It aligns consecutive frames using sparse optical flow and a robust homography
before measuring motion, reducing distractions from camera shake. Persistent,
coherent movement or expansion is required for initial acquisition. A two-second
identity memory checks size and retained appearance during reacquisition; strong
matching motion proposals can recover axial motion with little centroid change. It waits for a
resting ball to move; use `--detector appearance` for the older stationary
acquisition baseline.

The inference code contains no recording-specific ball coordinates or distance
profiles. Its default minimum radius is 3 pixels; the maximum derives from
image dimensions. Optional `--min-radius` and `--max-radius` are pixel limits,
not measured distances. Small circles inside a larger moving region are
penalized to reduce confusion with shoes and patches on clothing.

The automatic floor hint looks for a stable horizontal transition above a
smoother bottom-connected region. It reports no estimate in ambiguous scenes.
This is a floor-like image boundary, not semantic floor segmentation or a
calibrated ground plane. It only influences acquisition ranking and motion
requirements: the search still includes airborne balls. Use `--no-auto-floor`
to disable it or `--floor-region X0 Y0 X1 Y1` for an optional normalized
rectangle override.

Run the same defaults on another recording:

```bash
python3 track_ball.py \
  --video recordings/test_5m_onground.mp4 \
  --timestamps recordings/test_5m_onground_20261002_104807_492821.csv \
  --jsonl tracking_review/2026-10-02/5m_motion.jsonl \
  --review-video tracking_review/2026-10-02/5m_motion.mp4
```

Output directories are created automatically. Paths are relative to the working
directory; run from the project directory or use absolute paths. FFmpeg must be
installed for MP4 export. Replay automatically loads a same-name `.csv` sidecar
when available. If you rename only the MP4, specify its original CSV with
`--timestamps`. Invalid or mismatched timestamp files fail rather than silently
inventing timing. Without a sidecar, timestamps use frame index / video FPS.
CSV times mark frame arrival, not exposure. Velocity updates limit the effect
of unusually short arrival intervals using recent frame timing; pixel velocity
still requires exposure timestamps for accurate physical trajectory estimation.

The review MP4 includes every processed input frame at the input playback FPS.
Green circles show measured tracks, yellow circles show brief gap predictions,
and white dots show raw measurements. The trail shows recent observations.
The header reports frame index, timestamp, tracking state and floor confidence;
a dashed line marks the estimated floor boundary. An absent frame has no
position marker. Add `--show-candidates` for proposal boxes or `--display` to
watch during processing. Tracking `processing_ms` excludes export and display.

The appearance verifier is trained on labeled 2 m and 5 m crops of this ball
and environment, with lighting, rotation and resolution augmentation. The
11 m recording is excluded from model training. Model files are stored in
`robokeeper/models/`; copy that directory when deploying the module. Reproduce
training with `python3 tools/train_ball_verifier.py`. A high appearance score
or `confirmed` state does not guarantee correct ball identity. These clips do
not establish accuracy in other environments or with other ball designs.
`--no-appearance-verifier` runs the hybrid geometry/motion checks alone for
comparison, with a greater risk of confusing moving distractors with the ball.

Use `--detector motion` for the original background-difference baseline.
`--min-area`, `--max-area`, `--warmup-frames`, and `--difference-threshold` apply
only to that mode. `BallTracker()` retains its original motion default for API
compatibility. To use the hybrid pipeline directly:

```python
from robokeeper import BallTracker, HybridBallSegmenter
tracker = BallTracker(HybridBallSegmenter(), confirmation_hits=1, adaptive_gate=True)
result = tracker.process(frame, timestamp_s)
```

Each camera needs independent segmenter and tracker state. Frame timestamps
must increase; long gaps discard stale tracks. The live reader retains the
newest frame rather than queuing old frames. Camera frame rate is not inference
throughput: benchmark capture and processing on the intended computer before
relying on 100 fps operation. Distant small balls, partial visibility,
occlusions and reflections can still cause gaps or false detections.

Run synthetic, recording/export, model-crop and sparse real-clip checks with
the command below. One expected failure records the known small-ball classifier
miss at frame 130 of the 11 m clip; it is not counted as successful detection.

```bash
python3 -m unittest discover -s tests -v
```

The existing [camera stream instructions](camera_stream_instructions.md) cover
browser viewing of the USB camera.
