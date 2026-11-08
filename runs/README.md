# Live keeper runs

Each `kick_YYYYMMDD_HHMMSS.json` is one armed kick from `live_keeper.py`: the 3D
points, every prediction, timing per stage, exposure and the calibration used.
`live_keeper.py` saves new runs into a folder for the day (`runs/2026-11-08/`);
move them into a test folder below and add a row here.

Positions are in feet, as measured by hand: **x** from the midpoint between the
cameras along the camera line (positive = right, seen from behind the cameras);
**height** = ball center above the floor. In file names, `x+4.0ft` or
`height4.9ft` is the hand measurement, not the prediction.

Note: the Pi has no clock battery. The `kick_20261007_01*` runs were made on
2026-10-30 with the clock still set to Oct 7 (check `date` on the Pi before a
session).

## 2026-11-08: left lens refocused, new roll calibration, Pi at ~50-58 pairs/s

| Folder | Kick | Measured | Predicted | Error |
|---|---|---|---|---|
| `1_fast_ground_two_threads` | warm-ups 1-3 | not measured | +4.3, none (not tracked), +2.2 | |
| | right1 | x +4.0 | x +3.68 | -0.32 |
| | right2 | x +3.8 | x +3.15 | -0.65 |
| | left1 | x -4.2 | x -3.36 | +0.84 |
| | left2 | x -3 to -4 | x -4.03 | ~0 |
| `2_fast_ground_sequential_eyes` | right1 | x +6.0 | x +5.26 | -0.74 |
| | right2 | x +4.7 | x +5.21 | +0.51 |
| | left1 | x -4.5 | x -4.66 | -0.16 |
| | left2 | x -4.2 | x -4.34 | -0.14 |
| `3_chips_two_threads` | chip0 | not measured | x -3.38, height 5.38 | |
| | chip1 | height 4.9 | x -4.28, height 5.27 | +0.37 |
| | chip2 | height 3.0 | x -3.14, height 3.13 | +0.13 |
| | chip3 | height 3.8 | x -3.81, height 3.74 | -0.06 |

Ground shots: errors about +-0.5 ft, usable predictions 0.25-0.38 s before
crossing. Chips: first time tracked in both cameras; height within 0.5 ft about
0.4-0.5 s before crossing. Sequential eyes gave no speed gain on the Pi.

## Earlier days

| Folder | What it was |
|---|---|
| `2026-11-06/mixed_slow_fast_and_chips` | Live: slow ground shots accurate; fast ground shots and chips gave too few tracked frames to predict (blurry left lens, ~30 pairs/s). Matching clips: `recordings/stereo/*_20261106_*`, measurements in `tracking_review/2026-11-06/ground_truth.json` |
| `2026-11-05/first_successful_prediction` | First live prediction (x +1.52 m), after the new mount and roll calibration |
| `2026-11-04/headless_no_predictions` | Backlit hallway; camera aim had moved, so the old calibration matched nothing |
| `2026-10-30/1_headless_overexposed` | First headless runs; image saturated |
| `2026-10-30/2_headless_exposure_ok_no_detections` | Exposure fixed, still no predictions (eyes swapped since the cables were reseated) |
| `2026-10-07/house_kicks_no_servo` | Kicks at the house, servo not connected |
| `2026-10-06/1_first_live_kicks` | First live kicks with the web preview |
| `2026-10-06/2_indoor_kicks_with_recordings` | Same, with `--record` clips (`recordings/live/`) |
| `2026-10-06/3_indoor_kicks_fewer_points_before_prediction` | After lowering the points needed before the first prediction |
