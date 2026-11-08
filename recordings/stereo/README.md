# Stereo recordings

The video files are not in git (too large); copy them from the Pi with `scp`.
This index is. `record_stereo.py` saves into `recordings/stereo/<today>/`, or
`recordings/stereo/<today>/<n>_<test>/` with `--test chips_left`.

Each clip is four files with the same name: `.avi` (replay with
`track_stereo.py --video` or `live_keeper.py --video`), `.csv` (sensor
timestamps), `.json` (exposure, `swap_eyes`), `.mp4` (for viewing). A prefix
before `__` is the hand measurement of where the ball crossed the camera line:
**x** in feet from the midpoint between the cameras (positive = right, seen from
behind the cameras), **height** = ball center above the floor.

| Folder | Clips |
|---|---|
| `2026-10-06/rolls_from_known_distances` | Balls rolled from 2, 4 and 7 m; first stereo calibration (`calibrate_stereo.py kicks`) |
| `2026-11-03/rolls_11m_swapped_eyes` | Rolls from 11 m after the cables were reseated (eyes swapped). x from "35.5 in / 22 in left of the left camera, 33 in right of the right camera" |
| `2026-11-04/kicks_backlit_hallway` | Hallway lit from behind by sunlit doors. x from "33 in right of the right camera, 42 in left of the left camera" |
| `2026-11-06/1_ground_shots` | Slow, fast and short-exposure (498 us, gain 16) ground shots; measurements also in `tracking_review/2026-11-06/ground_truth.json` |
| `2026-11-06/2_chips` | Chips, some short-exposure; heights are rough estimates |
