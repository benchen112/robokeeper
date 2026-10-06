# Motion tracking review — 2026-10-02

The default hybrid detector uses scene-relative motion, circle edges and a grayscale appearance verifier. It compensates small camera movements, checks coherent proposals before initial acquisition, and keeps recent ball appearance/size for two seconds when reacquiring. A floor-like boundary adds a preference; it never masks the search. No video-specific floor rectangle or distance profile was supplied.

Open [the review player](motion_review.html) in a browser to switch between original and annotated video at the same playback time. Each review includes every decoded input frame. Green means a measured track, yellow means a short gap prediction, and white marks a detection. No marker means no current position.

| Recording | Full overlay video | Decoded frames | First measurement frame |
| --- | --- | ---: | ---: |
| 2m | [2m_motion.mp4](2m_motion.mp4) | 321 | 110 |
| 5m | [5m_motion.mp4](5m_motion.mp4) | 432 | 88 |
| 11m | [11m_motion.mp4](11m_motion.mp4) | 401 | 148 |

The 2 m kick has measurements on every frame from 110 through 140. Approximate manual centers at frames 110, 120 and 130 differ by about 3–9 pixels. There are no measured detections in the checked out-of-view interval, frames 155–205. Border reacquisition improves with size memory, but center/radius estimates remain less accurate than for a fully visible ball.

The 5 m clip still briefly acquires the moving foot before the kick, then follows much of the ball flight and return. Its first measurement in the table is therefore a false acquisition, not a correct ball detection. No measured detections occur after frame 295, when the ball has left. Gaps and border uncertainty remain. These sparse manually reviewed checks do not provide a full precision/recall benchmark.

Wider motion-search padding moved the first 11 m acquisition from frame 163 (radius about 27 px) to frame 148 (about 19 px), roughly 0.25 seconds earlier. Small motion blobs need surrounding image context to fit the outer ball circle. The 11 m clip remains the weak case: the classifier rejects some tiny real ball crops, and acquisition begins late. A known miss at frame 130 is recorded as an expected failure. The clip starts at 11 m; its first detection does not establish detection at that physical distance. There is no calibrated distance or 3D trajectory estimate.

The appearance model uses 2 m/5 m ball crops and reviewed shoe/window negatives, with augmented appearance. Crop centers are approximate. The 11 m clip was excluded from model training but used during algorithm development; it is development validation, not an untouched test set. Other environments and ball designs remain unvalidated. The floor boundary is a scene heuristic, not semantic floor segmentation or a calibrated ground plane. A resting ball waits for motion before initial acquisition.

The 5 m source advertises 433 frames and has 433 CSV timestamps, but both OpenCV and FFmpeg decode only 432 frames. Its review contains all 432 readable frames. Timestamp alignment needs caution for this source; no missing image was invented and the source was preserved. Other source/review counts agree. One interrupted MP4 conversion was recovered from the intact intermediate AVI; the export now uses a separate file and verifies frame count before replacing the final file.

| Recording | Median processing ms | 95th percentile ms |
| --- | ---: | ---: |
| 2m | 117.8 | 233.8 |
| 5m | 86.2 | 394.1 |
| 11m | 84.9 | 126.5 |

These CPU timings exclude decode, display and export. Portions of 2 m/5 m replay overlapped regression checks; the serial 11 m run is a less contended reference. This is not a Pi benchmark or a 100 fps pipeline: 100 fps permits 10 ms per frame. The live reader discards stale frames, protecting freshness while reducing observations. CSV times mark arrival, not exposure. Velocity updates now bound spikes from very short arrival intervals; accurate physical trajectory estimates still require appropriate camera timing and calibration.

Run another clip from the project directory:

```bash
python3 track_ball.py --video path/to/clip.mp4 \
  --jsonl tracking_review/new_clip.jsonl \
  --review-video tracking_review/new_clip.mp4
```

If the MP4 was renamed without its CSV, add `--timestamps path/to/original.csv`. Failed conversion retains the intermediate AVI for recovery. Successful export prints the absolute path.

Validation: the final full suite ran 32 tests: 31 passed and one known small-11 m classifier miss is an expected failure. Review frame counts match decoded inputs and JSONL rows. Synthetic checks cover stationary distractors, multiple ball sizes, camera translation, ambiguous floor fallback and identity-memory expiry. See [README](../../README.md) for options and deployment details.
