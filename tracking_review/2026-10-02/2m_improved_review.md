# 2 m tracking review

[Watch the complete annotated clip](2m_improved.mp4). [Sampled frames](2m_improved_contact.jpg).

The review contains all 321 input frames at 60.188 fps, H.264 MP4.
JSONL measurements: [2m_improved.jsonl](2m_improved.jsonl).
Replay uses the original CSV arrival timestamps; playback stays at source FPS.

## Observations

- Acquires the resting ball in the first frame and confirms it at frame 2 (0.040 s).
- Tracks the ball through the kick, with short measurement gaps.
- Does not report observed ball measurements on frames 160–200, when the ball is outside the view.
- Reacquires the partially visible ball at the right edge after it returns.
- Some edge/occlusion frames remain predicted or absent, including the final shoe occlusion.
- No furniture lock-on in the real-clip regression: every post-frame-150 measurement remains near the right edge.

Green circles: confirmed filtered positions. Yellow: tentative or gap prediction.
White dots: raw observations. The header says whether a fresh measurement exists.
No marker means no current position estimate; it does not prove no ball is present.

State counts: {'tentative': 4, 'confirmed': 208, 'predicted': 14, 'absent': 95}. These are **not accuracy/recall measurements**.
Seven manually reviewed center samples (frames 0, 20, 50, 80, 100, 120, 130)
are within 20 pixels in the regression check. This is sparse verification, not
full-frame ground truth. Partial-ball centers at the edge remain less certain.

## Validation and latency

20 automated tests pass, including the real 2 m replay, stationary acquisition,
airborne acquisition/tracking, distractor size rejection, irregular timestamps,
and frame-preserving H.264 export.

A separate tracking-only replay on this host measured roughly 15 ms median and
47 ms p95 processing per frame before export (not a Pi benchmark). The exported
run measured 19.23 ms median and 65.24 ms p95
while tests/export were also consuming resources. Capture/decode/export are not
included in processing_ms. Full-frame reacquisition is more costly than local
tracking; lost-state scans occur every six processed frames. Measure the live
capture-to-result path on the Pi before making latency claims.

The 30–140 px radius limits and soft normalized floor region (0, .45, 1, 1)
are tuned for this clip. The algorithm contains no clip path or ball coordinates.
The new detector has not been validated for 5 m or 11 m accuracy.

Reproduction command is in the project README. Prior JSONL/contact sheets are preserved.
