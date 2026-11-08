#!/usr/bin/env python3
"""Live kick test: press Start on the web page, kick, read the predicted crossing.

The stereo camera runs continuously, but ball tracking only runs between
Start and the result, so the Pi idles cool. After Start, both eyes are
tracked, ball centers are triangulated, and the trajectory predictor keeps
updating where the ball will cross the camera plane. The ball leaves the view
before the plane, so the session ends when the camera clock passes the
predicted crossing time (or after --max-wait-s) and the last prediction is
shown on the page and saved to --runs-dir.

Positions: origin midway between the two lenses, x to the right as seen from
the camera, height up (0 = center of a ball rolling on the ground, with the
provisional calibration), z = perpendicular distance from the camera plane
(the plane through both lenses).

On the Pi (Camarray stereo HAT); exposure is calibrated at startup (see below):

    python3 live_keeper.py
    # open http://<pi-ip>:8000/ and press Start just before kicking

Add --record to also save each armed kick's processed frames as a replayable
clip in --record-dir (AVI + timestamp CSV, as record_stereo.py writes), for
debugging a kick on the laptop with track_stereo.py or live_keeper.py --video.

Add --servo to aim an arm (SG90 on GPIO18 = header pin 12) at the predicted
crossing: it moves on every new prediction and returns to center afterwards.
Check its direction first with tools/servo_test.py.

Headless (tracking only, for timing on the Pi): no web page, preview or
recording. Press Enter in the terminal to arm a kick (q + Enter to quit); every
new prediction prints the servo angle and crossing point, and each result
prints the per-pair processing times. Run files are still saved to --runs-dir.

    python3 live_keeper.py --headless
    python3 live_keeper.py --headless --servo   # also move the arm

With the camera, a fan-header cooler (Pi 5 Active Cooler) runs at full speed
for as long as this runs, then goes back to temperature control (--no-fan to
leave it alone).

Exposure: without --exposure-us, a short calibration at startup picks a fixed
exposure and gain for --target-brightness (shortest exposure first, gain only
past --max-exposure-us) and keeps them for the whole run.

Laptop replay of a recorded clip (arms on the first frame, prints the result):

    python3 live_keeper.py --video recordings/stereo/kick_7m_20261006_161256.avi \
        --autostart --no-serve
"""

import argparse
import atexit
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from http.server import ThreadingHTTPServer
import json
import logging
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np

from camera_stream import Handler
from robokeeper.cooler import FanBoost, cpu_temp_c
from robokeeper.exposure import (add_exposure_arguments, check_exposure_arguments,
                                 open_calibrated_camera)
from robokeeper.live import KickSession
from robokeeper.servo import ServoArm, arm_angle_deg
from robokeeper.stereo import PreviewBuffer, clip_swaps_eyes, split_stereo
from record_stereo import StereoClip
from robokeeper.stereo3d import StereoCalibration
from track_ball import annotate, load_timestamps
from track_stereo import make_tracker


# Segmenter stages timed per eye-frame (robokeeper/hybrid.py); classifier is part of checks.
STAGES = ("blur", "motion", "floor", "hough", "template", "checks", "classifier", "acquisition")


FT_PER_M = 1 / 0.3048


def m_ft(value, signed=True):
    """'+1.52 m (+4.99 ft)': meters with feet alongside."""
    return f"{value:+.2f} m ({value * FT_PER_M:+.2f} ft)" if signed else \
        f"{value:.2f} m ({value * FT_PER_M:.2f} ft)"


class ReplaySource:
    """A recorded packed stereo clip, read as fast as it is processed."""

    def __init__(self, path, timestamps_path=None):
        self.video = cv2.VideoCapture(str(path))
        if not self.video.isOpened():
            raise RuntimeError(f"Cannot open video {path}")
        count = round(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        csv = Path(timestamps_path) if timestamps_path else Path(path).with_suffix(".csv")
        fps = self.video.get(cv2.CAP_PROP_FPS) or 60
        self.timestamps = load_timestamps(csv, count) if csv.exists() else [i / fps for i in range(count)]
        self.index = 0
        self.running = True
        self.error = None

    def read(self, previous_sequence):
        ok, frame = self.video.read()
        if not ok or self.index >= len(self.timestamps):
            self.running = False
            return None
        self.index += 1
        timestamp = self.timestamps[self.index - 1]
        # Exposure and gain are unknown in replay; 0 marks them as such.
        return self.index, frame, timestamp, None, {
            "timestamp_source": "recorded", "sensor_timestamp_ns": round(timestamp * 1e9),
            "exposure_us": 0, "analogue_gain": 0.0}

    def close(self):
        self.video.release()


class LiveKeeper:
    def __init__(self, source, calibration, args, servo=None):
        self.source = source
        self.servo = servo
        self.servo_timer = None
        self.servo_moves = []  # (seconds since Start, angle, predicted seconds to cross)
        self.aimed = None  # the crossing the servo was last pointed at
        self.calibration = calibration
        self.args = args
        min_row = None if args.any_height else (
            args.min_ball_row if args.min_ball_row is not None else round(calibration.cy - 100))
        options = SimpleNamespace(detector="hybrid", min_radius=3, max_radius=None,
                                  no_auto_floor=False, no_appearance_verifier=False,
                                  moving_camera=False, min_ball_row=min_row,
                                  max_gap_s=args.max_gap_s)
        logging.info("New balls must be centered below row %s; track kept over gaps up to %.2f s",
                     min_row, args.max_gap_s)
        self.trackers = [make_tracker(options)[0] for _ in range(2)]
        # Two threads track the eyes side by side; on the laptop the GIL made that
        # slower than one eye after the other (--sequential-eyes), so compare both.
        self.pool = ThreadPoolExecutor(1 if args.sequential_eyes else 2)
        logging.info("Eyes tracked %s", "one after the other" if args.sequential_eyes else "in 2 threads")
        self.lock = threading.Lock()
        self.session = None
        self.history = deque(maxlen=10)
        self.snapshot = None  # latest (views, results) for the preview thread
        self.trails = [deque(maxlen=30), deque(maxlen=30)]
        self.timings = []
        self.breakdown = []  # per processed pair: frame prep and per-eye stage ms
        self.exposures = []  # (exposure_us, analogue_gain) per processed pair
        self.clip = None
        self.encode_pool = ThreadPoolExecutor(2) if args.record else None
        self.skipped = 0
        self.first_ts = self.last_ts = None
        self.error = None
        self.stopping = False

    # --- controls (web threads) ---
    def start_kick(self):
        with self.lock:
            if self.session and not self.session.finished:
                return False, "A kick is already armed"
            for tracker in self.trackers:
                tracker.reset()
            for trail in self.trails:
                trail.clear()
            self.timings, self.exposures, self.breakdown = [], [], []
            self.skipped, self.first_ts, self.last_ts = 0, None, None
            self.session = KickSession(self.calibration, max_wait_s=self.args.max_wait_s,
                                       aim=self._aim)
            self.servo_moves, self.aimed = [], None
            if self.servo:
                if self.servo_timer:
                    self.servo_timer.cancel()
                self.servo.center()
            if self.args.record:
                self.args.record_dir.mkdir(parents=True, exist_ok=True)
                self.clip = StereoClip(self.args.record_dir, "kick", {
                    "requested_fps": self.args.fps, "jpeg_quality": self.args.record_quality,
                    "encoder_threads": 2, "keep_mjpg": False,
                    "frames": "only the pairs the live tracker processed, in order"},
                    self.encode_pool)
            logging.info("Armed: kick when ready")
            return True, "armed"

    def _aim(self, crossing):
        return arm_angle_deg(crossing.x_m, crossing.height_m, crossing.model,
                             pivot_x_m=self.args.servo_x_m,
                             pivot_height_m=self.args.servo_height_m,
                             ball_radius_m=self.args.ball_radius_m,
                             limit_deg=self.args.servo_limit_deg)

    def _point_servo(self, session, timestamp):
        """Move the arm whenever the prediction changes. Caller holds the lock."""
        if session.prediction is None or session.prediction[0] is self.aimed:
            return
        crossing, details = session.prediction
        self.aimed = crossing
        self.servo_moves.append((round(timestamp - session.start, 3), details["servo_deg"],
                                 round(crossing.time_s - timestamp, 3)))
        if self.args.headless:
            print(f"  t {timestamp - session.start:6.3f} s  servo {details['servo_deg']:+6.1f} deg  "
                  f"crossing x {m_ft(details['x_m'])} ±{m_ft(details['x_std_m'], False)}, "
                  f"height {m_ft(details['height_m'])}, in {details['time_to_cross_s']:.3f} s",
                  flush=True)
        if self.servo:
            self.servo.move(details["servo_deg"])

    def _rest_servo(self):
        if self.servo:
            self.servo.center()
            time.sleep(.6)  # let it get there before the pulses stop
            self.servo.relax()

    def cancel(self):
        with self.lock:
            if not self.session or self.session.finished:
                return False, "Nothing is armed"
            self.session.finish("cancelled")
            self._complete(self.session)
            return True, "cancelled"

    def status(self):
        with self.lock:
            session = self.session
            return {
                "state": session.state if session and not session.finished else "idle",
                "servo": None if not self.servo else {"angle_deg": self.servo.angle,
                                                      "dry_run": self.servo.dry_run},
                "session": session.summary() if session else None,
                "stats": self._stats(),
                "history": list(self.history),
                "error": self.error or getattr(self.source, "error", None),
            }

    # --- worker ---
    def _stats(self):
        if not self.timings:
            return None
        span = (self.last_ts - self.first_ts) if self.timings and len(self.timings) > 1 else 0
        exposure, gain = np.array(self.exposures, float).T if self.exposures else ([0], [0])
        return {"pairs_processed": len(self.timings),
                "exposure_us_median": round(float(np.median(exposure))),
                "exposure_us_range": [round(float(np.min(exposure))), round(float(np.max(exposure)))],
                "analogue_gain_median": round(float(np.median(gain)), 2),
                "processed_fps": round((len(self.timings) - 1) / span, 1) if span > 0 else None,
                "pair_ms_p50": round(float(np.percentile(self.timings, 50)), 1),
                "pair_ms_p95": round(float(np.percentile(self.timings, 95)), 1),
                "pair_ms_max": round(float(np.max(self.timings)), 1),
                "camera_frames_skipped": self.skipped,
                "cpu_temp_c": cpu_temp_c()}

    def _complete(self, session):
        """Record a finished session. Caller holds the lock."""
        summary = session.summary()
        summary["stats"] = self._stats()
        summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        summary["servo_moves"] = list(self.servo_moves)
        if self.servo and self.args.servo_return_s >= 0:
            self.servo_timer = threading.Timer(self.args.servo_return_s, self._rest_servo)
            self.servo_timer.daemon = True
            self.servo_timer.start()
        clip, self.clip = self.clip, None
        if clip is not None:
            summary["recording"] = clip.avi_path.name
            # Remuxing takes seconds; do it off the tracking and web threads.
            # Not a daemon, so Ctrl+C still waits for the clip to be written.
            threading.Thread(target=self._finish_clip, args=(clip, session.finished)).start()
        self.history.appendleft(summary)
        crossing = summary["crossing"]
        if crossing:
            logging.info("Result (%s): crosses at x %s ±%s, height %s; %d measurements, "
                         "last at z %s", session.finished, m_ft(crossing["x_m"]),
                         m_ft(crossing["x_std_m"], False), m_ft(crossing["height_m"]),
                         summary["measurements"], m_ft(summary["last_measurement"]["z_m"], False))
        else:
            logging.info("Result (%s): no crossing predicted; %d measurements",
                         session.finished, summary["measurements"])
        if summary["calibration_mismatch"]:
            logging.warning("%s", summary["calibration_mismatch"])
        stats = summary["stats"]
        if stats:
            logging.info("Processing: %d pairs at %s pairs/s, %d camera frames skipped; "
                         "per pair %.1f ms median, %.1f p95, %.1f max; CPU %s C",
                         stats["pairs_processed"], stats["processed_fps"],
                         stats["camera_frames_skipped"], stats["pair_ms_p50"],
                         stats["pair_ms_p95"], stats["pair_ms_max"], stats["cpu_temp_c"])
        stages = self._stage_summary()
        for label in ("ball_tracked", "no_ball"):
            if stages and label in stages:
                group = stages[label]
                logging.info("Stages, %s (%d eye-frames), ms mean/p95: %s", label.replace("_", " "),
                             group["eye_frames"], ", ".join(
                                 f"{k} {m:.1f}/{p:.1f}" for k, (m, p) in group["ms_mean_p95"].items()
                                 if p >= .05))
        if stages:
            logging.info("Per pair: frame prep %.1f ms, thread overhead%s %.1f ms",
                         stages["per_pair"]["frame_prep_ms_mean"],
                         "" if self.args.sequential_eyes else " + waiting for the slower eye",
                         stages["per_pair"]["threads_and_slower_eye_wait_ms_mean"])
        if self.args.runs_dir:
            # One folder per day (runs/2026-11-08/); sort into test folders afterwards.
            now = datetime.now()
            folder = self.args.runs_dir / f"{now:%Y-%m-%d}"
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"kick_{now:%Y%m%d_%H%M%S}.json"
            # Write then rename, so Ctrl+C mid-save cannot leave an empty run file.
            partial = path.with_suffix(".json.partial")
            partial.write_text(json.dumps({**session.record(), "stats": summary["stats"],
                                           "recording": summary.get("recording"),
                                           "servo_moves": summary["servo_moves"],
                                           "timings_ms": [round(t, 1) for t in self.timings],
                                           "stage_summary": stages,
                                           "stage_ms": self.breakdown,
                                           "exposure_us": [e for e, _ in self.exposures],
                                           "analogue_gain": [g for _, g in self.exposures]}) + "\n")
            partial.replace(path)
            logging.info("Saved %s", path)

    @staticmethod
    def _finish_clip(clip, reason):
        try:
            clip.finish(reason)
        except Exception:
            logging.exception("Saving the recording failed; partial files are in %s",
                              clip.avi_path.parent)

    def _track(self, pair):
        tracker, view, timestamp = pair
        tracked = tracker._position is not None  # a ball was being followed coming in
        start = time.perf_counter()
        result = tracker.process(view, timestamp)
        eye = {"ms": round((time.perf_counter() - start) * 1000, 2), "tracked": tracked}
        eye.update((k, round(v, 2)) for k, v in getattr(tracker.segmenter, "stage_ms", {}).items())
        return result, eye

    def _stage_summary(self):
        """Mean and p95 ms per stage, for eye-frames with and without a tracked ball."""
        if not self.breakdown:
            return None
        summary = {}
        eyes = [eye for pair in self.breakdown for eye in pair["eyes"]]
        for label, tracked in (("ball_tracked", True), ("no_ball", False)):
            group = [eye for eye in eyes if eye["tracked"] == tracked]
            if not group:
                continue
            stages = {}
            for stage in STAGES:
                values = np.array([eye.get(stage, 0.) for eye in group])
                stages[stage] = [round(float(values.mean()), 2), round(float(np.percentile(values, 95)), 2)]
            other = np.array([eye["ms"] - sum(eye.get(k, 0.) for k in STAGES if k != "classifier")
                              for eye in group])
            stages["other"] = [round(float(other.mean()), 2), round(float(np.percentile(other, 95)), 2)]
            summary[label] = {"eye_frames": len(group), "ms_mean_p95": stages}
        prep = np.array([pair["prep"] for pair in self.breakdown])
        # Pair time beyond the eyes' own work: the slower eye in 2 threads, both in sequence.
        combine = sum if self.args.sequential_eyes else max
        wait = np.array([pair["pair"] - combine(eye["ms"] for eye in pair["eyes"]) for pair in self.breakdown])
        summary["per_pair"] = {"frame_prep_ms_mean": round(float(prep.mean()), 2),
                               "threads_and_slower_eye_wait_ms_mean": round(float(wait.mean()), 2),
                               "sequential_eyes": bool(self.args.sequential_eyes)}
        return summary

    def run(self):
        sequence = 0
        try:
            while not self.stopping:
                sample = self.source.read(sequence)
                if sample is None:
                    if not self.source.running:
                        break
                    self.error = "No new stereo frame within 2 seconds"
                    continue
                previous = sequence
                sequence, frame, timestamp, _, metadata = sample
                self.error = None
                prep_start = time.perf_counter()
                gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                views = split_stereo(gray, (self.args.eye_width, self.args.eye_height),
                                     self.args.swap_eyes)
                prep_ms = (time.perf_counter() - prep_start) * 1000
                with self.lock:
                    session = self.session if self.session and not self.session.finished else None
                if session is None:
                    self.snapshot = (views, None)
                    continue
                start = time.monotonic()
                tracked = list(self.pool.map(self._track, [(t, v, timestamp)
                                                           for t, v in zip(self.trackers, views)]))
                elapsed_ms = (time.monotonic() - start) * 1000
                results = [result for result, _ in tracked]
                with self.lock:
                    if session is not self.session or session.finished:
                        continue  # cancelled or restarted while processing
                    if self.timings:
                        self.skipped += max(0, sequence - previous - 1)
                    self.first_ts = self.first_ts if self.first_ts is not None else timestamp
                    self.last_ts = timestamp
                    self.timings.append(elapsed_ms)
                    self.breakdown.append({"pair": round(elapsed_ms, 2), "prep": round(prep_ms, 2),
                                           "eyes": [eye for _, eye in tracked]})
                    self.exposures.append((metadata.get("exposure_us") or 0,
                                           metadata.get("analogue_gain") or 0))
                    if self.clip is not None:
                        self.clip.add(sequence, gray, metadata)
                    for trail, result in zip(self.trails, results):
                        if result.observed and result.filtered_center is not None:
                            trail.append(tuple(round(v) for v in result.filtered_center))
                    self.snapshot = (views, results)
                    done = session.update(timestamp, *results)
                    self._point_servo(session, timestamp)
                    if done:
                        self._complete(session)
        except Exception as exc:  # keep the page up to report it
            logging.exception("Tracking stopped")
            self.error = f"Tracking stopped: {exc}"
        finally:
            with self.lock:
                if self.session and not self.session.finished:
                    self.session.finish("video ended" if not self.source.running else "stopped")
                    self._complete(self.session)

    def preview_loop(self, preview):
        while not self.stopping:
            with self.lock:
                armed = self.session is not None and not self.session.finished
                trails = [tuple(t) for t in self.trails]
            fps = self.args.armed_preview_fps if armed else self.args.preview_fps
            if fps <= 0 or self.snapshot is None:
                time.sleep(.2)
                continue
            views, results = self.snapshot
            images = []
            for i, view in enumerate(views):
                image = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
                if results is not None:
                    annotate(image, results[i], trail=trails[i])
                cv2.putText(image, ("LEFT", "RIGHT")[i], (12, 95), cv2.FONT_HERSHEY_SIMPLEX,
                            1.2, (255, 255, 255), 2)
                images.append(image)
            shown = np.hstack(images)
            preview.publish(cv2.resize(shown, (1280, round(shown.shape[0] * 1280 / shown.shape[1]))))
            time.sleep(1 / fps)


PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Robokeeper live</title>
<style>
 body{margin:0 auto;max-width:1100px;padding:16px;background:#10151b;color:#e4ecf4;font:16px/1.45 system-ui,sans-serif}
 h1{font-size:22px;margin:0 0 10px} h2{font-size:17px;margin:18px 0 6px}
 img{width:100%;background:#000;border-radius:8px}
 button{font:inherit;font-size:20px;padding:12px 28px;margin:10px 10px 10px 0;border-radius:8px;border:1px solid #536475;background:#273340;color:inherit;cursor:pointer}
 #start{background:#1d7a3d;border-color:#4fd17d} button:disabled{opacity:.4;cursor:default}
 .state{font-size:20px;font-weight:600} .big{font-size:30px;font-weight:700}
 table{border-collapse:collapse;font-size:15px;width:100%} td,th{border-bottom:1px solid #2c3a48;padding:5px 10px 5px 0;text-align:left}
 .muted{color:#9fb0c0;font-size:14px} .err{color:#ff8a80}
</style></head><body>
<h1>Robokeeper live kick test</h1>
<img src="/stream.mjpg" alt="Live stereo view">
<div><button id="start">Start</button><button id="cancel">Cancel</button>
<span class="state" id="state">…</span> <span class="err" id="error"></span></div>
<div id="live" class="muted"></div>
<h2>Result</h2><div id="result" class="muted">No kick yet.</div>
<h2>Previous kicks</h2>
<table><thead><tr><th>Finished</th><th>Reason</th><th>Crossing x</th><th>Height</th><th>±</th><th>Measurements (z range)</th><th>First prediction</th><th>Processing</th></tr></thead><tbody id="history"></tbody></table>
<p class="muted">Positions are in metres from the midpoint between the two lenses: x is to the right as seen from the camera (negative = left), height is up, with 0 at the centre of a ball rolling on the ground (provisional calibration; a checkerboard calibration will make it relative to the cameras), and z is the perpendicular distance from the camera plane through both lenses. "Range" is the straight-line distance from that midpoint. The crossing is predicted: the ball leaves the view before it reaches the plane.</p>
<script>
const $ = id => document.getElementById(id);
const m = v => v == null ? '–' : (v >= 0 ? '+' : '') + v.toFixed(2) + ' m (' + (v >= 0 ? '+' : '') + (v / 0.3048).toFixed(2) + ' ft)';
const d = v => v.toFixed(2) + ' m (' + (v / 0.3048).toFixed(2) + ' ft)';
function crossingText(c) {
  if (!c) return 'No crossing predicted.';
  return `<div class="big">x ${m(c.x_m)} (${c.x_m >= 0 ? 'right' : 'left'}), height ${m(c.height_m)}</div>
   ±${d(c.x_std_m)} lateral · ${c.speed_mps.toFixed(1)} m/s · ${c.model} · servo ${c.servo_deg}° ·
   from ${c.samples_used} samples, last at z ${d(c.made_at.z_m)} (range ${d(c.made_at.range_m)}),
   ${c.time_to_cross_s.toFixed(2)} s before the predicted crossing`;
}
function stats(s) {
  return s ? `${s.pairs_processed} pairs · ${s.processed_fps ?? '–'} pairs/s · median ${s.pair_ms_p50} ms · p95 ${s.pair_ms_p95} ms · ${s.camera_frames_skipped} camera frames skipped` : '';
}
async function post(path) {
  const r = await fetch(path, {method: 'POST'});
  const body = await r.json();
  if (!body.ok) $('error').textContent = body.message;
  poll();
}
$('start').onclick = () => post('/start');
$('cancel').onclick = () => post('/cancel');
async function poll() {
  let s;
  try { s = await (await fetch('/status')).json(); } catch (e) { $('state').textContent = 'Disconnected'; return; }
  $('state').textContent = {idle: 'Idle – press Start, then kick', armed: 'Armed – waiting for the ball', tracking: 'Tracking'}[s.state];
  $('start').disabled = s.state !== 'idle';
  $('cancel').disabled = s.state === 'idle';
  $('error').textContent = s.error || '';
  const sess = s.session;
  if (sess && s.state !== 'idle') {
    const p = sess.last_measurement;
    $('live').innerHTML = (p ? `Latest ball position: x ${m(p.x_m)}, height ${m(p.height_m)}, z ${d(p.z_m)} (range ${d(p.range_m)}) · ${sess.measurements} measurements` : 'No 3D measurement yet') +
      '<br>' + stats(s.stats) + (sess.crossing ? '<br>Current prediction: ' + crossingText(sess.crossing) : '');
  } else $('live').textContent = '';
  const last = s.history[0];
  const moves = last && last.servo_moves && last.servo_moves.length ? `<br>Servo moved ${last.servo_moves.length}× · first at ${last.servo_moves[0][1]}°, ${last.servo_moves[0][2].toFixed(2)} s before the predicted crossing · final ${last.servo_moves.at(-1)[1]}°` : '';
  if (last) $('result').innerHTML = `<b>${last.reason}</b> · ${last.finished_at}<br>` + crossingText(last.crossing) + moves +
     '<br><span class="muted">' + stats(last.stats) + '</span>';
  $('history').innerHTML = s.history.map(h => `<tr><td>${h.finished_at.slice(11)}</td><td>${h.reason}</td>
    <td>${h.crossing ? m(h.crossing.x_m) : '–'}</td><td>${h.crossing ? m(h.crossing.height_m) : '–'}</td>
    <td>${h.crossing ? h.crossing.x_std_m.toFixed(2) : '–'}</td>
    <td>${h.measurements}${h.first_measurement ? ` (${h.first_measurement.z_m.toFixed(1)} → ${h.last_measurement.z_m.toFixed(1)} m)` : ''}</td>
    <td>${h.first_prediction ? `at z ${h.first_prediction.made_at.z_m.toFixed(1)} m, ${h.first_prediction.time_to_cross_s.toFixed(2)} s ahead, x ${m(h.first_prediction.x_m)}` : '–'}</td>
    <td>${h.stats ? `${h.stats.processed_fps ?? '–'} pairs/s` : '–'}</td></tr>`).join('');
}
setInterval(poll, 300); poll();
</script></body></html>
"""


def arm_from_terminal(keeper):
    """Headless control: Enter arms a kick, q + Enter quits."""
    print("Press Enter to arm a kick (q + Enter to quit)", flush=True)
    for line in sys.stdin:
        if line.strip().lower() in ("q", "quit", "exit"):
            break
        ok, message = keeper.start_kick()
        if not ok:
            print(message, flush=True)
    keeper.stopping = True


class KeeperHandler(Handler):
    keeper = None

    def _json(self, payload, code=200):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/status":
            self._json(self.keeper.status())
        else:
            super().do_GET()

    def do_POST(self):
        actions = {"/start": self.keeper.start_kick, "/cancel": self.keeper.cancel}
        if self.path not in actions:
            self.send_error(404)
            return
        ok, message = actions[self.path]()
        self._json({"ok": ok, "message": message})

    def log_message(self, *args):
        pass  # status polling would flood the terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, help="Replay a packed stereo clip instead of the camera")
    parser.add_argument("--timestamps", type=Path, help="Replay timestamp CSV (default: same name)")
    parser.add_argument("--autostart", action="store_true", help="Arm on the first frame")
    parser.add_argument("--no-serve", action="store_true", help="No web page (replay testing)")
    parser.add_argument("--headless", action="store_true",
                        help="Tracking only: no web page, preview or recording; arm with Enter")
    parser.add_argument("--calibration", type=Path, default=Path("calibration/stereo.json"))
    parser.add_argument("--camera-num", type=int, default=0)
    parser.add_argument("--eye-width", type=int, default=1280)
    parser.add_argument("--eye-height", type=int, default=800)
    parser.add_argument("--fps", type=float, default=60)
    add_exposure_arguments(parser)
    parser.add_argument("--tuning-file", help="Optional Picamera2 tuning JSON")
    parser.add_argument("--swap-eyes", action="store_true", help="Label the right half as left")
    parser.add_argument("--min-ball-row", type=int,
                        help="Ignore new balls centered above this pixel row, such as heads "
                             "(default: 100 px above the calibration's horizon row)")
    parser.add_argument("--any-height", action="store_true",
                        help="Acquire balls anywhere in the image (no --min-ball-row)")
    parser.add_argument("--max-gap-s", type=float, default=0.4,
                        help="Keep a track across processing gaps up to this long")
    parser.add_argument("--max-wait-s", type=float, default=8.0,
                        help="End an armed kick with no predicted crossing after this long")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"),
                        help="Save each kick as JSON in a dated folder here (runs/YYYY-MM-DD/)")
    parser.add_argument("--record", action="store_true",
                        help="Save each armed kick's processed frames as a replayable clip")
    parser.add_argument("--record-dir", type=Path, default=Path("recordings/live"))
    parser.add_argument("--record-quality", type=int, default=95, help="JPEG quality 1-100")
    servo = parser.add_argument_group("servo (arm in the goal plane; 0 deg = straight down)")
    servo.add_argument("--servo", action="store_true", help="Drive the servo on --servo-pin")
    servo.add_argument("--servo-dry-run", action="store_true",
                       help="Compute and log servo angles without GPIO (laptop testing)")
    servo.add_argument("--servo-pin", type=int, default=18, help="BCM GPIO number (18 = pin 12)")
    servo.add_argument("--servo-min-us", type=int, default=500, help="Pulse width at -90 deg")
    servo.add_argument("--servo-max-us", type=int, default=2500, help="Pulse width at +90 deg")
    servo.add_argument("--servo-invert", action="store_true",
                       help="Flip direction (see tools/servo_test.py)")
    servo.add_argument("--servo-limit-deg", type=float, default=90.0)
    servo.add_argument("--servo-x-m", type=float, default=0.0,
                       help="Servo shaft sideways from the camera midpoint (+ = camera's right)")
    servo.add_argument("--servo-height-m", type=float, default=0.30,
                       help="Servo shaft height above the ground")
    servo.add_argument("--ball-radius-m", type=float, default=0.11,
                       help="Ball radius (size 5 = 0.11 m)")
    servo.add_argument("--servo-return-s", type=float, default=3.0,
                       help="Return to center this long after a result (-1 = stay)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preview-fps", type=float, default=8, help="Preview rate while idle")
    parser.add_argument("--armed-preview-fps", type=float, default=3,
                        help="Preview rate while tracking (0 = off, frees CPU for tracking)")
    parser.add_argument("--opencv-threads", type=int, default=2)
    parser.add_argument("--sequential-eyes", action="store_true",
                        help="Track the two eyes one after the other instead of in 2 threads")
    parser.add_argument("--no-fan", action="store_true",
                        help="Leave the cooler under the kernel's temperature control")
    args = parser.parse_args()
    if args.headless and args.record:
        parser.error("--headless measures tracking alone; drop --record")
    if args.no_serve and not args.headless and not (args.video and args.autostart):
        parser.error("--no-serve is for replay with --video --autostart")
    args.no_serve = args.no_serve or args.headless
    args.autostart = args.autostart or bool(args.headless and args.video)
    check_exposure_arguments(parser, args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    cv2.setNumThreads(args.opencv_threads)
    calibration = StereoCalibration.load(args.calibration)
    fan = None if args.video or args.no_fan else FanBoost()
    if fan and fan.start():
        atexit.register(fan.close)  # also when startup fails before the finally below
    if args.video:
        source = ReplaySource(args.video, args.timestamps)
        if clip_swaps_eyes(args.video) and not args.swap_eyes:
            args.swap_eyes = True
            logging.info("Clip is labelled swap_eyes; swapping left and right halves")
    else:
        if calibration.swap_eyes and not args.swap_eyes:
            args.swap_eyes = True
            logging.info("Calibration says the eyes arrive swapped (cabling); swapping left and right")
        source = open_calibrated_camera(args)
        if args.exposure_us is not None:
            sample = source.read(0)
            if sample is not None and args.runs_dir:
                # A look at the calibrated view, since headless runs show no image.
                args.runs_dir.mkdir(parents=True, exist_ok=True)
                check = args.runs_dir / "camera_check.jpg"
                cv2.imwrite(str(check), cv2.resize(sample[1], (1280, 400), interpolation=cv2.INTER_AREA))
                logging.info("Saved %s (calibrated view, both eyes)", check)
    arm = None
    if args.servo or args.servo_dry_run:
        arm = ServoArm(args.servo_pin, min_pulse_us=args.servo_min_us,
                       max_pulse_us=args.servo_max_us, invert=args.servo_invert,
                       limit_deg=args.servo_limit_deg, dry_run=not args.servo)
        arm.center()
        logging.info("Servo on GPIO%d%s: shaft %.2f m up, %+.2f m sideways; ball radius %.2f m",
                     args.servo_pin, " (dry run)" if arm.dry_run else "", args.servo_height_m,
                     args.servo_x_m, args.ball_radius_m)
    keeper = LiveKeeper(source, calibration, args, arm)
    preview = server = None
    try:
        if not args.no_serve:
            preview = PreviewBuffer()
            handler = type("Handler", (KeeperHandler,), {"camera": preview, "keeper": keeper})
            server = ThreadingHTTPServer((args.host, args.port), handler)
            server.daemon_threads = True
            threading.Thread(target=server.serve_forever, daemon=True).start()
            threading.Thread(target=keeper.preview_loop, args=(preview,), daemon=True).start()
            logging.info("Open http://<pi-ip>:%d/ (find the IP with hostname -I); Ctrl+C to stop",
                         args.port)
        if args.autostart:
            keeper.start_kick()
        if args.headless and not args.video:
            threading.Thread(target=arm_from_terminal, args=(keeper,), daemon=True).start()
        keeper.run()
        if args.video and not args.no_serve:
            logging.info("Replay finished; page stays up until Ctrl+C")
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        keeper.stopping = True
        if server:
            server.shutdown()
            server.server_close()
        if preview:
            preview.close()
        source.close()
        keeper.pool.shutdown()
        if arm:
            if keeper.servo_timer:
                keeper.servo_timer.cancel()
            arm.close()
        if keeper.encode_pool:
            keeper.encode_pool.shutdown()
        if fan:
            fan.close()
    if args.no_serve and not args.headless:
        print(json.dumps(keeper.status()["history"][:1], indent=1))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        logging.error("%s", exc)
        sys.exit(1)
