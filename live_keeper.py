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

On the Pi (Camarray stereo HAT), same exposure as the recorded kicks:

    python3 live_keeper.py
    # open http://<pi-ip>:8000/ and press Start just before kicking

Add --record to also save each armed kick's processed frames as a replayable
clip in --record-dir (AVI + timestamp CSV, as record_stereo.py writes), for
debugging a kick on the laptop with track_stereo.py or live_keeper.py --video.

Laptop replay of a recorded clip (arms on the first frame, prints the result):

    python3 live_keeper.py --video recordings/stereo/kick_7m_20261006_161256.avi \
        --autostart --no-serve
"""

import argparse
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
from robokeeper.live import KickSession
from robokeeper.stereo import LatestPicameraStereo, PreviewBuffer, split_stereo
from record_stereo import StereoClip
from robokeeper.stereo3d import StereoCalibration
from track_ball import annotate, load_timestamps
from track_stereo import make_tracker


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
    def __init__(self, source, calibration, args):
        self.source = source
        self.calibration = calibration
        self.args = args
        options = SimpleNamespace(detector="hybrid", min_radius=3, max_radius=None,
                                  no_auto_floor=False, no_appearance_verifier=False,
                                  moving_camera=False)
        self.trackers = [make_tracker(options)[0] for _ in range(2)]
        self.pool = ThreadPoolExecutor(2)
        self.lock = threading.Lock()
        self.session = None
        self.history = deque(maxlen=10)
        self.snapshot = None  # latest (views, results) for the preview thread
        self.trails = [deque(maxlen=30), deque(maxlen=30)]
        self.timings = []
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
            self.timings, self.exposures = [], []
            self.skipped, self.first_ts, self.last_ts = 0, None, None
            self.session = KickSession(self.calibration, max_wait_s=self.args.max_wait_s)
            if self.args.record:
                self.args.record_dir.mkdir(parents=True, exist_ok=True)
                self.clip = StereoClip(self.args.record_dir, "kick", {
                    "requested_fps": self.args.fps, "jpeg_quality": self.args.record_quality,
                    "encoder_threads": 2, "keep_mjpg": False,
                    "frames": "only the pairs the live tracker processed, in order"},
                    self.encode_pool)
            logging.info("Armed: kick when ready")
            return True, "armed"

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
                "camera_frames_skipped": self.skipped}

    def _complete(self, session):
        """Record a finished session. Caller holds the lock."""
        summary = session.summary()
        summary["stats"] = self._stats()
        summary["finished_at"] = datetime.now().isoformat(timespec="seconds")
        clip, self.clip = self.clip, None
        if clip is not None:
            summary["recording"] = clip.avi_path.name
            # Remuxing takes seconds; do it off the tracking and web threads.
            # Not a daemon, so Ctrl+C still waits for the clip to be written.
            threading.Thread(target=self._finish_clip, args=(clip, session.finished)).start()
        self.history.appendleft(summary)
        crossing = summary["crossing"]
        if crossing:
            logging.info("Result (%s): crosses at x %+.2f m (±%.2f), height %+.2f m; "
                         "%d measurements, last at z %.2f m", session.finished, crossing["x_m"],
                         crossing["x_std_m"], crossing["height_m"], summary["measurements"],
                         summary["last_measurement"]["z_m"])
        else:
            logging.info("Result (%s): no crossing predicted; %d measurements",
                         session.finished, summary["measurements"])
        if self.args.runs_dir:
            self.args.runs_dir.mkdir(parents=True, exist_ok=True)
            path = self.args.runs_dir / f"kick_{datetime.now():%Y%m%d_%H%M%S}.json"
            # Write then rename, so Ctrl+C mid-save cannot leave an empty run file.
            partial = path.with_suffix(".json.partial")
            partial.write_text(json.dumps({**session.record(), "stats": summary["stats"],
                                           "recording": summary.get("recording"),
                                           "timings_ms": [round(t, 1) for t in self.timings],
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
        return tracker.process(view, timestamp)

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
                gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                views = split_stereo(gray, (self.args.eye_width, self.args.eye_height),
                                     self.args.swap_eyes)
                with self.lock:
                    session = self.session if self.session and not self.session.finished else None
                if session is None:
                    self.snapshot = (views, None)
                    continue
                start = time.monotonic()
                results = list(self.pool.map(self._track, [(t, v, timestamp)
                                                           for t, v in zip(self.trackers, views)]))
                elapsed_ms = (time.monotonic() - start) * 1000
                with self.lock:
                    if session is not self.session or session.finished:
                        continue  # cancelled or restarted while processing
                    if self.timings:
                        self.skipped += max(0, sequence - previous - 1)
                    self.first_ts = self.first_ts if self.first_ts is not None else timestamp
                    self.last_ts = timestamp
                    self.timings.append(elapsed_ms)
                    self.exposures.append((metadata.get("exposure_us") or 0,
                                           metadata.get("analogue_gain") or 0))
                    if self.clip is not None:
                        self.clip.add(sequence, gray, metadata)
                    for trail, result in zip(self.trails, results):
                        if result.observed and result.filtered_center is not None:
                            trail.append(tuple(round(v) for v in result.filtered_center))
                    self.snapshot = (views, results)
                    if session.update(timestamp, *results):
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
const m = v => v == null ? '–' : (v >= 0 ? '+' : '') + v.toFixed(2) + ' m';
function crossingText(c) {
  if (!c) return 'No crossing predicted.';
  return `<div class="big">x ${m(c.x_m)} (${c.x_m >= 0 ? 'right' : 'left'}), height ${m(c.height_m)}</div>
   ±${c.x_std_m.toFixed(2)} m lateral · ${c.speed_mps.toFixed(1)} m/s · ${c.model} · servo ${c.servo_deg}° ·
   from ${c.samples_used} samples, last at z ${c.made_at.z_m.toFixed(2)} m (range ${c.made_at.range_m.toFixed(2)} m),
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
    $('live').innerHTML = (p ? `Latest ball position: x ${m(p.x_m)}, height ${m(p.height_m)}, z ${p.z_m.toFixed(2)} m (range ${p.range_m.toFixed(2)} m) · ${sess.measurements} measurements` : 'No 3D measurement yet') +
      '<br>' + stats(s.stats) + (sess.crossing ? '<br>Current prediction: ' + crossingText(sess.crossing) : '');
  } else $('live').textContent = '';
  const last = s.history[0];
  if (last) $('result').innerHTML = `<b>${last.reason}</b> · ${last.finished_at}<br>` + crossingText(last.crossing) +
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
    parser.add_argument("--calibration", type=Path, default=Path("calibration/stereo.json"))
    parser.add_argument("--camera-num", type=int, default=0)
    parser.add_argument("--eye-width", type=int, default=1280)
    parser.add_argument("--eye-height", type=int, default=800)
    parser.add_argument("--fps", type=float, default=60)
    parser.add_argument("--exposure-us", type=int, default=100,
                        help="Locked exposure; 100 us matches the recorded kicks")
    parser.add_argument("--gain", type=float, default=8.0, help="Analogue gain with --exposure-us")
    parser.add_argument("--auto-exposure", action="store_true", help="Ignore --exposure-us/--gain")
    parser.add_argument("--tuning-file", help="Optional Picamera2 tuning JSON")
    parser.add_argument("--swap-eyes", action="store_true", help="Label the right half as left")
    parser.add_argument("--max-wait-s", type=float, default=8.0,
                        help="End an armed kick with no predicted crossing after this long")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"),
                        help="Save each kick here as JSON")
    parser.add_argument("--record", action="store_true",
                        help="Save each armed kick's processed frames as a replayable clip")
    parser.add_argument("--record-dir", type=Path, default=Path("recordings/live"))
    parser.add_argument("--record-quality", type=int, default=95, help="JPEG quality 1-100")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preview-fps", type=float, default=8, help="Preview rate while idle")
    parser.add_argument("--armed-preview-fps", type=float, default=3,
                        help="Preview rate while tracking (0 = off, frees CPU for tracking)")
    parser.add_argument("--opencv-threads", type=int, default=2)
    args = parser.parse_args()
    if args.no_serve and not (args.video and args.autostart):
        parser.error("--no-serve is for replay with --video --autostart")
    if not args.auto_exposure and args.exposure_us > 1e6 / args.fps:
        parser.error("--exposure-us must fit within the frame period")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        datefmt="%H:%M:%S")
    cv2.setNumThreads(args.opencv_threads)
    calibration = StereoCalibration.load(args.calibration)
    if args.video:
        source = ReplaySource(args.video, args.timestamps)
    else:
        source = LatestPicameraStereo(
            args.eye_width * 2, args.eye_height, args.fps, args.camera_num, args.tuning_file,
            None if args.auto_exposure else args.exposure_us, 1.0 if args.auto_exposure else args.gain)
    keeper = LiveKeeper(source, calibration, args)
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
        if keeper.encode_pool:
            keeper.encode_pool.shutdown()
    if args.no_serve:
        print(json.dumps(keeper.status()["history"][:1], indent=1))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, OSError, ImportError) as exc:
        logging.error("%s", exc)
        sys.exit(1)
