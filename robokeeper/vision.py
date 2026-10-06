"""Low latency 2D ball candidates and one-ball temporal tracking.

Each instance owns its background model and track. Create one instance per camera;
the frame/timestamp interface is also suitable for a future stereo capture layer.
"""

from collections import deque
from dataclasses import dataclass
from math import hypot, log
from typing import Protocol

import cv2
import numpy as np


@dataclass(frozen=True)
class BallCandidate:
    center: tuple[float, float]
    radius: float
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class TrackResult:
    timestamp: float
    state: str  # warming_up, absent, tentative, confirmed, or predicted
    observed: bool
    center: tuple[float, float] | None
    filtered_center: tuple[float, float] | None
    velocity_px_s: tuple[float, float] | None
    radius: float | None
    confidence: float
    candidates: tuple[BallCandidate, ...]


class Segmenter(Protocol):
    def find_candidates(self, frame: np.ndarray) -> tuple[BallCandidate, ...]: ...


class MotionBallSegmenter:
    """Propose moving, compact, locally distinct blobs from a fixed camera.

    This does not classify soccer balls by learned appearance. A stationary ball or
    substantial camera motion cannot be identified reliably by this segmenter.
    """

    def __init__(
        self,
        *,
        warmup_frames: int = 5,
        min_area: int = 5,
        max_area: int = 4000,
        min_confidence: float = 0.45,
        max_foreground_fraction: float = 0.25,
        difference_threshold: int = 18,
        background_alpha: float = 0.01,
    ) -> None:
        if (warmup_frames < 0 or min_area < 1 or max_area < min_area
                or not 1 <= difference_threshold <= 255
                or not 0 < background_alpha <= 1):
            raise ValueError("Invalid warmup or area limits")
        self.warmup_frames = warmup_frames
        self.min_area = min_area
        self.max_area = max_area
        self.min_confidence = min_confidence
        self.max_foreground_fraction = max_foreground_fraction
        self.difference_threshold = difference_threshold
        self.background_alpha = background_alpha
        self.frames_seen = 0
        self.ready = False
        self.background: np.ndarray | None = None

    def reset(self) -> None:
        self.frames_seen = 0
        self.ready = False
        self.background = None

    def find_candidates(self, frame: np.ndarray) -> tuple[BallCandidate, ...]:
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim == 2:
            gray = frame
        else:
            raise ValueError("Expected a grayscale or BGR frame")
        if gray.dtype != np.uint8:
            raise ValueError("Expected an 8-bit frame")

        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        if self.background is None:
            self.background = gray.astype(np.float32)
        background_u8 = cv2.convertScaleAbs(self.background)
        difference = cv2.absdiff(gray, background_u8)
        _, mask = cv2.threshold(difference, self.difference_threshold, 255, cv2.THRESH_BINARY)
        foreground_fraction = cv2.countNonZero(mask) / mask.size
        # Warm up and recover from global lighting changes; otherwise protect foreground.
        update_mask = (None if self.frames_seen < self.warmup_frames
                       or foreground_fraction > self.max_foreground_fraction
                       else cv2.bitwise_not(mask))
        cv2.accumulateWeighted(gray, self.background, self.background_alpha, mask=update_mask)
        self.frames_seen += 1
        self.ready = self.frames_seen > self.warmup_frames
        if not self.ready or foreground_fraction > self.max_foreground_fraction:
            return ()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        height, width = gray.shape
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if not self.min_area <= area <= self.max_area:
                continue
            # Shape is intentionally tolerant of small, partly segmented balls.
            aspect = min(w, h) / max(w, h)
            fill = area / (w * h)
            if aspect < 0.48 or fill < 0.28:
                continue

            pad = max(3, int(max(w, h) * 0.5))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(width, x + w + pad), min(height, y + h + pad)
            region = gray[y0:y1, x0:x1]
            component_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(component_mask, [contour - (x, y)], -1, 255, -1)
            component = component_mask > 0
            pixels = gray[y:y + h, x:x + w][component]
            surround = np.ones(region.shape, dtype=bool)
            surround[y - y0:y - y0 + h, x - x0:x - x0 + w] = False
            ring = region[surround]
            contrast = abs(float(pixels.mean()) - float(ring.mean())) if ring.size else 0.0
            texture = float(pixels.std()) if pixels.size > 1 else 0.0
            appearance = min(1.0, max(contrast, texture) / 22.0)
            shape = min(1.0, aspect / 0.85)
            solidity = min(1.0, fill / 0.70)
            confidence = 0.35 * shape + 0.25 * solidity + 0.40 * appearance
            if confidence < self.min_confidence:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
            cx = moments["m10"] / moments["m00"]
            cy = moments["m01"] / moments["m00"]
            candidates.append(
                BallCandidate((cx, cy), 0.5 * max(w, h), confidence, (x, y, w, h))
            )
        return tuple(sorted(candidates, key=lambda candidate: candidate.confidence, reverse=True))


class AppearanceBallSegmenter:
    """Circular edge proposals plus local appearance tracking, including resting balls.

    Floor coordinates are normalized (x0, y0, x1, y1) and only add an
    acquisition preference. They never restrict the search or an airborne track.
    Confidence is a ranking score, not a probability of soccer-ball identity.
    """

    ready = True

    def __init__(self, *, min_radius=8, max_radius=140, floor_region=None,
                 search_interval=6, min_edge_support=0.58):
        if (min_radius < 3 or max_radius < min_radius or search_interval < 1
                or not 0 < min_edge_support <= 1):
            raise ValueError("Invalid appearance detector limits")
        if floor_region is not None:
            x0, y0, x1, y1 = floor_region
            if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
                raise ValueError("Floor region must be normalized x0,y0,x1,y1")
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.floor_region = floor_region
        self.search_interval = search_interval
        self.min_edge_support = min_edge_support
        self.reset()

    def reset(self):
        self._hint = None
        self._template = None
        self._last_radius = None
        self._frames = 0
        self._gray = None

    def set_tracking_hint(self, center, radius, search_margin):
        self._hint = (center, radius, search_margin) if center is not None else None

    def observe(self, candidate):
        # Learn appearance only from selected measurements, never gap predictions.
        self._last_radius = candidate.radius
        patch, visible = self._patch(*candidate.center, candidate.radius)
        if visible >= 0.95 and candidate.confidence >= 0.58 and patch.std() >= 5:
            self._template = patch

    def _patch(self, x, y, radius):
        # Sample the inner disk's bounding square, avoiding background and shoes.
        axis = np.linspace(-0.70, 0.70, 32, dtype=np.float32) * radius
        mx, my = np.meshgrid(axis + x, axis + y)
        valid = (mx >= 0) & (mx < self._gray.shape[1]) & (my >= 0) & (my < self._gray.shape[0])
        patch = cv2.remap(self._gray, mx, my, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
        return patch, float(valid.mean())

    def _edge_support(self, x, y, radius, gx, gy):
        angles = np.linspace(0, 2 * np.pi, 96, endpoint=False)
        co, si = np.cos(angles), np.sin(angles)
        supported = np.zeros(96, dtype=bool)
        visible = np.zeros(96, dtype=bool)
        for offset in (-3, -1, 1, 3):
            xx = np.rint(x + (radius + offset) * co).astype(int)
            yy = np.rint(y + (radius + offset) * si).astype(int)
            valid = (xx >= 0) & (xx < gx.shape[1]) & (yy >= 0) & (yy < gx.shape[0])
            xx = np.clip(xx, 0, gx.shape[1] - 1)
            yy = np.clip(yy, 0, gx.shape[0] - 1)
            dx, dy = gx[yy, xx], gy[yy, xx]
            radial = np.abs(dx * co + dy * si)
            supported |= valid & (radial > 50) & (radial > np.hypot(dx, dy) * 0.85)
            visible |= valid
        if visible.mean() < 0.45:
            return 0.0
        return float(supported.sum() / visible.sum())

    def find_candidates(self, frame):
        if frame.dtype != np.uint8 or frame.ndim not in (2, 3):
            raise ValueError("Expected an 8-bit grayscale or BGR frame")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        self._gray = cv2.GaussianBlur(gray, (5, 5), 1)
        height, width = gray.shape
        self._frames += 1
        local = self._hint is not None
        if local:
            (cx, cy), radius, margin = self._hint
            extent = radius + margin
            x0, y0 = max(0, int(cx - extent)), max(0, int(cy - extent))
            x1, y1 = min(width, int(cx + extent)), min(height, int(cy + extent))
            rmin = max(self.min_radius, int(radius * 0.72))
            rmax = min(self.max_radius, int(radius * 1.40))
        else:
            # Reacquisition scans are throttled, but always cover the entire frame.
            if (self._frames - 1) % self.search_interval:
                return ()
            x0, y0, x1, y1 = 0, 0, width, height
            rmin, rmax = self.min_radius, self.max_radius
            if self._last_radius is not None:
                rmin = max(rmin, int(self._last_radius * 0.55))
                rmax = min(rmax, int(self._last_radius * 1.55))
        if x1 - x0 < 8 or y1 - y0 < 8:
            return ()
        region = self._gray[y0:y1, x0:x1]
        # Padding lets the circle estimator propose partially visible edge balls.
        pad = rmax if x0 == 0 or x1 == width else 0
        search = cv2.copyMakeBorder(region, 0, 0, pad, pad, cv2.BORDER_REPLICATE)
        circles = cv2.HoughCircles(search, cv2.HOUGH_GRADIENT, 1.2,
                                  max(12, rmin * 0.7), param1=100,
                                  param2=25 if local else 35,
                                  minRadius=rmin, maxRadius=rmax)
        proposals = [] if circles is None else [
            (float(x + x0 - pad), float(y + y0), float(r)) for x, y, r in circles[0]
        ]
        # Template matching fills short edge-detector gaps without using motion masks.
        if local and self._template is not None:
            for scale in (0.9, 1.0, 1.12):
                rr = radius * scale
                size = max(5, round(1.4 * rr))
                if not self.min_radius <= rr <= self.max_radius or min(region.shape) < size:
                    continue
                template = cv2.resize(self._template, (size, size))
                response = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
                _, similarity, _, location = cv2.minMaxLoc(response)
                if similarity >= 0.55:
                    proposals.append((x0 + location[0] + (size - 1) / 2,
                                      y0 + location[1] + (size - 1) / 2, rr))
        if local and (cx - radius < 15 or cx + radius > width - 15):
            # Refine the last circle against fresh edges, especially at image borders.
            # This is still a measurement: candidates must pass edge support below.
            step = max(4.0, radius * 0.10)
            proposals.extend((cx + dx, cy + dy, radius * scale)
                             for dx in (-step, 0, step) for dy in (-step, 0, step)
                             for scale in (0.92, 1.0, 1.08)
                             if self.min_radius <= radius * scale <= self.max_radius)
        gx = cv2.Sobel(self._gray, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(self._gray, cv2.CV_32F, 0, 1)
        candidates = []
        for x, y, radius in proposals:
            if not (-radius * 0.5 <= x <= width + radius * 0.5 and 0 <= y < height):
                continue
            support = self._edge_support(x, y, radius, gx, gy)
            patch, visible = self._patch(x, y, radius)
            similarity = 0.0
            if self._template is not None and visible >= 0.95:
                similarity = float(cv2.matchTemplate(patch, self._template,
                                                     cv2.TM_CCOEFF_NORMED)[0, 0])
            threshold = 0.38 if local and similarity >= 0.55 else self.min_edge_support
            if local and visible < 0.95:
                threshold = min(threshold, 0.50)
            if support < threshold:
                continue
            # After loss, remembered appearance helps reject unrelated round objects.
            if not local and self._template is not None and visible >= 0.95 and similarity < 0.50:
                continue
            confidence = 0.75 * support + 0.25 * max(0.0, similarity) if self._template is not None else support
            if not local and self.floor_region is not None:
                fx0, fy0, fx1, fy1 = self.floor_region
                if fx0 <= x / width <= fx1 and fy0 <= (y + radius) / height <= fy1:
                    confidence = min(1.0, confidence + 0.08)
            bx, by = max(0, int(x - radius)), max(0, int(y - radius))
            ex, ey = min(width, int(x + radius + 1)), min(height, int(y + radius + 1))
            if ex > bx and ey > by:
                candidates.append(BallCandidate((x, y), radius, confidence,
                                                (bx, by, ex - bx, ey - by)))
        return tuple(sorted(candidates, key=lambda c: c.confidence, reverse=True))


class BallTracker:
    """Associate candidates over time and report measured and filtered centroids."""

    def __init__(
        self,
        segmenter: Segmenter | None = None,
        *,
        confirmation_hits: int = 3,
        max_misses: int = 5,
        initial_gate_px: float = 70.0,
        adaptive_gate: bool = False,
    ) -> None:
        if confirmation_hits < 1 or max_misses < 0 or initial_gate_px <= 0:
            raise ValueError("Invalid tracking limits")
        self.segmenter = segmenter or MotionBallSegmenter()
        self.confirmation_hits = confirmation_hits
        self.max_misses = max_misses
        self.initial_gate_px = initial_gate_px
        self.adaptive_gate = adaptive_gate
        self._position: tuple[float, float] | None = None
        self._velocity = (0.0, 0.0)
        self._radius = 0.0
        self._confidence = 0.0
        self._hits = 0
        self._misses = 0
        self._last_time: float | None = None
        self._frame_intervals = deque(maxlen=15)

    def reset(self) -> None:
        self._position = None
        self._velocity = (0.0, 0.0)
        self._radius = 0.0
        self._confidence = 0.0
        self._hits = 0
        self._misses = 0
        self._last_time = None
        self._frame_intervals.clear()
        if hasattr(self.segmenter, "reset"):
            self.segmenter.reset()

    def process(self, frame: np.ndarray, timestamp: float) -> TrackResult:
        if self._last_time is not None and timestamp <= self._last_time:
            raise ValueError("Frame timestamps must increase")
        dt = timestamp - self._last_time if self._last_time is not None else 0.0
        if 0 < dt <= .2:
            self._frame_intervals.append(dt)
        velocity_dt = dt
        if len(self._frame_intervals) >= 3:
            # Arrival timestamps can bunch up after USB/decode buffering. Do not
            # turn pixel noise into enormous velocity on a near-zero interval.
            velocity_dt = max(dt, float(np.median(self._frame_intervals)) * .5)
        if dt > 0.2:
            self._frame_intervals.clear()
            self._position = None
            self._hits = 0
            self._velocity = (0.0, 0.0)
        if hasattr(self.segmenter, "set_frame_timestamp"):
            self.segmenter.set_frame_timestamp(timestamp)
        if hasattr(self.segmenter, "set_tracking_hint"):
            predicted = None if self._position is None else (
                self._position[0] + self._velocity[0] * dt,
                self._position[1] + self._velocity[1] * dt,
            )
            margin = max(30.0, self._radius * 0.8) + hypot(*self._velocity) * dt + 12 * self._misses
            self.segmenter.set_tracking_hint(predicted, self._radius, margin)
        candidates = self.segmenter.find_candidates(frame)
        verified = getattr(self.segmenter, "acquisition_candidates", ())
        if self.adaptive_gate and self._position is not None and verified:
            challenger = max(verified, key=lambda candidate: candidate.confidence)
            predicted = (self._position[0]+self._velocity[0]*dt,
                         self._position[1]+self._velocity[1]*dt)
            outside = (hypot(challenger.center[0]-predicted[0], challenger.center[1]-predicted[1])
                       > max(8., self._radius*1.3)
                       or not .65 <= challenger.radius/self._radius <= 1.55)
            if outside and challenger.confidence > max(.65, self._confidence+.12):
                self._position = None
                self._velocity = (0., 0.)
                self._hits = 0
                candidates = (challenger,) + tuple(c for c in candidates if c is not challenger)
        ready = getattr(self.segmenter, "ready", True)
        self._last_time = timestamp

        if not ready:
            return TrackResult(timestamp, "warming_up", False, None, None, None, None, 0.0, candidates)

        selected = None
        if self._position is None:
            if candidates:
                selected = candidates[0]
                self._position = selected.center
                self._velocity = (0.0, 0.0)
                self._radius = selected.radius
                self._confidence = selected.confidence
                self._hits = 1
                self._misses = 0
        else:
            px = self._position[0] + self._velocity[0] * dt
            py = self._position[1] + self._velocity[1] * dt
            gate = self.initial_gate_px + 2 * self._radius + 0.5 * hypot(*self._velocity) * dt
            if self.adaptive_gate:
                gate = max(8., self._radius * 1.3) + .5 * hypot(*self._velocity) * dt + 4 * self._misses
            eligible = [
                (candidate.confidence - 0.65 * hypot(candidate.center[0] - px, candidate.center[1] - py) / gate
                 - 0.35 * abs(log(candidate.radius / self._radius)),
                 candidate)
                for candidate in candidates
                if hypot(candidate.center[0] - px, candidate.center[1] - py) <= gate
                and 0.65 <= candidate.radius / self._radius <= 1.55
            ]
            if eligible:
                selected = max(eligible, key=lambda item: item[0])[1]
                residual_x = selected.center[0] - px
                residual_y = selected.center[1] - py
                self._position = (px + 0.75 * residual_x, py + 0.75 * residual_y)
                if dt > 0:
                    self._velocity = (
                        self._velocity[0] + 0.6 * residual_x / velocity_dt,
                        self._velocity[1] + 0.6 * residual_y / velocity_dt,
                    )
                self._radius = 0.7 * self._radius + 0.3 * selected.radius
                self._confidence = selected.confidence
                self._hits += 1
                self._misses = 0
            else:
                # A one-frame lookalike must not survive to become a track.
                if self._hits < self.confirmation_hits:
                    self._position = None
                    self._hits = 0
                    self._velocity = (0.0, 0.0)
                else:
                    self._position = (px, py)
                    self._misses += 1
                    self._confidence *= 0.75
                    if self._misses > self.max_misses:
                        self._position = None
                        self._hits = 0
                        self._velocity = (0.0, 0.0)

        if selected is not None and self._hits >= self.confirmation_hits and hasattr(self.segmenter, "observe"):
            self.segmenter.observe(selected)
        if self._position is None:
            return TrackResult(timestamp, "absent", False, None, None, None, None, 0.0, candidates)
        state = (
            "predicted" if selected is None else
            "confirmed" if self._hits >= self.confirmation_hits else "tentative"
        )
        return TrackResult(
            timestamp, state, selected is not None,
            selected.center if selected else None, self._position, self._velocity,
            self._radius, self._confidence, candidates,
        )
