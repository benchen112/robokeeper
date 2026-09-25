"""Low latency 2D ball candidates and one-ball temporal tracking.

Each instance owns its background model and track. Create one instance per camera;
the frame/timestamp interface is also suitable for a future stereo capture layer.
"""

from dataclasses import dataclass
from math import hypot
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
        cv2.accumulateWeighted(gray, self.background, self.background_alpha)
        self.frames_seen += 1
        self.ready = self.frames_seen > self.warmup_frames
        if not self.ready or cv2.countNonZero(mask) / mask.size > self.max_foreground_fraction:
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


class BallTracker:
    """Associate candidates over time and report measured and filtered centroids."""

    def __init__(
        self,
        segmenter: Segmenter | None = None,
        *,
        confirmation_hits: int = 3,
        max_misses: int = 5,
        initial_gate_px: float = 70.0,
    ) -> None:
        if confirmation_hits < 1 or max_misses < 0 or initial_gate_px <= 0:
            raise ValueError("Invalid tracking limits")
        self.segmenter = segmenter or MotionBallSegmenter()
        self.confirmation_hits = confirmation_hits
        self.max_misses = max_misses
        self.initial_gate_px = initial_gate_px
        self._position: tuple[float, float] | None = None
        self._velocity = (0.0, 0.0)
        self._radius = 0.0
        self._confidence = 0.0
        self._hits = 0
        self._misses = 0
        self._last_time: float | None = None

    def reset(self) -> None:
        self._position = None
        self._velocity = (0.0, 0.0)
        self._radius = 0.0
        self._confidence = 0.0
        self._hits = 0
        self._misses = 0
        self._last_time = None
        if hasattr(self.segmenter, "reset"):
            self.segmenter.reset()

    def process(self, frame: np.ndarray, timestamp: float) -> TrackResult:
        if self._last_time is not None and timestamp <= self._last_time:
            raise ValueError("Frame timestamps must increase")
        candidates = self.segmenter.find_candidates(frame)
        ready = getattr(self.segmenter, "ready", True)
        dt = min(timestamp - self._last_time, 0.2) if self._last_time is not None else 0.0
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
            eligible = [
                (candidate.confidence - 0.35 * hypot(candidate.center[0] - px, candidate.center[1] - py) / gate,
                 candidate)
                for candidate in candidates
                if hypot(candidate.center[0] - px, candidate.center[1] - py) <= gate
            ]
            if eligible:
                selected = max(eligible, key=lambda item: item[0])[1]
                residual_x = selected.center[0] - px
                residual_y = selected.center[1] - py
                self._position = (px + 0.75 * residual_x, py + 0.75 * residual_y)
                if dt > 0:
                    self._velocity = (
                        self._velocity[0] + 0.6 * residual_x / dt,
                        self._velocity[1] + 0.6 * residual_y / dt,
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
