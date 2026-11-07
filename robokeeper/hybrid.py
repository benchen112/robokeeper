"""Motion-supported ball proposals and conservative automatic floor hints."""
from collections import deque
from dataclasses import dataclass
from math import hypot
from time import perf_counter

import cv2
import numpy as np

from .appearance import BallAppearanceVerifier
from .vision import AppearanceBallSegmenter, BallCandidate

# Hough cost grows with the radius range. Acquisition scans of large motion
# regions search larger circles on a copy where they are at most this many pixels.
ACQUISITION_HOUGH_MAX_RADIUS = 48
# The remembered template is a 32 px patch, so matching it larger adds cost, not
# detail. Close balls are matched on a copy where the template is this wide.
TEMPLATE_MATCH_PX = 24
# Motion regions beyond these are skipped before Hough (see _hough_proposals).
MAX_BALL_BLOB_PX = 700
MAX_BALL_BLOB_ASPECT = 4.5
# Fixed-camera motion is computed only in tiles (half-resolution pixels) where
# the image changed recently; still scenes cost almost nothing.
MOTION_TILE_PX = 16


def _any_per_tile(image, tile):
    """True per tile x tile block (edge blocks padded with zeros) holding a 255 pixel."""
    h, w = image.shape
    th, tw = -(-h // tile), -(-w // tile)
    if (th * tile, tw * tile) != (h, w):
        image = cv2.copyMakeBorder(image, 0, th * tile - h, 0, tw * tile - w, cv2.BORDER_CONSTANT)
    # A block mean rounds 255 / tile**2 up to 1 (tile <= 22), so one 255 pixel counts.
    return cv2.resize(image, (tw, th), interpolation=cv2.INTER_AREA) > 0


@dataclass(frozen=True)
class FloorEstimate:
    boundary_y: float | None = None
    confidence: float = 0.0


class FloorEstimator:
    """Find a supported horizontal transition above a smoother bottom region.

    This estimates a floor-like image region, not a calibrated ground plane.
    Ambiguous scenes produce no estimate. It is only a ranking hint.

    The whole-frame line search costs 5-15 ms, all on one frame, so it runs
    every ``search_every`` frames until an estimate holds, then only every
    ``refresh_every`` frames: with a fixed camera the floor does not move.
    """

    def __init__(self, search_every=6, refresh_every=120):
        self.estimate = FloorEstimate()
        self._history = deque(maxlen=5)
        self.search_every = search_every
        self.refresh_every = refresh_every
        self._since = None  # frames since the last search; None runs the next one

    def update(self, gray):
        every = self.refresh_every if self.estimate.boundary_y is not None else self.search_every
        if self._since is not None and self._since + 1 < every:
            self._since += 1
            return self.estimate
        self._since = 0
        small = cv2.resize(gray, (320, 200), interpolation=cv2.INTER_AREA)
        edges = cv2.Canny(small, 40, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 35,
                                minLineLength=80, maxLineGap=15)
        choices = []
        if lines is not None:
            for x0, y0, x1, y1 in lines[:, 0]:
                if abs(y1 - y0) > max(3, abs(x1 - x0) * 0.04):
                    continue
                y = int((y0 + y1) / 2)
                if not 30 < y < 170:
                    continue
                above = float((edges[max(0, y - 30):y] > 0).mean())
                below = float((edges[y + 8:] > 0).mean())
                if above < 0.035 or below > 0.08 or above < below * 1.5:
                    continue
                support = min(1.0, abs(x1 - x0) / 200)
                confidence = support * min(1.0, (above - below) / 0.10)
                choices.append((y / 200, confidence))
        if not choices:
            self._history.clear()
            self.estimate = FloorEstimate()
            return self.estimate
        # Lowest supported boundary with a smoother bottom-connected region.
        y, confidence = max(choices, key=lambda item: item[0])
        self._history.append((y, confidence))
        if len(self._history) >= 3 and np.ptp([item[0] for item in self._history]) < .05:
            self.estimate = FloorEstimate(float(np.median([v[0] for v in self._history])),
                                           float(np.mean([v[1] for v in self._history])))
        else:
            self.estimate = FloorEstimate()
        return self.estimate


class HybridBallSegmenter(AppearanceBallSegmenter):
    """Require scene-relative motion to acquire, then retain ball appearance locally.

    Camera translation/rotation is estimated from sparse feature flow with RANSAC.
    A ``fixed_camera`` (rigid mount) skips that alignment and builds the motion
    mask at half resolution; circle checks still use full-resolution pixels.
    Background difference proposes regions; current-frame circles/appearance verify
    them. Resting objects cannot acquire a track through appearance alone by default.
    Radius limits are broad sensor limits, not a distance profile.
    ``acquisition_min_row`` (pixels) ignores new balls centered above that row,
    such as a standing person's head; a ball already tracked may rise above it.
    While the tracker reports a lock (``set_track_lock``), template matching alone
    proposes the ball; Hough runs every ``locked_hough_interval`` locked frames
    (0 = only on fallback) and whenever the template yields no verified candidate.
    """

    def __init__(self, *, min_radius=3, max_radius=None, floor_region=None,
                 auto_floor=True, difference_threshold=12, verify_appearance=True,
                 fixed_camera=False, acquisition_min_row=None,
                 locked_hough_interval=8):
        if not 1 <= difference_threshold <= 255:
            raise ValueError("Difference threshold must be between 1 and 255")
        super().__init__(min_radius=min_radius, max_radius=max_radius if max_radius is not None else 10000,
                         floor_region=floor_region, search_interval=1)
        self.verifier = BallAppearanceVerifier() if verify_appearance else None
        self.radius_limit = max_radius
        self.auto_floor = auto_floor and floor_region is None
        self.difference_threshold = difference_threshold
        self.floor_estimator = FloorEstimator()
        self.fixed_camera = fixed_camera
        self.acquisition_min_row = acquisition_min_row
        self.motion_scale = .5 if fixed_camera else 1.
        self.locked_hough_interval = locked_hough_interval
        self.ready = False

    def reset(self):
        super().reset()
        self._previous = None
        self._background = None
        self._recent_motion = None
        self._live_tiles = None
        self.motion_mask = None
        self._motion_integral = None
        self._mask_scale = 1
        self._mask_shape = (0, 0)
        self.camera_transform = np.eye(3, dtype=np.float32)
        self.camera_motion_reliable = True
        self._last_motion_frame = 0
        self._timestamp = None
        self._last_observed_timestamp = None
        self._observed_radii = deque(maxlen=2)
        self._memory_active = False
        self._tracklets = []
        self.raw_candidates = ()
        self.acquisition_candidates = ()
        self.ready = False
        self.stage_ms = {}
        self._locked = False
        self._locked_frames = 0
        self.template_only = False
        self.floor_estimator = FloorEstimator()

    def set_track_lock(self, locked):
        self._locked = locked

    def set_frame_timestamp(self, timestamp):
        self._timestamp = timestamp

    def observe(self, candidate):
        super().observe(candidate)
        self._last_observed_timestamp = self._timestamp
        self._observed_radii.append((self._timestamp, candidate.radius))
        if self._motion_support(*candidate.center, candidate.radius) * candidate.radius >= .40:
            self._last_motion_frame = self._frames

    def _camera_transform(self, previous, current):
        # Features across the frame make moving players outliers to camera motion.
        old = cv2.resize(previous, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        new = cv2.resize(current, (old.shape[1], old.shape[0]), interpolation=cv2.INTER_AREA)
        points = cv2.goodFeaturesToTrack(old, 160, .03, 12)
        identity = np.eye(3, dtype=np.float32)
        if points is None or len(points) < 12:
            return identity, True  # Fixed-camera, low-texture fallback; scene-change guard below.
        moved, status, error = cv2.calcOpticalFlowPyrLK(old, new, points, None,
                                                      winSize=(21, 21), maxLevel=2)
        if moved is None:
            return identity, False
        valid = (status[:, 0] > 0) & (error[:, 0] < 25)
        if valid.sum() < 12:
            return identity, False
        matrix, inliers = cv2.findHomography(points[valid], moved[valid], cv2.RANSAC, 1.0)
        if matrix is None or inliers.mean() < .55:
            return identity, False
        scaling = np.diag([2., 2., 1.])
        matrix = scaling @ matrix @ np.linalg.inv(scaling)
        corners = np.float32([[[0, 0], [previous.shape[1], 0],
                               [0, previous.shape[0]], [previous.shape[1], previous.shape[0]]]])
        mapped = cv2.perspectiveTransform(corners, matrix)
        if np.max(np.linalg.norm(mapped-corners, axis=2)) > previous.shape[1]*.05:
            return identity, False
        return matrix.astype(np.float32), True

    def _warp_point(self, point):
        projected = self.camera_transform @ np.array((*point, 1.), dtype=np.float32)
        return tuple(projected[:2] / projected[2])

    def _motion(self, full):
        h, w = full.shape
        if self.fixed_camera and self.motion_scale == .5 and h % 2 == 0 and w % 2 == 0:
            return self._motion_fixed(full)
        return self._motion_general(full)

    def _motion_general(self, full):
        h, w = full.shape
        gray = full if self.motion_scale == 1 else cv2.resize(
            full, None, fx=self.motion_scale, fy=self.motion_scale, interpolation=cv2.INTER_AREA)
        if self._previous is None:
            self._previous = gray.copy()
            self._background = gray.astype(np.float32)
            self._recent_motion = np.zeros(gray.shape, np.float32)
            self._set_mask(np.zeros(full.shape, np.uint8), 1)
            return False
        size = (gray.shape[1], gray.shape[0])
        if self.fixed_camera:
            reliable = True
            aligned, background, recent = self._previous, self._background, self._recent_motion
            valid = None
        else:
            transform, reliable = self._camera_transform(self._previous, gray)
            self.camera_transform = transform
            aligned = cv2.warpPerspective(self._previous, transform, size, borderMode=cv2.BORDER_REPLICATE)
            background = cv2.warpPerspective(self._background, transform, size, borderMode=cv2.BORDER_REPLICATE)
            recent = cv2.warpPerspective(self._recent_motion, transform, size)
            valid = cv2.warpPerspective(np.ones(gray.shape, np.uint8), transform, size)
        self.camera_motion_reliable = reliable
        # Remove uniform illumination changes before measuring scene-relative motion.
        offset = float(np.median(gray[::8, ::8].astype(float) - aligned[::8, ::8]))
        # A gradient allowance suppresses interpolation residuals at stationary edges.
        # Gradients per pixel grow as the mask resolution drops, so the allowance scales.
        dx = cv2.Sobel(aligned, cv2.CV_32F, 1, 0)
        dy = cv2.Sobel(aligned, cv2.CV_32F, 0, 1)
        edge_noise = (.25 * self.motion_scale) * cv2.magnitude(dx, dy)
        delta = np.abs(gray.astype(np.float32) - aligned.astype(np.float32) - offset)
        noise = float(np.percentile(delta[::8, ::8], 70))
        threshold = max(self.difference_threshold, noise * 3)
        short = (delta > threshold + edge_noise).astype(np.uint8) * 255
        if valid is not None:
            short[valid == 0] = 0
        recent = np.maximum(recent * .72, short.astype(np.float32))
        offset_bg = float(np.median(gray[::8, ::8].astype(float) - background[::8, ::8]))
        background += offset_bg
        foreground = np.abs(gray.astype(np.float32) - background) > threshold
        mask = (recent > 45) & foreground
        if valid is not None:
            mask &= valid > 0
        mask = mask.astype(np.uint8) * 255
        fraction = float((short > 0).mean())
        if not reliable or fraction > .20:
            mask.fill(0)
            background = gray.astype(np.float32)
            recent.fill(0)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cv2.accumulateWeighted(gray, background, .015, mask=cv2.bitwise_not(mask))
        self._previous = gray.copy()
        self._background = background
        self._recent_motion = recent
        if mask.shape != full.shape:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        self._set_mask(mask, 1)
        return reliable and fraction <= .20

    def _motion_fixed(self, full):
        """The fixed-camera half-resolution mask of ``_motion_general``, pixel for
        pixel, computed only where it can be non-zero.

        The mask needs a short-term difference within the last few frames, so
        tiles where nothing changed since then (a resting ball, an empty scene)
        are skipped: a tile is processed when any pixel may pass the short-term
        threshold now, or its decaying ``recent`` value can still exceed 45. The
        mask stays at half resolution; ``_mask_count`` reads full-resolution
        counts from its integral exactly.
        """
        gray = cv2.resize(full, None, fx=.5, fy=.5, interpolation=cv2.INTER_AREA)
        if self._previous is None:
            self._previous = gray.copy()
            self._background = gray.astype(np.float32)
            self._recent_motion = np.zeros(gray.shape, np.float32)
            self._live_tiles = None
            self._set_mask(np.zeros(gray.shape, np.uint8), 2)
            return False
        hh, ww = gray.shape
        tile = MOTION_TILE_PX
        th, tw = -(-hh // tile), -(-ww // tile)
        if self._live_tiles is None or self._live_tiles.shape != (th, tw):
            self._live_tiles = np.zeros((th, tw), bool)
        previous, background, recent = self._previous, self._background, self._recent_motion
        offset = float(np.median(gray[::8, ::8].astype(float) - previous[::8, ::8]))
        sub = np.abs(gray[::8, ::8].astype(np.float32) - previous[::8, ::8].astype(np.float32) - offset)
        noise = float(np.percentile(sub, 70))
        threshold = max(self.difference_threshold, noise * 3)
        # |gray - previous - offset| > threshold + edge noise (>= 0) needs
        # |gray - previous| > threshold - |offset|: a superset, cheap in uint8.
        floor_value = int(np.floor(threshold - abs(offset)))
        change = cv2.absdiff(gray, previous)
        if floor_value >= 0:
            change = cv2.threshold(change, floor_value, 255, cv2.THRESH_BINARY)[1]
        else:
            change.fill(255)
        active = _any_per_tile(change, tile) | self._live_tiles
        rectangles = self._tile_rectangles(active, tile, hh, ww)
        offset_bg = float(np.median(gray[::8, ::8].astype(float) - background[::8, ::8]))
        background += offset_bg
        raw = np.zeros(gray.shape, np.uint8)
        changed = 0
        for x0, y0, x1, y1 in rectangles:
            # Sobel needs one neighbouring pixel; take it from the frame, as the
            # whole-frame filter would, and keep only the rectangle.
            ex0, ey0, ex1, ey1 = max(0, x0 - 1), max(0, y0 - 1), min(ww, x1 + 1), min(hh, y1 + 1)
            around = previous[ey0:ey1, ex0:ex1]
            inner = (slice(y0 - ey0, y1 - ey0), slice(x0 - ex0, x1 - ex0))
            dx = cv2.Sobel(around, cv2.CV_32F, 1, 0)[inner]
            dy = cv2.Sobel(around, cv2.CV_32F, 0, 1)[inner]
            edge_noise = .125 * cv2.magnitude(dx, dy)
            g = gray[y0:y1, x0:x1].astype(np.float32)
            delta = np.abs(g - previous[y0:y1, x0:x1].astype(np.float32) - offset)
            short = (delta > threshold + edge_noise).astype(np.uint8) * 255
            changed += cv2.countNonZero(short)
            part = np.maximum(recent[y0:y1, x0:x1] * .72, short.astype(np.float32))
            recent[y0:y1, x0:x1] = part
            foreground = np.abs(g - background[y0:y1, x0:x1]) > threshold
            raw[y0:y1, x0:x1] = ((part > 45) & foreground).astype(np.uint8) * 255
            # recent * .72 > 45 needs recent > 62.5: such tiles stay live.
            live = _any_per_tile(cv2.threshold(part, 62, 255, cv2.THRESH_BINARY)[1].astype(np.uint8), tile)
            self._live_tiles[y0 // tile:y0 // tile + live.shape[0], x0 // tile:x0 // tile + live.shape[1]] = live
        fraction = changed / float(hh * ww)
        mask = raw
        if fraction > .20:
            mask.fill(0)
            background = gray.astype(np.float32)
            recent.fill(0)
            self._live_tiles.fill(False)
        elif rectangles:
            # Closing then opening reaches 4 pixels; beyond the rectangles the
            # mask is zero, so filtering their bounding box is exact.
            x0 = min(r[0] for r in rectangles); y0 = min(r[1] for r in rectangles)
            x1 = max(r[2] for r in rectangles); y1 = max(r[3] for r in rectangles)
            ex0, ey0, ex1, ey1 = max(0, x0 - 4), max(0, y0 - 4), min(ww, x1 + 4), min(hh, y1 + 4)
            part = raw[ey0:ey1, ex0:ex1]
            if cv2.countNonZero(part):
                kernel = np.ones((3, 3), np.uint8)
                part = cv2.morphologyEx(part, cv2.MORPH_CLOSE, kernel)
                part = cv2.morphologyEx(part, cv2.MORPH_OPEN, kernel)
                mask = np.zeros(gray.shape, np.uint8)
                bx0, by0 = max(0, x0 - 1) - ex0, max(0, y0 - 1) - ey0
                bx1, by1 = min(ww, x1 + 1) - ex0, min(hh, y1 + 1) - ey0
                # Closing can add pixels one beyond the raw mask.
                mask[ey0 + by0:ey0 + by1, ex0 + bx0:ex0 + bx1] = part[by0:by1, bx0:bx1]
        if mask is raw and not changed:
            cv2.accumulateWeighted(gray, background, .015)  # Empty mask: update everywhere.
        else:
            cv2.accumulateWeighted(gray, background, .015, mask=cv2.bitwise_not(mask))
        self._previous = gray
        self._background = background
        self._recent_motion = recent
        self._set_mask(mask, 2)
        return fraction <= .20

    @staticmethod
    def _tile_rectangles(active, tile, h, w):
        """Disjoint pixel rectangles covering the active tiles."""
        if not active.any():
            return []
        count, _, stats, _ = cv2.connectedComponentsWithStats(active.view(np.uint8), connectivity=8)
        boxes = [[x, y, x + bw, y + bh] for x, y, bw, bh, _ in stats[1:count]]
        merged = True
        while merged and len(boxes) > 1:
            merged = False
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    a, b = boxes[i], boxes[j]
                    if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                        boxes[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                        del boxes[j]
                        merged = True
                        break
                if merged:
                    break
        return [(x0 * tile, y0 * tile, min(w, x1 * tile), min(h, y1 * tile)) for x0, y0, x1, y1 in boxes]

    def _set_mask(self, mask, scale):
        """``motion_mask`` at 1/``scale`` resolution; counts read through ``_mask_count``."""
        self.motion_mask = mask
        self._mask_scale = scale
        self._mask_shape = (mask.shape[0] * scale, mask.shape[1] * scale)
        self._motion_integral = (cv2.integral(cv2.bitwise_and(mask, 1))
                                 if cv2.countNonZero(mask) else None)

    def _mask_count(self, x0, y0, x1, y1):
        """Motion pixels in a full-resolution rectangle, as if the mask were
        upsampled (nearest) to full resolution."""
        integral = self._motion_integral
        if integral is None:
            return 0.
        if self._mask_scale == 1:
            return float(integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0])
        def corner(y, x):
            a, ry = divmod(y, 2)
            b, rx = divmod(x, 2)
            base = int(integral[a, b])
            total = 4 * base
            if ry:
                total += 2 * (int(integral[a + 1, b]) - base)
            if rx:
                total += 2 * (int(integral[a, b + 1]) - base)
            if ry and rx:
                total += int(integral[a + 1, b + 1]) - int(integral[a + 1, b]) - int(integral[a, b + 1]) + base
            return total
        return float(corner(y1, x1) - corner(y0, x1) - corner(y1, x0) + corner(y0, x0))

    def _motion_support(self, x, y, radius, inner_fraction=0.0):
        # Integral rectangle/ring support bounds the cost even for large candidates.
        h, w = self._mask_shape
        def rectangle(r):
            x0, y0 = max(0, int(x-r)), max(0, int(y-r))
            x1, y1 = min(w, int(x+r+1)), min(h, int(y+r+1))
            if x1 <= x0 or y1 <= y0:
                return 0., 0
            return self._mask_count(x0, y0, x1, y1), (x1-x0)*(y1-y0)
        count, area = rectangle(radius)
        if inner_fraction:
            inner_count, inner_area = rectangle(radius*inner_fraction)
            count, area = count-inner_count, area-inner_area
        return count/area if area else 0.

    def _timed(self, stage, function, *args):
        """Run one pipeline stage, adding its milliseconds to ``stage_ms``."""
        start = perf_counter()
        result = function(*args)
        self.stage_ms[stage] = self.stage_ms.get(stage, 0.) + (perf_counter() - start) * 1e3
        return result

    def find_candidates(self, frame):
        if frame.dtype != np.uint8 or frame.ndim not in (2, 3):
            raise ValueError("Expected an 8-bit grayscale or BGR frame")
        # Coarse per-stage times for this frame (a few timer calls; negligible cost).
        self.stage_ms = {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = self._timed("blur", cv2.GaussianBlur, gray, (3, 3), 0)
        h, w = gray.shape
        self.max_radius = self.radius_limit or max(self.min_radius, min(h, w) // 4)
        self._gray = gray
        # Retain identity briefly across occlusion, then allow a fresh shot.
        self._memory_active = (self._timestamp is not None and
                               self._last_observed_timestamp is not None and
                               self._timestamp-self._last_observed_timestamp <= 2.)
        if self._hint is None and self._last_observed_timestamp is not None and not self._memory_active:
            self._template = None
            self._last_radius = None
            self._last_observed_timestamp = None
        self._frames += 1
        valid_motion = self._timed("motion", self._motion, gray)
        self.ready = self._frames > 1
        # The floor hint ranks new balls; while one is tracked it is not refreshed,
        # keeping its whole-frame search out of the frames that matter most.
        floor = (self.floor_estimator.estimate if not self.auto_floor or self._hint is not None
                 else self._timed("floor", self.floor_estimator.update, gray))
        if not self.auto_floor:
            floor = FloorEstimate()
        self.raw_candidates = ()
        self.acquisition_candidates = ()
        self.template_only = False
        if not self.ready or not valid_motion:
            self._tracklets.clear()
            return ()
        # Locked onto a predictable ball, the remembered template finds it near
        # the prediction without Hough. Hough runs on a periodic refresh (radius
        # re-anchoring), and immediately if the template finds nothing usable.
        template = self._timed("template", self._template_proposals, gray) if self._hint is not None else []
        self._locked_frames = self._locked_frames + 1 if self._locked and self._hint is not None else 0
        refresh = (self.locked_hough_interval > 0
                   and self._locked_frames % self.locked_hough_interval == 0)
        skip_hough = bool(template) and self._locked_frames > 0 and not refresh
        proposals = [] if skip_hough else self._timed("hough", self._hough_proposals, gray)
        proposals += template + self._border_proposals(w)
        candidates = self._timed("checks", self._verify, proposals, gray, floor)
        if skip_hough and not candidates:
            candidates = self._timed("checks", self._verify,
                                     self._timed("hough", self._hough_proposals, gray), gray, floor)
            self.template_only = False
        else:
            self.template_only = skip_hough
        candidates.sort(key=lambda c:c.confidence, reverse=True)
        self.raw_candidates = tuple(candidates)
        self.acquisition_candidates = self._timed("acquisition", self._coherent_acquisition, candidates)
        if self._hint is None:
            return self.acquisition_candidates
        return tuple(candidates)

    def _hough_proposals(self, gray):
        h, w = gray.shape
        proposals = []
        contours = (cv2.findContours(self.motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
                    if self._motion_integral is not None else ())
        regions = []
        for contour in contours:
            x, y, bw, bh = (v * self._mask_scale for v in cv2.boundingRect(contour))
            if self._contour_area(contour) < 8 or max(bw, bh) < self.min_radius * 2:
                continue
            # A reflection can connect the ball to a tall foreground component.
            # Keep its upper portion instead of dropping the entire component.
            if bh > h * .35 and bw / bh < .45:
                bh = min(bh, max(40, bw * 2))
            # Recorded ball blobs (blurred ball plus motion trail) reached 3.4:1 and
            # 550 px. Longer or larger ones are people or shadows, and the costliest
            # Hough regions. A ball entering at a side edge may be a thin sliver.
            at_side = x == 0 or x + bw == w
            if max(bw, bh) > MAX_BALL_BLOB_PX or (
                    not at_side and max(bw, bh) > MAX_BALL_BLOB_ASPECT * min(bw, bh)):
                continue
            pad = max(16, min(max(bw, bh) // 2, self.max_radius))
            regions.append((max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)))
        regions.sort(key=lambda box: -self._mask_count(*box))
        # Prioritize motion regions near the prediction; cap work during crowded scenes.
        if self._hint is not None:
            (cx, cy), radius, margin = self._hint
            regions.sort(key=lambda box: hypot((box[0]+box[2])/2-cx, (box[1]+box[3])/2-cy))
            extent = radius + margin
            regions.insert(0, (max(0, int(cx-extent)), max(0, int(cy-extent)),
                               min(w, int(cx+extent)), min(h, int(cy+extent))))
        for x0, y0, x1, y1 in regions[:1 if self._hint is not None else 6]:
            if self._hint is None and self.acquisition_min_row is not None:
                # Search only where an acquirable circle's edges can be; this also
                # skips most of a moving person, the slowest Hough regions.
                y0 = max(y0, int(self.acquisition_min_row) - 60)
            if min(x1-x0, y1-y0) < self.min_radius*2:
                continue
            region = gray[y0:y1, x0:x1]
            max_r = min(self.max_radius, min(region.shape)//2)
            if x0 == 0 or x1 == w:
                max_r = self.max_radius
            min_r = self.min_radius
            if self._hint is not None and (x0 <= cx <= x1 and y0 <= cy <= y1):
                min_r = max(min_r, int(radius*.60))
                max_r = min(self.max_radius, int(radius*1.5))
            # Extend only image-border sides to propose partly visible circles.
            left_pad = max_r if x0 == 0 else 0
            right_pad = max_r if x1 == w else 0
            padded = cv2.copyMakeBorder(region, 0, 0, left_pad, right_pad, cv2.BORDER_REPLICATE)
            # Bound Hough work in large close-up regions. Small distant-ball
            # regions retain their native pixels; validation uses native gradients.
            area_scale = min(1., np.sqrt(250000 / padded.size))
            bands = [(min_r, max_r, area_scale)]
            split = ACQUISITION_HOUGH_MAX_RADIUS / area_scale
            if self._hint is None and min_r < split * .5 < split < max_r:
                # Small radii at native scale; large radii (overlapping) on a smaller copy.
                bands = [(min_r, split, area_scale),
                         (split * .9, max_r, ACQUISITION_HOUGH_MAX_RADIUS / max_r)]
            for low, high, scale in bands:
                search = padded if scale == 1. else cv2.resize(
                    padded, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                cs = cv2.HoughCircles(search, cv2.HOUGH_GRADIENT, 1,
                                      max(5, low*scale), param1=80,
                                      param2=12 if high*scale < 40 else 22,
                                      minRadius=max(2, int(low*scale)),
                                      maxRadius=max(3, int(high*scale)))
                if cs is not None:
                    proposals.extend((float(x/scale+x0-left_pad), float(y/scale+y0), float(r/scale))
                                     for x,y,r in cs[0])

        return proposals

    def _contour_area(self, contour):
        """A mask contour's area as the full-resolution mask would give it, or a
        lower bound of at least 8 (the only threshold it is compared with)."""
        area = cv2.contourArea(contour)
        if self._mask_scale == 1:
            return area
        # Upsampled 2x, a blob's contour area is at least 4x + 1 its area here;
        # only small blobs, near the 8 px limit, need the exact value.
        if 4 * area + 1 >= 8:
            return 4 * area + 1
        x, y, bw, bh = cv2.boundingRect(contour)
        blob = np.zeros((bh + 2, bw + 2), np.uint8)
        cv2.drawContours(blob, [contour], -1, 255, cv2.FILLED, offset=(1 - x, 1 - y))
        blob = cv2.resize(blob, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
        found = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        return max((cv2.contourArea(c) for c in found), default=0.)

    def _template_proposals(self, gray):
        h, w = gray.shape
        proposals = []
        if self._hint is not None and self._template is not None:
            (cx, cy), radius, margin = self._hint
            radius = self._expected_radius(radius)
            extent = radius + margin
            x0, y0 = max(0, int(cx-extent)), max(0, int(cy-extent))
            x1, y1 = min(w, int(cx+extent)), min(h, int(cy+extent))
            region = gray[y0:y1, x0:x1]
            shrink = min(1., TEMPLATE_MATCH_PX / (1.4*radius))
            if shrink < 1 and region.size:
                # A sliver at the image edge must not shrink to zero pixels.
                size_xy = (max(1, round(region.shape[1] * shrink)), max(1, round(region.shape[0] * shrink)))
                region = cv2.resize(region, size_xy, interpolation=cv2.INTER_AREA)
            for scale in (.85, 1., 1.15):
                rr = radius * scale
                size = max(5, round(1.4*rr*shrink))
                if min(region.shape, default=0) < size:
                    continue
                patch = cv2.resize(self._template, (size,size), interpolation=cv2.INTER_AREA)
                _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(region,patch,cv2.TM_CCOEFF_NORMED))
                if score > .55:
                    # Pixel centers map back as (i + .5) / shrink - .5.
                    proposals.append((score, x0+(location[0]+(size-1)/2+.5)/shrink-.5,
                                      y0+(location[1]+(size-1)/2+.5)/shrink-.5, rr))
        if not proposals:
            return []
        # The template's radius steps are coarse; fit the ball's edge around the
        # best match so the circle checks see its true center and size.
        _, x, y, rr = max(proposals)
        refined = self._fit_circle(x, y, rr)
        return ([refined] if refined else []) + [p[1:] for p in proposals]

    def _expected_radius(self, radius):
        """Extrapolate the ball's size change: an approaching ball can grow a
        third between processed frames, beyond the template's scale steps."""
        if len(self._observed_radii) < 2 or self._timestamp is None:
            return radius
        (t0, r0), (t1, r1) = self._observed_radii
        if None in (t0, t1) or t1 <= t0 or self._timestamp <= t1:
            return radius
        growth = (r1/r0) ** ((self._timestamp-t1)/(t1-t0))
        return radius * float(np.clip(growth * r1/radius, .8, 1.8))

    def _fit_circle(self, x, y, radius):
        """Least-squares circle through the strongest radial edge on rays from (x, y)."""
        h, w = self._gray.shape
        angles = np.linspace(0, 2*np.pi, 64, endpoint=False)
        co, si = np.cos(angles)[:, None], np.sin(angles)[:, None]
        radii = radius * np.linspace(.7, 1.4, 15)
        fx, fy = x + co*radii, y + si*radii
        valid = (fx >= 0) & (fx < w) & (fy >= 0) & (fy < h)
        xx = np.clip(np.rint(fx).astype(int), 0, w-1)
        yy = np.clip(np.rint(fy).astype(int), 0, h-1)
        dx, dy = self._point_sobel(xx, yy)
        radial = np.abs(dx*co + dy*si) * valid
        best = radial.argmax(axis=1)
        rows = np.arange(len(angles))
        keep = radial[rows, best] > 50
        if keep.sum() < 16:
            return None
        px, py = fx[rows, best][keep], fy[rows, best][keep]
        for _ in range(2):
            a = np.column_stack((px, py, np.ones_like(px)))
            (bx, by, c), *_ = np.linalg.lstsq(a, px*px+py*py, rcond=None)
            cx, cy = bx/2, by/2
            r = float(np.sqrt(max(0., c + cx*cx + cy*cy)))
            # Drop rays that hit a shoe or background edge, then refit once.
            inliers = np.abs(np.hypot(px-cx, py-cy) - r) < max(2., r*.08)
            if inliers.sum() < 16:
                return None
            px, py = px[inliers], py[inliers]
        if not (.75 < r/radius < 1.35 and hypot(cx-x, cy-y) < radius*.5):
            return None
        return float(cx), float(cy), r

    def _border_proposals(self, w):
        proposals = []
        if self._hint is not None:
            (cx, cy), radius, _ = self._hint
            if cx-radius < 15 or cx+radius > w-15:
                step = max(4., radius*.10)
                proposals.extend((cx+dx, cy+dy, radius*scale)
                                 for dx in (-step, 0, step) for dy in (-step, 0, step)
                                 for scale in (.92, 1., 1.08)
                                 if self.min_radius <= radius*scale <= self.max_radius)
        return proposals

    def _verify(self, proposals, gray, floor):
        h, w = gray.shape
        candidates=[]
        # Checks run cheapest first. Rejecting a duplicate of an accepted circle
        # up front gives the same result as rejecting it after every other check.
        for x,y,radius in proposals:
            if not (-radius*.5 <= x <= w+radius*.5 and 0 <= y < h):
                continue
            if any(hypot(x-c.center[0],y-c.center[1])<max(4,radius*.3) for c in candidates):
                continue
            if self._hint is None and self.acquisition_min_row is not None and y < self.acquisition_min_row:
                continue
            # A ball coming at the camera grows fast between sparse frames; shrinking
            # (a smaller object taking over) stays tightly limited.
            if self._hint is None and self._memory_active and not .45 <= radius/self._last_radius <= 4.:
                continue
            motion=self._motion_support(x,y,radius)
            surrounding_motion=self._motion_support(x,y,radius*1.8, inner_fraction=.70)
            # A tiny circle inside a moving player is not an isolated moving ball.
            if surrounding_motion > .18 and motion < surrounding_motion * 1.2:
                continue
            local=False
            if self._hint is not None:
                (cx,cy), previous_radius, margin=self._hint
                local=hypot(x-cx,y-cy) < max(12,previous_radius*.7)
            if motion * radius < .75 and not local:
                continue
            support=self._edge_support(x,y,radius)
            if support < .35:  # Below every edge threshold below; skip the patch.
                continue
            patch,visible=self._patch(x,y,radius)
            if not local and patch.std() < 6 and support < .80:
                continue
            similarity=0.
            if self._template is not None and visible>.95:
                similarity=float(cv2.matchTemplate(patch,self._template,cv2.TM_CCOEFF_NORMED)[0,0])
            # Remembered appearance only vetoes at a similar size; a ball seen at
            # several times its last size (approaching) cannot be compared reliably.
            if (self._hint is None and self._template is not None and visible > .95
                    and similarity < .50 and (self._last_radius is None
                                              or .67 < radius/self._last_radius < 1.5)):
                continue
            edge_threshold = .35 if local and similarity > .55 else (.65 if radius < 10 else .56)
            if motion > .40 and radius >= 10:
                edge_threshold = min(edge_threshold, .35)
            if support < edge_threshold:
                continue
            if motion * radius < .40 and not (local and similarity>.65
                                               and self._frames-self._last_motion_frame <= 6):
                continue
            # The classifier is the costliest check, so it runs on survivors only.
            appearance_margin = self._timed("classifier", self.verifier.score, gray, x, y, radius) if self.verifier else 0.
            partial = x-radius < 0 or x+radius > w
            margin_threshold = -.6 if self._hint is not None else (-.6 if radius < 25 or partial else .60)
            above_floor = (self.auto_floor and floor.boundary_y is not None
                           and floor.confidence > .35 and (y+radius)/h < floor.boundary_y-.05)
            if self._hint is None and (radius < 10 or above_floor):
                margin_threshold = max(margin_threshold, .60)
            if self.verifier and appearance_margin < margin_threshold:
                continue
            score=.50*support+.35*min(1.,motion*radius/3.)+.15*max(0.,similarity)
            if self.verifier:
                score=.70*score+.30*float(1/(1+np.exp(-appearance_margin)))
            if self._hint is None:
                # Geometry only breaks ties; motion remains necessary above or below floor.
                if self.floor_region is not None:
                    fx0,fy0,fx1,fy1=self.floor_region
                    floor_bonus=fx0<=x/w<=fx1 and fy0<=(y+radius)/h<=fy1
                else:
                    floor_bonus=floor.boundary_y is not None and floor.confidence>.35 and (y+radius)/h>=floor.boundary_y - .05
                score += .15 * bool(floor_bonus)
            bx,by=max(0,int(x-radius)),max(0,int(y-radius))
            ex,ey=min(w,int(x+radius+1)),min(h,int(y+radius+1))
            if ex>bx and ey>by:
                candidates.append(BallCandidate((x,y),radius,min(1.,score),(bx,by,ex-bx,ey-by)))
        return candidates

    def _coherent_acquisition(self, candidates):
        """Follow competing proposals before committing; reject stationary flicker."""
        for track in self._tracklets:
            track["position"] = self._warp_point(track["position"])
            track["history"] = deque((
                (frame, self._warp_point(point), radius)
                for frame, point, radius in track["history"]), maxlen=5)
            track["misses"] += 1
        used = set()
        accepted = []
        now = self._timestamp
        # Gates were tuned for consecutive 60 fps frames. When processing falls
        # behind, predict each tracklet forward and widen the gate with the gap.
        expected = []
        for track in self._tracklets:
            gap = 1. if now is None or track["time"] is None else max(1., (now-track["time"])*60)
            position = track["position"]
            if track["velocity"] is not None and now is not None:
                position = (position[0]+track["velocity"][0]*(now-track["time"]),
                            position[1]+track["velocity"][1]*(now-track["time"]))
            expected.append((position, min(gap, 2. if track["velocity"] is not None else 6.)))
        for candidate in candidates[:40]:
            x, y = candidate.center
            choices = [(hypot(x-position[0], y-position[1]) / (max(8, candidate.radius*.7)*scale), index)
                       for index, (track, (position, scale)) in enumerate(zip(self._tracklets, expected))
                       if index not in used and .65 < candidate.radius/track["radius"] < 1.55]
            distance, index = min(choices, default=(float("inf"), -1))
            if distance <= 1:
                track = self._tracklets[index]
                used.add(index)
                if now is not None and track["time"] is not None and now > track["time"]:
                    track["velocity"] = ((x-track["position"][0])/(now-track["time"]),
                                         (y-track["position"][1])/(now-track["time"]))
                track["time"] = now
                track["position"] = candidate.center
                track["radius"] = candidate.radius
                track["misses"] = 0
                track["history"].append((self._frames, candidate.center, candidate.radius))
                history = list(track["history"])
                if len(history) >= 3:
                    first_frame, origin, first_radius = history[0]
                    span = self._frames - first_frame
                    net = hypot(x-origin[0], y-origin[1])
                    path = sum(hypot(b[1][0]-a[1][0], b[1][1]-a[1][1])
                               for a,b in zip(history,history[1:]))
                    floor = self.floor_estimator.estimate
                    above_floor = (self.auto_floor and floor.boundary_y is not None
                                   and floor.confidence > .35 and
                                   (y+candidate.radius)/self._gray.shape[0] < floor.boundary_y-.05)
                    # Slow round patches high in the scene need stronger motion proof.
                    # Fast coherent airborne candidates remain eligible.
                    speed_gate = (max(2.5,candidate.radius*.15) if above_floor else max(.6,candidate.radius*.04)) if self.verifier else (max(2.5, candidate.radius*.25) if above_floor else max(1.2, candidate.radius*.08))
                    translating = (net / max(1, span) > speed_gate
                                   and net / max(path,.01) > .70)
                    expanding = ((not above_floor or (candidate.confidence > .80 and len(history) >= 5))
                                 and candidate.radius >= 10 and
                                 candidate.radius/first_radius > 1 + .03*span and
                                 all(b[2] >= a[2]*.98 for a,b in zip(history,history[1:])))
                    # Pixel texture changes also indicate axial motion, whose
                    # center barely moves. A recent known ball needs matching size
                    # and strong appearance, rather than a new translation gate.
                    remembered = (self.verifier is not None and self._memory_active
                                  and .45 <= candidate.radius/self._last_radius <= 2.2
                                  and candidate.confidence > .75)
                    if translating or expanding or remembered:
                        accepted.append(candidate)
            else:
                self._tracklets.append(dict(position=candidate.center, radius=candidate.radius,
                                            history=deque([(self._frames,candidate.center,candidate.radius)],maxlen=5),
                                            misses=0, time=now, velocity=None))
                used.add(len(self._tracklets)-1)
        self._tracklets = [track for track in self._tracklets if track["misses"] < 3][-60:]
        return tuple(accepted)
