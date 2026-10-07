"""Motion-supported ball proposals and conservative automatic floor hints."""
from collections import deque
from dataclasses import dataclass
from math import hypot

import cv2
import numpy as np

from .appearance import BallAppearanceVerifier
from .vision import AppearanceBallSegmenter, BallCandidate

# Hough cost grows with the radius range. Acquisition scans of large motion
# regions search larger circles on a copy where they are at most this many pixels.
ACQUISITION_HOUGH_MAX_RADIUS = 48


@dataclass(frozen=True)
class FloorEstimate:
    boundary_y: float | None = None
    confidence: float = 0.0


class FloorEstimator:
    """Find a supported horizontal transition above a smoother bottom region.

    This estimates a floor-like image region, not a calibrated ground plane.
    Ambiguous scenes produce no estimate. It is only a ranking hint.
    """

    def __init__(self):
        self.estimate = FloorEstimate()
        self._history = deque(maxlen=5)
        self._frames = 0

    def update(self, gray):
        self._frames += 1
        if self._frames % 6 != 1:
            return self.estimate
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
    """

    def __init__(self, *, min_radius=3, max_radius=None, floor_region=None,
                 auto_floor=True, difference_threshold=12, verify_appearance=True,
                 fixed_camera=False):
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
        self.motion_scale = .5 if fixed_camera else 1.
        self.ready = False

    def reset(self):
        super().reset()
        self._previous = None
        self._background = None
        self._recent_motion = None
        self.motion_mask = None
        self.camera_transform = np.eye(3, dtype=np.float32)
        self.camera_motion_reliable = True
        self._last_motion_frame = 0
        self._timestamp = None
        self._last_observed_timestamp = None
        self._memory_active = False
        self._tracklets = []
        self.raw_candidates = ()
        self.acquisition_candidates = ()
        self.ready = False
        self.floor_estimator = FloorEstimator()

    def set_frame_timestamp(self, timestamp):
        self._timestamp = timestamp

    def observe(self, candidate):
        super().observe(candidate)
        self._last_observed_timestamp = self._timestamp
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
        gray = full if self.motion_scale == 1 else cv2.resize(
            full, None, fx=self.motion_scale, fy=self.motion_scale, interpolation=cv2.INTER_AREA)
        if self._previous is None:
            self._previous = gray.copy()
            self._background = gray.astype(np.float32)
            self._recent_motion = np.zeros(gray.shape, np.float32)
            self.motion_mask = np.zeros(full.shape, np.uint8)
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
        self.motion_mask = mask
        self._motion_integral = cv2.integral((mask > 0).astype(np.uint8))
        return reliable and fraction <= .20

    def _motion_support(self, x, y, radius, inner_fraction=0.0):
        # Integral rectangle/ring support bounds the cost even for large candidates.
        h, w = self.motion_mask.shape
        def rectangle(r):
            x0, y0 = max(0, int(x-r)), max(0, int(y-r))
            x1, y1 = min(w, int(x+r+1)), min(h, int(y+r+1))
            if x1 <= x0 or y1 <= y0:
                return 0., 0
            integral = self._motion_integral
            count = float(integral[y1,x1]-integral[y0,x1]-integral[y1,x0]+integral[y0,x0])
            return count, (x1-x0)*(y1-y0)
        count, area = rectangle(radius)
        if inner_fraction:
            inner_count, inner_area = rectangle(radius*inner_fraction)
            count, area = count-inner_count, area-inner_area
        return count/area if area else 0.

    def find_candidates(self, frame):
        if frame.dtype != np.uint8 or frame.ndim not in (2, 3):
            raise ValueError("Expected an 8-bit grayscale or BGR frame")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
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
        valid_motion = self._motion(gray)
        self.ready = self._frames > 1
        floor = self.floor_estimator.update(gray) if self.auto_floor else FloorEstimate()
        self.raw_candidates = ()
        self.acquisition_candidates = ()
        if not self.ready or not valid_motion:
            self._tracklets.clear()
            return ()
        proposals = []
        self.raw_candidates = ()
        self.acquisition_candidates = ()
        contours, _ = cv2.findContours(self.motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            if cv2.contourArea(contour) < 8 or max(bw, bh) < self.min_radius * 2:
                continue
            # A reflection can connect the ball to a tall foreground component.
            # Keep its upper portion instead of dropping the entire component.
            if bh > h * .35 and bw / bh < .45:
                bh = min(bh, max(40, bw * 2))
            pad = max(16, min(max(bw, bh) // 2, self.max_radius))
            regions.append((max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad)))
        regions.sort(key=lambda box: -float(self._motion_integral[box[3],box[2]]
                                           - self._motion_integral[box[1],box[2]]
                                           - self._motion_integral[box[3],box[0]]
                                           + self._motion_integral[box[1],box[0]]))
        # Prioritize motion regions near the prediction; cap work during crowded scenes.
        if self._hint is not None:
            (cx, cy), radius, margin = self._hint
            regions.sort(key=lambda box: hypot((box[0]+box[2])/2-cx, (box[1]+box[3])/2-cy))
            extent = radius + margin
            regions.insert(0, (max(0, int(cx-extent)), max(0, int(cy-extent)),
                               min(w, int(cx+extent)), min(h, int(cy+extent))))
        for x0, y0, x1, y1 in regions[:1 if self._hint is not None else 6]:
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

        if self._hint is not None and self._template is not None:
            (cx, cy), radius, margin = self._hint
            extent = radius + margin
            x0, y0 = max(0, int(cx-extent)), max(0, int(cy-extent))
            x1, y1 = min(w, int(cx+extent)), min(h, int(cy+extent))
            region = gray[y0:y1, x0:x1]
            for scale in (.85, 1., 1.15):
                rr = radius * scale
                size = max(5, round(1.4*rr))
                if min(region.shape, default=0) < size:
                    continue
                patch = cv2.resize(self._template, (size,size))
                _, score, _, location = cv2.minMaxLoc(cv2.matchTemplate(region,patch,cv2.TM_CCOEFF_NORMED))
                if score > .55:
                    proposals.append((x0+location[0]+(size-1)/2,y0+location[1]+(size-1)/2,rr))
        if self._hint is not None:
            (cx, cy), radius, _ = self._hint
            if cx-radius < 15 or cx+radius > w-15:
                step = max(4., radius*.10)
                proposals.extend((cx+dx, cy+dy, radius*scale)
                                 for dx in (-step, 0, step) for dy in (-step, 0, step)
                                 for scale in (.92, 1., 1.08)
                                 if self.min_radius <= radius*scale <= self.max_radius)
        candidates=[]
        for x,y,radius in proposals:
            if not (-radius*.5 <= x <= w+radius*.5 and 0 <= y < h):
                continue
            if self._hint is None and self._memory_active and not .45 <= radius/self._last_radius <= 2.2:
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
            patch,visible=self._patch(x,y,radius)
            if not local and patch.std() < 6 and support < .80:
                continue
            similarity=0.
            if self._template is not None and visible>.95:
                similarity=float(cv2.matchTemplate(patch,self._template,cv2.TM_CCOEFF_NORMED)[0,0])
            if self._hint is None and self._template is not None and visible > .95 and similarity < .50:
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
            appearance_margin = self.verifier.score(gray,x,y,radius) if self.verifier else 0.
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
            if any(hypot(x-c.center[0],y-c.center[1])<max(4,radius*.3) for c in candidates):
                continue
            bx,by=max(0,int(x-radius)),max(0,int(y-radius))
            ex,ey=min(w,int(x+radius+1)),min(h,int(y+radius+1))
            if ex>bx and ey>by:
                candidates.append(BallCandidate((x,y),radius,min(1.,score),(bx,by,ex-bx,ey-by)))
        candidates.sort(key=lambda c:c.confidence, reverse=True)
        self.raw_candidates = tuple(candidates)
        self.acquisition_candidates = self._coherent_acquisition(candidates)
        if self._hint is None:
            return self.acquisition_candidates
        return tuple(candidates)

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
        for candidate in candidates[:40]:
            x, y = candidate.center
            choices = [(hypot(x-track["position"][0], y-track["position"][1]), index)
                       for index, track in enumerate(self._tracklets)
                       if index not in used and .65 < candidate.radius/track["radius"] < 1.55]
            distance, index = min(choices, default=(float("inf"), -1))
            if distance <= max(8, candidate.radius*.7):
                track = self._tracklets[index]
                used.add(index)
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
                                            misses=0))
                used.add(len(self._tracklets)-1)
        self._tracklets = [track for track in self._tracklets if track["misses"] < 3][-60:]
        return tuple(accepted)
