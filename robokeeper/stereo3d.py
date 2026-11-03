"""Stereo calibration and triangulation of ball centers in meters.

World frame: origin midway between the two lenses, X to the right, height up
and Z forward (depth). The plane Z = 0 is the camera plane the ball crosses.
"""

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Point3D:
    x_m: float
    height_m: float
    z_m: float
    disparity_px: float


@dataclass(frozen=True)
class StereoCalibration:
    """Rectified pinhole model shared by both eyes.

    ``cy`` is the row of a level ray. For the provisional fit it is the horizon
    row measured from a rolling ball, so heights are level with the ground.
    ``vertical_offset_px`` is left y minus right y and ``disparity_offset_px``
    is the disparity of a point at infinity; both are zero after checkerboard
    rectification, whose per-eye undistort/rectify parameters are stored in
    ``rectification``. ``camera_height_m`` is added to triangulated heights:
    with ``cy`` at the true horizon it is the cameras' height above a rolling
    ball's center, so height 0 stays the rolling-ball center at any mount
    height. ``swap_eyes`` means the camera delivers the physical right camera
    in the left half of the packed frame; offsets are for the physical order.
    """

    focal_px: float
    cx: float
    cy: float
    baseline_m: float
    disparity_offset_px: float = 0.0
    vertical_offset_px: float = 0.0
    image_size: tuple[int, int] = (1280, 800)
    method: str = "provisional"
    notes: str = ""
    rectification: dict | None = field(default=None, compare=False)
    camera_height_m: float = 0.0
    swap_eyes: bool = False

    def __post_init__(self):
        if not self.focal_px > 0 or not self.baseline_m > 0:
            raise ValueError("Calibration needs a positive focal length and baseline")

    def rectify(self, left_xy, right_xy):
        """Map raw eye pixels to the rectified model."""
        if self.rectification is None:
            return ((float(left_xy[0]), float(left_xy[1])),
                    (float(right_xy[0]), float(right_xy[1]) + self.vertical_offset_px))
        import cv2

        rectified = []
        for eye, point in (("left", left_xy), ("right", right_xy)):
            params = {key: np.asarray(value, np.float64)
                      for key, value in self.rectification[eye].items()}
            mapped = cv2.undistortPoints(np.array([[point]], np.float64), params["K"],
                                         params["D"], R=params["R"], P=params["P"])
            rectified.append(tuple(float(v) for v in mapped[0, 0]))
        return tuple(rectified)

    def triangulate(self, left_xy, right_xy, min_disparity_px=1.0, radius_px=None):
        """Ball center in meters, or None when the disparity is too small.

        With ``radius_px``, also None when the eyes disagree on height by more
        than the provisional calibration allows: the eyes saw different objects.
        """
        (xl, yl), (xr, yr) = self.rectify(left_xy, right_xy)
        if radius_px is not None and abs(yl - yr) > max(40.0, 0.6 * radius_px):
            return None
        disparity = xl - xr - self.disparity_offset_px
        if disparity < min_disparity_px:
            return None
        z = self.focal_px * self.baseline_m / disparity
        return Point3D(
            x_m=z * ((xl + xr + self.disparity_offset_px) / 2 - self.cx) / self.focal_px,
            height_m=self.camera_height_m - z * ((yl + yr) / 2 - self.cy) / self.focal_px,
            z_m=z, disparity_px=disparity)

    def project(self, point):
        """Raw eye pixels for a world point (inverse of triangulate, no distortion)."""
        if self.rectification is not None:
            raise ValueError("project() supports the provisional model only")
        x, height, z = point
        y = self.cy - self.focal_px * (height - self.camera_height_m) / z
        left = (self.cx + self.focal_px * (x + self.baseline_m / 2) / z, y)
        right = (self.cx + self.focal_px * (x - self.baseline_m / 2) / z
                 - self.disparity_offset_px, y - self.vertical_offset_px)
        return left, right

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        data["image_size"] = tuple(data["image_size"])
        return cls(**data)


def read_stereo_track(path, image_size=(1280, 800)):
    """Full-ball stereo pairs from a ``track_stereo.py --jsonl`` file.

    Returns ``(frame_index, timestamp_s, (left_xy, left_radius, right_xy,
    right_radius))`` for frames where both eyes confirmed the ball and neither
    view is clipped by the image edge.
    """
    pairs = []
    with open(path, encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if not row.get("both_confirmed_observed"):
                continue
            left, right = row["left"], row["right"]
            views = [(tuple(eye["centroid_px"]), eye["radius_px"]) for eye in (left, right)]
            if any(touches_border(center, radius, image_size) for center, radius in views):
                continue
            pairs.append((row["frame_index"], row["timestamp_s"],
                          (views[0][0], views[0][1], views[1][0], views[1][1])))
    return pairs


def touches_border(center, radius, image_size, margin=2.0):
    """True when a circle is clipped by the image edge, biasing its centroid."""
    x, y = center
    width, height = image_size
    return (x - radius < margin or y - radius < margin
            or x + radius > width - margin or y + radius > height - margin)


def fit_kick_calibration(clips, baseline_m, image_size=(1280, 800), start_frames=5):
    """Provisional calibration from rolling-ball stereo tracks.

    ``clips`` is a list of ``(pairs, start_distance_m)``; each pair is
    ``(left_xy, left_radius, right_xy, right_radius)`` for frames where both
    eyes see the whole ball, in time order. The start distance is the depth
    of the first frames; for a kick, where the ball was kicked from.

    Assumes parallel optical axes (zero disparity at infinity), a principal
    point at the image center horizontally, and a ball rolling on level
    ground, so its center height is constant. Returns the calibration and a
    diagnostics dict.
    """
    if not clips:
        raise ValueError("Need at least one clip")
    offsets, focal_estimates, mean_rows, disparities = [], [], [], []
    for pairs, start_distance in clips:
        if len(pairs) < start_frames:
            raise ValueError(f"Each clip needs at least {start_frames} full-ball stereo pairs")
        rows = np.array([(lx, ly, rx, ry) for (lx, ly), _, (rx, ry), _ in pairs], np.float64)
        offsets.extend(rows[:, 1] - rows[:, 3])
        if start_distance is not None:
            # Depth is proportional to 1/disparity and changes nearly linearly
            # over the first frames, so extrapolate 1/disparity back to the
            # first tracked frame (the ball is already moving by the later
            # ones). Theil-Sen resists one bad early segmentation.
            early = rows[:2 * start_frames]
            inverse = 1 / (early[:, 0] - early[:, 2])
            index = np.arange(len(early))
            i, j = np.triu_indices(len(early), 1)
            slope = np.median((inverse[j] - inverse[i]) / (index[j] - index[i]))
            first_inverse = np.median(inverse - slope * index)
            focal_estimates.append(start_distance / first_inverse / baseline_m)
        disparities.append(rows[:, 0] - rows[:, 2])
        mean_rows.append(rows[:, [1, 3]])
    if not focal_estimates:
        raise ValueError("At least one clip needs a known start distance")
    vertical_offset = float(np.median(offsets))
    disparity = np.concatenate(disparities)
    rows = np.concatenate(mean_rows)
    mean_y = (rows[:, 0] + rows[:, 1] + vertical_offset) / 2
    # A constant-height ball gives y = horizon + (height below camera / B) * disparity.
    slope, horizon = np.polyfit(disparity, mean_y, 1)
    focal = float(np.mean(focal_estimates))
    calibration = StereoCalibration(
        focal_px=focal, cx=image_size[0] / 2, cy=float(horizon), baseline_m=baseline_m,
        vertical_offset_px=vertical_offset, image_size=tuple(image_size),
        method="provisional_from_kicks",
        notes="Fitted from rolling-ball clips with known start distances; assumes "
              "parallel cameras and no lens distortion. Replace with a checkerboard "
              "calibration for accurate 3D.")
    diagnostics = {
        "focal_estimates_px": [round(float(v), 1) for v in focal_estimates],
        "vertical_offset_spread_px": round(float(np.percentile(offsets, 75)
                                                 - np.percentile(offsets, 25)), 1),
        "ball_center_below_camera_m": round(float(slope * baseline_m), 3),
        "horizontal_fov_deg": round(float(np.degrees(2 * np.arctan(image_size[0] / 2 / focal))), 1),
    }
    return calibration, diagnostics


def fit_roll_calibration(pairs, base, ball_radius_m=0.11, rolling=True, min_pairs=20):
    """Refit how the two cameras are aimed from a ball rolled toward them.

    ``pairs`` are ``(a_xy, a_radius, b_xy, b_radius)`` with ``a`` the left half
    of the packed frame as captured (no swap), for frames where both halves see
    the whole ball. Focal length and baseline come from ``base`` (they do not
    change when the rig is moved); refit are the eye order (the sign of the
    parallax: disparity must grow as the ball, and its radius, grows), the
    disparity offset (disparity vs. the ball size's ``baseline * r / R``), the
    vertical offset (row difference) and, for a ball rolling on the ground, the
    horizon row and the cameras' height above the ball's center. Returns the
    calibration and a diagnostics dict.
    """
    if base.rectification is not None:
        raise ValueError("This calibration is checkerboard-rectified; after moving the "
                         "cameras, redo the checkerboard calibration")
    rows = np.array([(ax, ay, ar, bx, by, br) for (ax, ay), ar, (bx, by), br in pairs], np.float64)
    if len(rows) < min_pairs:
        raise ValueError(f"Only {len(rows)} pairs with the ball in both eyes; need {min_pairs}. "
                         "Roll the ball toward the cameras through both views")
    radius = (rows[:, 2] + rows[:, 5]) / 2
    # The same object in both eyes: similar size and a consistent row difference.
    keep = np.minimum(rows[:, 2], rows[:, 5]) / np.maximum(rows[:, 2], rows[:, 5]) > .7
    dy = rows[:, 1] - rows[:, 4]
    keep &= np.abs(dy - np.median(dy[keep])) <= np.maximum(8., .5 * radius)
    rows, radius = rows[keep], radius[keep]
    if len(rows) < min_pairs:
        raise ValueError(f"Only {len(rows)} consistent pairs; need {min_pairs}")
    if radius.max() / radius.min() < 1.5:
        raise ValueError(f"The ball barely changed size ({radius.min():.0f}-{radius.max():.0f} px): "
                         "roll it from far away toward the cameras")
    expected = base.baseline_m * radius / ball_radius_m  # disparity at the size's distance
    raw = rows[:, 0] - rows[:, 3]
    slope = float(np.polyfit(expected, raw, 1)[0])
    swap = slope < 0
    left, right = (rows[:, 3:], rows[:, :3]) if swap else (rows[:, :3], rows[:, 3:])
    residual = left[:, 0] - right[:, 0] - expected
    disparity_offset = float(np.median(residual))
    vertical = left[:, 1] - right[:, 1]
    vertical_offset = float(np.median(vertical))
    disparity = left[:, 0] - right[:, 0] - disparity_offset
    mean_y = (left[:, 1] + right[:, 1] + vertical_offset) / 2
    cy, camera_height = base.cy, base.camera_height_m
    if rolling:
        # A ball at constant height: row = horizon + (camera above ball / baseline) * disparity.
        k, cy = (float(v) for v in np.polyfit(disparity, mean_y, 1))
        camera_height = k * base.baseline_m
    spread = lambda v: round(float(np.percentile(v, 75) - np.percentile(v, 25)), 1)
    calibration = replace(
        base, cy=cy, camera_height_m=camera_height, disparity_offset_px=disparity_offset,
        vertical_offset_px=vertical_offset, swap_eyes=bool(swap), method="provisional_from_roll",
        notes=("Eye order, offsets" + (", horizon and camera height" if rolling else "")
               + " refit from a rolled ball (tools/calibrate_stereo.py roll); focal length and "
               "baseline kept from the previous calibration. Assumes parallel-ish cameras and "
               "no lens distortion; a checkerboard calibration is more accurate."))
    diagnostics = {
        "pairs_used": int(len(rows)), "pairs_dropped": int((~keep).sum()),
        "ball_radius_px": [round(float(radius.min()), 1), round(float(radius.max()), 1)],
        "eyes_swapped": bool(swap), "parallax_slope": round(abs(slope), 2),
        "disparity_offset_spread_px": spread(residual), "vertical_offset_spread_px": spread(vertical),
    }
    if rolling:
        diagnostics["camera_above_ball_center_m"] = round(camera_height, 3)
    if not .6 < abs(slope) < 1.5:
        diagnostics["warning"] = ("Parallax does not match the ball's size change (slope "
                                  f"{abs(slope):.2f}, expected ~1): check --ball-radius-m, "
                                  "the baseline, or that one ball was tracked")
    return calibration, diagnostics
