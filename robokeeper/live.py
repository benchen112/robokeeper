"""One armed kick: stereo tracks in, a predicted camera-plane crossing out.

Positions use the stereo3d world frame: origin midway between the two lenses,
X right (as seen from the camera), height up, Z forward. Height zero is the
calibration's level row: with the provisional kick calibration that is the
center of a ball rolling on the ground; after checkerboard calibration it is
the cameras' own height. Z is the perpendicular distance from the camera plane (the plane
through both lenses), not the straight-line distance from a lens.

The ball is not visible at Z = 0 (it leaves the field of view first), so the
crossing is always predicted. A session finishes when the camera clock passes
the latest predicted crossing time, or after ``max_wait_s`` without one.
"""

from dataclasses import asdict
import math

from .stereo3d import touches_border
from .servo import arm_angle_deg
from .trajectory import TrajectoryPredictor


def _confirmed(result):
    return result.state == "confirmed" and result.observed and result.center is not None


def _point_json(timestamp, point, start):
    return {"t_s": round(timestamp - start, 4), "x_m": round(point.x_m, 3),
            "height_m": round(point.height_m, 3), "z_m": round(point.z_m, 3),
            "range_m": round(math.dist((0, 0, 0), (point.x_m, point.height_m, point.z_m)), 3)}


def _crossing_json(crossing, now, point, start, aim):
    return {"x_m": round(crossing.x_m, 3), "height_m": round(crossing.height_m, 3),
            "x_std_m": round(crossing.x_std_m, 3), "speed_mps": round(crossing.speed_mps, 2),
            "model": crossing.model, "samples_used": crossing.samples_used,
            "time_to_cross_s": round(crossing.time_s - now, 3),
            "made_at_s": round(now - start, 4), "made_at": _point_json(now, point, start),
            "servo_deg": round(aim(crossing), 1)}


class KickSession:
    """Feed every processed stereo pair to ``update``; read ``summary()``."""

    def __init__(self, calibration, *, max_wait_s=8.0, window_s=0.6, min_samples=4, aim=None):
        self.calibration = calibration
        # Crossing -> servo arm angle; defaults to an arm hanging at the cameras.
        self.aim = aim or (lambda c: arm_angle_deg(c.x_m, c.height_m, c.model))
        self.max_wait_s = max_wait_s
        # Live processing on the Pi can fall to ~4 pairs/s during a kick, when
        # the ball may be in view for only a handful of processed pairs.
        self.predictor = TrajectoryPredictor(window_s=window_s, min_samples=min_samples)
        self.start = None
        self.points = []
        self.first_prediction = None
        self.prediction = None
        self.finished = None  # reason, once done

    @property
    def state(self):
        if self.finished:
            return "done"
        return "tracking" if self.points else "armed"

    def update(self, timestamp, left, right):
        """Returns True once the session has finished."""
        if self.finished:
            return True
        if self.start is None:
            self.start = timestamp
        point = None
        if _confirmed(left) and _confirmed(right) and not any(
                touches_border(r.center, r.radius, self.calibration.image_size)
                for r in (left, right)):
            point = self.calibration.triangulate(left.center, right.center,
                                                 radius_px=(left.radius + right.radius) / 2)
        if point is not None and point.z_m > 0:
            self.points.append(_point_json(timestamp, point, self.start))
            self.predictor.add(timestamp, (point.x_m, point.height_m, point.z_m))
            crossing = self.predictor.predict()
            if crossing is not None:
                self.prediction = (crossing, _crossing_json(crossing, timestamp, point, self.start, self.aim))
                self.first_prediction = self.first_prediction or self.prediction[1]
            elif len(self.predictor.samples) >= self.predictor.min_samples and self.prediction:
                # Enough fresh samples and no approach: the ball stopped or turned away.
                self.prediction = None
        if self.prediction and timestamp >= self.prediction[0].time_s:
            self.finished = "crossed"
        elif timestamp - self.start > self.max_wait_s:
            self.finished = "timeout"
        return bool(self.finished)

    def finish(self, reason):
        if not self.finished:
            self.finished = reason

    def summary(self):
        latest = self.prediction[1] if self.prediction else None
        return {
            "state": self.state, "reason": self.finished,
            "measurements": len(self.points),
            "first_measurement": self.points[0] if self.points else None,
            "last_measurement": self.points[-1] if self.points else None,
            "first_prediction": self.first_prediction,
            "crossing": latest,
            "frame": ("Origin midway between the lenses; x right, height up (0 = rolling-ball "
                      "center with the provisional calibration), z = perpendicular distance "
                      "from the camera plane; range = straight-line distance from the midpoint"),
        }

    def record(self):
        """Everything for the saved run file."""
        return {**self.summary(), "points": self.points, "calibration": asdict(self.calibration)}
