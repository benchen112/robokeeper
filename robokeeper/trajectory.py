"""Predict where a ball crosses the camera plane, and aim a servo at it.

Uses the stereo3d world frame: X right, height up, Z forward; the camera
plane is Z = 0. The predictor is causal (it only sees past samples), so the
same code serves offline replay and live tracking.
"""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

GRAVITY_MPS2 = 9.81


@dataclass(frozen=True)
class Crossing:
    time_s: float           # predicted crossing time on the sample clock
    time_to_cross_s: float  # from the latest sample
    x_m: float              # lateral position in the camera plane (+ = right)
    height_m: float         # ball-center height relative to the camera
    x_std_m: float          # 1-sigma lateral uncertainty from the fit
    velocity_mps: tuple[float, float, float]
    speed_mps: float
    model: str              # "rolling" or "airborne"
    samples_used: int


def _weighted_line(t, values, weights):
    """Weighted least-squares a + b t; returns coefficients, covariance, residuals."""
    design = np.column_stack([np.ones_like(t), t])
    w = np.sqrt(weights)
    coefficients, *_ = np.linalg.lstsq(design * w[:, None], values * w, rcond=None)
    residuals = values - design @ coefficients
    dof = max(1, len(t) - 2)
    scale = float(np.sum(weights * residuals**2) / dof)
    covariance = scale * np.linalg.inv(design.T @ (design * weights[:, None]))
    return coefficients, covariance, residuals


class TrajectoryPredictor:
    """Fit recent 3D ball positions and extrapolate to the plane Z = plane_z.

    Lateral and forward motion are fitted as constant velocity over the last
    ``window_s`` seconds, which suits a rolling ball and the short horizon of
    a shot. Height is constant for a rolling ball, or follows gravity when
    that fits clearly better (a lofted ball). Samples are weighted by stereo
    depth noise, which grows with distance squared, and gross outliers are
    dropped once before the final fit. When processing is slow and fewer than
    ``min_samples`` fall in the window, it stretches to the latest
    ``min_samples`` samples, up to ``max_window_s``.
    """

    def __init__(self, plane_z=0.0, window_s=0.6, min_samples=5,
                 min_approach_mps=0.3, max_history=240, max_window_s=1.2):
        self.plane_z = plane_z
        self.window_s = window_s
        self.max_window_s = max_window_s
        self.min_samples = min_samples
        self.min_approach_mps = min_approach_mps
        self.samples = deque(maxlen=max_history)

    def reset(self):
        self.samples.clear()

    def add(self, timestamp, point):
        """Add a triangulated ball center (x, height, z) in meters."""
        if self.samples and timestamp <= self.samples[-1][0]:
            raise ValueError("Trajectory samples must have increasing timestamps")
        x, height, z = point
        self.samples.append((timestamp, x, height, z))

    def predict(self):
        if len(self.samples) < self.min_samples:
            return None
        data = np.array(self.samples, np.float64)
        recent = data[data[:, 0] >= data[-1, 0] - self.window_s]
        if len(recent) < self.min_samples:
            recent = data[-self.min_samples:]
            recent = recent[recent[:, 0] >= data[-1, 0] - self.max_window_s]
        data = recent
        if len(data) < self.min_samples:
            return None
        latest = data[-1, 0]
        t = data[:, 0] - latest
        weights = 1 / np.maximum(data[:, 3], 0.2) ** 4
        keep = np.ones(len(t), bool)
        for _ in range(2):
            z_fit, z_cov, z_res = _weighted_line(t[keep], data[keep, 3], weights[keep])
            normalized = np.abs(z_res) * np.sqrt(weights[keep])
            if keep.sum() <= self.min_samples:
                break
            limit = 3.5 * max(float(np.median(normalized)), 1e-9)
            inliers = normalized <= limit
            if inliers.all():
                break
            keep[np.flatnonzero(keep)[~inliers]] = False
        t, data, weights = t[keep], data[keep], weights[keep]
        if len(t) < self.min_samples:
            return None
        z_fit, z_cov, _ = _weighted_line(t, data[:, 3], weights)
        vz = z_fit[1]
        if vz > -self.min_approach_mps:
            return None
        t_cross = (self.plane_z - z_fit[0]) / vz
        x_fit, x_cov, _ = _weighted_line(t, data[:, 1], weights)
        # Lateral uncertainty: fit covariance at the crossing time plus the
        # effect of crossing-time uncertainty from the depth fit.
        basis = np.array([1.0, t_cross])
        z_std = math.sqrt(max(0.0, basis @ z_cov @ basis))
        t_std = z_std / abs(vz)
        x_std = math.sqrt(max(0.0, basis @ x_cov @ basis) + (x_fit[1] * t_std) ** 2)

        heights = data[:, 2]
        rolling_level = float(np.average(heights, weights=weights))
        rolling_rms = math.sqrt(float(np.average((heights - rolling_level) ** 2, weights=weights)))
        lifted = heights + 0.5 * GRAVITY_MPS2 * t**2
        h_fit, _, h_res = _weighted_line(t, lifted, weights)
        airborne_rms = math.sqrt(float(np.average(h_res**2, weights=weights)))
        airborne = rolling_rms > 0.03 and airborne_rms < 0.6 * rolling_rms
        if airborne:
            height = h_fit[0] + h_fit[1] * t_cross - 0.5 * GRAVITY_MPS2 * t_cross**2
            vh = h_fit[1]
        else:
            height, vh = rolling_level, 0.0
        vx = x_fit[1]
        return Crossing(
            time_s=float(latest + t_cross), time_to_cross_s=float(t_cross),
            x_m=float(x_fit[0] + vx * t_cross), height_m=float(height),
            x_std_m=float(x_std), velocity_mps=(float(vx), float(vh), float(vz)),
            speed_mps=float(math.sqrt(vx**2 + vh**2 + vz**2)),
            model="airborne" if airborne else "rolling", samples_used=len(t))


def servo_angle_deg(x_m, height_m, pivot_x_m=0.0, pivot_height_m=0.0, limit_deg=90.0):
    """Angle for an arm rotating in the camera plane to point at (x, height).

    0 degrees points straight up from the pivot; positive turns toward +X,
    the right side as seen from the camera (the keeper's right when the
    camera faces the kicker). The result is clamped to +/- ``limit_deg``.
    """
    angle = math.degrees(math.atan2(x_m - pivot_x_m, height_m - pivot_height_m))
    return max(-limit_deg, min(limit_deg, angle))
