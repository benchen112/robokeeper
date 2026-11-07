"""Lens sharpness feedback for focusing a camera by hand (tools/focus_check.py).

Sharpness is the variance of the Laplacian over a region of interest: fine
detail (edges, text) raises it, defocus lowers it. Exposure must stay fixed
while focusing, since brightness and sensor noise change the number too. The
value only compares settings of one lens looking at one unchanged scene; it has
no absolute meaning.
"""

from collections import deque

import cv2
import numpy as np


def sharpness(gray, roi=(.3, .3, .7, .7)):
    """Laplacian variance of the normalized ``roi`` (x0, y0, x1, y1) of ``gray``."""
    h, w = gray.shape[:2]
    x0, y0, x1, y1 = roi
    region = gray[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]
    # A light blur keeps sensor noise (which rises with gain) from counting as detail.
    region = cv2.GaussianBlur(region, (3, 3), 0)
    return float(cv2.Laplacian(region, cv2.CV_32F).var())


class FocusMeter:
    """Smoothed sharpness, the best value seen, and which way it is heading.

    ``update`` takes one frame's sharpness; ``trend`` compares the last
    ``window`` frames with the ``window`` frames ``lag`` frames earlier, and
    calls a change only beyond ``tolerance`` and beyond the frame-to-frame noise.
    """

    def __init__(self, window=6, lag=30, tolerance=.03):
        self.window = window
        self.lag = lag
        self.tolerance = tolerance
        self.values = deque(maxlen=window + lag)
        self.best = 0.

    def update(self, value):
        self.values.append(value)
        if len(self.values) >= self.window:
            self.best = max(self.best, self.current)

    def reset_best(self):
        self.best = self.current if len(self.values) >= self.window else 0.

    @property
    def current(self):
        recent = list(self.values)[-self.window:]
        return float(np.mean(recent)) if recent else 0.

    @property
    def fraction_of_best(self):
        return self.current / self.best if self.best else 0.

    @property
    def trend(self):
        """'sharper', 'blurrier', 'steady', or '' until enough frames."""
        if len(self.values) < self.values.maxlen:
            return ""
        values = list(self.values)
        before, now = np.array(values[:self.window]), np.array(values[-self.window:])
        change = now.mean() - before.mean()
        noise = 3 * np.sqrt((before.var(ddof=1) + now.var(ddof=1)) / self.window)
        if abs(change) <= max(self.tolerance * before.mean(), noise):
            return "steady"
        return "sharper" if change > 0 else "blurrier"


def bar(fraction, width=10):
    filled = int(round(min(max(fraction, 0.), 1.) * width))
    return "#" * filled + "." * (width - filled)
