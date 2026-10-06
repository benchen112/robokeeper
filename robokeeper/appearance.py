"""Grayscale HOG verifier; labeled training data remains outside inference."""
from pathlib import Path

import cv2
import numpy as np


class BallAppearanceVerifier:
    def __init__(self, model_path=None):
        path = Path(model_path) if model_path else Path(__file__).parent / "models/ball_hog_trees.xml.gz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing ball appearance model: {path}")
        self.tree = cv2.ml.RTrees_load(str(path))
        self.hog = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)
        if self.tree.getVarCount() != self.hog.getDescriptorSize():
            raise ValueError("Appearance model does not match feature dimensions")

    @staticmethod
    def patch(gray, x, y, radius):
        axis = np.linspace(-1.15, 1.15, 32, dtype=np.float32) * radius
        mx, my = np.meshgrid(axis+x, axis+y)
        return cv2.remap(gray, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def score(self, gray, x, y, radius):
        feature = self.hog.compute(self.patch(gray, x, y, radius)).reshape(1, -1)
        votes = self.tree.getVotes(feature, 0)
        positive = int(np.flatnonzero(votes[0] == 1)[0])
        fraction = np.clip(float(votes[1, positive] / votes[1].sum()), .001, .999)
        # A log vote ratio, not a calibrated probability of ball identity.
        return float(np.log(fraction / (1-fraction)))
