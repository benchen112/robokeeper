"""Pick a fixed exposure and gain from the live image before tracking starts.

Tracking wants a fixed exposure (auto exposure changing mid-kick changes the
ball's appearance) that is as short as possible (a kicked ball smears across
the exposure). So exposure is the control, capped at ``max_exposure_us``; gain
stays at the preferred value and only rises once exposure hits the cap (or
falls once exposure is at its minimum). The level is the mean of a subsampled
frame; ``clipped`` is the fraction of saturated pixels. Clipping sets a ceiling:
if a bright area (a window, a light) saturates before the mean reaches the
target, the brightest setting that does not clip is used instead.

``source`` needs ``read(sequence)`` (as LatestPicameraStereo) and
``set_exposure(exposure_us, gain)``.
"""

import numpy as np

# The Camarray YUV stream is limited range: white is 235, not 255.
SATURATED = 230


def _measure(source, sequence, exposure_us, gain, settle_frames=12):
    """Mean level and clipped fraction once frames report the requested settings."""
    levels, clipped = [], []
    for _ in range(settle_frames + 2):
        sample = source.read(sequence)
        if sample is None:
            return sequence, None, None
        sequence, frame, _, _, metadata = sample
        reported_us = metadata.get("exposure_us")
        reported_gain = metadata.get("analogue_gain")
        applied = (reported_us is None or abs(reported_us - exposure_us) <= max(20, .03 * exposure_us)) and (
            reported_gain is None or abs(reported_gain - gain) <= max(.1, .03 * gain))
        if not applied:
            continue
        pixels = frame[::4, ::4]
        levels.append(float(pixels.mean()))
        clipped.append(float((pixels >= SATURATED).mean()))
        if len(levels) == 2:
            break
    if not levels:  # Settings never reported as applied; use the latest frame anyway.
        pixels = frame[::4, ::4]
        levels, clipped = [float(pixels.mean())], [float((pixels >= SATURATED).mean())]
    return sequence, float(np.mean(levels)), float(np.mean(clipped))


def calibrate_exposure(source, *, exposure_us=2000, gain=8.0, target=120, max_exposure_us=3000,
                       min_exposure_us=30, min_gain=1.0, max_gain=16.0, max_clipped=.02,
                       rounds=8):
    """Returns (exposure_us, gain, level, clipped, converged); leaves the camera set to them."""
    preferred_gain = gain
    sequence, level, clipped = 0, None, None
    ceiling = float("inf")  # exposure x gain at which saturation appeared
    for _ in range(rounds):
        source.set_exposure(exposure_us, gain)
        sequence, level, clipped = _measure(source, sequence, exposure_us, gain)
        if level is None:
            raise RuntimeError("No camera frames during exposure calibration")
        current = exposure_us * gain
        if clipped > max_clipped:
            ceiling = min(ceiling, current)
        elif abs(level - target) <= target * .12:
            return exposure_us, gain, level, clipped, True
        ratio = target / max(level, 1.)
        if clipped > max_clipped:
            ratio = min(ratio, .7)
        light = min(current * float(np.clip(ratio, .25, 4.)), ceiling * .9)
        if ratio > 1 and clipped <= max_clipped and light <= current * 1.03:
            return exposure_us, gain, level, clipped, True  # As bright as clipping allows.
        new_exposure = round(float(np.clip(light / preferred_gain, min_exposure_us, max_exposure_us)))
        new_gain = round(float(np.clip(light / new_exposure, min_gain, max_gain)), 2)
        if (new_exposure, new_gain) == (exposure_us, gain):
            break  # At a limit: as dark/bright as the caps allow.
        exposure_us, gain = new_exposure, new_gain
    source.set_exposure(exposure_us, gain)
    return exposure_us, gain, level, clipped, False
