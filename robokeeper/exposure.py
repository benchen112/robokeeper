"""Pick a fixed exposure and gain from the live image before tracking starts.

Tracking wants a fixed exposure (auto exposure changing mid-kick changes the
ball's appearance) that is as short as possible (a kicked ball smears across
the exposure). So exposure is the control, capped at ``max_exposure_us``; gain
stays at the preferred value and only rises once exposure hits the cap (or
falls once exposure is at its minimum). The level is the mean of a subsampled
frame; ``clipped`` is the fraction of saturated pixels. Areas that stay saturated
however short the exposure (windows, sunlit doors) are light sources and are
left out; any other clipping sets a ceiling, so if a bright area saturates
before the mean reaches the target, the brightest unclipped setting is used.

``source`` needs ``read(sequence)`` (as LatestPicameraStereo) and
``set_exposure(exposure_us, gain)``.
"""

import numpy as np

# The Camarray YUV stream is limited range: white is 235, not 255.
SATURATED = 230


def _measure(source, sequence, exposure_us, gain, settle_frames=12):
    """A subsampled frame once frames report the requested settings."""
    pixels, applied_frames = None, 0
    for _ in range(settle_frames + 2):
        sample = source.read(sequence)
        if sample is None:
            return sequence, None
        sequence, frame, _, _, metadata = sample
        pixels = frame[::4, ::4]
        reported_us = metadata.get("exposure_us")
        reported_gain = metadata.get("analogue_gain")
        applied = (reported_us is None or abs(reported_us - exposure_us) <= max(20, .03 * exposure_us)) and (
            reported_gain is None or abs(reported_gain - gain) <= max(.1, .03 * gain))
        applied_frames += applied
        if applied_frames == 2:
            break
    return sequence, pixels  # Settings never reported as applied: the latest frame anyway.


def _level(pixels, ignore):
    """Mean level and saturated fraction of the pixels that are not light sources."""
    valid = pixels[~ignore]
    return float(valid.mean()), float((valid >= SATURATED).mean())


def calibrate_exposure(source, *, exposure_us=2000, gain=8.0, target=120, max_exposure_us=3000,
                       min_exposure_us=30, min_gain=1.0, max_gain=16.0, max_clipped=.02,
                       max_light_sources=.25, rounds=8, info=None):
    """Returns (exposure_us, gain, level, clipped, converged); leaves the camera set to them.

    Pixels still saturated after the light (exposure x gain) dropped 3x are light
    sources (windows, lamps, sunlit doors): they are left out of the level and the
    clipping check, so they cannot drag the exposure down, up to
    ``max_light_sources`` of the frame. ``info``, a dict, receives the fraction
    left out as ``light_source_fraction``.
    """
    preferred_gain = gain
    sequence, level, clipped = 0, None, None
    history = []  # (exposure x gain, saturated mask) per round
    ignore = None
    for _ in range(rounds):
        source.set_exposure(exposure_us, gain)
        sequence, pixels = _measure(source, sequence, exposure_us, gain)
        if pixels is None:
            raise RuntimeError("No camera frames during exposure calibration")
        current = exposure_us * gain
        saturated = pixels >= SATURATED
        if ignore is None:
            ignore = np.zeros(saturated.shape, bool)
        for light, earlier in history:
            if light >= 3 * current:
                widened = ignore | (earlier & saturated)
                if widened.mean() <= max_light_sources:
                    ignore = widened
        history.append((current, saturated))
        level, clipped = _level(pixels, ignore)
        # Clipping sets a ceiling: the least light at which non-light-source pixels saturated.
        ceiling = min((light for light, mask in history
                       if (mask & ~ignore).sum() > max_clipped * (~ignore).sum()), default=float("inf"))
        if info is not None:
            info["light_source_fraction"] = round(float(ignore.mean()), 3)
        if clipped <= max_clipped and abs(level - target) <= target * .12:
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


def add_exposure_arguments(parser):
    """Exposure options shared by live_keeper.py and record_stereo.py, so a clip
    recorded with the defaults is calibrated exactly as a live run would be."""
    parser.add_argument("--exposure-us", type=int,
                        help="Locked exposure (default: calibrate at startup, then lock)")
    parser.add_argument("--gain", type=float, default=8.0,
                        help="Analogue gain; calibration raises it only past --max-exposure-us")
    parser.add_argument("--target-brightness", type=float, default=120,
                        help="Startup calibration: mean pixel level to aim for (0-255)")
    parser.add_argument("--max-exposure-us", type=int, default=3000,
                        help="Startup calibration: longest exposure (motion blur) before more gain")
    parser.add_argument("--auto-exposure", action="store_true",
                        help="Camera's continuous auto exposure (changes during a kick)")


def check_exposure_arguments(parser, args):
    if not args.auto_exposure and max(args.exposure_us or 0, args.max_exposure_us) > 1e6 / args.fps:
        parser.error("--exposure-us and --max-exposure-us must fit within the frame period")


def open_calibrated_camera(args):
    """Open the stereo camera with fixed exposure, calibrating it unless
    --exposure-us or --auto-exposure was given. args.exposure_us and args.gain
    are set to what the camera ended up with (None with --auto-exposure)."""
    import logging

    from .stereo import LatestPicameraStereo

    calibrate = not args.auto_exposure and args.exposure_us is None
    start_us = min(2000, args.max_exposure_us) if calibrate else args.exposure_us
    source = LatestPicameraStereo(
        args.eye_width * 2, args.eye_height, args.fps, args.camera_num, args.tuning_file,
        None if args.auto_exposure else start_us, 1.0 if args.auto_exposure else args.gain)
    if args.auto_exposure:
        args.exposure_us = args.gain = None
        return source
    if calibrate:
        try:
            info = {}
            exposure_us, gain, level, clipped, converged = calibrate_exposure(
                source, exposure_us=start_us, gain=args.gain, target=args.target_brightness,
                max_exposure_us=args.max_exposure_us, info=info)
        except BaseException:
            source.close()
            raise
        logging.log(logging.INFO if converged else logging.WARNING,
                    "Exposure %s: %d us, gain %.2f -> brightness %.0f (target %.0f), "
                    "%.1f%% clipped%s", "calibrated" if converged else "at its limits",
                    exposure_us, gain, level, args.target_brightness, clipped * 100,
                    "" if converged else "; scene too dark/bright for the caps")
        if info.get("light_source_fraction"):
            logging.info("Left %.0f%% of the frame out as light sources (saturated at any exposure)",
                         info["light_source_fraction"] * 100)
        logging.info("To reuse these settings exactly: --exposure-us %d --gain %.2f",
                     exposure_us, gain)
        args.exposure_us, args.gain = exposure_us, gain
    return source
