"""Pick a fixed exposure and gain from the live image before tracking starts.

Tracking wants a fixed exposure (auto exposure changing mid-kick changes the
ball's appearance) that is short (a kicked ball smears across the exposure).
Exposure is the control up to ``blur_limit_us``; past that, gain rises (up to
``max_gain``) before exposure goes on to ``max_exposure_us``, since extra gain
cost no visible noise on the Nov 6 clips while longer exposure adds blur.

Only ``meter_rows`` (fractions of the image height; default the lower 60%,
where the ball and floor are) are metered: ceiling lights and the sky above
the scene neither set the level nor stop it rising by clipping. The level is
the mean of the subsampled metered rows; ``clipped`` is their fraction of
saturated pixels. Areas that stay saturated however short the exposure
(windows, sunlit doors) are light sources and are left out; any other clipping
sets a ceiling, so if a bright area saturates before the mean reaches the
target, the brightest unclipped setting is used.

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


def _settings(light, preferred_gain, blur_limit_us, min_exposure_us, max_exposure_us, min_gain, max_gain):
    """(exposure_us, gain) for exposure x gain = ``light``: exposure first up to
    the blur limit, then gain up to its maximum, then exposure up to its maximum."""
    exposure_us = float(np.clip(light / preferred_gain, min_exposure_us, min(blur_limit_us, max_exposure_us)))
    gain = float(np.clip(light / exposure_us, min_gain, max_gain))
    if gain >= max_gain and light / max_gain > exposure_us:
        exposure_us = float(np.clip(light / max_gain, exposure_us, max_exposure_us))
    return round(exposure_us), round(gain, 2)


def calibrate_exposure(source, *, exposure_us=2000, gain=8.0, target=120, max_exposure_us=3000,
                       blur_limit_us=1000, meter_rows=(.4, 1.), min_exposure_us=30, min_gain=1.0,
                       max_gain=16.0, max_clipped=.02, max_light_sources=.25, rounds=8, info=None):
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
        rows = pixels.shape[0]
        pixels = pixels[int(meter_rows[0] * rows):max(int(meter_rows[1] * rows), int(meter_rows[0] * rows) + 1)]
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
        new_exposure, new_gain = _settings(light, preferred_gain, blur_limit_us, min_exposure_us,
                                           max_exposure_us, min_gain, max_gain)
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
                        help="Analogue gain; calibration raises it only past --blur-limit-us")
    parser.add_argument("--target-brightness", type=float, default=120,
                        help="Startup calibration: mean pixel level to aim for (0-255)")
    parser.add_argument("--blur-limit-us", type=int, default=1000,
                        help="Startup calibration: longest exposure before raising gain (motion blur)")
    parser.add_argument("--max-exposure-us", type=int, default=3000,
                        help="Startup calibration: longest exposure once gain is at its maximum")
    parser.add_argument("--meter-rows", type=_fraction_pair, default=(.4, 1.),
                        help="Startup calibration: image rows to meter, as fractions top,bottom "
                             "(default .4,1 = lower 60%%, where the ball is; 0,1 = whole frame)")
    parser.add_argument("--auto-exposure", action="store_true",
                        help="Camera's continuous auto exposure (changes during a kick)")


def _fraction_pair(value):
    import argparse
    try:
        top, bottom = (float(v) for v in value.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected two fractions, e.g. .4,1") from None
    if not 0 <= top < bottom <= 1:
        raise argparse.ArgumentTypeError("need 0 <= top < bottom <= 1")
    return top, bottom


def check_exposure_arguments(parser, args):
    if not args.auto_exposure and max(args.exposure_us or 0, args.max_exposure_us,
                                      args.blur_limit_us) > 1e6 / args.fps:
        parser.error("--exposure-us, --blur-limit-us and --max-exposure-us must fit within the frame period")


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
                max_exposure_us=args.max_exposure_us, blur_limit_us=args.blur_limit_us,
                meter_rows=args.meter_rows, info=info)
        except BaseException:
            source.close()
            raise
        logging.log(logging.INFO if converged else logging.WARNING,
                    "Exposure %s: %d us, gain %.2f -> brightness %.0f (target %.0f), "
                    "%.1f%% clipped, metering rows %.0f-%.0f%%%s", "calibrated" if converged else "at its limits",
                    exposure_us, gain, level, args.target_brightness, clipped * 100,
                    args.meter_rows[0] * 100, args.meter_rows[1] * 100,
                    "" if converged else "; scene too dark/bright for the caps")
        if info.get("light_source_fraction"):
            logging.info("Left %.0f%% of the frame out as light sources (saturated at any exposure)",
                         info["light_source_fraction"] * 100)
        logging.info("To reuse these settings exactly: --exposure-us %d --gain %.2f",
                     exposure_us, gain)
        args.exposure_us, args.gain = exposure_us, gain
    return source
