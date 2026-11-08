"""Folders and labels for runs and recordings: one folder per day, one per test.

    runs/2026-11-08/4_chips_left_center_right/left1_x-4.5ft_height2.4ft__kick_20261108_164940.json

``session_folder(root, "chips left")`` gives ``root/<today>/<n>_chips_left``,
reusing today's folder of that name or numbering a new one after the existing
ones. A label typed after a kick ("left1 x-4.5 h2.4", feet) becomes the file
name prefix, before ``__``; the original timestamped name is kept after it.
"""

from datetime import datetime
from pathlib import Path
import re

LABEL_SEPARATOR = "__"


def _safe(text):
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", text.strip()).strip("_")


def session_folder(root, test=None, now=None):
    """``root/YYYY-MM-DD``, or ``root/YYYY-MM-DD/<n>_<test>`` (not created)."""
    day = Path(root) / f"{(now or datetime.now()):%Y-%m-%d}"
    if not test:
        return day
    name = _safe(test)
    if not name:
        raise ValueError("Test name needs letters or digits")
    numbers = []
    if day.is_dir():
        for folder in day.iterdir():
            match = re.fullmatch(r"(\d+)_(.+)", folder.name)
            if folder.is_dir() and match:
                if match.group(2) == name:
                    return folder
                numbers.append(int(match.group(1)))
    return day / f"{max(numbers, default=0) + 1}_{name}"


def parse_measurement(text):
    """'left1 x-4.5 h2.4' -> ('left1_x-4.5ft_height2.4ft', {'name':..., 'x_ft':..., 'height_ft':...}).

    Tokens ``x<feet>`` and ``h<feet>`` (or ``height<feet>``) are measurements;
    other words name the kick. Raises ValueError when nothing usable is given.
    """
    names, measured = [], {}
    for token in text.split():
        match = re.fullmatch(r"(x|h|height)=?([+-]?\d+(?:\.\d+)?)(?:ft)?", token, re.IGNORECASE)
        if match:
            key = "x_ft" if match.group(1).lower() == "x" else "height_ft"
            measured[key] = float(match.group(2))
        else:
            names.append(token)
    name = _safe("_".join(names))
    parts = [name] if name else []
    if "x_ft" in measured:
        parts.append(f"x{measured['x_ft']:+.1f}ft")
    if "height_ft" in measured:
        parts.append(f"height{measured['height_ft']:.1f}ft")
    if not parts:
        raise ValueError(f"Nothing to label in {text!r}")
    if name:
        measured["name"] = name
    return "_".join(parts), measured


def labelled_path(path, label):
    """``path`` renamed to ``<label>__<original name>`` (an existing label is replaced)."""
    path = Path(path)
    original = path.name.split(LABEL_SEPARATOR)[-1]
    return path.with_name(f"{label}{LABEL_SEPARATOR}{original}")
