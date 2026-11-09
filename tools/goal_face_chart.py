#!/usr/bin/env python3
"""Goal-face chart: measured vs predicted crossing for labelled live runs.

Looks at every run JSON under the given folders that has a "measured" x (and,
for chips, a height) and a predicted crossing, and draws the camera line as
seen from behind the cameras: one hollow ring per measured crossing, one dot
per prediction, joined by the error. Ground shots measured with x only are
placed at ball-center height (0.36 ft).

    python3 tools/goal_face_chart.py runs/2026-11-08 \
        --output tracking_review/2026-11-08/goal_face.png
"""

import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FT = 0.3048
BALL_RADIUS_FT = 0.11 / FT  # ball center above the floor when rolling
LENS_X_FT = 0.15 / FT       # each lens from the midpoint

# Reference palette (dataviz skill), light mode: categorical slots 1-2.
PREDICTED = "#2a78d6"
MEASURED = "#eb6834"
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def load_kicks(folders):
    kicks = []
    for folder in folders:
        for path in sorted(Path(folder).rglob("*.json")):
            try:
                run = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            measured, crossing = run.get("measured") or {}, run.get("crossing")
            if not isinstance(measured, dict) or "x_ft" not in measured or not crossing:
                continue
            if "note" in measured and "to" in measured["note"]:
                continue  # A range ("-3 to -4 ft"), not a measurement.
            chip = "height_ft" in measured
            kicks.append({
                "name": measured.get("name", path.stem),
                "chip": chip,
                "measured": (measured["x_ft"], measured["height_ft"] if chip else BALL_RADIUS_FT),
                # Tracker height 0 = rolling-ball center; add the radius for height above the floor.
                "predicted": (crossing["x_m"] / FT, crossing["height_m"] / FT + BALL_RADIUS_FT),
            })
    return kicks


def draw(kicks, output, title):
    errors = [((p[0] - m[0]) ** 2 + (p[1] - m[1]) ** 2) ** .5
              for m, p in ((k["measured"], k["predicted"]) for k in kicks)]
    chips = [k for k in kicks if k["chip"]]
    fig, ax = plt.subplots(figsize=(16, 9), dpi=120)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    xs = [v for k in kicks for v in (k["measured"][0], k["predicted"][0])]
    hs = [v for k in kicks for v in (k["measured"][1], k["predicted"][1])]
    span = max(7., max(abs(v) for v in xs) + 1.)
    ax.set_xlim(-span, span)
    ax.set_ylim(0, max(8.5, max(hs) + 1.))
    ax.set_aspect("equal")

    # Floor and the two lenses: the reference the measurements are taken from.
    ax.axhline(0, color=TEXT_SECONDARY, lw=2, zorder=1)
    for x in (-LENS_X_FT, LENS_X_FT):
        ax.plot([x], [0.12], marker="s", ms=9, color=TEXT_SECONDARY, zorder=3, clip_on=False)
    ax.annotate("cameras", (0, 0.12), xytext=(0, 14), textcoords="offset points",
                ha="center", color=TEXT_SECONDARY, fontsize=13)

    for k in kicks:
        (mx, mh), (px, ph) = k["measured"], k["predicted"]
        ax.plot([mx, px], [mh, ph], color=TEXT_SECONDARY, lw=1.5, zorder=2)
        ax.scatter([mx], [mh], s=260, facecolors=SURFACE, edgecolors=MEASURED, linewidths=3, zorder=4)
        ax.scatter([px], [ph], s=110, color=PREDICTED, edgecolors=SURFACE, linewidths=2, zorder=5)

    # Recessive grid in feet; no box.
    ax.set_xticks(range(-int(span), int(span) + 1))
    ax.set_yticks(range(0, int(ax.get_ylim()[1]) + 1))
    ax.grid(color=GRID, lw=1, zorder=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=13, length=0)
    ax.set_xlabel("Across the camera line (ft, seen from behind the cameras)", color=TEXT_SECONDARY, fontsize=14)
    ax.set_ylabel("Ball center above the floor (ft)", color=TEXT_SECONDARY, fontsize=14)

    fig.text(0.06, 0.94, title, fontsize=24, fontweight="bold", color=TEXT, ha="left")
    fig.text(0.06, 0.895,
             f"{len(kicks)} kicks ({len(kicks) - len(chips)} ground shots, {len(chips)} chips) · "
             f"median error {statistics.median(errors):.2f} ft · largest {max(errors):.2f} ft "
             f"(ball diameter 0.72 ft)", fontsize=15, color=TEXT_SECONDARY, ha="left")
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=15, markerfacecolor=SURFACE, markeredgecolor=MEASURED,
                   markeredgewidth=3, label="Measured crossing"),
        plt.Line2D([], [], marker="o", ls="", ms=10, color=PREDICTED, label="Predicted crossing"),
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.052, 0.875), ncol=2, frameon=False,
               fontsize=14, labelcolor=TEXT, columnspacing=2.5)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.82, bottom=0.1)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor=SURFACE)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("folders", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("tracking_review/goal_face.png"))
    parser.add_argument("--title", default="Where the ball crossed vs. where we predicted")
    args = parser.parse_args()
    kicks = load_kicks(args.folders)
    if not kicks:
        parser.error("No runs with a measured x and a predicted crossing")
    errors = draw(kicks, args.output, args.title)
    for k, e in zip(kicks, errors):
        print(f"{k['name']:10s} {'chip' if k['chip'] else 'ground':6s} measured {k['measured'][0]:+5.2f},"
              f"{k['measured'][1]:4.2f}  predicted {k['predicted'][0]:+5.2f},{k['predicted'][1]:4.2f}  "
              f"error {e:.2f} ft")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
