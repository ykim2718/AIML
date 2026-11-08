"""Draw the R2 retention probability P_ret(m) against the measured wafer count m for several model R2 values.

Each curve is Eq. (9), P_ret(m) = F_chi2(m; m (1 - R2_min) / (1 - R2_p)), for one model R2 prediction R2_p at a
fixed floor R2_min. A dashed line marks the target retention probability gamma, and each curve's first crossing of
that line, the wafer measurement count m*, is marked and given in the legend.

Changelog:
    0.0.0 - Initial release.
    0.1.0 - Draw the retention probability on a log y axis.
    0.2.0 - Return to a linear y axis from 0.4 to 1.
    0.3.0 - Show m* in the legend instead of a label beside each marker.
"""

__author__ = 'yRocket'
__version__ = "0.3.1+20261108"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import pathlib
import sys
from typing import Final

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import TABLEAU_COLORS

from r2_sampling_ratio import required_measured, retention_probability

__all__ = ['retention_curves', 'draw_chart']

FIGURE_NAME: Final[str] = 'fig1-retention-probability.png'
DATA_NAME: Final[str] = 'fig1-retention-probability.csv'
FIGSIZE: Final[tuple] = (9.0, 5.5)
REFERENCE_WIDTH: Final[float] = 9.0
FONT_SIZES: Final[dict] = {'text': 12.0, 'axis_label': 14.0, 'text_box': 12.0, 'pane_caption': 14.0}  # points
DPI: Final[int] = 300
Y_RANGE: Final[tuple] = (0.4, 1.02)  # curves below 0.4 leave the plot; the CSV keeps their full values


def retention_curves(r2_models: list = None, r2_min: float = None, max_measured: int = None) -> pd.DataFrame:
    """P_ret(m) for m = 1..max_measured and every model R2.

    Returns:
        pd.DataFrame with a RangeIndex and columns r2_model, r2_min, measured, retention.
    """
    if max_measured < 1:
        raise ValueError(f"max_measured must be at least 1: {max_measured=}")
    rows = [{'r2_model': r2_model, 'r2_min': r2_min, 'measured': m,
             'retention': retention_probability(r2_model=r2_model, r2_min=r2_min, measured=m)}
            for r2_model in r2_models for m in range(1, max_measured + 1)]
    return pd.DataFrame(rows)


def draw_chart(curves: pd.DataFrame = None, confidence: float = None, figure_path: pathlib.Path = None) -> list:
    """Draw one line per model R2, the target line, and the m* marker of each curve that reaches the target; the legend gives each m*.

    Returns:
        list of (r2_model, m*) pairs; m* is None for a curve that does not reach the target within the plotted m.
    """
    if curves.empty:
        raise ValueError("no curve to draw")
    scale = FIGSIZE[0] / REFERENCE_WIDTH
    plt.rcParams.update({'font.size': FONT_SIZES['text'] * scale,
                         'axes.labelsize': FONT_SIZES['axis_label'] * scale})
    colors = list(TABLEAU_COLORS.values())
    max_measured = int(curves['measured'].max())
    r2_min = float(curves['r2_min'].iloc[0])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.axhline(confidence, color='0.45', linestyle='--', linewidth=1.2, zorder=1)
    ax.text(max_measured / 2.0, confidence - 0.008, f"target gamma = {confidence:.2f}", ha='center', va='top',
            color='0.30', fontsize=FONT_SIZES['text_box'] * scale)

    crossings = []
    for index, (r2_model, curve) in enumerate(curves.groupby('r2_model', sort=True)):
        color = colors[index]
        plan = required_measured(r2_model=r2_model, r2_min=r2_min, confidence=confidence,
                                 total_wafers=max_measured)
        crossings.append((r2_model, plan.measured))
        reached = f"m* = {plan.measured}" if plan.measured is not None else "not reached"
        ax.plot(curve['measured'], curve['retention'], color=color, linewidth=2.0, zorder=2,
                label=f"R2_p = {r2_model:.2f}, {reached}")
        if plan.measured is not None:
            ax.plot(plan.measured, plan.retention, marker='o', markersize=8, color=color,
                    markeredgecolor='white', markeredgewidth=1.5, zorder=3)

    ax.set_xlim(0, max_measured + 1)
    ax.set_ylim(*Y_RANGE)
    ax.set_xlabel('Measured wafer count m')
    ax.set_ylabel('Retention probability P_ret(m)')
    ax.grid(True, color='0.90', linewidth=0.8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(title=f"R2_min = {r2_min:.2f}", loc='center right', frameon=False)
    fig.tight_layout()
    fig.savefig(figure_path, dpi=DPI)
    plt.close(fig)
    return crossings


def parse_args() -> argparse.Namespace:
    script = pathlib.Path(__file__).name
    parser = argparse.ArgumentParser(
        prog=script,
        usage=f"{script} {__version__}\n  python {script} [options]",
        description="Draw P_ret(m) against the measured wafer count m for several model R2 values.",
        epilog=f"example: python {script} --r2-model 0.80 0.85 0.90 0.95 --r2-min 0.85 --output-folder fig",
    )
    parser.add_argument('-v', '--version', action='version', version=f"{script} {__version__}")
    parser.add_argument('--r2-model', type=float, nargs='+', required=True,
                        help="model R2 prediction of each curve, at most 10 curves")
    parser.add_argument('--r2-min', type=float, required=True, help="R2 floor to keep")
    parser.add_argument('--confidence', type=float, default=0.95, help="target retention probability (default: 0.95)")
    parser.add_argument('--max-measured', type=int, default=60, help="largest m on the x axis (default: 60)")
    parser.add_argument('--output-folder', type=pathlib.Path, required=True,
                        help=f"folder for {FIGURE_NAME} and its data {DATA_NAME}")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()

    if not 1 <= len(args.r2_model) <= len(TABLEAU_COLORS):
        parser.error(f"--r2-model takes 1 to {len(TABLEAU_COLORS)} values: {args.r2_model}")
    if any(not 0.0 < r < 1.0 for r in args.r2_model):
        parser.error(f"--r2-model values must lie in (0, 1): {args.r2_model}")
    if not args.r2_min < 1.0:
        parser.error(f"--r2-min must be below 1: {args.r2_min}")
    if not 0.0 < args.confidence < 1.0:
        parser.error(f"--confidence must lie in (0, 1): {args.confidence}")
    if args.max_measured < 1:
        parser.error(f"--max-measured must be at least 1: {args.max_measured}")
    if args.output_folder.exists() and not args.output_folder.is_dir():
        parser.error(f"--output-folder is not a folder: {args.output_folder}")
    return args


if __name__ == '__main__':
    args = parse_args()
    args.output_folder.mkdir(parents=True, exist_ok=True)
    curves = retention_curves(r2_models=args.r2_model, r2_min=args.r2_min, max_measured=args.max_measured)
    curves.to_csv(args.output_folder / DATA_NAME, index=False)
    crossings = draw_chart(curves=curves, confidence=args.confidence,
                           figure_path=args.output_folder / FIGURE_NAME)
    for r2_model, m_star in crossings:
        print(f"R2_p = {r2_model:.2f}: m* = {m_star if m_star is not None else 'not reached'}")
    unreached = sum(m_star is None for _, m_star in crossings)
    if unreached:
        print(f"{unreached}/{len(crossings)} curves do not reach the target within m <= {args.max_measured}",
              file=sys.stderr)
    print(f"wrote {args.output_folder / FIGURE_NAME} and {args.output_folder / DATA_NAME}")
