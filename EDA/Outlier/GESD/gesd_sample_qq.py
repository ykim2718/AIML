"""Normal quantile-quantile charts for the 15-observation measurement sample.

The sample is drawn three ways: as it stands, with its one extreme value removed, and on a log
scale. The first shows how far that value sits from a normal model, the second shows what the
rest of the sample looks like once it is gone, and the third asks whether a log transform
rescues the normality that the generalized ESD procedure assumes.

Changelog:
    0.0.0 - Initial release.
"""

__author__ = 'yRocket'
__version__ = "0.0.0.2026.8.17"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys
from dataclasses import dataclass

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

from gesd_sample_outliers import SAMPLE

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

COLOR_POINT = TABLEAU_COLORS['tab:blue']
COLOR_EXTREME = TABLEAU_COLORS['tab:red']
COLOR_REFERENCE = TABLEAU_COLORS['tab:gray']

# Blom plotting position. The 3/8 offset is the usual choice for a normal quantile plot because
# it makes the plotted points close to unbiased estimates of the expected order statistics.
BLOM_OFFSET = 0.375

# The normal quartiles the reference line is anchored on.
QUARTILE_PROBABILITIES = (0.25, 0.75)


@dataclass
class QqPanel:
    """One quantile-quantile panel.

    Attributes:
        key: short identifier used in the chart data file.
        title: panel title, already carrying its (a), (b) or (c) label.
        axis_label: name of the quantity on the vertical axis.
        values: the sample this panel plots, shape (m,).
        highlight: value to draw in the extreme colour, or None when the panel has none.
    """

    key: str
    title: str
    axis_label: str
    values: np.ndarray
    highlight: float = None


def normal_quantiles(count: int = None, offset: float = BLOM_OFFSET) -> np.ndarray:
    """Theoretical normal quantiles for a sample of the given size, in ascending order.

    Args:
        count: number of observations.
        offset: plotting-position offset; 0.375 is the Blom choice.

    Returns:
        An array of shape (count,) holding the quantiles.
    """
    if count < 2:
        raise ValueError(f"a quantile plot needs at least 2 observations, got {count}.")
    return stats.norm.ppf((np.arange(1, count + 1) - offset) / (count + 1.0 - 2.0 * offset))


def quartile_line(values: np.ndarray = None) -> tuple[float, float]:
    """Intercept and slope of the reference line through the first and third quartiles.

    The line is anchored on quartiles rather than fitted by least squares so that one extreme
    observation cannot rotate it and hide the very departure the chart is drawn to show.

    Returns:
        The intercept and the slope.
    """
    sample_quartiles = np.quantile(values, QUARTILE_PROBABILITIES)
    normal_quartile_pair = stats.norm.ppf(QUARTILE_PROBABILITIES)
    spread = normal_quartile_pair[1] - normal_quartile_pair[0]
    slope = (sample_quartiles[1] - sample_quartiles[0]) / spread
    if slope <= 0.0:
        raise ValueError(f"the interquartile range of this sample is {sample_quartiles[1] - sample_quartiles[0]}, "
                         f"so no reference line can be drawn; the sample is too heavily tied to plot.")
    return float(sample_quartiles[0] - slope * normal_quartile_pair[0]), float(slope)


def build_panels(data: np.ndarray = None) -> list[QqPanel]:
    """Assemble the three views of the sample the chart draws."""
    extreme = float(data.max())
    if np.any(data <= 0.0):
        raise ValueError(f"the sample carries {int((data <= 0.0).sum())} non-positive values, so the log panel "
                         f"cannot be drawn; supply a positive sample or drop the log view.")
    return [
        QqPanel(key='full', title=f"(a) All {data.size} observations", axis_label='ordered value',
                values=data, highlight=extreme),
        QqPanel(key='without_extreme', title=f"(b) Without {extreme:g}", axis_label='ordered value',
                values=data[data < extreme]),
        QqPanel(key='log10', title=f"(c) All {data.size} observations, log10", axis_label='ordered log10(value)',
                values=np.log10(data), highlight=float(np.log10(extreme))),
    ]


def shapiro_probability(values: np.ndarray = None) -> float:
    """Shapiro-Wilk p-value for the null hypothesis that the sample is normal."""
    return float(stats.shapiro(values).pvalue)


def plot_panels(panels: list[QqPanel] = None, output_path: pathlib.Path = None) -> pd.DataFrame:
    """Draw one quantile-quantile panel per view and return the points that were plotted.

    Returns:
        A pd.DataFrame with columns 'panel', 'order', 'theoretical_quantile', 'value' and
        'highlighted', holding one row per plotted observation.
    """
    figure, axes = plt.subplots(1, len(panels), figsize=(17, 5.4))
    rows = []
    for axis, panel in zip(axes, panels):
        ordered = np.sort(panel.values)
        quantiles = normal_quantiles(count=ordered.size)
        marked = np.isclose(ordered, panel.highlight) if panel.highlight is not None else np.zeros_like(ordered, bool)

        axis.scatter(quantiles[~marked], ordered[~marked], s=52, color=COLOR_POINT,
                     edgecolors='white', linewidths=0.8, zorder=3, label='observation')
        if marked.any():
            axis.scatter(quantiles[marked], ordered[marked], s=130, color=COLOR_EXTREME, marker='D',
                         edgecolors='white', linewidths=0.8, zorder=4, label='extreme value')
        intercept, slope = quartile_line(values=ordered)
        grid = np.linspace(quantiles.min() - 0.4, quantiles.max() + 0.4, 50)
        axis.plot(grid, intercept + slope * grid, color=COLOR_REFERENCE, linewidth=1.5,
                  label='line through the quartiles')
        axis.plot([], [], ' ', label=f"Shapiro-Wilk p = {shapiro_probability(values=ordered):.2e}")

        axis.set_xlabel('theoretical normal quantile')
        axis.set_ylabel(panel.axis_label)
        axis.set_title(panel.title)
        axis.legend(loc='upper left', frameon=False, fontsize=9)
        axis.grid(alpha=0.25, linewidth=0.6)

        rows.append(pd.DataFrame({'panel': panel.key, 'order': np.arange(1, ordered.size + 1),
                                  'theoretical_quantile': quantiles, 'value': ordered,
                                  'highlighted': marked}))

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"[2] Figure written to {output_path}")
    return pd.concat(rows, ignore_index=True)


def parse_args() -> argparse.Namespace:
    """Parse and validate the command line options."""
    parser = argparse.ArgumentParser(
        description='Draw normal quantile-quantile charts for the measurement sample.')
    parser.add_argument('--save-figure', choices=['true', 'false'], default='true',
                        help='write the figure and the points behind it (default: %(default)s)')
    parser.add_argument('--output-folder', type=pathlib.Path, default=None,
                        help='folder for the figure (default: generalized-esd-outlier-detection_fig '
                             'next to this script)')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    args.save_figure = args.save_figure == 'true'
    if args.output_folder is None:
        # The figure belongs with the document that discusses this sample.
        args.output_folder = (pathlib.Path(__file__).resolve().parent
                              / 'generalized-esd-outlier-detection_fig')
    if args.save_figure:
        args.output_folder.mkdir(parents=True, exist_ok=True)

    return args


if __name__ == '__main__':
    options = parse_args()

    sample_panels = build_panels(data=SAMPLE)
    for one_panel in sample_panels:
        print(f"[1] {one_panel.key:<16} n = {one_panel.values.size:2d}  "
              f"Shapiro-Wilk p = {shapiro_probability(values=one_panel.values):.3e}")

    if options.save_figure:
        stem = pathlib.Path(__file__).stem
        plotted = plot_panels(panels=sample_panels, output_path=options.output_folder / f"{stem}.png")
        plotted.to_csv(options.output_folder / f"{stem}.csv", index=False)
        print(f"[3] Chart data written to {options.output_folder / f'{stem}.csv'}")
