"""Worked example of the Hampel identifier on a 15-observation measurement sample.

The sample carries one value nearly fifty times the smallest, on a distribution that is not
normal with or without it. One figure is drawn, the normal quantile panels behind the normality
claim, and the sample is written out with the score each observation receives.

Changelog:
    0.2.0 - Drop the boundary, sensitivity and breakdown figures; keep the quantile panels.
    0.1.1 - Draw only the untransformed views; the log view stays in the statistics.
    0.1.0 - Add the normal quantile panel and the normality statistics behind it.
    0.0.0 - Initial release.
"""

__author__ = 'yRocket'
__version__ = "0.2.0.2026.8.18"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

from hampel_identifier import DEFAULT_THRESHOLD, hampel_test

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

# The measurement sample under test. Only seven distinct values occur among the fifteen
# observations, and the largest tie is five, against the eight at which the MAD would be 0.
SAMPLE = np.array([0.0232, 0.0232, 0.0232, 0.0220, 0.0232, 0.0232, 0.6532, 0.0403,
                   0.0293, 0.0159, 0.0134, 0.0134, 0.0134, 0.0134, 0.0134])

COLOR_POINT = TABLEAU_COLORS['tab:blue']
COLOR_FLAGGED = TABLEAU_COLORS['tab:red']
COLOR_REFERENCE = TABLEAU_COLORS['tab:gray']

# Plotting position offset for the quantile panel. The 3/8 offset is the usual choice for a
# normal quantile plot, because it makes the plotted points close to unbiased estimates of the
# expected order statistics.
BLOM_OFFSET = 0.375

# The quartiles the quantile reference line is anchored on.
QUARTILE_PROBABILITIES = (0.25, 0.75)

# Views drawn as quantile panels. The log view is reported in the statistics but not plotted,
# because the transform does not change the verdict and the panel repeats the first one.
PLOTTED_VIEWS = ('full', 'without_extreme')


def normality_views(data: np.ndarray = None) -> dict:
    """The three views of the sample the statistics cover, keyed by a short name."""
    if np.any(data <= 0.0):
        raise ValueError(f"the sample carries {int((data <= 0.0).sum())} non-positive values, so the log view "
                         f"cannot be formed; supply a positive sample or drop that view.")
    return {'full': data, 'without_extreme': data[data < data.max()], 'log10': np.log10(data)}


def normality_frame(data: np.ndarray = None) -> pd.DataFrame:
    """Skewness and a normality test for the sample, the sample without its extreme value, and its log.

    Args:
        data: the sample, shape (n,).

    Returns:
        A pd.DataFrame indexed by 'view' (one of 'full', 'without_extreme', 'log10'), with columns
        'count', 'skewness' and 'shapiro_p'.
    """
    rows = {name: {'count': values.size, 'skewness': float(stats.skew(values)),
                   'shapiro_p': float(stats.shapiro(values).pvalue)}
            for name, values in normality_views(data=data).items()}
    return pd.DataFrame(rows).T.rename_axis('view')


def normal_quantiles(count: int = None, offset: float = BLOM_OFFSET) -> np.ndarray:
    """Theoretical normal quantiles for a sample of the given size, in ascending order."""
    if count < 2:
        raise ValueError(f"a quantile plot needs at least 2 observations, got {count}.")
    return stats.norm.ppf((np.arange(1, count + 1) - offset) / (count + 1.0 - 2.0 * offset))


def quartile_line(values: np.ndarray = None) -> tuple[float, float]:
    """Intercept and slope of the reference line through the first and third quartiles.

    The line is anchored on quartiles rather than fitted by least squares, so one extreme
    observation cannot rotate it and flatten the departure the panel is drawn to show.
    """
    sample_quartiles = np.quantile(values, QUARTILE_PROBABILITIES)
    normal_pair = stats.norm.ppf(QUARTILE_PROBABILITIES)
    slope = (sample_quartiles[1] - sample_quartiles[0]) / (normal_pair[1] - normal_pair[0])
    if slope <= 0.0:
        raise ValueError(f"the interquartile range of this sample is {sample_quartiles[1] - sample_quartiles[0]}, "
                         f"so no reference line can be drawn; the sample is too heavily tied to plot.")
    return float(sample_quartiles[0] - slope * normal_pair[0]), float(slope)


def plot_normality(data: np.ndarray = None, statistics: pd.DataFrame = None,
                   drawn: tuple = PLOTTED_VIEWS, output_path: pathlib.Path = None) -> pd.DataFrame:
    """Draw a normal quantile panel for each view named in drawn.

    Returns:
        A pd.DataFrame with columns 'view', 'order', 'theoretical_quantile' and 'value', holding
        one row per plotted observation.
    """
    available = normality_views(data=data)
    missing = [name for name in drawn if name not in available]
    if missing:
        raise ValueError(f"no such view: {missing}; available views are {sorted(available)}.")
    views = {name: available[name] for name in drawn}
    titles = {'full': f"(a) All {data.size} observations", 'without_extreme': f"(b) Without {data.max():g}",
              'log10': f"(c) All {data.size} observations, log10"}
    labels = {'full': 'ordered value', 'without_extreme': 'ordered value', 'log10': 'ordered log10(value)'}
    figure, axes = plt.subplots(1, len(views), figsize=(11.5, 5.4))
    rows = []
    for axis, (name, values) in zip(axes, views.items()):
        ordered = np.sort(values)
        quantiles = normal_quantiles(count=ordered.size)
        extreme = np.isclose(ordered, ordered.max()) & (name != 'without_extreme')
        axis.scatter(quantiles[~extreme], ordered[~extreme], s=52, color=COLOR_POINT,
                     edgecolors='white', linewidths=0.8, zorder=3, label='observation')
        if extreme.any():
            axis.scatter(quantiles[extreme], ordered[extreme], s=130, color=COLOR_FLAGGED, marker='D',
                         edgecolors='white', linewidths=0.8, zorder=4, label='flagged observation')
        intercept, slope = quartile_line(values=ordered)
        grid = np.linspace(quantiles.min() - 0.4, quantiles.max() + 0.4, 50)
        axis.plot(grid, intercept + slope * grid, color=COLOR_REFERENCE, linewidth=1.5,
                  label='line through the quartiles')
        axis.plot([], [], ' ', label=f"Shapiro-Wilk p = {statistics.loc[name, 'shapiro_p']:.2e}")
        axis.plot([], [], ' ', label=f"skewness = {statistics.loc[name, 'skewness']:.3f}")
        axis.set_xlabel('theoretical normal quantile')
        axis.set_ylabel(labels[name])
        axis.set_title(titles[name])
        axis.legend(loc='upper left', frameon=False, fontsize=9)
        axis.grid(alpha=0.25, linewidth=0.6)
        rows.append(pd.DataFrame({'view': name, 'order': np.arange(1, ordered.size + 1),
                                  'theoretical_quantile': quantiles, 'value': ordered}))

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"[2] Figure written to {output_path}")
    return pd.concat(rows, ignore_index=True)


def parse_args() -> argparse.Namespace:
    """Parse and validate the command line options."""
    parser = argparse.ArgumentParser(
        description='Reproduce the worked example of the Hampel identifier document.')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help='cut-off on the absolute modified z-score (default: %(default)s)')
    parser.add_argument('--save-figure', choices=['true', 'false'], default='true',
                        help='write the figure and the samples behind it (default: %(default)s)')
    parser.add_argument('--output-folder', type=pathlib.Path, default=None,
                        help='folder for the figure (default: hampel-identifier_fig next to this script)')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    args.save_figure = args.save_figure == 'true'
    if args.output_folder is None:
        # The figure is referenced from hampel-identifier.md, whose images live in that folder.
        args.output_folder = pathlib.Path(__file__).resolve().parent / 'hampel-identifier_fig'

    if args.threshold <= 0.0:
        parser.error(f"--threshold must be positive, got {args.threshold}.")
    if args.save_figure:
        args.output_folder.mkdir(parents=True, exist_ok=True)

    return args


if __name__ == '__main__':
    options = parse_args()

    outcome = hampel_test(data=SAMPLE, threshold=options.threshold)
    print(f"[1] Flagged {outcome.count}: {np.sort(SAMPLE[outcome.positions]).tolist()}   "
          f"median = {outcome.centre:.4f}, MAD = {outcome.mad:.4f}, scale = {outcome.scale:.6f}, "
          f"sd = {SAMPLE.std(ddof=1):.6f}")

    if options.save_figure:
        normality = normality_frame(data=SAMPLE)
        quantile_points = plot_normality(data=SAMPLE, statistics=normality,
                                         output_path=options.output_folder / 'hampel_normality.png')
        # One observation per row for the sample and for the plotted points, one view per row for
        # the statistics.
        outcome.to_frame().to_csv(options.output_folder / 'hampel_sample.csv')
        normality.to_csv(options.output_folder / 'hampel_normality.csv')
        quantile_points.to_csv(options.output_folder / 'hampel_quantiles.csv', index=False)
        print(f"[3] Chart data written to {options.output_folder}")
