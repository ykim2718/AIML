"""Worked example of the Hampel identifier on a 15-observation measurement sample.

The sample carries one value nearly fifty times the smallest, on a distribution that is not
normal with or without it. Two figures are drawn: the sample under the robust and the classical
rule with the threshold swept, and the breakdown behaviour that separates the two rules as one
observation is pushed further out.

Changelog:
    0.1.1 - Draw only the untransformed views; the log view stays in the statistics.
    0.1.0 - Add the normal quantile panel and the normality statistics behind it.
    0.0.0 - Initial release.
"""

__author__ = 'yRocket'
__version__ = "0.1.1.2026.8.18"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

from hampel_identifier import (DEFAULT_THRESHOLD, classical_z_scores, hampel_test, max_attainable_z,
                               threshold_sweep)

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

# The measurement sample under test. Only seven distinct values occur among the fifteen
# observations, and the largest tie is five, against the eight at which the MAD would be 0.
SAMPLE = np.array([0.0232, 0.0232, 0.0232, 0.0220, 0.0232, 0.0232, 0.6532, 0.0403,
                   0.0293, 0.0159, 0.0134, 0.0134, 0.0134, 0.0134, 0.0134])

COLOR_ROBUST = TABLEAU_COLORS['tab:blue']
COLOR_CLASSICAL = TABLEAU_COLORS['tab:orange']
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

# Sample sizes the ceiling panel walks over, and the range of contamination the sweep applies.
CEILING_SIZES = np.arange(3, 61)
CONTAMINATION_GRID = np.geomspace(0.03, 20.0, 120)


def contamination_sweep(clean: np.ndarray = None, grid: np.ndarray = None,
                        threshold: float = DEFAULT_THRESHOLD) -> pd.DataFrame:
    """Score one contaminating observation under both rules as its value is pushed outward.

    Args:
        clean: the uncontaminated observations, shape (n-1,).
        grid: the values the contaminant takes, shape (m,).
        threshold: the cut-off both rules are compared against.

    Returns:
        A pd.DataFrame indexed by 'contaminant' (the value inserted), with columns 'modified_z'
        and 'classical_z' holding that observation's score under each rule.
    """
    modified, classical = [], []
    for value in grid:
        contaminated = np.append(clean, value)
        modified.append(float(hampel_test(data=contaminated, threshold=threshold).scores[-1]))
        classical.append(float(classical_z_scores(data=contaminated)[-1]))
    return pd.DataFrame({'modified_z': modified, 'classical_z': classical},
                        index=pd.Index(grid, name='contaminant'))


def normality_frame(data: np.ndarray = None) -> pd.DataFrame:
    """Skewness and a normality test for the sample, the sample without its extreme value, and its log.

    Args:
        data: the sample, shape (n,).

    Returns:
        A pd.DataFrame indexed by 'view' (one of 'full', 'without_extreme', 'log10'), with columns
        'count', 'skewness' and 'shapiro_p'.
    """
    rows = {}
    for name, values in normality_views(data=data).items():
        rows[name] = {'count': values.size, 'skewness': float(stats.skew(values)),
                      'shapiro_p': float(stats.shapiro(values).pvalue)}
    return pd.DataFrame(rows).T.rename_axis('view')


def normality_views(data: np.ndarray = None) -> dict:
    """The three views of the sample the quantile panel draws, keyed by a short name."""
    if np.any(data <= 0.0):
        raise ValueError(f"the sample carries {int((data <= 0.0).sum())} non-positive values, so the log view "
                         f"cannot be drawn; supply a positive sample or drop that view.")
    return {'full': data, 'without_extreme': data[data < data.max()], 'log10': np.log10(data)}


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
        axis.scatter(quantiles[~extreme], ordered[~extreme], s=52, color=COLOR_ROBUST,
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


def plot_sample(result=None, sweep: pd.DataFrame = None, output_path: pathlib.Path = None) -> None:
    """Draw the sample under both rules, the two score scales, and the threshold sensitivity."""
    figure, axes = plt.subplots(1, 3, figsize=(17, 5.2))
    data = result.values
    positions = np.arange(1, data.size + 1)
    flagged = np.zeros(data.size, dtype=bool)
    flagged[result.positions] = True
    classical = classical_z_scores(data=data)

    # Panel (a): where each rule draws its boundary, on a log axis so both fit with the data.
    axes[0].scatter(positions[~flagged], data[~flagged], s=48, color=COLOR_ROBUST,
                    edgecolors='white', linewidths=0.8, zorder=3, label='retained')
    axes[0].scatter(positions[flagged], data[flagged], s=130, color=COLOR_FLAGGED, marker='D',
                    edgecolors='white', linewidths=0.8, zorder=4, label='flagged')
    robust_upper = result.bounds()[1]
    classical_upper = float(data.mean() + result.threshold * data.std(ddof=1))
    axes[0].axhline(robust_upper, color=COLOR_ROBUST, linestyle='--', linewidth=1.5,
                    label=f"median + {result.threshold} x robust scale = {robust_upper:.4f}")
    axes[0].axhline(classical_upper, color=COLOR_CLASSICAL, linestyle=':', linewidth=2,
                    label=f"mean + {result.threshold} x sd = {classical_upper:.4f}")
    axes[0].set_yscale('log')
    axes[0].set_xlabel('observation number')
    axes[0].set_ylabel('value (log scale)')
    axes[0].set_title('(a) Where each rule draws the boundary')
    axes[0].legend(loc='center right', frameon=False, fontsize=9)
    axes[0].grid(alpha=0.25, linewidth=0.6)

    # Panel (b): the same observations scored by each rule, against the shared threshold.
    axes[1].scatter(positions, np.abs(result.scores), s=48, color=COLOR_ROBUST, marker='o',
                    edgecolors='white', linewidths=0.8, zorder=3, label='modified z (median, MAD)')
    axes[1].scatter(positions, np.abs(classical), s=48, color=COLOR_CLASSICAL, marker='s',
                    edgecolors='white', linewidths=0.8, zorder=3, label='classical z (mean, sd)')
    axes[1].axhline(result.threshold, color=COLOR_REFERENCE, linewidth=1.5,
                    label=f"threshold = {result.threshold}")
    axes[1].axhline(max_attainable_z(sample_size=data.size), color=COLOR_FLAGGED, linestyle='--', linewidth=1.5,
                    label=f"ceiling of classical z = {max_attainable_z(sample_size=data.size):.4f}")
    axes[1].set_yscale('log')
    # Headroom above the largest score, so the legend does not sit on top of the flagged point.
    axes[1].set_ylim(top=float(np.abs(result.scores).max()) * 12.0)
    axes[1].set_xlabel('observation number')
    axes[1].set_ylabel('absolute score (log scale)')
    axes[1].set_title('(b) The same observations under each score')
    axes[1].legend(loc='upper left', frameon=False, fontsize=9)
    axes[1].grid(alpha=0.25, linewidth=0.6)

    # Panel (c): how many each rule reports as the threshold moves.
    axes[2].step(sweep.index, sweep['hampel'], where='mid', color=COLOR_ROBUST, linewidth=3.2,
                 marker='o', markersize=8, label='modified z')
    axes[2].step(sweep.index, sweep['classical'], where='mid', color=COLOR_CLASSICAL, linewidth=2,
                 marker='s', markersize=6, label='classical z')
    axes[2].axvline(DEFAULT_THRESHOLD, color=COLOR_REFERENCE, linewidth=1.5,
                    label=f"conventional threshold = {DEFAULT_THRESHOLD}")
    axes[2].set_xlabel('threshold')
    axes[2].set_ylabel('observations flagged')
    axes[2].set_title('(c) Sensitivity to the threshold')
    axes[2].legend(loc='upper right', frameon=False, fontsize=9)
    axes[2].grid(alpha=0.25, linewidth=0.6)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"[3] Figure written to {output_path}")


def plot_breakdown(sweep: pd.DataFrame = None, sizes: np.ndarray = None, sample_size: int = None,
                   threshold: float = DEFAULT_THRESHOLD, output_path: pathlib.Path = None) -> None:
    """Draw the saturation of the classical score and the ceiling that causes it."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    # Panel (a): one observation is pushed outward and each rule is asked how extreme it is.
    axes[0].plot(sweep.index, sweep['modified_z'], color=COLOR_ROBUST, linewidth=2.4,
                 label='modified z (median, MAD)')
    axes[0].plot(sweep.index, sweep['classical_z'], color=COLOR_CLASSICAL, linewidth=2.4,
                 label='classical z (mean, sd)')
    ceiling = max_attainable_z(sample_size=sample_size)
    axes[0].axhline(ceiling, color=COLOR_FLAGGED, linestyle='--', linewidth=1.5,
                    label=f"ceiling (n-1)/sqrt(n) = {ceiling:.3f}")
    axes[0].axhline(threshold, color=COLOR_REFERENCE, linewidth=1.5, label=f"threshold = {threshold}")
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    axes[0].set_xlabel('value of the contaminating observation (log scale)')
    axes[0].set_ylabel('its score (log scale)')
    axes[0].set_title(f"(a) One observation pushed outward, n = {sample_size}")
    axes[0].legend(loc='upper left', frameon=False, fontsize=9)
    axes[0].grid(alpha=0.25, linewidth=0.6)

    # Panel (b): the ceiling is a function of n alone, and below a certain n it defeats the rule.
    ceilings = np.array([max_attainable_z(sample_size=int(n)) for n in sizes])
    axes[1].plot(sizes, ceilings, color=COLOR_CLASSICAL, linewidth=2.4, label='(n-1)/sqrt(n)')
    axes[1].axhline(threshold, color=COLOR_REFERENCE, linewidth=1.5, label=f"threshold = {threshold}")
    unusable = sizes[ceilings < threshold]
    axes[1].fill_between(sizes, 0, ceilings.max() * 1.05, where=(ceilings < threshold),
                         color=COLOR_FLAGGED, alpha=0.12,
                         label=f"n <= {int(unusable.max())}: the rule can never flag anything")
    axes[1].scatter([sample_size], [max_attainable_z(sample_size=sample_size)], s=110, zorder=4,
                    color=COLOR_FLAGGED, marker='D', edgecolors='white', linewidths=0.8,
                    label=f"this sample, n = {sample_size}")
    axes[1].set_ylim(0, ceilings.max() * 1.05)
    axes[1].set_xlabel('sample size n')
    axes[1].set_ylabel('largest attainable classical z')
    axes[1].set_title('(b) The ceiling depends on the sample size alone')
    axes[1].legend(loc='lower right', frameon=False, fontsize=9)
    axes[1].grid(alpha=0.25, linewidth=0.6)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"[4] Figure written to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse and validate the command line options."""
    parser = argparse.ArgumentParser(
        description='Reproduce the worked example of the Hampel identifier document.')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help='cut-off on the absolute modified z-score (default: %(default)s)')
    parser.add_argument('--sweep-steps', type=int, default=41,
                        help='how many thresholds the sensitivity panel walks over (default: %(default)s)')
    parser.add_argument('--save-figure', choices=['true', 'false'], default='true',
                        help='write the figures and the samples behind them (default: %(default)s)')
    parser.add_argument('--output-folder', type=pathlib.Path, default=None,
                        help='folder for the figures (default: hampel-identifier_fig next to this script)')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    args.save_figure = args.save_figure == 'true'
    if args.output_folder is None:
        # The figures are referenced from hampel-identifier.md, whose images live in that folder.
        args.output_folder = pathlib.Path(__file__).resolve().parent / 'hampel-identifier_fig'

    if args.threshold <= 0.0:
        parser.error(f"--threshold must be positive, got {args.threshold}.")
    if args.sweep_steps < 2:
        parser.error(f"--sweep-steps must be at least 2, got {args.sweep_steps}.")
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
        grid = np.linspace(0.5, 10.5, options.sweep_steps)
        sensitivity = threshold_sweep(data=SAMPLE, thresholds=grid)
        pushed = contamination_sweep(clean=np.delete(SAMPLE, outcome.positions),
                                     grid=CONTAMINATION_GRID, threshold=options.threshold)
        normality = normality_frame(data=SAMPLE)
        quantile_points = plot_normality(data=SAMPLE, statistics=normality,
                                         output_path=options.output_folder / 'hampel_normality.png')
        plot_sample(result=outcome, sweep=sensitivity,
                    output_path=options.output_folder / 'hampel_sample.png')
        plot_breakdown(sweep=pushed, sizes=CEILING_SIZES, sample_size=SAMPLE.size,
                       threshold=options.threshold,
                       output_path=options.output_folder / 'hampel_breakdown.png')
        # One observation per row for the sample, one contaminant per row for the sweep.
        outcome.to_frame().to_csv(options.output_folder / 'hampel_sample.csv')
        normality.to_csv(options.output_folder / 'hampel_normality.csv')
        quantile_points.to_csv(options.output_folder / 'hampel_quantiles.csv', index=False)
        sensitivity.to_csv(options.output_folder / 'hampel_sensitivity.csv')
        pushed.to_csv(options.output_folder / 'hampel_breakdown.csv')
        print(f"[5] Chart data written to {options.output_folder}")
