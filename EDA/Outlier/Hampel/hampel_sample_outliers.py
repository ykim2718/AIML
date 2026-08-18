"""Worked example of the Hampel identifier on a 15-observation measurement sample.

The sample is the one the generalized ESD document also uses, so the two methods can be compared
on identical data. Two figures are drawn: the sample under both rules with the threshold swept,
and the breakdown behaviour that separates them as one observation is pushed further out.

Changelog:
    0.0.0 - Initial release.
"""

__author__ = 'yRocket'
__version__ = "0.0.1.2026.8.17"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd

from hampel_identifier import (DEFAULT_THRESHOLD, classical_z_scores, hampel_test, max_attainable_z,
                               threshold_sweep)

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

# The measurement sample under test, shared with the generalized ESD document.
SAMPLE = np.array([0.0232, 0.0232, 0.0232, 0.0220, 0.0232, 0.0232, 0.6532, 0.0403,
                   0.0293, 0.0159, 0.0134, 0.0134, 0.0134, 0.0134, 0.0134])

COLOR_ROBUST = TABLEAU_COLORS['tab:blue']
COLOR_CLASSICAL = TABLEAU_COLORS['tab:orange']
COLOR_FLAGGED = TABLEAU_COLORS['tab:red']
COLOR_REFERENCE = TABLEAU_COLORS['tab:gray']

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
    print(f"[2] Figure written to {output_path}")


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
    print(f"[3] Figure written to {output_path}")


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
        plot_sample(result=outcome, sweep=sensitivity,
                    output_path=options.output_folder / 'hampel_sample.png')
        plot_breakdown(sweep=pushed, sizes=CEILING_SIZES, sample_size=SAMPLE.size,
                       threshold=options.threshold,
                       output_path=options.output_folder / 'hampel_breakdown.png')
        # One observation per row for the sample, one contaminant per row for the sweep.
        outcome.to_frame().to_csv(options.output_folder / 'hampel_sample.csv')
        sensitivity.to_csv(options.output_folder / 'hampel_sensitivity.csv')
        pushed.to_csv(options.output_folder / 'hampel_breakdown.csv')
        print(f"[4] Chart data written to {options.output_folder}")
