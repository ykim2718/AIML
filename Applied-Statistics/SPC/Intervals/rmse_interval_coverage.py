#!/usr/bin/env python3
"""Compute the probability that plus/minus a multiple of the RMSE covers a new Normal error.

The multiplier 1.96 is the two-sided 95 percent point of the standard Normal, so the interval
covers 95 percent of the errors only when the scale it is multiplied by is the true sigma. When
the scale is the RMSE of a finite record, the ratio of a new error to the RMSE follows Student's
t, and the coverage falls below 95 percent by an amount that depends on the sample count. This
script tabulates that coverage, the bias of the RMSE as an estimate of sigma, and the multiplier
that restores 95 percent, and checks the closed-form coverage against a Monte Carlo draw.
"""
__author__ = 'yRocket'
__version__ = "0.0.0+20261005"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from scipy import stats
from scipy.special import gammaln

matplotlib.use('Agg')

__all__ = ['coverage_known_sigma', 'coverage_estimated_sigma', 'rmse_bias_factor',
           'multiplier_for_coverage', 'coverage_table', 'sample_count_window',
           'simulate_coverage', 'draw_coverage']

COLORS: list = list(TABLEAU_COLORS.values())
FIGSIZE: tuple = (12.0, 5.0)
REFERENCE_WIDTH: float = 9.0  # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 9.0
PANEL_LABEL_Y: float = 0.02  # one height for every panel label, below the axis label
LEGEND_CORNERS: tuple = ('lower right', 'upper right')  # one corner per panel, away from the curve

NORMAL_MULTIPLIER: float = 1.96  # the two-sided 95 percent point of the standard Normal, rounded
TARGET_COVERAGE: float = 0.95
CLAIMED_COVERAGE: float = 0.94  # the figure the claim under study carries
TABLE_SAMPLE_COUNTS: tuple = (5, 10, 20, 25, 28, 30, 40, 50, 100, 200, 1000)
CURVE_SAMPLE_COUNTS: tuple = (3, 200)  # the span the figure draws
TRIAL_COUNT: int = 2_000_000
RANDOM_SEED: int = 11


def _degrees_of_freedom(sample_count: int = None, parameter_count: int = None) -> int:
    """The degrees of freedom left after fitting, with the inputs checked."""
    if sample_count is None:
        raise ValueError('sample_count is required.')
    if parameter_count is None:
        raise ValueError('parameter_count is required.')
    if sample_count < 1:
        raise ValueError(f"sample_count must be at least 1, got {sample_count}.")
    if parameter_count < 0:
        raise ValueError(f"parameter_count must not be negative, got {parameter_count}.")
    if parameter_count >= sample_count:
        raise ValueError(f"{parameter_count} parameters leave no degrees of freedom in "
                         f"{sample_count} samples.")
    return sample_count - parameter_count


def coverage_known_sigma(multiplier: float = None) -> float:
    """The probability that plus/minus multiplier times the true sigma covers a new error."""
    if multiplier is None:
        raise ValueError('multiplier is required.')
    if multiplier <= 0.0:
        raise ValueError(f"multiplier must be positive, got {multiplier}.")
    return 2.0 * stats.norm.cdf(multiplier) - 1.0


def coverage_estimated_sigma(multiplier: float = None, sample_count: int = None,
                             parameter_count: int = 0) -> float:
    """The probability that plus/minus multiplier times the RMSE covers a new error.

    The RMSE divides the sum of squares by sample_count, while the sum of squares carries
    sample_count - parameter_count degrees of freedom, so the two counts enter separately.
    """
    if multiplier is None:
        raise ValueError('multiplier is required.')
    if multiplier <= 0.0:
        raise ValueError(f"multiplier must be positive, got {multiplier}.")
    freedom = _degrees_of_freedom(sample_count=sample_count, parameter_count=parameter_count)
    scale = np.sqrt(freedom / sample_count)  # RMSE is the unbiased estimate shrunk by this factor
    return 2.0 * stats.t.cdf(multiplier * scale, df=freedom) - 1.0


def rmse_bias_factor(sample_count: int = None, parameter_count: int = 0) -> float:
    """The ratio of the expected RMSE to the true sigma."""
    freedom = _degrees_of_freedom(sample_count=sample_count, parameter_count=parameter_count)
    log_ratio = gammaln((freedom + 1.0) / 2.0) - gammaln(freedom / 2.0)
    return np.sqrt(2.0 / sample_count) * np.exp(log_ratio)


def multiplier_for_coverage(coverage: float = None, sample_count: int = None,
                            parameter_count: int = 0) -> float:
    """The multiplier on the RMSE that attains the requested coverage."""
    if coverage is None:
        raise ValueError('coverage is required.')
    if not 0.0 < coverage < 1.0:
        raise ValueError(f"coverage must lie between 0 and 1, got {coverage}.")
    freedom = _degrees_of_freedom(sample_count=sample_count, parameter_count=parameter_count)
    scale = np.sqrt(freedom / sample_count)
    return stats.t.ppf((1.0 + coverage) / 2.0, df=freedom) / scale


def coverage_table(sample_counts: tuple = None, multiplier: float = None,
                   coverage: float = None) -> pd.DataFrame:
    """One row per sample count, holding the bias factor, the coverage and the multiplier.

    Columns: sample_count, bias_factor, coverage, multiplier_for_coverage.
    """
    if sample_counts is None:
        raise ValueError('sample_counts is required.')
    if not sample_counts:
        raise ValueError('sample_counts holds no entry.')
    rows = [{
        'sample_count': count,
        'bias_factor': rmse_bias_factor(sample_count=count),
        'coverage': coverage_estimated_sigma(multiplier=multiplier, sample_count=count),
        'multiplier_for_coverage': multiplier_for_coverage(coverage=coverage, sample_count=count),
    } for count in sample_counts]
    return pd.DataFrame(rows).set_index('sample_count')


def sample_count_window(multiplier: float = None, coverage: float = None,
                        half_width: float = None, limit: int = None) -> tuple:
    """The first and the last sample count whose coverage lies within half_width of coverage."""
    if coverage is None:
        raise ValueError('coverage is required.')
    if half_width is None:
        raise ValueError('half_width is required.')
    if limit is None:
        raise ValueError('limit is required.')
    inside = [count for count in range(2, limit + 1)
              if abs(coverage_estimated_sigma(multiplier=multiplier, sample_count=count)
                     - coverage) <= half_width]
    if not inside:
        raise ValueError(f"no sample count up to {limit} covers {coverage} within {half_width}.")
    return inside[0], inside[-1]


def simulate_coverage(multiplier: float = None, sample_count: int = None, trial_count: int = None,
                      seed: int = None) -> float:
    """The share of Monte Carlo trials in which plus/minus multiplier times the RMSE covers."""
    if multiplier is None:
        raise ValueError('multiplier is required.')
    if trial_count is None:
        raise ValueError('trial_count is required.')
    if seed is None:
        raise ValueError('seed is required.')
    _degrees_of_freedom(sample_count=sample_count, parameter_count=0)
    generator = np.random.default_rng(seed)
    record = generator.standard_normal((trial_count, sample_count))
    rmse = np.sqrt(np.mean(record ** 2, axis=1))
    fresh = generator.standard_normal(trial_count)
    return float(np.mean(np.abs(fresh) <= multiplier * rmse))


def draw_coverage(sample_counts: tuple = None, multiplier: float = None, coverage: float = None,
                  claimed: float = None, output_path: pathlib.Path = None) -> pathlib.Path:
    """Draw the coverage curve and the multiplier curve, returning the path written."""
    if sample_counts is None:
        raise ValueError('sample_counts is required.')
    if claimed is None:
        raise ValueError('claimed is required.')
    if output_path is None:
        raise ValueError('output_path is required.')
    if len(sample_counts) != 2:
        raise ValueError(f"sample_counts must hold the first and the last count, got {sample_counts}.")
    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    counts = np.arange(sample_counts[0], sample_counts[1] + 1)
    covered = np.array([coverage_estimated_sigma(multiplier=multiplier, sample_count=int(count))
                        for count in counts])
    needed = np.array([multiplier_for_coverage(coverage=coverage, sample_count=int(count))
                       for count in counts])
    figure, axes = plt.subplots(1, 2, figsize=FIGSIZE)
    figure.subplots_adjust(bottom=0.20, wspace=0.24)
    axes[0].plot(counts, covered, color=COLORS[0], linewidth=1.8,
                 label=f'coverage of ± {multiplier} × RMSE')
    axes[0].axhline(coverage, color=COLORS[2], linewidth=1.2, linestyle='--',
                    label=f'{coverage:.2f}, the coverage with sigma known')
    axes[0].axhline(claimed, color=COLORS[3], linewidth=1.2, linestyle=':',
                    label=f'{claimed:.2f}, the claimed coverage')
    crossing = int(np.argmin(np.abs(covered - claimed)))
    axes[0].plot(counts[crossing], covered[crossing], marker='o', markersize=6,
                 color=COLORS[3], linestyle='none',
                 label=f'n = {counts[crossing]}, coverage {covered[crossing]:.4f}')
    axes[0].set_ylim(0.86, 0.96)
    axes[0].set_ylabel('Coverage', fontsize=font_size)
    axes[1].plot(counts, needed, color=COLORS[0], linewidth=1.8,
                 label=f'multiplier for {coverage:.2f} coverage')
    axes[1].axhline(multiplier, color=COLORS[2], linewidth=1.2, linestyle='--',
                    label=f'{multiplier}, the multiplier in use')
    axes[1].set_ylim(1.9, 3.2)
    axes[1].set_ylabel('Multiplier on the RMSE', fontsize=font_size)
    for axis, corner in zip(axes, LEGEND_CORNERS):
        axis.set_xscale('log')
        axis.set_xlabel('Sample count n', fontsize=font_size)
        axis.tick_params(labelsize=font_size * 0.9)
        axis.grid(alpha=0.25, linewidth=0.6)
        axis.legend(loc=corner, fontsize=font_size * 0.80, framealpha=0.85)
    for position, label in enumerate('(a) (b)'.split()):
        box = axes[position].get_position()
        figure.text(box.x0 + box.width / 2.0, PANEL_LABEL_Y, label,
                    ha='center', va='bottom', fontsize=font_size)
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f'{pathlib.Path(__file__).name} {__version__}\n'
                    'Compute the coverage of plus/minus a multiple of the RMSE for Normal errors.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--output-folder', type=pathlib.Path, required=True,
                        help='folder that receives the figure and the csv tables; created if absent')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    arguments = parser.parse_args()
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    return arguments


if __name__ == '__main__':
    args = parse_args()
    table = coverage_table(sample_counts=TABLE_SAMPLE_COUNTS, multiplier=NORMAL_MULTIPLIER,
                           coverage=TARGET_COVERAGE)
    table.to_csv(args.output_folder / 'rmse_interval_coverage.csv')
    written = draw_coverage(sample_counts=CURVE_SAMPLE_COUNTS, multiplier=NORMAL_MULTIPLIER,
                            coverage=TARGET_COVERAGE, claimed=CLAIMED_COVERAGE,
                            output_path=args.output_folder / 'rmse_interval_coverage.png')
    first, last = sample_count_window(multiplier=NORMAL_MULTIPLIER, coverage=CLAIMED_COVERAGE,
                                      half_width=0.0025, limit=10_000)
    print(table.round(4).to_string())
    print(f"coverage with sigma known: {coverage_known_sigma(multiplier=NORMAL_MULTIPLIER):.6f}")
    print(f"coverage rounds to {CLAIMED_COVERAGE:.2f} for n = {first} to {last}")
    for count in (10, 28, 100):
        closed = coverage_estimated_sigma(multiplier=NORMAL_MULTIPLIER, sample_count=count)
        drawn = simulate_coverage(multiplier=NORMAL_MULTIPLIER, sample_count=count,
                                  trial_count=TRIAL_COUNT, seed=RANDOM_SEED)
        print(f"n = {count:4d}  closed form {closed:.4f}  Monte Carlo {drawn:.4f}")
    print(f"wrote {written}")
