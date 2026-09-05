#!/usr/bin/env python3
"""Compute the tables and figures of the xbar-s control chart document.

The chart constants, the signal rates of a pure dispersion increase, the average run length a mean
chart loses when the spread grows, and the capability that a constant mean cannot protect are all
produced here, so the numbers in the text and the numbers a run prints come from the same place.
"""
__author__ = 'yRocket'
__version__ = "0.0.0.2026.9.5"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

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

__all__ = ['c4_constant', 'chart_constants', 'simulate_run', 'signal_rates', 'mean_shift_arl',
           'capability_at_multiplier', 'coupling_samples', 'draw_run', 'draw_diagnostics']

COLORS: list = list(TABLEAU_COLORS.values())
FIGSIZE_RUN: tuple = (12.0, 5.0)
FIGSIZE_DIAGNOSTIC: tuple = (13.5, 4.5)
REFERENCE_WIDTH: float = 9.0  # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 9.0
PANEL_LABEL_Y: float = 0.02  # one height for every panel label, below the axis label

SUBGROUP_SIZE: int = 5
SUBGROUP_COUNT: int = 40
SHIFT_AT: int = 21  # the first subgroup drawn from the widened process
PROCESS_MEAN: float = 100.0
PROCESS_SIGMA: float = 2.0
SIGMA_MULTIPLIERS: tuple = (1.0, 1.25, 1.5, 2.0, 3.0)
MEAN_SHIFTS: tuple = (0.5, 1.0, 1.5, 2.0)
BASELINE_CP: float = 1.33  # the capability of the process before the spread grows
LEVEL_SIGMA: float = 20.0  # the wander of the level in the coupled process
LEVEL_CV: float = 0.02  # the fixed fraction of the level the coupled spread holds
RANDOM_SEED: int = 27


def c4_constant(subgroup_size: int = None) -> float:
    """The expectation of s in units of sigma for a normal population of the given subgroup size."""
    if subgroup_size is None:
        raise ValueError('subgroup_size is required.')
    if subgroup_size < 2:
        raise ValueError(f"subgroup_size must be at least 2, got {subgroup_size}.")
    half = subgroup_size / 2.0
    return float(np.sqrt(2.0 / (subgroup_size - 1)) * np.exp(gammaln(half) - gammaln(half - 0.5)))


def chart_constants(sizes: tuple = None) -> pd.DataFrame:
    """The xbar-s chart constants at each subgroup size.

    Returns a pd.DataFrame indexed by 'n' with columns 'c4', 'B3', 'B4', 'A3' and 'cv_s', the last
    being the coefficient of variation of a single s value.
    """
    if not sizes:
        raise ValueError('sizes is empty; there is nothing to tabulate.')
    rows = []
    for size in sizes:
        constant = c4_constant(subgroup_size=size)
        spread = 3.0 / constant * np.sqrt(1.0 - constant ** 2)
        rows.append({'n': size, 'c4': constant, 'B3': max(0.0, 1.0 - spread), 'B4': 1.0 + spread,
                     'A3': 3.0 / (constant * np.sqrt(size)),
                     'cv_s': np.sqrt(1.0 - constant ** 2) / constant})
    return pd.DataFrame(rows).set_index('n')


def simulate_run(subgroup_size: int = SUBGROUP_SIZE, subgroup_count: int = SUBGROUP_COUNT,
                 shift_at: int = SHIFT_AT, sigma_multiplier: float = 2.0,
                 seed: int = RANDOM_SEED) -> pd.DataFrame:
    """One run whose standard deviation is multiplied at shift_at while the mean is held fixed.

    Returns a pd.DataFrame indexed by 'subgroup' with columns 'xbar' and 's'.
    """
    if shift_at < 2 or shift_at > subgroup_count:
        raise ValueError(f"shift_at must lie in [2, {subgroup_count}], got {shift_at}.")
    generator = np.random.default_rng(seed)
    sigma = np.where(np.arange(1, subgroup_count + 1) < shift_at,
                     PROCESS_SIGMA, PROCESS_SIGMA * sigma_multiplier)[:, None]
    observations = PROCESS_MEAN + sigma * generator.standard_normal((subgroup_count, subgroup_size))
    frame = pd.DataFrame({'subgroup': np.arange(1, subgroup_count + 1),
                          'xbar': observations.mean(axis=1),
                          's': observations.std(axis=1, ddof=1)})
    return frame.set_index('subgroup')


def signal_rates(subgroup_size: int = SUBGROUP_SIZE,
                 multipliers: tuple = SIGMA_MULTIPLIERS) -> pd.DataFrame:
    """The chance that one subgroup signals when only the spread grows, on limits held at baseline.

    Returns a pd.DataFrame indexed by 'sigma_multiplier' with columns 's_signal', 's_arl',
    'xbar_signal' and 'xbar_arl'.
    """
    if not multipliers:
        raise ValueError('multipliers is empty; there is nothing to evaluate.')
    constant = c4_constant(subgroup_size=subgroup_size)
    upper_s = constant + 3.0 * np.sqrt(1.0 - constant ** 2)  # the s chart UCL in units of sigma
    lower_s = max(0.0, constant - 3.0 * np.sqrt(1.0 - constant ** 2))
    degrees = subgroup_size - 1
    rows = []
    for multiplier in multipliers:
        above = stats.chi2.sf(degrees * (upper_s / multiplier) ** 2, degrees)
        below = stats.chi2.cdf(degrees * (lower_s / multiplier) ** 2, degrees)
        signal_s = float(above + below)
        signal_x = float(2.0 * stats.norm.cdf(-3.0 / multiplier))
        rows.append({'sigma_multiplier': multiplier, 's_signal': signal_s, 's_arl': 1.0 / signal_s,
                     'xbar_signal': signal_x, 'xbar_arl': 1.0 / signal_x})
    return pd.DataFrame(rows).set_index('sigma_multiplier')


def mean_shift_arl(subgroup_size: int = SUBGROUP_SIZE, shifts: tuple = MEAN_SHIFTS,
                   multipliers: tuple = (1.0, 2.0)) -> pd.DataFrame:
    """The average run length of the mean chart to a mean shift, once its limits carry the spread.

    The shift is stated in units of the original sigma, and the limits are recomputed at the widened
    spread, which is what happens after a dispersion increase is accepted as the new baseline.

    Returns a pd.DataFrame indexed by 'mean_shift' with one column per multiplier, named 'sigma x
    1.0', 'sigma x 2.0' and so on.
    """
    if not shifts or not multipliers:
        raise ValueError('shifts and multipliers must both be non-empty.')
    rows = []
    for shift in shifts:
        row = {'mean_shift': shift}
        for multiplier in multipliers:
            distance = shift * np.sqrt(subgroup_size) / multiplier  # the shift in limit widths
            signal = stats.norm.cdf(distance - 3.0) + stats.norm.cdf(-distance - 3.0)
            row[f'sigma x {multiplier}'] = 1.0 / float(signal)
        rows.append(row)
    return pd.DataFrame(rows).set_index('mean_shift')


def capability_at_multiplier(baseline_cp: float = BASELINE_CP,
                             multipliers: tuple = SIGMA_MULTIPLIERS) -> pd.DataFrame:
    """The capability and the defect rate of a centred process whose spread grows.

    Returns a pd.DataFrame indexed by 'sigma_multiplier' with columns 'cp' and 'ppm'.
    """
    if baseline_cp <= 0:
        raise ValueError(f"baseline_cp must be positive, got {baseline_cp}.")
    rows = []
    for multiplier in multipliers:
        capability = baseline_cp / multiplier
        rows.append({'sigma_multiplier': multiplier, 'cp': capability,
                     'ppm': float(2.0 * stats.norm.cdf(-3.0 * capability) * 1e6)})
    return pd.DataFrame(rows).set_index('sigma_multiplier')


def coupling_samples(subgroup_size: int = SUBGROUP_SIZE, subgroup_count: int = 200,
                     seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Subgroup statistics from a normal process and from a process whose spread scales with level.

    The first pair is the independence a normal population gives; the second is drawn with a
    constant coefficient of variation, which couples the two statistics.

    Returns a pd.DataFrame indexed by 'subgroup' with columns 'xbar_normal', 's_normal',
    'xbar_scaled' and 's_scaled'.
    """
    generator = np.random.default_rng(seed + 1)
    normal = PROCESS_MEAN + PROCESS_SIGMA * generator.standard_normal((subgroup_count, subgroup_size))
    # a level that wanders, with a spread that stays a fixed fraction of that level
    level = PROCESS_MEAN + LEVEL_SIGMA * generator.standard_normal((subgroup_count, 1))
    scaled = level * (1.0 + LEVEL_CV * generator.standard_normal((subgroup_count, subgroup_size)))
    frame = pd.DataFrame({'subgroup': np.arange(1, subgroup_count + 1),
                          'xbar_normal': normal.mean(axis=1),
                          's_normal': normal.std(axis=1, ddof=1),
                          'xbar_scaled': scaled.mean(axis=1),
                          's_scaled': scaled.std(axis=1, ddof=1)})
    return frame.set_index('subgroup')


def _panel_labels(figure: plt.Figure = None, axes: list = None) -> None:
    """Place (a), (b), ... below each panel, all at one height."""
    font_size = BASE_FONT_SIZE * figure.get_size_inches()[0] / REFERENCE_WIDTH
    for index, axis in enumerate(axes):
        box = axis.get_position()
        figure.text(box.x0 + box.width / 2.0, PANEL_LABEL_Y, f"({chr(ord('a') + index)})",
                    ha='center', va='bottom', fontsize=font_size + 1)


def draw_run(run: pd.DataFrame = None, output_path: pathlib.Path = None,
             subgroup_size: int = SUBGROUP_SIZE, shift_at: int = SHIFT_AT) -> pathlib.Path:
    """Draw the mean chart and the s chart of a run whose spread grows at a fixed mean."""
    if run is None or output_path is None:
        raise ValueError('run and output_path are both required.')
    baseline = run.loc[:shift_at - 1]
    constant = c4_constant(subgroup_size=subgroup_size)
    mean_s = baseline['s'].mean()
    centre = baseline['xbar'].mean()
    spread = 3.0 / constant * np.sqrt(1.0 - constant ** 2)
    limits = {'xbar': (centre - 3.0 / (constant * np.sqrt(subgroup_size)) * mean_s, centre,
                       centre + 3.0 / (constant * np.sqrt(subgroup_size)) * mean_s),
              's': (max(0.0, 1.0 - spread) * mean_s, mean_s, (1.0 + spread) * mean_s)}
    font_size = BASE_FONT_SIZE * FIGSIZE_RUN[0] / REFERENCE_WIDTH
    figure, axes = plt.subplots(1, 2, figsize=FIGSIZE_RUN)
    for axis, (column, label) in zip(axes, (('xbar', 'Subgroup mean'), ('s', 'Subgroup s'))):
        lower, middle, upper = limits[column]
        axis.plot(run.index, run[column], marker='o', markersize=4, linewidth=1.0, color=COLORS[0])
        outside = run[(run[column] > upper) | (run[column] < lower)]
        axis.scatter(outside.index, outside[column], s=90, facecolors='none',
                     edgecolors=COLORS[3], linewidths=1.6, zorder=3)
        for value, style in ((upper, '--'), (middle, '-'), (lower, '--')):
            axis.axhline(value, linestyle=style, linewidth=1.0, color=COLORS[7])
        axis.axvline(shift_at - 0.5, linestyle=':', linewidth=1.4, color=COLORS[2])
        axis.set_xlabel('Subgroup', fontsize=font_size)
        axis.set_ylabel(label, fontsize=font_size)
        axis.tick_params(labelsize=font_size - 1)
        axis.text(0.01, 0.03, f"UCL {upper:.2f}\nCL {middle:.2f}\nLCL {lower:.2f}",
                  transform=axis.transAxes, ha='left', va='bottom', fontsize=font_size - 1,
                  bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.85})
    figure.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    _panel_labels(figure=figure, axes=list(axes))
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def draw_diagnostics(arl: pd.DataFrame = None, samples: pd.DataFrame = None,
                     output_path: pathlib.Path = None) -> pathlib.Path:
    """Draw the run length cost of a widened chart and the two mean against s scatters."""
    if arl is None or samples is None or output_path is None:
        raise ValueError('arl, samples and output_path are all required.')
    font_size = BASE_FONT_SIZE * FIGSIZE_DIAGNOSTIC[0] / REFERENCE_WIDTH
    figure, axes = plt.subplots(1, 3, figsize=FIGSIZE_DIAGNOSTIC)
    for index, column in enumerate(arl.columns):
        axes[0].plot(arl.index, arl[column], marker='o', markersize=5, linewidth=1.4,
                     color=COLORS[index], label=column)
    axes[0].set_yscale('log')
    axes[0].set_xlabel('Mean shift, in original sigma', fontsize=font_size)
    axes[0].set_ylabel('Average run length', fontsize=font_size)
    axes[0].legend(fontsize=font_size - 1)
    for index, (mean_column, s_column, title) in enumerate(
            (('xbar_normal', 's_normal', 'Normal process'),
             ('xbar_scaled', 's_scaled', 'Constant CV process'))):
        axis = axes[index + 1]
        correlation, p_value = stats.pearsonr(samples[mean_column], samples[s_column])
        axis.scatter(samples[mean_column], samples[s_column], s=14, alpha=0.7, color=COLORS[index])
        axis.set_xlabel('Subgroup mean', fontsize=font_size)
        axis.set_ylabel('Subgroup s', fontsize=font_size)
        axis.text(0.03, 0.96, f"{title}\nr = {correlation:.3f}\np = {p_value:.1e}",
                  transform=axis.transAxes, ha='left', va='top', fontsize=font_size)
    for axis in axes:
        axis.tick_params(labelsize=font_size - 1)
    figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
    _panel_labels(figure=figure, axes=list(axes))
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f'{pathlib.Path(__file__).name} {__version__}\n'
                    'Compute the tables and draw the figures of the xbar-s control chart document.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--output-folder', type=pathlib.Path, required=True,
                        help='folder that receives the figures and the csv tables; created if absent')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    arguments = parser.parse_args()
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    return arguments


if __name__ == '__main__':
    args = parse_args()
    constants = chart_constants(sizes=(2, 3, 4, 5, 6, 8, 10, 15, 20, 25))
    run = simulate_run()
    rates = signal_rates()
    arl_table = mean_shift_arl()
    capability = capability_at_multiplier()
    samples = coupling_samples()
    for name, table in (('chart_constants', constants), ('run', run), ('signal_rates', rates),
                        ('mean_shift_arl', arl_table), ('capability', capability),
                        ('coupling_samples', samples)):
        table.to_csv(args.output_folder / f'{name}.csv')
    draw_run(run=run, output_path=args.output_folder / 'xbar_s_run.png')
    draw_diagnostics(arl=arl_table, samples=samples,
                     output_path=args.output_folder / 'xbar_s_diagnostics.png')
    print(constants.round(4).to_string())
    print(rates.round(4).to_string())
    print(arl_table.round(1).to_string())
    print(capability.round(3).to_string())
    baseline_s = run.loc[:SHIFT_AT - 1, 's'].mean()
    baseline_centre = run.loc[:SHIFT_AT - 1, 'xbar'].mean()
    half_width = constants.loc[SUBGROUP_SIZE, 'A3'] * baseline_s
    after = run.loc[SHIFT_AT:]
    s_signals = after.index[after['s'] > constants.loc[SUBGROUP_SIZE, 'B4'] * baseline_s]
    xbar_signals = after.index[(after['xbar'] - baseline_centre).abs() > half_width]
    print(f"baseline sbar {baseline_s:.3f}  centre {baseline_centre:.3f}")
    print(f"s signals {list(s_signals)}")
    print(f"xbar signals {list(xbar_signals)}")
    for pair in (('xbar_normal', 's_normal'), ('xbar_scaled', 's_scaled')):
        print(f"{pair[0]} against {pair[1]}: r {stats.pearsonr(samples[pair[0]], samples[pair[1]])}")
