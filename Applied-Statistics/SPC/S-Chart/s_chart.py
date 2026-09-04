#!/usr/bin/env python3
"""Draw an s chart that catches a step increase in the process spread, and the R against s efficiency.

The figure, the chart constants and the efficiency numbers the accompanying document quotes are all
produced here, so the picture and the text come from one run.
"""
__author__ = 'yRocket'
__version__ = "0.0.0.2026.9.4"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from scipy import special

__all__ = ['c4', 'chart_constants', 'subgroup_spreads', 'range_efficiency', 'draw_chart']

FIGSIZE: tuple = (11.5, 4.4)
REFERENCE_WIDTH: float = 11.5    # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 10.0
DPI: int = 300
CONSTANT_SIZES: tuple = (2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25)
EFFICIENCY_SIZES: tuple = (2, 3, 4, 5, 6, 8, 10, 15, 25)
EFFICIENCY_TRIALS: int = 400_000
SUBGROUP_SIZE: int = 5
N_SUBGROUPS: int = 40
SHIFT_AT: int = 20               # zero-based index of the first subgroup drawn with the larger spread
SIGMA_RATIO: float = 2.0         # the process spread is multiplied by this from SHIFT_AT onwards
PROCESS_MEAN: float = 100.0
PROCESS_SIGMA: float = 2.0
CURVE_COLORS: tuple = tuple(TABLEAU_COLORS.values())


def c4(n: int) -> float:
    """Expected value of the sample standard deviation of n normal draws, in units of sigma."""
    if n < 2:
        raise ValueError(f"c4 needs a subgroup of at least 2; got {n}")
    return float(np.sqrt(2.0 / (n - 1)) * special.gamma(n / 2.0) / special.gamma((n - 1) / 2.0))


def chart_constants(sizes: tuple = CONSTANT_SIZES) -> pd.DataFrame:
    """The s chart constants.

    Returns a pd.DataFrame indexed by 'n' with columns 'c4', 'B3', 'B4' and 'A3'. B3 is clipped at
    zero because a standard deviation cannot be negative.
    """
    rows = {}
    for n in sizes:
        constant = c4(n)
        spread = 3.0 / constant * np.sqrt(1.0 - constant ** 2)
        rows[n] = {'c4': constant,
                   'B3': max(0.0, 1.0 - spread),
                   'B4': 1.0 + spread,
                   'A3': 3.0 / (constant * np.sqrt(n))}
    frame = pd.DataFrame(rows).T
    frame.index.name = 'n'
    return frame


def subgroup_spreads(n_subgroups: int = N_SUBGROUPS, subgroup_size: int = SUBGROUP_SIZE,
                     shift_at: int = SHIFT_AT, sigma_ratio: float = SIGMA_RATIO,
                     mean: float = PROCESS_MEAN, sigma: float = PROCESS_SIGMA,
                     seed: int = 11) -> pd.DataFrame:
    """One run of a process whose spread steps up part way through.

    Returns a pd.DataFrame with a RangeIndex and columns 'subgroup', 's' and 'widened', the last
    marking the subgroups drawn after the step.
    """
    if not 0 < shift_at < n_subgroups:
        raise ValueError(f"shift_at must fall inside the run; got {shift_at} of {n_subgroups}")
    if sigma_ratio <= 0:
        raise ValueError(f"sigma_ratio must be positive; got {sigma_ratio}")
    rng = np.random.default_rng(seed)
    widened = np.arange(n_subgroups) >= shift_at
    scales = sigma * np.where(widened, sigma_ratio, 1.0)
    draws = rng.normal(loc=mean, scale=scales[:, None], size=(n_subgroups, subgroup_size))
    return pd.DataFrame({'subgroup': np.arange(1, n_subgroups + 1),
                         's': draws.std(axis=1, ddof=1),
                         'widened': widened})


def range_efficiency(sizes: tuple = EFFICIENCY_SIZES, trials: int = EFFICIENCY_TRIALS,
                     seed: int = 7) -> pd.DataFrame:
    """Efficiency of the range estimator of sigma relative to the s estimator.

    Returns a pd.DataFrame indexed by 'n' with columns 'd2' and 'efficiency', the latter being the
    variance ratio of the two estimators; 1 means the two carry the same information.
    """
    rng = np.random.default_rng(seed)
    rows = {}
    for n in sizes:
        draws = rng.standard_normal((trials, n))
        d2 = float((draws.max(axis=1) - draws.min(axis=1)).mean())
        from_range = (draws.max(axis=1) - draws.min(axis=1)) / d2
        from_s = draws.std(axis=1, ddof=1) / c4(n)
        rows[n] = {'d2': d2, 'efficiency': float(from_s.var() / from_range.var())}
    frame = pd.DataFrame(rows).T
    frame.index.name = 'n'
    return frame


def draw_chart(run: pd.DataFrame, efficiency: pd.DataFrame, output_path: pathlib.Path,
               sigma: float = PROCESS_SIGMA, subgroup_size: int = SUBGROUP_SIZE) -> pathlib.Path:
    """Draw the s chart and the efficiency curve side by side and save the figure."""
    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    constant = c4(subgroup_size)
    spread = 3.0 / constant * np.sqrt(1.0 - constant ** 2)
    centre = constant * sigma
    upper = (1.0 + spread) * centre
    lower = max(0.0, (1.0 - spread) * centre)
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE)
    figure.subplots_adjust(bottom=0.26, wspace=0.28)
    outside = (run['s'] > upper) | (run['s'] < lower)
    axes[0].plot(run['subgroup'], run['s'], color=CURVE_COLORS[0], marker='o', markersize=3.5,
                 linewidth=1.0, zorder=2)
    axes[0].scatter(run.loc[outside, 'subgroup'], run.loc[outside, 's'], s=55, facecolors='none',
                    edgecolors=CURVE_COLORS[3], linewidths=1.4, zorder=3)
    for level, style, text in ((upper, '--', 'UCL'), (centre, '-', 'CL'), (lower, '--', 'LCL')):
        axes[0].axhline(level, color='black', linestyle=style, linewidth=0.9)
        axes[0].text(run['subgroup'].max() + 0.6, level, text, fontsize=font_size * 0.85, va='center')
    axes[0].axvline(SHIFT_AT + 0.5, color=CURVE_COLORS[2], linestyle=':', linewidth=1.4)
    axes[0].set_xlabel('Subgroup', fontsize=font_size)
    axes[0].set_ylabel('Subgroup standard deviation', fontsize=font_size)
    axes[0].set_xlim(0, run['subgroup'].max() + 3.5)
    axes[0].tick_params(labelsize=font_size * 0.85)
    axes[0].grid(visible=True, alpha=0.25)
    axes[1].plot(efficiency.index, efficiency['efficiency'], color=CURVE_COLORS[1], marker='o',
                 markersize=4.0, linewidth=1.6)
    axes[1].axhline(1.0, color='black', linestyle='--', linewidth=0.9)
    axes[1].set_xlabel('Subgroup size', fontsize=font_size)
    axes[1].set_ylabel('Efficiency of the range', fontsize=font_size)
    axes[1].set_ylim(0.55, 1.05)
    axes[1].tick_params(labelsize=font_size * 0.85)
    axes[1].grid(visible=True, alpha=0.25)
    for axis, label in zip(axes, ('(a)', '(b)')):
        position = axis.get_position()
        figure.text(position.x0 + position.width / 2.0, 0.06, label, ha='center', va='center',
                    fontsize=font_size)
    figure.savefig(output_path, dpi=DPI)
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f'{pathlib.Path(__file__).name} {__version__}\n'
                    'Draw the s chart and the efficiency curve, and write the constants the document quotes.',
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
    constants = chart_constants()
    run = subgroup_spreads()
    efficiency = range_efficiency()
    constants.to_csv(args.output_folder / 'chart_constants.csv')
    run.to_csv(args.output_folder / 'subgroup_spreads.csv', index=False)
    efficiency.to_csv(args.output_folder / 'range_efficiency.csv')
    figure_path = draw_chart(run=run, efficiency=efficiency,
                             output_path=args.output_folder / 's_chart.png')
    print(f'figure  {figure_path}')
    constant = c4(SUBGROUP_SIZE)
    spread = 3.0 / constant * np.sqrt(1.0 - constant ** 2)
    centre = constant * PROCESS_SIGMA
    print(f'n={SUBGROUP_SIZE}: c4={constant:.4f} CL={centre:.4f} '
          f'UCL={(1 + spread) * centre:.4f} LCL={max(0.0, 1 - spread) * centre:.4f}')
    first = run.loc[run['s'] > (1 + spread) * centre, 'subgroup']
    print(f'signals at subgroups {list(first)}; the spread doubles at {SHIFT_AT + 1}')
    print(constants.round(4).to_string())
    print(efficiency.round(4).to_string())
