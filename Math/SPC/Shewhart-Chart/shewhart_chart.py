#!/usr/bin/env python3
"""Draw an xbar chart that catches a sustained mean shift, and the ARL curve that says how fast.

The figure and the numbers the accompanying document quotes are produced here, so the picture and
the text come from one run.
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
from scipy import stats

__all__ = ['subgroup_means', 'signal_probability', 'arl_frame', 'draw_chart']

FIGSIZE: tuple = (11.5, 4.4)
REFERENCE_WIDTH: float = 11.5    # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 10.0
DPI: int = 300
LIMIT_SIGMA: float = 3.0
SUBGROUP_SIZE: int = 5
N_SUBGROUPS: int = 40
SHIFT_AT: int = 20               # zero-based index of the first shifted subgroup
SHIFT_SIZE: float = 1.5          # in units of the process standard deviation
PROCESS_MEAN: float = 100.0
PROCESS_SIGMA: float = 2.0
ARL_SHIFTS: tuple = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
ARL_SIZES: tuple = (1, 4, 5, 9)
CURVE_COLORS: tuple = tuple(TABLEAU_COLORS.values())


def subgroup_means(n_subgroups: int = N_SUBGROUPS, subgroup_size: int = SUBGROUP_SIZE,
                   shift_at: int = SHIFT_AT, shift_size: float = SHIFT_SIZE,
                   mean: float = PROCESS_MEAN, sigma: float = PROCESS_SIGMA,
                   seed: int = 3) -> pd.DataFrame:
    """One run of a process whose mean steps up part way through.

    Returns a pd.DataFrame with a RangeIndex and columns 'subgroup', 'xbar' and 'shifted', the last
    marking the subgroups drawn after the step.
    """
    if not 0 < shift_at < n_subgroups:
        raise ValueError(f"shift_at must fall inside the run; got {shift_at} of {n_subgroups}")
    if subgroup_size < 2:
        raise ValueError(f"subgroup_size must be at least 2; got {subgroup_size}")
    rng = np.random.default_rng(seed)
    shifted = np.arange(n_subgroups) >= shift_at
    centres = mean + np.where(shifted, shift_size * sigma, 0.0)
    draws = rng.normal(loc=centres[:, None], scale=sigma, size=(n_subgroups, subgroup_size))
    return pd.DataFrame({'subgroup': np.arange(1, n_subgroups + 1),
                         'xbar': draws.mean(axis=1),
                         'shifted': shifted})


def signal_probability(shift_size: float, subgroup_size: int, limit_sigma: float = LIMIT_SIGMA) -> float:
    """Probability that one subgroup falls outside the limits when the mean has moved by shift_size."""
    if subgroup_size < 1:
        raise ValueError(f"subgroup_size must be at least 1; got {subgroup_size}")
    distance = shift_size * np.sqrt(subgroup_size)
    return float(stats.norm.cdf(distance - limit_sigma) + stats.norm.cdf(-distance - limit_sigma))


def arl_frame(shifts: tuple = ARL_SHIFTS, sizes: tuple = ARL_SIZES) -> pd.DataFrame:
    """Average run length to a signal, by shift size and subgroup size.

    Returns a pd.DataFrame indexed by 'shift_sigma' with one column 'n_<size>' per subgroup size,
    holding the expected number of subgroups until the chart signals.
    """
    frame = pd.DataFrame(index=pd.Index(shifts, name='shift_sigma'))
    for size in sizes:
        frame[f'n_{size}'] = [1.0 / signal_probability(shift_size=s, subgroup_size=size) for s in shifts]
    return frame


def draw_chart(run: pd.DataFrame, arl: pd.DataFrame, output_path: pathlib.Path,
               mean: float = PROCESS_MEAN, sigma: float = PROCESS_SIGMA,
               subgroup_size: int = SUBGROUP_SIZE) -> pathlib.Path:
    """Draw the xbar chart and the ARL curves side by side and save the figure."""
    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    standard_error = sigma / np.sqrt(subgroup_size)
    upper = mean + LIMIT_SIGMA * standard_error
    lower = mean - LIMIT_SIGMA * standard_error
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE)
    figure.subplots_adjust(bottom=0.26, wspace=0.28)
    outside = (run['xbar'] > upper) | (run['xbar'] < lower)
    axes[0].plot(run['subgroup'], run['xbar'], color=CURVE_COLORS[0], marker='o', markersize=3.5,
                 linewidth=1.0, zorder=2)
    axes[0].scatter(run.loc[outside, 'subgroup'], run.loc[outside, 'xbar'], s=55, facecolors='none',
                    edgecolors=CURVE_COLORS[3], linewidths=1.4, zorder=3)
    for level, style, text in ((upper, '--', 'UCL'), (mean, '-', 'CL'), (lower, '--', 'LCL')):
        axes[0].axhline(level, color='black', linestyle=style, linewidth=0.9)
        axes[0].text(run['subgroup'].max() + 0.6, level, text, fontsize=font_size * 0.85, va='center')
    axes[0].axvline(SHIFT_AT + 0.5, color=CURVE_COLORS[2], linestyle=':', linewidth=1.4)
    axes[0].set_xlabel('Subgroup', fontsize=font_size)
    axes[0].set_ylabel('Subgroup mean', fontsize=font_size)
    axes[0].set_xlim(0, run['subgroup'].max() + 3.5)
    axes[0].tick_params(labelsize=font_size * 0.85)
    axes[0].grid(visible=True, alpha=0.25)
    for column, color in zip(arl.columns, CURVE_COLORS):
        axes[1].plot(arl.index, arl[column], color=color, marker='o', markersize=3.5, linewidth=1.4,
                     label=column.replace('n_', 'n = '))
    axes[1].set_yscale('log')
    axes[1].set_xlabel('Mean shift (process sigma)', fontsize=font_size)
    axes[1].set_ylabel('Average run length', fontsize=font_size)
    axes[1].tick_params(labelsize=font_size * 0.85)
    axes[1].legend(fontsize=font_size * 0.9, frameon=False)
    axes[1].grid(visible=True, alpha=0.25, which='both')
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
                    'Draw the xbar chart and the ARL curves the document quotes.',
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
    run = subgroup_means()
    arl = arl_frame()
    run.to_csv(args.output_folder / 'subgroup_means.csv', index=False)
    arl.to_csv(args.output_folder / 'average_run_length.csv')
    figure_path = draw_chart(run=run, arl=arl, output_path=args.output_folder / 'shewhart_chart.png')
    print(f'figure  {figure_path}')
    limit = PROCESS_SIGMA / np.sqrt(SUBGROUP_SIZE) * LIMIT_SIGMA
    print(f'UCL {PROCESS_MEAN + limit:.3f}  CL {PROCESS_MEAN:.3f}  LCL {PROCESS_MEAN - limit:.3f}')
    first = run.loc[(run['xbar'] > PROCESS_MEAN + limit) | (run['xbar'] < PROCESS_MEAN - limit), 'subgroup']
    print(f'signals at subgroups {list(first)}; shift starts at {SHIFT_AT + 1}')
    print(arl.round(2).to_string())
