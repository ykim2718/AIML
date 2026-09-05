#!/usr/bin/env python3
"""Compute the capability indices and draw the processes the accompanying document quotes.

The figures and the numbers the document quotes are produced here, so the pictures and the text
come from one run.

Changelog:
    0.1.0 Draw the foundry priority grades as a second figure.
"""
__author__ = 'yRocket'
__version__ = "0.1.0.2026.9.4"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from scipy import stats

__all__ = ['capability_indices', 'index_table', 'priority_process', 'priority_table',
           'draw_cases', 'draw_priorities']

FIGSIZE: tuple = (12.0, 4.2)
REFERENCE_WIDTH: float = 12.0    # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 10.0
DPI: int = 300
LOWER_SPEC: float = 90.0
UPPER_SPEC: float = 110.0
CASES: tuple = ((102.5, 2.5), (100.0, 2.5), (100.0, 3.3333333333333335))
INDEX_LEVELS: tuple = (0.67, 1.00, 1.33, 1.67, 2.00)
# one row per foundry control priority: grade, control objective, Cpk minimum, k maximum
PRIORITIES: tuple = ((0, 'Product Yield', 1.67, 0.10),
                     (1, 'Device Performance', 1.50, 0.15),
                     (2, 'Process Performance', 1.33, 0.20),
                     (3, 'Monitoring', 1.00, 0.25))
CURVE_COLORS: tuple = tuple(TABLEAU_COLORS.values())


def _draw_panel(axis: plt.Axes = None, mean: float = None, sigma: float = None,
                grid: np.ndarray = None, top: float = None, font_size: float = None,
                color: str = None, lower_spec: float = LOWER_SPEC,
                upper_spec: float = UPPER_SPEC) -> pd.Series:
    """Draw one density against the specification limits and return that process's indices.

    Returns the pd.Series that capability_indices produces for the process, so the caller can label
    the panel from the same numbers the curve was drawn from.
    """
    indices = capability_indices(mean=mean, sigma=sigma, lower_spec=lower_spec, upper_spec=upper_spec)
    axis.plot(grid, stats.norm.pdf(grid, mean, sigma), color=color, linewidth=1.6)
    for level, text in ((lower_spec, 'LSL'), (upper_spec, 'USL')):
        axis.axvline(level, color='black', linestyle='--', linewidth=0.9)
        # the limit line runs through the label, so the box keeps the line off the letters
        axis.text(level, top * 0.94, text, ha='center', fontsize=font_size * 0.85,
                  bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.0})
    axis.axvline(mean, color=color, linestyle=':', linewidth=1.2)
    axis.set_ylim(0.0, top)
    axis.tick_params(labelsize=font_size * 0.85)
    axis.grid(visible=True, alpha=0.25)
    return indices


def _panel_labels(figure: plt.Figure = None, axes: np.ndarray = None, font_size: float = None) -> None:
    """Write (a), (b), ... below each panel, all at one height."""
    for index, axis in enumerate(axes):
        position = axis.get_position()
        figure.text(position.x0 + position.width / 2.0, 0.05, f'({chr(ord("a") + index)})',
                    ha='center', va='center', fontsize=font_size)


def _ppm_text(ppm: float = None) -> str:
    """The defect rate as it is written on a panel, without exponent notation."""
    return f'{ppm:,.0f}' if ppm >= 10.0 else f'{ppm:.3g}'


def capability_indices(mean: float, sigma: float, lower_spec: float = LOWER_SPEC,
                       upper_spec: float = UPPER_SPEC) -> pd.Series:
    """The capability indices of one process against one two-sided specification.

    Returns a pd.Series indexed by 'Cp', 'k', 'CPU', 'CPL', 'Cpk' and 'ppm', the last being the
    fraction outside the specification in parts per million under a normal model.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive; got {sigma}")
    if upper_spec <= lower_spec:
        raise ValueError(f"upper_spec must exceed lower_spec; got {upper_spec} and {lower_spec}")
    midpoint = (upper_spec + lower_spec) / 2.0
    half_width = (upper_spec - lower_spec) / 2.0
    upper_index = (upper_spec - mean) / (3.0 * sigma)
    lower_index = (mean - lower_spec) / (3.0 * sigma)
    outside = stats.norm.sf(upper_spec, mean, sigma) + stats.norm.cdf(lower_spec, mean, sigma)
    return pd.Series({'Cp': (upper_spec - lower_spec) / (6.0 * sigma),
                      'k': abs(midpoint - mean) / half_width,
                      'CPU': upper_index,
                      'CPL': lower_index,
                      'Cpk': min(upper_index, lower_index),
                      'ppm': outside * 1e6})


def index_table(levels: tuple = INDEX_LEVELS) -> pd.DataFrame:
    """Defect rate against index value.

    Returns a pd.DataFrame indexed by 'index_value' with columns 'ppm_centred', the two-sided rate a
    centred process with that Cp produces, and 'ppm_near_tail', the one-sided rate implied by that
    Cpk.
    """
    if any(v <= 0 for v in levels):
        raise ValueError(f"every index value must be positive; got {levels}")
    return pd.DataFrame({'ppm_centred': [2.0 * stats.norm.sf(3.0 * v) * 1e6 for v in levels],
                         'ppm_near_tail': [stats.norm.sf(3.0 * v) * 1e6 for v in levels]},
                        index=pd.Index(levels, name='index_value'))


def draw_cases(cases: tuple, output_path: pathlib.Path, lower_spec: float = LOWER_SPEC,
               upper_spec: float = UPPER_SPEC) -> pathlib.Path:
    """Draw one density per case against the specification limits and save the figure."""
    if not cases:
        raise ValueError('cases is empty; there is nothing to draw.')
    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    grid = np.linspace(lower_spec - 8.0, upper_spec + 8.0, 800)
    figure, axes = plt.subplots(nrows=1, ncols=len(cases), figsize=FIGSIZE, sharey=True)
    figure.subplots_adjust(bottom=0.30, wspace=0.12)
    # one y limit for every panel, so that the densities are compared on the same scale
    top = max(stats.norm.pdf(grid, mean, sigma).max() for mean, sigma in cases) * 1.20
    for axis, (mean, sigma), color in zip(axes, cases, CURVE_COLORS):
        indices = _draw_panel(axis=axis, mean=mean, sigma=sigma, grid=grid, top=top,
                              font_size=font_size, color=color, lower_spec=lower_spec,
                              upper_spec=upper_spec)
        axis.set_xlabel(f"Cp = {indices['Cp']:.2f}   k = {indices['k']:.2f}   "
                        f"Cpk = {indices['Cpk']:.2f}\n{_ppm_text(indices['ppm'])} ppm outside",
                        fontsize=font_size * 0.95)
    axes[0].set_ylabel('Density', fontsize=font_size)
    _panel_labels(figure=figure, axes=axes, font_size=font_size)
    figure.savefig(output_path, dpi=DPI)
    plt.close(figure)
    return output_path


def priority_process(cpk_min: float = None, k_max: float = None, lower_spec: float = LOWER_SPEC,
                     upper_spec: float = UPPER_SPEC) -> tuple:
    """The process that sits exactly on one priority grade, as a (mean, sigma) pair.

    The grade fixes Cpk and k, and Cpk = (1 - k) Cp inverts to the width the pair demands. The mean
    is placed below the midpoint by k half-widths, so the nearer limit is the lower one.
    """
    if cpk_min is None or cpk_min <= 0:
        raise ValueError(f"cpk_min must be a positive index; got {cpk_min}")
    if k_max is None or not 0.0 <= k_max < 1.0:
        raise ValueError(f"k_max must lie in [0, 1); got {k_max}")
    if upper_spec <= lower_spec:
        raise ValueError(f"upper_spec must exceed lower_spec; got {upper_spec} and {lower_spec}")
    half_width = (upper_spec - lower_spec) / 2.0
    cp = cpk_min / (1.0 - k_max)
    return (upper_spec + lower_spec) / 2.0 - k_max * half_width, (upper_spec - lower_spec) / (6.0 * cp)


def priority_table(priorities: tuple = PRIORITIES, lower_spec: float = LOWER_SPEC,
                   upper_spec: float = UPPER_SPEC) -> pd.DataFrame:
    """The indices each priority grade demands.

    Returns a pd.DataFrame indexed by 'priority' with columns 'objective', 'Cp', 'k', 'CPU', 'CPL',
    'Cpk' and 'ppm', the indices realised by the process that sits exactly on that grade.
    """
    if not priorities:
        raise ValueError('priorities is empty; a grade table needs at least one grade.')
    rows = []
    for grade, objective, cpk_min, k_max in priorities:
        mean, sigma = priority_process(cpk_min=cpk_min, k_max=k_max, lower_spec=lower_spec,
                                       upper_spec=upper_spec)
        rows.append(pd.Series({'objective': objective,
                               **capability_indices(mean=mean, sigma=sigma, lower_spec=lower_spec,
                                                    upper_spec=upper_spec)}))
    return pd.DataFrame(rows, index=pd.Index([grade for grade, _, _, _ in priorities], name='priority'))


def draw_priorities(priorities: tuple, output_path: pathlib.Path, lower_spec: float = LOWER_SPEC,
                    upper_spec: float = UPPER_SPEC) -> pathlib.Path:
    """Draw the process that sits exactly on each priority grade and save the figure."""
    if not priorities:
        raise ValueError('priorities is empty; there is nothing to draw.')
    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    grid = np.linspace(lower_spec - 8.0, upper_spec + 8.0, 800)
    cases = [priority_process(cpk_min=cpk, k_max=k, lower_spec=lower_spec, upper_spec=upper_spec)
             for _, _, cpk, k in priorities]
    figure, axes = plt.subplots(nrows=1, ncols=len(priorities), figsize=FIGSIZE, sharey=True)
    figure.subplots_adjust(bottom=0.32, top=0.86, wspace=0.12)
    # one y limit for every panel, so that the densities are compared on the same scale
    top = max(stats.norm.pdf(grid, mean, sigma).max() for mean, sigma in cases) * 1.20
    for axis, (grade, objective, _, _), (mean, sigma), color in zip(axes, priorities, cases, CURVE_COLORS):
        indices = _draw_panel(axis=axis, mean=mean, sigma=sigma, grid=grid, top=top,
                              font_size=font_size, color=color, lower_spec=lower_spec,
                              upper_spec=upper_spec)
        axis.set_title(f'Priority {grade}   {objective}', fontsize=font_size * 0.95)
        axis.set_xlabel(f"Cpk = {indices['Cpk']:.2f}   k = {indices['k']:.2f}   "
                        f"Cp = {indices['Cp']:.2f}\n{_ppm_text(indices['ppm'])} ppm outside",
                        fontsize=font_size * 0.95)
    axes[0].set_ylabel('Density', fontsize=font_size)
    _panel_labels(figure=figure, axes=axes, font_size=font_size)
    figure.savefig(output_path, dpi=DPI)
    plt.close(figure)
    return output_path



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f'{pathlib.Path(__file__).name} {__version__}\n'
                    'Compute the capability indices and draw the cases the document quotes.',
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
    cases = pd.DataFrame([capability_indices(mean=m, sigma=s) for m, s in CASES],
                         index=pd.Index([f'mean={m} sigma={s:.4f}' for m, s in CASES], name='case'))
    levels = index_table()
    grades = priority_table()
    cases.to_csv(args.output_folder / 'capability_cases.csv')
    levels.to_csv(args.output_folder / 'index_defect_rate.csv')
    grades.to_csv(args.output_folder / 'priority_grades.csv')
    figure_path = draw_cases(cases=CASES, output_path=args.output_folder / 'process_capability_index.png')
    priority_path = draw_priorities(priorities=PRIORITIES,
                                    output_path=args.output_folder / 'priority_grades.png')
    print(f'figure  {figure_path}')
    print(f'figure  {priority_path}')
    print(f'LSL {LOWER_SPEC}  USL {UPPER_SPEC}')
    print(cases.round(4).to_string())
    print(levels.round(6).to_string())
    print(grades.round(4).to_string())
