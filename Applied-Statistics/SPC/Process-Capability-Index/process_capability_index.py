#!/usr/bin/env python3
"""Compute the capability indices and draw three processes that separate what Cp, k and Cpk say.

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

__all__ = ['capability_indices', 'index_table', 'draw_cases']

FIGSIZE: tuple = (12.0, 4.2)
REFERENCE_WIDTH: float = 12.0    # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 10.0
DPI: int = 300
LOWER_SPEC: float = 90.0
UPPER_SPEC: float = 110.0
CASES: tuple = ((102.5, 2.5), (100.0, 2.5), (100.0, 3.3333333333333335))
INDEX_LEVELS: tuple = (0.67, 1.00, 1.33, 1.67, 2.00)
CURVE_COLORS: tuple = tuple(TABLEAU_COLORS.values())


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
    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    grid = np.linspace(lower_spec - 8.0, upper_spec + 8.0, 800)
    figure, axes = plt.subplots(nrows=1, ncols=len(cases), figsize=FIGSIZE, sharey=True)
    figure.subplots_adjust(bottom=0.30, wspace=0.12)
    # one y limit for every panel, so that the three densities are compared on the same scale
    top = max(stats.norm.pdf(grid, mean, sigma).max() for mean, sigma in cases) * 1.20
    for axis, (mean, sigma), color in zip(axes, cases, CURVE_COLORS):
        density = stats.norm.pdf(grid, mean, sigma)
        indices = capability_indices(mean=mean, sigma=sigma, lower_spec=lower_spec, upper_spec=upper_spec)
        axis.plot(grid, density, color=color, linewidth=1.6)
        for level, text in ((lower_spec, 'LSL'), (upper_spec, 'USL')):
            axis.axvline(level, color='black', linestyle='--', linewidth=0.9)
            axis.text(level, top * 0.94, text, ha='center', fontsize=font_size * 0.85)
        axis.axvline(mean, color=color, linestyle=':', linewidth=1.2)
        axis.set_xlabel(f"Cp = {indices['Cp']:.2f}   k = {indices['k']:.2f}   "
                        f"Cpk = {indices['Cpk']:.2f}\n{indices['ppm']:.0f} ppm outside",
                        fontsize=font_size * 0.95)
        axis.set_ylim(0.0, top)
        axis.tick_params(labelsize=font_size * 0.85)
        axis.grid(visible=True, alpha=0.25)
    axes[0].set_ylabel('Density', fontsize=font_size)
    for axis, label in zip(axes, ('(a)', '(b)', '(c)')):
        position = axis.get_position()
        figure.text(position.x0 + position.width / 2.0, 0.05, label, ha='center', va='center',
                    fontsize=font_size)
    figure.savefig(output_path, dpi=DPI)
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f'{pathlib.Path(__file__).name} {__version__}\n'
                    'Compute the capability indices and draw the three cases the document quotes.',
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
    cases.to_csv(args.output_folder / 'capability_cases.csv')
    levels.to_csv(args.output_folder / 'index_defect_rate.csv')
    figure_path = draw_cases(cases=CASES, output_path=args.output_folder / 'process_capability_index.png')
    print(f'figure  {figure_path}')
    print(f'LSL {LOWER_SPEC}  USL {UPPER_SPEC}')
    print(cases.round(4).to_string())
    print(levels.round(6).to_string())
