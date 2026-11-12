#!/usr/bin/env python3
"""Draw the range of the minimum location R2 that eq (26) allows for each number of locations N.

For a given average of location R2, R2_avgn, eq (26) puts the minimum location R2, R2_minn, between
1 - N (1 - R2_avgn) and R2_avgn. The figure draws that range as a band over R2_avgn for a few values of N.
"""
__author__ = 'yRocket'
__version__ = "0.0.0+20261112"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import pathlib
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

__all__ = ['minn_bounds', 'draw_bounds']

FIGSIZE: tuple = (9.0, 6.4)
REFERENCE_WIDTH: float = 9.0    # the width FONT_SIZES were chosen for
FONT_SIZES: dict = {'text': 12.0, 'axis_label': 14.0, 'text_box': 12.0, 'pane_caption': 14.0}  # points
DPI: int = 300
N_VALUES: tuple = (2, 5, 13)    # 13 is the number of locations in the measured data of Appendix E
BAND_COLORS: tuple = (TABLEAU_COLORS['tab:blue'], TABLEAU_COLORS['tab:orange'], TABLEAU_COLORS['tab:green'])
Y_RANGE: tuple = (-1.0, 1.0)
GRID_POINTS: int = 401


def minn_bounds(r2_avgn: np.ndarray = None, n: int = None) -> tuple:
    """Return (lower, upper) bounds of R2_minn from eq (26) for R2_avgn values and N locations."""
    if r2_avgn is None or n is None:
        raise ValueError('r2_avgn and n are both required.')
    if n < 2:
        raise ValueError(f"N must be at least 2: {n=}")
    if np.any(r2_avgn > 1.0):
        raise ValueError(f"R2_avgn cannot exceed 1: max={r2_avgn.max()}")
    return 1.0 - n * (1.0 - r2_avgn), r2_avgn.copy()


def draw_bounds(n_values: tuple = N_VALUES, out_path: pathlib.Path = None) -> pathlib.Path:
    """Draw one band per N between the lower and upper bounds of R2_minn, and save the figure."""
    if out_path is None:
        raise ValueError('out_path is required.')
    if len(n_values) > len(BAND_COLORS) or list(n_values) != sorted(n_values):
        raise ValueError(f"need at most {len(BAND_COLORS)} N values in increasing order: {n_values=}")
    scale = FIGSIZE[0] / REFERENCE_WIDTH
    plt.rcParams.update({'font.size': FONT_SIZES['text'] * scale, 'axes.labelsize': FONT_SIZES['axis_label'] * scale})
    x = np.linspace(0.0, 1.0, GRID_POINTS)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    previous_lower = x    # the band of each N starts where the band of the previous, smaller N ends
    for n, color in zip(n_values, BAND_COLORS):
        lower, upper = minn_bounds(x, n)
        ax.fill_between(x, lower, previous_lower, facecolor=color, alpha=0.35, linewidth=0)
        ax.plot(x, lower, color=color, linewidth=2, label=f"Lower bound for $N$ = {n}: $1 - {n}(1 - R^2_{{avgn}})$")
        previous_lower = lower
    ax.plot(x, x, color='black', linewidth=2, linestyle='--', label='Upper bound: $R^2_{minn} = R^2_{avgn}$')
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(*Y_RANGE)
    ax.set_xlabel('$R^2_{avgn}$')
    ax.set_ylabel('$R^2_{minn}$')
    ax.grid(color='0.88', linewidth=0.8)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


parser = argparse.ArgumentParser(prog=pathlib.Path(__file__).name,
                                 description=f"{pathlib.Path(__file__).name} {__version__}: "
                                             f"draw the R2_minn range of eq (26) for each N.")
parser.add_argument('-v', '--version', action='version', version=f"%(prog)s {__version__}")
parser.add_argument('-o', '--out', type=pathlib.Path,
                    default=pathlib.Path(__file__).resolve().parent.parent / 'r2-aggregation-ko_fig' / 'minn-bounds.png',
                    help='output PNG path')

if __name__ == '__main__':
    args = parser.parse_args()
    saved = draw_bounds(N_VALUES, args.out)
    for n in N_VALUES:
        x_zero = 1.0 - 1.0 / n    # R2_avgn at which the lower bound reaches 0
        print(f"N={n}: lower bound is 0 at R2_avgn={x_zero:.4f}")
    print(f"saved {saved}")
    sys.exit(0)
