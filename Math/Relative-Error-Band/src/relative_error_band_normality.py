#!/usr/bin/env python3
"""Test the normality of the relative error e_ij = (x_ij - y_ij) / y_ij at each measurement location, and chart it.

The CSV holds one wafer per row with columns true_p1..true_pN (measured values y_ij) and pred_p1..pred_pN (predicted
values x_ij). For each location j the M relative errors of the M wafers are tested separately with Shapiro-Wilk,
D'Agostino-Pearson and Anderson-Darling. The script prints one row per location, writes the samples e_ij to a CSV
(one row per wafer, one column per location), and saves one figure with the normal Q-Q plot of every location.

Requires scipy and matplotlib.

Changelog:
    0.1.0  The figure is saved as SVG, or as PNG when the SVG is larger.
"""
__author__ = 'yRocket'
__version__ = "0.1.1+20261117"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import math
import pathlib
import sys

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from scipy import stats

from relative_error_band_figure import save_svg_or_png

__all__ = ['load_relative_error', 'normality_tests', 'draw']

SAMPLE_CSV_NAME: str = 'relative-error-band-fig4-relative-error.csv'
FIGURE_STEM: str = 'relative-error-band-fig4-relative-error-qq'  # suffix .svg or .png chosen by size
PERCENT: float = 100.0
ALPHA: float = 0.05
GRID_COLUMNS: int = 5
FIGSIZE: tuple = (12.0, 7.6)
REFERENCE_WIDTH: float = 12.0
FONT_SIZES: dict = {'text': 12.0, 'axis_label': 14.0, 'text_box': 12.0, 'pane_caption': 14.0}  # points
DATA_COLOR: str = mcolors.TABLEAU_COLORS['tab:blue']
REFERENCE_COLOR: str = mcolors.TABLEAU_COLORS['tab:gray']
DPI: int = 300  # matrix chart


def load_relative_error(csv_path: pathlib.Path = None) -> pd.DataFrame:
    """Return e_ij as a frame with index 'wafer' (row number) and columns 'p1'..'pN' (measurement locations)."""
    if csv_path is None:
        raise ValueError('csv_path is required.')
    frame = pd.read_csv(csv_path, encoding='utf-8-sig')
    true_cols = sorted([c for c in frame.columns if c.startswith('true_p')], key=lambda c: int(c[6:]))
    pred_cols = sorted([c for c in frame.columns if c.startswith('pred_p')], key=lambda c: int(c[6:]))
    if not true_cols or [c[6:] for c in true_cols] != [c[6:] for c in pred_cols]:
        raise ValueError(f"need matching true_p*/pred_p* columns: {list(frame.columns)}")
    y, x = frame[true_cols].to_numpy(float), frame[pred_cols].to_numpy(float)
    if len(y) < 8 or np.isnan(y).any() or np.isnan(x).any() or (y <= 0).any():
        raise ValueError(f"need eight or more wafers with positive, complete values: {y.shape=}")
    names = [c[5:] for c in true_cols]
    return pd.DataFrame((x - y) / y, columns=names).rename_axis('wafer')


def normality_tests(r: pd.DataFrame = None) -> pd.DataFrame:
    """Return one row per location (index 'location') with columns n, mean_pct, stdev_pct, skewness,
    excess_kurtosis, shapiro_W, shapiro_p, dagostino_K2, dagostino_p, anderson_A2, anderson_p."""
    rows = {}
    for name, col in r.items():
        v = col.to_numpy()
        sw, dp, ad = stats.shapiro(v), stats.normaltest(v), stats.anderson(v, method='interpolate')
        rows[name] = {'n': len(v), 'mean_pct': v.mean() * PERCENT, 'stdev_pct': v.std(ddof=1) * PERCENT,
                      'skewness': stats.skew(v), 'excess_kurtosis': stats.kurtosis(v),
                      'shapiro_W': sw.statistic, 'shapiro_p': sw.pvalue,
                      'dagostino_K2': dp.statistic, 'dagostino_p': dp.pvalue,
                      'anderson_A2': ad.statistic, 'anderson_p': ad.pvalue}
    return pd.DataFrame(rows).T.rename_axis('location')


def draw(r: pd.DataFrame = None, tests: pd.DataFrame = None, stem: pathlib.Path = None) -> pathlib.Path:
    """Save and return one figure with the normal Q-Q plot of e (in percent) at every location."""
    scale = FIGSIZE[0] / REFERENCE_WIDTH
    plt.rcParams.update({'font.size': FONT_SIZES['text'] * scale,
                         'axes.labelsize': FONT_SIZES['axis_label'] * scale})
    rows = math.ceil(r.shape[1] / GRID_COLUMNS)
    fig, axes = plt.subplots(rows, GRID_COLUMNS, figsize=FIGSIZE, sharex=True, sharey=True)
    for ax, (name, col) in zip(axes.ravel(), r.items()):
        (theory, ordered), (slope, intercept, _) = stats.probplot(col.to_numpy() * PERCENT, dist='norm')
        ax.plot(theory, slope * theory + intercept, color=REFERENCE_COLOR, linewidth=2)
        ax.scatter(theory, ordered, s=16, color=DATA_COLOR, edgecolor='white', linewidth=0.5, zorder=3)
        ax.text(0.04, 0.96, f"{name}  p = {tests.loc[name, 'shapiro_p']:.3f}", transform=ax.transAxes,
                ha='left', va='top', fontsize=FONT_SIZES['text_box'] * scale)
        ax.grid(color='#e6e6e6', linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    for ax in axes.ravel()[r.shape[1]:]:
        ax.set_visible(False)
    fig.supxlabel('Standard normal quantile', fontsize=FONT_SIZES['axis_label'] * scale)
    fig.supylabel('Ordered relative error e (%)', fontsize=FONT_SIZES['axis_label'] * scale)
    fig.tight_layout()
    path = save_svg_or_png(fig=fig, stem=stem, dpi=DPI)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{pathlib.Path(__file__).name} {__version__}\n"
                    'Normality tests and Q-Q plots of the relative error (x - y) / y at each location.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--csv', type=pathlib.Path, help='wafer CSV with true_p*/pred_p* columns')
    parser.add_argument('--output-folder', type=pathlib.Path, help='folder for the figure and the sample CSV')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    if args.csv is None or not args.csv.is_file():
        parser.print_help()
        parser.error(f"--csv must be an existing file: {args.csv}")
    if args.output_folder is None or not args.output_folder.is_dir():
        parser.error(f"--output-folder must be an existing folder: {args.output_folder}")
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    R = load_relative_error(csv_path=ARGS.csv)
    TESTS = normality_tests(r=R)
    pd.set_option('display.width', 200)
    print(TESTS.to_string(float_format='{:.4f}'.format))
    print(f"\nlocations with any p < {ALPHA}: "
          f"{list(TESTS.index[(TESTS[['shapiro_p', 'dagostino_p', 'anderson_p']] < ALPHA).any(axis=1)])}")
    R.to_csv(ARGS.output_folder / SAMPLE_CSV_NAME, float_format='%.8f')
    draw(r=R, tests=TESTS, stem=ARGS.output_folder / FIGURE_STEM)
