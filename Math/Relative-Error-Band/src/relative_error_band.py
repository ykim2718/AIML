#!/usr/bin/env python3
"""Compute the +/- p95 relative error band around the 1:1 line of predicted vs. measured values.

The CSV holds one wafer per row with columns true_p1..true_pN (measured values y) and pred_p1..pred_pN (predicted
values x). All T = M * N pairs are pooled and the relative error e = (x - y) / y is formed. The script prints

    with raw data      p95 = 0.95 quantile of |e|, and the conformal rank ceil(0.95 (T + 1)) of the sorted |r|
    without raw data   normal, mean 0         p95 = z * RMSRE                 (z: 0.975 standard normal quantile)
                       normal, mean != 0      p95 solves Phi((p - mu)/sd) - Phi((-p - mu)/sd) = 0.95
                       distribution free      p95 <= RMSRE / sqrt(0.05)       (Markov inequality on e^2)

the share of pairs inside each band, the Shapiro-Wilk test of the pooled e, and RMSE, R2_all and RMSE / y_bar.
It also checks whether the error grows in proportion to the measured value: it fits ln|y - x| = ln a + b ln y by
least squares and tests b = 0 (constant error) and b = 1 (proportional error), and it prints the RMS error and the
RMSRE in five bins of equal count along y.
With --output-folder it also saves one figure with a Pred vs. True panel per method, showing the pairs, the 1:1 line
and the band x = (1 +/- p95) y, and writes the band of each method to a CSV.

Requires scipy and matplotlib.

Changelog:
    0.1.0  Added the band chart per method (--output-folder).
    0.2.0  The chart is saved as SVG, or as PNG when the SVG is larger.
    0.3.0  Added the proportionality check of the error (error_scaling).
"""
__author__ = 'yRocket'
__version__ = "0.3.0+20261117"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import math
import pathlib
import sys

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import lines as mlines
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy import optimize, stats

from relative_error_band_figure import save_svg_or_png

__all__ = ['METHODS', 'load_pairs', 'relative_error_band', 'error_scaling', 'band_table', 'draw']

COVERAGE: float = 0.95
PERCENT: float = 100.0
SCALING_BINS: int = 5  # bins of equal count along y for the proportionality check
# key of relative_error_band() -> (number, method name, equation number, section) as in Table 3 of the document
METHODS: dict = {'empirical': (1, 'Empirical Quantile', 8, '4.2'),
                 'conformal': (2, 'Conformal Rank', 9, '4.2'),
                 'normal_zero_mean': (3, 'Normal with Zero Mean', 10, '4.3'),
                 'normal': (4, 'Normal with Nonzero Mean', 11, '4.4'),
                 'distribution_free': (5, 'Distribution-free Upper Bound', 12, '4.5')}
SAMPLE_CSV_NAME: str = 'relative-error-band-fig2-band-by-method.csv'
FIGURE_STEM: str = 'relative-error-band-fig2-band-by-method'  # suffix .svg or .png chosen by size
GRID: tuple = (2, 3)  # rows, columns
FIGSIZE: tuple = (12.0, 9.0)
REFERENCE_WIDTH: float = 12.0
FONT_SIZES: dict = {'text': 12.0, 'axis_label': 14.0, 'text_box': 12.0, 'pane_caption': 14.0}  # points
PANE_CAPTION_Y: float = -0.25  # axes fraction, below the x axis label
INSIDE_COLOR: str = mcolors.TABLEAU_COLORS['tab:blue']
OUTSIDE_COLOR: str = mcolors.TABLEAU_COLORS['tab:red']
BAND_COLOR: str = mcolors.TABLEAU_COLORS['tab:green']
REFERENCE_COLOR: str = mcolors.TABLEAU_COLORS['tab:gray']
BAND_ALPHA: float = 0.12
DPI: int = 300  # matrix chart


def load_pairs(csv_path: pathlib.Path = None) -> tuple:
    """Return the pooled (x, y) arrays of predicted and measured values, each of length T = M * N."""
    if csv_path is None:
        raise ValueError('csv_path is required.')
    frame = pd.read_csv(csv_path, encoding='utf-8-sig')
    true_cols = sorted([c for c in frame.columns if c.startswith('true_p')], key=lambda c: int(c[6:]))
    pred_cols = sorted([c for c in frame.columns if c.startswith('pred_p')], key=lambda c: int(c[6:]))
    if not true_cols or [c[6:] for c in true_cols] != [c[6:] for c in pred_cols]:
        raise ValueError(f"need matching true_p*/pred_p* columns: {list(frame.columns)}")
    y, x = frame[true_cols].to_numpy(float).ravel(), frame[pred_cols].to_numpy(float).ravel()
    if len(y) < 20 or np.isnan(y).any() or np.isnan(x).any() or (y <= 0).any():
        raise ValueError(f"need twenty or more pairs with positive, complete measured values: {len(y)=}")
    return x, y


def relative_error_band(x: np.ndarray = None, y: np.ndarray = None, coverage: float = COVERAGE) -> dict:
    """Return the relative error statistics, the four p95 estimates and the share of pairs inside each band."""
    if not 0 < coverage < 1:
        raise ValueError(f"coverage must be in (0, 1): {coverage}")
    r = (x - y) / y
    a = np.abs(r)
    t = len(r)
    mu, sd = r.mean(), r.std()
    rmsre = math.sqrt((r ** 2).mean())
    rank = math.ceil(coverage * (t + 1))
    z = stats.norm.ppf((1 + coverage) / 2)
    bands = {'p_empirical': np.quantile(a, coverage),
             'p_conformal': np.sort(a)[rank - 1] if rank <= t else np.inf,
             'p_normal_zero_mean': z * rmsre,
             'p_normal': optimize.brentq(
                 lambda p: stats.norm.cdf((p - mu) / sd) - stats.norm.cdf((-p - mu) / sd) - coverage,
                 0.0, abs(mu) + 10 * sd),
             'p_distribution_free': rmsre / math.sqrt(1 - coverage)}
    e = y - x
    rmse = math.sqrt((e ** 2).mean())
    out = {'T': t, 'mean_r': mu, 'stdev_r': sd, 'RMSRE': rmsre, 'conformal_rank': rank, 'z': z,
           'shapiro_W': stats.shapiro(r).statistic, 'shapiro_p': stats.shapiro(r).pvalue,
           'RMSE': rmse, 'y_bar': y.mean(), 'RMSE/y_bar': rmse / y.mean(), 'cv_y': y.std() / y.mean(),
           'R2_all': 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), 'y_min': y.min(), 'y_max': y.max()}
    for name, p in bands.items():
        out[name] = p
        out[name.replace('p_', 'inside_')] = (a <= p).mean()
    return out


def error_scaling(x: np.ndarray = None, y: np.ndarray = None, bins: int = SCALING_BINS,
                  coverage: float = COVERAGE) -> tuple:
    """Return the fit of ln|y - x| = ln a + b ln y and the RMS error per bin of y.

    The dict holds b, its standard error, its two-sided coverage interval and the p-values of b = 0 and b = 1.
    The frame has index 'bin' (1..bins, by increasing y) and columns n, y_min, y_max, rms_error, rmsre_pct.
    """
    d = np.abs(y - x)
    if (d <= 0).any():
        raise ValueError(f"ln|y - x| needs nonzero errors: {(d <= 0).sum()} pairs have y == x")
    fit = stats.linregress(np.log(y), np.log(d))
    dof = len(d) - 2
    half = stats.t.ppf((1 + coverage) / 2, dof) * fit.stderr
    slope = {'b': fit.slope, 'b_stderr': fit.stderr, 'b_low': fit.slope - half, 'b_high': fit.slope + half,
             'p_b0': 2 * stats.t.sf(abs(fit.slope / fit.stderr), dof),
             'p_b1': 2 * stats.t.sf(abs((fit.slope - 1) / fit.stderr), dof)}
    label = pd.qcut(y, bins, labels=False)
    rows = {}
    for k in range(bins):
        m = label == k
        rows[k + 1] = {'n': int(m.sum()), 'y_min': y[m].min(), 'y_max': y[m].max(),
                       'rms_error': math.sqrt((d[m] ** 2).mean()),
                       'rmsre_pct': math.sqrt(((d[m] / y[m]) ** 2).mean()) * PERCENT}
    return slope, pd.DataFrame(rows).T.rename_axis('bin')


def band_table(x: np.ndarray = None, y: np.ndarray = None, result: dict = None) -> pd.DataFrame:
    """Return one row per method (index 'number') with columns method, equation, section, p95_pct, pairs_inside,
    share_inside_pct."""
    a = np.abs((x - y) / y)
    rows = {}
    for key, (number, name, equation, section) in METHODS.items():
        p = result[f"p_{key}"]
        rows[number] = {'method': name, 'equation': equation, 'section': section, 'p95_pct': p * PERCENT,
                        'pairs_inside': int((a <= p).sum()), 'share_inside_pct': (a <= p).mean() * PERCENT}
    return pd.DataFrame(rows).T.rename_axis('number')


def draw(x: np.ndarray = None, y: np.ndarray = None, table: pd.DataFrame = None,
         stem: pathlib.Path = None) -> pathlib.Path:
    """Save and return one figure with a Pred vs. True panel per method: pairs inside and outside the band, the 1:1
    line and the band x = (1 +/- p95) y."""
    if len(table) >= GRID[0] * GRID[1]:
        raise ValueError(f"the grid {GRID} needs one free panel for the legend: {len(table)=}")
    scale = FIGSIZE[0] / REFERENCE_WIDTH
    plt.rcParams.update({'font.size': FONT_SIZES['text'] * scale,
                         'axes.labelsize': FONT_SIZES['axis_label'] * scale})
    fig, axes = plt.subplots(*GRID, figsize=FIGSIZE, sharex=True, sharey=True)
    a = np.abs((x - y) / y)
    line_y = np.array([y.min(), y.max()])
    for ax, (number, row) in zip(axes.ravel(), table.iterrows()):
        p = row['p95_pct'] / PERCENT
        inside = a <= p
        ax.fill_between(line_y, (1 - p) * line_y, (1 + p) * line_y, color=BAND_COLOR, alpha=BAND_ALPHA, linewidth=0)
        for sign in (-1, 1):
            ax.plot(line_y, (1 + sign * p) * line_y, color=BAND_COLOR, linewidth=1.6)
        ax.plot(line_y, line_y, color=REFERENCE_COLOR, linewidth=1.4, linestyle='--')
        ax.scatter(y[inside], x[inside], s=10, color=INSIDE_COLOR, edgecolor='none', alpha=0.7, zorder=3)
        ax.scatter(y[~inside], x[~inside], s=18, color=OUTSIDE_COLOR, edgecolor='none', zorder=4)
        ax.text(0.04, 0.96, f"p95 = {row['p95_pct']:.4f} %\ninside {row['pairs_inside']} / {len(a)}",
                transform=ax.transAxes, ha='left', va='top', fontsize=FONT_SIZES['text_box'] * scale)
        ax.set_xlabel('True y (nm)')
        ax.xaxis.set_tick_params(labelbottom=True)
        ax.grid(color='#e6e6e6', linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.text(0.5, PANE_CAPTION_Y, f"({chr(ord('a') + number - 1)}) {row['method']}",
                transform=ax.transAxes, ha='center', va='top', fontsize=FONT_SIZES['pane_caption'] * scale)
    for ax in axes[:, 0]:
        ax.set_ylabel('Pred x (nm)')
    legend_ax = axes.ravel()[len(table)]
    legend_ax.axis('off')
    legend_ax.legend(handles=[
        mlines.Line2D([], [], marker='o', linestyle='none', color=INSIDE_COLOR, label='pair inside the band'),
        mlines.Line2D([], [], marker='o', linestyle='none', color=OUTSIDE_COLOR, label='pair outside the band'),
        mlines.Line2D([], [], color=REFERENCE_COLOR, linestyle='--', label='1:1 line'),
        patches.Patch(facecolor=mcolors.to_rgba(BAND_COLOR, BAND_ALPHA), edgecolor=BAND_COLOR,
                      label=r'band $x = (1 \pm p_{95})\, y$')], loc='center', frameon=False)
    for ax in axes.ravel()[len(table) + 1:]:
        ax.set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.14, wspace=0.12, hspace=0.42)
    path = save_svg_or_png(fig=fig, stem=stem, dpi=DPI)
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{pathlib.Path(__file__).name} {__version__}\n"
                    'The +/- p95 relative error band around the 1:1 line of predicted vs. measured values.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--csv', type=pathlib.Path, help='wafer CSV with true_p*/pred_p* columns')
    parser.add_argument('--output-folder', type=pathlib.Path, help='folder for the band chart and its CSV (optional)')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    if args.csv is None or not args.csv.is_file():
        parser.print_help()
        parser.error(f"--csv must be an existing file: {args.csv}")
    if args.output_folder is not None and not args.output_folder.is_dir():
        parser.error(f"--output-folder must be an existing folder: {args.output_folder}")
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    X, Y = load_pairs(csv_path=ARGS.csv)
    RESULT = relative_error_band(x=X, y=Y)
    for KEY, VALUE in RESULT.items():
        print(f"{KEY:24s} {VALUE:.6f}" if isinstance(VALUE, float) else f"{KEY:24s} {VALUE}")
    SLOPE, BINS = error_scaling(x=X, y=Y)
    for KEY, VALUE in SLOPE.items():
        print(f"{KEY:24s} {VALUE:.6g}")
    print(BINS.to_string(float_format='{:.4f}'.format))
    if ARGS.output_folder is not None:
        TABLE = band_table(x=X, y=Y, result=RESULT)
        print(TABLE.to_string(float_format='{:.4f}'.format))
        TABLE.to_csv(ARGS.output_folder / SAMPLE_CSV_NAME, float_format='%.6f')
        draw(x=X, y=Y, table=TABLE, stem=ARGS.output_folder / FIGURE_STEM)
