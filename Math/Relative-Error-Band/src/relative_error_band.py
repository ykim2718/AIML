#!/usr/bin/env python3
"""Compute the +/- p95 relative error band around the 1:1 line of predicted vs. measured values.

The CSV holds one wafer per row with columns true_p1..true_pN (measured values y) and pred_p1..pred_pN (predicted
values x). All T = M * N pairs are pooled and the relative error e = (x - y) / y is formed. The script prints

    with raw data      p95 = 0.95 quantile of |e|, and the conformal rank ceil(0.95 (T + 1)) of the sorted |r|
    without raw data   normal, mean 0         p95 = z * RMSRE                 (z: 0.975 standard normal quantile)
                       normal, mean != 0      p95 solves Phi((p - mu)/sd) - Phi((-p - mu)/sd) = 0.95
                       distribution free      p95 <= RMSRE / sqrt(0.05)       (Markov inequality on e^2)

the share of pairs inside each band, the Shapiro-Wilk test of the pooled e, and RMSE, R2_all and RMSE / y_bar.
It also checks whether the error grows in proportion to the measured value: it runs a PCA on the pairs
(y, |x - y|) and prints the first and second principal vectors and the angle of the first from the y axis, and it
prints the RMSE and the RMSRE in five bins of equal count along y. For an error that is not proportional, the same
five estimates are computed on the error x - y with the RMSE in place of the RMSRE, giving the band x = y +/- q95 in
the unit of y.
With --output-folder the error x - y and the relative error are also drawn against y with the RMS of each bin.
With --output-folder it also saves one figure with a Pred vs. True panel per method, showing the pairs, the 1:1 line
and the band x = (1 +/- p95) y, and writes the band of each method to a CSV.

Requires scipy and matplotlib.

Changelog:
    0.1.0  Added the band chart per method (--output-folder).
    0.2.0  The chart is saved as SVG, or as PNG when the SVG is larger.
    0.3.0  Added the proportionality check of the error (error_scaling).
    0.4.0  Added the absolute error band (absolute_error_band) and the chart of the RMSE by bin.
    0.5.0  The chart of the RMSE by bin is replaced by the scatter of the error against y (draw_error); the band
           table and chart also cover the absolute band (band='absolute').
    0.6.0  draw_error draws two panels, the error and the relative error against y, with the RMS of each bin.
    0.7.0  The proportionality check uses the PCA of (y, |x - y|) instead of the fit of ln|y - x| on ln y.
"""
__author__ = 'yRocket'
__version__ = "0.7.0+20261117"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import math
import pathlib
import sys
from typing import Literal, TypeAlias, get_args

import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import lines as mlines
from matplotlib import patches
from matplotlib import pyplot as plt
from scipy import optimize, stats

from relative_error_band_figure import save_svg_or_png

__all__ = ['METHODS', 'load_pairs', 'band_estimates', 'relative_error_band', 'absolute_error_band', 'error_scaling',
           'band_table', 'draw', 'draw_error']

COVERAGE: float = 0.95
PERCENT: float = 100.0
SCALING_BINS: int = 5  # bins of equal count along y for the proportionality check
# key of band_estimates() -> (number, method name, equation number, section) as in Table 4 of the document
METHODS: dict = {'empirical': (1, 'Empirical Quantile', 8, '4.2'),
                 'conformal': (2, 'Conformal Rank', 9, '4.2'),
                 'normal_zero_mean': (3, 'Normal with Zero Mean', 10, '4.3'),
                 'normal': (4, 'Normal with Nonzero Mean', 11, '4.4'),
                 'distribution_free': (5, 'Distribution-free Upper Bound', 12, '4.5')}
SAMPLE_CSV_NAME: str = 'relative-error-band-fig4-band-by-method.csv'
FIGURE_STEM: str = 'relative-error-band-fig4-band-by-method'  # suffix .svg or .png chosen by size
ABSOLUTE_CSV_NAME: str = 'relative-error-band-fig3-absolute-band-by-method.csv'
ABSOLUTE_FIGURE_STEM: str = 'relative-error-band-fig3-absolute-band-by-method'  # suffix .svg or .png chosen by size
Band: TypeAlias = Literal['relative', 'absolute']  # x = (1 +/- p95) y, or x = y +/- q95
BANDS: tuple = get_args(Band)
# band -> (key prefix in the result, table column, scale of the column, text-box format, legend label)
BAND_SPEC: dict = {'relative': ('p_', 'p95_pct', PERCENT, 'p95 = {:.4f} %', r'band $x = (1 \pm p_{95})\, y$'),
                   'absolute': ('q_', 'q95_nm', 1.0, 'q95 = {:.4f} nm', r'band $x = y \pm q_{95}$')}
ERROR_CSV_NAME: str = 'relative-error-band-fig2-error-vs-measured.csv'
ERROR_FIGURE_STEM: str = 'relative-error-band-fig2-error-vs-measured'  # suffix .svg or .png chosen by size
ERROR_FIGSIZE: tuple = (12.0, 10.0)
ERROR_REFERENCE_WIDTH: float = 12.0
ERROR_CAPTION_Y: float = -0.22  # axes fraction, below the x axis label
RIGHT_ANGLE: float = 90.0  # degrees
PCA_ARROW_SIGMAS: float = 2.0  # arrow length in standard deviations along each principal vector
PCA_COLORS: tuple = (mcolors.TABLEAU_COLORS['tab:red'], mcolors.TABLEAU_COLORS['tab:green'])
MODEL_COLORS: dict = {'constant': mcolors.TABLEAU_COLORS['tab:gray'],
                     'proportional': mcolors.TABLEAU_COLORS['tab:orange']}
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


def band_estimates(r: np.ndarray = None, coverage: float = COVERAGE) -> dict:
    """Return the five band half-widths of the residuals r (keys p_<method>) and the share inside (inside_<method>).

    r is the relative error (x - y) / y for the relative band, or the error x - y for the absolute band; the root
    mean square of r takes the place of the RMSRE.
    """
    if not 0 < coverage < 1:
        raise ValueError(f"coverage must be in (0, 1): {coverage}")
    a = np.abs(r)
    t = len(r)
    mu, sd = r.mean(), r.std()
    rms = math.sqrt((r ** 2).mean())
    rank = math.ceil(coverage * (t + 1))
    z = stats.norm.ppf((1 + coverage) / 2)
    bands = {'p_empirical': np.quantile(a, coverage),
             'p_conformal': np.sort(a)[rank - 1] if rank <= t else np.inf,
             'p_normal_zero_mean': z * rms,
             'p_normal': optimize.brentq(
                 lambda p: stats.norm.cdf((p - mu) / sd) - stats.norm.cdf((-p - mu) / sd) - coverage,
                 0.0, abs(mu) + 10 * sd),
             'p_distribution_free': rms / math.sqrt(1 - coverage)}
    out = {'mean': mu, 'stdev': sd, 'rms': rms, 'conformal_rank': rank, 'z': z}
    for name, p in bands.items():
        out[name] = p
        out[name.replace('p_', 'inside_')] = (a <= p).mean()
    return out


def relative_error_band(x: np.ndarray = None, y: np.ndarray = None, coverage: float = COVERAGE) -> dict:
    """Return the relative error statistics, the five p95 estimates and the share of pairs inside each band."""
    r = (x - y) / y
    band = band_estimates(r=r, coverage=coverage)
    e = y - x
    rmse = math.sqrt((e ** 2).mean())
    out = {'T': len(r), 'mean_r': band.pop('mean'), 'stdev_r': band.pop('stdev'), 'RMSRE': band.pop('rms'),
           'conformal_rank': band.pop('conformal_rank'), 'z': band.pop('z'),
           'shapiro_W': stats.shapiro(r).statistic, 'shapiro_p': stats.shapiro(r).pvalue,
           'RMSE': rmse, 'y_bar': y.mean(), 'RMSE/y_bar': rmse / y.mean(), 'cv_y': y.std() / y.mean(),
           'R2_all': 1 - (e ** 2).sum() / ((y - y.mean()) ** 2).sum(), 'y_min': y.min(), 'y_max': y.max()}
    out.update(band)
    return out


def absolute_error_band(x: np.ndarray = None, y: np.ndarray = None, coverage: float = COVERAGE) -> dict:
    """Return the statistics of the error x - y, the five q95 estimates (keys q_<method>, unit of y) and the share
    of pairs inside each band x = y +/- q95."""
    band = band_estimates(r=x - y, coverage=coverage)
    out = {'mean_d': band.pop('mean'), 'stdev_d': band.pop('stdev'), 'RMSE': band.pop('rms')}
    band.pop('conformal_rank'), band.pop('z')
    for name, value in band.items():
        out[name.replace('p_', 'q_')] = value
    return out


def error_scaling(x: np.ndarray = None, y: np.ndarray = None, bins: int = SCALING_BINS) -> tuple:
    """Return the PCA of the pairs (y, |x - y|) and the RMS error per bin of y.

    The dict holds the mean point (y_mean, d_mean), the eigenvalues (lambda1 >= lambda2) of the covariance matrix,
    the first and second principal vectors (v1_y, v1_d, v2_y, v2_d; v1_y >= 0, v2_d >= 0) and the angle of v1 from
    the y axis, theta_deg = atan2(v1_d, v1_y) in degrees, with theta_pct = theta_deg / 90 in percent.
    The frame has index 'bin' (1..bins, by increasing y) and columns n, y_min, y_max, rms_y, rms_error, rmsre_pct.
    """
    d = np.abs(y - x)
    values, vectors = np.linalg.eigh(np.cov(np.column_stack([y, d]), rowvar=False))
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    if vectors[0, 0] < 0:
        vectors[:, 0] = -vectors[:, 0]
    if vectors[1, 1] < 0:
        vectors[:, 1] = -vectors[:, 1]
    theta = math.degrees(math.atan2(vectors[1, 0], vectors[0, 0]))
    pca = {'y_mean': y.mean(), 'd_mean': d.mean(), 'lambda1': values[0], 'lambda2': values[1],
           'v1_y': vectors[0, 0], 'v1_d': vectors[1, 0], 'v2_y': vectors[0, 1], 'v2_d': vectors[1, 1],
           'theta_deg': theta, 'theta_pct': theta / RIGHT_ANGLE * PERCENT}
    label = pd.qcut(y, bins, labels=False)
    rows = {}
    for k in range(bins):
        m = label == k
        rows[k + 1] = {'n': int(m.sum()), 'y_min': y[m].min(), 'y_max': y[m].max(),
                       'rms_y': math.sqrt((y[m] ** 2).mean()),
                       'rms_error': math.sqrt((d[m] ** 2).mean()),
                       'rmsre_pct': math.sqrt(((d[m] / y[m]) ** 2).mean()) * PERCENT}
    return pca, pd.DataFrame(rows).T.rename_axis('bin')


def _residual(x: np.ndarray = None, y: np.ndarray = None, band: Band = 'relative') -> np.ndarray:
    """Return |x - y| / y for the relative band or |x - y| for the absolute band."""
    if band not in BANDS:
        raise ValueError(f"band must be one of {BANDS}: {band!r}")
    return np.abs((x - y) / y) if band == 'relative' else np.abs(x - y)


def band_table(x: np.ndarray = None, y: np.ndarray = None, result: dict = None,
               band: Band = 'relative') -> pd.DataFrame:
    """Return one row per method (index 'number') with columns method, equation, section, p95_pct (relative) or
    q95_nm (absolute), pairs_inside, share_inside_pct. result is relative_error_band() or absolute_error_band()."""
    a = _residual(x=x, y=y, band=band)
    prefix, column, factor = BAND_SPEC[band][:3]
    rows = {}
    for key, (number, name, equation, section) in METHODS.items():
        p = result[f"{prefix}{key}"]
        rows[number] = {'method': name, 'equation': equation, 'section': section, column: p * factor,
                        'pairs_inside': int((a <= p).sum()), 'share_inside_pct': (a <= p).mean() * PERCENT}
    return pd.DataFrame(rows).T.rename_axis('number')


def draw(x: np.ndarray = None, y: np.ndarray = None, table: pd.DataFrame = None, stem: pathlib.Path = None,
         band: Band = 'relative') -> pathlib.Path:
    """Save and return one figure with a Pred vs. True panel per method: pairs inside and outside the band, the 1:1
    line and the band x = (1 +/- p95) y (relative) or x = y +/- q95 (absolute)."""
    if len(table) >= GRID[0] * GRID[1]:
        raise ValueError(f"the grid {GRID} needs one free panel for the legend: {len(table)=}")
    scale = FIGSIZE[0] / REFERENCE_WIDTH
    plt.rcParams.update({'font.size': FONT_SIZES['text'] * scale,
                         'axes.labelsize': FONT_SIZES['axis_label'] * scale})
    fig, axes = plt.subplots(*GRID, figsize=FIGSIZE, sharex=True, sharey=True)
    a = _residual(x=x, y=y, band=band)
    _, column, factor, text_format, legend_label = BAND_SPEC[band]
    line_y = np.array([y.min(), y.max()])
    for ax, (number, row) in zip(axes.ravel(), table.iterrows()):
        p = row[column] / factor
        inside = a <= p
        lower, upper = ((1 - p) * line_y, (1 + p) * line_y) if band == 'relative' else (line_y - p, line_y + p)
        ax.fill_between(line_y, lower, upper, color=BAND_COLOR, alpha=BAND_ALPHA, linewidth=0)
        for edge in (lower, upper):
            ax.plot(line_y, edge, color=BAND_COLOR, linewidth=1.6)
        ax.plot(line_y, line_y, color=REFERENCE_COLOR, linewidth=1.4, linestyle='--')
        ax.scatter(y[inside], x[inside], s=10, color=INSIDE_COLOR, edgecolor='none', alpha=0.7, zorder=3)
        ax.scatter(y[~inside], x[~inside], s=18, color=OUTSIDE_COLOR, edgecolor='none', zorder=4)
        ax.text(0.04, 0.96, text_format.format(row[column]) + f"\ninside {row['pairs_inside']} / {len(a)}",
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
                      label=legend_label)], loc='center', frameon=False)
    for ax in axes.ravel()[len(table) + 1:]:
        ax.set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.98, bottom=0.14, wspace=0.12, hspace=0.42)
    path = save_svg_or_png(fig=fig, stem=stem, dpi=DPI)
    plt.close(fig)
    return path


def draw_error(x: np.ndarray = None, y: np.ndarray = None, bins: pd.DataFrame = None, pca: dict = None,
               stem: pathlib.Path = None) -> pathlib.Path:
    """Save and return the residual plot against the measured value y: (a) the error x - y with the RMSE of each
    bin of y, (b) the relative error (x - y) / y in percent with the RMSRE of each bin, both drawn at +/-, and
    (c) the size of the error |x - y| with the first and second principal vectors of its PCA and the angle of v1."""
    if x is None or y is None or bins is None or pca is None or stem is None:
        raise ValueError('x, y, bins, pca and stem are required.')
    scale = ERROR_FIGSIZE[0] / ERROR_REFERENCE_WIDTH
    plt.rcParams.update({'font.size': FONT_SIZES['text'] * scale,
                         'axes.labelsize': FONT_SIZES['axis_label'] * scale})
    fig = plt.figure(figsize=ERROR_FIGSIZE)
    grid = fig.add_gridspec(2, 2)
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
    panels = ((axes[0], x - y, bins['rms_error'], 'Error x − y (nm)', 'bin RMSE', '(a) Error'),
              (axes[1], (x - y) / y * PERCENT, bins['rmsre_pct'], 'Relative error e (%)', 'bin RMSRE',
               '(b) Relative error'))
    for ax, residual, spread, ylabel, spread_label, caption in panels:
        ax.axhline(0, color=REFERENCE_COLOR, linewidth=1.0)
        ax.scatter(y, residual, s=12, color=INSIDE_COLOR, edgecolor='none', alpha=0.6, zorder=3, label='pair')
        for sign in (-1, 1):
            for k, (lo, hi, v) in enumerate(zip(bins['y_min'], bins['y_max'], spread)):
                ax.plot([lo, hi], [sign * v] * 2, color=MODEL_COLORS['proportional'], linewidth=2.2, zorder=4,
                        label=f"±{spread_label}" if sign > 0 and k == 0 else None)
        ax.set_ylabel(ylabel)
    ax = axes[2]
    ax.scatter(y, np.abs(x - y), s=12, color=INSIDE_COLOR, edgecolor='none', alpha=0.6, zorder=3, label='pair')
    center = np.array([pca['y_mean'], pca['d_mean']])
    for k, color in ((1, PCA_COLORS[0]), (2, PCA_COLORS[1])):
        vector = np.array([pca[f"v{k}_y"], pca[f"v{k}_d"]]) * PCA_ARROW_SIGMAS * math.sqrt(pca[f"lambda{k}"])
        ax.annotate('', xy=center + vector, xytext=center, zorder=5,
                    arrowprops={'arrowstyle': '-|>', 'color': color, 'linewidth': 2.4, 'mutation_scale': 18})
        ax.plot([], [], color=color, linewidth=2.4, label=f"v{k}: principal vector {k}")
    ax.text(0.02, 0.95, f"θ = {pca['theta_deg']:.2f}°,  θ / 90° = {pca['theta_pct']:.2f} %", transform=ax.transAxes,
            ha='left', va='top', fontsize=FONT_SIZES['text_box'] * scale)
    ax.set_aspect('equal', adjustable='datalim')
    ax.set_ylabel('Error size |x − y| (nm)')
    for ax, caption in zip(axes, ('(a) Error', '(b) Relative error', '(c) PCA of the error size')):
        ax.set_xlabel('Measured value y (nm)')
        ax.legend(loc='upper right', frameon=False, fontsize=FONT_SIZES['text'] * scale)
        ax.grid(color='#e6e6e6', linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.text(0.5, ERROR_CAPTION_Y, caption, transform=ax.transAxes, ha='center', va='top',
                fontsize=FONT_SIZES['pane_caption'] * scale)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.12, wspace=0.28, hspace=0.55)
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
    ABSOLUTE = absolute_error_band(x=X, y=Y)
    for KEY, VALUE in ABSOLUTE.items():
        print(f"{KEY:24s} {VALUE:.6f}")
    PCA, BINS = error_scaling(x=X, y=Y)
    for KEY, VALUE in PCA.items():
        print(f"{KEY:24s} {VALUE:.6g}")
    print(BINS.to_string(float_format='{:.4f}'.format))
    if ARGS.output_folder is not None:
        TABLE = band_table(x=X, y=Y, result=RESULT)
        print(TABLE.to_string(float_format='{:.4f}'.format))
        TABLE.to_csv(ARGS.output_folder / SAMPLE_CSV_NAME, float_format='%.6f')
        draw(x=X, y=Y, table=TABLE, stem=ARGS.output_folder / FIGURE_STEM)
        ABSOLUTE_TABLE = band_table(x=X, y=Y, result=ABSOLUTE, band='absolute')
        print(ABSOLUTE_TABLE.to_string(float_format='{:.4f}'.format))
        ABSOLUTE_TABLE.to_csv(ARGS.output_folder / ABSOLUTE_CSV_NAME, float_format='%.6f')
        draw(x=X, y=Y, table=ABSOLUTE_TABLE, stem=ARGS.output_folder / ABSOLUTE_FIGURE_STEM, band='absolute')
        pd.DataFrame({'y': Y, 'error': X - Y, 'relative_error': (X - Y) / Y}).to_csv(
            ARGS.output_folder / ERROR_CSV_NAME, index=False, float_format='%.6f')
        draw_error(x=X, y=Y, bins=BINS, pca=PCA, stem=ARGS.output_folder / ERROR_FIGURE_STEM)
