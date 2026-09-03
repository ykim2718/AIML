"""Variance decomposition of a wafer measurement table into within-wafer and wafer-to-wafer parts.

The script reads a table whose rows are wafers and whose columns are measurement sites, reports the
one-way ANOVA and the variance components, and draws the three figures the document carries.

Changelog:
- 0.0.0: initial release.
"""

__author__ = 'yRocket'
__version__ = "0.0.0.2026.9.3"

import argparse
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
from scipy import stats

__all__ = [
    'read_measurements',
    'variance_components',
    'flag_inflated_wafers',
    'draw_site_value_violin',
    'draw_cumulative_stdev',
    'draw_flagged_wafer_means',
]

WAFER_ID_COLUMN: str = 'wafer_id'
SITE_COLUMN_PATTERN: str = r'^S\d+$'
BIN_SIZE: int = 15                       # wafers per violin
MOVING_WINDOW: int = 15                  # wafers per moving average
FIGSIZE: tuple = (12.0, 5.4)
REFERENCE_WIDTH: float = 12.0            # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 9.0
SAVE_DPI: int = 300

COLOR = list(TABLEAU_COLORS.values())
COLOR_OBSERVED: str = COLOR[0]           # tab:blue
COLOR_TREND: str = COLOR[1]              # tab:orange
COLOR_LEFT_TERM: str = COLOR[2]          # tab:green
COLOR_FLAGGED: str = COLOR[3]            # tab:red
COLOR_RIGHT_TERM: str = COLOR[4]         # tab:purple
COLOR_INK: str = COLOR[7]                # tab:gray


def read_measurements(csv_path: pathlib.Path) -> pd.DataFrame:
    """Read the measurement table.

    Returns a pd.DataFrame indexed by `wafer_id` whose columns are the site columns (`S1`, `S2`, ...)
    in file order.
    """
    frame = pd.read_csv(csv_path)
    if WAFER_ID_COLUMN not in frame.columns:
        raise ValueError(f"{csv_path} has no '{WAFER_ID_COLUMN}' column; columns are {list(frame.columns)}")
    site_columns = [name for name in frame.columns if re.match(SITE_COLUMN_PATTERN, name)]
    if not site_columns:
        raise ValueError(f"{csv_path} has no site column matching {SITE_COLUMN_PATTERN}")
    frame = frame.set_index(WAFER_ID_COLUMN)[site_columns]
    missing = int(frame.isna().sum().sum())
    if missing:
        raise ValueError(f"{csv_path} has {missing} missing values; the decomposition needs a complete table")
    return frame


def variance_components(values: np.ndarray) -> dict:
    """Split the variance of the measurements into a within-wafer and a wafer-to-wafer component.

    `values` has one row per wafer and one column per site.
    """
    wafer_count, site_count = values.shape
    wafer_mean = values.mean(axis=1)
    ms_within = ((values - wafer_mean[:, None]) ** 2).sum() / (wafer_count * (site_count - 1))
    ms_between = site_count * ((wafer_mean - values.mean()) ** 2).sum() / (wafer_count - 1)
    f_statistic = ms_between / ms_within
    var_wafer = max((ms_between - ms_within) / site_count, 0.0)
    return {
        'wafer_count': wafer_count,
        'site_count': site_count,
        'ms_within': ms_within,
        'ms_between': ms_between,
        'f_statistic': f_statistic,
        'p_value': stats.f.sf(f_statistic, wafer_count - 1, wafer_count * (site_count - 1)),
        'sigma_within': np.sqrt(ms_within),
        'sigma_between': np.sqrt(var_wafer),
        'icc': var_wafer / (var_wafer + ms_within),
        'sigma_total': values.std(ddof=1),
    }


def flag_inflated_wafers(values: np.ndarray, alpha: float = 0.05) -> pd.DataFrame:
    """Flag the wafers whose within-wafer variance exceeds the pooled one.

    Returns a pd.DataFrame indexed by wafer position (0-based) with columns `mean`, `sd_within`,
    `standard_error`, `worst_site`, `shift_drop_worst`, `p_value` and `flagged`; the test is a chi-square
    against the pooled mean square with a Bonferroni correction over the wafers.
    """
    wafer_count, site_count = values.shape
    wafer_mean = values.mean(axis=1)
    sd_within = values.std(axis=1, ddof=1)
    ms_within = ((values - wafer_mean[:, None]) ** 2).sum() / (wafer_count * (site_count - 1))
    worst = np.abs(values - wafer_mean[:, None]).argmax(axis=1)
    trimmed = np.array([np.delete(values[i], worst[i]).mean() for i in range(wafer_count)])
    p_value = stats.chi2.sf(sd_within ** 2 * (site_count - 1) / ms_within, site_count - 1)
    return pd.DataFrame({
        'mean': wafer_mean,
        'sd_within': sd_within,
        'standard_error': sd_within / np.sqrt(site_count),
        'worst_site': worst + 1,
        'shift_drop_worst': wafer_mean - trimmed,
        'p_value': p_value,
        'flagged': p_value < alpha / wafer_count,
    })


def _style_axes(axes: plt.Axes, title: str, xlabel: str, ylabel: str, font_size: float) -> None:
    """Apply the shared look: no top or right spine, a recessive horizontal grid, muted tick labels."""
    axes.set_title(title, fontsize=font_size * 1.4, color='black', pad=12, loc='left')
    axes.set_xlabel(xlabel, fontsize=font_size * 1.1, color=COLOR_INK)
    axes.set_ylabel(ylabel, fontsize=font_size * 1.1, color=COLOR_INK)
    for side in ('top', 'right'):
        axes.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        axes.spines[side].set_color('#d9d8d2')
    axes.grid(axis='y', color='#ebeae5', lw=0.9)
    axes.set_axisbelow(True)
    axes.tick_params(colors=COLOR_INK, labelsize=font_size)


def draw_site_value_violin(frame: pd.DataFrame, figure_path: pathlib.Path, sample_path: pathlib.Path) -> None:
    """Draw the distribution of the site values in bins of consecutive wafers, with the wafer-mean trend."""
    values = frame.to_numpy(dtype=float)
    wafer_count = values.shape[0]
    wafer_mean = values.mean(axis=1)
    order = np.arange(1, wafer_count + 1)
    starts = range(0, wafer_count, BIN_SIZE)
    bins = [values[start:start + BIN_SIZE].ravel() for start in starts]
    labels = [f"{start + 1}-{min(start + BIN_SIZE, wafer_count)}" for start in starts]
    samples = pd.DataFrame({
        'bin': np.repeat(labels, [len(one) for one in bins]),
        'site_value': np.concatenate(bins),
    })
    samples.to_csv(sample_path, index=False)                      # the samples the violins were drawn from
    slope, intercept, r_value, _, _ = stats.linregress(order, wafer_mean)

    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    figure, axes = plt.subplots(figsize=FIGSIZE)
    parts = axes.violinplot(bins, positions=np.arange(len(bins)), widths=0.85,
                            showextrema=False, showmedians=True)
    for body in parts['bodies']:
        body.set_facecolor(COLOR_OBSERVED)
        body.set_alpha(0.40)
        body.set_edgecolor(COLOR_OBSERVED)
        body.set_linewidth(1.2)
    parts['cmedians'].set_color(COLOR_INK)
    parts['cmedians'].set_linewidth(2)
    axes.plot([], [], color=COLOR_OBSERVED, lw=6, alpha=0.40, label="site values in the bin (violin)")
    axes.plot([], [], color=COLOR_INK, lw=2, label="bin median")
    axes.plot((order - (BIN_SIZE + 1) / 2) / BIN_SIZE, intercept + slope * order, color=COLOR_TREND, lw=2.4,
              zorder=6, label=f"wafer-mean trend {slope:+.3f}/wafer (r$^2$={r_value ** 2:.2f})")
    axes.set_xticks(np.arange(len(bins)))
    axes.set_xticklabels(labels, rotation=45, ha='right', fontsize=font_size * 0.95, color=COLOR_INK)
    axes.set_xlim(-0.8, len(bins) - 0.2)
    _style_axes(axes=axes, title="Distribution of site values along run order",
                xlabel=f"wafer index range (run order, {BIN_SIZE} wafers per bin)",
                ylabel="site value", font_size=font_size)
    legend = axes.legend(loc='lower right', frameon=True, fontsize=font_size, edgecolor='#e3e2dd')
    for text in legend.get_texts():
        text.set_color(COLOR_INK)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=SAVE_DPI)
    plt.close(figure)


def draw_cumulative_stdev(frame: pd.DataFrame, figure_path: pathlib.Path) -> None:
    """Draw the cumulative standard deviation of the wafer means beside the two terms it is built from.

    Every curve at step n uses the first n wafers only, so no step reads a wafer it has not reached yet.
    """
    values = frame.to_numpy(dtype=float)
    wafer_count, site_count = values.shape
    wafer_mean = values.mean(axis=1)
    order = np.arange(1, wafer_count + 1)
    sigma_within = np.sqrt(np.cumsum(values.var(axis=1, ddof=1)) / order)
    left_term = sigma_within / np.sqrt(site_count)
    observed = np.array([np.nan, np.nan] + [wafer_mean[:i].std(ddof=1) for i in range(3, wafer_count + 1)])
    sigma_total = np.array([np.nan, np.nan] + [values[:i].std(ddof=1) for i in range(3, wafer_count + 1)])
    gap = observed ** 2 - left_term ** 2
    # the right term is undefined where the observed value sits below the left term
    right_term = np.where(gap > 0, np.sqrt(np.abs(gap)), np.nan)

    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    figure, axes = plt.subplots(figsize=FIGSIZE)
    axes.plot(order, observed, color=COLOR_OBSERVED, lw=3.0, zorder=4,
              label=r"observed  $\sigma_{\mu_n}$  (stdev of wafer means 1..n)")
    axes.plot(order, right_term, color=COLOR_RIGHT_TERM, lw=1.6, ls=(0, (5, 3)), zorder=6,
              label=r"eq (9) right term:  $\sqrt{s_\mu^2(1..n)}$")
    axes.plot(order, left_term, color=COLOR_LEFT_TERM, lw=2.2, ls=(0, (6, 4)), zorder=5,
              label=r"eq (9) left term:  $\sqrt{\sigma_{within}^2(1..n)/N}$")
    axes.plot(order, sigma_total / np.sqrt(site_count * order), color=COLOR_TREND, lw=2.0, ls=(0, (6, 4)),
              zorder=3, label=r"$\sigma_{total}(1..n)/\sqrt{Nn}$")
    axes.axhline(observed[-1], color=COLOR_INK, lw=1.5, ls=(0, (2, 3)), zorder=2,
                 label=r"$\sigma_{\mu_K}$ = %.2f  (value at n = K)" % observed[-1])
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_xlim(2.5, wafer_count * 1.25)
    axes.set_ylim(0.4, 45)
    axes.xaxis.set_major_locator(FixedLocator([3, 5, 10, 20, 50, 100, wafer_count]))
    axes.yaxis.set_major_locator(FixedLocator([0.5, 1, 2, round(left_term[-1], 1), 10, 20, round(observed[-1], 1)]))
    for axis in (axes.xaxis, axes.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_formatter(NullFormatter())
    _style_axes(axes=axes,
                title="Cumulative stdev of the wafer means and the two terms of eq (9), each from the first n wafers",
                xlabel="n  (cumulative wafer count, run order) - log",
                ylabel="standard deviation - log", font_size=font_size)
    axes.grid(which='both', axis='both', color='#ebeae5', lw=0.9)
    legend = axes.legend(loc='lower left', frameon=True, fontsize=font_size, edgecolor='#e3e2dd')
    for text in legend.get_texts():
        text.set_color(COLOR_INK)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=SAVE_DPI)
    plt.close(figure)


def draw_flagged_wafer_means(frame: pd.DataFrame, report: pd.DataFrame, figure_path: pathlib.Path) -> None:
    """Draw the wafer means over run order, marking the wafers whose within-wafer spread drives the mean."""
    values = frame.to_numpy(dtype=float)
    wafer_count = values.shape[0]
    order = np.arange(1, wafer_count + 1)
    wafer_mean = report['mean'].to_numpy()
    sd_within = report['sd_within'].to_numpy()
    flagged = report['flagged'].to_numpy()
    moving = pd.Series(wafer_mean).rolling(MOVING_WINDOW, center=True, min_periods=MOVING_WINDOW // 3).mean()

    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    figure, axes = plt.subplots(figsize=FIGSIZE)
    axes.errorbar(order[~flagged], wafer_mean[~flagged], yerr=sd_within[~flagged], fmt='none',
                  ecolor=COLOR_OBSERVED, elinewidth=1.4, alpha=0.35, zorder=1)
    axes.errorbar(order[flagged], wafer_mean[flagged], yerr=sd_within[flagged], fmt='none',
                  ecolor=COLOR_FLAGGED, elinewidth=1.8, alpha=0.35, zorder=2)
    axes.plot(order, wafer_mean, color=COLOR_OBSERVED, lw=1.1, alpha=0.35, zorder=2)
    axes.scatter(order[~flagged], wafer_mean[~flagged], s=20, color=COLOR_OBSERVED, edgecolor='white',
                 linewidth=0.8, zorder=3, label=r"wafer mean $\pm 1\sigma$ (within)")
    axes.scatter(order[flagged], wafer_mean[flagged], s=60, marker='D', color=COLOR_FLAGGED, edgecolor='white',
                 linewidth=1.0, zorder=6, label=f"within-wafer variance inflated: {int(flagged.sum())} wafers")
    axes.plot(order, moving, color=COLOR_TREND, lw=2.2, zorder=4, label=f"{MOVING_WINDOW}-wafer moving average")
    for position in np.argsort(-sd_within)[:4]:
        axes.annotate(frame.index[position], (order[position], wafer_mean[position]), textcoords="offset points",
                      xytext=(6, 10), fontsize=font_size * 0.85, color=COLOR_FLAGGED)
    _style_axes(axes=axes, title="Wafers whose within-wafer spread drives the mean",
                xlabel="wafer index (run order)", ylabel="wafer mean", font_size=font_size)
    legend = axes.legend(loc='lower right', frameon=True, fontsize=font_size * 0.95, edgecolor='#e3e2dd')
    for text in legend.get_texts():
        text.set_color(COLOR_INK)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=SAVE_DPI)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse the command line and check that the input file and the output folder are usable."""
    parser = argparse.ArgumentParser(
        description=f"wiw_w2w_anova.py {__version__} - decompose a wafer measurement table and draw its figures.")
    parser.add_argument('-v', '--version', action='version', version=f"wiw_w2w_anova.py {__version__}")
    parser.add_argument('--input-csv', type=pathlib.Path, default=pathlib.Path('example.csv'),
                        help="measurement table: one row per wafer, a wafer_id column and S1, S2, ... site columns")
    parser.add_argument('--output-folder', type=pathlib.Path, default=pathlib.Path('wiw-w2w-anova_fig'),
                        help="folder the figures and the chart samples are written to")
    parsed = parser.parse_args()
    if not parsed.input_csv.is_file():
        parser.error(f"--input-csv {parsed.input_csv} is not a file")
    parsed.output_folder.mkdir(parents=True, exist_ok=True)
    return parsed


if __name__ == '__main__':
    args = parse_args()
    measurements = read_measurements(csv_path=args.input_csv)
    measured = measurements.to_numpy(dtype=float)
    components = variance_components(values=measured)
    print(f"wafers {components['wafer_count']}, sites {components['site_count']}")
    print(f"F = {components['f_statistic']:.2f}, p = {components['p_value']:.3e}")
    print(f"sigma_within = {components['sigma_within']:.3f}, sigma_between = {components['sigma_between']:.3f}, "
          f"sigma_total = {components['sigma_total']:.3f}, ICC = {components['icc']:.3f}")
    wafer_report = flag_inflated_wafers(values=measured)
    wafer_report.index = measurements.index
    wafer_report.to_csv(args.output_folder / 'wafer_report.csv')
    print(f"wafers with inflated within-wafer variance: {int(wafer_report['flagged'].sum())}")
    draw_site_value_violin(frame=measurements,
                           figure_path=args.output_folder / 'site_value_violin.png',
                           sample_path=args.output_folder / 'site_value_violin_samples.csv')
    draw_cumulative_stdev(frame=measurements, figure_path=args.output_folder / 'cum_stdev.png')
    draw_flagged_wafer_means(frame=measurements, report=wafer_report,
                             figure_path=args.output_folder / 'wafer_mean_flagged.png')
    print(f"figures written to {args.output_folder}")
