"""Variance decomposition of a wafer measurement table into within-wafer and wafer-to-wafer parts.

The script reads a table whose rows are wafers and whose columns are measurement sites, reports the
one-way ANOVA and the variance components, and draws the three figures the document carries.

Changelog:
- 0.0.0: initial release.
- 0.1.0: mark the w2w detection point on the cumulative figure.
- 0.2.0: drop the flagged-wafer figure and move the cumulative legend to the lower right.
- 0.3.0: hold the measurement table in a class and derive every quantity from it.
- 0.4.0: give the w2w detection point its own class.
- 0.5.0: draw the site value figure one violin per wafer.
- 0.6.0: add the rolling components and the figure that draws them over run order.
- 0.7.0: trace the wafer means on the site value figure instead of their linear trend.
- 0.8.0: scale the rolling figure's right axis as uniformity.
- 0.9.0: draw the per-wafer uniformity on the right axis instead of rescaling the left one.
- 0.10.0: take the components over an expanding window instead of a sliding one.
- 0.11.0: screen each wafer against the expanding within-wafer component of the wafers before it.
- 0.12.0: keep the flagged wafers out of the running baseline and draw the screen instead of the components.
"""

__author__ = 'yRocket'
__version__ = "0.12.0.2026.9.3"

import argparse
import pathlib
import re
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
from scipy import stats

__all__ = ['VarianceComponents', 'W2WDetectionPoint', 'WaferMeasurements']

WAFER_ID_COLUMN: str = 'wafer_id'
SITE_COLUMN_PATTERN: str = r'^S\d+$'
VIOLIN_WIDTH: float = 1.6                # width of one wafer's violin in wafer index units
DETECTION_RATIO: float = 0.98            # the right term counts as the whole spread above this share of it
ALPHA: float = 0.05                      # family-wise error rate of the within-wafer variance test
SCREEN_CONFIDENCE: float = 0.999         # confidence of the chi-square limit a wafer is screened against
SCREEN_WARMUP: int = 20                  # wafers the running baseline is built on before any wafer is judged
FIGSIZE: tuple = (12.0, 5.4)
REFERENCE_WIDTH: float = 12.0            # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 9.0
SAVE_DPI: int = 300

COLOR = list(TABLEAU_COLORS.values())
COLOR_OBSERVED: str = COLOR[0]           # tab:blue
COLOR_TREND: str = COLOR[1]              # tab:orange
COLOR_LEFT_TERM: str = COLOR[2]          # tab:green
COLOR_MARK: str = COLOR[3]               # tab:red
COLOR_RIGHT_TERM: str = COLOR[4]         # tab:purple
COLOR_INK: str = COLOR[7]                # tab:gray


@dataclass(frozen=True)
class VarianceComponents:
    """The one-way ANOVA of a wafer measurement table and the components it splits the variance into."""

    wafer_count: int
    site_count: int
    ms_within: float
    ms_between: float
    f_statistic: float
    p_value: float
    sigma_within: float
    sigma_between: float
    sigma_total: float

    @property
    def icc(self) -> float:
        """Share of the variance of a single site value that the wafer it sits on accounts for."""
        return self.sigma_between ** 2 / (self.sigma_between ** 2 + self.sigma_within ** 2)


class W2WDetectionPoint:
    """The first n from which the wafer-level term carries the whole spread of the wafer means.

    Below that n the site error still accounts for a visible share of the observed spread, so the
    wafer-to-wafer part cannot be told apart from it. The cumulative terms are injected once.
    """

    def __init__(self, terms: pd.DataFrame, ratio: float = DETECTION_RATIO) -> None:
        for column in ('observed', 'right_term'):
            if column not in terms.columns:
                raise ValueError(f"the cumulative terms have no '{column}' column; columns are {list(terms.columns)}")
        if not 0.0 < ratio <= 1.0:
            raise ValueError(f"ratio {ratio} is outside (0, 1]")
        self.terms = terms
        self.ratio = ratio

    @property
    def share(self) -> pd.Series:
        """Share of the observed spread that the right term carries, at each n."""
        return self.terms['right_term'] / self.terms['observed']

    @property
    def n(self) -> int:
        """The wafer count at the detection point."""
        reached = np.flatnonzero(np.nan_to_num(self.share.to_numpy()) >= self.ratio)
        if reached.size == 0:
            raise ValueError(f"the right term never reaches {self.ratio:.0%} of the observed spread")
        return int(self.terms.index[reached[0]])


class WaferMeasurements:
    """A measurement table whose rows are wafers and whose columns are sites, with what it decomposes into.

    The table is injected once; every quantity and every figure is derived from that one table.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        if frame.empty:
            raise ValueError("the measurement table is empty")
        missing = int(frame.isna().sum().sum())
        if missing:
            raise ValueError(f"the measurement table has {missing} missing values; the decomposition needs all of them")
        self.frame = frame
        self.values = frame.to_numpy(dtype=float)
        self.wafer_count, self.site_count = self.values.shape
        self.wafer_mean = self.values.mean(axis=1)
        self.order = np.arange(1, self.wafer_count + 1)

    @classmethod
    def from_csv(cls, csv_path: pathlib.Path) -> 'WaferMeasurements':
        """Read the table from a CSV with a wafer_id column and one column per site."""
        frame = pd.read_csv(csv_path)
        if WAFER_ID_COLUMN not in frame.columns:
            raise ValueError(f"{csv_path} has no '{WAFER_ID_COLUMN}' column; columns are {list(frame.columns)}")
        site_columns = [name for name in frame.columns if re.match(SITE_COLUMN_PATTERN, name)]
        if not site_columns:
            raise ValueError(f"{csv_path} has no site column matching {SITE_COLUMN_PATTERN}")
        return cls(frame=frame.set_index(WAFER_ID_COLUMN)[site_columns])

    def components(self) -> VarianceComponents:
        """Split the variance into a within-wafer and a wafer-to-wafer component."""
        ms_within = ((self.values - self.wafer_mean[:, None]) ** 2).sum() / (self.wafer_count * (self.site_count - 1))
        ms_between = self.site_count * ((self.wafer_mean - self.values.mean()) ** 2).sum() / (self.wafer_count - 1)
        f_statistic = ms_between / ms_within
        var_between = max((ms_between - ms_within) / self.site_count, 0.0)
        return VarianceComponents(
            wafer_count=self.wafer_count,
            site_count=self.site_count,
            ms_within=ms_within,
            ms_between=ms_between,
            f_statistic=f_statistic,
            p_value=stats.f.sf(f_statistic, self.wafer_count - 1, self.wafer_count * (self.site_count - 1)),
            sigma_within=np.sqrt(ms_within),
            sigma_between=np.sqrt(var_between),
            sigma_total=self.values.std(ddof=1),
        )

    def cumulative_terms(self) -> pd.DataFrame:
        """Return the cumulative curves of the wafer means, each step using the first n wafers only.

        Returns a pd.DataFrame indexed by `n` with columns `observed` (the standard deviation of the first n
        wafer means), `left_term` and `right_term` (the two terms of the formula that observed is built from)
        and `sigma_total`; the first two rows are NaN because a standard deviation needs three wafers.
        """
        sigma_within = np.sqrt(np.cumsum(self.values.var(axis=1, ddof=1)) / self.order)
        left_term = sigma_within / np.sqrt(self.site_count)
        head = [np.nan, np.nan]
        observed = np.array(head + [self.wafer_mean[:i].std(ddof=1) for i in range(3, self.wafer_count + 1)])
        sigma_total = np.array(head + [self.values[:i].std(ddof=1) for i in range(3, self.wafer_count + 1)])
        gap = observed ** 2 - left_term ** 2
        # the right term is undefined where the observed value sits below the left term
        right_term = np.where(gap > 0, np.sqrt(np.abs(gap)), np.nan)
        return pd.DataFrame({'observed': observed, 'left_term': left_term, 'right_term': right_term,
                             'sigma_total': sigma_total}, index=pd.Index(self.order, name='n'))

    def wafer_uniformity(self) -> pd.Series:
        """Return each wafer's uniformity, its own site standard deviation over its own mean, in percent."""
        return pd.Series(100 * self.values.std(axis=1, ddof=1) / self.wafer_mean, index=self.frame.index,
                         name='uniformity_percent')

    def expanding_components(self) -> pd.DataFrame:
        """Return the two components over an expanding window: the first wafer to the wafer at each step.

        Returns a pd.DataFrame indexed by `n`, the wafer at the right edge, with columns `sigma_within` and
        `sigma_between`; a single wafer carries no wafer-to-wafer part, so the window starts at two wafers.
        """
        site_variance = self.values.var(axis=1, ddof=1)
        right_edge, within, between = [], [], []
        for stop in range(2, self.wafer_count + 1):
            ms_within = site_variance[:stop].mean()
            ms_between = self.site_count * self.wafer_mean[:stop].var(ddof=1)
            right_edge.append(stop)
            within.append(np.sqrt(ms_within))
            between.append(np.sqrt(max((ms_between - ms_within) / self.site_count, 0.0)))
        return pd.DataFrame({'sigma_within': within, 'sigma_between': between},
                            index=pd.Index(right_edge, name='n'))

    def running_screen(self, confidence: float = SCREEN_CONFIDENCE, warmup: int = SCREEN_WARMUP) -> pd.DataFrame:
        """Flag each wafer whose site standard deviation exceeds the limit set by the wafers before it.

        The baseline at wafer n is the within-wafer component of the wafers before it that were not flagged,
        so a wafer is judged neither against itself nor against anything measured after it, and an excursion
        does not raise the baseline the wafers after it are judged against. The limit is that baseline times
        the chi-square factor of the site count. Returns a pd.DataFrame indexed by the table's wafer id with
        columns `sd_within`, `baseline`, `limit` and `exceeded`; the baseline and the limit are NaN over the
        warm-up wafers, which are left unjudged because their baseline rests on too few wafers.
        """
        if not 0.0 < confidence < 1.0:
            raise ValueError(f"confidence {confidence} is outside (0, 1)")
        if not 2 <= warmup < self.wafer_count:
            raise ValueError(f"warmup {warmup} is outside [2, {self.wafer_count})")
        site_variance = self.values.var(axis=1, ddof=1)
        factor = np.sqrt(stats.chi2.ppf(confidence, self.site_count - 1) / (self.site_count - 1))
        accepted = list(site_variance[:warmup])
        baseline = np.full(self.wafer_count, np.nan)
        exceeded = np.zeros(self.wafer_count, dtype=bool)
        for index in range(warmup, self.wafer_count):
            baseline[index] = np.sqrt(np.mean(accepted))
            exceeded[index] = np.sqrt(site_variance[index]) > baseline[index] * factor
            if not exceeded[index]:
                accepted.append(site_variance[index])
        return pd.DataFrame({'sd_within': np.sqrt(site_variance), 'baseline': baseline,
                             'limit': baseline * factor, 'exceeded': exceeded}, index=self.frame.index)

    def detection_point(self, ratio: float = DETECTION_RATIO) -> int:
        """Return the wafer count at the w2w detection point of this table."""
        return W2WDetectionPoint(terms=self.cumulative_terms(), ratio=ratio).n

    def wafer_report(self, alpha: float = ALPHA) -> pd.DataFrame:
        """Report each wafer and flag the ones whose within-wafer variance exceeds the pooled one.

        Returns a pd.DataFrame indexed by the table's wafer id with columns `mean`, `sd_within`,
        `standard_error`, `worst_site`, `shift_drop_worst`, `p_value` and `flagged`; the test is a chi-square
        against the pooled mean square with a Bonferroni correction over the wafers.
        """
        sd_within = self.values.std(axis=1, ddof=1)
        worst = np.abs(self.values - self.wafer_mean[:, None]).argmax(axis=1)
        trimmed = np.array([np.delete(self.values[i], worst[i]).mean() for i in range(self.wafer_count)])
        chi_square = sd_within ** 2 * (self.site_count - 1) / self.components().ms_within
        p_value = stats.chi2.sf(chi_square, self.site_count - 1)
        return pd.DataFrame({
            'mean': self.wafer_mean,
            'sd_within': sd_within,
            'standard_error': sd_within / np.sqrt(self.site_count),
            'worst_site': worst + 1,
            'shift_drop_worst': self.wafer_mean - trimmed,
            'p_value': p_value,
            'flagged': p_value < alpha / self.wafer_count,
        }, index=self.frame.index)

    @staticmethod
    def _font_size() -> float:
        """Scale the text with the figure so that the layout holds when FIGSIZE changes."""
        return BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH

    def _finish(self, axes: plt.Axes, title: str, xlabel: str, ylabel: str, legend_location: str) -> None:
        """Apply the shared look: no top or right spine, a recessive grid, muted tick and legend text."""
        font_size = self._font_size()
        axes.set_title(title, fontsize=font_size * 1.4, color='black', pad=12, loc='left')
        axes.set_xlabel(xlabel, fontsize=font_size * 1.1, color=COLOR_INK)
        axes.set_ylabel(ylabel, fontsize=font_size * 1.1, color=COLOR_INK)
        for side in ('top', 'right'):
            axes.spines[side].set_visible(False)
        for side in ('left', 'bottom'):
            axes.spines[side].set_color('#d9d8d2')
        axes.set_axisbelow(True)
        axes.tick_params(colors=COLOR_INK, labelsize=font_size)
        legend = axes.legend(loc=legend_location, frameon=True, fontsize=font_size, edgecolor='#e3e2dd')
        for text in legend.get_texts():
            text.set_color(COLOR_INK)

    def draw_site_value_violin(self, figure_path: pathlib.Path, sample_path: pathlib.Path) -> None:
        """Draw one violin per wafer over run order, with the site values on it and the wafer-mean trend."""
        samples = self.frame.stack().rename('site_value').reset_index()
        samples.columns = [WAFER_ID_COLUMN, 'site', 'site_value']
        samples.to_csv(sample_path, index=False)                  # the samples the violins were drawn from

        font_size = self._font_size()
        figure, axes = plt.subplots(figsize=FIGSIZE)
        parts = axes.violinplot([row for row in self.values], positions=self.order, widths=VIOLIN_WIDTH,
                                showextrema=False, showmedians=False)
        for body in parts['bodies']:
            body.set_facecolor(COLOR_OBSERVED)
            body.set_alpha(0.35)
            body.set_linewidth(0)
        axes.scatter(np.repeat(self.order, self.site_count), self.values.ravel(), s=1.5, color=COLOR_INK,
                     alpha=0.55, zorder=4, label=f"site values ({self.site_count} per wafer)")
        axes.plot([], [], color=COLOR_OBSERVED, lw=6, alpha=0.35, label="per-wafer violin")
        axes.plot(self.order, self.wafer_mean, color=COLOR_TREND, lw=1.2, zorder=6, label="wafer mean")
        axes.set_xlim(0, self.wafer_count + 1)
        axes.grid(axis='y', color='#ebeae5', lw=0.9)
        self._finish(axes=axes, title="Distribution of site values on each wafer along run order",
                     xlabel="wafer index (run order)", ylabel="site value", legend_location='lower right')
        figure.tight_layout()
        figure.savefig(figure_path, dpi=SAVE_DPI)
        plt.close(figure)

    def draw_screening(self, figure_path: pathlib.Path, confidence: float = SCREEN_CONFIDENCE) -> None:
        """Draw each wafer's site spread against the running baseline and the limit that judges it."""
        screen = self.running_screen(confidence=confidence)
        warmup = int(screen['limit'].isna().sum())
        exceeded = screen['exceeded'].to_numpy()

        font_size = self._font_size()
        figure, axes = plt.subplots(figsize=FIGSIZE)
        axes.vlines(self.order, 0, screen['sd_within'], color=COLOR_INK, lw=0.8, alpha=0.45, zorder=2)
        axes.scatter(self.order[~exceeded], screen['sd_within'][~exceeded], s=9, color=COLOR_INK, zorder=4,
                     label=r"wafer spread  $s_i$  (within the limit)")
        axes.scatter(self.order[exceeded], screen['sd_within'][exceeded], s=26, color=COLOR_MARK, zorder=6,
                     label=f"WiW excursion ({int(exceeded.sum())} wafers)")
        axes.plot(self.order, screen['limit'], color=COLOR_MARK, lw=1.8, zorder=5,
                  label=r"eq (13) limit  $\sigma_{within}(1..i-1)\,\sqrt{\chi^2_{p,N-1}/(N-1)}$"
                        f"  at p = {confidence}")
        axes.plot(self.order, screen['baseline'], color=COLOR_LEFT_TERM, lw=2.4, zorder=5,
                  label=r"running baseline  $\sigma_{within}$(1..i-1), excursions left out")
        axes.axvspan(0, warmup + 0.5, color=COLOR_INK, alpha=0.07, zorder=1)
        axes.text(0.42, 0.96, f"shaded: warm-up over wafer 1 to {warmup}, baseline only", va='top',
                  transform=axes.transAxes, fontsize=font_size, color=COLOR_INK)
        axes.set_xlim(0, self.wafer_count + 1)
        # leave the top of the axes to the legend so that it never sits on a spike
        axes.set_ylim(0, screen['sd_within'].max() * 1.35)
        axes.grid(axis='y', color='#ebeae5', lw=0.9)
        self._finish(axes=axes, title="Per-wafer spread against the running within-wafer limit of eq (13)",
                     xlabel="i  (wafer index, run order)", ylabel=r"standard deviation of the site values",
                     legend_location='upper left')
        figure.tight_layout()
        figure.savefig(figure_path, dpi=SAVE_DPI)
        plt.close(figure)

    def draw_cumulative_stdev(self, figure_path: pathlib.Path) -> None:
        """Draw the cumulative standard deviation of the wafer means beside the two terms it is built from."""
        terms = self.cumulative_terms()
        observed = terms['observed'].to_numpy()
        left_term = terms['left_term'].to_numpy()
        detection = self.detection_point()

        font_size = self._font_size()
        figure, axes = plt.subplots(figsize=FIGSIZE)
        axes.plot(self.order, observed, color=COLOR_OBSERVED, lw=3.0, zorder=4,
                  label=r"observed  $\sigma_{\mu_n}$  (stdev of wafer means 1..n)")
        axes.plot(self.order, terms['right_term'], color=COLOR_RIGHT_TERM, lw=1.6, ls=(0, (5, 3)), zorder=6,
                  label=r"eq (9) right term:  $\sqrt{s_\mu^2(1..n)}$")
        axes.plot(self.order, left_term, color=COLOR_LEFT_TERM, lw=2.2, ls=(0, (6, 4)), zorder=5,
                  label=r"eq (9) left term:  $\sqrt{\sigma_{within}^2(1..n)/N}$")
        axes.plot(self.order, terms['sigma_total'] / np.sqrt(self.site_count * self.order), color=COLOR_TREND,
                  lw=2.0, ls=(0, (6, 4)), zorder=3, label=r"$\sigma_{total}(1..n)/\sqrt{Nn}$")
        axes.axhline(observed[-1], color=COLOR_INK, lw=1.5, ls=(0, (2, 3)), zorder=2,
                     label=r"$\sigma_{\mu_K}$ = %.2f  (value at n = K)" % observed[-1])
        axes.axvline(detection, color=COLOR_MARK, lw=1.6, ls=(0, (4, 3)), zorder=7,
                     label=f"w2w detection point (n = {detection})")
        axes.annotate(f"n = {detection}", (detection, observed[detection - 1]), textcoords="offset points",
                      xytext=(8, -4), fontsize=font_size, color=COLOR_MARK)
        axes.set_xscale('log')
        axes.set_yscale('log')
        axes.set_xlim(2.5, self.wafer_count * 1.25)
        axes.set_ylim(0.4, 45)
        axes.xaxis.set_major_locator(FixedLocator([3, 5, 10, 20, 50, 100, self.wafer_count]))
        axes.yaxis.set_major_locator(
            FixedLocator([0.5, 1, 2, round(left_term[-1], 1), 10, 20, round(observed[-1], 1)]))
        for axis in (axes.xaxis, axes.yaxis):
            axis.set_major_formatter(ScalarFormatter())
            axis.set_minor_formatter(NullFormatter())
        axes.grid(which='both', color='#ebeae5', lw=0.9)
        self._finish(
            axes=axes,
            title="Cumulative stdev of the wafer means and the two terms of eq (9), each from the first n wafers",
            xlabel="n  (cumulative wafer count, run order) - log",
            ylabel="standard deviation - log", legend_location='lower right')
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
    measurements = WaferMeasurements.from_csv(csv_path=args.input_csv)
    components = measurements.components()
    print(f"wafers {components.wafer_count}, sites {components.site_count}")
    print(f"F = {components.f_statistic:.2f}, p = {components.p_value:.3e}")
    print(f"sigma_within = {components.sigma_within:.3f}, sigma_between = {components.sigma_between:.3f}, "
          f"sigma_total = {components.sigma_total:.3f}, ICC = {components.icc:.3f}")
    report = measurements.wafer_report()
    report.to_csv(args.output_folder / 'wafer_report.csv')
    print(f"wafers with inflated within-wafer variance: {int(report['flagged'].sum())}")
    print(f"w2w detection point: n = {measurements.detection_point()}")
    screen = measurements.running_screen()
    screen.to_csv(args.output_folder / 'running_screen.csv')
    judged = int(screen['limit'].notna().sum())
    print(f"wafers over the running limit: {int(screen['exceeded'].sum())} of {judged} judged")
    measurements.draw_site_value_violin(figure_path=args.output_folder / 'site_value_violin.png',
                                        sample_path=args.output_folder / 'site_value_violin_samples.csv')
    measurements.wafer_uniformity().to_csv(args.output_folder / 'wafer_uniformity.csv')
    measurements.expanding_components().to_csv(args.output_folder / 'expanding_components.csv')
    measurements.draw_screening(figure_path=args.output_folder / 'wafer_screening.png')
    measurements.draw_cumulative_stdev(figure_path=args.output_folder / 'cum_stdev.png')
    print(f"figures written to {args.output_folder}")
