"""Generalized ESD outlier detection following the algorithm of ISO 16269-4:2010, Annex A.

The script implements the many-outlier procedure of Rosner (1983): the extreme studentized
deviate is computed r times, one observation being removed at each step, and the number of
outliers is the largest step whose statistic still exceeds its own critical value. Running the
script reproduces the worked example of the accompanying document and writes the two figures
together with the samples they were drawn from.

Changelog:
    0.0.0 - Initial release.
"""

__author__ = 'yRocket'
__version__ = "0.0.1.2026.8.17"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Union

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

# The 54-observation sample used as the worked example. It carries three outliers that the
# single-outlier form of the test cannot reach, which is the reason the many-outlier form exists.
EXAMPLE_DATA = np.array([
    -0.25, 0.68, 0.94, 1.15, 1.20, 1.26, 1.26, 1.34, 1.38, 1.43, 1.49, 1.49, 1.55, 1.56,
    1.58, 1.65, 1.69, 1.70, 1.76, 1.77, 1.81, 1.91, 1.94, 1.96, 1.99, 2.06, 2.09, 2.10,
    2.14, 2.15, 2.23, 2.24, 2.26, 2.35, 2.37, 2.40, 2.47, 2.54, 2.62, 2.64, 2.90, 2.92,
    2.92, 2.93, 3.21, 3.26, 3.30, 3.59, 3.68, 4.30, 4.64, 5.34, 5.42, 6.01,
])

PALETTE = list(TABLEAU_COLORS.values())

# Significance levels the sensitivity panel sweeps the upper bound against.
SENSITIVITY_ALPHAS = (0.01, 0.05, 0.10)


@dataclass
class GesdStep:
    """One iteration of the procedure, before the number of outliers has been decided.

    Attributes:
        step: the iteration index i, counted from 1.
        position: position of the removed observation in the original array, counted from 0.
        value: the removed observation.
        mean: mean of the observations still present at the start of the iteration.
        deviation: sample standard deviation of those observations, computed with ddof = 1.
        statistic: the extreme studentized deviate R_i.
        critical: the critical value lambda_i of that same iteration.
    """

    step: int
    position: int
    value: float
    mean: float
    deviation: float
    statistic: float
    critical: float


@dataclass
class GesdResult:
    """Outcome of the procedure over all r iterations.

    Attributes:
        steps: the r iterations in the order they were computed.
        count: the number of outliers, that is the largest i whose statistic exceeds its critical value.
        positions: positions of those outliers in the original array, sorted ascending.
        alpha: the significance level the critical values were computed at.
    """

    steps: list[GesdStep]
    count: int
    positions: np.ndarray
    alpha: float

    def to_frame(self) -> pd.DataFrame:
        """Tabulate every iteration.

        Returns:
            A pd.DataFrame indexed by 'step' (the iteration index i), with columns
            'position', 'value', 'mean', 'deviation', 'statistic', 'critical' and 'exceeds'.
        """
        frame = pd.DataFrame([vars(one_step) for one_step in self.steps])
        frame['exceeds'] = frame['statistic'] > frame['critical']
        return frame.set_index('step')


def gesd_critical_value(sample_size: int = None, step: int = None, alpha: float = None) -> float:
    """Critical value lambda_i of one iteration of the generalized ESD procedure.

    The percentile is Bonferroni-corrected by the number of observations still present, which is
    what keeps the overall type I error at alpha across the r iterations rather than at each one.

    Args:
        sample_size: n, the size of the original sample.
        step: the iteration index i, counted from 1.
        alpha: significance level of the whole procedure.

    Returns:
        The critical value lambda_i.
    """
    degrees_of_freedom = sample_size - step - 1
    if degrees_of_freedom < 1:
        raise ValueError(f"step {step} leaves {degrees_of_freedom} degrees of freedom for n = {sample_size}; "
                         f"the largest usable step is n - 2 = {sample_size - 2}.")
    percentile = 1.0 - alpha / (2.0 * (sample_size - step + 1))
    quantile = float(stats.t.ppf(percentile, degrees_of_freedom))
    return ((sample_size - step) * quantile
            / np.sqrt((degrees_of_freedom + quantile ** 2) * (sample_size - step + 1)))


def gesd_test(data: np.ndarray = None, max_outliers: int = None, alpha: float = 0.05) -> GesdResult:
    """Detect up to max_outliers outliers in an approximately normal univariate sample.

    The procedure never stops early. All r iterations are computed and the decision is taken
    afterwards, because an iteration whose statistic falls below its critical value can still be
    followed by one that exceeds it; stopping at the first failure is what masking exploits.

    Args:
        data: the sample, shape (n,).
        max_outliers: r, the upper bound on the number of outliers. Must satisfy 1 <= r <= n - 2.
        alpha: significance level of the whole procedure.

    Returns:
        A GesdResult holding every iteration and the resulting outlier count.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"data carries {int((~np.isfinite(values)).sum())} non-finite values; "
                         f"remove or impute them before testing.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly between 0 and 1, got {alpha}.")
    sample_size = values.size
    if not 1 <= max_outliers <= sample_size - 2:
        raise ValueError(f"max_outliers must lie between 1 and n - 2 = {sample_size - 2}, got {max_outliers}.")

    survivors = np.arange(sample_size)                       # positions in the original array
    steps: list[GesdStep] = []
    for step in range(1, max_outliers + 1):
        present = values[survivors]
        mean = float(present.mean())
        deviation = float(present.std(ddof=1))
        if deviation == 0.0:
            raise ValueError(f"the {present.size} observations left at step {step} are all equal, so the "
                             f"studentized deviate is undefined; the sample is not usable for this test.")
        offsets = np.abs(present - mean)
        extreme = int(np.argmax(offsets))
        steps.append(GesdStep(step=step, position=int(survivors[extreme]), value=float(present[extreme]),
                              mean=mean, deviation=deviation, statistic=float(offsets[extreme] / deviation),
                              critical=gesd_critical_value(sample_size=sample_size, step=step, alpha=alpha)))
        survivors = np.delete(survivors, extreme)

    exceeding = [one_step.step for one_step in steps if one_step.statistic > one_step.critical]
    count = max(exceeding) if exceeding else 0
    positions = np.sort(np.array([one_step.position for one_step in steps[:count]], dtype=int))
    return GesdResult(steps=steps, count=count, positions=positions, alpha=alpha)


def sensitivity_frame(data: np.ndarray = None, max_outliers: int = None,
                      alphas: tuple = SENSITIVITY_ALPHAS) -> pd.DataFrame:
    """Number of outliers the procedure reports as the upper bound and the significance level move.

    Args:
        data: the sample, shape (n,).
        max_outliers: the largest upper bound to sweep to.
        alphas: the significance levels to sweep.

    Returns:
        A pd.DataFrame indexed by 'max_outliers' (the upper bound r), with one column per
        significance level named 'alpha=<value>', holding the reported outlier count.
    """
    columns = {}
    for alpha in alphas:
        full = gesd_test(data=data, max_outliers=max_outliers, alpha=alpha)
        # Truncating the r iterations at bound reproduces exactly the run made with that bound,
        # because the iterations themselves do not depend on r.
        counts = []
        for bound in range(1, max_outliers + 1):
            exceeding = [one.step for one in full.steps[:bound] if one.statistic > one.critical]
            counts.append(max(exceeding) if exceeding else 0)
        columns[f"alpha={alpha:g}"] = counts
    frame = pd.DataFrame(columns, index=pd.Index(range(1, max_outliers + 1), name='max_outliers'))
    return frame


def report(data: np.ndarray = None, result: GesdResult = None) -> None:
    """Print the sample summary and every iteration of the procedure."""
    sample_size = data.size
    print(f"[1] Sample: n = {sample_size}, mean = {data.mean():.4f}, sd = {data.std(ddof=1):.4f}, "
          f"min = {data.min():.2f}, max = {data.max():.2f}")
    print(f"[2] Procedure: r = {len(result.steps)}, alpha = {result.alpha}\n")
    frame = result.to_frame()
    with pd.option_context('display.float_format', '{:.4f}'.format, 'display.width', 120):
        print(frame[['value', 'mean', 'deviation', 'statistic', 'critical', 'exceeds']].to_string())
    if result.count == 0:
        print(f"\n[3] No observation exceeded its critical value; the sample carries no detectable outlier.")
        return
    print(f"\n[3] Outliers: {result.count} at positions {result.positions.tolist()}, "
          f"values {np.sort(data[result.positions]).tolist()}")


def plot_procedure(data: np.ndarray = None, result: GesdResult = None, output_path: pathlib.Path = None) -> None:
    """Draw the sample with its flagged observations and the statistic against the critical value."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    flagged = np.zeros(data.size, dtype=bool)
    flagged[result.positions] = True

    axes[0].scatter(np.arange(data.size)[~flagged], data[~flagged], s=28, color=PALETTE[0], label='retained')
    axes[0].scatter(np.arange(data.size)[flagged], data[flagged], s=70, color=PALETTE[3], marker='D',
                    label=f"flagged ({result.count})")
    clean = data[~flagged]
    axes[0].axhline(clean.mean(), color=PALETTE[7], linewidth=1.2,
                    label=f"mean without flagged = {clean.mean():.3f}")
    axes[0].set_xlabel('position in the sample')
    axes[0].set_ylabel('observed value')
    axes[0].set_title('(a) Sample and the flagged observations')
    axes[0].legend(loc='upper left', frameon=False)

    frame = result.to_frame()
    axes[1].plot(frame.index, frame['statistic'], marker='o', color=PALETTE[0], label='statistic R_i')
    axes[1].plot(frame.index, frame['critical'], marker='s', linestyle='--', color=PALETTE[3],
                 label=f"critical value lambda_i (alpha = {result.alpha})")
    if result.count > 0:
        axes[1].axvline(result.count, color=PALETTE[7], linewidth=1.2,
                        label=f"largest exceeding step = {result.count}")
    axes[1].set_xticks(frame.index)
    axes[1].set_xlabel('step i')
    axes[1].set_ylabel('value')
    axes[1].set_title('(b) Statistic against critical value at each step')
    axes[1].legend(loc='lower left', frameon=False)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"[4] Figure written to {output_path}")


def plot_diagnostics(data: np.ndarray = None, result: GesdResult = None, sensitivity: pd.DataFrame = None,
                     output_path: pathlib.Path = None) -> None:
    """Draw the normal quantile plot before and after removal, and the sensitivity to r and alpha."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    flagged = np.zeros(data.size, dtype=bool)
    flagged[result.positions] = True

    for label, sample, color in (('full sample', data, PALETTE[0]), ('flagged removed', data[~flagged], PALETTE[2])):
        quantiles = stats.norm.ppf((np.arange(1, sample.size + 1) - 0.375) / (sample.size + 0.25))
        axes[0].scatter(quantiles, np.sort(sample), s=24, color=color, label=label)
    reference = data[~flagged]
    grid = np.linspace(-3, 3, 100)
    axes[0].plot(grid, reference.mean() + reference.std(ddof=1) * grid, color=PALETTE[7], linewidth=1.2,
                 label='normal line fitted without the flagged')
    axes[0].set_xlabel('theoretical normal quantile')
    axes[0].set_ylabel('ordered observation')
    axes[0].set_title('(a) Normal quantile plot before and after removal')
    axes[0].legend(loc='upper left', frameon=False)

    # The curves coincide wherever two significance levels agree, so the later ones are drawn
    # thinner and their markers smaller; otherwise the last curve would hide the others entirely.
    for order, (column, color) in enumerate(zip(sensitivity.columns, PALETTE)):
        axes[1].step(sensitivity.index, sensitivity[column], where='mid', marker='o', color=color, label=column,
                     linewidth=3.6 - 1.2 * order, markersize=11 - 3 * order, alpha=0.9)
    axes[1].set_xticks(sensitivity.index)
    axes[1].set_yticks(range(0, int(sensitivity.to_numpy().max()) + 2))
    axes[1].set_xlabel('upper bound r')
    axes[1].set_ylabel('outliers reported')
    axes[1].set_title('(b) Sensitivity to the upper bound and the significance level')
    axes[1].legend(loc='upper left', frameon=False)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"[5] Figure written to {output_path}")


def write_chart_data(data: np.ndarray = None, result: GesdResult = None, sensitivity: pd.DataFrame = None,
                     output_folder: pathlib.Path = None) -> None:
    """Write the samples the figures were drawn from, one observation per row, without summaries."""
    flagged = np.zeros(data.size, dtype=bool)
    flagged[result.positions] = True
    sample_frame = pd.DataFrame({'value': data, 'flagged': flagged},
                                index=pd.Index(np.arange(data.size), name='position'))
    targets = {
        'sample.csv': sample_frame,
        'steps.csv': result.to_frame(),
        'sensitivity.csv': sensitivity,
    }
    for name, frame in targets.items():
        frame.to_csv(output_folder / name)
        print(f"[6] Chart data written to {output_folder / name}")


def parse_args() -> argparse.Namespace:
    """Parse and validate the command line options."""
    parser = argparse.ArgumentParser(
        description='Detect outliers with the generalized ESD procedure of ISO 16269-4:2010, Annex A.')
    parser.add_argument('--input-csv', type=pathlib.Path, default=None,
                        help='CSV holding the sample; the column named by --column is tested '
                             '(default: the built-in worked example)')
    parser.add_argument('--column', type=str, default='value',
                        help='column of --input-csv to test (default: %(default)s)')
    parser.add_argument('--max-outliers', type=int, default=10,
                        help='upper bound r on the number of outliers (default: %(default)s)')
    parser.add_argument('--alpha', type=float, default=0.05,
                        help='significance level of the whole procedure (default: %(default)s)')
    parser.add_argument('--save-figure', choices=['true', 'false'], default='true',
                        help='write the two figures and the samples behind them (default: %(default)s)')
    parser.add_argument('--output-folder', type=pathlib.Path, default=None,
                        help='folder for the figures (default: generalized-esd_fig '
                             'next to this script)')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    args.save_figure = args.save_figure == 'true'
    if args.output_folder is None:
        args.output_folder = pathlib.Path(__file__).resolve().parent / 'generalized-esd_fig'

    if args.input_csv is not None and not args.input_csv.is_file():
        parser.error(f"--input-csv is not a file: {args.input_csv}")
    if not 0.0 < args.alpha < 1.0:
        parser.error(f"--alpha must lie strictly between 0 and 1, got {args.alpha}.")
    if args.max_outliers < 1:
        parser.error(f"--max-outliers must be at least 1, got {args.max_outliers}.")
    if args.save_figure:
        args.output_folder.mkdir(parents=True, exist_ok=True)

    return args


def load_sample(input_csv: Union[pathlib.Path, None] = None, column: str = None) -> np.ndarray:
    """Read the sample to test, falling back to the built-in worked example."""
    if input_csv is None:
        return EXAMPLE_DATA
    frame = pd.read_csv(input_csv)
    if column not in frame.columns:
        raise ValueError(f"column '{column}' is absent from {input_csv}; available: {list(frame.columns)}")
    return frame[column].to_numpy(dtype=float)


if __name__ == '__main__':
    options = parse_args()

    sample = load_sample(input_csv=options.input_csv, column=options.column)
    outcome = gesd_test(data=sample, max_outliers=options.max_outliers, alpha=options.alpha)
    report(data=sample, result=outcome)

    if options.save_figure:
        sweep = sensitivity_frame(data=sample, max_outliers=options.max_outliers)
        plot_procedure(data=sample, result=outcome,
                       output_path=options.output_folder / 'gesd_procedure.png')
        plot_diagnostics(data=sample, result=outcome, sensitivity=sweep,
                         output_path=options.output_folder / 'gesd_diagnostics.png')
        write_chart_data(data=sample, result=outcome, sensitivity=sweep, output_folder=options.output_folder)
