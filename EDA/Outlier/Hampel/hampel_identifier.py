"""Hampel identifier: outlier detection from the median and the median absolute deviation.

The classical z-score divides a deviation from the mean by the standard deviation, and an outlier
contaminates both. This module replaces the pair with the median and the MAD, which a minority of
contaminating observations cannot move, and flags an observation whose rescaled deviation exceeds
a threshold. The classical z-score is provided alongside so the two can be compared on the same
sample, which is what the accompanying document does.

Changelog:
    0.0.0 - Initial release.
"""

__author__ = 'yRocket'
__version__ = "0.0.0.2026.8.17"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys
from dataclasses import dataclass
from typing import Union

import numpy as np
import pandas as pd
from scipy import stats

# The MAD of a normal sample converges to this multiple of sigma, so dividing by it puts the
# modified score on the same scale as a classical z-score. It is the third quartile of N(0, 1).
NORMAL_QUARTILE = float(stats.norm.ppf(0.75))

# The cut-off recommended by Iglewicz and Hoaglin for the modified z-score.
DEFAULT_THRESHOLD = 3.5


@dataclass
class HampelResult:
    """Outcome of the Hampel identifier on one sample.

    Attributes:
        values: the sample, shape (n,).
        centre: the median of the sample.
        mad: the median absolute deviation, before any rescaling.
        scale: mad divided by NORMAL_QUARTILE, which estimates sigma for a normal sample.
        scores: the modified z-score of each observation, shape (n,).
        threshold: the cut-off the absolute scores were compared against.
        positions: positions of the flagged observations, sorted ascending.
    """

    values: np.ndarray
    centre: float
    mad: float
    scale: float
    scores: np.ndarray
    threshold: float
    positions: np.ndarray

    @property
    def count(self) -> int:
        """Number of flagged observations."""
        return int(self.positions.size)

    def bounds(self) -> tuple[float, float]:
        """The interval outside which an observation is flagged."""
        return self.centre - self.threshold * self.scale, self.centre + self.threshold * self.scale

    def to_frame(self) -> pd.DataFrame:
        """Tabulate the sample with its scores.

        Returns:
            A pd.DataFrame indexed by 'position' (counted from 0), with columns 'value',
            'modified_z', 'classical_z' and 'flagged'.
        """
        flagged = np.zeros(self.values.size, dtype=bool)
        flagged[self.positions] = True
        return pd.DataFrame({'value': self.values, 'modified_z': self.scores,
                             'classical_z': classical_z_scores(data=self.values), 'flagged': flagged},
                            index=pd.Index(np.arange(self.values.size), name='position'))


def median_absolute_deviation(data: np.ndarray = None) -> float:
    """Median of the absolute deviations from the sample median.

    Args:
        data: the sample, shape (n,).

    Returns:
        The MAD, unscaled.
    """
    values = np.asarray(data, dtype=float)
    return float(np.median(np.abs(values - np.median(values))))


def classical_z_scores(data: np.ndarray = None) -> np.ndarray:
    """Deviation from the mean divided by the sample standard deviation, with ddof = 1.

    The result of this function cannot exceed max_attainable_z for the sample size, whatever the
    data are, because the extreme observation enters both the mean and the standard deviation.

    Args:
        data: the sample, shape (n,).

    Returns:
        The scores, shape (n,).
    """
    values = np.asarray(data, dtype=float)
    deviation = values.std(ddof=1)
    if deviation == 0.0:
        raise ValueError(f"all {values.size} observations are equal, so the classical z-score is undefined.")
    return (values - values.mean()) / deviation


def max_attainable_z(sample_size: int = None) -> float:
    """Largest absolute classical z-score a sample of this size can produce.

    The bound is (n - 1) / sqrt(n). A rule that compares the classical z-score against a threshold
    at or above this value can never flag anything, no matter how extreme an observation is.

    Args:
        sample_size: n.

    Returns:
        The bound.
    """
    if sample_size < 2:
        raise ValueError(f"a z-score needs at least 2 observations, got {sample_size}.")
    return (sample_size - 1) / np.sqrt(sample_size)


def hampel_test(data: np.ndarray = None, threshold: float = DEFAULT_THRESHOLD,
                quartile: float = NORMAL_QUARTILE) -> HampelResult:
    """Flag observations whose modified z-score exceeds the threshold in absolute value.

    Args:
        data: the sample, shape (n,).
        threshold: the cut-off, 3.5 by convention.
        quartile: the constant the MAD is divided by to estimate sigma; the default makes the
            score agree with a classical z-score on normal data.

    Returns:
        A HampelResult holding the centre, the scale, every score, and the flagged positions.
    """
    values = np.asarray(data, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {values.shape}.")
    if values.size < 3:
        raise ValueError(f"the identifier needs at least 3 observations to have a usable median, got {values.size}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"data carries {int((~np.isfinite(values)).sum())} non-finite values; "
                         f"remove or impute them before testing.")
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}.")
    if quartile <= 0.0:
        raise ValueError(f"quartile must be positive, got {quartile}.")

    centre = float(np.median(values))
    mad = median_absolute_deviation(data=values)
    if mad == 0.0:
        # More than half the sample sits on one value, so the MAD carries no scale at all.
        # Returning zero scores here would report a clean sample, which is the opposite of the truth.
        repeated = int((values == centre).sum())
        raise ValueError(f"the MAD is 0 because {repeated} of {values.size} observations equal the median "
                         f"{centre}; the modified z-score is undefined. Use a scale estimator that tolerates "
                         f"ties, such as Sn or Qn, or report that the sample cannot support the test.")

    scale = mad / quartile
    scores = (values - centre) / scale
    return HampelResult(values=values, centre=centre, mad=mad, scale=scale, scores=scores,
                        threshold=threshold, positions=np.flatnonzero(np.abs(scores) > threshold))


def threshold_sweep(data: np.ndarray = None, thresholds: np.ndarray = None) -> pd.DataFrame:
    """Number of observations each rule flags as the threshold moves.

    Args:
        data: the sample, shape (n,).
        thresholds: the cut-offs to sweep, shape (m,).

    Returns:
        A pd.DataFrame indexed by 'threshold', with columns 'hampel' and 'classical' holding the
        count each rule reports at that cut-off.
    """
    modified = hampel_test(data=data, threshold=float(thresholds[0])).scores
    classical = classical_z_scores(data=data)
    return pd.DataFrame({'hampel': [int((np.abs(modified) > t).sum()) for t in thresholds],
                         'classical': [int((np.abs(classical) > t).sum()) for t in thresholds]},
                        index=pd.Index(thresholds, name='threshold'))


def report(result: HampelResult = None) -> None:
    """Print the centre, the scale, and every flagged observation."""
    lower, upper = result.bounds()
    size = result.values.size
    print(f"[1] Sample: n = {size}, median = {result.centre:.6f}, MAD = {result.mad:.6f}, "
          f"scale = {result.scale:.6f}")
    print(f"[2] Rule: |modified z| > {result.threshold}, so the retained interval is "
          f"[{lower:.6f}, {upper:.6f}]")
    print(f"[3] Classical scale for comparison: sd = {result.values.std(ddof=1):.6f}, "
          f"which is {result.values.std(ddof=1) / result.scale:.1f} times the robust scale")
    print(f"[4] The classical z-score of this sample cannot exceed {max_attainable_z(sample_size=size):.4f}\n")
    frame = result.to_frame()
    with pd.option_context('display.float_format', '{:.4f}'.format, 'display.width', 120):
        print(frame.to_string())
    if result.count == 0:
        print(f"\n[5] No observation exceeded the threshold.")
        return
    print(f"\n[5] Flagged: {result.count} at positions {result.positions.tolist()}, "
          f"values {np.sort(result.values[result.positions]).tolist()}")


def parse_args() -> argparse.Namespace:
    """Parse and validate the command line options."""
    parser = argparse.ArgumentParser(
        description='Flag outliers with the Hampel identifier, built on the median and the MAD.')
    parser.add_argument('--input-csv', type=pathlib.Path, required=True,
                        help='CSV holding the sample; the column named by --column is tested')
    parser.add_argument('--column', type=str, default='value',
                        help='column of --input-csv to test (default: %(default)s)')
    parser.add_argument('--threshold', type=float, default=DEFAULT_THRESHOLD,
                        help='cut-off on the absolute modified z-score (default: %(default)s)')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    if not args.input_csv.is_file():
        parser.error(f"--input-csv is not a file: {args.input_csv}")
    if args.threshold <= 0.0:
        parser.error(f"--threshold must be positive, got {args.threshold}.")

    return args


def load_sample(input_csv: Union[pathlib.Path, None] = None, column: str = None) -> np.ndarray:
    """Read the column to test from a CSV."""
    frame = pd.read_csv(input_csv)
    if column not in frame.columns:
        raise ValueError(f"column '{column}' is absent from {input_csv}; available: {list(frame.columns)}")
    return frame[column].to_numpy(dtype=float)


if __name__ == '__main__':
    options = parse_args()

    sample = load_sample(input_csv=options.input_csv, column=options.column)
    report(result=hampel_test(data=sample, threshold=options.threshold))
