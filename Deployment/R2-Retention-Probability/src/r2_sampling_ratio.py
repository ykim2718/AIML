"""Compute the R2 retention probability and the smallest measured wafer count that keeps it above a target.

A deposition thickness model predicts every wafer, and only m of N wafers are measured. The measured R2 uses the
prediction errors of the m wafers and the thickness variance known from past measurement:

    R2_m = 1 - s2_m / sigma2_y,    s2_m = mean of (y_measured - y_predicted)^2

With independent normal errors of mean 0 and the predicted model R2 R2_p standing in for the unknown true R2 R2_t, the
retention probability P(R2_m >= R2_min) is exact for every m >= 1:

    P_ret(m) = F_chi2(m; m (1 - R2_min) / (1 - R2_p))

and the wafer measurement count m* is the smallest m in 1..N with P_ret(m) >= gamma; r* = m* / N.

Changelog:
    0.0.0 - Initial release.
    0.1.0 - Replace the large-sample normal approximation with the chi-square retention probability, valid from m = 1.
"""

__author__ = 'yRocket'
__version__ = "0.1.1+20261108"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import math
import pathlib
import sys
from dataclasses import asdict, dataclass
from typing import Final, Optional

import numpy as np
import pandas as pd

__all__ = ['SamplingPlan', 'chi2_cdf', 'estimate_r2', 'retention_probability', 'required_measured', 'build_table']

OUTPUT_CSV_NAME: Final[str] = 'r2_sampling_ratio.csv'
SERIES_TOLERANCE: Final[float] = 1e-15
SERIES_MAX_TERMS: Final[int] = 100000


@dataclass(frozen=True)
class SamplingPlan:
    """The wafer measurement count and ratio for one condition. measured is None when no m <= N reaches the target."""
    total_wafers: int
    r2_model: float
    r2_min: float
    confidence: float
    measured: Optional[int]  # m*
    retention: Optional[float]  # P_ret(m*)
    ratio: Optional[float]  # r* = m* / N


def chi2_cdf(x: float = None, dof: int = None) -> float:
    """Cumulative distribution of the chi-square distribution with `dof` degrees of freedom at `x`.

    Uses the series of the regularized lower incomplete gamma function P(dof / 2, x / 2).
    """
    if dof < 1:
        raise ValueError(f"dof must be at least 1: {dof=}")
    if x <= 0.0:
        return 0.0
    a, t = dof / 2.0, x / 2.0
    term = total = 1.0 / a
    for n in range(1, SERIES_MAX_TERMS):
        term *= t / (a + n)
        total += term
        if term < SERIES_TOLERANCE * total:
            return math.exp(-t + a * math.log(t) - math.lgamma(a)) * total
    raise ArithmeticError(f"chi2_cdf series did not converge in {SERIES_MAX_TERMS} terms: {x=}, {dof=}")


def estimate_r2(measured: np.ndarray = None, predicted: np.ndarray = None, thickness_variance: float = None) -> float:
    """Measured R2 of the m measured wafers: 1 - mean squared prediction error / thickness variance."""
    y_m = np.asarray(measured, dtype=float)
    y_c = np.asarray(predicted, dtype=float)
    if y_m.ndim != 1 or y_m.shape != y_c.shape:
        raise ValueError(f"measured and predicted must be 1-D arrays of the same length: {y_m.shape=}, {y_c.shape=}")
    if y_m.size < 1:
        raise ValueError("at least one measured wafer is needed")
    if not thickness_variance > 0.0:
        raise ValueError(f"thickness_variance must be positive: {thickness_variance=}")
    return 1.0 - float(np.mean((y_m - y_c) ** 2)) / thickness_variance


def retention_probability(r2_model: float = None, r2_min: float = None, measured: int = None) -> float:
    """P(R2_m >= r2_min) with `measured` wafers when the model R2 is r2_model."""
    if not 0.0 < r2_model < 1.0:
        raise ValueError(f"r2_model must lie in (0, 1): {r2_model=}")
    if not r2_min < 1.0:
        raise ValueError(f"r2_min must be below 1: {r2_min=}")
    if measured < 1:
        raise ValueError(f"measured must be at least 1: {measured=}")
    return chi2_cdf(x=measured * (1.0 - r2_min) / (1.0 - r2_model), dof=measured)


def required_measured(
        r2_model: float = None,
        r2_min: float = None,
        confidence: float = None,
        total_wafers: int = None,
) -> SamplingPlan:
    """Smallest m in 1..total_wafers with retention_probability >= confidence; measured is None if none reaches it."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0, 1): {confidence=}")
    if total_wafers < 1:
        raise ValueError(f"total_wafers must be at least 1: {total_wafers=}")
    for m in range(1, total_wafers + 1):
        p = retention_probability(r2_model=r2_model, r2_min=r2_min, measured=m)
        if p >= confidence:
            return SamplingPlan(total_wafers=total_wafers, r2_model=r2_model, r2_min=r2_min,
                                confidence=confidence, measured=m, retention=p, ratio=m / total_wafers)
    return SamplingPlan(total_wafers=total_wafers, r2_model=r2_model, r2_min=r2_min,
                        confidence=confidence, measured=None, retention=None, ratio=None)


def build_table(
        r2_models: list = None,
        margins: list = None,
        r2_mins: list = None,
        confidences: list = None,
        total_wafers: list = None,
) -> pd.DataFrame:
    """Plans for every combination of the given conditions.

    Pass exactly one of `margins` (r2_min = r2_model - margin) or `r2_mins` (fixed floors).
    Conditions that no m <= N can reach keep empty measured, retention and ratio, and are counted on stderr.

    Returns:
        pd.DataFrame with a RangeIndex and columns
        total_wafers, r2_model, r2_min, confidence, measured, retention, ratio.
    """
    if (margins is None) == (r2_mins is None):
        raise ValueError("pass exactly one of margins or r2_mins")
    plans = []
    for n_total in total_wafers:
        for confidence in confidences:
            for r2_model in r2_models:
                floors = [round(r2_model - m, 10) for m in margins] if margins is not None else r2_mins
                for r2_min in floors:
                    plans.append(asdict(required_measured(
                        r2_model=r2_model, r2_min=r2_min, confidence=confidence, total_wafers=n_total)))
    if not plans:
        raise ValueError("no condition to compute")
    table = pd.DataFrame(plans)
    unreached = int(table['measured'].isna().sum())
    if unreached:
        print(f"{unreached}/{len(table)} conditions do not reach the target retention within N wafers",
              file=sys.stderr)
    return table


def parse_args() -> argparse.Namespace:
    script = pathlib.Path(__file__).name
    parser = argparse.ArgumentParser(
        prog=script,
        usage=f"{script} {__version__}\n  python {script} [options]",
        description="R2 retention probability and the smallest measured wafer count that keeps it above a target.",
        epilog=f"example: python {script} --r2-model 0.85 0.90 0.95 --margin 0.03 0.05 0.10 --total-wafers 100",
    )
    parser.add_argument('-v', '--version', action='version', version=f"{script} {__version__}")
    parser.add_argument('--r2-model', type=float, nargs='+', required=True,
                        help="model R2 prediction of each model, e.g. 0.85 0.90 0.95")
    floor = parser.add_mutually_exclusive_group(required=True)
    floor.add_argument('--margin', type=float, nargs='+',
                       help="allowed drop below the model R2; r2_min = r2_model - margin")
    floor.add_argument('--r2-min', type=float, nargs='+', help="fixed R2 floors to keep")
    parser.add_argument('--confidence', type=float, nargs='+', default=[0.95],
                        help="target retention probability gamma (default: 0.95)")
    parser.add_argument('--total-wafers', type=int, nargs='+', required=True,
                        help="wafers per lot or period, N")
    parser.add_argument('--output-folder', type=pathlib.Path, default=None,
                        help=f"if given, write {OUTPUT_CSV_NAME} into this folder")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()

    if any(not 0.0 < r < 1.0 for r in args.r2_model):
        parser.error(f"--r2-model values must lie in (0, 1): {args.r2_model}")
    if args.margin is not None and any(m <= 0.0 for m in args.margin):
        parser.error(f"--margin values must be positive: {args.margin}")
    if args.r2_min is not None and any(not r < 1.0 for r in args.r2_min):
        parser.error(f"--r2-min values must be below 1: {args.r2_min}")
    if any(not 0.0 < c < 1.0 for c in args.confidence):
        parser.error(f"--confidence values must lie in (0, 1): {args.confidence}")
    if any(n < 1 for n in args.total_wafers):
        parser.error(f"--total-wafers values must be at least 1: {args.total_wafers}")
    if args.output_folder is not None and args.output_folder.exists() and not args.output_folder.is_dir():
        parser.error(f"--output-folder is not a folder: {args.output_folder}")
    return args


if __name__ == '__main__':
    args = parse_args()
    table = build_table(
        r2_models=args.r2_model,
        margins=args.margin,
        r2_mins=args.r2_min,
        confidences=args.confidence,
        total_wafers=args.total_wafers,
    )
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.output_folder is not None:
        args.output_folder.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_folder / OUTPUT_CSV_NAME
        table.to_csv(csv_path, index=False)
        print(f"wrote {len(table)} rows to {csv_path}")
