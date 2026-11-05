"""Find the smallest ratio of measured wafers that keeps the model R2 above a floor with a given probability.

A deposition thickness model predicts every wafer, and only n of N wafers are measured. R2 computed on the n
measured wafers varies from sample to sample, with the large-sample variance (Olkin-Finn, delta method) times
the finite population correction:

    Var(R2_hat) ~= 4 R2 (1 - R2)^2 / n * (N - n) / (N - 1)

Requiring R2 - z * sqrt(Var) >= R2_min gives

    n0 = (2 z sqrt(R2) (1 - R2) / (R2 - R2_min))^2
    n* = max(ceil(n0 / (1 + (n0 - 1) / N)), min_measured)
    r* = n* / N

Changelog:
    0.0.0 - Initial release.
"""

__author__ = 'yRocket'
__version__ = "0.0.0+20261105"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import math
import pathlib
import sys
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Final

import numpy as np
import pandas as pd

__all__ = ['SamplingPlan', 'estimate_r2', 'r2_variance', 'required_measured', 'build_table']

OUTPUT_CSV_NAME: Final[str] = 'r2_sampling_ratio.csv'


@dataclass(frozen=True)
class SamplingPlan:
    """The measured wafer count and ratio for one condition."""
    total_wafers: int
    r2_model: float
    r2_min: float
    confidence: float
    n0: float  # sample size before the finite population correction
    measured: int  # n*, after the correction and the floor
    floored: bool  # True when min_measured, not the formula, set the count
    ratio: float  # r* = n* / N


def estimate_r2(measured: np.ndarray = None, predicted: np.ndarray = None) -> float:
    """R2 of the predictions on the measured wafers: 1 - SSE / SST."""
    y = np.asarray(measured, dtype=float)
    y_hat = np.asarray(predicted, dtype=float)
    if y.ndim != 1 or y.shape != y_hat.shape:
        raise ValueError(f"measured and predicted must be 1-D arrays of the same length: {y.shape=}, {y_hat.shape=}")
    if y.size < 2:
        raise ValueError(f"R2 needs at least 2 measured wafers: {y.size=}")
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst == 0.0:
        raise ValueError("all measured thicknesses are equal, so SST is 0 and R2 is undefined")
    return 1.0 - float(np.sum((y - y_hat) ** 2)) / sst


def r2_variance(r2_model: float = None, measured: int = None, total_wafers: int = None) -> float:
    """Large-sample variance of R2 on `measured` of `total_wafers` wafers, with the finite population correction."""
    _check_r2(name='r2_model', value=r2_model)
    if not 1 <= measured <= total_wafers or total_wafers < 2:
        raise ValueError(f"need 1 <= measured <= total_wafers and total_wafers >= 2: {measured=}, {total_wafers=}")
    fpc = (total_wafers - measured) / (total_wafers - 1)
    return 4.0 * r2_model * (1.0 - r2_model) ** 2 / measured * fpc


def required_measured(
        r2_model: float = None,
        r2_min: float = None,
        confidence: float = None,
        total_wafers: int = None,
        min_measured: int = None,
) -> SamplingPlan:
    """Smallest measured wafer count n* that keeps P(R2_hat >= r2_min) >= confidence."""
    _check_r2(name='r2_model', value=r2_model)
    _check_r2(name='r2_min', value=r2_min)
    if r2_min >= r2_model:
        raise ValueError(f"r2_min must be below r2_model, otherwise no sample size can hold it: {r2_min=}, {r2_model=}")
    if not 0.5 < confidence < 1.0:
        raise ValueError(f"confidence must lie in (0.5, 1): {confidence=}")
    if not 2 <= min_measured <= total_wafers:
        raise ValueError(f"need 2 <= min_measured <= total_wafers: {min_measured=}, {total_wafers=}")

    z = NormalDist().inv_cdf(confidence)
    n0 = (2.0 * z * math.sqrt(r2_model) * (1.0 - r2_model) / (r2_model - r2_min)) ** 2
    corrected = math.ceil(n0 / (1.0 + (n0 - 1.0) / total_wafers))
    measured = max(corrected, min_measured)
    return SamplingPlan(
        total_wafers=total_wafers,
        r2_model=r2_model,
        r2_min=r2_min,
        confidence=confidence,
        n0=n0,
        measured=measured,
        floored=corrected < min_measured,
        ratio=measured / total_wafers,
    )


def build_table(
        r2_models: list = None,
        margins: list = None,
        r2_mins: list = None,
        confidences: list = None,
        total_wafers: list = None,
        min_measured: int = None,
) -> pd.DataFrame:
    """Plans for every combination of the given conditions.

    Pass exactly one of `margins` (r2_min = r2_model - margin) or `r2_mins` (fixed floors).
    Combinations whose r2_min is not below r2_model are skipped and counted on stderr.

    Returns:
        pd.DataFrame with a RangeIndex and columns
        total_wafers, r2_model, r2_min, confidence, n0, measured, floored, ratio.
    """
    if (margins is None) == (r2_mins is None):
        raise ValueError("pass exactly one of margins or r2_mins")
    plans, skipped = [], 0
    for n_total in total_wafers:
        for confidence in confidences:
            for r2_model in r2_models:
                floors = [round(r2_model - m, 10) for m in margins] if margins is not None else r2_mins
                for r2_min in floors:
                    if r2_min >= r2_model:
                        skipped += 1
                        continue
                    plans.append(asdict(required_measured(
                        r2_model=r2_model, r2_min=r2_min, confidence=confidence,
                        total_wafers=n_total, min_measured=min_measured)))
    if skipped:
        print(f"skipped {skipped} combinations whose r2_min is not below r2_model", file=sys.stderr)
    if not plans:
        raise ValueError("no valid combination: every r2_min is at or above its r2_model")
    return pd.DataFrame(plans)


def _check_r2(name: str = None, value: float = None) -> None:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie in (0, 1): {value=}")


def parse_args() -> argparse.Namespace:
    script = pathlib.Path(__file__).name
    parser = argparse.ArgumentParser(
        prog=script,
        usage=f"{script} {__version__}\n  python {script} [options]",
        description="Smallest measured wafer ratio that keeps the model R2 above a floor with a given probability.",
        epilog=f"example: python {script} --r2-model 0.85 0.90 0.95 --margin 0.03 0.05 0.10 --total-wafers 25 100",
    )
    parser.add_argument('-v', '--version', action='version', version=f"{script} {__version__}")
    parser.add_argument('--r2-model', type=float, nargs='+', required=True,
                        help="expected R2 of each model, e.g. 0.85 0.90 0.95")
    floor = parser.add_mutually_exclusive_group(required=True)
    floor.add_argument('--margin', type=float, nargs='+',
                       help="allowed drop below the model R2; r2_min = r2_model - margin")
    floor.add_argument('--r2-min', type=float, nargs='+', help="fixed R2 floors to keep")
    parser.add_argument('--confidence', type=float, nargs='+', default=[0.95],
                        help="probability of keeping R2 >= r2_min (default: 0.95)")
    parser.add_argument('--total-wafers', type=int, nargs='+', required=True,
                        help="wafers per lot or period, N")
    parser.add_argument('--min-measured', type=int, default=10,
                        help="lower bound on n so that the large-sample approximation holds (default: 10)")
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
    if args.r2_min is not None and any(not 0.0 < r < 1.0 for r in args.r2_min):
        parser.error(f"--r2-min values must lie in (0, 1): {args.r2_min}")
    if any(not 0.5 < c < 1.0 for c in args.confidence):
        parser.error(f"--confidence values must lie in (0.5, 1): {args.confidence}")
    if args.min_measured < 2:
        parser.error(f"--min-measured must be at least 2: {args.min_measured}")
    if any(n < args.min_measured for n in args.total_wafers):
        parser.error(f"--total-wafers values must be >= --min-measured {args.min_measured}: {args.total_wafers}")
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
        min_measured=args.min_measured,
    )
    print(table.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if args.output_folder is not None:
        args.output_folder.mkdir(parents=True, exist_ok=True)
        csv_path = args.output_folder / OUTPUT_CSV_NAME
        table.to_csv(csv_path, index=False)
        print(f"wrote {len(table)} rows to {csv_path}")
