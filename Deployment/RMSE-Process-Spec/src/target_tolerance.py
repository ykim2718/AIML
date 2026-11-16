#!/usr/bin/env python3
"""Estimate the stdev and the tolerance of the measured thickness when the predicted thickness equals the target.

The CSV holds one wafer per row with columns true_p1..true_pN (measured values y) and pred_p1..pred_pN (predicted
values x at the same locations). All T = M * N pairs (x, y) are pooled and the measured value is regressed on the
predicted value:

    y = a + beta * x + error,   s = residual standard deviation (denominator T - 2)

At the target x0 the measured value has

    mean      mu0 = a + beta * x0
    stdev     s_p = s * sqrt(1 + 1/T + (x0 - x_bar)^2 / Sxx)
    tolerance h   = k * s_p              (interval mu0 +/- h)

The script also prints the error bias and stdev of e = y - x, and the measured values whose prediction lies within
--window of the target, as a check of the regression at the target.
"""
__author__ = 'yRocket'
__version__ = "0.0.0+20261116"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import pathlib
import sys
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

__all__ = ['TargetInput', 'load_pairs', 'target_tolerance']


@dataclass(frozen=True)
class TargetInput:
    """Settings. target: x0, k: coverage factor of the tolerance, window: half width of the local check around x0."""
    target: float = None
    k: float = 1.96
    window: float = 10.0

    def __post_init__(self):
        if self.target is None or self.k <= 0 or self.window <= 0:
            raise ValueError(f"need a target, k > 0 and window > 0: {self}")


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
    if len(y) < 3 or np.isnan(y).any() or np.isnan(x).any():
        raise ValueError(f"need three or more pairs without missing values: {len(y)=}")
    return x, y


def target_tolerance(x: np.ndarray = None, y: np.ndarray = None, setting: TargetInput = None) -> dict:
    """Return the regression of y on x and the mean, stdev and tolerance of y at x = target."""
    t = len(y)
    e = y - x
    x_bar = x.mean()
    sxx = ((x - x_bar) ** 2).sum()
    if sxx <= 0:
        raise ValueError('predicted values have no spread; the regression is undefined.')
    beta = ((x - x_bar) * (y - y.mean())).sum() / sxx
    a = y.mean() - beta * x_bar
    s = np.sqrt(((y - a - beta * x) ** 2).sum() / (t - 2))
    x0 = setting.target
    mu0 = a + beta * x0
    s_p = s * np.sqrt(1 + 1 / t + (x0 - x_bar) ** 2 / sxx)
    h = setting.k * s_p
    near = np.abs(x - x0) <= setting.window
    out = {'T': t, 'bias': e.mean(), 'sigma_e': e.std(), 'RMSE': np.sqrt((e ** 2).mean()),
           'a': a, 'beta': beta, 's': s, 'x_bar': x_bar, 'x_min': x.min(), 'x_max': x.max(),
           'target': x0, 'mu0': mu0, 's_p': s_p, 'k': setting.k, 'h': h, 'lower': mu0 - h, 'upper': mu0 + h,
           'coverage': 1 - 2 * (1 - NormalDist().cdf(setting.k)),
           'near_count': int(near.sum())}
    if near.sum() >= 2:
        out.update({'near_mean': y[near].mean(), 'near_stdev': y[near].std(ddof=1)})
    else:
        print(f"warning: {near.sum()} pairs within {setting.window} of the target; local check skipped",
              file=sys.stderr)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{pathlib.Path(__file__).name} {__version__}\n"
                    'Stdev and tolerance of the measured value when the prediction equals the target.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--csv', type=pathlib.Path, help='wafer CSV with true_p*/pred_p* columns')
    parser.add_argument('--target', type=float, help='target thickness x0')
    parser.add_argument('--k', type=float, default=TargetInput.k,
                        help='coverage factor of the tolerance (default 1.96)')
    parser.add_argument('--window', type=float, default=TargetInput.window,
                        help='half width around the target for the local check (default 10)')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    if args.csv is None or not args.csv.is_file():
        parser.print_help()
        parser.error(f"--csv must be an existing file: {args.csv}")
    if args.target is None:
        parser.error('--target is required.')
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    X, Y = load_pairs(csv_path=ARGS.csv)
    RESULT = target_tolerance(x=X, y=Y, setting=TargetInput(target=ARGS.target, k=ARGS.k, window=ARGS.window))
    for KEY, VALUE in RESULT.items():
        print(f"{KEY:12s} {VALUE:.4f}" if isinstance(VALUE, float) else f"{KEY:12s} {VALUE}")
