#!/usr/bin/env python3
"""Compute the +/- p95 relative error band around the 1:1 line of predicted vs. measured values.

The CSV holds one wafer per row with columns true_p1..true_pN (measured values y) and pred_p1..pred_pN (predicted
values x). All T = M * N pairs are pooled and the relative error e = (x - y) / y is formed. The script prints

    with raw data      p95 = 0.95 quantile of |e|, and the conformal rank ceil(0.95 (T + 1)) of the sorted |r|
    without raw data   normal, mean 0         p95 = z * RMSRE                 (z: 0.975 standard normal quantile)
                       normal, mean != 0      p95 solves Phi((p - mu)/sd) - Phi((-p - mu)/sd) = 0.95
                       distribution free      p95 <= RMSRE / sqrt(0.05)       (Markov inequality on e^2)

the share of pairs inside each band, the Shapiro-Wilk test of the pooled e, and RMSE, R2_all and RMSE / y_bar.

Requires scipy.
"""
__author__ = 'yRocket'
__version__ = "0.0.1+20261116"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import math
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats

__all__ = ['load_pairs', 'relative_error_band']

COVERAGE: float = 0.95


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{pathlib.Path(__file__).name} {__version__}\n"
                    'The +/- p95 relative error band around the 1:1 line of predicted vs. measured values.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--csv', type=pathlib.Path, help='wafer CSV with true_p*/pred_p* columns')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    if args.csv is None or not args.csv.is_file():
        parser.print_help()
        parser.error(f"--csv must be an existing file: {args.csv}")
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    X, Y = load_pairs(csv_path=ARGS.csv)
    for KEY, VALUE in relative_error_band(x=X, y=Y).items():
        print(f"{KEY:24s} {VALUE:.6f}" if isinstance(VALUE, float) else f"{KEY:24s} {VALUE}")
