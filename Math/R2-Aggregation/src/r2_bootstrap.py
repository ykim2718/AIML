#!/usr/bin/env python3
"""Estimate the sampling distribution of R2_all, R2_avgw, R2_avgn and R2_minn by a wafer bootstrap.

Each bootstrap sample draws M wafers with replacement and keeps all N location values of a drawn wafer together,
because the location values of one wafer are correlated. For each sample the four R2 are recomputed with the
definitions of the document, and the probability of meeting each target and the alpha quantile are reported.
"""
__author__ = 'yRocket'
__version__ = "0.0.0+20261114"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import pathlib
import sys

import numpy as np

from r2_three_views import load

__all__ = ['four_r2', 'bootstrap', 'summarize']

TARGETS: dict = {'R2_all': 0.90, 'R2_avgw': 0.95, 'R2_avgn': 0.85, 'R2_minn': 0.80}
N_BOOT: int = 2000
SEED: int = 0
ALPHA: float = 0.05


def four_r2(y: np.ndarray = None, y_hat: np.ndarray = None, y_hat_w: np.ndarray = None) -> dict:
    """Return R2_all, R2_avgw, R2_avgn and R2_minn for an M x N measured array and its predictions."""
    if y is None or y_hat is None or y_hat_w is None:
        raise ValueError('y, y_hat and y_hat_w are all required.')
    if y.shape != y_hat.shape or y.ndim != 2 or y_hat_w.shape != (y.shape[0],):
        raise ValueError(f"shape mismatch: {y.shape=}, {y_hat.shape=}, {y_hat_w.shape=}")
    y_i = y.mean(axis=1)
    sst_all = ((y - y.mean()) ** 2).sum()
    sst_avgw = ((y_i - y.mean()) ** 2).sum()
    sst_j = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    if sst_all <= 0 or sst_avgw <= 0 or np.any(sst_j <= 0):
        raise ValueError(f"an SST is not positive: {sst_all=}, {sst_avgw=}, {sst_j.min()=}")
    r2_j = 1 - ((y - y_hat) ** 2).sum(axis=0) / sst_j
    return {'R2_all': 1 - ((y - y_hat) ** 2).sum() / sst_all, 'R2_avgw': 1 - ((y_i - y_hat_w) ** 2).sum() / sst_avgw,
            'R2_avgn': r2_j.mean(), 'R2_minn': r2_j.min()}


def bootstrap(y: np.ndarray = None, y_hat: np.ndarray = None, y_hat_w: np.ndarray = None, n_boot: int = N_BOOT,
              seed: int = SEED) -> dict:
    """Return {name: array of n_boot bootstrap values} for the four R2, resampling whole wafers."""
    if n_boot < 1:
        raise ValueError(f"n_boot must be positive: {n_boot=}")
    rng = np.random.default_rng(seed)
    m = y.shape[0]
    draws = {name: np.empty(n_boot) for name in TARGETS}
    for b in range(n_boot):
        idx = rng.integers(0, m, size=m)
        for name, value in four_r2(y[idx], y_hat[idx], y_hat_w[idx]).items():
            draws[name][b] = value
    return draws


def summarize(draws: dict = None, targets: dict = TARGETS, alpha: float = ALPHA) -> dict:
    """Return {name: (p_meet, alpha_quantile)} and the joint probability under key 'joint'."""
    if draws is None:
        raise ValueError('draws is required.')
    out = {name: ((draws[name] >= t).mean(), np.quantile(draws[name], alpha)) for name, t in targets.items()}
    out['joint'] = (np.all([draws[name] >= t for name, t in targets.items()], axis=0).mean(), np.nan)
    return out


parser = argparse.ArgumentParser(prog=pathlib.Path(__file__).name,
                                 description=f"{pathlib.Path(__file__).name} {__version__}: "
                                             f"wafer bootstrap of the four R2 against the targets.")
parser.add_argument('-v', '--version', action='version', version=f"%(prog)s {__version__}")
parser.add_argument('csv', type=pathlib.Path, nargs='?',
                    default=pathlib.Path(__file__).resolve().parent.parent / 'data' / 'u0_actuals_and_predictions.csv',
                    help='input CSV')

if __name__ == '__main__':
    args = parser.parse_args()
    y, y_hat, y_hat_w, _ = load(args.csv)
    cases = {'Original': y_hat, 'Offset correction': y_hat + (y - y_hat).mean(axis=0)}
    for label, pred in cases.items():
        point = four_r2(y, pred, y_hat_w)
        stats = summarize(bootstrap(y, pred, y_hat_w))
        for name in TARGETS:
            p, low = stats[name]
            print(f"{label:18s} {name:8s} R2={point[name]:.4f} q{ALPHA:.2f}={low:.4f} P(>={TARGETS[name]})={p:.4f}")
        print(f"{label:18s} joint    P(all four)={stats['joint'][0]:.4f}")
    sys.exit(0)
