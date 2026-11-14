#!/usr/bin/env python3
"""Estimate the sampling distribution of R2_all, R2_avgw, R2_avgn and R2_minn by a wafer bootstrap.

Each bootstrap sample draws M wafers with replacement and keeps all N location values of a drawn wafer together,
because the location values of one wafer are correlated. For each sample the four R2 are recomputed with the
definitions of the document, and the probability of meeting each target and the alpha quantile are reported.
The targets come from the four target setting methods of the document: conversion of T_all, a common SSE reduction
ratio, removal of the location mean error (offset) and the bootstrap lower quantile.

Changelog:
- 0.1.0: add method_targets for the four target setting methods.
"""
__author__ = 'yRocket'
__version__ = "0.1.0+20261114"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import pathlib
import sys

import numpy as np

from r2_three_views import load

__all__ = ['four_r2', 'bootstrap', 'summarize', 'method_targets']

NAMES: tuple = ('R2_all', 'R2_avgw', 'R2_avgn', 'R2_minn')
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
    draws = {name: np.empty(n_boot) for name in NAMES}
    for b in range(n_boot):
        idx = rng.integers(0, m, size=m)
        for name, value in four_r2(y[idx], y_hat[idx], y_hat_w[idx]).items():
            draws[name][b] = value
    return draws


def summarize(draws: dict = None, targets: dict = None, alpha: float = ALPHA) -> dict:
    """Return {name: (p_meet, alpha_quantile)} and the joint probability under key 'joint'."""
    if draws is None or targets is None:
        raise ValueError('draws and targets are both required.')
    if set(targets) != set(NAMES):
        raise ValueError(f"targets must have exactly the keys {NAMES}: {sorted(targets)=}")
    out = {name: ((draws[name] >= t).mean(), np.quantile(draws[name], alpha)) for name, t in targets.items()}
    out['joint'] = (np.all([draws[name] >= t for name, t in targets.items()], axis=0).mean(), np.nan)
    return out


def method_targets(y: np.ndarray = None, y_hat: np.ndarray = None, y_hat_w: np.ndarray = None,
                   alpha: float = ALPHA, n_boot: int = N_BOOT, seed: int = SEED) -> tuple:
    """Return ({method: {name: target}}, eta, offset prediction) for the four target setting methods.

    The input of methods 1 to 3 is the improvement that removes the location mean error (offset): eta is the share
    of the offset component in E, and T_all of method 1 is the R2_all that this eta gives.
    """
    current = four_r2(y, y_hat, y_hat_w)
    e = y - y_hat
    sst_j = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    q = (e ** 2).sum() / sst_j.sum()
    lam = 1 - sst_j.sum() / ((y - y.mean()) ** 2).sum()
    kappa = (1 - current['R2_avgw']) / q
    delta = 1 - current['R2_avgn'] - q
    gamma_n = (1 - current['R2_minn']) / (1 - current['R2_avgn'])
    eta = y.shape[0] * (e.mean(axis=0) ** 2).sum() / (e ** 2).sum()
    t_all = 1 - (1 - eta) * (1 - current['R2_all'])
    t_avgn = 1 - (1 - t_all) / (1 - lam) - delta
    offset_pred = y_hat + e.mean(axis=0)
    targets = {
        'Method 1': {'R2_all': t_all, 'R2_avgw': 1 - kappa / (1 - lam) * (1 - t_all), 'R2_avgn': t_avgn,
                     'R2_minn': 1 - gamma_n * (1 - t_avgn)},
        'Method 2': {name: 1 - (1 - eta) * (1 - value) for name, value in current.items()},
        'Method 3': four_r2(y, offset_pred, y_hat_w),
    }
    for label, pred in (('Method 4, current', y_hat), ('Method 4, offset removed', offset_pred)):
        draws = bootstrap(y, pred, y_hat_w, n_boot, seed)
        targets[label] = {name: np.quantile(draws[name], alpha) for name in NAMES}
    return targets, eta, offset_pred


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
    targets, eta, offset_pred = method_targets(y, y_hat, y_hat_w)
    print(f"eta (offset share of E) = {eta:.4f}")
    for label, t in targets.items():
        print(f"{label:26s} " + ' '.join(f"{name}={t[name]:.4f}" for name in NAMES))
    for label, pred in (('Current', y_hat), ('Offset removed', offset_pred)):
        draws = bootstrap(y, pred, y_hat_w)
        point = four_r2(y, pred, y_hat_w)
        print(f"{label}: " + ' '.join(f"{name}={point[name]:.4f}" for name in NAMES))
        for method, t in targets.items():
            stats = summarize(draws, t)
            print(f"  P vs {method:24s} " + ' '.join(f"{name}={stats[name][0]:.4f}" for name in NAMES)
                  + f" joint={stats['joint'][0]:.4f}")
    sys.exit(0)
