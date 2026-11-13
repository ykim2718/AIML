#!/usr/bin/env python3
"""Compute R2_all, R2_avgw, R2_avgn with lambda, kappa and the SST weights from a wafer CSV, and chart the three R2.

The CSV holds one wafer per row with columns true_p1..true_pN, pred_p1..pred_pN and pred_avg (the direct prediction of
the wafer average). The wafer average of the measured values is recomputed from true_p1..true_pN, as in the definition.
"""
__author__ = 'yRocket'
__version__ = "0.0.2+20261113"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

POINT_COLOR: str = '#2a78b8'
MEAN_COLOR: str = '#555555'
WEIGHTED_COLOR: str = '#c2571a'
DPI: int = 300
R2_J_RANGE: tuple = (0.4, 1.0)    # y-axis range of panel (c)


def load(csv_path: pathlib.Path = None) -> tuple:
    """Return (y, y_hat, y_hat_w, true_avg) arrays read from csv_path."""
    if csv_path is None:
        raise ValueError('csv_path is required.')
    frame = pd.read_csv(csv_path)
    true_cols = sorted([c for c in frame.columns if c.startswith('true_p')], key=lambda c: int(c[6:]))
    pred_cols = sorted([c for c in frame.columns if c.startswith('pred_p')], key=lambda c: int(c[6:]))
    if not true_cols or [c[5:] for c in true_cols] != [c[5:] for c in pred_cols] or 'pred_avg' not in frame:
        raise ValueError(f"need matching true_p*/pred_p* columns and pred_avg: {list(frame.columns)}")
    y, y_hat = frame[true_cols].to_numpy(float), frame[pred_cols].to_numpy(float)
    if np.isnan(y).any() or np.isnan(y_hat).any() or frame['pred_avg'].isna().any():
        raise ValueError('the CSV has missing values.')
    return y, y_hat, frame['pred_avg'].to_numpy(float), frame['true_avg'].to_numpy(float)


def quantities(y: np.ndarray = None, y_hat: np.ndarray = None, y_hat_w: np.ndarray = None) -> dict:
    """Return the sums of squares, the R2 values, lambda, kappa, r_y and omega_j as defined in the document."""
    m, n = y.shape
    y_bar, y_i, y_j = y.mean(), y.mean(axis=1), y.mean(axis=0)
    e, e_w = y - y_hat, y_i - y_hat_w
    sst_all, sse_all = ((y - y_bar) ** 2).sum(), (e ** 2).sum()
    sst_avgw, sse_avgw = ((y_i - y_bar) ** 2).sum(), (e_w ** 2).sum()
    sst_j, sse_j = ((y - y_j) ** 2).sum(axis=0), (e ** 2).sum(axis=0)
    r2_j = 1 - sse_j / sst_j
    dev = y - y_j
    cross = (dev.sum(axis=1) ** 2).sum() - (dev ** 2).sum()
    q = {'M': m, 'N': n, 'SST_all': sst_all, 'SSE_all': sse_all, 'SST_avgw': sst_avgw, 'SSE_avgw': sse_avgw,
         'R2_all': 1 - sse_all / sst_all, 'R2_avgw': 1 - sse_avgw / sst_avgw, 'R2_j': r2_j, 'R2_avgn': r2_j.mean(),
         'R2_wavgn': 1 - sse_j.sum() / sst_j.sum(), 'lambda': m * ((y_j - y_bar) ** 2).sum() / sst_all,
         'kappa': (sse_avgw / sst_avgw) * (sst_j.sum() / sse_j.sum()), 'r_y': cross / ((n - 1) * sst_j.sum()),
         'omega_j': sst_j / sst_j.sum()}
    identity_all = 1 - (1 - q['lambda']) * (1 - q['R2_wavgn'])
    identity_avgw = 1 - q['kappa'] * (1 - q['R2_wavgn'])
    if not (np.isclose(identity_all, q['R2_all']) and np.isclose(identity_avgw, q['R2_avgw'])):
        raise RuntimeError(f"identities fail: {identity_all=}, {q['R2_all']=}, {identity_avgw=}, {q['R2_avgw']=}")
    return q


def draw(y: np.ndarray = None, y_hat: np.ndarray = None, y_hat_w: np.ndarray = None, q: dict = None,
         out_path: pathlib.Path = None) -> pathlib.Path:
    """Draw the three R2 views side by side and save the figure."""
    if out_path is None:
        raise ValueError('out_path is required.')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), gridspec_kw={'width_ratios': [1, 1, 1.25]})
    fig.subplots_adjust(left=0.05, right=0.99, top=0.88, bottom=0.2, wspace=0.28)
    views = ((axes[0], y.ravel(), y_hat.ravel(), q['R2_all'], 'R$^2_{all}$', f"T = {y.size} measured values"),
             (axes[1], y.mean(axis=1), y_hat_w, q['R2_avgw'], 'R$^2_{avgw}$', f"M = {q['M']} wafer averages"))
    for ax, actual, pred, r2, name, what in views:
        lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
        pad = 0.04 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MEAN_COLOR, linewidth=1, linestyle='--')
        ax.scatter(actual, pred, s=14 if actual.size > 100 else 28, color=POINT_COLOR, alpha=0.7, edgecolors='none')
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect('equal')
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        ax.set_title(f"{name} = {r2:.4f}   ({what})", fontsize=11)
        ax.grid(color='0.9', linewidth=0.8)
    ax = axes[2]
    loc = np.arange(1, q['N'] + 1)
    ax.vlines(loc, 0, q['R2_j'], color='0.8', linewidth=1)
    ax.scatter(loc, q['R2_j'], s=10 + 400 * q['omega_j'], color=POINT_COLOR, zorder=3, edgecolors='white')
    ax.axhline(q['R2_avgn'], color=MEAN_COLOR, linewidth=1.5, label=f"R$^2_{{avgn}}$ = {q['R2_avgn']:.4f}")
    ax.axhline(q['R2_wavgn'], color=WEIGHTED_COLOR, linewidth=1.5, linestyle='--',
               label=f"R$^2_{{wavgn}}$ = {q['R2_wavgn']:.4f}")
    ax.set_xticks(loc)
    ax.set_xlabel('Measurement location j (marker area $\\propto$ $\\omega_j$)')
    ax.set_ylabel('R$^2_j$')
    ax.set_ylim(*R2_J_RANGE)
    ax.set_title(f"R$^2_{{avgn}}$ = {q['R2_avgn']:.4f}   (mean of N = {q['N']} R$^2_j$)", fontsize=11)
    ax.legend(loc='lower right', frameon=False)
    ax.grid(axis='y', color='0.9', linewidth=0.8)
    fig.suptitle(f"$\\lambda$ = {q['lambda']:.4f},   $\\kappa$ = {q['kappa']:.4f},   1 - $\\lambda$ = {1 - q['lambda']:.4f},"
                 f"   $\\bar{{r}}_y$ = {q['r_y']:.4f}", fontsize=12)
    for ax, caption in zip(axes, ('(a) R2 over all measured values', '(b) R2 over wafer averages',
                                  '(c) R2 average over N locations')):
        box = ax.get_position()
        fig.text((box.x0 + box.x1) / 2, 0.03, caption, ha='center', fontsize=12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    return out_path


parser = argparse.ArgumentParser(prog=pathlib.Path(__file__).name,
                                 description=f"{pathlib.Path(__file__).name} {__version__}: chart the three R2 of a wafer CSV.")
parser.add_argument('-v', '--version', action='version', version=f"%(prog)s {__version__}")
parser.add_argument('csv', type=pathlib.Path, nargs='?', default=pathlib.Path(__file__).resolve().parent.parent / 'data' / 'u0_actuals_and_predictions.csv',
                    help='input CSV')
parser.add_argument('-o', '--out', type=pathlib.Path, default=pathlib.Path(__file__).resolve().parent.parent / 'r2-aggregation-ko_fig' / 'fig2-u0-r2.png',
                    help='output PNG path')

if __name__ == '__main__':
    args = parser.parse_args()
    y, y_hat, y_hat_w, true_avg = load(args.csv)
    bad = np.flatnonzero(~np.isclose(true_avg, y.mean(axis=1), atol=1e-3))
    if bad.size:
        print(f"WARNING: true_avg differs from the mean of true_p* in rows {bad.tolist()}; the mean is used.")
    q = quantities(y, y_hat, y_hat_w)
    for key in ('M', 'N', 'SST_all', 'SSE_all', 'SST_avgw', 'SSE_avgw', 'R2_all', 'R2_avgw', 'R2_avgn', 'R2_wavgn',
                'lambda', 'kappa', 'r_y'):
        print(f"{key:9s} = {q[key]:.4f}" if isinstance(q[key], float) else f"{key:9s} = {q[key]}")
    print('R2_j     =', np.round(q['R2_j'], 4).tolist())
    print('omega_j  =', np.round(q['omega_j'], 4).tolist())
    print(f"saved {draw(y, y_hat, y_hat_w, q, args.out)}")
    sys.exit(0)
