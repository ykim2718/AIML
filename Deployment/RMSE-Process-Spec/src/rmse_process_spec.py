#!/usr/bin/env python3
"""Compute the prediction RMSE from a wafer CSV and derive the thickness spec and the wafer uniformity spec from it.

The CSV holds one wafer per row with columns true_p1..true_pN (measured values) and pred_p1..pred_pN (predicted
values at the same locations). Two quantities are judged with the prediction:

    thickness   x = y_ij                       two-sided spec  target +/- h   (target: --target, default y_bar)
    uniformity  x = U_i = s_i / ybar_i         one-sided spec  U_bar + h      (s_i: within-wafer std, ddof=1)

For each quantity the tolerance h is the larger of a process requirement and a prediction requirement:

    h_process    = 3 * cp_min * sigma + |mean - target|   (Cpk or Cpu of the measured values >= cp_min)
    h_prediction = n_sigma * RMSE                         (at least n_sigma RMSE inside the tolerance h)
    h            = max(h_process, h_prediction)

and the acceptance limit of a predicted value is the spec moved inward by the guard band g = guard_k * RMSE.
The script prints both quantities for the current prediction and for the prediction whose location mean error
(offset) is removed, and writes the per-wafer uniformity values to a CSV.

Changelog:
    0.1.0 - Center the thickness spec on a given target and use Cpk for the process requirement.
    0.2.0 - Replace --pt-max with --n-sigma, the least number of RMSE inside the tolerance.
"""
__author__ = 'yRocket'
__version__ = "0.2.0+20261116"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import pathlib
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ['SpecInput', 'load', 'uniformity', 'spec', 'build_tables']

UNIFORMITY_CSV_NAME: str = 'uniformity_by_wafer.csv'
PERCENT: float = 100.0


@dataclass(frozen=True)
class SpecInput:
    """Settings of the spec rule. cp_min: least Cpk or Cpu, n_sigma: least RMSE count in h, guard_k: guard in RMSE."""
    cp_min: float = 1.33
    n_sigma: float = 10.0
    guard_k: float = 1.645

    def __post_init__(self):
        if self.cp_min <= 0 or self.n_sigma <= 0 or self.guard_k < 0:
            raise ValueError(f"need cp_min > 0, n_sigma > 0, guard_k >= 0: {self}")


def load(csv_path: pathlib.Path = None) -> tuple:
    """Return (wafer keys, y, y_hat) with y and y_hat of shape (M, N) read from csv_path."""
    if csv_path is None:
        raise ValueError('csv_path is required.')
    frame = pd.read_csv(csv_path, encoding='utf-8-sig')
    true_cols = sorted([c for c in frame.columns if c.startswith('true_p')], key=lambda c: int(c[6:]))
    pred_cols = sorted([c for c in frame.columns if c.startswith('pred_p')], key=lambda c: int(c[6:]))
    if len(true_cols) < 2 or [c[6:] for c in true_cols] != [c[6:] for c in pred_cols]:
        raise ValueError(f"need matching true_p*/pred_p* columns, at least two: {list(frame.columns)}")
    y, y_hat = frame[true_cols].to_numpy(float), frame[pred_cols].to_numpy(float)
    if len(y) < 2 or np.isnan(y).any() or np.isnan(y_hat).any():
        raise ValueError(f"need two or more wafers without missing values: {y.shape=}")
    return frame.iloc[:, 0].to_numpy(), y, y_hat


def uniformity(values: np.ndarray = None) -> np.ndarray:
    """Return the wafer uniformity 1 sigma / mean of each row, with the sample std over the N locations (ddof=1)."""
    means = values.mean(axis=1)
    if (means <= 0).any():
        raise ValueError('wafer means must be positive to define the uniformity.')
    return values.std(axis=1, ddof=1) / means


def spec(target: float = None, mean: float = None, sigma: float = None, rmse: float = None,
         setting: SpecInput = None, two_sided: bool = None) -> dict:
    """Return the tolerances, the spec limits and the acceptance limits for one quantity centered on target."""
    h_process = 3 * setting.cp_min * sigma + abs(mean - target)
    h_prediction = setting.n_sigma * rmse
    h = max(h_process, h_prediction)
    g = setting.guard_k * rmse
    out = {'target': target, 'mean': mean, 'sigma': sigma, 'RMSE': rmse, 'RMSE/sigma': rmse / sigma,
           'h_process': h_process, 'h_prediction': h_prediction, 'h': h,
           'governed_by': 'prediction' if h_prediction > h_process else 'process', 'USL': target + h,
           'guard_band': g, 'accept_tolerance': h - g, 'accept_upper': target + h - g}
    if two_sided:
        out.update({'LSL': target - h, 'accept_lower': target - h + g})
    return out


def build_tables(y: np.ndarray = None, y_hat: np.ndarray = None, setting: SpecInput = None,
                 target: float = None) -> tuple:
    """Return (thickness table, uniformity table, per-wafer frame) with the thickness spec centered on target.

    Both tables have index name 'quantity' and columns 'current' and 'offset_removed'. The per-wafer frame has columns
    'U', 'U_hat' and 'U_hat_offset_removed' in percent.
    """
    offset = (y - y_hat).mean(axis=0)
    predictions = {'current': y_hat, 'offset_removed': y_hat + offset}
    u = uniformity(values=y)
    sigma_y = np.sqrt(((y - y.mean()) ** 2).mean())
    sigma_u = np.sqrt(((u - u.mean()) ** 2).mean())
    thickness, uniform, per_wafer = {}, {}, {'U': u * PERCENT}
    for name, pred in predictions.items():
        rmse = np.sqrt(((y - pred) ** 2).mean())
        thickness[name] = spec(target=target, mean=y.mean(), sigma=sigma_y, rmse=rmse, setting=setting, two_sided=True)
        thickness[name]['R2_all'] = 1 - (rmse / sigma_y) ** 2
        u_hat = uniformity(values=pred)
        rmse_u = np.sqrt(((u - u_hat) ** 2).mean())
        uniform[name] = {k: (v * PERCENT if isinstance(v, float) and k != 'RMSE/sigma' else v)
                         for k, v in spec(target=u.mean(), mean=u.mean(), sigma=sigma_u, rmse=rmse_u, setting=setting,
                                          two_sided=False).items()}
        uniform[name]['U_max'] = u.max() * PERCENT
        per_wafer['U_hat' if name == 'current' else 'U_hat_offset_removed'] = u_hat * PERCENT
    frames = [pd.DataFrame(t).rename_axis('quantity') for t in (thickness, uniform)]
    return frames[0], frames[1], pd.DataFrame(per_wafer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"{pathlib.Path(__file__).name} {__version__}\n"
                    'Derive the thickness spec and the wafer uniformity spec from the prediction RMSE.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--csv', type=pathlib.Path, help='wafer CSV with true_p*/pred_p* columns')
    parser.add_argument('--target', type=float, help='thickness target (default: mean of the measured values)')
    parser.add_argument('--cp-min', type=float, default=SpecInput.cp_min, help='required Cpk or Cpu (default 1.33)')
    parser.add_argument('--n-sigma', type=float, default=SpecInput.n_sigma,
                        help='least number of RMSE inside the tolerance (default 10)')
    parser.add_argument('--guard-k', type=float, default=SpecInput.guard_k, help='guard band in RMSE (default 1.645)')
    parser.add_argument('--output-folder', type=pathlib.Path, help='folder for the per-wafer uniformity CSV')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    args = parser.parse_args()
    if args.csv is None or not args.csv.is_file():
        parser.print_help()
        parser.error(f"--csv must be an existing file: {args.csv}")
    if args.output_folder is not None and not args.output_folder.is_dir():
        parser.error(f"--output-folder must be an existing folder: {args.output_folder}")
    return args


if __name__ == '__main__':
    ARGS = parse_args()
    SETTING = SpecInput(cp_min=ARGS.cp_min, n_sigma=ARGS.n_sigma, guard_k=ARGS.guard_k)
    _, Y, Y_HAT = load(csv_path=ARGS.csv)
    TARGET = Y.mean() if ARGS.target is None else ARGS.target
    THICKNESS, UNIFORM, PER_WAFER = build_tables(y=Y, y_hat=Y_HAT, setting=SETTING, target=TARGET)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(f"M = {Y.shape[0]}, N = {Y.shape[1]}, {SETTING}")
    print('\nThickness (two-sided spec)\n', THICKNESS, sep='')
    print('\nUniformity 1 sigma / mean in % (one-sided spec)\n', UNIFORM, sep='')
    if ARGS.output_folder is not None:
        PATH = ARGS.output_folder / UNIFORMITY_CSV_NAME
        PER_WAFER.to_csv(PATH, index_label='wafer', float_format='%.6f')
        print(f"\nwrote {PATH}")
