"""Generate the wafer measurement table that the variance decomposition document analyses.

Each wafer draws one level of its own and each site adds measurement noise on top of it, so the
table carries a known within-wafer component and a known wafer-to-wafer component. A run order
drift, a few wafers with a shifted level, and a few wafers with inflated site noise are laid over
that model. The seed is fixed, so the same table comes out every time.

Changelog:
- 0.0.0: initial release.
"""

__author__ = 'yRocket'
__version__ = "0.0.1+20260922"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import pathlib

import numpy as np
import pandas as pd

WAFER_COUNT = 200
SITE_COUNT = 13
SEED = 137

GRAND_MEAN = 609.0          # process level of the first wafer, before the drift
LEVEL_SD = 19.0             # standard deviation of the wafer level around that process level
LEVEL_DRIFT = 0.13          # added to the wafer level per wafer along run order
SITE_SD_BASE = 7.6          # site noise of the first wafer
SITE_SD_SLOPE = 0.036       # added to the site noise per wafer along run order

SHIFTED_WAFERS = 6          # wafers whose level is pushed away from the drift
SHIFT_RANGE = (55.0, 160.0)
SHIFT_UP_SHARE = 0.25       # share of the shifted wafers pushed up instead of down

NOISY_WAFERS = 20           # wafers whose site noise is inflated
NOISE_MULTIPLIER = (2.0, 3.8)

FIRST_CLEAN_WAFER = 8       # wafers before this one carry neither a shift nor inflated noise
DECIMALS = 3


def wafer_levels(rng: np.random.Generator, wafer_count: int) -> np.ndarray:
    """Return the level of each wafer: a random level, the run order drift, and a few shifts."""
    order = np.arange(wafer_count)
    level = rng.normal(0.0, LEVEL_SD, wafer_count) + LEVEL_DRIFT * order
    shifted = rng.choice(np.arange(FIRST_CLEAN_WAFER, wafer_count), size=SHIFTED_WAFERS,
                         replace=False)
    direction = rng.choice([-1.0, 1.0], size=SHIFTED_WAFERS,
                           p=[1.0 - SHIFT_UP_SHARE, SHIFT_UP_SHARE])
    level[shifted] += direction * rng.uniform(*SHIFT_RANGE, size=SHIFTED_WAFERS)
    return level


def site_noise_sd(rng: np.random.Generator, wafer_count: int) -> np.ndarray:
    """Return the site noise of each wafer: a drifting base with a few wafers inflated."""
    order = np.arange(wafer_count)
    noise_sd = SITE_SD_BASE + SITE_SD_SLOPE * order
    noisy = rng.choice(np.arange(FIRST_CLEAN_WAFER, wafer_count), size=NOISY_WAFERS,
                       replace=False)
    noise_sd[noisy] *= rng.uniform(*NOISE_MULTIPLIER, size=NOISY_WAFERS)
    return noise_sd


def build_table(seed: int, wafer_count: int, site_count: int) -> pd.DataFrame:
    """Return the measurement table, one row per wafer and one column per site."""
    if wafer_count < 2:
        raise ValueError(f"wafer_count must be at least 2 to carry a wafer-to-wafer component; got {wafer_count}.")
    if site_count < 2:
        raise ValueError(f"site_count must be at least 2 to carry a within-wafer component; got {site_count}.")
    rng = np.random.default_rng(seed)
    level = wafer_levels(rng, wafer_count)
    noise_sd = site_noise_sd(rng, wafer_count)
    noise = rng.normal(0.0, 1.0, (wafer_count, site_count)) * noise_sd[:, None]
    values = np.round(GRAND_MEAN + level[:, None] + noise, DECIMALS)
    table = pd.DataFrame(values, columns=[f"S{j + 1}" for j in range(site_count)])
    table.insert(0, 'wafer_id', [f"wf{i + 1:04d}" for i in range(wafer_count)])
    return table


def report(table: pd.DataFrame) -> str:
    """Return the variance components of the table, so the caller sees what was written."""
    values = table.filter(regex=r"^S\d+$").to_numpy()
    site_count = values.shape[1]
    ms_within = values.var(axis=1, ddof=1).mean()
    ms_between = site_count * values.mean(axis=1).var(ddof=1)
    sigma_within = np.sqrt(ms_within)
    sigma_between = np.sqrt(max(ms_between - ms_within, 0.0) / site_count)
    icc = sigma_between ** 2 / (sigma_between ** 2 + sigma_within ** 2)
    return (f"sigma_within = {sigma_within:.3f}, sigma_between = {sigma_between:.3f}, "
            f"sigma_total = {values.std(ddof=1):.3f}, ICC = {icc:.3f}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"make_example_csv.py {__version__} - generate the wafer measurement table.")
    parser.add_argument('-v', '--version', action='version',
                        version=f"make_example_csv.py {__version__}")
    parser.add_argument('--output-csv', default='example.csv', help="file to write (default: example.csv)")
    parser.add_argument('--seed', type=int, default=SEED, help=f"random seed (default: {SEED})")
    parser.add_argument('--wafer-count', type=int, default=WAFER_COUNT,
                        help=f"number of wafers (default: {WAFER_COUNT})")
    parser.add_argument('--site-count', type=int, default=SITE_COUNT,
                        help=f"number of sites per wafer (default: {SITE_COUNT})")
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_arguments()
    measurement_table = build_table(arguments.seed, arguments.wafer_count, arguments.site_count)
    output_path = pathlib.Path(arguments.output_csv)
    measurement_table.to_csv(output_path, index=False)
    print(f"{output_path} written: {len(measurement_table)} wafers x {arguments.site_count} sites")
    print(report(measurement_table))
