#!/usr/bin/env python3
"""Draw the figure that compares a stationary record with the three ways a record stops being one.

One stationary series and three non-stationary ones are simulated from the same innovation
sequence, so the four panels differ only in what was added to it. Each panel carries the rolling
mean and the rolling standard deviation, and the first-half and second-half statistics that a
split-record comparison would read, so the figure and the document report the same numbers.
"""
__author__ = 'yRocket'
__version__ = "0.0.0.2026.9.7"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

matplotlib.use('Agg')

__all__ = ['ar1_series', 'build_series', 'half_statistics', 'rolling_statistics', 'draw_comparison']

COLORS: list = list(TABLEAU_COLORS.values())
FIGSIZE: tuple = (12.0, 7.0)
REFERENCE_WIDTH: float = 9.0  # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 9.0
PANEL_LABEL_Y: float = 0.02  # one height for every panel label, below the axis label
ANNOTATION_HEADROOM: float = 0.30  # share of the data span left free below it for the summary box

SAMPLE_COUNT: int = 600
ROLLING_WINDOW: int = 60
AR_COEFFICIENT: float = 0.6  # the autocorrelation the stationary series carries
STEP_SIZE: float = 2.0  # the mean shift the second case takes at the midpoint
SPREAD_START: float = 0.5  # the multiplier on the innovation at the first sample
SPREAD_END: float = 2.5  # the multiplier on the innovation at the last sample
RANDOM_SEED: int = 11

PANEL_TITLES: dict = {
    'stationary': 'Stationary  AR(1)',
    'mean_shift': 'Non-stationary  mean shift',
    'spread_growth': 'Non-stationary  growing spread',
    'random_walk': 'Non-stationary  random walk',
}


def ar1_series(innovation: np.ndarray = None, phi: float = None) -> np.ndarray:
    """The AR(1) path driven by the given innovation sequence, started at its stationary mean."""
    if innovation is None:
        raise ValueError('innovation is required.')
    if phi is None:
        raise ValueError('phi is required.')
    if abs(phi) >= 1.0:
        raise ValueError(f"phi must be smaller than 1 in absolute value for a stationary series, got {phi}.")
    path = np.empty_like(innovation)
    path[0] = innovation[0] / np.sqrt(1.0 - phi ** 2)  # start at the stationary variance
    for index in range(1, innovation.size):
        path[index] = phi * path[index - 1] + innovation[index]
    return path


def build_series(count: int = None, seed: int = None) -> pd.DataFrame:
    """The four records as columns of a frame indexed by sample number.

    Columns: stationary, mean_shift, spread_growth, random_walk.
    """
    if count is None:
        raise ValueError('count is required.')
    if seed is None:
        raise ValueError('seed is required.')
    if count < 2 * ROLLING_WINDOW:
        raise ValueError(f"count must hold at least two rolling windows, got {count}.")
    generator = np.random.default_rng(seed)
    innovation = generator.standard_normal(count)
    stationary = ar1_series(innovation=innovation, phi=AR_COEFFICIENT)
    step = np.where(np.arange(count) >= count // 2, STEP_SIZE, 0.0)
    spread = np.linspace(SPREAD_START, SPREAD_END, count)
    frame = pd.DataFrame({
        'stationary': stationary,
        'mean_shift': stationary + step,
        'spread_growth': innovation * spread,
        'random_walk': np.cumsum(innovation),
    })
    frame.index.name = 'sample'
    return frame


def half_statistics(frame: pd.DataFrame = None) -> pd.DataFrame:
    """The mean and the standard deviation of each column over the first and the second half.

    Index: the column names of the input frame. Columns: mean_first, mean_second, sd_first, sd_second.
    """
    if frame is None:
        raise ValueError('frame is required.')
    if frame.empty:
        raise ValueError('frame holds no rows.')
    split = len(frame) // 2
    first, second = frame.iloc[:split], frame.iloc[split:]
    table = pd.DataFrame({
        'mean_first': first.mean(),
        'mean_second': second.mean(),
        'sd_first': first.std(ddof=1),
        'sd_second': second.std(ddof=1),
    })
    table.index.name = 'series'
    return table


def rolling_statistics(frame: pd.DataFrame = None, window: int = None) -> tuple:
    """The rolling mean and the rolling standard deviation of every column, as two frames."""
    if frame is None:
        raise ValueError('frame is required.')
    if window is None:
        raise ValueError('window is required.')
    if window > len(frame):
        raise ValueError(f"window {window} exceeds the {len(frame)} samples of the record.")
    rolling = frame.rolling(window=window, center=True, min_periods=window)
    return rolling.mean(), rolling.std(ddof=1)


def draw_comparison(frame: pd.DataFrame = None, window: int = None, halves: pd.DataFrame = None,
                    output_path: pathlib.Path = None) -> pathlib.Path:
    """Draw one panel per record and save the figure, returning the path written."""
    if frame is None:
        raise ValueError('frame is required.')
    if window is None:
        raise ValueError('window is required.')
    if halves is None:
        raise ValueError('halves is required.')
    if output_path is None:
        raise ValueError('output_path is required.')
    missing = [name for name in frame.columns if name not in PANEL_TITLES]
    if missing:
        raise ValueError(f"no panel title for {missing}.")
    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    means, sds = rolling_statistics(frame=frame, window=window)
    figure, axes = plt.subplots(2, 2, figsize=FIGSIZE, sharex=True)
    figure.subplots_adjust(bottom=0.13, hspace=0.30, wspace=0.16)
    for position, name in enumerate(frame.columns):
        axis = axes.flat[position]
        axis.plot(frame.index, frame[name], color=COLORS[0], linewidth=0.7, alpha=0.55,
                  label='record')
        axis.plot(means.index, means[name], color=COLORS[3], linewidth=1.8,
                  label=f'rolling mean, {window} samples')
        axis.fill_between(means.index, means[name] - sds[name], means[name] + sds[name],
                          color=COLORS[1], alpha=0.28, linewidth=0.0,
                          label='rolling mean ± rolling sd')
        low, high = frame[name].min(), frame[name].max()
        span = high - low
        axis.set_ylim(low - ANNOTATION_HEADROOM * span, high + 0.05 * span)  # room for the summary box
        row = halves.loc[name]
        summary = (f"first half   mean {row['mean_first']:+.2f}   sd {row['sd_first']:.2f}\n"
                   f"second half  mean {row['mean_second']:+.2f}   sd {row['sd_second']:.2f}")
        axis.text(0.02, 0.03, summary, transform=axis.transAxes, fontsize=font_size * 0.82,
                  family='monospace', va='bottom', ha='left',
                  bbox=dict(boxstyle='round,pad=0.35', facecolor='white', alpha=0.80, linewidth=0.0))
        axis.set_title(PANEL_TITLES[name], fontsize=font_size * 1.02)
        axis.tick_params(labelsize=font_size * 0.9)
        axis.grid(alpha=0.25, linewidth=0.6)
        if position >= 2:
            axis.set_xlabel('Sample', fontsize=font_size)
        if position % 2 == 0:
            axis.set_ylabel('Value', fontsize=font_size)
    axes.flat[0].legend(loc='upper right', fontsize=font_size * 0.78, framealpha=0.85)
    for position, label in enumerate('(a) (b) (c) (d)'.split()):
        box = axes.flat[position].get_position()
        figure.text(box.x0 + box.width / 2.0, PANEL_LABEL_Y if position >= 2 else box.y0 - 0.055,
                    label, ha='center', va='bottom', fontsize=font_size)
    figure.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f'{pathlib.Path(__file__).name} {__version__}\n'
                    'Draw the figure that compares a stationary record with three non-stationary ones.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--output-folder', type=pathlib.Path, required=True,
                        help='folder that receives the figure and the csv tables; created if absent')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    arguments = parser.parse_args()
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    return arguments


if __name__ == '__main__':
    args = parse_args()
    series = build_series(count=SAMPLE_COUNT, seed=RANDOM_SEED)
    halves_table = half_statistics(frame=series)
    series.to_csv(args.output_folder / 'stationarity_series.csv')
    halves_table.to_csv(args.output_folder / 'stationarity_half_statistics.csv')
    written = draw_comparison(frame=series, window=ROLLING_WINDOW, halves=halves_table,
                              output_path=args.output_folder / 'stationarity_comparison.png')
    print(halves_table.round(3).to_string())
    print(f"wrote {written}")
