"""Draw one noisy sample fitted by three models of growing capacity.

Panel (a) fits a straight line, panel (b) a quadratic that matches the true curve, and panel (c) a cubic
spline that passes through every training point. The held-out RMSE of each fit is printed so the text can
quote it.
"""
__author__ = 'yRocket'
__version__ = "0.0.0+20261101"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

__all__ = ['noisy_quadratic', 'fit_curves', 'draw_fit_capacity']

FIGSIZE: tuple = (12.0, 3.6)
REFERENCE_WIDTH: float = 12.0        # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 11.0
COLORS: list = list(matplotlib.colors.TABLEAU_COLORS.values())
PANEL_LABELS: tuple = ('(a) Underfitted', '(b) Well-fitted', '(c) Overfitted')


def noisy_quadratic(n_samples: int, noise: float, seed: int) -> pd.DataFrame:
    """Sample y = 4 (x - 0.5)^2 + noise on evenly spaced x in [0, 1].

    Returns a DataFrame with a RangeIndex and columns 'x' and 'y'.
    """
    if n_samples < 4:
        raise ValueError(f"n_samples must be at least 4, got {n_samples}")
    if noise <= 0:
        raise ValueError(f"noise must be positive, got {noise}")
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, n_samples)
    y = 4.0 * (x - 0.5) ** 2 + rng.normal(scale=noise, size=n_samples)
    return pd.DataFrame({'x': x, 'y': y})


def fit_curves(train: pd.DataFrame) -> list:
    """Fit a line, a quadratic and an interpolating cubic spline. Returns three callables of x."""
    line = np.polynomial.Polynomial.fit(train['x'], train['y'], deg=1)
    quadratic = np.polynomial.Polynomial.fit(train['x'], train['y'], deg=2)
    spline = CubicSpline(train['x'], train['y'])
    return [line, quadratic, spline]


def draw_fit_capacity(train: pd.DataFrame, test: pd.DataFrame, output_folder: pathlib.Path) -> pathlib.Path:
    """Write the three-panel figure and print the train and held-out RMSE. Returns the figure path."""
    output_folder.mkdir(parents=True, exist_ok=True)
    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    plt.rcParams.update({'font.size': font_size})

    curves = fit_curves(train=train)
    grid = np.linspace(0.0, 1.0, 400)
    margin = 0.25 * (train['y'].max() - train['y'].min())
    y_limits = (train['y'].min() - margin, train['y'].max() + margin)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=FIGSIZE, sharey=True)
    for axis, curve in zip(axes, curves):
        axis.plot(grid, curve(grid), color=COLORS[1], linewidth=2.5, zorder=1)
        axis.scatter(train['x'], train['y'], s=30, color=COLORS[0], zorder=2)
        axis.set_ylim(*y_limits)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel('x')
        axis.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('y')

    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.22, top=0.97, wspace=0.12)
    for axis, label in zip(axes, PANEL_LABELS):
        position = axis.get_position()
        fig.text(position.x0 + position.width / 2, 0.03, label, ha='center', fontsize=font_size)

    figure_path = output_folder / 'fit-capacity.png'
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    for label, curve in zip(PANEL_LABELS, curves):
        train_rmse = float(np.sqrt(np.mean((curve(train['x']) - train['y']) ** 2)))
        test_rmse = float(np.sqrt(np.mean((curve(test['x']) - test['y']) ** 2)))
        print(f"{label:<16} train RMSE {train_rmse:.3f}  held-out RMSE {test_rmse:.3f}")
    print(f"wrote {figure_path}")
    return figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f"{pathlib.Path(__file__).name} {__version__}\n"
                    f"Draw one noisy sample fitted by three models of growing capacity.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--output-folder', type=pathlib.Path, required=True,
                        help="folder the figure is written to")
    parser.add_argument('--n-train', type=int, default=16, help="points used for fitting")
    parser.add_argument('--noise', type=float, default=0.12, help="standard deviation of the noise")
    parser.add_argument('--seed', type=int, default=3, help="seed of the training sample")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if args.n_train < 4:
        parser.error(f"--n-train must be at least 4, got {args.n_train}")
    if args.noise <= 0:
        parser.error(f"--noise must be positive, got {args.noise}")
    return args


if __name__ == '__main__':
    arguments = parse_args()
    train_frame = noisy_quadratic(n_samples=arguments.n_train, noise=arguments.noise, seed=arguments.seed)
    test_frame = noisy_quadratic(n_samples=2001, noise=arguments.noise, seed=arguments.seed + 1000)
    draw_fit_capacity(train=train_frame, test=test_frame, output_folder=arguments.output_folder)
