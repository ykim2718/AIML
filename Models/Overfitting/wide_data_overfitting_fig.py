"""Draw why wide data overfits and when each defense helps.

Panel (a) simulates the largest absolute correlation between a response and p unrelated columns.
Panels (b) and (c) fit four models on a wide design and compare their held-out error as the column
count grows, first when the signal sits in a few columns and then when it is spread over many.
"""
__author__ = 'yRocket'
__version__ = "0.0.2.2026.9.6"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.metrics import root_mean_squared_error

__all__ = ['chance_correlation', 'sparse_design', 'factor_design', 'held_out_error',
           'draw_wide_data_overfitting']

FIGSIZE: tuple = (13.5, 4.0)
REFERENCE_WIDTH: float = 13.5        # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 10.0
COLORS: list = list(matplotlib.colors.TABLEAU_COLORS.values())
FITS: tuple = ('least squares', 'ridge', 'lasso', 'random subspace')


def chance_correlation(n_samples: int, column_counts: list, n_repeat: int = 200,
                       seed: int = 0) -> pd.DataFrame:
    """Largest absolute correlation between a random response and p unrelated columns.

    Returns a DataFrame indexed by 'p' with columns 'median' and 'bound'.
    """
    if n_samples < 3:
        raise ValueError(f"n_samples must be at least 3, got {n_samples}")
    rng = np.random.default_rng(seed)
    rows = []
    for p in column_counts:
        largest = np.empty(n_repeat)
        for repeat in range(n_repeat):
            x = rng.normal(size=(n_samples, p))
            y = rng.normal(size=n_samples)
            x = (x - x.mean(axis=0)) / x.std(axis=0)
            y = (y - y.mean()) / y.std()
            largest[repeat] = np.abs(x.T @ y / n_samples).max()
        rows.append({'p': p, 'median': float(np.median(largest)),
                     'bound': float(np.sqrt(2 * np.log(2 * p) / n_samples))})
    return pd.DataFrame(rows).set_index('p')


def sparse_design(n_train: int, n_test: int, n_columns: int, n_informative: int, noise_sd: float,
                  rng: np.random.Generator) -> tuple:
    """Train and test draws whose response depends on the first n_informative columns.

    Returns (x_train, y_train, x_test, y_test). One coefficient vector is drawn and both sets use it.
    """
    if n_columns < n_informative:
        raise ValueError(f"n_columns must be at least n_informative, got {n_columns}")
    beta = np.zeros(n_columns)
    beta[:n_informative] = rng.normal(loc=1.0, scale=0.2, size=n_informative)
    draws = []
    for n_rows in (n_train, n_test):
        x = rng.normal(size=(n_rows, n_columns))
        draws += [x, x @ beta + rng.normal(scale=noise_sd, size=n_rows)]
    return tuple(draws)


def factor_design(n_train: int, n_test: int, n_columns: int, n_factor: int, noise_sd: float,
                  rng: np.random.Generator) -> tuple:
    """Train and test draws whose columns share a few latent factors that the response follows.

    Returns (x_train, y_train, x_test, y_test). One loading matrix is drawn and both sets use it.
    """
    if n_factor < 1:
        raise ValueError(f"n_factor must be at least 1, got {n_factor}")
    loading = rng.normal(size=(n_factor, n_columns))
    gamma = rng.normal(size=n_factor)
    draws = []
    for n_rows in (n_train, n_test):
        factor = rng.normal(size=(n_rows, n_factor))
        draws += [factor @ loading + rng.normal(scale=0.5, size=(n_rows, n_columns)),
                  factor @ gamma + rng.normal(scale=noise_sd, size=n_rows)]
    return tuple(draws)


def held_out_error(design: str, n_train: int, n_test: int, column_counts: list,
                   noise_sd: float = 1.0, seed: int = 0) -> pd.DataFrame:
    """Held-out RMSE of the four fits as the column count grows.

    `design` is 'sparse' or 'factor'. Returns a DataFrame indexed by 'p' with one column per fit.
    """
    if design not in ('sparse', 'factor'):
        raise ValueError(f"design must be 'sparse' or 'factor', got {design!r}")
    rng = np.random.default_rng(seed)
    rows = []
    for p in column_counts:
        if design == 'sparse':
            x_train, y_train, x_test, y_test = sparse_design(
                n_train=n_train, n_test=n_test, n_columns=p, n_informative=5, noise_sd=noise_sd,
                rng=rng)
        else:
            x_train, y_train, x_test, y_test = factor_design(
                n_train=n_train, n_test=n_test, n_columns=p, n_factor=4, noise_sd=noise_sd, rng=rng)
        models = {
            'least squares': LinearRegression(),
            'ridge': RidgeCV(alphas=np.logspace(-3, 4, 40)),
            'lasso': LassoCV(max_iter=20000, random_state=0),
            'random subspace': BaggingRegressor(
                estimator=LinearRegression(), n_estimators=200,
                max_features=min(p, max(2, int(np.sqrt(p)))),
                bootstrap=False, bootstrap_features=False, random_state=0),
        }
        row = {'p': p}
        for name, model in models.items():
            model.fit(x_train, y_train)
            row[name] = float(root_mean_squared_error(y_test, model.predict(x_test)))
        rows.append(row)
    return pd.DataFrame(rows).set_index('p')


def draw_wide_data_overfitting(chance: pd.DataFrame, sparse: pd.DataFrame, factor: pd.DataFrame,
                               output_folder: pathlib.Path) -> pathlib.Path:
    """Write the three-panel figure and the series it was drawn from. Returns the figure path."""
    output_folder.mkdir(parents=True, exist_ok=True)
    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    plt.rcParams.update({'font.size': font_size})

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=FIGSIZE)

    axes[0].plot(chance.index, chance['median'], marker='o', markersize=4, color=COLORS[0],
                 label='simulated median')
    axes[0].plot(chance.index, chance['bound'], linestyle='--', color='0.4',
                 label=r'$\sqrt{2\ln(2p)/n}$')
    axes[0].set_xlabel('columns p, no real signal')
    axes[0].set_ylabel('largest |correlation| with y')
    axes[0].set_ylim(0, None)

    palette = ('0.4', COLORS[0], COLORS[1], COLORS[2])
    for axis, frame, title in ((axes[1], sparse, 'signal in 5 columns'),
                               (axes[2], factor, 'signal shared by all columns')):
        for name, color in zip(FITS, palette):
            axis.plot(frame.index, frame[name], marker='o', markersize=4, color=color, label=name)
        axis.set_xlabel(f"columns p, {title}")
        axis.set_ylabel('held-out RMSE')

    for axis in axes:
        axis.set_xscale('log')
        axis.legend(frameon=False, fontsize=font_size * 0.85)

    fig.subplots_adjust(bottom=0.22, top=0.96, wspace=0.30)
    for axis, label in zip(axes, ('(a)', '(b)', '(c)')):
        position = axis.get_position()
        fig.text(position.x0 + position.width / 2, 0.04, label, ha='center', fontsize=font_size)

    figure_path = output_folder / 'wide-data-overfitting.png'
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    chance.to_csv(output_folder / 'chance-correlation.csv')
    sparse.to_csv(output_folder / 'held-out-error-sparse.csv')
    factor.to_csv(output_folder / 'held-out-error-factor.csv')
    print(f"wrote {figure_path}")
    print(chance.round(3).to_string())
    print(sparse.round(3).to_string())
    print(factor.round(3).to_string())
    return figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f"{pathlib.Path(__file__).name} {__version__}\n"
                    f"Draw why wide data overfits and when each defense helps.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--output-folder', type=pathlib.Path, required=True,
                        help="folder the figure and its series are written to")
    parser.add_argument('--n-train', type=int, default=100, help="rows used for fitting")
    parser.add_argument('--n-repeat', type=int, default=200, help="repeats of the chance simulation")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if args.n_train < 10:
        parser.error(f"--n-train must be at least 10, got {args.n_train}")
    return args


if __name__ == '__main__':
    arguments = parse_args()
    chance_frame = chance_correlation(n_samples=arguments.n_train,
                                      column_counts=[10, 30, 100, 300, 1000, 3000, 10000],
                                      n_repeat=arguments.n_repeat)
    widths = [10, 30, 60, 90, 200, 500, 1000]
    sparse_frame = held_out_error(design='sparse', n_train=arguments.n_train, n_test=2000,
                                  column_counts=widths)
    factor_frame = held_out_error(design='factor', n_train=arguments.n_train, n_test=2000,
                                  column_counts=widths, seed=1)
    draw_wide_data_overfitting(chance=chance_frame, sparse=sparse_frame, factor=factor_frame,
                               output_folder=arguments.output_folder)
