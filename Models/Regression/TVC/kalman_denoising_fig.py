"""Draw how much of the observation noise a Kalman filter removes.

A local level model is simulated, fitted with statsmodels, and the true state, the observations, the
filtered estimate and the smoothed estimate are drawn together with the error each one carries.
"""
__author__ = 'yRocket'
__version__ = "0.0.1.2026.8.31"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

__all__ = ['simulate_local_level', 'fit_local_level', 'draw_denoising']

FIGSIZE: tuple = (10.0, 3.8)
REFERENCE_WIDTH: float = 10.0        # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 10.0
COLORS: list = list(matplotlib.colors.TABLEAU_COLORS.values())


def simulate_local_level(n_points: int = 200, state_sd: float = 0.10, noise_sd: float = 1.0,
                         seed: int = 0) -> pd.DataFrame:
    """Simulate a random-walk state observed through additive noise.

    Returns a DataFrame indexed by 't' with columns 'true_state' and 'observation'.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points}")
    if state_sd <= 0 or noise_sd <= 0:
        raise ValueError(f"state_sd and noise_sd must be positive, got {state_sd} and {noise_sd}")
    rng = np.random.default_rng(seed)
    state = np.cumsum(rng.normal(scale=state_sd, size=n_points))
    observation = state + rng.normal(scale=noise_sd, size=n_points)
    return pd.DataFrame({'true_state': state, 'observation': observation},
                        index=pd.RangeIndex(start=1, stop=n_points + 1, name='t'))


def fit_local_level(observation: pd.Series) -> pd.DataFrame:
    """Fit the local level model and return its filtered and smoothed state.

    Returns a DataFrame on the index of `observation` with columns 'filtered' and 'smoothed'.
    """
    if observation.isna().any():
        raise ValueError("observation carries missing values; the filter needs a complete series.")
    result = sm.tsa.UnobservedComponents(observation.to_numpy(), level='local level').fit(disp=False)
    if not result.mle_retvals['converged']:
        raise RuntimeError("maximum likelihood estimation of Q and R did not converge.")
    return pd.DataFrame({'filtered': result.filtered_state[0], 'smoothed': result.smoothed_state[0]},
                        index=observation.index)


def draw_denoising(series: pd.DataFrame, output_folder: pathlib.Path) -> pathlib.Path:
    """Write the two-panel figure and the series it was drawn from. Returns the figure path."""
    output_folder.mkdir(parents=True, exist_ok=True)
    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    plt.rcParams.update({'font.size': font_size})

    rmse = {name: float(np.sqrt(np.mean((series[name] - series['true_state']) ** 2)))
            for name in ('observation', 'filtered', 'smoothed')}

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE, gridspec_kw={'width_ratios': [2.0, 1.0]})

    axes[0].plot(series.index, series['observation'], marker='.', linestyle='none', markersize=3.5,
                 color='0.6', label='observation')
    axes[0].plot(series.index, series['true_state'], color='0.15', linewidth=1.6, label='true state')
    axes[0].plot(series.index, series['filtered'], color=COLORS[0], linewidth=1.4, label='filtered')
    axes[0].plot(series.index, series['smoothed'], color=COLORS[1], linewidth=1.4, label='smoothed')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('value')
    axes[0].legend(frameon=False, ncol=2, fontsize=font_size * 0.9)

    names = ['observation', 'filtered', 'smoothed']
    axes[1].bar(names, [rmse[name] for name in names], color=['0.6', COLORS[0], COLORS[1]], width=0.6)
    for index, name in enumerate(names):
        axes[1].text(index, rmse[name], f"{rmse[name]:.2f}", ha='center', va='bottom', fontsize=font_size * 0.9)
    axes[1].set_ylabel('RMSE against the true state')
    axes[1].set_ylim(0, max(rmse.values()) * 1.25)

    fig.subplots_adjust(bottom=0.24, top=0.95)
    for axis, label in zip(axes, ('(a)', '(b)')):
        position = axis.get_position()
        fig.text(position.x0 + position.width / 2, 0.045, label, ha='center', fontsize=font_size)

    figure_path = output_folder / 'kalman-denoising.png'
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    series.to_csv(output_folder / 'kalman-denoising.csv')
    print(f"wrote {figure_path}")
    print("RMSE " + ", ".join(f"{name} {value:.3f}" for name, value in rmse.items()))
    return figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f"{pathlib.Path(__file__).name} {__version__}\n"
                    f"Draw how much observation noise a Kalman filter removes.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--output-folder', type=pathlib.Path, required=True,
                        help="folder the figure and its series are written to")
    parser.add_argument('--n-points', type=int, default=200, help="number of time points to simulate")
    parser.add_argument('--seed', type=int, default=0, help="seed of the random number generator")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if args.n_points < 2:
        parser.error(f"--n-points must be at least 2, got {args.n_points}")
    return args


if __name__ == '__main__':
    arguments = parse_args()
    simulated = simulate_local_level(n_points=arguments.n_points, seed=arguments.seed)
    estimates = fit_local_level(observation=simulated['observation'])
    draw_denoising(series=simulated.join(estimates), output_folder=arguments.output_folder)
