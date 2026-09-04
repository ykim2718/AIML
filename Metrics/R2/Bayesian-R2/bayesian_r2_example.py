"""Reproduce the worked example that compares Pearson R2, standard R2 and Bayesian R2.

The script fits an 8-point dataset by OLS, draws from the conjugate posterior of the
Bayesian linear model, and reports R2 as a distribution with a credible interval.
Deliberately miscalibrated predictors are evaluated as well, to show what Pearson R2
and standard R2 do when the predictions are biased or shrunk.

Changelog:
    0.0.0 - Initial release.
"""

__author__ = 'yRocket'
__version__ = "0.0.0.2026.8.16"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Union

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS

# The worked example dataset. x is a controlled setting, y is the measured response.
EXAMPLE_X = np.array([1., 2., 3., 4., 5., 6., 7., 8.])
EXAMPLE_Y = np.array([1.8, 4.9, 5.2, 8.9, 9.4, 13.6, 12.8, 17.1])

# Predictors constructed by hand, not drawn from the posterior. Each is an affine
# function of x, so every one of them shares a single Pearson R2 with the data.
PROBE_PREDICTORS = {
    'well calibrated': (0.10, 2.05),
    'mildly shrunk': (1.10, 1.83),
    'strongly shrunk': (4.60, 1.05),
    'collapsed to a wrong center': (12.00, 0.20),
}

PALETTE = list(TABLEAU_COLORS.values())

# The R2 posterior is steep near its ceiling and trails off to the left, so plotting the
# full range leaves every draw stacked in one bin. The histogram is drawn from this lower
# quantile upward, and the draws left outside are counted in the legend rather than hidden.
FIGURE_TAIL_QUANTILE = 0.002


class Variant(StrEnum):
    """Name of the residual variance used in the denominator of Bayesian R2."""

    EMPIRICAL = auto()
    MODEL_BASED = auto()


@dataclass
class PosteriorDraws:
    """Draws from the conjugate posterior of a Bayesian simple linear regression.

    Attributes:
        intercept: shape (S,).
        slope: shape (S,).
        sigma: shape (S,), the noise scale, not the variance.
        fitted: shape (S, n), the predictive mean of each draw at each data point.
    """

    intercept: np.ndarray
    slope: np.ndarray
    sigma: np.ndarray
    fitted: np.ndarray


def ols_fit(x: np.ndarray = None, y: np.ndarray = None) -> tuple[float, float]:
    """Fit y = intercept + slope * x by ordinary least squares.

    Returns:
        The intercept and the slope.
    """
    design = np.column_stack([np.ones_like(x), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(intercept), float(slope)


def pearson_r2(y_true: np.ndarray = None, y_pred: np.ndarray = None) -> float:
    """Square of the Pearson correlation coefficient between observations and predictions.

    The value is invariant under any affine transform of y_pred with a non-zero slope,
    which is why it cannot detect bias or a wrong scale.
    """
    return float(np.corrcoef(y_true, y_pred)[0, 1] ** 2)


def standard_r2(y_true: np.ndarray = None, y_pred: np.ndarray = None) -> float:
    """Coefficient of determination, 1 - SS_res / SS_tot.

    The value falls below 0 whenever the predictions are worse than the mean of y_true.
    """
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot


def bayesian_r2_empirical(y_pred_draws: np.ndarray = None, y_true: np.ndarray = None) -> np.ndarray:
    """Bayesian R2 using the variance of the residuals actually left over.

    Args:
        y_pred_draws: posterior draws of the predictive mean, shape (S, n).
        y_true: observations, shape (n,).

    Returns:
        R2 draws of shape (S,), every element inside [0, 1].
    """
    _check_draw_matrix(y_pred_draws=y_pred_draws, n_points=y_true.shape[0])
    var_fit = y_pred_draws.var(axis=1, ddof=1)                       # variance over data points
    var_res = (y_true[None, :] - y_pred_draws).var(axis=1, ddof=1)
    return var_fit / (var_fit + var_res)


def bayesian_r2_model_based(y_pred_draws: np.ndarray = None, sigma_draws: np.ndarray = None) -> np.ndarray:
    """Bayesian R2 using the noise variance the model claims for itself.

    Args:
        y_pred_draws: posterior draws of the predictive mean, shape (S, n).
        sigma_draws: posterior draws of the noise scale, shape (S,).

    Returns:
        R2 draws of shape (S,), every element inside [0, 1].
    """
    _check_draw_matrix(y_pred_draws=y_pred_draws, n_points=None)
    if sigma_draws.shape[0] != y_pred_draws.shape[0]:
        raise ValueError(f"draw count mismatch: {y_pred_draws.shape[0]} predictions vs {sigma_draws.shape[0]} sigmas.")
    var_fit = y_pred_draws.var(axis=1, ddof=1)
    return var_fit / (var_fit + sigma_draws ** 2)


def _check_draw_matrix(y_pred_draws: np.ndarray = None, n_points: Union[int, None] = None) -> None:
    """Fail loudly on a draw matrix whose shape cannot mean what the caller intends."""
    if y_pred_draws.ndim != 2:
        raise ValueError(f"y_pred_draws must be 2-D (S, n), got shape {y_pred_draws.shape}.")
    if y_pred_draws.shape[1] < 2:
        raise ValueError(f"need at least 2 data points to take a variance, got {y_pred_draws.shape[1]}.")
    if n_points is not None and y_pred_draws.shape[1] != n_points:
        raise ValueError(f"y_pred_draws has {y_pred_draws.shape[1]} points but y_true has {n_points}.")


def sample_posterior(x: np.ndarray = None, y: np.ndarray = None, draws: int = None,
                     seed: int = None) -> PosteriorDraws:
    """Draw from the conjugate posterior of a linear model under the reference prior.

    The prior is p(beta, sigma^2) proportional to 1 / sigma^2, for which the posterior is
    available in closed form. Sampling is therefore exact and needs no MCMC convergence check.

    Args:
        x: predictor, shape (n,).
        y: response, shape (n,).
        draws: number of posterior draws.
        seed: seed of the random generator, so the report is reproducible.

    Returns:
        The posterior draws and their fitted values.
    """
    n_points = x.shape[0]
    n_params = 2
    if n_points <= n_params:
        raise ValueError(f"need more than {n_params} points to identify the residual scale, got {n_points}.")

    design = np.column_stack([np.ones_like(x), x])
    beta_hat = np.linalg.lstsq(design, y, rcond=None)[0]
    residual_ss = float(np.sum((y - design @ beta_hat) ** 2))
    degrees_of_freedom = n_points - n_params

    rng = np.random.default_rng(seed=seed)
    # sigma^2 | y follows an inverse gamma, sampled here through a chi-square variate.
    sigma_sq = residual_ss / rng.chisquare(df=degrees_of_freedom, size=draws)
    sigma = np.sqrt(sigma_sq)

    # beta | sigma^2, y is normal with covariance sigma^2 * (X'X)^-1.
    unit_cov = np.linalg.inv(design.T @ design)
    cov_root = np.linalg.cholesky(unit_cov)
    standard_normal = rng.standard_normal(size=(draws, n_params))
    beta = beta_hat[None, :] + (standard_normal @ cov_root.T) * sigma[:, None]

    return PosteriorDraws(intercept=beta[:, 0], slope=beta[:, 1], sigma=sigma,
                          fitted=beta @ design.T)


def summarize(values: np.ndarray = None, interval: float = None) -> dict:
    """Summarize a posterior sample as a median and an equal-tailed credible interval."""
    tail = (1.0 - interval) / 2.0
    lower, upper = np.quantile(values, [tail, 1.0 - tail])
    return {'median': float(np.median(values)), 'lower': float(lower), 'upper': float(upper),
            'mean': float(np.mean(values)), 'min': float(np.min(values)), 'max': float(np.max(values))}


def report_dataset(x: np.ndarray = None, y: np.ndarray = None) -> None:
    """Print the dataset alongside its OLS fit, Pearson R2 and standard R2."""
    intercept, slope = ols_fit(x=x, y=y)
    fitted = intercept + slope * x
    print(f"\n[1] Dataset and OLS reference fit  (n = {x.shape[0]})")
    print(f"    intercept = {intercept:+.4f}   slope = {slope:+.4f}")
    print(f"    mean(y) = {np.mean(y):.4f}   var(y) = {np.var(y, ddof=1):.4f}")
    print(f"    {'i':>3} {'x':>7} {'y':>8} {'y_hat':>9} {'residual':>10}")
    for index, (x_value, y_value, fit_value) in enumerate(zip(x, y, fitted), start=1):
        print(f"    {index:>3} {x_value:>7.1f} {y_value:>8.2f} {fit_value:>9.3f} {y_value - fit_value:>+10.3f}")
    print(f"    Pearson R2  = {pearson_r2(y_true=y, y_pred=fitted):.4f}")
    print(f"    standard R2 = {standard_r2(y_true=y, y_pred=fitted):.4f}   (identical by construction for OLS)")


def report_posterior(y: np.ndarray = None, posterior: PosteriorDraws = None, interval: float = None,
                     preview: int = None) -> dict:
    """Print the first draws in full and then the posterior summary of both variants."""
    empirical = bayesian_r2_empirical(y_pred_draws=posterior.fitted, y_true=y)
    model_based = bayesian_r2_model_based(y_pred_draws=posterior.fitted, sigma_draws=posterior.sigma)

    print(f"\n[2] First {preview} posterior draws  (empirical variant)")
    print(f"    {'s':>3} {'a':>8} {'b':>8} {'sigma':>8} {'Var_fit':>10} {'Var_res':>10} {'R2':>9} {'PearsonR2':>11}")
    var_fit = posterior.fitted.var(axis=1, ddof=1)
    var_res = (y[None, :] - posterior.fitted).var(axis=1, ddof=1)
    for index in range(preview):
        print(f"    {index + 1:>3} {posterior.intercept[index]:>+8.3f} {posterior.slope[index]:>+8.3f} "
              f"{posterior.sigma[index]:>8.3f} {var_fit[index]:>10.3f} {var_res[index]:>10.3f} "
              f"{empirical[index]:>9.4f} {pearson_r2(y_true=y, y_pred=posterior.fitted[index]):>11.4f}")

    percent = int(round(interval * 100))
    summaries = {Variant.EMPIRICAL: summarize(values=empirical, interval=interval),
                 Variant.MODEL_BASED: summarize(values=model_based, interval=interval)}
    print(f"\n[3] Bayesian R2 over {empirical.shape[0]:,} draws  ({percent}% equal-tailed credible interval)")
    for variant, stats in summaries.items():
        print(f"    {variant:<12} median = {stats['median']:.4f}   "
              f"{percent}% CI = [{stats['lower']:.4f}, {stats['upper']:.4f}]   "
              f"range = [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"    out-of-range draws (outside [0, 1]): "
          f"{int(np.sum((empirical < 0) | (empirical > 1)))} of {empirical.shape[0]:,}")
    return {'empirical': empirical, 'model_based': model_based, 'summaries': summaries}


def report_probes(x: np.ndarray = None, y: np.ndarray = None) -> None:
    """Print the three metrics for the hand-constructed predictors."""
    print("\n[4] Hand-constructed predictors, not posterior draws")
    print(f"    {'predictor':<30} {'a':>7} {'b':>6} {'PearsonR2':>11} {'standardR2':>12} {'BayesianR2':>12}")
    for label, (intercept, slope) in PROBE_PREDICTORS.items():
        prediction = intercept + slope * x
        single_draw = prediction[None, :]
        bayes = float(bayesian_r2_empirical(y_pred_draws=single_draw, y_true=y)[0])
        print(f"    {label:<30} {intercept:>7.2f} {slope:>6.2f} "
              f"{pearson_r2(y_true=y, y_pred=prediction):>11.4f} "
              f"{standard_r2(y_true=y, y_pred=prediction):>12.4f} {bayes:>12.4f}")
    print("    Pearson R2 is constant above because every predictor is affine in the same x.")


def plot_report(x: np.ndarray = None, y: np.ndarray = None, posterior: PosteriorDraws = None,
                r2_draws: np.ndarray = None, interval: float = None, lines: int = None,
                output_path: pathlib.Path = None) -> None:
    """Save a two-panel figure showing the posterior fits and the resulting R2 distribution."""
    stats = summarize(values=r2_draws, interval=interval)
    percent = int(round(interval * 100))
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4.8))

    axes[0].plot(x, posterior.fitted[:lines].T, color=PALETTE[0], alpha=0.06, linewidth=1.0)
    axes[0].plot(x, posterior.fitted[:lines].T[:, :1], color=PALETTE[0], alpha=0.5, linewidth=1.0,
                 label=f"{lines} posterior fits")
    axes[0].scatter(x, y, color=PALETTE[3], zorder=3, s=45, label='observations')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('(a) Data and posterior fits')
    axes[0].legend(loc='upper left', frameon=False)

    display_low = float(np.quantile(r2_draws, FIGURE_TAIL_QUANTILE))
    below_axis = int(np.sum(r2_draws < display_low))
    axes[1].hist(r2_draws, bins=60, range=(display_low, float(np.max(r2_draws))), color=PALETTE[0], alpha=0.75)
    axes[1].axvline(stats['median'], color=PALETTE[3], linewidth=2.0,
                    label=f"median = {stats['median']:.4f}")
    for bound, name in ((stats['lower'], 'lower'), (stats['upper'], 'upper')):
        axes[1].axvline(bound, color=PALETTE[1], linewidth=1.6, linestyle='--',
                        label=f"{percent}% CI {name} = {bound:.4f}")
    axes[1].plot([], [], ' ', label=f"{below_axis} draws left of the axis")
    axes[1].set_xlabel('Bayesian R2')
    axes[1].set_ylabel('draw count')
    axes[1].set_title('(b) Posterior distribution of Bayesian R2')
    axes[1].legend(loc='upper left', frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"\n[5] Figure written to {output_path}")


def parse_args() -> argparse.Namespace:
    """Parse and validate the command line options."""
    parser = argparse.ArgumentParser(description='Compare Pearson R2, standard R2 and Bayesian R2 on a small dataset.')
    parser.add_argument('--draws', type=int, default=4000,
                        help='number of posterior draws (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=20260531,
                        help='random generator seed, fixed so the report is reproducible (default: %(default)s)')
    parser.add_argument('--interval', type=float, default=0.90,
                        help='width of the equal-tailed credible interval (default: %(default)s)')
    parser.add_argument('--preview-draws', type=int, default=5,
                        help='how many individual draws to print in full (default: %(default)s)')
    parser.add_argument('--figure-lines', type=int, default=200,
                        help='how many posterior fits to overlay in panel (a) (default: %(default)s)')
    parser.add_argument('--save-figure', choices=['true', 'false'], default='true',
                        help='write the two-panel figure (default: %(default)s)')
    parser.add_argument('--output-folder', type=pathlib.Path, default=None,
                        help='folder for the figure (default: bayesian-r2_fig next to this script)')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()
    args.save_figure = args.save_figure == 'true'
    if args.output_folder is None:
        args.output_folder = pathlib.Path(__file__).resolve().parent / 'bayesian-r2_fig'

    if args.draws < 100:
        parser.error(f"--draws must be at least 100 for a usable quantile, got {args.draws}.")
    if not 0.0 < args.interval < 1.0:
        parser.error(f"--interval must lie strictly between 0 and 1, got {args.interval}.")
    if not 1 <= args.preview_draws <= args.draws:
        parser.error(f"--preview-draws must lie between 1 and --draws ({args.draws}), got {args.preview_draws}.")
    if not 1 <= args.figure_lines <= args.draws:
        parser.error(f"--figure-lines must lie between 1 and --draws ({args.draws}), got {args.figure_lines}.")
    if args.save_figure:
        args.output_folder.mkdir(parents=True, exist_ok=True)
        if not args.output_folder.is_dir():
            parser.error(f"--output-folder is not a folder: {args.output_folder}")

    return args


if __name__ == '__main__':
    options = parse_args()

    report_dataset(x=EXAMPLE_X, y=EXAMPLE_Y)
    draws_of_posterior = sample_posterior(x=EXAMPLE_X, y=EXAMPLE_Y, draws=options.draws, seed=options.seed)
    posterior_report = report_posterior(y=EXAMPLE_Y, posterior=draws_of_posterior, interval=options.interval,
                                        preview=options.preview_draws)
    report_probes(x=EXAMPLE_X, y=EXAMPLE_Y)

    if options.save_figure:
        plot_report(x=EXAMPLE_X, y=EXAMPLE_Y, posterior=draws_of_posterior,
                    r2_draws=posterior_report['empirical'], interval=options.interval,
                    lines=options.figure_lines, output_path=options.output_folder / 'bayesian_r2_posterior.png')
