#!/usr/bin/env python3
"""Show a fault that two univariate charts miss and that Hotelling T2 and the PCA residual catch.

The figure and the numbers the accompanying document quotes are produced here, so the picture and
the text come from one run.
"""
__author__ = 'yRocket'
__version__ = "0.0.0.2026.9.4"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from scipy import stats

__all__ = ['reference_data', 'hotelling_t2', 'squared_prediction_error', 'residual_limit',
           'false_alarm_frame', 'draw_monitoring']

FIGSIZE: tuple = (12.5, 4.4)
REFERENCE_WIDTH: float = 12.5    # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 10.0
DPI: int = 300
N_SAMPLES: int = 120
CORRELATION: float = 0.92
SENSOR_MEANS: tuple = (400.0, 25.0)
SENSOR_SIGMAS: tuple = (8.0, 1.5)
FAULT_AT: int = 90               # zero-based index of the fault sample
FAULT_OFFSET: tuple = (2.0, -2.0)   # in units of each sensor standard deviation
N_COMPONENTS: int = 1
ALPHA: float = 0.01
UNIVARIATE_ALPHA: float = 0.0027
SENSOR_COUNTS: tuple = (1, 2, 5, 10, 20, 50, 100)
CURVE_COLORS: tuple = tuple(TABLEAU_COLORS.values())


def reference_data(n_samples: int = N_SAMPLES, correlation: float = CORRELATION,
                   means: tuple = SENSOR_MEANS, sigmas: tuple = SENSOR_SIGMAS,
                   fault_at: int = FAULT_AT, fault_offset: tuple = FAULT_OFFSET,
                   seed: int = 5) -> pd.DataFrame:
    """Two correlated sensors, with one sample whose correlation is broken but whose values are not.

    Returns a pd.DataFrame with a RangeIndex and columns 'sample', 'sensor_1', 'sensor_2' and
    'faulted'.
    """
    if not -1.0 < correlation < 1.0:
        raise ValueError(f"correlation must lie strictly between -1 and 1; got {correlation}")
    if not 0 < fault_at < n_samples:
        raise ValueError(f"fault_at must fall inside the run; got {fault_at} of {n_samples}")
    rng = np.random.default_rng(seed)
    covariance = np.array([[1.0, correlation], [correlation, 1.0]])
    standard = rng.multivariate_normal(mean=[0.0, 0.0], cov=covariance, size=n_samples)
    standard[fault_at] = np.array(fault_offset)
    faulted = np.zeros(n_samples, dtype=bool)
    faulted[fault_at] = True
    return pd.DataFrame({'sample': np.arange(1, n_samples + 1),
                         'sensor_1': means[0] + sigmas[0] * standard[:, 0],
                         'sensor_2': means[1] + sigmas[1] * standard[:, 1],
                         'faulted': faulted})


def hotelling_t2(values: np.ndarray, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """Squared Mahalanobis distance of each row from the reference mean."""
    values = np.atleast_2d(values)
    if covariance.shape[0] != values.shape[1]:
        raise ValueError(f"covariance is {covariance.shape} but the data has {values.shape[1]} columns")
    centred = values - mean
    return np.einsum('ij,jk,ik->i', centred, np.linalg.inv(covariance), centred)


def squared_prediction_error(scaled: np.ndarray, loadings: np.ndarray) -> np.ndarray:
    """Squared distance of each row from the subspace the retained loadings span."""
    if loadings.shape[0] != scaled.shape[1]:
        raise ValueError(f"loadings hold {loadings.shape[0]} variables but the data has {scaled.shape[1]}")
    reconstructed = scaled @ loadings @ loadings.T
    return ((scaled - reconstructed) ** 2).sum(axis=1)


def residual_limit(discarded_eigenvalues: np.ndarray, alpha: float = ALPHA) -> float:
    """Jackson and Mudholkar limit for the squared prediction error."""
    if discarded_eigenvalues.size == 0:
        raise ValueError("no components were discarded, so the residual carries no variation")
    theta = [float((discarded_eigenvalues ** power).sum()) for power in (1, 2, 3)]
    h0 = 1.0 - 2.0 * theta[0] * theta[2] / (3.0 * theta[1] ** 2)
    deviate = stats.norm.isf(alpha)
    bracket = (deviate * np.sqrt(2.0 * theta[1] * h0 ** 2) / theta[0]
               + 1.0 + theta[1] * h0 * (h0 - 1.0) / theta[0] ** 2)
    return float(theta[0] * bracket ** (1.0 / h0))


def false_alarm_frame(counts: tuple = SENSOR_COUNTS, alpha: float = UNIVARIATE_ALPHA) -> pd.DataFrame:
    """Chance that at least one of p independent univariate charts signals on a healthy process.

    Returns a pd.DataFrame indexed by 'n_sensors' with one column 'false_alarm_rate'.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must lie strictly between 0 and 1; got {alpha}")
    return pd.DataFrame({'false_alarm_rate': [1.0 - (1.0 - alpha) ** p for p in counts]},
                        index=pd.Index(counts, name='n_sensors'))


def draw_monitoring(data: pd.DataFrame, t2: np.ndarray, spe: np.ndarray, t2_limit: float,
                    spe_limit: float, output_path: pathlib.Path) -> pathlib.Path:
    """Draw the sensor scatter, the T2 chart and the residual chart, and save the figure."""
    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    columns = ['sensor_1', 'sensor_2']
    values = data[columns].to_numpy()
    healthy = values[~data['faulted'].to_numpy()]
    mean, covariance = healthy.mean(axis=0), np.cov(healthy, rowvar=False)
    figure, axes = plt.subplots(nrows=1, ncols=3, figsize=FIGSIZE)
    figure.subplots_adjust(bottom=0.26, wspace=0.34)
    axes[0].scatter(healthy[:, 0], healthy[:, 1], s=12, color=CURVE_COLORS[0], alpha=0.7)
    fault = values[data['faulted'].to_numpy()][0]
    axes[0].scatter([fault[0]], [fault[1]], s=90, facecolors='none', edgecolors=CURVE_COLORS[3],
                    linewidths=1.8, zorder=3)
    angles = np.linspace(0.0, 2.0 * np.pi, 400)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    circle = np.column_stack([np.cos(angles), np.sin(angles)]) * np.sqrt(eigenvalues * t2_limit)
    ellipse = circle @ eigenvectors.T + mean
    axes[0].plot(ellipse[:, 0], ellipse[:, 1], color=CURVE_COLORS[2], linewidth=1.6)
    for index, axis_line in enumerate((axes[0].axvline, axes[0].axhline)):
        spread = np.sqrt(covariance[index, index])
        for sign in (-3.0, 3.0):
            axis_line(mean[index] + sign * spread, color='black', linestyle='--', linewidth=0.9)
    axes[0].set_xlabel('Sensor 1', fontsize=font_size)
    axes[0].set_ylabel('Sensor 2', fontsize=font_size)
    axes[0].tick_params(labelsize=font_size * 0.85)
    axes[0].grid(visible=True, alpha=0.25)
    for axis, series, limit, name in ((axes[1], t2, t2_limit, 'Hotelling $T^2$'),
                                      (axes[2], spe, spe_limit, 'SPE')):
        axis.plot(data['sample'], series, color=CURVE_COLORS[0], linewidth=1.0, marker='o',
                  markersize=2.5)
        beyond = series > limit
        axis.scatter(data.loc[beyond, 'sample'], series[beyond], s=55, facecolors='none',
                     edgecolors=CURVE_COLORS[3], linewidths=1.4, zorder=3)
        axis.axhline(limit, color='black', linestyle='--', linewidth=0.9)
        axis.set_yscale('log')
        axis.set_xlabel('Sample', fontsize=font_size)
        axis.set_ylabel(name, fontsize=font_size)
        axis.tick_params(labelsize=font_size * 0.85)
        axis.grid(visible=True, alpha=0.25, which='both')
    for axis, label in zip(axes, ('(a)', '(b)', '(c)')):
        position = axis.get_position()
        figure.text(position.x0 + position.width / 2.0, 0.055, label, ha='center', va='center',
                    fontsize=font_size)
    figure.savefig(output_path, dpi=DPI)
    plt.close(figure)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f'{pathlib.Path(__file__).name} {__version__}\n'
                    'Draw the multivariate monitoring figure and write the tables the document quotes.',
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
    data = reference_data()
    columns = ['sensor_1', 'sensor_2']
    values = data[columns].to_numpy()
    healthy = values[~data['faulted'].to_numpy()]
    mean, covariance = healthy.mean(axis=0), np.cov(healthy, rowvar=False)
    t2 = hotelling_t2(values=values, mean=mean, covariance=covariance)

    scaled = (values - healthy.mean(axis=0)) / healthy.std(axis=0, ddof=1)
    correlation = np.corrcoef(healthy, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    loadings = eigenvectors[:, :N_COMPONENTS]
    spe = squared_prediction_error(scaled=scaled, loadings=loadings)

    n_variables = len(columns)
    n_healthy = len(healthy)
    factor = n_variables * (n_healthy + 1) * (n_healthy - 1) / (n_healthy * (n_healthy - n_variables))
    t2_limit = factor * stats.f.isf(ALPHA, n_variables, n_healthy - n_variables)
    spe_limit = residual_limit(discarded_eigenvalues=eigenvalues[N_COMPONENTS:], alpha=ALPHA)

    alarms = false_alarm_frame()
    data.assign(t2=t2, spe=spe).to_csv(args.output_folder / 'monitoring_statistics.csv', index=False)
    alarms.to_csv(args.output_folder / 'false_alarm_rate.csv')
    figure_path = draw_monitoring(data=data, t2=t2, spe=spe, t2_limit=t2_limit, spe_limit=spe_limit,
                                  output_path=args.output_folder / 'multivariate_spc.png')
    print(f'figure  {figure_path}')
    fault = data['faulted'].to_numpy()
    spread = healthy.std(axis=0, ddof=1)
    print(f'fault sample {int(data.loc[fault, "sample"].iloc[0])}: '
          f'sensor_1 z={(values[fault][0][0] - mean[0]) / spread[0]:+.2f}, '
          f'sensor_2 z={(values[fault][0][1] - mean[1]) / spread[1]:+.2f}')
    print(f'eigenvalues of the correlation matrix: {np.round(eigenvalues, 4)}')
    print(f'T2  limit {t2_limit:.3f}  fault value {t2[fault][0]:.3f}  signals {int((t2 > t2_limit).sum())}')
    print(f'SPE limit {spe_limit:.4f}  fault value {spe[fault][0]:.4f}  signals {int((spe > spe_limit).sum())}')
    print(alarms.round(4).to_string())
