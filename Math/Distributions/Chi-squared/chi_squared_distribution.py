#!/usr/bin/env python3
"""Draw the chi-squared density and distribution functions and tabulate their upper-tail quantiles.

The figure and the tables that the accompanying document quotes are both produced here, so the
numbers in the text and the curve in the picture come from one run.
"""
__author__ = 'yRocket'
__version__ = "0.0.0.2026.9.3"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from scipy import stats

__all__ = ['curve_frame', 'quantile_frame', 'moment_frame', 'draw_distribution']

FIGSIZE: tuple = (11.0, 4.6)
REFERENCE_WIDTH: float = 11.0    # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 10.0
DPI: int = 300
DEGREES_OF_FREEDOM: tuple = (1, 2, 3, 5, 10)
TABLE_DEGREES: tuple = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30)
UPPER_TAIL: tuple = (0.10, 0.05, 0.01)
X_MAX: float = 20.0
X_POINTS: int = 2000
PDF_Y_LIMIT: float = 0.55        # the k=1 density diverges at zero, so the axis is capped for display
CURVE_COLORS: tuple = tuple(TABLEAU_COLORS.values())


def curve_frame(degrees: tuple = DEGREES_OF_FREEDOM, x_max: float = X_MAX,
                x_points: int = X_POINTS) -> pd.DataFrame:
    """Density and distribution values on a common grid.

    Returns a pd.DataFrame with a RangeIndex and columns 'x', 'pdf_k<d>' and 'cdf_k<d>' for every d
    in degrees. Values are unclipped, so the k=1 density near zero is far above the plotted axis.
    """
    if x_max <= 0:
        raise ValueError(f"x_max must be positive; got {x_max}")
    if any(d < 1 for d in degrees):
        raise ValueError(f"degrees of freedom must be at least 1; got {degrees}")
    x = np.linspace(x_max / x_points, x_max, x_points)
    frame = pd.DataFrame({'x': x})
    for d in degrees:
        frame[f'pdf_k{d}'] = stats.chi2.pdf(x, df=d)
        frame[f'cdf_k{d}'] = stats.chi2.cdf(x, df=d)
    return frame


def quantile_frame(degrees: tuple = TABLE_DEGREES, upper_tail: tuple = UPPER_TAIL) -> pd.DataFrame:
    """Upper-tail critical values.

    Returns a pd.DataFrame indexed by 'degrees_of_freedom' with one column 'upper_<alpha>' per alpha
    in upper_tail, each holding the value the statistic exceeds with probability alpha.
    """
    if any(not 0.0 < a < 1.0 for a in upper_tail):
        raise ValueError(f"every upper tail probability must lie strictly between 0 and 1; got {upper_tail}")
    frame = pd.DataFrame(index=pd.Index(degrees, name='degrees_of_freedom'))
    for alpha in upper_tail:
        frame[f'upper_{alpha}'] = stats.chi2.isf(alpha, df=list(degrees))
    return frame


def moment_frame(degrees: tuple = DEGREES_OF_FREEDOM) -> pd.DataFrame:
    """Moments of the distribution, taken from scipy rather than from the closed forms.

    Returns a pd.DataFrame indexed by 'degrees_of_freedom' with columns 'mean', 'variance',
    'skewness' and 'excess_kurtosis'.
    """
    mean, variance, skewness, kurtosis = stats.chi2.stats(df=list(degrees), moments='mvsk')
    return pd.DataFrame({'mean': mean, 'variance': variance, 'skewness': skewness,
                         'excess_kurtosis': kurtosis},
                        index=pd.Index(degrees, name='degrees_of_freedom'))


def draw_distribution(curves: pd.DataFrame, degrees: tuple, output_path: pathlib.Path) -> pathlib.Path:
    """Draw the density and the distribution function side by side and save the figure."""
    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    colors = CURVE_COLORS[:len(degrees)]
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=FIGSIZE)
    figure.subplots_adjust(bottom=0.22)  # reserve room below the axis labels for the panel labels
    for axis, kind, y_label in zip(axes, ('pdf', 'cdf'), ('Density', 'Cumulative probability')):
        for d, color in zip(degrees, colors):
            axis.plot(curves['x'], curves[f'{kind}_k{d}'], color=color, linewidth=1.4, label=f'k = {d}')
        axis.set_xlim(0.0, curves['x'].max())
        axis.set_xlabel('x', fontsize=font_size)
        axis.set_ylabel(y_label, fontsize=font_size)
        axis.tick_params(labelsize=font_size * 0.9)
        axis.legend(fontsize=font_size * 0.9, frameon=False)
        axis.grid(visible=True, alpha=0.25)
    axes[0].set_ylim(0.0, PDF_Y_LIMIT)
    axes[1].set_ylim(0.0, 1.02)
    for axis, label in zip(axes, ('(a)', '(b)')):
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
                    'Draw the chi-squared density and distribution functions and write the tables '
                    'the document quotes.',
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
    curves = curve_frame()
    quantiles = quantile_frame()
    moments = moment_frame()
    curves.to_csv(args.output_folder / 'chi_squared_curves.csv', index=False)
    quantiles.to_csv(args.output_folder / 'chi_squared_quantiles.csv')
    moments.to_csv(args.output_folder / 'chi_squared_moments.csv')
    figure_path = draw_distribution(curves=curves, degrees=DEGREES_OF_FREEDOM,
                                    output_path=args.output_folder / 'chi_squared_distribution.png')
    print(f'figure  {figure_path}')
    print(moments.round(4).to_string())
    print(quantiles.round(3).to_string())
