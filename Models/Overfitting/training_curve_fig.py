"""Draw the concept of training and validation accuracy against iterations.

The training-set curve rises and levels off near its ceiling, while the validation-set curve rises, peaks
and then falls. The curves are drawn from closed-form shapes, not from a fitted model, and the axes carry
no ticks because only the shape is meant to be read.
"""
__author__ = 'yRocket'
__version__ = "0.1.0+20261101"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ['concept_curves', 'draw_training_curve']

FIGSIZE: tuple = (7.0, 5.1)
REFERENCE_WIDTH: float = 7.0         # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 14.0
COLORS: dict = matplotlib.colors.TABLEAU_COLORS
TRAINING_COLOR: str = COLORS['tab:blue']
VALIDATION_COLOR: str = COLORS['tab:red']

# Shapes in axis units, with iterations t in [0, 1] and accuracy in [0, 1].
TRAINING_START: float = 0.22
TRAINING_RISE: float = 0.75
TRAINING_SCALE: float = 0.25         # iterations over which the training curve closes 63 % of its rise
VALIDATION_PEAK_AT: float = 0.65
VALIDATION_PEAK: float = 0.57
VALIDATION_START: float = 0.13
VALIDATION_END: float = 0.35


def concept_curves(n_points: int = 400) -> pd.DataFrame:
    """Evaluate the two concept curves on evenly spaced iterations.

    Returns a DataFrame indexed by 'iteration' (in [0, 1]) with columns 'training' and 'validation'.
    """
    if n_points < 2:
        raise ValueError(f"n_points must be at least 2, got {n_points}")
    t = np.linspace(0.0, 1.0, n_points)
    training = TRAINING_START + TRAINING_RISE * (1.0 - np.exp(-t / TRAINING_SCALE))

    # A cubic in u = t - peak with zero slope at the peak, passing through the start and end values.
    u_start, u_end = -VALIDATION_PEAK_AT, 1.0 - VALIDATION_PEAK_AT
    system = np.array([[u_start ** 2, u_start ** 3], [u_end ** 2, u_end ** 3]])
    target = np.array([VALIDATION_START - VALIDATION_PEAK, VALIDATION_END - VALIDATION_PEAK])
    quadratic, cubic = np.linalg.solve(system, target)
    u = t - VALIDATION_PEAK_AT
    validation = VALIDATION_PEAK + quadratic * u ** 2 + cubic * u ** 3
    return pd.DataFrame({'training': training, 'validation': validation},
                        index=pd.Index(t, name='iteration'))


def draw_training_curve(curve: pd.DataFrame, output_folder: pathlib.Path) -> pathlib.Path:
    """Write the concept figure. Returns the figure path."""
    output_folder.mkdir(parents=True, exist_ok=True)
    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH

    fig, axis = plt.subplots(figsize=FIGSIZE)
    axis.set_xlim(-0.12, 1.29)
    axis.set_ylim(-0.17, 1.17)
    axis.set_axis_off()

    # Two crossing axis lines, as in a hand-drawn sketch: each runs a little past the other.
    axis.plot([0.0, 0.0], [-0.12, 1.0], color='black', linewidth=1.5)
    axis.plot([-0.05, 1.23], [0.0, 0.0], color='black', linewidth=1.5)
    axis.text(-0.11, 1.07, 'Accuracy', fontsize=font_size * 1.5, ha='left', va='center')
    axis.text(1.23, -0.10, 'Iterations', fontsize=font_size * 1.5, ha='right', va='center')

    t = curve.index.to_numpy()
    axis.plot(t, curve['training'], color=TRAINING_COLOR, linewidth=3.0)
    axis.plot(t, curve['validation'], color=VALIDATION_COLOR, linewidth=3.0)
    axis.text(0.92, curve['training'].iloc[-1] + 0.05, 'training-set', color=TRAINING_COLOR,
              fontsize=font_size, ha='center', va='bottom')
    axis.text(1.25, VALIDATION_PEAK - 0.06, 'validation-set', color=VALIDATION_COLOR,
              fontsize=font_size, ha='right', va='center')

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
    figure_path = output_folder / 'training-curve.png'
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)
    print(f"wrote {figure_path}")
    return figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f"{pathlib.Path(__file__).name} {__version__}\n"
                    f"Draw the concept of training and validation accuracy against iterations.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--output-folder', type=pathlib.Path, required=True,
                        help="folder the figure is written to")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


if __name__ == '__main__':
    arguments = parse_args()
    draw_training_curve(curve=concept_curves(), output_folder=arguments.output_folder)
