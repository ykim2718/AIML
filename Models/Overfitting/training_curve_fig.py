"""Draw the concept of training and validation accuracy against iterations.

The training-set curve rises and levels off near its ceiling, while the validation-set curve rises, peaks
and then falls. The curves are drawn from closed-form shapes, not from a fitted model, and the axes carry
no ticks because only the shape is meant to be read.
"""
__author__ = 'yRocket'
__version__ = "0.2.0+20261101"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import argparse
import pathlib
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ['concept_curves', 'draw_training_curve']

AXES_SIZE: tuple = (4.0, 3.0)        # x and y length of the chart's axis lines, in inches
MARGINS: dict = {'left': 0.45, 'right': 0.15, 'bottom': 0.50, 'top': 0.40}  # inches around the chart
DPI: int = 200
REFERENCE_AXES_WIDTH: float = 4.0    # the chart width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 9.0
X_AXIS_END: float = 1.23             # iterations where the x axis line ends; the curves end at 1
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
    font_size = BASE_FONT_SIZE * AXES_SIZE[0] / REFERENCE_AXES_WIDTH
    fig_width = MARGINS['left'] + AXES_SIZE[0] + MARGINS['right']
    fig_height = MARGINS['bottom'] + AXES_SIZE[1] + MARGINS['top']

    # The axes box spans exactly the two axis lines, so the chart is AXES_SIZE inches.
    fig = plt.figure(figsize=(fig_width, fig_height))
    axis = fig.add_axes((MARGINS['left'] / fig_width, MARGINS['bottom'] / fig_height,
                         AXES_SIZE[0] / fig_width, AXES_SIZE[1] / fig_height))
    axis.set_xlim(0.0, X_AXIS_END)
    axis.set_ylim(0.0, 1.0)
    axis.set_axis_off()

    # Two crossing axis lines, as in a hand-drawn sketch: each runs a little past the other.
    axis.plot([0.0, 0.0], [-0.12, 1.0], color='black', linewidth=1.2, clip_on=False)
    axis.plot([-0.05, X_AXIS_END], [0.0, 0.0], color='black', linewidth=1.2, clip_on=False)
    axis.text(-0.11, 1.07, 'Accuracy', fontsize=font_size * 1.5, ha='left', va='center')
    axis.text(X_AXIS_END, -0.10, 'Iterations', fontsize=font_size * 1.5, ha='right', va='center')

    t = curve.index.to_numpy()
    axis.plot(t, curve['training'], color=TRAINING_COLOR, linewidth=2.5)
    axis.plot(t, curve['validation'], color=VALIDATION_COLOR, linewidth=2.5)
    axis.text(0.92, curve['training'].iloc[-1] + 0.05, 'training-set', color=TRAINING_COLOR,
              fontsize=font_size, ha='center', va='bottom')
    axis.text(1.25, VALIDATION_PEAK - 0.06, 'validation-set', color=VALIDATION_COLOR,
              fontsize=font_size, ha='right', va='center')

    figure_path = output_folder / 'training-curve.png'
    fig.savefig(figure_path, dpi=DPI)
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
