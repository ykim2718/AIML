"""Draw training and validation accuracy against boosting iterations.

A gradient boosting classifier is fitted on a sample whose labels are partly reassigned at random, and its accuracy is
read after every iteration on the training half and on the held-out validation half. Training accuracy keeps
rising toward 1 while validation accuracy peaks and then falls, which marks where the model starts to fit
the reassigned labels. The two curves are written to CSV so the text can quote them.
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
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

__all__ = ['accuracy_by_iteration', 'draw_training_curve']

FIGSIZE: tuple = (7.0, 4.2)
REFERENCE_WIDTH: float = 7.0         # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 11.0
COLORS: list = list(matplotlib.colors.TABLEAU_COLORS.values())


def accuracy_by_iteration(n_samples: int, flip_fraction: float, n_iterations: int, seed: int) -> pd.DataFrame:
    """Fit gradient boosting and read the accuracy after each iteration.

    Returns a DataFrame indexed by 'iteration' (starting at 1) with columns 'training' and 'validation'.
    """
    if n_samples < 20:
        raise ValueError(f"n_samples must be at least 20, got {n_samples}")
    if not 0 <= flip_fraction < 0.5:
        raise ValueError(f"flip_fraction must be in [0, 0.5), got {flip_fraction}")
    x, y = make_classification(n_samples=n_samples, n_features=20, n_informative=4, flip_y=flip_fraction,
                               random_state=seed)
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.5, random_state=seed)
    model = GradientBoostingClassifier(n_estimators=n_iterations, learning_rate=0.1, max_depth=4,
                                       random_state=seed)
    model.fit(x_train, y_train)
    frame = pd.DataFrame({
        'training': [float(np.mean(p == y_train)) for p in model.staged_predict(x_train)],
        'validation': [float(np.mean(p == y_valid)) for p in model.staged_predict(x_valid)],
    })
    frame.index = pd.RangeIndex(start=1, stop=len(frame) + 1, name='iteration')
    return frame


def draw_training_curve(curve: pd.DataFrame, output_folder: pathlib.Path) -> pathlib.Path:
    """Write the figure and the curve it was drawn from. Returns the figure path."""
    output_folder.mkdir(parents=True, exist_ok=True)
    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    plt.rcParams.update({'font.size': font_size})

    best = int(curve['validation'].idxmax())
    fig, axis = plt.subplots(figsize=FIGSIZE)
    axis.plot(curve.index, curve['training'], color=COLORS[0], linewidth=2.0, label='training set')
    axis.plot(curve.index, curve['validation'], color=COLORS[3], linewidth=2.0, label='validation set')
    axis.axvline(best, color='0.4', linestyle='--', linewidth=1.0)
    label_height = (curve['validation'].max() + curve['training'].min()) / 2
    axis.annotate(f"best validation\niteration {best}", xy=(best, label_height),
                  xytext=(6, 0), textcoords='offset points', color='0.3', fontsize=font_size * 0.9)
    axis.set_xlabel('iterations')
    axis.set_ylabel('accuracy')
    axis.set_xlim(0, curve.index.max())
    axis.spines[['top', 'right']].set_visible(False)
    axis.legend(frameon=False, loc='center right')
    fig.tight_layout()

    figure_path = output_folder / 'training-curve.png'
    fig.savefig(figure_path, dpi=300)
    plt.close(fig)

    curve.to_csv(output_folder / 'training-curve.csv')
    print(f"wrote {figure_path}")
    print(curve.loc[[1, best, curve.index.max()]].round(3).to_string())
    return figure_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f"{pathlib.Path(__file__).name} {__version__}\n"
                    f"Draw training and validation accuracy against boosting iterations.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--output-folder', type=pathlib.Path, required=True,
                        help="folder the figure and its curve are written to")
    parser.add_argument('--n-samples', type=int, default=600, help="rows, split half for training")
    parser.add_argument('--flip-fraction', type=float, default=0.25, help="fraction of labels reassigned at random")
    parser.add_argument('--n-iterations', type=int, default=300, help="boosting iterations")
    parser.add_argument('--seed', type=int, default=1, help="seed of the sample and the model")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    if args.n_samples < 20:
        parser.error(f"--n-samples must be at least 20, got {args.n_samples}")
    if not 0 <= args.flip_fraction < 0.5:
        parser.error(f"--flip-fraction must be in [0, 0.5), got {args.flip_fraction}")
    if args.n_iterations < 2:
        parser.error(f"--n-iterations must be at least 2, got {args.n_iterations}")
    return args


if __name__ == '__main__':
    arguments = parse_args()
    curve_frame = accuracy_by_iteration(n_samples=arguments.n_samples, flip_fraction=arguments.flip_fraction,
                                        n_iterations=arguments.n_iterations, seed=arguments.seed)
    draw_training_curve(curve=curve_frame, output_folder=arguments.output_folder)
