#!/usr/bin/env python3
"""Sweep the multioutput settings of sklearn.metrics.r2_score over one multioutput problem.

The tables the accompanying document quotes are produced here, so the numbers in the text and the
numbers a run prints come from the same place.
"""
__author__ = 'yRocket'
__version__ = "0.0.0.2026.9.5"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

__all__ = ['output_weights', 'multioutput_scores', 'constant_output_scores']

# two outputs on the same four samples: a wide one in tens, a narrow one about 1
Y_TRUE: np.ndarray = np.array([[10.0, 1.00], [20.0, 1.02], [30.0, 0.98], [40.0, 1.01]])
Y_PRED: np.ndarray = np.array([[12.0, 1.00], [19.0, 1.05], [31.0, 0.95], [39.0, 1.02]])
# the settings the document walks through, in the order it walks through them
SETTINGS: tuple = ('raw_values', 'uniform_average', 'variance_weighted', None, (1.0, 3.0))
# one output holds a constant, which is where force_finite decides what the score is
Y_TRUE_CONSTANT: np.ndarray = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
Y_PRED_CONSTANT: np.ndarray = np.array([[1.0, 5.1], [2.0, 5.0], [3.0, 5.0]])


def output_weights(y_true: np.ndarray = None, sample_weight: np.ndarray = None) -> np.ndarray:
    """The per-output weights that 'variance_weighted' averages with.

    They are the weighted total sums of squares of each output, which is what r2_score puts in the
    denominator of that output's score.
    """
    if y_true is None:
        raise ValueError('y_true is required.')
    y_true = np.asarray(y_true, dtype=float)
    if y_true.ndim != 2:
        raise ValueError(f"y_true must be two-dimensional, got shape {y_true.shape}.")
    weight = 1.0 if sample_weight is None else np.asarray(sample_weight, dtype=float)[:, None]
    return np.sum(weight * (y_true - np.average(y_true, axis=0, weights=sample_weight)) ** 2, axis=0)


def multioutput_scores(y_true: np.ndarray = None, y_pred: np.ndarray = None,
                       settings: tuple = SETTINGS, sample_weight: np.ndarray = None) -> pd.DataFrame:
    """The score r2_score returns under each multioutput setting.

    Returns a pd.DataFrame indexed by 'setting' with columns 'value', the score written out, and
    'shape', either 'scalar' or the length of the returned array.
    """
    if y_true is None or y_pred is None:
        raise ValueError('y_true and y_pred are both required.')
    if not settings:
        raise ValueError('settings is empty; there is nothing to sweep.')
    rows = []
    for setting in settings:
        # a tuple of weights has to reach r2_score as a list, which is what it accepts
        argument = list(setting) if isinstance(setting, tuple) else setting
        score = r2_score(y_true, y_pred, multioutput=argument, sample_weight=sample_weight)
        rows.append({'setting': repr(setting),
                     'value': np.array2string(np.asarray(score), precision=4),
                     'shape': 'scalar' if np.ndim(score) == 0 else str(np.size(score))})
    return pd.DataFrame(rows).set_index('setting')


def constant_output_scores(y_true: np.ndarray = None, y_pred: np.ndarray = None) -> pd.DataFrame:
    """The raw scores of a problem whose second output is constant, with force_finite both ways.

    Returns a pd.DataFrame indexed by 'force_finite' with one column per output, named 'output 0',
    'output 1' and so on.
    """
    if y_true is None or y_pred is None:
        raise ValueError('y_true and y_pred are both required.')
    rows = {}
    for force_finite in (True, False):
        scores = r2_score(y_true, y_pred, multioutput='raw_values', force_finite=force_finite)
        rows[force_finite] = np.asarray(scores)
    columns = [f'output {index}' for index in range(np.shape(y_true)[1])]
    return pd.DataFrame.from_dict(rows, orient='index', columns=columns).rename_axis('force_finite')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=pathlib.Path(__file__).name,
        description=f'{pathlib.Path(__file__).name} {__version__}\n'
                    'Sweep the multioutput settings of r2_score over the problem the document uses.',
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-v', '--version', action='version', version=__version__)
    parser.add_argument('--output-folder', type=pathlib.Path, required=True,
                        help='folder that receives the csv tables; created if absent')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    arguments = parser.parse_args()
    arguments.output_folder.mkdir(parents=True, exist_ok=True)
    return arguments


if __name__ == '__main__':
    args = parse_args()
    scores = multioutput_scores(y_true=Y_TRUE, y_pred=Y_PRED)
    weights = output_weights(y_true=Y_TRUE)
    constant = constant_output_scores(y_true=Y_TRUE_CONSTANT, y_pred=Y_PRED_CONSTANT)
    scores.to_csv(args.output_folder / 'multioutput_scores.csv')
    constant.to_csv(args.output_folder / 'constant_output_scores.csv')
    print(f'total sum of squares per output  {np.array2string(weights, precision=6)}')
    print(scores.to_string())
    print(constant.to_string())
