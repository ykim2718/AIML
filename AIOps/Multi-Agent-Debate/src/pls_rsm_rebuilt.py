"""Rebuild of the PLS second-order response surface, written from the final spec of the spec debate only.

Changelog:
- 0.0.0+20261118: initial release
"""

__author__ = 'yRocket'
__version__ = "0.0.0+20261118"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import itertools
import math
import sys
import warnings

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict

__all__ = [
    'expand_quadratic',
    'stationary_point',
    'fit_surface',
]

DEFAULT_MAX_COMPONENTS: int = 9
DEFAULT_N_SPLITS: int = 5
FOLD_RANDOM_STATE: int = 0


def expand_quadratic(x: np.ndarray) -> tuple:
    """FR-1: return the expanded matrix (linear, squares, then x_i x_j for i < j) and its column names."""
    if x.ndim != 2:
        raise ValueError(f"x must be a 2-D array of shape (n, k), got shape {x.shape}.")
    n_factors = x.shape[1]
    linear_names = [f"x{i + 1}" for i in range(n_factors)]
    square_names = [f"x{i + 1}^2" for i in range(n_factors)]
    pairs = list(itertools.combinations(range(n_factors), 2))
    pair_names = [f"x{i + 1}x{j + 1}" for i, j in pairs]
    columns = [x, x ** 2]
    if pairs:
        columns.append(np.column_stack([x[:, i] * x[:, j] for i, j in pairs]))
    return np.hstack(columns), linear_names + square_names + pair_names


def stationary_point(b: np.ndarray, matrix: np.ndarray) -> tuple:
    """FR-7: return the stationary point -0.5 B^-1 b and the ascending eigenvalues of B; a singular B raises."""
    eigenvalues = np.linalg.eigvalsh(matrix)
    largest = np.max(np.abs(eigenvalues))
    if np.min(np.abs(eigenvalues)) < np.finfo(float).eps * max(1.0, largest):
        raise np.linalg.LinAlgError("B is singular; the surface has no isolated stationary point.")
    return -0.5 * np.linalg.solve(matrix, b), eigenvalues


def _surface_terms(coefficients: np.ndarray, names: list, n_factors: int) -> tuple:
    """FR-6: split the expanded coefficients into the linear vector b and the symmetric matrix B."""
    position = {name: index for index, name in enumerate(names)}
    b = np.array([coefficients[position[f"x{i + 1}"]] for i in range(n_factors)])
    matrix = np.zeros((n_factors, n_factors))
    for i in range(n_factors):
        matrix[i, i] = coefficients[position[f"x{i + 1}^2"]]
    for i, j in itertools.combinations(range(n_factors), 2):
        matrix[i, j] = matrix[j, i] = 0.5 * coefficients[position[f"x{i + 1}x{j + 1}"]]
    return b, matrix


def fit_surface(x: np.ndarray, y: np.ndarray, max_components: int = DEFAULT_MAX_COMPONENTS,
                n_splits: int = DEFAULT_N_SPLITS) -> dict:
    """FR-2 to FR-7: choose the PLS component count by cross-validated RMSE, refit, and locate the stationary point.

    Returns a dict with keys coefficients, n_components, cv_rmse, b, B, stationary_point and eigenvalues.
    """
    if max_components < 1:
        raise ValueError(f"max_components must be at least 1, got {max_components}.")
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}.")
    if y.shape != (x.shape[0],):
        raise ValueError(f"y must have shape ({x.shape[0]},), got {y.shape}.")
    x_expanded, names = expand_quadratic(x=x)
    n_rows = x_expanded.shape[0]
    folds = KFold(n_splits=min(n_splits, n_rows), shuffle=True, random_state=FOLD_RANDOM_STATE)
    held_out = math.ceil(n_rows / folds.get_n_splits())
    rank = int(np.linalg.matrix_rank(x_expanded - x_expanded.mean(axis=0)))
    upper = min(max_components, rank, n_rows - held_out - 1)
    if upper < 1:
        raise ValueError(f"no component count is available: upper count {upper} for {n_rows} rows of rank {rank}.")
    cv_rmse = []
    for count in range(1, upper + 1):
        with warnings.catch_warnings():
            # Non-finite predictions are detected below and end the search, so the warning itself adds nothing.
            warnings.simplefilter('ignore', category=RuntimeWarning)
            prediction = cross_val_predict(PLSRegression(n_components=count), x_expanded, y, cv=folds).ravel()
        if not np.all(np.isfinite(prediction)):
            print(f"non-finite prediction at {count} components; search stops at {count - 1}.", file=sys.stderr)
            break
        cv_rmse.append(float(np.sqrt(np.mean((y - prediction) ** 2))))
    if not cv_rmse:
        raise ValueError(f"no usable component count for {n_rows} rows.")
    n_components = int(np.argmin(cv_rmse)) + 1
    coefficients = PLSRegression(n_components=n_components).fit(x_expanded, y).coef_.ravel()
    b, matrix = _surface_terms(coefficients=coefficients, names=names, n_factors=x.shape[1])
    point, eigenvalues = stationary_point(b=b, matrix=matrix)
    return {
        'coefficients': coefficients,
        'n_components': n_components,
        'cv_rmse': cv_rmse,
        'b': b,
        'B': matrix,
        'stationary_point': point,
        'eigenvalues': eigenvalues,
    }
