"""Rebuild of the PLS response surface, written from the draft spec v0 only, before the spec debate.

Where v0 leaves a choice open, this module takes the reading a coding agent would take from v0 alone:
- FR-1 does not fix the column order, so the expansion is sklearn PolynomialFeatures(degree=2).
- FR-2 does not fix the folds, the score or the search range, so it uses 5 unshuffled folds, the mean fold RMSE,
  and every count up to the number of expanded columns.
- FR-3 does not say how a cross-product coefficient enters B, so B_ij = B_ji = that coefficient.

Changelog:
- 0.0.0+20261118: initial release
"""

__author__ = 'yRocket'
__version__ = "0.0.0+20261118"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import PolynomialFeatures

__all__ = [
    'fit_surface',
]

N_FOLDS: int = 5


def fit_surface(x: np.ndarray, y: np.ndarray) -> dict:
    """Return the outputs v0 names: the stationary point, the eigenvalues of B and the component count."""
    if x.ndim != 2 or y.shape != (x.shape[0],):
        raise ValueError(f"x must be (n, k) and y (n,), got {x.shape} and {y.shape}.")
    expansion = PolynomialFeatures(degree=2, include_bias=False)
    x_expanded = expansion.fit_transform(x)
    scores = []
    for count in range(1, x_expanded.shape[1] + 1):
        fold_scores = cross_val_score(PLSRegression(n_components=count), x_expanded, y, cv=N_FOLDS,
                                      scoring='neg_root_mean_squared_error')
        scores.append(float(np.mean(fold_scores)))
    n_components = int(np.argmax(scores)) + 1
    coefficients = PLSRegression(n_components=n_components).fit(x_expanded, y).coef_.ravel()
    n_factors = x.shape[1]
    b = np.zeros(n_factors)
    matrix = np.zeros((n_factors, n_factors))
    for coefficient, powers in zip(coefficients, expansion.powers_):
        factors = np.flatnonzero(powers)
        if powers.sum() == 1:
            b[factors[0]] = coefficient
        elif len(factors) == 1:
            matrix[factors[0], factors[0]] = coefficient
        else:
            matrix[factors[0], factors[1]] = matrix[factors[1], factors[0]] = coefficient
    eigenvalues = np.linalg.eigvalsh(matrix)
    return {
        'n_components': n_components,
        'stationary_point': -0.5 * np.linalg.solve(matrix, b),
        'eigenvalues': eigenvalues,
    }
