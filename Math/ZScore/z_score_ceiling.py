"""Ceiling of the classical z-score: the largest score a sample of a given size can produce.

A classical z-score divides a deviation from the mean by a standard deviation the same
observation helped compute, so the score cannot run away from the sample it is taken in. The
largest absolute score is fixed by the sample size alone, and this module computes both the score
and that limit. Tabulating and drawing them belong to the caller.

Changelog:
    0.0.0 - Initial release.
"""

__author__ = 'yRocket'
__version__ = "0.0.0.2026.8.18"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

# Everything this module offers. The names beginning with an underscore are internal.
__all__ = [
    'DEFAULT_DDOF',
    'classical_z_scores',
    'max_attainable_z',
]

import numpy as np

# The divisor of the sum of squares is n - ddof. The sample standard deviation uses ddof = 1, and
# the ceiling below is written for it; ddof = 0 is the population form and has its own ceiling.
DEFAULT_DDOF = 1


def _as_ddof(ddof: int = DEFAULT_DDOF) -> int:
    """Return the divisor correction, refusing a value the ceiling is not defined for."""
    if ddof not in (0, 1):
        raise ValueError(f"ddof must be 0 or 1, got {ddof}.")
    return int(ddof)


def _as_sample(data: np.ndarray = None) -> np.ndarray:
    """Return the sample as a 1-D float array, refusing input a z-score cannot be taken in."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {values.shape}.")
    if values.size < 2:
        raise ValueError(f"a z-score needs at least 2 observations to have a scale, got {values.size}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"data carries {int((~np.isfinite(values)).sum())} non-finite values; "
                         f"remove or impute them before scoring.")
    return values


def classical_z_scores(data: np.ndarray = None, ddof: int = DEFAULT_DDOF) -> np.ndarray:
    """Deviation of every observation from the mean, divided by the standard deviation.

    Args:
        data: the sample, shape (n,).
        ddof: the divisor correction, 1 for the sample standard deviation and 0 for the
            population form.

    Returns:
        The classical z-scores, shape (n,).
    """
    values = _as_sample(data=data)
    correction = _as_ddof(ddof=ddof)
    scale = float(values.std(ddof=correction))
    if scale == 0.0:
        # Every observation equals the mean, so no observation is any distance from it at all.
        # Returning zeros would report a clean sample where the score is simply not defined.
        raise ValueError(f"the standard deviation is 0 because all {values.size} observations are "
                         f"{float(values[0])}; the classical score is undefined.")
    return (values - values.mean()) / scale


def max_attainable_z(size: int = None, ddof: int = DEFAULT_DDOF) -> float:
    """Largest absolute z-score a sample of the given size can produce, whatever its values.

    The bound is arithmetic rather than a property of any data, and it is attained: one
    observation set apart from size - 1 tied ones reaches it exactly.

    Args:
        size: the number of observations.
        ddof: the divisor correction, as in classical_z_scores.

    Returns:
        The ceiling on the absolute z-score.
    """
    correction = _as_ddof(ddof=ddof)
    if not isinstance(size, (int, np.integer)):
        raise ValueError(f"size must be an integer, got {type(size).__name__}.")
    if size < 2:
        raise ValueError(f"size must be at least 2, got {size}.")
    if correction == 1:
        return float((size - 1) / np.sqrt(size))
    return float(np.sqrt(size - 1))
