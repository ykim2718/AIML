"""Hampel identifier: outlier scoring from the median and the median absolute deviation.

The classical z-score divides a deviation from the mean by the standard deviation, and an outlier
contaminates both. This module replaces the pair with the median and the MAD, which a minority of
contaminating observations cannot move. Scoring and deciding are separate: hampel_score computes
the modified z-score of every observation and takes no threshold, and the caller compares the
absolute score against one. Tabulating and printing the scores belong to the caller as well, so
the module offers no command line interface.

Changelog:
    0.9.0 - Move score_frame to the caller that needs it and drop the command line interface.
    0.8.0 - Keep __all__ to the scoring surface so every exported name is defined beside it.
    0.7.0 - Declare __all__ so `from hampel_identifier import *` exports only the documented surface.
    0.6.0 - Make the scale private as _hampel_scale; callers use the score or the interval.
    0.5.0 - Drop classical_z_scores and max_attainable_z, and the column they fed.
    0.4.0 - Replace HampelResult with hampel_score, hampel_scale and retained_interval.
    0.3.0 - Rename the HampelResult field modified_z to modified_z_scores.
    0.2.0 - Rename the HampelResult field scores to modified_z.
    0.1.0 - Drop threshold_sweep; the figure it fed is no longer part of the document.
    0.0.0 - Initial release.
"""

__author__ = 'yRocket'
__version__ = "0.9.1.2026.8.18"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)

# Everything this module offers. The names beginning with an underscore are internal.
__all__ = [
    'NORMAL_QUARTILE',
    'DEFAULT_THRESHOLD',
    'median_absolute_deviation',
    'hampel_score',
    'retained_interval',
]

import numpy as np
from scipy import stats

# The MAD of a normal sample converges to this multiple of sigma, so dividing by it puts the
# modified score on the same scale as a classical z-score. It is the third quartile of N(0, 1).
NORMAL_QUARTILE = float(stats.norm.ppf(0.75))

# The cut-off recommended by Iglewicz and Hoaglin for the modified z-score.
DEFAULT_THRESHOLD = 3.5


def _as_sample(data: np.ndarray = None) -> np.ndarray:
    """Return the sample as a 1-D float array, refusing input the identifier cannot read."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {values.shape}.")
    if values.size < 3:
        raise ValueError(f"the identifier needs at least 3 observations to have a usable median, got {values.size}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"data carries {int((~np.isfinite(values)).sum())} non-finite values; "
                         f"remove or impute them before scoring.")
    return values


def median_absolute_deviation(data: np.ndarray = None) -> float:
    """Median of the absolute deviations from the sample median.

    Args:
        data: the sample, shape (n,).

    Returns:
        The MAD, before any rescaling.
    """
    values = np.asarray(data, dtype=float)
    return float(np.median(np.abs(values - np.median(values))))


def _hampel_scale(data: np.ndarray = None, quartile: float = NORMAL_QUARTILE) -> float:
    """Robust scale of the sample, which is the MAD divided by the consistency constant.

    Args:
        data: the sample, shape (n,).
        quartile: the constant the MAD is divided by; the default makes the scale estimate the
            standard deviation on a normal sample.

    Returns:
        The robust scale.
    """
    values = _as_sample(data=data)
    if quartile <= 0.0:
        raise ValueError(f"quartile must be positive, got {quartile}.")
    mad = median_absolute_deviation(data=values)
    if mad == 0.0:
        # More than half the sample sits on one value, so the MAD carries no scale at all.
        # Returning zero here would make every score infinite or undefined further down.
        centre = float(np.median(values))
        repeated = int((values == centre).sum())
        raise ValueError(f"the MAD is 0 because {repeated} of {values.size} observations equal the median "
                         f"{centre}; the robust scale is undefined. Every 50% breakdown estimator returns 0 "
                         f"here, Sn and Qn included, so report that the sample cannot support the method or "
                         f"take a scale that does not rest on a median of the deviations.")
    return mad / quartile


def hampel_score(data: np.ndarray = None, quartile: float = NORMAL_QUARTILE) -> np.ndarray:
    """Modified z-score of every observation, which is its distance from the median in robust scales.

    No threshold is taken, because the score does not depend on one. A caller decides by comparing
    the absolute score against a cut-off, 3.5 by convention.

    Args:
        data: the sample, shape (n,).
        quartile: the constant the MAD is divided by, as in median_absolute_deviation.

    Returns:
        The modified z-scores, shape (n,).
    """
    values = _as_sample(data=data)
    return (values - np.median(values)) / _hampel_scale(data=values, quartile=quartile)


def retained_interval(data: np.ndarray = None, threshold: float = DEFAULT_THRESHOLD,
                      quartile: float = NORMAL_QUARTILE) -> tuple[float, float]:
    """The decision rule written in the units of the data rather than in scales.

    An observation inside the interval is retained and an observation outside it is flagged.

    Args:
        data: the sample, shape (n,).
        threshold: the cut-off on the absolute modified z-score.
        quartile: the constant the MAD is divided by, as in median_absolute_deviation.

    Returns:
        The lower and the upper end of the interval.
    """
    values = _as_sample(data=data)
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}.")
    centre = float(np.median(values))
    half_width = threshold * _hampel_scale(data=values, quartile=quartile)
    return centre - half_width, centre + half_width
