# Hampel Identifier — Flagging Outliers with the Median and the MAD
Rev. 39 | Created: 2026-08-17 | Updated: 2026-09-04 20:10 UTC

> A note on the modified z-score built from the median and the median absolute deviation,
> organized as the score and then the robustness that the score rests on.

## 1. Scope

The classical way to call an observation extreme is to divide its deviation from the mean by
the standard deviation and compare the result against a threshold. Both the centre and the scale
in that ratio are computed from the same sample the observation belongs to, so an outlier
inflates the quantity it is being measured against.

The Hampel identifier replaces the pair with the median and the median absolute deviation, the
MAD. A minority of contaminating observations cannot move either, so the scale keeps describing
the bulk of the sample rather than the observation under test.

## 2. Modified z-score

### 2.1. Definition

$$\mathrm{MAD} = \mathrm{median}\left( \left| x_1 - \tilde{x} \right|, \ldots, \left| x_n - \tilde{x} \right| \right)$$

$$M_i = \frac{x_i - \tilde{x}}{\mathrm{MAD} / \Phi^{-1}(0.75)}$$

- $x_1, \ldots, x_n$ — the sample, and $x_i$ its $i$-th observation.
- $\tilde{x}$ (x tilde) — the median of the sample, which the deviations are taken from and which the score is centred on.
- $\mathrm{MAD}$ — the median of those absolute deviations, which is the raw robust scale before any rescaling.
- $\Phi^{-1}(0.75) = 0.674490$ — the third quartile of the standard normal distribution, which the MAD is divided by. It is the subject of section 2.2.
- $M_i$ — the modified z-score of observation $i$.

The denominator is the robust scale. The score $M_i$ is an ordinary z-score with the median put in
place of the mean and the robust scale in place of the standard deviation, which is why it is
called the modified z-score.

### 2.2. Consistency Constant

For a normal sample the MAD converges to $\Phi^{-1}(0.75) \sigma$ rather than to $\sigma$, so
the raw MAD understates the spread by about a third. Dividing by $\Phi^{-1}(0.75)$, equivalently
multiplying by 1.482602, removes that bias in the limit and makes the modified score comparable
to a classical z-score. Without it the score would sit on a scale of its own and no threshold
could be carried over from the classical rule. A short sample still lands below the limit, which
is the second thing the appendix works out.
[Appendix D. Consistency Constant of the MAD](#appendix-d-consistency-constant-of-the-mad) derives
the number and shows what a short sample does to it.

The constant is the only place normality enters the method, and it is a calibration rather than
an assumption: changing it rescales every score by the same factor and reorders nothing. Its
purpose is to let the threshold of section 2.3 be read as a false-positive rate.

### 2.3. Outlier Flag

An observation is flagged as an outlier when its modified z-score exceeds the threshold $k$ in
absolute value.

$$\left| M_i \right| \gt k$$

The threshold $k$ is 3.5 by convention. The value comes from Iglewicz and Hoaglin (1993), who set
it high enough that a sample which really is normal almost never produces a flag. On such a
sample the expected number of false flags in fifteen observations is 0.007. A threshold of 3.0
is also used and is roughly six times looser. Fix the threshold before looking at the data, and
report it, because a verdict that changes between 3.0 and 3.5 rests on the choice rather than on
the sample.

Every observation is scored once, against one centre and one scale. There is no iteration and
nothing is removed part-way, because the estimates the scores are built on were never
contaminated in the first place.

### 2.4. Retained Interval

The same rule can be written in the units of the data rather than in scales. Multiplying the
inequality through by the robust scale turns it into an interval around the median.

$$\tilde{x} - k \cdot \frac{\mathrm{MAD}}{\Phi^{-1}(0.75)} \le x_i \le \tilde{x} + k \cdot \frac{\mathrm{MAD}}{\Phi^{-1}(0.75)}$$

An observation inside that interval is retained and an observation outside it is flagged. The
interval is therefore where the rule draws its boundary on the measurement scale itself.

The interval is not fixed by the method. The median and the scale are both computed from the
sample, so two samples tested at the same threshold have different intervals.

## 3. Robustness

### 3.1. Assumptions

The sample is univariate. Contamination is a minority: fewer than half the observations are
outliers.

No value may be taken by more than half the sample. If one is, the MAD is 0, the score divides by
zero, and it is undefined. This is a hard failure rather than a loss of accuracy, and an
implementation that returns zeros instead of raising an error reports a clean sample on data it
cannot read. Reaching for a different robust estimator does not help: Sn and Qn have the same 50%
breakdown point and return 0 on exactly the same samples. A positive scale can only come from an
estimator that does not rest on a median of the deviations, such as the interquartile range,
which buys the scale by dropping to a 25% breakdown point.

### 3.2. Contaminated Scale

A sample standard deviation is a mean of squared deviations, so a single distant observation
enters it quadratically. Pushing that observation further out enlarges its own deviation and the
standard deviation together, so the ratio of the two grows far more slowly than the distance does.

The median absolute deviation is a median of absolute deviations. Sending one observation off to
infinity moves the middle of that list by one position and no further.

The same self-reference, an observation inflating the scale that it is measured against, caps the
classical score outright: in a sample of size $n$ no absolute z-score can exceed $(n-1)/\sqrt{n}$,
whatever the data are, a bound due to Shiffler (1988). The modified score has no such ceiling,
because its denominator stops responding to the observation in its numerator once that
observation is far enough out.

### 3.3. Breakdown Point

One corrupted observation is enough to break the mean and the standard deviation. Breaking the
median or the MAD takes more than half the sample.

An estimate breaks down when it stops describing the sample and describes the corrupted
observations instead. The calculation still returns a number. Setting one observation of the
worked example to a billion takes the mean past 66 million while the median stays at 0.0232, and
a mean of 66 million says nothing about the fourteen observations that still sit near 0.02.

The breakdown point is the fraction of the sample that has to be corrupted for that to happen.

**Table 1. Breakdown point of the two pairs**

| Estimator | Breakdown point | What it takes to break it |
|---|---|---|
| Mean and standard deviation | 0% | One observation is enough, if it is moved far enough. |
| Median and MAD | 50% | Eight of the fifteen are needed. Seven leave the median at 0.0403, which is still one of the real observations. |

The breakdown point is the whole reason the method works. Section 2 is nothing more than the
arithmetic that puts the robust pair on a scale a threshold can be read against.

## References

<a id="ref-1"></a>
[1] Hampel, F. R. (1974). [The Influence Curve and Its Role in Robust Estimation](https://doi.org/10.1080/01621459.1974.10482962). *Journal of the American Statistical Association*, 69(346), 383–393.<br>
<a id="ref-2"></a>
[2] Iglewicz, B., & Hoaglin, D. C. (1993). *How to Detect and Handle Outliers*. The ASQC Basic References in Quality Control: Statistical Techniques, Vol. 16. ASQC Quality Press, Milwaukee. [https://asq.org/quality-press](https://asq.org/quality-press). ISBN 978-0-87389-247-6.<br>
<a id="ref-3"></a>
[3] Shiffler, R. E. (1988). [Maximum Z Scores and Outliers](https://doi.org/10.1080/00031305.1988.10475530). *The American Statistician*, 42(1), 79–80.<br>
<a id="ref-4"></a>
[4] Rousseeuw, P. J., & Croux, C. (1993). [Alternatives to the Median Absolute Deviation](https://doi.org/10.1080/01621459.1993.10476408). *Journal of the American Statistical Association*, 88(424), 1273–1283.<br>
<a id="ref-5"></a>
[5] Leys, C., Ley, C., Klein, O., Bernard, P., & Licata, L. (2013). [Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median](https://doi.org/10.1016/j.jesp.2013.03.013). *Journal of Experimental Social Psychology*, 49(4), 764–766.

---

## Appendix A. Terminology

- **breakdown point** — The fraction of a sample that has to be corrupted before a statistic becomes unusable. The mean has a breakdown point of 0%, because one corrupted observation is enough to take it anywhere. The median has 50%, because more than half the sample has to be corrupted before it leaves the uncontaminated data.
- **consistency constant** — A factor applied to a robust scale estimate so that it converges to the standard deviation under an assumed distribution.
- **contamination** — The fraction of a sample that does not come from the assumed distribution.
- **cumulative distribution function** — The function giving, for each value, the probability of falling at or below it. The standard normal one is written $\Phi$, and its inverse turns a probability back into a number of standard deviations.
- **MAD** — The abbreviation used throughout for the median absolute deviation.
- **median absolute deviation (MAD)** — The median of the absolute deviations of the observations from the sample median, used as a scale estimate that a minority of extreme observations cannot inflate. On a normal sample it converges to 0.674490 times the standard deviation rather than to the standard deviation itself, which is why section 2.2 divides it by that constant before using it as a scale.
- **modified z-score** — The deviation from the median divided by the rescaled MAD.
- **Qn** — A robust scale estimator of Rousseeuw and Croux (1993), taking an order statistic of the pairwise distances between observations. It shares the 50% breakdown point of the MAD and needs no centre to measure from, and it is far more efficient on a normal sample, at 82% against 37%.
- **Sn** — A robust scale estimator of Rousseeuw and Croux (1993), built from a median of medians of pairwise distances. It shares the 50% breakdown point of the MAD and assumes no symmetry, and it reaches 58% efficiency on a normal sample against the 37% of the MAD.
- **z-score** — The deviation from the mean divided by the sample standard deviation.

## Appendix B. Reference Implementation

The block below is `hampel_identifier.py`, in the folder of this document. It is that file as
written, without the module docstring and the `if __name__ == '__main__':` guard.

The file holds the method of section 2 and nothing else. It scores a sample and reports the
interval; tabulating, printing and reading the scores are left to whatever calls it.

```python
# EDA/Outlier/Hampel/hampel_identifier.py
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
```

## Appendix C. Worked Example

Every number in this appendix is produced by `hampel_worked_example.py`, invoked as
`python3 hampel_worked_example.py --threshold 3.5`. The script reads the sample from
`EDA/Outlier/data/1d_esc_current.csv` and writes the points behind the figure beside the figure
as CSV.

Two rules are set against each other in C.1. **The identifier** is the method of section 2, which
measures an observation against the median and the MAD. **The classical rule** measures it against
the mean and the standard deviation instead.

### C.1. Sample

**Table 2. Every observation with its score and its verdict**

| Observation | Value | Value − median | Modified z | Flagged |
|---|---|---|---|---|
| 1 | 0.0232 | 0.0000 | 0.0000 | No |
| 2 | 0.0232 | 0.0000 | 0.0000 | No |
| 3 | 0.0232 | 0.0000 | 0.0000 | No |
| 4 | 0.0220 | −0.0012 | −0.1109 | No |
| 5 | 0.0232 | 0.0000 | 0.0000 | No |
| 6 | 0.0232 | 0.0000 | 0.0000 | No |
| 7 | **0.6532** | **+0.6300** | **58.2094** | **Yes** |
| 8 | 0.0403 | +0.0171 | 1.5800 | No |
| 9 | 0.0293 | +0.0061 | 0.5636 | No |
| 10 | 0.0159 | −0.0073 | −0.6745 | No |
| 11 | 0.0134 | −0.0098 | −0.9055 | No |
| 12 | 0.0134 | −0.0098 | −0.9055 | No |
| 13 | 0.0134 | −0.0098 | −0.9055 | No |
| 14 | 0.0134 | −0.0098 | −0.9055 | No |
| 15 | 0.0134 | −0.0098 | −0.9055 | No |

The flag is the rule of section 2.3 applied to the modified z beside it: an observation
is flagged when the absolute value of that score exceeds the threshold of 3.5. Only observation 7
does, at 58.2094. The next largest is 1.5800 at observation 8, less than half the threshold.

The two middle columns lay out the arithmetic of section 2.1, so the flag can be checked rather
than taken. The deviation column is the numerator of the modified z, and the modified z is that
deviation divided by the scale of 0.010823 that Table 3 works out. Observation 7 reads
0.6300 / 0.010823 = 58.2094, which is 16.6 times the threshold it has to clear.

**Table 3. The centre and the scale each rule computes**

| Quantity | Classical rule | Identifier |
|---|---|---|
| Centre | mean = 0.062913 | median = 0.023200 |
| Raw scale | standard deviation = 0.163469 | MAD = 0.007300 |
| Scale used in the score | standard deviation = 0.163469 | MAD / 0.674490 = 0.010823 |
| Upper boundary at 3.5 | mean + 3.5 x standard deviation = 0.635053 | median + 3.5 x (MAD / 0.674490) = 0.061080 |

The MAD is the median of the absolute deviations from 0.023200. Sorted, those 15 deviations are
0 five times, then 0.0012, 0.0061, 0.0073, 0.0098 five times, 0.0171 and 0.6300. The eighth of
fifteen is 0.0073, which is the MAD, and it comes from the observation 0.0159. Dividing it by
0.674490 as section 2.2 requires gives the 0.010823 the score divides by.

The two scales differ by a factor of 15.1, and the whole of that gap is the single observation
0.6532 entering the standard deviation. The scale of the identifier describes the other 14
observations; the classical scale describes the outlier.

### C.2. Normality

Normality enters the method only through the constant of section 2.2, and detection itself does
not need it. This sample is where that distinction matters, because it is not normal.

![Fig 1](hampel-identifier_fig/hampel_normality.png)

**Fig 1. Normal quantile plots of the sample and of the sample without its extreme value**

The observation is on the horizontal axis and the quantile a normal sample would put there is on
the vertical one, so a normal sample would fall along the line. The line runs through the first
and third quartiles rather than being fitted by least squares, which keeps the extreme
observation from rotating it toward itself and hiding the departure the panel is drawn to show.

**Table 4. Normality under three views of the sample**

| View | Count | Skewness | Shapiro-Wilk p |
|---|---|---|---|
| All observations | 15 | 3.462 | 1.70e-07 |
| Without 0.6532 | 14 | 1.042 | 9.91e-03 |
| All observations, log10 | 15 | 2.780 | 1.97e-05 |

The Shapiro-Wilk test takes as its null hypothesis that the sample was drawn from a normal
distribution. Its p-value is the probability of a departure at least as large as the one observed
if that hypothesis were true, so a p-value below the chosen level says the observed shape is too
unlikely under normality to be put down to sampling variation, and the hypothesis is rejected.

All three p-values fall below 0.05, and all three fall below 0.01 as well, though the middle row
falls below the stricter level by only 0.0001 and should not be read as decisive there. What puts
each view below the level is a different feature of the data.

- The full sample fails because of one observation. The value 0.6532 is 28 times the median, so panel (a) compresses the other 14 into a narrow column on the left and strands that one point far to the right of the line. The skewness of 3.462 is that point.
- The sample without it fails because the data are discrete. The remaining 14 observations take only 6 distinct values, with 5 tied at 0.0134 and 5 at 0.0232, so panel (b) stacks those ties into vertical columns instead of following the line. A normal distribution is continuous and produces no ties at all, so a sample carrying ten of them cannot look normal however its extremes are treated.
- The log view, computed but not plotted, fails for the same reason as the first. A logarithm compresses ratios, but 0.6532 remains 48.7 times the smallest observation afterwards, so it stays isolated and the skewness falls only from 3.462 to 2.780.

The second row is the one that matters for this document, because it is the reason the sample
cannot be rescued by treating 0.6532. The p-values are also approximate rather than exact here,
since two thirds of the observations are tied and the Shapiro-Wilk statistic assumes a continuous
distribution; the size of the departure does not rest on that approximation.

## Appendix D. Consistency Constant of the MAD

Section 2.2 divides the MAD by 0.674490 and says the raw MAD converges to that multiple of
$\sigma$ on a normal sample. This appendix derives the number, says how well a small sample
reaches it, and shows that it belongs to the normal distribution rather than to the MAD.

### D.1. Derivation

Take $X$ from a normal distribution with mean $\mu$ and standard deviation $\sigma$. Its median is
$\mu$, so the population MAD is the number $m$ with half the probability lying within $m$ of the
centre.

$$P\left( \left| X - \mu \right| \le m \right) = \frac{1}{2}$$

The probability inside that band is what lies below $\mu + m$ less what lies below $\mu - m$.
Symmetry about $\mu$ is what then collapses the two terms into one, since the lower tail is the
mirror of the upper.

$$\Phi\left( \frac{m}{\sigma} \right) - \Phi\left( -\frac{m}{\sigma} \right) = 2 \Phi\left( \frac{m}{\sigma} \right) - 1 = \frac{1}{2}$$

- $\Phi$ (capital phi) — the cumulative distribution function of the standard normal, so $\Phi(u)$ is the probability of falling below $u$ standard deviations.
- $m$ — the population MAD, the half-width of the band holding half the probability.

Rearranging leaves $\Phi(m/\sigma) = 3/4$, and inverting it leaves the constant.

$$m = \Phi^{-1}(0.75) \sigma = 0.674490 \sigma$$

That is where the third quartile comes from: **half the probability inside $\pm m$ is the same
statement as three quarters of it below $+m$.** The quartile is not chosen for the MAD. It is what
the MAD turns out to be.

The same number placed twice gives the interquartile range, since the first and the third quartile
each sit $0.674490 \sigma$ from the centre.

$$\mathrm{IQR} = 2 \cdot 0.674490 \sigma = 1.348980 \sigma$$

A rule built on the interquartile range therefore rescales by the same constant this one does,
taken twice rather than once.

### D.2. Finite Samples

The derivation is a statement about the distribution, not about a sample drawn from it. The sample
MAD converges to $m$, but in a small sample it lands low: the median of a short list of deviations
sits below the population median more often than above it, and rescaling does not repair that.

**Table 5. Mean of the rescaled MAD over 60,000 normal samples, true $\sigma = 1$**

| Sample size n | Mean of MAD / 0.674490 | Bias |
|---|---|---|
| 10 | 0.9129 | −8.7% |
| 15 | 0.9459 | −5.4% |
| 30 | 0.9739 | −2.6% |
| 100 | 0.9920 | −0.8% |
| 1000 | 0.9992 | −0.1% |

The bias runs one way, so the robust scale of a short sample is too small and every modified
z-score built on it is too large. Fifteen observations is a common size for a measurement run, and
at that size the scale is understated by about 5%, which lifts a score of 3.3 to 3.5 with nothing
having happened to the data.

### D.3. Other Distributions

The constant is a property of the normal distribution and nothing else. Repeating the derivation
under another shape gives another number, and using 0.674490 there would rescale the MAD to
something that is not the standard deviation of anything.

**Table 6. The ratio of the population MAD to the standard deviation**

| Distribution | MAD / $\sigma$ | Value |
|---|---|---|
| Normal | $\Phi^{-1}(0.75)$ | 0.674490 |
| Laplace | $\ln 2 / \sqrt{2}$ | 0.490129 |
| Uniform | $\sqrt{3} / 2$ | 0.866025 |

This is the sense in which section 2.2 calls the constant the one place normality enters the
method. Detection does not need it, since scaling every score by one factor leaves the order
alone. Reading a threshold as a false-positive rate does need it, and that reading is only as
good as the shape assumed here.
