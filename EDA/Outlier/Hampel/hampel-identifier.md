# Hampel Identifier — Flagging Outliers with the Median and the MAD
Rev. 23 | Created: 2026-08-17 | Updated: 2026-08-18 01:45 CDT

> A note on the modified z-score built from the median and the median absolute deviation,
> organized as the score, its robustness, and the treatment of what it flags.

## 1. Scope

The classical way to call an observation extreme is to divide its deviation from the mean by
the standard deviation and compare the result against a threshold. Both the centre and the scale
in that ratio are computed from the same sample the observation belongs to, so an outlier
inflates the quantity it is being measured against.

The Hampel identifier replaces the pair with the median and the median absolute deviation, the
MAD. Those two are unmoved by a minority of contaminating observations, so the scale stays a description of
the bulk of the sample rather than of the observation under test.

## 2. Modified z-score

### 2.1. Definition

Let $\tilde{x}$ be the median of the sample.

$$\mathrm{MAD} = \mathrm{median}\left( \left| x_1 - \tilde{x} \right|, \ldots, \left| x_n - \tilde{x} \right| \right)$$

$$M_i = \frac{x_i - \tilde{x}}{\mathrm{MAD} / \Phi^{-1}(0.75)}$$

Here $\Phi^{-1}(0.75) = 0.674490$ is the third quartile of the standard normal distribution, and
the denominator as a whole is the robust scale. The score $M_i$ is an ordinary z-score with the
median put in place of the mean and the robust scale in place of the standard deviation, which is
why it is called the modified z-score.

### 2.2. Consistency Constant

For a normal sample the MAD converges to $\Phi^{-1}(0.75)\,\sigma$ rather than to $\sigma$, so
the raw MAD understates the spread by about a third. Dividing by $\Phi^{-1}(0.75)$, equivalently
multiplying by 1.482602, removes that bias and is what makes the modified score comparable to a
classical z-score.

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

$$\tilde{x} - k \cdot \frac{\mathrm{MAD}}{\Phi^{-1}(0.75)} \; \le \; x_i \; \le \; \tilde{x} + k \cdot \frac{\mathrm{MAD}}{\Phi^{-1}(0.75)}$$

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
cannot read. A sample that fails this way needs a scale estimator that tolerates ties, such as Sn
or Qn.

### 3.2. Contaminated Scale

A sample standard deviation is a mean of squared deviations, so a single distant observation
enters it quadratically. The larger the outlier, the larger the denominator it is divided by,
and the two effects work against each other.

The median absolute deviation is a median of absolute deviations. Moving one observation
arbitrarily far changes which value sits at the middle of that list by at most one position, and
if the sample is large enough it does not change it at all.

The same self-reference, an observation inflating the scale that it is measured against, also
caps the classical score at a value no sample can exceed.
[Appendix D. Ceiling of the Classical Score](#appendix-d-ceiling-of-the-classical-score) works
that cap out.

### 3.3. Breakdown Point

The mean and the standard deviation break down at one observation. The median and the MAD break
down only when more than half the sample is corrupted.

An estimate breaks down when it stops describing the sample and describes the corrupted
observations instead. It does not stop computing. Setting one observation of the worked example
to a billion takes the mean past 66 million while the median stays at 0.0232, and a mean of 66
million says nothing about fifteen observations that otherwise sit near 0.02.

The breakdown point is the fraction of the sample that has to be corrupted for that to happen.

**Table 1. Breakdown point of the two pairs**

| Estimator | Breakdown point | What it takes to break it |
|---|---|---|
| Mean and standard deviation | 0% | One observation, moved far enough |
| Median and MAD | 50% | Eight of the fifteen; seven leave the median at 0.0403, a real observation |

This is the whole of the reason the method works. Section 2 is the arithmetic of putting the
robust pair on a scale that a threshold can be read against.

## 4. Treatment

The identifier settles whether an observation is far from the bulk. It does not settle whether
the observation is wrong, and the two questions should not be merged.

- Investigate the cause before acting. A flag traced to a recording error, an instrument fault, or a documented disturbance can be corrected or removed on that evidence.
- Retain a flag with no assignable cause, because a value that is genuinely part of the process carries information that removing it destroys.
- Report what was removed, the threshold used, and the change removal made to the estimates.
- Prefer accommodation to deletion when flags recur. A robust estimator limits the influence of extreme observations without discarding them, and the median and MAD used here are already such estimators.

Re-running the identifier on the sample left after removal is not a treatment. The centre and the
scale are recomputed on cleaned data, so the second pass measures a different thing from the
first.

## References

<a id="ref-1"></a>
[1] Hampel, F. R. (1974). [The Influence Curve and Its Role in Robust Estimation](https://doi.org/10.1080/01621459.1974.10482962). *Journal of the American Statistical Association*, 69(346), 383–393.

<a id="ref-2"></a>
[2] Iglewicz, B., & Hoaglin, D. C. (1993). *How to Detect and Handle Outliers*. The ASQC Basic References in Quality Control: Statistical Techniques, Vol. 16. ASQC Quality Press, Milwaukee. [https://asq.org/quality-press](https://asq.org/quality-press). ISBN 978-0-87389-247-6.

<a id="ref-3"></a>
[3] Shiffler, R. E. (1988). [Maximum Z Scores and Outliers](https://doi.org/10.1080/00031305.1988.10475530). *The American Statistician*, 42(1), 79–80.

<a id="ref-4"></a>
[4] Rousseeuw, P. J., & Croux, C. (1993). [Alternatives to the Median Absolute Deviation](https://doi.org/10.1080/01621459.1993.10476408). *Journal of the American Statistical Association*, 88(424), 1273–1283.

<a id="ref-5"></a>
[5] Leys, C., Ley, C., Klein, O., Bernard, P., & Licata, L. (2013). [Detecting outliers: Do not use standard deviation around the mean, use absolute deviation around the median](https://doi.org/10.1016/j.jesp.2013.03.013). *Journal of Experimental Social Psychology*, 49(4), 764–766.

---

## Appendix A. Terminology

- **accommodation** — Handling an outlier by limiting its influence on the estimates rather than by removing it.
- **breakdown point** — The fraction of a sample that has to be corrupted before a statistic becomes unusable. The mean has a breakdown point of 0%, because one corrupted observation is enough to take it anywhere. The median has 50%, because more than half the sample has to be corrupted before it leaves the uncontaminated data.
- **consistency constant** — A factor applied to a robust scale estimate so that it converges to the standard deviation under an assumed distribution.
- **contamination** — The fraction of a sample that does not come from the assumed distribution.
- **MAD** — Median absolute deviation; see that entry.
- **median absolute deviation (MAD)** — The median of the absolute deviations of the observations from the sample median, used as a scale estimate that a minority of extreme observations cannot inflate. On a normal sample it converges to 0.674490 times the standard deviation rather than to the standard deviation itself, which is why section 2.2 divides it by that constant before using it as a scale.
- **modified z-score** — The deviation from the median divided by the rescaled MAD.
- **Qn** — A robust scale estimator taking an order statistic of the pairwise distances between observations, with a 50% breakdown point and better behaviour than the MAD under ties.
- **Sn** — A robust scale estimator built from a median of medians of pairwise distances, with a 50% breakdown point and no assumption of symmetry.
- **z-score** — The deviation from the mean divided by the sample standard deviation.

## Appendix B. Reference Implementation

The implementation is `hampel_identifier.py`, in the folder of this document. The block below is
an excerpt: the docstrings are abridged to their opening paragraph, and the tabulation,
reporting and command line parts of the file are omitted. It is otherwise the file as written and
runs as printed.

```python
# EDA/Outlier/Hampel/hampel_identifier.py
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
    """Median of the absolute deviations from the sample median."""
    values = np.asarray(data, dtype=float)
    return float(np.median(np.abs(values - np.median(values))))


def hampel_scale(data: np.ndarray = None, quartile: float = NORMAL_QUARTILE) -> float:
    """Robust scale of the sample, which is the MAD divided by the consistency constant."""
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
                         f"{centre}; the robust scale is undefined. Use a scale estimator that tolerates "
                         f"ties, such as Sn or Qn, or report that the sample cannot support the method.")
    return mad / quartile


def hampel_score(data: np.ndarray = None, quartile: float = NORMAL_QUARTILE) -> np.ndarray:
    """Modified z-score of every observation, which is its distance from the median in robust scales."""
    values = _as_sample(data=data)
    return (values - np.median(values)) / hampel_scale(data=values, quartile=quartile)


def retained_interval(data: np.ndarray = None, threshold: float = DEFAULT_THRESHOLD,
                      quartile: float = NORMAL_QUARTILE) -> tuple[float, float]:
    """The decision rule written in the units of the data rather than in scales."""
    values = _as_sample(data=data)
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}.")
    centre = float(np.median(values))
    half_width = threshold * hampel_scale(data=values, quartile=quartile)
    return centre - half_width, centre + half_width


def classical_z_scores(data: np.ndarray = None) -> np.ndarray:
    """Deviation from the mean divided by the sample standard deviation, with ddof = 1."""
    values = _as_sample(data=data)
    deviation = values.std(ddof=1)
    if deviation == 0.0:
        raise ValueError(f"all {values.size} observations are equal, so the classical z-score is undefined.")
    return (values - values.mean()) / deviation


def max_attainable_z(sample_size: int = None) -> float:
    """Largest absolute classical z-score a sample of this size can produce."""
    if sample_size < 2:
        raise ValueError(f"a z-score needs at least 2 observations, got {sample_size}.")
    return (sample_size - 1) / np.sqrt(sample_size)
```

## Appendix C. Worked Example

Every number in this appendix is produced by `hampel_sample_outliers.py`, invoked as
`python3 hampel_sample_outliers.py --threshold 3.5`. The sample is fixed inside the script and
the points behind the figure are written beside it as CSV.

Two rules are applied to the same data throughout. **The identifier** is the method of section 2,
which scores an observation against the median and the MAD. **The classical rule** scores it
against the mean and the standard deviation instead, and flags at the same threshold of 3.5. The
scores they produce are the modified z of section 2.1 and the ordinary z-score.

### C.1. Sample

**Table 2. Every observation with its score under each rule**

| Observation | Value | Value − median | Modified z | Classical z | Flagged |
|---|---|---|---|---|---|
| 1 | 0.0232 | 0.0000 | 0.0000 | −0.2429 | No |
| 2 | 0.0232 | 0.0000 | 0.0000 | −0.2429 | No |
| 3 | 0.0232 | 0.0000 | 0.0000 | −0.2429 | No |
| 4 | 0.0220 | −0.0012 | −0.1109 | −0.2503 | No |
| 5 | 0.0232 | 0.0000 | 0.0000 | −0.2429 | No |
| 6 | 0.0232 | 0.0000 | 0.0000 | −0.2429 | No |
| 7 | **0.6532** | **+0.6300** | **58.2094** | 3.6110 | **Yes** |
| 8 | 0.0403 | +0.0171 | 1.5800 | −0.1383 | No |
| 9 | 0.0293 | +0.0061 | 0.5636 | −0.2056 | No |
| 10 | 0.0159 | −0.0073 | −0.6745 | −0.2876 | No |
| 11 | 0.0134 | −0.0098 | −0.9055 | −0.3029 | No |
| 12 | 0.0134 | −0.0098 | −0.9055 | −0.3029 | No |
| 13 | 0.0134 | −0.0098 | −0.9055 | −0.3029 | No |
| 14 | 0.0134 | −0.0098 | −0.9055 | −0.3029 | No |
| 15 | 0.0134 | −0.0098 | −0.9055 | −0.3029 | No |

The flag is the rule of section 2.3 applied to the modified z beside it: an observation
is flagged when the absolute value of that score exceeds the threshold of 3.5. Only observation 7
does, at 58.2094. The next largest is 1.5800 at observation 8, less than half the threshold.

The two middle columns lay out the arithmetic of section 2.1, so the flag can be checked rather
than taken. The deviation column is the numerator of the modified z, and the modified z is that
deviation divided by the scale of 0.010823 that Table 3 works out. Observation 7 reads
0.6300 / 0.010823 = 58.2094, which is 16.6 times the threshold it has to clear.

The classical z is carried alongside for contrast and takes no part in the flag. It is negative
at all fourteen retained observations, because the mean has been pulled above every one of them
by the fifteenth.

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

The identifier does not need normality for its detection to be meaningful, and section 2.2 says
that normality is what makes the threshold readable as a false-positive rate. This sample is the
case where that distinction matters, because it is not normal.

![Fig 1](hampel-identifier_fig/hampel_normality.png)

**Fig 1. Normal quantile plots of the sample and of the sample without its extreme value**

The reference line runs through the first and third quartiles rather than being fitted by least
squares, so the extreme observation cannot rotate it and flatten the departure the panel is drawn
to show.

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
falls below the stricter level by only 0.0001 and should not be read as decisive there. What puts each
view below the level is a different feature of the data.

- The full sample fails because of one observation. The value 0.6532 is 28 times the median, and panel (a) shows it far off the line while the other 14 lie almost flat against it. The skewness of 3.462 is that one point.
- The sample without it fails because the data are discrete. The remaining 14 observations take only 6 distinct values, with 5 tied at 0.0134 and 5 at 0.0232, so panel (b) rises in steps rather than along the line. A normal distribution is continuous and produces no ties at all, so a sample with ten of them cannot look normal however the extremes are treated.
- The log view, computed but not plotted, fails for the same reason as the first. A logarithm compresses ratios, but 0.6532 remains 48.7 times the smallest observation afterwards, so it stays isolated and the skewness falls only from 3.462 to 2.780.

The second row is the one that matters for this document, because it is the reason the sample
cannot be rescued by treating 0.6532. The p-values are also approximate rather than exact here,
since two thirds of the observations are tied and the Shapiro-Wilk statistic assumes a continuous
distribution; the size of the departure does not rest on that approximation.

## Appendix D. Ceiling of the Classical Score

The self-reference of section 3.2 has a consequence that is easy to miss. For a sample of size $n$ the
largest attainable absolute z-score is bounded, whatever the data are.

$$\max_i \left| z_i \right| \le \frac{n-1}{\sqrt{n}}$$

The bound is due to Shiffler (1988). It does not describe the data; it is arithmetic, and it
holds for every sample of that size.

**Table 5. The bound at several sample sizes**

| Sample size n | Largest attainable absolute z | A threshold of 3.5 |
|---|---|---|
| 10 | 2.8460 | Can never be reached |
| 14 | 3.4744 | Can never be reached |
| 15 | 3.6148 | Reachable, barely |
| 20 | 4.2485 | Reachable |
| 54 | 7.2124 | Reachable |

For $n \le 14$ a rule that flags an observation when its classical z-score exceeds 3.5 cannot
flag anything, no matter how extreme the sample is. The modified score of section 2.1 carries no such
bound, because its denominator stops responding to the observation in the numerator.
