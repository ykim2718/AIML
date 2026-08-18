# Hampel Identifier — Finding Outliers with the Median and the MAD
Rev. 14 | Created: 2026-08-17 | Updated: 2026-08-18 01:12 CDT

> A rule that flags an observation sitting more than 3.5 robust scales away from the median.
> The procedure comes first and the reasoning behind it comes last.

## 1. Procedure

The whole method is five steps.

```text
1. Take the median of the sample.
2. Take the distance of every observation from that median.
3. The MAD is the median of those distances.
4. The scale is the MAD divided by 0.674490.
5. Flag any observation further than 3.5 scales from the median.
```

Appendix C runs those steps on a sample of fifteen measurements. They give the following.

**Table 1. The five steps on the worked example**

| Step | Result |
|---|---|
| Median | 0.023200 |
| MAD | 0.007300 |
| Scale, which is MAD / 0.674490 | 0.010823 |
| Half-width, which is 3.5 x scale | 0.037880 |
| Retained interval | [−0.014680, 0.061080] |

One observation of the fifteen lies outside that interval. It is 0.6532, and it is flagged. The
other fourteen lie between 0.0134 and 0.0403, and they are retained.

Written out, with $\tilde{x}$ standing for the median:

$$\mathrm{MAD} = \mathrm{median}\left( \left| x_1 - \tilde{x} \right|, \ldots, \left| x_n - \tilde{x} \right| \right)$$

$$M_i = \frac{x_i - \tilde{x}}{\mathrm{MAD} / \Phi^{-1}(0.75)}$$

An observation is flagged when $\left| M_i \right| \gt 3.5$. The score $M_i$ is called the
modified z-score, and the constant $\Phi^{-1}(0.75) = 0.674490$ is explained in section 3.1.

Every observation is scored once, against one centre and one scale. Nothing is removed part-way
and the calculation is not repeated.

### 1.1. What a Flag Means

A flag says that an observation sits far from the bulk of the sample. It does not say that the
observation is wrong. Those are separate questions and the rule settles only the first.

- Look for a cause before acting. A flag traced to a recording error, an instrument fault, or a documented disturbance can be corrected or removed on that evidence.
- Keep a flag that has no assignable cause. The value may be a genuine part of the process, and removing it throws away what it carries.
- Report what was removed, which threshold was used, and how the removal changed the estimates.
- When flags keep appearing, limit the influence of extreme values rather than deleting them. This is called accommodation, and the median and the MAD already do it.

Running the rule again on what is left is not a treatment. The second pass computes a fresh
centre and a fresh scale from cleaned data, so it answers a different question from the first.

## 2. Failure Modes

**Table 2. Conditions under which the rule cannot be used**

| Condition | What goes wrong | What to use instead |
|---|---|---|
| More than half the sample takes one value | The MAD is 0, so the score divides by zero and is undefined | A scale estimator that tolerates ties, such as Sn or Qn |
| Half the sample or more is contaminated | The median and the MAD describe the contamination rather than the sample | Nothing recovers this. The sample needs a separate source of truth |
| Several variables at once | The score reads one variable at a time | A distance that accounts for covariance, such as Mahalanobis distance |
| Observations correlated in sequence | Neighbouring values are not exchangeable | Model the correlation, or score the residuals |
| The sample is not normal | The threshold stops reading as a false-positive rate, though the scores still rank correctly | Report the score and its margin instead of a nominal rate |

The first row is a hard failure rather than a loss of accuracy. The score is undefined, and an
implementation that returns zeros instead of raising an error reports a clean sample on data it
cannot read. Coarse measurement resolution pushes a sample toward that row. The worked example
has five tied observations out of fifteen, and the MAD reaches zero at eight.

The last row limits how a result may be read rather than the result itself. On a sample that is
not normal the score still measures how far an observation sits from the bulk, in units of the
spread of that bulk. What is lost is the reading of 3.5 as a false-positive rate.

## 3. Threshold

The threshold applies to the score and not to the observation. An observation is flagged when
$\left| M_i \right|$ exceeds 3.5, so the cut-off is counted in scales rather than in the units of
the data. The boundary this places on the data itself moves from sample to sample, because the
scale is computed from the sample. Table 1 works one boundary out.

**Table 3. What the threshold costs on genuinely normal data**

| Threshold | Probability that one observation is flagged | Expected false flags in fifteen observations |
|---|---|---|
| 2.5 | 0.012419 | 0.186 |
| 3.0 | 0.002700 | 0.041 |
| 3.5 | 0.000465 | 0.007 |

The value 3.5 comes from Iglewicz and Hoaglin (1993). It is deliberately conservative: a hundred
normal observations produce one false flag about once in twenty samples. A threshold of 3.0 is
also used, and it is roughly six times looser.

Fix the threshold before looking at the data. A verdict that changes between 3.0 and 3.5 rests on
that choice rather than on the sample, and the report should say so.

### 3.1. The Constant 0.674490

Step 4 divides the MAD by 0.674490 rather than using it directly. On a normal sample the MAD
converges to 0.674490 times the standard deviation, so the raw MAD understates the spread by
about a third. Dividing by the constant removes that bias and puts the modified z-score on the
same footing as an ordinary z-score. The factor is called the consistency constant.

This is the only place where normality enters the method, and it is a calibration rather than an
assumption. Changing the constant multiplies every score by the same factor and reorders nothing.
Its purpose is to let the numbers in Table 3 mean what they say.

## 4. Breakdown Point

The mean and the standard deviation become unusable as soon as one observation is corrupted. The
median and the MAD keep working until more than half the sample is corrupted. The breakdown point
is the fraction of a sample that has to be corrupted for a statistic to become unusable.

Set one of the fifteen observations to a billion. The mean rises past 66 million while the median
stays at 0.023200. The mean still returns a number, but that number describes the one corrupted
value rather than the fourteen observations near 0.02. Corrupt seven of the fifteen and the median
is 0.0403, which is still one of the real observations. Corrupt eight and the median becomes the
corrupted value.

**Table 4. Breakdown point of the two pairs**

| Estimator | Breakdown point | Corrupted observations needed, out of fifteen |
|---|---|---|
| Mean and standard deviation | 0% | One |
| Median and MAD | 50% | Eight |

This is why the procedure of section 1 needs no iteration. A rule built on the mean and the
standard deviation has to remove outliers one at a time, because every outlier it has not yet
removed is still inflating the scale that the next one is measured against. The median and the
MAD were never inflated, so a single pass is enough.

The same self-reference caps the classical z-score at a value that no sample can exceed. Appendix
D works that cap out.

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
- **exchangeable** — A property of observations whose joint distribution is unchanged by reordering them, which serial correlation breaks.
- **MAD** — The abbreviation used throughout for the median absolute deviation.
- **Mahalanobis distance** — A distance from the centre of a multivariate sample that accounts for the covariance among the variables.
- **median absolute deviation (MAD)** — The median of the absolute deviations of the observations from the sample median, used as a scale estimate that a minority of extreme observations cannot inflate. On a normal sample it converges to 0.674490 times the standard deviation rather than to the standard deviation itself, which is why section 3.1 divides it by that constant.
- **modified z-score** — The deviation from the median divided by the rescaled MAD.
- **Qn** — A robust scale estimator taking an order statistic of the pairwise distances between observations, with a 50% breakdown point and better behaviour than the MAD under ties.
- **Sn** — A robust scale estimator built from a median of medians of pairwise distances, with a 50% breakdown point and no assumption of symmetry.
- **z-score** — The deviation from the mean divided by the sample standard deviation.

## Appendix B. Reference Implementation

The implementation is `hampel_identifier.py`, in the folder of this document. The block below is
an excerpt: the docstrings are abridged to their opening paragraph, and the tabulation,
reporting and command line parts of the file are omitted. It is otherwise the file as written and
runs as printed.

### B.1. Implementation

```python
# EDA/Outlier/Hampel/hampel_identifier.py
from dataclasses import dataclass

import numpy as np
from scipy import stats

# The MAD of a normal sample converges to this multiple of sigma, so dividing by it puts the
# modified score on the same scale as a classical z-score. It is the third quartile of N(0, 1).
NORMAL_QUARTILE = float(stats.norm.ppf(0.75))

# The cut-off recommended by Iglewicz and Hoaglin for the modified z-score.
DEFAULT_THRESHOLD = 3.5


@dataclass
class HampelResult:
    """Outcome of the Hampel identifier on one sample."""

    values: np.ndarray
    centre: float
    mad: float
    scale: float
    scores: np.ndarray
    threshold: float
    positions: np.ndarray

    @property
    def count(self) -> int:
        """Number of flagged observations."""
        return int(self.positions.size)

    def bounds(self) -> tuple[float, float]:
        """The interval outside which an observation is flagged."""
        return self.centre - self.threshold * self.scale, self.centre + self.threshold * self.scale


def median_absolute_deviation(data: np.ndarray = None) -> float:
    """Median of the absolute deviations from the sample median."""
    values = np.asarray(data, dtype=float)
    return float(np.median(np.abs(values - np.median(values))))


def classical_z_scores(data: np.ndarray = None) -> np.ndarray:
    """Deviation from the mean divided by the sample standard deviation, with ddof = 1."""
    values = np.asarray(data, dtype=float)
    deviation = values.std(ddof=1)
    if deviation == 0.0:
        raise ValueError(f"all {values.size} observations are equal, so the classical z-score is undefined.")
    return (values - values.mean()) / deviation


def max_attainable_z(sample_size: int = None) -> float:
    """Largest absolute classical z-score a sample of this size can produce."""
    if sample_size < 2:
        raise ValueError(f"a z-score needs at least 2 observations, got {sample_size}.")
    return (sample_size - 1) / np.sqrt(sample_size)


def hampel_test(data: np.ndarray = None, threshold: float = DEFAULT_THRESHOLD,
                quartile: float = NORMAL_QUARTILE) -> HampelResult:
    """Flag observations whose modified z-score exceeds the threshold in absolute value."""
    values = np.asarray(data, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {values.shape}.")
    if values.size < 3:
        raise ValueError(f"the identifier needs at least 3 observations to have a usable median, got {values.size}.")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"data carries {int((~np.isfinite(values)).sum())} non-finite values; "
                         f"remove or impute them before testing.")
    if threshold <= 0.0:
        raise ValueError(f"threshold must be positive, got {threshold}.")
    if quartile <= 0.0:
        raise ValueError(f"quartile must be positive, got {quartile}.")

    centre = float(np.median(values))
    mad = median_absolute_deviation(data=values)
    if mad == 0.0:
        # More than half the sample sits on one value, so the MAD carries no scale at all.
        # Returning zero scores here would report a clean sample, which is the opposite of the truth.
        repeated = int((values == centre).sum())
        raise ValueError(f"the MAD is 0 because {repeated} of {values.size} observations equal the median "
                         f"{centre}; the modified z-score is undefined. Use a scale estimator that tolerates "
                         f"ties, such as Sn or Qn, or report that the sample cannot support the test.")

    scale = mad / quartile
    scores = (values - centre) / scale
    return HampelResult(values=values, centre=centre, mad=mad, scale=scale, scores=scores,
                        threshold=threshold, positions=np.flatnonzero(np.abs(scores) > threshold))
```

### B.2. Design Notes

The consistency constant is computed from `scipy.stats` rather than written as 0.674490, so the
calibration of section 3.1 cannot drift from the value the code actually applies.

`HampelResult` carries the centre and the scale alongside the scores rather than discarding them,
because a score on its own cannot be checked against anything. Its omitted `to_frame` method
tabulates the sample against both scores, which is what the worked example prints.

The zero-MAD branch is the one that matters. It is the failure of section 2, and the alternative
to raising is to divide by zero or to return scores of zero, either of which reports a clean
sample on data the method cannot read. The message names the count of tied observations so the
caller can see why.

`classical_z_scores` and `max_attainable_z` exist for the comparison rather than for the method.
Carrying the bound as a function rather than as a number in the document keeps Appendix D
checkable: the claim about any sample size can be evaluated instead of trusted.

### B.3. Invocation

```bash
python3 hampel_identifier.py --input-csv <PATH> --column value --threshold 3.5
python3 hampel_sample_outliers.py --threshold 3.5
```

Running either with no option prints the usage. The first tests a column of a file; the second
reproduces Appendix C exactly and writes the figure with the samples behind it to
`hampel-identifier_fig/`.

## Appendix C. Worked Example

Every number in this appendix is produced by `hampel_sample_outliers.py`, invoked as
`python3 hampel_sample_outliers.py --threshold 3.5`. The sample is fixed inside the script and
the points behind the figure are written beside it as CSV.

Two rules are applied to the same data throughout. **The identifier** is the method of section 1,
which scores an observation against the median and the MAD. **The classical rule** scores it
against the mean and the standard deviation instead, and flags at the same threshold of 3.5. The
scores they produce are the modified z of section 1 and the ordinary z-score.

### C.1. Sample

**Table 5. Every observation with its score under each rule**

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

The flag is the decision rule of section 1 applied to the modified z beside it: an observation
is flagged when the absolute value of that score exceeds the threshold of 3.5. Only observation 7
does, at 58.2094. The next largest is 1.5800 at observation 8, less than half the threshold.

The two middle columns lay out the arithmetic of section 1, so the flag can be checked rather
than taken. The deviation column is the numerator of the modified z, and the modified z is that
deviation divided by the scale of 0.010823 that Table 6 works out. Observation 7 reads
0.6300 / 0.010823 = 58.2094, which is 16.6 times the threshold it has to clear.

The classical z is carried alongside for contrast and takes no part in the flag. It is negative
at all fourteen retained observations, because the mean has been pulled above every one of them
by the fifteenth.

**Table 6. The centre and the scale each rule computes**

| Quantity | Classical rule | Identifier |
|---|---|---|
| Centre | mean = 0.062913 | median = 0.023200 |
| Raw scale | standard deviation = 0.163469 | MAD = 0.007300 |
| Scale used in the score | standard deviation = 0.163469 | MAD / 0.674490 = 0.010823 |
| Upper boundary at 3.5 | mean + 3.5 x standard deviation = 0.635053 | median + 3.5 x (MAD / 0.674490) = 0.061080 |

The MAD is the median of the absolute deviations from 0.023200. Sorted, those 15 deviations are
0 five times, then 0.0012, 0.0061, 0.0073, 0.0098 five times, 0.0171 and 0.6300. The eighth of
fifteen is 0.0073, which is the MAD, and it comes from the observation 0.0159. Dividing it by
0.674490 as section 3.1 requires gives the 0.010823 the score divides by.

The two scales differ by a factor of 15.1, and the whole of that gap is the single observation
0.6532 entering the standard deviation. The scale of the identifier describes the other 14
observations; the classical scale describes the outlier.

### C.2. Normality

The last row of Table 2 says that the identifier still works on a sample that is not normal,
and that what normality buys is the reading of the threshold as a false-positive rate. This
sample is where that distinction matters, because it is not normal.

![Fig 1](hampel-identifier_fig/hampel_normality.png)

**Fig 1. Normal quantile plots of the sample and of the sample without its extreme value**

The reference line runs through the first and third quartiles rather than being fitted by least
squares, so the extreme observation cannot rotate it and flatten the departure the panel is drawn
to show.

**Table 7. Normality under three views of the sample**

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

For a sample of size $n$ the largest absolute classical z-score that can occur is bounded,
whatever the data are.

$$\max_i \left| z_i \right| \le \frac{n-1}{\sqrt{n}}$$

The bound is due to Shiffler (1988). It does not describe the data; it is arithmetic, and it
holds for every sample of that size.

**Table 8. The bound at several sample sizes**

| Sample size n | Largest attainable absolute z | A threshold of 3.5 |
|---|---|---|
| 10 | 2.8460 | Can never be reached |
| 14 | 3.4744 | Can never be reached |
| 15 | 3.6148 | Reachable, barely |
| 20 | 4.2485 | Reachable |
| 54 | 7.2124 | Reachable |

For $n \le 14$ a rule that flags an observation when its classical z-score exceeds 3.5 cannot
flag anything, no matter how extreme the sample is. The modified z-score of section 1 has no
such bound, because its denominator stops responding to the observation in the numerator.
