# The 1.5 Sigma Shift Between Long-Term and Short-Term Capability
Rev. 0 | Created: 2026-08-30 | Updated: 2026-08-30 16:31 CDT

> A note on why a Six Sigma capability statement subtracts 1.5 from the short-term sigma level,
> what model that subtraction assumes, where the constant came from, and what the constant cannot
> be asked to do.

## 1. Scope

A Six Sigma process is described in two ways at once. It is said to hold six standard deviations
between its mean and the nearer specification limit, and it is said to produce 3.4 defective parts
per million. Those two statements do not agree. A normal distribution centred six standard
deviations from a limit puts about $0.002$ parts per million outside it, which is smaller than
3.4 by three orders of magnitude. The gap is closed by a convention: the long-term sigma level is
taken to be the short-term sigma level minus 1.5.

This document sets out what that convention asserts, derives the two distinct models that can
produce it, gives the arguments offered for the value 1.5, and states where the constant stops
being defensible. It is about the structure of the adjustment, not about how to run a capability
study.

## 2. Two Estimates of Sigma

### 2.1. Within-Subgroup and Overall Variation

A capability study collects observations in subgroups. Each subgroup is small and is taken over a
short interval, so the causes that act inside one subgroup are only those that act quickly. The
causes that act slowly, such as tool wear, ambient drift, batch-to-batch change of material, and
operator change, move one subgroup relative to the next and leave no trace inside any single one.

Two estimates of the standard deviation follow from the same data.

- Short-term standard deviation $\sigma_{st}$: it is pooled from the within-subgroup spread, by
  $\bar{R}/d_2$ or by the pooled subgroup standard deviation, and it sees the fast causes only.
- Long-term standard deviation $\sigma_{lt}$: it is the ordinary standard deviation of every
  observation in the study, taken without regard to the subgroup structure, and it sees the fast
  and the slow causes together.

Because the second contains everything the first contains and more, $\sigma_{lt} \ge \sigma_{st}$
in expectation. The two feed different capability indices: $C_p$ and $C_{pk}$ are built on
$\sigma_{st}$, while $P_p$ and $P_{pk}$ are built on $\sigma_{lt}$.

### 2.2. Sigma Level and the Shift

Write $\mu$ for the process mean and `USL`, `LSL` for the specification limits. The sigma level is
the standardised distance from the mean to the nearer limit, computed with one or the other scale.

$$Z_{st} = \min\left( \frac{USL - \mu}{\sigma_{st}},\ \frac{\mu - LSL}{\sigma_{st}} \right), \qquad Z_{lt} = \min\left( \frac{USL - \mu}{\sigma_{lt}},\ \frac{\mu - LSL}{\sigma_{lt}} \right)$$

The shift is the difference between them.

$$Z_{shift} = Z_{st} - Z_{lt}$$

This quantity is defined for any data set that has been subgrouped, and it is estimated from that
data set without any assumption. The Six Sigma convention does something else: it fixes
$Z_{shift} = 1.5$ in advance and uses it to convert a short-term study into a long-term claim. The
rest of this document concerns that fixed value.

## 3. Shift Model

### 3.1. Statement

The first reading of the convention is a statement about location. The process spread is held
constant at $\sigma_{st}$, and the mean is allowed to sit anywhere in a band of width
$1.5\sigma_{st}$ around the target. The capability claim is made at the worst position in that
band, so the mean is placed $1.5\sigma_{st}$ from the target, towards one limit, for the whole
period.

$$Z_{lt} = Z_{st} - 1.5$$

The defective fraction is then the normal tail at the shifted mean. Take the specification to be
symmetric about the target, so the mean sits at $Z_{st} - 1.5$ from the near limit and at
$Z_{st} + 1.5$ from the far one. The near tail carries the whole of the fraction; the far tail
contributes $3.2 \times 10^{-14}$ at $Z_{st} = 6$ and is dropped.

$$p_{lt} = \Phi\left( -(Z_{st} - 1.5) \right)$$

### 3.2. Tail Probability

Table 1 gives both readings of the same process. The short-term column is the two-sided tail of a
centred process at $Z_{st}$; the long-term column is the one-sided tail after the shift.

Table 1. Defect rates under the shift model, in parts per million.

| Z_st | Z_lt | Short-term, centred | Long-term, shifted |
|---:|---:|---:|---:|
| 3.0 | 1.5 | 2700 | 66807 |
| 3.5 | 2.0 | 465 | 22750 |
| 4.0 | 2.5 | 63.3 | 6210 |
| 4.5 | 3.0 | 6.80 | 1350 |
| 5.0 | 3.5 | 0.573 | 233 |
| 5.5 | 4.0 | 0.0380 | 31.7 |
| 6.0 | 4.5 | 0.00197 | 3.40 |

The last row is the origin of the familiar pair. Six sigma short term with the 1.5 allowance is
4.5 sigma long term, and the one-sided normal tail at 4.5 is 3.4 parts per million. In index terms
the same row reads $C_p = 2.00$ and $P_{pk} = 1.50$.

## 4. Inflation Model

### 4.1. Statement

The second reading is a statement about dispersion. The mean stays on target, and the slow causes
are treated as a random offset $M$ applied to each subgroup, with $E[M] = 0$ and
$\mathrm{Var}[M] = \tau^2$, independent of the within-subgroup deviation. The variances add.

$$\sigma_{lt}^{2} = \sigma_{st}^{2} + \tau^{2}$$

The sigma level then contracts by the ratio of the two scales rather than by a fixed amount.

$$Z_{lt} = Z_{st} \cdot \frac{\sigma_{st}}{\sigma_{lt}} = \frac{Z_{st}}{\sqrt{1 + \left( \tau / \sigma_{st} \right)^{2}}}$$

### 4.2. Why the Two Models Differ

The two models are not two derivations of one result. They disagree in two ways that matter.

The first disagreement is in how the gap scales. In the shift model $Z_{st} - Z_{lt}$ is 1.5 at
every capability level by construction. In the inflation model the gap is
$Z_{st}(1 - \sigma_{st}/\sigma_{lt})$, which is proportional to $Z_{st}$ for a fixed ratio of
scales. Forcing the inflation model to reproduce a gap of 1.5 therefore requires a different
amount of drift at every capability level, as Table 2 shows. A process cannot obey both models
across a range of $Z_{st}$ with one description of its drift.

Table 2. Drift required by the inflation model to reproduce a gap of 1.5 in sigma level.

| Z_st | Z_lt | Scale ratio | Drift ratio |
|---:|---:|---:|---:|
| 3.0 | 1.5 | 2.000 | 1.732 |
| 4.0 | 2.5 | 1.600 | 1.249 |
| 5.0 | 3.5 | 1.429 | 1.020 |
| 6.0 | 4.5 | 1.333 | 0.882 |

The scale ratio is $\sigma_{lt}/\sigma_{st}$ and the drift ratio is $\tau/\sigma_{st}$.

The second disagreement is in the defect count at a given $Z_{lt}$. The shift model puts the mean
off centre and counts one tail; the inflation model keeps the mean centred and counts two. At
$Z_{lt} = 4.5$ the shift model gives 3.4 parts per million and the inflation model gives 6.8. The
factor of two is not a rounding difference, and a quoted defect rate is meaningless until the
model behind it is named.

## 5. Origin of the Constant

### 5.1. Control Chart Detection Limit

The one argument that produces 1.5 from a stated premise rather than from experience is due to
Bothe [[1](#ref-1)], and it turns on how large a mean shift a control chart can miss.

Take an $\bar{X}$ chart with subgroup size $n$ and the usual three-sigma limits, and let the
process mean move by $\delta\sigma_{st}$ and stay there. Subgroup means have standard deviation
$\sigma_{st}/\sqrt{n}$, so in the chart's own units the mean has moved by $\delta\sqrt{n}$. The
probability $P$ that the next subgroup falls outside the limits is the sum of the two tails
beyond the displaced centre.

$$P = \Phi\left( \delta\sqrt{n} - 3 \right) + \Phi\left( -\delta\sqrt{n} - 3 \right)$$

The far term is negligible. The shift that the chart catches half the time therefore satisfies
$\delta\sqrt{n} = 3$, that is $\delta = 3/\sqrt{n}$. For the common subgroup size of four this is
exactly 1.5.

Table 3. Probability that the next subgroup signals a sustained mean shift, three-sigma limits.

| Subgroup size | Shift 1.0 | Shift 1.5 | Shift 2.0 | Shift at 50 percent power |
|---:|---:|---:|---:|---:|
| 1 | 0.023 | 0.067 | 0.159 | 3.00 |
| 2 | 0.056 | 0.190 | 0.432 | 2.12 |
| 3 | 0.102 | 0.344 | 0.679 | 1.73 |
| 4 | 0.159 | 0.500 | 0.841 | 1.50 |
| 5 | 0.223 | 0.638 | 0.930 | 1.34 |
| 9 | 0.500 | 0.933 | 0.999 | 1.00 |

Shifts are in units of $\sigma_{st}$. The last column is $3/\sqrt{n}$.

The reading is that a shift of $1.5\sigma_{st}$ is the largest one a chart with $n = 4$ has no
better than an even chance of catching, so a shift of that size can persist while the chart shows
nothing wrong. A capability claim made from short-term data must then carry an allowance for it.
The argument is sound, but it delivers $3/\sqrt{n}$, not 1.5. The constant belongs to a subgroup
size of four and to a chart run on the single-point rule alone; run rules raise the power and
lower the shift that can hide behind it.

### 5.2. Tolerance Stack-Up

A second 1.5 appears in the same literature from a different source. Bender, working on statistical
tolerancing for assemblies [[2](#ref-2)], proposed multiplying the root-sum-square combination of
component tolerances by 1.5, on the grounds that component distributions in practice are neither
normal nor centred and that the plain root-sum-square result is optimistic. Harry and Lawson
carried that correction into the producibility analysis behind the Motorola capability model
[[3](#ref-3)], and Raval and Muralidharan trace the resulting ancestry [[4](#ref-4)].

The two constants are not the same quantity. Bender's factor multiplies one spread to give
another, so it is a ratio and carries no units. The capability constant is subtracted from a
standardised distance, so it is a displacement measured in units of $\sigma_{st}$. Their numerical
agreement is a coincidence, and treating either as evidence for the other is an error.

### 5.3. Reverse Reading of 3.4 DPMO

The third argument runs backwards and is circular. If 3.4 defects per million is taken as the
definition of Six Sigma quality, the one-sided normal tail equal to $3.4 \times 10^{-6}$ sits at
$Z = 4.5$, and the shift follows as $6 - 4.5 = 1.5$. This derives nothing. The 3.4 figure is a
consequence of choosing 1.5, so it cannot serve as evidence for the choice.

## 6. Limits of the Constant

The convention is an allowance, and it holds only under conditions that it never states.

- Subgroup size: 1.5 is $3/\sqrt{n}$ at $n = 4$ alone. Nine per subgroup makes it 1.0, two makes
  it 2.1.
- Rational subgrouping: $\sigma_{st}$ is whatever the subgrouping admits. Subgroups spread over a
  longer interval absorb drift into $\sigma_{st}$ and shrink the true shift, while the constant
  stays fixed.
- Study length: a shift is long term only relative to a period of observation. A study covering
  one shift on one machine estimates nothing long term, whatever it is labelled.
- Distribution: the tail figures assume normality and a stable spread. A drifting spread, a
  skewed distribution, or a bounded characteristic voids the arithmetic of both models in
  section 3 and section 4.
- Direction: the shift model gives the worst position in the band, not the expected one. Applied
  to a process whose mean is in fact steady, it understates capability.

The decisive limitation is that $Z_{shift}$ is not a constant of nature but a measurable property
of a process. Any study that yields both $\sigma_{st}$ and $\sigma_{lt}$ yields an estimate of it
directly, and that estimate carries information the constant cannot.

## 7. Practice

Report $Z_{st}$ and $Z_{lt}$ from the same data set, together with the subgrouping that separates
them, and report $Z_{shift}$ as an estimate rather than as an assumption. Use 1.5 only when no
long-term data exist, and then state it as an assumption in the report, not as a property of the
process. Whenever a sigma level or a defect rate is quoted, name which of the two scales it uses
and which of the two models in section 3 and section 4 produced it; without that, the number is
ambiguous by a factor of two at best and by three orders of magnitude at worst.

## References

<a id="ref-1"></a>
[1] Bothe, D. R. (2002). Statistical Reason for the 1.5σ Shift. *Quality Engineering*, 14(3),
479–487. [https://doi.org/10.1081/QEN-120001884](https://doi.org/10.1081/QEN-120001884)

<a id="ref-2"></a>
[2] Bender, A. (1968). Statistical Tolerancing as it Relates to Quality Control and the Designer.
*SAE Technical Paper 680490*.
[https://www.sae.org/publications/technical-papers/content/680490/](https://www.sae.org/publications/technical-papers/content/680490/)

<a id="ref-3"></a>
[3] Harry, M. J., & Lawson, J. R. (1992). *Six Sigma Producibility Analysis and Process
Characterization*. Addison-Wesley. ISBN 978-0-201-63412-9.

<a id="ref-4"></a>
[4] Raval, N., & Muralidharan, K. (2016). A Note on 1.5 Sigma Shift in Performance Evaluation.
*International Journal of Reliability, Quality and Safety Engineering*, 23(6), 1640007.
[https://doi.org/10.1142/S0218539316400076](https://doi.org/10.1142/S0218539316400076)

---

## Appendix A. Terminology

- **DPMO**: defects per million opportunities, the defective fraction multiplied by $10^{6}$.
- **Long-term capability**: capability computed with $\sigma_{lt}$, covering a period long enough
  for the slow causes to act.
- **Rational subgroup**: a set of observations taken so that only the fast causes can vary inside
  it, which is what makes $\sigma_{st}$ separable from $\sigma_{lt}$.
- **Short-term capability**: capability computed with $\sigma_{st}$, covering only the variation
  present inside a subgroup.
- **Sigma level**: the standardised distance from the process mean to the nearer specification
  limit.
- **Three-sigma limits**: control limits placed three standard deviations of the plotted statistic
  either side of its centre line.
