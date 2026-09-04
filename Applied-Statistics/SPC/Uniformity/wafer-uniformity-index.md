# The Wafer Uniformity Index
Rev. 3 | Created: 2026-09-04 | Updated: 2026-09-04 14:40 CDT

> A note on the scalar index that semiconductor process control uses to summarise how much a layer
> varies across a wafer: the two standard formulas, what the number does and does not carry, and how
> it is used in deposition, etch and CMP.

## 1. Scope

A process step leaves a quantity on every part of the wafer, and no process leaves the same value
everywhere. Deposition leaves a thickness, etch leaves a depth, CMP leaves a remaining thickness,
implant leaves a sheet resistance. In each case the wafer is measured at a fixed set of sites and
the whole map has to be reduced to one number, because a number is what a specification can hold
and what a control chart can plot.

That number is the uniformity index. This document gives the two definitions the industry uses,
what each one measures physically, what neither of them can see, and how the index enters process
control. Terms used without definition in the body are collected in
[Appendix A](#appendix-a-terminology).

## 2. Measurement Basis

### 2.1. Measurement Pattern and Edge Exclusion

The index is computed from a fixed pattern of measurement sites, most often laid out on concentric
rings so that the radial direction is sampled evenly. Common patterns are 9, 13, 25, 49 and 121
points. Sites within a few millimetres of the wafer edge are excluded, typically 3 mm, because the
edge carries handling damage and a steep process roll-off that would otherwise dominate the number.

Both the pattern and the exclusion are part of the definition, not incidental to it. The same wafer
measured at 9 points and at 49 points yields different indices, and section 4.2 gives the size of
that effect.

### 2.2. What the Index Normalises

Every definition below divides a measure of spread by the mean of the same measurements. The
division is what makes the index dimensionless and therefore comparable across thickness targets,
across layers and across tools; a 30 Å spread means something different on a 300 Å film than on a
3000 Å film, and the ratio removes that difference.

The name is also the reverse of what the number does. A larger index means a less uniform wafer, so
what is called uniformity is measured as non-uniformity. Some metrology tools report
$100 - \mathrm{NU}$ instead, which is why a reported number is only interpretable together with the
convention that produced it.

## 3. Standard Definitions

### 3.1. Range Method

The range method uses the largest and smallest measured values and the mean of all of them.

$$\mathrm{NU}_{\mathrm{range}} = \frac{Max - Min}{2 \times Mean} \times 100 \hspace{19em} (1)$$

The factor 2 in the denominator makes the index a half-range, so it reads as a plus-or-minus
statement: with the extremes placed symmetrically about the mean, every measured value lies within
$Mean \times (1 \pm \mathrm{NU}/100)$. Two variants are also in use. Dividing by $Max + Min$ instead
of by $2 \times Mean$ gives the same value to within the asymmetry of the extremes, but dropping the
factor 2 doubles it. A number that does not name its formula is therefore ambiguous by a factor of
two.

The method has one strength and one weakness, and both come from the same place. It reads only two
of the measurements, so it is quick to compute and easy to explain; and it reads only two of the
measurements, so one bad site, one particle, one metrology misread moves the whole index.

### 3.2. Standard Deviation Method

The standard deviation method uses the spread of all the measurements rather than the two extremes.

$$\mathrm{NU}_{1\sigma} = \frac{\sigma}{Mean} \times 100, \qquad \mathrm{NU}_{3\sigma} = \frac{3\sigma}{Mean} \times 100 \hspace{19em} (2)$$

The one-sigma form is the coefficient of variation of the site measurements. The three-sigma form
is the same quantity scaled so that, for a normally distributed set of sites, it spans the interval
that holds about 99.7 percent of them; it is the form usually quoted for critical dimension
uniformity. Whether $\sigma$ is the sample standard deviation with $n-1$ in the denominator or the
population form with $n$ is a further convention that changes the number, by 1 percent of itself at
$n = 49$ and by more at small $n$.

Table 1. The two standard definitions compared.

| Aspect | Range method | Standard deviation method |
|---|---|---|
| Reads | Two measurements | Every measurement |
| Sensitive to one outlier | Strongly | Weakly |
| Grows with the point count | Yes | No |
| Sampling distribution | No simple closed form | Chi-squared, from the sample variance |
| Typical use | Quick tool checks, incoming reports | SPC charting, capability work |

### 3.3. Choosing Between Them

Both indices measure the same physical thing, and on a well-behaved wafer they track each other.
For a set of $n$ measurements drawn from a normal distribution, the expected range is $d_2(n)$
standard deviations, where $d_2$ is the control chart constant [[3](#ref-3)], so the two indices are
related through the point count.

$$\mathrm{NU}_{\mathrm{range}} \approx \frac{d_2(n)}{2} \times \mathrm{NU}_{1\sigma} \hspace{19em} (3)$$

The relation holds only when the wafer has no spatial signature, that is when the site-to-site
variation is random. A wafer with a strong radial trend has its extremes at the centre and the edge
by construction, and the range then reflects the trend rather than the scatter.

## 4. Physical Meaning

### 4.1. What the Index Cannot See

The index is a summary of dispersion. It carries no information about where on the wafer the
variation sits, and two wafers with opposite process signatures can return the same number.

<img src="wafer-uniformity-index_fig/wafer_uniformity_index.png" width="1000" style="max-width: 100%;" alt="Fig 1">

Fig 1. A centre-thick wafer (a) and an edge-thick wafer (b) with the radial profile of each (c). The
colour scale is thickness in angstrom, the dots are the 49 measurement sites, and both wafers have a
mean of 1000 Å, a standard deviation of 36.05 Å, a range index of 5.415 percent and a one-sigma
index of 3.605 percent.

The two wafers in Fig 1 are indistinguishable to every definition in section 3, and they call for
opposite corrective actions. A centre-thick deposition asks for less precursor at the centre or more
at the edge; an edge-thick one asks for the reverse. The index says that something is wrong and how
much; it never says what. That is the reason a uniformity number is reported next to a contour map
or a radial trend and not on its own.

The index also mixes sources that a diagnosis has to separate. The measured spread contains the
genuine process signature, the site-to-site randomness of the process, and the repeatability of the
metrology tool, and these add as variances.

$$\sigma_{\mathrm{measured}}^{2} = \sigma_{\mathrm{process}}^{2} + \sigma_{\mathrm{metrology}}^{2} \hspace{19em} (4)$$

Where the metrology term is not small against the process term, part of the reported
non-uniformity belongs to the measurement rather than to the wafer, and improving the process cannot
move it.

### 4.2. Dependence on the Point Count

Equation (3) has a consequence that is easy to miss in practice. The range index grows with the
number of measurement sites even when nothing about the process changes, because more draws from
the same distribution are more likely to include an extreme one.

Table 2. Half-range index of a wafer whose one-sigma index is exactly 1 percent, by point count.

| Points | Expected range in sigma | Half-range index |
|---:|---:|---:|
| 5 | 2.324 | 1.162 |
| 9 | 2.971 | 1.485 |
| 13 | 3.336 | 1.668 |
| 17 | 3.588 | 1.794 |
| 21 | 3.779 | 1.889 |
| 25 | 3.932 | 1.966 |
| 49 | 4.483 | 2.241 |
| 121 | 5.149 | 2.575 |

The same wafer measured at 9 sites reports 1.49 percent and at 49 sites reports 2.24 percent, a
change of half again for no physical reason. Range-based numbers from different measurement recipes
are therefore not comparable, and a recipe change that adds sites will look like a process
degradation on a range chart and like nothing at all on a sigma chart. The one-sigma index does not
have this behaviour, since the sample standard deviation estimates the same population quantity at
every $n$.

## 5. Application

### 5.1. Process Steps

Table 3. The index by process step.

| Step | Measured quantity | Common form |
|---|---|---|
| Deposition | Film thickness | Range or one-sigma |
| Etch | Etch depth, remaining thickness | Range or one-sigma |
| CMP | Removal rate, remaining thickness | One-sigma, as within-wafer non-uniformity |
| Implant | Sheet resistance | One-sigma |
| Lithography | Critical dimension | Three-sigma |

The physical cause differs by step and each has its own radial signature. Deposition uniformity
follows gas flow, showerhead design and substrate temperature; etch follows plasma density and
temperature; CMP follows pad pressure and slurry distribution, and its within-wafer non-uniformity
is the term the literature analyses most closely [[1](#ref-1)], [[2](#ref-2)]. In every case the
index is the observable and the signature is the diagnosis.

### 5.2. Use in Process Control

The index is itself a statistic, computed once per wafer, and so it can be charted over wafers and
lots exactly like any other measurement. Because it is a dispersion statistic rather than a location
statistic, its natural chart is the one built for standard deviations rather than the one built for
means.

Charting it also picks out one term of a larger variance budget. The total variation of a
population of measurements across a lot separates into a part within each wafer and a part between
wafers, and the uniformity index tracks only the first.

$$\sigma_{\mathrm{total}}^{2} = \sigma_{\mathrm{within}}^{2} + \sigma_{\mathrm{between}}^{2} \hspace{19em} (5)$$

A process whose uniformity index is stable and small can still be out of control through the second
term, from chamber-to-chamber differences or from drift between lots. The index is a necessary
statistic, not a sufficient one, and it belongs next to a chart of the wafer mean rather than in
place of one.

### 5.3. Reporting

Because so much of the number depends on convention rather than on the wafer, a reported uniformity
value carries the following with it, or it cannot be compared with anything.

- Formula: range or standard deviation, one-sigma or three-sigma, with or without the factor 2.
- Measurement pattern: the number of sites and their layout.
- Edge exclusion: the width of the excluded annulus.
- Divisor of the standard deviation: $n$ or $n-1$.
- Metrology tool and its repeatability, so that equation (4) can be read.

## References

<a id="ref-1"></a>
[1] Davis, J. C., Sherer, J. M., Poole, S. J., & Loewenstein, L. M. (1996). [A Robust Metric for
Measuring Within-Wafer Uniformity](https://doi.org/10.1109/3476.558556). *IEEE Transactions on Components, Packaging, and Manufacturing
Technology — Part C*, 19(4), 283–289.

<a id="ref-2"></a>
[2] [A Study of Within-Wafer Non-Uniformity Metrics](https://ieeexplore.ieee.org/document/773193).
*1999 4th International Workshop on Statistical Metrology*.

<a id="ref-3"></a>
[3] Montgomery, D. C. (2020). *Introduction to Statistical Quality Control* (8th ed.). Wiley.
ISBN 978-1-119-72309-7.

---

## Appendix A. Terminology

- **Coefficient of variation**: the standard deviation divided by the mean, the quantity equation
  (2) reports as a percentage.
- **Critical dimension**: the width of a printed feature, the quantity lithography controls.
- **Edge exclusion**: the annulus at the wafer edge that carries no measurement sites.
- **Measurement site**: one location on the wafer at which the process result is measured.
- **NU**: non-uniformity, the symbol equations (1) to (3) use for the index.
- **Radial signature**: the systematic dependence of the measured quantity on distance from the
  wafer centre.
- **Repeatability**: the spread a metrology tool returns on repeated measurement of one unchanged
  site.
- **Within-wafer non-uniformity**: the uniformity index computed across the sites of a single wafer,
  as distinct from the variation between wafers.
