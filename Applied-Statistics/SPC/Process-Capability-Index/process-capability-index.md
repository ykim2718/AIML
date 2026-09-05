# Process Capability Indices
Rev. 14 | Created: 2026-09-04 | Updated: 2026-09-04 21:20 CDT

> A note on the three indices that compare a process against its specification: $C_p$ for the
> spread, $k$ for the centring, and $C_{pk}$ for what the two produce together.

## 1. Scope

A control chart says whether a process is stable. It says nothing about whether the process makes
acceptable product, because it is built from the process own variation and never looks at the
specification. Capability indices are the other half of that pair: they hold the process
distribution against the specification limits and return a number that says how much room is left.

This document defines $C_p$, $k$ and $C_{pk}$, gives what each one measures, shows why the three are
read together rather than singly, and sets out the assumptions under which the numbers mean
anything. Terms used without definition in the body are collected in
[Appendix A](#appendix-a-terminology).

## 2. Definitions

Throughout, the process is normal with mean $\mu$ and standard deviation $\sigma$, and the
specification has a lower limit `LSL` and an upper limit `USL`. Write $m$ for the midpoint of the
specification.

### 2.1. Potential Capability

$C_p$ compares the width the specification allows against the width the process occupies, where the
process width is taken as $6\sigma$ [[1](#ref-1)].

$$C_p = \frac{USL - LSL}{6\sigma} \hspace{19em} (1)$$

The index does not contain $\mu$. It therefore answers a question about the spread alone: if the
process were centred, how much room would there be. That is why it is called the potential
capability. A process with $C_p = 2$ occupies half the specification width and could be moved a long
way off centre before making anything out of specification; a process with $C_p = 1$ exactly fills
the specification and has no room at all.

### 2.2. Centring

$k$ measures how far the process mean sits from the middle of the specification, expressed as a
fraction of the half-width.

$$m = \frac{USL + LSL}{2}, \qquad k = \frac{\left| m - \mu \right|}{(USL - LSL)/2} \hspace{19em} (2)$$

The index runs from 0 to 1 within the specification. At $k = 0$ the process is perfectly centred, at
$k = 0.25$ the mean has moved a quarter of the way from the middle towards a limit, and at $k = 1$
the mean sits on a limit. Like $C_p$, it answers half the question: it contains no $\sigma$ and so
says nothing about how wide the process is.

### 2.3. Achieved Capability

$C_{pk}$ measures the distance from the mean to the nearer specification limit in units of
$3\sigma$, so it carries both the spread and the centring.

$$CPU = \frac{USL - \mu}{3\sigma}, \qquad CPL = \frac{\mu - LSL}{3\sigma}, \qquad C_{pk} = \min\left( CPU, CPL \right) \hspace{19em} (3)$$

The three indices are not independent. Substituting equations (1) and (2) into equation (3) gives
$C_{pk}$ as $C_p$ discounted by the centring.

$$C_{pk} = (1 - k) C_p \hspace{19em} (4)$$

Equation (4) is the whole relationship between them. $C_p$ is what the spread makes possible, $k$ is
the fraction of that potential the off-centring gives away, and $C_{pk}$ is what is left. It follows
that $C_{pk} \le C_p$ always, with equality only when the process is centred.

## 3. Physical Meaning

### 3.1. Defect Rate

Under the normal model the indices convert directly into a fraction outside the specification. For a
centred process both tails contribute and the fraction follows $C_p$; off centre, the near tail
dominates and the fraction follows $C_{pk}$. The derivation is in
[Appendix B](#appendix-b-derivation-of-equation-5).

$$p = \Phi\left( -3 CPL \right) + \Phi\left( -3 CPU \right) \hspace{19em} (5)$$

Table 1. Defect rate of a centred process, in parts per million.

| Cp | Cpk | k | Sigma level | Both tails, ppm | Near tail, ppm |
|---:|---:|---:|---:|---:|---:|
| 0.67 | 0.67 | 0 | 2.0 | 44431 | 22216 |
| 1.00 | 1.00 | 0 | 3.0 | 2700 | 1350 |
| 1.33 | 1.33 | 0 | 4.0 | 66.1 | 33.0 |
| 1.67 | 1.67 | 0 | 5.0 | 0.544 | 0.272 |
| 2.00 | 2.00 | 0 | 6.0 | 0.00197 | 0.000987 |

Every row is a centred process, so $k$ is 0 and equation (4) makes $C_{pk}$ equal to $C_p$. The
sigma level is the distance from the mean to the nearer limit in standard deviations, which is
$3 C_{pk}$ and is exactly the number equation (5) hands to $\Phi$. Moving such a process off centre
collapses the far tail, and what is left is the near tail column read at the smaller $C_{pk}$ the
move produces.

The top row worked through shows where the two rates come from. Equation (4) gives
$C_{pk} = (1 - 0) \times 0.67 = 0.67$, so $CPU$ and $CPL$ are both 0.67 and the sigma level is
$3 \times 0.67 = 2.01$, which the table rounds to 2.0. Equation (5) then reads
$p = \Phi(-2.01) + \Phi(-2.01)$, and $\Phi(-2.01) = 0.0222156$, so the near tail is 22216 ppm and
the two together are 44431 ppm.

The 1.33 row is why $C_p = 1.33$ became a common minimum requirement and $C_{pk} = 1.33$ a
common goal: it puts four standard deviations between the mean and the nearer limit, which leaves
room for the ordinary drift of a real process. The bottom row is the arithmetic behind the name of
the six sigma programme.

### 3.2. Reading the Three Together

No single index determines the defect rate, and Fig 1 is the demonstration.

<img src="process-capability-index_fig/process_capability_index.png" width="1000" style="max-width: 100%;" alt="Fig 1">

Fig 1. Three processes against the same specification, LSL 90 and USL 110, drawn on one density
scale. The dotted line is the process mean and each panel is labelled with its indices and with the
fraction outside the specification, which is far too small an area to see at this scale.

Panels (a) and (b) have the same $C_p$ of 1.33 and differ only in centring, and that difference
costs a factor of twenty in defect rate: 1350 ppm against 63 ppm. The spread is not the problem in
(a), and grinding on the spread would be wasted work; moving the mean back to 100 turns (a) into (b)
with no change to the process variation at all.

Panels (a) and (c) are the opposite case. Both have $C_{pk} = 1.00$, yet (a) makes 1350 ppm and (c)
makes 2700 ppm, and the two call for entirely different work. In (a) the process is narrow and
misaligned, which is usually a matter of adjusting a setpoint. In (c) the process is centred and too
wide, which requires reducing the variation itself, and no amount of adjustment will help. A
$C_{pk}$ quoted on its own hides that distinction.

Table 2. What the pair of indices indicates.

| Condition | Reading | Action |
|---|---|---|
| $C_p$ high, $k$ near 0 | Capable and centred | Hold |
| $C_p$ high, $k$ large | Capable but misaligned | Move the mean |
| $C_p$ low, $k$ near 0 | Centred but too wide | Reduce the variation |
| $C_p$ low, $k$ large | Both wrong | Centre first, then reduce |

The order in the last row matters. Centring is usually a setpoint change and is cheap; reducing
variation usually means changing hardware, recipe or material and is expensive. Equation (4) says
the cheap move recovers the factor $1/(1-k)$ immediately.

## 4. Application

### 4.1. Short-Term and Long-Term

The indices are computed from an estimate of $\sigma$, and there are two of those. Estimating
$\sigma$ from within-subgroup variation captures only the causes acting over a short interval;
estimating it from all the data across a long period captures the drift between subgroups as well.
The first gives $C_p$ and $C_{pk}$, the second gives the performance indices $P_p$ and $P_{pk}$,
which are defined by the same equations (1) and (3) with the long-term estimate substituted.

Because the long-term estimate is the larger of the two, $P_{pk} \le C_{pk}$ in practice. The gap
between them is not a defect in the calculation; it is the amount of drift the process has, and a
large gap is itself the finding. Quoting one of the two without saying which was computed makes the
number unusable.

### 4.2. Assumptions

Equation (5) and Table 1 hold under conditions that a real process meets only sometimes.

- Normality: the tail areas come from the normal distribution, and a skewed or bounded
  characteristic gives a different defect rate at the same index value.
- Stability: the indices describe the next period only if the process is in control, so the control
  chart comes first and the capability study second.
- Estimation: $\mu$ and $\sigma$ are estimated, so the indices are estimates with their own
  uncertainty, which is wide at the sample sizes commonly used.
- Two-sided specification: $C_p$ needs both limits. Where only one exists, $CPU$ or $CPL$ is
  reported alone and $C_p$ and $k$ do not apply.

### 4.3. In the Fab

For semiconductor process steps the specification is usually on a measured layer property, and the
capability study runs on one measurement per wafer, most often the wafer mean. The within-wafer
spread is a separate concern with its own index, and folding it into $\sigma$ mixes two variances
that call for different action, in the same way that equation (4) separates spread from centring.

The practical use is threefold: qualifying a new process or tool against its specification, comparing
chambers that run the same recipe, and setting the sampling rate, since a step with a high $C_{pk}$
needs measuring less often than one running near its limits.

### 4.4. Priority Classes

A fab measures far more parameters than it can hold to one standard, so it grades them by what
their control is for and gives each grade its own requirement.

Table 3. Foundry Control Priority and Wafer Acceptance Criteria

| Priority | Control objective | Cpk min | k max | Implied Cp min | Near tail, ppm | Acceptance |
|---|---|---:|---:|---:|---:|---:|
| 0 | Product Yield | 1.67 | 0.10 | 1.86 | 0.272 | 100% |
| 1 | Device Performance | 1.50 | 0.15 | 1.76 | 3.40 | 100% |
| 2 | Process Performance | 1.33 | 0.20 | 1.66 | 33.0 | 80% |
| 3 | Monitoring | 1.00 | 0.25 | 1.33 | 1350 | None |

The $C_{pk}$ column follows the long-standing ladder of minimum capability values: 1.33 as the
general minimum, 1.50 for a critical parameter, and 1.67 for one on a process still new
[[2](#ref-2)]. Monitoring sits below that ladder at 1.00, where the near tail is already 1350 ppm,
because the grade exists to keep a parameter charted rather than to protect the product. The $k$
column is the other half of the requirement, and equation (4) gives what it buys. A process
satisfies $C_{pk}$ alone by sitting exactly on target at exactly the required width, and it fails
as soon as the mean moves; capping $k$ raises the required $C_p$ to $C_{pk}/(1 - k)$,
which is the margin that holds the $C_{pk}$ through the drift the process will have.

<img src="process-capability-index_fig/priority_grades.png" width="1000" style="max-width: 100%;" alt="Fig 2">

Fig 2. The process that sits exactly on each grade, against the same specification, LSL 90 and
USL 110, drawn on one density scale. The dotted line is the process mean, which the $k$ maximum
holds below the midpoint, and each panel carries the indices that process realises. The ppm figure
is the whole fraction outside the specification, so it exceeds the near tail of Table 3 by the far
tail, which is negligible at every grade.

The Acceptance column carries the grade over from the parameter to the lot: it is the share of the
measured wafers that must meet the specification before the lot moves on. Priority 0 and
Priority 1 admit no failing wafer, Priority 2 admits one in five, and Priority 3 gates nothing at
all, since that grade charts a parameter rather than accepting product on it.

That implied width changes little across the three protecting grades, 1.86 against 1.66, while the
centring allowance at Priority 0 is twice as tight as at Priority 2, so what a higher grade demands
is centring rather than width — a setpoint rather than hardware. Monitoring is the one grade that
loosens both at once. The grade also fixes the action on a miss: hold the lot and take the chamber
out of production at Priority 0, engineering review before the lot moves on at Priority 1, a note
for the next scheduled maintenance at Priority 2, and a trend entry with no disposition at
Priority 3. The labels and the numbers are the convention of each fab rather than a published
standard, and what transfers is the shape.

## References

<a id="ref-1"></a>
[1] Kane, V. E. (1986). [Process Capability Indices](https://doi.org/10.1080/00224065.1986.11978984). *Journal of Quality Technology*, 18(1), 41–52.<br>
<a id="ref-2"></a>
[2] Montgomery, D. C. (2020). *Introduction to Statistical Quality Control* (8th ed.). Wiley.
ISBN 978-1-119-72309-7.

---

## Appendix A. Terminology

- **Capability study**: the exercise of estimating the indices from a sample of a stable process.
- **In control**: showing no control chart signal of a cause outside the ordinary variation.
- **Lot**: the group of wafers that moves through the process together.
- **Parts per million**: the fraction outside the specification multiplied by $10^{6}$.
- **Performance index**: a capability index computed with a long-term estimate of the standard
  deviation, written $P_p$ and $P_{pk}$.
- **Priority class**: a grade assigned to a measured parameter, carrying its own capability
  requirement and its own action on failure.
- **Sigma level**: the distance from the mean to the nearer specification limit in standard
  deviations, equal to $3 C_{pk}$.
- **Specification limit**: a bound the product must satisfy, set by design rather than estimated
  from the process.

## Appendix B. Derivation of Equation (5)

The process is normal with mean $\mu$ and standard deviation $\sigma$, so its density is the
following.

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right) \hspace{19em} (6)$$

A unit is defective when it falls below $LSL$ or above $USL$. The two events cannot both happen, so
the fraction outside the specification is the sum of the two areas.

$$p = \int_{-\infty}^{LSL} f(x) \mathrm{d}x + \int_{USL}^{\infty} f(x) \mathrm{d}x \hspace{19em} (7)$$

Substituting $z = (x - \mu)/\sigma$, so that $x = \mu + \sigma z$ and
$\mathrm{d}x = \sigma \mathrm{d}z$, turns the integrand into the standard normal density
$\varphi$, because the $\sigma$ the substitution brings in cancels the one in the denominator.

$$f(x) \mathrm{d}x = \frac{1}{\sigma\sqrt{2\pi}} \exp\left( -\frac{z^2}{2} \right) \sigma \mathrm{d}z = \varphi(z) \mathrm{d}z, \qquad \Phi(a) = \int_{-\infty}^{a} \varphi(z) \mathrm{d}z \hspace{19em} (8)$$

The limits move with the substitution, $x = LSL$ becoming $z = (LSL - \mu)/\sigma$ and $x = USL$
becoming $z = (USL - \mu)/\sigma$, so equation (7) becomes the following.

$$p = \int_{-\infty}^{(LSL - \mu)/\sigma} \varphi(z) \mathrm{d}z + \int_{(USL - \mu)/\sigma}^{\infty} \varphi(z) \mathrm{d}z \hspace{19em} (9)$$

The first integral is $\Phi$ by its definition in equation (8). The second runs to $+\infty$ and
has to be turned round first. Substituting $u = -z$ in it reverses the limits and leaves the
integrand unchanged, since $\varphi(-u) = \varphi(u)$, which gives an identity holding for any
$a$.

$$\int_{a}^{\infty} \varphi(z) \mathrm{d}z = \int_{-\infty}^{-a} \varphi(u) \mathrm{d}u = \Phi(-a) \hspace{19em} (10)$$

Applying it with $a = (USL - \mu)/\sigma$ puts both terms of equation (9) on the lower side.

$$p = \Phi\left( \frac{LSL - \mu}{\sigma} \right) + \Phi\left( -\frac{USL - \mu}{\sigma} \right) \hspace{19em} (11)$$

Equation (3) defines $CPL = (\mu - LSL)/(3\sigma)$ and $CPU = (USL - \mu)/(3\sigma)$, which
rearrange to $(LSL - \mu)/\sigma = -3 CPL$ and $(USL - \mu)/\sigma = 3 CPU$. Substituting the
two into equation (11) gives equation (5). The $\sigma$ has left the expression because the two
indices already carry it.

The last two columns of Table 1 are the two cases of that result. A centred process has $\mu = m$, so
$CPU = CPL = C_p$ by equations (2) and (3) and the two tails are equal, which gives
$p = 2\Phi(-3 C_p)$. Off centre the smaller of the two indices is $C_{pk}$, so $\Phi(-3 C_{pk})$
is the near tail on its own, and the far tail is small enough beside it to be dropped.
