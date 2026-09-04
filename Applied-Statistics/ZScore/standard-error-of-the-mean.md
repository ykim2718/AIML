# Standard Deviation of a Population and of Its Sample Mean
Rev. 1 | Created: 2026-08-30 | Updated: 2026-09-04 16:10 CDT

> A note on the relation between the standard deviation of an original distribution and the
> standard deviation of the mean of a sample drawn from it, on what the sample size does to that
> relation, and on the several symbols that are all read as sigma.

## 1. Scope

Two quantities are both called a standard deviation and are both written with a sigma, yet they
describe different things. One is the spread of the individual values in a population. The other
is the spread of the sample mean around the population mean, over repeated samples of the same
size. They are connected by the sample size alone.

This document states that connection, gives what it means for each of the two quantities, and
separates the symbols that share the name sigma. The derivation is in
[Appendix B](#appendix-b-derivation).

## 2. Relation

### 2.1. Statement

Let a sample of size $n$ be drawn independently from a population with standard deviation
$\sigma$, and let $\bar{X}$ be the mean of that sample. The standard deviation of $\bar{X}$ is
written $\sigma_{\bar{X}}$ and is called the standard error of the mean.

$$\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}}$$

Here $n$ is the count of individual observations drawn, so the relation holds the two standard
deviations and nothing else besides that count.

### 2.2. What the Two Describe

Table 1. The two standard deviations compared.

| Aspect | Population standard deviation | Standard error of the mean |
|---|---|---|
| Symbol | $\sigma$ | $\sigma_{\bar{X}}$ |
| Object measured | Individual values of the population | Sample means over repeated samples |
| Sample size | Not involved | Present as a factor $1/\sqrt{n}$ |
| Relative size | Larger | Smaller, for $n \gt 1$ |

The second is smaller because averaging cancels. A single draw can land far out in either tail
with nothing to offset it. A mean moves that far only when the extremes agree with each other,
and agreement is rarer than one extreme draw.

## 3. Effect of the Sample Size

The standard error falls as the square root of the sample size, not as the sample size itself.
Table 2 gives the factor by which it falls.

Table 2. Standard error as a fraction of the population standard deviation.

| Sample size | Square root | Standard error |
|---:|---:|---:|
| 1 | 1.000 | 1.000 |
| 2 | 1.414 | 0.707 |
| 4 | 2.000 | 0.500 |
| 9 | 3.000 | 0.333 |
| 16 | 4.000 | 0.250 |
| 25 | 5.000 | 0.200 |
| 100 | 10.000 | 0.100 |

Two rows carry the whole of the behaviour. At $n = 1$ the mean is the single observation itself,
so the standard error equals the population standard deviation and the two quantities coincide. At
$n = 100$ the standard error is one tenth of it.

The square root is what makes precision expensive. Halving the standard error costs four times the
sample, and reducing it by a factor of ten costs a hundred times the sample. Against that, the
relation also says that the sample mean is a sharper statement about the population mean than any
single observation is, and that its sharpness is known in advance from $n$ and $\sigma$ without
looking at the data.

## 4. Symbols Read as Sigma

Three of the symbols below are spoken as sigma, and the fourth is the one that is reached for when
a sigma is not wanted. They are not interchangeable.

Table 3. Symbols read as sigma.

| Symbol | Name | Meaning |
|---|---|---|
| $\sum$ | Capital sigma | Summation operator, an instruction to add terms |
| $\sigma$ | Lower-case sigma | Standard deviation of a population |
| $s$ | Latin s | Standard deviation computed from one sample |
| $\sigma_{\bar{X}}$ | Sigma with a subscript | Standard deviation of the sample mean |

The distinction between $\sigma$ and $s$ is the one that is most often lost. Both measure the
spread of individual values, but $\sigma$ is a property of the population and is unknown in
practice, while $s$ is computed from the observations at hand and changes from sample to sample.
When $\sigma$ is unknown, the standard error is estimated by replacing it with $s$, which gives
$s/\sqrt{n}$; this is an estimate and carries its own uncertainty, whereas $\sigma/\sqrt{n}$ does
not.

## References

<a id="ref-1"></a>
[1] Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury.
ISBN 978-0-534-24312-8.

<a id="ref-2"></a>
[2] Cochran, W. G. (1977). *Sampling Techniques* (3rd ed.). Wiley. ISBN 978-0-471-16240-7.
[https://www.wiley.com/en-us/Sampling+Techniques,+3rd+Edition-p-9780471162407](https://www.wiley.com/en-us/Sampling+Techniques,+3rd+Edition-p-9780471162407)

---

## Appendix A. Terminology

- **Population**: the complete set of values about which a statement is to be made.
- **Sample**: a subset of the population that is actually observed.
- **Sample mean**: the arithmetic mean of the observations in one sample, written $\bar{X}$.
- **Standard error**: the standard deviation of a statistic computed from a sample, here the
  standard deviation of the sample mean.
- **Variance**: the square of the standard deviation.

## Appendix B. Derivation

Let $X_1, \ldots, X_n$ be drawn independently from a population with mean $\mu$ and variance
$\sigma^2$, so that each draw has the same distribution and no draw carries information about
another.

$$E[X_i] = \mu, \qquad \mathrm{Var}[X_i] = \sigma^{2}, \qquad i = 1, \ldots, n$$

The sample mean is their sum divided by the count.

$$\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i$$

Two properties of the variance are needed. Scaling a variable by a constant scales its variance by
the square of that constant, and the variance of a sum of independent variables is the sum of
their variances.

$$\mathrm{Var}[aY] = a^{2} \mathrm{Var}[Y], \qquad \mathrm{Var}\left[ \sum_{i=1}^{n} X_i \right] = \sum_{i=1}^{n} \mathrm{Var}[X_i]$$

Apply the first with $a = 1/n$, then the second.

$$\mathrm{Var}\left[ \bar{X} \right] = \frac{1}{n^{2}} \mathrm{Var}\left[ \sum_{i=1}^{n} X_i \right] = \frac{1}{n^{2}} \sum_{i=1}^{n} \sigma^{2} = \frac{n\sigma^{2}}{n^{2}} = \frac{\sigma^{2}}{n}$$

The standard deviation is the positive square root of the variance, which gives the relation of
section 2.1.

$$\sigma_{\bar{X}} = \sqrt{\mathrm{Var}\left[ \bar{X} \right]} = \frac{\sigma}{\sqrt{n}}$$

Taking expectations of the same sum shows that the sample mean is centred on the population mean,
which is what makes the standard error a statement about accuracy rather than about bias
[[1](#ref-1)].

$$E\left[ \bar{X} \right] = \frac{1}{n} \sum_{i=1}^{n} E[X_i] = \frac{n\mu}{n} = \mu$$

The derivation uses independence only at the second variance property. Two cases break it.
Correlated draws add covariance terms that the sum of variances omits, and the result no longer
holds. Sampling without replacement from a finite population of size $N$ makes the draws slightly
dependent, and the variance acquires the finite population correction factor [[2](#ref-2)].

$$\mathrm{Var}\left[ \bar{X} \right] = \frac{\sigma^{2}}{n} \cdot \frac{N-n}{N-1}$$

The factor tends to one as $N$ grows with $n$ fixed, so the plain relation is the limiting case of
a population large enough that removing $n$ items does not change it.
