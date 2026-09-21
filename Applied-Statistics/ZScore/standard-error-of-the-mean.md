# Standard Deviation of a Population and of Its Sample Mean
Rev. 9 | Created: 2026-08-30 | Updated: 2026-09-21 17:21 CDT

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
[Appendix B](#appendix-b-derivation), and the terms are defined in
[Appendix A](#appendix-a-terminology).

## 2. Relation

### 2.1. Statement

Let a sample of size $n$ be drawn independently from a population with standard deviation
$\sigma$, and let $\bar{X}$ be the mean of that sample. The standard deviation of $\bar{X}$ is
written $\sigma_{\bar{X}}$ and is called the standard error of the mean.

```math
\sigma_{\bar{X}} = \frac{\sigma}{\sqrt{n}} \hspace{19em} (1)
```

Here $n$ is the count of individual observations drawn, so the relation holds the two standard
deviations and nothing else besides that count. It assumes independent draws from a single
population; [Appendix B](#appendix-b-derivation) gives the two cases that break that
assumption.

### 2.2. What the Two Describe

Table 1. The two standard deviations compared.

| Aspect          | Population standard deviation       | Standard error of the mean         |
| :-------------: | :---------------------------------: | :--------------------------------: |
| Symbol          | $\sigma$                            | $\sigma_{\bar{X}}$                 |
| Object measured | Individual values of the population | Sample means over repeated samples |
| Sample size     | Not involved                        | Present as a factor $1/\sqrt{n}$   |
| Relative size   | Larger                              | Smaller, for $n \gt 1$             |

Averaging cancels the extremes. A single draw can land far out in either tail with no other value
to offset it, while a mean moves that far only when several of the $X_1, \ldots, X_n$ that make
up a sample of size $n$ lean the same way, and such leaning is rarer than one extreme draw.

## 3. Effect of the Sample Size

The standard error falls as the square root of the sample size, not as the sample size itself.
Table 2 gives the factor by which it falls.

Table 2. Standard error as a fraction of the population standard deviation.

| Sample size | Square root | Standard error |
| :---------: | :---------: | :------------: |
| 1           | 1.000       | 1.000          |
| 2           | 1.414       | 0.707          |
| 4           | 2.000       | 0.500          |
| 9           | 3.000       | 0.333          |
| 16          | 4.000       | 0.250          |
| 25          | 5.000       | 0.200          |
| 100         | 10.000      | 0.100          |

Two rows carry the whole of the behaviour. At $n = 1$ the mean is the single observation itself,
so the standard error equals the population standard deviation and the two quantities coincide. At
$n = 100$ the standard error is one tenth of the population standard deviation.

The square root sets the price of precision. Halving the standard error costs four times the
sample, and reducing it by a factor of ten costs a hundred times the sample. Against that, the
relation also says that the sample mean is a sharper statement about the population mean than any
single observation is, and that its sharpness is known in advance from $n$ and $\sigma$ without
looking at the data.

## 4. Symbols Read as Sigma

Table 3 lists four symbols. Three of them are spoken as sigma, and the fourth, Latin $s$, carries
the spread of individual values computed from one sample. Each names a different quantity, so one
cannot stand in for another.

Table 3. Symbols read as sigma.

| Symbol             | Name                   | Meaning                                         |
| :----------------: | :--------------------: | :---------------------------------------------: |
| $\sum$             | Capital sigma          | Summation operator, an instruction to add terms |
| $\sigma$           | Lower-case sigma       | Standard deviation of a population              |
| $s$                | Latin s                | Standard deviation computed from one sample     |
| $\sigma_{\bar{X}}$ | Sigma with a subscript | Standard deviation of the sample mean           |

The distinction between $\sigma$ and $s$ is the one that is most often lost. Both measure the
spread of individual values, but $\sigma$ is a property of the population and is unknown in
practice, while $s$ is computed from the observations at hand and changes from sample to sample.
When $\sigma$ is unknown, the standard error is estimated by replacing it with $s$, which gives
$s/\sqrt{n}$.

```math
\hat{\sigma}_{\bar{X}} = \frac{s}{\sqrt{n}} \hspace{19em} (2)
```

Equation (2) is an estimate and carries its own uncertainty, which is why an interval built on it
uses the $t$ distribution with $n - 1$ degrees of freedom rather than the normal quantiles that
equation (1) admits [[1](#ref-1)].

## References

<a id="ref-1"></a>
[1] Casella, G., & Berger, R. L. (2002). *Statistical Inference* (2nd ed.). Duxbury.
ISBN 978-0-534-24312-8.<br>
<a id="ref-2"></a>
[2] Cochran, W. G. (1977). [*Sampling Techniques*](https://www.wiley.com/en-us/Sampling+Techniques,+3rd+Edition-p-9780471162407) (3rd ed.). Wiley. ISBN 978-0-471-16240-7.

---

## Appendix A. Terminology

- **Covariance**: the expected product of the deviations of two variables from their means.
- **Draw**: one observation taken from the population, or the act of taking it.
- **Population**: the complete set of values about which a statement is to be made.
- **Sample**: a subset of the population that is actually observed.
- **Sample mean**: the arithmetic mean of the observations in one sample, written $\bar{X}$.
- **Standard error**: the standard deviation of a statistic computed from a sample, here the
  standard deviation of the sample mean.
- **Variance**: the square of the standard deviation.

## Appendix B. Derivation

Let $X_1, \ldots, X_n$ be drawn from a population with mean $\mu$ and variance $\sigma^2$, with
every draw following the same distribution (identically distributed) and no draw carrying
information about another (independent).

```math
E[X_i] = \mu, \qquad \mathrm{Var}[X_i] = \sigma^{2}, \qquad i = 1, \ldots, n \hspace{19em} (3)
```

The sample mean is their sum divided by the count.

```math
\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i \hspace{19em} (4)
```

Two properties of the variance are needed. Scaling a variable by a constant scales its variance by
the square of that constant, and the variance of a sum of independent variables is the sum of
their variances. [B.1](#b1-the-two-variance-properties) derives both.

```math
\mathrm{Var}[aY] = a^{2} \mathrm{Var}[Y], \qquad \mathrm{Var}\left[ \sum_{i=1}^{n} X_i \right] = \sum_{i=1}^{n} \mathrm{Var}[X_i] \hspace{19em} (5)
```

Apply the first property of equation (5) with $a = 1/n$, then the second.

```math
\mathrm{Var}\left[ \bar{X} \right] = \frac{1}{n^{2}} \mathrm{Var}\left[ \sum_{i=1}^{n} X_i \right] = \frac{1}{n^{2}} \sum_{i=1}^{n} \sigma^{2} = \frac{n\sigma^{2}}{n^{2}} = \frac{\sigma^{2}}{n} \hspace{19em} (6)
```

The standard deviation is the positive square root of the variance, which gives equation (1).

```math
\sigma_{\bar{X}} = \sqrt{\mathrm{Var}\left[ \bar{X} \right]} = \frac{\sigma}{\sqrt{n}} \hspace{19em} (7)
```

Taking expectations of equation (4) shows that the sample mean is centred on the population mean,
so equation (7) measures spread alone [[1](#ref-1)].

```math
E\left[ \bar{X} \right] = \frac{1}{n} \sum_{i=1}^{n} E[X_i] = \frac{n\mu}{n} = \mu \hspace{19em} (8)
```

The two assumptions are used in different places. Identically distributed is what lets equation
(6) put the same $\sigma^{2}$ in every term of the sum, and independent is what the second
property of equation (5) needs, and only as far as a zero covariance between draws.
Two cases give a covariance other than zero. 1) Correlated draws add the covariance terms that
the sum of variances omits, and equation (6) no longer holds. 2) Sampling without replacement,
where a drawn item is held out instead of being returned to the population, leaves fewer values
for the draws that follow, so each draw shifts the distribution of the next one. The variance of
the mean then carries the finite population correction factor $(N-n)/(N-1)$, for a population of
size $N$ [[2](#ref-2)].

```math
\mathrm{Var}\left[ \bar{X} \right] = \frac{\sigma^{2}}{n} \cdot \frac{N-n}{N-1} \hspace{19em} (9)
```

The factor tends to one as $N$ grows with $n$ fixed, so equation (1) is the limiting case of a
population large enough that removing $n$ items does not change it. At $N = 1000$ and $n = 100$ the
factor is 0.901, so the standard error is 0.949 times what equation (1) gives. At $n = N$ the
factor is zero, since a sample that holds the whole population leaves the sample mean equal to the
population mean with nothing left to vary.

### B.1 The Two Variance Properties

Both properties of equation (5) follow from the definition of the variance,
$\mathrm{Var}[Y] = E[(Y - E[Y])^{2}]$. Taking the constant out of the square gives the first.

```math
\mathrm{Var}[aY] = E\left[ (aY - aE[Y])^{2} \right] = a^{2} E\left[ (Y - E[Y])^{2} \right] = a^{2} \mathrm{Var}[Y] \hspace{19em} (10)
```

Expanding the square of the sum leaves every pair of terms, which collect into the variances on
the diagonal and the covariances off it.

```math
\mathrm{Var}\left[ \sum_{i=1}^{n} X_i \right] = \sum_{i=1}^{n} \sum_{j=1}^{n} \mathrm{Cov}(X_i, X_j) = \sum_{i=1}^{n} \mathrm{Var}[X_i] + \sum_{i \ne j} \mathrm{Cov}(X_i, X_j) \hspace{19em} (11)
```

Independent draws split the expectation of a product into the product of the expectations, and
each deviation has expectation zero, so every term off the diagonal vanishes and equation (11)
leaves the second property of equation (5).
[B.2](#b2-covariance-and-the-product-of-expectations) reaches the same split from a zero
covariance, which is all that the second property needs.

```math
\mathrm{Cov}(X_i, X_j) = E\left[ (X_i - \mu)(X_j - \mu) \right] = E[X_i - \mu] \cdot E[X_j - \mu] = 0, \qquad i \ne j \hspace{19em} (12)
```

### B.2 Covariance and the Product of Expectations

Multiplying out the definition of the covariance and then taking the expectation term by term,
which the linearity of $E[\cdot]$ allows, leaves the expectation of the product against the
product of the means $\mu_i = E[X_i]$ and $\mu_j = E[X_j]$.

```math
\begin{aligned}
\mathrm{Cov}(X_i, X_j) &= E\left[ (X_i - \mu_i)(X_j - \mu_j) \right] \\
&= E\left[ X_i X_j - \mu_j X_i - \mu_i X_j + \mu_i \mu_j \right] \\
&= E[X_i X_j] - \mu_j E[X_i] - \mu_i E[X_j] + \mu_i \mu_j \\
&= E[X_i X_j] - \mu_i \mu_j
\end{aligned}
\hspace{19em} (13)
```

With a covariance of zero, equation (13) is zero.

```math
E[X_i X_j] - \mu_i \mu_j = 0 \hspace{19em} (14)
```

Moving $\mu_i \mu_j$ to the other side splits the expectation of the product into the product of
the expectations. Equation (12) reaches the same split from independent draws, which is the
stronger assumption.

```math
E[X_i X_j] = E[X_i] \cdot E[X_j] \hspace{19em} (15)
```
