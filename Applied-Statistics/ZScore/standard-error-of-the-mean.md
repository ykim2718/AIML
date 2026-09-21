# Standard Deviation of a Population and of Its Sample Mean
Rev. 4 | Created: 2026-08-30 | Updated: 2026-09-21 16:40 CDT

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
to offset it, while a mean moves that far only when several draws agree in direction, and such
agreement is rarer than one extreme draw.

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

```math
E[X_i] = \mu, \qquad \mathrm{Var}[X_i] = \sigma^{2}, \qquad i = 1, \ldots, n \hspace{19em} (3)
```

The sample mean is their sum divided by the count.

```math
\bar{X} = \frac{1}{n} \sum_{i=1}^{n} X_i \hspace{19em} (4)
```

Two properties of the variance are needed. Scaling a variable by a constant scales its variance by
the square of that constant, and the variance of a sum of independent variables is the sum of
their variances.

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

The derivation uses independence only at the second property of equation (5). Two cases break that
independence. Correlated draws add covariance terms that the sum of variances omits, and equation
(6) no longer holds. Sampling without replacement from a finite population of size $N$ makes the
draws slightly dependent, and the variance acquires the finite population correction factor
[[2](#ref-2)].

```math
\mathrm{Var}\left[ \bar{X} \right] = \frac{\sigma^{2}}{n} \cdot \frac{N-n}{N-1} \hspace{19em} (9)
```

The factor tends to one as $N$ grows with $n$ fixed, so equation (1) is the limiting case of a
population large enough that removing $n$ items does not change it. At $N = 1000$ and $n = 100$ the
factor is 0.901, so the standard error is 0.949 times what equation (1) gives.
