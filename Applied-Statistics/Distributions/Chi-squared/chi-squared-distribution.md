# The Chi-Squared Distribution
Rev. 0 | Created: 2026-09-03 | Updated: 2026-09-03 23:58 CDT

> A note on the distribution of a sum of squared standard normal variables: how it is built, what
> its density and moments are, how it relates to the other sampling distributions, and why it turns
> up in the tests that use it.

## 1. Scope

The chi-squared distribution is not usually met as a model of data. It is met as the distribution
of a statistic, and it earns its place because two very different quantities happen to follow it.
One is the scaled sample variance of a normal population. The other is Pearson's measure of the
gap between observed and expected counts. Both reduce to a sum of squares of quantities that are
approximately standard normal, and that sum is what the distribution describes.

This document defines the distribution, gives its density and moments, sets out its relations to
the normal, gamma, t and F distributions, and shows the two settings above. The derivations are in
[Appendix B](#appendix-b-derivations).

## 2. Definition

### 2.1. Construction from Normal Variables

Let $Z_1, \ldots, Z_k$ be independent standard normal variables. The chi-squared distribution with
$k$ degrees of freedom is the distribution of their sum of squares, written $X \sim \chi^2_k$.

$$X = \sum_{i=1}^{k} Z_i^{2} \hspace{19em} (1)$$

The construction fixes three things at once. The support is $x \ge 0$, because a sum of squares
cannot be negative. The distribution has one parameter, $k$, and it counts the squares being added.
The distribution is not symmetric, because squaring folds the two tails of the normal onto one
side.

### 2.2. Density Function

The density of $\chi^2_k$ on $x \gt 0$ involves the gamma function $\Gamma$.

$$f(x) = \frac{1}{2^{k/2}\,\Gamma(k/2)}\, x^{k/2 - 1} e^{-x/2} \hspace{19em} (2)$$

Nothing in this expression requires $k$ to be a whole number. The counting construction of
section 2.1 gives integer $k$, but the density is a proper density for every real $k \gt 0$, and
that is what lets the distribution be used with the fractional degrees of freedom that some
approximations produce [[4](#ref-4)].

The moment generating function is simpler than the density and is the tool used for most of what
follows.

$$M(t) = E\left[ e^{tX} \right] = (1 - 2t)^{-k/2}, \qquad t \lt \tfrac{1}{2} \hspace{19em} (3)$$

## 3. Properties

### 3.1. Moments

Differentiating equation (3) at the origin gives the moments. The mean is the degrees of freedom
and the variance is twice it.

$$E[X] = k, \qquad \mathrm{Var}[X] = 2k, \qquad \gamma_1 = \sqrt{8/k}, \qquad \gamma_2 = 12/k \hspace{19em} (4)$$

Here $\gamma_1$ is the skewness and $\gamma_2$ the excess kurtosis. Both fall as $k$ grows, which
is the whole of the distribution's large-sample behaviour in two numbers.

Table 1. Moments of the chi-squared distribution.

| Degrees of freedom | Mean | Variance | Skewness | Excess kurtosis |
|---:|---:|---:|---:|---:|
| 1 | 1 | 2 | 2.8284 | 12.0 |
| 2 | 2 | 4 | 2.0000 | 6.0 |
| 3 | 3 | 6 | 1.6330 | 4.0 |
| 5 | 5 | 10 | 1.2649 | 2.4 |
| 10 | 10 | 20 | 0.8944 | 1.2 |

That the mean equals $k$ is worth holding on to. A test statistic that is supposed to be $\chi^2_k$
and comes out near $k$ is unremarkable; one that comes out at several times $k$ is the signal the
test is looking for.

### 3.2. Additivity

If $X \sim \chi^2_m$ and $Y \sim \chi^2_n$ are independent, their sum is chi-squared and the
degrees of freedom add.

$$X + Y \sim \chi^2_{m+n} \hspace{19em} (5)$$

This follows immediately from equation (1), since the sum of two independent sums of squares is
again a sum of squares, and just as directly from equation (3), since the product of the two moment
generating functions is the one with exponent $-(m+n)/2$. Degrees of freedom behave like a count
because they are one.

### 3.3. Shape

The shape changes qualitatively with $k$. For $k = 1$ the density diverges at the origin, and for
$k = 2$ it is the exponential density, which is finite at the origin but still decreasing. From
$k = 3$ the density vanishes at the origin and is unimodal, with the mode at $k - 2$.

<img src="chi-squared-distribution_fig/chi_squared_distribution.png" width="900" style="max-width: 100%;" alt="Fig 1">

Fig 1. Chi-squared density (a) and distribution function (b) for five values of the degrees of
freedom. The density for $k = 1$ leaves the top of the panel; it is unbounded at the origin.

As $k$ grows the density becomes more symmetric, and the standardised variable converges to the
standard normal.

$$\frac{X - k}{\sqrt{2k}} \xrightarrow{d} N(0, 1) \hspace{19em} (6)$$

The convergence is slow, because the skewness in equation (4) falls only as $k^{-1/2}$, and that
slowness is why tables of the distribution stayed in use rather than a normal approximation. At
$k = 10$ the upper 5 percent point is 18.307, while equation (6) gives 17.356, an error of 5
percent. The cube-root transform of Wilson and Hilferty [[1](#ref-1)] is far better at the same
$k$, giving 18.292.

Table 2. Upper-tail critical values: the value the statistic exceeds with the stated probability.

| Degrees of freedom | 0.10 | 0.05 | 0.01 |
|---:|---:|---:|---:|
| 1 | 2.706 | 3.841 | 6.635 |
| 2 | 4.605 | 5.991 | 9.210 |
| 3 | 6.251 | 7.815 | 11.345 |
| 4 | 7.779 | 9.488 | 13.277 |
| 5 | 9.236 | 11.070 | 15.086 |
| 6 | 10.645 | 12.592 | 16.812 |
| 7 | 12.017 | 14.067 | 18.475 |
| 8 | 13.362 | 15.507 | 20.090 |
| 9 | 14.684 | 16.919 | 21.666 |
| 10 | 15.987 | 18.307 | 23.209 |
| 15 | 22.307 | 24.996 | 30.578 |
| 20 | 28.412 | 31.410 | 37.566 |
| 30 | 40.256 | 43.773 | 50.892 |

## 4. Relation to Other Distributions

The chi-squared distribution sits at the centre of the normal-theory sampling distributions, and
most of the others are built from it.

Table 3. Relations to other distributions.

| Distribution | Relation | Note |
|---|---|---|
| Normal | $Z^2 \sim \chi^2_1$ | The case $k = 1$ |
| Exponential | $\chi^2_2$ is exponential with mean 2 | The case $k = 2$ |
| Gamma | $\chi^2_k$ is gamma with shape $k/2$ and scale 2 | The general form of the density |
| Student t | $Z / \sqrt{V/k}$ with $V \sim \chi^2_k$ independent of $Z$ | The chi-squared is the denominator |
| F | $(V_1/k_1) / (V_2/k_2)$ with $V_1, V_2$ independent | A ratio of two chi-squared variables |
| Noncentral chi-squared | Sum of squares of normals with nonzero means | Used for the power of the tests below |

The first row can be checked against Table 2. The two-sided 5 percent point of the standard normal
is 1.96, and $1.96^2 = 3.8415$, which is the 5 percent point of $\chi^2_1$. A two-sided test on a
normal mean and a one-sided test on its square are the same test.

## 5. Role in Sampling

### 5.1. Sample Variance of a Normal Population

For $x_1, \ldots, x_n$ drawn independently from $N(\mu, \sigma^2)$, with sample mean $\bar{x}$ and
sample variance $s^2$, the scaled sample variance is chi-squared with one degree of freedom fewer
than the sample size, and it is independent of $\bar{x}$.

$$\frac{(n-1)s^{2}}{\sigma^{2}} \sim \chi^2_{n-1} \hspace{19em} (7)$$

The degree of freedom that is lost is the one spent estimating $\mu$. The deviations
$x_i - \bar{x}$ sum to zero by construction, so only $n-1$ of them can be chosen freely, and the
sum of their squares behaves like a sum of $n-1$ independent squares rather than $n$. That the
result is exactly chi-squared, and exactly independent of the mean, is Cochran's theorem
[[2](#ref-2)].

Inverting equation (7) gives the confidence interval for a normal variance, which is asymmetric
because the distribution is.

$$\left[ \frac{(n-1)s^{2}}{\chi^2_{n-1,\,\alpha/2}}, \ \frac{(n-1)s^{2}}{\chi^2_{n-1,\,1-\alpha/2}} \right] \hspace{19em} (8)$$

### 5.2. Degrees of Freedom

Across every use of the distribution, the degrees of freedom count the quantities that are free to
vary once the constraints are imposed, and each parameter estimated from the same data removes one
more. In equation (7) the single constraint is the estimated mean; in the tests of section 6 the
constraints come from the totals the fitted counts must reproduce. Getting this count wrong does
not make the statistic wrong; it makes the reference distribution wrong, which is harder to
notice.

## 6. Tests Built on the Distribution

### 6.1. Goodness of Fit

Given counts $O_1, \ldots, O_m$ falling into $m$ cells and the counts $E_1, \ldots, E_m$ that a
hypothesis expects, Pearson's statistic measures the gap [[3](#ref-3)].

$$X^{2} = \sum_{j=1}^{m} \frac{(O_j - E_j)^{2}}{E_j} \hspace{19em} (9)$$

Under the hypothesis, each term is roughly a squared standardised deviation, so the sum is roughly
a sum of squared standard normals and equation (1) applies. With no parameter estimated from the
data the degrees of freedom are $m - 1$, the one constraint being that the expected counts total
the observed count.

For 300 rolls of a die giving 43, 52, 54, 61, 48 and 42, every expected count is 50 and the
statistic is 5.160 on 5 degrees of freedom. Table 2 gives 11.070 as the 5 percent point, so the
counts are unremarkable; the exact upper-tail probability is 0.397.

The approximation behind the test is the normal approximation to each cell count, and it fails when
the expected counts are small. The usual working rule is that every expected count should be at
least 5.

### 6.2. Independence in a Contingency Table

For a table of counts with $r$ rows and $c$ columns, the hypothesis that the row and column
classifications are independent predicts each cell from the margins alone, and equation (9) is
applied to those predictions.

$$E_{ij} = \frac{R_i C_j}{n} \hspace{19em} (10)$$

The degrees of freedom follow section 5.2. The table has $rc$ cells, and the row and column
proportions estimated from the margins remove $(r-1) + (c-1)$ of them, leaving
$rc - 1 - (r-1) - (c-1) = (r-1)(c-1)$.

A two-by-two table with rows 38, 62 and 51, 49 has every expected count equal to 44.5 or 55.5, a
statistic of 3.421 on 1 degree of freedom, and an upper-tail probability of 0.064. Against the 3.841
of Table 2 the difference between the two rows is not significant at the 5 percent level.

## 7. Computation

The distribution and its inverse come from `scipy.stats.chi2`. The survival function `sf` gives the
upper tail directly and is preferred over `1 - cdf`, which loses precision far out in the tail,
while `isf` inverts it and produces the entries of Table 2.

```python
# Python
from scipy import stats

print(stats.chi2.isf(0.05, df=5))       # upper 5 percent point
print(stats.chi2.sf(5.16, df=5))        # upper tail probability of an observed statistic
print(stats.chi2.stats(df=5, moments='mv'))
```

```text
11.070497693516355
0.3966674666097388
(np.float64(5.0), np.float64(10.0))
```

The two tests of section 6 are each one call, and both return the statistic with its upper-tail
probability.

```python
# Python
import numpy as np
from scipy import stats

print(stats.chisquare(f_obs=np.array([43, 52, 54, 61, 48, 42])))
statistic, p_value, degrees, expected = stats.chi2_contingency(np.array([[38, 62], [51, 49]]),
                                                               correction=False)
print(round(statistic, 4), degrees, round(p_value, 4))
```

```text
Power_divergenceResult(statistic=np.float64(5.16), pvalue=np.float64(0.3966674666097388))
3.4214 1 0.0644
```

Two defaults are worth knowing. `stats.chisquare` assumes equal expected counts unless `f_exp` is
given, and it takes the degrees of freedom as $m - 1$ unless `ddof` says how many parameters were
estimated. `stats.chi2_contingency` applies Yates's continuity correction by default on a
two-by-two table, so reproducing the uncorrected statistic of section 6.2 needs
`correction=False`.

## References

<a id="ref-1"></a>
[1] Wilson, E. B., & Hilferty, M. M. (1931). The Distribution of Chi-Square. *Proceedings of the
National Academy of Sciences*, 17(12), 684–688.
[https://doi.org/10.1073/pnas.17.12.684](https://doi.org/10.1073/pnas.17.12.684)

<a id="ref-2"></a>
[2] Cochran, W. G. (1934). The Distribution of Quadratic Forms in a Normal System, with
Applications to the Analysis of Covariance. *Mathematical Proceedings of the Cambridge
Philosophical Society*, 30(2), 178–191.
[https://doi.org/10.1017/S0305004100016595](https://doi.org/10.1017/S0305004100016595)

<a id="ref-3"></a>
[3] Pearson, K. (1900). On the Criterion that a Given System of Deviations from the Probable in the
Case of a Correlated System of Variables is Such that it Can be Reasonably Supposed to have Arisen
from Random Sampling. *The London, Edinburgh, and Dublin Philosophical Magazine and Journal of
Science*, 50(302), 157–175.
[https://doi.org/10.1080/14786440009463897](https://doi.org/10.1080/14786440009463897)

<a id="ref-4"></a>
[4] Johnson, N. L., Kotz, S., & Balakrishnan, N. (1994). *Continuous Univariate Distributions*
(Vol. 1, 2nd ed.). Wiley. ISBN 978-0-471-58495-7.

---

## Appendix A. Terminology

- **Cell**: one class of a categorical classification, holding a count.
- **Contingency table**: a table of counts cross-classified by two categorical variables.
- **Degrees of freedom**: the number of quantities free to vary once the constraints on them are
  imposed; the parameter of the chi-squared distribution.
- **Excess kurtosis**: the fourth standardised moment less 3, so that the normal distribution has
  the value 0.
- **Margin**: a row total or a column total of a table of counts.
- **Moment generating function**: $E[e^{tX}]$ as a function of $t$, whose derivatives at the origin
  are the moments.
- **Skewness**: the third standardised moment, zero for a symmetric distribution.
- **Support**: the set of values a distribution gives positive probability to.
- **Survival function**: one minus the distribution function, that is the upper-tail probability.

## Appendix B. Derivations

### B.1. Density for One Degree of Freedom

Take $X = Z^2$ with $Z$ standard normal. For $x \gt 0$ the event $X \le x$ is the event
$-\sqrt{x} \le Z \le \sqrt{x}$, so the distribution function of $X$ follows from that of $Z$, using
the symmetry of the normal density $\phi$ in the second step.

$$F(x) = \Phi(\sqrt{x}) - \Phi(-\sqrt{x}) = 2\Phi(\sqrt{x}) - 1 \hspace{19em} (11)$$

Differentiating, with $\phi(z) = e^{-z^2/2}/\sqrt{2\pi}$ and the chain rule contributing
$1/(2\sqrt{x})$, gives the density.

$$f(x) = 2\,\phi(\sqrt{x}) \cdot \frac{1}{2\sqrt{x}} = \frac{1}{\sqrt{2\pi}}\, x^{-1/2} e^{-x/2} \hspace{19em} (12)$$

This is equation (2) at $k = 1$, since $2^{1/2}\Gamma(1/2) = \sqrt{2\pi}$.

### B.2. Moment Generating Function

For a single square, substitute $u = z\sqrt{1-2t}$ in the defining integral. The substitution is
legitimate only for $t \lt 1/2$, which is where the exponent stays negative and the integral
converges.

$$E\left[ e^{tZ^{2}} \right] = \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi}} e^{tz^{2}} e^{-z^{2}/2}\, dz = \frac{1}{\sqrt{1-2t}} \int_{-\infty}^{\infty} \frac{e^{-u^{2}/2}}{\sqrt{2\pi}}\, du = (1-2t)^{-1/2} \hspace{19em} (13)$$

The remaining integral is the total mass of the standard normal density and equals 1. Since the
$Z_i$ in equation (1) are independent, the moment generating function of their sum is the product
of $k$ such factors, which is equation (3). Equation (5) is the same statement read backwards: the
product of $(1-2t)^{-m/2}$ and $(1-2t)^{-n/2}$ is $(1-2t)^{-(m+n)/2}$.

### B.3. Mean and Variance

Expand equation (3) about the origin, or differentiate it twice. The first two derivatives at
$t = 0$ give the first two raw moments.

$$M'(t) = k(1-2t)^{-k/2 - 1}, \qquad M''(t) = k(k+2)(1-2t)^{-k/2 - 2} \hspace{19em} (14)$$

Setting $t = 0$ gives $E[X] = k$ and $E[X^2] = k(k+2)$, so the variance is
$k(k+2) - k^2 = 2k$, which is equation (4).

### B.4. Loss of One Degree of Freedom

Write the sum of squared deviations from the true mean and add and subtract $\bar{x}$ inside each
square. The expansion has three pieces: the sum of squared deviations from $\bar{x}$, a cross term,
and $n(\bar{x} - \mu)^2$.

$$\sum_{i=1}^{n} (x_i - \mu)^{2} = \sum_{i=1}^{n} (x_i - \bar{x})^{2} + 2(\bar{x} - \mu)\sum_{i=1}^{n}(x_i - \bar{x}) + n(\bar{x} - \mu)^{2} \hspace{19em} (15)$$

The cross term vanishes because the deviations from $\bar{x}$ sum to zero, leaving two pieces.
Dividing by $\sigma^2$, the left side is a sum of $n$ squared standard normals and so is
$\chi^2_n$. The last term is $\left( (\bar{x}-\mu)/(\sigma/\sqrt{n}) \right)^2$, a single squared
standard normal, and so is $\chi^2_1$. Cochran's theorem [[2](#ref-2)] says that the two terms on
the right are independent and each chi-squared, so the degrees of freedom subtract and the first
term is $\chi^2_{n-1}$. That first term is $(n-1)s^2/\sigma^2$, which is equation (7).
