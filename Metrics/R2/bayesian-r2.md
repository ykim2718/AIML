# Bayesian R² — Obtaining R² as a Distribution Instead of a Point
Rev. 1 | Created: 2026-05-31 | Updated: 2026-08-16 18:20 CDT

> A note on computing one R² per posterior draw to obtain R² as a distribution,
> organized as definition, computation, interpretation, and applicability.

## 1. Motivation

Standard R² returns a single number. When a fit reports `R² = 0.87`, that number
alone does not say whether the value would collapse to 0.60 under a slightly
different sample or hold firmly near 0.85. The magnitude of explanatory power and
the confidence in that magnitude are two different pieces of information, and
standard R² carries only the first.

Bayesian R² propagates the predictive uncertainty of the model into R² itself, so
that R² arrives as a distribution rather than a point. This is why the result takes
the form `0.87 [0.81, 0.91]`.

**Table 1. Standard R² vs Bayesian R²**

| Aspect | Standard R² | Bayesian R² |
|---|---|---|
| Output | A single scalar | A distribution over S draws |
| Uncertainty | Absent | Expressed as an interval |
| Range | Can be negative, never exceeds 1 | Always within [0, 1] |
| Input required | One set of predictions | S sets of posterior draws |
| Failure mode | Instability stays invisible | Without draws the metric cannot be computed at all |

## 2. Core Idea

For each draw s taken from the posterior distribution, one R² is computed. With S
draws there are S values of R², and that collection is the posterior distribution
of R². When the model parameters are uncertain, the predictions differ from draw to
draw, and when the predictions differ, so does the amount of variation the model
explains. Bayesian R² does not erase that movement; it reports it.

## 3. Definition

### 3.1. Divergence Problem of the Standard Form

Applying the standard definition `R² = 1 − SS_res / SS_tot` to individual posterior
draws produces values outside [0, 1]. Under a least-squares fit the residual sum of
squares cannot exceed the total sum of squares, but an individual posterior draw is
not the least-squares solution for that data. When a prior or the shrinkage of a
hierarchical model pulls predictions away from the data, the residual sum of squares
of a single draw can exceed the total sum of squares, and the R² of that draw turns
negative. A collection of values whose sign flips from draw to draw yields neither an
interpretable median nor an interpretable interval. A concrete instance appears in
[Appendix B. Worked Example](#appendix-b-worked-example).

### 3.2. Gelman Formulation

The remedy proposed by Gelman et al. (2019) is to replace the denominator with a sum
of two non-negative quantities. Removing the subtraction removes the place where the
value could diverge.

```text
             Var_fit^(s)                    explained variation
R2^(s) = ─────────────────────────  =  ────────────────────────────────────────
         Var_fit^(s) + Var_res^(s)      explained variation + residual variation
```

- `Var_fit` — the variance of the predictions `y_hat_i^(s)` of draw s taken across the data points i, which is the variation the model explains.
- `Var_res` — the variation that draw s leaves unexplained.

Both terms are variances and therefore non-negative, and because the denominator is
their sum, the ratio is structurally confined to [0, 1]. The value is 0 when the
numerator vanishes and approaches 1 as `Var_res` approaches 0. Unlike the standard
form, no posterior draw can push it outside its definition.

Here `Var` denotes the sample variance taken over the data points i = 1 ... n.

$$\mathrm{Var}(z) = \frac{1}{n-1}\sum_{i=1}^{n}\left(z_i - \bar{z}\right)^2$$

### 3.3. Choice of Residual Variance

There are two ways to define `Var_res`, and the choice changes what R² means.

**Table 2. Two choices of residual variance**

| Variant | Definition | Meaning | Note |
|---|---|---|---|
| Empirical | `Var(y_i − y_hat_i^(s))` | The variance of the residuals actually left over | Requires the observations y and works for any model family |
| Model-based | `(sigma^(s))^2` | The noise variance the model claims for itself | Used with Gaussian likelihoods and computed without y |

The two values agree when the model fits well, and any gap between them is itself a
diagnostic. A model-based value that is visibly smaller means the model underestimates
its own noise, and the model-based R² then reads higher than the empirical one. The
two variants should not be mixed within one document, and the choice should be stated.

## 4. Computation

### 4.1. Procedure

```text
1. Draw S posterior samples (MCMC, ensemble members, or dropout passes)
2. For each sample s:
     Var_fit  = variance of the predicted values over data points
     Var_res  = variance of the residuals, or the model noise variance
     R2^(s)   = Var_fit / (Var_fit + Var_res)
3. Collect {R2^(1) ... R2^(S)} as the posterior distribution of R2
     median   -> point estimate
     quantiles-> credible interval
```

The input is a single matrix of shape `(S, n)` whose rows are posterior draws and
whose columns are data points. The variances run along the column direction, that is
across data points, never across draws. Taking the other axis measures how much the
prediction for one point moves from draw to draw, which is a different quantity
altogether and not R².

### 4.2. Implementation

```python
# Python
import numpy as np


def bayesian_r2(y_pred_draws: np.ndarray,
                y_true: np.ndarray = None,
                sigma_draws: np.ndarray = None) -> np.ndarray:
    """Compute the posterior distribution of Bayesian R2.

    Args:
        y_pred_draws: posterior draws of the fitted mean, shape (S, n).
        y_true: observed values, shape (n,). Selects the empirical variant.
        sigma_draws: posterior draws of the noise scale, shape (S,).
                     Selects the model-based variant.

    Returns:
        R2 draws of shape (S,), every element inside [0, 1].
    """
    if (y_true is None) == (sigma_draws is None):
        raise ValueError("y_true and sigma_draws are mutually exclusive; pass exactly one.")
    if y_pred_draws.ndim != 2:
        raise ValueError(f"y_pred_draws must be 2-D (S, n), got {y_pred_draws.shape}.")

    var_fit = y_pred_draws.var(axis=1, ddof=1)          # variance over data points
    if y_true is not None:
        var_res = (y_true[None, :] - y_pred_draws).var(axis=1, ddof=1)
    else:
        var_res = sigma_draws ** 2

    return var_fit / (var_fit + var_res)


# posterior_mean_draws: shape (S, n), y_observed: shape (n,)
r2_draws = bayesian_r2(y_pred_draws=posterior_mean_draws, y_true=y_observed)
point = np.median(r2_draws)
lower, upper = np.quantile(r2_draws, [0.05, 0.95])      # 90% credible interval
```

The arithmetic is two variances and one division; the cost of the method sits entirely
in obtaining the posterior draws.

## 5. Interpretation

### 5.1. Point Estimate and Credible Interval

The posterior distribution of R² is not symmetric. The closer R² comes to 1, the more
the upper side is blocked by the ceiling at 1, which leaves a long tail on the left.
In such a distribution the mean is dragged toward the tail, so the median serves as
the point estimate and quantiles define the interval.

The width of that interval is the confidence in the explanatory power.
`0.87 [0.85, 0.89]` and `0.87 [0.61, 0.95]` share a point estimate but are entirely
different results, and the latter means the data are too thin to accept a model on the
strength of R².

This interval is a credible interval, not a confidence interval. Reading it directly as
"R² lies in this interval with probability 90%" is the property that distinguishes a
credible interval, defined in [Appendix A. Terminology](#appendix-a-terminology).

### 5.2. Uncertainty Decomposition

A single result carries two separable kinds of uncertainty.

- The width of the R² distribution — uncertainty from not having pinned down the parameters, which shrinks as more data arrive.
- The magnitude of `Var_res` — noise inherent in the data, which does not shrink with more data.

A wide interval therefore calls for a larger sample, whereas a narrow interval around a
low R² calls for different input variables or a different model structure. Standard R²
cannot separate the two and returns the same number in both situations.

## 6. Tools

**Table 3. Available implementations**

| Environment | Interface | Note |
|---|---|---|
| R | `bayes_R2(fit)` — rstanarm, brms | Maintained by the authors of the paper and taken as the reference implementation |
| R | `loo_R2(fit)` — rstanarm, brms | The variant carrying an out-of-sample correction |
| Python | `az.r2_score()` — ArviZ | Takes predictive samples and observations and returns a summary |
| Any | Custom implementation | Two variances and a division, so it ports with little effort |

A custom implementation should first be reconciled against the reference implementation
on the same data. A mismatch usually traces back to the variant chosen in 3.3 or to the
axis direction described in 4.1.

## 7. Prerequisites

The binding constraint lies in applicability rather than in the definition. Bayesian R²
requires genuine posterior draws.

### 7.1. Applicable Models

**Table 4. Model applicability**

| Model type | Applicable | Reason |
|---|---|---|
| Bayesian models such as Stan, PyMC, brms | Yes | MCMC supplies posterior draws directly |
| Deep ensemble | Yes | Each member plays the role of one draw |
| MC dropout | Yes | Repeated forward passes generate the draws |
| Bootstrap ensemble | Yes | Each resample is refitted to produce a draw |
| Single Gaussian head emitting one mu and one sigma | No | Without draws no distribution can be formed |
| Point-estimate models such as a plain GBM or a single network | No | Only one set of predictions exists |

Deep ensembles and MC dropout do not yield posterior draws in the strict sense, but they
act as samples from the predictive distribution, so the same computation holds. When the
member count is as low as five, however, the quantiles are unstable and the interval
should not be read narrowly.

### 7.2. Inapplicable Models

A model that emits a single `mu` and a single `sigma` has a predictive distribution but
no posterior draws. The `sigma` expresses aleatoric uncertainty, yet the parameter
uncertainty that would make R² move is absent, so R² is fixed at one value regardless.
Such a case calls for a metric that does not demand draws, such as the CRPS Skill Score
discussed in 8. Comparison.

## 8. Comparison

**Table 5. Bayesian R² vs CRPS Skill Score**

| Aspect | Bayesian R² | CRPS Skill Score |
|---|---|---|
| Output | Point estimate and interval | A single score |
| Interval | Present | Absent |
| Sample requirement | S posterior draws | Only a predictive distribution |
| Question answered | How large is the explanatory power and how certain is it | How well did the predictive distribution match |
| Baseline | The total variation of the data | An explicitly designated baseline model |

The two metrics are not competitors; their applicability differs. Where posterior draws
can be obtained, Bayesian R² carries more information because it also supplies an
interval, and where they cannot, the CRPS Skill Score is what remains.

## 9. Pitfalls

**Table 6. Common mistakes**

| Mistake | Consequence | Fix |
|---|---|---|
| Feeding posterior predictive samples into `Var_fit` | Noise enters the numerator and inflates R² | Use posterior draws of the predictive mean |
| Taking the variance along the draw axis | The result is a different quantity, not R² | Take it along the data-point axis |
| Using the mean as the point estimate | Skewness drags the value downward | Use the median |
| Reporting a training-data value as generalization performance | The value is optimistically biased | Use out-of-sample data or the loo variant |
| Reporting a narrow interval computed from few draws | The interval itself is unstable | Increase the draw count and confirm convergence first |
| Mixing the two variants when comparing | Model-to-model comparison loses meaning | Fix one variant and state it |

The first row is the most frequent. Most tools expose both the posterior draws of the
predictive mean and the posterior predictive samples that add observation noise on top,
and `Var_fit` takes the former. Supplying the latter moves unexplained variation into
the explained term, so R² reads higher than it is.

## 10. Summary

Bayesian R² computes `Var_fit / (Var_fit + Var_res)` for each posterior draw s and thereby
obtains R² as a distribution. Because the denominator is a sum of two non-negative terms,
the divergence problem of standard R² disappears structurally, and the method yields a
point estimate together with a credible interval. In exchange, it applies only to Bayesian
models and ensemble-style models from which draws can be taken.

---

## Appendix A. Terminology

- **aleatoric uncertainty** — Uncertainty originating in the noise of the data itself, which does not shrink as more data arrive.
- **credible interval** — An interval defined by quantiles of the posterior distribution, read directly as the probability that the parameter lies within it.
- **CRPS** — Continuous Ranked Probability Score. A score comparing an entire predictive distribution against a single observation, where smaller is better.
- **CRPS Skill Score** — CRPS normalized by the CRPS of a baseline model into the form `1 − CRPS_model / CRPS_baseline`, where larger is better.
- **deep ensemble** — A predictive distribution built from several networks trained independently under different initializations or data orderings.
- **epistemic uncertainty** — Uncertainty arising from not having identified the model and its parameters, which shrinks as more data arrive.
- **LOO** — Leave-One-Out cross-validation. Estimating out-of-sample performance by holding out one observation at a time.
- **MC dropout** — Obtaining predictive samples by repeating forward passes with dropout left active at inference time.
- **MCMC** — Markov Chain Monte Carlo. The standard family of algorithms for drawing samples from a posterior distribution.
- **OLS** — Ordinary Least Squares. The fit minimizing the sum of squared residuals.
- **Pearson R²** — The square of the Pearson correlation coefficient between the observations and the predictions.
- **posterior distribution** — The distribution of the parameters after the data have been observed.
- **posterior draw** — One sample taken from the posterior distribution.
- **posterior predictive sample** — A sample at the scale of an observation, generated by adding observation noise on top of a posterior draw.
- **shrinkage** — The pull a prior or a hierarchical structure exerts on estimates toward a common center.

## Appendix B. Worked Example

This appendix computes both metrics by hand on one small dataset so that the difference
between them is visible in numbers rather than in argument.

### B.1. Data and Reference Fit

The dataset has n = 8 points. The OLS fit gives an intercept of −0.018 and a slope of
2.051, and the residuals are the values the model failed to reproduce.

**Table 7. Worked example data and OLS fit**

| i | x | y | y_hat (OLS) | Residual |
|---|---|---|---|---|
| 1 | 1 | 1.8 | 2.033 | −0.233 |
| 2 | 2 | 4.9 | 4.085 | 0.815 |
| 3 | 3 | 5.2 | 6.136 | −0.936 |
| 4 | 4 | 8.9 | 8.187 | 0.713 |
| 5 | 5 | 9.4 | 10.238 | −0.838 |
| 6 | 6 | 13.6 | 12.289 | 1.311 |
| 7 | 7 | 12.8 | 14.340 | −1.540 |
| 8 | 8 | 17.1 | 16.392 | 0.708 |

The observations have a mean of 9.213 and a sample variance of 26.301. On this fit the
Pearson R² is 0.9598, and the standard R² is also 0.9598. The two coincide here because
an OLS fit with an intercept forces them to agree, which is exactly why they are so often
treated as the same quantity.

### B.2. Bayesian R² over Posterior Draws

Five posterior draws of the intercept a, the slope b, and the noise scale sigma are
listed below. Each row is a complete model, so each row produces its own R². The
empirical variant of 3.3 is used, meaning `Var_res` is the variance of the residuals of
that draw.

**Table 8. Per-draw computation**

| s | a | b | sigma | Var_fit | Var_res | Bayesian R² | Pearson R² |
|---|---|---|---|---|---|---|---|
| 1 | 0.10 | 2.05 | 1.05 | 25.215 | 1.057 | 0.9598 | 0.9598 |
| 2 | 0.60 | 1.92 | 1.20 | 22.118 | 1.160 | 0.9502 | 0.9598 |
| 3 | −0.40 | 2.14 | 0.95 | 27.478 | 1.104 | 0.9614 | 0.9598 |
| 4 | 1.10 | 1.83 | 1.40 | 20.093 | 1.351 | 0.9370 | 0.9598 |
| 5 | 4.60 | 1.05 | 2.60 | 6.615 | 7.071 | 0.4833 | 0.9598 |

Draw 1 is worked through explicitly. Its predictions are `0.10 + 2.05 x`, whose variance
across the eight points is 25.215; its residuals have variance 1.057; the ratio
`25.215 / (25.215 + 1.057)` gives 0.9598.

Sorting the five values gives 0.4833, 0.9370, 0.9502, 0.9598, 0.9614, so the median is
0.9502 and the values span 0.4833 to 0.9614. Five draws are far too few to quote a
quantile-based interval, as noted in 7.1 and in Table 6; the spread is shown here only to
make the distribution visible.

Switching to the model-based variant, which divides by `sigma^2` instead of the residual
variance, moves the same five draws to 0.9581, 0.9389, 0.9682, 0.9111, and 0.4946. The
values shift in both directions and the median moves from 0.9502 to 0.9389, so a figure
produced under one variant cannot be set against a figure produced under the other.

### B.3. Why Pearson R² Cannot See the Difference

The Pearson R² column is constant at 0.9598 across every draw, including the badly
calibrated draw 5. This is not a coincidence. The Pearson correlation is invariant under
an affine transformation of the predictions, and every draw here is an affine function of
the same x, so all of them are affine transformations of one another and share a single
correlation with y.

The consequence is that Pearson R² measures only whether the predictions move in step
with the observations. It is blind to bias and to scale. Draw 5 predicts 5.65 where the
observation is 1.8 and predicts 13.0 where the observation is 17.1, yet Pearson R² still
reports 0.9598 while Bayesian R² reports 0.4833.

### B.4. A Draw That Breaks the Standard Formula

The claim in 3.1 becomes concrete with a heavily shrunk draw. Suppose a prior pulls the
fit toward a center of roughly 12.9 and nearly flattens the slope, giving a = 12.00 and
b = 0.20.

**Table 9. Three definitions on a heavily shrunk draw**

| Metric | Value | Reading |
|---|---|---|
| Pearson R² | 0.9598 | Unchanged, since the predictions remain affine in x |
| Standard R² | −0.4128 | Below 0, outside the range the metric is supposed to occupy |
| Bayesian R² | 0.0110 | Near 0 and still inside [0, 1] |

The predictions of this draw are worse than simply reporting the mean of y, which is what
drives the standard form negative. The Gelman form registers the same failure as a value
near 0 without leaving its range, which is the property that makes the values poolable
across draws. Pearson R² records no failure at all.

### B.5. Summary of the Comparison

**Table 10. What each metric reports on this dataset**

| Metric | Value | What it answers |
|---|---|---|
| Pearson R² on any draw | 0.9598 | Do predictions and observations move together |
| Standard R² on the OLS fit | 0.9598 | How much variation does the single best fit explain |
| Bayesian R², median | 0.9502 | How much variation does a typical posterior draw explain |
| Bayesian R², spread | 0.4833 to 0.9614 | How much does that explanatory power depend on which draw is taken |

Pearson R² and standard R² agree on the OLS fit and diverge everywhere else. Bayesian R²
is the only entry that reports a spread, and on this dataset that spread is the finding:
a single number near 0.96 would have hidden the presence of draw 5 entirely.
