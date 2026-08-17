# Bayesian R² — Obtaining R² as a Distribution Instead of a Point
Rev. 2 | Created: 2026-05-31 | Updated: 2026-08-16 18:45 CDT

> A note on computing one R² per posterior draw to obtain R² as a distribution,
> organized as definition, computation, interpretation, and applicability.

## 1. Motivation

Standard R² is the coefficient of determination `1 − SS_res / SS_tot`, where `SS_res`
is the sum of squared residuals and `SS_tot` is the sum of squared deviations of the
observations from their own mean. It is computed from one single set of predictions and
therefore returns a single number. When a fit reports `R² = 0.87`, that number alone
does not say whether the value would collapse to 0.60 under a slightly different sample
or hold firmly near 0.85. The magnitude of explanatory power and the confidence in that
magnitude are two different pieces of information, and standard R² carries only the first.

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


def bayesian_r2_empirical(y_pred_draws: np.ndarray = None, y_true: np.ndarray = None) -> np.ndarray:
    """Bayesian R2 using the variance of the residuals actually left over.

    Args:
        y_pred_draws: posterior draws of the predictive mean, shape (S, n).
        y_true: observations, shape (n,).

    Returns:
        R2 draws of shape (S,), every element inside [0, 1].
    """
    var_fit = y_pred_draws.var(axis=1, ddof=1)           # variance over data points
    var_res = (y_true[None, :] - y_pred_draws).var(axis=1, ddof=1)
    return var_fit / (var_fit + var_res)


def bayesian_r2_model_based(y_pred_draws: np.ndarray = None, sigma_draws: np.ndarray = None) -> np.ndarray:
    """Bayesian R2 using the noise variance the model claims for itself."""
    var_fit = y_pred_draws.var(axis=1, ddof=1)
    return var_fit / (var_fit + sigma_draws ** 2)


# posterior_mean_draws: shape (S, n), y_observed: shape (n,)
r2_draws = bayesian_r2_empirical(y_pred_draws=posterior_mean_draws, y_true=y_observed)
point = np.median(r2_draws)
lower, upper = np.quantile(r2_draws, [0.05, 0.95])       # 90% credible interval
```

The arithmetic is two variances and one division; the cost of the method sits entirely
in obtaining the posterior draws. The two variants of 3.3 are two separate functions
rather than one function with a switch, so a caller cannot silently supply the inputs of
one variant and receive the other.

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
- **conjugate posterior** — A posterior that stays in the same distribution family as the prior, so it has a closed form and can be sampled directly without MCMC.
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
- **reference prior** — A prior chosen to carry as little information as possible, so the posterior is driven by the data rather than by the prior.
- **shrinkage** — The pull a prior or a hierarchical structure exerts on estimates toward a common center.

## Appendix B. Worked Example

Every number in this appendix is produced by `bayesian_r2_example.py`, invoked as
`python3 bayesian_r2_example.py --draws 4000`. The seed is fixed inside the script, so the
values reproduce exactly.

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

### B.2. Posterior Draws

The posterior is the conjugate posterior of the linear model under the reference prior
`p(beta, sigma^2)` proportional to `1 / sigma^2`, which has a closed form and can be sampled
exactly without MCMC. Four thousand draws are taken. Each draw is a complete model, so each
produces its own R²; the first five are shown in full.

**Table 8. First five posterior draws and their empirical-variant R²**

| s | a | b | sigma | Var_fit | Var_res | Bayesian R² | Pearson R² |
|---|---|---|---|---|---|---|---|
| 1 | 0.369 | 1.903 | 1.826 | 21.740 | 1.188 | 0.9482 | 0.9598 |
| 2 | 0.556 | 1.952 | 0.922 | 22.857 | 1.116 | 0.9534 | 0.9598 |
| 3 | 0.754 | 1.757 | 1.142 | 18.524 | 1.576 | 0.9216 | 0.9598 |
| 4 | 1.151 | 1.866 | 0.743 | 20.900 | 1.262 | 0.9431 | 0.9598 |
| 5 | 0.266 | 2.007 | 0.844 | 24.164 | 1.069 | 0.9576 | 0.9598 |

Draw 1 is worked through explicitly. Its predictions are `0.369 + 1.903 x`, whose variance
across the eight points is 21.740; its residuals have variance 1.188; the ratio
`21.740 / (21.740 + 1.188)` gives 0.9482.

### B.3. Credible Interval

The point of the method is the interval, so the summary over all 4,000 draws is the result
that matters. None of the draws left [0, 1].

**Table 9. Bayesian R² over 4,000 posterior draws**

| Variant | Median | 90% credible interval | Full range of draws |
|---|---|---|---|
| Empirical | 0.9571 | [0.9112, 0.9613] | [0.0983, 0.9614] |
| Model-based | 0.9482 | [0.8294, 0.9780] | [0.0771, 0.9893] |

The reported result for this dataset is therefore `0.9571 [0.9112, 0.9613]` under the
empirical variant. The interval is what standard R² cannot supply: the single OLS number
0.9598 gives no indication that a posterior draw consistent with these eight points can
explain as little as 91% of the variation, or that the extreme tail reaches 0.10.

![Fig 1](bayesian-r2_fig/bayesian_r2_posterior.png)

**Fig 1. Posterior fits and the resulting distribution of Bayesian R²**

Panel (a) overlays 200 posterior fits on the data, and the fan of slopes is the source of
the spread. Panel (b) is the distribution those fits produce. Its shape confirms the claim
of 5.1: the mass piles against a ceiling near 0.961 and trails away to the left, so the mean
sits below the median and the median is the honest point estimate. The interval is strongly
asymmetric for the same reason, with the lower bound 0.046 away from the median and the
upper bound only 0.004 away.

The two variants disagree by more than rounding. The model-based median is 0.9482 against
0.9571, and its interval is roughly three times wider, because `sigma` is itself uncertain
and that uncertainty enters the denominator directly. The direction of the gap is the
reverse of the case sketched in 3.3: with only eight points the posterior of `sigma^2` is
right-skewed and its median, 1.387, sits above the median empirical residual variance of
1.152, so here the model claims more noise than the residuals show rather than less. A
figure produced under one variant cannot be set against a figure produced under the other.

### B.4. What Pearson R² and Standard R² Do Instead

Four predictors are constructed by hand, ranging from well calibrated to collapsed. These
are not posterior draws; they are chosen to move bias and scale on purpose.

**Table 10. Three metrics on hand-constructed predictors**

| Predictor | a | b | Pearson R² | Standard R² | Bayesian R² |
|---|---|---|---|---|---|
| Well calibrated | 0.10 | 2.05 | 0.9598 | 0.9593 | 0.9598 |
| Mildly shrunk | 1.10 | 1.83 | 0.9598 | 0.9480 | 0.9370 |
| Strongly shrunk | 4.60 | 1.05 | 0.9598 | 0.7306 | 0.4833 |
| Collapsed to a wrong center | 12.00 | 0.20 | 0.9598 | −0.4128 | 0.0110 |

The Pearson column never moves. This is not a coincidence: the Pearson correlation is
invariant under any affine transform of the predictions with a non-zero slope, and all four
predictors are affine in the same x, so they are affine transforms of one another and share
one correlation with y. Pearson R² measures only whether the predictions move in step with
the observations, and it is blind to bias and to scale. The last predictor puts 12.2 where
the observation is 1.8, and Pearson R² still reports 0.9598.

The last row is also where the standard form leaves its own range. Those predictions are
worse than simply reporting the mean of y, which drives `SS_res` above `SS_tot` and the
value to −0.4128. This is the failure described in 3.1, and it is why the values of a
standard R² computed per draw cannot be pooled into a distribution. The Gelman form records
the same failure as 0.0110, near the bottom of a range it never leaves.

### B.5. Summary of the Comparison

**Table 11. What each metric reports on this dataset**

| Metric | Value | What it answers |
|---|---|---|
| Pearson R² on any predictor | 0.9598 | Do predictions and observations move together |
| Standard R² on the OLS fit | 0.9598 | How much variation does the single best fit explain |
| Bayesian R², median | 0.9571 | How much variation does a typical posterior draw explain |
| Bayesian R², 90% credible interval | [0.9112, 0.9613] | How far can that explanatory power be from its median |

The three agree on a well-calibrated fit and separate as soon as the predictions are biased
or shrunk. Bayesian R² is the only entry carrying an interval, and on this dataset that
interval is the finding: eight points support a median of 0.957 but not the precision that
the bare number 0.9598 appears to promise.
