# Predictive Uncertainty
Rev. 0 | Created: 2026-09-10 | Updated: 2026-09-10 23:40 UTC

## 1. Purpose

- **Problem Statement**: A fitted regression model returns one number per row, and that number carries no statement of how far the outcome may fall from it.
- **Goal**: Present the ways of attaching a probability to a predicted value as a taxonomy and a hierarchy, so that a reader can name the level a model reaches and what it costs to reach the next one.
- **Non-Goal**: The probability a classifier emits, and the calibration of that probability, are not covered.
- **Non-Goal**: Improving the accuracy of the point prediction is not covered.

## 2. Summary

The loss and the wrapper decide what probability can be reported, not the model family. Ridge and LightGBM both emit a bare number as they come; both reach a quantile once the loss is swapped for the pinball loss, and both reach an interval with a stated coverage once a calibration split is spent. Table 1 is the three axes this places a method on, and the first of them orders the rest.

That first axis is a hierarchy of four levels, drawn in Fig 1, and each level answers strictly more questions than the one below. A point answers what to expect. An interval answers whether the outcome falls inside one range. A quantile function answers that for a range whose width moves with the input. A distribution answers what the probability is that the outcome passes any stated limit, which is the question a specification limit asks.

Two failures separate the form of an answer from its correctness, and both are measured in Appendix B on the same 600 held-out rows. A constant-width interval covers 0.908 of all rows while covering 1.000 of the quietest third and 0.785 of the noisiest third, so coverage measured over all rows is met by over-covering easy rows and under-covering hard ones. Quantile boosting, whose width does move with the input, covers 0.822 to 0.832 against a nominal 0.90, so a width that adapts carries no guarantee that it is wide enough.

Sections 5 and 6 record, for the linear and the tree families, what each library emits and which parameter changes it. Section 7 is the one route that attaches a coverage guarantee and attaches it to any family. Section 8 is how a reported interval is judged, and section 9 is which method to use.

## 3. Taxonomy

### 3.1 Three Axes

Three questions place any method of attaching a probability to a predicted value, and they are answered independently of one another. Table 1 is the three.

Table 1. The three axes of a predictive uncertainty method

| Axis | Question | Values |
|------|----------|--------|
| Form | What the model emits per row | Point, interval, quantile function, distribution |
| Source | What the emitted spread is made of | Noise in the outcome, ignorance about the fit |
| Route | How the emission is obtained | Native, loss swap, resampling, calibration wrapper |

The Form axis orders the other two, because it decides which questions can be asked at all. Section 3.2 gives that axis as a hierarchy, and the remaining two axes then say whether the emitted number means what it claims.

### 3.2 Hierarchy Of The Answer

Fig 1 is the Form axis of Table 1, drawn out.

```text
Probability attached to a predicted value
|
+-- Level 1: point ............................ one number per row
|   +-- Squared-error loss .................... the conditional mean
|   +-- Absolute-error loss ................... the conditional median
|   +-- Huber or epsilon-insensitive loss ..... a robust centre
|
+-- Level 2: interval ......................... two numbers at one stated level
|   +-- Residual standard deviation ........... one width for every row, no guarantee
|   +-- Spread across resampled members ....... width moves with the input, no guarantee
|   +-- Split conformal ....................... one width for every row, coverage guaranteed
|
+-- Level 3: quantile function ................ a bound per level, each a function of the input
|   +-- Pinball loss, one fit per level ....... linear, GBR, HistGBR, LightGBM
|   +-- Pinball loss, one fit for all levels .. XGBoost
|   +-- Leaf observations of a forest ......... random forest, no refit
|   +-- Conformalized quantile regression ..... any of the above, coverage guaranteed
|
+-- Level 4: distribution ..................... a density and a full CDF per row
    +-- Gaussian mean and variance ............ BayesianRidge, ARDRegression, Gaussian process
    +-- Fitted distribution parameters ........ NGBoost
    +-- Dense grid of quantiles ............... any level 3 method fitted at many levels
```

Fig 1. Hierarchy of the answer a model gives about its predicted value

The four levels are ordered by the questions they answer, and Table 2 is that ordering. Each level contains the one below it: an interval is two quantiles at a fixed pair of levels, a quantile function is the inverse of a CDF read at chosen points, and a distribution is that inverse read anywhere.

Table 2. What each level of Fig 1 answers

| Level | Emits | Answers | Needs |
|-------|-------|---------|-------|
| 1. Point | One number | What to expect | Nothing beyond the fit |
| 2. Interval | Two numbers at one level | Whether the outcome falls inside one range | A residual spread, or a calibration split |
| 3. Quantile function | One bound per requested level | The same, with a range that moves with the input | One fit per level, or one fit that takes a vector of levels |
| 4. Distribution | A density or CDF | The probability of passing any stated limit, and any moment or quantile of the outcome | A distribution family, or a dense grid of levels |

Level 4 is the level a specification limit needs. A limit arrives after the model is fitted and is moved without warning, so the answer has to be a function that any limit can be substituted into rather than a bound fitted at one level.

### 3.3 Source Of The Spread

An emitted spread is made of two parts that behave differently under more data, and the predictive variance splits into exactly those two. For parameters $\theta$ and an input $x$, the law of total variance gives the split [[7](#ref-7)].

$$\operatorname{Var}[y \mid x] = \mathbb{E}_{\theta}\left[\operatorname{Var}(y \mid x, \theta)\right] + \operatorname{Var}_{\theta}\left(\mathbb{E}[y \mid x, \theta]\right) \hspace{19em} (1)$$

The first term is the noise in the outcome and the second is the ignorance about the fit. Three consequences decide which methods can be used for what.

- First term, unchanged by more rows. Irreducible at a fixed set of inputs.
- Second term, shrinking as rows accumulate. Zero in the limit of a known fit.
- An interval about a future outcome, needing both. An interval about the fitted mean, needing only the second.

The distinction is what separates two methods that both look like a spread. Table 3 is which term each route measures, and the last column is what happens when the other term is the one that matters.

Table 3. Which term of equation (1) each route measures

| Route | Term measured | Width moves with the input | What it misses |
|-------|---------------|----------------------------|----------------|
| Residual standard deviation | First, as one constant | No | The change of noise across the input space |
| BayesianRidge `return_std` | Both, the first as one constant | Only through the second term | The change of noise across the input space |
| Spread across the trees of a forest | Second | Yes | The noise, except where a leaf holds one row |
| Pinball loss at two levels | Both, as fitted | Yes | A guarantee that the fitted bound is wide enough |
| Leaf observations of a forest | Both | Yes | The same guarantee |
| Split conformal, conformalized quantile regression | Both, through the calibration residuals | Only for the quantile form | Conditional coverage, which is not guaranteed |

The second row of Table 3 is the trap this section exists to name. A Gaussian mean and standard deviation is a level 4 answer in form, and BayesianRidge emits one, but its noise term is a single fitted constant. Appendix B measures a standard deviation between 1.889 and 1.901 across the test rows, where the noise standard deviation of the construction runs from 0.51 to 5.25 on those same rows, so the emitted density is the right shape with the wrong width nearly everywhere.

The third row is the same trap read the other way. A forest's trees disagree about the mean, and that disagreement is the second term of equation (1) alone. Appendix B measures 0.858 coverage from the tree spread of a default forest and 0.650 once each leaf is made to hold at least 20 rows, so the first number comes from leaves so small that a tree's prediction is close to one noisy observation rather than from a predictive distribution.

### 3.4 Route

Four routes reach the levels of Fig 1, and they differ in what they spend. Table 4 sets them side by side.

Table 4. The four routes of Table 1

| Route | Mechanism | Spends | Reaches |
|-------|-----------|--------|---------|
| Native | The model is fitted as a distribution from the start | A distributional assumption | Level 4 |
| Loss swap | The squared-error loss is replaced by the pinball loss of equation (2) | One fit per level, unless the library takes a vector | Level 3 |
| Resampling | The members of an ensemble, or the rows in their leaves, are read as a sample | Nothing beyond a fitted ensemble | Level 2 or level 3 |
| Calibration wrapper | The bounds of an already fitted model are shifted by a quantile of held-out scores | A calibration split | Level 2 or level 3, with a guarantee |

The loss swap is the route that reaches the most families, because it changes which functional of the conditional distribution the fit estimates rather than changing the model. The pinball loss at level $\alpha$ is minimized by the $\alpha$ quantile [[1](#ref-1)].

$$L_{\alpha}(y, q) = \max\left\{\alpha\,(y - q),\ (\alpha - 1)(y - q)\right\} \hspace{19em} (2)$$

Equation (2) also explains the level 1 rows of Fig 1. At $\alpha = 0.5$ it is the absolute-error loss up to a factor of two, so a fit under absolute error already emits a quantile, the median. The squared-error loss emits the mean instead, and the Huber and epsilon-insensitive losses emit a centre that is neither, which is why a robust point fit says nothing about the spread it is robust to.

## 4. Form Against Correctness

The level a method reaches in Fig 1 and the correctness of what it emits are independent, and a method can be high on one and wrong on the other. Table 5 is the four combinations, with the row from Appendix B that occupies each.

Table 5. Level reached against coverage delivered

| Level reached | Coverage close to nominal | Coverage far from nominal |
|---------------|---------------------------|---------------------------|
| Level 2 | `Ridge` with a residual standard deviation, 0.912 | Forest tree spread with 20 rows per leaf, 0.650 |
| Level 3 or 4 | Conformalized quantile regression, 0.917 | LightGBM quantile, 0.815 |

Two failures fill the right column and they have separate causes. A fitted quantile is the minimizer of equation (2) on the training rows, and nothing in that minimization forces the held-out frequency to match the level, so a regularized or under-trained fit lands inside the true bounds. A resampling spread measures the wrong term of equation (1), as section 3.3 records.

Marginal coverage is one of two ways to be wrong, and the left column of Table 5 answers only the first. `BayesianRidge` sits in that column at level 4 with a coverage of 0.908, and its width stands still where the noise moves, which shows up as the conditional coverage of 1.000 against 0.785 that section 8 measures.

The left column is reached in two ways that are not interchangeable. A constant width can be made to cover 0.90 of all rows by choosing one number, and it then over-covers where the noise is small and under-covers where it is large. A width that moves with the input can cover 0.90 of all rows and also of each band of rows. Section 8 is the measurement that separates the two, and it is the reason coverage is never reported alone.

## 5. Linear Family

Two of the linear regressors emit a spread of their own and the rest emit a bare number, so for the rest the level in Fig 1 is set entirely by the route of Table 4. Table 6 records what each one emits and how it reaches level 3.

Table 6. What the linear regressors emit

| Estimator | `predict` signature | Level as fitted | Functional estimated | Route to level 3 |
|-----------|---------------------|-----------------|----------------------|------------------|
| `Ridge` | `predict(X)` | 1 | Conditional mean | `QuantileRegressor`, or a wrapper of section 7 |
| `Lasso` | `predict(X)` | 1 | Conditional mean | The same |
| `ElasticNet` | `predict(X)` | 1 | Conditional mean | The same |
| `HuberRegressor` | `predict(X)` | 1 | Robust centre | The same |
| `LinearSVR` | `predict(X)` | 1 | Centre of an epsilon-insensitive tube | The same |
| `QuantileRegressor` | `predict(X)` | 3, one level per fit | Conditional quantile | Already there |
| `BayesianRidge` | `predict(X, return_std=False)` | 4, Gaussian | Conditional mean and variance | Already above it, with the width of Table 3 |
| `ARDRegression` | `predict(X, return_std=False)` | 4, Gaussian | The same | The same |

`QuantileRegressor` carries an L1 penalty like `Lasso` and its `alpha` defaults to 1.0, so an unpenalized quantile fit needs `alpha=0` set explicitly. The `quantile` argument is the level and `alpha` is the penalty, which is the reverse of the naming in the gradient boosting libraries of section 6.

The two Bayesian rows emit a variance because they fit the noise and the coefficients together under the evidence framework [[6](#ref-6)]. The emitted variance is the sum of a fitted noise term and a term that grows with the distance of the input from the fitted data.

$$\operatorname{Var}[y \mid x] = \sigma^{2} + x^{\top} \Sigma\, x \hspace{19em} (3)$$

Equation (3) is equation (1) with the second term made explicit for a linear fit, and it says where the emitted width can and cannot move. The first term is one scalar, so all of the movement comes from the second, which is small wherever the fit is well determined. Appendix B measures that width moving by 0.012 across 600 rows, against a true noise scale that changes by a factor of ten on the same rows.

## 6. Tree And Ensemble Family

All four of the common tree models reach level 3 through the pinball loss, and the random forest is the only one that reaches it without a second fit. Table 7 is the parameter that changes in each library.

Table 7. How the tree models emit a quantile

| Estimator | Parameter | Levels per fit | Note |
|-----------|-----------|----------------|------|
| `GradientBoostingRegressor` | `loss="quantile"`, `alpha` | One | `loss` also takes `squared_error`, `absolute_error`, `huber` |
| `HistGradientBoostingRegressor` | `loss="quantile"`, `quantile` | One | `loss` also takes `squared_error`, `absolute_error`, `poisson`, `gamma` |
| `LGBMRegressor` | `objective="quantile"`, `alpha` | One | `alpha` is the level here, not a penalty |
| `XGBRegressor` | `objective="reg:quantileerror"`, `quantile_alpha` | Several | `quantile_alpha` takes an array, and `predict` returns one column per level |
| `RandomForestRegressor` | None | All | Through the resampling route of Table 4, section 6.1 |

One fit per level is the cost that decides how this scales. A ten-level answer from `LGBMRegressor` is ten fitted models to train, store and serve, while `XGBRegressor` takes the ten levels as an array and returns a matrix of ten columns from one booster. Appendix B confirms the shape as `(600, 2)` for a two-level fit.

Separately fitted levels are also free to cross. Each fit minimizes equation (2) at its own level with no term that ties it to the others, so a 0.05 bound can land above a 0.95 bound on some rows, and the crossing has to be repaired by sorting the emitted columns before they are used.

### 6.1 The Forest Without A Second Fit

A random forest already holds the rows it was fitted on, sorted into leaves, so a quantile is read off without changing the loss. For an input, the rows in the leaves it lands in across all trees form a weighted sample of the outcome, and any quantile of that sample is a level 3 answer [[2](#ref-2)].

The route through the tree predictions instead of the leaf rows is the one that fails, and section 3.3 gives the reason. Appendix B measures both on the same forest: the leaf rows give 0.893 coverage with the width moving by 2.89, and the tree spread gives 0.650 once the leaves are large enough to average.

The cost of the leaf route is the store rather than the fit. The fitted rows have to be kept and their leaf membership indexed, and a prediction touches every tree's leaf pool instead of one number per tree, which is why the library route is a separate implementation rather than a call on the fitted forest.

## 7. Model-Agnostic Route

Conformal prediction is the one route that attaches a coverage guarantee, and it attaches to a fitted model of any family without refitting it. A calibration split held out from the fit is scored, and one quantile of those scores widens the emitted bounds by an amount that makes the coverage hold in finite samples under exchangeability [[3](#ref-3)] [[8](#ref-8)].

For a point model, the score is the absolute residual on the calibration rows and the correction is one number, the $\lceil (n+1)(1-\alpha) \rceil$-th smallest of the $n$ scores, which is the 541st of the 600 calibration scores at $\alpha = 0.10$.

$$\hat q = s_{\left(\left\lceil (n+1)(1-\alpha) \right\rceil\right)} \hspace{19em} (4)$$

The interval is then the prediction plus and minus $\hat q$, which is one width for every row. Appendix B measures $\hat q = 3.087$ and a coverage of 0.908 against a nominal 0.90, and the conditional coverage of 1.000 and 0.785 that a constant width forces.

Conformalized quantile regression (CQR) keeps the width moving by conformalizing the bounds instead of the point [[4](#ref-4)]. The score is how far outside the fitted bounds the calibration outcome fell, negative where it fell inside.

$$E_{i} = \max\left\{\hat q_{\alpha/2}(x_{i}) - y_{i},\ y_{i} - \hat q_{1-\alpha/2}(x_{i})\right\} \hspace{19em} (5)$$

Equation (4) applied to those scores gives one correction that is added to the upper bound and subtracted from the lower. Appendix B measures a correction of 0.483, which lifts the coverage of the gradient boosting quantiles from 0.832 to 0.917 while the width keeps moving by 2.68, so the guarantee and the adaptation are held at the same time.

Exchangeability between the calibration rows and the rows to be predicted is the condition all of this rests on, and it is the condition a drifting process breaks. The guarantee is also marginal: it holds over the draw of rows, not within a band of them, which is why Appendix B reports coverage per band as well.

Two families reach level 4 natively and belong here as the alternative to a wrapper. A Gaussian process emits a mean and a variance from the kernel it was fitted with, at a cost that grows as the cube of the rows. NGBoost fits the parameters of a chosen distribution by boosting them together under a natural gradient, which gives a density per row from a tree model [[5](#ref-5)]. Both replace the guarantee of this section with a distributional assumption.

## 8. Calibration

Coverage measured over all rows is met by over-covering the quiet rows and under-covering the noisy ones, so an interval is judged by three numbers together. Table 8 is the three, with the failure each one catches.

Table 8. What an emitted interval is judged by

| Measure | Definition | Catches |
|---------|------------|---------|
| Marginal coverage | Fraction of all rows inside the interval | Bounds too narrow or too wide overall |
| Conditional coverage | The same, within a band of rows grouped by an input or by the emitted width | A constant width standing in for a varying one |
| Sharpness | Mean width of the interval | Bounds made to cover by being useless |

Coverage and sharpness are read against each other, because either one alone is trivially satisfied. The rule is to maximize sharpness subject to calibration: among the methods whose coverage holds, the one with the narrowest interval wins [[10](#ref-10)]. Appendix B is arranged as that comparison, and the two constant-width rows lose it despite their coverage.

A single number that scores both at once is a proper scoring rule, which is a score whose expected value is optimized by the true distribution and by nothing else [[9](#ref-9)]. Two of them cover the levels of Fig 1.

- Pinball loss of equation (2), proper for one quantile. Available as `mean_pinball_loss` and `d2_pinball_score`.
- Continuous ranked probability score, proper for a full distribution. The integral of the squared difference between the emitted CDF and the step function at the outcome.

Neither replaces the coverage table. A proper score ranks methods against one another without saying whether the best of them is right, and a coverage of 0.65 is a fact about the interval that no ranking reports.

## 9. Selection

The choice follows from two questions: which level of Fig 1 the answer has to reach, and whether a calibration split can be spared. Table 9 reads from the left column.

Table 9. Which method to use

| Use | When | Why |
|-----|------|-----|
| Residual standard deviation | A rough interval is wanted and the noise is known to be even across the inputs | One number over the fit already made |
| `BayesianRidge` or `ARDRegression` | A linear fit, and a density is wanted at level 4 | Native, and no split spent |
| Gaussian process | Few rows, a smooth response, and a density wanted | Native, and the width grows where the data thins |
| Pinball loss in the model's own library | The noise changes across the inputs and the level is fixed in advance | Level 3 at the cost of one fit per level |
| Leaf observations of a forest | A forest is already fitted and more than one level is wanted | Every level from one fit |
| NGBoost | A tree model, and a density is wanted at level 4 | Level 4 from trees, at the price of a distribution family |
| Split conformal | A coverage guarantee is required and the emitted width may be constant | Marginal coverage in finite samples, over any fitted model |
| Conformalized quantile regression | A coverage guarantee is required and the width must move with the input | The only row that holds both |

Appendix B measures every row of Table 9 against the others on one split, except the Gaussian process and NGBoost rows, which need a library outside the four of section 6.

## 10. Further Work

- **Coverage under drift** — The guarantee of section 7 rests on exchangeability between the calibration rows and the rows to be predicted, which a process whose distribution moves breaks silently, since the emitted width keeps its nominal label. Adaptive and weighted conformal methods replace the fixed quantile of equation (4) with one that is re-estimated from the recent coverage error, which restores the guarantee in a long-run average sense rather than per row. It needs the outcomes to arrive with a bounded delay and a decision on how fast the correction may move, since a correction that chases noise is worse than a fixed one.
- **Distributional boosting as the default level 4** — The tree family reaches level 4 in this document either through a dense grid of separately fitted quantiles, at one fit per level, or through NGBoost's distributional assumption. Boosting the parameters of a distribution jointly is what the natural gradient made stable, and the same treatment is now available as a distributional loss inside the mainstream boosting libraries. It needs the residuals of the fitted model examined first, since the choice of family is what the emitted density inherits, and a wrong family is a level 4 answer with a level 2 amount of information in it.

## References

<a id="ref-1"></a>[1] Koenker, R. and Bassett, G. (1978). [Regression Quantiles](https://doi.org/10.2307/1913643). *Econometrica*, 46(1), 33-50.<br>
<a id="ref-2"></a>[2] Meinshausen, N. (2006). [Quantile Regression Forests](https://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf). *Journal of Machine Learning Research*, 7, 983-999.<br>
<a id="ref-3"></a>[3] Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. and Wasserman, L. (2018). [Distribution-Free Predictive Inference for Regression](https://doi.org/10.1080/01621459.2017.1307116). *Journal of the American Statistical Association*, 113(523), 1094-1111.<br>
<a id="ref-4"></a>[4] Romano, Y., Patterson, E. and Candès, E. J. (2019). [Conformalized Quantile Regression](https://papers.neurips.cc/paper/8613-conformalized-quantile-regression.pdf). *Advances in Neural Information Processing Systems*, 32.<br>
<a id="ref-5"></a>[5] Duan, T., Avati, A., Ding, D. Y., Thai, K. K., Basu, S., Ng, A. Y. and Schuler, A. (2020). [NGBoost: Natural Gradient Boosting for Probabilistic Prediction](https://proceedings.mlr.press/v119/duan20a/duan20a.pdf). *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, 2690-2700.<br>
<a id="ref-6"></a>[6] MacKay, D. J. C. (1992). [Bayesian Interpolation](https://doi.org/10.1162/neco.1992.4.3.415). *Neural Computation*, 4(3), 415-447.<br>
<a id="ref-7"></a>[7] Kendall, A. and Gal, Y. (2017). [What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?](https://papers.neurips.cc/paper/7141-what-uncertainties-do-we-need-in-bayesian-deep-learning-for-computer-vision.pdf) *Advances in Neural Information Processing Systems*, 30, 5574-5584.<br>
<a id="ref-8"></a>[8] Angelopoulos, A. N. and Bates, S. (2023). [Conformal Prediction: A Gentle Introduction](https://www.nowpublishers.com/article/Details/MAL-101). *Foundations and Trends in Machine Learning*, 16(4), 494-591.<br>
<a id="ref-9"></a>[9] Gneiting, T. and Raftery, A. E. (2007). [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437). *Journal of the American Statistical Association*, 102(477), 359-378.<br>
<a id="ref-10"></a>[10] Gneiting, T., Balabdaoui, F. and Raftery, A. E. (2007). [Probabilistic Forecasts, Calibration and Sharpness](https://doi.org/10.1111/j.1467-9868.2007.00587.x). *Journal of the Royal Statistical Society: Series B*, 69(2), 243-268.

---

## Appendix A. Terminology

- **aleatoric uncertainty**: The noise in the outcome, the first term of equation (1).
- **calibration split**: Rows held out from the fit and used only to size a conformal correction.
- **conditional coverage**: The coverage of an interval within a band of rows rather than over all of them.
- **conformal prediction**: A procedure that widens the bounds of any fitted model by a quantile of held-out scores so that the coverage holds in finite samples.
- **continuous ranked probability score**: The integral of the squared difference between an emitted CDF and the step function at the outcome.
- **coverage**: The fraction of rows whose outcome falls inside the emitted interval.
- **epistemic uncertainty**: The ignorance about the fit, the second term of equation (1).
- **exchangeability**: The property that the joint distribution of the rows is unchanged by reordering them.
- **leaf observations**: The fitted rows sorted into the leaf an input lands in, read as a sample of the outcome.
- **marginal coverage**: The coverage of an interval over all rows.
- **pinball loss**: The asymmetric absolute loss of equation (2), minimized by one quantile.
- **prediction interval**: Two numbers between which a future outcome is claimed to fall at a stated level.
- **predictive distribution**: A density or CDF for the outcome at one input.
- **proper scoring rule**: A score whose expected value is optimized by the true distribution and by no other.
- **quantile function**: The level of a distribution read as a function of the input.
- **sharpness**: The narrowness of an emitted interval or density, judged only among methods whose coverage holds.
- **split conformal**: Conformal prediction whose correction is one quantile of the absolute residuals on a calibration split.

## Appendix B. Worked Example

Every number in this appendix comes from one dataset and one pair of splits. The noise is built to widen with the first feature, so that a method whose width cannot move is visibly wrong without being wrong on average.

```python
import numpy as np
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
n, p = 3000, 5
X = rng.normal(size=(n, p))
# the noise scale is a function of the first feature, so no one width serves every row
y = 3.0 * X[:, 0] + 2.0 * X[:, 1] - 1.5 * X[:, 2] + rng.normal(scale=0.5 + 1.5 * np.abs(X[:, 0]))

X_fit, X_rest, y_fit, y_rest = train_test_split(X, y, test_size=0.4, random_state=0)
X_cal, X_test, y_cal, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)
```

The split is 1800 rows to fit on, 600 to calibrate the wrappers of section 7 on, and 600 to measure on. Every interval is asked for the 0.05 and 0.95 levels, so the nominal coverage is 0.90 throughout. The three boosting models are given the same 300 trees, the same learning rate of 0.05 and the same depth of 3, so that the comparison between them is a comparison of objectives rather than of defaults.

The true width at those levels follows from the construction as $2 \times 1.645 \times (0.5 + 1.5\,|x_{1}|)$. Averaged over the three bands of the test rows, sorted by $|x_{1}|$, that width is 2.85, 5.21 and 9.28, so no one number serves all three. The two band columns of Table 10 are the outer two of those bands.

Table 10. Eleven methods on the same 600 held-out rows, nominal coverage 0.90

| Method | Level | Coverage | Mean width | Width sd | Coverage, quiet third | Coverage, noisy third | Pinball |
|--------|-------|----------|------------|----------|-----------------------|-----------------------|---------|
| `Ridge`, residual standard deviation | 2 | 0.912 | 6.38 | 0.00 | 1.000 | 0.790 | 0.2078 |
| `BayesianRidge`, `return_std` | 4 | 0.908 | 6.22 | 0.01 | 1.000 | 0.785 | 0.2073 |
| `QuantileRegressor` | 3 | 0.897 | 5.96 | 0.49 | 1.000 | 0.770 | 0.2104 |
| `GradientBoostingRegressor`, quantile | 3 | 0.832 | 5.42 | 2.68 | 0.775 | 0.875 | 0.2052 |
| `LGBMRegressor`, quantile | 3 | 0.815 | 5.24 | 2.69 | 0.750 | 0.870 | 0.1963 |
| `XGBRegressor`, `reg:quantileerror` | 3 | 0.822 | 5.37 | 2.69 | 0.770 | 0.880 | 0.1952 |
| `RandomForestRegressor`, tree spread | 2 | 0.858 | 6.01 | 2.85 | 0.885 | 0.880 | 0.1981 |
| `RandomForestRegressor`, tree spread, 20 rows per leaf | 2 | 0.650 | 3.63 | 1.07 | 0.765 | 0.605 | 0.3006 |
| `RandomForestRegressor`, leaf observations | 3 | 0.893 | 6.46 | 2.89 | 0.935 | 0.895 | 0.2019 |
| `Ridge` and split conformal | 2 | 0.908 | 6.17 | 0.00 | 1.000 | 0.785 | 0.2072 |
| `GradientBoostingRegressor` quantile and CQR | 3 | 0.917 | 6.39 | 2.68 | 0.940 | 0.885 | 0.1990 |

Five readings come out of Table 10.

- Three rows at a coverage near 0.90 with a width standard deviation of 0.01 or less, and a conditional coverage of 1.000 against 0.785. Marginal coverage met by trading the bands against each other.
- `BayesianRidge` emitting a standard deviation between 1.889 and 1.901 across the 600 rows. A level 4 form, moving by 0.012 where the truth moves by a factor of ten, as equation (3) predicts.
- The three quantile boosting rows at 0.815 to 0.832 with a width standard deviation of 2.68. Adaptation without a guarantee.
- Tree spread at 0.858 with one row per leaf and 0.650 with twenty. The first number coming from leaves too small to average rather than from a predictive distribution.
- The last row at 0.917 with a width standard deviation of 2.68. The guarantee of section 7 and the adaptation of section 6 held together, at the price of the 600 calibration rows.

Only one linear row adapts at all, and its width standard deviation of 0.49 is the limit of what a straight line can do against a noise scale that is a function of the input. `QuantileRegressor` fits a linear quantile, so its two bounds may tilt apart but not bend, which is why its conditional coverage sits at 1.000 and 0.770 like the rows that do not adapt.

### B.1 Level 3 In Four Libraries

The parameter names of Table 7, on the same data.

```python
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb

LO, HI = 0.05, 0.95
TREES, LR, DEPTH = 300, 0.05, 3

# alpha is the L1 penalty here, not the level; it defaults to 1.0
ql = QuantileRegressor(quantile=LO, alpha=0.0).fit(X_fit, y_fit).predict(X_test)

# one fit per level
g = {a: GradientBoostingRegressor(loss="quantile", alpha=a, n_estimators=TREES,
                                  learning_rate=LR, max_depth=DEPTH,
                                  random_state=0).fit(X_fit, y_fit) for a in (LO, HI)}

# alpha is the level here
ll = lgb.LGBMRegressor(objective="quantile", alpha=LO, n_estimators=TREES,
                       learning_rate=LR, max_depth=DEPTH, verbose=-1,
                       random_state=0).fit(X_fit, y_fit).predict(X_test)

# both levels from one booster, returned as one column each
xq = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array([LO, HI]),
                      n_estimators=TREES, learning_rate=LR, max_depth=DEPTH, random_state=0)
xp = xq.fit(X_fit, y_fit).predict(X_test)
print(xp.shape)          # (600, 2)
```

The `alpha` argument means the L1 penalty in the first block and the level in the third, which is the naming collision section 5 records.

### B.2 The Two Forest Routes

The same fitted forest, read twice.

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(X_fit, y_fit)

# route 1: the spread of the tree predictions, the second term of equation (1) alone
per_tree = np.stack([t.predict(X_test) for t in rf.estimators_])
lo_spread, hi_spread = np.percentile(per_tree, [5, 95], axis=0)

# route 2: the fitted rows in the leaves this input lands in, both terms
leaf_fit, leaf_test = rf.apply(X_fit), rf.apply(X_test)
pools = []
for t in range(rf.n_estimators):
    d = {}
    for i, leaf in enumerate(leaf_fit[:, t]):
        d.setdefault(leaf, []).append(i)
    pools.append(d)

lo_leaf = np.empty(len(y_test))
hi_leaf = np.empty(len(y_test))
for i in range(len(y_test)):
    rows = []
    for t in range(rf.n_estimators):
        rows.extend(pools[t].get(leaf_test[i, t], ()))
    lo_leaf[i], hi_leaf[i] = np.percentile(y_fit[np.asarray(rows)], [5, 95])
```

Route 1 reaches 0.858 coverage on a default forest and 0.650 once `min_samples_leaf=20` makes each leaf hold at least twenty rows. Route 2 reaches 0.893 on the same forest and needs the fitted outcomes kept in memory.

### B.3 The Two Wrappers

Equations (4) and (5), on the 600 calibration rows.

```python
from sklearn.linear_model import Ridge

alpha = 0.10
k = np.ceil((len(y_cal) + 1) * (1 - alpha)) / len(y_cal)

# split conformal on a point model: one width for every row
ridge = Ridge().fit(X_fit, y_fit)
scores = np.abs(y_cal - ridge.predict(X_cal))
q_hat = np.quantile(scores, min(k, 1.0), method="higher")
lo_conf, hi_conf = ridge.predict(X_test) - q_hat, ridge.predict(X_test) + q_hat

# CQR on the fitted quantiles: the width keeps moving
E = np.maximum(g[LO].predict(X_cal) - y_cal, y_cal - g[HI].predict(X_cal))
q_cqr = np.quantile(E, min(k, 1.0), method="higher")
lo_cqr = g[LO].predict(X_test) - q_cqr
hi_cqr = g[HI].predict(X_test) + q_cqr
```

The corrections are 3.087 and 0.483. The first is large because it has to carry the whole width of a point model, and the second is small because the fitted quantiles already carry most of it and are only short of the level.
