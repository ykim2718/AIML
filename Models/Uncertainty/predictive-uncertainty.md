# Predictive Uncertainty
Rev. 1 | Created: 2026-09-10 | Updated: 2026-09-10 19:10 CDT

## 1. Purpose

- **Problem Statement**: 학습된 regression model의 예측 에 대한 확률을 구하여 ensemble 모델등에 활용하고 싶다.
- **Goal**: 예측 값의 확률의 taxonomy 와 hierarchy를 통한 방법을 제시한다.
- **Non-Goal**: classifier의 결과를 다루지 않는다.

## 2. Summary

Three implementations produce a probability from a fitted regression model, and Table 1 is the choice between them. Probabilistic regression fits a mean and a variance per row and hands back a density. Quantile regression fits a bound per requested level and hands back an interval. Bootstrap and resampling read the disagreement among refitted members as a spread.

The three do not deliver the same thing, and Table 2 orders what they deliver into four levels. A point answers what to expect, an interval answers whether the outcome falls inside one range, a set of quantiles answers that for a range whose width moves with the input, and a density answers what the probability is that the outcome passes any limit. Only the last level answers the question the Problem Statement asks of an ensemble, because a per-row weight is a function of a per-row variance.

Two of the three implementations measure the wrong quantity if used as they are usually written. Bootstrapping a point model measures how much the fitted mean moves, not how much the outcome moves, and Appendix B measures its coverage at 0.115 against a nominal 0.90. A forest's tree spread has the same defect, and it is hidden at default settings, where a leaf holds one row and the tree spread accidentally resembles the noise.

Section 10 is the payoff for the Problem Statement. Two members whose feature sets leave each of them ignorant in a different place are combined at 2.3910 RMSE by equal weights and at 2.1689 by weights built from their own per-row variances, against 2.0643 and 3.0808 for the members alone.

## 3. Taxonomy

Table 1 is the three implementations, with what each emits and when each is used.

Table 1. The three implementations

| Implementation | Emits | Recommended when |
|----------------|-------|------------------|
| Probabilistic regression | Mean $\mu$ and standard deviation $\sigma$ per row | The probability of passing a stated limit is wanted, and a distribution family can be assumed |
| Quantile regression | One bound per requested level, such as 0.10, 0.50 and 0.90 | A stated interval is wanted at levels fixed in advance, with no distribution assumed |
| Bootstrap and resampling | Spread of the predictions of refitted members | The members already exist, and what is wanted is how much the fit itself is in doubt |

Fig 1 is the same three with the route each takes in a library.

```text
Probability attached to the prediction of a regression model
|
+-- Probabilistic regression ....... a loss that fits the variance as well as the mean
|   +-- Gaussian NLL loss .......... custom two-output objective in XGBoost, section 5.1
|   +-- Evidence framework ......... BayesianRidge, ARDRegression, Gaussian process
|   +-- Natural gradient boosting .. NGBoost
|
+-- Quantile regression ............ the pinball loss, one level at a time
|   +-- Linear .................... QuantileRegressor
|   +-- Boosted trees ............. GradientBoostingRegressor, HistGradientBoostingRegressor, LightGBM
|   +-- Boosted trees, vectorized .. XGBoost, several levels from one booster
|
+-- Bootstrap and resampling ....... the members of an ensemble read as a sample
    +-- Refits on resampled rows ... any estimator, the spread of the refitted means
    +-- Tree spread ................ RandomForestRegressor, the spread over its trees
    +-- Leaf observations .......... RandomForestRegressor, the fitted rows in the leaves
```

Fig 1. The three implementations and the route each takes in a library

The third branch differs from the other two in what it is a spread of. The first two fit a spread of the outcome; the third measures a spread of the model, and section 7 is what follows from that.

## 4. Hierarchy

The three implementations of Table 1 do not answer the same questions, and Table 2 orders what they answer into four levels. Each level contains the one below it, so a level is reached by every method that reaches a level above it.

Table 2. What each level answers

| Level | Emits | Answers | Reached by |
|-------|-------|---------|------------|
| 1. Point | One number | What to expect | Any fitted regression model |
| 2. Interval | Two numbers at one level | Whether the outcome falls inside one range | A residual standard deviation, or any method at a higher level |
| 3. Quantile set | One bound per requested level | The same, with a range that moves with the input | Quantile regression, leaf observations |
| 4. Distribution | A density and a full CDF | The probability of passing any limit, and any moment or quantile | Probabilistic regression, or a dense grid of quantiles |

Level 4 is what the Problem Statement needs. An ensemble weight has to be a number per row, so it has to come from a variance per row, and a variance is what level 4 emits and level 3 leaves implicit in a pair of bounds. Level 4 is also the level at which the question "what is the probability that this prediction exceeds the limit" is answered for a limit that was not known when the model was fitted.

The level a method reaches says nothing about whether what it emits is correct. `BayesianRidge` reaches level 4 and Appendix B measures its standard deviation between 1.889 and 1.901 across rows whose true noise standard deviation runs from 0.51 to 5.25, a correlation of 0.420 with the truth. Section 11 is the measurement that separates the two.

## 5. Probabilistic Regression

A loss that reads the variance as a second output is what turns a point model into a density. For a Gaussian assumption the loss is the negative log-likelihood, written here with $s = \log \sigma$ so that the second output is unconstrained [[6](#ref-6)].

$$\mathrm{NLL}(y, \mu, s) = s + \frac{(y - \mu)^{2}}{2}e^{-2s} \hspace{19em} (1)$$

Equation (1) gives $\mu$ and $\sigma$ per row, and the probability of passing a limit $L$ follows as $1 - \Phi\!\left((L - \mu)/\sigma\right)$. Appendix B puts the limit at 6.0 and measures 0.0476 from this fit against an actual rate of 0.0767.

### 5.1 What The Boosting Libraries Provide

Neither boosting library has a built-in objective that emits a variance, and the objective names that suggest one do not exist. In XGBoost 3.2.0 both `reg:normal` and `reg:gaussian` fail with `Unknown objective function`, and of the objectives that do exist, `reg:squarederror` fits the mean, `reg:absoluteerror` fits the median, and `reg:pseudohubererror` fits a robust centre, each of them a single number per row. In LightGBM 4.7.0 the names `gaussian` and `normal` fail the same way.

The route that works is a custom objective with two outputs, which Appendix B fits and measures in XGBoost. The gradient and Hessian of equation (1) are supplied for both outputs at once, and `multi_strategy="one_output_per_tree"` grows a tree per output, so `predict` returns a matrix of shape `(600, 2)` whose columns are $\mu$ and $\log \sigma$. The same route is closed in LightGBM, whose estimator rejects a two-column label with `y should be a 1d array`, so a variance from that library means NGBoost or a grid of quantile fits.

Two libraries supply the same thing already assembled. NGBoost boosts the parameters of a chosen distribution together under a natural gradient [[3](#ref-3)]. `BayesianRidge` and `ARDRegression` fit the noise and the coefficients together under the evidence framework and expose the result through `predict(X, return_std=True)` [[4](#ref-4)].

### 5.2 The Variance Head Overfits

The fitted variance shrinks as trees are added, and the coverage falls with it. Appendix B measures 0.852 at 100 trees, 0.817 at 200 and 0.765 at 400, with the mean width falling from 5.40 to 4.68 over the same range.

The cause is in equation (1). Lowering $s$ is rewarded wherever the residual on the training rows is small, and a boosted model drives the training residual toward zero, so the variance head is fitted against a residual that keeps shrinking for reasons the test rows do not share. Holding out rows to stop on, or bounding $\sigma$ from below, is what keeps the second output honest.

`BayesianRidge` has the opposite failure and it is structural rather than a matter of tuning. Its emitted variance is a fitted noise constant plus a term that grows with the distance of the input from the fitted data, so on well-determined rows the width is one number. Appendix B measures it moving by 0.012 across 600 rows.

## 6. Quantile Regression

Replacing the squared-error loss with the pinball loss changes which functional of the conditional distribution the same model fits, and at level $\alpha$ that functional is the $\alpha$ quantile [[1](#ref-1)].

$$L_{\alpha}(y, q) = \max\left\{\alpha\,(y - q),\ (\alpha - 1)(y - q)\right\} \hspace{19em} (2)$$

Equation (2) is minimized by a bound below which a fraction $\alpha$ of the training outcomes falls, so a fit at 0.10 and one at 0.90 bracket 80 per cent of those. Nothing in the fit ties the two together: each is a separate minimization, and Appendix B measures the resulting coverage at 0.815 to 0.832 for the three boosting libraries against a nominal 0.90.

Table 3 is the parameter that selects the level in each library. The name `alpha` means the level in LightGBM and in `GradientBoostingRegressor`, and the L1 penalty in `QuantileRegressor`, where the level is `quantile` and the penalty defaults to 1.0.

Table 3. How each library fits a quantile

| Estimator | Parameter | Levels per fit |
|-----------|-----------|----------------|
| `QuantileRegressor` | `quantile`, with `alpha` as the L1 penalty | One |
| `GradientBoostingRegressor` | `loss="quantile"`, `alpha` | One |
| `HistGradientBoostingRegressor` | `loss="quantile"`, `quantile` | One |
| `LGBMRegressor` | `objective="quantile"`, `alpha` | One |
| `XGBRegressor` | `objective="reg:quantileerror"`, `quantile_alpha` | Several, one column returned per level |

One fit per level is the cost that decides how this scales. Ten levels from `LGBMRegressor` are ten models to train, store and serve, while `XGBRegressor` takes the levels as an array and returns a matrix of one column each from a single booster.

Separately fitted levels are also free to cross, since no term ties one fit to another, and a 0.05 bound may land above a 0.95 bound on some rows. Appendix B counts the crossings on its 600 test rows and finds none, which is a property of that fit rather than a guarantee, so the emitted columns are sorted before use.

## 7. Bootstrap And Resampling

Refitting a model on resampled rows measures how much the fitted mean moves, and that is a different quantity from how much the outcome moves [[7](#ref-7)]. The gap is not a matter of degree. Appendix B refits `Ridge` on 200 bootstrap resamples and takes the 5th and 95th percentiles of the 200 predictions per row, which gives a mean width of 0.40 and a coverage of 0.115 against a nominal 0.90, while the residual standard deviation of the same model is 1.940.

The spread across the trees of a random forest is the same quantity and carries the same defect. It is harder to see at default settings, where a fully grown leaf holds about one row, so each tree's prediction is close to one noisy observation and the spread over trees resembles the noise by accident. Appendix B measures 0.858 coverage there and 0.650 once `min_samples_leaf=20` makes each leaf average its rows, and the second number is what the route is actually worth.

The forest has a second route that does measure the outcome. The fitted rows in the leaves an input lands in, pooled across trees, are a weighted sample of the outcome at that input, and any quantile of that sample is a level 3 answer [[2](#ref-2)]. Appendix B measures 0.893 coverage with a width that moves by 2.89, against 0.650 for the tree spread on the same forest.

Nothing above rules out the resampling route; it fixes what it is for. The spread of refitted members is the right quantity for asking how much the fit itself is in doubt, which is what grows where the training data thins. Added to a fitted noise term it gives the two parts of a predictive variance, and used alone it gives the smaller of the two.

## 8. Tree And Ensemble Family

Three of the four tree models reach level 3 by swapping in the pinball loss, and the fourth reaches it without changing its loss at all. Table 4 is that difference and what each reaches beyond it.

Table 4. What each tree model reaches

| Estimator | Level 3 | Level 4 | Second fit needed |
|-----------|---------|---------|-------------------|
| `LGBMRegressor` | `objective="quantile"` | A dense grid of levels, or NGBoost | One fit per level |
| `XGBRegressor` | `objective="reg:quantileerror"` | Custom two-output objective, section 5.1 | No, the level array is taken at once |
| `GradientBoostingRegressor` | `loss="quantile"` | A dense grid of levels | One fit per level |
| `RandomForestRegressor` | Leaf observations, section 7 | A dense grid read off the same leaves | No, one fit serves every level |

The random forest is the only row whose one fit serves both levels, because the sample it reads is already stored in the fitted trees, and the `No` in the `XGBRegressor` row covers its level 3 entry alone. What it spends instead is memory and prediction time: the fitted outcomes have to be kept and a prediction touches every tree's leaf pool rather than one number per tree.

`XGBRegressor` is the only row that takes several levels in one fit, and the two entries in its row are separate mechanisms. `quantile_alpha` takes an array and returns one column per level; the level 4 entry is the custom objective of section 5.1, which returns two columns that are a mean and a log standard deviation rather than two bounds.

## 9. Linear Family

Two of the linear regressors emit a mean and a variance, one emits a quantile when asked for a level, and the remaining five emit a centre and nothing else. Table 5 is what each emits.

Table 5. What the linear regressors emit

| Estimator | `predict` signature | Level as fitted | Functional estimated |
|-----------|---------------------|-----------------|----------------------|
| `Ridge` | `predict(X)` | 1 | Conditional mean |
| `Lasso` | `predict(X)` | 1 | Conditional mean |
| `ElasticNet` | `predict(X)` | 1 | Conditional mean |
| `HuberRegressor` | `predict(X)` | 1 | Robust centre |
| `LinearSVR` | `predict(X)` | 1 | Centre of an epsilon-insensitive tube |
| `QuantileRegressor` | `predict(X)` | 3, one level per fit | Conditional quantile |
| `BayesianRidge` | `predict(X, return_std=False)` | 4, Gaussian | Conditional mean and variance |
| `ARDRegression` | `predict(X, return_std=False)` | 4, Gaussian | Conditional mean and variance |

The first five rows differ in the centre they estimate and not in what they say about the spread, which is nothing. `Ridge`, `Lasso` and `ElasticNet` differ from one another only in the penalty and all three fit the conditional mean; `HuberRegressor` and `LinearSVR` fit a centre that resists outliers, which makes the point more reliable and leaves the question of the spread exactly where it was.

Reaching level 3 from any of those five means `QuantileRegressor`, which is a different fit rather than an option on the existing one, since no linear estimator in the family takes a loss argument. Reaching level 4 means `BayesianRidge` or `ARDRegression`, with the constant noise term of section 5.2.

## 10. Weighting An Ensemble

A per-row variance turns a fixed ensemble weight into one that moves with the input, which is what the Problem Statement asks for. For two members with means $\mu_{1}, \mu_{2}$ and variances $\sigma_{1}^{2}, \sigma_{2}^{2}$ at the same row, the weights that minimize the variance of the combination are inverse to those variances.

$$w_{1} = \frac{1/\sigma_{1}^{2}}{1/\sigma_{1}^{2} + 1/\sigma_{2}^{2}}, \qquad \hat y = w_{1}\mu_{1} + (1 - w_{1})\mu_{2} \hspace{19em} (3)$$

Appendix B fits two members whose feature sets leave each of them blind to a different driver of the outcome and combines them by equation (3). The members score 2.0643 and 3.0808 RMSE alone, equal weights score 2.3910, and the weights of equation (3) score 2.1689 with $w_{1}$ moving between 0.36 and 0.93 across rows.

The gain comes from the members differing in where they are uncertain, and it disappears when they do not. A second pair in Appendix B, whose members share the feature that carries most of the noise, scores 2.7213 by equation (3) against 2.6942 by equal weights, because both members report almost the same $\sigma$ and the weights it produces are near 0.5 everywhere while one member is simply worse. A variance that is wrong is worse than no variance at all, since equation (3) then pays the wrong member.

## 11. Calibration

Coverage measured over all rows is met by over-covering the quiet rows and under-covering the noisy ones, so an emitted interval is judged by three numbers together. Table 6 is the three, with the failure each one catches.

Table 6. What an emitted interval is judged by

| Measure | Definition | Catches |
|---------|------------|---------|
| Marginal coverage | Fraction of all rows inside the interval | Bounds too narrow or too wide overall |
| Conditional coverage | The same, within a band of rows grouped by an input | A constant width standing in for a varying one |
| Sharpness | Mean width of the interval | Bounds made to cover by being useless |

Coverage and sharpness are read against each other, because either alone is trivially met. The rule is to maximize sharpness subject to calibration: among the methods whose coverage holds, the narrowest interval wins [[8](#ref-8)]. Appendix B is arranged as that comparison.

A single number that scores both at once is a proper scoring rule, whose expected value is optimized by the true distribution and by no other [[5](#ref-5)]. Two of them cover the levels of Table 2.

- Pinball loss of equation (2), proper for one quantile. Available as `mean_pinball_loss` and `d2_pinball_score`.
- Continuous ranked probability score, proper for a full distribution. The integral of the squared difference between the emitted CDF and the step function at the outcome.

Neither replaces the coverage table. A proper score ranks methods against one another without saying whether the best of them is right, and the coverage of 0.115 in Appendix B is a fact about that interval which no ranking reports.

## 12. Selection

The choice follows from what the probability is for. Table 7 reads from the left column.

Table 7. Which implementation to use

| Use | When | Why |
|-----|------|-----|
| Probabilistic regression, custom NLL objective | A tree model, and a per-row variance is needed for a limit or a weight | Level 4 from the library already in use, at the price of section 5.2 |
| NGBoost | The same, and a maintained implementation is preferred to a custom objective | The natural gradient and the distribution families come assembled |
| `BayesianRidge` or `ARDRegression` | A linear fit, and a density is wanted with no split spent | Level 4 natively, with a noise term that does not move |
| Quantile regression | The levels are fixed in advance and no distribution should be assumed | Level 3 without a distribution family, at one fit per level |
| Leaf observations of a forest | A forest is already fitted and more than one level is wanted | Every level from one fit, and no assumption |
| Bootstrap and resampling | The question is how much the fit is in doubt, not how much the outcome varies | The only one of the three that answers that question, section 7 |

## 13. Further Work

- **A coverage guarantee over any of the three** — Every method in Table 1 is measured in Appendix B against a nominal 0.90 and none of them is guaranteed to reach it; the three that adapt land between 0.815 and 0.893. Conformal prediction sets the correction from a held-out split so that the marginal coverage holds in finite samples for any underlying model, which makes it a wrapper over any row of Table 1 rather than a fourth implementation. It needs a calibration split exchangeable with the rows to be predicted, which is the condition a drifting process breaks.
- **A variance head that does not shrink** — Section 5.2 measures the coverage of the NLL fit falling from 0.852 to 0.765 as trees are added, because the variance is fitted against a training residual that keeps shrinking. Separating the rows the mean is fitted on from the rows the variance is fitted on is the treatment the mean-variance literature settles on, and it is now cheap enough to run inside a boosting loop. It needs a rule for splitting those rows and a measurement of what the split costs the mean, since rows spent on the variance are rows the mean does not see.

## References

<a id="ref-1"></a>[1] Koenker, R. and Bassett, G. (1978). [Regression Quantiles](https://doi.org/10.2307/1913643). *Econometrica*, 46(1), 33-50.<br>
<a id="ref-2"></a>[2] Meinshausen, N. (2006). [Quantile Regression Forests](https://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf). *Journal of Machine Learning Research*, 7, 983-999.<br>
<a id="ref-3"></a>[3] Duan, T., Avati, A., Ding, D. Y., Thai, K. K., Basu, S., Ng, A. Y. and Schuler, A. (2020). [NGBoost: Natural Gradient Boosting for Probabilistic Prediction](https://proceedings.mlr.press/v119/duan20a/duan20a.pdf). *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, 2690-2700.<br>
<a id="ref-4"></a>[4] MacKay, D. J. C. (1992). [Bayesian Interpolation](https://doi.org/10.1162/neco.1992.4.3.415). *Neural Computation*, 4(3), 415-447.<br>
<a id="ref-5"></a>[5] Gneiting, T. and Raftery, A. E. (2007). [Strictly Proper Scoring Rules, Prediction, and Estimation](https://doi.org/10.1198/016214506000001437). *Journal of the American Statistical Association*, 102(477), 359-378.<br>
<a id="ref-6"></a>[6] Nix, D. A. and Weigend, A. S. (1994). [Estimating the Mean and Variance of the Target Probability Distribution](https://doi.org/10.1109/ICNN.1994.374138). *Proceedings of 1994 IEEE International Conference on Neural Networks*, 1, 55-60.<br>
<a id="ref-7"></a>[7] Efron, B. (1979). [Bootstrap Methods: Another Look at the Jackknife](https://doi.org/10.1214/aos/1176344552). *The Annals of Statistics*, 7(1), 1-26.<br>
<a id="ref-8"></a>[8] Gneiting, T., Balabdaoui, F. and Raftery, A. E. (2007). [Probabilistic Forecasts, Calibration and Sharpness](https://doi.org/10.1111/j.1467-9868.2007.00587.x). *Journal of the Royal Statistical Society: Series B*, 69(2), 243-268.

---

## Appendix A. Terminology

- **conditional coverage**: The coverage of an interval within a band of rows rather than over all of them.
- **continuous ranked probability score**: The integral of the squared difference between an emitted CDF and the step function at the outcome.
- **coverage**: The fraction of rows whose outcome falls inside the emitted interval.
- **inverse-variance weighting**: The ensemble weights of equation (3), each inverse to that member's variance at that row.
- **leaf observations**: The fitted rows sorted into the leaf an input lands in, read as a sample of the outcome.
- **marginal coverage**: The coverage of an interval over all rows.
- **negative log-likelihood**: The loss of equation (1), minimized by the mean and variance of the assumed family.
- **pinball loss**: The asymmetric absolute loss of equation (2), minimized by one quantile.
- **prediction interval**: Two numbers between which a future outcome is claimed to fall at a stated level.
- **predictive distribution**: A density or CDF for the outcome at one input.
- **proper scoring rule**: A score whose expected value is optimized by the true distribution and by no other.
- **sharpness**: The narrowness of an emitted interval or density, judged only among methods whose coverage holds.
- **variance head**: The second output of a two-output model, which carries the log standard deviation.

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

The split is 1800 rows to fit on, 600 held for measurements that need rows outside the fit, and 600 to measure on. Every interval is asked for the 0.05 and 0.95 levels, so the nominal coverage is 0.90 throughout. The boosting models are given the same 300 trees, the same learning rate of 0.05 and the same depth of 3, so that the comparison between them is one of objectives rather than of defaults.

The true noise standard deviation on the test rows runs from 0.51 to 5.25. Averaged over the three bands of those rows sorted by $|x_{1}|$, the true 90 per cent width is 2.85, 5.21 and 9.28, so no one number serves all three. The two band columns of Table 8 are the outer two of those bands.

Table 8. Eleven methods on the same 600 held-out rows, nominal coverage 0.90

| Method | Implementation | Coverage | Mean width | Width sd | Coverage, quiet third | Coverage, noisy third | Pinball |
|--------|----------------|----------|------------|----------|-----------------------|-----------------------|---------|
| `Ridge`, residual standard deviation | None, baseline | 0.912 | 6.38 | 0.00 | 1.000 | 0.790 | 0.2078 |
| XGBoost Gaussian NLL, 100 trees | Probabilistic | 0.852 | 5.40 | 2.53 | 0.875 | 0.860 | 0.1999 |
| `BayesianRidge`, `return_std` | Probabilistic | 0.908 | 6.22 | 0.01 | 1.000 | 0.785 | 0.2073 |
| `QuantileRegressor` | Quantile | 0.897 | 5.96 | 0.49 | 1.000 | 0.770 | 0.2104 |
| `GradientBoostingRegressor`, quantile | Quantile | 0.832 | 5.42 | 2.68 | 0.775 | 0.875 | 0.2052 |
| `LGBMRegressor`, quantile | Quantile | 0.815 | 5.24 | 2.69 | 0.750 | 0.870 | 0.1963 |
| `XGBRegressor`, `reg:quantileerror` | Quantile | 0.822 | 5.37 | 2.69 | 0.770 | 0.880 | 0.1952 |
| `Ridge`, 200 bootstrap refits | Resampling | 0.115 | 0.40 | 0.12 | 0.170 | 0.090 | 0.5746 |
| `RandomForestRegressor`, tree spread | Resampling | 0.858 | 6.01 | 2.85 | 0.885 | 0.880 | 0.1981 |
| `RandomForestRegressor`, tree spread, `min_samples_leaf=20` | Resampling | 0.650 | 3.63 | 1.07 | 0.765 | 0.605 | 0.3006 |
| `RandomForestRegressor`, leaf observations | Resampling | 0.893 | 6.46 | 2.89 | 0.935 | 0.895 | 0.2019 |

Five readings come out of Table 8.

- Bootstrap of a point model at 0.115 coverage with a mean width of 0.40, against a residual standard deviation of 1.940 for the same model. The spread of the fitted mean, not of the outcome.
- Tree spread at 0.858 with one row per leaf and 0.650 with twenty. The first number coming from leaves too small to average rather than from a predictive distribution.
- Three rows at a coverage near 0.90 with a width standard deviation of 0.49 or less, and a conditional coverage of 1.000 against 0.770. Marginal coverage met by trading the bands against each other.
- The NLL fit and the leaf observations covering 0.852 and 0.893 with the width moving by 2.53 and 2.89, and a conditional coverage within 0.04 of itself across the bands. The two rows whose width follows the noise.
- The three quantile boosting rows at 0.815 to 0.832. Adaptation without a guarantee, since equation (2) is minimized on the training rows and nothing forces the held-out frequency to match the level.

### B.1 The Variance Head

Equation (1) supplied to XGBoost as a two-output objective.

```python
import xgboost as xgb

def gaussian_nll(y_true, raw):
    y_true = np.asarray(y_true)[:, 0]
    mu, s = raw[:, 0], raw[:, 1]           # s is the log standard deviation
    inv = np.exp(-2.0 * s)
    r = y_true - mu
    grad = np.stack([-r * inv, 1.0 - r ** 2 * inv], axis=1)
    hess = np.stack([inv, 2.0 * r ** 2 * inv], axis=1)
    return grad, hess

m = xgb.XGBRegressor(objective=gaussian_nll, n_estimators=100, learning_rate=0.05,
                     max_depth=3, min_child_weight=20, base_score=0.0,
                     multi_strategy="one_output_per_tree", random_state=0)
# the label is duplicated because the model has two outputs
m.fit(X_fit, np.stack([y_fit, y_fit], axis=1))

raw = m.predict(X_test)                    # (600, 2)
mu, sd = raw[:, 0], np.exp(raw[:, 1])
```

The emitted standard deviation runs from 0.77 to 5.33 and correlates 0.962 with the true noise standard deviation, against 0.420 for `BayesianRidge` on the same rows.

Table 9 is the coverage of that fit as trees are added, at the same learning rate and depth.

Table 9. Coverage of the NLL fit against the tree count

| Trees | Coverage | Mean width | Coverage, quiet third | Coverage, noisy third |
|-------|----------|------------|-----------------------|-----------------------|
| 100 | 0.852 | 5.40 | 0.875 | 0.860 |
| 200 | 0.817 | 5.05 | 0.810 | 0.860 |
| 400 | 0.765 | 4.68 | 0.750 | 0.835 |

The probability of passing a limit is read off the same two columns, and Table 10 is that probability at two limits against the rate actually observed.

Table 10. Probability of exceeding a limit, averaged over the 600 test rows

| Limit | XGBoost Gaussian NLL | `BayesianRidge` | Observed rate |
|-------|----------------------|-----------------|---------------|
| 4.0 | 0.1318 | 0.1754 | 0.1650 |
| 6.0 | 0.0476 | 0.0762 | 0.0767 |

`BayesianRidge` is closer in both rows of Table 10 while being the worse model of the two at the level of a single row, and the reason is in what the table averages. Its one constant standard deviation of about 1.89 is near the average noise of the dataset, which is the right number for a rate averaged over all rows and the wrong number for any row in particular.

### B.2 Quantile Regression In Four Libraries

The parameters of Table 3, on the same data.

```python
from sklearn.linear_model import QuantileRegressor
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb

LO, HI = 0.05, 0.95
TREES, LR, DEPTH = 300, 0.05, 3

# alpha is the L1 penalty here, not the level, and it defaults to 1.0
ql = QuantileRegressor(quantile=LO, alpha=0.0).fit(X_fit, y_fit).predict(X_test)

# one fit per level
g = {a: GradientBoostingRegressor(loss="quantile", alpha=a, n_estimators=TREES,
                                  learning_rate=LR, max_depth=DEPTH,
                                  random_state=0).fit(X_fit, y_fit) for a in (LO, HI)}

# alpha is the level here
ll = lgb.LGBMRegressor(objective="quantile", alpha=LO, n_estimators=TREES,
                       learning_rate=LR, max_depth=DEPTH, verbose=-1,
                       random_state=0).fit(X_fit, y_fit).predict(X_test)

# both levels from one booster, one column returned per level
xq = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array([LO, HI]),
                      n_estimators=TREES, learning_rate=LR, max_depth=DEPTH, random_state=0)
xp = xq.fit(X_fit, y_fit).predict(X_test)
print(xp.shape)                            # (600, 2)
```

The LightGBM bounds cross on none of the 600 test rows.

### B.3 The Two Forest Routes

The same fitted forest, read twice.

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(X_fit, y_fit)

# route 1: the spread of the tree predictions, which is a spread of the fitted mean
per_tree = np.stack([t.predict(X_test) for t in rf.estimators_])
lo_spread, hi_spread = np.percentile(per_tree, [5, 95], axis=0)

# route 2: the fitted rows in the leaves this input lands in, which is a sample of the outcome
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

Route 1 reaches 0.858 coverage on this forest and 0.650 once `min_samples_leaf=20` makes each leaf hold at least twenty rows. Route 2 reaches 0.893 on the same forest and needs the fitted outcomes kept in memory.

### B.4 Weighting Two Members By Their Variances

Equation (3), over two members fitted by the objective of B.1 on different feature sets.

```python
from sklearn.metrics import mean_squared_error

def fit_nll(columns, seed):
    m = xgb.XGBRegressor(objective=gaussian_nll, n_estimators=100, learning_rate=0.05,
                         max_depth=3, min_child_weight=20, base_score=0.0,
                         multi_strategy="one_output_per_tree", random_state=seed)
    m.fit(X_fit[:, columns], np.stack([y_fit, y_fit], axis=1))
    return m

# each member is blind to a different driver of the outcome
c1, c2 = [0, 1, 2], [0, 3, 4]
r1 = fit_nll(c1, 1).predict(X_test[:, c1])
r2 = fit_nll(c2, 2).predict(X_test[:, c2])
mu1, sd1 = r1[:, 0], np.exp(r1[:, 1])
mu2, sd2 = r2[:, 0], np.exp(r2[:, 1])

w1 = (1 / sd1 ** 2) / (1 / sd1 ** 2 + 1 / sd2 ** 2)
combined = w1 * mu1 + (1 - w1) * mu2
```

Table 11 is that combination against the alternatives, scored as the root mean squared error on the 600 test rows.

Table 11. Two members combined, root mean squared error

| Combination | Columns (0, 1, 2) and (0, 3, 4) | Columns (0, 1) and (1, 2) |
|-------------|---------------------------------|---------------------------|
| Member 1 alone | 2.0643 | 2.5430 |
| Member 2 alone | 3.0808 | 3.6136 |
| Equal weights | 2.3910 | 2.6942 |
| Inverse-variance weights, equation (3) | 2.1689 | 2.7213 |

The first column is the case the Problem Statement describes and the weights move between 0.36 and 0.93 across rows. The second column is the same procedure over two members that share the feature carrying the noise: both report nearly the same standard deviation, the weights sit near 0.5 everywhere, and equation (3) loses to equal weights because it has no information left to act on.
