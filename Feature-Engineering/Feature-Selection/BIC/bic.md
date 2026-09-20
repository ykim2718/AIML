# Bayesian Information Criterion
Rev. 1 | Created: 2026-09-20 | Updated: 2026-09-20 08:45 CDT

## 1. Purpose

- **Problem Statement**: A model with more parameters reaches a higher likelihood on the data it was fitted to, so a ranking by likelihood alone always puts the largest candidate first.
- **Goal**: Give the definition of BIC, the values its formula reads, the conditions under which its answer holds and the conditions that break it, so that a reader can decide whether to size a model with BIC or with a resampling score.
- **Non-Goal**: Prior design and the exact computation of a Bayes factor are left out. BIC is used as their large-sample approximation.

## 2. Summary

BIC scores a fitted model by its deviance plus a penalty of $\ln n$ per parameter, and the candidate with the smallest score is selected.

```math
\mathrm{BIC} = -2 \ln \hat{L} + k \ln n \hspace{19em} (1)
```

Table 1. The three values the formula reads

| Symbol      | Meaning                                                           | Read from                    |
| :---------: | :---------------------------------------------------------------: | :--------------------------: |
| $\hat{L}$   | Likelihood at the fitted parameter values                          | The fit that is scored       |
| $k$         | Count of free parameters, intercept and noise variance included    | The model specification      |
| $n$         | Count of observations the likelihood was computed on               | The dataset                  |

A BIC value is read against another BIC value computed on the same observations and the same response, and the difference between the two is the whole result.

Fig 1 puts the criterion beside AIC, which charges a fixed 2 per parameter on the same deviance.

![Fig 1](bic_fig/fig1.png)

Fig 1. AIC and BIC as one fit term under two penalties

AIC charges 2 per parameter at every sample size and BIC charges $\ln n$, so the two charges cross where $\ln n = 2$. [Appendix B](#appendix-b-aic-and-bic) sets the two criteria against each other; the rest of this document stays on BIC.

## 3. Principle

The penalty $k \ln n$ is the large-sample approximation of the marginal likelihood, so the smallest BIC marks the candidate with the largest posterior probability.

### 3.1 From The Marginal Likelihood

A Bayesian comparison ranks candidate model $M$ by its posterior probability given data $D$.

```math
p(M \mid D) \propto p(D \mid M)\, p(M) \hspace{19em} (2)
```

The marginal likelihood $p(D \mid M)$ integrates the likelihood over the prior of the parameters. Expanding the log integrand to second order around the maximum likelihood estimate and integrating the resulting Gaussian — the Laplace approximation — leaves a term that grows with $n$ and a remainder that stays bounded.

```math
-2 \ln p(D \mid M) = -2 \ln \hat{L} + k \ln n + O(1) \hspace{19em} (3)
```

Schwarz derived the criterion in the form of equation (3) and dropped the bounded remainder [[1](#ref-1)]. Under equal prior probability over the candidates, the ranking by equation (1) is the ranking by posterior probability up to terms that do not grow with $n$.

### 3.2 Conditions

- Assumption: a likelihood the model can write down, observations drawn independently, a Fisher information matrix that stays non-singular, and a parameter count $k$ held fixed while $n$ grows.
- Setting: $k$ counts every estimated parameter, the intercept and the noise variance included; $n$ counts observations.
- Breaking condition: a sample too small for the expansion, a candidate set whose members are all misspecified, a penalized fit whose parameter count is not an integer, and grouped data where the count of observations and the count of independent units differ.
- Where it is met: choosing a subset size along a selection path, the component count of a mixture, the lag order of a time series model, and the cluster count of a model-based clustering.

## 4. Application

BIC is read as a difference between candidates scored on one dataset, and the fit that produces each score is the one fit the candidate already needs.

### 4.1 Linear Regression Form

For a linear model with Gaussian errors, the maximized likelihood is a function of the residual sum of squares alone, which turns equation (1) into a form computed from a fit residual.

```math
\mathrm{BIC} = n \ln \frac{\mathrm{RSS}}{n} + k \ln n + n (\ln 2\pi + 1) \hspace{19em} (4)
```

The trailing term is the same for every candidate fitted to the same $n$ observations, so it cancels in every comparison and is dropped by most implementations. A score that omits it is therefore not comparable with a score from a library that keeps it. The step from the Gaussian likelihood to that form is in [Appendix C](#appendix-c-the-deviance-of-a-gaussian-linear-model).

### 4.2 Use In Feature Selection

The criterion sizes a model; the search that produces the candidates is a separate choice.

- Path: a forward or backward selection path, a regularization path, or an explicit list of subsets produces one candidate per size.
- Score: each candidate is fitted once and scored by equation (1), and the size with the smallest score is kept.
- Cost: one fit per candidate, against one fit per candidate per fold for a cross-validated score.
- Regularized path: the count of non-zero coefficients stands in for the degrees of freedom of a Lasso fit, and the score of `LassoLarsIC` reads the same count [[4](#ref-4)].

### 4.3 Reading A Difference

A difference of BIC between two candidates is twice the log of the Bayes factor the two imply, and the scale of the Bayes factor fixes what a given gap is worth [[3](#ref-3)].

Table 2. What a BIC difference in favour of the smaller score is worth

| Difference | Evidence against the higher-scoring candidate |
| :--------: | :-------------------------------------------: |
| 0 to 2     | Not worth more than a bare mention            |
| 2 to 6     | Positive                                      |
| 6 to 10    | Strong                                        |
| Over 10    | Very strong                                   |

A gap under 2 leaves the two candidates tied, and the candidate with fewer parameters is kept.

## References

<a id="ref-1"></a>[1] Schwarz, G. (1978). [Estimating the Dimension of a Model](https://doi.org/10.1214/aos/1176344136). *The Annals of Statistics*, 6(2), 461–464.<br>
<a id="ref-2"></a>[2] Akaike, H. (1974). [A New Look at the Statistical Model Identification](https://doi.org/10.1109/TAC.1974.1100705). *IEEE Transactions on Automatic Control*, 19(6), 716–723.<br>
<a id="ref-3"></a>[3] Kass, R. E., & Raftery, A. E. (1995). [Bayes Factors](https://doi.org/10.1080/01621459.1995.10476572). *Journal of the American Statistical Association*, 90(430), 773–795.<br>
<a id="ref-4"></a>[4] scikit-learn developers. [Lasso model selection: AIC-BIC / cross-validation](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html?utm_source=gemini). scikit-learn examples.<br>
<a id="ref-5"></a>[5] Displayr. [Information Criteria](https://docs.displayr.com/wiki/Information_Criteria?utm_source=gemini). Displayr documentation wiki.

---

## Appendix A. Terminology

- **Bayes factor**: The ratio of the marginal likelihoods of two candidate models on the same data.
- **Consistency**: The property that the probability of selecting the data-generating model tends to 1 as the sample grows, when that model is among the candidates.
- **Deviance**: Minus twice the maximized log-likelihood of a fitted model.
- **Efficiency**: The property that the prediction error of the selected model tends to the error of the best available candidate as the sample grows.
- **Laplace approximation**: The value of an integral obtained by expanding the log integrand to second order around its maximum and integrating the resulting Gaussian.
- **Marginal likelihood**: The likelihood of the data under a model with its parameters integrated out over their prior.
- **Regular model**: A model whose Fisher information matrix stays non-singular at the maximum likelihood estimate.

## Appendix B. AIC And BIC

Both criteria score a model by the same deviance and differ in what one parameter costs [[5](#ref-5)].

```math
\mathrm{AIC} = -2 \ln \hat{L} + 2k \hspace{19em} (5)
```

Table 3. The two criteria against each other

| Criterion | Penalty per parameter | Target quantity                       | Large-sample property                         | Selected size   |
| :-------: | :-------------------: | :-----------------------------------: | :-------------------------------------------: | :-------------: |
| AIC       | 2                     | Expected out-of-sample deviance        | Efficient, not consistent                  | Larger          |
| BIC       | $\ln n$               | Posterior probability of the model     | Consistent when a candidate generated the data | Smaller         |

AIC estimates the deviance the fitted model would reach on a fresh sample of the same size, which makes it a prediction criterion [[2](#ref-2)]. BIC estimates the posterior probability of the model, which makes it an identification criterion [[1](#ref-1)]. The two answer different questions on the same fit, so a disagreement between them is read by asking which of the two questions was being asked.

The penalties cross at $n = e^2 \approx 7.4$, so from $n = 8$ upward BIC charges more per parameter than AIC, and along one nested path the size BIC selects is never larger than the size AIC selects. [Appendix D](#appendix-d-worked-example) measures the size difference on one design.

## Appendix C. The Deviance Of A Gaussian Linear Model

The deviance that equation (4) carries is the Gaussian likelihood evaluated at the least squares estimate and at the noise variance that maximizes the likelihood.

A linear model with independent Gaussian errors writes the joint density of $n$ observations as a product whose exponent collects into the residual sum of squares $\mathrm{RSS}(\beta) = \sum_i (y_i - x_i^{\top} \beta)^2$.

```math
L(\beta, \sigma^2) = (2\pi\sigma^2)^{-n/2} \exp\left(-\frac{\mathrm{RSS}(\beta)}{2\sigma^2}\right) \hspace{19em} (6)
```

The log turns the product into a sum of three terms.

```math
\ln L(\beta, \sigma^2) = -\frac{n}{2}\ln(2\pi) - \frac{n}{2}\ln \sigma^2 - \frac{\mathrm{RSS}(\beta)}{2\sigma^2} \hspace{19em} (7)
```

The coefficient vector $\beta$ enters through $\mathrm{RSS}(\beta)$ alone, so the maximizing $\beta$ is the least squares estimate and $\mathrm{RSS}$ below is the residual sum of squares at that estimate. Setting the derivative in $\sigma^2$ to zero gives the maximizing variance.

```math
\frac{\partial \ln L}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{\mathrm{RSS}}{2\sigma^4} = 0
\quad \Longrightarrow \quad
\hat{\sigma}^2 = \frac{\mathrm{RSS}}{n} \hspace{19em} (8)
```

Putting $\hat{\sigma}^2$ back into equation (7) turns the last term into $n / 2$ and the middle term into $\ln(\mathrm{RSS} / n)$, and multiplying by $-2$ leaves the deviance.

```math
-2 \ln \hat{L} = n \left(\ln 2\pi + \ln \frac{\mathrm{RSS}}{n} + 1\right) \hspace{19em} (9)
```

Adding $k \ln n$ to equation (9) gives equation (4). The third row of Table 4, with $\mathrm{RSS} = 218.3$ and $n = 200$, returns 585.1 from equation (9).

A model outside the Gaussian family keeps equation (1) unchanged and puts the deviance of its own likelihood in place of equation (9). A fitting library reports that deviance beside the fitted parameters.

## Appendix D. Worked Example

The class in [`src/bic_model_selection.py`](src/bic_model_selection.py) builds a dataset whose generating features are known, grows a linear model one feature at a time, and scores every size with both criteria. Running `python3 src/bic_model_selection.py` from this folder prints the table below, counts how often each criterion recovers the generating features over repeated draws, and writes Fig 2.

The data are 200 observations of 10 independent standard normal features with $y = 3 x_0 - 2 x_1 + 1.5 x_2 + \varepsilon$ and $\varepsilon \sim N(0, 1)$, drawn under seed 2. Each step adds the feature that lowers the residual sum of squares most, and the parameter count of a step is its feature count plus the intercept and the noise variance.

Table 4. The forward path scored by both criteria

| Features | Added | RSS    | Deviance | AIC   | BIC   | Minimum |
| :------: | :---: | :----: | :------: | :---: | :---: | :-----: |
| 1        | x0    | 1549.9 | 977.1    | 983.1 | 993.0 |         |
| 2        | x1    | 767.5  | 836.5    | 844.5 | 857.7 |         |
| 3        | x2    | 218.3  | 585.1    | 595.1 | 611.6 | BIC     |
| 4        | x3    | 213.2  | 580.4    | 592.4 | 612.2 |         |
| 5        | x7    | 210.2  | 577.5    | 591.5 | 614.6 | AIC     |
| 6        | x6    | 208.5  | 575.9    | 591.9 | 618.3 |         |
| 7        | x4    | 207.1  | 574.5    | 592.5 | 622.2 |         |
| 8        | x8    | 206.5  | 574.0    | 594.0 | 626.9 |         |
| 9        | x5    | 206.1  | 573.6    | 595.6 | 631.8 |         |
| 10       | x9    | 206.1  | 573.5    | 597.5 | 637.1 |         |

The AIC minimum sits at five features, 591.5 against 595.1 for the generating three. Each of the two extra features lowers the deviance by 3.8 on average, above the 2 per parameter AIC charges and below the $\ln 200 = 5.3$ BIC charges.

![Fig 2](bic_fig/fig2.png)

Fig 2. Both criteria along the forward path, with the minimum of each circled

Both curves fall steeply while a generating feature is still missing and separate once the path passes three features: the BIC curve turns up immediately, and the AIC curve stays flat to five features before it turns.

Over 200 draws of the same design under consecutive seeds, BIC stops on exactly $x_0$, $x_1$ and $x_2$ in 166 of them and AIC in 46. The 166 against 46 is the consistency of Table 3 measured on one design rather than a claim about every design.
