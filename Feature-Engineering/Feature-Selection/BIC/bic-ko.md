# Bayesian Information Criterion
Rev. 0 | Created: 2026-09-20 | Updated: 2026-09-20 07:13 CDT

## 1. Purpose

- **Problem Statement**: Parameter 가 많은 model 일수록 적합에 쓴 data 에서 더 높은 likelihood 에 닿으므로, likelihood 만으로 순위를 매기면 언제나 가장 큰 후보가 1 위가 된다.
- **Goal**: BIC 의 정의, 그 식이 읽는 값, 답이 성립하는 조건과 깨지는 조건을 정리하여, 주어진 data 에서 model 의 크기를 BIC 로 정할지 resampling 점수로 정할지 독자가 스스로 가릴 수 있게 한다.
- **Non-Goal**: Prior 설계와 Bayes factor 의 정확한 계산은 다루지 않는다. BIC 를 그 둘의 대표본 근사로 쓴다.

## 2. Summary

BIC 는 적합이 끝난 model 을 deviance 에 parameter 하나당 $\ln n$ 의 penalty 를 더한 값으로 채점하고, 점수가 가장 작은 후보를 고른다.

```math
\mathrm{BIC} = -2 \ln \hat{L} + k \ln n \hspace{19em} (1)
```

Table 1. The three values the formula reads

| Symbol      | Meaning                                                           | Read from                    |
| :---------: | :---------------------------------------------------------------: | :--------------------------: |
| $\hat{L}$   | 적합이 끝난 parameter 값에서의 likelihood                          | 채점하는 적합 자체           |
| $k$         | 자유 parameter 의 개수, intercept 와 noise variance 포함           | Model 의 설정                |
| $n$         | Likelihood 를 계산한 관측의 개수                                   | Data 집합                    |

BIC 값은 같은 관측과 같은 response 에서 계산한 다른 BIC 값과 견주어 읽으며, 두 값의 차이가 결과의 전부다.

Fig 1 은 같은 deviance 에 parameter 하나당 2 를 물리는 AIC 를 BIC 옆에 나란히 둔다.

![Fig 1](bic_fig/fig1.png)

Fig 1. AIC and BIC as one fit term under two penalties

AIC 는 표본 크기와 무관하게 parameter 하나당 2 를 물리고 BIC 는 $\ln n$ 을 물리므로, 두 값은 $\ln n = 2$ 인 자리에서 교차한다. 두 기준의 대조는 [Appendix B](#appendix-b-aic-and-bic) 에 두고, 본문은 BIC 를 따라간다.

## 3. Principle

Penalty $k \ln n$ 은 marginal likelihood 의 대표본 근사에서 나오며, 그래서 BIC 가 가장 작은 후보가 posterior probability 가 가장 큰 후보가 된다.

### 3.1 From The Marginal Likelihood

Bayesian 비교는 data $D$ 가 주어졌을 때의 posterior probability 로 후보 model $M$ 의 순위를 매긴다.

```math
p(M \mid D) \propto p(D \mid M)\, p(M) \hspace{19em} (2)
```

Marginal likelihood $p(D \mid M)$ 는 parameter 의 prior 에 대해 likelihood 를 적분한 값이다. 적분 안의 log 를 maximum likelihood 추정값 둘레에서 2 차까지 전개하고 그 Gaussian 을 적분하면 (Laplace approximation), $n$ 과 함께 커지는 항과 유계로 남는 나머지가 갈린다.

```math
-2 \ln p(D \mid M) = -2 \ln \hat{L} + k \ln n + O(1) \hspace{19em} (3)
```

Schwarz 는 식 (3) 의 형태로 기준을 유도하고 유계인 나머지를 버렸다 [[1](#ref-1)]. 후보들의 prior probability 가 같으면, 식 (1) 의 순위는 $n$ 과 함께 커지지 않는 항을 빼고 posterior probability 의 순위와 같다.

### 3.2 Conditions

- 가정: model 이 적을 수 있는 likelihood, 독립으로 뽑힌 관측, 특이하지 않게 유지되는 Fisher information matrix, 그리고 $n$ 이 커지는 동안 고정된 parameter 개수 $k$.
- 설정값: $k$ 는 intercept 와 noise variance 를 포함하여 추정하는 parameter 를 모두 센다. $n$ 은 관측의 개수를 센다.
- 깨지는 조건: 전개가 성립하기에 너무 작은 표본, 후보가 모두 잘못 설정된 경우, parameter 개수가 정수가 아닌 penalized 적합, 그리고 관측의 개수와 독립 단위의 개수가 다른 집단 자료.
- 만나는 자리: 선택 경로에서의 부분집합 크기, mixture 의 component 개수, 시계열 model 의 lag 차수, model 기반 clustering 의 cluster 개수.

## 4. Application

BIC 는 한 data 집합에서 채점한 후보들 사이의 차이로 읽으며, 각 점수를 내는 적합은 그 후보가 어차피 한 번 해야 하는 적합이다.

### 4.1 Linear Regression Form

Gaussian error 를 가정한 linear model 에서 maximized likelihood 는 residual sum of squares 만의 함수이므로, 식 (1) 은 적합 잔차에서 바로 계산되는 형태가 된다.

```math
\mathrm{BIC} = n \ln \frac{\mathrm{RSS}}{n} + k \ln n + n (\ln 2\pi + 1) \hspace{19em} (4)
```

마지막 항은 같은 $n$ 개의 관측에 적합한 모든 후보에서 같은 값이므로 비교할 때마다 상쇄되고, 대부분의 구현이 마지막 항을 뺀 값을 내놓는다. 마지막 항을 뺀 점수와 남긴 library 의 점수는 서로 견줄 수 없다.

### 4.2 Use In Feature Selection

기준은 model 의 크기를 정하고, 후보를 만들어 내는 탐색은 따로 고른다.

- 경로: forward 또는 backward selection 경로, regularization 경로, 또는 부분집합의 명시적 목록이 크기마다 후보 하나를 낸다.
- 채점: 후보마다 한 번 적합하고 식 (1) 로 채점하여, 점수가 가장 작은 크기를 남긴다.
- 비용: 후보 하나당 적합 한 번이며, cross-validation 점수는 후보 하나당 fold 수만큼 적합한다.
- Regularized 경로: Lasso 적합의 degrees of freedom 자리에 0 이 아닌 계수의 개수를 넣으며, `LassoLarsIC` 의 점수도 같은 개수를 읽는다 [[4](#ref-4)].

### 4.3 Reading A Difference

두 후보의 BIC 차이는 두 후보가 함의하는 Bayes factor 의 log 에 2 를 곱한 값이고, Bayes factor 의 눈금이 주어진 간격의 값어치를 정한다 [[3](#ref-3)].

Table 2. What a BIC difference in favour of the smaller score is worth

| Difference | Evidence against the higher-scoring candidate |
| :--------: | :-------------------------------------------: |
| 0 to 2     | 언급할 가치를 넘지 않음                       |
| 2 to 6     | 긍정적                                        |
| 6 to 10    | 강함                                          |
| Over 10    | 매우 강함                                     |

간격이 2 미만이면 두 후보는 동률이며, parameter 가 적은 쪽을 남긴다.

## References

<a id="ref-1"></a>[1] Schwarz, G. (1978). [Estimating the Dimension of a Model](https://doi.org/10.1214/aos/1176344136). *The Annals of Statistics*, 6(2), 461–464.<br>
<a id="ref-2"></a>[2] Akaike, H. (1974). [A New Look at the Statistical Model Identification](https://doi.org/10.1109/TAC.1974.1100705). *IEEE Transactions on Automatic Control*, 19(6), 716–723.<br>
<a id="ref-3"></a>[3] Kass, R. E., & Raftery, A. E. (1995). [Bayes Factors](https://doi.org/10.1080/01621459.1995.10476572). *Journal of the American Statistical Association*, 90(430), 773–795.<br>
<a id="ref-4"></a>[4] scikit-learn developers. [Lasso model selection: AIC-BIC / cross-validation](https://scikit-learn.org/stable/auto_examples/linear_model/plot_lasso_model_selection.html?utm_source=gemini). scikit-learn examples.<br>
<a id="ref-5"></a>[5] Displayr. [Information Criteria](https://docs.displayr.com/wiki/Information_Criteria?utm_source=gemini). Displayr documentation wiki.

---

## Appendix A. Terminology

- **Bayes factor**: 같은 data 에서 두 후보 model 의 marginal likelihood 비.
- **Consistency**: Data 를 만든 model 이 후보 안에 있을 때, 표본이 커질수록 그 model 을 고를 확률이 1 로 가는 성질.
- **Deviance**: 적합이 끝난 model 의 maximized log-likelihood 에 $-2$ 를 곱한 값.
- **Efficiency**: 표본이 커질수록 고른 model 의 예측 오차가 후보 가운데 가장 좋은 것의 오차로 가는 성질.
- **Laplace approximation**: 적분 안의 log 를 최댓값 둘레에서 2 차까지 전개하고 그 Gaussian 을 적분하여 얻는 적분값.
- **Marginal likelihood**: Parameter 를 그 prior 에 대해 적분하여 없앤 model 아래에서의 data 의 likelihood.
- **Regular model**: Maximum likelihood 추정값에서 Fisher information matrix 가 특이하지 않게 유지되는 model.

## Appendix B. AIC And BIC

두 기준은 같은 deviance 로 model 을 채점하고, parameter 하나의 값이 얼마인지에서 갈린다 [[5](#ref-5)].

```math
\mathrm{AIC} = -2 \ln \hat{L} + 2k \hspace{19em} (5)
```

Table 3. The two criteria against each other

| Criterion | Penalty per parameter | Target quantity                       | Large-sample property                         | Selected size   |
| :-------: | :-------------------: | :-----------------------------------: | :-------------------------------------------: | :-------------: |
| AIC       | 2                     | 표본 밖 deviance 의 기댓값             | Efficient 하고 consistent 하지 않음            | 더 큼           |
| BIC       | $\ln n$               | Model 의 posterior probability         | 후보가 data 를 만들었으면 consistent 함        | 더 작음         |

AIC 는 적합한 model 이 같은 크기의 새 표본에서 낼 deviance 를 추정하므로 예측 기준이다 [[2](#ref-2)]. BIC 는 model 의 posterior probability 를 추정하므로 식별 기준이다 [[1](#ref-1)]. 같은 적합에 대해 서로 다른 질문에 답하므로, 두 기준이 엇갈리면 둘 중 어느 질문을 하고 있었는지로 읽는다.

Penalty 는 $n = e^2 \approx 7.4$ 에서 교차하며, $n = 8$ 부터는 BIC 가 AIC 보다 parameter 하나당 더 많이 물린다. 그래서 같은 nested 경로에서 BIC 가 고르는 크기는 AIC 가 고르는 크기보다 커지지 않는다. [Appendix C](#appendix-c-worked-example) 는 크기 차이를 설계 하나에서 잰다.

## Appendix C. Worked Example

[`src/bic_model_selection.py`](src/bic_model_selection.py) 의 class 는 어느 feature 가 자료를 만들었는지 아는 data 를 만들고, linear model 을 feature 하나씩 키우며, 크기마다 두 기준으로 채점한다. 이 folder 에서 `python3 src/bic_model_selection.py` 를 실행하면 아래 표를 출력하고, 반복 추출에서 각 기준이 생성 feature 를 되찾는 횟수를 세며, Fig 2 를 쓴다.

Data 는 독립인 standard normal feature 10 개의 관측 200 개이고, $y = 3 x_0 - 2 x_1 + 1.5 x_2 + \varepsilon$ 에 $\varepsilon \sim N(0, 1)$ 이며, seed 2 로 뽑았다. 각 단계는 residual sum of squares 를 가장 많이 낮추는 feature 를 더하고, 그 단계의 parameter 개수는 feature 개수에 intercept 와 noise variance 를 더한 값이다.

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

AIC 의 최솟값은 feature 다섯 개 자리에 있으며, 생성 feature 세 개의 595.1 에 대해 591.5 다. 더해진 두 feature 는 deviance 를 평균 3.8 씩 낮추는데, AIC 가 물리는 parameter 당 2 보다는 크고 BIC 가 물리는 $\ln 200 = 5.3$ 보다는 작다.

![Fig 2](bic_fig/fig2.png)

Fig 2. Both criteria along the forward path, with the minimum of each circled

두 곡선은 생성 feature 가 아직 빠져 있는 동안 가파르게 내려가고, 경로가 feature 세 개를 지나면서 갈린다. BIC 곡선은 곧바로 올라가고, AIC 곡선은 feature 다섯 개까지 평평하게 가다가 돌아선다.

같은 설계를 연속한 seed 로 200 번 뽑으면 BIC 는 그 가운데 166 번을 정확히 $x_0$, $x_1$, $x_2$ 에서 멈추고 AIC 는 46 번 멈춘다. 166 대 46 의 차이는 Table 3 의 consistency 를 설계 하나에서 잰 값이며, 모든 설계에 대한 주장이 아니다.
