# Predictive Uncertainty (Korean)
Rev. 0 | Created: 2026-09-10 | Updated: 2026-09-10 23:40 UTC

## 1. Purpose

- **Problem Statement**: 학습된 regression model 은 한 행마다 숫자 하나를 내놓으며, 그 숫자에는 실제 결과가 그것에서 얼마나 벗어날 수 있는지에 대한 진술이 없다.
- **Goal**: 예측값에 확률을 붙이는 방법을 taxonomy 와 hierarchy 로 제시하여, 읽는 이가 자기 model 이 닿은 level 과 그 다음 level 로 올라가는 값을 스스로 댈 수 있게 한다.
- **Non-Goal**: Classifier 가 내놓는 확률과 그 확률의 calibration 은 다루지 않는다.
- **Non-Goal**: 점 예측의 정확도를 높이는 일은 다루지 않는다.

## 2. Summary

무슨 확률을 보고할 수 있는지는 loss 와 wrapper 가 정하며, model 계열이 정하지 않는다. Ridge 와 LightGBM 은 그대로 두면 둘 다 맨 숫자 하나를 내놓고, loss 를 pinball loss 로 바꾸면 둘 다 quantile 에 닿으며, calibration split 을 쓰면 둘 다 coverage 를 명시한 interval 에 닿는다. Table 1 이 방법을 놓는 세 축이고, 그 가운데 첫째 축이 나머지를 정렬한다.

그 첫째 축은 Fig 1 에 그린 네 level 의 hierarchy 이며, 각 level 은 아래 level 보다 엄격히 더 많은 질문에 답한다. Point 는 무엇을 기대할지에 답한다. Interval 은 결과가 한 구간 안에 드는지에 답한다. Quantile function 은 같은 질문에, 폭이 입력에 따라 움직이는 구간으로 답한다. Distribution 은 결과가 임의로 제시된 한계를 넘을 확률에 답하며, 이것이 specification limit 이 묻는 질문이다.

답의 형태와 답의 옳음을 갈라놓는 실패가 둘 있고, Appendix B 가 같은 600 개의 held-out 행에서 둘을 모두 측정한다. 폭이 고정된 interval 은 전체 행의 0.908 을 덮으면서 가장 조용한 3분의 1 은 1.000, 가장 시끄러운 3분의 1 은 0.785 를 덮으므로, 전체 행에서 잰 coverage 는 쉬운 행을 과도하게 덮고 어려운 행을 덜 덮어서 달성된다. 폭이 입력에 따라 실제로 움직이는 quantile boosting 은 nominal 0.90 에 대해 0.822 에서 0.832 를 덮으므로, 폭이 적응한다는 것이 그 폭이 충분히 넓다는 보장은 아니다.

Section 5 와 section 6 은 linear 계열과 tree 계열에 대해 각 library 가 무엇을 내놓고 어느 parameter 가 그것을 바꾸는지 적는다. Section 7 은 coverage 보장을 붙이는 유일한 경로이며, 그 보장을 계열에 상관없이 붙인다. Section 8 은 보고된 interval 을 무엇으로 판정하는지이고, section 9 는 어느 방법을 쓸지이다.

## 3. Taxonomy

### 3.1 Three Axes

예측값에 확률을 붙이는 방법은 세 질문으로 자리가 정해지며, 세 질문은 서로 독립으로 답해진다. Table 1 이 그 셋이다.

Table 1. The three axes of a predictive uncertainty method

| Axis | Question | Values |
|------|----------|--------|
| Form | 한 행마다 model 이 내놓는 것 | Point, interval, quantile function, distribution |
| Source | 내놓은 폭이 무엇으로 이루어졌는지 | 결과의 noise, fit 에 대한 무지 |
| Route | 그 출력을 얻는 방법 | Native, loss swap, resampling, calibration wrapper |

Form 축이 나머지 두 축을 정렬하는데, 어떤 질문을 아예 할 수 있는지를 그 축이 정하기 때문이다. Section 3.2 가 그 축을 hierarchy 로 펼치고, 남은 두 축이 내놓은 숫자가 스스로 주장하는 뜻을 실제로 갖는지를 말한다.

### 3.2 Hierarchy Of The Answer

Fig 1 은 Table 1 의 Form 축을 펼친 것이다.

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

네 level 은 그것이 답하는 질문으로 정렬되며, Table 2 가 그 정렬이다. 각 level 은 아래 level 을 품는다. Interval 은 고정된 두 level 의 quantile 두 개이고, quantile function 은 CDF 의 역함수를 고른 지점에서 읽은 것이며, distribution 은 그 역함수를 어디서든 읽은 것이다.

Table 2. What each level of Fig 1 answers

| Level | Emits | Answers | Needs |
|-------|-------|---------|-------|
| 1. Point | 숫자 하나 | 무엇을 기대할지 | Fit 외에 아무것도 |
| 2. Interval | 한 level 에서의 숫자 둘 | 결과가 한 구간 안에 드는지 | Residual 의 폭, 또는 calibration split |
| 3. Quantile function | 요청한 level 마다 경계 하나 | 같은 질문에, 입력에 따라 움직이는 구간으로 | Level 당 한 번의 fit, 또는 level vector 를 받는 한 번의 fit |
| 4. Distribution | Density 또는 CDF | 임의로 제시된 한계를 넘을 확률, 그리고 결과의 임의의 moment 와 quantile | Distribution family, 또는 촘촘한 level grid |

Specification limit 이 필요로 하는 것은 level 4 이다. 한계는 model 을 학습한 뒤에 오고 예고 없이 옮겨지므로, 답은 한 level 에서 학습된 경계가 아니라 어떤 한계든 대입할 수 있는 함수여야 한다.

### 3.3 Source Of The Spread

내놓은 폭은 데이터가 늘어날 때 서로 다르게 움직이는 두 부분으로 이루어지며, predictive variance 는 정확히 그 둘로 갈라진다. Parameter $\theta$ 와 입력 $x$ 에 대해 law of total variance 가 그 분해를 준다 [[7](#ref-7)].

$$\operatorname{Var}[y \mid x] = \mathbb{E}_{\theta}\left[\operatorname{Var}(y \mid x, \theta)\right] + \operatorname{Var}_{\theta}\left(\mathbb{E}[y \mid x, \theta]\right) \hspace{19em} (1)$$

첫째 항은 결과의 noise 이고 둘째 항은 fit 에 대한 무지이다. 세 가지 결과가 어느 방법을 무엇에 쓸 수 있는지를 정한다.

- 첫째 항, 행이 늘어도 그대로. 입력이 고정된 자리에서는 줄일 수 없음.
- 둘째 항, 행이 쌓이면 줄어듦. Fit 이 알려진 극한에서 0.
- 미래 결과에 대한 interval, 두 항이 모두 필요. 학습된 평균에 대한 interval, 둘째 항만 필요.

이 구별이, 둘 다 폭처럼 보이는 두 방법을 갈라놓는다. Table 3 이 각 경로가 어느 항을 재는지이고, 마지막 열은 다른 항이 정작 중요한 항일 때 무슨 일이 생기는지이다.

Table 3. Which term of equation (1) each route measures

| Route | Term measured | Width moves with the input | What it misses |
|-------|---------------|----------------------------|----------------|
| Residual standard deviation | 첫째 항, 상수 하나로 | 아니오 | 입력 공간에 걸친 noise 의 변화 |
| BayesianRidge `return_std` | 두 항 모두, 첫째 항은 상수 하나로 | 둘째 항을 통해서만 | 입력 공간에 걸친 noise 의 변화 |
| Spread across the trees of a forest | 둘째 항 | 예 | Leaf 가 한 행만 담은 자리를 빼면 noise |
| Pinball loss at two levels | 두 항 모두, 학습된 대로 | 예 | 학습된 경계가 충분히 넓다는 보장 |
| Leaf observations of a forest | 두 항 모두 | 예 | 같은 보장 |
| Split conformal, conformalized quantile regression | 두 항 모두, calibration residual 을 통해 | Quantile 형태에서만 | Conditional coverage, 보장되지 않음 |

Table 3 의 둘째 행이 이 절이 이름을 붙이려 존재하는 함정이다. Gaussian 평균과 표준편차는 형태로는 level 4 의 답이고 BayesianRidge 가 그것을 내놓지만, 그 noise 항은 학습된 상수 하나이다. Appendix B 는 test 행 전체에서 표준편차가 1.889 에서 1.901 사이임을 재는데, 같은 행에서 구성상의 noise 표준편차는 0.51 에서 5.25 까지 움직이므로, 내놓은 density 는 거의 모든 자리에서 모양은 맞고 폭은 틀리다.

셋째 행은 같은 함정을 반대로 읽은 것이다. Forest 의 tree 들은 평균을 두고 서로 어긋나며, 그 어긋남은 equation (1) 의 둘째 항뿐이다. Appendix B 는 기본 forest 의 tree spread 에서 0.858 의 coverage 를, 각 leaf 가 최소 20 행을 담게 하면 0.650 을 재므로, 앞의 숫자는 predictive distribution 에서 온 것이 아니라 tree 하나의 예측이 noise 섞인 관측 하나에 가까울 만큼 leaf 가 작았던 데서 온다.

### 3.4 Route

네 경로가 Fig 1 의 level 에 닿으며, 무엇을 지불하는지가 서로 다르다. Table 4 가 그 넷을 나란히 놓는다.

Table 4. The four routes of Table 1

| Route | Mechanism | Spends | Reaches |
|-------|-----------|--------|---------|
| Native | Model 을 처음부터 distribution 으로 학습 | Distribution 에 대한 가정 | Level 4 |
| Loss swap | Squared-error loss 를 equation (2) 의 pinball loss 로 교체 | Level 당 한 번의 fit, library 가 vector 를 받지 않는 한 | Level 3 |
| Resampling | Ensemble 의 member, 또는 그 leaf 안의 행을 표본으로 읽음 | 학습된 ensemble 외에 없음 | Level 2 또는 level 3 |
| Calibration wrapper | 이미 학습된 model 의 경계를 held-out score 의 quantile 만큼 이동 | Calibration split | Level 2 또는 level 3, 보장과 함께 |

Loss swap 이 가장 많은 계열에 닿는 경로인데, model 을 바꾸는 것이 아니라 fit 이 conditional distribution 의 어느 functional 을 추정하는지를 바꾸기 때문이다. Level $\alpha$ 의 pinball loss 는 $\alpha$ quantile 에서 최소가 된다 [[1](#ref-1)].

$$L_{\alpha}(y, q) = \max\left\{\alpha\,(y - q),\ (\alpha - 1)(y - q)\right\} \hspace{19em} (2)$$

Equation (2) 는 Fig 1 의 level 1 행들도 설명한다. $\alpha = 0.5$ 에서 이 식은 상수배를 빼면 absolute-error loss 이므로, absolute error 로 학습한 fit 은 이미 quantile 하나인 median 을 내놓고 있다. Squared-error loss 는 대신 평균을 내놓고, Huber 와 epsilon-insensitive loss 는 둘 다 아닌 중심을 내놓는데, 그래서 robust 한 점 추정은 자기가 무엇에 robust 한지 그 폭에 대해 아무 말도 하지 않는다.

## 4. Form Against Correctness

한 방법이 Fig 1 에서 닿은 level 과 그것이 내놓은 값의 옳음은 서로 독립이며, 한쪽이 높으면서 다른 쪽이 틀릴 수 있다. Table 5 가 그 네 조합이고, 각 자리를 차지하는 Appendix B 의 행을 함께 적었다.

Table 5. Level reached against coverage delivered

| Level reached | Coverage close to nominal | Coverage far from nominal |
|---------------|---------------------------|---------------------------|
| Level 2 | Residual 표준편차를 쓴 `Ridge`, 0.912 | Leaf 당 20 행인 forest tree spread, 0.650 |
| Level 3 또는 4 | Conformalized quantile regression, 0.917 | LightGBM quantile, 0.815 |

오른쪽 열을 채우는 실패는 둘이고 원인이 서로 다르다. 학습된 quantile 은 훈련 행에서 equation (2) 를 최소화한 것이며, 그 최소화에는 held-out 빈도가 level 과 맞도록 강제하는 항이 없으므로, 규제가 걸리거나 덜 학습된 fit 은 참 경계보다 안쪽에 내려앉는다. Resampling 의 폭은 section 3.3 이 적은 대로 equation (1) 의 틀린 항을 잰다.

틀리는 길은 둘이고 Table 5 의 왼쪽 열은 그 가운데 첫째만 답한다. `BayesianRidge` 는 level 4 로 coverage 0.908 을 내며 그 열에 앉아 있고, 그 폭은 noise 가 움직이는 자리에서 가만히 있는데, 이것이 section 8 이 재는 1.000 대 0.785 의 conditional coverage 로 드러난다.

왼쪽 열에는 서로 바꿔 쓸 수 없는 두 가지 방법으로 닿는다. 고정된 폭은 숫자 하나를 골라 전체 행의 0.90 을 덮게 만들 수 있고, 그러면 noise 가 작은 자리는 과도하게 덮고 큰 자리는 덜 덮는다. 입력에 따라 움직이는 폭은 전체 행의 0.90 을, 그리고 각 band 의 0.90 도 함께 덮을 수 있다. Section 8 이 그 둘을 갈라내는 측정이고, coverage 를 결코 혼자 보고하지 않는 이유이다.

## 5. Linear Family

Linear regressor 가운데 둘은 자기 폭을 내놓고 나머지는 맨 숫자를 내놓으므로, 나머지의 Fig 1 상 level 은 전적으로 Table 4 의 경로가 정한다. Table 6 이 각각이 무엇을 내놓고 어떻게 level 3 에 닿는지를 적는다.

Table 6. What the linear regressors emit

| Estimator | `predict` signature | Level as fitted | Functional estimated | Route to level 3 |
|-----------|---------------------|-----------------|----------------------|------------------|
| `Ridge` | `predict(X)` | 1 | Conditional mean | `QuantileRegressor`, 또는 section 7 의 wrapper |
| `Lasso` | `predict(X)` | 1 | Conditional mean | 같음 |
| `ElasticNet` | `predict(X)` | 1 | Conditional mean | 같음 |
| `HuberRegressor` | `predict(X)` | 1 | Robust 한 중심 | 같음 |
| `LinearSVR` | `predict(X)` | 1 | Epsilon-insensitive tube 의 중심 | 같음 |
| `QuantileRegressor` | `predict(X)` | 3, fit 당 level 하나 | Conditional quantile | 이미 도달 |
| `BayesianRidge` | `predict(X, return_std=False)` | 4, Gaussian | Conditional mean 과 variance | 이미 그 위, 폭은 Table 3 대로 |
| `ARDRegression` | `predict(X, return_std=False)` | 4, Gaussian | 같음 | 같음 |

`QuantileRegressor` 는 `Lasso` 처럼 L1 penalty 를 가지며 그 `alpha` 의 기본값이 1.0 이므로, 규제 없는 quantile fit 에는 `alpha=0` 을 명시해야 한다. `quantile` 인자가 level 이고 `alpha` 가 penalty 인데, 이는 section 6 의 gradient boosting library 들과 이름이 반대로 쓰인 것이다.

Bayesian 계열의 두 행이 variance 를 내놓는 것은 evidence framework 아래에서 noise 와 coefficient 를 함께 학습하기 때문이다 [[6](#ref-6)]. 내놓는 variance 는 학습된 noise 항과, 입력이 학습 데이터에서 멀어질수록 커지는 항의 합이다.

$$\operatorname{Var}[y \mid x] = \sigma^{2} + x^{\top} \Sigma\, x \hspace{19em} (3)$$

Equation (3) 은 linear fit 에 대해 equation (1) 의 둘째 항을 드러낸 것이며, 내놓는 폭이 어디서 움직일 수 있고 어디서 못 움직이는지를 말한다. 첫째 항은 scalar 하나이므로 움직임은 전부 둘째 항에서 오고, 그 항은 fit 이 잘 정해진 자리에서는 작다. Appendix B 는 그 폭이 600 행에 걸쳐 0.012 만큼 움직이는 것을 재며, 같은 행에서 참 noise scale 은 열 배로 변한다.

## 6. Tree And Ensemble Family

흔히 쓰는 네 tree model 은 모두 pinball loss 를 통해 level 3 에 닿으며, 두 번째 fit 없이 닿는 것은 random forest 뿐이다. Table 7 이 각 library 에서 바뀌는 parameter 이다.

Table 7. How the tree models emit a quantile

| Estimator | Parameter | Levels per fit | Note |
|-----------|-----------|----------------|------|
| `GradientBoostingRegressor` | `loss="quantile"`, `alpha` | 하나 | `loss` 는 `squared_error`, `absolute_error`, `huber` 도 받음 |
| `HistGradientBoostingRegressor` | `loss="quantile"`, `quantile` | 하나 | `loss` 는 `squared_error`, `absolute_error`, `poisson`, `gamma` 도 받음 |
| `LGBMRegressor` | `objective="quantile"`, `alpha` | 하나 | 여기서 `alpha` 는 level 이며 penalty 가 아님 |
| `XGBRegressor` | `objective="reg:quantileerror"`, `quantile_alpha` | 여럿 | `quantile_alpha` 는 array 를 받고 `predict` 는 level 마다 열 하나를 돌려줌 |
| `RandomForestRegressor` | 없음 | 전부 | Table 4 의 resampling 경로로, section 6.1 |

Level 당 한 번의 fit 이 규모를 정하는 값이다. `LGBMRegressor` 로 열 개 level 을 답하려면 학습하고 저장하고 serving 할 model 이 열 개이며, `XGBRegressor` 는 열 개 level 을 array 로 받아 하나의 booster 에서 열 개 열의 행렬을 돌려준다. Appendix B 가 두 level fit 에서 그 shape 을 `(600, 2)` 로 확인한다.

따로 학습된 level 은 서로 교차할 수도 있다. 각 fit 은 자기 level 에서 equation (2) 를 최소화하며 다른 fit 과 묶어 주는 항이 없으므로, 어떤 행에서는 0.05 경계가 0.95 경계보다 위에 올 수 있고, 그 교차는 쓰기 전에 내놓은 열들을 정렬하여 고쳐야 한다.

### 6.1 The Forest Without A Second Fit

Random forest 는 학습에 쓴 행을 leaf 로 갈라 이미 담고 있으므로, loss 를 바꾸지 않고 quantile 을 읽어 낸다. 한 입력에 대해, 모든 tree 에서 그 입력이 떨어지는 leaf 안의 행들이 결과의 가중 표본을 이루며, 그 표본의 어떤 quantile 이든 level 3 의 답이다 [[2](#ref-2)].

Leaf 안의 행 대신 tree 의 예측을 거치는 경로가 실패하는 쪽이고, 이유는 section 3.3 이 준다. Appendix B 가 같은 forest 에서 둘을 모두 잰다. Leaf 의 행은 0.893 의 coverage 를 폭 2.89 의 움직임과 함께 주고, tree spread 는 leaf 가 평균을 낼 만큼 커지면 0.650 을 준다.

Leaf 경로의 값은 fit 이 아니라 저장이다. 학습에 쓴 행을 남겨 두고 그 leaf 소속을 색인해야 하며, 예측 한 번이 tree 당 숫자 하나가 아니라 모든 tree 의 leaf pool 을 건드리는데, 그래서 library 경로가 학습된 forest 위의 호출이 아니라 별도의 구현으로 되어 있다.

## 7. Model-Agnostic Route

Conformal prediction 은 coverage 보장을 붙이는 유일한 경로이며, 어느 계열의 학습된 model 에든 다시 학습하지 않고 붙는다. Fit 에서 떼어 둔 calibration split 을 채점하고, 그 score 의 quantile 하나가 내놓은 경계를 넓히는데, 그 넓힘의 크기가 exchangeability 아래에서 유한 표본에서도 coverage 가 성립하게 만든다 [[3](#ref-3)] [[8](#ref-8)].

점 예측 model 에서는 score 가 calibration 행의 절대 residual 이고 보정은 숫자 하나, 즉 $n$ 개 score 가운데 $\lceil (n+1)(1-\alpha) \rceil$ 번째로 작은 값이며, $\alpha = 0.10$ 에서 600 개 calibration score 가운데 541 번째이다.

$$\hat q = s_{\left(\left\lceil (n+1)(1-\alpha) \right\rceil\right)} \hspace{19em} (4)$$

Interval 은 예측에 $\hat q$ 를 더하고 뺀 것이며, 이는 모든 행에 폭 하나이다. Appendix B 는 $\hat q = 3.087$ 과 nominal 0.90 에 대한 0.908 의 coverage 를, 그리고 고정된 폭이 강제하는 1.000 과 0.785 의 conditional coverage 를 잰다.

Conformalized quantile regression (CQR) 은 점 대신 경계를 conformalize 하여 폭이 계속 움직이게 둔다 [[4](#ref-4)]. Score 는 calibration 결과가 학습된 경계 밖으로 얼마나 벗어났는지이며, 안쪽에 들었으면 음수이다.

$$E_{i} = \max\left\{\hat q_{\alpha/2}(x_{i}) - y_{i},\ y_{i} - \hat q_{1-\alpha/2}(x_{i})\right\} \hspace{19em} (5)$$

그 score 에 equation (4) 를 적용하면 보정 하나가 나오고, 그것을 위 경계에 더하고 아래 경계에서 뺀다. Appendix B 는 0.483 의 보정을 재는데, 이것이 gradient boosting quantile 의 coverage 를 0.832 에서 0.917 로 올리는 동안 폭은 2.68 로 계속 움직이므로, 보장과 적응이 함께 유지된다.

Calibration 행과 예측할 행 사이의 exchangeability 가 이 모든 것이 놓인 조건이며, drift 하는 공정이 깨뜨리는 조건이 바로 그것이다. 보장은 또한 marginal 이다. 행을 뽑는 과정 전체에 대해 성립하고 그 안의 band 마다 성립하지 않는데, 그래서 Appendix B 가 band 별 coverage 도 함께 보고한다.

두 계열은 level 4 에 native 로 닿으며, wrapper 의 대안으로서 여기에 속한다. Gaussian process 는 학습에 쓴 kernel 에서 평균과 variance 를 내놓고, 그 값은 행 수의 세제곱으로 커진다. NGBoost 는 고른 distribution 의 parameter 들을 natural gradient 아래에서 함께 boosting 하여, tree model 에서 한 행마다의 density 를 준다 [[5](#ref-5)]. 둘 다 이 절의 보장을 distribution 에 대한 가정으로 바꿔 놓는다.

## 8. Calibration

전체 행에서 잰 coverage 는 조용한 행을 과도하게 덮고 시끄러운 행을 덜 덮어서 달성되므로, interval 은 세 숫자를 함께 놓고 판정한다. Table 8 이 그 셋이고, 각각이 잡아내는 실패를 함께 적었다.

Table 8. What an emitted interval is judged by

| Measure | Definition | Catches |
|---------|------------|---------|
| Marginal coverage | Interval 안에 든 전체 행의 비율 | 전체적으로 너무 좁거나 너무 넓은 경계 |
| Conditional coverage | 같은 값을, 입력이나 내놓은 폭으로 묶은 band 안에서 | 변하는 폭 자리를 고정된 폭이 대신하는 것 |
| Sharpness | Interval 의 평균 폭 | 쓸모없이 넓혀서 덮게 만든 경계 |

Coverage 와 sharpness 는 서로에 대해 읽는데, 어느 하나만이면 무의미하게 충족되기 때문이다. 규칙은 calibration 을 조건으로 두고 sharpness 를 최대화하는 것이다. Coverage 가 성립하는 방법들 가운데 interval 이 가장 좁은 것이 이긴다 [[10](#ref-10)]. Appendix B 가 그 비교로 짜여 있으며, 폭이 고정된 두 행은 coverage 에도 불구하고 그 비교에서 진다.

둘을 한꺼번에 채점하는 숫자 하나가 proper scoring rule 이며, 참 distribution 에서만 그 기대값이 최적이 되는 점수이다 [[9](#ref-9)]. 그 가운데 둘이 Fig 1 의 level 들을 덮는다.

- Equation (2) 의 pinball loss, quantile 하나에 대해 proper. `mean_pinball_loss` 와 `d2_pinball_score` 로 제공.
- Continuous ranked probability score, distribution 전체에 대해 proper. 내놓은 CDF 와 결과에서의 step function 사이 차이의 제곱을 적분한 값.

어느 것도 coverage 표를 대신하지 않는다. Proper score 는 방법들을 서로 견주어 줄 세우면서 그 가운데 최고가 옳은지는 말하지 않으며, 0.65 의 coverage 는 어떤 줄 세우기도 보고하지 않는 그 interval 에 대한 사실이다.

## 9. Selection

선택은 두 질문에서 따라 나온다. 답이 Fig 1 의 어느 level 에 닿아야 하는지, 그리고 calibration split 을 뗄 수 있는지이다. Table 9 는 왼쪽 열에서 읽는다.

Table 9. Which method to use

| Use | When | Why |
|-----|------|-----|
| Residual standard deviation | 거친 interval 로 족하고 noise 가 입력에 걸쳐 고르다고 알려진 경우 | 이미 한 fit 위에 숫자 하나 |
| `BayesianRidge` or `ARDRegression` | Linear fit 이고 level 4 의 density 를 원하는 경우 | Native 이고 split 을 쓰지 않음 |
| Gaussian process | 행이 적고 응답이 매끄러우며 density 를 원하는 경우 | Native 이고 데이터가 얇아지는 자리에서 폭이 커짐 |
| Pinball loss in the model's own library | Noise 가 입력에 따라 변하고 level 이 미리 정해진 경우 | Level 당 한 번의 fit 을 값으로 치른 level 3 |
| Leaf observations of a forest | Forest 가 이미 학습되어 있고 level 을 둘 이상 원하는 경우 | 한 번의 fit 에서 모든 level |
| NGBoost | Tree model 이고 level 4 의 density 를 원하는 경우 | Distribution family 를 값으로 치른, tree 에서의 level 4 |
| Split conformal | Coverage 보장이 필요하고 내놓는 폭이 고정되어도 되는 경우 | 학습된 어떤 model 위에서든 유한 표본의 marginal coverage |
| Conformalized quantile regression | Coverage 보장이 필요하고 폭이 입력에 따라 움직여야 하는 경우 | 둘을 함께 지키는 유일한 행 |

Appendix B 가 Table 9 의 모든 행을 한 split 에서 서로 견주어 재며, section 6 의 네 library 밖의 library 를 필요로 하는 Gaussian process 와 NGBoost 행만 뺀다.

## 10. Further Work

- **Coverage under drift** — Section 7 의 보장은 calibration 행과 예측할 행 사이의 exchangeability 에 놓여 있는데, distribution 이 움직이는 공정은 그것을 소리 없이 깨뜨린다. 내놓는 폭이 nominal 이라는 이름표를 그대로 달고 있기 때문이다. Adaptive 와 weighted conformal 방법은 equation (4) 의 고정된 quantile 을 최근 coverage 오차에서 다시 추정한 값으로 바꾸어, 행마다가 아니라 장기 평균의 뜻으로 보장을 되살린다. 결과가 유한한 지연 안에 도착해야 하고 보정이 얼마나 빨리 움직여도 되는지에 대한 결정이 필요한데, noise 를 따라다니는 보정은 고정된 것보다 나쁘기 때문이다.
- **Distributional boosting as the default level 4** — 이 문서에서 tree 계열이 level 4 에 닿는 길은 따로 학습한 quantile 을 촘촘한 grid 로 두어 level 당 한 번의 fit 을 치르는 것, 또는 NGBoost 의 distribution 가정을 받는 것 둘이다. Distribution 의 parameter 들을 함께 boosting 하는 것을 안정하게 만든 것이 natural gradient 이며, 같은 처리가 이제 주류 boosting library 안에서 distributional loss 로 제공된다. 학습된 model 의 residual 을 먼저 살펴야 하는데, 내놓는 density 가 물려받는 것이 family 의 선택이고, 틀린 family 는 level 2 만큼의 정보가 담긴 level 4 의 답이기 때문이다.

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

- **aleatoric uncertainty**: 결과의 noise, equation (1) 의 첫째 항.
- **calibration split**: Fit 에서 떼어 두고 conformal 보정의 크기를 정하는 데만 쓰는 행.
- **conditional coverage**: 전체 행이 아니라 행의 한 band 안에서 잰 interval 의 coverage.
- **conformal prediction**: 학습된 어떤 model 의 경계든 held-out score 의 quantile 만큼 넓혀 유한 표본에서 coverage 가 성립하게 하는 절차.
- **continuous ranked probability score**: 내놓은 CDF 와 결과에서의 step function 사이 차이의 제곱을 적분한 값.
- **coverage**: 결과가 내놓은 interval 안에 든 행의 비율.
- **epistemic uncertainty**: Fit 에 대한 무지, equation (1) 의 둘째 항.
- **exchangeability**: 행의 순서를 바꾸어도 그 결합 분포가 변하지 않는 성질.
- **leaf observations**: 입력이 떨어지는 leaf 로 갈라 담긴 학습 행을, 결과의 표본으로 읽은 것.
- **marginal coverage**: 전체 행에 대해 잰 interval 의 coverage.
- **pinball loss**: Equation (2) 의 비대칭 절대 손실로, quantile 하나에서 최소가 된다.
- **prediction interval**: 미래 결과가 명시된 level 로 그 사이에 든다고 주장하는 두 숫자.
- **predictive distribution**: 한 입력에서의 결과에 대한 density 또는 CDF.
- **proper scoring rule**: 참 distribution 에서만 그 기대값이 최적이 되는 점수.
- **quantile function**: Distribution 의 level 을 입력의 함수로 읽은 것.
- **sharpness**: 내놓은 interval 이나 density 의 좁음이며, coverage 가 성립하는 방법들 사이에서만 판정한다.
- **split conformal**: 보정이 calibration split 의 절대 residual 의 quantile 하나인 conformal prediction.

## Appendix B. Worked Example

이 appendix 의 모든 숫자는 하나의 데이터와 한 벌의 split 에서 나온다. Noise 는 첫 feature 에 따라 넓어지도록 만들었으며, 그래야 폭이 움직이지 못하는 방법이 평균적으로 틀리지 않으면서도 눈에 보이게 틀린다.

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

Split 은 학습에 쓸 1800 행, section 7 의 wrapper 를 calibration 할 600 행, 측정할 600 행이다. 모든 interval 은 0.05 와 0.95 level 로 요청하므로 nominal coverage 는 전부 0.90 이다. Boosting model 셋에는 같은 300 그루의 tree, 같은 0.05 의 learning rate, 같은 3 의 depth 를 주어, 그들 사이의 비교가 기본값의 비교가 아니라 objective 의 비교가 되게 했다.

그 level 에서의 참 폭은 구성에서 $2 \times 1.645 \times (0.5 + 1.5\,|x_{1}|)$ 로 따라 나온다. $|x_{1}|$ 로 정렬한 test 행의 세 band 에 걸쳐 평균하면 그 폭은 2.85, 5.21, 9.28 이므로 어느 숫자 하나도 세 band 를 다 맡지 못한다. Table 10 의 두 band 열은 그 band 들의 바깥 둘이다.

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

Table 10 에서 다섯 가지가 읽힌다.

- Coverage 가 0.90 에 가깝고 폭의 표준편차가 0.01 이하인 세 행, 그리고 1.000 대 0.785 의 conditional coverage. Band 를 서로 상쇄시켜 달성한 marginal coverage.
- 600 행에 걸쳐 1.889 에서 1.901 사이의 표준편차를 내놓는 `BayesianRidge`. Equation (3) 이 예측하는 대로, 참값이 열 배로 움직이는 자리에서 0.012 만큼 움직이는 level 4 의 형태.
- 폭의 표준편차가 2.68 이면서 0.815 에서 0.832 에 있는 quantile boosting 세 행. 보장 없는 적응.
- Leaf 당 한 행에서 0.858, 스물에서 0.650 인 tree spread. 앞의 숫자가 predictive distribution 이 아니라 평균을 내기에 너무 작은 leaf 에서 온 것.
- 폭의 표준편차 2.68 로 0.917 인 마지막 행. Section 7 의 보장과 section 6 의 적응이 함께 유지된 것이며, 값은 calibration 에 쓴 600 행.

Linear 계열에서 조금이라도 적응하는 행은 하나뿐이고, 그 폭의 표준편차 0.49 가 입력의 함수인 noise scale 을 상대로 직선이 할 수 있는 한계이다. `QuantileRegressor` 는 linear quantile 을 학습하므로 두 경계가 서로 기울어 벌어질 수는 있어도 휘지는 못하는데, 그래서 그 conditional coverage 가 적응하지 않는 행들처럼 1.000 과 0.770 에 있다.

### B.1 Level 3 In Four Libraries

같은 데이터 위에서 본 Table 7 의 parameter 이름들이다.

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

`alpha` 인자는 첫 block 에서 L1 penalty 를 뜻하고 셋째 block 에서 level 을 뜻하는데, 이것이 section 5 가 적은 이름 충돌이다.

### B.2 The Two Forest Routes

같은 학습된 forest 를 두 번 읽는다.

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

경로 1 은 기본 forest 에서 0.858 의 coverage 에 닿고, `min_samples_leaf=20` 으로 각 leaf 가 최소 스무 행을 담게 하면 0.650 에 닿는다. 경로 2 는 같은 forest 에서 0.893 에 닿으며 학습 결과를 memory 에 남겨 두어야 한다.

### B.3 The Two Wrappers

600 개의 calibration 행 위에서 본 equation (4) 와 equation (5) 이다.

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

보정은 3.087 과 0.483 이다. 앞의 것이 큰 까닭은 점 예측 model 의 폭을 통째로 져야 하기 때문이고, 뒤의 것이 작은 까닭은 학습된 quantile 이 그 폭의 대부분을 이미 지고 있어 level 에만 조금 못 미치기 때문이다.
