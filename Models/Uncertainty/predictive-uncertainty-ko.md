# Predictive Uncertainty (Korean)
Rev. 0 | Created: 2026-09-10 | Updated: 2026-09-11 00:30 UTC

## 1. Purpose

- **Problem Statement**: 학습된 regression model의 예측 에 대한 확률을 구하여 ensemble 모델등에 활용하고 싶다.
- **Goal**: 예측 값의 확률의 taxonomy 와 hierarchy를 통한 방법을 제시한다.
- **Non-Goal**: classifier의 결과를 다루지 않는다.

## 2. Summary

학습된 regression model 에서 확률을 얻는 구현은 셋이며, Table 1 이 그 선택이다. Probabilistic regression 은 한 행마다 평균과 분산을 학습하여 density 를 돌려준다. Quantile regression 은 요청한 level 마다 경계를 학습하여 interval 을 돌려준다. Bootstrap 과 resampling 은 다시 학습한 member 들의 어긋남을 폭으로 읽는다.

셋이 내놓는 것은 같지 않으며, Table 2 가 그것을 네 level 로 정렬한다. Point 는 무엇을 기대할지에, interval 은 결과가 한 구간 안에 드는지에, quantile 의 집합은 같은 질문에 폭이 입력에 따라 움직이는 구간으로, density 는 결과가 임의의 한계를 넘을 확률에 답한다. Problem Statement 가 ensemble 에 대해 묻는 질문에 답하는 것은 마지막 level 뿐인데, 행마다의 weight 는 행마다의 분산의 함수이기 때문이다.

셋 가운데 둘은 보통 쓰이는 방식대로 쓰면 엉뚱한 양을 잰다. 점 예측 model 을 bootstrap 하면 학습된 평균이 얼마나 움직이는지를 재지 결과가 얼마나 움직이는지를 재지 않으며, Appendix B 는 그 coverage 를 nominal 0.90 에 대해 0.115 로 잰다. Forest 의 tree spread 도 같은 결함을 지니는데, 기본 설정에서는 leaf 가 한 행을 담아 tree spread 가 우연히 noise 를 닮으므로 그 결함이 가려진다.

Section 10 이 Problem Statement 에 대한 보답이다. Feature 구성이 서로 달라 각각 다른 자리에서 무지한 두 member 를 equal weight 로 묶으면 RMSE 2.3910 이고, 각자의 행마다의 분산으로 만든 weight 로 묶으면 2.1689 이며, member 를 따로 쓰면 2.0643 과 3.0808 이다.

## 3. Taxonomy

Table 1 이 세 구현이며, 각각이 무엇을 내놓고 언제 쓰이는지를 함께 적었다.

Table 1. The three implementations

| Implementation | Emits | Recommended when |
|----------------|-------|------------------|
| Probabilistic regression | 한 행마다 평균 $\mu$ 와 표준편차 $\sigma$ | 제시된 한계를 넘을 확률이 필요하고 distribution family 를 가정할 수 있을 때 |
| Quantile regression | 요청한 level 마다 경계 하나, 예를 들어 0.10, 0.50, 0.90 | 미리 정한 level 의 interval 이 필요하고 distribution 을 가정하지 않을 때 |
| Bootstrap and resampling | 다시 학습한 member 들의 예측의 폭 | Member 가 이미 있고, 알고 싶은 것이 fit 자체가 얼마나 의심스러운지일 때 |

Fig 1 은 같은 셋을, library 에서 각각이 지나는 경로와 함께 놓은 것이다.

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

셋째 가지는 무엇의 폭인가에서 앞의 둘과 다르다. 앞의 둘은 결과의 폭을 학습하고, 셋째는 model 의 폭을 재며, 거기서 따라 나오는 것이 section 7 이다.

## 4. Hierarchy

Table 1 의 세 구현이 답하는 질문은 같지 않으며, Table 2 가 그 답을 네 level 로 정렬한다. 각 level 은 아래 level 을 품으므로, 위 level 에 닿는 방법은 아래 level 에도 닿는다.

Table 2. What each level answers

| Level | Emits | Answers | Reached by |
|-------|-------|---------|------------|
| 1. Point | 숫자 하나 | 무엇을 기대할지 | 학습된 모든 regression model |
| 2. Interval | 한 level 에서의 숫자 둘 | 결과가 한 구간 안에 드는지 | Residual 표준편차, 또는 더 높은 level 의 모든 방법 |
| 3. Quantile set | 요청한 level 마다 경계 하나 | 같은 질문에, 입력에 따라 움직이는 구간으로 | Quantile regression, leaf observations |
| 4. Distribution | Density 와 CDF 전체 | 임의의 한계를 넘을 확률, 그리고 임의의 moment 와 quantile | Probabilistic regression, 또는 촘촘한 quantile grid |

Problem Statement 가 필요로 하는 것은 level 4 이다. Ensemble weight 는 행마다의 숫자여야 하므로 행마다의 분산에서 나와야 하고, 그 분산을 내놓는 것이 level 4 이며 level 3 은 그것을 경계 한 쌍 안에 묻어 둔다. Model 을 학습할 때 알지 못했던 한계에 대해 "이 예측이 그 한계를 넘을 확률은 얼마인가" 에 답하는 것도 level 4 이다.

한 방법이 닿은 level 은 그것이 내놓은 값이 옳은지에 대해 아무 말도 하지 않는다. `BayesianRidge` 는 level 4 에 닿으며, Appendix B 는 참 noise 표준편차가 0.51 에서 5.25 까지 움직이는 행들에서 그 표준편차를 1.889 와 1.901 사이로, 참값과의 상관 0.420 으로 잰다. 그 둘을 갈라내는 측정이 section 11 이다.

## 5. Probabilistic Regression

점 예측 model 을 density 로 바꾸는 것은 분산을 둘째 출력으로 읽는 loss 이다. Gaussian 을 가정하면 그 loss 는 negative log-likelihood 이며, 둘째 출력이 제약 없이 움직이도록 $s = \log \sigma$ 로 적는다 [[6](#ref-6)].

$$\mathrm{NLL}(y, \mu, s) = s + \frac{(y - \mu)^{2}}{2}e^{-2s} \hspace{19em} (1)$$

Equation (1) 은 한 행마다 $\mu$ 와 $\sigma$ 를 주며, 한계 $L$ 을 넘을 확률은 $1 - \Phi\!\left((L - \mu)/\sigma\right)$ 로 따라 나온다. Appendix B 는 한계를 6.0 에 두고 이 fit 에서 0.0476 을, 실제 비율로는 0.0767 을 잰다.

### 5.1 What The Boosting Libraries Provide

두 boosting library 어느 쪽에도 분산을 내놓는 내장 objective 는 없으며, 그런 것을 연상시키는 이름은 실재하지 않는다. XGBoost 3.2.0 에서 `reg:normal` 과 `reg:gaussian` 은 둘 다 `Unknown objective function` 으로 실패하고, 실재하는 objective 가운데 `reg:squarederror` 는 평균을, `reg:absoluteerror` 는 median 을, `reg:pseudohubererror` 는 robust 한 중심을 학습하며, 셋 다 한 행에 숫자 하나이다. LightGBM 4.7.0 에서 `gaussian` 과 `normal` 이라는 이름도 같은 방식으로 실패한다.

작동하는 경로는 출력이 둘인 custom objective 이며, Appendix B 가 그것을 XGBoost 에서 학습하고 측정한다. Equation (1) 의 gradient 와 Hessian 을 두 출력에 대해 한꺼번에 주고 `multi_strategy="one_output_per_tree"` 로 출력마다 tree 를 기르면, `predict` 는 열이 $\mu$ 와 $\log \sigma$ 인 shape `(600, 2)` 의 행렬을 돌려준다. LightGBM 에서는 이 경로가 막혀 있다. 그 estimator 가 열이 둘인 label 을 `y should be a 1d array` 로 물리치므로, 그 library 에서 분산을 얻으려면 NGBoost 나 quantile fit 의 grid 를 쓴다.

같은 것을 이미 조립해 둔 library 가 둘 있다. NGBoost 는 고른 distribution 의 parameter 들을 natural gradient 아래에서 함께 boosting 한다 [[3](#ref-3)]. `BayesianRidge` 와 `ARDRegression` 은 evidence framework 아래에서 noise 와 coefficient 를 함께 학습하고 그 결과를 `predict(X, return_std=True)` 로 내준다 [[4](#ref-4)].

### 5.2 The Variance Head Overfits

학습된 분산은 tree 를 더할수록 줄어들고 coverage 도 따라 내려간다. Appendix B 는 tree 100 그루에서 0.852, 200 에서 0.817, 400 에서 0.765 를 재며, 같은 구간에서 평균 폭은 5.40 에서 4.68 로 줄어든다.

원인은 equation (1) 안에 있다. 학습 행의 residual 이 작은 자리에서는 $s$ 를 낮추는 것이 보상을 받는데, boosting 된 model 은 학습 residual 을 0 쪽으로 몰고 가므로, 분산 head 는 test 행이 공유하지 않는 이유로 계속 줄어드는 residual 을 상대로 학습된다. 멈출 기준이 될 행을 따로 떼어 두거나 $\sigma$ 에 아래 한계를 두는 것이 둘째 출력을 정직하게 지킨다.

`BayesianRidge` 의 실패는 반대이며, 조정의 문제가 아니라 구조의 문제다. 내놓는 분산이 학습된 noise 상수에 입력이 학습 데이터에서 멀어질수록 커지는 항을 더한 것이므로, fit 이 잘 정해진 행에서는 폭이 숫자 하나이다. Appendix B 는 그것이 600 행에 걸쳐 0.012 만큼 움직이는 것을 잰다.

## 6. Quantile Regression

Squared-error loss 를 pinball loss 로 바꾸면 같은 model 이 conditional distribution 의 어느 functional 을 학습하는지가 바뀌며, level $\alpha$ 에서 그 functional 은 $\alpha$ quantile 이다 [[1](#ref-1)].

$$L_{\alpha}(y, q) = \max\left\{\alpha\,(y - q),\ (\alpha - 1)(y - q)\right\} \hspace{19em} (2)$$

Equation (2) 는 학습 결과의 $\alpha$ 만큼이 그 아래 떨어지는 경계에서 최소가 되므로, 0.10 에서의 fit 과 0.90 에서의 fit 이 그 가운데 80 퍼센트를 감싼다. 둘을 묶어 주는 것은 fit 안에 없다. 각각이 별개의 최소화이고, Appendix B 는 boosting library 셋에서 그 결과 coverage 를 nominal 0.90 에 대해 0.815 에서 0.832 로 잰다.

Table 3 이 각 library 에서 level 을 고르는 parameter 이다. `alpha` 라는 이름이 LightGBM 과 `GradientBoostingRegressor` 에서는 level 을 뜻하고 `QuantileRegressor` 에서는 L1 penalty 를 뜻하는데, 거기서 level 은 `quantile` 이고 penalty 의 기본값은 1.0 이다.

Table 3. How each library fits a quantile

| Estimator | Parameter | Levels per fit |
|-----------|-----------|----------------|
| `QuantileRegressor` | `quantile`, `alpha` 는 L1 penalty | 하나 |
| `GradientBoostingRegressor` | `loss="quantile"`, `alpha` | 하나 |
| `HistGradientBoostingRegressor` | `loss="quantile"`, `quantile` | 하나 |
| `LGBMRegressor` | `objective="quantile"`, `alpha` | 하나 |
| `XGBRegressor` | `objective="reg:quantileerror"`, `quantile_alpha` | 여럿, level 마다 열 하나를 돌려줌 |

Level 당 한 번의 fit 이 규모를 정하는 값이다. `LGBMRegressor` 로 열 개 level 을 얻으려면 학습하고 저장하고 serving 할 model 이 열 개이며, `XGBRegressor` 는 level 들을 array 로 받아 하나의 booster 에서 열 하나씩의 행렬을 돌려준다.

따로 학습된 level 은 서로 교차할 수도 있는데, 한 fit 을 다른 fit 에 묶어 주는 항이 없으므로 어떤 행에서는 0.05 경계가 0.95 경계 위에 올 수 있다. Appendix B 는 600 개 test 행에서 그 교차를 세어 하나도 찾지 못하는데, 이는 보장이 아니라 그 fit 의 성질이므로 내놓은 열은 쓰기 전에 정렬한다.

## 7. Bootstrap And Resampling

Resampling 한 행으로 model 을 다시 학습하는 것은 학습된 평균이 얼마나 움직이는지를 재며, 그것은 결과가 얼마나 움직이는지와 다른 양이다 [[7](#ref-7)]. 그 차이는 정도의 문제가 아니다. Appendix B 는 `Ridge` 를 bootstrap resample 200 벌에 다시 학습하고 행마다 200 개 예측의 5, 95 백분위수를 취하는데, 평균 폭 0.40 에 coverage 0.115 가 나오며 같은 model 의 residual 표준편차는 1.940 이다.

Random forest 의 tree 들 사이의 폭도 같은 양이고 같은 결함을 지닌다. 기본 설정에서는 다 자란 leaf 가 한 행쯤을 담으므로 tree 하나의 예측이 noise 섞인 관측 하나에 가깝고 tree 들의 폭이 우연히 noise 를 닮아, 결함이 잘 보이지 않는다. Appendix B 는 거기서 coverage 0.858 을, `min_samples_leaf=20` 으로 leaf 가 자기 행들을 평균하게 하면 0.650 을 재며, 이 경로의 실제 값어치는 뒤의 숫자이다.

Forest 에는 결과를 재는 둘째 경로가 있다. 입력이 떨어지는 leaf 안의 학습 행들을 tree 전체에 걸쳐 모으면 그 입력에서의 결과의 가중 표본이 되고, 그 표본의 어떤 quantile 이든 level 3 의 답이다 [[2](#ref-2)]. Appendix B 는 폭이 2.89 만큼 움직이면서 coverage 0.893 을 재는데, 같은 forest 의 tree spread 는 0.650 이다.

위의 어느 것도 resampling 경로를 버리라는 말이 아니라 그것의 쓰임을 고쳐 잡는 말이다. 다시 학습한 member 들의 폭은 fit 자체가 얼마나 의심스러운지를 묻는 데 맞는 양이며, 그것이 학습 데이터가 얇아지는 자리에서 커지는 값이다. 학습된 noise 항에 더하면 predictive variance 의 두 부분이 되고, 혼자 쓰면 그 둘 가운데 작은 쪽이 된다.

## 8. Tree And Ensemble Family

네 tree model 가운데 셋은 loss 를 pinball loss 로 바꾸어 level 3 에 닿고, 넷째는 loss 를 전혀 바꾸지 않고 닿는다. Table 4 가 그 차이와, 각각이 그 너머로 무엇에 닿는지이다.

Table 4. What each tree model reaches

| Estimator | Level 3 | Level 4 | Second fit needed |
|-----------|---------|---------|-------------------|
| `LGBMRegressor` | `objective="quantile"` | 촘촘한 level grid, 또는 NGBoost | Level 당 한 번의 fit |
| `XGBRegressor` | `objective="reg:quantileerror"` | 출력이 둘인 custom objective, section 5.1 | 아니오, level array 를 한꺼번에 받음 |
| `GradientBoostingRegressor` | `loss="quantile"` | 촘촘한 level grid | Level 당 한 번의 fit |
| `RandomForestRegressor` | Leaf observations, section 7 | 같은 leaf 에서 읽는 촘촘한 grid | 아니오, 한 번의 fit 이 모든 level 을 맡음 |

한 번의 fit 이 두 level 을 모두 맡는 행은 random forest 뿐인데, 그것이 읽는 표본이 학습된 tree 안에 이미 저장되어 있기 때문이며, `XGBRegressor` 행의 `아니오` 는 그 행의 level 3 항목에만 걸린다. 대신 치르는 것은 memory 와 예측 시간이다. 학습된 결과를 남겨 두어야 하고, 예측 한 번이 tree 당 숫자 하나가 아니라 모든 tree 의 leaf pool 을 건드린다.

여러 level 을 한 번의 fit 으로 받는 행은 `XGBRegressor` 뿐이며, 그 행의 두 항목은 서로 다른 기법이다. `quantile_alpha` 는 array 를 받아 level 마다 열 하나를 돌려주고, level 4 항목은 section 5.1 의 custom objective 로서 경계 둘이 아니라 평균과 log 표준편차인 열 둘을 돌려준다.

## 9. Linear Family

Linear regressor 가운데 둘은 평균과 분산을 내놓고, 하나는 level 을 요청하면 quantile 을 내놓으며, 남은 다섯은 중심만 내놓는다. Table 5 가 각각이 내놓는 것이다.

Table 5. What the linear regressors emit

| Estimator | `predict` signature | Level as fitted | Functional estimated |
|-----------|---------------------|-----------------|----------------------|
| `Ridge` | `predict(X)` | 1 | Conditional mean |
| `Lasso` | `predict(X)` | 1 | Conditional mean |
| `ElasticNet` | `predict(X)` | 1 | Conditional mean |
| `HuberRegressor` | `predict(X)` | 1 | Robust 한 중심 |
| `LinearSVR` | `predict(X)` | 1 | Epsilon-insensitive tube 의 중심 |
| `QuantileRegressor` | `predict(X)` | 3, fit 당 level 하나 | Conditional quantile |
| `BayesianRidge` | `predict(X, return_std=False)` | 4, Gaussian | Conditional mean 과 variance |
| `ARDRegression` | `predict(X, return_std=False)` | 4, Gaussian | Conditional mean 과 variance |

앞의 다섯 행은 어떤 중심을 추정하는지에서 서로 다를 뿐, 폭에 대해 하는 말은 모두 같으며 그것은 아무 말도 하지 않는 것이다. `Ridge`, `Lasso`, `ElasticNet` 은 penalty 에서만 서로 다르고 셋 다 conditional mean 을 학습한다. `HuberRegressor` 와 `LinearSVR` 은 outlier 에 견디는 중심을 학습하는데, 그래서 점 추정이 더 믿을 만해질 뿐 폭에 대한 물음은 있던 자리에 그대로 남는다.

그 다섯에서 level 3 에 닿는 길은 `QuantileRegressor` 이며, 이는 기존 fit 의 option 이 아니라 다른 fit 이다. 이 계열의 어떤 linear estimator 도 loss 를 인자로 받지 않기 때문이다. Level 4 에 닿는 길은 `BayesianRidge` 나 `ARDRegression` 이고, section 5.2 의 상수 noise 항을 함께 받는다.

## 10. Weighting An Ensemble

행마다의 분산은 고정된 ensemble weight 를 입력에 따라 움직이는 weight 로 바꾸며, 그것이 Problem Statement 가 요구하는 것이다. 같은 행에서 평균이 $\mu_{1}, \mu_{2}$ 이고 분산이 $\sigma_{1}^{2}, \sigma_{2}^{2}$ 인 두 member 에 대해, 결합의 분산을 최소로 만드는 weight 는 그 분산에 반비례한다.

$$w_{1} = \frac{1/\sigma_{1}^{2}}{1/\sigma_{1}^{2} + 1/\sigma_{2}^{2}}, \qquad \hat y = w_{1}\mu_{1} + (1 - w_{1})\mu_{2} \hspace{19em} (3)$$

Appendix B 는 feature 구성 때문에 각각 결과의 다른 동인에 눈이 먼 두 member 를 학습하고 equation (3) 으로 묶는다. Member 를 따로 쓰면 RMSE 2.0643 과 3.0808 이고, equal weight 는 2.3910 이며, equation (3) 의 weight 는 2.1689 로 $w_{1}$ 이 행에 따라 0.36 과 0.93 사이를 움직인다.

이 이득은 member 들이 서로 다른 자리에서 불확실한 데서 오며, 그렇지 않으면 사라진다. Appendix B 의 둘째 쌍은 noise 를 지고 있는 feature 를 두 member 가 공유하는데, equation (3) 이 2.7213 으로 equal weight 의 2.6942 에 진다. 두 member 가 거의 같은 $\sigma$ 를 보고하여 거기서 나오는 weight 가 어디서나 0.5 근처인데 한쪽 member 는 그냥 더 나쁘기 때문이다. 틀린 분산은 분산이 없는 것보다 나쁜데, equation (3) 이 그러면 틀린 member 에게 값을 치르기 때문이다.

## 11. Calibration

전체 행에서 잰 coverage 는 조용한 행을 과도하게 덮고 시끄러운 행을 덜 덮어서 달성되므로, 내놓은 interval 은 세 숫자를 함께 놓고 판정한다. Table 6 이 그 셋이고, 각각이 잡아내는 실패를 함께 적었다.

Table 6. What an emitted interval is judged by

| Measure | Definition | Catches |
|---------|------------|---------|
| Marginal coverage | Interval 안에 든 전체 행의 비율 | 전체적으로 너무 좁거나 너무 넓은 경계 |
| Conditional coverage | 같은 값을, 입력으로 묶은 band 안에서 | 변하는 폭 자리를 고정된 폭이 대신하는 것 |
| Sharpness | Interval 의 평균 폭 | 쓸모없이 넓혀서 덮게 만든 경계 |

Coverage 와 sharpness 는 서로에 대해 읽는데, 어느 하나만이면 무의미하게 충족되기 때문이다. 규칙은 calibration 을 조건으로 두고 sharpness 를 최대화하는 것이며, coverage 가 성립하는 방법들 가운데 interval 이 가장 좁은 것이 이긴다 [[8](#ref-8)]. Appendix B 가 그 비교로 짜여 있다.

둘을 한꺼번에 채점하는 숫자 하나가 proper scoring rule 이며, 참 distribution 에서만 그 기대값이 최적이 되는 점수이다 [[5](#ref-5)]. 그 가운데 둘이 Table 2 의 level 들을 덮는다.

- Equation (2) 의 pinball loss, quantile 하나에 대해 proper. `mean_pinball_loss` 와 `d2_pinball_score` 로 제공.
- Continuous ranked probability score, distribution 전체에 대해 proper. 내놓은 CDF 와 결과에서의 step function 사이 차이의 제곱을 적분한 값.

어느 것도 coverage 표를 대신하지 않는다. Proper score 는 방법들을 서로 견주어 줄 세우면서 그 가운데 최고가 옳은지는 말하지 않으며, Appendix B 의 coverage 0.115 는 어떤 줄 세우기도 보고하지 않는 그 interval 에 대한 사실이다.

## 12. Selection

선택은 그 확률을 무엇에 쓸 것인가에서 따라 나온다. Table 7 은 왼쪽 열에서 읽는다.

Table 7. Which implementation to use

| Use | When | Why |
|-----|------|-----|
| Probabilistic regression, custom NLL objective | Tree model 이고 한계나 weight 를 위해 행마다의 분산이 필요한 경우 | 이미 쓰는 library 에서 얻는 level 4, 값은 section 5.2 |
| NGBoost | 같은 경우이면서 custom objective 보다 관리되는 구현을 택할 때 | Natural gradient 와 distribution family 가 조립되어 옴 |
| `BayesianRidge` or `ARDRegression` | Linear fit 이고 split 을 쓰지 않고 density 를 원하는 경우 | Native 로 level 4, 다만 noise 항이 움직이지 않음 |
| Quantile regression | Level 이 미리 정해져 있고 distribution 을 가정하지 않아야 하는 경우 | Distribution family 없이 level 3, 값은 level 당 한 번의 fit |
| Leaf observations of a forest | Forest 가 이미 학습되어 있고 level 을 둘 이상 원하는 경우 | 한 번의 fit 에서 모든 level, 가정도 없음 |
| Bootstrap and resampling | 묻는 것이 결과의 변동이 아니라 fit 이 얼마나 의심스러운지인 경우 | 셋 가운데 그 질문에 답하는 유일한 것, section 7 |

## 13. Further Work

- **A coverage guarantee over any of the three** — Table 1 의 모든 방법이 Appendix B 에서 nominal 0.90 에 대해 측정되었고 그에 닿는 것이 보장된 것은 하나도 없으며, 적응하는 셋은 0.815 와 0.893 사이에 내려앉는다. Conformal prediction 은 떼어 둔 split 에서 보정을 정하여, 바탕 model 이 무엇이든 유한 표본에서 marginal coverage 가 성립하게 만들며, 그래서 넷째 구현이 아니라 Table 1 의 어느 행에든 씌우는 wrapper 가 된다. 예측할 행과 exchangeable 한 calibration split 이 필요한데, 그것이 drift 하는 공정이 깨뜨리는 조건이다.
- **A variance head that does not shrink** — Section 5.2 는 tree 를 더할수록 NLL fit 의 coverage 가 0.852 에서 0.765 로 내려가는 것을 재는데, 분산이 계속 줄어드는 학습 residual 을 상대로 학습되기 때문이다. 평균을 학습하는 행과 분산을 학습하는 행을 나누는 것이 mean-variance 문헌이 이르른 처방이고, 이제는 boosting loop 안에서 돌릴 만큼 값이 싸다. 그 행을 나누는 규칙과 그 나눔이 평균에 무엇을 치르게 하는지에 대한 측정이 필요한데, 분산에 쓴 행은 평균이 보지 못하는 행이기 때문이다.

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

- **conditional coverage**: 전체 행이 아니라 행의 한 band 안에서 잰 interval 의 coverage.
- **continuous ranked probability score**: 내놓은 CDF 와 결과에서의 step function 사이 차이의 제곱을 적분한 값.
- **coverage**: 결과가 내놓은 interval 안에 든 행의 비율.
- **inverse-variance weighting**: Equation (3) 의 ensemble weight 로, 각 member 의 그 행에서의 분산에 반비례한다.
- **leaf observations**: 입력이 떨어지는 leaf 로 갈라 담긴 학습 행을, 결과의 표본으로 읽은 것.
- **marginal coverage**: 전체 행에 대해 잰 interval 의 coverage.
- **negative log-likelihood**: Equation (1) 의 loss 로, 가정한 family 의 평균과 분산에서 최소가 된다.
- **pinball loss**: Equation (2) 의 비대칭 절대 손실로, quantile 하나에서 최소가 된다.
- **prediction interval**: 미래 결과가 명시된 level 로 그 사이에 든다고 주장하는 두 숫자.
- **predictive distribution**: 한 입력에서의 결과에 대한 density 또는 CDF.
- **proper scoring rule**: 참 distribution 에서만 그 기대값이 최적이 되는 점수.
- **sharpness**: 내놓은 interval 이나 density 의 좁음이며, coverage 가 성립하는 방법들 사이에서만 판정한다.
- **variance head**: 출력이 둘인 model 의 둘째 출력으로, log 표준편차를 지닌다.

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

Split 은 학습에 쓸 1800 행, fit 밖의 행이 필요한 측정을 위해 떼어 둔 600 행, 측정할 600 행이다. 모든 interval 은 0.05 와 0.95 level 로 요청하므로 nominal coverage 는 전부 0.90 이다. Boosting model 에는 같은 300 그루의 tree, 같은 0.05 의 learning rate, 같은 3 의 depth 를 주어, 그들 사이의 비교가 기본값의 비교가 아니라 objective 의 비교가 되게 했다.

Test 행에서 참 noise 표준편차는 0.51 에서 5.25 까지 움직인다. 그 행을 $|x_{1}|$ 로 정렬한 세 band 에 걸쳐 평균하면 참 90 퍼센트 폭은 2.85, 5.21, 9.28 이므로 어느 숫자 하나도 세 band 를 다 맡지 못한다. Table 8 의 두 band 열은 그 band 들의 바깥 둘이다.

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

Table 8 에서 다섯 가지가 읽힌다.

- 점 예측 model 의 bootstrap 이 평균 폭 0.40 에 coverage 0.115, 같은 model 의 residual 표준편차는 1.940. 결과가 아니라 학습된 평균의 폭.
- Leaf 당 한 행에서 0.858, 스물에서 0.650 인 tree spread. 앞의 숫자가 predictive distribution 이 아니라 평균을 내기에 너무 작은 leaf 에서 온 것.
- Coverage 가 0.90 에 가깝고 폭의 표준편차가 0.49 이하인 세 행, 그리고 1.000 대 0.770 의 conditional coverage. Band 를 서로 상쇄시켜 달성한 marginal coverage.
- 폭이 2.53 과 2.89 만큼 움직이면서 0.852 와 0.893 을 덮고 band 사이 conditional coverage 가 0.04 안에 드는 NLL fit 과 leaf observations. 폭이 noise 를 따라가는 두 행.
- 0.815 에서 0.832 에 있는 quantile boosting 세 행. 보장 없는 적응이며, equation (2) 가 학습 행에서 최소화될 뿐 held-out 빈도가 level 과 맞도록 강제하는 것이 없기 때문.

### B.1 The Variance Head

Equation (1) 을 출력이 둘인 objective 로 XGBoost 에 준 것이다.

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

내놓는 표준편차는 0.77 에서 5.33 까지 움직이며 참 noise 표준편차와 0.962 로 상관하는데, 같은 행에서 `BayesianRidge` 는 0.420 이다.

Table 9 는 같은 learning rate 와 depth 에서 tree 를 더할 때 그 fit 의 coverage 이다.

Table 9. Coverage of the NLL fit against the tree count

| Trees | Coverage | Mean width | Coverage, quiet third | Coverage, noisy third |
|-------|----------|------------|-----------------------|-----------------------|
| 100 | 0.852 | 5.40 | 0.875 | 0.860 |
| 200 | 0.817 | 5.05 | 0.810 | 0.860 |
| 400 | 0.765 | 4.68 | 0.750 | 0.835 |

한계를 넘을 확률은 같은 두 열에서 읽으며, Table 10 이 두 한계에서의 그 확률과 실제로 관측된 비율이다.

Table 10. Probability of exceeding a limit, averaged over the 600 test rows

| Limit | XGBoost Gaussian NLL | `BayesianRidge` | Observed rate |
|-------|----------------------|-----------------|---------------|
| 4.0 | 0.1318 | 0.1754 | 0.1650 |
| 6.0 | 0.0476 | 0.0762 | 0.0767 |

`BayesianRidge` 는 한 행 수준에서는 둘 가운데 나쁜 model 이면서 Table 10 의 두 행 모두에서 더 가까운데, 그 까닭은 이 표가 무엇을 평균하는지에 있다. 약 1.89 인 그 하나의 상수 표준편차가 이 데이터의 평균 noise 에 가까운데, 그것은 전체 행에 걸쳐 평균한 비율에는 맞는 숫자이고 어느 한 행에는 틀린 숫자이다.

### B.2 Quantile Regression In Four Libraries

같은 데이터 위에서 본 Table 3 의 parameter 이다.

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

LightGBM 의 두 경계는 600 개 test 행 가운데 어디서도 교차하지 않는다.

### B.3 The Two Forest Routes

같은 학습된 forest 를 두 번 읽는다.

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

경로 1 은 이 forest 에서 coverage 0.858 에 닿고, `min_samples_leaf=20` 으로 각 leaf 가 최소 스무 행을 담게 하면 0.650 에 닿는다. 경로 2 는 같은 forest 에서 0.893 에 닿으며 학습된 결과를 memory 에 남겨 두어야 한다.

### B.4 Weighting Two Members By Their Variances

B.1 의 objective 로 서로 다른 feature 구성에서 학습한 두 member 에 equation (3) 을 적용한 것이다.

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

Table 11 이 그 결합과 대안들을 600 개 test 행의 root mean squared error 로 채점한 것이다.

Table 11. Two members combined, root mean squared error

| Combination | Columns (0, 1, 2) and (0, 3, 4) | Columns (0, 1) and (1, 2) |
|-------------|---------------------------------|---------------------------|
| Member 1 alone | 2.0643 | 2.5430 |
| Member 2 alone | 3.0808 | 3.6136 |
| Equal weights | 2.3910 | 2.6942 |
| Inverse-variance weights, equation (3) | 2.1689 | 2.7213 |

첫째 열이 Problem Statement 가 말하는 경우이며 weight 는 행에 따라 0.36 과 0.93 사이를 움직인다. 둘째 열은 noise 를 지고 있는 feature 를 공유하는 두 member 에 같은 절차를 쓴 것이다. 둘이 거의 같은 표준편차를 보고하여 weight 가 어디서나 0.5 근처에 앉고, equation (3) 은 더 쓸 정보가 남지 않아 equal weight 에 진다.
