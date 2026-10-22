# PLS Bootstrap Ensemble
Rev. 1 | Created: 2026-10-19 | Updated: 2026-10-22 18:20 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Comparison](#31-comparison)
- [4. Principle](#4-principle)
  - [4.1 Algorithm](#41-algorithm)
  - [4.2 Spread and Prediction Interval](#42-spread-and-prediction-interval)
  - [4.3 Accuracy Gain](#43-accuracy-gain)
  - [4.4 Combined Quantity](#44-combined-quantity)
- [5. Application](#5-application)
  - [5.1 Assumptions](#51-assumptions)
  - [5.2 Settings](#52-settings)
  - [5.3 Failure Conditions](#53-failure-conditions)
  - [5.4 Use Cases](#54-use-cases)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Worked Example](#appendix-b-worked-example)

## 1. Purpose

- **Problem Statement**: PLS model 하나만 적합하면 예측값과 회귀계수가 학습 표본에 따라 얼마나 달라지는지 알 수 없다.
- **Goal**: 행을 다시 뽑은 표본마다 PLS 를 적합하여 예측값, 예측 구간, 회귀계수의 안정성을 구하고, 그 값을 믿을 수 있는 조건을 가린다.
- **Non-Goal**: PLS 알고리즘의 유도, boosting 과 stacking 같은 다른 ensemble framework 는 다루지 않는다.

## 2. Summary

PLS bootstrap ensemble 은 행을 복원추출한 표본마다 PLS 를 적합하고 member 의 예측을 평균하는 bagging 이다 [[1](#ref-1)].
PLS 는 안정한 학습기여서 평균으로 얻는 정확도 이득은 작으며 [[2](#ref-2)], 쓸모는 member 사이의 퍼짐, out-of-bag (OOB) 오차, 계수의 bootstrap 분포에 있다.
퍼짐은 적합한 평균의 불확실성만 재므로, 예측 구간을 만들 때는 잔차 분산을 더한다.

## 3. Taxonomy and its Hierarchy

PLS bootstrap ensemble 의 변형은 다시 뽑는 단위, 성분 수를 정하는 방식, 합치는 양의 세 축으로 나뉜다. 세 축과 축마다의 값은 아래 [Fig 1](#fig-1) 과 같다.

```text
PLS bootstrap ensemble
|
+-- Resampled unit
|   +-- Rows with replacement ............ bagging
|   +-- Rows without replacement ......... subagging
|   +-- Groups or blocks of rows ......... group / block bootstrap
|
+-- Number of components
|   +-- Fixed for all members ............ chosen once by CV
|   +-- Chosen per member ................ inner CV in each member
|
+-- Combined quantity
    +-- Prediction ....................... mean and spread
    +-- Regression coefficient ........... mean and percentile interval
```

<a id="fig-1"></a>Fig 1. Three axes of a PLS bootstrap ensemble

세 축은 서로 독립이다. 예를 들어 group bootstrap 에 성분 수를 member 마다 정하고 계수를 합치는 조합도 쓸 수 있다.

변형 사이의 계층은 단일 PLS 에서 출발하며, 한 단계 내려갈 때마다 행에 대한 가정을 하나 버리고 계산을 더한다. 계층은 아래 [Fig 2](#fig-2) 와 같다.

```text
Single PLS ................................ one fit, no spread
|   assumes: the training rows are the only sample
|
+-- Row bootstrap PLS ..................... bagging, subagging
    |   assumes: rows are independent and exchangeable
    |
    +-- Group / block bootstrap PLS ....... lot, wafer or time-block resampling
            assumes: groups are independent; rows inside a group need not be
```

<a id="fig-2"></a>Fig 2. Assumptions held at each level

- Single PLS 에서 row bootstrap 으로 내려가면 퍼짐, OOB 오차, 계수 구간을 얻고, 계산량은 member 수 $B$ 배로 는다.
- Row bootstrap 에서 group bootstrap 으로 내려가면 행이 서로 독립이라는 가정을 group 이 서로 독립이라는 가정으로 바꾼다. 대신 다시 뽑는 단위의 수가 행 수에서 group 수로 줄어 퍼짐의 추정이 거칠어진다.

### 3.1 Comparison

변형은 자료의 구조와 원하는 산출물로 고른다.

Table 1. Variant by situation

| Situation                         | Variant                         | Reason                                               |
| :-------------------------------: | :-----------------------------: | :--------------------------------------------------: |
| 행이 서로 독립인 calibration      | Row bootstrap, 성분 수 고정     | 가장 단순하며 행 단위 OOB 오차를 그대로 씀           |
| 적합 횟수를 줄여야 할 때          | Subagging                       | Bagging 과 비슷한 성능, 더 적은 계산량 [[3](#ref-3)] |
| 같은 lot·wafer·시간 구간의 행     | Group / block bootstrap         | 같은 group 의 행이 학습과 OOB 에 나뉘지 않음         |
| 계수의 부호와 크기를 해석할 때    | Coefficient percentile interval | 성분의 부호와 무관한 회귀계수를 합침                 |
| 성분 수 선택의 불확실성도 넣을 때 | 성분 수를 member 마다 선택      | 성분 수가 달라지는 몫이 퍼짐에 들어감                |

Group bootstrap 이 필요한지는 행 단위 OOB 오차와 group 단위 cross-validation (CV) 오차를 견주어 정한다. 행 단위 OOB 오차가 뚜렷이 작으면 행 사이 상관이 OOB 오차를 낮춘 것이다.

## 4. Principle

PLS bootstrap ensemble 은 PLS 를 학습기로 둔 bagging 이다 [[1](#ref-1)]. PLS 는 $X$ 를 $y$ 와의 공분산이 큰 방향으로 투영한 성분 $A$ 개로 회귀하므로, 열이 많고 서로 상관된 자료에서도 적합이 된다 [[4](#ref-4)].

### 4.1 Algorithm

절차는 성분 수를 한 번 정한 뒤 $B$ 개의 member 를 적합하는 것이다.

```text
# Pseudocode
choose A by K-fold CV on (X, y)
for b in 1..B:
    I_b <- n row indices drawn with replacement
    O_b <- rows not in I_b                      # out-of-bag rows of member b
    f_b <- PLS with A components fitted on (X[I_b], y[I_b])
predict : mean of f_b(x) over b                  # equation (1)
spread  : variance of f_b(x) over b              # equation (2)
OOB     : for row i, mean of f_b(x_i) over b with i in O_b
```

행 하나가 bootstrap 표본에 한 번도 뽑히지 않을 확률은 $(1 - 1/n)^n$ 이고 $n$ 이 커지면 $e^{-1} \approx 0.368$ 에 가까워진다. 따라서 member 마다 행의 약 37 % 가 OOB 로 남고, 행 하나의 OOB 예측에는 member 약 $0.368B$ 개가 쓰인다.

### 4.2 Spread and Prediction Interval

예측값은 member 예측의 평균이고, 퍼짐은 member 예측의 분산이다.

```math
\bar{f}(x) = \frac{1}{B}\sum_{b=1}^{B}\hat{f}_b(x) \hspace{19em} (1)
```

```math
s_f^2(x) = \frac{1}{B-1}\sum_{b=1}^{B}\left(\hat{f}_b(x) - \bar{f}(x)\right)^2 \hspace{19em} (2)
```

$s_f^2(x)$ 는 학습 표본이 바뀔 때 적합한 평균 $\bar{f}(x)$ 가 움직이는 정도이다. 새 관측값은 $y = f(x) + \varepsilon$ 이므로, 관측값이 들어갈 구간에는 잡음 분산 $\sigma^2$ 이 더해져야 한다. $\sigma^2$ 의 추정값으로 OOB 평균제곱오차를 쓴다.

```math
\hat{\sigma}^2_{\mathrm{OOB}} = \frac{1}{n'}\sum_{i=1}^{n'}\left(y_i - \hat{y}_i^{\mathrm{OOB}}\right)^2 \hspace{19em} (3)
```

```math
\bar{f}(x) \pm z_{1-\alpha/2}\sqrt{s_f^2(x) + \hat{\sigma}^2_{\mathrm{OOB}}} \hspace{19em} (4)
```

$n'$ 은 OOB 예측을 하나 이상 받은 행의 수이다. $\hat{\sigma}^2_{\mathrm{OOB}}$ 에는 잡음과 적합 오차가 함께 들어 있으므로, 식 (4) 는 적합 오차를 두 번 더해 구간이 약간 넓어지는 쪽으로 어긋난다.

[Appendix B](#appendix-b-worked-example) 의 예제에서 member 예측의 5–95 percentile 구간은 새 관측값의 33.3 % 만 담았고, 식 (4) 의 90 % 구간은 90.7 % 를 담았다.

### 4.3 Accuracy Gain

Bagging 이 평균으로 줄이는 것은 학습기의 분산이므로, 학습 표본이 조금 바뀔 때 적합이 크게 바뀌는 학습기일수록 이득이 크다 [[1](#ref-1)]. PLS 는 성분 수가 적을 때 적합이 안정하다 [[2](#ref-2)].

- 성분 수가 적거나 중간일 때 bagging 한 PLS 는 단일 PLS 보다 뚜렷이 낫지 않다 [[2](#ref-2)].
- 성분 수가 많아 단일 PLS 가 overfitting 할 때 bagging 한 PLS 는 overfitting 의 영향을 덜 받는다 [[2](#ref-2)].
- [Appendix B](#appendix-b-worked-example) 의 예제에서 test RMSE 는 단일 PLS 0.525, bagging 한 PLS 0.522 로, 잡음 표준편차 0.5 보다 각각 0.025 와 0.022 컸다.

### 4.4 Combined Quantity

Member 사이에서 합치는 양은 예측값과 회귀계수이며, weight 와 loading 은 합치지 않는다. PLS 성분은 부호를 바꾸어도 같은 적합이 되므로 member 마다 weight 와 loading 의 부호가 다를 수 있고, 성분의 차례도 바뀔 수 있다. 회귀계수 $\hat{\beta}_b$ 는 $x$ 로 $\hat{y}$ 를 구하는 선형식의 계수여서 성분의 부호와 차례에 무관하므로, member 사이에서 평균하고 percentile 을 구할 수 있다.

계수 $j$ 의 95 % 구간은 $\hat{\beta}_{b,j}$ 의 2.5 와 97.5 percentile 이다. 구간이 0 을 포함하지 않는 계수는 학습 표본이 바뀌어도 부호가 유지된다.

## 5. Application

[Section 4](#4-principle) 의 절차를 쓸 수 있는 조건, 설정값, 깨지는 조건, 쓰이는 공정은 아래와 같다.

### 5.1 Assumptions

- 행이 서로 독립이고 같은 분포에서 나옴. Group 구조가 있으면 group 이 그 단위.
- 예측할 $x$ 가 학습 자료의 범위 안에 있음.
- 잡음 분산 $\sigma^2$ 이 $x$ 에 따라 크게 달라지지 않음. 식 (4) 에서는 모든 $x$ 에 같은 $\hat{\sigma}^2_{\mathrm{OOB}}$ 가 쓰임.

### 5.2 Settings

Table 2. Settings

| Parameter          | Setting                                  | Basis                                               |
| :----------------: | :--------------------------------------: | :-------------------------------------------------: |
| Member 수 $B$      | $\bar{f}$ 와 구간 폭이 더 변하지 않는 값 | $B$ 를 두 배로 늘려 두 값의 변화를 확인. 예제는 200 |
| 성분 수 $A$        | 전체 자료에 K-fold CV 로 한 번           | Member 마다 정하면 $B$ 배의 CV 가 더 듦             |
| Scaling            | Member 안에서 다시 계산                  | 전체 자료의 평균과 분산을 쓰면 OOB 행의 정보가 섞임 |
| 구간 수준 $\alpha$ | 식 (4) 의 $z_{1-\alpha/2}$               | 90 % 구간은 $z = 1.645$                             |

### 5.3 Failure Conditions

- **행 사이 상관**: 같은 lot·wafer·시간 구간의 행이 학습 표본과 OOB 에 함께 들어가 OOB 오차가 실제보다 작게 나옴. 식 (4) 의 구간도 그만큼 좁아짐. Group bootstrap 으로 바꿈.
- **외삽**: 학습 범위 밖의 $x$ 에서는 member 가 모두 같은 선형 외삽을 하여 $s_f^2(x)$ 가 작게 나옴. 퍼짐이 작다고 예측을 믿을 수 없으며, score 의 Hotelling $T^2$ 로 $x$ 가 학습 범위 안에 있는지 먼저 확인.
- **작은 $n$**: $\hat{\sigma}^2_{\mathrm{OOB}}$ 가 $n'$ 개 행에서 나오므로 학습 표본마다 크게 흔들림. 예제의 $n = 60$ 에서 OOB RMSE 는 0.406 부터 0.612 까지 나왔고, 식 (4) 구간의 coverage 는 0.820 부터 0.962 까지 나왔다.
- **고정한 성분 수**: $A$ 를 전체 자료로 한 번 정하면 성분 수 선택의 불확실성은 퍼짐에 들어가지 않음. 예제에서 CV 가 고른 $A$ 는 학습 표본에 따라 1 부터 5 까지 달랐다.

### 5.4 Use Cases

- **Spectrum calibration**: NIR spectrum 처럼 열이 수백 개이고 서로 상관된 자료에서 성분 함량을 예측. Subagging 을 diesel 품질 parameter 의 NIR calibration 에 적용한 예가 있다 [[3](#ref-3)].
- **Soft sensor 와 virtual metrology**: 공정 sensor 값으로 계측값을 예측할 때, 식 (4) 의 구간 폭으로 계측을 건너뛰어도 되는 wafer 와 실제로 계측할 wafer 를 가름.
- **변수 해석**: 계수 구간이 0 을 포함하지 않는 열만 공정 변수 후보로 남김.

## References

<a id="ref-1"></a>[1] Breiman, L. (1996). [Bagging Predictors](https://doi.org/10.1007/BF00058655). *Machine Learning*, 24(2), 123–140.<br>
<a id="ref-2"></a>[2] Mevik, B.-H., Segtnan, V. H. and Næs, T. (2004). [Ensemble methods and partial least squares regression](https://doi.org/10.1002/cem.895). *Journal of Chemometrics*, 18(11), 498–507.<br>
<a id="ref-3"></a>[3] Galvão, R. K. H., Araújo, M. C. U., Martins, M. N., José, G. E., Pontes, M. J. C., Silva, E. C. and Saldanha, T. C. B. (2006). [An application of subagging for the improvement of prediction accuracy of multivariate calibration models](https://doi.org/10.1016/j.chemolab.2005.09.005). *Chemometrics and Intelligent Laboratory Systems*, 81(1), 60–67.<br>
<a id="ref-4"></a>[4] Wold, S., Sjöström, M. and Eriksson, L. (2001). [PLS-regression: a basic tool of chemometrics](https://doi.org/10.1016/S0169-7439(01)00155-1). *Chemometrics and Intelligent Laboratory Systems*, 58(2), 109–130.

---

## Appendix A. Terminology

- **bagging**: Bootstrap aggregating. 행을 복원추출한 표본마다 학습기를 적합하고 예측을 평균하는 ensemble.
- **bootstrap**: 크기 $n$ 의 자료에서 행 $n$ 개를 복원추출하여 새 표본을 만드는 방법.
- **coverage**: 구간이 새 관측값을 담은 비율.
- **CV**: Cross-validation. 자료를 나누어 일부로 적합하고 나머지로 오차를 재는 검증.
- **Hotelling $T^2$**: Score 가 잠재공간의 중심에서 떨어진 정도를 재는 통계량.
- **member**: Ensemble 을 이루는 model 하나. 여기서는 bootstrap 표본 하나에 적합한 PLS.
- **NIR**: Near-infrared. 근적외선 영역의 spectrum.
- **OOB**: Out-of-bag. 한 member 의 bootstrap 표본에 뽑히지 않은 행.
- **PLS**: Partial Least Squares. $X$ 를 $y$ 와의 공분산이 큰 방향으로 투영한 성분으로 회귀하는 방법.
- **RMSE**: Root mean squared error. 오차 제곱 평균의 제곱근.
- **subagging**: Subsample aggregating. 행을 비복원추출한 부분 표본으로 bagging 하는 방법.

## Appendix B. Worked Example

예제는 열 200 개가 잠재 인자 셋에 따라 움직이는 spectrum 모양의 자료에서 단일 PLS 와 PLS bootstrap ensemble 을 견준다. 학습 행 60 개, test 행 2,000 개, 잡음 표준편차 0.5 로 학습 표본을 20 번 새로 만들고, 표본마다 member 200 개를 적합한다. 실행에는 numpy 와 scikit-learn 이 필요하며, scikit-learn 1.9.1 에서 약 13 초 걸렸다.

```python
# Python
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV

N_TRAIN, N_TEST, P, SIGMA = 60, 2000, 200, 0.5
B, REPS, Z90 = 200, 20, 1.645
SHAPES = np.stack([np.exp(-((np.linspace(0, 1, P) - c) / 0.08) ** 2) for c in (0.25, 0.5, 0.75)])


def simulate(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return collinear, spectrum-like X driven by three latent factors, and y."""
    t = rng.normal(size=(n, 3))
    x = t @ SHAPES + 0.05 * rng.normal(size=(n, P))
    y = t @ np.array([1.0, -0.7, 0.4]) + SIGMA * rng.normal(size=n)
    return x, y


def run(seed: int) -> dict[str, float]:
    """Fit a single PLS and a PLS bootstrap ensemble on one training set and score both."""
    rng = np.random.default_rng(seed)
    x_tr, y_tr = simulate(rng, N_TRAIN)
    x_te, y_te = simulate(rng, N_TEST)

    # single PLS, number of components chosen by 5-fold CV
    search = GridSearchCV(PLSRegression(scale=False), {"n_components": range(1, 11)},
                          cv=5, scoring="neg_mean_squared_error").fit(x_tr, y_tr)
    a = search.best_params_["n_components"]
    single = search.best_estimator_.predict(x_te).ravel()

    # ensemble: B members, each fitted on a bootstrap resample of the rows
    pred = np.empty((B, N_TEST))
    coef = np.empty((B, P))
    oob_sum, oob_cnt = np.zeros(N_TRAIN), np.zeros(N_TRAIN)
    for b in range(B):
        idx = rng.integers(0, N_TRAIN, N_TRAIN)
        oob = np.setdiff1d(np.arange(N_TRAIN), idx)
        member = PLSRegression(n_components=a, scale=False).fit(x_tr[idx], y_tr[idx])
        pred[b] = member.predict(x_te).ravel()
        coef[b] = member.coef_.ravel()
        oob_sum[oob] += member.predict(x_tr[oob]).ravel()
        oob_cnt[oob] += 1

    bagged = pred.mean(axis=0)
    seen = oob_cnt > 0
    oob_mse = np.mean((y_tr[seen] - oob_sum[seen] / oob_cnt[seen]) ** 2)

    # 90 % intervals: member spread alone vs member spread plus OOB error, equation (4)
    lo, hi = np.percentile(pred, [5, 95], axis=0)
    half = Z90 * np.sqrt(pred.var(axis=0, ddof=1) + oob_mse)
    c_lo, c_hi = np.percentile(coef, [2.5, 97.5], axis=0)
    return {
        "components": a,
        "rmse_single": np.sqrt(np.mean((y_te - single) ** 2)),
        "rmse_bagged": np.sqrt(np.mean((y_te - bagged) ** 2)),
        "rmse_oob": np.sqrt(oob_mse),
        "cover_spread": np.mean((y_te >= lo) & (y_te <= hi)),
        "cover_noise": np.mean(np.abs(y_te - bagged) <= half),
        "stable_coef": np.sum((c_lo > 0) | (c_hi < 0)),
    }


rows = [run(seed) for seed in range(REPS)]
for key in rows[0]:
    v = np.array([r[key] for r in rows], dtype=float)
    print(f"{key:13s} mean {v.mean():7.3f}  min {v.min():7.3f}  max {v.max():7.3f}")
```

```text
components    mean   2.700  min   1.000  max   5.000
rmse_single   mean   0.525  min   0.492  max   0.566
rmse_bagged   mean   0.522  min   0.493  max   0.560
rmse_oob      mean   0.520  min   0.406  max   0.612
cover_spread  mean   0.333  min   0.263  max   0.467
cover_noise   mean   0.907  min   0.820  max   0.962
stable_coef   mean 122.250  min  19.000  max 158.000
```

Table 3. Results over 20 training sets

| Quantity                                | Mean   | Min   | Max   |
| :-------------------------------------: | :----: | :---: | :---: |
| CV 가 고른 성분 수                      | 2.700  | 1     | 5     |
| Test RMSE, single PLS                   | 0.525  | 0.492 | 0.566 |
| Test RMSE, ensemble                     | 0.522  | 0.493 | 0.560 |
| OOB RMSE                                | 0.520  | 0.406 | 0.612 |
| 90 % coverage, member percentile        | 0.333  | 0.263 | 0.467 |
| 90 % coverage, equation (4)             | 0.907  | 0.820 | 0.962 |
| 95 % 구간이 0 을 뺀 계수 수 (200 개 중) | 122.25 | 19    | 158   |

- Test RMSE 는 두 방법 모두 잡음 표준편차 0.5 에 가깝고, 둘의 차이는 평균 0.003 이다.
- OOB RMSE 의 평균 0.520 은 test RMSE 와 같은 수준이지만, 학습 표본 하나에서는 0.406 부터 0.612 까지 흩어졌다.
- Member percentile 구간에는 잡음 분산이 빠져 있어 90 % 구간이 새 관측값의 33.3 % 만 담았다. 식 (4) 로 OOB 오차를 더하면 평균 coverage 가 90.7 % 이다.
- 계수 구간이 0 을 뺀 열의 수는 19 부터 158 까지 달랐다. CV 가 성분 2 개를 고른 학습 표본에서는 141–158 개, 4 개 이상을 고른 표본에서는 19–25 개였다.
