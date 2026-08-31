# TVC (Time-Varying Coefficient)
Rev. 3 | Created: 2026-08-30 | Updated: 2026-08-31 01:48 CDT

보통의 회귀는 계수를 상수 하나로 고정한다. TVC 는 그 계수를 시간의 함수 $\beta(t)$ 로 확장한 model 이며, 같은 $X$ 라도 그것이 언제 있었느냐에 따라 결과에 미치는 영향이 달라지는 자료를 위한 것이다.

$$Y(t) = \beta_0(t) + \beta_1(t) X + \epsilon(t)$$

이 문서는 계수가 왜 움직이는지, $\beta(t)$ 를 어떤 형태로 두는지, 그것을 Kalman filter 로 어떻게 추정하는지, 그리고 그 추정을 PLS model 위에 얹는 방법을 차례로 정리한다. 마지막 것은 [Appendix B](#appendix-b-applying-tvp-to-a-pls-model) 에 둔다.

## 1. Scope

- 다루는 범위: $\beta(t)$ 의 함수 형태, 상태공간 표현과 Kalman filter 추정, 생존분석과 거시경제에서의 용례, PLS score 위에서의 적용.
- 다루지 않는 범위: 상수 계수 회귀의 추정 이론, Bayesian sampling 기반 추정, 비선형 filter.
- 전제: 관측에 시간 순서가 있고, 그 순서가 정보를 담고 있는 자료.

## 2. Why The Coefficient Moves

계수가 움직이는 이유는 자료마다 다르지만, 실제로 마주치는 형태는 몇 가지로 좁혀진다. Table 1 은 그 형태와 그것이 $\beta(t)$ 에 남기는 모양이다.

Table 1. Patterns of a moving coefficient

| Pattern | Example | Shape of $\beta(t)$ |
|---------|---------|---------------------|
| Decay | Advertising effect right after a launch | Large at first, monotonically shrinking |
| Delayed and inverted | Transplant surgery, hazardous early and protective later | Sign reversal after a crossing point |
| Structural change | Interest rate effect on growth across market regimes | Level shifts at regime boundaries |

세 형태는 계수를 상수로 두었을 때 무엇을 잃는지도 함께 말해 준다. 감쇄를 상수로 적합하면 전 구간의 평균 효과가 나와 초기의 큰 효과도 후기의 작은 효과도 모두 틀리고, 부호가 뒤집히는 경우에는 두 구간이 상쇄되어 효과가 없다는 결론까지 나온다.

## 3. Forms Of The Coefficient

$\beta(t)$ 를 어떤 형태로 두느냐가 곧 model 의 유연성과 추정 부담을 정한다. 셋 중 어느 것을 고를지는 계수의 변화 모양을 사전에 얼마나 아는지에 달려 있다.

### 3.1 Parametric Form

변화의 모양을 안다면 그것을 직접 적는 것이 가장 단순하다.

$$\beta(t) = \beta_0 + \beta_1 t \qquad \text{or} \qquad \beta(t) = \beta_0 + \beta_1 \ln t$$

앞의 것은 영향력이 일정한 속도로 늘거나 줄어드는 경우, 뒤의 것은 초기에 빠르게 변하다 완만해지는 감쇄에 맞는다. 추정할 parameter 가 둘뿐이라 표본이 적어도 안정적이고, 계수의 해석도 상수 회귀만큼 직관적이다. 대신 가정한 모양이 틀리면 그 틀림이 그대로 결론이 된다.

### 3.2 Spline And GAM

모양을 모른다면 기저함수의 합으로 두고 자료가 곡선을 고르게 한다.

$$\beta(t) = \sum_{k=1}^{K} \gamma_k B_k(t)$$

$B_k$ 는 spline 기저이고 $\gamma_k$ 는 추정 대상이다. GAM 의 틀에서 매끄러움 벌점을 함께 두면 $K$ 를 넉넉히 잡아도 과적합이 통제된다 [[1](#ref-1)]. 부호 반전이나 여러 번의 굴곡처럼 모양을 미리 적기 어려운 경우가 이 방법의 자리이다.

### 3.3 Random Walk In A State Space

시간이 이산적이고 관측이 순서대로 도착한다면, 계수를 함수로 적는 대신 시점마다 하나씩 두고 그것들이 서서히 움직인다고 두는 방법이 있다. 이때 model 은 상태공간 형태를 얻는다. 4 장이 이 형태와 그 추정을 다룬다.

## 4. Estimation By Kalman Filter

### 4.1 State-Space Form

TVP model 은 두 방정식으로 정의된다. 둘을 가르는 기준은 그 값이 자료에 적혀 있는지이다. $y_t$ 와 $X_t$ 는 자료를 열면 숫자로 그대로 읽히지만, 계수 $\beta_t$ 는 어디에도 적혀 있지 않아 그 둘로부터 추정할 수밖에 없다. 그래서 앞의 것들을 잇는 아래 첫 식을 관측 방정식이라 하고, 뒤엣것이 시간에 따라 어떻게 움직이는지를 적은 둘째 식을 상태 방정식이라 하며, 추정 대상인 $\beta_t$ 가 이 상태공간의 상태 변수이다.

상수 계수 회귀에서도 계수는 자료에 적혀 있지 않다. 다만 미지수가 하나뿐이라 전체 자료로 한 번 추정하면 끝나므로 굳이 상태라 부르지 않는다. TVP 는 시점마다 다른 $\beta_t$ 를 두어 미지수를 시점 수만큼 만들고, 그것을 관측이 도착할 때마다 따라가며 추정해야 할 대상으로 삼는다.

$$y_t = X_t \beta_t + \epsilon_t, \qquad \epsilon_t \sim N(0, R)$$

$$\beta_t = \beta_{t-1} + v_t, \qquad v_t \sim N(0, Q)$$

Table 2. Terms of the state-space form

| Term | Role |
|------|------|
| $y_t$ | Observed response at time $t$ |
| $X_t$ | Observed covariates at time $t$ |
| $\beta_t$ | Coefficient at time $t$, the state variable |
| $R$ | Observation noise variance |
| $Q$ | State noise variance, how fast the coefficient may move |

$Q$ 가 이 model 의 조절 손잡이이다. $Q$ 가 0 이면 계수는 움직이지 않아 보통의 회귀로 되돌아가고, $Q$ 가 크면 계수가 관측을 그대로 따라가 잡음까지 계수의 변화로 읽는다.

### 4.2 The Loop

Kalman filter 는 시점마다 예측과 갱신 두 단계를 반복하여 $\beta_t$ 의 추정치와 그 불확실성을 함께 갱신한다 [[2](#ref-2)]. Fig 1 이 그 한 바퀴이다.

```text
   [ estimate at t-1 ]
            |
            v
   +--------------------+   beta_hat(t|t-1) = beta_hat(t-1|t-1)
   | 1. Predict         |   P(t|t-1)        = P(t-1|t-1) + Q
   +--------------------+
            |
            v   y_t arrives
   +--------------------+   e_t = y_t - X_t beta_hat(t|t-1)
   | 2. Update          |   K_t = P(t|t-1) X_t' / (X_t P(t|t-1) X_t' + R)
   +--------------------+   beta_hat(t|t) = beta_hat(t|t-1) + K_t e_t
            |               P(t|t)        = (I - K_t X_t) P(t|t-1)
            v
   [ estimate at t ] --> next step
```

Fig 1. One step of the Kalman filter

예측 단계는 직전 추정치를 그대로 옮기고, 그 추정치의 분산 $P$ 에만 $Q$ 를 더한다. 계수가 무작위 보행을 한다고 두었으므로 다음 값에 대한 최선의 추측은 직전 값이며, 다만 그동안 움직였을 수 있으니 확신은 줄어든다.

갱신 단계는 실제 관측 $y_t$ 가 도착한 뒤 예측 오차 $e_t$ 를 계산하고, 그 오차를 얼마나 반영할지를 Kalman gain $K_t$ 로 정한다. $K_t$ 는 두 불확실성의 비율이다. 관측 잡음 $R$ 이 크면 $K_t$ 가 작아져 새 관측을 덜 믿고 기존 예측을 지키며, 추정치의 분산 $P$ 가 크면 $K_t$ 가 커져 새 관측을 적극 반영한다.

### 4.3 Filtering And Smoothing

Filter 는 $t$ 시점까지의 정보만으로 $\beta_t$ 를 추정한다. 실시간으로 판단해야 하는 자리에는 이것이 맞지만, 자료를 모두 모아 놓고 지나간 계수의 궤적을 그리는 자리에서는 전체 표본을 쓰는 smoother 를 쓴다. 같은 시점의 추정이라도 뒤의 관측까지 반영하므로 궤적이 덜 흔들린다.

## 5. Where It Is Used

### 5.1 Non-Proportional Hazards

Cox 비례위험 model 은 위험비가 시간에 무관하게 일정하다는 가정 위에 서 있다. 어떤 변수가 이 가정을 위반하면, 즉 위험비가 추적 기간 동안 변하면, 그 변수의 계수를 $\beta(t)$ 로 확장하여 왜곡을 막는다. 가정 위반은 Schoenfeld 잔차를 시간에 대해 회귀하여 검정하며, 그 잔차의 기울기가 곧 $\beta(t)$ 가 어느 방향으로 움직이는지를 알려 준다 [[3](#ref-3)].

### 5.2 TVP-VAR

거시경제 자료에서는 변수들 사이의 관계 자체가 시대에 따라 달라진다. TVP-VAR 은 VAR 의 계수를 4.1 의 상태공간 형태로 두어, 정책 기조나 시장 구조가 바뀔 때 계수가 어떻게 이동했는지를 추정한다 [[4](#ref-4)]. 계수뿐 아니라 충격의 분산까지 함께 시변으로 두는 것이 보통이며, 그래야 계수의 변화와 잡음 크기의 변화가 서로 섞이지 않는다.

## 6. Strengths And Limits

Table 3. What the model gains and what it costs

| Gain | Cost |
|------|------|
| Temporal change of the effect, read directly off the trajectory | More parameters, so a higher risk of overfitting |
| A way past a violated proportional hazards assumption | A flexible curve that resists a one-sentence interpretation |
| A basis for timing an intervention | A need for long enough follow-up and enough data |

세 가지 비용은 모두 같은 뿌리를 가진다. 계수를 시점마다 두면 자유도가 표본 크기만큼 늘어나므로, 그것을 무엇으로 묶어 둘지가 이 model 의 실제 설계 문제이다. Spline 이라면 벌점의 세기, 상태공간이라면 $Q$ 가 그 묶는 장치이며, 둘 중 어느 쪽이든 그 값을 자료로 정할 때는 검증 구간이 학습 시점 이후에 있어야 한다.

## References

<a id="ref-1"></a>[1] Hastie, T. and Tibshirani, R., "[Varying-Coefficient Models](https://doi.org/10.1111/j.2517-6161.1993.tb01939.x)", Journal of the Royal Statistical Society: Series B (Methodological), 55(4), 757-779, 1993.

<a id="ref-2"></a>[2] Kalman, R. E., "[A New Approach to Linear Filtering and Prediction Problems](https://doi.org/10.1115/1.3662552)", Journal of Basic Engineering, 82(1), 35-45, 1960.

<a id="ref-3"></a>[3] Grambsch, P. M. and Therneau, T. M., "[Proportional hazards tests and diagnostics based on weighted residuals](https://doi.org/10.1093/biomet/81.3.515)", Biometrika, 81(3), 515-526, 1994.

<a id="ref-4"></a>[4] Primiceri, G. E., "[Time Varying Structural Vector Autoregressions and Monetary Policy](https://doi.org/10.1111/j.1467-937X.2005.00353.x)", The Review of Economic Studies, 72(3), 821-852, 2005.

---

## Appendix A. Terminology

- **basis function**: 곡선을 몇 개의 정해진 함수의 가중합으로 적을 때 그 정해진 함수 하나. 기저함수.
- **Cox proportional hazards model**: 위험비가 시간에 무관하게 일정하다고 두고 생존 시간을 설명하는 회귀 model.
- **GAM**: Generalized Additive Model. 각 설명변수의 효과를 매끄러운 함수로 두고 그 합으로 응답을 설명하는 model.
- **hazard ratio**: 두 집단의 순간 위험률의 비. 위험비.
- **Kalman filter**: 관측 잡음이 있는 자료에서 관측되지 않는 상태를 예측과 갱신의 반복으로 추정하는 알고리즘.
- **Kalman gain**: 갱신 단계에서 예측 오차를 얼마나 반영할지 정하는 가중치.
- **PLS**: Partial Least Squares. 응답과의 공분산이 큰 방향으로 투영하는 지도 학습형 축약.
- **Schoenfeld residual**: Cox model 에서 사건 시점마다 관측된 공변량과 그 기대값의 차이. 비례위험 가정의 검정에 쓰인다.
- **score**: 관측을 PLS 성분 방향으로 투영한 값.
- **smoother**: 전체 표본을 모두 쓴 뒤 각 시점의 상태를 다시 추정하는 절차.
- **spline**: 구간마다 다항식을 잇되 이음매에서 매끄럽게 맞춘 곡선.
- **state space model**: 관측 방정식과 상태 방정식의 쌍으로 자료를 정의하는 model.
- **TVC**: Time-Varying Coefficient. 회귀계수를 시간의 함수로 둔 model.
- **TVP**: Time-Varying Parameter. 계수를 상태공간의 상태 변수로 둔 TVC 의 이산 시간 형태.
- **VAR**: Vector Autoregression. 여러 계열이 서로의 과거에 회귀하는 model.

## Appendix B. Applying TVP To A PLS Model

TVP 를 PLS 위에 얹을 수 있는가 — 얹을 수 있고, 방법은 둘이다.

Table 4. Two ways to make a PLS model time-varying

| Approach | What moves | When it fits |
|----------|------------|--------------|
| TVP on the scores | The regression from scores to $y$ | Latent directions stable, only their effect drifting |
| Adaptive PLS | The projection itself, refitted or updated | New $X$ no longer looking like the old $X$ |

앞의 것을 권한다. PLS 는 $X$ 를 응답과의 공분산이 큰 방향으로 투영하여 성분 수만큼의 score 로 줄이는데, TVP 를 그 score 위에 얹으면 추정할 상태가 성분 수에 절편 하나를 더한 개수로 묶인다. 원래 변수 위에 바로 얹으면 상태가 변수 수만큼 필요하고, 변수들이 서로 강하게 상관된 자료에서 그 상태들은 개별적으로 식별되지 않는다. 즉 PLS 가 차원을 줄이는 일을, TVP 가 그 줄어든 공간에서 시간을 다루는 일을 맡는 분담이다.

이 구성에는 성격이 다른 parameter 가 두 벌 있고, 둘의 취급이 정반대이다. Table 5 가 그 구분이며, 이름의 time-varying 은 아래쪽 행을 가리킨다.

Table 5. What stays fixed and what varies with time

| Quantity | Fitted on | Moves with time |
|----------|-----------|-----------------|
| Projection weights of the PLS model | The early segment, once | No |
| Coefficient from the scores to $y$ | The whole series, by the filter | Yes |

Score 에서 $y$ 로 가는 계수가 이 model 에서 시간에 따라 변하는 유일한 대상이며, 관측이 도착할 때마다 4.2 의 loop 이 그것을 갱신한다. 자료가 아무리 쌓여도 다시 적합하지 않는 쪽은 투영 가중치이다.

투영 가중치까지 함께 갱신하지 않는 데에는 이유가 있다. 예측이 score 와 계수의 곱이므로, 둘을 동시에 움직이면 투영을 두 배로 키우고 계수를 절반으로 줄인 것이 원래 것과 똑같은 예측을 낸다. 그러면 계수의 궤적이 효과가 변한 것인지 투영이 돌아간 것인지 구분되지 않아, 시변 계수를 읽겠다는 목적 자체가 사라진다.

구성은 세 단계이다. 초기 구간으로 PLS 를 적합하여 투영을 고정하고, 전체 구간을 그 투영으로 score 로 바꾸고, score 를 설명변수로 하는 상태공간 model 을 Kalman filter 로 추정한다. 아래는 `statsmodels` 의 `MLEModel` 로 그 상태공간을 정의하고 PLS score 에 적용한 예이며, `obs_cov` 와 `state_cov` 를 제곱으로 두어 분산이 음수가 되지 않게 한다.

```python
# Python
import numpy as np
import statsmodels.api as sm
from sklearn.cross_decomposition import PLSRegression


class TVPRegression(sm.tsa.statespace.MLEModel):
    """Random-walk time-varying coefficients on a fixed design matrix."""

    def __init__(self, endog, exog):
        exog = np.asarray(exog, dtype=float)
        if exog.ndim != 2:
            raise ValueError(f"exog must be 2-D, got shape {exog.shape}")
        k = exog.shape[1]
        super().__init__(endog, k_states=k, initialization="diffuse")
        self["design"] = exog.T.reshape(1, k, -1)
        self["transition"] = np.eye(k)
        self["selection"] = np.eye(k)

    @property
    def start_params(self):
        return np.r_[1.0, np.full(self.k_states, 0.1)]

    def update(self, params, **kwargs):
        params = super().update(params, **kwargs)
        self["obs_cov", 0, 0] = params[0] ** 2
        self["state_cov"] = np.diag(params[1:] ** 2)


# Fix the projection on the early segment, then let only its effect move.
pls = PLSRegression(n_components=2, scale=True).fit(X_early, y_early)
design = np.column_stack([np.ones(len(y)), pls.transform(X)])
result = TVPRegression(y, design).fit(disp=False)
beta = result.smoothed_state.T          # one coefficient trajectory per column
```

$Q$ 를 최대가능도로 추정하게 두면 자료가 계수의 이동 속도를 스스로 정한다. 추정된 $Q$ 가 0 에 가깝게 나오면 그것 자체가 답이다. 계수가 움직인다는 증거가 자료에 없다는 뜻이므로, 상수 계수 PLS 를 그대로 쓰면 된다.

한 가지 주의가 있다. 투영을 초기 구간으로 고정했으므로, 새로 들어온 $X$ 가 이전 $X$ 와 다르게 생기면 score 의 의미가 달라지고 그 위의 $\beta_t$ 는 해석을 잃는다. Score 의 분산이나 잔차가 후반부에서 체계적으로 커지는지를 확인하고, 커진다면 Table 4 의 두 번째 방법으로 옮겨야 한다.
