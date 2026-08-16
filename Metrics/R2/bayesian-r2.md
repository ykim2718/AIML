# Bayesian R² — R² 를 점이 아니라 분포로 얻는 방법
Rev. 0 | Created: 2026-08-16 | Updated: 2026-08-16 18:15 CDT

> 사후표본마다 R² 를 하나씩 계산해 R² 의 분포를 얻는 방법을 정의, 계산 절차,
> 해석, 적용 조건의 순서로 정리한 문서.

## 1. Motivation

표준 R² 는 숫자 하나만 준다. `R² = 0.87` 이라는 값이 나왔을 때, 그 값이 데이터를
조금만 바꿔도 0.60 으로 떨어지는 값인지 아니면 0.85 근처에서 단단하게 유지되는
값인지를 그 숫자 자체는 말해주지 않는다. 설명력의 크기와 설명력에 대한 확신은
서로 다른 정보인데, 표준 R² 는 앞의 하나만 담는다.

Bayesian R² 는 모델의 예측 불확실성을 R² 자체로 전파해서, R² 를 점이 아니라
분포로 얻는다. 결과가 `0.87 [0.81, 0.91]` 형태로 나오는 이유가 이것이다.

**Table 1. Standard R² vs Bayesian R²**

| Aspect | Standard R² | Bayesian R² |
|---|---|---|
| Output | 스칼라 하나 | S 개 표본으로 이루어진 분포 |
| Uncertainty | 없음 | 구간으로 표현 |
| Range | 음수 가능, 1 초과 불가 | 항상 [0, 1] |
| Input required | 예측값 1 벌 | 사후표본 S 벌 |
| Failure mode | 값이 흔들려도 알 수 없음 | 표본을 못 뽑으면 계산 자체가 불가 |

## 2. Core Idea

사후분포에서 뽑은 표본 s 마다 R² 를 하나씩 계산한다. 표본이 S 개면 R² 도 S 개가
나오고, 그 S 개의 집합이 곧 R² 의 사후분포다. 모델 파라미터가 불확실하면 예측값이
표본마다 달라지고, 예측값이 달라지면 설명된 변동의 크기도 달라진다. Bayesian R² 는
그 흔들림을 지우지 않고 그대로 출력으로 내보낸다.

## 3. Definition

### 3.1. Divergence Problem of the Standard Form

표준 정의 `R² = 1 − SS_res / SS_tot` 를 사후표본에 그대로 적용하면 값이 [0, 1] 을
벗어난다. 최소제곱 적합에서는 잔차제곱합이 전체제곱합을 넘을 수 없지만, 사후표본
하나하나는 그 데이터에 대한 최소제곱해가 아니다. 사전분포나 계층모델의 shrinkage 가
예측값을 데이터에서 끌어당기면 개별 표본의 잔차제곱합이 전체제곱합을 넘어설 수
있고, 그 표본의 R² 는 음수가 된다. 표본별로 부호가 뒤집히는 값들을 모아 놓으면
분포의 중앙값도 구간도 해석할 수 없다.

### 3.2. Gelman Formulation

Gelman et al. (2019) 의 해법은 분모를 "두 양의 합" 으로 바꾸는 것이다. 뺄셈을
없애면 발산할 자리가 사라진다.

```text
             Var_fit^(s)                    explained variation
R2^(s) = ─────────────────────────  =  ────────────────────────────────────────
         Var_fit^(s) + Var_res^(s)      explained variation + residual variation
```

- `Var_fit` — 표본 s 의 예측값 `y_hat_i^(s)` 들이 데이터 포인트 i 를 따라 갖는 분산이며, 모델이 설명한 변동에 해당한다.
- `Var_res` — 표본 s 가 설명하지 못하고 남긴 변동이다.

두 항 모두 분산이므로 항상 0 이상이고, 분모가 두 양의 합이라 비율은 구조적으로
[0, 1] 안에 갇힌다. 분자가 0 이면 R² 는 0 이고, `Var_res` 가 0 으로 가면 R² 는 1 로
간다. 표준 R² 와 달리 어떤 사후표본에서도 정의를 벗어나지 않는다.

`Var` 는 데이터 포인트 i = 1 ... n 에 대한 표본분산이다.

$$\mathrm{Var}(z) = \frac{1}{n-1}\sum_{i=1}^{n}\left(z_i - \bar{z}\right)^2$$

### 3.3. Choice of Residual Variance

`Var_res` 를 잡는 방법은 두 가지이고, 어느 쪽을 쓰는지에 따라 R² 의 의미가 달라진다.

**Table 2. Two choices of residual variance**

| Variant | Definition | Meaning | Note |
|---|---|---|---|
| Empirical | `Var(y_i − y_hat_i^(s))` | 실제로 남은 잔차의 분산 | 관측치 y 가 필요하며 모델 종류를 가리지 않는다 |
| Model-based | `(sigma^(s))^2` | 모델이 스스로 주장하는 noise 분산 | Gaussian likelihood 계열에 쓰며 y 없이 계산된다 |

두 값은 모델이 잘 맞을 때 비슷해지고, 어긋나면 그 차이 자체가 진단 신호다.
model-based 쪽이 눈에 띄게 작으면 모델이 자기 noise 를 과소평가하고 있다는 뜻이며,
이때 model-based R² 는 실제보다 높게 나온다. 두 변형을 한 문서 안에서 섞어 쓰지 않고
어느 쪽인지 명시한다.

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

입력은 `(S, n)` 모양의 행렬 하나다. 행이 사후표본, 열이 데이터 포인트다. 분산은
열 방향으로, 즉 데이터 포인트를 따라 계산하며 표본을 따라 계산하지 않는다. 축을
바꿔 잡으면 "예측값이 표본마다 얼마나 흔들리는가" 를 재게 되어 R² 가 아닌 다른
양이 나온다.

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

계산 자체는 분산 두 개와 나눗셈 한 번이며, 비용은 사후표본을 얻는 단계에 있다.

## 5. Interpretation

### 5.1. Point Estimate and Credible Interval

R² 의 사후분포는 대칭이 아니다. R² 가 1 에 가까울수록 위쪽이 1 이라는 벽에 막혀
왼쪽으로 긴 꼬리가 생긴다. 이런 분포에서 평균은 꼬리에 끌려가므로 점추정에는
중앙값을 쓰고, 구간은 분위수로 잡는다.

구간의 폭이 곧 설명력에 대한 확신의 정도다. `0.87 [0.85, 0.89]` 와
`0.87 [0.61, 0.95]` 는 점추정이 같아도 완전히 다른 결과이며, 뒤쪽은 R² 를 근거로
모델을 채택하기에는 데이터가 부족하다는 뜻이다.

이 구간은 credible interval 이지 confidence interval 이 아니다. "R² 가 이 구간에 있을
확률이 90% 다" 로 직접 읽을 수 있는 것이 credible interval 의 성질이다.
정의는 [Appendix A. Terminology](#appendix-a-terminology) 를 참조한다.

### 5.2. Uncertainty Decomposition

한 결과 안에 두 종류의 불확실성이 분리되어 들어 있다.

- R² 분포의 폭 — 파라미터를 특정하지 못해 생기는 불확실성이며, 데이터를 더 모으면 줄어든다.
- `Var_res` 의 크기 — 데이터 자체에 있는 noise 이며, 데이터를 더 모아도 줄지 않는다.

따라서 구간이 넓으면 표본을 늘리는 것이 답이고, 구간은 좁은데 R² 자체가 낮으면
입력 변수나 모델 구조를 바꿔야 한다. 표준 R² 는 이 둘을 구분하지 못해 두 상황에
같은 숫자를 준다.

## 6. Tools

**Table 3. Available implementations**

| Environment | Interface | Note |
|---|---|---|
| R | `bayes_R2(fit)` — rstanarm, brms | 논문 저자가 관리하는 구현이며 기준 구현으로 삼는다 |
| R | `loo_R2(fit)` — rstanarm, brms | out-of-sample 보정을 적용한 변형 |
| Python | `az.r2_score()` — ArviZ | 예측 표본과 관측치를 받아 요약을 반환한다 |
| Any | 직접 구현 | 분산 두 개와 나눗셈이므로 이식 부담이 작다 |

직접 구현할 때는 기준 구현과 같은 데이터로 값을 맞춰 보고 시작한다. 값이 어긋나면
대개 3.3 의 변형 선택이나 4.1 의 축 방향이 원인이다.

## 7. Prerequisites

가장 중요한 제약은 정의가 아니라 적용 조건에 있다. Bayesian R² 는 진짜 사후표본을
뽑을 수 있어야 쓴다.

### 7.1. Applicable Models

**Table 4. Model applicability**

| Model type | Applicable | Reason |
|---|---|---|
| Stan, PyMC, brms 등 Bayesian model | Yes | MCMC 가 사후표본을 직접 준다 |
| Deep ensemble | Yes | member 하나가 표본 하나 역할을 한다 |
| MC dropout | Yes | forward pass 를 반복해 표본을 만든다 |
| Bootstrap ensemble | Yes | 재표본마다 적합해 표본을 만든다 |
| 단일 Gaussian head (mu, sigma 한 벌) | No | 표본이 없어 분포를 만들 수 없다 |
| 점추정만 내는 model (일반 GBM, 단일 신경망) | No | 예측값이 한 벌뿐이다 |

Deep ensemble 과 MC dropout 은 엄밀한 의미의 사후표본은 아니지만 예측 분포의 표본
역할을 하므로 같은 계산이 성립한다. 다만 member 수가 5 개 정도로 적으면 분위수가
불안정하므로 구간을 좁게 읽지 않는다.

### 7.2. Inapplicable Models

`mu` 와 `sigma` 를 한 벌만 내놓는 모델은 예측 분포는 있어도 사후 "표본" 이 없다.
`sigma` 로 aleatoric 불확실성은 표현되지만, R² 를 흔들 재료인 파라미터 불확실성이
없으므로 R² 는 어차피 값 하나로 고정된다. 이 경우에는 8. Comparison 에서 다루는
CRPS Skill Score 처럼 표본을 요구하지 않는 지표를 쓴다.

## 8. Comparison

**Table 5. Bayesian R² vs CRPS Skill Score**

| Aspect | Bayesian R² | CRPS Skill Score |
|---|---|---|
| Output | 점추정 + 구간 | 점수 하나 |
| Interval | 있음 | 없음 |
| Sample requirement | 사후표본 S 벌 필요 | 예측 분포만 있으면 됨 |
| Question answered | 설명력이 얼마이고 얼마나 확실한가 | 예측 분포가 얼마나 잘 맞았는가 |
| Baseline | 데이터의 전체 변동 | 명시적으로 지정한 baseline model |

두 지표는 경쟁 관계가 아니라 적용 조건이 다르다. 사후표본을 뽑을 수 있으면
Bayesian R² 가 구간까지 주므로 더 많은 정보를 담고, 표본을 뽑을 수 없으면 CRPS
Skill Score 가 남는 선택지다.

## 9. Pitfalls

**Table 6. Common mistakes**

| Mistake | Consequence | Fix |
|---|---|---|
| 사후예측표본을 `Var_fit` 에 사용 | noise 가 분자에 섞여 R² 가 부풀려짐 | 예측 평균의 사후표본을 쓴다 |
| 분산을 표본 축으로 계산 | R² 가 아닌 다른 양이 나옴 | 데이터 포인트 축으로 계산한다 |
| 평균으로 점추정 | 왜도에 끌려 값이 낮게 나옴 | 중앙값을 쓴다 |
| 학습 데이터로 계산한 값을 일반화 성능으로 보고 | 낙관적으로 편향됨 | out-of-sample 데이터를 쓰거나 loo 변형을 쓴다 |
| S 가 작은 상태에서 좁은 구간 보고 | 구간 자체가 불안정 | 표본 수를 늘리고 수렴을 먼저 확인한다 |
| 두 변형을 섞어 비교 | 모델 간 비교가 무의미해짐 | 한 변형으로 고정하고 명시한다 |

첫 번째 항목이 가장 자주 나온다. 대부분의 도구는 예측 평균의 사후표본과 관측
noise 까지 더한 사후예측표본을 모두 제공하는데, `Var_fit` 에 들어갈 것은 앞쪽이다.
뒤쪽을 넣으면 설명하지 못한 변동이 설명된 변동으로 옮겨 가서 R² 가 실제보다 높게
나온다.

## 10. Summary

Bayesian R² 는 사후표본 s 마다 `Var_fit / (Var_fit + Var_res)` 를 계산해 R² 를 분포로
얻는 방법이다. 분모가 두 양의 합이라 표준 R² 의 발산 문제가 구조적으로 사라지고,
점추정과 credible interval 을 함께 얻는다. 대신 사후표본을 뽑을 수 있는 Bayesian
model 이나 ensemble 계열에서만 쓸 수 있다.

---

## Appendix A. Terminology

- **aleatoric uncertainty** — 데이터 자체의 noise 에서 오는 불확실성이며, 데이터를 더 모아도 줄지 않는다.
- **credible interval** — 사후분포의 분위수로 정의한 구간이며, 파라미터가 그 구간에 있을 확률로 직접 해석한다.
- **CRPS** — Continuous Ranked Probability Score. 예측 분포 전체와 관측값 하나를 비교하는 점수이며, 값이 작을수록 좋다.
- **CRPS Skill Score** — CRPS 를 baseline model 의 CRPS 로 정규화해 `1 − CRPS_model / CRPS_baseline` 형태로 만든 지표이며, 값이 클수록 좋다.
- **deep ensemble** — 초기값이나 데이터 순서를 달리해 독립적으로 학습시킨 신경망 여러 개를 모아 예측 분포를 만드는 방법이다.
- **epistemic uncertainty** — 모델과 파라미터를 특정하지 못해 생기는 불확실성이며, 데이터를 더 모으면 줄어든다.
- **LOO** — Leave-One-Out cross-validation. 관측치 하나씩을 빼고 평가해 out-of-sample 성능을 추정하는 방법이다.
- **MC dropout** — 추론 단계에서도 dropout 을 켠 채 forward pass 를 반복해 예측 표본을 얻는 방법이다.
- **MCMC** — Markov Chain Monte Carlo. 사후분포에서 표본을 뽑는 표준 알고리즘 계열이다.
- **posterior distribution** — 데이터를 관측한 뒤의 파라미터 분포이며, 사후분포로 표기한다.
- **posterior draw** — 사후분포에서 뽑은 표본 하나이며, 사후표본으로 표기한다.
- **posterior predictive sample** — 사후표본에 관측 noise 까지 더해 생성한 관측값 수준의 표본이다.
- **shrinkage** — 사전분포나 계층 구조가 추정값을 전체 평균 쪽으로 끌어당기는 효과다.
