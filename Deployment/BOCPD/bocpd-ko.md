# Bayesian Online Change Point Detection (BOCPD)
Version 0 | Created: 2026-11-08 | Updated: 2026-11-08 02:51 CST

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Comparison](#31-comparison)
- [4. Principle](#4-principle)
  - [4.1 Run Length](#41-run-length)
  - [4.2 Run Length Recursion](#42-run-length-recursion)
  - [4.3 Predictive Distribution](#43-predictive-distribution)
  - [4.4 Hazard Function](#44-hazard-function)
  - [4.5 Decision Rule](#45-decision-rule)
- [5. Application](#5-application)
  - [5.1 VM Residual Monitoring](#51-vm-residual-monitoring)
  - [5.2 Settings](#52-settings)
  - [5.3 Breaking Conditions](#53-breaking-conditions)
  - [5.4 Demo Script](#54-demo-script)
- [6. Further Work](#6-further-work)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Derivation](#appendix-b-derivation)
  - [B.1 Change Probability under Constant Hazard](#b1-change-probability-under-constant-hazard)
  - [B.2 Normal-Gamma Update](#b2-normal-gamma-update)

## 1. Purpose

- **Problem Statement**: 반도체 공정의 tool shift, chamber drift, recipe 변경은 VM (Virtual Metrology) 잔차의 평균이나 분산을 바꾸지만, 한 점이 관리 한계를 넘는지만 보는 판정은 일시적인 outlier 와 지속되는 체제 변화를 가리지 못한다.
- **Goal**: 잔차가 wafer 하나씩 들어올 때마다 "마지막 변화점 이후 몇 장째인가" 의 사후 확률을 갱신하고, 그 확률로 변화점을 판정하는 BOCPD 의 원리, 판정 규칙, 설정값, 깨지는 조건을 제시한다.
- **Non-Goal**: 변화점을 찾은 뒤 VM model 을 다시 학습하거나 보정하는 절차는 다루지 않는다.

## 2. Summary

BOCPD 는 Adams 와 MacKay 가 2007 년에 제안한 알고리즘으로, 새 관측 $x_t$ 가 들어올 때마다 run length $r_t$ 의 사후 분포 $P(r_t \mid x_{1:t})$ 를 베이즈 정리로 한 단계씩 갱신한다 [[1](#ref-1)].
Constant hazard 에서는 $P(r_t = 0 \mid x_{1:t})$ 가 언제나 hazard $H$ 와 같으므로, 판정은 $w$ 장 뒤의 $P(r_t \le w \mid x_{1:t})$ 나 MAP run length 의 급락으로 한다.
표준 Gaussian BOCPD 는 큰 outlier 하나도 변화점으로 판정하므로, 오탐을 줄이려면 판정 지연 $w$ 와 robust likelihood 를 함께 정한다.

## 3. Taxonomy and its Hierarchy

변화점 탐지 방법은 처리 방식, 추론 방식, 출력 형태의 세 축으로 갈리고, BOCPD 는 online, Bayesian, 확률 출력에 해당한다.

<a id="fig-1"></a>

```text
Change point detection
|
+-- Processing : offline (whole series)    | online (one sample at a time)
+-- Inference  : frequentist statistic     | Bayesian posterior
+-- Output     : alarm time (point)        | probability of change
```

Fig 1. Taxonomy axes of change point detection

- Processing: offline 은 series 전체를 받은 뒤 분할하고 (PELT 등), online 은 관측 하나가 들어올 때마다 판정한다.
- Inference: frequentist 방법은 통계량이 관리 한계를 넘는지 보고, Bayesian 방법은 변화점 위치의 사후 확률을 계산한다.
- Output: 경보 시점 하나를 주는지, 변화가 일어났을 확률을 주는지로 갈린다.

Online 방법을 가정이 강한 것부터 차례로 놓으면 아래 Fig 2 이다.

<a id="fig-2"></a>

```text
Shewhart chart        known in-control mean and sigma; judges one point
  |  drops: memoryless judgement (accumulates past deviations)
  v
EWMA / CUSUM          known in-control mean and sigma; shift size set in advance
  |  drops: known parameters (learns them per regime from a prior)
  v
BOCPD                 unknown parameters per regime; posterior over run length
  |  drops: Gaussian likelihood sensitive to outliers
  v
Robust BOCPD          beta-divergence or heavy-tailed likelihood
```

Fig 2. Hierarchy of online change point detection methods

- 한 단계 내려갈 때마다 가정 하나를 버리고, 그 대가로 계산량과 설정할 prior 가 늘어난다.
- Shewhart chart 는 현재 점 하나만 보므로 작은 평균 이동을 늦게 잡는다. EWMA 와 CUSUM 은 과거 편차를 누적해 작은 이동을 잡지만, 관리 상태의 평균과 표준편차를 미리 알아야 한다.
- BOCPD 는 체제마다 parameter 를 모른다고 두고 prior 에서 학습하며, 출력이 run length 의 확률 분포이다.
- Robust BOCPD 는 BOCPD 의 Gaussian likelihood 가 outlier 하나에 크게 반응하는 약점을 β-divergence 로 줄인다 [[2](#ref-2)].

### 3.1 Comparison

다섯 방법을 Fig 2 의 계층 위에서 견주면 아래 Table 1 이다.

Table 1. Online change point detection methods

| Method       | Statistic                              | Settings                    | Output                  |
| :----------: | :------------------------------------: | :-------------------------: | :---------------------: |
| Shewhart     | 현재 점의 관리 한계 이탈 여부          | 관리 한계 (보통 3σ)         | 경보 시점 (point)       |
| EWMA         | 지수 가중 이동평균의 관리 한계 이탈    | 가중치, 관리 한계 폭        | 경보 시점 (point)       |
| CUSUM        | 누적 편차합의 결정 구간 h 초과         | 기준값 k, 결정 구간 h       | 경보 시점 (point)       |
| BOCPD        | run length 의 사후 분포                | hazard H, 관측 분포의 prior | 변화 확률 (probability) |
| Robust BOCPD | β-divergence 로 갱신한 run length 분포 | H, prior, β                 | 변화 확률 (probability) |

- 다섯 방법 모두 설정값이 있다. BOCPD 의 설정값은 기대 run length $\lambda = 1/H$ 와 관측 분포의 prior 이며, 둘 다 "평균 몇 장마다 변화가 오는가", "잔차의 크기는 대략 얼마인가" 처럼 공정 지식으로 정할 수 있는 양이다.
- EWMA 와 CUSUM 은 관리 상태의 평균과 표준편차를 고정한 채 판정하고, BOCPD 는 새 체제가 시작될 때마다 평균과 분산을 다시 추정한다.
- 출력이 확률이면 경보 문턱을 "변화 확률 0.9 이상" 처럼 공정 위험도에 맞춰 정할 수 있다.

## 4. Principle

BOCPD 는 run length 의 사후 분포를 관측 하나마다 한 번의 곱셈과 정규화로 갱신한다 [[1](#ref-1)].

이 꼭지에서 쓰는 기호는 아래이다.

- $t$: 관측 순번 (wafer 번호), $t = 1, 2, \ldots$
- $x_t$: 시점 $t$ 의 관측 하나 (예: 잔차 $e_t$), $x_{1:t}$ 는 $x_1$ 부터 $x_t$ 까지의 관측
- $r_t$: 시점 $t$ 의 run length, 0 이상의 정수
- $H$: hazard, 다음 관측에서 변화가 일어날 확률, $0 \lt H \lt 1$
- $\theta$: 한 체제 안에서 관측이 따르는 분포의 parameter (Gaussian 이면 평균과 분산)
- $\pi_t^{(r)}$: run length 가 $r$ 인 체제의 관측으로 본 $x_t$ 의 predictive 확률

### 4.1 Run Length

> **Definition: Run length**<br>
> Run length $r_t$ 는 시점 $t$ 에서 마지막 변화점 이후 현재 체제에 속한 관측의 개수이다.

- 변화가 없으면 $r_t = r_{t-1} + 1$ 로 하나 늘어난다 (growth).
- 시점 $t$ 에서 변화가 일어나면 $r_t = 0$ 으로 돌아간다 (change).
- BOCPD 는 $r_t$ 를 하나로 정하지 않고, $r_t = 0, 1, \ldots, t$ 의 확률 분포 $P(r_t \mid x_{1:t})$ 를 유지한다.

### 4.2 Run Length Recursion

$x_t$ 가 들어오면 joint 확률 $P(r_t, x_{1:t})$ 를 두 경우로 나누어 계산한다 [[1](#ref-1)].
Growth 는 run length 가 $r_{t-1}$ 인 체제가 $x_t$ 를 예측한 확률에 변화가 없을 확률 $1 - H$ 를 곱한다.

```math
P(r_t = r_{t-1} + 1,\ x_{1:t}) = P(r_{t-1},\ x_{1:t-1})\ \pi_t^{(r_{t-1})}\ (1 - H) \hspace{19em} (1)
```

Change 는 모든 $r_{t-1}$ 에 대해 같은 곱에 $H$ 를 곱해 더한다.

```math
P(r_t = 0,\ x_{1:t}) = \sum_{r_{t-1}} P(r_{t-1},\ x_{1:t-1})\ \pi_t^{(r_{t-1})}\ H \hspace{19em} (2)
```

사후 분포는 식 (1) 과 식 (2) 를 모든 $r_t$ 에 대해 더한 evidence $P(x_{1:t})$ 로 나눈 값이다.

```math
P(r_t \mid x_{1:t}) = \frac{P(r_t,\ x_{1:t})}{\sum_{r_t} P(r_t,\ x_{1:t})} \hspace{19em} (3)
```

시작은 $P(r_0 = 0) = 1$ 이다. 관측 하나마다 run length 의 후보가 하나 늘어나므로, 시점 $t$ 의 계산량과 memory 는 $O(t)$ 이다.

절차를 pseudocode 로 적으면 아래이다.

```text
# Pseudocode
P[0] = 1                                   # P(r_0 = 0)
for each new x_t:
    pi[r]     = predictive(x_t | stats[r])  # for every run length r
    growth[r+1] = P[r] * pi[r] * (1 - H)    # Eq. (1)
    change      = sum_r P[r] * pi[r] * H    # Eq. (2)
    P = normalize([change] + growth)        # Eq. (3)
    stats = [prior] + update(stats, x_t)    # new run starts from the prior
    decide with P(r_t <= w)                 # Section 4.5
```

### 4.3 Predictive Distribution

Run length 마다 그 체제에 속한 관측만으로 $\theta$ 의 사후 분포를 두고, 그 분포로 $x_t$ 의 predictive 확률 $\pi_t^{(r)}$ 를 계산한다.
켤레 사전 분포 (conjugate prior) 를 쓰면 사후 분포가 prior 와 같은 계열에 남아, 충분통계량 몇 개만 갱신하면 된다.

평균과 분산을 모두 모르는 Gaussian 관측에는 Normal-Gamma prior 를 쓴다. 분산 대신 정밀도 (분산의 역수) 에 Gamma 를 두는 형태이며, 분산에 Inverse-Gamma 를 두는 Normal-Inverse-Gamma 와 같은 model 이다.
Prior 의 parameter 는 $\mu_0$ (평균의 사전값), $\kappa_0$ (그 평균에 주는 가상 관측 수), $\alpha_0$, $\beta_0$ (정밀도의 Gamma 분포 shape 와 rate) 이다.
이때 predictive 분포는 자유도 $2\alpha$, 위치 $\mu$, 척도 제곱 $\beta(\kappa + 1)/(\alpha\kappa)$ 인 Student-t 분포이다.

```math
\pi_t^{(r)} = \mathrm{St}\left(x_t \;\middle|\; 2\alpha_r,\ \mu_r,\ \frac{\beta_r(\kappa_r + 1)}{\alpha_r \kappa_r}\right) \hspace{19em} (4)
```

$\mu_r, \kappa_r, \alpha_r, \beta_r$ 는 run length $r$ 인 체제의 관측 $r$ 개로 갱신한 parameter 이고, 갱신식은 [Appendix B.2](#b2-normal-gamma-update) 에 있다.
새 체제 ($r_t = 0$) 는 prior 의 parameter 에서 다시 시작한다.

### 4.4 Hazard Function

> **Definition: Hazard function**<br>
> Hazard function $H(r)$ 는 run length 가 $r$ 까지 변화 없이 이어졌을 때 다음 관측에서 변화가 일어날 확률이다.

- Constant hazard $H(r) = 1/\lambda$ 는 변화 사이의 간격이 평균 $\lambda$ 인 기하 분포를 따른다고 두는 것이며, BOCPD 에서 가장 많이 쓰는 형태이다 [[1](#ref-1)].
- $H(r)$ 가 $r$ 과 함께 커지게 두면, 오래 이어진 체제일수록 변화 확률을 높게 본다. 부품 수명처럼 사용 시간이 길수록 교체가 가까워지는 경우에 쓴다.

### 4.5 Decision Rule

Constant hazard 에서는 $P(r_t = 0 \mid x_{1:t}) = H$ 가 관측과 상관없이 성립하므로, 이 값에 문턱을 두면 판정이 되지 않는다.
식 (2) 의 합이 식 (1) 의 합과 같은 항 $P(r_{t-1}, x_{1:t-1})\,\pi_t^{(r_{t-1})}$ 에 각각 $H$ 와 $1 - H$ 를 곱한 것이어서, 정규화하면 $H$ 만 남는다. 유도는 [Appendix B.1](#b1-change-probability-under-constant-hazard) 에 있다.
새 체제가 시작되었다는 증거는 $x_{t+1}, x_{t+2}, \ldots$ 가 들어와 짧은 run length 의 predictive 가 긴 run length 의 predictive 보다 높아지면서 쌓인다.

판정 규칙은 아래 둘 가운데 하나를 쓴다.

- **지연 판정**: $P(r_t \le w \mid x_{1:t})$ 가 문턱 $p^{\ast}$ 를 넘으면 시점 $t - r_t + 1$ 에서 변화가 일어났다고 판정한다. $w$ 는 판정을 미루는 관측 수이다.
- **MAP run length 급락**: 사후 확률이 가장 큰 run length 가 앞 시점보다 크게 줄면 변화점으로 판정한다.

$w$ 가 크면 outlier 하나에 덜 반응하지만 판정이 $w$ 장만큼 늦어진다.

## 5. Application

BOCPD 는 VM 잔차처럼 하나씩 들어오는 단변량 series 에서 tool shift 와 chamber drift 의 시점을 확률로 판정하는 데 쓴다.

### 5.1 VM Residual Monitoring

- **가정**: 잔차 $e_t = y_t - \hat{y}_t$ 가 한 체제 안에서 서로 독립이고 같은 Gaussian 분포를 따른다. $y_t$ 는 wafer $t$ 의 실측값 (RM, Real Metrology), $\hat{y}_t$ 는 VM 예측값이다.
- **판정 대상**: 파츠 교체나 chamber 상태 변화로 잔차의 평균이나 분산이 바뀐 시점.
- **만나는 곳**: 증착과 식각 공정에서 VM 을 운영하며 일부 wafer 만 실측하는 단계. 실측된 wafer 의 잔차가 들어올 때마다 BOCPD 를 한 번 갱신한다.
- 판정한 변화점은 VM model 의 재학습이나 bias 보정을 시작하는 신호로 쓴다.

### 5.2 Settings

Table 2. BOCPD settings

| Setting             | Meaning                                  | Choice                                                   |
| :-----------------: | :--------------------------------------: | :------------------------------------------------------: |
| $\lambda = 1/H$     | 변화 사이의 기대 관측 수                 | PM (Preventive Maintenance) 간격의 wafer 수 등 공정 기록 |
| $\mu_0$             | 잔차 평균의 사전값                       | 0 (VM 이 unbiased 일 때)                                 |
| $\kappa_0$          | $\mu_0$ 에 주는 가상 관측 수             | 1 (작을수록 새 체제의 평균을 빨리 따라감)                |
| $\alpha_0, \beta_0$ | 정밀도의 Gamma prior, 잔차 분산의 사전값 | $\beta_0/\alpha_0$ 를 관리 상태 잔차 분산에 맞춤         |
| $w$                 | 판정을 미루는 관측 수                    | 허용 지연과 outlier 빈도로 정함                          |
| $p^{\ast}$          | $P(r_t \le w)$ 의 판정 문턱              | 경보 비용과 놓침 비용으로 정함                           |

- Prior 가 약하면 ($\alpha_0, \beta_0$ 가 작으면) 새 체제의 predictive 분포가 넓어져, 관리 상태와 크게 다른 값 하나를 새 체제가 더 잘 예측한다. 그 결과 outlier 하나에도 MAP run length 가 1 로 떨어진다.
- Run length 후보가 시점마다 하나씩 늘어나므로, 오래 운영할 때는 확률이 아주 작은 run length 를 버리는 pruning 으로 계산량을 일정하게 묶는다 [[1](#ref-1)].

### 5.3 Breaking Conditions

- **Outlier**: 표준 Gaussian BOCPD 는 큰 outlier 하나를 변화점으로 판정하고, 그 앞의 긴 run length 의 확률을 되찾지 못한다. [꼭지 5.4](#54-demo-script) 의 예에서 outlier 하나의 $P(r_t \le 3)$ 은 0.984 로, 실제 3σ 평균 이동의 0.471 보다 높다. Robust likelihood 나 큰 $w$ 를 쓴다.
- **자기상관**: 잔차가 체제 안에서 자기상관을 가지면 독립 가정이 깨져, BOCPD 가 천천히 움직이는 drift 를 여러 개의 변화점으로 나눈다. 자기상관을 model 에 넣거나 (AR 관측 model 등) 잔차를 먼저 whitening 한다.
- **점진적 drift**: BOCPD 는 parameter 가 한 시점에 바뀌는 piecewise-constant model 이므로, 수백 장에 걸친 완만한 drift 는 늦게 잡거나 계단 여러 개로 나눈다.
- **Prior 와 hazard 의 오지정**: $\lambda$ 를 실제 변화 간격보다 크게 두면 판정이 늦어지고, $\beta_0/\alpha_0$ 를 실제 분산보다 작게 두면 오탐이 늘어난다.

### 5.4 Demo Script

[`src/bocpd_demo.py`](src/bocpd_demo.py) 는 식 (1) 부터 식 (4) 를 Normal-Gamma prior 와 constant hazard 로 구현하고, 합성 잔차 series 에 적용한 결과를 시점마다 출력한다.
Series 는 $N(0, 1)$ 60 개 다음 $N(3, 1)$ 40 개이고, $t = 31$ 의 값을 outlier 6 으로 바꾼 것이다. 설정은 $\lambda = 100$, $\mu_0 = 0$, $\kappa_0 = \alpha_0 = \beta_0 = 1$, $w = 3$ 이다.

```bash
python3 src/bocpd_demo.py --seed 0
```

출력에서 outlier 와 평균 이동 전후를 뽑으면 아래 Table 3 이다.

Table 3. Run length posterior around the outlier and the mean shift

| t   | x_t    | P(r_t = 0) | P(r_t ≤ 3) | MAP r_t |
| :-: | :----: | :--------: | :--------: | :-----: |
| 30  | 0.220  | 0.0100     | 0.023      | 30      |
| 31  | 6.000  | 0.0100     | 0.984      | 1       |
| 32  | -0.209 | 0.0100     | 0.926      | 2       |
| 33  | -0.159 | 0.0100     | 0.733      | 3       |
| 61  | 2.564  | 0.0100     | 0.046      | 30      |
| 62  | 1.830  | 0.0100     | 0.052      | 31      |
| 63  | 4.739  | 0.0100     | 0.471      | 3       |

- $P(r_t = 0)$ 은 모든 시점에서 $H = 0.01$ 이며, [꼭지 4.5](#45-decision-rule) 의 성질이 출력에 그대로 나온다.
- $t = 31$ 의 outlier 에서 MAP run length 가 30 에서 1 로 떨어지고, 그 뒤에도 outlier 이전부터 이어진 run length 로 돌아가지 않는다. 표준 BOCPD 가 outlier 하나를 변화점으로 판정한 것이다.
- $t = 61$ 의 평균 이동은 $t = 63$ 에서 MAP run length 3 으로 잡혀, 판정 지연이 2 장이다. 이때 $P(r_t \le 3)$ 은 0.471 로 outlier 보다 낮다.

## 6. Further Work

- **Robust BOCPD 적용**
  - 무엇을: Gaussian likelihood 대신 β-divergence 로 run length 분포를 갱신하는 robust BOCPD 를 VM 잔차에 적용한다 [[2](#ref-2)].
  - 왜 지금: [꼭지 5.4](#54-demo-script) 의 예에서 표준 BOCPD 가 outlier 하나에 실제 평균 이동보다 강하게 반응하므로, 오탐을 줄이는 것이 가장 큰 과제이다.
  - 무엇이 필요한가: 실측 잔차 series 와 tool shift, PM 이 일어난 시점의 기록.
- **다변량 확장**
  - 무엇을: 여러 계측 항목의 잔차를 함께 보는 다변량 Gaussian 관측 model 로 바꾼다.
  - 왜 지금: 한 chamber 에서 두께와 균일도처럼 여러 VM 을 함께 운영하면, 단변량 BOCPD 를 항목마다 돌린 판정이 서로 어긋난다.
  - 무엇이 필요한가: 항목 사이의 공분산을 추정할 만큼의 실측 wafer 수.

## References

<a id="ref-1"></a>
[1] Adams, R. P., & MacKay, D. J. C. (2007). [Bayesian Online Changepoint Detection](https://arxiv.org/abs/0710.3742). *arXiv preprint* arXiv:0710.3742.<br>
<a id="ref-2"></a>
[2] Knoblauch, J., Jewson, J., & Damoulas, T. (2018). [Doubly Robust Bayesian Inference for Non-Stationary Streaming Data with β-Divergences](https://arxiv.org/abs/1806.02261). *Advances in Neural Information Processing Systems 31*.

---

## Appendix A. Terminology

- **Changepoint**: 시계열에서 관측이 따르는 분포의 parameter (평균, 분산 등) 가 바뀌는 시점.
- **Conjugate prior**: 사후 분포가 사전 분포와 같은 계열에 남도록 likelihood 와 짝을 맞춘 사전 분포.
- **CUSUM (Cumulative Sum)**: 기준값 k 를 뺀 편차를 누적하고, 누적합이 결정 구간 h 를 넘는 시점을 경보로 판정하는 관리도.
- **EWMA (Exponentially Weighted Moving Average)**: 가중치로 과거 관측을 지수적으로 줄여 평균한 값이 관리 한계를 넘는 시점을 경보로 판정하는 관리도.
- **MAP (Maximum A Posteriori)**: 사후 확률이 가장 큰 값.
- **Outlier**: 한 시점에만 분포에서 크게 벗어나고 다음 관측부터는 원래 분포로 돌아오는 관측.
- **PELT (Pruned Exact Linear Time)**: series 전체를 받아 비용 함수를 최소로 하는 변화점 위치를 pruning 으로 계산하는 offline 알고리즘.
- **Predictive distribution**: parameter 의 사후 분포로 적분해 얻은 다음 관측의 분포.
- **Pruning**: 확률이 문턱보다 작은 run length 후보를 계산에서 버리는 것.
- **Shewhart chart**: 관측 하나가 관리 한계 (보통 평균 ± 3σ) 를 넘는지로 판정하는 관리도.
- **Tool shift**: 파츠 교체, PM, chamber 상태 변화 등으로 장비의 출력 수준이 한 시점에 바뀌는 것.
- **VM (Virtual Metrology)**: 장비 sensor 값으로 계측값을 예측하는 model. 잔차 $e_t$ 는 실측값 (RM, Real Metrology) 에서 VM 예측값을 뺀 값이다.
- **β-divergence**: 두 분포의 차이를 재는 divergence 의 한 계열로, 확률이 작은 관측의 영향을 β 에 따라 줄인다.

## Appendix B. Derivation

### B.1 Change Probability under Constant Hazard

$H$ 가 run length 와 상관없는 상수이면, 식 (1) 과 식 (2) 의 공통 항을 $a_r = P(r_{t-1} = r,\ x_{1:t-1})\,\pi_t^{(r)}$ 로 두어 식 (3) 의 분모를 아래처럼 쓴다.

```math
\sum_{r_t} P(r_t,\ x_{1:t}) = H \sum_{r} a_r + (1 - H) \sum_{r} a_r = \sum_{r} a_r \hspace{19em} (5)
```

식 (2) 의 분자는 $H \sum_r a_r$ 이므로, 식 (5) 로 나누면 $\sum_r a_r$ 이 약분된다.

```math
P(r_t = 0 \mid x_{1:t}) = \frac{H \sum_{r} a_r}{\sum_{r} a_r} = H \hspace{19em} (6)
```

$a_r$ 은 관측 $x_t$ 에 따라 바뀌지만 분자와 분모에 같이 들어 있어 결과에 남지 않는다. 그래서 변화의 증거는 $r_t = 0$ 이 아니라, 다음 시점들에서 짧은 run length 로 옮겨 가는 확률에 나타난다.

### B.2 Normal-Gamma Update

Run length $r$ 인 체제의 parameter $\mu_r, \kappa_r, \alpha_r, \beta_r$ 에 관측 $x_t$ 하나를 더하면, run length $r + 1$ 의 parameter 는 아래이다.

```math
\mu_{r+1} = \frac{\kappa_r \mu_r + x_t}{\kappa_r + 1} \hspace{19em} (7)
```

```math
\kappa_{r+1} = \kappa_r + 1 \hspace{19em} (8)
```

```math
\alpha_{r+1} = \alpha_r + \frac{1}{2} \hspace{19em} (9)
```

```math
\beta_{r+1} = \beta_r + \frac{\kappa_r (x_t - \mu_r)^2}{2(\kappa_r + 1)} \hspace{19em} (10)
```

$r = 0$ 의 parameter 는 prior $\mu_0, \kappa_0, \alpha_0, \beta_0$ 이다. 식 (7) 부터 식 (10) 은 [`src/bocpd_demo.py`](src/bocpd_demo.py) 의 `NormalGammaBocpd.update` 에 있다.
