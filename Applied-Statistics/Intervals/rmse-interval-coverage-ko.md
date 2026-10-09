# Coverage of the RMSE Interval
Rev. 36 | Created: 2026-10-05 | Updated: 2026-10-09 16:30 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Placement](#31-placement)
- [4. RMSE Interval](#4-rmse-interval)
  - [4.1 Interval Calculation](#41-interval-calculation)
  - [4.2 Interval Coverage](#42-interval-coverage)
  - [4.3 Coverage Loss](#43-coverage-loss)
  - [4.4 Coverage Stated with 1.96](#44-coverage-stated-with-196)
  - [4.5 Degrees of Freedom with Fitted Parameters](#45-degrees-of-freedom-with-fitted-parameters)
  - [4.6 Conditions](#46-conditions)
- [5. Prediction Interval](#5-prediction-interval)
  - [5.1 The Multiplier for 95 Percent](#51-the-multiplier-for-95-percent)
  - [5.2 Leverage](#52-leverage)
  - [5.3 Conditions](#53-conditions)
- [6. Confidence Interval](#6-confidence-interval)
- [7. Tolerance Interval](#7-tolerance-interval)
- [8. Spec Setting for SPC](#8-spec-setting-for-spc)
  - [8.1 What the Limit Is Drawn On](#81-what-the-limit-is-drawn-on)
  - [8.2 The Limit from the RMSE](#82-the-limit-from-the-rmse)
  - [8.3 The Guard Band](#83-the-guard-band)
  - [8.4 The Spec and the Control Limit](#84-the-spec-and-the-control-limit)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Quantile Functions of the Normal and t Distributions](#appendix-b-quantile-functions-of-the-normal-and-t-distributions)
  - [B.1 Normal Distribution](#b1-normal-distribution)
  - [B.2 t Distribution](#b2-t-distribution)
- [Appendix C. Chi-squared Distribution](#appendix-c-chi-squared-distribution)
- [Appendix D. Derivation of Equations (4) to (8)](#appendix-d-derivation-of-equations-4-to-8)

## 1. Purpose

- **Problem Statement**: ML 의 metric 인 RMSE 의 물리적 의미를 이해하기 힘들다.
- **Goal**: model 의 오차인 RMSE 를 이용해 SPC 에서 spec 과 함께 쓸 한계 (error limit 과 guard band) 를 정하는 방법의 이론과 실행을 담은 guide 를 만든다.
- **Non-Goal**: 오차가 정규분포가 아닐 때 쓸 구간은 다루지 않는다.

## 2. Summary

RMSE 의 1.96 배로 그린 구간이 새 오차 하나를 담는 비율은 오차의 참 표준편차를 알 때 95.00% 이고, RMSE 를 오차 n 개에서 구했으면 그보다 낮으며, 94% 가 되는 것은 n 이 28 일 때다 (n 이 19 부터 55 까지면 94% 로 반올림된다).

새 오차를 RMSE 로 나눈 값이 자유도 n 의 Student t distribution 을 따르므로 비율이 95% 아래로 깎인다.

95% 를 지키려면 배율을 1.96 대신 그 t distribution 의 97.5% 점으로 둔다. n 이 28 이면 2.05 이고, n 이 커지면 1.96 으로 돌아간다. Model 의 오차에 긋는 한계 (error limit) 는 그 배율로 긋고, 예측값이 계측값을 대신해 판정에 쓰이면 이미 있는 spec 에서 guard band 만큼 안쪽으로 물러선다.

## 3. Taxonomy and its Hierarchy

구간의 크기는 model 의 오차인 RMSE 의 배율로 구한다. 구간의 두 끝은 중심에서 RMSE 의 배율 배만큼 떨어진 자리이며, $\sigma$ 를 아는 경우에는 RMSE 대신 그 참값에 배율을 곱한다. 배율을 정하는 것은 두 가지다. 오차의 표준편차 (scale) 를 아는가 추정하는가, 그리고 구간이 담는 대상 (covered quantity) 이 무엇인가이다. 아래 <a href="#fig-1">Fig 1</a> 이 그 두 축을 담는다.

```text
INTERVAL around a prediction:  center +/- k * scale
|
+-- axis 1: the scale
|   sigma                        the true error scale, known
|   RMSE of n errors             an estimate, itself random
|
+-- axis 2: the covered quantity
    one future error             prediction
    the mean of the errors       confidence
    a fraction of the population tolerance
```

<a id="fig-1"></a>
Fig 1. The two axes an interval around a prediction is placed on

첫째 축은 구간의 폭을 정하는 오차의 표준편차가 참값인가 추정값인가를 가른다. 둘째 축은 같은 폭에 어떤 확률을 붙이는가를 가르며, 담는 대상이 오차 하나에서 오차의 평균으로, 다시 모집단의 비율로 옮겨 간다.

두 축 위의 구간은 가정의 세기에 따라 계층을 이루며, 아래 <a href="#fig-2">Fig 2</a> 가 그 계층을 담는다.

```text
HIERARCHY of the intervals, with what each step down changes

interval                    scale          multiplier        coverage of the next error
Normal interval             σ, known       z(1 - α/2)        1 - α for every n
|
| assumption dropped: σ is known. The RMSE of n errors takes its place.
|
+-- RMSE interval           RMSE           z(1 - α/2)        below 1 - α
|                                                            0.9400 at α = 0.05, n = 28
+-- prediction interval     RMSE           t(ν, 1 - α/2)     1 - α for every n
    |
    | requirement added: the covered fraction must hold with a stated confidence
    |
    +-- tolerance interval  RMSE           tolerance factor  a stated fraction of the
                                                             population, held with a
                                                             stated confidence
```

<a id="fig-2"></a>
Fig 2. The hierarchy the intervals form, and the assumption each step drops or the requirement it adds

- **σ**: 오차의 참 표준편차. Normal interval 은 그 값을 아는 경우이고, RMSE interval 과 prediction interval 과 tolerance interval 은 모르는 경우다.
- **RMSE**: 오차 n 개로 구한 $\sigma$ 의 추정값. 식 (1) 이 정의다.
- **n**: Model 의 예측값과 계측값을 짝지어 오차를 구한 횟수. 식 (1) 에서 사용한 n 개다.
- **ν**: RMSE 의 제곱합에 남은 자유도. 그림은 parameter 를 추정하지 않아 $\nu = n$ 인 경우다.
- **multiplier**: 표준편차에 곱하는 수. 구간의 두 끝이 중심에서 그만큼 떨어진다.
- **coverage**: Model 이 다음에 한 번 더 예측했을 때 그 오차가 구간 안에 들 확률. 그 오차는 RMSE 를 구하던 때에는 아직 생기지 않았으므로 RMSE 를 구한 n 개와 독립이고, 문서는 이것을 새 오차라 적는다.
- **1 - α**: 구간이 담을 확률로 정한 값. $\alpha$ 는 그 나머지이고, 이 문서는 $\alpha = 0.05$ 를 쓴다.
- **z**: 표준정규분포의 분위수 함수 (quantile function, inverse CDF). 식 (3) 의 누적분포함수 $\Phi$ 의 역함수여서 $z = \Phi^{-1}$ 이고, $z(q)$ 는 누적확률이 $q$ 가 되는 지점이다. $z(1 - \alpha/2)$ 는 양쪽 꼬리에 합쳐 $\alpha$ 를 남기며, $\alpha = 0.05$ 에서 1.96 이다. [Appendix B](#appendix-b-quantile-functions-of-the-normal-and-t-distributions) 가 이것을 자세히 적는다.
- **t**: 자유도 $\nu$ 인 Student t distribution 의 분위수 함수. $t(\nu, q)$ 는 누적확률이 $q$ 가 되는 지점이고, $t(\nu, 1 - \alpha/2)$ 는 $\alpha = 0.05$ 와 $n = 28$ 에서 2.05 다.
- **tolerance factor**: 모집단의 어느 비율까지 담을지와 그것을 보장하는 신뢰수준, 그리고 n 으로 정해지는 배율. 신뢰수준을 0.5 보다 높게 두면 prediction interval 의 배율보다 크다.
- 그림의 $n = 28$ 은 담는 확률이 꼭 0.9400 이 되는 오차 개수다. 다른 n 에서 나오는 값은 Table 2 에 있다.

계층은 한 단계 내려갈 때마다 아는 것을 내려놓거나 요구를 하나 더 붙인다. 맨 위는 오차의 표준편차를 알고, 그 아래는 표준편차를 n 개의 오차로 추정하며, 맨 아래는 추정한 표준편차로 모집단의 비율까지 말한다. $\sigma$ 를 안다는 가정을 버리면 <a href="#fig-2">Fig 2</a> 의 둘째 줄 (RMSE interval) 과 셋째 줄 (prediction interval) 로 갈린다. 배율을 $z(1 - \alpha/2)$ 에 그대로 두면 담는 확률이 $1 - \alpha$ 아래로 떨어지고, $1 - \alpha$ 를 지키려면 배율을 식 (2) 의 $\sqrt{n/\nu}\; t_{\nu}(1 - \alpha/2)$ 로 키우며, $\nu = n$ 이면 $t_{n}(1 - \alpha/2)$ 다. 넷째 줄 (tolerance interval) 은 담는 확률에 신뢰수준을 하나 더 붙이며, 그 신뢰수준을 0.5 보다 높게 두면 배율이 셋째 줄보다 커진다.

### 3.1 Placement

Placement 는 <a href="#fig-1">Fig 1</a> 의 두 축과 <a href="#fig-2">Fig 2</a> 의 계층 위에서 각 구간이 어느 자리에 놓이는지를 뜻한다. Table 1 은 구간마다 쓰는 표준편차와 곱하는 배율과 확률이 가리키는 대상을 적으며, 손에 있는 자료에 어느 구간을 쓸지는 이 표에서 고른다.

Table 1. Intervals around a prediction

| #   | Interval            | Scale          | Multiplier                             | Covered quantity       | Probability                                |
| :-: | :-----------------: | :------------: | :------------------------------------: | :--------------------: | :----------------------------------------: |
| 1   | Normal interval     | $\sigma$       | $z(1 - \alpha/2)$                      | 새 오차 하나           | $1 - \alpha$                               |
| 2   | RMSE interval       | RMSE           | $z(1 - \alpha/2)$                      | 새 오차 하나           | [식 (7) 의 값](#4-rmse-interval)           |
| 3   | Prediction interval | RMSE           | $\sqrt{n/\nu}\; t_{\nu}(1 - \alpha/2)$ | 새 오차 하나           | [$1 - \alpha$](#5-prediction-interval)     |
| 4   | Confidence interval | $s / \sqrt{n}$ | $t_{n-1}(1 - \alpha/2)$                | 오차의 평균            | [$1 - \alpha$](#6-confidence-interval)     |
| 5   | Tolerance interval  | RMSE           | tolerance factor                       | 모집단의 비율 $P$ 이상 | [신뢰수준 $\gamma$](#7-tolerance-interval) |

$\alpha = 0.05$ 에서 1 행과 2 행의 배율이 1.96, 3 행의 배율이 $\sqrt{n/\nu}\; t_{\nu}(0.975)$, 4 행의 배율이 $t_{n-1}(0.975)$ 다. 1 행은 $\sigma$ 를 아는 경우에만 쓸 수 있고, 2 행부터는 모두 표본에서 구한 값으로 $\sigma$ 를 대신한다. 2 행과 3 행은 같은 표준편차에 다른 배율을 곱한 것이며, 3 행의 배율이 식 (2) 대로 자유도 $\nu$ 의 t distribution 에서 나온 값이라 비율이 n 과 무관하게 $1 - \alpha$ 로 유지된다. 4 행은 폭이 3 행의 약 $1/\sqrt{n}$ 이고, 담는 대상이 새 오차가 아니라 오차의 평균이므로 2 행과 바꾸어 쓸 수 없다. 4 행의 $s$ 는 식 (1) 의 RMSE 와 달리 오차의 평균을 뺀 뒤 n - 1 로 나누어 구한 표준편차이고, 자유도는 n - 1 이다. 5 행은 다른 네 행과 확률의 뜻이 다르다. 3 행은 새 오차 하나가 구간에 들 확률이 평균적으로 $1 - \alpha$ 라는 뜻이고, 5 행은 구간이 모집단의 $P$ 이상을 담는다는 것을 신뢰수준 $\gamma$ 로 보장한다는 뜻이어서 담는 비율에 다시 확률이 붙는다. 배율은 $P$ 와 $\gamma$ 와 n 의 세 값으로 정해지는 tolerance factor 다 [[1](#ref-1)]. Tolerance factor 는 닫힌 형태가 없어 세 값의 조합마다 표에서 읽는다 [[2](#ref-2)].

## 4. RMSE Interval

구간을 긋는 식과 그 구간이 담는 확률을 구하는 식이 이 꼭지에 있다. 구간은 식 (2), 담는 확률은 식 (7) 이며, 모르는 $\sigma$ 는 유도 과정에서 약분되어 확률이 오차의 개수 n 으로만 정해진다.

### 4.1 Interval Calculation

구간은 중심에서 RMSE 의 배율 배만큼 떨어진 두 자리이고, RMSE 는 오차 n 개의 제곱평균제곱근이다.

```math
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} e_i^2} \hspace{19em} (1)
```

식 (1) 의 오차 $e_i$ 는 model 의 i 번째 예측값에서 i 번째 계측값을 뺀 차이이고, $e_1, \dots, e_n$ 은 평균 0, 분산 $\sigma^2$ 의 정규분포에서 독립으로 나온 그 차이 n 개다. $\sigma$ 대신 RMSE 를 쓸 수 있는 근거는 둘이다. 첫째, 오차의 평균이 0 이므로 $\sigma^2 = E[e_i^2]$ 이고, 식 (1) 의 제곱근 안이 바로 $e_i^2$ 의 표본평균이어서 $E[\mathrm{RMSE}^2] = \sigma^2$ 이 성립한다. 곧 $\mathrm{RMSE}^2$ 은 $\sigma^2$ 의 불편추정량 (unbiased estimator) 이고, n 이 커지면 표본평균이 $\sigma^2$ 로 모이므로 RMSE 도 $\sigma$ 로 모인다. 둘째, 유한한 n 에서 RMSE 가 $\sigma$ 와 어긋나는 몫을 식 (6) 의 t distribution 이 그대로 셈에 넣으므로, 담는 확률을 어림잡지 않고 식 (7) 로 구할 수 있다.

```math
\mathrm{LSL},\ \mathrm{USL} = \mu_0 \mp k\,\mathrm{RMSE}, \qquad k = \sqrt{\frac{n}{\nu}}\; t_{\nu}\!\left(1 - \frac{\alpha}{2}\right) \hspace{19em} (2)
```

식 (2) 의 $\mu_0$ 는 구간의 중심이고, 오차에 긋는 구간에서는 0 이다. $k$ 는 담을 확률 $1 - \alpha$ 가 정하는 배율이며, $\sqrt{n/\nu}$ 는 RMSE 가 제곱합을 $\nu$ 가 아니라 n 으로 나눈 것을 되돌리는 몫이다. Parameter 를 추정하지 않아 $\nu = n$ 이면 그 몫이 1 이 되어 $k = t_n(1 - \alpha/2)$ 로 줄고, $\alpha = 0.05$ 와 n = 28 에서 2.05 다. Parameter 를 p 개 추정했으면 $\nu = n - p$ 를 넣어 다시 구하며, n = 28 에 p = 2 이면 2.13 이다. $k$ 를 관례대로 1.96 에 두면 담는 확률이 $1 - \alpha$ 에 미치지 못하고, 얼마나 모자라는지를 꼭지 4.2 가 적는다.

### 4.2 Interval Coverage

$\sigma$ 를 알면 1.96 배로 그린 구간이 담는 확률은 n 과 무관하게 95.00% 이고, $\sigma$ 자리에 RMSE 를 넣으면 그보다 낮아진다.

```math
P\left(|e| \le 1.96\,\sigma\right) = 2\Phi(1.96) - 1 = 0.9500 \hspace{19em} (3)
```

$\Phi$ 는 표준정규분포의 누적분포함수이고, 1.96 은 그 분포의 양측 95% 점을 소수 둘째 자리에서 끊은 값이다. 끊지 않은 값은 1.95996 이며, 1.96 이 내는 비율 0.950004 는 95% 와 소수 여섯째 자리에서 갈린다. 계측 두 방법의 차이를 견줄 때 차이의 평균에 표준편차의 1.96 배를 더하고 빼어 한계를 적는 관례가 이 값을 쓴다 [[3](#ref-3)]. 1.96 은 담을 확률을 95% 로 정했을 때의 값이며, 다른 확률을 정하면 1.96 대신 그 확률의 양측 점이 들어간다. 이 문서는 관례대로 95% 를 놓고 적는다.

RMSE 는 자료마다 달라지는 확률변수여서 식 (3) 의 $\sigma$ 를 그대로 대신하지 못한다. 제곱합을 $\sigma^2$ 으로 나눈 값은 자유도 n 의 chi-squared distribution 을 따르며, 그 정의와 확률밀도함수와 누적분포함수는 [Appendix C](#appendix-c-chi-squared-distribution) 에 있다.

```math
\frac{n\,\mathrm{RMSE}^2}{\sigma^2} \sim \chi^2_n \hspace{19em} (4)
```

식 (4) 에서 RMSE 의 기댓값이 나오며, 그 값은 $\sigma$ 에 1 보다 작은 상수를 곱한 것이다.

```math
E[\mathrm{RMSE}] = c_n\,\sigma, \qquad c_n = \sqrt{\frac{2}{n}}\;\frac{\Gamma\!\left(\frac{n+1}{2}\right)}{\Gamma\!\left(\frac{n}{2}\right)} \hspace{19em} (5)
```

$c_n$ 은 n 이 커지면 1 로 간다. n 이 10 이면 0.9754, 28 이면 0.9911, 100 이면 0.9975 다. 유도는 [Appendix D](#appendix-d-derivation-of-equations-4-to-8) 에 있다.

새 오차를 RMSE 로 나눈 값은 자유도 n 의 Student t distribution 을 따른다 [[4](#ref-4)].

```math
\frac{e^{\ast}}{\mathrm{RMSE}} \sim t_n \hspace{19em} (6)
```

식 (6) 의 $e^{\ast}$ 는 model 이 다음에 내놓을 예측 하나의 오차다. RMSE 를 구하던 때에는 아직 생기지 않았으므로 RMSE 를 구한 n 개와 독립이다. 분모가 상수 $\sigma$ 에서 확률변수 RMSE 로 바뀌면서 비 (ratio) 의 분포가 표준정규분포에서 t distribution 으로 옮겨 가고, 담는 확률도 그만큼 달라진다.

```math
P\left(|e^{\ast}| \le 1.96\,\mathrm{RMSE}\right) = 2F_{t_n}(1.96) - 1 \hspace{19em} (7)
```

식 (7) 의 $F_{t_n}$ 은 자유도 n 의 t distribution 의 누적분포함수다. Table 2 가 n 마다 그 값과, 95% 를 지키려면 1.96 대신 쓸 배율을 적는다.

Table 2. Coverage of the RMSE interval by sample count

| #   | n    | $c_n$  | Coverage | Multiplier for 0.95 |
| :-: | :--: | :----: | :------: | :-----------------: |
| 1   | 5    | 0.9515 | 0.8927   | 2.5706              |
| 2   | 10   | 0.9754 | 0.9216   | 2.2281              |
| 3   | 20   | 0.9876 | 0.9359   | 2.0860              |
| 4   | 25   | 0.9901 | 0.9388   | 2.0595              |
| 5   | 28   | 0.9911 | 0.9400   | 2.0484              |
| 6   | 30   | 0.9917 | 0.9407   | 2.0423              |
| 7   | 40   | 0.9938 | 0.9430   | 2.0211              |
| 8   | 50   | 0.9950 | 0.9444   | 2.0086              |
| 9   | 100  | 0.9975 | 0.9472   | 1.9840              |
| 10  | 200  | 0.9988 | 0.9486   | 1.9719              |
| 11  | 1000 | 0.9998 | 0.9497   | 1.9623              |

Coverage 열은 n 이 커질수록 올라가 95% 에 다가가지만 어느 n 에서도 95% 에 닿지 않는다. 94% 로 반올림되는 구간은 n 이 19 부터 55 까지이고, 가장 가까운 값은 5 행의 0.9400 이다. 94.5% 를 넘으려면 n 이 56 이상, 94.9% 를 넘으려면 277 이상이어야 한다. Table 2 는 parameter 를 추정하지 않아 $\nu = n$ 인 경우이고, 추정했으면 꼭지 4.5 의 식 (8) 로 구한다. 식 (7) 의 값은 200 만 회의 Monte Carlo 와 n 이 10, 28, 100 인 세 경우에서 소수 셋째 자리까지 같다.

아래 <a href="#fig-3">Fig 3</a> 가 식 (7) 과 95% 를 지키는 배율을 n 에 대해 그린다.

<img src="rmse-interval-coverage-ko_fig/rmse_interval_coverage.png" width="900" style="max-width: 100%;" alt="Fig 3">

<a id="fig-3"></a>
Fig 3. Coverage of the RMSE interval and the multiplier that restores 95 percent

- (a) 식 (7) 의 확률을 n 에 대해 그린 것이다. 위의 가로선이 오차의 표준편차를 알 때의 0.9500, 아래의 가로선이 0.9400 이며, 표시한 점이 곡선과 아래 가로선이 만나는 n = 28 이다.
- (b) 95% 를 지키는 배율 $t_n(0.975)$ 를 n 에 대해 그린 것이다. 가로선이 1.96 이고, 곡선은 n 이 커지면서 그 선으로 내려온다.

### 4.3 Coverage Loss

두 가지가 확률을 깎는다. RMSE 의 기댓값이 $\sigma$ 보다 작은 것이 하나이고, RMSE 가 자료마다 흔들리는 것이 다른 하나다.

RMSE 가 언제나 기댓값 $c_n\sigma$ 와 같다면 담는 확률은 $2\Phi(1.96\,c_n) - 1$ 이 된다. n 이 28 이면 그 값이 0.9479 이므로, 기댓값이 작은 데서 오는 몫은 0.0021 이다. 실제 확률이 0.9400 이니 나머지 0.0079 는 RMSE 가 흔들리는 데서 온다.

담는 확률을 표준편차의 함수로 본 $g(u) = 2\Phi(1.96\,u) - 1$ 은 $u \gt 0$ 에서 concave 이다. Jensen's inequality 가 $E[g(U)] \lt g(E[U])$ 를 주므로, RMSE 가 기댓값보다 커질 때 얻는 확률이 작아질 때 잃는 확률보다 적고 평균이 내려간다.

두 몫은 n 이 커지면 함께 줄어든다. n 이 10 이면 0.0059 와 0.0225, 100 이면 0.0006 과 0.0022 다.

### 4.4 Coverage Stated with 1.96

1.96 과 95% 를 함께 적은 글은 Table 2 대로 n 이 수백 이상일 때만 근사로 맞다. 1.96 과 94% 를 함께 적은 글은 RMSE 로 오차의 표준편차를 추정했을 때 n 이 30 안팎이면 나오는 값을 적은 것이며, $\sigma$ 를 알 때 94% 를 담는 배율은 1.88 이다.

### 4.5 Degrees of Freedom with Fitted Parameters

Model 을 자료에 맞추어 parameter 를 p 개 추정하고 그 잔차로 RMSE 를 구했으면 자유도가 $\nu = n - p$ 로 줄고, 같은 1.96 이 담는 비율이 더 내려간다.

```math
P\left(|e^{\ast}| \le 1.96\,\mathrm{RMSE}\right) = 2F_{t_{\nu}}\!\left(1.96\sqrt{\frac{\nu}{n}}\right) - 1, \qquad \nu = n - p \hspace{19em} (8)
```

식 (8) 은 p 가 0 이면 식 (7) 로 돌아간다. n 이 28 일 때 p 가 2 이면 0.9299, 5 이면 0.9111 이다. n 이 100 이고 p 가 5 이면 0.9409 로, 같은 0.94 를 세 배가 넘는 자료로 얻는다.

### 4.6 Conditions

- **가정**: 오차가 평균 0 의 정규분포에서 독립으로 나오고 분산이 모두 같다. RMSE 를 구한 오차와 담을 대상인 새 오차가 서로 독립이다. 예측값 자체가 흔들리는 몫인 leverage 는 뺀다 (꼭지 5.2).
- **설정값**: 배율은 1.96 으로 고정한다. 같은 자료로 parameter 를 p 개 추정했으면 자유도는 $\nu = n - p$ 이고, 담는 확률은 식 (8) 로 구한다.
- **깨지는 조건**: 오차의 평균이 0 이 아니면 RMSE 가 흩어짐과 치우침을 함께 담아 구간이 필요보다 넓어지고, 식 (7) 의 비율은 더 이상 그 구간을 설명하지 않는다. 분산이 자리마다 다르면 (heteroscedasticity) 하나의 RMSE 가 모든 자리를 대표하지 못해 분산이 큰 자리에서 비율이 떨어진다. 오차가 서로 상관되어 있으면 제곱합에 남는 자유도가 n 보다 작아 식 (7) 이 비율을 높게 낸다. 꼬리가 정규분포보다 두꺼우면 같은 배율이 담는 비율이 더 낮아, 분산이 같은 Laplace distribution 에서 $\pm 1.96\sigma$ 가 담는 비율은 93.7% 다.
- **만나는 자리**: 계측 두 방법의 차이에 한계를 긋는 Bland-Altman 한계 [[3](#ref-3)].

## 5. Prediction Interval

Prediction interval 은 식 (2) 에 그 식이 적은 $k$ 를 그대로 넣은 구간이고, 다음 오차 하나가 그 안에 들 확률이 n 과 무관하게 $1 - \alpha$ 로 유지된다.

꼭지 4 의 RMSE interval 과는 쓰는 표준편차가 같고 배율만 다르다. RMSE interval 은 같은 식에서 $k$ 를 관례인 1.96 으로 고정해 담는 확률이 식 (7) 로 내려가고, prediction interval 은 $k = \sqrt{n/\nu}\; t_{\nu}(1 - \alpha/2)$ 를 그대로 써서 그 확률을 $1 - \alpha$ 에 붙여 둔다. Parameter 를 추정하지 않았으면 $\nu = n$ 이어서 배율이 $t_{n}(1 - \alpha/2)$ 로 줄고, 그 값이 Table 2 의 마지막 열이다.

### 5.1 The Multiplier for 95 Percent

$\alpha = 0.05$ 에서 배율은 n 이 28 이면 2.05, 50 이면 2.01, 100 이면 1.98 이다.

### 5.2 Leverage

이 문서의 prediction interval 은 leverage 를 뺀 간이 구간이며, 식 (2) 의 배율과 식 (8) 의 확률은 새 오차가 RMSE 를 구한 잔차와 독립이고 분산이 $\sigma^2$ 인 경우의 값이다. 새 입력 $x^{\ast}$ 에서 model 의 예측값을 빼고 남는 오차는 분산이 $\sigma^2$ 이 아니라 $\sigma^2(1 + h)$ 이고, $h$ 는 그 입력의 leverage, 곧 예측값 자체가 흔들리는 몫이다. 선형 회귀라면 $h = 1/n + (x^{\ast} - \bar{x})^2 / \sum (x_i - \bar{x})^2$ 이다. $h$ 가 0 보다 크면 실제로 담는 확률은 식 (2) 의 구간에서 $1 - \alpha$ 보다, 1.96 배 구간에서 식 (8) 의 값보다 낮고, 자료의 중심에서 멀어질수록 더 낮다.

### 5.3 Conditions

- **가정**: 꼭지 4.6 과 같다.
- **설정값**: 배율은 식 (2) 의 $\sqrt{n/\nu}\; t_{\nu}(0.975)$ 이고 자유도는 $\nu = n - p$ 이며, p 는 같은 자료로 추정한 parameter 의 개수다. 오차 개수 n 은 RMSE 의 정의인 식 (1) 의 분모에 따로 남아 $\sqrt{n/\nu}$ 로 들어간다.
- **깨지는 조건**: 꼭지 4.6 과 같다. 꼭지 4.6 의 네 조건에서는 식 (2) 의 배율로도 담는 확률이 $1 - \alpha$ 로 유지되지 않는다.
- **만나는 자리**: 회귀 model 의 예측 오차 범위 보고, 공정 자료의 예측값에 붙이는 오차 막대, 꼭지 8 의 error limit.

## 6. Confidence Interval

Confidence interval 은 오차 하나가 아니라 오차의 평균이 들어 있을 구간이고, 같은 자유도와 같은 표준편차를 쓰면 폭이 prediction interval 의 $1/\sqrt{n}$ 이다.

구간은 $\bar{e} \pm t_{\nu}(1 - \alpha/2)\, s / \sqrt{n}$ 이며, $\bar{e}$ 는 오차 n 개의 평균, $s$ 는 그 평균을 뺀 뒤 구한 표준편차, 자유도는 $\nu = n - 1$ 이다. 담는 대상이 평균이므로 n 이 커지면 폭이 0 으로 줄지만, prediction interval 의 폭은 $\sigma$ 가 남아 0 으로 줄지 않는다. 둘을 바꾸어 쓰면 오차 하나를 담아야 할 때 $\sqrt{n}$ 배 좁은 구간을 긋게 된다.

Model 의 치우침이 0 인지 보는 데 이 구간을 쓴다. 구간이 0 을 담지 않으면 오차의 평균이 0 이 아니라는 뜻이고, 꼭지 8.2 의 첫 항목이 그 치우침을 먼저 빼라고 적는다.

## 7. Tolerance Interval

Tolerance interval 은 모집단의 비율 $P$ 를 담는다는 것을 신뢰수준 $\gamma$ 로 보장하는 구간이고, 담는 비율 자체에 확률을 한 번 더 붙인 것이다.

Prediction interval 은 다음 오차 하나가 들어올 확률이 평균적으로 $1 - \alpha$ 라고 말하고, tolerance interval 은 그 구간이 모집단의 $P$ 이상을 담을 확률이 $\gamma$ 라고 말한다. 신뢰수준을 0.5 보다 높게 둔 경우 요구가 하나 늘어난 만큼 배율도 커서, 같은 n 과 같은 비율에서 prediction interval 보다 넓다. 배율은 $P$ 와 $\gamma$ 와 n 으로 정해지는 tolerance factor 이고, 닫힌 형태가 없어 조합마다 표에서 읽는다 [[1](#ref-1)] [[2](#ref-2)].

"신뢰수준 95% 로 모집단의 99% 이상" 처럼 구간이 모집단의 몇 할을 담아야 하는지와 그것을 얼마의 신뢰수준으로 보장해야 하는지를 함께 요구할 때 쓴다. 다음 한 점이 구간에 드는지만 볼 때는 prediction interval 이 맞다.

## 8. Spec Setting for SPC

예측값으로 SPC 를 할 때 긋는 세 한계는 각각 prediction interval, RMSE interval, Normal interval 을 쓰고, 그에 앞서 confidence interval 로 model 의 치우침을 확인한다.

- **Prediction interval**: Error limit (꼭지 8.2). 다음 예측 하나의 오차를 n 과 무관하게 $1 - \alpha$ 로 담아야 하므로 식 (2) 의 배율을 쓴다.
- **RMSE interval**: Acceptance limit 의 guard band (꼭지 8.3). ISO 14253-1 이 표준불확도에 고정 배율 2 를 곱한 확장 불확도 (expanded uncertainty) 를 spec 에서 빼라고 정하므로, RMSE 에 고정 배율을 곱하는 RMSE interval 의 꼴이 된다. 고정 배율이 담는 확률은 식 (7) 처럼 n 이 작을수록 낮아, n 이 작으면 배율을 prediction interval 의 배율 쪽으로 키운다.
- **Normal interval**: Control limit (꼭지 8.4). 관리 한계는 관측 표준편차를 아는 값으로 보고 그 3 배를 더하고 빼므로, Table 1 의 1 행에서 배율이 3, 곧 $\alpha = 0.0027$ 인 경우다.
- **Confidence interval**: Bias 확인 (꼭지 8.2 의 첫 항목). RMSE 를 구하기 전에 오차의 평균이 0 인지를 꼭지 6 의 구간으로 본다.
- **Tolerance interval**: 쓰지 않는다. Spec 이 "신뢰수준 $\gamma$ 로 모집단의 $P$ 이상" 을 요구하면 error limit 의 배율을 꼭지 7 의 tolerance factor 로 바꾼다.

### 8.1 What the Limit Is Drawn On

Error limit 과 guard band 는 RMSE 로 구하고, control limit 은 공정의 변동으로 구한다. Table 3 이 세 한계를 견준다.

Table 3. Limits around a predicted value

| #   | Limit            | Set from     | Offset               | What it decides               |
| :-: | :--------------: | :----------: | :------------------: | :---------------------------: |
| 1   | Error limit      | RMSE         | $k\,\mathrm{RMSE}$   | 한 예측의 오차를 받아들일지   |
| 2   | Acceptance limit | Spec 과 RMSE | $g\,\mathrm{RMSE}$   | 예측값으로 규격 판정을 내릴지 |
| 3   | Control limit    | 공정의 변동  | 관측 표준편차의 3 배 | 공정이 평소와 달라졌는지      |

1 행은 model 이 계측을 대신해도 되는 범위다. 2 행은 이미 있는 spec 에서 안쪽으로 물러선 자리이며, 물러선 폭이 guard band 다. 3 행의 한계는 spec 에서 나오지 않지만, 예측값으로 chart 를 그리면 그 폭에 RMSE 가 식 (10) 으로 섞여 든다.

### 8.2 The Limit from the RMSE

Table 3 의 1 행인 error limit 은 꼭지 4.1 의 식 (2) 에 $\mu_0 = 0$ 을 넣은 것이다. Spec 은 요구에서 따로 정해지며, RMSE 는 꼭지 8.3 의 guard band 로만 spec 에 들어간다. $\alpha = 0.05$ 와 $\nu = n = 28$ 이면 배율이 2.05 이고, Table 2 의 마지막 열이 그 값이다. 쓰기 전에 아래 세 가지를 확인한다.

- **Bias 를 먼저 뺀다**: 오차의 평균이 0 이 아니면 RMSE 가 치우침과 흩어짐을 함께 담아 (꼭지 4.6) 한계가 필요보다 넓어진다. 치우침을 model 에서 고친 뒤 RMSE 를 다시 구한다.
- **Model 을 맞추는 데 쓰지 않은 자료로 구한다**: 그 자료로 추정한 parameter 가 없으므로 $\nu = n$ 이다. Model 을 맞춘 자료의 잔차로 구하면 자유도가 $\nu = n - p$ 로 줄어 식 (7) 이 확률을 높게 내므로, 확률은 식 (8) 로, 배율은 식 (2) 에 그 $\nu$ 를 넣어 구한다.
- **오차 개수를 먼저 센다**: n 이 30 안팎이면 1.96 이 담는 확률이 94% 이므로, 95% 를 error limit 에 적으려면 배율을 Table 2 에서 바꾸어 읽는다.

### 8.3 The Guard Band

예측값이 계측값을 대신해 규격 판정을 내리면 spec 안쪽으로 물러선 수락 한계 (acceptance limit) 를 쓴다.

```math
A_{\mathrm{L}} = \mathrm{LSL} + g\,\mathrm{RMSE}, \qquad A_{\mathrm{U}} = \mathrm{USL} - g\,\mathrm{RMSE} \hspace{19em} (9)
```

식 (9) 의 두 값 사이에 들어온 것만 받아들이고, spec 과 수락 한계 사이의 폭 $g\,\mathrm{RMSE}$ 가 guard band 다. Spec 바로 안쪽에서 측정된 것도 참값은 밖에 있을 수 있고, 그 확률은 RMSE 가 클수록 커진다. Guard band 가 그 확률, 곧 consumer's risk 를 내린다 [[5](#ref-5)]. ISO 14253-1 의 기본 규칙은 확장 불확도 한 배를 spec 에서 빼는 것이다 [[6](#ref-6)]. RMSE 를 표준불확도로 보면 $g$ 가 곧 포함인자 (coverage factor) 이고, 관례인 포함인자 2 가 $g = 2$ 를 준다. n 이 작으면 식 (2) 의 $k$ 가 커지는 것과 같은 이유로 2 보다 큰 값을 쓴다. $g$ 를 키우면 consumer's risk 가 내려가는 대신 규격 안의 것을 거부할 확률, 곧 producer's risk 가 올라간다.

### 8.4 The Spec and the Control Limit

관리 한계는 공정의 변동에서 구하고 spec 은 요구에서 구하므로, 둘을 같은 수로 두지 않는다. 예측값으로 chart 를 그리면 관측되는 변동이 공정의 변동과 model 의 오차를 함께 담는다.

```math
\sigma_{\mathrm{obs}}^2 = \sigma_{\mathrm{proc}}^2 + \mathrm{RMSE}^2 \hspace{19em} (10)
```

식 (10) 의 $\sigma_{\mathrm{obs}}$ 가 관리 한계의 폭을 정하므로, RMSE 가 공정 표준편차의 절반이면 한계가 1.12 배 넓어지고 같은 자료로 잰 공정능력지수 (process capability index) 는 0.89 배로 내려간다. RMSE 가 공정 표준편차와 같으면 각각 1.41 배와 0.71 배다. 식 (10) 은 model 의 오차가 공정의 변동과 독립일 때 성립하며, 오차가 공정 수준에 따라 달라지면 (heteroscedasticity) 하나의 RMSE 로 모든 자리의 한계를 정하지 못한다.

## References

<a id="ref-1"></a>
[1] Meeker, W. Q., Hahn, G. J., & Escobar, L. A. (2017). [Statistical Intervals: A Guide for Practitioners and Researchers](https://wqmeeker.stat.iastate.edu/other_pages/hahn_meeker.html) (2nd ed.). John Wiley & Sons. ISBN 978-0471687177.<br>
<a id="ref-2"></a>
[2] NIST/SEMATECH. (2012). [e-Handbook of Statistical Methods](https://doi.org/10.18434/M32189). NIST Handbook 151, National Institute of Standards and Technology.<br>
<a id="ref-3"></a>
[3] Bland, J. M., & Altman, D. G. (1986). [Statistical Methods for Assessing Agreement Between Two Methods of Clinical Measurement](https://doi.org/10.1016/S0140-6736(86)90837-8). *The Lancet*, 327(8476), 307–310.<br>
<a id="ref-4"></a>
[4] Student. (1908). [The Probable Error of a Mean](https://doi.org/10.2307/2331554). *Biometrika*, 6(1), 1–25.<br>
<a id="ref-5"></a>
[5] JCGM. (2012). [Evaluation of Measurement Data — The Role of Measurement Uncertainty in Conformity Assessment](https://doi.org/10.59161/JCGM106-2012). JCGM 106:2012, Joint Committee for Guides in Metrology.<br>
<a id="ref-6"></a>
[6] ISO. (2017). [Geometrical Product Specifications (GPS) — Inspection by Measurement of Workpieces and Measuring Equipment — Part 1: Decision Rules for Verifying Conformity or Nonconformity with Specifications](https://www.iso.org/standard/70137.html). ISO 14253-1:2017, International Organization for Standardization.

---

## Appendix A. Terminology

- **chi-squared distribution**: 독립인 표준정규분포 값 여러 개를 제곱해 더한 값이 따르는 분포. 식 (14) 가 정의다.
- **concave**: 2차 도함수가 음수여서 곡선이 위로 볼록한 함수의 성질.
- **confidence interval**: 추정하려는 모수가 들어 있을 확률을 정해 둔 구간.
- **consumer's risk**: 받아들인 것이 규격을 벗어나 있을 확률.
- **control limit**: 공정이 평소와 같은지를 가르는 chart 의 한계. 공정의 변동에서 구한다.
- **coverage**: 구간이 담으려는 대상을 실제로 담을 확률.
- **coverage factor**: 표준불확도에 곱해 확장 불확도를 얻는 수. 관례는 2 다.
- **degrees of freedom**: 제곱합에 남아 있는 독립한 성분의 개수.
- **expanded uncertainty**: 표준불확도에 coverage factor 를 곱한 값.
- **gamma function**: 식 (18) 의 적분으로 정의되는 함수. 양의 정수에서 계승 (factorial) 을 확장한 값이 된다.
- **guard band**: Spec 과 수락 한계 사이의 폭. 식 (9) 가 정의다.
- **heteroscedasticity**: 분산이 자리마다 다른 상태.
- **Jensen's inequality**: concave 함수에서 기댓값의 함수가 함수의 기댓값보다 크다는 부등식.
- **Laplace distribution**: 양쪽 꼬리가 지수함수로 줄어드는 대칭 분포. 분산이 같은 정규분포보다 꼬리가 두껍다.
- **leverage**: 새 입력이 자료의 중심에서 떨어진 정도를 재는 값. 예측값 자체가 흔들리는 몫을 정한다.
- **lower incomplete gamma function**: gamma function 의 적분 구간을 유한한 위끝에서 끊은 함수. 식 (17) 이 정의다.
- **Monte Carlo**: 난수로 표본을 만들어 확률을 추정하는 방법.
- **prediction interval**: 새 관측 하나가 들어 있을 확률을 정해 둔 구간.
- **process capability index**: Spec 의 폭을 공정 변동의 6 배로 나눈 값.
- **producer's risk**: 거부한 것이 규격 안에 있을 확률.
- **quantile function**: 누적분포함수의 역함수. 확률을 받아 그 확률이 되는 지점을 낸다. 식 (12) 가 정의다.
- **regularized incomplete gamma function**: lower incomplete gamma function 을 gamma function 으로 나눈 값. 식 (16) 의 우변이 그 값이다.
- **RMSE**: 오차 제곱의 평균에 제곱근을 취한 값. 식 (1) 이 정의다.
- **spec limit**: 요구에서 나온 상한과 하한. 공정의 변동과는 따로 정해진다.
- **Student t distribution**: 표준정규분포 값을, 그와 독립인 chi-squared 값을 자유도로 나눈 것의 제곱근으로 나눈 비가 따르는 분포.
- **tolerance factor**: tolerance interval 에서 표준편차에 곱하는 배율. 비율과 신뢰수준과 표본 수로 정해진다.
- **tolerance interval**: 모집단의 정해진 비율을 담는다는 것을 정해진 신뢰수준으로 보장하는 구간.
- **unbiased estimator**: 기댓값이 추정 대상과 같은 추정량.

## Appendix B. Quantile Functions of the Normal and t Distributions

### B.1 Normal Distribution

#### From the Cumulative Distribution Function to Its Inverse

누적분포함수는 지점을 받아 확률을 내고, 분위수 함수는 반대로 확률을 받아 지점을 낸다. 표준정규분포의 누적분포함수 $\Phi$ 가 앞의 것이다.

```math
\Phi(x) = P(Z \le x), \qquad Z \sim N(0, 1) \hspace{19em} (11)
```

식 (11) 의 $\Phi$ 는 연속이고 $x$ 가 커지면 늘기만 하므로, 0 과 1 사이의 확률 $q$ 마다 $\Phi(x) = q$ 를 만족하는 $x$ 가 하나로 정해진다. 그 $x$ 를 $z(q)$ 로 적으며, 이것이 $\Phi$ 의 역함수다.

```math
z(q) = \Phi^{-1}(q), \qquad \Phi(z(q)) = q \hspace{19em} (12)
```

$z(0.5)$ 는 0 이고 $z(0.975)$ 는 1.96 이다. 앞엣것은 중앙값이고, 뒤엣것은 그 아래에 전체의 97.5% 가 놓이는 지점이다.

#### The Two-sided Point

양쪽 꼬리에 합쳐 $\alpha$ 를 남기는 구간의 두 끝이 $\pm z(1 - \alpha/2)$ 다. 분포가 대칭이므로 한쪽 꼬리에 $\alpha/2$ 씩 남기고, 위쪽 꼬리가 $\alpha/2$ 인 지점은 그 아래의 누적확률이 $1 - \alpha/2$ 인 지점이다.

```math
P\left(|Z| \le z(1 - \alpha/2)\right) = 1 - \alpha \hspace{19em} (13)
```

$\alpha = 0.05$ 를 넣으면 $z(0.975) = 1.96$ 이고, 식 (13) 이 식 (3) 이 된다.

### B.2 t Distribution

자유도 $\nu$ 의 Student t distribution 도 같은 방식으로 $t(\nu, q)$ 를 쓴다. 그 분포의 누적분포함수 $F_{t_{\nu}}$ 의 역함수이며, $t(\nu, 1 - \alpha/2)$ 가 양쪽 꼬리에 합쳐 $\alpha$ 를 남기는 지점이다. t distribution 은 정규분포보다 꼬리가 두꺼워 같은 $q$ 에서 $z(q)$ 보다 크고, $\nu$ 가 커지면 $z(q)$ 로 다가간다. $\alpha = 0.05$ 에서 $t(28, 0.975)$ 는 2.05, $t(1000, 0.975)$ 는 1.96 이다.

## Appendix C. Chi-squared Distribution

### C.1 Definition

서로 독립이고 표준정규분포 $N(0, 1)$ 를 따르는 $k$ 개의 확률변수 $Z_1, Z_2, \dots, Z_k$ 가 있을 때, 이 변수들의 제곱합으로 정의되는 확률변수 $X$ 는 자유도가 $k$ 인 chi-squared distribution 을 따른다.

```math
X = \sum_{i=1}^{k} Z_i^2 = Z_1^2 + Z_2^2 + \dots + Z_k^2 \sim \chi^2(k) \hspace{19em} (14)
```

- $k$ (자유도, degrees of freedom): 합산되는 독립 표준정규분포 변수의 개수.

본문은 같은 자유도를 n 과 $\nu$ 로 적는다. 식 (19) 의 $S$ 는 식 (14) 의 $X$ 에 $k = n$ 을 넣은 것이고, 식 (22) 의 $S_{\nu}$ 는 $k = \nu$ 를 넣은 것이다.

### C.2 Probability Density Function

자유도가 $k$ 인 chi-squared distribution 의 확률밀도함수 (probability density function) 는 $x \gt 0$ 에서 아래와 같다.

```math
f(x; k) = \frac{1}{2^{k/2}\,\Gamma(k/2)}\; x^{(k/2) - 1}\, e^{-x/2} \hspace{19em} (15)
```

식 (15) 의 $\Gamma$ 는 gamma function 이며, 식 (18) 이 gamma function 의 정의다.

### C.3 Cumulative Distribution Function

누적분포함수 $F(x; k)$ 는 확률변수 $X$ 가 특정 값 $x$ 이하일 확률 $P(X \le x)$ 를 뜻하며, 확률밀도함수를 0 부터 $x$ 까지 적분하여 구한다.

```math
F(x; k) = P(X \le x) = \frac{1}{\Gamma(k/2)}\; \gamma\!\left(\frac{k}{2}, \frac{x}{2}\right) \quad (x \ge 0) \hspace{19em} (16)
```

식 (16) 의 $\gamma(s, t)$ 는 하부 불완전 감마 함수 (lower incomplete gamma function) 이고, $\Gamma(s)$ 는 적분의 위끝을 무한대로 늘린 gamma function 이다.

```math
\gamma(s, t) = \int_0^t u^{s-1} e^{-u}\, du \hspace{19em} (17)
```

```math
\Gamma(s) = \int_0^{\infty} u^{s-1} e^{-u}\, du \hspace{19em} (18)
```

Chi-squared distribution 의 누적분포함수는 닫힌 형태 (elementary function) 로 단순하게 표현되지 않는다. 그래서 식 (16) 처럼 정규화 불완전 감마 함수 (regularized incomplete gamma function) 의 꼴로 정의하고, 실제 계산에는 numerical 방법이나 R, Python, Excel 이 담은 통계 함수를 쓴다.

### C.4 Properties

- **값의 범위**: $X \ge 0$. 제곱합이므로 음수가 되지 않는다.
- **평균**: $E(X) = k$.
- **분산**: $\mathrm{Var}(X) = 2k$.
- **모양**: $k$ 가 작을수록 오른쪽으로 긴 꼬리를 가진 비대칭 형태이며, $k$ 가 커질수록 점점 정규분포 모양에 가깝게 대칭형으로 변한다.

## Appendix D. Derivation of Equations (4) to (8)

출발점은 오차 하나를 $\sigma$ 로 나눈 값이 표준정규분포를 따른다는 것 하나다. n 개를 제곱해 더하면 자유도 n 의 chi-squared distribution 이 된다.

```math
S = \sum_{i=1}^{n} \left(\frac{e_i}{\sigma}\right)^2 \sim \chi^2_n, \qquad S = \frac{n\,\mathrm{RMSE}^2}{\sigma^2} \hspace{19em} (19)
```

두 번째 등식은 식 (1) 의 양변을 제곱해 n 을 곱한 것이며, 이것이 식 (4) 가다. 식 (5) 는 여기서 제곱근의 기댓값으로 나온다. 자유도 n 의 chi-squared 값에 제곱근을 취한 값의 기댓값이 아래와 같다.

```math
E\left[\sqrt{S}\right] = \sqrt{2}\;\frac{\Gamma\!\left(\frac{n+1}{2}\right)}{\Gamma\!\left(\frac{n}{2}\right)} \hspace{19em} (20)
```

식 (19) 에서 $\mathrm{RMSE} = \sigma\sqrt{S/n}$ 이므로 양변에 기댓값을 취하고 식 (20) 을 넣으면 식 (5) 의 $c_n$ 이 그대로 나온다.

식 (6) 은 t distribution 의 정의에서 나온다. 새 오차 $e^{\ast}$ 는 $e_1, \dots, e_n$ 과 독립이므로 $Z = e^{\ast}/\sigma$ 는 S 와 독립인 표준정규분포 값이다.

```math
\frac{e^{\ast}}{\mathrm{RMSE}} = \frac{\sigma Z}{\sigma\sqrt{S/n}} = \frac{Z}{\sqrt{S/n}} \sim t_n \hspace{19em} (21)
```

식 (21) 의 가운데에서 $\sigma$ 가 약분되므로 담는 비율은 모르는 $\sigma$ 와 무관해지고 n 하나로 정해진다. 좌변의 절댓값이 1.96 이하일 확률을 t distribution 의 누적분포함수로 적은 것이 식 (7) 이다.

Parameter 를 p 개 추정한 경우에는 제곱합의 자유도가 $\nu = n - p$ 로 줄지만, RMSE 는 식 (1) 대로 n 으로 나눈다. 자유도만큼의 chi-squared 값을 $S_{\nu}$ 로 두면 두 수가 따로 남는다.

```math
\frac{e^{\ast}}{\mathrm{RMSE}} = \frac{Z}{\sqrt{S_{\nu}/n}} = \sqrt{\frac{n}{\nu}}\;\frac{Z}{\sqrt{S_{\nu}/\nu}} \sim \sqrt{\frac{n}{\nu}}\;t_{\nu} \hspace{19em} (22)
```

식 (22) 의 좌변이 1.96 이하일 조건은 $t_{\nu}$ 가 $1.96\sqrt{\nu/n}$ 이하일 조건과 같고, 이것을 누적분포함수로 적은 것이 식 (8) 이다. 식 (5) 의 $c_n$ 도 n 을 $\nu$ 로 바꾸면 $\sqrt{\nu/n}$ 배만큼 더 작아진다.
