# Coverage of the RMSE Interval
Rev. 10 | Created: 2026-10-05 | Updated: 2026-10-09 10:45 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Placement](#31-placement)
- [4. Coverage of the Interval](#4-coverage-of-the-interval)
  - [4.1 With the Scale Known](#41-with-the-scale-known)
  - [4.2 The RMSE as an Estimate of the Scale](#42-the-rmse-as-an-estimate-of-the-scale)
  - [4.3 The Coverage with the RMSE](#43-the-coverage-with-the-rmse)
  - [4.4 Where the Loss Comes From](#44-where-the-loss-comes-from)
- [5. Application](#5-application)
  - [5.1 The Multiplier for 95 Percent](#51-the-multiplier-for-95-percent)
  - [5.2 Degrees of Freedom with Fitted Parameters](#52-degrees-of-freedom-with-fitted-parameters)
  - [5.3 Conditions](#53-conditions)
- [6. Spec Setting for SPC](#6-spec-setting-for-spc)
  - [6.1 What the Limit Is Drawn On](#61-what-the-limit-is-drawn-on)
  - [6.2 The Limit from the RMSE](#62-the-limit-from-the-rmse)
  - [6.3 The Guard Band](#63-the-guard-band)
  - [6.4 The Spec and the Control Limit](#64-the-spec-and-the-control-limit)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Chi-squared Distribution](#appendix-b-chi-squared-distribution)
- [Appendix C. Derivation of Equations (3) to (7)](#appendix-c-derivation-of-equations-3-to-7)

## 1. Purpose

- **Problem Statement**: ML 의 metric 인 RMSE 의 물리적 의미를 이해하기 힘들다.
- **Goal**: model 의 오차인 RMSE 를 이용해 SPC 를 위한 spec 을 만드는 방법의 이론과 실행을 담은 guide 를 만든다.
- **Non-Goal**: 오차가 정규분포가 아닐 때 쓸 구간은 다루지 않는다.

## 2. Summary

RMSE 의 1.96 배로 그린 구간이 새 오차 하나를 담는 비율은 오차의 참 표준편차를 알 때 95.00% 이고, RMSE 를 오차 n 개에서 구했으면 그보다 낮으며, 94% 가 되는 것은 n 이 28 일 때다 (n 이 23 부터 37 까지면 94% 로 반올림된다).

비율이 깎이는 까닭은 새 오차를 RMSE 로 나눈 값이 표준정규분포가 아니라 자유도 n 의 Student t distribution 을 따르는 데 있다.

95% 를 지키려면 배율을 1.96 대신 그 t distribution 의 97.5% 점으로 둔다. n 이 28 이면 2.05 이고, n 이 커지면 1.96 으로 돌아간다. SPC 의 spec 은 그 배율로 긋고, 예측값이 계측값을 대신해 판정에 쓰이면 guard band 만큼 안쪽으로 물러선다.

## 3. Taxonomy and its Hierarchy

구간의 크기는 model 의 오차인 RMSE 의 배율로 구한다. 구간의 두 끝은 중심에서 RMSE 의 배율 배만큼 떨어진 자리이며, $\sigma$ 를 아는 자리에서는 RMSE 대신 그 참값에 배율을 곱한다. 배율을 정하는 것은 두 가지다. 오차의 표준편차 (scale) 를 아는가 추정하는가, 그리고 구간이 담는 대상 (covered quantity) 이 무엇인가이다. 아래 <a href="#fig-1">Fig 1</a> 이 그 두 축을 담는다.

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
HIERARCHY of the intervals, each step down dropping an assumption
|
Normal interval    scale sigma   k = 1.96            coverage 0.9500 for every n
|
+-- RMSE interval  scale RMSE    k = 1.96            coverage 2 * F_t(1.96; nu) - 1
|                                                    0.9400 at n = 28
+-- prediction     scale RMSE    k = t(nu, 0.975)    coverage 0.9500 for every n
    |
    +-- tolerance  scale RMSE    k = tolerance factor
                                                     a stated fraction, held with
                                                     a stated confidence
```

<a id="fig-2"></a>
Fig 2. The hierarchy the intervals form, from the strongest assumption down

계층은 위에서 아래로 내려갈수록 아는 것을 하나씩 내려놓는다. 맨 위는 오차의 표준편차를 알고, 그 아래는 표준편차를 n 개의 오차로 추정하며, 맨 아래는 추정한 표준편차로 모집단의 비율까지 말하려 하므로 구간이 가장 넓다.

### 3.1 Placement

Placement 는 <a href="#fig-1">Fig 1</a> 의 두 축과 <a href="#fig-2">Fig 2</a> 의 계층 위에서 각 구간이 어느 자리에 놓이는지를 뜻한다. Table 1 이 그 자리를 구간이 쓰는 표준편차, 그 표준편차에 곱하는 배율, 확률이 가리키는 대상의 세 가지로 적으며, 손에 있는 자료에 어느 구간을 쓸지는 이 표에서 고른다.

Table 1. Intervals around a prediction

| #   | Interval            | Scale             | Multiplier       | Covered quantity     | Probability       |
| :-: | :-----------------: | :---------------: | :--------------: | :------------------: | :---------------: |
| 1   | Normal interval     | $\sigma$          | 1.96             | 새 오차 하나         | 0.9500            |
| 2   | RMSE interval       | RMSE              | 1.96             | 새 오차 하나         | 식 (6) 의 값      |
| 3   | Prediction interval | RMSE              | $t_{\nu}(0.975)$ | 새 오차 하나         | 0.9500            |
| 4   | Confidence interval | RMSE / $\sqrt{n}$ | $t_{\nu}(0.975)$ | 오차의 평균          | 0.9500            |
| 5   | Tolerance interval  | RMSE              | tolerance factor | 모집단의 정해진 비율 | 신뢰수준으로 보장 |

1 행은 $\sigma$ 를 아는 경우에만 쓸 수 있고, 2 행부터는 모두 RMSE 로 $\sigma$ 를 대신한다. 2 행과 3 행은 같은 표준편차에 다른 배율을 곱한 것이며, 3 행의 배율이 자유도 $\nu$ 의 t distribution 에서 나온 값이라 비율이 n 과 무관하게 0.9500 으로 유지된다. 4 행은 폭이 $\sqrt{n}$ 배만큼 좁고, 담는 대상이 새 오차가 아니라 오차의 평균이므로 2 행과 바꾸어 쓸 수 없다. 4 행의 RMSE 는 오차의 평균을 뺀 뒤 구한 값이고 자유도는 $\nu = n - 1$ 이다. 5 행은 비율 자체에 신뢰수준을 붙이는 구간이고, 배율은 비율과 신뢰수준과 n 의 세 값으로 정해지는 tolerance factor 다 [[1](#ref-1)]. Tolerance factor 는 닫힌 형태가 없어 세 값의 조합마다 표에서 읽는다 [[2](#ref-2)].

## 4. Coverage of the Interval

RMSE 의 1.96 배로 그린 구간이 새 오차 하나를 담는 확률은 식 (6) 하나로 정해지고, 그 값은 모르는 $\sigma$ 가 아니라 RMSE 를 구한 오차의 개수 n 으로 정해진다.

### 4.1 With the Scale Known

오차가 평균 0, 분산 $\sigma^2$ 의 정규분포를 따르고 $\sigma$ 를 알면 구간이 담는 비율은 n 과 무관하게 95.00% 이다.

```math
P\left(|e| \le 1.96\,\sigma\right) = 2\Phi(1.96) - 1 = 0.9500 \hspace{19em} (1)
```

$\Phi$ 는 표준정규분포의 누적분포함수이고, 1.96 은 그 분포의 양측 95% 점을 소수 둘째 자리에서 끊은 값이다. 끊지 않은 값은 1.95996 이며, 1.96 이 내는 비율 0.950004 는 95% 와 소수 여섯째 자리에서 갈린다. 계측 두 방법의 차이를 견주는 자리에서 차이의 평균에 표준편차의 1.96 배를 더하고 빼어 한계를 적는 관례가 이 값을 쓴다 [[3](#ref-3)].

### 4.2 The RMSE as an Estimate of the Scale

RMSE 는 $\sigma$ 를 모를 때 그 자리에 넣는 추정값이며, 자료마다 달라지는 확률변수다.

```math
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n} e_i^2} \hspace{19em} (2)
```

식 (2) 의 오차 $e_i$ 는 model 이 i 번째 자리에 내놓은 예측값에서 그 자리의 계측값을 뺀 차이이고, $e_1, \dots, e_n$ 은 평균 0, 분산 $\sigma^2$ 의 정규분포에서 독립으로 나온 그 차이 n 개다. 제곱합을 $\sigma^2$ 으로 나눈 값은 자유도 n 의 chi-squared distribution 을 따른다. Chi-squared distribution 의 정의와 확률밀도함수와 누적분포함수는 [Appendix B](#appendix-b-chi-squared-distribution) 에 있다.

```math
\frac{n\,\mathrm{RMSE}^2}{\sigma^2} \sim \chi^2_n \hspace{19em} (3)
```

식 (3) 에서 RMSE 의 기댓값이 나오며, 그 값은 $\sigma$ 에 1 보다 작은 상수를 곱한 것이다.

```math
E[\mathrm{RMSE}] = c_n\,\sigma, \qquad c_n = \sqrt{\frac{2}{n}}\;\frac{\Gamma\!\left(\frac{n+1}{2}\right)}{\Gamma\!\left(\frac{n}{2}\right)} \hspace{19em} (4)
```

$c_n$ 은 n 이 커지면 1 로 간다. n 이 10 이면 0.9754, 28 이면 0.9911, 100 이면 0.9975 다. 유도는 [Appendix C](#appendix-c-derivation-of-equations-3-to-7) 에 있다.

### 4.3 The Coverage with the RMSE

새 오차를 RMSE 로 나눈 값은 표준정규분포가 아니라 자유도 n 의 Student t distribution 을 따른다 [[4](#ref-4)].

```math
\frac{e^{\ast}}{\mathrm{RMSE}} \sim t_n \hspace{19em} (5)
```

식 (5) 의 $e^{\ast}$ 는 RMSE 를 구한 n 개와 독립인 새 오차 하나다. 분모가 상수 $\sigma$ 에서 확률변수 RMSE 로 바뀌면서 비 (ratio) 의 분포가 표준정규분포에서 t distribution 으로 옮겨 가고, 담는 비율도 그만큼 달라진다.

```math
P\left(|e^{\ast}| \le 1.96\,\mathrm{RMSE}\right) = 2F_{t_n}(1.96) - 1 \hspace{19em} (6)
```

식 (6) 의 $F_{t_n}$ 은 자유도 n 의 t distribution 의 누적분포함수다. Table 2 가 n 마다 그 값과, 95% 를 지키려면 1.96 자리에 넣어야 할 배율을 적는다.

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

Coverage 열은 n 이 커질수록 올라가 95% 에 다가가지만 어느 n 에서도 95% 에 닿지 않는다. 94% 로 반올림되는 구간은 n 이 23 부터 37 까지이고, 가장 가까운 값은 5 행의 0.9400 이다. 94.5% 를 넘으려면 n 이 56 이상, 94.9% 를 넘으려면 277 이상이어야 한다. 식 (6) 의 값은 200 만 회의 Monte Carlo 와 n 이 10, 28, 100 인 세 자리에서 소수 셋째 자리까지 같다.

아래 <a href="#fig-3">Fig 3</a> 가 식 (6) 과 95% 를 지키는 배율을 n 에 대해 그린다.

<img src="rmse-interval-coverage-ko_fig/rmse_interval_coverage.png" width="900" style="max-width: 100%;" alt="Fig 3">

<a id="fig-3"></a>
Fig 3. Coverage of the RMSE interval and the multiplier that restores 95 percent

- (a) 식 (6) 의 비율을 n 에 대해 그린 것이다. 위의 가로선이 오차의 표준편차를 알 때의 0.9500, 아래의 가로선이 0.9400 이며, 표시한 점이 곡선과 아래 가로선이 만나는 n = 28 이다.
- (b) 95% 를 지키는 배율 $t_n(0.975)$ 를 n 에 대해 그린 것이다. 가로선이 1.96 이고, 곡선은 n 이 커지면서 그 선으로 내려온다.

### 4.4 Where the Loss Comes From

두 가지가 비율을 깎는다. RMSE 의 기댓값이 $\sigma$ 보다 작은 것이 하나이고, RMSE 가 자료마다 흔들리는 것이 다른 하나다.

RMSE 가 언제나 기댓값 $c_n\sigma$ 와 같다면 담는 비율은 $2\Phi(1.96\,c_n) - 1$ 이 된다. n 이 28 이면 그 값이 0.9479 이므로, 기댓값이 작은 데서 오는 몫은 0.0021 이다. 실제 비율이 0.9400 이니 나머지 0.0079 는 RMSE 가 흔들리는 데서 온다.

담는 비율을 표준편차의 함수로 본 $g(u) = 2\Phi(1.96\,u) - 1$ 은 $u \gt 0$ 에서 concave 이다. Jensen's inequality 가 $E[g(U)] \lt g(E[U])$ 를 주므로, RMSE 가 기댓값보다 커질 때 얻는 비율이 작아질 때 잃는 비율보다 적고 평균이 내려간다.

두 몫은 n 이 커지면 함께 줄어든다. n 이 10 이면 0.0059 와 0.0225, 100 이면 0.0006 과 0.0022 다.

## 5. Application

95% 를 지키려면 RMSE 에 곱하는 배율을 1.96 이 아니라 그 자유도의 t distribution 의 97.5% 점으로 둔다.

### 5.1 The Multiplier for 95 Percent

배율은 Table 2 의 마지막 열에 있다. n 이 28 이면 2.05, 50 이면 2.01, 100 이면 1.98 이다. 1.96 을 그대로 쓰면서 95% 라 적는 글은 n 이 수백 이상일 때만 맞고, n 이 277 이상이어야 비율이 94.9% 를 넘는다.

반대로 94% 를 노린 구간을 $\sigma$ 를 아는 자리에서 그리려면 배율은 1.88 이다. 1.96 과 94% 를 함께 적은 글은 이 배율을 말하는 것이 아니라, RMSE 로 오차의 표준편차를 추정한 자리에서 n 이 30 안팎일 때 나오는 값을 적은 것이다.

### 5.2 Degrees of Freedom with Fitted Parameters

Model 을 자료에 맞추어 parameter 를 p 개 추정하고 그 잔차로 RMSE 를 구했으면 자유도가 $\nu = n - p$ 로 줄고, 같은 1.96 이 담는 비율이 더 내려간다.

```math
P\left(|e^{\ast}| \le 1.96\,\mathrm{RMSE}\right) = 2F_{t_{\nu}}\!\left(1.96\sqrt{\frac{\nu}{n}}\right) - 1, \qquad \nu = n - p \hspace{19em} (7)
```

식 (7) 은 p 가 0 이면 식 (6) 으로 돌아간다. n 이 28 일 때 p 가 2 이면 0.9299, 5 이면 0.9111 이다. n 이 100 이고 p 가 5 이면 0.9409 로, 같은 0.94 를 세 배가 넘는 자료로 얻는다.

식 (7) 은 새 오차가 RMSE 를 구한 잔차와 독립이고 분산이 $\sigma^2$ 인 경우다. 같은 자료 안의 점을 예측하면 model 이 그 점에 맞춰져 있어 잔차가 더 작고, 예측값 자체가 흔들리는 몫이 더해져 분산이 $\sigma^2$ 보다 크므로 식 (7) 의 값은 실제보다 높게 나온다.

### 5.3 Conditions

- **가정**: 오차가 평균 0 의 정규분포에서 독립으로 나오고 분산이 모두 같다. RMSE 를 구한 오차와 담을 대상인 새 오차가 서로 독립이다.
- **설정값**: 배율은 $t_{\nu}(0.975)$ 이고 자유도는 $\nu = n - p$ 이며, p 는 같은 자료로 추정한 parameter 의 개수다. 오차 개수 n 은 RMSE 의 정의인 식 (2) 의 분모에 따로 남는다.
- **깨지는 조건**: 오차의 평균이 0 이 아니면 RMSE 가 흩어짐과 치우침을 함께 담아 구간이 필요보다 넓어지고, 식 (6) 의 비율은 더 이상 그 구간을 설명하지 않는다. 분산이 자리마다 다르면 (heteroscedasticity) 하나의 RMSE 가 모든 자리를 대표하지 못해 분산이 큰 자리에서 비율이 떨어진다. 오차가 서로 상관되어 있으면 제곱합에 남는 자유도가 n 보다 작아 식 (6) 이 비율을 높게 낸다. 꼬리가 정규분포보다 두꺼우면 같은 배율이 담는 비율이 더 낮아, 분산이 같은 Laplace distribution 에서 $\pm 1.96\sigma$ 가 담는 비율은 93.7% 다.
- **만나는 자리**: 계측 두 방법의 차이에 한계를 긋는 Bland-Altman 한계 [[3](#ref-3)], 회귀 model 의 예측 오차 범위 보고, 공정 자료의 예측값에 붙이는 오차 막대.

## 6. Spec Setting for SPC

RMSE 로 spec 을 정하는 일은 네 걸음이다. 한계를 무엇에 긋는지 가르고, 담을 비율에서 배율을 읽고, 예측값이 판정에 쓰이면 guard band 만큼 물러서고, 그 spec 을 관리 한계와 따로 둔다.

### 6.1 What the Limit Is Drawn On

RMSE 가 정하는 것은 model 의 오차에 긋는 한계와 그 오차 때문에 spec 에서 물러서는 폭 둘이고, 공정이 평소와 같은지를 가르는 관리 한계는 공정의 변동에서 따로 나온다.

Table 3. Limits around a predicted value

| #   | Limit            | Set from     | Offset               | What it decides               |
| :-: | :--------------: | :----------: | :------------------: | :---------------------------: |
| 1   | Error limit      | RMSE         | $k\,\mathrm{RMSE}$   | 한 예측의 오차를 받아들일지   |
| 2   | Acceptance limit | Spec 과 RMSE | $g\,\mathrm{RMSE}$   | 예측값으로 규격 판정을 내릴지 |
| 3   | Control limit    | 공정의 변동  | 관측 표준편차의 3 배 | 공정이 평소와 달라졌는지      |

1 행은 model 이 계측을 대신해도 되는 범위다. 2 행은 이미 있는 spec 에서 안쪽으로 물러선 자리이며, 물러선 폭이 guard band 다. 3 행의 한계는 spec 에서 나오지 않지만, 예측값으로 chart 를 그리면 그 폭에 RMSE 가 식 (10) 으로 섞여 든다.

### 6.2 The Limit from the RMSE

한계는 목표값에서 배율과 RMSE 의 곱만큼 떨어진 두 자리다.

```math
\mathrm{LSL},\ \mathrm{USL} = \mu_0 \mp k\,\mathrm{RMSE}, \qquad k = \sqrt{\frac{n}{\nu}}\; t_{\nu}\!\left(1 - \frac{\alpha}{2}\right) \hspace{19em} (8)
```

식 (8) 의 $\mu_0$ 는 Table 3 의 1 행에서 0 이고 2 행에서 공정의 목표값이며, $\alpha$ 는 1 에서 담을 비율을 뺀 값이다. $\alpha$ 가 0.05 이면 $k$ 는 Table 2 의 마지막 열이고, n 이 28 이면 2.05 다. 쓰기 전에 아래 세 가지를 확인한다.

- **Bias 를 먼저 뺀다**: 오차의 평균이 0 이 아니면 RMSE 가 치우침과 흩어짐을 함께 담아 (꼭지 5.3) 한계가 필요보다 넓어진다. 치우침을 model 에서 고친 뒤 RMSE 를 다시 구한다.
- **Model 을 맞추는 데 쓰지 않은 자료로 구한다**: 같은 자료의 잔차로 RMSE 를 구하면 식 (7) 이 비율을 높게 낸다. 자유도는 $\nu = n - p$ 이고, p 는 그 자료로 추정한 parameter 의 개수다.
- **오차 개수를 먼저 센다**: n 이 30 안팎이면 1.96 이 담는 비율이 94% 이므로, 95% 를 spec 에 적으려면 배율을 Table 2 에서 바꾸어 읽는다.

### 6.3 The Guard Band

예측값이 계측값을 대신해 규격 판정을 내리면 spec 안쪽으로 물러선 수락 한계 (acceptance limit) 를 쓴다.

```math
A_{\mathrm{L}} = \mathrm{LSL} + g\,\mathrm{RMSE}, \qquad A_{\mathrm{U}} = \mathrm{USL} - g\,\mathrm{RMSE} \hspace{19em} (9)
```

식 (9) 의 두 값 사이에 들어온 것만 받아들이고, spec 과 수락 한계 사이의 폭 $g\,\mathrm{RMSE}$ 가 guard band 다. Spec 바로 안쪽에서 측정된 것도 실제로는 밖에 있을 확률이 RMSE 만큼 남아 있고, guard band 가 그 확률, 곧 consumer's risk 를 내린다 [[5](#ref-5)]. ISO 14253-1 의 기본 규칙은 확장 불확도 (expanded uncertainty) 한 배를 spec 에서 빼는 것이고 [[6](#ref-6)], 확장 불확도를 RMSE 의 두 배로 잡으면 $g = 2$ 가 된다. $g$ 를 키우면 consumer's risk 가 내려가는 대신 규격 안의 것을 거부할 확률, 곧 producer's risk 가 올라간다.

### 6.4 The Spec and the Control Limit

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

- **chi-squared distribution**: 독립인 표준정규분포 값 여러 개를 제곱해 더한 값이 따르는 분포. 식 (11) 이 정의다.
- **concave**: 2차 도함수가 음수여서 곡선이 위로 볼록한 함수의 성질.
- **confidence interval**: 추정하려는 모수가 들어 있을 확률을 정해 둔 구간.
- **consumer's risk**: 받아들인 것이 규격을 벗어나 있을 확률.
- **control limit**: 공정이 평소와 같은지를 가르는 chart 의 한계. 공정의 변동에서 구한다.
- **coverage**: 구간이 담으려는 대상을 실제로 담을 확률.
- **degrees of freedom**: 제곱합에 남아 있는 독립한 성분의 개수.
- **expanded uncertainty**: 표준불확도에 포함인자를 곱한 값. 포함인자 2 가 관례다.
- **gamma function**: 식 (15) 의 적분으로 정의되는 함수. 양의 정수에서 계승 (factorial) 을 확장한 값이 된다.
- **guard band**: Spec 과 수락 한계 사이의 폭. 식 (9) 가 정의다.
- **heteroscedasticity**: 분산이 자리마다 다른 상태.
- **Jensen's inequality**: concave 함수에서 기댓값의 함수가 함수의 기댓값보다 크다는 부등식.
- **Laplace distribution**: 양쪽 꼬리가 지수함수로 줄어드는 대칭 분포. 분산이 같은 정규분포보다 꼬리가 두껍다.
- **lower incomplete gamma function**: gamma function 의 적분 구간을 유한한 위끝에서 끊은 함수. 식 (14) 이 정의다.
- **Monte Carlo**: 난수로 표본을 만들어 확률을 추정하는 방법.
- **prediction interval**: 새 관측 하나가 들어 있을 확률을 정해 둔 구간.
- **process capability index**: Spec 의 폭을 공정 변동의 6 배로 나눈 값.
- **producer's risk**: 거부한 것이 규격 안에 있을 확률.
- **regularized incomplete gamma function**: lower incomplete gamma function 을 gamma function 으로 나눈 값. 식 (13) 의 우변이 그 값이다.
- **RMSE**: 오차 제곱의 평균에 제곱근을 취한 값. 식 (2) 가 정의다.
- **spec limit**: 요구에서 나온 상한과 하한. 공정의 변동과는 따로 정해진다.
- **Student t distribution**: 표준정규분포 값을, 그와 독립인 chi-squared 값을 자유도로 나눈 것의 제곱근으로 나눈 비가 따르는 분포.
- **tolerance factor**: tolerance interval 에서 표준편차에 곱하는 배율. 비율과 신뢰수준과 표본 수로 정해진다.
- **tolerance interval**: 모집단의 정해진 비율을 담는다는 것을 정해진 신뢰수준으로 보장하는 구간.

## Appendix B. Chi-squared Distribution

### B.1 Definition

서로 독립이고 표준정규분포 $N(0, 1)$ 를 따르는 $k$ 개의 확률변수 $Z_1, Z_2, \dots, Z_k$ 가 있을 때, 이 변수들의 제곱합으로 정의되는 확률변수 $X$ 는 자유도가 $k$ 인 chi-squared distribution 을 따른다.

```math
X = \sum_{i=1}^{k} Z_i^2 = Z_1^2 + Z_2^2 + \dots + Z_k^2 \sim \chi^2(k) \hspace{19em} (11)
```

- $k$ (자유도, degrees of freedom): 합산되는 독립 표준정규분포 변수의 개수.

본문은 같은 자유도를 n 과 $\nu$ 로 적는다. 식 (16) 의 $S$ 는 식 (11) 의 $X$ 에 $k = n$ 을 넣은 것이고, 식 (19) 의 $S_{\nu}$ 는 $k = \nu$ 를 넣은 것이다.

### B.2 Probability Density Function

자유도가 $k$ 인 chi-squared distribution 의 확률밀도함수 (probability density function) 는 $x \gt 0$ 에서 아래와 같다.

```math
f(x; k) = \frac{1}{2^{k/2}\,\Gamma(k/2)}\; x^{(k/2) - 1}\, e^{-x/2} \hspace{19em} (12)
```

식 (12) 의 $\Gamma$ 는 gamma function 이며, 식 (15) 가 gamma function 의 정의다.

### B.3 Cumulative Distribution Function

누적분포함수 $F(x; k)$ 는 확률변수 $X$ 가 특정 값 $x$ 이하일 확률 $P(X \le x)$ 를 뜻하며, 확률밀도함수를 0 부터 $x$ 까지 적분하여 구한다.

```math
F(x; k) = P(X \le x) = \frac{1}{\Gamma(k/2)}\; \gamma\!\left(\frac{k}{2}, \frac{x}{2}\right) \quad (x \ge 0) \hspace{19em} (13)
```

식 (13) 의 $\gamma(s, t)$ 는 하부 불완전 감마 함수 (lower incomplete gamma function) 이고, $\Gamma(s)$ 는 적분의 위끝을 무한대로 늘린 gamma function 이다.

```math
\gamma(s, t) = \int_0^t u^{s-1} e^{-u}\, du \hspace{19em} (14)
```

```math
\Gamma(s) = \int_0^{\infty} u^{s-1} e^{-u}\, du \hspace{19em} (15)
```

Chi-squared distribution 의 누적분포함수는 닫힌 형태 (elementary function) 로 단순하게 표현되지 않는다. 그래서 식 (13) 처럼 정규화 불완전 감마 함수 (regularized incomplete gamma function) 의 꼴로 정의하고, 실제 계산에는 numerical 방법이나 R, Python, Excel 이 담은 통계 함수를 쓴다.

### B.4 Properties

- **값의 범위**: $X \ge 0$. 제곱합이므로 음수가 되지 않는다.
- **평균**: $E(X) = k$.
- **분산**: $\mathrm{Var}(X) = 2k$.
- **모양**: $k$ 가 작을수록 오른쪽으로 긴 꼬리를 가진 비대칭 형태이며, $k$ 가 커질수록 점점 정규분포 모양에 가깝게 대칭형으로 변한다.

## Appendix C. Derivation of Equations (3) to (7)

출발점은 오차 하나를 $\sigma$ 로 나눈 값이 표준정규분포를 따른다는 것 하나다. n 개를 제곱해 더하면 자유도 n 의 chi-squared distribution 이 된다.

```math
S = \sum_{i=1}^{n} \left(\frac{e_i}{\sigma}\right)^2 \sim \chi^2_n, \qquad S = \frac{n\,\mathrm{RMSE}^2}{\sigma^2} \hspace{19em} (16)
```

두 번째 등식은 식 (2) 의 양변을 제곱해 n 을 곱한 것이며, 이것이 식 (3) 이다. 식 (4) 는 여기서 제곱근의 기댓값으로 나온다. 자유도 n 의 chi-squared 값에 제곱근을 취한 값의 기댓값이 아래와 같다.

```math
E\left[\sqrt{S}\right] = \sqrt{2}\;\frac{\Gamma\!\left(\frac{n+1}{2}\right)}{\Gamma\!\left(\frac{n}{2}\right)} \hspace{19em} (17)
```

식 (16) 에서 $\mathrm{RMSE} = \sigma\sqrt{S/n}$ 이므로 양변에 기댓값을 취하고 식 (17) 를 넣으면 식 (4) 의 $c_n$ 이 그대로 나온다.

식 (5) 는 t distribution 의 정의에서 나온다. 새 오차 $e^{\ast}$ 는 $e_1, \dots, e_n$ 과 독립이므로 $Z = e^{\ast}/\sigma$ 는 S 와 독립인 표준정규분포 값이다.

```math
\frac{e^{\ast}}{\mathrm{RMSE}} = \frac{\sigma Z}{\sigma\sqrt{S/n}} = \frac{Z}{\sqrt{S/n}} \sim t_n \hspace{19em} (18)
```

식 (18) 의 가운데에서 $\sigma$ 가 약분되므로 담는 비율은 모르는 $\sigma$ 와 무관해지고 n 하나로 정해진다. 좌변의 절댓값이 1.96 이하일 확률을 t distribution 의 누적분포함수로 적은 것이 식 (6) 이다.

Parameter 를 p 개 추정한 경우에는 제곱합의 자유도가 $\nu = n - p$ 로 줄지만, RMSE 는 식 (2) 대로 n 으로 나눈다. 자유도만큼의 chi-squared 값을 $S_{\nu}$ 로 두면 두 수가 따로 남는다.

```math
\frac{e^{\ast}}{\mathrm{RMSE}} = \frac{Z}{\sqrt{S_{\nu}/n}} = \sqrt{\frac{n}{\nu}}\;\frac{Z}{\sqrt{S_{\nu}/\nu}} \sim \sqrt{\frac{n}{\nu}}\;t_{\nu} \hspace{19em} (19)
```

식 (19) 의 좌변이 1.96 이하일 조건은 $t_{\nu}$ 가 $1.96\sqrt{\nu/n}$ 이하일 조건과 같고, 이것을 누적분포함수로 적은 것이 식 (7) 이다. 식 (4) 의 $c_n$ 도 같은 자리를 바꾸어 $\sqrt{\nu/n}$ 배만큼 더 작아진다.
