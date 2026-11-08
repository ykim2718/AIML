# Measured Wafer Ratio for R2 Retention
Version 15 | Created: 2026-11-05 | Updated: 2026-11-08 03:17 UTC

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Notation](#3-notation)
- [4. Definitions](#4-definitions)
- [5. Interpretation](#5-interpretation)
  - [5.1 Variance of R2](#51-variance-of-r2)
  - [5.2 Retention Condition](#52-retention-condition)
  - [5.3 Measured Count and Interval](#53-measured-count-and-interval)
- [6. Worked Example](#6-worked-example)
  - [6.1 Retention Probability by Measured Count](#61-retention-probability-by-measured-count)
  - [6.2 Measured Count and Interval](#62-measured-count-and-interval)
  - [6.3 Condition Comparison](#63-condition-comparison)
- [7. Application](#7-application)
  - [7.1 Calculation Script](#71-calculation-script)
  - [7.2 Breaking Conditions](#72-breaking-conditions)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Derivation](#appendix-b-derivation)

## 1. Purpose

- **Problem Statement**: 반도체 증착 공정을 진행하면서 주기적으로 일부 wafer 에서 두께를 측정하면, 실측과 예측의 비교를 통해 예측값의 R² 를 동적으로 추출할 수 있고, 측정하지 않은 wafer 에 대한 예측값의 정확도를 알 수 있다. 그러나 예측값의 정확도를 검증하는 데 필요한 measurement interval 이나 측정 wafer 수를 계산하는 방법은 없다.
- **Goal**: model 의 R² 예측값 $\hat{R}^2_c$ 로 유지 확률 $P_{\mathrm{ret}}(m)$ 을 구하고, 그 유지 확률로부터 wafer measurement count $m^{\ast}$ 와 measurement interval $T^{\ast}$ 를 구하는 수학 model 을 제시한다.
- **Non-Goal**: 두께 예측 model 을 만들거나 고르는 방법은 다루지 않는다.

## 2. Summary

실측 wafer $m$ 장으로 계산한 R² 가 최저 R² $R^2_{\min}$ 이상일 유지 확률 $P_{\mathrm{ret}}(m)$ 은 model 의 R² 예측값 $\hat{R}^2_c$ 와 $m$ 으로 정해지며, $m$ 이 늘수록 커진다.
$P_{\mathrm{ret}}(m)$ 이 목표 유지 확률 $\gamma$ 이상이 되는 가장 작은 $m$ 이 wafer measurement count $m^{\ast}$ 이고, measurement interval 은 $T^{\ast} = N/m^{\ast}$ 이다.
$\hat{R}^2_c$ 가 높을수록, $\hat{R}^2_c - R^2_{\min}$ 이 클수록, $\gamma$ 가 낮을수록 $m^{\ast}$ 가 줄고 $T^{\ast}$ 가 길어진다.

## 3. Notation

Table 1. Symbols

|         Symbol          |                         Meaning                          |  Shape  |            Assumption             |
| :---------------------: | :------------------------------------------------------: | :-----: | :-------------------------------: |
|           $N$           |           한 lot 또는 한 기간의 전체 wafer 수            | integer |             $N \ge 2$             |
|           $m$           |           실측 wafer 수. 예측 wafer 수는 $N-m$           | integer |          $1 \le m \le N$          |
|           $r$           |                     실측 wafer 비율                      | scalar  |          $0 \lt r \le 1$          |
|          $y_i$          |             $i$ 번째 실측 wafer 의 실측 두께             | scalar  |                 -                 |
|       $\hat{y}_i$       |       $i$ 번째 실측 wafer 에 대한 model 예측 두께        | scalar  |                 -                 |
|        $\bar{y}$        |                 실측 두께 $m$ 개의 평균                  | scalar  |                 -                 |
|        $R^2_m$          |   실측 R². 실측 wafer $m$ 장의 실측 두께로 계산한 R²   | scalar  |                 -                 |
|         $R^2$           | 참값 R². 측정하지 않은 wafer 를 포함한 전체 wafer 에서 예측값의 R² | scalar  |          $0 \lt R^2 \lt 1$          |
|         $\hat{R}^2_c$         |  model 의 R² 예측값. model 이 과거 실측으로 계산해 둔 값  | scalar  |        $0 \lt \hat{R}^2_c \lt 1$        |
|      $R^2_{\min}$       |                   유지해야 할 최저 R²                    | scalar  |      $R^2_{\min} \lt \hat{R}^2_c$       |
|        $\Delta$         |              허용 여유 $\hat{R}^2_c - R^2_{\min}$              | scalar  |          $\Delta \gt 0$           |
| $P_{\mathrm{ret}}(m)$   |  유지 확률. $R^2_m \ge R^2_{\min}$ 이 될 확률       | scalar  | $0 \le P_{\mathrm{ret}}(m) \le 1$ |
|        $\alpha$         |        $R^2_m \lt R^2_{\min}$ 이 될 허용 확률        | scalar  |      $0 \lt \alpha \lt 0.5$       |
|        $\gamma$         |         목표 유지 확률. $\gamma = 1 - \alpha$          | scalar  |      $0.5 \lt \gamma \lt 1$       |
|       $P(\cdot)$        |                           확률                           | scalar  |                 -                 |
|      $\Phi(\cdot)$      |               표준정규분포의 누적분포함수                | scalar  |                 -                 |
|  $\mathrm{Var}(\cdot)$  |                           분산                           | scalar  |                 -                 |
|      $z_{\gamma}$       |   표준정규분포의 $\gamma$ 분위수. $\Phi^{-1}(\gamma)$    | scalar  |                 -                 |
|         $\rho$          |             실측 두께와 예측 두께의 상관계수             | scalar  |        $0 \lt \rho \lt 1$         |
|      $\hat{\rho}$       |          실측 wafer $m$ 장으로 추정한 상관계수           | scalar  |                 -                 |
|          $m_0$          |            유한 모집단 보정 전의 필요 실측 수            | scalar  |                 -                 |
|       $m_{\min}$        |             대표본 근사를 위한 실측 수 하한              | integer | $2 \le m_{\min} \le N$. 기본값 10 |
|       $m^{\ast}$        |     wafer measurement count. 최소 실측 wafer 수      | integer |                 -                 |
|       $r^{\ast}$        |                      최소 실측 비율                      | scalar  |                 -                 |
|       $T^{\ast}$        | Measurement interval. 실측 wafer 한 장당 공정을 거치는 wafer 수 | scalar  |         $T^{\ast} \ge 1$          |

R² 의 아래첨자는 그 R² 를 무엇으로 구했는지를 나타낸다. $m$ 은 measured 로 실측 두께와 견주어 확인한 값을, $c$ 는 calculated by model 로 model 이 계산해 둔 값을 뜻한다. $\min$ 은 유지해야 할 최저값을, $\mathrm{ret}$ 은 retention 을 뜻한다. 목표 유지 확률은 허용 확률 $\alpha$ 를 1 에서 뺀 값 $\gamma = 1 - \alpha$ 하나로 적는다.

Hat ( ˆ ) 은 추정값, 예측값, 적합값에만 붙인다. $\hat{y}_i$ 와 $\hat{R}^2_c$ 는 model 의 예측값이고, $\hat{\rho}$ 는 실측 wafer $m$ 장으로 구한 추정값이다. Hat 이 없는 $R^2_m$ 은 실측 wafer $m$ 장에서 계산한 실측값이고, $R^2$ 과 $\rho$ 는 참값이다.

- $\hat{y}$: 영어로 "y hat", 한국어로 "와이 햇" 이라 읽는다.
- $\bar{y}$: 영어로 "y bar", 한국어로 "와이 바" 라 읽는다.
- $\hat{R}^2$: 영어로 "R hat squared", 한국어로 "알 햇 제곱" 이라 읽는다.
- $\hat{\rho}$: 영어로 "rho hat", 한국어로 "로 햇" 이라 읽는다.
- $\Phi^{-1}$: 영어로 "Phi inverse", 한국어로 "파이 역함수" 라 읽는다.
- $\sum$: 영어로 "sum", 한국어로 "합" 이라 읽는다.
- $\sqrt{x}$: 영어로 "square root of x", 한국어로 "루트 x" 라 읽는다.
- $\lceil x \rceil$: 영어로 "ceiling of x", 한국어로 "x 의 올림" 이라 읽는다.
- $\iff$: 영어로 "if and only if", 한국어로 "필요충분조건" 또는 "동치" 라 읽는다.
- $\{ r \mid \cdot \}$: 영어로 "the set of r such that", 한국어로 "조건을 만족하는 r 의 집합" 이라 읽는다.
- $m^{\ast}$: 영어로 "m star", 한국어로 "엠 스타" 라 읽는다.

## 4. Definitions

> **Definition: Measured R2**<br>
> 실측 R² $R^2_m$ 은 실측 wafer $m$ 장의 실측 두께 $y_i$ 와 model 예측 두께 $\hat{y}_i$ 로 계산한 R² 이며, 1 에서 예측 오차의 제곱합 (SSE) 을 실측 두께가 평균 $\bar{y}$ 에서 벗어난 제곱합 (SST) 으로 나눈 값을 뺀 값이다.

> **Definition: Measured ratio**<br>
> 실측 비율 $r$ 은 전체 wafer $N$ 장 가운데 실측한 wafer $m$ 장이 차지하는 비율이다.

> **Definition: Retention probability**<br>
> 유지 확률 $P_{\mathrm{ret}}(m)$ 은 실측 wafer $m$ 장으로 계산한 실측 R² $R^2_m$ 이 최저 R² $R^2_{\min}$ 이상일 확률이다.

> **Definition: Minimum measured ratio**<br>
> 최소 실측 비율 $r^{\ast}$ 는 유지 확률 $P_{\mathrm{ret}}(m)$ 이 목표 유지 확률 $\gamma$ 이상이 되는 실측 비율 $r$ 가운데 가장 작은 값이고, 그때의 실측 wafer 수가 wafer measurement count $m^{\ast}$ 이다.

> **Definition: Measurement interval**<br>
> Measurement interval $T^{\ast}$ 는 최소 실측 비율 $r^{\ast}$ 로 실측할 때 실측 wafer 한 장당 공정을 거치는 wafer 수의 평균이며, $r^{\ast}$ 의 역수이다.

다섯 정의 식은 아래 식 (1) 부터 식 (5) 이다. Problem Statement 의 측정 wafer 수는 wafer measurement count $m^{\ast}$ 이고, measurement interval 은 식 (5) 의 $T^{\ast}$ 이다.

```math
R^2_m = 1 - \frac{\sum_{i=1}^{m} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{m} (y_i - \bar{y})^2}
\hspace{19em} (1)
```

```math
r = \frac{m}{N}
\hspace{19em} (2)
```

```math
P_{\mathrm{ret}}(m) = P\left( R^2_m \ge R^2_{\min} \right)
\hspace{19em} (3)
```

```math
r^{\ast} = \min \left\{ r = \frac{m}{N} \;\middle|\; P_{\mathrm{ret}}(m) \ge \gamma \right\}
\hspace{19em} (4)
```

```math
T^{\ast} = \frac{1}{r^{\ast}}
\hspace{19em} (5)
```

- 식 (1) 의 분자는 SSE, 분모는 SST 이다.
- 실측 wafer 가 바뀌면 분자와 분모가 함께 바뀌므로, model 이 같아도 $R^2_m$ 은 참값 $R^2$ 주위에서 변동한다. 그래서 $m \lt N$ 이면 식 (3) 의 유지 확률은 1 보다 작다.

## 5. Interpretation

식 (3) 의 유지 확률은 식 (6) 으로 계산한다.

```math
P_{\mathrm{ret}}(m) \approx \Phi\left( \frac{\hat{R}^2_c - R^2_{\min}}{\sqrt{\mathrm{Var}(R^2_m)}} \right)
\hspace{19em} (6)
```

식 (6) 에서 분자는 허용 여유 $\Delta$ 이고, 분모는 식 (7) 로 구하는 $R^2_m$ 의 표준편차이다. $m$ 이 늘면 분모가 줄어 $P_{\mathrm{ret}}(m)$ 이 커지며, 유도는 [Appendix B](#appendix-b-derivation) 에 있다.

### 5.1 Variance of R2

$R^2_m$ 의 분산은 식 (7) 이다. 유도는 [Appendix B](#appendix-b-derivation) 에 있다.

```math
\mathrm{Var}(R^2_m) \approx \frac{4 \hat{R}^2_c (1 - \hat{R}^2_c)^2}{m} \cdot \frac{N - m}{N - 1}
\hspace{19em} (7)
```

- 앞 인자: 대표본 근사 (large-sample approximation) 에서 delta method 로 구한 R² 의 분산
- 뒤 인자: 유한 모집단 보정 (finite population correction, FPC). $m = N$ 이면 0 이 되어, 전체를 실측하면 변동이 없다.
- $\hat{R}^2_c$ 가 1 에 가까울수록 $(1 - \hat{R}^2_c)^2$ 이 작아져 분산이 줄어든다.

식 (6) 과 식 (7) 은 아래 조건에서 성립한다.

- 예측 두께에 bias 가 없고 실측 두께에 대한 기울기가 1 이어서, 참값 $R^2$ 이 실측과 예측의 상관계수 제곱 $\rho^2$ 과 같다.
- 실측 R² $R^2_m$ 의 기댓값이 참값 $R^2$ 이고, model 의 R² 예측값 $\hat{R}^2_c$ 가 참값 $R^2$ 과 같다.
- $m$ 이 충분히 커서 $R^2_m$ 이 정규분포에 가깝다. 이 문서는 $m \ge m_{\min} = 10$ 으로 둔다.
- 실측 wafer 를 전체 wafer 에서 무작위로 고른다.

### 5.2 Retention Condition

유지 확률이 목표 유지 확률 $\gamma$ 이상이라는 조건은 식 (8) 이다.

```math
P_{\mathrm{ret}}(m) \ge \gamma
\iff
\hat{R}^2_c - z_{\gamma} \sqrt{\mathrm{Var}(R^2_m)} \ge R^2_{\min}
\hspace{19em} (8)
```

- 좌변: 식 (6) 의 유지 확률이 목표 유지 확률 이상이라는 조건
- 우변: $R^2_m$ 분포의 아래쪽 $\alpha$ 분위수가 $R^2_{\min}$ 이상이라는 조건. $z_{\gamma} = \Phi^{-1}(\gamma)$ 는 식 (6) 의 $\Phi$ 를 거꾸로 푼 값이다.

### 5.3 Measured Count and Interval

식 (8) 을 만족하는 가장 작은 $m$ 은 식 (9), 그 계산에 쓰는 $m_0$ 는 식 (10) 이다. 최소 실측 비율과 measurement interval 은 식 (9) 의 $m^{\ast}$ 를 식 (2) 와 식 (5) 에 넣은 식 (11) 과 식 (12) 이다.

```math
m^{\ast} = \max\left( \left\lceil \frac{m_0}{1 + (m_0 - 1)/N} \right\rceil,\ m_{\min} \right)
\hspace{19em} (9)
```

```math
m_0 = \left( \frac{2\, z_{\gamma} \sqrt{\hat{R}^2_c}\, (1 - \hat{R}^2_c)}{\hat{R}^2_c - R^2_{\min}} \right)^2
\hspace{19em} (10)
```

```math
r^{\ast} = \frac{m^{\ast}}{N}
\hspace{19em} (11)
```

```math
T^{\ast} = \frac{N}{m^{\ast}}
\hspace{19em} (12)
```

- $m_0$: 전체 wafer 수가 무한할 때의 필요 실측 수
- $m_0 / (1 + (m_0 - 1)/N)$: 유한 모집단 보정을 넣은 필요 실측 수. $N$ 이 작을수록 $m_0$ 보다 많이 줄어든다.
- $m_{\min}$: 5.1 의 대표본 근사가 성립하도록 둔 하한

식 (9) 는 $\hat{R}^2_c \gt R^2_{\min}$ 일 때만 성립한다. $\hat{R}^2_c \le R^2_{\min}$ 이면 식 (6) 의 분자가 0 이하여서 $m$ 을 $N$ 까지 늘려도 $P_{\mathrm{ret}}(m)$ 이 0.5 를 넘지 않고, 식 (8) 을 만족하지 못한다.

## 6. Worked Example

### 6.1 Retention Probability by Measured Count

$N = 100$, $\hat{R}^2_c = 0.90$, $R^2_{\min} = 0.85$ 로 식 (6) 과 식 (7) 을 $m$ 마다 계산하면 아래 Table 2 이다.

Table 2. Retention probability by measured count, N = 100

|  m  | Std of R2 |   z   | P_ret(m) | Meets gamma = 0.95 |
| :-: | :-------: | :---: | :------: | :----------------: |
| 10  |  0.0572   | 0.874 |  0.809   |         No         |
| 20  |  0.0381   | 1.311 |  0.905   |         No         |
| 28  |  0.0306   | 1.635 |  0.949   |         No         |
| 29  |  0.0298   | 1.676 |  0.953   |        Yes         |
| 40  |  0.0234   | 2.141 |  0.984   |        Yes         |
| 60  |  0.0156   | 3.211 |  0.999   |        Yes         |

`Std of R2` 는 식 (7) 의 제곱근이고, `z` 는 허용 여유 0.05 를 그 값으로 나눈 값이다. $P_{\mathrm{ret}}(m)$ 은 $m$ 이 늘수록 커지며, 목표 유지 확률 0.95 이상이 되는 가장 작은 $m$ 은 29 이다.

### 6.2 Measured Count and Interval

같은 조건에 $\gamma = 0.95$ 를 더해 식 (9) 부터 식 (12) 를 계산한다.

1. $z_{0.95} = \Phi^{-1}(0.95) = 1.645$
2. $m_0 = (2 \times 1.645 \times \sqrt{0.90} \times 0.10 / 0.05)^2 = (0.3121 / 0.05)^2 = 38.96$
3. $m_0 / (1 + (m_0 - 1)/N) = 38.96 / 1.3796 = 28.24$, 올림하면 29
4. $m^{\ast} = \max(29,\ 10) = 29$, $r^{\ast} = 29/100 = 0.29$
5. $T^{\ast} = 100/29 = 3.45$ 이므로, wafer 평균 3.45 장마다 한 장을 실측한다.

식 (9) 로 구한 $m^{\ast} = 29$ 는 Table 2 에서 $P_{\mathrm{ret}}(m)$ 이 처음으로 0.95 이상이 되는 $m$ 과 같다.

### 6.3 Condition Comparison

조건별 $m^{\ast}$ 와 $r^{\ast}$ 는 아래 Table 3, Table 4, Table 5 이다. Cell 은 `m* (r*)` 이고, * 는 $m_{\min} = 10$ 이 $m^{\ast}$ 를 정한 경우이다. 각 cell 의 measurement interval $T^{\ast}$ 는 괄호 안 $r^{\ast}$ 의 역수이다.

Table 3. Measured count by model R2 and margin, N = 100, gamma = 95 %

| R2_c |  Margin = 0.03  |  Margin = 0.05  |  Margin = 0.10  |
| :--: | :-------------: | :-------------: | :-------------: |
| 0.80 |   80 (0.80)     |   59 (0.59)     |   26 (0.26)     |
| 0.85 |   70 (0.70)     |   46 (0.46)     |   18 (0.18)     |
| 0.90 |   53 (0.53)     |   29 (0.29)     |   10\* (0.10)   |
| 0.95 |   23 (0.23)     |   10 (0.10)     |   10\* (0.10)   |

$\Delta$ 를 0.03 에서 0.10 으로 넓히면 $\hat{R}^2_c = 0.80$ 에서 $r^{\ast}$ 가 0.80 에서 0.26 으로 줄어든다.

Table 4. Measured count by retention probability, N = 100, margin = 0.05

| R2_c |     90 %      |     95 %      |     99 %      |
| :--: | :-----------: | :-----------: | :-----------: |
| 0.80 |  46 (0.46)    |  59 (0.59)    |  74 (0.74)    |
| 0.85 |  34 (0.34)    |  46 (0.46)    |  63 (0.63)    |
| 0.90 |  20 (0.20)    |  29 (0.29)    |  45 (0.45)    |
| 0.95 | 10\* (0.10)   |  10 (0.10)    |  18 (0.18)    |

목표 유지 확률을 95 % 에서 99 % 로 올리면 $\hat{R}^2_c = 0.90$ 에서 $r^{\ast}$ 가 0.29 에서 0.45 로 늘어난다.

Table 5. Measured count by total wafers, gamma = 95 %, margin = 0.05

| R2_c |    N = 25     |    N = 100    |
| :--: | :-----------: | :-----------: |
| 0.80 |  22 (0.88)    |  59 (0.59)    |
| 0.85 |  20 (0.80)    |  46 (0.46)    |
| 0.90 |  16 (0.64)    |  29 (0.29)    |
| 0.95 | 10\* (0.40)   |  10 (0.10)    |

$N$ 이 25 이면 $m^{\ast}$ 는 줄지만 $r^{\ast}$ 는 커진다. $\hat{R}^2_c = 0.90$ 에서 $N = 25$ 의 $r^{\ast}$ 는 0.64 로 $N = 100$ 의 0.29 보다 두 배 넘게 크다.

## 7. Application

### 7.1 Calculation Script

식 (1), 식 (7), 식 (9) 부터 식 (11) 을 계산하는 Python script 는 [src/r2_sampling_ratio.py](src/r2_sampling_ratio.py) 이다.

- `estimate_r2(measured=..., predicted=...)`: 식 (1) 의 $R^2_m$
- `r2_variance(r2_model=..., measured=..., total_wafers=...)`: 식 (7) 의 분산
- `required_measured(r2_model=..., r2_min=..., confidence=..., total_wafers=..., min_measured=...)`: 식 (9) 부터 식 (11) 의 $m_0$, $m^{\ast}$, $r^{\ast}$
- `build_table(...)`: 여러 조건의 $m^{\ast}$, $r^{\ast}$ 를 한 표로 계산

식 (6) 의 유지 확률은 `r2_variance` 의 결과를 식 (6) 에 넣어 구하고, 식 (12) 의 measurement interval 은 결과 열 `ratio` 의 역수로 구한다. 허용 여유는 `--margin` 으로 $\Delta$ 를 주거나 `--r2-min` 으로 $R^2_{\min}$ 을 직접 주며, 둘을 함께 주면 오류로 끝난다. Table 3 은 아래 명령으로 계산한다.

```bash
python src/r2_sampling_ratio.py --r2-model 0.80 0.85 0.90 0.95 --margin 0.03 0.05 0.10 --total-wafers 100 --confidence 0.95
```

`--output-folder <FOLDER>` 를 주면 결과를 `<FOLDER>/r2_sampling_ratio.csv` 로 저장한다. 열은 `total_wafers`, `r2_model`, `r2_min`, `confidence`, `n0`, `measured`, `floored`, `ratio` 이다.

### 7.2 Breaking Conditions

- $\hat{R}^2_c \le R^2_{\min}$: 식 (10) 의 분모가 0 이하가 되어 $m^{\ast}$ 가 없다. `required_measured` 는 오류로 끝나고, `build_table` 은 이 조건을 건너뛰고 건너뛴 개수를 출력한다.
- $\hat{R}^2_c$ 가 과거 실측의 값과 달라짐: 공정 변화로 model 의 실제 R² 가 $\hat{R}^2_c$ 보다 낮아지면, 식 (9) 의 $m^{\ast}$ 에서 실제 유지 확률이 $\gamma$ 보다 낮다. $\hat{R}^2_c$ 는 최근 실측으로 다시 추정한다.
- 예측에 bias 가 있음: 식 (1) 의 R² 가 상관계수 제곱보다 작아져 식 (7) 이 맞지 않는다.
- $m$ 이 작음: 식 (6) 과 식 (7) 의 정규 근사가 맞지 않는다. $m_{\min}$ 은 이 경우를 막는 하한이다.

---

## Appendix A. Terminology

- **bias**: 예측 두께의 평균이 실측 두께의 평균에서 벗어난 양.
- **delta method**: 추정량의 함수가 갖는 분산을 그 함수의 미분과 추정량의 분산으로 근사하는 방법.
- **R² (coefficient of determination)**: 1 에서 SSE 를 SST 로 나눈 값을 뺀 정확도 지표.
- **SSE (sum of squared errors)**: 실측 두께와 예측 두께의 차이를 제곱해 더한 값.
- **SST (total sum of squares)**: 실측 두께와 그 평균의 차이를 제곱해 더한 값.
- **대표본 근사 (large-sample approximation)**: 표본 수가 충분히 클 때 추정량의 분포를 정규분포로 보는 근사.
- **분위수 (quantile)**: 누적확률이 주어진 값이 되는 확률변수의 값.
- **유한 모집단 보정 (finite population correction, FPC)**: 전체 $N$ 개에서 비복원으로 $m$ 개를 뽑을 때 분산에 곱하는 $(N-m)/(N-1)$.

## Appendix B. Derivation

5.1 의 조건에서 참값 $R^2$ 은 상관계수 제곱 $\rho^2$ 과 같고 model 의 R² 예측값 $\hat{R}^2_c$ 와도 같다. 실측 wafer $m$ 장으로 추정한 상관계수 $\hat{\rho}$ 의 대표본 분산은 $(1 - \rho^2)^2 / m$ 이다. 실측 R² $R^2_m$ 은 상관계수 추정값의 제곱 $\hat{\rho}^2$ 과 같으므로, 이 사실들로 $R^2_m$ 의 분산을 구한다.

```math
\begin{aligned}
\mathrm{Var}(R^2_m)
&\approx (2\rho)^2\, \mathrm{Var}(\hat{\rho}) \cdot \frac{N - m}{N - 1} \\
&\approx 4 \rho^2\, \frac{(1 - \rho^2)^2}{m} \cdot \frac{N - m}{N - 1} \\
&= \frac{4 \hat{R}^2_c (1 - \hat{R}^2_c)^2}{m} \cdot \frac{N - m}{N - 1}
\end{aligned}
\hspace{19em} (13)
```

- 첫째 근사: delta method 이다. $\rho^2$ 을 $\rho$ 로 미분하면 $2\rho$ 이므로 $\hat{\rho}^2$ 의 분산은 $\hat{\rho}$ 의 분산에 $(2\rho)^2$ 을 곱한 값이다. 전체 $N$ 장에서 비복원으로 $m$ 장을 뽑으므로 유한 모집단 보정 $(N - m)/(N - 1)$ 을 곱한다.
- 둘째 근사: $\hat{\rho}$ 의 대표본 분산 $(1 - \rho^2)^2 / m$ 을 넣는다.
- 셋째 등호: $\rho^2 = R^2 = \hat{R}^2_c$ 를 넣는다.

따라서 식 (7) 이 성립한다.

```math
\mathrm{Var}(R^2_m) \approx \frac{4 \hat{R}^2_c (1 - \hat{R}^2_c)^2}{m} \cdot \frac{N - m}{N - 1}
\hspace{19em} (7)
```

이어서 5.1 의 조건에서 $R^2_m$ 이 평균 $\hat{R}^2_c$, 분산이 식 (7) 인 정규분포를 따른다고 보고, 식 (3) 의 유지 확률을 구한다.

```math
\begin{aligned}
P_{\mathrm{ret}}(m)
&= P\left( \frac{R^2_m - \hat{R}^2_c}{\sqrt{\mathrm{Var}(R^2_m)}} \ge \frac{R^2_{\min} - \hat{R}^2_c}{\sqrt{\mathrm{Var}(R^2_m)}} \right) \\
&\approx 1 - \Phi\left( \frac{R^2_{\min} - \hat{R}^2_c}{\sqrt{\mathrm{Var}(R^2_m)}} \right) \\
&= \Phi\left( \frac{\hat{R}^2_c - R^2_{\min}}{\sqrt{\mathrm{Var}(R^2_m)}} \right)
\end{aligned}
\hspace{19em} (14)
```

- 첫째 등호: 식 (3) 의 부등식 양변에서 $\hat{R}^2_c$ 를 빼고 양수인 $\sqrt{\mathrm{Var}(R^2_m)}$ 로 나눈다. 부등호의 방향은 바뀌지 않는다.
- 둘째 근사: 왼쪽의 표준화한 값이 표준정규분포를 따르므로, 그 값이 어떤 수 이상일 확률은 1 에서 그 수까지의 누적확률을 뺀 값이다.
- 셋째 등호: 표준정규분포는 0 에 대해 대칭이어서, 임의의 실수 $z$ 에 대해 $1 - \Phi(-z) = \Phi(z)$ 이다.

따라서 식 (6) 이 성립한다.

```math
P_{\mathrm{ret}}(m) \approx \Phi\left( \frac{\hat{R}^2_c - R^2_{\min}}{\sqrt{\mathrm{Var}(R^2_m)}} \right)
\hspace{19em} (6)
```

마지막으로 $P_{\mathrm{ret}}(m) \ge \gamma$ 를 $m$ 에 대해 푼다.

```math
\begin{aligned}
P_{\mathrm{ret}}(m) \ge \gamma
&\iff \frac{\hat{R}^2_c - R^2_{\min}}{\sqrt{\mathrm{Var}(R^2_m)}} \ge z_{\gamma} \\
&\iff z_{\gamma}^2\, \mathrm{Var}(R^2_m) \le \Delta^2 \\
&\iff m_0\, \frac{N - m}{N - 1} \le m \\
&\iff m \ge \frac{m_0}{1 + (m_0 - 1)/N}
\end{aligned}
\hspace{19em} (15)
```

- 첫째 동치: 식 (6) 을 넣고, $\Phi$ 가 증가함수이므로 양변에 $\Phi^{-1}$ 을 적용한다. $z_{\gamma} = \Phi^{-1}(\gamma)$ 이다.
- 둘째 동치: $\hat{R}^2_c - R^2_{\min} = \Delta$ 로 옮기면 $z_{\gamma} \sqrt{\mathrm{Var}(R^2_m)} \le \Delta$ 이고, 양변이 0 이상이므로 제곱해도 부등호가 유지된다.
- 셋째 동치: 식 (7) 을 넣고, $m_0 = 4 z_{\gamma}^2 \hat{R}^2_c (1 - \hat{R}^2_c)^2 / \Delta^2$ 로 묶은 뒤 양변에 $m$ 을 곱한다.
- 넷째 동치: 양변에 $N - 1$ 을 곱해 정리하면 $m (N - 1 + m_0) \ge m_0 N$ 이고, 양변을 $N - 1 + m_0$ 로 나눈 뒤 분자와 분모를 $N$ 으로 나눈다.

마지막 줄의 $m$ 을 정수로 올림하고 $m_{\min}$ 과 견주어 큰 쪽을 고르면 $m^{\ast}$ 이다. 따라서 식 (9) 가 성립한다.

```math
m^{\ast} = \max\left( \left\lceil \frac{m_0}{1 + (m_0 - 1)/N} \right\rceil,\ m_{\min} \right)
\hspace{19em} (9)
```
