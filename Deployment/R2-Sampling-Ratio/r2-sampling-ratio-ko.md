# Measured Wafer Ratio for R2 Retention
Rev. 0 | Created: 2026-11-05 | Updated: 2026-11-05 05:32 CST

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Notation](#3-notation)
- [4. Interpretation](#4-interpretation)
  - [4.1 Measured R2](#41-measured-r2)
  - [4.2 Variance of R2](#42-variance-of-r2)
  - [4.3 Retention Condition](#43-retention-condition)
  - [4.4 Minimum Measured Count](#44-minimum-measured-count)
- [5. Worked Example](#5-worked-example)
  - [5.1 One Condition](#51-one-condition)
  - [5.2 Condition Comparison](#52-condition-comparison)
- [6. Application](#6-application)
  - [6.1 Calculation Script](#61-calculation-script)
  - [6.2 Breaking Conditions](#62-breaking-conditions)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Derivation](#appendix-b-derivation)

## 1. Purpose

- **Problem Statement**: 증착 공정의 두께를 model 이 예측하고 일부 wafer 만 실측하면, 실측 wafer 로 계산한 R² 가 실측 wafer 가 늘 때마다 변동한다.
- **Goal**: 실측 R² 가 최저 R² 이상으로 유지될 확률을 정한 값 이상으로 두는 가장 작은 실측 wafer 비율을 model 마다 계산한다.
- **Non-Goal**: 두께 예측 model 을 만들거나 고르는 방법은 다루지 않는다.

## 2. Summary

전체 wafer $N$ 장 가운데 실측할 wafer 수 $n^{\ast}$ 는 model 의 기대 R² 인 $R^2_m$, 유지할 최저 R² 인 $R^2_{\min}$, 유지 확률 $1-\alpha$ 로 정해지며, 실측 비율은 $r^{\ast} = n^{\ast}/N$ 이다.
$R^2_m$ 이 높을수록, $R^2_m - R^2_{\min}$ 이 클수록, $1-\alpha$ 가 낮을수록, $N$ 이 클수록 $r^{\ast}$ 가 줄어든다.

## 3. Notation

Table 1. Symbols

|       Symbol        |                    Meaning                     |   Shape    |              Assumption               |
| :-----------------: | :--------------------------------------------: | :--------: | :-----------------------------------: |
|         $N$         |      한 lot 또는 한 기간의 전체 wafer 수        |  integer   |              $N \ge 2$                |
|         $n$         |   실측 wafer 수. 예측 wafer 수는 $N-n$         |  integer   |           $1 \le n \le N$             |
|        $y_i$        |         $i$ 번째 실측 wafer 의 실측 두께        |   scalar   |                  -                    |
|     $\hat{y}_i$     |    $i$ 번째 실측 wafer 에 대한 model 예측 두께   |   scalar   |                  -                    |
|      $\bar{y}$      |           실측 두께 $n$ 개의 평균               |   scalar   |                  -                    |
|    $\hat{R}^2_n$    |       실측 wafer $n$ 장으로 계산한 R²            |   scalar   |                  -                    |
|      $R^2_m$        |    model $m$ 의 기대 R². 과거 실측으로 추정      |   scalar   |          $0 \lt R^2_m \lt 1$          |
|    $R^2_{\min}$     |              유지해야 할 최저 R²                 |   scalar   |       $R^2_{\min} \lt R^2_m$          |
|      $\Delta$       |      허용 여유 $R^2_m - R^2_{\min}$              |   scalar   |            $\Delta \gt 0$             |
|     $1-\alpha$      |        $\hat{R}^2_n \ge R^2_{\min}$ 의 유지 확률  |   scalar   |       $0.5 \lt 1-\alpha \lt 1$        |
|   $z_{1-\alpha}$    |       표준정규분포의 $1-\alpha$ 분위수           |   scalar   |                  -                    |
|       $n_0$         |   유한 모집단 보정 전의 필요 실측 수             |   scalar   |                  -                    |
|   $n_{\min}$        |   대표본 근사를 위한 실측 수 하한                |  integer   |   $2 \le n_{\min} \le N$. 기본값 10    |
|     $n^{\ast}$      |               최소 실측 wafer 수                 |  integer   |                  -                    |
|     $r^{\ast}$      |                  최소 실측 비율                  |   scalar   |                  -                    |

- $\hat{y}$: 영어로 "y hat", 한국어로 "와이 햇" 이라 읽는다.
- $\bar{y}$: 영어로 "y bar", 한국어로 "와이 바" 라 읽는다.
- $\hat{R}^2$: 영어로 "R hat squared", 한국어로 "알 햇 제곱" 이라 읽는다.
- $\sum$: 영어로 "sum", 한국어로 "합" 이라 읽는다.
- $\sqrt{x}$: 영어로 "square root of x", 한국어로 "루트 x" 라 읽는다.
- $\lceil x \rceil$: 영어로 "ceiling of x", 한국어로 "x 의 올림" 이라 읽는다.
- $n^{\ast}$: 영어로 "n star", 한국어로 "엔 스타" 라 읽는다.

## 4. Interpretation

최소 실측 wafer 수와 실측 비율은 식 (1) 과 식 (2) 이다.

```math
n^{\ast} = \max\left( \left\lceil \frac{n_0}{1 + (n_0 - 1)/N} \right\rceil,\ n_{\min} \right),
\qquad
n_0 = \left( \frac{2\, z_{1-\alpha} \sqrt{R^2_m}\, (1 - R^2_m)}{R^2_m - R^2_{\min}} \right)^2
\hspace{19em} (1)
```

```math
r^{\ast} = \frac{n^{\ast}}{N}
\hspace{19em} (2)
```

식 (1) 은 식 (3) 의 R² 가 식 (4) 의 분산으로 변동할 때 식 (5) 의 조건을 만족하는 가장 작은 $n$ 이며, 유도는 [Appendix B](#appendix-b-derivation) 에 있다.

### 4.1 Measured R2

실측 wafer $n$ 장의 실측 두께와 예측 두께로 계산한 R² 는 식 (3) 이다.

```math
\hat{R}^2_n = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}
\hspace{19em} (3)
```

- 분자: 예측 오차의 제곱합 (SSE)
- 분모: 실측 두께가 평균에서 벗어난 제곱합 (SST)
- 실측 wafer 가 바뀌면 분자와 분모가 함께 바뀌므로, model 이 같아도 $\hat{R}^2_n$ 은 $R^2_m$ 주위에서 변동한다.

### 4.2 Variance of R2

$\hat{R}^2_n$ 의 분산은 식 (4) 이다. 유도는 [Appendix B](#appendix-b-derivation) 에 있다.

```math
\mathrm{Var}(\hat{R}^2_n) \approx \frac{4 R^2_m (1 - R^2_m)^2}{n} \cdot \frac{N - n}{N - 1}
\hspace{19em} (4)
```

- 앞 인자: 대표본 근사 (large-sample approximation) 에서 delta method 로 구한 R² 의 분산
- 뒤 인자: 유한 모집단 보정 (finite population correction, FPC). $n = N$ 이면 0 이 되어, 전체를 실측하면 변동이 없다.
- $R^2_m$ 이 1 에 가까울수록 $(1 - R^2_m)^2$ 이 작아져 분산이 줄어든다.

식 (4) 는 아래 조건에서 성립한다.

- 예측 두께에 bias 가 없고 실측 두께에 대한 기울기가 1 이어서, 식 (3) 의 R² 가 실측과 예측의 상관계수 제곱과 같다.
- $n$ 이 충분히 커서 $\hat{R}^2_n$ 이 정규분포에 가깝다. 이 문서는 $n \ge n_{\min} = 10$ 으로 둔다.
- 실측 wafer 를 전체 wafer 에서 무작위로 고른다.

### 4.3 Retention Condition

$\hat{R}^2_n$ 이 $R^2_{\min}$ 이상일 확률을 $1-\alpha$ 이상으로 유지하는 조건은 식 (5) 이다.

```math
P\left( \hat{R}^2_n \ge R^2_{\min} \right) \ge 1 - \alpha
\iff
R^2_m - z_{1-\alpha} \sqrt{\mathrm{Var}(\hat{R}^2_n)} \ge R^2_{\min}
\hspace{19em} (5)
```

- 좌변: 실측 R² 가 최저 R² 이상일 확률
- 우변: $\hat{R}^2_n$ 분포의 아래쪽 $\alpha$ 분위수가 $R^2_{\min}$ 이상이라는 조건

### 4.4 Minimum Measured Count

식 (5) 에 식 (4) 를 넣고 $n$ 에 대해 풀면 식 (1) 이 된다.

- $n_0$: 전체 wafer 수가 무한할 때의 필요 실측 수
- $n_0 / (1 + (n_0 - 1)/N)$: 유한 모집단 보정을 넣은 필요 실측 수. $N$ 이 작을수록 $n_0$ 보다 많이 줄어든다.
- $n_{\min}$: 4.2 의 대표본 근사가 성립하도록 둔 하한

식 (1) 은 $R^2_m \gt R^2_{\min}$ 일 때만 성립한다. $R^2_m \le R^2_{\min}$ 이면 $n$ 을 $N$ 까지 늘려도 식 (5) 를 만족하지 못한다.

## 5. Worked Example

### 5.1 One Condition

$N = 100$, $R^2_m = 0.90$, $R^2_{\min} = 0.85$, $1-\alpha = 0.95$ 로 식 (1) 과 식 (2) 를 계산한다.

1. $z_{0.95} = 1.645$
2. $n_0 = (2 \times 1.645 \times \sqrt{0.90} \times 0.10 / 0.05)^2 = (0.3121 / 0.05)^2 = 38.96$
3. $n_0 / (1 + (n_0 - 1)/N) = 38.96 / 1.3796 = 28.24$, 올림하면 29
4. $n^{\ast} = \max(29,\ 10) = 29$, $r^{\ast} = 29/100 = 0.29$

식 (5) 의 좌변을 $n = 28$ 과 $n = 29$ 에서 계산하면 아래 Table 2 와 같다.

Table 2. Retention check around n = 29

|  n  | Var of R2 | Std of R2 | R2_m - z Std | Meets R2_min = 0.85 |
| :-: | :-------: | :-------: | :----------: | :-----------------: |
| 28  | 0.000935  |  0.03058  |    0.8497    |         No          |
| 29  | 0.000890  |  0.02984  |    0.8509    |         Yes         |

$n = 28$ 에서는 아래쪽 5 % 분위수가 0.8497 로 0.85 보다 작고, $n = 29$ 에서 처음으로 0.85 이상이 된다.

### 5.2 Condition Comparison

조건별 $n^{\ast}$ 와 $r^{\ast}$ 는 아래 Table 3, Table 4, Table 5 와 같다. Cell 은 `n* (r*)` 이고, * 는 $n_{\min} = 10$ 이 $n^{\ast}$ 를 정한 경우이다.

Table 3. Measured count by model R2 and margin, N = 100, 1 - alpha = 95 %

| R2_m |  Margin = 0.03  |  Margin = 0.05  |  Margin = 0.10  |
| :--: | :-------------: | :-------------: | :-------------: |
| 0.80 |   80 (0.80)     |   59 (0.59)     |   26 (0.26)     |
| 0.85 |   70 (0.70)     |   46 (0.46)     |   18 (0.18)     |
| 0.90 |   53 (0.53)     |   29 (0.29)     |   10\* (0.10)   |
| 0.95 |   23 (0.23)     |   10 (0.10)     |   10\* (0.10)   |

$\Delta$ 를 0.03 에서 0.10 으로 넓히면 $R^2_m = 0.80$ 에서 $r^{\ast}$ 가 0.80 에서 0.26 으로 줄어든다.

Table 4. Measured count by retention probability, N = 100, margin = 0.05

| R2_m |     90 %      |     95 %      |     99 %      |
| :--: | :-----------: | :-----------: | :-----------: |
| 0.80 |  46 (0.46)    |  59 (0.59)    |  74 (0.74)    |
| 0.85 |  34 (0.34)    |  46 (0.46)    |  63 (0.63)    |
| 0.90 |  20 (0.20)    |  29 (0.29)    |  45 (0.45)    |
| 0.95 | 10\* (0.10)   |  10 (0.10)    |  18 (0.18)    |

유지 확률을 95 % 에서 99 % 로 올리면 $R^2_m = 0.90$ 에서 $r^{\ast}$ 가 0.29 에서 0.45 로 늘어난다.

Table 5. Measured count by total wafers, 1 - alpha = 95 %, margin = 0.05

| R2_m |    N = 25     |    N = 100    |
| :--: | :-----------: | :-----------: |
| 0.80 |  22 (0.88)    |  59 (0.59)    |
| 0.85 |  20 (0.80)    |  46 (0.46)    |
| 0.90 |  16 (0.64)    |  29 (0.29)    |
| 0.95 | 10\* (0.40)   |  10 (0.10)    |

$N$ 이 25 이면 $n^{\ast}$ 는 줄지만 $r^{\ast}$ 는 커진다. $R^2_m = 0.90$ 에서 $N = 25$ 의 $r^{\ast}$ 는 0.64 로 $N = 100$ 의 0.29 보다 두 배 넘게 크다.

## 6. Application

### 6.1 Calculation Script

식 (1) 부터 식 (4) 를 계산하는 Python script 는 [src/r2_sampling_ratio.py](src/r2_sampling_ratio.py) 이다.

- `estimate_r2(measured=..., predicted=...)`: 식 (3) 의 $\hat{R}^2_n$
- `r2_variance(r2_model=..., measured=..., total_wafers=...)`: 식 (4) 의 분산
- `required_measured(r2_model=..., r2_min=..., confidence=..., total_wafers=..., min_measured=...)`: 식 (1) 과 식 (2) 의 $n^{\ast}$, $r^{\ast}$
- `build_table(...)`: 여러 조건의 $n^{\ast}$, $r^{\ast}$ 를 한 표로 계산

허용 여유는 `--margin` 으로 $\Delta$ 를 주거나 `--r2-min` 으로 $R^2_{\min}$ 을 직접 주며, 둘을 함께 주면 오류로 끝난다. Table 3 은 아래 명령으로 계산한다.

```bash
python src/r2_sampling_ratio.py --r2-model 0.80 0.85 0.90 0.95 --margin 0.03 0.05 0.10 --total-wafers 100 --confidence 0.95
```

`--output-folder <FOLDER>` 를 주면 결과를 `<FOLDER>/r2_sampling_ratio.csv` 로 저장한다. 열은 `total_wafers`, `r2_model`, `r2_min`, `confidence`, `n0`, `measured`, `floored`, `ratio` 이다.

### 6.2 Breaking Conditions

- $R^2_m \le R^2_{\min}$: 식 (1) 의 분모가 0 이하가 되어 $n^{\ast}$ 가 없다. `required_measured` 는 오류로 끝나고, `build_table` 은 이 조건을 건너뛰고 건너뛴 개수를 출력한다.
- $R^2_m$ 이 과거 실측의 값과 달라짐: 공정 변화로 model 의 실제 R² 가 $R^2_m$ 보다 낮아지면, 식 (1) 의 $n^{\ast}$ 로는 유지 확률이 $1-\alpha$ 보다 낮다. $R^2_m$ 은 최근 실측으로 다시 추정한다.
- 예측에 bias 가 있음: 식 (3) 의 R² 가 상관계수 제곱보다 작아져 식 (4) 가 맞지 않는다.
- $n$ 이 작음: 식 (4) 의 정규 근사가 맞지 않는다. $n_{\min}$ 은 이 경우를 막는 하한이다.

---

## Appendix A. Terminology

- **bias**: 예측 두께의 평균이 실측 두께의 평균에서 벗어난 양.
- **delta method**: 추정량의 함수가 갖는 분산을 그 함수의 미분과 추정량의 분산으로 근사하는 방법.
- **R² (coefficient of determination)**: 1 에서 SSE 를 SST 로 나눈 값을 뺀 정확도 지표.
- **SSE (sum of squared errors)**: 실측 두께와 예측 두께의 차이를 제곱해 더한 값.
- **SST (total sum of squares)**: 실측 두께와 그 평균의 차이를 제곱해 더한 값.
- **대표본 근사 (large-sample approximation)**: 표본 수가 충분히 클 때 추정량의 분포를 정규분포로 보는 근사.
- **분위수 (quantile)**: 누적확률이 주어진 값이 되는 확률변수의 값.
- **유한 모집단 보정 (finite population correction, FPC)**: 전체 $N$ 개에서 비복원으로 $n$ 개를 뽑을 때 분산에 곱하는 $(N-n)/(N-1)$.

## Appendix B. Derivation

실측과 예측의 상관계수를 $\rho$ 라 하면 4.2 의 조건에서 $R^2_m = \rho^2$ 이다.

```math
\begin{aligned}
\mathrm{Var}(\hat{\rho}) &\approx \frac{(1 - \rho^2)^2}{n} && \text{large-sample variance of the correlation} \\
\mathrm{Var}(\hat{\rho}^2) &\approx (2\rho)^2\, \mathrm{Var}(\hat{\rho}) = \frac{4 \rho^2 (1 - \rho^2)^2}{n} && \text{delta method with } d(\rho^2)/d\rho = 2\rho \\
\mathrm{Var}(\hat{R}^2_n) &\approx \frac{4 R^2_m (1 - R^2_m)^2}{n} \cdot \frac{N - n}{N - 1} && \text{finite population correction, Eq. (4)} \\
z_{1-\alpha}^2\, \mathrm{Var}(\hat{R}^2_n) &\le \Delta^2 && \text{square both sides of Eq. (5)} \\
n_0\, \frac{N - n}{N - 1} &\le n && n_0 = 4 z_{1-\alpha}^2 R^2_m (1 - R^2_m)^2 / \Delta^2 \\
n &\ge \frac{n_0 N}{N - 1 + n_0} = \frac{n_0}{1 + (n_0 - 1)/N} && \text{solve for } n,\ \text{Eq. (1)}
\end{aligned}
```

마지막 줄의 $n$ 을 정수로 올림하고 $n_{\min}$ 과 견주어 큰 쪽을 고르면 식 (1) 의 $n^{\ast}$ 가 된다.
