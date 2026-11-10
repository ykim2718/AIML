# R2 by Aggregation Level
Version 2 | Created: 2026-11-09 | Updated: 2026-11-10 05:32 CST

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Notation](#3-notation)
- [4. Definitions](#4-definitions)
- [5. Interpretation](#5-interpretation)
  - [5.1 Total Form](#51-total-form)
  - [5.2 Wafer Average Form](#52-wafer-average-form)
  - [5.3 Location Average Form](#53-location-average-form)
  - [5.4 Relation without the Pooled R2](#54-relation-without-the-pooled-r2)
- [6. Worked Example](#6-worked-example)
- [7. Application](#7-application)
  - [7.1 Reporting R2 of a Wafer Model](#71-reporting-r2-of-a-wafer-model)
  - [7.2 Breaking Conditions](#72-breaking-conditions)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Derivation](#appendix-b-derivation)
  - [B.1 Total Form](#b1-total-form)
  - [B.2 Wafer Average Form](#b2-wafer-average-form)
  - [B.3 Location Average Form](#b3-location-average-form)
  - [B.4 Relation without the Pooled R2](#b4-relation-without-the-pooled-r2)

## 1. Purpose

- **Problem Statement**: wafer $`M`$ 장에서 wafer 마다 측정 위치 (measurement location) $`N`$ 곳을 측정하면 측정값이 $`T = MN`$ 개 나온다. 같은 예측으로 R² 를 측정값 $`T`$ 개, wafer 평균 $`M`$ 개, 측정 위치마다의 wafer $`M`$ 장에서 구하면 세 R² 의 값이 서로 다르지만, 세 값 사이의 관계식은 없다.
- **Goal**: 전체 $`R^2_{\mathrm{all}}`$, wafer 평균 $`R^2_{\mathrm{avgw}}`$, 측정 위치 평균 $`R^2_{\mathrm{avgn}}`$ 사이의 항등식을 구하고 증명한다.
- **Non-Goal**: 예측 model 을 만들거나 고르는 방법, R² 의 표본 분포는 다루지 않는다.

## 2. Summary

세 R² 는 측정 위치마다의 오차를 합쳐 구한 pooled $`R^2_{\mathrm{pool}}`$ 하나로 이어지며, $`1 - R^2_{\mathrm{all}} = (1 - \lambda)(1 - R^2_{\mathrm{pool}})`$, $`1 - R^2_{\mathrm{avgw}} = \kappa (1 - R^2_{\mathrm{pool}})`$ 이다.
$`R^2_{\mathrm{pool}}`$ 은 측정 위치마다의 $`R^2_j`$ 를 $`\mathrm{SST}_j`$ 로 가중 평균한 값이고, $`\mathrm{SST}_j`$ 가 모두 같으면 $`R^2_{\mathrm{avgn}}`$ 과 같다.
그래서 $`R^2_{\mathrm{all}} \ge R^2_{\mathrm{pool}}`$ 은 언제나 성립하고, $`R^2_{\mathrm{avgw}} \ge R^2_{\mathrm{pool}}`$ 은 측정 위치 사이 오차 상관 $`\bar{r}_e`$ 가 측정값 상관 $`\bar{r}_y`$ 이하일 때 성립한다.

## 3. Notation

Table 1. Symbols

| Symbol                             | Meaning                                                                              | Shape   | Assumption                             |
| :--------------------------------: | :----------------------------------------------------------------------------------: | :-----: | :------------------------------------: |
| $`M`$                              | wafer 수                                                                             | integer | $`M \ge 2`$                            |
| $`N`$                              | wafer 한 장의 측정 위치 수                                                           | integer | $`N \ge 2`$, 모든 wafer 에서 같음      |
| $`T`$                              | 측정값의 개수 $`MN`$                                                                 | integer | -                                      |
| $`i`$                              | wafer 번호                                                                           | integer | $`1 \le i \le M`$                      |
| $`j`$, $`k`$                       | 측정 위치 번호                                                                       | integer | $`1 \le j, k \le N`$                   |
| $`y_{ij}`$                         | wafer $`i`$ 의 측정 위치 $`j`$ 에서 측정값                                           | scalar  | -                                      |
| $`\hat{y}_{ij}`$                   | wafer $`i`$ 의 측정 위치 $`j`$ 에 대한 예측값                                        | scalar  | -                                      |
| $`e_{ij}`$                         | 예측 오차 $`y_{ij} - \hat{y}_{ij}`$                                                  | scalar  | -                                      |
| $`\bar{y}`$                        | 측정값 $`T`$ 개의 평균                                                               | scalar  | -                                      |
| $`\bar{y}_{i\cdot}`$               | wafer $`i`$ 의 wafer 평균                                                            | scalar  | -                                      |
| $`\bar{\hat{y}}_{i\cdot}`$         | wafer $`i`$ 의 예측값 $`N`$ 개의 평균                                                | scalar  | -                                      |
| $`\bar{e}_{i\cdot}`$               | wafer $`i`$ 의 wafer 평균 예측 오차                                                  | scalar  | -                                      |
| $`\bar{y}_{\cdot j}`$              | 측정 위치 $`j`$ 에서 wafer $`M`$ 장의 측정값 평균                                    | scalar  | -                                      |
| $`\mathrm{SST}`$, $`\mathrm{SSE}`$ | 평균에서 벗어난 제곱합 (total sum of squares) 과 오차 제곱합 (sum of squared errors) | scalar  | -                                      |
| $`R^2_{\mathrm{all}}`$             | 측정값 $`T`$ 개로 구한 전체 R²                                                       | scalar  | $`\mathrm{SST}_{\mathrm{all}} \gt 0`$  |
| $`R^2_{\mathrm{avgw}}`$            | wafer 평균 $`M`$ 개로 구한 wafer 평균 R²                                             | scalar  | $`\mathrm{SST}_{\mathrm{avgw}} \gt 0`$ |
| $`R^2_j`$                          | 측정 위치 $`j`$ 에서 wafer $`M`$ 장으로 구한 측정 위치 R²                            | scalar  | $`\mathrm{SST}_j \gt 0`$               |
| $`R^2_{\mathrm{avgn}}`$            | $`R^2_j`$ 의 $`N`$ 개 산술 평균                                                      | scalar  | -                                      |
| $`R^2_{\mathrm{pool}}`$            | 측정 위치 $`N`$ 곳의 SSE 와 SST 를 각각 더해 구한 pooled R²                          | scalar  | -                                      |
| $`\lambda`$                        | 측정 위치 평균의 차이가 $`\mathrm{SST}_{\mathrm{all}}`$ 에서 차지하는 비율           | scalar  | $`0 \le \lambda \lt 1`$                |
| $`\bar{r}_y`$                      | 측정 위치 사이 측정값 상관                                                           | scalar  | $`-1/(N-1) \lt \bar{r}_y \le 1`$       |
| $`\bar{r}_e`$                      | 측정 위치 사이 예측 오차 상관                                                        | scalar  | $`-1/(N-1) \lt \bar{r}_e \le 1`$       |
| $`\kappa`$                         | 상관 비 (correlation ratio)                                                          | scalar  | $`\kappa \gt 0`$                       |

SST, SSE, R² 의 아래첨자 $`\mathrm{all}`$ 은 측정값 $`T`$ 개, $`\mathrm{avgw}`$ 는 wafer 평균, $`j`$ 는 측정 위치 $`j`$, $`\mathrm{avgn}`$ 은 측정 위치 $`N`$ 곳의 평균, $`\mathrm{pool}`$ 은 측정 위치 $`N`$ 곳을 합친 것을 뜻한다. $`\bar{r}`$ 의 아래첨자 $`y`$ 와 $`e`$ 는 측정값과 예측 오차를 뜻한다. 이 문서의 SST, SSE, R², $`\lambda`$, $`\bar{r}`$, $`\kappa`$ 는 주어진 측정값과 예측값으로 정의하는 양이며 모수를 추정한 값이 아니므로 hat 을 붙이지 않는다. Hat 을 붙인 $`\hat{y}_{ij}`$ 와 $`\bar{\hat{y}}_{i\cdot}`$ 는 예측값이다.

- $`\hat{y}_{ij}`$: 영어로 "y hat i j", 한국어로 "와이 햇 아이 제이" 라 읽는다.
- $`\bar{y}`$: 영어로 "y bar", 한국어로 "와이 바" 라 읽는다.
- $`\bar{y}_{i\cdot}`$: 영어로 "y bar i dot", 한국어로 "와이 바 아이 닷" 이라 읽는다.
- $`\bar{y}_{\cdot j}`$: 영어로 "y bar dot j", 한국어로 "와이 바 닷 제이" 라 읽는다.
- $`\bar{\hat{y}}_{i\cdot}`$: 영어로 "y hat bar i dot", 한국어로 "와이 햇 바 아이 닷" 이라 읽는다.
- $`\sum`$: 영어로 "sum", 한국어로 "합" 이라 읽는다.
- $`\sum_{j \ne k}`$: 영어로 "sum over j not equal to k", 한국어로 "제이와 케이가 다른 모든 순서쌍에 대한 합" 이라 읽는다.

## 4. Definitions

> **Definition: Prediction error**<br>
> 예측 오차 $`e_{ij}`$ 는 wafer $`i`$ 의 측정 위치 $`j`$ 에 대해, 측정값 $`y_{ij}`$ 에서 예측값 $`\hat{y}_{ij}`$ 를 뺀 값이다.

> **Definition: Wafer average**<br>
> Wafer 평균 $`\bar{y}_{i\cdot}`$ 는 wafer $`i`$ 의 측정값 $`N`$ 개의 산술 평균이다. Wafer 평균의 예측값 $`\bar{\hat{y}}_{i\cdot}`$ 는 같은 wafer 의 예측값 $`N`$ 개의 산술 평균이고, wafer 평균 예측 오차 $`\bar{e}_{i\cdot}`$ 는 $`\bar{y}_{i\cdot}`$ 에서 $`\bar{\hat{y}}_{i\cdot}`$ 를 뺀 값이다.

> **Definition: Location mean**<br>
> 측정 위치 평균 $`\bar{y}_{\cdot j}`$ 는 측정 위치 $`j`$ 에서 wafer $`M`$ 장의 측정값의 산술 평균이다.

정의 식은 아래 식 (1) 부터 식 (6) 이며, $`\bar{y}`$ 는 측정값 $`T`$ 개의 평균이다.

```math
e_{ij} = y_{ij} - \hat{y}_{ij}
\hspace{19em} (1)
```

```math
\bar{y} = \frac{1}{MN} \sum_{i=1}^{M} \sum_{j=1}^{N} y_{ij}
\hspace{19em} (2)
```

```math
\bar{y}_{i\cdot} = \frac{1}{N} \sum_{j=1}^{N} y_{ij}
\hspace{19em} (3)
```

```math
\bar{\hat{y}}_{i\cdot} = \frac{1}{N} \sum_{j=1}^{N} \hat{y}_{ij}
\hspace{19em} (4)
```

```math
\bar{e}_{i\cdot} = \bar{y}_{i\cdot} - \bar{\hat{y}}_{i\cdot}
\hspace{19em} (5)
```

```math
\bar{y}_{\cdot j} = \frac{1}{M} \sum_{i=1}^{M} y_{ij}
\hspace{19em} (6)
```

> **Definition: Total R2**<br>
> 전체 $`R^2_{\mathrm{all}}`$ 은 wafer $`M`$ 장, 측정 위치 $`N`$ 곳의 측정값 $`T = MN`$ 개에서, 1 에서 예측 오차 제곱합 $`\mathrm{SSE}_{\mathrm{all}}`$ 을 측정값과 전체 평균의 차이의 제곱합 $`\mathrm{SST}_{\mathrm{all}}`$ 로 나눈 값을 뺀 값이다.

> **Definition: Wafer average R2**<br>
> Wafer 평균 $`R^2_{\mathrm{avgw}}`$ 는 wafer 평균 $`M`$ 개에서, 1 에서 wafer 평균 예측 오차의 제곱합 $`\mathrm{SSE}_{\mathrm{avgw}}`$ 를 wafer 평균과 전체 평균의 차이의 제곱합 $`\mathrm{SST}_{\mathrm{avgw}}`$ 로 나눈 값을 뺀 값이다. Wafer 평균의 예측값은 같은 wafer 의 예측값 $`N`$ 개의 산술 평균이다.

> **Definition: Location R2**<br>
> 측정 위치 $`R^2_j`$ 는 측정 위치 $`j`$ 의 wafer $`M`$ 장에서, 1 에서 예측 오차 제곱합 $`\mathrm{SSE}_j`$ 를 측정값과 측정 위치 평균의 차이의 제곱합 $`\mathrm{SST}_j`$ 로 나눈 값을 뺀 값이다.

> **Definition: Location average R2**<br>
> 측정 위치 평균 $`R^2_{\mathrm{avgn}}`$ 은 측정 위치 $`N`$ 곳의 측정 위치 $`R^2_j`$ 를 산술 평균한 값이다.

네 R² 의 정의 식은 아래 식 (7) 부터 식 (16) 이다.

```math
\mathrm{SST}_{\mathrm{all}} = \sum_{i=1}^{M} \sum_{j=1}^{N} \left( y_{ij} - \bar{y} \right)^2
\hspace{19em} (7)
```

```math
\mathrm{SSE}_{\mathrm{all}} = \sum_{i=1}^{M} \sum_{j=1}^{N} e_{ij}^2
\hspace{19em} (8)
```

```math
R^2_{\mathrm{all}} = 1 - \frac{\mathrm{SSE}_{\mathrm{all}}}{\mathrm{SST}_{\mathrm{all}}}
\hspace{19em} (9)
```

```math
\mathrm{SST}_{\mathrm{avgw}} = \sum_{i=1}^{M} \left( \bar{y}_{i\cdot} - \bar{y} \right)^2
\hspace{19em} (10)
```

```math
\mathrm{SSE}_{\mathrm{avgw}} = \sum_{i=1}^{M} \bar{e}_{i\cdot}^2
\hspace{19em} (11)
```

```math
R^2_{\mathrm{avgw}} = 1 - \frac{\mathrm{SSE}_{\mathrm{avgw}}}{\mathrm{SST}_{\mathrm{avgw}}}
\hspace{19em} (12)
```

```math
\mathrm{SST}_j = \sum_{i=1}^{M} \left( y_{ij} - \bar{y}_{\cdot j} \right)^2
\hspace{19em} (13)
```

```math
\mathrm{SSE}_j = \sum_{i=1}^{M} e_{ij}^2
\hspace{19em} (14)
```

```math
R^2_j = 1 - \frac{\mathrm{SSE}_j}{\mathrm{SST}_j}
\hspace{19em} (15)
```

```math
R^2_{\mathrm{avgn}} = \frac{1}{N} \sum_{j=1}^{N} R^2_j
\hspace{19em} (16)
```

- 네 R² 는 $`1 - \mathrm{SSE}/\mathrm{SST}`$ 꼴이며, 측정값과 예측값의 상관계수를 제곱한 값과는 다르다.
- 예측 오차가 모두 0 이면 네 R² 는 모두 1 이다. 예측이 평균보다 나쁘면 R² 는 음수일 수 있다.

> **Definition: Pooled location R2**<br>
> Pooled $`R^2_{\mathrm{pool}}`$ 은 측정 위치 $`N`$ 곳의 예측 오차 제곱합 $`\mathrm{SSE}_j`$ 를 모두 더한 값을, 측정값과 측정 위치 평균의 차이의 제곱합 $`\mathrm{SST}_j`$ 를 모두 더한 값으로 나누어 1 에서 뺀 값이다.

> **Definition: Location pattern fraction**<br>
> 측정 위치 패턴 비율 $`\lambda`$ 는 측정 위치 평균 $`\bar{y}_{\cdot j}`$ 와 전체 평균 $`\bar{y}`$ 의 차이의 제곱을 측정 위치 $`N`$ 곳에서 더하고 wafer 수 $`M`$ 을 곱한 값을, 측정값과 전체 평균의 차이의 제곱합 $`\mathrm{SST}_{\mathrm{all}}`$ 로 나눈 값이다.

> **Definition: Between-location correlation**<br>
> 측정 위치 사이 측정값 상관 $`\bar{r}_y`$ 는 측정값에서 측정 위치 평균을 뺀 값을 서로 다른 두 측정 위치 $`j \ne k`$ 에서 곱해 wafer $`M`$ 장과 모든 순서쌍 $`(j, k)`$ 에서 더한 값을, $`N - 1`$ 과 $`\mathrm{SST}_j`$ 의 합을 곱한 값으로 나눈 것이다. 측정 위치 사이 예측 오차 상관 $`\bar{r}_e`$ 는 같은 계산을 예측 오차 $`e_{ij}`$ 와 $`\mathrm{SSE}_j`$ 로 한 것이다.

> **Definition: Correlation ratio**<br>
> 상관 비 $`\kappa`$ 는 $`1 + (N-1)\bar{r}_e`$ 를 $`1 + (N-1)\bar{r}_y`$ 로 나눈 값이다.

정의 식은 아래 식 (17) 부터 식 (21) 이다.

```math
R^2_{\mathrm{pool}} = 1 - \frac{\sum_{j=1}^{N} \mathrm{SSE}_j}{\sum_{j=1}^{N} \mathrm{SST}_j}
\hspace{19em} (17)
```

```math
\lambda = \frac{M \sum_{j=1}^{N} \left( \bar{y}_{\cdot j} - \bar{y} \right)^2}{\mathrm{SST}_{\mathrm{all}}}
\hspace{19em} (18)
```

```math
\bar{r}_y = \frac{\sum_{j \ne k} \sum_{i=1}^{M} \left( y_{ij} - \bar{y}_{\cdot j} \right) \left( y_{ik} - \bar{y}_{\cdot k} \right)}{(N-1) \sum_{j=1}^{N} \mathrm{SST}_j}
\hspace{19em} (19)
```

```math
\bar{r}_e = \frac{\sum_{j \ne k} \sum_{i=1}^{M} e_{ij}\, e_{ik}}{(N-1) \sum_{j=1}^{N} \mathrm{SSE}_j}
\hspace{19em} (20)
```

```math
\kappa = \frac{1 + (N-1)\, \bar{r}_e}{1 + (N-1)\, \bar{r}_y}
\hspace{19em} (21)
```

- $`\mathrm{SST}_j`$ 가 모두 같으면 $`\bar{r}_y`$ 는 측정 위치 두 곳의 측정값 상관계수를 모든 쌍에서 산술 평균한 값이다. $`\mathrm{SSE}_j`$ 가 모두 같으면 $`\bar{r}_e`$ 도 예측 오차 상관계수의 산술 평균이다.
- $`\bar{r}_y = 1`$ 은 wafer 마다 측정값이 모든 측정 위치에서 같은 양만큼 움직이는 경우이고, $`\bar{r}_e = 0`$ 은 측정 위치마다의 예측 오차가 서로 상관이 없는 경우이다.
- $`1 + (N-1)\bar{r}_y`$ 와 $`1 + (N-1)\bar{r}_e`$ 는 0 이상이다. 0 이 되는 것은 wafer 평균의 SST 또는 SSE 가 0 인 경우뿐이며, 그 증명은 [B.2](#b2-wafer-average-form) 에 있다.

## 5. Interpretation

세 R² 는 pooled $`R^2_{\mathrm{pool}}`$ 을 거쳐 아래 식 (22), 식 (23), 식 (24) 로 이어진다. 세 식은 예측값을 어떤 model 로 구했는지와 상관없이 성립하는 항등식이며, 증명은 [Appendix B](#appendix-b-derivation) 에 있다.

```math
1 - R^2_{\mathrm{all}} = (1 - \lambda) \left( 1 - R^2_{\mathrm{pool}} \right)
\hspace{19em} (22)
```

```math
1 - R^2_{\mathrm{avgw}} = \kappa \left( 1 - R^2_{\mathrm{pool}} \right)
\hspace{19em} (23)
```

```math
R^2_{\mathrm{pool}} = \sum_{j=1}^{N} \frac{\mathrm{SST}_j}{\sum_{k=1}^{N} \mathrm{SST}_k}\, R^2_j
\hspace{19em} (24)
```

세 식은 모든 wafer 가 같은 측정 위치 $`N`$ 곳을 갖는다고 두며, 식 (23) 은 여기에 더해 wafer 평균의 예측값을 식 (4) 처럼 예측값 $`N`$ 개의 산술 평균으로 구해야 성립한다.

### 5.1 Total Form

식 (22) 의 $`1 - \lambda`$ 는 $`\mathrm{SST}_{\mathrm{all}}`$ 가운데 측정 위치마다의 wafer 사이 변동 $`\sum_j \mathrm{SST}_j`$ 가 차지하는 비율이다. 식 (29) 에서 $`\mathrm{SST}_{\mathrm{all}}`$ 은 이 변동과 측정 위치 평균의 차이 $`\lambda\, \mathrm{SST}_{\mathrm{all}}`$ 로 나뉘지만, 식 (30) 에서 $`\mathrm{SSE}_{\mathrm{all}}`$ 은 $`\mathrm{SSE}_j`$ 의 합과 같아 그대로 남는다. 그래서 $`R^2_{\mathrm{all}}`$ 은 측정 위치 평균의 차이를 맞힌 몫 $`\lambda (1 - R^2_{\mathrm{pool}})`$ 만큼 $`R^2_{\mathrm{pool}}`$ 보다 크다.

- $`R^2_{\mathrm{pool}} \le 1`$ 이고 $`0 \lt 1 - \lambda \le 1`$ 이므로 $`R^2_{\mathrm{all}} \ge R^2_{\mathrm{pool}}`$ 은 언제나 성립한다.
- 두 값이 같은 경우는 측정 위치 평균이 모두 같은 $`\lambda = 0`$ 이거나 예측 오차가 모두 0 인 경우뿐이다.
- 성립 조건은 $`\sum_j \mathrm{SST}_j \gt 0`$ 이며, 이때 $`\lambda \lt 1`$ 이다.

### 5.2 Wafer Average Form

식 (23) 의 $`\kappa`$ 는 wafer 평균을 구할 때 측정값의 변동과 예측 오차가 각각 얼마나 줄어드는지의 비이다. 측정 위치 $`N`$ 곳을 평균하면 식 (35) 와 식 (38) 에서 SST 는 $`\sum_j \mathrm{SST}_j`$ 의 $`[1 + (N-1)\bar{r}_y]/N^2`$ 배가 되고, SSE 는 $`\sum_j \mathrm{SSE}_j`$ 의 $`[1 + (N-1)\bar{r}_e]/N^2`$ 배가 된다. 그래서 측정 위치 사이 상관이 예측 오차에서 측정값보다 작으면 평균하는 동안 SSE 가 SST 보다 더 줄어 $`R^2_{\mathrm{avgw}}`$ 가 커진다.

- $`R^2_{\mathrm{avgw}} \ge R^2_{\mathrm{pool}}`$ 은 $`\kappa \le 1`$, 곧 $`\bar{r}_e \le \bar{r}_y`$ 일 때 성립한다.
- Wafer 사이 변동이 모든 측정 위치에서 같은 양인 $`\bar{r}_y = 1`$ 이고 예측 오차가 측정 위치 사이에서 상관이 없는 $`\bar{r}_e = 0`$ 이면 $`\kappa = 1/N`$ 이고, $`1 - R^2_{\mathrm{avgw}}`$ 는 $`1 - R^2_{\mathrm{pool}}`$ 의 $`1/N`$ 배이다.
- 예측 오차가 모든 측정 위치에서 같은 양인 $`\bar{r}_e = 1`$ 이면 $`\kappa \ge 1`$ 이고, 평균해도 SSE 가 SST 보다 덜 줄어 $`R^2_{\mathrm{avgw}} \le R^2_{\mathrm{pool}}`$ 이다.
- 성립 조건은 $`\mathrm{SST}_{\mathrm{avgw}} \gt 0`$ 과 $`\sum_j \mathrm{SSE}_j \gt 0`$ 이다. $`\sum_j \mathrm{SSE}_j = 0`$ 이면 $`\bar{r}_e`$ 가 정의되지 않지만, 그때는 $`R^2_{\mathrm{avgw}} = R^2_{\mathrm{pool}} = 1`$ 이다.

### 5.3 Location Average Form

식 (24) 에서 $`R^2_{\mathrm{pool}}`$ 은 $`R^2_j`$ 를 $`\mathrm{SST}_j`$ 에 비례하는 가중치로 평균한 값이고, $`R^2_{\mathrm{avgn}}`$ 은 같은 $`R^2_j`$ 를 가중치 $`1/N`$ 로 평균한 값이다. 그래서 $`\mathrm{SST}_j`$ 가 모두 같으면 식 (25) 가 성립한다.

```math
R^2_{\mathrm{pool}} = R^2_{\mathrm{avgn}}
\hspace{19em} (25)
```

- $`\mathrm{SST}_j`$ 가 큰 측정 위치에서 $`R^2_j`$ 가 크면 $`R^2_{\mathrm{pool}} \gt R^2_{\mathrm{avgn}}`$ 이고, 작으면 $`R^2_{\mathrm{pool}} \lt R^2_{\mathrm{avgn}}`$ 이다.
- 성립 조건은 모든 $`j`$ 에서 $`\mathrm{SST}_j \gt 0`$ 이다.

### 5.4 Relation without the Pooled R2

식 (22) 를 식 (23) 으로 나누어 $`1 - R^2_{\mathrm{pool}}`$ 을 지우면 $`R^2_{\mathrm{all}}`$ 과 $`R^2_{\mathrm{avgw}}`$ 의 관계는 식 (26) 이다.

```math
1 - R^2_{\mathrm{all}} = \frac{1 - \lambda}{\kappa} \left( 1 - R^2_{\mathrm{avgw}} \right)
\hspace{19em} (26)
```

- $`(1 - \lambda)/\kappa \lt 1`$ 이면 $`R^2_{\mathrm{all}} \gt R^2_{\mathrm{avgw}}`$ 이고, $`(1 - \lambda)/\kappa \gt 1`$ 이면 $`R^2_{\mathrm{all}} \lt R^2_{\mathrm{avgw}}`$ 이다.
- $`\mathrm{SST}_j`$ 가 모두 같으면 식 (25) 에 따라 식 (22) 와 식 (23) 의 $`R^2_{\mathrm{pool}}`$ 을 $`R^2_{\mathrm{avgn}}`$ 으로 바꿀 수 있어, 세 R² 가 $`\lambda`$ 와 $`\kappa`$ 만으로 이어진다.
- 성립 조건은 5.1 과 5.2 의 조건에 $`\mathrm{SSE}_{\mathrm{avgw}} \gt 0`$ 을 더한 것이다. 이 조건에서 $`\kappa \gt 0`$ 이다.

## 6. Worked Example

Wafer $`M = 3`$ 장, 측정 위치 $`N = 2`$ 곳의 측정값과 예측값은 아래 Table 2 이다.

Table 2. Measured and predicted values

| Wafer i | y_i1 | y_i2 | Predicted y_i1 | Predicted y_i2 | e_i1 | e_i2 | Wafer average | Wafer average error |
| :-----: | :--: | :--: | :------------: | :------------: | :--: | :--: | :-----------: | :-----------------: |
| 1       | 10   | 12   | 11             | 12             | -1   | 0    | 11            | -0.5                |
| 2       | 13   | 15   | 13             | 14             | 0    | 1    | 14            | 0.5                 |
| 3       | 16   | 20   | 15             | 19             | 1    | 1    | 18            | 1.0                 |

전체 평균은 $`\bar{y} = 86/6 = 14.3333`$ 이고, 측정 위치 평균은 $`\bar{y}_{\cdot 1} = 13`$, $`\bar{y}_{\cdot 2} = 15.6667`$ 이다. 식 (7) 부터 식 (21) 로 구한 값은 아래 Table 3 이다.

Table 3. Sums of squares and R2 values

| Quantity             | Value                           | R2 or ratio      |
| :------------------: | :-----------------------------: | :--------------: |
| All                  | SST = 61.3333, SSE = 4          | R2_all = 0.9348  |
| Wafer average        | SST = 24.6667, SSE = 1.5        | R2_avgw = 0.9392 |
| Location 1           | SST = 18, SSE = 2               | R2_1 = 0.8889    |
| Location 2           | SST = 32.6667, SSE = 2          | R2_2 = 0.9388    |
| Location average     | -                               | R2_avgn = 0.9138 |
| Pooled               | SST = 50.6667, SSE = 4          | R2_pool = 0.9211 |
| Location pattern     | 3 × (1.7778 + 1.7778) = 10.6667 | lambda = 0.1739  |
| Measured correlation | cross sum = 24 (each order)     | r_y = 0.9474     |
| Error correlation    | cross sum = 1 (each order)      | r_e = 0.5        |
| Correlation ratio    | 1.5 / 1.9474                    | kappa = 0.7703   |

세 관계식에 Table 3 의 값을 넣으면 아래처럼 양변이 같다.

- 식 (22): 좌변 $`1 - 0.9348 = 0.0652`$, 우변 $`(1 - 0.1739)(1 - 0.9211) = 0.8261 \times 0.0789 = 0.0652`$.
- 식 (23): 좌변 $`1 - 0.9392 = 0.0608`$, 우변 $`0.7703 \times 0.0789 = 0.0608`$.
- 식 (24): 가중치는 $`18/50.6667 = 0.3553`$ 과 $`32.6667/50.6667 = 0.6447`$ 이고, $`0.3553 \times 0.8889 + 0.6447 \times 0.9388 = 0.9211`$.
- 식 (26): 좌변 $`0.0652`$, 우변 $`(0.8261/0.7703) \times 0.0608 = 1.0725 \times 0.0608 = 0.0652`$.

$`\bar{r}_e = 0.5`$ 가 $`\bar{r}_y = 0.9474`$ 보다 작아 $`\kappa \lt 1`$ 이므로 $`R^2_{\mathrm{avgw}} = 0.9392`$ 가 $`R^2_{\mathrm{pool}} = 0.9211`$ 보다 크다. $`\mathrm{SST}_j`$ 가 큰 측정 위치 2 의 $`R^2_2`$ 가 더 커서 $`R^2_{\mathrm{pool}} = 0.9211`$ 이 $`R^2_{\mathrm{avgn}} = 0.9138`$ 보다 크다. $`\lambda = 0.1739`$ 로 측정 위치 평균의 차이를 맞힌 몫이 더해져 $`R^2_{\mathrm{all}} = 0.9348`$ 이 $`R^2_{\mathrm{pool}}`$ 보다 크다.

## 7. Application

### 7.1 Reporting R2 of a Wafer Model

측정 위치마다 값을 예측하는 wafer model 은 같은 예측으로 세 R² 를 모두 구할 수 있으므로, 어느 R² 를 보고하는지에 따라 같은 model 의 성능이 다르게 보인다.

- $`R^2_{\mathrm{all}}`$: 측정 위치 평균의 차이 (wafer 안의 고정된 패턴) 를 맞힌 몫 $`\lambda`$ 가 들어 있다. 측정 위치 평균만 맞히고 wafer 사이 변동을 전혀 맞히지 못한 예측도 $`R^2_{\mathrm{all}} = \lambda`$ 이다.
- $`R^2_{\mathrm{avgw}}`$: 측정 위치 사이 예측 오차 상관 $`\bar{r}_e`$ 가 작을수록 커진다. Wafer 평균을 관리하는 공정에서 쓴다.
- $`R^2_{\mathrm{avgn}}`$ 과 $`R^2_{\mathrm{pool}}`$: wafer 사이 변동을 측정 위치마다 얼마나 맞히는지를 나타낸다. 측정 위치 평균의 차이와 평균할 때 줄어드는 오차가 들어 있지 않다.

### 7.2 Breaking Conditions

- Wafer 마다 측정 위치의 수나 위치가 다르면 식 (23) 이 성립하지 않는다. 식 (32) 에서 전체 평균을 측정 위치 평균의 산술 평균으로 바꾸는 단계에 모든 wafer 의 측정 위치가 같다는 가정이 쓰인다.
- Wafer 평균의 예측값을 측정 위치 예측값의 평균이 아니라 다른 model 로 구하면 식 (23) 과 식 (26) 이 성립하지 않는다.
- R² 를 측정값과 예측값의 상관계수 제곱으로 구하면 SSE 와 SST 의 비가 아니므로 식 (22) 부터 식 (26) 이 모두 성립하지 않는다.
- 어떤 측정 위치에서 측정값이 wafer 사이에서 같아 $`\mathrm{SST}_j = 0`$ 이면 $`R^2_j`$ 와 $`R^2_{\mathrm{avgn}}`$ 이 정의되지 않는다.

---

## Appendix A. Terminology

- **correlation ratio**: 식 (21) 의 $`\kappa`$. 측정 위치 $`N`$ 곳을 평균할 때 SSE 가 줄어드는 비율을 SST 가 줄어드는 비율로 나눈 값.
- **location pattern fraction**: 식 (18) 의 $`\lambda`$. 측정 위치 평균이 서로 달라 생기는 제곱합이 $`\mathrm{SST}_{\mathrm{all}}`$ 에서 차지하는 비율.
- **measurement location**: wafer 위에서 측정값을 얻는 지점. 모든 wafer 에서 같은 번호 $`j`$ 는 같은 지점이다.
- **pooled R2**: 여러 묶음의 SSE 와 SST 를 각각 더한 뒤 $`1 - \mathrm{SSE}/\mathrm{SST}`$ 로 구한 R².
- **R2 (coefficient of determination)**: $`1 - \mathrm{SSE}/\mathrm{SST}`$. 예측 오차 제곱합이 평균에서 벗어난 제곱합의 몇 분의 일인지를 1 에서 뺀 값.
- **sum of squared errors (SSE)**: 예측 오차를 제곱해 더한 값.
- **total sum of squares (SST)**: 측정값과 그 평균의 차이를 제곱해 더한 값.
- **wafer average**: wafer 한 장의 측정값 $`N`$ 개를 산술 평균한 값.

## Appendix B. Derivation

### B.1 Total Form

식 (22) 는 $`\mathrm{SST}_{\mathrm{all}}`$ 을 측정 위치마다 나누어 증명한다. 쓰는 정의 식은 아래 식 (6), 식 (7), 식 (8), 식 (9), 식 (13), 식 (14), 식 (17), 식 (18) 이다.

```math
\bar{y}_{\cdot j} = \frac{1}{M} \sum_{i=1}^{M} y_{ij}
\hspace{19em} (6)
```

```math
\mathrm{SST}_{\mathrm{all}} = \sum_{i=1}^{M} \sum_{j=1}^{N} \left( y_{ij} - \bar{y} \right)^2
\hspace{19em} (7)
```

```math
\mathrm{SSE}_{\mathrm{all}} = \sum_{i=1}^{M} \sum_{j=1}^{N} e_{ij}^2
\hspace{19em} (8)
```

```math
R^2_{\mathrm{all}} = 1 - \frac{\mathrm{SSE}_{\mathrm{all}}}{\mathrm{SST}_{\mathrm{all}}}
\hspace{19em} (9)
```

```math
\mathrm{SST}_j = \sum_{i=1}^{M} \left( y_{ij} - \bar{y}_{\cdot j} \right)^2
\hspace{19em} (13)
```

```math
\mathrm{SSE}_j = \sum_{i=1}^{M} e_{ij}^2
\hspace{19em} (14)
```

```math
R^2_{\mathrm{pool}} = 1 - \frac{\sum_{j=1}^{N} \mathrm{SSE}_j}{\sum_{j=1}^{N} \mathrm{SST}_j}
\hspace{19em} (17)
```

```math
\lambda = \frac{M \sum_{j=1}^{N} \left( \bar{y}_{\cdot j} - \bar{y} \right)^2}{\mathrm{SST}_{\mathrm{all}}}
\hspace{19em} (18)
```

```math
\mathrm{SST}_{\mathrm{all}} = \sum_{j=1}^{N} \sum_{i=1}^{M} \left[ \left( y_{ij} - \bar{y}_{\cdot j} \right) + \left( \bar{y}_{\cdot j} - \bar{y} \right) \right]^2
\hspace{19em} (27)
```

- 식 (27): 식 (7) 의 합의 순서를 바꾸고, $`y_{ij} - \bar{y}`$ 에 $`\bar{y}_{\cdot j}`$ 를 빼고 더했다.

```math
\mathrm{SST}_{\mathrm{all}} = \sum_{j=1}^{N} \mathrm{SST}_j + 2 \sum_{j=1}^{N} \left( \bar{y}_{\cdot j} - \bar{y} \right) \sum_{i=1}^{M} \left( y_{ij} - \bar{y}_{\cdot j} \right) + M \sum_{j=1}^{N} \left( \bar{y}_{\cdot j} - \bar{y} \right)^2
\hspace{19em} (28)
```

- 식 (28): 제곱을 전개하고, 첫 항에 식 (13) 을 썼다. $`\bar{y}_{\cdot j} - \bar{y}`$ 는 $`i`$ 에 따라 바뀌지 않으므로 $`i`$ 에 대한 합 밖으로 나온다.

```math
\mathrm{SST}_{\mathrm{all}} = \sum_{j=1}^{N} \mathrm{SST}_j + \lambda\, \mathrm{SST}_{\mathrm{all}}
\hspace{19em} (29)
```

- 식 (29): 식 (6) 에 따라 $`\sum_i (y_{ij} - \bar{y}_{\cdot j}) = M \bar{y}_{\cdot j} - M \bar{y}_{\cdot j} = 0`$ 이므로 가운데 항이 0 이다. 마지막 항은 식 (18) 로 $`\lambda\, \mathrm{SST}_{\mathrm{all}}`$ 이다.

```math
\mathrm{SSE}_{\mathrm{all}} = \sum_{j=1}^{N} \mathrm{SSE}_j
\hspace{19em} (30)
```

- 식 (30): 식 (8) 의 합의 순서를 바꾸고 식 (14) 를 썼다.

```math
1 - R^2_{\mathrm{all}} = \frac{\sum_{j} \mathrm{SST}_j}{\mathrm{SST}_{\mathrm{all}}} \cdot \frac{\sum_{j} \mathrm{SSE}_j}{\sum_{j} \mathrm{SST}_j}
\hspace{19em} (31)
```

- 식 (31): 식 (9) 에 식 (30) 을 넣고, 분자와 분모에 $`\sum_j \mathrm{SST}_j \gt 0`$ 을 곱했다.

식 (29) 에서 $`\sum_j \mathrm{SST}_j / \mathrm{SST}_{\mathrm{all}} = 1 - \lambda`$ 이고, 식 (17) 에서 $`\sum_j \mathrm{SSE}_j / \sum_j \mathrm{SST}_j = 1 - R^2_{\mathrm{pool}}`$ 이다. 따라서 식 (22) 가 성립한다.

```math
1 - R^2_{\mathrm{all}} = (1 - \lambda) \left( 1 - R^2_{\mathrm{pool}} \right)
\hspace{19em} (22)
```

### B.2 Wafer Average Form

식 (23) 은 wafer 평균의 SST 와 SSE 를 측정 위치마다의 제곱합과 교차합으로 나누어 증명한다. 쓰는 정의 식은 아래 식 (1) 부터 식 (6), 식 (10) 부터 식 (14), 식 (19), 식 (20), 식 (21) 이다.

```math
e_{ij} = y_{ij} - \hat{y}_{ij}
\hspace{19em} (1)
```

```math
\bar{y} = \frac{1}{MN} \sum_{i=1}^{M} \sum_{j=1}^{N} y_{ij}
\hspace{19em} (2)
```

```math
\bar{y}_{i\cdot} = \frac{1}{N} \sum_{j=1}^{N} y_{ij}
\hspace{19em} (3)
```

```math
\bar{\hat{y}}_{i\cdot} = \frac{1}{N} \sum_{j=1}^{N} \hat{y}_{ij}
\hspace{19em} (4)
```

```math
\bar{e}_{i\cdot} = \bar{y}_{i\cdot} - \bar{\hat{y}}_{i\cdot}
\hspace{19em} (5)
```

```math
\bar{y}_{\cdot j} = \frac{1}{M} \sum_{i=1}^{M} y_{ij}
\hspace{19em} (6)
```

```math
\mathrm{SST}_{\mathrm{avgw}} = \sum_{i=1}^{M} \left( \bar{y}_{i\cdot} - \bar{y} \right)^2
\hspace{19em} (10)
```

```math
\mathrm{SSE}_{\mathrm{avgw}} = \sum_{i=1}^{M} \bar{e}_{i\cdot}^2
\hspace{19em} (11)
```

```math
R^2_{\mathrm{avgw}} = 1 - \frac{\mathrm{SSE}_{\mathrm{avgw}}}{\mathrm{SST}_{\mathrm{avgw}}}
\hspace{19em} (12)
```

```math
\mathrm{SST}_j = \sum_{i=1}^{M} \left( y_{ij} - \bar{y}_{\cdot j} \right)^2
\hspace{19em} (13)
```

```math
\mathrm{SSE}_j = \sum_{i=1}^{M} e_{ij}^2
\hspace{19em} (14)
```

```math
\bar{r}_y = \frac{\sum_{j \ne k} \sum_{i=1}^{M} \left( y_{ij} - \bar{y}_{\cdot j} \right) \left( y_{ik} - \bar{y}_{\cdot k} \right)}{(N-1) \sum_{j=1}^{N} \mathrm{SST}_j}
\hspace{19em} (19)
```

```math
\bar{r}_e = \frac{\sum_{j \ne k} \sum_{i=1}^{M} e_{ij}\, e_{ik}}{(N-1) \sum_{j=1}^{N} \mathrm{SSE}_j}
\hspace{19em} (20)
```

```math
\kappa = \frac{1 + (N-1)\, \bar{r}_e}{1 + (N-1)\, \bar{r}_y}
\hspace{19em} (21)
```

```math
\bar{y}_{i\cdot} - \bar{y} = \frac{1}{N} \sum_{j=1}^{N} \left( y_{ij} - \bar{y}_{\cdot j} \right)
\hspace{19em} (32)
```

- 식 (32): 식 (3) 으로 $`\bar{y}_{i\cdot} = \frac{1}{N} \sum_j y_{ij}`$ 이고, 식 (2) 와 식 (6) 으로 $`\bar{y} = \frac{1}{N} \sum_j \bar{y}_{\cdot j}`$ 이다.

```math
\mathrm{SST}_{\mathrm{avgw}} = \frac{1}{N^2} \sum_{i=1}^{M} \left[ \sum_{j=1}^{N} \left( y_{ij} - \bar{y}_{\cdot j} \right) \right]^2
\hspace{19em} (33)
```

- 식 (33): 식 (10) 에 식 (32) 를 넣었다.

```math
\mathrm{SST}_{\mathrm{avgw}} = \frac{1}{N^2} \left[ \sum_{j=1}^{N} \mathrm{SST}_j + \sum_{j \ne k} \sum_{i=1}^{M} \left( y_{ij} - \bar{y}_{\cdot j} \right) \left( y_{ik} - \bar{y}_{\cdot k} \right) \right]
\hspace{19em} (34)
```

- 식 (34): 합의 제곱 $`(\sum_j a_j)^2 = \sum_j a_j^2 + \sum_{j \ne k} a_j a_k`$ 를 쓰고, $`i`$ 에 대한 합을 안으로 넣었다. 첫 항은 식 (13) 이다.

```math
\mathrm{SST}_{\mathrm{avgw}} = \frac{1}{N^2} \left[ 1 + (N-1)\, \bar{r}_y \right] \sum_{j=1}^{N} \mathrm{SST}_j
\hspace{19em} (35)
```

- 식 (35): 식 (19) 로 교차합은 $`(N-1)\, \bar{r}_y \sum_j \mathrm{SST}_j`$ 이다.

```math
\bar{e}_{i\cdot} = \frac{1}{N} \sum_{j=1}^{N} e_{ij}
\hspace{19em} (36)
```

- 식 (36): 식 (5) 에 식 (3) 과 식 (4) 를 넣고 식 (1) 을 썼다. 식 (4) 의 성립 조건이 여기에 쓰인다.

```math
\mathrm{SSE}_{\mathrm{avgw}} = \frac{1}{N^2} \left[ \sum_{j=1}^{N} \mathrm{SSE}_j + \sum_{j \ne k} \sum_{i=1}^{M} e_{ij}\, e_{ik} \right]
\hspace{19em} (37)
```

- 식 (37): 식 (11) 에 식 (36) 을 넣고, 식 (34) 와 같이 합의 제곱을 전개했다. 첫 항은 식 (14) 이다.

```math
\mathrm{SSE}_{\mathrm{avgw}} = \frac{1}{N^2} \left[ 1 + (N-1)\, \bar{r}_e \right] \sum_{j=1}^{N} \mathrm{SSE}_j
\hspace{19em} (38)
```

- 식 (38): 식 (20) 으로 교차합은 $`(N-1)\, \bar{r}_e \sum_j \mathrm{SSE}_j`$ 이다.

```math
1 - R^2_{\mathrm{avgw}} = \frac{1 + (N-1)\, \bar{r}_e}{1 + (N-1)\, \bar{r}_y} \cdot \frac{\sum_j \mathrm{SSE}_j}{\sum_j \mathrm{SST}_j}
\hspace{19em} (39)
```

- 식 (39): 식 (12) 에 식 (35) 와 식 (38) 을 넣고 $`1/N^2`$ 을 약분했다.

식 (39) 의 첫 분수는 식 (21) 의 $`\kappa`$ 이고, 둘째 분수는 식 (17) 의 $`1 - R^2_{\mathrm{pool}}`$ 이다. 따라서 식 (23) 이 성립한다.

```math
1 - R^2_{\mathrm{avgw}} = \kappa \left( 1 - R^2_{\mathrm{pool}} \right)
\hspace{19em} (23)
```

식 (35) 의 $`\mathrm{SST}_{\mathrm{avgw}}`$ 와 식 (38) 의 $`\mathrm{SSE}_{\mathrm{avgw}}`$ 는 제곱합이므로 0 이상이다. $`\sum_j \mathrm{SST}_j \gt 0`$ 과 $`\sum_j \mathrm{SSE}_j \gt 0`$ 에서 $`1 + (N-1)\bar{r}_y \ge 0`$ 과 $`1 + (N-1)\bar{r}_e \ge 0`$ 이고, 각각 $`\mathrm{SST}_{\mathrm{avgw}} = 0`$ 과 $`\mathrm{SSE}_{\mathrm{avgw}} = 0`$ 일 때만 0 이다.

### B.3 Location Average Form

식 (24) 는 식 (17) 의 비를 측정 위치마다의 비로 나누어 증명한다. 쓰는 정의 식은 아래 식 (15) 와 식 (17) 이다.

```math
R^2_j = 1 - \frac{\mathrm{SSE}_j}{\mathrm{SST}_j}
\hspace{19em} (15)
```

```math
R^2_{\mathrm{pool}} = 1 - \frac{\sum_{j=1}^{N} \mathrm{SSE}_j}{\sum_{j=1}^{N} \mathrm{SST}_j}
\hspace{19em} (17)
```

```math
1 - R^2_{\mathrm{pool}} = \sum_{j=1}^{N} \frac{\mathrm{SST}_j}{\sum_{k} \mathrm{SST}_k} \cdot \frac{\mathrm{SSE}_j}{\mathrm{SST}_j}
\hspace{19em} (40)
```

- 식 (40): 식 (17) 의 분자의 항마다 $`\mathrm{SST}_j \gt 0`$ 을 곱하고 나누었다.

```math
1 - R^2_{\mathrm{pool}} = \sum_{j=1}^{N} \frac{\mathrm{SST}_j}{\sum_{k} \mathrm{SST}_k} \left( 1 - R^2_j \right)
\hspace{19em} (41)
```

- 식 (41): 식 (15) 로 $`\mathrm{SSE}_j / \mathrm{SST}_j = 1 - R^2_j`$ 이다.

식 (41) 의 가중치 $`\mathrm{SST}_j / \sum_k \mathrm{SST}_k`$ 는 $`j`$ 에 대해 더하면 1 이므로, 1 을 빼고 부호를 바꾸면 식 (24) 가 성립한다.

```math
R^2_{\mathrm{pool}} = \sum_{j=1}^{N} \frac{\mathrm{SST}_j}{\sum_{k=1}^{N} \mathrm{SST}_k}\, R^2_j
\hspace{19em} (24)
```

$`\mathrm{SST}_j`$ 가 모두 같으면 가중치가 모두 $`1/N`$ 이 되고, 식 (16) 에 따라 식 (25) 가 성립한다.

```math
R^2_{\mathrm{avgn}} = \frac{1}{N} \sum_{j=1}^{N} R^2_j
\hspace{19em} (16)
```

```math
R^2_{\mathrm{pool}} = R^2_{\mathrm{avgn}}
\hspace{19em} (25)
```

### B.4 Relation without the Pooled R2

식 (26) 은 B.1 의 식 (22) 와 B.2 의 식 (23) 에서 $`1 - R^2_{\mathrm{pool}}`$ 을 지워 증명한다.

```math
1 - R^2_{\mathrm{all}} = (1 - \lambda) \left( 1 - R^2_{\mathrm{pool}} \right)
\hspace{19em} (22)
```

```math
1 - R^2_{\mathrm{avgw}} = \kappa \left( 1 - R^2_{\mathrm{pool}} \right)
\hspace{19em} (23)
```

```math
1 - R^2_{\mathrm{pool}} = \frac{1 - R^2_{\mathrm{avgw}}}{\kappa}
\hspace{19em} (42)
```

- 식 (42): 식 (23) 의 양변을 $`\kappa`$ 로 나누었다. $`\mathrm{SSE}_{\mathrm{avgw}} \gt 0`$ 이면 B.2 의 끝 문단에 따라 $`\kappa \gt 0`$ 이다.

식 (42) 를 식 (22) 에 넣으면 식 (26) 이 성립한다.

```math
1 - R^2_{\mathrm{all}} = \frac{1 - \lambda}{\kappa} \left( 1 - R^2_{\mathrm{avgw}} \right)
\hspace{19em} (26)
```
