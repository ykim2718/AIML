# R2 by Aggregation Level
Version 47 | Created: 2026-11-09 | Updated: 2026-11-12 05:16 UTC

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Notation](#3-notation)
- [4. Definitions](#4-definitions)
- [5. Interpretation](#5-interpretation)
  - [5.1 R2 over All Measured Values Form](#51-r2-over-all-measured-values-form)
  - [5.2 R2 over Wafer Averages Form](#52-r2-over-wafer-averages-form)
  - [5.3 R2 Average over N Locations Form](#53-r2-average-over-n-locations-form)
  - [5.4 Relation without the SST-weighted R2 Average over N Locations](#54-relation-without-the-sst-weighted-r2-average-over-n-locations)
  - [5.5 Inequalities](#55-inequalities)
  - [5.6 Inequalities among the Goal Quantities](#56-inequalities-among-the-goal-quantities)
  - [5.7 Wafer Average Form of Lambda and Kappa](#57-wafer-average-form-of-lambda-and-kappa)
- [6. Worked Example](#6-worked-example)
- [7. Application](#7-application)
  - [7.1 Reporting R2 of a Wafer Model](#71-reporting-r2-of-a-wafer-model)
  - [7.2 Breaking Conditions](#72-breaking-conditions)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Derivation](#appendix-b-derivation)
  - [B.1 R2 over All Measured Values Form](#b1-r2-over-all-measured-values-form)
  - [B.2 R2 over Wafer Averages Form](#b2-r2-over-wafer-averages-form)
  - [B.3 R2 Average over N Locations Form](#b3-r2-average-over-n-locations-form)
  - [B.4 Relation without the SST-weighted R2 Average over N Locations](#b4-relation-without-the-sst-weighted-r2-average-over-n-locations)
  - [B.5 Inequalities](#b5-inequalities)
  - [B.6 Wafer Average Form of Lambda and Kappa](#b6-wafer-average-form-of-lambda-and-kappa)
- [Appendix C. Intuitive Meaning](#appendix-c-intuitive-meaning)

## 1. Purpose

- **Problem Statement**: 반도체 wafer $`M`$ 장에서 wafer 마다 측정 위치 (measurement location) $`N`$ 곳을 측정하면 측정값이 $`T = MN`$ 개 나온다. 같은 예측으로 R² 를 측정값 $`T`$ 개, wafer 평균 $`M`$ 개, 측정 위치마다의 wafer $`M`$ 장에서 구하면 세 R² 의 값이 서로 다르지만, 세 값 사이의 관계식은 없다.
- **Goal**: 전체 측정값으로 구한 $`R^2_{\mathrm{all}}`$, wafer 평균값으로 구한 $`R^2_{\mathrm{avgw}}`$, 측정 위치 R² 평균 $`R^2_{\mathrm{avgn}}`$ 사이의 항등식과 부등식을 구하고 증명한다.
- **Non-Goal**: 예측 model 을 만들거나 고르는 방법, R² 의 표본 분포는 다루지 않는다.

## 2. Summary

세 R² 는 측정 위치 R² 의 SST 가중 평균 $`R^2_{\mathrm{wavgn}}`$ 하나로 이어지며, $`1 - R^2_{\mathrm{all}} = (1 - \lambda)(1 - R^2_{\mathrm{wavgn}})`$, $`1 - R^2_{\mathrm{avgw}} = \kappa (1 - R^2_{\mathrm{wavgn}})`$ 이다.
$`R^2_{\mathrm{wavgn}}`$ 은 측정 위치마다의 $`R^2_j`$ 를 $`\mathrm{SST}_j`$ 로 가중 평균한 값이고, $`\mathrm{SST}_j`$ 가 모두 같으면 $`R^2_{\mathrm{avgn}}`$ 과 같다.
그래서 $`R^2_{\mathrm{all}} \ge R^2_{\mathrm{wavgn}}`$ 은 언제나 성립하고, $`R^2_{\mathrm{avgw}} \ge R^2_{\mathrm{wavgn}}`$ 은 오차 비 $`\kappa`$ 가 1 이하일 때 성립한다.

## 3. Notation

Table 1. Symbols

| Symbol                             | Meaning                                                                                                                                                                                                 | Scope                        | Shape   | Assumption                        |
| :--------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------: | :-----: | :-------------------------------: |
| $`M`$                              | wafer 수                                                                                                                                                                                                | -                            | integer | $`M \ge 2`$                       |
| $`N`$                              | wafer 한 장에서 측정하는 위치의 수                                                                                                                                                                      | -                            | integer | $`N \ge 2`$, 모든 wafer 에서 같음 |
| $`T`$                              | 전체 (all) 측정값의 개수 $`MN`$                                                                                                                                                                         | -                            | integer | -                                 |
| $`i`$                              | wafer 번호                                                                                                                                                                                              | -                            | integer | $`1 \le i \le M`$                 |
| $`j`$, $`k`$                       | 측정 위치의 번호                                                                                                                                                                                        | -                            | integer | $`1 \le j, k \le N`$              |
| $`y_{ij}`$                         | wafer $`i`$ 의 측정 위치 $`j`$ 에서 측정값                                                                                                                                                              | per wafer,<br>per location   | scalar  | -                                 |
| $`\hat{y}_{ij}`$                   | wafer $`i`$ 의 측정 위치 $`j`$ 에 대한 예측값                                                                                                                                                           | per wafer,<br>per location   | scalar  | -                                 |
| $`e_{ij}`$                         | 예측 오차 $`y_{ij} - \hat{y}_{ij}`$                                                                                                                                                                     | per wafer,<br>per location   | scalar  | -                                 |
| $`\bar{y}`$                        | 측정값 $`T`$ 개의 평균                                                                                                                                                                                  | all wafers,<br>all locations | scalar  | -                                 |
| $`\bar{y}_{i\cdot}`$               | wafer $`i`$ 의 wafer 평균 (average of wafer, avgw)                                                                                                                                                      | per wafer,<br>all locations  | scalar  | -                                 |
| $`\hat{y}_{iw}`$                   | wafer $`i`$ 의 wafer 평균을 직접 예측한 값                                                                                                                                                              | per wafer,<br>all locations  | scalar  | -                                 |
| $`e_{iw}`$                         | wafer 평균에서 그 예측값을 뺀 오차 $`\bar{y}_{i\cdot} - \hat{y}_{iw}`$                                                                                                                                  | per wafer,<br>all locations  | scalar  | -                                 |
| $`\bar{y}_{\cdot j}`$              | 측정 위치 $`j`$ 에서 wafer $`M`$ 장의 측정값 평균                                                                                                                                                       | per location,<br>all wafers  | scalar  | -                                 |
| $`\mathrm{SST}`$, $`\mathrm{SSE}`$ | 평균에서 벗어난 제곱합 (total sum of squares) 과 오차 제곱합 (sum of squared errors)                                                                                                                    | 아래첨자에 따름              | scalar  | -                                 |
| $`R^2_j`$                          | 측정 위치 $`j`$ 에서 wafer $`M`$ 장으로 구한 R². $`N`$ 곳에서 산술 평균하면 avgn (average over $`N`$ locations)                                                                                         | per location,<br>all wafers  | scalar  | $`\mathrm{SST}_j \gt 0`$          |
| $`\omega_j`$                       | 측정 위치 $`j`$ 의 SST 가중치. $`\mathrm{SST}_j`$ 를 측정 위치 $`N`$ 곳의 $`\mathrm{SST}_k`$ 의 합으로 나눈 값                                                                                          | per location,<br>all wafers  | scalar  | $`\sum_j \omega_j = 1`$           |
| $`R^2_{\mathrm{wavgn}}`$           | 측정 위치마다 구한 $`R^2_j`$ 를 $`\mathrm{SST}_j`$ 에 비례하는 가중치로 평균한 값 (SST-weighted R2 average over $`N`$ locations, wavgn)                                                                 | all wafers,<br>all locations | scalar  | -                                 |
| $`\lambda`$                        | 측정 위치 평균의 차이가 $`\mathrm{SST}_{\mathrm{all}}`$ 에서 차지하는 비율 (location pattern fraction)                                                                                                  | all wafers,<br>all locations | scalar  | $`0 \le \lambda \lt 1`$           |
| $`\bar{r}_y`$                      | 서로 다른 두 측정 위치에서 측정값이 이루는 상관                                                                                                                                                         | all wafers,<br>all locations | scalar  | $`\bar{r}_y \gt -1/(N-1)`$        |
| $`\kappa`$                         | 오차 비 (error ratio). wafer 평균 예측의 오차 몫 $`\mathrm{SSE}_{\mathrm{avgw}}/\mathrm{SST}_{\mathrm{avgw}}`$ 를 측정 위치 예측의 오차 몫 $`\sum_j \mathrm{SSE}_j / \sum_j \mathrm{SST}_j`$ 로 나눈 값 | all wafers,<br>all locations | scalar  | $`\kappa \gt 0`$                  |

🥑 아래첨자의 점 ($`\cdot`$) 은 점 표기 (dot notation) 로, 점으로 바꾼 첨자에 대해 모두 더했다는 뜻이다. 예를 들어 $`y_{i\cdot} = \sum_{j} y_{ij}`$ 는 wafer $`i`$ 의 측정값 $`N`$ 개를 모두 더한 값이다. 점 표기에 bar 를 얹으면 합 대신 평균이 되어, $`\bar{y}_{i\cdot}`$ 는 $`j`$ 에 대한 평균이고 $`\bar{y}_{\cdot j}`$ 는 $`i`$ 에 대한 평균이다.

SST, SSE, R² 의 아래첨자 $`\mathrm{all}`$ 은 측정값 $`T`$ 개, $`\mathrm{avgw}`$ 는 wafer 평균값, $`j`$ 는 측정 위치 $`j`$, $`\mathrm{avgn}`$ 은 측정 위치 R² 평균, $`\mathrm{wavgn}`$ 은 측정 위치 $`N`$ 곳의 R² 를 SST 로 가중 평균한 것 (weighted average over $`N`$ locations) 을 뜻한다. $`\bar{r}_y`$ 의 아래첨자 $`y`$ 는 측정값을 뜻하고, $`\hat{y}_{iw}`$ 와 $`e_{iw}`$ 의 둘째 아래첨자 $`w`$ 는 측정 위치 번호가 아니라, 측정 위치마다 구하지 않고 wafer 한 장에 하나로 직접 예측한 값임을 뜻한다. 이 문서의 SST, SSE, R², $`\lambda`$, $`\bar{r}_y`$, $`\kappa`$ 는 주어진 측정값과 예측값으로 정의하는 양이며 모수를 추정한 값이 아니므로 hat 을 붙이지 않는다. Hat 을 붙인 $`\hat{y}_{ij}`$ 와 $`\hat{y}_{iw}`$ 는 예측값이다.

- $`\hat{y}_{ij}`$: 영어로 "y hat i j", 한국어로 "와이 햇 아이 제이" 라 읽는다.
- $`\bar{y}`$: 영어로 "y bar", 한국어로 "와이 바" 라 읽는다.
- $`\bar{y}_{i\cdot}`$: 영어로 "y bar i dot", 한국어로 "와이 바 아이 닷" 이라 읽는다.
- $`\bar{y}_{\cdot j}`$: 영어로 "y bar dot j", 한국어로 "와이 바 닷 제이" 라 읽는다.
- $`\hat{y}_{iw}`$: 영어로 "y hat i w", 한국어로 "와이 햇 아이 더블유" 라 읽는다.
- $`\lambda`$: 영어로 "lambda", 한국어로 "람다" 라 읽는다.
- $`\kappa`$: 영어로 "kappa", 한국어로 "카파" 라 읽는다.
- $`\omega_j`$: 영어로 "omega j", 한국어로 "오메가 제이" 라 읽는다.
- $`\sum`$: 영어로 "sum", 한국어로 "합" 이라 읽는다.
- $`\min_j`$, $`\max_j`$: 영어로 "minimum over j", "maximum over j", 한국어로 "제이에 대한 최솟값", "제이에 대한 최댓값" 이라 읽는다.
- $`\iff`$: 영어로 "if and only if", 한국어로 "필요충분조건" 이라 읽는다.
- $`\sum_{j \ne k}`$: 영어로 "sum over j not equal to k", 한국어로 "제이와 케이가 다른 모든 순서쌍에 대한 합" 이라 읽는다.

## 4. Definitions

> **Definition: R2 over all measured values**<br>
> 전체 측정값으로 구한 $`R^2_{\mathrm{all}}`$ 은 wafer $`M`$ 장, 측정 위치 $`N`$ 곳의 측정값 $`T = MN`$ 개에서, 1 에서 예측 오차 제곱합 $`\mathrm{SSE}_{\mathrm{all}}`$ 을 측정값과 전체 평균의 차이의 제곱합 $`\mathrm{SST}_{\mathrm{all}}`$ 로 나눈 값을 뺀 값이다.

> **Definition: R2 over wafer averages**<br>
> Wafer 평균값으로 구한 $`R^2_{\mathrm{avgw}}`$ 는 wafer 평균 $`M`$ 개에서, 1 에서 wafer 평균 예측 오차의 제곱합 $`\mathrm{SSE}_{\mathrm{avgw}}`$ 를 wafer 평균과 전체 평균의 차이의 제곱합 $`\mathrm{SST}_{\mathrm{avgw}}`$ 로 나눈 값을 뺀 값이다. Wafer 평균의 예측값은 wafer 평균 하나를 직접 예측한 값이다.

> **Definition: R2 average over N locations**<br>
> 측정 위치 R² 평균 $`R^2_{\mathrm{avgn}}`$ 은 측정 위치 $`j`$ 마다 wafer $`M`$ 장의 측정값과 예측값으로 구한 R² $`R^2_j`$ 를, 측정 위치 $`N`$ 곳에서 산술 평균한 값이다.

세 R² 의 정의 식은 아래 식 (1) 부터 식 (15) 이다. 식 (1) 부터 식 (5) 는 Table 1 의 $`e_{ij}`$, $`\bar{y}`$, $`\bar{y}_{i\cdot}`$, $`e_{iw}`$, $`\bar{y}_{\cdot j}`$ 를 식으로 적은 것이다. $`\hat{y}_{iw}`$ 는 측정 위치 예측값 $`\hat{y}_{ij}`$ 와 따로 구한 값이어서 식이 없다.

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
e_{iw} = \bar{y}_{i\cdot} - \hat{y}_{iw}
\hspace{19em} (4)
```

```math
\bar{y}_{\cdot j} = \frac{1}{M} \sum_{i=1}^{M} y_{ij}
\hspace{19em} (5)
```

식 (6) 부터 식 (8) 은 $`R^2_{\mathrm{all}}`$, 식 (9) 부터 식 (11) 은 $`R^2_{\mathrm{avgw}}`$, 식 (12) 부터 식 (15) 는 $`R^2_{\mathrm{avgn}}`$ 과, $`R^2_{\mathrm{avgn}}`$ 이 평균하는 $`R^2_j`$ 의 식이다.

```math
\mathrm{SST}_{\mathrm{all}} = \sum_{i=1}^{M} \sum_{j=1}^{N} \left( y_{ij} - \bar{y} \right)^2
\hspace{19em} (6)
```

```math
\mathrm{SSE}_{\mathrm{all}} = \sum_{i=1}^{M} \sum_{j=1}^{N} e_{ij}^2
\hspace{19em} (7)
```

```math
R^2_{\mathrm{all}} = 1 - \frac{\mathrm{SSE}_{\mathrm{all}}}{\mathrm{SST}_{\mathrm{all}}}
\hspace{19em} (8)
```

```math
\mathrm{SST}_{\mathrm{avgw}} = \sum_{i=1}^{M} \left( \bar{y}_{i\cdot} - \bar{y} \right)^2
\hspace{19em} (9)
```

```math
\mathrm{SSE}_{\mathrm{avgw}} = \sum_{i=1}^{M} e_{iw}^2
\hspace{19em} (10)
```

```math
R^2_{\mathrm{avgw}} = 1 - \frac{\mathrm{SSE}_{\mathrm{avgw}}}{\mathrm{SST}_{\mathrm{avgw}}}
\hspace{19em} (11)
```

```math
\mathrm{SST}_j = \sum_{i=1}^{M} \left( y_{ij} - \bar{y}_{\cdot j} \right)^2
\hspace{19em} (12)
```

```math
\mathrm{SSE}_j = \sum_{i=1}^{M} e_{ij}^2
\hspace{19em} (13)
```

```math
R^2_j = 1 - \frac{\mathrm{SSE}_j}{\mathrm{SST}_j}
\hspace{19em} (14)
```

```math
R^2_{\mathrm{avgn}} = \frac{1}{N} \sum_{j=1}^{N} R^2_j
\hspace{19em} (15)
```

- $`R^2_{\mathrm{all}}`$, $`R^2_{\mathrm{avgw}}`$, $`R^2_j`$ 는 $`1 - \mathrm{SSE}/\mathrm{SST}`$ 꼴이며, 측정값과 예측값의 상관계수를 제곱한 값과는 다르다.
- $`e_{ij}`$ 와 $`e_{iw}`$ 가 모두 0 이면 $`R^2_{\mathrm{all}}`$, $`R^2_{\mathrm{avgw}}`$, $`R^2_j`$, $`R^2_{\mathrm{avgn}}`$ 은 모두 1 이다. 예측이 평균보다 나쁘면 R² 는 음수일 수 있다.

Table 1 의 $`R^2_{\mathrm{wavgn}}`$, $`\lambda`$, $`\kappa`$, $`\omega_j`$ 를 식으로 적으면 아래 식 (16), 식 (17), 식 (18), 식 (19) 이다. $`\bar{r}_y`$ 는 B.2 에서만 쓰므로 그 정의 식은 [B.2](#b2-r2-over-wafer-averages-form) 에 있다.

```math
R^2_{\mathrm{wavgn}} = 1 - \frac{\sum_{j=1}^{N} \mathrm{SSE}_j}{\sum_{j=1}^{N} \mathrm{SST}_j}
\hspace{19em} (16)
```

```math
\lambda = \frac{M \sum_{j=1}^{N} \left( \bar{y}_{\cdot j} - \bar{y} \right)^2}{\mathrm{SST}_{\mathrm{all}}}
\hspace{19em} (17)
```

```math
\kappa = \frac{\mathrm{SSE}_{\mathrm{avgw}}}{\mathrm{SST}_{\mathrm{avgw}}} \cdot \frac{\sum_{j=1}^{N} \mathrm{SST}_j}{\sum_{j=1}^{N} \mathrm{SSE}_j}
\hspace{19em} (18)
```

```math
\omega_j = \frac{\mathrm{SST}_j}{\sum_{k=1}^{N} \mathrm{SST}_k}
\hspace{19em} (19)
```

## 5. Interpretation

세 R² 는 측정 위치 R² 의 SST 가중 평균 $`R^2_{\mathrm{wavgn}}`$ 을 거쳐 아래 식 (20), 식 (21), 식 (22) 로 이어진다. 세 식은 예측값을 어떤 model 로 구했는지와 상관없이 성립하는 항등식이며, 증명은 [Appendix B](#appendix-b-derivation) 에 있다.

```math
1 - R^2_{\mathrm{all}} = (1 - \lambda) \left( 1 - R^2_{\mathrm{wavgn}} \right)
\hspace{19em} (20)
```

```math
1 - R^2_{\mathrm{avgw}} = \kappa \left( 1 - R^2_{\mathrm{wavgn}} \right)
\hspace{19em} (21)
```

```math
R^2_{\mathrm{wavgn}} = \sum_{j=1}^{N} \omega_j R^2_j
\hspace{19em} (22)
```

세 식은 모든 wafer 가 같은 측정 위치 $`N`$ 곳을 갖는다고 둔다. 식 (21) 은 wafer 평균의 예측값 $`\hat{y}_{iw}`$ 를 측정 위치 예측값과 어떻게 관련지어 구했는지와 상관없이 성립한다.

### 5.1 R2 over All Measured Values Form

식 (20) 의 $`1 - \lambda`$ 는 $`\mathrm{SST}_{\mathrm{all}}`$ 가운데 측정 위치마다의 wafer 사이 변동 $`\sum_j \mathrm{SST}_j`$ 가 차지하는 비율이다. 식 (35) 에서 $`\mathrm{SST}_{\mathrm{all}}`$ 은 이 변동과 측정 위치 평균의 차이 $`\lambda\, \mathrm{SST}_{\mathrm{all}}`$ 로 나뉘지만, 식 (36) 에서 $`\mathrm{SSE}_{\mathrm{all}}`$ 은 $`\mathrm{SSE}_j`$ 의 합과 같아 그대로 남는다. 그래서 $`R^2_{\mathrm{all}}`$ 은 측정 위치 평균의 차이를 맞힌 몫 $`\lambda (1 - R^2_{\mathrm{wavgn}})`$ 만큼 $`R^2_{\mathrm{wavgn}}`$ 보다 크다.

- 성립 조건은 $`\sum_j \mathrm{SST}_j \gt 0`$ 이며, 이때 $`\lambda \lt 1`$ 이다.

### 5.2 R2 over Wafer Averages Form

식 (21) 의 $`\kappa`$ 는 식 (18) 대로 wafer 평균 예측의 오차 몫 $`\mathrm{SSE}_{\mathrm{avgw}} / \mathrm{SST}_{\mathrm{avgw}}`$ 를 측정 위치 예측의 오차 몫 $`\sum_j \mathrm{SSE}_j / \sum_j \mathrm{SST}_j`$ 로 나눈 값이다. 그래서 $`\kappa \lt 1`$ 이면 wafer 평균 예측의 오차 몫이 측정 위치 예측의 오차 몫보다 작아 $`R^2_{\mathrm{avgw}}`$ 가 $`R^2_{\mathrm{wavgn}}`$ 보다 커진다. 식 (43) 은 같은 $`\kappa`$ 를 측정값의 상관 $`\bar{r}_y`$ 로 적은 것이다.

- Wafer 사이 변동이 모든 측정 위치에서 같은 양인 $`\bar{r}_y = 1`$ 이면 식 (43) 에서 $`\kappa = N \mathrm{SSE}_{\mathrm{avgw}} / \sum_j \mathrm{SSE}_j`$ 이다. 이때 wafer 평균 예측의 SSE 가 측정 위치 SSE 합의 $`1/N`$ 보다 작으면 $`R^2_{\mathrm{avgw}} \gt R^2_{\mathrm{wavgn}}`$ 이다.
- 성립 조건은 $`\mathrm{SST}_{\mathrm{avgw}} \gt 0`$ 과 $`\sum_j \mathrm{SSE}_j \gt 0`$ 이다. $`\sum_j \mathrm{SSE}_j = 0`$ 이면 $`\kappa`$ 가 정의되지 않는다.

### 5.3 R2 Average over N Locations Form

식 (22) 에서 $`R^2_{\mathrm{wavgn}}`$ 은 $`R^2_j`$ 를 $`\mathrm{SST}_j`$ 에 비례하는 가중치로 평균한 값이고, $`R^2_{\mathrm{avgn}}`$ 은 같은 $`R^2_j`$ 를 가중치 $`1/N`$ 로 평균한 값이다. 그래서 $`\mathrm{SST}_j`$ 가 모두 같으면 식 (23) 이 성립한다. $`R^2_{\mathrm{wavgn}}`$ 의 직관적 의미는 [Appendix C](#appendix-c-intuitive-meaning) 에 있다.

```math
R^2_{\mathrm{wavgn}} = R^2_{\mathrm{avgn}}
\hspace{19em} (23)
```

- 성립 조건은 모든 $`j`$ 에서 $`\mathrm{SST}_j \gt 0`$ 이다.

### 5.4 Relation without the SST-weighted R2 Average over N Locations

식 (20) 을 식 (21) 로 나누어 $`1 - R^2_{\mathrm{wavgn}}`$ 을 지우면 $`R^2_{\mathrm{all}}`$ 과 $`R^2_{\mathrm{avgw}}`$ 의 관계는 식 (24) 이다.

```math
1 - R^2_{\mathrm{all}} = \frac{1 - \lambda}{\kappa} \left( 1 - R^2_{\mathrm{avgw}} \right)
\hspace{19em} (24)
```

- $`\mathrm{SST}_j`$ 가 모두 같으면 식 (23) 에 따라 식 (20) 과 식 (21) 의 $`R^2_{\mathrm{wavgn}}`$ 을 $`R^2_{\mathrm{avgn}}`$ 으로 바꿀 수 있어, 세 R² 가 $`\lambda`$ 와 $`\kappa`$ 만으로 이어진다.
- 성립 조건은 5.1 과 5.2 의 조건에 $`\mathrm{SSE}_{\mathrm{avgw}} \gt 0`$ 을 더한 것이다. 이 조건에서 식 (18) 의 네 제곱합이 양수이므로 $`\kappa \gt 0`$ 이다.

### 5.5 Inequalities

세 R² 와 $`R^2_{\mathrm{wavgn}}`$ 사이의 부등식은 아래 식 (25) 부터 식 (30) 이며, 식 (20), 식 (21), 식 (22), 식 (24) 의 두 변의 차이에서 나온다. 증명은 [B.5](#b5-inequalities) 에 있다.

```math
R^2_{\mathrm{all}} \ge R^2_{\mathrm{wavgn}}
\hspace{19em} (25)
```

- 식 (25): 언제나 성립한다. 두 값의 차이는 $`\lambda (1 - R^2_{\mathrm{wavgn}})`$ 이며, 측정 위치 평균이 모두 같은 $`\lambda = 0`$ 이거나 예측 오차가 모두 0 일 때만 두 값이 같다.

```math
R^2_{\mathrm{wavgn}} \ge \min_{j} R^2_j
\hspace{19em} (26)
```

```math
R^2_{\mathrm{wavgn}} \le \max_{j} R^2_j
\hspace{19em} (27)
```

- 식 (26), 식 (27): 언제나 성립한다. $`R^2_{\mathrm{wavgn}}`$ 은 식 (22) 에서 $`R^2_j`$ 의 가중 평균이므로 $`R^2_j`$ 의 최솟값과 최댓값 사이에 있다. 산술 평균인 $`R^2_{\mathrm{avgn}}`$ 도 같은 범위에 있다.
- 식 (25) 와 식 (26) 을 이으면 $`R^2_{\mathrm{all}} \ge \min_j R^2_j`$ 이다. 전체 측정값으로 구한 $`R^2_{\mathrm{all}}`$ 은 가장 낮은 측정 위치 $`R^2_j`$ 보다 낮아지지 않는다.

```math
R^2_{\mathrm{wavgn}} \ge R^2_{\mathrm{avgn}} \iff \sum_{j=1}^{N} \left( \omega_j - \frac{1}{N} \right) R^2_j \ge 0
\hspace{19em} (28)
```

- 식 (28): 우변의 괄호는 측정 위치 $`j`$ 의 가중치가 $`1/N`$ 보다 큰 정도이다. $`\mathrm{SST}_j`$ 가 큰 측정 위치에서 $`R^2_j`$ 가 크면 우변이 0 이상이어서 $`R^2_{\mathrm{wavgn}} \ge R^2_{\mathrm{avgn}}`$ 이고, 작으면 $`R^2_{\mathrm{wavgn}} \lt R^2_{\mathrm{avgn}}`$ 일 수 있다.

```math
R^2_{\mathrm{avgw}} \ge R^2_{\mathrm{wavgn}} \iff \kappa \le 1
\hspace{19em} (29)
```

- 식 (29): 오차 비 $`\kappa`$ 가 1 이하이면 $`R^2_{\mathrm{avgw}}`$ 가 $`R^2_{\mathrm{wavgn}}`$ 이상이다. Wafer 평균 예측값 $`\hat{y}_{iw}`$ 는 측정 위치 예측과 따로 구하므로, 측정 위치 예측이 아무리 좋아도 $`\hat{y}_{iw}`$ 의 오차가 크면 $`\kappa \gt 1`$ 이 되어 $`R^2_{\mathrm{avgw}}`$ 가 낮아진다.

```math
R^2_{\mathrm{all}} \ge R^2_{\mathrm{avgw}} \iff \kappa \ge 1 - \lambda
\hspace{19em} (30)
```

- 식 (30): 측정 위치 평균의 차이가 차지하는 비율 $`\lambda`$ 가 클수록, 오차 비 $`\kappa`$ 가 클수록 $`R^2_{\mathrm{all}}`$ 이 $`R^2_{\mathrm{avgw}}`$ 이상이 되기 쉽다.
- 성립 조건: 식 (25) 는 $`\sum_j \mathrm{SST}_j \gt 0`$, 식 (26) 부터 식 (28) 은 모든 $`j`$ 에서 $`\mathrm{SST}_j \gt 0`$, 식 (29) 는 $`\mathrm{SST}_{\mathrm{avgw}} \gt 0`$ 과 $`\sum_j \mathrm{SSE}_j \gt 0`$ 이다. 식 (30) 은 여기에 $`\mathrm{SSE}_{\mathrm{avgw}} \gt 0`$ 을 더한다.

### 5.6 Inequalities among the Goal Quantities

Goal 의 세 R² 사이에는 언제나 성립하는 순서가 없고, 두 값의 순서는 아래 Table 2 의 조건으로 정해진다.

Table 2. Inequalities among the three R2 in the Goal

| Pair                                             | Inequality                                      | Condition                                           | From             |
| :----------------------------------------------: | :---------------------------------------------: | :-------------------------------------------------: | :--------------: |
| $`R^2_{\mathrm{all}}`$, $`R^2_{\mathrm{avgw}}`$  | $`R^2_{\mathrm{all}} \ge R^2_{\mathrm{avgw}}`$  | $`\kappa \ge 1 - \lambda`$ 일 때, 그리고 그때만     | 식 (30)          |
| $`R^2_{\mathrm{all}}`$, $`R^2_{\mathrm{avgn}}`$  | $`R^2_{\mathrm{all}} \ge R^2_{\mathrm{avgn}}`$  | 식 (28) 의 우변이 0 이상일 때                       | 식 (25), 식 (28) |
| $`R^2_{\mathrm{avgw}}`$, $`R^2_{\mathrm{avgn}}`$ | $`R^2_{\mathrm{avgw}} \ge R^2_{\mathrm{avgn}}`$ | $`\kappa \le 1`$ 이고 식 (28) 의 우변이 0 이상일 때 | 식 (29), 식 (28) |

- 첫 행의 조건 $`\kappa \ge 1 - \lambda`$ 는 $`\lambda \ge 1 - \kappa`$ 와 같다. 식 (20) 과 식 (21) 에서 두 R² 는 모두 $`1 - R^2_{\mathrm{wavgn}}`$ 에 한 계수를 곱한 오차 몫을 1 에서 뺀 값이며, 그 계수가 $`R^2_{\mathrm{all}}`$ 은 $`1 - \lambda`$, $`R^2_{\mathrm{avgw}}`$ 는 $`\kappa`$ 이다. $`\lambda`$ 는 wafer 마다 되풀이되는 측정 위치 평균의 차이, 곧 wafer 안의 고정된 측정 위치 패턴이 $`\mathrm{SST}_{\mathrm{all}}`$ 에서 차지하는 비율이고, 예측이 이 패턴을 맞히는 만큼 $`R^2_{\mathrm{all}}`$ 의 오차 몫이 줄어든다. $`1 - \kappa`$ 는 측정 위치 $`N`$ 곳을 wafer 평균 하나로 모을 때 SSE 가 SST 보다 더 줄어드는 비율이고, wafer 평균의 예측이 측정 위치 사이 오차를 덜 남길수록 커진다. 따라서 $`R^2_{\mathrm{all}} \ge R^2_{\mathrm{avgw}}`$ 는 wafer 안의 고정된 측정 위치 패턴이 차지하는 몫이 wafer 평균으로 모아 줄어드는 오차의 몫 이상이라는 뜻이다. Table 4 의 예에서는 $`\lambda = 0.1739`$ 가 $`1 - \kappa = 0.2297`$ 보다 작아 $`R^2_{\mathrm{avgw}}`$ 가 더 크다.
- 첫 행은 필요충분조건이고, 둘째 행과 셋째 행은 충분조건이다. $`R^2_{\mathrm{all}}`$ 과 $`R^2_{\mathrm{avgn}}`$, $`R^2_{\mathrm{avgw}}`$ 와 $`R^2_{\mathrm{avgn}}`$ 은 두 값을 바로 잇는 항등식이 없어, $`R^2_{\mathrm{wavgn}}`$ 을 사이에 두고 두 부등식을 이어서 얻는다.
- $`\mathrm{SST}_j`$ 가 모두 같으면 식 (23) 에 따라 $`R^2_{\mathrm{wavgn}} = R^2_{\mathrm{avgn}}`$ 이므로, 둘째 행은 언제나 성립하고 셋째 행은 $`\kappa \le 1`$ 과 필요충분조건이 된다.
- 조건이 없어도 식 (25) 와 식 (26) 을 이으면 $`R^2_{\mathrm{all}} \ge \min_j R^2_j`$ 이고, $`R^2_{\mathrm{avgn}}`$ 도 $`\min_j R^2_j`$ 와 $`\max_j R^2_j`$ 사이에 있다.
- Table 4 의 예에서는 $`\kappa = 0.7703`$ 이 $`1 - \lambda = 0.8261`$ 보다 작고 식 (28) 의 우변이 0.0072 로 0 이상이어서, $`R^2_{\mathrm{avgw}} = 0.9392 \gt R^2_{\mathrm{all}} = 0.9348 \gt R^2_{\mathrm{avgn}} = 0.9138`$ 이다.

### 5.7 Wafer Average Form of Lambda and Kappa

식 (17) 의 $`\lambda`$ 와 식 (18) 의 $`\kappa`$ 는 wafer 평균 $`\bar{y}_{i\cdot}`$ 과, wafer 안에서 측정값과 wafer 평균의 차이 $`y_{ij} - \bar{y}_{i\cdot}`$ 로 적을 수 있다. 먼저 $`\sum_j \mathrm{SST}_j`$ 와 $`\sum_j \mathrm{SSE}_j`$ 를 wafer 평균의 몫과 wafer 평균과의 차이의 몫으로 나누면 아래 식 (31) 과 식 (32) 이다.

```math
\sum_{j=1}^{N} \mathrm{SST}_j = N\, \mathrm{SST}_{\mathrm{avgw}} + \sum_{i=1}^{M} \sum_{j=1}^{N} \left[ \left( y_{ij} - \bar{y}_{i\cdot} \right) - \left( \bar{y}_{\cdot j} - \bar{y} \right) \right]^2
\hspace{19em} (31)
```

```math
\sum_{j=1}^{N} \mathrm{SSE}_j = N \sum_{i=1}^{M} \left( \frac{1}{N} \sum_{k=1}^{N} e_{ik} \right)^2 + \sum_{i=1}^{M} \sum_{j=1}^{N} \left( e_{ij} - \frac{1}{N} \sum_{k=1}^{N} e_{ik} \right)^2
\hspace{19em} (32)
```

식 (35) 와 식 (17) 에서 $`\mathrm{SST}_{\mathrm{all}} = \sum_j \mathrm{SST}_j + M \sum_j (\bar{y}_{\cdot j} - \bar{y})^2`$ 이다. 그래서 식 (31) 을 넣으면 식 (17) 의 $`\lambda`$ 가, 식 (31) 과 식 (32) 를 식 (18) 에 넣으면 $`\kappa`$ 가 wafer 평균과 wafer 평균과의 차이로 적힌다.

- 식 (31): 첫 항은 wafer 평균 사이의 변동이고, 둘째 항은 측정값과 wafer 평균의 차이에서 측정 위치 패턴 $`\bar{y}_{\cdot j} - \bar{y}`$ 를 뺀 나머지 변동이다. 측정 위치 패턴은 측정값과 wafer 평균의 차이를 모든 wafer 에서 평균한 값이다.
- 식 (32): 첫 항은 wafer 마다 측정 위치 예측 오차의 평균 $`\frac{1}{N} \sum_k e_{ik}`$ 의 몫이고, 둘째 항은 그 평균과의 차이의 몫이다.
- 식 (17): 식 (31) 을 넣으면 분모 $`\mathrm{SST}_{\mathrm{all}}`$ 은 wafer 평균 사이의 변동, 측정 위치 패턴, 나머지 변동의 세 항으로 나뉘며, $`\lambda`$ 는 그 가운데 측정 위치 패턴의 비율이다. 측정값만 들어가므로 예측값과 무관하다.
- 식 (18): 첫 분수는 wafer 평균의 오차 몫 $`1 - R^2_{\mathrm{avgw}}`$ 이고, 둘째 분수는 식 (31) 을 식 (32) 로 나눈 값이다. 예측 오차가 들어가므로 예측값에 따라 달라진다.
- Table 4 의 예에서 식 (31) 은 2 × 24.6667 + 1.3333 = 50.6667, 식 (32) 는 2 × 1.5 + 1.0 = 4.0 이다. 식 (17) 의 분모는 10.6667 + 50.6667 = 61.3333 이어서 $`\lambda`$ = 10.6667 / 61.3333 = 0.1739 이고, 식 (18) 은 (1.5 / 24.6667) × (50.6667 / 4.0) = 0.7703 으로 Table 4 의 값과 같다.
- 성립 조건은 식 (17) 과 식 (18) 과 같다. 유도는 [B.6](#b6-wafer-average-form-of-lambda-and-kappa) 에 있다.

## 6. Worked Example

Wafer $`M = 3`$ 장, 측정 위치 $`N = 2`$ 곳의 측정값과 예측값은 아래 Table 3 이다.

Table 3. Measured and predicted values

| Wafer i | y_i1 | y_i2 | Predicted y_i1 | Predicted y_i2 | e_i1 | e_i2 | Wafer average | Predicted wafer average | Wafer average error |
| :-----: | :--: | :--: | :------------: | :------------: | :--: | :--: | :-----------: | :---------------------: | :-----------------: |
| 1       | 10   | 12   | 11             | 12             | -1   | 0    | 11            | 11.5                    | -0.5                |
| 2       | 13   | 15   | 13             | 14             | 0    | 1    | 14            | 13.0                    | 1.0                 |
| 3       | 16   | 20   | 15             | 19             | 1    | 1    | 18            | 17.5                    | 0.5                 |

Wafer 평균의 예측값 $`\hat{y}_{iw}`$ 는 측정 위치 예측값과 따로 구한 값이어서, wafer 2 의 $`\hat{y}_{2w} = 13.0`$ 은 측정 위치 예측값 평균 13.5 와 다르다. 전체 평균은 $`\bar{y} = 86/6 = 14.3333`$ 이고, 측정 위치 평균은 $`\bar{y}_{\cdot 1} = 13`$, $`\bar{y}_{\cdot 2} = 15.6667`$ 이다. 식 (6) 부터 식 (18) 로 구한 값은 아래 Table 4 이다.

Table 4. Sums of squares and R2 values

| Quantity                                 | Value                           | R2 or ratio       |
| :--------------------------------------: | :-----------------------------: | :---------------: |
| R2 over all measured values              | SST = 61.3333, SSE = 4          | R2_all = 0.9348   |
| R2 over wafer averages                   | SST = 24.6667, SSE = 1.5        | R2_avgw = 0.9392  |
| Location 1                               | SST = 18, SSE = 2               | R2_1 = 0.8889     |
| Location 2                               | SST = 32.6667, SSE = 2          | R2_2 = 0.9388     |
| R2 average over N locations              | -                               | R2_avgn = 0.9138  |
| SST-weighted R2 average over N locations | SST = 50.6667, SSE = 4          | R2_wavgn = 0.9211 |
| Location pattern                         | 3 × (1.7778 + 1.7778) = 10.6667 | lambda = 0.1739   |
| Measured correlation                     | cross sum = 24 (each order)     | r_y = 0.9474      |
| Error ratio                              | (1.5 / 24.6667) × (50.6667 / 4) | kappa = 0.7703    |

식 (20) 부터 식 (24) 의 항등식에 Table 4 의 값을 넣으면 아래처럼 양변이 같다. 식 (23) 은 $`\mathrm{SST}_1 \ne \mathrm{SST}_2`$ 이어서 쓰지 않는다.

- 식 (20): 좌변 $`1 - 0.9348 = 0.0652`$, 우변 $`(1 - 0.1739)(1 - 0.9211) = 0.8261 \times 0.0789 = 0.0652`$.
- 식 (21): 좌변 $`1 - 0.9392 = 0.0608`$, 우변 $`0.7703 \times 0.0789 = 0.0608`$.
- 식 (22): 가중치는 $`\omega_1 = 18/50.6667 = 0.3553`$ 과 $`\omega_2 = 32.6667/50.6667 = 0.6447`$ 이고, $`0.3553 \times 0.8889 + 0.6447 \times 0.9388 = 0.9211`$.
- 식 (24): 좌변 $`0.0652`$, 우변 $`(0.8261/0.7703) \times 0.0608 = 1.0725 \times 0.0608 = 0.0652`$.

부등식 식 (25) 부터 식 (30) 에 Table 4 의 값을 넣으면 아래처럼 모두 성립한다.

- 식 (25): $`R^2_{\mathrm{all}} = 0.9348`$ 이 $`R^2_{\mathrm{wavgn}} = 0.9211`$ 보다 크고, 차이 $`0.0137`$ 은 $`\lambda (1 - R^2_{\mathrm{wavgn}}) = 0.1739 \times 0.0789 = 0.0137`$ 이다.
- 식 (26), 식 (27): $`R^2_1 = 0.8889 \le R^2_{\mathrm{wavgn}} = 0.9211 \le R^2_2 = 0.9388`$.
- 식 (28): $`\omega_j`$ 에서 $`1/N = 0.5`$ 를 뺀 값은 $`-0.1447`$ 과 $`0.1447`$ 이고, $`-0.1447 \times 0.8889 + 0.1447 \times 0.9388 = 0.0072 \ge 0`$ 이므로 $`R^2_{\mathrm{wavgn}} = 0.9211`$ 이 $`R^2_{\mathrm{avgn}} = 0.9138`$ 보다 $`0.0072`$ 크다.
- 식 (29): $`\kappa = 0.7703 \le 1`$ 이므로 $`R^2_{\mathrm{avgw}} = 0.9392`$ 가 $`R^2_{\mathrm{wavgn}} = 0.9211`$ 이상이다.
- 식 (30): $`\kappa = 0.7703`$ 이 $`1 - \lambda = 0.8261`$ 보다 작으므로 $`R^2_{\mathrm{all}} = 0.9348`$ 이 $`R^2_{\mathrm{avgw}} = 0.9392`$ 보다 작다.

## 7. Application

### 7.1 Reporting R2 of a Wafer Model

측정 위치마다 값을 예측하는 wafer model 은 같은 예측으로 세 R² 를 모두 구할 수 있으므로, 어느 R² 를 보고하는지에 따라 같은 model 의 성능이 다르게 보인다.

- $`R^2_{\mathrm{all}}`$: 측정 위치 평균의 차이 (wafer 안의 고정된 패턴) 를 맞힌 몫 $`\lambda`$ 가 들어 있다. 측정 위치 평균만 맞히고 wafer 사이 변동을 전혀 맞히지 못한 예측도 $`R^2_{\mathrm{all}} = \lambda`$ 이다.
- $`R^2_{\mathrm{avgw}}`$: wafer 평균을 직접 예측한 값으로 구하므로, 측정 위치 예측과 따로 오차 비 $`\kappa`$ 가 작을수록 커진다. Wafer 평균을 관리하는 공정에서 쓴다.
- $`R^2_{\mathrm{avgn}}`$ 과 $`R^2_{\mathrm{wavgn}}`$: wafer 사이 변동을 측정 위치마다 얼마나 맞히는지를 나타낸다. 측정 위치 평균의 차이와 평균할 때 줄어드는 오차가 들어 있지 않다.

### 7.2 Breaking Conditions

- Wafer 마다 측정 위치의 수나 위치가 다르면 식 (21) 이 성립하지 않는다. 식 (39) 에서 전체 평균을 측정 위치 평균의 산술 평균으로 바꾸는 단계에 모든 wafer 의 측정 위치가 같다는 가정이 쓰인다.
- R² 를 측정값과 예측값의 상관계수 제곱으로 구하면 SSE 와 SST 의 비가 아니므로 식 (20) 부터 식 (24) 가 모두 성립하지 않는다.
- 어떤 측정 위치에서 측정값이 wafer 사이에서 같아 $`\mathrm{SST}_j = 0`$ 이면 $`R^2_j`$ 와 $`R^2_{\mathrm{avgn}}`$ 이 정의되지 않는다.

---

## Appendix A. Terminology

- **dot notation**: 아래첨자 하나를 점 ($`\cdot`$) 으로 바꾸어 그 첨자에 대해 모두 더했음을 나타내는 표기. 예: $`y_{i\cdot} = \sum_{j} y_{ij}`$.
- **error ratio**: 식 (18) 의 $`\kappa`$. wafer 평균 예측의 오차 몫을 측정 위치 예측의 오차 몫으로 나눈 값.
- **location pattern fraction**: 식 (17) 의 $`\lambda`$. 측정 위치 평균이 서로 달라 생기는 제곱합이 $`\mathrm{SST}_{\mathrm{all}}`$ 에서 차지하는 비율.
- **measurement location**: wafer 위에서 측정값을 얻는 지점. 모든 wafer 에서 같은 번호 $`j`$ 는 같은 지점이다.
- **R2 (coefficient of determination)**: $`1 - \mathrm{SSE}/\mathrm{SST}`$. 예측 오차 제곱합이 평균에서 벗어난 제곱합의 몇 분의 일인지를 1 에서 뺀 값.
- **SST-weighted R2 average over N locations**: 식 (16) 의 $`R^2_{\mathrm{wavgn}}`$. 측정 위치 R² $`R^2_j`$ 를 $`\mathrm{SST}_j`$ 에 비례하는 가중치로 평균한 값이며, 측정 위치 $`N`$ 곳의 SSE 합과 SST 합으로 구한 R² 와 같다.
- **sum of squared errors (SSE)**: 예측 오차를 제곱해 더한 값.
- **total sum of squares (SST)**: 측정값과 그 평균의 차이를 제곱해 더한 값.
- **wafer average**: wafer 한 장의 측정값 $`N`$ 개를 산술 평균한 값.

## Appendix B. Derivation

### B.1 R2 over All Measured Values Form

식 (20) 은 $`\mathrm{SST}_{\mathrm{all}}`$ 을 측정 위치마다 나누어 증명한다. 쓰는 정의 식은 아래 식 (5), 식 (6), 식 (7), 식 (8), 식 (12), 식 (13), 식 (16), 식 (17) 이다.

```math
\bar{y}_{\cdot j} = \frac{1}{M} \sum_{i=1}^{M} y_{ij}
\hspace{19em} (5)
```

```math
\mathrm{SST}_{\mathrm{all}} = \sum_{i=1}^{M} \sum_{j=1}^{N} \left( y_{ij} - \bar{y} \right)^2
\hspace{19em} (6)
```

```math
\mathrm{SSE}_{\mathrm{all}} = \sum_{i=1}^{M} \sum_{j=1}^{N} e_{ij}^2
\hspace{19em} (7)
```

```math
R^2_{\mathrm{all}} = 1 - \frac{\mathrm{SSE}_{\mathrm{all}}}{\mathrm{SST}_{\mathrm{all}}}
\hspace{19em} (8)
```

```math
\mathrm{SST}_j = \sum_{i=1}^{M} \left( y_{ij} - \bar{y}_{\cdot j} \right)^2
\hspace{19em} (12)
```

```math
\mathrm{SSE}_j = \sum_{i=1}^{M} e_{ij}^2
\hspace{19em} (13)
```

```math
R^2_{\mathrm{wavgn}} = 1 - \frac{\sum_{j=1}^{N} \mathrm{SSE}_j}{\sum_{j=1}^{N} \mathrm{SST}_j}
\hspace{19em} (16)
```

```math
\lambda = \frac{M \sum_{j=1}^{N} \left( \bar{y}_{\cdot j} - \bar{y} \right)^2}{\mathrm{SST}_{\mathrm{all}}}
\hspace{19em} (17)
```

```math
\mathrm{SST}_{\mathrm{all}} = \sum_{j=1}^{N} \sum_{i=1}^{M} \left[ \left( y_{ij} - \bar{y}_{\cdot j} \right) + \left( \bar{y}_{\cdot j} - \bar{y} \right) \right]^2
\hspace{19em} (33)
```

- 식 (33): 식 (6) 의 합의 순서를 바꾸고, $`y_{ij} - \bar{y}`$ 에 $`\bar{y}_{\cdot j}`$ 를 빼고 더했다.

```math
\mathrm{SST}_{\mathrm{all}} = \sum_{j=1}^{N} \mathrm{SST}_j + 2 \sum_{j=1}^{N} \left( \bar{y}_{\cdot j} - \bar{y} \right) \sum_{i=1}^{M} \left( y_{ij} - \bar{y}_{\cdot j} \right) + M \sum_{j=1}^{N} \left( \bar{y}_{\cdot j} - \bar{y} \right)^2
\hspace{19em} (34)
```

- 식 (34): 제곱을 전개하고, 첫 항에 식 (12) 를 썼다. $`\bar{y}_{\cdot j} - \bar{y}`$ 는 $`i`$ 에 따라 바뀌지 않으므로 $`i`$ 에 대한 합 밖으로 나온다.

```math
\mathrm{SST}_{\mathrm{all}} = \sum_{j=1}^{N} \mathrm{SST}_j + \lambda\, \mathrm{SST}_{\mathrm{all}}
\hspace{19em} (35)
```

- 식 (35): 식 (5) 에 따라 $`\sum_i (y_{ij} - \bar{y}_{\cdot j}) = M \bar{y}_{\cdot j} - M \bar{y}_{\cdot j} = 0`$ 이므로 가운데 항이 0 이다. 마지막 항은 식 (17) 로 $`\lambda\, \mathrm{SST}_{\mathrm{all}}`$ 이다.

```math
\mathrm{SSE}_{\mathrm{all}} = \sum_{j=1}^{N} \mathrm{SSE}_j
\hspace{19em} (36)
```

- 식 (36): 식 (7) 의 합의 순서를 바꾸고 식 (13) 을 썼다.

```math
1 - R^2_{\mathrm{all}} = \frac{\sum_{j} \mathrm{SST}_j}{\mathrm{SST}_{\mathrm{all}}} \cdot \frac{\sum_{j} \mathrm{SSE}_j}{\sum_{j} \mathrm{SST}_j}
\hspace{19em} (37)
```

- 식 (37): 식 (8) 에 식 (36) 을 넣고, 분자와 분모에 $`\sum_j \mathrm{SST}_j \gt 0`$ 을 곱했다.

식 (35) 에서 $`\sum_j \mathrm{SST}_j / \mathrm{SST}_{\mathrm{all}} = 1 - \lambda`$ 이고, 식 (16) 에서 $`\sum_j \mathrm{SSE}_j / \sum_j \mathrm{SST}_j = 1 - R^2_{\mathrm{wavgn}}`$ 이다. 따라서 식 (20) 이 성립한다.

```math
1 - R^2_{\mathrm{all}} = (1 - \lambda) \left( 1 - R^2_{\mathrm{wavgn}} \right)
\hspace{19em} (20)
```

### B.2 R2 over Wafer Averages Form

식 (21) 은 식 (11) 의 오차 몫을 식 (18) 과 식 (16) 의 곱으로 나누어 증명한다. 식 (18) 의 $`\kappa`$ 를 $`\bar{r}_y`$ 로 적은 식 (43) 은 $`\mathrm{SST}_{\mathrm{avgw}}`$ 를 측정 위치마다의 제곱합과 교차합으로 나누어 구한다. 쓰는 정의 식은 아래 식 (2), 식 (3), 식 (9), 식 (11), 식 (18) 과 이 절에서 정의하는 식 (38) 이고, 식 (5), 식 (12), 식 (16) 은 B.1 에 있다.

```math
\bar{y} = \frac{1}{MN} \sum_{i=1}^{M} \sum_{j=1}^{N} y_{ij}
\hspace{19em} (2)
```

```math
\bar{y}_{i\cdot} = \frac{1}{N} \sum_{j=1}^{N} y_{ij}
\hspace{19em} (3)
```

```math
\mathrm{SST}_{\mathrm{avgw}} = \sum_{i=1}^{M} \left( \bar{y}_{i\cdot} - \bar{y} \right)^2
\hspace{19em} (9)
```

```math
R^2_{\mathrm{avgw}} = 1 - \frac{\mathrm{SSE}_{\mathrm{avgw}}}{\mathrm{SST}_{\mathrm{avgw}}}
\hspace{19em} (11)
```

> **Definition: Between-location correlation**<br>
> $`\bar{r}_y`$ 는 서로 다른 두 측정 위치에서 측정값이 이루는 상관이다. 측정 위치 평균에서 벗어난 양의 곱을 서로 다른 측정 위치의 모든 순서쌍과 wafer $`M`$ 장에서 더한 값을, $`N-1`$ 과 측정 위치 SST 의 합의 곱으로 나눈 값이다.

$`\bar{r}_y`$ 의 정의 식은 아래 식 (38) 이다.

```math
\bar{r}_y = \frac{\sum_{j \ne k} \sum_{i=1}^{M} \left( y_{ij} - \bar{y}_{\cdot j} \right) \left( y_{ik} - \bar{y}_{\cdot k} \right)}{(N-1) \sum_{j=1}^{N} \mathrm{SST}_j}
\hspace{19em} (38)
```

- $`\mathrm{SST}_j`$ 가 모두 같으면 $`\bar{r}_y`$ 는 측정 위치 두 곳의 측정값 상관계수를 모든 쌍에서 산술 평균한 값이다.
- $`\bar{r}_y = 1`$ 은 wafer 마다 측정값이 모든 측정 위치에서 같은 양만큼 움직이는 경우이다.

```math
\kappa = \frac{\mathrm{SSE}_{\mathrm{avgw}}}{\mathrm{SST}_{\mathrm{avgw}}} \cdot \frac{\sum_{j=1}^{N} \mathrm{SST}_j}{\sum_{j=1}^{N} \mathrm{SSE}_j}
\hspace{19em} (18)
```

```math
\bar{y}_{i\cdot} - \bar{y} = \frac{1}{N} \sum_{j=1}^{N} \left( y_{ij} - \bar{y}_{\cdot j} \right)
\hspace{19em} (39)
```

- 식 (39): 식 (3) 으로 $`\bar{y}_{i\cdot} = \frac{1}{N} \sum_j y_{ij}`$ 이고, 식 (2) 와 식 (5) 로 $`\bar{y} = \frac{1}{N} \sum_j \bar{y}_{\cdot j}`$ 이다.

```math
\mathrm{SST}_{\mathrm{avgw}} = \frac{1}{N^2} \sum_{i=1}^{M} \left[ \sum_{j=1}^{N} \left( y_{ij} - \bar{y}_{\cdot j} \right) \right]^2
\hspace{19em} (40)
```

- 식 (40): 식 (9) 에 식 (39) 를 넣었다.

```math
\mathrm{SST}_{\mathrm{avgw}} = \frac{1}{N^2} \left[ \sum_{j=1}^{N} \mathrm{SST}_j + \sum_{j \ne k} \sum_{i=1}^{M} \left( y_{ij} - \bar{y}_{\cdot j} \right) \left( y_{ik} - \bar{y}_{\cdot k} \right) \right]
\hspace{19em} (41)
```

- 식 (41): 합의 제곱 $`(\sum_j a_j)^2 = \sum_j a_j^2 + \sum_{j \ne k} a_j a_k`$ 를 쓰고, $`i`$ 에 대한 합을 안으로 넣었다. 첫 항은 식 (12) 이다.

```math
\mathrm{SST}_{\mathrm{avgw}} = \frac{1}{N^2} \left[ 1 + (N-1)\, \bar{r}_y \right] \sum_{j=1}^{N} \mathrm{SST}_j
\hspace{19em} (42)
```

- 식 (42): 식 (38) 로 교차합은 $`(N-1)\, \bar{r}_y \sum_j \mathrm{SST}_j`$ 이다.

```math
\kappa = \frac{N^2\, \mathrm{SSE}_{\mathrm{avgw}}}{\left[ 1 + (N-1)\, \bar{r}_y \right] \sum_{j=1}^{N} \mathrm{SSE}_j}
\hspace{19em} (43)
```

- 식 (43): 식 (18) 의 $`\mathrm{SST}_{\mathrm{avgw}}`$ 에 식 (42) 를 넣었다.

```math
1 - R^2_{\mathrm{avgw}} = \frac{\mathrm{SSE}_{\mathrm{avgw}}}{\mathrm{SST}_{\mathrm{avgw}}} \cdot \frac{\sum_{j=1}^{N} \mathrm{SST}_j}{\sum_{j=1}^{N} \mathrm{SSE}_j} \cdot \frac{\sum_{j=1}^{N} \mathrm{SSE}_j}{\sum_{j=1}^{N} \mathrm{SST}_j}
\hspace{19em} (44)
```

- 식 (44): 식 (11) 의 $`\mathrm{SSE}_{\mathrm{avgw}} / \mathrm{SST}_{\mathrm{avgw}}`$ 에 $`\sum_j \mathrm{SST}_j / \sum_j \mathrm{SSE}_j`$ 와 그 역수를 곱했다.

식 (44) 의 앞 두 분수의 곱은 식 (18) 의 $`\kappa`$ 이고, 마지막 분수는 식 (16) 의 $`1 - R^2_{\mathrm{wavgn}}`$ 이다. 따라서 식 (21) 이 성립한다.

```math
1 - R^2_{\mathrm{avgw}} = \kappa \left( 1 - R^2_{\mathrm{wavgn}} \right)
\hspace{19em} (21)
```

### B.3 R2 Average over N Locations Form

식 (22) 는 식 (16) 의 비를 측정 위치마다의 비로 나누어 증명한다. 쓰는 정의 식은 아래 식 (14) 와 식 (19) 이고, 식 (16) 은 B.1 에 있다.

```math
R^2_j = 1 - \frac{\mathrm{SSE}_j}{\mathrm{SST}_j}
\hspace{19em} (14)
```

```math
\omega_j = \frac{\mathrm{SST}_j}{\sum_{k=1}^{N} \mathrm{SST}_k}
\hspace{19em} (19)
```

```math
1 - R^2_{\mathrm{wavgn}} = \sum_{j=1}^{N} \omega_j \cdot \frac{\mathrm{SSE}_j}{\mathrm{SST}_j}
\hspace{19em} (45)
```

- 식 (45): 식 (16) 의 분자의 항마다 $`\mathrm{SST}_j \gt 0`$ 을 곱하고 나눈 뒤, 식 (19) 의 $`\omega_j`$ 로 적었다.

```math
1 - R^2_{\mathrm{wavgn}} = \sum_{j=1}^{N} \omega_j \left( 1 - R^2_j \right)
\hspace{19em} (46)
```

- 식 (46): 식 (14) 로 $`\mathrm{SSE}_j / \mathrm{SST}_j = 1 - R^2_j`$ 이다.

식 (46) 의 가중치 $`\omega_j`$ 는 $`j`$ 에 대해 더하면 1 이므로, 1 을 빼고 부호를 바꾸면 식 (22) 가 성립한다.

```math
R^2_{\mathrm{wavgn}} = \sum_{j=1}^{N} \omega_j R^2_j
\hspace{19em} (22)
```

$`\mathrm{SST}_j`$ 가 모두 같으면 가중치가 모두 $`1/N`$ 이 되고, 식 (15) 에 따라 식 (23) 이 성립한다.

```math
R^2_{\mathrm{avgn}} = \frac{1}{N} \sum_{j=1}^{N} R^2_j
\hspace{19em} (15)
```

```math
R^2_{\mathrm{wavgn}} = R^2_{\mathrm{avgn}}
\hspace{19em} (23)
```

### B.4 Relation without the SST-weighted R2 Average over N Locations

식 (24) 는 B.1 의 식 (20) 과 B.2 의 식 (21) 에서 $`1 - R^2_{\mathrm{wavgn}}`$ 을 지워 증명한다.

```math
1 - R^2_{\mathrm{wavgn}} = \frac{1 - R^2_{\mathrm{avgw}}}{\kappa}
\hspace{19em} (47)
```

- 식 (47): 식 (21) 의 양변을 $`\kappa`$ 로 나누었다. $`\mathrm{SSE}_{\mathrm{avgw}} \gt 0`$ 이고 $`\mathrm{SST}_{\mathrm{avgw}} \gt 0`$ 이면 식 (18) 의 네 제곱합이 양수이므로 $`\kappa \gt 0`$ 이다.

식 (47) 을 식 (20) 에 넣으면 식 (24) 가 성립한다.

```math
1 - R^2_{\mathrm{all}} = \frac{1 - \lambda}{\kappa} \left( 1 - R^2_{\mathrm{avgw}} \right)
\hspace{19em} (24)
```

### B.5 Inequalities

식 (25) 부터 식 (30) 은 항등식의 두 변의 차이를 구하고 그 부호를 보아 증명한다. 쓰는 식은 식 (15) 부터 식 (17), 식 (18) 부터 식 (22), 식 (24) 이며, 모두 B.1 부터 B.4 에 있다.

식 (16) 의 $`\mathrm{SSE}_j`$ 와 $`\mathrm{SST}_j`$ 는 제곱합이므로 $`1 - R^2_{\mathrm{wavgn}} \ge 0`$ 이고, 식 (17) 의 분자와 분모도 제곱합이므로 $`\lambda \ge 0`$ 이다. 아래 증명은 이 두 부호를 쓴다.

#### R2 over All Measured Values and SST-weighted R2 Average over N Locations

```math
R^2_{\mathrm{all}} - R^2_{\mathrm{wavgn}} = \left( 1 - R^2_{\mathrm{wavgn}} \right) - \left( 1 - R^2_{\mathrm{all}} \right)
\hspace{19em} (48)
```

- 식 (48): 좌변에 1 을 더하고 뺐다.

```math
R^2_{\mathrm{all}} - R^2_{\mathrm{wavgn}} = \lambda \left( 1 - R^2_{\mathrm{wavgn}} \right)
\hspace{19em} (49)
```

- 식 (49): 식 (48) 의 둘째 괄호에 식 (20) 을 넣었다.

식 (49) 의 우변은 $`\lambda \ge 0`$ 과 $`1 - R^2_{\mathrm{wavgn}} \ge 0`$ 의 곱이므로 0 이상이다. 따라서 식 (25) 가 성립한다.

```math
R^2_{\mathrm{all}} \ge R^2_{\mathrm{wavgn}}
\hspace{19em} (25)
```

#### Range of the SST-weighted R2 Average over N Locations

```math
R^2_{\mathrm{wavgn}} - \min_{k} R^2_k = \sum_{j=1}^{N} \omega_j \left( R^2_j - \min_{k} R^2_k \right)
\hspace{19em} (50)
```

- 식 (50): 식 (22) 의 가중치를 $`j`$ 에 대해 더하면 1 이므로, $`\min_k R^2_k`$ 에 가중치의 합을 곱해 뺐다.

```math
\max_{k} R^2_k - R^2_{\mathrm{wavgn}} = \sum_{j=1}^{N} \omega_j \left( \max_{k} R^2_k - R^2_j \right)
\hspace{19em} (51)
```

- 식 (51): 식 (50) 과 같은 방법으로 $`\max_k R^2_k`$ 에서 식 (22) 를 뺐다.

식 (50) 과 식 (51) 에서 가중치는 $`\mathrm{SST}_j \gt 0`$ 이므로 양수이고, 괄호는 최솟값과 최댓값의 정의로 0 이상이다. 두 우변이 0 이상이므로 식 (26) 과 식 (27) 이 성립한다.

```math
R^2_{\mathrm{wavgn}} \ge \min_{j} R^2_j
\hspace{19em} (26)
```

```math
R^2_{\mathrm{wavgn}} \le \max_{j} R^2_j
\hspace{19em} (27)
```

#### SST-weighted and Unweighted R2 Averages over N Locations

```math
R^2_{\mathrm{wavgn}} - R^2_{\mathrm{avgn}} = \sum_{j=1}^{N} \left( \omega_j - \frac{1}{N} \right) R^2_j
\hspace{19em} (52)
```

- 식 (52): 식 (22) 에서 식 (15) 를 빼고 같은 $`R^2_j`$ 의 계수를 묶었다.

$`R^2_{\mathrm{wavgn}} - R^2_{\mathrm{avgn}} \ge 0`$ 은 식 (52) 의 우변이 0 이상인 것과 같으므로 식 (28) 이 성립한다.

```math
R^2_{\mathrm{wavgn}} \ge R^2_{\mathrm{avgn}} \iff \sum_{j=1}^{N} \left( \omega_j - \frac{1}{N} \right) R^2_j \ge 0
\hspace{19em} (28)
```

#### R2 over Wafer Averages and SST-weighted R2 Average over N Locations

```math
R^2_{\mathrm{avgw}} - R^2_{\mathrm{wavgn}} = (1 - \kappa) \left( 1 - R^2_{\mathrm{wavgn}} \right)
\hspace{19em} (53)
```

- 식 (53): 식 (48) 과 같이 1 을 더하고 뺀 뒤 $`1 - R^2_{\mathrm{avgw}}`$ 에 식 (21) 을 넣었다.

$`\sum_j \mathrm{SSE}_j \gt 0`$ 이면 $`1 - R^2_{\mathrm{wavgn}} \gt 0`$ 이므로, 식 (53) 의 좌변이 0 이상인 것은 $`\kappa \le 1`$ 과 같다. 따라서 식 (29) 가 성립한다.

```math
R^2_{\mathrm{avgw}} \ge R^2_{\mathrm{wavgn}} \iff \kappa \le 1
\hspace{19em} (29)
```

#### R2 over All Measured Values and R2 over Wafer Averages

```math
R^2_{\mathrm{all}} - R^2_{\mathrm{avgw}} = \left( 1 - R^2_{\mathrm{avgw}} \right) \left( 1 - \frac{1 - \lambda}{\kappa} \right)
\hspace{19em} (54)
```

- 식 (54): 식 (48) 과 같이 1 을 더하고 뺀 뒤 $`1 - R^2_{\mathrm{all}}`$ 에 식 (24) 를 넣었다.

$`\mathrm{SSE}_{\mathrm{avgw}} \gt 0`$ 이면 $`1 - R^2_{\mathrm{avgw}} \gt 0`$ 이고 $`\kappa \gt 0`$ 이다. 그래서 식 (54) 의 좌변이 0 이상인 것은 $`(1 - \lambda)/\kappa \le 1`$, 곧 $`\kappa \ge 1 - \lambda`$ 와 같고, 식 (30) 이 성립한다.

```math
R^2_{\mathrm{all}} \ge R^2_{\mathrm{avgw}} \iff \kappa \ge 1 - \lambda
\hspace{19em} (30)
```

### B.6 Wafer Average Form of Lambda and Kappa

식 (31) 과 식 (32) 는 측정값과 예측 오차를 wafer 평균과 그 차이로 나누어 증명한다.

```math
y_{ij} - \bar{y}_{\cdot j} = \left( \bar{y}_{i\cdot} - \bar{y} \right) + \left[ \left( y_{ij} - \bar{y}_{i\cdot} \right) - \left( \bar{y}_{\cdot j} - \bar{y} \right) \right]
\hspace{19em} (55)
```

- 식 (55): 우변의 두 항을 더하면 $`\bar{y}_{i\cdot}`$ 과 $`\bar{y}`$ 가 지워진다.

식 (12) 를 측정 위치에 대해 더하고 식 (55) 를 넣어 제곱을 전개한다. 첫 항은 측정 위치에 따라 바뀌지 않으므로 그 합은 식 (9) 의 $`N`$ 배이다. 식 (2), 식 (3), 식 (5) 로 $`\sum_j (y_{ij} - \bar{y}_{i\cdot}) = 0`$ 이고 $`\sum_j (\bar{y}_{\cdot j} - \bar{y}) = 0`$ 이므로 교차항은 0 이다. 따라서 식 (31) 이 성립한다.

식 (13) 을 측정 위치에 대해 더하고, $`e_{ij}`$ 에 $`\frac{1}{N} \sum_k e_{ik}`$ 를 빼고 더한 뒤 제곱을 전개한다. 이 차이를 측정 위치에 대해 더하면 0 이므로 교차항은 0 이고, 따라서 식 (32) 가 성립한다.

## Appendix C. Intuitive Meaning

SST 가중 평균 $`R^2_{\mathrm{wavgn}}`$ 과 가중하지 않은 산술 평균 $`R^2_{\mathrm{avgn}}`$ 을 측정 위치 두 곳의 예로 견주면 아래 Fig 1 이다.

<img src="r2-aggregation-ko_fig/fig1-weighting.png" width="800" style="max-width: 100%;" alt="Fig 1">

Fig 1. Plain and SST-weighted averages of location R2

- 원 하나가 측정 위치 하나이다. 바깥 원의 넓이는 $`\mathrm{SST}_j`$, 안쪽 주황 원의 넓이는 $`\mathrm{SSE}_j`$ 이고, 파란 고리가 바깥 원에서 차지하는 몫이 $`R^2_j`$ 이다. 측정 위치 1 은 $`\mathrm{SST}_1 = 100`$, $`\mathrm{SSE}_1 = 10`$ 이고 측정 위치 2 는 $`\mathrm{SST}_2 = 1`$, $`\mathrm{SSE}_2 = 0.5`$ 이며, Table 3 와 다른 예시이다.
- (a): 두 측정 위치를 같은 크기로 그렸다. 측정 위치 1 의 두 화살표는 중심에서 바깥 원과 안쪽 원의 가장자리까지 그은 반지름이며, 두 원의 넓이가 각각 $`\mathrm{SST}_1`$ 과 $`\mathrm{SSE}_1`$ 이다. $`R^2_{\mathrm{avgn}}`$ 은 이처럼 측정 위치마다 가중치 $`1/N`$ 을 주어 $`(0.90 + 0.50)/2 = 0.70`$ 이다.
- (b): 두 측정 위치를 $`\mathrm{SST}_j`$ 그대로의 크기로 그린 뒤 한 원으로 합쳤다. $`R^2_{\mathrm{wavgn}}`$ 은 합친 원에서 파란 고리의 몫이며 $`1 - 10.5/101 = 0.896`$ 이다.

두 값의 차이는 측정 위치 2 를 얼마나 크게 세는지에서 생긴다. 측정 위치 2 는 wafer 사이에서 측정값이 거의 변하지 않아 ($`\mathrm{SST}_2 = 1`$) 예측이 맞힐 변동이 거의 없다. (a) 는 이 측정 위치를 측정 위치 1 과 같은 크기의 원으로 부풀려서, 크기가 0.5 인 오차가 원의 절반을 차지하게 만든다. (b) 에서는 측정 위치 2 가 제 크기인 작은 점으로 남고, 합친 원의 파란 몫은 측정 위치 1 의 몫과 거의 같다.

SST 가중 평균이 의미를 갖는 까닭은 아래 세 가지이다.

- **맞힐 변동이 큰 측정 위치에 무게를 둔다.** 측정 위치마다 담긴 변동을 한 그릇에 부어 합치면, 그릇의 비율은 많이 부은 측정 위치가 더 많이 정한다. 그 무게가 아래 식 (22) 의 가중치 $`\omega_j`$ 이며, 측정 위치 $`j`$ 가 측정 위치 $`N`$ 곳의 wafer 사이 변동 전체에서 차지하는 몫이다. Fig 1 의 예에서는 0.99 대 0.01 이다.
- **변동이 거의 없는 측정 위치의 $`R^2_j`$ 에 흔들리지 않는다.** $`\mathrm{SST}_j`$ 가 작으면 $`R^2_j`$ 는 작은 오차에도 크게 바뀐다. Fig 1 의 예에서 $`\mathrm{SSE}_2`$ 를 0.5 에서 2 로 늘리면 $`R^2_2`$ 는 0.50 에서 -1.00 으로, $`R^2_{\mathrm{avgn}}`$ 은 0.70 에서 -0.05 로 떨어지지만, $`R^2_{\mathrm{wavgn}}`$ 은 0.896 에서 0.881 로 거의 그대로이다.
- **하나의 비율로 읽힌다.** $`R^2_{\mathrm{wavgn}} = 1 - \sum_j \mathrm{SSE}_j / \sum_j \mathrm{SST}_j`$ 는 측정 위치 $`N`$ 곳의 wafer 사이 변동 전체 가운데 예측이 맞힌 몫이다. 그래서 [5.1](#51-r2-over-all-measured-values-form) 과 [5.2](#52-r2-over-wafer-averages-form) 의 항등식에서 $`R^2_{\mathrm{all}}`$, $`R^2_{\mathrm{avgw}}`$ 와 $`\lambda`$, $`\kappa`$ 만으로 이어진다. $`R^2_{\mathrm{avgn}}`$ 은 $`\mathrm{SST}_j`$ 가 모두 같을 때만 이 값과 같아진다.

```math
R^2_{\mathrm{wavgn}} = \sum_{j=1}^{N} \omega_j R^2_j
\hspace{19em} (22)
```

가중치를 $`\mathrm{SST}_j`$ 가 아니라 $`\mathrm{SSE}_j`$ 로 두지 않는 까닭은 아래 두 가지이다.

- **비율 하나로 줄어들지 않는다.** $`R^2_j = 1 - \mathrm{SSE}_j / \mathrm{SST}_j`$ 의 분모가 $`\mathrm{SST}_j`$ 이므로, $`\mathrm{SST}_j`$ 에 비례하는 가중치를 곱하면 분모가 약분되어 가중 평균이 $`1 - \sum_j \mathrm{SSE}_j / \sum_j \mathrm{SST}_j`$ 하나로 줄어든다. $`\mathrm{SSE}_j`$ 에 비례하는 가중치를 곱하면 분모가 남아 그런 비율이 나오지 않고, 예측 오차가 큰 측정 위치일수록 무게가 커진다. Fig 1 의 예에서 SSE 가중 평균은 $`(10 \times 0.90 + 0.5 \times 0.50)/10.5 = 0.881`$ 이며, 어떤 변동의 몇 분의 일인지로 읽히지 않는다.
- **가중치가 예측 model 에 따라 바뀐다.** $`\mathrm{SST}_j`$ 는 측정값만으로 정해져 어떤 예측 model 을 쓰든 같지만, $`\mathrm{SSE}_j`$ 는 예측값에 따라 바뀐다. Fig 1 의 예에서 $`\mathrm{SSE}_2`$ 를 0.5 에서 2 로 바꾸면 SSE 가중치는 0.952 대 0.048 에서 0.833 대 0.167 로 바뀌고, SSE 가중 평균은 0.881 에서 0.583 으로 떨어진다. 같은 측정값에서 두 예측 model 을 견줄 때 가중치까지 함께 바뀌면, R² 의 차이가 예측이 나아진 몫인지 가중치가 바뀐 몫인지 가릴 수 없다.

Fig 1 은 같은 folder 의 [src/r2_wavgn_figure.py](src/r2_wavgn_figure.py) 로 그렸다. 이 script 는 위 예시의 $`\mathrm{SST}_j`$ 와 $`\mathrm{SSE}_j`$ 로 $`R^2_j`$, $`R^2_{\mathrm{avgn}}`$, $`R^2_{\mathrm{wavgn}}`$ 을 계산해 그림을 저장하며, 문서 folder 에서 `python3 src/r2_wavgn_figure.py` 로 실행한다.
