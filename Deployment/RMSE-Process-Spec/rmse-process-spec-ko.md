# Process Spec from Prediction RMSE
Version 18 | Created: 2026-11-16 | Updated: 2026-11-16 05:08 CST

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Principle](#3-principle)
    - [3.1 Notation](#31-notation)
    - [3.2 RMSE](#32-rmse)
    - [3.3 Tolerance](#33-tolerance)
    - [3.4 Guard Band](#34-guard-band)
    - [3.5 Wafer Uniformity Spec](#35-wafer-uniformity-spec)
- [4. Worked Example](#4-worked-example)
    - [4.1 RMSE of the Data](#41-rmse-of-the-data)
    - [4.2 Thickness Spec](#42-thickness-spec)
    - [4.3 Wafer Uniformity Spec](#43-wafer-uniformity-spec)
    - [4.4 Calculation Script](#44-calculation-script)
- [5. Application](#5-application)
    - [5.1 Conditions](#51-conditions)
    - [5.2 Breaking Conditions](#52-breaking-conditions)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. RMSE and R2_all](#appendix-b-rmse-and-r2_all)

## 1. Purpose

**Problem Statement**: 측정하지 않은 wafer 를 예측값으로 판정하면 예측 오차가 공정 산포에 더해지는데, 그 오차를 반영해 thickness spec 과 wafer uniformity spec 을 구하는 방법이 없다.

**Goal**: 예측값의 RMSE 로 아래 세 가지를 새로 정한다.

1. 예측값으로 판정할 수 있는 thickness spec 을 target ± tolerance 로 제시하고, 그 tolerance 의 가장 작은 값과 근거
2. Wafer uniformity (1σ/mean) 의 upper spec limit
3. 예측값을 판정할 acceptance limit

**Non-Goal**:

- 이미 정해진 품질 spec 에 대해 예측값으로 합격을 판정하는 방법은 다루지 않는다. 이 문서는 RMSE 를 반영해 spec 을 새로 정한다.
- 예측 model 을 만들거나 고치는 방법은 다루지 않는다.
- 제품 성능에서 정하는 design spec 은 다루지 않는다. 이 문서의 spec 은 공정 산포와 예측 오차로 정하는 공정 spec 이다.
- 요구 공정능력지수 (process capability index), tolerance 안에 들어가야 하는 RMSE 의 최소 개수 $`n_\sigma`$, guard band 계수의 값을 정하는 방법은 다루지 않는다. 세 값은 입력으로 받는다.

## 2. Summary

Tolerance 는 공정 요건 $`h_P`$ 와 예측 요건 $`h_E = n_\sigma\, \mathrm{RMSE}`$ 가운데 큰 값 $`h = \max(h_P, h_E)`$ 이고, spec 은 target ± $`h`$, acceptance limit 은 target ± $`(h - k\, \mathrm{RMSE})`$ 이며, 비 $`\rho = h_E / h_P`$ 가 1 보다 크면 예측 요건이 tolerance 를 정한다.
Wafer 44 장, 측정 위치 13 곳의 데이터에서 thickness spec 은 600 ± 148.3056 nm ($`\rho`$ = 0.8124, 공정 요건이 정함), wafer uniformity spec 은 2.4962 % + 7.7251 % ($`\rho`$ = 1.9417, 예측 요건이 정함) 이다.
RMSE 에는 예측 오차의 bias 가 들어 있어, $`n_\sigma`$ 와 $`k`$ 는 설계 규칙이며 정규분포의 coverage 를 보장하지 않는다.

## 3. Principle

### 3.1 Notation

이 문서의 기호는 아래 Table 1 이다.

Table 1. Symbols

| Symbol                 | Meaning                                                          | Shape   | Assumption                 |
| :--------------------: | :--------------------------------------------------------------: | :-----: | :------------------------: |
| $`M`$                  | wafer 수                                                         | integer | $`M \ge 2`$                |
| $`N`$                  | wafer 한 장의 측정 위치 수                                       | integer | $`N \ge 2`$                |
| $`T`$                  | 전체 측정값의 개수 $`MN`$                                        | integer | -                          |
| $`y_{ij}`$             | wafer $`i`$ 의 측정 위치 $`j`$ 에서 측정값                       | scalar  | -                          |
| $`\hat{y}_{ij}`$       | wafer $`i`$ 의 측정 위치 $`j`$ 에서 예측값                       | scalar  | -                          |
| $`e_{ij}`$             | 예측 오차 $`y_{ij} - \hat{y}_{ij}`$                              | scalar  | -                          |
| $`b_e`$                | 예측 오차 $`T`$ 개의 평균 (bias)                                 | scalar  | -                          |
| $`\sigma_e`$           | 예측 오차 $`T`$ 개의 표준편차 (분모 $`T`$)                       | scalar  | -                          |
| $`\bar{y}`$            | 측정값 $`T`$ 개의 평균                                           | scalar  | -                          |
| $`\bar{y}_{i\cdot}`$   | wafer $`i`$ 의 측정값 $`N`$ 개의 평균                            | scalar  | $`\bar{y}_{i\cdot} \gt 0`$ |
| $`\sigma_y`$           | 측정값 $`T`$ 개의 표준편차 (분모 $`T`$)                          | scalar  | $`\sigma_y \gt 0`$         |
| $`\mathrm{RMSE}`$      | 측정 위치 예측값의 root mean squared error                       | scalar  | -                          |
| $`R^2_{\mathrm{all}}`$ | 측정값 $`T`$ 개로 구한 R²                                        | scalar  | -                          |
| $`s_i`$                | wafer $`i`$ 의 측정값 $`N`$ 개의 표준편차 (분모 $`N-1`$)         | scalar  | -                          |
| $`U_i`$                | wafer $`i`$ 의 wafer uniformity                                  | scalar  | -                          |
| $`\hat{U}_i`$          | wafer $`i`$ 의 wafer uniformity 예측값                           | scalar  | -                          |
| $`\bar{U}`$            | $`U_i`$ $`M`$ 개의 평균                                          | scalar  | -                          |
| $`\sigma_U`$           | $`U_i`$ $`M`$ 개의 표준편차 (분모 $`M`$)                         | scalar  | $`\sigma_U \gt 0`$         |
| $`\mathrm{RMSE}_U`$    | wafer uniformity 예측값의 RMSE                                   | scalar  | -                          |
| $`\mathrm{USL}`$       | upper spec limit                                                 | scalar  | -                          |
| $`\mathrm{LSL}`$       | lower spec limit                                                 | scalar  | -                          |
| $`y_0`$                | thickness 의 target                                              | scalar  | -                          |
| $`h`$                  | thickness 의 tolerance. target 에서 spec limit 까지의 거리       | scalar  | $`h \gt 0`$                |
| $`h_P`$                | thickness 의 공정 요건. 공정능력으로 정한 tolerance 의 하한      | scalar  | -                          |
| $`h_E`$                | thickness 의 예측 요건. RMSE 로 정한 tolerance 의 하한           | scalar  | -                          |
| $`\rho`$               | 예측 요건을 공정 요건으로 나눈 비                                | scalar  | $`\rho \gt 0`$             |
| $`h_U`$                | wafer uniformity 의 tolerance. $`\bar{U}`$ 에서 USL 까지의 거리  | scalar  | $`h_U \gt 0`$              |
| $`h_{P,U}`$            | wafer uniformity 의 공정 요건                                    | scalar  | -                          |
| $`h_{E,U}`$            | wafer uniformity 의 예측 요건                                    | scalar  | -                          |
| $`\mathrm{UAL}_U`$     | wafer uniformity 의 upper acceptance limit                       | scalar  | -                          |
| $`C_{pk,\min}`$        | thickness 에서 요구하는 공정능력지수 $`C_{pk}`$ 의 최솟값        | scalar  | $`C_{pk,\min} \gt 0`$      |
| $`C_{pu,\min}`$        | wafer uniformity 에서 요구하는 공정능력지수 $`C_{pu}`$ 의 최솟값 | scalar  | $`C_{pu,\min} \gt 0`$      |
| $`n_\sigma`$           | tolerance 안에 들어가야 하는 RMSE 의 최소 개수                   | scalar  | $`n_\sigma \gt 0`$         |
| $`k`$                  | guard band 계수                                                  | scalar  | $`k \ge 0`$                |
| $`g`$                  | guard band $`k\, \mathrm{RMSE}`$                                 | scalar  | -                          |
| $`\alpha`$             | 허용하는 오판정 확률                                             | scalar  | $`0 \lt \alpha \lt 1`$     |

### 3.2 RMSE

RMSE 는 예측 오차 $`e_{ij} = y_{ij} - \hat{y}_{ij}`$ 의 제곱 평균의 제곱근이며, 예측 오차의 평균 $`b_e`$ 와 표준편차 $`\sigma_e`$ 로 나뉜다.

```math
\mathrm{RMSE} = \sqrt{\frac{1}{T} \sum_{i=1}^{M} \sum_{j=1}^{N} e_{ij}^2}
\hspace{19em} (1)
```

```math
b_e = \frac{1}{T} \sum_{i=1}^{M} \sum_{j=1}^{N} e_{ij}
\hspace{19em} (2)
```

```math
\sigma_e = \sqrt{\frac{1}{T} \sum_{i=1}^{M} \sum_{j=1}^{N} \left( e_{ij} - b_e \right)^2}
\hspace{19em} (3)
```

식 (3) 의 제곱을 풀면 $`\sigma_e^2 = \frac{1}{T} \sum e_{ij}^2 - b_e^2`$ 이므로, 식 (1) 과 함께 아래 식 (4) 가 성립한다.

```math
\mathrm{RMSE}^2 = b_e^2 + \sigma_e^2
\hspace{19em} (4)
```

RMSE 는 $`b_e = 0`$ 일 때만 예측 오차의 표준편차 $`\sigma_e`$ 와 같고, 그 밖에는 $`\sigma_e`$ 보다 크다. 이 문서는 tolerance 를 정할 때 RMSE 를 예측 정확도의 보수적인 설계 지표로 쓰며, RMSE 의 몇 배를 정규분포의 몇 σ coverage 로 읽지 않는다. RMSE 와 $`R^2_{\mathrm{all}}`$ 의 관계는 [Appendix B](#appendix-b-rmse-and-r2_all) 에 있다.

### 3.3 Tolerance

Thickness spec 은 target $`y_0`$ 에 대칭인 $`y_0 \pm h`$ 로 두며, tolerance $`h`$ 는 공정 요건 $`h_P`$ 와 예측 요건 $`h_E`$ 가운데 큰 값이다.

공정 요건은 측정값의 표준편차 $`\sigma_y`$ 와 공정능력지수 $`C_{pk} = \min\left( (\mathrm{USL} - \bar{y}),\ (\bar{y} - \mathrm{LSL}) \right) / (3\sigma_y)`$ [[1](#ref-1)] 에서 나온다.

```math
\sigma_y = \sqrt{\frac{1}{T} \sum_{i=1}^{M} \sum_{j=1}^{N} \left( y_{ij} - \bar{y} \right)^2}
\hspace{19em} (5)
```

$`\mathrm{USL} = y_0 + h`$, $`\mathrm{LSL} = y_0 - h`$ 를 넣으면 두 값 가운데 작은 쪽은 $`h - |\bar{y} - y_0|`$ 이므로, $`C_{pk} = (h - |\bar{y} - y_0|) / (3\sigma_y) \ge C_{pk,\min}`$ 를 만족하는 가장 작은 $`h`$ 가 아래 식 (6) 의 공정 요건이다. 측정값 평균이 target 과 같으면 ($`\bar{y} = y_0`$) $`C_{pk}`$ 는 $`C_p = (\mathrm{USL} - \mathrm{LSL}) / (6\sigma_y)`$ [[1](#ref-1)] 와 같고, 평균이 target 에서 벗어난 만큼 공정 요건이 커진다.

```math
h_P = 3\, C_{pk,\min}\, \sigma_y + |\bar{y} - y_0|
\hspace{19em} (6)
```

예측 요건은 tolerance 안에 RMSE 가 $`n_\sigma`$ 개 이상 들어가야 한다는 설계 규칙이다. 이 요건은 target 과 측정값 평균의 차이에 따라 달라지지 않는다.

```math
h_E = n_\sigma\, \mathrm{RMSE}
\hspace{19em} (7)
```

Tolerance 는 아래 식 (8) 이고, spec 은 $`y_0 \pm h`$ 이다.

```math
h = \max\left( h_P,\ h_E \right)
\hspace{19em} (8)
```

두 요건 가운데 어느 쪽이 tolerance 를 정하는지는 아래 식 (9) 의 비 $`\rho`$ 로 가린다.

```math
\rho = \frac{h_E}{h_P}
\hspace{19em} (9)
```

$`\rho`$ 에 따라 tolerance 를 정하는 요건과 tolerance 를 좁히는 방법은 아래 Fig 1 이다.

```text
                         rho = h_E / h_P

  0                             1
  +-----------------------------+-------------------------------->  rho
  |          rho < 1            |            rho > 1
  |                             |
  |  process sets h             |  prediction sets h
  |  h = h_P                    |  h = h_E
  |                             |
  |  to narrow h:               |  to narrow h:
  |  - reduce sigma_y           |  - reduce RMSE of the
  |  - move y_bar to y_0        |    prediction model
                                |
                       rho = 1: h_P = h_E
```

<a id="fig-1"></a>Fig 1. Requirement that sets the tolerance

- 가로축은 $`\rho`$ 이며, $`\rho = 1`$ 에서 두 요건이 같다.
- $`\rho \lt 1`$ 이면 공정 요건이 tolerance 를 정한다. Tolerance 를 좁히려면 $`\sigma_y`$ 를 줄이거나 측정값 평균을 target 에 맞춘다.
- $`\rho \gt 1`$ 이면 예측 요건이 tolerance 를 정한다. Tolerance 를 좁히려면 예측 model 의 RMSE 를 줄인다.

$`\rho`$ 는 이 문서가 정의한 비이며 산업 표준 지표가 아니다. 측정값 평균이 target 과 같을 때 $`\rho \gt 1`$ 을 $`R^2_{\mathrm{all}}`$ 로 바꾼 조건은 [Appendix B](#appendix-b-rmse-and-r2_all) 에 있다.

### 3.4 Guard Band

예측값의 acceptance limit 은 spec limit 을 guard band $`g = k\, \mathrm{RMSE}`$ 만큼 안쪽으로 좁힌 값이다.

```math
\lvert \hat{y}_{ij} - y_0 \rvert \le h - g
\hspace{19em} (10)
```

Acceptance 구간이 비어 있지 않으려면 $`h \ge g`$ 이어야 한다. $`h \ge h_E = n_\sigma\, \mathrm{RMSE}`$ 이므로 $`n_\sigma \ge k`$ 이면 이 조건은 언제나 만족한다.

Guard band 가 지켜야 하는 것은 합격으로 판정한 예측값에서 측정값이 spec 밖에 있을 조건부 확률이다.

```math
P\left( y_{ij} \gt \mathrm{USL} \mid \hat{y}_{ij} \right) \le \alpha
\hspace{19em} (11)
```

식 (10) 의 $`g = k\, \mathrm{RMSE}`$ 는 아래 세 조건이 모두 성립할 때 식 (11) 을 $`\alpha = 1 - \Phi(k)`$ 로 만족한다. 여기서 $`\Phi`$ 는 표준정규분포의 누적분포함수이며, $`k = 1.645`$ 이면 $`\alpha = 0.05`$ 이다.

- 예측 오차의 평균이 0 이다 ($`b_e = 0`$, 측정 위치마다의 mean error (offset) 도 0).
- 예측 오차가 정규분포를 따른다.
- 예측값의 크기와 측정 위치에 상관없이 예측 오차의 표준편차가 RMSE 로 같다.

### 3.5 Wafer Uniformity Spec

Wafer uniformity 는 wafer 한 장의 측정값 $`N`$ 개의 표준편차를 그 평균으로 나눈 값이며, 예측값 $`\hat{U}_i`$ 는 같은 식에 $`\hat{y}_{ij}`$ 를 넣어 구한다.

```math
s_i = \sqrt{\frac{1}{N-1} \sum_{j=1}^{N} \left( y_{ij} - \bar{y}_{i\cdot} \right)^2}
\hspace{19em} (12)
```

```math
U_i = \frac{s_i}{\bar{y}_{i\cdot}}
\hspace{19em} (13)
```

```math
\mathrm{RMSE}_U = \sqrt{\frac{1}{M} \sum_{i=1}^{M} \left( U_i - \hat{U}_i \right)^2}
\hspace{19em} (14)
```

Wafer uniformity 는 작을수록 좋으므로 upper spec limit 하나만 두고, 정해 둔 target 이 없으므로 기준값을 측정값의 평균 $`\bar{U}`$ 로 둔다. 한쪽 공정능력지수 $`C_{pu} = (\mathrm{USL} - \bar{U}) / (3\sigma_U)`$ [[1](#ref-1)] 에서 공정 요건이 나오고, 나머지는 thickness 와 같은 구조이다.

```math
h_{P,U} = 3\, C_{pu,\min}\, \sigma_U
\hspace{19em} (15)
```

```math
h_{E,U} = n_\sigma\, \mathrm{RMSE}_U
\hspace{19em} (16)
```

```math
h_U = \max\left( h_{P,U},\ h_{E,U} \right)
\hspace{19em} (17)
```

```math
\mathrm{USL}_U = \bar{U} + h_U
\hspace{19em} (18)
```

```math
\mathrm{UAL}_U = \mathrm{USL}_U - k\, \mathrm{RMSE}_U
\hspace{19em} (19)
```

Thickness 와 wafer uniformity 의 식을 견주면 아래 Table 2 이다.

Table 2. Thickness and wafer uniformity spec rules

| Item                   | Thickness                                                    | Wafer uniformity                       |
| :--------------------: | :----------------------------------------------------------: | :------------------------------------: |
| Spec type              | 양측                                                         | 상측                                   |
| Reference              | $`y_0`$                                                      | $`\bar{U}`$                            |
| Process requirement    | $`3\, C_{pk,\min}\, \sigma_y + \lvert \bar{y} - y_0 \rvert`$ | $`3\, C_{pu,\min}\, \sigma_U`$         |
| Prediction requirement | $`n_\sigma\, \mathrm{RMSE}`$                                 | $`n_\sigma\, \mathrm{RMSE}_U`$         |
| Tolerance              | 두 요건의 최댓값                                             | 두 요건의 최댓값                       |
| Acceptance             | 양쪽에서 $`k\, \mathrm{RMSE}`$ 를 뺌                         | 상한에서 $`k\, \mathrm{RMSE}_U`$ 를 뺌 |

- Wafer uniformity 에서는 기준값이 측정값의 평균이어서 공정 요건에 평균과 기준값의 차이가 더해지지 않는다.
- 두 경우 모두 RMSE 가 커지면 예측 요건이 커져 tolerance 가 넓어질 수 있다. 이 문서는 RMSE 로 spec 을 새로 정하므로 이 결과를 그대로 두며, spec 을 넓힐 수 없는 경우는 [꼭지 5.2](#52-breaking-conditions) 에 있다.

## 4. Worked Example

반도체 wafer 44 장의 측정 위치 13 곳에서 측정한 값과 예측값 ([data/u0_actuals_and_predictions.csv](data/u0_actuals_and_predictions.csv)) 으로 두 spec 을 구한다. 이 데이터는 R2 by Aggregation Level 문서의 Appendix D 와 같은 데이터이며, 측정값은 `true_p1` 부터 `true_p13`, 예측값은 `pred_p1` 부터 `pred_p13` 열이다. Thickness 의 단위는 nm 이고, target 은 $`y_0 = 600`$ nm 이다.

입력은 $`C_{pk,\min} = 1.33`$, $`C_{pu,\min} = 1.33`$, $`n_\sigma = 10`$, $`k = 1.645`$ 로 두었다. $`C_{pk} = C_p = 1.33`$ 은 중심이 맞는 정규분포 공정에서 spec 밖 비율이 약 64 ppm 인 값이다 [[1](#ref-1)]. $`n_\sigma = 10`$ 은 이 예에서 고른 설계 parameter 이다. Gauge R&R 의 측정 system 평가 기준 [[2](#ref-2)] 은 tolerance 에 대한 측정 system 변동의 비율로 정의되어 이 문서의 $`h / \mathrm{RMSE}`$ 와 정의가 다르므로, $`n_\sigma`$ 의 근거로 쓰지 않는다.

### 4.1 RMSE of the Data

현재 예측과 측정 위치마다의 mean error (offset) 를 없앤 예측으로 구한 RMSE 는 아래 Table 3 이다.

Table 3. RMSE of the location predictions

| Quantity               | Current prediction | Offset removed |
| :--------------------: | :----------------: | :------------: |
| $`\bar{y}`$            | 634.5735           | 634.5735       |
| $`\sigma_y`$           | 28.5043            | 28.5043        |
| RMSE                   | 12.0490            | 8.7503         |
| $`b_e`$                | -2.3737            | 0.0000         |
| $`\sigma_e`$           | 11.8129            | 8.7503         |
| $`R^2_{\mathrm{all}}`$ | 0.8213             | 0.9058         |

- Offset removed 는 측정 위치 $`j`$ 의 예측값에 측정 위치 $`j`$ 의 mean error (offset), 곧 $`e_{ij}`$ 를 wafer 44 장에 대해 평균한 값을 더한 예측이다. 같은 데이터의 측정값으로 구한 보정을 같은 데이터에 쓴 분석값이며, offset 을 없앤 개선의 효과를 보인다.
- 현재 예측에서 $`12.0490^2 = (-2.3737)^2 + 11.8129^2`$ 로 식 (4) 가 성립한다. 전체 bias $`b_e`$ 는 작지만, 측정 위치마다의 offset 은 $`\sigma_e`$ 에 들어 있어 offset 을 없애면 $`\sigma_e`$ 가 11.8129 에서 8.7503 으로 줄어든다.

### 4.2 Thickness Spec

식 (5) 부터 식 (10) 으로 구한 thickness spec 은 아래 Table 4 이며, 값은 $`\rho`$ 를 빼고 모두 nm 이다.

Table 4. Thickness spec from the RMSE

| Quantity                          | Current prediction | Offset removed |
| :-------------------------------: | :----------------: | :------------: |
| $`h_P`$                           | 148.3056           | 148.3056       |
| $`h_E`$                           | 120.4904           | 87.5032        |
| $`\rho`$                          | 0.8124             | 0.5900         |
| Governed by                       | Process            | Process        |
| Spec (target ± $`h`$)             | 600 ± 148.3056     | 600 ± 148.3056 |
| Guard band $`g`$                  | 19.8207            | 14.3943        |
| Acceptance (target ± $`(h - g)`$) | 600 ± 128.4850     | 600 ± 133.9114 |

Tolerance 의 근거는 아래이다.

- $`h_P`$ 는 식 (6) 의 $`3 \times 1.33 \times 28.5043 + |634.5735 - 600| = 113.7321 + 34.5735`$ 이다. 앞 항은 측정값의 산포로 $`C_{pk} = 1.33`$ 을 지키는 폭이고, 뒤 항은 측정값 평균이 target 에서 벗어난 만큼 더한 폭이다.
- $`h_E`$ 는 식 (7) 의 $`10 \times 12.0490 = 120.4904`$ 이다.
- $`\rho = 120.4904 / 148.3056 = 0.8124`$ 로 1 보다 작아, $`h = h_P = 148.3056`$ 이다. Offset 을 없애도 tolerance 는 그대로이고, guard band 만 줄어 acceptance 의 폭이 넓어진다.
- Acceptance 의 limit 은 식 (10) 대로 $`148.3056 - 19.8207 = 128.4850`$ 이다. $`n_\sigma = 10 \ge k = 1.645`$ 이어서 $`h \ge g`$ 이다.
- 측정값 평균이 target 과 같으면 ($`y_0 = 634.5735`$) $`h_P`$ 는 113.7321 로 줄어 $`\rho = 1.0594`$ 가 되고, spec 은 634.5735 ± 120.4904 이다.
- $`n_\sigma = 30`$ 이면 $`h_E`$ 는 3 배인 361.4712 가 되어 예측 요건이 tolerance 를 정한다.

### 4.3 Wafer Uniformity Spec

식 (12) 부터 식 (19) 로 구한 wafer uniformity spec 은 아래 Table 5 이며, 값은 $`\rho`$ 를 빼고 모두 % 이다.

Table 5. Wafer uniformity spec from the RMSE

| Quantity                                         | Current prediction | Offset removed  |
| :----------------------------------------------: | :----------------: | :-------------: |
| $`\sigma_U`$                                     | 0.9971             | 0.9971          |
| $`\mathrm{RMSE}_U`$                              | 0.7725             | 0.7980          |
| $`h_{P,U}`$                                      | 3.9785             | 3.9785          |
| $`h_{E,U}`$                                      | 7.7251             | 7.9800          |
| $`\rho`$                                         | 1.9417             | 2.0057          |
| Governed by                                      | Prediction         | Prediction      |
| Spec ($`\bar{U}`$ + $`h_U`$)                     | 2.4962 + 7.7251    | 2.4962 + 7.9800 |
| Guard band $`g`$                                 | 1.2708             | 1.3127          |
| $`\mathrm{UAL}_U`$ ($`\bar{U}`$ + $`(h_U - g)`$) | 2.4962 + 6.4543    | 2.4962 + 6.6673 |
| Largest $`U_i`$                                  | 6.3171             | 6.3171          |

Tolerance 의 근거는 아래이다.

- 기준값 2.4962 % 는 측정한 $`U_i`$ 의 평균 $`\bar{U}`$ 이다.
- $`h_{P,U}`$ 는 식 (15) 의 $`3 \times 1.33 \times 0.9971 = 3.9785`$ 이고, $`h_{E,U}`$ 는 식 (16) 의 $`10 \times 0.7725 = 7.7251`$ 이다.
- $`\rho = 7.7251 / 3.9785 = 1.9417`$ 로 1 보다 커서 $`h_U = h_{E,U} = 7.7251`$ 이다. 예측 요건이 공정 요건 이하가 되는 $`\mathrm{RMSE}_U`$ 는 $`3.9785 / 10 = 0.3979`$ %p 이고, 현재 예측은 그 1.94 배이다.
- $`\mathrm{UAL}_U`$ 는 식 (19) 대로 $`7.7251 - 1.2708 = 6.4543`$ 을 기준값에 더한 값이다.
- 현재 RMSE 와 설정한 공정능력 및 예측 정확도 요건을 동시에 만족하는 최소 USL 은 $`2.4962 + 7.7251 = 10.2213`$ % 이다. 더 작은 USL 을 적용하려면 요건을 변경하거나 예측 오차를 줄여야 한다.
- Offset 을 없애면 thickness RMSE 는 27 % 줄지만 $`\mathrm{RMSE}_U`$ 는 0.7725 에서 0.7980 으로 오른다. 측정 위치마다 wafer 44 장에 공통인 값을 더하는 보정은 wafer 사이에서 달라지는 $`U_i`$ 의 오차를 줄이지 않는다.

### 4.4 Calculation Script

Table 3 부터 Table 5 의 값은 같은 folder 의 [src/rmse_process_spec.py](src/rmse_process_spec.py) 로 구했다. 이 script 는 CSV 에서 식 (1) 의 RMSE, 식 (2) 와 식 (3) 의 $`b_e`$ 와 $`\sigma_e`$, 식 (14) 의 $`\mathrm{RMSE}_U`$ 를 구하고, `--target` 으로 받은 thickness target 에 대해 현재 예측과 offset 을 없앤 예측마다 공정 요건, 예측 요건, $`\rho`$, spec limit, acceptance limit 을 출력한다. `--output-folder` 를 주면 wafer 마다의 $`U_i`$ 와 $`\hat{U}_i`$ 를 `uniformity_by_wafer.csv` 로 저장한다. 문서 folder 에서 아래처럼 실행한다.

```bash
python3 src/rmse_process_spec.py --csv data/u0_actuals_and_predictions.csv --target 600 --cp-min 1.33 --n-sigma 10 --guard-k 1.645
```

## 5. Application

### 5.1 Conditions

이 방법은 RMSE 를 구한 wafer 와 판정할 wafer 의 예측 오차 분포가 같을 때 쓴다.

- 측정값과 예측값이 같은 wafer 의 같은 측정 위치 $`N`$ 곳에 있다.
- RMSE 를 구한 wafer 가 판정할 기간의 장비, recipe, 제품을 대표한다.
- $`C_{pk,\min}`$, $`C_{pu,\min}`$, $`n_\sigma`$, $`k`$ 는 공정과 품질 담당이 정해 입력한다. Script 의 `--cp-min` 은 두 최솟값에 같은 값을 준다.
- $`\rho \lt 1`$ 이면 spec 이 측정값과 target 만으로 정해지고, RMSE 는 guard band 에만 쓰인다.

### 5.2 Breaking Conditions

아래 경우에는 Table 4 와 Table 5 의 spec 과 acceptance limit 을 그대로 믿을 수 없다.

- **Spec 을 넓힐 수 없는 경우**: 식 (8) 과 식 (17) 에서 RMSE 가 커지면 tolerance 가 넓어질 수 있다. 품질 요구로 spec 이 이미 정해져 있으면 정해진 spec 에 대한 판정 방법을 써야 하며, 이 문서는 그 방법을 다루지 않는다.
- **Bias 와 offset 이 있는 예측 오차**: 식 (4) 에서 RMSE 는 bias 를 포함하므로, $`n_\sigma`$ 와 $`k`$ 를 정규분포의 coverage 로 읽을 수 없다. 이 데이터는 offset 을 없애면 RMSE 가 12.0490 에서 8.7503 으로 줄어, 측정 위치마다의 offset 이 RMSE 에 큰 비율로 들어 있다. 이때 식 (10) 의 guard band 는 식 (11) 을 $`\alpha = 0.05`$ 로 보장하지 않는다.
- **예측값에 따라 다른 오차**: 예측 오차의 분산이 예측값의 크기에 따라 다르거나, 한 wafer 안의 측정 위치 오차가 서로 상관되어 있으면 식 (11) 의 조건부 확률이 RMSE 하나로 정해지지 않는다.
- **정규분포 가정이 의심스러운 wafer uniformity**: $`U_i`$ 의 최댓값 6.3171 % 는 둘째로 큰 값 4.4633 % 보다 1.85 %p 크고, $`U_i`$ 44 개의 왜도는 1.6453 이다. 같은 표본으로 평균과 표준편차를 추정했으므로 이 값만으로 분포가 오른쪽 꼬리가 길다고 확정할 수는 없지만, 정규분포를 가정한 $`C_{pu}`$ 의 spec 밖 비율은 실제와 다를 수 있다. 이때는 Q-Q plot 으로 정규성을 확인하고, bootstrap 또는 비모수 분위수로 USL 을 함께 구해 견준다.
- **표본 수**: RMSE, $`\sigma_y`$, $`\sigma_U`$ 는 wafer 44 장으로 구한 추정값이며, 공정능력 추정에는 일반적으로 표본 100 개 이상이 필요하다 [[1](#ref-1)]. Wafer uniformity 는 wafer 한 장에 값이 하나여서 표본이 44 개이다.
- **예측 model 의 변경**: model 을 다시 학습하거나 공정 조건이 바뀌면 RMSE 를 다시 구하고 spec 을 다시 계산한다.

## References

<a id="ref-1"></a>[1] NIST/SEMATECH. [6.1.6. What is Process Capability?](https://www.itl.nist.gov/div898/handbook/pmc/section1/pmc16.htm). *NIST/SEMATECH e-Handbook of Statistical Methods*.<br>
<a id="ref-2"></a>[2] Wikipedia. [ANOVA gauge R&R](https://en.wikipedia.org/wiki/ANOVA_gauge_R%26R). 측정 system 평가 기준의 출처로 Burdick, R. K., Borror, C. M., and Montgomery, D. C. (2005). *Design and Analysis of Gauge R and R Studies*. SIAM-ASA, p. 4, ISBN 0898715881 을 인용한다.

---

## Appendix A. Terminology

- **$`\rho`$**: 예측 요건을 공정 요건으로 나눈 비. 1 보다 크면 예측 요건이 tolerance 를 정한다
- **$`n_\sigma`$**: tolerance 안에 들어가야 하는 RMSE 의 최소 개수. 설계 규칙이며 정규분포의 coverage 를 뜻하지 않는다
- **acceptance limit**: 예측값으로 합격을 판정하는 한계. Spec limit 을 guard band 만큼 안쪽으로 좁힌 값
- **bias**: 예측 오차의 평균. 이 문서에서는 $`b_e`$
- **guard band**: 예측 오차 때문에 spec limit 을 안쪽으로 좁히는 폭. 이 문서에서는 $`k\, \mathrm{RMSE}`$
- **mean error (offset)**: 한 측정 위치에서 예측 오차를 wafer 에 대해 평균한 값
- **process capability index**: 공정능력지수. Spec limit 까지의 거리를 공정 산포로 나눈 값. 양쪽 spec 은 $`C_p`$ 와 $`C_{pk}`$, upper spec 하나는 $`C_{pu}`$ 를 쓴다
- **RMSE**: root mean squared error. 예측 오차 제곱 평균의 제곱근
- **spec limit**: 공정 결과가 합격인 범위의 경계. Upper spec limit (USL) 과 lower spec limit (LSL)
- **target ± tolerance**: spec 을 중심값 target 과 그 중심에서 spec limit 까지의 거리 tolerance 로 적는 꼴
- **wafer uniformity**: wafer 한 장의 측정값 표준편차를 그 평균으로 나눈 값 (1σ/mean). 이 문서에서는 % 로 적는다

## Appendix B. RMSE and R2_all

$`R^2_{\mathrm{all}}`$ 은 1 에서 식 (1) 의 제곱합을 식 (5) 의 제곱합으로 나눈 값을 뺀 값이다. 두 식의 분모가 모두 $`T`$ 이므로 그 비는 $`\mathrm{RMSE}^2 / \sigma_y^2`$ 와 같고, 아래 식 (20) 이 성립한다.

```math
\frac{\mathrm{RMSE}}{\sigma_y} = \sqrt{1 - R^2_{\mathrm{all}}}
\hspace{19em} (20) 🌳
```

데이터 셋에 따라 임의로 변하는 쪽은 $`R^2_{\mathrm{all}}`$ 이다. RMSE 는 식 (1) 에서 예측 오차 $`y_{ij} - \hat{y}_{ij}`$ 만으로 정해지므로, 예측 오차가 같으면 $`\sigma_y`$ 와 상관없이 같은 값이다. 반대로 $`R^2_{\mathrm{all}}`$ 은 식 (20) 에서 RMSE 를 $`\sigma_y`$ 와 견준 값이라, 같은 예측 오차라도 $`\sigma_y`$ 가 큰 데이터 셋에서는 커지고 작은 데이터 셋에서는 작아진다. 그러니 "$`R^2_{\mathrm{all}}`$ 이 그대로인데 $`\sigma_y`$ 가 커진다" 는 상황은 예측 오차 자체가 $`\sigma_y`$ 와 같은 비율로 커진 경우이고, RMSE 가 커지는 것은 그 오차가 실제로 커졌기 때문이다. 이 문서가 예측 요건 식 (7) 에 $`R^2_{\mathrm{all}}`$ 대신 RMSE 를 쓰는 이유도 여기에 있다.

측정값 평균이 target 과 같으면 ($`\bar{y} = y_0`$) 식 (9) 의 $`\rho \gt 1`$ 은 $`n_\sigma\, \mathrm{RMSE} \gt 3\, C_{pk,\min}\, \sigma_y`$ 이고, 양변을 $`n_\sigma\, \sigma_y`$ 로 나누고 식 (20) 을 넣어 제곱하면 아래 식 (21) 이다.

```math
R^2_{\mathrm{all}} \lt 1 - \left( \frac{3\, C_{pk,\min}}{n_\sigma} \right)^2
\hspace{19em} (21)
```

식 (21) 이 성립하면 $`h`$ 는 식 (7) 의 $`h_E = n_\sigma\, \mathrm{RMSE}`$ 이고, 성립하지 않으면 식 (6) 의 $`h_P = 3\, C_{pk,\min}\, \sigma_y + |\bar{y} - y_0|`$ 이다. 식 (21) 으로 판단할 수 있는 것은 $`\bar{y} = y_0`$ 일 때뿐이고, 그렇지 않으면 식 (9) 의 $`\rho`$ 로 판단한다. $`C_{pk,\min} = 1.33`$, $`n_\sigma = 10`$ 이면 $`3\, C_{pk,\min} / n_\sigma = 0.399`$ 이어서, $`R^2_{\mathrm{all}}`$ 이 0.8408 보다 낮으면 예측 요건이 tolerance 를 정한다.
