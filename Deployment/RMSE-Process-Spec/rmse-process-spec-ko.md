# Process Spec from Prediction RMSE
Version 2 | Created: 2026-11-16 | Updated: 2026-11-16 03:10 CST

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

## 1. Purpose

**Problem Statement**: 측정하지 않은 wafer 를 예측값으로 판정하면 예측 오차가 공정 산포에 더해지는데, 그 오차를 반영해 thickness spec 과 wafer uniformity spec 을 구하는 방법이 없다.

**Goal**: 예측값의 RMSE 로 아래 세 가지를 구한다.

1. 예측값으로 판정할 수 있는 thickness spec 을 target ± tolerance 로 제시하고, 그 tolerance 의 가장 작은 값과 근거
2. Wafer uniformity (1σ/mean) 의 upper spec limit
3. 예측값을 판정할 acceptance limit

**Non-Goal**:

- 예측 model 을 만들거나 고치는 방법은 다루지 않는다.
- 제품 성능에서 정하는 design spec 은 다루지 않는다. 이 문서의 spec 은 공정 산포와 예측 오차로 정하는 공정 spec 이다.
- 요구 공정능력지수 (process capability index), P/T (precision-to-tolerance ratio, 측정 system 표준편차의 6 배를 spec 폭으로 나눈 값) 상한, guard band 계수의 값을 정하는 방법은 다루지 않는다. 세 값은 입력으로 받는다.

## 2. Summary

Spec 은 target ± tolerance 이며, tolerance 는 Cpk 로 구한 공정 요건 $`3\, C_{\min}\, \sigma_y + |\bar{y} - y_0|`$ 와 예측 요건 $`3\, \mathrm{RMSE} / P_{\max}`$ 가운데 큰 값이고, 예측값의 acceptance limit 은 spec 을 $`k\, \mathrm{RMSE}`$ 만큼 안쪽으로 좁힌 값이다.
Wafer 44 장, 측정 위치 13 곳의 데이터에서 thickness spec 은 600 ± 148.3056 nm 로 공정 요건이 tolerance 를 정하고, wafer uniformity spec 은 2.4962 % + 7.7251 % 로 RMSE 0.7725 %p 의 예측 요건이 tolerance 를 정한다.
Thickness 는 측정값 평균이 target 에서 34.5735 nm 벗어나 공정 요건이 커졌고, RMSE 가 14.8306 nm 를 넘어야 예측 요건이 tolerance 를 정한다.

## 3. Principle

### 3.1 Notation

이 문서의 기호는 아래 Table 1 이다.

Table 1. Symbols

| Symbol                 | Meaning                                                    | Shape   | Assumption                 |
| :--------------------: | :--------------------------------------------------------: | :-----: | :------------------------: |
| $`M`$                  | wafer 수                                                   | integer | $`M \ge 2`$                |
| $`N`$                  | wafer 한 장의 측정 위치 수                                 | integer | $`N \ge 2`$                |
| $`T`$                  | 전체 측정값의 개수 $`MN`$                                  | integer | -                          |
| $`y_{ij}`$             | wafer $`i`$ 의 측정 위치 $`j`$ 에서 측정값                 | scalar  | -                          |
| $`\hat{y}_{ij}`$       | wafer $`i`$ 의 측정 위치 $`j`$ 에서 예측값                 | scalar  | -                          |
| $`\bar{y}`$            | 측정값 $`T`$ 개의 평균                                     | scalar  | -                          |
| $`\bar{y}_{i\cdot}`$   | wafer $`i`$ 의 측정값 $`N`$ 개의 평균                      | scalar  | $`\bar{y}_{i\cdot} \gt 0`$ |
| $`\sigma_y`$           | 측정값 $`T`$ 개의 표준편차 (분모 $`T`$)                    | scalar  | $`\sigma_y \gt 0`$         |
| $`\mathrm{RMSE}`$      | 측정 위치 예측값의 root mean squared error                 | scalar  | -                          |
| $`R^2_{\mathrm{all}}`$ | 측정값 $`T`$ 개로 구한 R²                                  | scalar  | -                          |
| $`s_i`$                | wafer $`i`$ 의 측정값 $`N`$ 개의 표준편차 (분모 $`N-1`$)   | scalar  | -                          |
| $`U_i`$                | wafer $`i`$ 의 wafer uniformity                            | scalar  | -                          |
| $`\hat{U}_i`$          | wafer $`i`$ 의 wafer uniformity 예측값                     | scalar  | -                          |
| $`\bar{U}`$            | $`U_i`$ $`M`$ 개의 평균                                    | scalar  | -                          |
| $`\sigma_U`$           | $`U_i`$ $`M`$ 개의 표준편차 (분모 $`M`$)                   | scalar  | $`\sigma_U \gt 0`$         |
| $`\mathrm{RMSE}_U`$    | wafer uniformity 예측값의 RMSE                             | scalar  | -                          |
| $`\mathrm{USL}`$       | upper spec limit                                           | scalar  | -                          |
| $`\mathrm{LSL}`$       | lower spec limit                                           | scalar  | -                          |
| $`y_0`$                | thickness 의 target                                        | scalar  | -                          |
| $`h`$                  | tolerance. target 에서 spec limit 까지의 거리              | scalar  | $`h \gt 0`$                |
| $`C_{\min}`$           | 요구하는 공정능력지수 $`C_{pk}`$ 또는 $`C_{pu}`$ 의 최솟값 | scalar  | $`C_{\min} \gt 0`$         |
| $`P_{\max}`$           | 허용하는 P/T 의 최댓값                                     | scalar  | $`0 \lt P_{\max} \le 1`$   |
| $`k`$                  | guard band 계수                                            | scalar  | $`k \ge 0`$                |
| $`g`$                  | guard band $`k\, \mathrm{RMSE}`$                           | scalar  | -                          |

### 3.2 RMSE

RMSE 는 측정 위치 예측 오차 $`y_{ij} - \hat{y}_{ij}`$ 의 제곱 평균의 제곱근이고, 측정값의 표준편차 $`\sigma_y`$ 도 같은 분모 $`T`$ 로 구한다.

```math
\mathrm{RMSE} = \sqrt{\frac{1}{T} \sum_{i=1}^{M} \sum_{j=1}^{N} \left( y_{ij} - \hat{y}_{ij} \right)^2}
\hspace{19em} (1)
```

```math
\sigma_y = \sqrt{\frac{1}{T} \sum_{i=1}^{M} \sum_{j=1}^{N} \left( y_{ij} - \bar{y} \right)^2}
\hspace{19em} (2)
```

$`R^2_{\mathrm{all}}`$ 은 1 에서 식 (1) 의 제곱합을 식 (2) 의 제곱합으로 나눈 값을 뺀 값이다. 두 식의 분모가 모두 $`T`$ 이므로 그 비는 $`\mathrm{RMSE}^2 / \sigma_y^2`$ 와 같고, 아래 식 (3) 이 성립한다.

```math
\frac{\mathrm{RMSE}}{\sigma_y} = \sqrt{1 - R^2_{\mathrm{all}}}
\hspace{19em} (3)
```

### 3.3 Tolerance

Thickness spec 은 target $`y_0`$ 에 대칭인 $`y_0 \pm h`$ 로 두며, tolerance $`h`$ 는 측정값의 공정능력지수가 $`C_{\min}`$ 이상이라는 공정 요건과, 예측값의 P/T 가 $`P_{\max}`$ 이하라는 예측 요건을 함께 만족하는 가장 작은 값이다.

공정 요건은 공정능력지수 $`C_{pk} = \min\left( (\mathrm{USL} - \bar{y}),\ (\bar{y} - \mathrm{LSL}) \right) / (3\sigma_y)`$ [[1](#ref-1)] 에서 나온다. $`\mathrm{USL} = y_0 + h`$, $`\mathrm{LSL} = y_0 - h`$ 를 넣으면 두 값 가운데 작은 쪽은 $`h - |\bar{y} - y_0|`$ 이므로, $`C_{pk} = (h - |\bar{y} - y_0|) / (3\sigma_y) \ge C_{\min}`$ 에서 아래 식 (4) 이다.

```math
h \ge 3\, C_{\min}\, \sigma_y + |\bar{y} - y_0|
\hspace{19em} (4)
```

측정값 평균이 target 과 같으면 ($`\bar{y} = y_0`$) $`C_{pk}`$ 는 $`C_p = (\mathrm{USL} - \mathrm{LSL}) / (6\sigma_y)`$ [[1](#ref-1)] 와 같고, 식 (4) 는 $`h \ge 3\, C_{\min}\, \sigma_y`$ 이다. 평균이 target 에서 벗어난 만큼 tolerance 가 넓어져야 한다.

예측 요건은 예측값을 표준편차가 RMSE 인 측정 system 으로 보고 구한다. 측정 system 의 P/T 는 $`6\, \mathrm{RMSE} / (\mathrm{USL} - \mathrm{LSL}) = 3\, \mathrm{RMSE} / h`$ 이고, 이것이 $`P_{\max}`$ 이하이므로 아래 식 (5) 이다. 이 요건은 target 과 측정값 평균의 차이에 따라 달라지지 않는다.

```math
h \ge \frac{3\, \mathrm{RMSE}}{P_{\max}}
\hspace{19em} (5)
```

두 요건을 함께 만족하는 가장 작은 값이 tolerance 이며, spec 은 $`y_0 \pm h`$ 이다.

```math
h = \max\left( 3\, C_{\min}\, \sigma_y + |\bar{y} - y_0|,\ \frac{3\, \mathrm{RMSE}}{P_{\max}} \right)
\hspace{19em} (6)
```

식 (5) 의 하한이 식 (4) 의 하한보다 크면 예측 오차가 tolerance 를 정한다. 두 하한을 견주면, 그 조건은 아래 식 (7) 이다.

```math
\mathrm{RMSE} \gt P_{\max} \left( C_{\min}\, \sigma_y + \frac{|\bar{y} - y_0|}{3} \right)
\hspace{19em} (7)
```

$`\bar{y} = y_0`$ 일 때 식 (7) 의 양변을 $`\sigma_y`$ 로 나누고 식 (3) 을 넣어 제곱하면, 같은 조건은 $`R^2_{\mathrm{all}}`$ 로 아래 식 (8) 이다.

```math
R^2_{\mathrm{all}} \lt 1 - \left( C_{\min}\, P_{\max} \right)^2
\hspace{19em} (8)
```

$`C_{\min} = 1.33`$, $`P_{\max} = 0.3`$ 이면 $`C_{\min} P_{\max} = 0.399`$ 이어서, 평균이 target 과 같을 때 $`R^2_{\mathrm{all}}`$ 이 0.8408 보다 낮으면 예측 오차가 tolerance 를 정한다.

### 3.4 Guard Band

예측값의 acceptance limit 은 spec limit 을 guard band $`g = k\, \mathrm{RMSE}`$ 만큼 안쪽으로 좁힌 값이다.

```math
\mathrm{LSL} + k\, \mathrm{RMSE} \le \hat{y}_{ij} \le \mathrm{USL} - k\, \mathrm{RMSE}
\hspace{19em} (9)
```

예측 오차가 평균 0, 표준편차 RMSE 인 정규분포를 따르면, 예측값이 upper acceptance limit 과 같은 측정 위치에서 측정값이 USL 을 넘을 확률은 표준정규분포에서 $`k`$ 를 넘을 확률이다. $`k = 1.645`$ 이면 그 확률은 0.05 이다.

### 3.5 Wafer Uniformity Spec

Wafer uniformity 는 wafer 한 장의 측정값 $`N`$ 개의 표준편차를 그 평균으로 나눈 값이며, 예측값 $`\hat{U}_i`$ 는 같은 식에 $`\hat{y}_{ij}`$ 를 넣어 구한다.

```math
s_i = \sqrt{\frac{1}{N-1} \sum_{j=1}^{N} \left( y_{ij} - \bar{y}_{i\cdot} \right)^2}
\hspace{19em} (10)
```

```math
U_i = \frac{s_i}{\bar{y}_{i\cdot}}
\hspace{19em} (11)
```

```math
\mathrm{RMSE}_U = \sqrt{\frac{1}{M} \sum_{i=1}^{M} \left( U_i - \hat{U}_i \right)^2}
\hspace{19em} (12)
```

Wafer uniformity 는 작을수록 좋으므로 upper spec limit 하나만 두고, spec 을 target + tolerance 로 적는다. 정해 둔 target 이 없으므로 target 은 측정값의 평균 $`\bar{U}`$ 로 둔다. 한쪽 공정능력지수 $`C_{pu} = (\mathrm{USL} - \bar{U}) / (3\sigma_U)`$ [[1](#ref-1)] 에서 $`h = \mathrm{USL} - \bar{U}`$ 로 쓰면, 식 (4) 부터 식 (6) 이 $`y_0`$ 와 $`\bar{y}`$ 를 $`\bar{U}`$, $`\sigma_y`$ 를 $`\sigma_U`$, RMSE 를 $`\mathrm{RMSE}_U`$ 로 바꾸어 그대로 성립한다.

```math
\mathrm{USL}_U = \bar{U} + \max\left( 3\, C_{\min}\, \sigma_U,\ \frac{3\, \mathrm{RMSE}_U}{P_{\max}} \right)
\hspace{19em} (13)
```

예측값의 acceptance limit 은 $`\hat{U}_i \le \mathrm{USL}_U - k\, \mathrm{RMSE}_U`$ 이다.

## 4. Worked Example

반도체 wafer 44 장의 측정 위치 13 곳에서 측정한 값과 예측값 ([data/u0_actuals_and_predictions.csv](data/u0_actuals_and_predictions.csv)) 으로 두 spec 을 구한다. 이 데이터는 R2 by Aggregation Level 문서의 Appendix D 와 같은 데이터이며, 측정값은 `true_p1` 부터 `true_p13`, 예측값은 `pred_p1` 부터 `pred_p13` 열이다. Thickness 의 단위는 nm 이고, target 은 $`y_0 = 600`$ nm 이다.

입력은 $`C_{\min} = 1.33`$, $`P_{\max} = 0.3`$, $`k = 1.645`$ 로 두었다. $`C_{pk} = C_p = 1.33`$ 은 중심이 맞는 정규분포 공정에서 spec 밖 비율이 약 64 ppm 인 값이다 [[1](#ref-1)].

### 4.1 RMSE of the Data

현재 예측과 측정 위치마다의 mean error (offset) 를 없앤 예측으로 구한 RMSE 는 아래 Table 2 이다.

Table 2. RMSE of the location predictions

| Quantity                     | Current prediction | Offset removed |
| :--------------------------: | :----------------: | :------------: |
| $`\bar{y}`$                  | 634.5735           | 634.5735       |
| $`\sigma_y`$                 | 28.5043            | 28.5043        |
| RMSE                         | 12.0490            | 8.7503         |
| $`\mathrm{RMSE} / \sigma_y`$ | 0.4227             | 0.3070         |
| $`R^2_{\mathrm{all}}`$       | 0.8213             | 0.9058         |

- Offset removed 는 측정 위치 $`j`$ 의 예측값에 측정 위치 $`j`$ 의 mean error (offset), 곧 $`y_{ij} - \hat{y}_{ij}`$ 를 wafer 44 장에 대해 평균한 값을 더한 예측이다. 같은 데이터의 측정값으로 구한 보정을 같은 데이터에 쓴 분석값이며, offset 을 없앤 개선의 효과를 보인다.
- 두 열에서 $`\mathrm{RMSE} / \sigma_y`$ 의 제곱은 $`1 - R^2_{\mathrm{all}}`$ 과 같아 식 (3) 이 성립한다.

### 4.2 Thickness Spec

식 (4) 부터 식 (9) 로 구한 thickness spec 은 아래 Table 3 이며, 값은 모두 nm 이다.

Table 3. Thickness spec from the RMSE

| Quantity                    | Current prediction | Offset removed |
| :-------------------------: | :----------------: | :------------: |
| Spec (target ± tolerance)   | 600 ± 148.3056     | 600 ± 148.3056 |
| Acceptance (target ± limit) | 600 ± 128.4850     | 600 ± 133.9114 |
| Process requirement         | 148.3056           | 148.3056       |
| Prediction requirement      | 120.4904           | 87.5032        |
| Governed by                 | Process            | Process        |
| Guard band $`g`$            | 19.8207            | 14.3943        |

Tolerance 의 근거는 아래이다.

- Tolerance 148.3056 은 식 (6) 에서 공정 요건과 예측 요건 가운데 큰 공정 요건이다.
- 공정 요건은 식 (4) 의 $`3 \times 1.33 \times 28.5043 + |634.5735 - 600| = 113.7321 + 34.5735`$ 이다. 앞 항은 측정값의 산포로 $`C_{pk} = 1.33`$ 을 지키는 폭이고, 뒤 항은 측정값 평균이 target 에서 벗어난 만큼 더한 폭이다.
- 예측 요건은 식 (5) 의 $`3 \times 12.0490 / 0.3 = 120.4904`$ 이며, 예측값으로 판정할 때 P/T 가 0.3 이하가 되는 가장 작은 tolerance 이다.
- 현재 예측의 RMSE 12.0490 은 식 (7) 의 $`0.3 \times (1.33 \times 28.5043 + 34.5735 / 3) = 14.8306`$ 보다 작아, 공정 산포와 평균의 벗어남이 tolerance 를 정한다. Offset 을 없애도 tolerance 는 그대로이고, guard band 만 줄어 acceptance 의 폭이 넓어진다.
- Acceptance 의 limit 은 식 (9) 대로 tolerance 에서 guard band 를 뺀 값이며, 현재 예측에서 $`148.3056 - 19.8207 = 128.4850`$ 이다.
- 측정값 평균이 target 과 같으면 ($`y_0 = 634.5735`$) 공정 요건은 113.7321 로 줄어 예측 요건 120.4904 가 tolerance 를 정하고, spec 은 634.5735 ± 120.4904 이다.
- $`P_{\max} = 0.1`$ 이면 예측 요건은 3 배가 되어 현재 예측에서 361.4712 이고, 예측 요건이 tolerance 를 정한다.

### 4.3 Wafer Uniformity Spec

식 (10) 부터 식 (13) 로 구한 wafer uniformity spec 은 아래 Table 4 이며, 값은 모두 % 이다.

Table 4. Wafer uniformity spec from the RMSE

| Quantity                    | Current prediction | Offset removed  |
| :-------------------------: | :----------------: | :-------------: |
| Spec (target + tolerance)   | 2.4962 + 7.7251    | 2.4962 + 7.9800 |
| Acceptance (target + limit) | 2.4962 + 6.4543    | 2.4962 + 6.6673 |
| Process requirement         | 3.9785             | 3.9785          |
| Prediction requirement      | 7.7251             | 7.9800          |
| Governed by                 | Prediction         | Prediction      |
| $`\sigma_U`$                | 0.9971             | 0.9971          |
| $`\mathrm{RMSE}_U`$         | 0.7725             | 0.7980          |
| Guard band $`g`$            | 1.2708             | 1.3127          |
| Largest $`U_i`$             | 6.3171             | 6.3171          |

Tolerance 의 근거는 아래이다.

- Target 2.4962 % 는 측정한 $`U_i`$ 의 평균 $`\bar{U}`$ 이다. Target 과 평균이 같아 공정 요건에 더할 폭이 없다.
- Tolerance 7.7251 % 는 식 (13) 에서 공정 요건과 예측 요건 가운데 큰 예측 요건 $`3 \times 0.7725 / 0.3`$ 이다.
- 공정 요건은 $`3 \times 1.33 \times 0.9971 = 3.9785`$ 로, 측정값의 산포만 보면 tolerance 는 이 값으로 충분하다.
- Acceptance 의 limit 은 tolerance 에서 guard band 를 뺀 $`7.7251 - 1.2708 = 6.4543`$ 이다.
- $`\mathrm{RMSE}_U / \sigma_U`$ 는 0.7747 로 thickness 의 0.4227 보다 크다. Wafer 사이에서 $`U_i`$ 가 달라지는 정도에 견주어 그 예측 오차가 크다는 뜻이다.
- 예측 요건이 공정 요건 이하가 되는 $`\mathrm{RMSE}_U`$ 는 $`0.399 \times 0.9971 = 0.3979`$ %p 이고, 현재 예측은 그 1.94 배이다.
- Upper spec limit $`\mathrm{USL}_U = 2.4962 + 7.7251 = 10.2213`$ % 는 측정한 $`U_i`$ 의 최댓값 6.3171 % 보다 높다. 현재 예측으로는 10.2213 % 보다 좁은 uniformity spec 을 판정할 수 없다.
- Offset 을 없애면 thickness RMSE 는 27 % 줄지만 $`\mathrm{RMSE}_U`$ 는 0.7725 에서 0.7980 으로 오른다. 측정 위치마다 wafer 44 장에 공통인 값을 더하는 보정은 wafer 사이에서 달라지는 $`U_i`$ 의 오차를 줄이지 않는다.

### 4.4 Calculation Script

Table 2 부터 Table 4 의 값은 같은 folder 의 [src/rmse_process_spec.py](src/rmse_process_spec.py) 로 구했다. 이 script 는 CSV 에서 식 (1) 의 RMSE 와 식 (12) 의 $`\mathrm{RMSE}_U`$ 를 구하고, `--target` 으로 받은 thickness target 에 대해 현재 예측과 offset 을 없앤 예측마다 spec limit 과 acceptance limit 을 출력한다. `--output-folder` 를 주면 wafer 마다의 $`U_i`$ 와 $`\hat{U}_i`$ 를 `uniformity_by_wafer.csv` 로 저장한다. 문서 folder 에서 아래처럼 실행한다.

```bash
python3 src/rmse_process_spec.py --csv data/u0_actuals_and_predictions.csv --target 600 --cp-min 1.33 --pt-max 0.3 --guard-k 1.645
```

## 5. Application

### 5.1 Conditions

이 방법은 RMSE 를 구한 wafer 와 판정할 wafer 의 예측 오차 분포가 같을 때 쓴다.

- 측정값과 예측값이 같은 wafer 의 같은 측정 위치 $`N`$ 곳에 있다.
- RMSE 를 구한 wafer 가 판정할 기간의 장비, recipe, 제품을 대표한다.
- $`C_{\min}`$, $`P_{\max}`$, $`k`$ 는 공정과 품질 담당이 정해 입력한다.
- 공정 요건이 tolerance 를 정하는 경우 (식 (7) 이 성립하지 않는 경우) 에는 spec 이 측정값과 target 만으로 정해지고, RMSE 는 guard band 에만 쓰인다.

### 5.2 Breaking Conditions

아래 경우에는 Table 3 과 Table 4 의 spec 과 acceptance limit 을 그대로 믿을 수 없다.

- **Offset 이 있는 예측 오차**: 측정 위치마다 mean error (offset) 가 있으면 오차의 평균이 0 이 아니어서, 식 (9) 의 guard band 가 spec 밖 확률 0.05 를 보장하지 않는다. 이 데이터는 offset 을 없애면 RMSE 가 12.0490 에서 8.7503 으로 줄어, offset 이 RMSE 에 큰 비율로 들어 있다.
- **정규분포가 아닌 wafer uniformity**: $`U_i`$ 의 최댓값 6.3171 % 는 $`\bar{U}`$ 보다 $`\sigma_U`$ 의 3.83 배 크다. 정규분포에서 평균보다 표준편차의 3.83 배 넘게 클 확률은 약 0.00006 인데 wafer 44 장 가운데 하나가 그만큼 크므로, $`U_i`$ 는 오른쪽 꼬리가 긴 분포를 따른다. 이때 $`C_{pu}`$ 로 구한 spec 밖 비율은 실제보다 낮다.
- **표본 수**: RMSE, $`\sigma_y`$, $`\sigma_U`$ 는 wafer 44 장으로 구한 추정값이며, 공정능력 추정에는 일반적으로 표본 100 개 이상이 필요하다 [[1](#ref-1)]. Wafer uniformity 는 wafer 한 장에 값이 하나여서 표본이 44 개이다.
- **예측 model 의 변경**: model 을 다시 학습하거나 공정 조건이 바뀌면 RMSE 를 다시 구하고 spec 을 다시 계산한다.

## References

<a id="ref-1"></a>[1] NIST/SEMATECH. [6.1.6. What is Process Capability?](https://www.itl.nist.gov/div898/handbook/pmc/section1/pmc16.htm). *NIST/SEMATECH e-Handbook of Statistical Methods*.

---

## Appendix A. Terminology

- **acceptance limit**: 예측값으로 합격을 판정하는 한계. Spec limit 을 guard band 만큼 안쪽으로 좁힌 값
- **guard band**: 예측 오차 때문에 spec limit 을 안쪽으로 좁히는 폭. 이 문서에서는 $`k\, \mathrm{RMSE}`$
- **mean error (offset)**: 한 측정 위치에서 예측 오차를 wafer 에 대해 평균한 값
- **P/T**: precision-to-tolerance ratio. 측정 system 표준편차의 6 배를 spec 폭 $`\mathrm{USL} - \mathrm{LSL}`$ 으로 나눈 값
- **process capability index**: 공정능력지수. Spec limit 까지의 거리를 공정 산포로 나눈 값. 양쪽 spec 은 $`C_p`$ 와 $`C_{pk}`$, upper spec 하나는 $`C_{pu}`$ 를 쓴다
- **RMSE**: root mean squared error. 예측 오차 제곱 평균의 제곱근
- **spec limit**: 공정 결과가 합격인 범위의 경계. Upper spec limit (USL) 과 lower spec limit (LSL)
- **target ± tolerance**: spec 을 중심값 target 과 그 중심에서 spec limit 까지의 거리 tolerance 로 적는 꼴
- **wafer uniformity**: wafer 한 장의 측정값 표준편차를 그 평균으로 나눈 값 (1σ/mean). 이 문서에서는 % 로 적는다
