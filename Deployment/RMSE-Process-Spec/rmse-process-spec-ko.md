# Tolerance at Target from Thickness Prediction
Version 39 | Created: 2026-11-16 | Updated: 2026-11-16 12:18 UTC

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Principle](#3-principle)
    - [3.1 Notation](#31-notation)
    - [3.2 Prediction Error](#32-prediction-error)
    - [3.3 Measured Value at Target](#33-measured-value-at-target)
- [4. Worked Example](#4-worked-example)
    - [4.1 Data](#41-data)
    - [4.2 Result](#42-result)
    - [4.3 Calculation Script](#43-calculation-script)
- [5. Application](#5-application)
    - [5.1 Conditions](#51-conditions)
    - [5.2 Breaking Conditions](#52-breaking-conditions)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

**Problem Statement**: 반도체 wafer에 증착한 박막의 두께를 예측한 데이터로 부터, 예측이 목표값 일떄의 tolerance 와 stdev를 구하는 방법이 정리되어 있지 않다.

**Goal**: 박막 두께의 예측값과 측정값의 짝으로 아래 두 가지를 구한다.

1. 예측값이 target 일 때 측정값의 stdev
2. 예측값이 target 일 때 측정값의 tolerance, 곧 측정값이 들어갈 범위

**Non-Goal**:

- 측정값이 없는 예측값만으로 구하는 방법은 다루지 않는다. 예측값과 측정값의 짝이 있어야 한다.
- 예측 model 을 만들거나 고치는 방법은 다루지 않는다.
- 공정 spec 이나 공정능력지수를 정하는 방법은 다루지 않는다.

## 2. Summary

예측값을 입력, 측정값을 출력으로 둔 회귀 직선 $`y = a + \beta x`$ 에서, 예측값이 target $`x_0`$ 일 때 측정값의 평균은 $`\mu_0 = a + \beta x_0`$ 이다.
그때 측정값의 stdev 는 잔차 (residual) 의 표준편차 $`s`$ 에 회귀 직선의 불확도를 더한 $`s_p`$ 이고, tolerance 는 $`h = k\, s_p`$ 로 측정값은 $`\mu_0 \pm h`$ 안에 든다.
$`\beta = 1`$ 이고 예측 오차의 평균이 0 이면 $`\mu_0 = x_0`$ 이고, $`s_p`$ 는 예측 오차의 표준편차와 거의 같다.

## 3. Principle

### 3.1 Notation

이 문서의 기호는 아래 Table 1 이다.

Table 1. Symbols

| Symbol       | Meaning                                                                    | Shape    | Assumption        |
| :----------: | :------------------------------------------------------------------------: | :------: | :---------------: |
| $`T`$        | 예측값과 측정값의 짝의 수. wafer 수와 wafer 한 장의 측정 위치 수를 곱한 값 | integer  | $`T \ge 3`$       |
| $`i`$        | 짝의 번호                                                                  | integer  | $`1 \le i \le T`$ |
| $`x_i`$      | $`i`$ 번째 짝의 예측값                                                     | scalar   | -                 |
| $`y_i`$      | $`i`$ 번째 짝의 측정값                                                     | scalar   | -                 |
| $`e_i`$      | 예측 오차 $`y_i - x_i`$                                                    | scalar   | -                 |
| $`b_e`$      | 예측 오차 $`T`$ 개의 평균 (bias)                                           | scalar   | -                 |
| $`\sigma_e`$ | 예측 오차 $`T`$ 개의 표준편차 (분모 $`T`$)                                 | scalar   | -                 |
| $`\bar{x}`$  | 예측값 $`T`$ 개의 평균                                                     | scalar   | -                 |
| $`\bar{y}`$  | 측정값 $`T`$ 개의 평균                                                     | scalar   | -                 |
| $`S_{xx}`$   | 예측값이 그 평균에서 벗어난 제곱합                                         | scalar   | $`S_{xx} \gt 0`$  |
| $`a`$        | 회귀 직선의 절편                                                           | scalar   | -                 |
| $`\beta`$    | 회귀 직선의 기울기                                                         | scalar   | -                 |
| $`s`$        | 잔차의 표준편차 (분모 $`T - 2`$)                                           | scalar   | -                 |
| $`x_0`$      | 예측값의 target                                                            | scalar   | -                 |
| $`\mu_0`$    | 예측값이 $`x_0`$ 일 때 측정값의 평균                                       | scalar   | -                 |
| $`s_p`$      | 예측값이 $`x_0`$ 일 때 측정값의 stdev                                      | scalar   | $`s_p \gt 0`$     |
| $`k`$        | coverage factor                                                            | scalar   | $`k \gt 0`$       |
| $`h`$        | tolerance. $`\mu_0`$ 에서 측정값 범위의 경계까지의 거리                    | scalar   | $`h \gt 0`$       |
| $`\Phi`$     | 표준정규분포의 누적분포함수                                                | function | -                 |

### 3.2 Prediction Error

예측값과 측정값의 차이인 예측 오차 $`e_i = y_i - x_i`$ 의 평균과 표준편차는 모든 예측값을 한데 모은 오차의 크기이다.

```math
b_e = \frac{1}{T} \sum_{i=1}^{T} e_i
\hspace{19em} (1)
```

```math
\sigma_e = \sqrt{\frac{1}{T} \sum_{i=1}^{T} \left( e_i - b_e \right)^2}
\hspace{19em} (2)
```

$`b_e`$ 와 $`\sigma_e`$ 는 예측값이 어떤 값이든 오차의 평균과 퍼짐이 같다고 볼 때 쓸 수 있다. 예측값이 target 일 때의 측정값을 구하려면, 측정값이 예측값에 따라 어떻게 달라지는지를 [꼭지 3.3](#33-measured-value-at-target) 의 회귀 직선으로 함께 본다.

### 3.3 Measured Value at Target

예측값 $`x_i`$ 를 입력, 측정값 $`y_i`$ 를 출력으로 두고 짝 $`T`$ 개에 맞는 직선 $`y = a + \beta x`$ 를 구한다. 측정값은 바꾸지 않으며, 이 직선으로 예측값이 어떤 값일 때 측정값이 평균적으로 얼마인지를 읽는다. 직선의 기울기, 절편, 잔차의 표준편차는 아래이다.

```math
S_{xx} = \sum_{i=1}^{T} (x_i - \bar{x})^2
\hspace{19em} (3)
```

```math
\beta = \frac{1}{S_{xx}} \sum_{i=1}^{T} (x_i - \bar{x})(y_i - \bar{y})
\hspace{19em} (4)
```

```math
a = \bar{y} - \beta\, \bar{x}
\hspace{19em} (5)
```

```math
s = \sqrt{\frac{1}{T - 2} \sum_{i=1}^{T} \left( y_i - a - \beta x_i \right)^2}
\hspace{19em} (6)
```

예측값이 target $`x_0`$ 일 때 측정값의 평균은 회귀 직선의 값이다.

```math
\mu_0 = a + \beta\, x_0
\hspace{19em} (7)
```

그때 측정값의 stdev 는 측정값 하나의 퍼짐 $`s`$ 와, 회귀 직선을 데이터로 추정해서 생기는 불확도를 제곱합으로 합한 값이다 [[1](#ref-1)].

```math
s_p = s \sqrt{1 + \frac{1}{T} + \frac{(x_0 - \bar{x})^2}{S_{xx}}}
\hspace{19em} (8)
```

Tolerance 는 stdev 에 coverage factor $`k`$ 를 곱한 값이며, 예측값이 $`x_0`$ 일 때 측정값은 $`\mu_0 \pm h`$ 안에 든다.

```math
h = k\, s_p
\hspace{19em} (9)
```

잔차가 정규분포를 따르면 측정값이 $`\mu_0 \pm h`$ 안에 들 확률은 $`1 - 2\left(1 - \Phi(k)\right)`$ 이고, $`k = 1.96`$ 이면 0.95 이다.

- $`\beta = 1`$ 이면 식 (5) 에서 $`a = \bar{y} - \bar{x} = b_e`$ 이어서 $`\mu_0 = x_0 + b_e`$ 이고, $`T`$ 가 크면 $`s_p`$ 는 식 (2) 의 $`\sigma_e`$ 와 거의 같다.
- $`\beta \ne 1`$ 이면 예측값에 따라 오차의 평균이 달라지므로, target 의 측정값은 식 (1) 과 식 (2) 대신 식 (7) 과 식 (8) 로 구한다.
- 식 (8) 의 셋째 항은 $`x_0`$ 이 예측값의 평균 $`\bar{x}`$ 에서 멀수록 커진다. $`x_0`$ 이 예측값의 범위 밖이면 직선이 그 범위에서도 맞는지 데이터로 확인할 수 없다.

## 4. Worked Example

### 4.1 Data

반도체 wafer 44 장의 측정 위치 13 곳에서 측정한 박막 두께와 그 예측값 ([data/u0_actuals_and_predictions.csv](data/u0_actuals_and_predictions.csv)) 을 쓴다. 측정값은 `true_p1` 부터 `true_p13`, 예측값은 `pred_p1` 부터 `pred_p13` 열이며, 측정 위치를 가리지 않고 짝 $`T = 572`$ 개를 한데 모은다. 단위는 nm 이고, target 은 $`x_0 = 600`$ nm, coverage factor 는 $`k = 1.96`$ 으로 두었다.

### 4.2 Result

식 (1) 부터 식 (9) 로 구한 값은 아래 Table 2 이며, $`\beta`$ 를 빼고 모두 nm 이다.

Table 2. Measured value when the prediction equals the target 600 nm

| Quantity     | Value   | Quantity    | Value    |
| :----------: | :-----: | :---------: | :------: |
| $`b_e`$      | -2.3737 | $`\mu_0`$   | 603.2369 |
| $`\sigma_e`$ | 11.8129 | $`s_p`$     | 10.8704  |
| $`a`$        | 94.3506 | $`h`$       | 21.3059  |
| $`\beta`$    | 0.8481  | Lower bound | 581.9310 |
| $`s`$        | 10.8475 | Upper bound | 624.5429 |

- 예측값이 600 nm 이면 측정값의 stdev 는 $`s_p = 10.8704`$ nm 이고, 측정값은 $`603.2369 \pm 21.3059`$ nm, 곧 581.9310 nm 부터 624.5429 nm 사이에 확률 0.95 로 든다.
- $`\beta = 0.8481`$ 로 1 보다 작아, 측정값은 예측값만큼 넓게 변하지 않는다. 예측값의 평균 636.9473 nm 보다 낮은 600 nm 에서는 측정값의 평균 603.2369 nm 가 target 보다 3.2369 nm 높다.
- 모든 예측값을 한데 모은 $`b_e = -2.3737`$ nm 로 600 nm 의 측정값 평균을 잡으면 $`600 - 2.3737 = 597.6263`$ nm 로, 회귀 직선의 값과 5.6106 nm 차이가 난다.
- $`s_p / s = 1.0021`$ 로, 식 (8) 의 둘째와 셋째 항은 $`T = 572`$ 에서 거의 영향이 없다.
- 예측값이 600 nm 에서 10 nm 이내인 짝 60 개의 측정값은 평균 603.6948 nm, stdev 12.2185 nm 로, 회귀로 구한 $`\mu_0`$ 와 $`s_p`$ 에 가깝다.

### 4.3 Calculation Script

Table 2 의 값은 같은 folder 의 [src/target_tolerance.py](src/target_tolerance.py) 로 구했다. 이 script 는 CSV 의 짝을 한데 모아 식 (1) 부터 식 (9) 의 값을 출력하고, `--window` 로 정한 폭 안에서 예측값이 target 에 가까운 짝의 측정값 평균과 stdev 를 함께 출력한다. 문서 folder 에서 아래처럼 실행한다.

```bash
python3 src/target_tolerance.py --csv data/u0_actuals_and_predictions.csv --target 600 --k 1.96 --window 10
```

## 5. Application

### 5.1 Conditions

이 방법은 짝을 구한 wafer 와 앞으로 예측할 wafer 의 예측 오차가 같은 성질을 가질 때 쓴다.

- 예측값과 측정값이 같은 wafer 의 같은 측정 위치에서 짝을 이룬다.
- Target 이 예측값의 범위 안에 있다. 이 데이터의 예측값은 497.5943 nm 부터 731.5830 nm 이다.
- 짝을 구한 wafer 가 예측할 기간의 장비, recipe, 제품을 대표한다.

### 5.2 Breaking Conditions

아래 경우에는 $`s_p`$ 와 $`h`$ 를 그대로 믿을 수 없다.

- **Wafer 안의 상관**: 같은 wafer 의 측정 위치 13 곳은 서로 독립이 아니다. 식 (8) 은 짝 $`T`$ 개가 독립이라고 보므로 회귀 직선의 불확도를 작게 잡는다.
- **예측값에 따라 다른 퍼짐**: 잔차의 퍼짐이 예측값의 크기에 따라 다르면 $`s`$ 하나로 target 의 stdev 를 정할 수 없다. 이때는 target 근처 짝의 stdev 를 함께 견준다.
- **정규분포가 아닌 잔차**: 잔차가 정규분포를 따르지 않으면 $`\mu_0 \pm h`$ 안에 들 확률이 $`1 - 2(1 - \Phi(k))`$ 와 다르다.
- **직선이 아닌 관계**: 측정값과 예측값의 관계가 직선이 아니면 식 (7) 의 $`\mu_0`$ 가 target 에서 측정값의 평균과 다르다.

## References

<a id="ref-1"></a>[1] NIST/SEMATECH. [4.5.1.2. How can I predict the value and estimate the uncertainty of a single response?](https://www.itl.nist.gov/div898/handbook/pmd/section5/pmd512.htm). *NIST/SEMATECH e-Handbook of Statistical Methods*.

---

## Appendix A. Terminology

- **bias**: 예측 오차의 평균. 이 문서에서는 $`b_e`$
- **coverage factor**: stdev 에 곱해 tolerance 를 정하는 수. 이 문서에서는 $`k`$
- **residual**: 잔차. 측정값에서 회귀 직선의 값을 뺀 값
- **tolerance**: 예측값이 target 일 때 측정값의 평균에서 측정값 범위의 경계까지의 거리. 이 문서에서는 $`h`$
