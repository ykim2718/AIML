# Centered R² vs Uncentered R² (Korean)
Rev. 2 | Created: 2026-09-05 | Updated: 2026-09-05 21:47 CDT

## 1. Introduction: R² and Its Relation to RSQ

결정계수 (coefficient of determination), 곧 R² 는 예측값 $\hat{y}$ 가 관측값
$y_{true}$ 와 얼마나 잘 맞는지를 재는 지표로, 통계와 machine learning (ML) 에서
가장 널리 쓰이는 검증 지표 가운데 하나이다. Spreadsheet software 와 많은
engineering 보고서에서 R² 는 흔히 **RSQ** (Excel 함수 이름 `RSQ()`) 로 불리며,
이는 두 변수 사이의 Pearson 상관계수를 제곱한 값을 계산한다. Excel 의 기본 RSQ 와
대부분의 회귀 library 가 내놓는 표준 R² 는 같은 양, 곧 **centered R²** 를
가리키며, 이는 $y$ 의 변동 가운데 model 이 평균 baseline 대비 설명한 비율을 잰다.

그런데 덜 다루어지는 변형인 **uncentered R²** 는 평균 기준의 baseline 을 0 기준의
baseline 으로 바꾼다. 이 글은 둘을 견주고, uncentered 식을 유도하고, 기하학적
해석을 주고, 각각이 언제 알맞은지를 다루며, Python 구현은
[Appendix B](#appendix-b-python-code) 에 모았다. 특히 같은 물리량끼리 견주는
자리에서 $y_{true}$ 와 $\hat{y}$ 의 1:1 line 일치를 평가하는 데 uncentered R² 가
더 적합한지를 살핀다
[[3](#ref-3)][[4](#ref-4)].

## 2. Comparison of the Two Formulas

$$R^2_c = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

$$R^2_u = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n y_i^2}$$

- 차이는 온전히 분모에 있다. Centered R² 는 "평균 둘레의 변동을 얼마나
  설명했는가" 를 묻고, uncentered R² 는 "0 둘레의 전체 제곱합을 얼마나
  설명했는가" 를 묻는다. Centered R² 는 평균 예측을 baseline 으로 삼고,
  uncentered R² 는 0 예측을 baseline 으로 삼는다 [[3](#ref-3)].
- Centered R² 가 묻는 것: "평균을 그대로 내놓는 baseline 보다 model 이 얼마나
  나은가?"
- Uncentered R² 가 묻는 것: "model 의 예측이 1:1 identity line (y=x) 에 얼마나
  가까운가?"

## 3. Pros and Cons

### 3.1 Centered R²

- **Pros**: 척도와 단위에 무관하여 서로 다른 물리량의 model 을 견줄 수 있다.
  Pearson 상관계수의 제곱과 같으므로 통계적 직관이 풍부하다.
- **Cons**: $y$ 의 분산이 작으면 (거의 일정하면) 분모가 0 에 가까워져 R² 가
  무너지거나 음수가 된다. 평균 예측기 대비 상대 성능만 재므로 계통적 bias 에
  둔감하다.

### 3.2 Uncentered R²

- **Pros**: 같은 물리량 수준에서 model, dataset, 실험을 견줄 때 uncentered R² 는
  일관된 절대 baseline (0 예측) 을 주어 서로 견주는 일이 어긋나지 않게 한다. 저
  분산 자료에 강건하다. 1:1 line 에서 벗어나는 것에 민감하여 계통적 bias 를 제대로
  벌한다 [[4](#ref-4)].
- **Cons**: 척도에 종속되므로 단위나 값 범위가 다른 model 을 견주는 데는 맞지
  않는다. $\bar{y}$ 가 크면 $R^2_u$ 는 인위적으로 1 에 가까워진다. Uncentered
  $R^2$ 값은 대체로 훨씬 높게 (흔히 0.99 이상) 나오는데, 분모 $\sum y_i^2$ 이
  centered 분모 $\sum (y_i - \bar{y})^2$ 보다 일반적으로 훨씬 크기
  때문이다. 이것만 떼어 읽으면 지나치게 낙관적인 평가로 이어진다. 다만 min-max
  정규화된 자료에서는 척도가 고르므로 uncentered $R^2$ 가 관측값과 예측값 사이의
  절대 일치를 재는 데 더 나은 지표가 된다.

**Key question**: 같은 물리량의 자료에서 model 을 견줄 때 uncentered R² 가 실제로
더 정확한가? 답은 사실상 "그렇다" 이다. 같은 단위의 자료에서 $y_{true}$ 와
$\hat{y}$ 의 절대 일치를 평가할 때 uncentered R² 는 (1) dataset 분산이 달라져도
왜곡되지 않고, (2) 일정한 bias 에 민감하며, (3) dataset 사이에 절대 baseline (0
예측) 을 공유하여 model 끼리 일관되게 견줄 수 있다.

## 4. Applications

### 4.1 Centered R²

- 일반적인 회귀와 ML modeling 의 표준 성능 지표.
- 단위나 영역이 다른 model 의 성능을 정규화된 형태로 견주는 일.
- Pearson 상관에 바탕을 둔 변동 설명력 평가
  [[1](#ref-1)].

### 4.2 Uncentered R²

- **Regression Through the Origin (RTO)**: Hooke 의 법칙이나 방사성 붕괴처럼
  물리적으로 0 을 지나야 하는 model. Stata, R, EViews 의 `noconstant` option 은
  자동으로 uncentered R² 를 내놓는다
  [[2](#ref-2)]. 절편을 0 으로 고정하면 표준 (centered) $R^2$ 는
  음수가 되거나 잘못된 해석을 낳는다. 이런 자리에서 uncentered $R^2$ 는 원점
  기준으로 model 의 적합을 평가하는 강건한 대안으로 쓰인다.
- **계량경제학의 진단 검정**: Breusch-Pagan 검정과 White 검정의 보조 회귀는
  $nR^2$ 를 chi-square 통계량으로 쓰는데, 이때의 계산이
  uncentered 형태이다 [[6](#ref-6)].
- **$Y_{true}$ 대 $Y_{pred}$ 그림에서 1:1 line 이탈 평가**:
  같은 물리량 자료를 다루는 ML 예측 과제에서 uncentered R² 는
  절대 일치를 잰다. Calibration, Computational Fluid
  Dynamics (CFD) 와 Finite Element Method (FEM) 검증, sensor 교정,
  재현성 연구에 알맞다.

## 5. Derivation of Uncentered R²

### 5.1 Sum of Squares Decomposition

관측값 $y_i$, 예측값 $\hat{y}_i$, 잔차
$e_i = y_i - \hat{y}_i$ ($i = 1, \ldots, n$) 에 대하여 uncentered 제곱합인
Total Sum of Squares (TSS), Explained Sum of Squares (ESS),
Residual Sum of Squares (RSS) 를 다음과 같이 정의한다.

$$TSS_u = \sum_{i=1}^n y_i^2, \quad ESS_u = \sum_{i=1}^n \hat{y}_i^2, \quad RSS = \sum_{i=1}^n e_i^2$$

### 5.2 Ordinary Least Squares (OLS) Orthogonality Condition

OLS 정규방정식 $X^T(y - X\hat{\beta}) = 0$ 에서, 설계행렬 $X$ 의 모든 열은 잔차
vector $e$ 와 직교한다.

$$\sum_{i=1}^n \hat{y}_i \cdot e_i = \hat{\beta}^T X^T e = 0$$

적합값 vector 와 잔차 vector 는 직교한다. 이는 model 에 절편이 있든 없든
성립한다.

### 5.3 Uncentered Decomposition

$y_i = \hat{y}_i + e_i$ 를 제곱하여 $i$ 에 대해 더하면 다음과 같다.

$$\sum y_i^2 = \sum \hat{y}_i^2 + 2\sum \hat{y}_i e_i + \sum e_i^2$$

교차항은 직교성 $\sum \hat{y}_i e_i = 0$ 으로 사라지고, 다음이 남는다.

$$\sum y_i^2 = \sum \hat{y}_i^2 + \sum e_i^2 \quad \Longleftrightarrow \quad TSS_u = ESS_u + RSS$$

### 5.4 Definition of Uncentered R²

양변을 $TSS_u$ 로 나누면 $1 = ESS_u/TSS_u + RSS/TSS_u$ 가 되고, uncentered R² 는
설명된 비율로 정의된다.

$$R^2_u \equiv \frac{ESS_u}{TSS_u} = \frac{\sum \hat{y}_i^2}{\sum y_i^2} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum y_i^2}$$

### 5.5 Relation to Centered R²

$\sum y_i^2 = \sum (y_i - \bar{y})^2 + n\bar{y}^2$ 이므로
$TSS_u = TSS_c + n\bar{y}^2$ 이다. 절편이 있는 model 에서는 다음이 성립한다.

$$R^2_u = \frac{TSS_c \cdot R^2_c + n\bar{y}^2}{TSS_c + n\bar{y}^2}$$

$\bar{y}$ 가 크거나 $n$ 이 크면 $R^2_u$ 는 $R^2_c$ 보다 1 에 가까워진다.
$\bar{y} = 0$ 이면 둘은 일치한다.

## 6. Geometric Interpretation

### 6.1 Vector Space Setup

$n$ 개의 관측을 vector
$\mathbf{y}, \hat{\mathbf{y}}, \mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} \in \mathbb{R}^n$
로 다루면 모든 제곱합이 norm 의 제곱이 된다: $\sum y_i^2 = \|\mathbf{y}\|^2$,
$\sum \hat{y}_i^2 = \|\hat{\mathbf{y}}\|^2$, $\sum e_i^2 = \|\mathbf{e}\|^2$.

### 6.2 OLS as Orthogonal Projection

설계행렬 $X$ 의 열공간을 $\mathcal{C}(X)$ 라 하자. OLS 추정값은
$\hat{\mathbf{y}} = P_X \mathbf{y}$ 이고 여기서
$P_X = X(X^TX)^{-1}X^T$ 는 $\mathbf{y}$ 를 $\mathcal{C}(X)$ 위로 내리는 직교
사영이다. 잔차 $\mathbf{e}$ 는 직교여공간에 놓이므로
$\mathbf{e} \perp \hat{\mathbf{y}}$ 가 저절로 성립한다 [[5](#ref-5)].

### 6.3 Uncentered Decomposition as the Pythagorean Theorem

$\mathbf{y} = \hat{\mathbf{y}} + \mathbf{e}$ 이고
$\hat{\mathbf{y}} \perp \mathbf{e}$ 이므로 직각삼각형이 만들어진다.

```text
              y
             /|
            / |
           /  |  e  (perpendicular residual)
          /   |
         /____|
      origin  ŷ  (on the C(X) plane)
```

$$\|\mathbf{y}\|^2 = \|\hat{\mathbf{y}}\|^2 + \|\mathbf{e}\|^2$$

이것이 $TSS_u = ESS_u + RSS$ 뒤에 있는 기하학적 항등식이며, 단순한 대수적
항등식이 아니라 문자 그대로 피타고라스 정리이다. Uncentered R² 의 기하학적 뜻은
다음이 된다.

$$R^2_u = \frac{\|\hat{\mathbf{y}}\|^2}{\|\mathbf{y}\|^2} = \cos^2\theta_u$$

여기서 $\theta_u$ 는 원점에서 잰 $\mathbf{y}$ 와 $\hat{\mathbf{y}}$ 사이의
각이다. $R^2_u \to 1$ 은 두 vector 가 같은 방향을 가리킨다는 뜻이고,
$R^2_u \to 0$ 은 둘이 직교한다는 뜻이다.

### 6.4 Centered R²: Subtracting the Mean Vector

평균 vector $\bar{\mathbf{y}} = \bar{y} \cdot \mathbf{1}$ 을 빼는 것은
초평면 $\mathbf{1}^\perp$ 위로 사영하는 것과 기하학적으로 같으며, 상수 vector
$\mathbf{1}$ 방향의 성분을 없앤다. Model 에 절편이 있으면
$\mathbf{1} \in \mathcal{C}(X)$ 이므로 $\mathbf{1}^\perp$ 안에서 직각삼각형이
그대로 만들어진다.

$$R^2_c = \frac{\|\tilde{\hat{\mathbf{y}}}\|^2}{\|\tilde{\mathbf{y}}\|^2} = \cos^2\theta_c$$

여기서 $\theta_c$ 는 평균을 없앤 뒤 두 vector 사이의 각이며, Pearson 상관계수의
제곱과 같다.

### 6.5 Geometric Distinction Between the Two R²

```text
              y
             /|
            / |
           /  |
          /   | e
         /    |
        /     |
       *------ŷ-----→ (C(X) plane)
   origin
     /
    *────────── 1 direction (constant vector)
    ȳ·1
```

- **Uncentered R²**: 원점에서 잰 $\mathbf{y}$ 와 $\hat{\mathbf{y}}$ 사이의
  각이다.
- **Centered R²**: 기준점을 $\bar{y}\cdot\mathbf{1}$ 로 옮긴 뒤에 잰
  각이다.

근본적인 차이는 기준점이 원점인가 평균 vector 인가에 있다. 같은 자료라도 기준점을
달리 하여 보면 두 vector 사이의 각이 달라진다.

### 6.6 Comparison Table

Table 1. Geometric distinction between uncentered and centered R²

| Aspect | Uncentered | Centered |
|---|---|---|
| Reference point | 원점 0 | 평균 vector ȳ·1 |
| Right triangle | y, ŷ, e | ỹ, ŷ̃, e |
| Angle interpretation | cos²(∠yŷ) | cos²(∠ỹŷ̃) = r² |
| Required condition | e ⊥ ŷ | e ⊥ ŷ 이고 1 ∈ C(X) |
| Sensitivity to bias | 민감함 (원점 고정) | 둔감함 (평균 이동에 불변) |

**1:1 line 검증의 기하학적 뜻**: 일정한 bias
$\hat{\mathbf{y}} = \mathbf{y} + c\mathbf{1}$ 아래에서 uncentered 각은 벌어지지만
centered 각은 그대로이다. 따라서 절대 일치를 평가할 때 uncentered R² 가 더
민감하게 반응한다.

## References

<a id="ref-1"></a>[1] Draper, N. R., & Smith, H. (1998). [*Applied Regression Analysis*](https://doi.org/10.1002/9781118625590) (3rd ed.). Wiley.<br>
<a id="ref-2"></a>[2] Eisenhauer, J. G. (2003). [Regression through the origin](https://doi.org/10.1111/1467-9639.00136). *Teaching Statistics*, 25(3), 76–80.<br>
<a id="ref-3"></a>[3] Kvalseth, T. O. (1985). [Cautionary note about R²](https://doi.org/10.1080/00031305.1985.10479448). *The American Statistician*, 39(4), 279–285.<br>
<a id="ref-4"></a>[4] Legates, D. R., & McCabe, G. J. (1999). [Evaluating the use of "goodness-of-fit" measures in hydrologic and hydroclimatic model validation](https://doi.org/10.1029/1998WR900018). *Water Resources Research*, 35(1), 233–241.<br>
<a id="ref-5"></a>[5] Strang, G. (2009). [*Introduction to Linear Algebra*](https://wellesleycambridge.com/) (4th ed.). Wellesley-Cambridge Press. ISBN 978-0-9802327-1-4.<br>
<a id="ref-6"></a>[6] Wooldridge, J. M. (2010). [*Econometric Analysis of Cross Section and Panel Data*](https://mitpress.mit.edu/9780262232586/econometric-analysis-of-cross-section-and-panel-data/) (2nd ed.). MIT Press. ISBN 978-0-262-23258-6.

---

## Appendix A. Terminology

- **centered R²**: 표준 R² 이며, 분모가 자료 평균 둘레의 제곱합인 것.
- **CFD**: Computational Fluid Dynamics. 유체 흐름의 수치 모사.
- **ESS**: Explained Sum of Squares. Uncentered 형태에서는 $\sum \hat{y}_i^2$.
- **FEM**: Finite Element Method. 이산화된 영역에서 장 문제를 수치로 푸는 방법.
- **MAE**: Mean Absolute Error. 잔차 절댓값의 평균.
- **MAPE**: Mean Absolute Percentage Error. 잔차 절댓값을 관측값에 대한 백분율로 적은 것의 평균. 관측값이 0 인 자리에서는 정의되지 않는다.
- **OLS**: Ordinary Least Squares. 잔차 제곱합을 최소로 하는 추정량.
- **RMSE**: Root Mean Squared Error. 잔차 제곱 평균의 제곱근.
- **RSQ**: Pearson 상관계수의 제곱을 뜻하는 Excel 함수 이름이며, centered R² 와 같다.
- **RSS**: Residual Sum of Squares. $\sum e_i^2$.
- **RTO**: Regression Through the Origin. 절편을 0 으로 고정한 회귀.
- **TSS**: Total Sum of Squares. Uncentered 형태에서는 $\sum y_i^2$, centered 형태에서는 $\sum (y_i - \bar{y})^2$.
- **uncentered R²**: 분모가 평균이 아니라 0 둘레의 제곱합인 R².

## Appendix B. Python Code

### B.1 Centered R²

```python
import numpy as np


def centered_r2(y_true, y_pred):
    """
    Centered R² (standard R²)
    R²_c = 1 - SSE / SST_centered
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    sse = np.sum((y_true - y_pred) ** 2)
    sst_c = np.sum((y_true - np.mean(y_true)) ** 2)

    if sst_c == 0:
        return np.nan  # undefined when y has zero variance
    return 1.0 - sse / sst_c

# Example usage
y_true = np.array([10.1, 10.2, 10.3, 10.4, 10.5])
y_pred = np.array([10.0, 10.3, 10.2, 10.5, 10.4])

print(f"Centered R²:   {centered_r2(y_true, y_pred):.6f}")
# Confirm equivalence with scikit-learn
from sklearn.metrics import r2_score
print(f"sklearn r2:    {r2_score(y_true, y_pred):.6f}")
```

### B.2 Uncentered R²

```python
def uncentered_r2(y_true, y_pred):
    """
    Uncentered R² (origin-based R²)
    R²_u = 1 - SSE / SST_uncentered
         = ⟨ŷ, ŷ⟩ / ⟨y, y⟩  (when e ⊥ ŷ)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    sse = np.sum((y_true - y_pred) ** 2)
    sst_u = np.sum(y_true ** 2)

    if sst_u == 0:
        return np.nan
    return 1.0 - sse / sst_u

# Example usage — comparing predictions on a same-physical-quantity scale (e.g., voltage)
y_true = np.array([10.1, 10.2, 10.3, 10.4, 10.5])
y_pred = np.array([10.0, 10.3, 10.2, 10.5, 10.4])

print(f"Uncentered R²: {uncentered_r2(y_true, y_pred):.6f}")

# Constant-bias scenario: contrast between the two metrics
y_pred_biased = y_true + 0.5  # all predictions offset by +0.5
print(f"\nConstant bias +0.5 scenario:")
print(f"  Centered R²:   {centered_r2(y_true, y_pred_biased):.6f}  "
      f"(insensitive to bias)")
print(f"  Uncentered R²: {uncentered_r2(y_true, y_pred_biased):.6f}  "
      f"(sensitive to bias)")
```

### B.3 Integrated Evaluation for 1:1-Line Agreement

```python
def evaluate_1_to_1_line_agreement(y_true, y_pred):
    """Evaluate 1:1-line agreement on same-physical-quantity data."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    err = y_pred - y_true

    rmse = np.sqrt(np.mean(err ** 2))
    mae  = np.mean(np.abs(err))                       # Mean Absolute Error
    bias = np.mean(err)                               # Mean Error (systematic bias)

    # MAPE: Mean Absolute Percentage Error (%) — undefined when y_true contains 0
    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs(err[mask] / y_true[mask])) * 100.0
    else:
        mape = np.nan

    return {
        "CenteredR2":   centered_r2(y_true, y_pred),
        "UncenteredR2": uncentered_r2(y_true, y_pred),
        "RMSE":         rmse,
        "MAE":          mae,
        "MAPE":         mape,
        "Bias":         bias,
    }


# Example usage
y_true = np.array([10.1, 10.2, 10.3, 10.4, 10.5])
y_pred = np.array([10.0, 10.3, 10.2, 10.5, 10.4])

results = evaluate_1_to_1_line_agreement(y_true, y_pred)
for k, v in results.items():
    print(f"{k:>15s}: {v:.6f}")
```
