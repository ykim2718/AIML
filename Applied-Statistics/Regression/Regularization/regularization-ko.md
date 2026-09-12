# Regularization
Rev. 2 | Created: 2026-09-12 | Updated: 2026-09-12 14:24 CDT

## 1. Purpose

- **Problem Statement**: 선형 회귀에서 $X^\top X$ 의 역행렬이 존재하지 않거나, 존재하더라도 불안정한 현상은 데이터 분석에서 자주 마주치는 문제다. 앞의 경우 계수를 산출할 수 없고, 뒤의 경우 데이터의 작은 변화에도 회귀계수가 크게 흔들린다.
- **Goal**: 역행렬이 존재하지 않는 조건과 다중공선성 (multicollinearity) 이 높은 조건을 구분하고, 비용함수에 penalty 를 더하는 세 가지 방법 (Ridge, Lasso, ElasticNet) 중 무엇을 어느 데이터에 쓸지 정한다.
- **Non-Goal**: Penalty 계수 $\lambda$ 를 고르는 절차는 다루지 않는다.

## 2. Summary

세 규제화 (regularization) 방법은 penalty 항의 형태로 갈리며, 그 형태가 계수를 0 으로 만들 수 있는지를 정한다. Ridge 는 L2 penalty 로 계수를 0 에 가깝게 줄이되 0 으로 만들지는 않고, Lasso 는 L1 penalty 로 불필요한 변수의 계수를 완전히 0 으로 만들며, ElasticNet 은 둘을 결합하여 상관관계가 높은 변수 그룹을 함께 다룬다. 선택 기준은 Table 1 에 있다. 규제화 대신 변수를 직접 줄이는 방법은 section 6 에 있다.

## 3. Cause of the Problem

### 3.1 Singular Matrix

행렬 $X^\top X$ 의 역행렬 $(X^\top X)^{-1}$ 이 존재하려면 $X^\top X$ 가 전위수 (full rank) 여야 한다. 다음과 같은 경우 역행렬이 존재하지 않으며 (특이 행렬, singular matrix), 계수를 산출할 수 없다.

- **특성 (feature) 수 $\gt$ 관측치 수 ($p \gt n$)**: 변수가 데이터 개수보다 많은 경우. 유전자 데이터, 텍스트 고차원 데이터.
- **완전 다중공선성 (perfect multicollinearity)**: 한 변수가 다른 변수들의 완벽한 선형 조합으로 표현되는 경우 ($x_2 = 2x_1$).

### 3.2 High Multicollinearity

변수 간 상관관계가 매우 높으면 $X^\top X$ 의 행렬식 (determinant) 이 0 에 가까워진다. 이로 인해 역행렬 요소가 매우 커져, 데이터의 작은 변화에도 회귀계수 $\hat{\beta}$ 가 극단적으로 널뛰는 현상이 발생하며 model 의 분산 (variance) 이 커진다.

## 4. Penalized Loss

Section 3 의 두 문제를 해결하는 대표적인 방법은 비용함수 (loss function) 에 계수의 크기에 대한 penalty 를 더해, $X^\top X$ 행렬을 강제로 역행렬 계산이 가능한 구조로 변형하는 것이다.

### 4.1 Ridge Regression

Ridge 는 기존 잔차 제곱합 ($RSS$) 에 계수 제곱합 (L2 norm) 을 penalty 로 추가한다.

```math
\mathrm{Loss}_{\mathrm{Ridge}} = \|y - X\beta\|_2^2 + \lambda \|\beta\|_2^2 = (y - X\beta)^\top (y - X\beta) + \lambda \beta^\top \beta
\hspace{19em} (1)
```

정규방정식의 해는 다음과 같다.

```math
\hat{\beta}_{\mathrm{Ridge}} = (X^\top X + \lambda I)^{-1} X^\top y
\hspace{19em} (2)
```

- **역행렬 보장**: 대각선에 더한 $\lambda I$ ($\lambda \gt 0$) 로 항상 역행렬을 갖는 $X^\top X + \lambda I$.
- **계수 축소 (shrinkage)**: 계수의 크기를 전체적으로 0 에 가깝게 줄이되, 0 으로 완전히 만들지는 않는 효과.
- **적용 대상**: 변수의 개수가 많고 다중공선성이 전반적으로 존재하는 경우.

### 4.2 Lasso Regression

Lasso 는 잔차 제곱합 ($RSS$) 에 계수의 절댓값 합 (L1 norm) 을 penalty 로 추가한다.

```math
\mathrm{Loss}_{\mathrm{Lasso}} = \|y - X\beta\|_2^2 + \lambda \|\beta\|_1 = (y - X\beta)^\top (y - X\beta) + \lambda \sum_{j=1}^{p} |\beta_j|
\hspace{19em} (3)
```

- **변수 선택 (feature selection)**: 중요하지 않은 변수의 회귀계수를 완전한 0 으로 만드는 성질.
- **희소성 (sparsity)**: 결과 model 이 단순해져 높아지는 해석력.
- **최적화**: 미분 불가능한 지점 ($\beta = 0$) 때문에 정규방정식 대신 쓰는 좌표 하강법 (coordinate descent) 등의 수치적 최적화 방법.

### 4.3 ElasticNet

ElasticNet 은 L1 규제와 L2 규제를 결합한 방식이다.

```math
\mathrm{Loss}_{\mathrm{ElasticNet}} = \|y - X\beta\|_2^2 + \lambda_1 \|\beta\|_1 + \lambda_2 \|\beta\|_2^2
\hspace{19em} (4)
```

- **결합**: Lasso 와 Ridge 의 장점을 함께 취한 형태.
- **그룹 효과 (group effect)**: 상관관계가 높은 변수 집합에서 그룹 전체를 함께 선택하거나 축소하는 성질. Lasso 는 그중 하나만 임의로 선택하고 나머지를 0 으로 만드는 경향.

## 5. Comparison

세 방법을 가르는 것은 계수를 완전한 0 으로 만드는지이며, 그 차이는 penalty 가 그리는 제약 영역의 모양에서 온다. Penalty 를 더해 비용함수를 최소화하는 것은 계수 벡터를 일정한 크기 안에 묶어 두고 $RSS$ 를 최소화하는 것과 같고, 그 크기를 재는 norm 이 영역의 모양을 정한다.

두 penalty 의 제약 영역은 Fig 1 과 같다.

```text
              b2                                    b2
              |                                     |
             /|\                                _.--+--._
            / | \                              /    |    \
      -----+--+--+----- b1              ------+-----+-----+------ b1
            \ | /                              \    |    /
             \|/                                '--_+_--'
              |                                     |

             (a)                                   (b)
```

Fig 1. Constraint regions of the L1 and L2 penalties

- **(a)**: L1 의 제약 영역. 축 위에 꼭짓점이 있는 마름모이며, $RSS$ 의 등고선이 커지다가 처음 닿는 곳이 대개 그 꼭짓점이다. 꼭짓점에서는 한 계수가 정확히 0 이다.
- **(b)**: L2 의 제약 영역. 꼭짓점이 없는 원이며, 등고선이 닿는 점은 축에서 벗어나 있어 두 계수 모두 0 이 아니다.

Table 1. Comparison of the three penalties

| Aspect | Ridge (L2) | Lasso (L1) | ElasticNet (L1 + L2) |
| --- | --- | --- | --- |
| Penalty form | $\lambda \sum \beta_j^2$ | $\lambda \sum \lvert\beta_j\rvert$ | $\lambda_1 \sum \lvert\beta_j\rvert + \lambda_2 \sum \beta_j^2$ |
| Zero coefficient | 없음. 0 에 가깝게 축소 | 있음. 완전한 0 가능 | 있음. 완전한 0 가능 |
| Main role | 다중공선성 완화, 분산 감소 | 변수 선택, model 단순화 | 다중공선성 상황에서의 변수 선택 |
| Suited data | 대부분의 변수가 유의미할 때 | 불필요한 변수가 많을 때 | 상관관계가 높은 변수 그룹이 많을 때 |

이 세 방법과 OLS 를 같은 데이터에 적용한 결과는 [Appendix B](#appendix-b-worked-example) 에 있다.

## 6. Dimension Reduction and Variable Removal

Penalty 를 더하는 대신 변수 자체를 줄이는 방법이 두 가지 있다.

- **VIF (Variance Inflation Factor) 기반 변수 제거**: 공선성이 높은, VIF 가 10 이상인 변수의 직접 제거.
- **PCA (주성분 분석) 회귀**: 서로 직교 (orthogonal) 하는 주성분 (principal component) 으로 기존 변수를 대체하여 다중공선성을 없애는 방법.

---

## Appendix A. Terminology

- **contour**: 같은 $RSS$ 값을 주는 계수 조합들이 그리는 선.
- **determinant**: 정방행렬에 대응하는 스칼라 값. 0 이면 역행렬이 존재하지 않는다.
- **full rank**: 행렬의 rank 가 그 행렬이 가질 수 있는 최댓값과 같은 상태.
- **L1 norm**: 벡터 원소의 절댓값 합.
- **L2 norm**: 벡터 원소의 제곱합의 제곱근.
- **multicollinearity**: 설명 변수 사이의 선형 상관관계.
- **regularization**: 비용함수에 계수의 크기에 대한 penalty 를 더해 해를 안정시키는 방법.
- **RSS**: Residual Sum of Squares. 잔차의 제곱합.
- **shrinkage**: Penalty 로 계수의 크기를 0 쪽으로 줄이는 효과.
- **singular matrix**: 역행렬이 존재하지 않는 정방행렬.
- **VIF**: Variance Inflation Factor. 한 변수가 다른 변수들로 얼마나 설명되는지를 나타내는 공선성 지표.

## Appendix B. Worked Example

아래 code 는 상관계수가 0.99 를 넘는 변수 두 개와 응답에 관여하지 않는 변수 세 개를 담은 자료를 만들어, section 3 의 두 조건과 section 4 의 세 penalty 가 계수에 남기는 차이를 한 번에 보인다.

```python
# Python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

# 1. Two nearly collinear features, one useful feature, three that carry nothing
np.random.seed(0)
n_samples = 60

x1 = np.random.randn(n_samples)
x2 = x1 + 0.01 * np.random.randn(n_samples)
x3 = np.random.randn(n_samples)
unrelated = np.random.randn(n_samples, 3)
X = np.column_stack([x1, x2, x3, unrelated])

# Only x1, x2 and x3 drive the response
y = 3 * x1 + 3 * x2 + 2 * x3 + 0.5 * np.random.randn(n_samples)

print(f"corr(x1, x2): {np.corrcoef(x1, x2)[0, 1]:.6f}")
print(f"condition number of X^T X: {np.linalg.cond(X.T @ X):.3e}")
print("-" * 60)

# 2. More features than observations: X^T X cannot be full rank
X_wide = np.random.randn(5, 8)
XtX_wide = X_wide.T @ X_wide
print(f"p > n: X^T X is {XtX_wide.shape[0]} x {XtX_wide.shape[1]}, "
      f"rank {np.linalg.matrix_rank(XtX_wide)}")
print(f"p > n: determinant of X^T X: {np.linalg.det(XtX_wide):.3e}")
print("-" * 60)

# 3. The same data under the four estimators
models = {
    "OLS": LinearRegression(),
    "Ridge (alpha=1.0)": Ridge(alpha=1.0),
    "Lasso (alpha=0.1)": Lasso(alpha=0.1),
    "ElasticNet (alpha=0.1, l1_ratio=0.5)": ElasticNet(alpha=0.1, l1_ratio=0.5),
}
for name, model in models.items():
    model.fit(X, y)
    print(f"{name}")
    print(f"  coefficients: {np.round(model.coef_, 3)}")
    print(f"  exact zeros: {int(np.sum(model.coef_ == 0))}")
print("-" * 60)

# 4. How far each coefficient vector moves when noise of sd 0.01 is added to y
y_perturbed = y + 0.01 * np.random.randn(n_samples)
for name in ("OLS", "Ridge (alpha=1.0)"):
    before = models[name].coef_.copy()
    after = models[name].fit(X, y_perturbed).coef_
    print(f"{name}: max coefficient shift {np.max(np.abs(after - before)):.4f}")
```

NumPy 2.4.6 과 scikit-learn 1.9.1 에서 실행한 결과는 다음과 같다.

```text
corr(x1, x2): 0.999953
condition number of X^T X: 4.444e+04
------------------------------------------------------------
p > n: X^T X is 8 x 8, rank 5
p > n: determinant of X^T X: -3.766e-45
------------------------------------------------------------
OLS
  coefficients: [-0.07   6.225  1.898 -0.105 -0.029  0.039]
  exact zeros: 0
Ridge (alpha=1.0)
  coefficients: [ 3.047  3.063  1.867 -0.09  -0.03   0.041]
  exact zeros: 0
Lasso (alpha=0.1)
  coefficients: [ 6.067e+00  2.000e-03  1.790e+00 -0.000e+00 -0.000e+00  0.000e+00]
  exact zeros: 3
ElasticNet (alpha=0.1, l1_ratio=0.5)
  coefficients: [ 2.996  2.991  1.755 -0.012 -0.004  0.   ]
  exact zeros: 1
------------------------------------------------------------
OLS: max coefficient shift 0.1059
Ridge (alpha=1.0): max coefficient shift 0.0008
```

결과에서 읽을 것은 다섯 가지다.

- **역행렬 없음**: 관측치 5 개에 변수 8 개인 $X^\top X$ 의 rank 5 와 $-3.766 \times 10^{-45}$ 인 행렬식. Section 3.1 이 말한 전위수 미달.
- **OLS 의 널뜀**: 상관 0.999953 인 두 변수에 $-0.07$ 과 $6.225$ 로 갈린 계수. 참값은 둘 다 3 이며, 합은 지켜지고 배분이 무너진 모양.
- **Ridge**: 3.047 과 3.063 으로 참값 가까이 모인 두 계수. 완전한 0 은 없음.
- **Lasso**: 두 변수 중 하나만 6.067 로 남기고 다른 하나를 0.002 로 누른 결과, 그리고 응답에 관여하지 않는 변수 세 개의 완전한 0.
- **ElasticNet**: 2.996 과 2.991 로 두 변수를 함께 남긴 그룹 효과.

응답에 표준편차 0.01 의 잡음을 더해 다시 적합했을 때 계수의 최대 이동은 OLS 가 0.1059, Ridge 가 0.0008 이다. Section 3.2 가 말한 분산의 차이가 이 두 수에 그대로 나타난다.
