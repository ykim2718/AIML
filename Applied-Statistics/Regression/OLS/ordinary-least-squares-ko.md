# Ordinary Least Squares
Rev. 12 | Created: 2026-09-12 | Updated: 2026-09-13 23:45 CDT

## 1. Purpose

- **Problem Statement**: 선형 회귀의 계수는 자료가 정하지만, 계수를 관측치마다 따로 맞추면 관측치 수만큼의 식이 남는다. 자료와 항을 하나의 행렬로 세워 최소화 문제를 한 번에 풀어야 계수가 유일하게 정해진다.
- **Goal**: design matrix $X$ 와 응답 벡터 $y$ 에서 normal equation 을 유도하여 회귀계수 추정값 $\hat{\beta}$ 를 닫힌 형태로 구하고, 그 계산을 NumPy 로 재현한다.
- **Non-Goal**: $X^\top X$ 의 역행렬이 없거나 불안정할 때 쓰는 regularization 은 다루지 않는다.

## 2. Summary

회귀 model 을 적합 (fitting) 한다는 것은 design matrix $X$ 와 응답 벡터 $y$ 로부터 식 (7) 의 행렬 연산으로 회귀계수 벡터 $\hat{\beta}$ 를 산출하는 과정이다. 그 계수는 잔차 제곱합 $RSS$ 를 가장 작게 만드는 값이며, $RSS$ 를 $\beta$ 에 대해 미분하여 0 으로 두면 normal equation 이 남는다. 해는 $X^\top X$ 의 역행렬이 존재할 때 유일하게 결정된다.

## 3. Design Matrix

Design matrix $X$ 는 관측 데이터와 model 의 항 (terms) 을 선형대수 형태로 표현한 행렬이다.

- **행 (Rows)**: 개별 관측치 (observation) $n$ 개
- **열 (Columns)**: model 이 사용하는 각 항, 절편 및 특성 (feature) $p + 1$ 개

```math
X =
\begin{bmatrix}
1 & x_{11} & x_{12} & \dots & x_{1p} \\
1 & x_{21} & x_{22} & \dots & x_{2p} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_{n1} & x_{n2} & \dots & x_{np}
\end{bmatrix}
\hspace{19em} (1)
```

첫 번째 열의 1 은 절편 (intercept, $\beta_0$) 에 대응하는 항이다.

## 4. Least Squares Criterion

선형 회귀 model 은 행렬식으로 표현된다.

```math
y = X\beta + \epsilon
\hspace{19em} (2)
```

Table 1. Symbols of the matrix form

| Symbol | Meaning | Shape |
| --- | --- | --- |
| $y$ | 응답 변수 벡터 | $n \times 1$ |
| $X$ | Design matrix | $n \times (p+1)$ |
| $\beta$ | 회귀계수 벡터 | $(p+1) \times 1$ |
| $\epsilon$ | 잔차 (오차) 벡터 | $n \times 1$ |

최소제곱법 (Ordinary Least Squares, OLS) 의 목적은 잔차 제곱합 (Residual Sum of Squares, $RSS$) 을 가장 작게 만드는 $\beta$ 를 찾는 것이다.

```math
RSS(\beta) = \sum_{i=1}^{n} e_i^2 = \epsilon^\top \epsilon = (y - X\beta)^\top (y - X\beta)
\hspace{19em} (3)
```

$\epsilon^\top \epsilon$ 이 성분의 제곱합이 되는 까닭은 [Appendix C](#appendix-c-self-inner-product-of-a-vector) 에 있다.

## 5. Normal Equation

$RSS(\beta)$ 를 최소화하는 점은 $\beta$ 에 대한 미분이 0 이 되는 지점이다. 식 (3) 을 전개하면 $\beta$ 의 이차식이 되고, 그 경사도를 0 으로 두면 연립방정식 하나가 남는다.

식 (3) 의 전개는 다음과 같으며, 그 과정은 [Appendix D](#appendix-d-expanding-the-residual-sum-of-squares) 에 있다.

```math
RSS(\beta) = y^\top y - 2\beta^\top X^\top y + \beta^\top X^\top X \beta
\hspace{19em} (4)
```

$\beta$ 에 대해 미분하여 0 으로 설정한다. 여기에 쓴 벡터 미분 규칙은 [Appendix E](#appendix-e-vector-derivatives-for-the-gradient) 에 있다.

```math
\frac{\partial RSS}{\partial \beta} = -2X^\top y + 2X^\top X\beta = 0
\hspace{19em} (5)
```

식 (5) 를 정리한 것이 normal equation 이다.

```math
X^\top X \beta = X^\top y
\hspace{19em} (6)
```

이름의 normal 은 직교를 뜻하며, 식 (6) 은 $X^\top (y - X\beta) = 0$ 과 같은 말, 곧 잔차가 design matrix 의 모든 열과 직교한다는 조건이다. 직교하면 $X\beta$ 는 $X$ 의 열들이 만드는 공간 (column space) 안에서 $y$ 에 가장 가까운 점, 곧 $y$ 에서 그 공간에 내린 수선 (perpendicular) 의 발 (정사영, orthogonal projection) 이 되고, 다른 어떤 계수를 넣어도 잔차는 그보다 길어진다. 잔차 제곱합이 최소가 되는 자리가 바로 이 지점이다.

이 조건을 그림으로 옮긴 것이 Fig 1 이다.

```text
                            y
                           /|
                          / |
                         /  |
                        /   |  residual
                       /    |  (perpendicular to the column space)
                      /     |
        O -----------+------+----------------------  column space
                            X beta
                            (foot of the perpendicular)
```

Fig 1. The residual as the perpendicular from y to the column space

- **평면**: $X\beta$ 로 만들 수 있는 모든 벡터가 놓이는 자리인 열공간.
- **수선 (perpendicular)**: $y$ 에서 그 평면에 90° 로 내린 선분. 그 길이의 제곱이 $RSS$ 이다.
- **수선의 발**: 평면 위에서 $y$ 에 가장 가까운 점 $X\beta$. 평면 위의 다른 점을 고르면 빗변이 되어 $y$ 까지의 거리가 길어진다.

## 6. Fitting

Model 을 적합 (fit) 한다는 것은 자료가 지시하는 최적의 계수 값 $\hat{\beta}$ 를 구하는 과정이며, 이는 식 (6) 의 normal equation 을 푸는 문제와 같다.

$X^\top X$ 의 역행렬이 존재할 때 해 $\hat{\beta}$ 는 유일하게 결정되며, 그 닫힌 형태인 식 (7) 이 이 문서의 결론이다.

```math
\hat{\beta} = (X^\top X)^{-1} X^\top y
\hspace{19em} (7)
```

식 (7) 의 계산은 [Appendix B](#appendix-b-computation) 에 있다.

---

## Appendix A. Terminology

- **column space**: Design matrix 의 열들이 만드는 벡터 공간.
- **design matrix**: 관측치를 행으로, model 의 항을 열로 놓은 $n \times (p+1)$ 행렬.
- **fitting**: 자료가 지시하는 최적의 계수 값을 구하는 과정.
- **intercept**: Design matrix 의 1 로 채워진 첫 열에 대응하는 계수 $\beta_0$.
- **L2 norm**: 벡터 원소의 제곱합의 제곱근.
- **linear form**: $\beta^\top a$ 처럼 계수의 1 차 항만으로 이루어진 스칼라 함수.
- **normal equation**: $RSS$ 의 경사도를 0 으로 두어 얻은 연립방정식 $X^\top X \beta = X^\top y$.
- **orthogonal projection**: 한 벡터에서 어떤 공간에 내린 수선의 발. 그 공간 안에서 원래 벡터에 가장 가까운 점이다.
- **perpendicular**: 한 점에서 직선이나 평면에 90° 로 내리그은 선분. 그것이 닿는 점이 수선의 발이다.
- **quadratic form**: $\beta^\top A \beta$ 처럼 계수의 2 차 항으로 이루어진 스칼라 함수.
- **residual**: 관측값과 model 예측값의 차이.
- **RSS**: Residual Sum of Squares. 잔차의 제곱합.
- **symmetric matrix**: 전치해도 자기 자신인 정방행렬.
- **transpose**: 행과 열을 맞바꾼 행렬. 합에서는 그대로, 곱에서는 차례가 뒤집힌다.

## Appendix B. Computation

식 (7) 은 NumPy 의 행렬 연산으로 그대로 옮겨진다. 아래 code 는 계수를 알고 있는 가상 데이터를 만들고, design matrix 를 세우고, normal equation 으로 구한 $\hat{\beta}$ 를 scikit-learn 의 `LinearRegression` 결과와 대조한다.

```python
# Python
import numpy as np
from sklearn.linear_model import LinearRegression

# 1. Generate synthetic data (y = 3 + 2*x1 + 5*x2 + noise)
np.random.seed(42)
n_samples = 100

x1 = np.random.rand(n_samples, 1) * 10
x2 = np.random.rand(n_samples, 1) * 5
noise = np.random.randn(n_samples, 1)

# True coefficients: beta_0 = 3, beta_1 = 2, beta_2 = 5
y = 3 + 2 * x1 + 5 * x2 + noise

# 2. Build the design matrix X
# Prepend a column of ones for the intercept (bias) term
X_raw = np.hstack([x1, x2])
ones = np.ones((n_samples, 1))
X = np.hstack([ones, X_raw])

print(f"Shape of design matrix X: {X.shape}")  # (100, 3)
print("First 3 rows of the design matrix:\n", X[:3])
print("-" * 50)

# 3. Solve for beta with the normal equation
# beta_hat = (X^T * X)^(-1) * X^T * y

# Method A: direct inverse with np.linalg.inv
X_transpose = X.T
beta_hat_inv = np.linalg.inv(X_transpose @ X) @ X_transpose @ y

# Method B: linear solve, preferred for numerical stability
# (X^T * X) * beta = X^T * y
beta_hat_solve = np.linalg.solve(X_transpose @ X, X_transpose @ y)

print("Normal equation (direct inverse):")
print(f"  intercept (beta_0): {beta_hat_inv[0][0]:.4f}")
print(f"  coefficient (beta_1): {beta_hat_inv[1][0]:.4f}")
print(f"  coefficient (beta_2): {beta_hat_inv[2][0]:.4f}")
print("-" * 50)

# 4. Cross-check against scikit-learn LinearRegression
model = LinearRegression()
model.fit(X_raw, y)

print("scikit-learn fit:")
print(f"  intercept: {model.intercept_[0]:.4f}")
print(f"  coefficients: {model.coef_[0]}")
```

NumPy 2.4.6 과 scikit-learn 1.9.1 에서 실행한 결과는 다음과 같다.

```text
Shape of design matrix X: (100, 3)
First 3 rows of the design matrix:
 [[1.         3.74540119 0.15714593]
 [1.         9.50714306 3.18205206]
 [1.         7.31993942 1.57177991]]
--------------------------------------------------
Normal equation (direct inverse):
  intercept (beta_0): 2.9106
  coefficient (beta_1): 1.9658
  coefficient (beta_2): 5.1439
--------------------------------------------------
scikit-learn fit:
  intercept: 2.9106
  coefficients: [1.96582747 5.14386228]
```

정규방정식으로 구한 계수와 `LinearRegression` 이 적합한 계수는 소수 넷째 자리까지 같다. 둘 다 참값 $(3, 2, 5)$ 에서 조금 벗어나 있는데, code 가 응답에 더한 noise 때문이다.

Code 에서 짚을 곳은 세 군데다.

- `np.hstack([ones, X_raw])`: 원본 특성 데이터 앞에 1 로 채워진 열을 붙여 만든 design matrix $X$. 그 첫 번째 열이 절편 $\beta_0$ 와 곱해지는 항.
- `@` 연산자: NumPy 의 행렬 곱셈 (matrix multiplication). `X.T @ X` 는 $X^\top X$.
- `np.linalg.solve` 와 `np.linalg.inv`: 연립방정식을 푸는 계산과 $(X^\top X)^{-1}$ 역행렬을 직접 구하는 계산. 수치 오차와 속도 측면에서 `np.linalg.solve(A, b)` 형태가 안정적.

## Appendix C. Self Inner Product of a Vector

열벡터 $v$ 를 자기 자신과 곱한 $v^\top v$ 는 성분 제곱의 합인 스칼라다.

$v$ 가 $n \times 1$ 열벡터이면 $v^\top$ 는 $1 \times n$ 행벡터이고, 행렬 곱 $(1 \times n)(n \times 1)$ 의 결과는 $1 \times 1$ 이다. 행렬 곱의 정의에 따라 그 하나뿐인 성분은 $v^\top$ 의 각 성분과 $v$ 의 같은 자리 성분을 곱해 더한 값이며, 두 벡터가 같으므로 각 항이 제곱이 된다.

```math
v^\top v =
\begin{bmatrix} v_1 & v_2 & \dots & v_n \end{bmatrix}
\begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}
= v_1^2 + v_2^2 + \dots + v_n^2 = \sum_{i=1}^{n} v_i^2
\hspace{19em} (8)
```

이 값은 $v$ 의 L2 norm 의 제곱 $\|v\|_2^2$ 이며, 모든 성분이 0 일 때만 0 이고 그 밖에는 언제나 양수다.

식 (3) 의 $\epsilon^\top \epsilon$ 은 식 (8) 의 $v$ 자리에 잔차 벡터 $\epsilon$ 을 넣은 것이므로 $\sum e_i^2$ 과 같다.

## Appendix D. Expanding the Residual Sum of Squares

식 (4) 는 식 (3) 의 곱을 풀어 쓴 것이며, 전치의 두 규칙과 스칼라의 성질 하나를 쓴다.

전치는 합을 그대로 따라가고 ($(A + B)^\top = A^\top + B^\top$), 곱에서는 차례를 뒤집는다 ($(AB)^\top = B^\top A^\top$). 이를 $(y - X\beta)$ 에 적용한다.

```math
(y - X\beta)^\top = y^\top - \beta^\top X^\top
\hspace{19em} (9)
```

식 (9) 를 식 (3) 에 넣고 두 괄호를 분배하면 항이 넷 나온다.

```math
(y - X\beta)^\top (y - X\beta) = y^\top y - y^\top X\beta - \beta^\top X^\top y + \beta^\top X^\top X \beta
\hspace{19em} (10)
```

가운데 두 항은 같은 값이다. $y^\top X\beta$ 는 $(1 \times n)(n \times (p+1))((p+1) \times 1)$ 의 곱이라 $1 \times 1$ 스칼라이고, 스칼라는 전치해도 자기 자신이므로 전치를 취해 순서를 뒤집어도 값이 바뀌지 않는다.

```math
y^\top X\beta = (y^\top X\beta)^\top = \beta^\top X^\top y
\hspace{19em} (11)
```

식 (11) 로 가운데 두 항을 합치면 $-2\beta^\top X^\top y$ 가 되고, 식 (10) 은 식 (4) 가 된다.

## Appendix E. Vector Derivatives for the Gradient

식 (5) 는 세 가지 규칙에서 나온다. 상수항의 미분, 일차형 (linear form) 의 미분, 이차형 (quadratic form) 의 미분이며, 여기서 미분은 $\beta$ 의 각 성분에 대한 편미분을 모아 놓은 열벡터를 뜻한다.

```math
\frac{\partial f}{\partial \beta} =
\begin{bmatrix}
\partial f / \partial \beta_0 \\
\vdots \\
\partial f / \partial \beta_p
\end{bmatrix}
\hspace{19em} (12)
```

첫째, 상수의 미분은 0 이다. $y^\top y$ 에는 $\beta$ 가 들어 있지 않다.

```math
\frac{\partial (y^\top y)}{\partial \beta} = 0
\hspace{19em} (13)
```

둘째, 일차형 $\beta^\top a$ 는 성분으로 쓰면 $\sum_j \beta_j a_j$ 이므로, $\beta_k$ 로 미분하면 $a_k$ 하나만 남는다. 성분을 다시 모으면 벡터 $a$ 가 된다.

```math
\frac{\partial (\beta^\top a)}{\partial \beta} = a
\hspace{19em} (14)
```

셋째, 이차형 $\beta^\top A \beta$ 는 성분으로 쓰면 $\sum_i \sum_j \beta_i A_{ij} \beta_j$ 이다. $\beta_k$ 로 미분하면 $i = k$ 인 항에서 $\sum_j A_{kj} \beta_j$ 가, $j = k$ 인 항에서 $\sum_i \beta_i A_{ik}$ 가 남으며, 이는 $(A + A^\top)\beta$ 의 $k$ 번째 성분이다.

```math
\frac{\partial (\beta^\top A \beta)}{\partial \beta} = (A + A^\top)\beta
\hspace{19em} (15)
```

$A$ 가 대칭이면 두 항이 같아 $2A\beta$ 가 된다. $X^\top X$ 는 전치가 $(X^\top X)^\top = X^\top (X^\top)^\top = X^\top X$ 로 자기 자신이므로 대칭이다.

식 (4) 의 세 항에 이 규칙을 차례로 넣은 것이 식 (5) 이다.

Table 2. The three terms of equation (4) under the rules

| Term of equation (4) | Rule | Derivative |
| --- | --- | --- |
| $y^\top y$ | 식 (13) | $0$ |
| $-2\beta^\top X^\top y$ | 식 (14), $a = X^\top y$ | $-2X^\top y$ |
| $\beta^\top X^\top X \beta$ | 식 (15), $A = X^\top X$ 는 대칭 | $2X^\top X \beta$ |
