# Conditions Under Which PLS Works Where OLS Fails
Rev. 0 | Created: 2026-11-02 | Updated: 2026-11-02 11:14 UTC

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Comparison](#31-comparison)
- [4. Singular Cross-Product Matrix](#4-singular-cross-product-matrix)
  - [4.1 OLS](#41-ols)
  - [4.2 PLS](#42-pls)
- [5. High Multicollinearity](#5-high-multicollinearity)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Worked Example](#appendix-b-worked-example)

## 1. Purpose

- **Problem Statement**: 입력 변수가 sample 보다 많거나 입력 변수끼리 선형으로 얽힌 데이터에서는 OLS (Ordinary Least Squares) 의 계수를 구할 수 없거나 계수가 크게 흔들린다.
- **Goal**: OLS 가 계수를 구하지 못하는 조건과 구하지만 불안정한 조건을 가르고, 각 조건에서 PLS (Partial Least Squares) 가 계수를 구하는 이유를 독자가 식으로 확인할 수 있게 한다.
- **Non-Goal**: 성분 수 $K$ 를 고르는 절차와 Ridge 같은 penalty 방법은 다루지 않는다.

## 2. Summary

OLS 가 계수를 구할 수 없고 PLS 는 구할 수 있는 조건은 cross-product matrix $\mathbf{X}^\top \mathbf{X}$ 의 역행렬이 없을 때, 곧 입력 변수 수가 sample 수 이상이거나 ($P \ge N$) 완전 다중공선성 (perfect multicollinearity) 이 있을 때이다. 역행렬은 있지만 다중공선성 (multicollinearity) 이 높을 때는 OLS 도 계수를 구하지만 계수가 크게 흔들리며, PLS 는 같은 데이터에서 안정적인 계수를 구한다. 조건별 결과는 Table 1 에 있다.

## 3. Taxonomy and its Hierarchy

OLS 가 계수를 구할 수 있는지는 sample 수와 입력 변수 수의 관계, 그리고 입력 변수 열 사이의 선형 종속 정도의 두 축으로 정해진다. 두 축은 아래 Fig 1 과 같다.

```text
Axis 1. Size of X (N x P)          Axis 2. Dependence among the columns of X
-------------------------          -----------------------------------------
P < N  : fewer columns than rows   independent      : no column near a mix of others
P >= N : as many columns or more   nearly dependent : a column close to a mix of others
                                   exactly dependent: a column equal to a mix of others
```

<a id="fig-1"></a>
Fig 1. Two axes that decide whether OLS has a unique solution

$P \ge N$ 이거나 열이 정확히 종속이면 $\mathbf{X}^\top \mathbf{X}$ 의 역행렬이 없다. $P \lt N$ 이고 열이 거의 종속이면 역행렬은 있지만 그 역행렬의 원소가 매우 크다. Centering 한 $\mathbf{X}$ 에서는 rank 가 $N - 1$ 이하이므로, $P = N - 1$ 이어도 역행렬이 없다.

OLS 에서 PLS 로 내려가면 입력 변수의 모든 방향을 쓰는 대신 고른 방향 $K$ 개만 쓰며, 그 대가로 역행렬이 필요 없어진다. 그 계층은 아래 Fig 2 와 같다.

```text
OLS               regresses y on all P columns of X
 |                needs   : X^T X invertible (rank P)
 |
 +-- PLS          regresses y on K orthogonal scores, K <= rank(X)
                  gives up: directions of X beyond the K chosen ones
                  gains   : no inverse of X^T X; works when P >= N or columns depend
```

<a id="fig-2"></a>
Fig 2. Hierarchy from OLS to PLS

OLS 는 $P$ 개 열 모두를 쓰므로 $\mathbf{X}^\top \mathbf{X}$ 의 rank 가 $P$ 여야 한다. PLS 는 score $K$ 개에만 회귀하므로 $K$ 가 $\mathbf{X}$ 의 rank 이하이기만 하면 되고, 고르지 않은 방향의 정보는 쓰지 않는다.

### 3.1 Comparison

데이터 조건마다 OLS 와 PLS 의 결과는 Table 1 과 같다.

Table 1. OLS and PLS by data condition

| Condition            | Rank of $\mathbf{X}^\top \mathbf{X}$ | OLS                         | PLS           | Section   |
| :---:                | :---:                                | :---:                       | :---:         | :---:     |
| $P \ge N$            | $P$ 미만                             | 계수를 하나로 정할 수 없음  | 계수를 구함   | section 4 |
| 완전 다중공선성      | $P$ 미만                             | 계수를 하나로 정할 수 없음  | 계수를 구함   | section 4 |
| 높은 다중공선성      | $P$                                  | 계수를 구하지만 크게 흔들림 | 안정적인 계수 | section 5 |
| $P \lt N$, 열이 독립 | $P$                                  | 안정적인 계수               | 안정적인 계수 | -         |

$N$ 은 sample 수이고 $P$ 는 입력 변수의 수이다. 앞의 두 행이 OLS 는 못하고 PLS 는 할 수 있는 조건이고, 셋째 행은 둘 다 하지만 PLS 가 더 안정적인 조건이다.

방법마다 가정, 설정값, 깨지는 조건, 쓰이는 곳은 Table 2 와 같다.

Table 2. Assumption, setting, failure condition and usage of each method

| Method | Assumption                                    | Setting                                   | Breaks when                     | Where used                                |
| :---:  | :---:                                         | :---:                                     | :---:                           | :---:                                     |
| OLS    | $\mathbf{X}^\top \mathbf{X}$ 의 역행렬이 있음 | 없음                                      | $P \ge N$ 이거나 열이 종속일 때 | 변수가 적고 서로 독립인 데이터            |
| PLS    | $\mathbf{X}$ 와 $\mathbf{y}$ 의 관계가 선형   | 성분 수 $K \le \mathrm{rank}(\mathbf{X})$ | 관계가 비선형일 때              | 센서 수백 개, 스펙트럼 같은 고차원 데이터 |

## 4. Singular Cross-Product Matrix

$P \ge N$ 이거나 완전 다중공선성이 있으면 $\mathbf{X}^\top \mathbf{X}$ 가 singular matrix 가 되어 OLS 의 계수는 하나로 정해지지 않고, PLS 는 역행렬 없이 계수를 구한다. $\mathbf{X} \in \mathbb{R}^{N \times P}$ 는 sample $N$ 개와 입력 변수 $P$ 개로 이루어진 입력 행렬이고, $\mathbf{y} \in \mathbb{R}^{N}$ 은 같은 sample 의 타겟이며, 두 행렬의 열은 평균을 빼서 centering 한다.

### 4.1 OLS

OLS 의 계수 $\hat{\boldsymbol{\beta}} \in \mathbb{R}^{P}$ 는 잔차 제곱합을 최소로 하는 값이며, 정규방정식 (normal equation) $\mathbf{X}^\top \mathbf{X} \hat{\boldsymbol{\beta}} = \mathbf{X}^\top \mathbf{y}$ 의 해이다. $\mathbf{X}^\top \mathbf{X}$ 의 역행렬이 있으면 해는 하나이다.

```math
\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y} \hspace{19em} (1)
```

$\mathbf{X}^\top \mathbf{X}$ 는 $P \times P$ 행렬이지만 그 rank 는 $\mathbf{X}$ 의 rank 와 같고, centering 한 $\mathbf{X}$ 의 rank 는 $N - 1$ 과 $P$ 를 넘지 못한다.

```math
\mathrm{rank}(\mathbf{X}^\top \mathbf{X}) = \mathrm{rank}(\mathbf{X}) \le \min(N - 1, \, P) \hspace{19em} (2)
```

$P \ge N$ 이면 식 (2) 의 rank 가 $P$ 보다 작고, 완전 다중공선성이 있으면 한 열이 다른 열들의 선형결합이어서 rank 가 $P$ 보다 작다. 두 경우 모두 $\mathbf{X}\mathbf{v} = \mathbf{0}$ 을 만족하는 $\mathbf{0}$ 이 아닌 벡터 $\mathbf{v}$ 가 있으므로, 정규방정식의 해 $\hat{\boldsymbol{\beta}}$ 하나에 $\mathbf{v}$ 를 몇 배 더해도 예측값이 같다.

```math
\mathbf{X}(\hat{\boldsymbol{\beta}} + a\mathbf{v}) = \mathbf{X}\hat{\boldsymbol{\beta}} \quad \text{for every real } a \hspace{19em} (3)
```

식 (3) 대로 잔차 제곱합이 같은 계수가 끝없이 많으므로, OLS 는 계수를 하나로 정할 수 없다. $P \ge N$ 이면 잔차가 0 이 되는 계수까지 있어서, 학습 데이터에는 완전히 맞지만 새 데이터의 예측 오차가 커진다.

### 4.2 PLS

PLS 는 $\mathbf{X}$ 와 $\mathbf{y}$ 의 공분산이 가장 큰 방향부터 score $K$ 개를 뽑고, $\mathbf{y}$ 를 그 score 에 회귀한다. X-score 를 열로 모은 $\mathbf{T} \in \mathbb{R}^{N \times K}$ 는 원래 $\mathbf{X}$ 의 선형결합 $\mathbf{T} = \mathbf{X}\mathbf{R}$ 이고, $\mathbf{R} \in \mathbb{R}^{P \times K}$ 는 각 score 의 방향을 열로 모은 행렬이다. Score 끼리는 서로 직교하므로 $\mathbf{T}^\top \mathbf{T}$ 가 대각행렬이고, score 마다의 계수 $q_k$ 는 역행렬 없이 나눗셈 하나로 구해진다.

```math
q_k = \frac{\mathbf{t}_k^\top \mathbf{y}}{\mathbf{t}_k^\top \mathbf{t}_k}, \qquad k = 1, \ldots, K \hspace{19em} (4)
```

$\mathbf{t}_k$ 는 $\mathbf{T}$ 의 $k$ 번째 열이다. 원래 입력 변수에 대한 계수는 score 의 계수에 방향 행렬 $\mathbf{R}$ 을 곱해 얻는다.

```math
\hat{\boldsymbol{\beta}}_{\mathrm{PLS}} = \mathbf{R}\,\mathbf{q}, \qquad \mathbf{q} = (q_1, \ldots, q_K)^\top \hspace{19em} (5)
```

식 (4) 의 분모 $\mathbf{t}_k^\top \mathbf{t}_k$ 는 score 가 0 이 아니면 양수이고, score 는 $K$ 가 $\mathbf{X}$ 의 rank 이하이고 deflation 후 남은 $\mathbf{X}$ 와 $\mathbf{y}$ 의 공분산이 0 이 아닌 동안 0 이 되지 않는다. 따라서 PLS 는 $\mathbf{X}^\top \mathbf{X}$ 의 rank 가 $P$ 보다 작아도 $K \le \mathrm{rank}(\mathbf{X})$ 이면 계수를 하나로 구한다. [Appendix B](#appendix-b-worked-example) 의 예제에서 센서 50 개, wafer 20 장인 $\mathbf{X}$ 의 $\mathbf{X}^\top \mathbf{X}$ rank 는 50 중 19 이고, OLS 계수 두 개가 서로 5.00 떨어져 있어도 잔차가 모두 $10^{-14}$ 수준으로 같으며, PLS 는 $K = 2$ 로 $R^2 = 0.994$ 를 얻는다.

## 5. High Multicollinearity

다중공선성이 높지만 완전하지 않으면 $\mathbf{X}^\top \mathbf{X}$ 의 역행렬이 있어 OLS 도 식 (1) 로 계수를 구하지만, 그 계수는 데이터의 작은 변화에도 크게 흔들린다. 타겟의 잡음이 서로 독립이고 분산이 $\sigma^2$ 로 같을 때, OLS 계수의 공분산 행렬은 아래와 같다. $\lambda_j$ 와 $\mathbf{v}_j$ 는 $\mathbf{X}^\top \mathbf{X}$ 의 $j$ 번째 eigenvalue 와 eigenvector 이다.

```math
\mathrm{Cov}(\hat{\boldsymbol{\beta}}) = \sigma^2 (\mathbf{X}^\top \mathbf{X})^{-1} = \sigma^2 \sum_{j=1}^{P} \frac{\mathbf{v}_j \mathbf{v}_j^\top}{\lambda_j} \hspace{19em} (6)
```

열이 거의 종속이면 가장 작은 eigenvalue 가 0 에 가깝고, 식 (6) 에서 $1/\lambda_j$ 가 커지므로 그 eigenvector 방향으로 계수의 분산이 커진다. PLS 는 공분산이 큰 방향부터 $K$ 개만 쓰므로, $K$ 가 작으면 eigenvalue 가 0 에 가까운 방향이 계수에 거의 들어가지 않는다. [Appendix B](#appendix-b-worked-example) 의 예제에서 $\mathbf{X}^\top \mathbf{X}$ 의 rank 는 10 중 10 이지만 eigenvalue 의 최대와 최소의 비가 $3.2 \times 10^{3}$ 이고, 타겟의 잡음만 바꾸어 200 번 다시 적합하면 계수 표준편차의 평균이 OLS 0.164, PLS 0.003 이다.

---

## Appendix A. Terminology

- **centering**: 각 열에서 그 열의 평균을 빼서 평균을 0 으로 만드는 처리.
- **cross-product matrix**: $\mathbf{X}^\top \mathbf{X}$ 로, 입력 변수 열끼리의 내적을 원소로 가진 $P \times P$ 행렬.
- **deflation**: 찾은 score 가 설명한 부분을 $\mathbf{X}$ 에서 빼서, 다음 score 를 남은 정보에서 찾게 하는 단계.
- **eigenvector**: 행렬 $\mathbf{A}$ 에 곱해도 방향이 바뀌지 않는 벡터 $\mathbf{v}$ 이며, $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$ 의 $\lambda$ 를 그 eigenvector 의 eigenvalue 라 한다.
- **OLS (Ordinary Least Squares)**: 잔차 제곱합을 최소로 하는 계수를 구하는 선형 회귀.
- **PLS (Partial Least Squares)**: $\mathbf{X}$ 와 타겟의 공분산이 큰 방향으로 score 를 뽑고 그 score 에 회귀하는 방법.
- **rank**: 행렬의 선형독립인 열의 수.
- **score**: Sample 을 한 방향으로 투영한 좌표값.
- **singular matrix**: 역행렬이 없는 정사각행렬이며, rank 가 행렬의 크기보다 작다.
- **다중공선성 (multicollinearity)**: 입력 변수 열 사이의 선형 상관이 높아, 한 열이 다른 열들의 선형결합에 가까운 상태.
- **완전 다중공선성 (perfect multicollinearity)**: 한 열이 다른 열들의 선형결합과 정확히 같은 상태.

## Appendix B. Worked Example

아래 code 는 숨은 요인 두 개가 센서를 움직이고 그 가운데 하나가 두께를 정하는 데이터를 만들어, Table 1 의 앞 세 조건에서 OLS 와 PLS 를 견준다. 실행 환경은 NumPy 2.5.3, scikit-learn 1.9.1 이다.

```python
# Python
import numpy as np
from sklearn.cross_decomposition import PLSRegression

rng = np.random.default_rng(0)


def sensors(n_wafers: int = None, n_sensors: int = None) -> tuple:
    """Return X driven by two hidden factors and a thickness y driven by the second one."""
    factors = rng.standard_normal((n_wafers, 2))
    loadings = rng.standard_normal((2, n_sensors))
    X = factors @ loadings + 0.1 * rng.standard_normal((n_wafers, n_sensors))
    y = 2.0 * factors[:, 1] + 0.1 * rng.standard_normal(n_wafers)
    return X - X.mean(axis=0), y - y.mean()


# 1. P > N: 50 sensors, 20 wafers
X, y = sensors(n_wafers=20, n_sensors=50)
print(f"P > N           : X {X.shape}, rank of X^T X = {np.linalg.matrix_rank(X.T @ X)} of {X.shape[1]}")

# Two different OLS coefficient vectors that fit the data equally well
beta_a = np.linalg.lstsq(X, y, rcond=None)[0]
null_direction = np.linalg.svd(X)[2][-1]  # X @ null_direction is zero
beta_b = beta_a + 5.0 * null_direction
print(f"                  ||X beta_a - y|| = {np.linalg.norm(X @ beta_a - y):.2e}, "
      f"||X beta_b - y|| = {np.linalg.norm(X @ beta_b - y):.2e}, "
      f"||beta_a - beta_b|| = {np.linalg.norm(beta_a - beta_b):.2f}")

pls = PLSRegression(n_components=2, scale=False).fit(X, y)
print(f"                  PLS with K = 2: R^2 = {pls.score(X, y):.3f}")
print("-" * 72)

# 2. Perfect multicollinearity: one column is twice another
X, y = sensors(n_wafers=40, n_sensors=10)
X[:, 1] = 2.0 * X[:, 0]
print(f"perfect collin. : X {X.shape}, rank of X^T X = {np.linalg.matrix_rank(X.T @ X)} of {X.shape[1]}")
pls = PLSRegression(n_components=2, scale=False).fit(X, y)
print(f"                  PLS with K = 2: R^2 = {pls.score(X, y):.3f}")
print("-" * 72)

# 3. High multicollinearity: X^T X has an inverse, but a tiny eigenvalue
X, y = sensors(n_wafers=40, n_sensors=10)
eigenvalues = np.linalg.eigvalsh(X.T @ X)
print(f"high collin.    : rank of X^T X = {np.linalg.matrix_rank(X.T @ X)} of {X.shape[1]}, "
      f"eigenvalue max/min = {eigenvalues[-1] / eigenvalues[0]:.1e}")

# Refit 200 times with fresh noise in y, and measure how far the coefficients move
ols_runs, pls_runs = [], []
for _ in range(200):
    y_new = y + 0.1 * rng.standard_normal(len(y))
    ols_runs.append(np.linalg.solve(X.T @ X, X.T @ y_new))
    pls_runs.append(PLSRegression(n_components=2, scale=False).fit(X, y_new).coef_.ravel())
print(f"                  mean sd of coefficients: OLS {np.std(ols_runs, axis=0).mean():.3f}, "
      f"PLS {np.std(pls_runs, axis=0).mean():.3f}")
```

실행 결과는 아래와 같다.

```text
P > N           : X (20, 50), rank of X^T X = 19 of 50
                  ||X beta_a - y|| = 6.29e-15, ||X beta_b - y|| = 1.06e-14, ||beta_a - beta_b|| = 5.00
                  PLS with K = 2: R^2 = 0.994
------------------------------------------------------------------------
perfect collin. : X (40, 10), rank of X^T X = 9 of 10
                  PLS with K = 2: R^2 = 0.996
------------------------------------------------------------------------
high collin.    : rank of X^T X = 10 of 10, eigenvalue max/min = 3.2e+03
                  mean sd of coefficients: OLS 0.164, PLS 0.003
```

- 첫 묶음: wafer 20 장을 centering 하여 rank 가 $N - 1 = 19$ 이다. 서로 5.00 떨어진 두 OLS 계수의 잔차가 모두 $10^{-14}$ 수준이어서 식 (3) 대로 계수가 하나로 정해지지 않고, 잔차가 0 이어서 학습 데이터에 완전히 맞는다.
- 둘째 묶음: 둘째 열을 첫째 열의 2 배로 두어 rank 가 10 중 9 이지만, PLS 는 $K = 2$ 로 계수를 구한다.
- 셋째 묶음: 숨은 요인 두 개가 센서 10 개를 움직여 열이 거의 종속이다. 역행렬은 있지만, 잡음만 바꾼 200 번의 적합에서 OLS 계수가 PLS 계수보다 약 50 배 크게 흔들린다.
