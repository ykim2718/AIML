# Regularization
Rev. 0 | Created: 2026-09-12 | Updated: 2026-09-12 12:30 CDT

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

Table 1. Comparison of the three penalties

| Aspect | Ridge (L2) | Lasso (L1) | ElasticNet (L1 + L2) |
| --- | --- | --- | --- |
| Penalty form | $\lambda \sum \beta_j^2$ | $\lambda \sum \lvert\beta_j\rvert$ | $\lambda_1 \sum \lvert\beta_j\rvert + \lambda_2 \sum \beta_j^2$ |
| Zero coefficient | 없음. 0 에 가깝게 축소 | 있음. 완전한 0 가능 | 있음. 완전한 0 가능 |
| Main role | 다중공선성 완화, 분산 감소 | 변수 선택, model 단순화 | 다중공선성 상황에서의 변수 선택 |
| Suited data | 대부분의 변수가 유의미할 때 | 불필요한 변수가 많을 때 | 상관관계가 높은 변수 그룹이 많을 때 |

## 6. Dimension Reduction and Variable Removal

Penalty 를 더하는 대신 변수 자체를 줄이는 방법이 두 가지 있다.

- **VIF (Variance Inflation Factor) 기반 변수 제거**: 공선성이 높은, VIF 가 10 이상인 변수의 직접 제거.
- **PCA (주성분 분석) 회귀**: 서로 직교 (orthogonal) 하는 주성분 (principal component) 으로 기존 변수를 대체하여 다중공선성을 없애는 방법.

---

## Appendix A. Terminology

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
