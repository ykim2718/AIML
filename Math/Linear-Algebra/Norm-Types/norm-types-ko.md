# Types of Norms
Rev. 14 | Created: 2026-11-02 | Updated: 2026-11-03 00:48 UTC

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Comparison](#31-comparison)
- [4. Definition](#4-definition)
- [5. Application](#5-application)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Derivation](#appendix-b-derivation)
- [Appendix C. Triangle Inequality](#appendix-c-triangle-inequality)

## 1. Purpose

- **Problem Statement**: 선형대수와 머신러닝에서 norm 을 vector 의 길이로만 읽으면, L1 regularization 이나 원소별 한계처럼 다른 norm 을 쓰는 곳에서 결과가 왜 달라지는지 설명할 수 없다.
- **Goal**: 독자가 norm 의 조건을 알고, vector norm 과 matrix norm 의 종류를 정의와 숫자로 가려, 손실과 regularization 과 해의 선택에 어느 norm 이 쓰이는지 알 수 있게 한다.
- **Non-Goal**: 무한 차원 함수 공간의 norm 과 복소 vector 의 norm 은 다루지 않는다.

## 2. Summary

Norm 은 양수성, 동차성, 삼각부등식을 만족하는 크기의 함수이며, 우리가 아는 직선 길이는 그 가운데 L2 norm (Euclidean norm) 이다.
Vector norm 은 Lp norm 의 $p$ 로 갈리며 L1, L2, L∞ 가 대표이고, matrix norm 은 원소로 재는 Frobenius norm 과 특이값으로 재는 spectral norm, nuclear norm 이 대표이다.
최소제곱 손실, ridge, 최소 노름 해는 L2 를, lasso 는 L1 을, 원소별 한계는 L∞ 를 쓴다.

## 3. Taxonomy and its Hierarchy

이 문서에서 $\mathbf{x}$ 는 원소 $x_1, \dots, x_n$ 으로 이루어진 $n$ 차원 열 vector, $\mathbf{A}$ 는 $m \times n$ 실수 행렬, $\sigma_1 \ge \sigma_2 \ge \cdots$ 는 $\mathbf{A}$ 의 특이값이다. Norm 은 무엇의 크기를 재는지와 무엇으로 크기를 계산하는지의 두 축으로 나뉘며, 축마다의 값은 아래 [Fig 1](#fig-1) 이다.

```text
Axis 1. Object measured
  vector               x in R^n
  matrix               A in R^(m x n)

Axis 2. Values computed from
  elements             sum, root of sum of squares, or maximum of |x_i| or |a_ij|
  singular values      sum, root of sum of squares, or maximum of sigma_i
```

<a id="fig-1"></a>
Fig 1. Two axes of norms

- Axis 1 은 크기를 재는 대상이다. Vector norm 은 vector 의 크기를 재고, matrix norm 은 행렬의 크기를 잰다.
- Axis 2 는 크기를 계산하는 데 쓰는 값이다. 원소로 계산하는 norm 은 원소 절댓값의 합, 제곱합의 제곱근, 최댓값을 쓰고, 특이값으로 계산하는 norm 은 특이값의 합, 제곱합의 제곱근, 최댓값을 쓴다. Vector norm 은 원소로만 계산하고, matrix norm 은 원소나 특이값으로 계산한다.

Norm 의 종류는 아래 [Fig 2](#fig-2) 의 차례로 내려간다.

```text
Norm axioms: positivity, homogeneity, triangle inequality
 |
 +-- Vector norm
 |    +-- Lp norm (p >= 1) .......... (sum |x_i|^p)^(1/p)
 |         +-- L1 (Manhattan) ....... sum |x_i|
 |         +-- L2 (Euclidean) ....... (sum x_i^2)^(1/2), the length
 |         +-- L-infinity (max) ..... max |x_i|
 |
 +-- Matrix norm
      +-- by elements
      |    +-- Frobenius ............ (sum a_ij^2)^(1/2)
      +-- by singular values
           +-- spectral ............. sigma_1, the largest singular value
           +-- nuclear .............. sum sigma_i

Not a norm: "L0" = number of nonzero elements (fails homogeneity)
```

<a id="fig-2"></a>
Fig 2. Hierarchy of norms

- 양수성, 동차성, 삼각부등식의 세 조건을 만족해야 norm 이다. 세 조건의 식은 [꼭지 4](#4-definition) 의 식 (1), 식 (2), 식 (3) 에 있다. 조건만 지키면 되므로 크기를 재는 방식이 여럿이다.
- Lp norm 은 $p \ge 1$ 일 때 norm 이다. $p = 1, 2, \infty$ 가 L1, L2, L∞ 이다.
- Frobenius norm 은 행렬을 vector 로 펼친 L2 norm 이고, 특이값 제곱합의 제곱근과도 같다.
- L0 는 이름에 norm 이 붙지만 동차성을 만족하지 않아 norm 이 아니다.

### 3.1 Comparison

Vector norm 을 $\mathbf{x} = [3, 4]^\top$ 에서 견주면 아래 Table 1 이다.

Table 1. Vector norms of x = [3, 4]

| Norm           | Definition                 | Value                | Unit ball | Used in                       |
| :------------: | :------------------------: | :------------------: | :-------: | :---------------------------: |
| L1 (Manhattan) | $\sum_i \lvert x_i \rvert$ | $3 + 4 = 7$          | Diamond   | Lasso, MAE                    |
| L2 (Euclidean) | $(\sum_i x_i^2)^{1/2}$     | $(9 + 16)^{1/2} = 5$ | Circle    | 최소제곱, ridge, 최소 노름 해 |
| L∞ (max)       | $\max_i \lvert x_i \rvert$ | $\max(3, 4) = 4$     | Square    | 원소별 한계                   |

Unit ball 은 norm 이 1 이하인 vector 의 집합이며, Table 1 의 Unit ball 열은 2 차원에서 그 집합의 경계가 이루는 모양이다. "ball" 은 L2 norm 일 때 집합이 둥근 모양인 데서 온 이름이며, 모양과 관계없이 norm 이 1 이하인 집합을 모두 unit ball 이라 부른다. 그래서 같은 정의라도 L1 norm 이면 Diamond, L∞ norm 이면 Square 가 된다. L1 은 좌표축을 따라 가로와 세로로만 움직인 거리이고, L2 는 원점에서 잰 직선 거리이며, L∞ 는 원소 가운데 가장 크게 벗어난 하나만 본다. 세 값은 식 (6) 대로 $4 \le 5 \le 7$ 이다.

Matrix norm 을 식 (10) 의 $\mathbf{A}$ 에서 견주면 아래 Table 2 이다. $\mathbf{A}$ 의 특이값은 $\sigma_1 = \sqrt{3} \approx 1.732$, $\sigma_2 = 1$ 이다.

Table 2. Matrix norms of A in equation (10)

| Norm      | Definition                                                | Value                        | Used in                                |
| :-------: | :-------------------------------------------------------: | :--------------------------: | :------------------------------------: |
| Frobenius | $(\sum_{i,j} a_{ij}^2)^{1/2} = (\sum_i \sigma_i^2)^{1/2}$ | $4^{1/2} = 2$                | 저차원 근사의 오차                     |
| Spectral  | $\sigma_1$                                                | $\sqrt{3} \approx 1.732$     | 가장 크게 늘어나는 배율, 계산의 안정성 |
| Nuclear   | $\sum_i \sigma_i$                                         | $1 + \sqrt{3} \approx 2.732$ | rank 를 줄이는 regularization          |

## 4. Definition

> **Definition (norm)**: $\mathbb{R}^n$ 을 원소가 실수 $n$ 개인 열 vector 전체의 집합이라 할 때, $\mathbb{R}^n$ 위의 norm 은 $\mathbb{R}^n$ 의 vector 마다 실수 하나를 주는 함수 $\lVert \cdot \rVert : \mathbb{R}^n \to \mathbb{R}$ 로서, 모든 $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ 과 모든 실수 $c$ 에 대해 양수성 (식 (1)), 동차성 (식 (2)), 삼각부등식 (식 (3)) 의 세 조건을 모두 만족하는 함수이다. $m \times n$ 행렬 전체의 집합 $\mathbb{R}^{m \times n}$ 위의 norm 도 vector 대신 행렬을 넣어 같은 세 조건으로 정의한다.

세 조건은 아래이다.

양수성 (positivity) 은 norm 이 0 이상이고, 0 이 되는 것은 zero vector 뿐이라는 조건이다.

```math
\lVert \mathbf{x} \rVert \ge 0, \quad \lVert \mathbf{x} \rVert = 0 \iff \mathbf{x} = \mathbf{0}
\hspace{19em} (1)
```

동차성 (homogeneity) 은 vector 에 상수를 곱하면 norm 에 그 상수의 절댓값이 곱해진다는 조건이다.

```math
\lVert c\,\mathbf{x} \rVert = \lvert c \rvert \, \lVert \mathbf{x} \rVert
\hspace{19em} (2)
```

삼각부등식 (triangle inequality) 은 두 vector 의 합의 norm 이 두 norm 의 합을 넘지 않는다는 조건이다.

```math
\lVert \mathbf{x} + \mathbf{y} \rVert \le \lVert \mathbf{x} \rVert + \lVert \mathbf{y} \rVert
\hspace{19em} (3)
```

삼각부등식의 이름과 숫자 예시는 [Appendix C](#appendix-c-triangle-inequality) 에 있다. 두 vector $\mathbf{x}$, $\mathbf{y}$ 사이의 거리는 두 vector 의 차 $\mathbf{x} - \mathbf{y}$ 의 norm $\lVert \mathbf{x} - \mathbf{y} \rVert$ 로 정의한다. 거리는 어느 norm 을 쓰느냐에 따라 달라져, L2 norm 이면 원점에서 잰 직선 거리처럼 두 점을 곧게 이은 거리가 되고 L1 norm 이면 좌표축을 따라 가로와 세로로만 움직인 거리가 된다.

Lp norm 은 아래 식이며, $p \to \infty$ 의 극한이 L∞ norm 이다.

```math
\lVert \mathbf{x} \rVert_p = \left( \sum_{i=1}^{n} \lvert x_i \rvert^p \right)^{1/p} \quad (p \ge 1)
\hspace{19em} (4)
```

```math
\lVert \mathbf{x} \rVert_\infty = \max_i \lvert x_i \rvert
\hspace{19em} (5)
```

아래첨자 없이 $\lVert \mathbf{x} \rVert$ 로 적으면 L2 norm 이다. 같은 vector 에서 세 norm 은 언제나 아래 순서이며, 유도는 [Appendix B](#appendix-b-derivation) 에 있다.

```math
\lVert \mathbf{x} \rVert_\infty \le \lVert \mathbf{x} \rVert_2 \le \lVert \mathbf{x} \rVert_1
\hspace{19em} (6)
```

Matrix norm 은 특이값으로 아래처럼 적는다. $r$ 은 $\mathbf{A}$ 의 rank 이다.

```math
\lVert \mathbf{A} \rVert_F = \left( \sum_{i=1}^{r} \sigma_i^2 \right)^{1/2}
\hspace{19em} (7)
```

```math
\lVert \mathbf{A} \rVert_2 = \sigma_1
\hspace{19em} (8)
```

```math
\lVert \mathbf{A} \rVert_\ast = \sum_{i=1}^{r} \sigma_i
\hspace{19em} (9)
```

특이값은 vector norm 에는 필요 없고, matrix norm 가운데 spectral norm 과 nuclear norm 을 정의할 때 필요하다. 식 (8) 의 spectral norm 은 가장 큰 특이값 $\sigma_1$ 이고 식 (9) 의 nuclear norm 은 특이값의 합이므로, 이 두 norm 은 특이값 없이 적을 수 없다. Frobenius norm 은 원소 제곱합의 제곱근으로 정의되어 특이값 없이도 구할 수 있지만, 특이값 제곱합의 제곱근과 같은 값이어서 세 matrix norm 을 식 (7), 식 (8), 식 (9) 처럼 모두 특이값으로 적을 수 있다.

Table 2 의 숫자는 아래 행렬에서 구했다.

```math
\mathbf{A} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}
\hspace{19em} (10)
```

식 (10) 의 $\mathbf{A}$ 에서도 Frobenius norm 2 는 $(\sigma_1^2 + \sigma_2^2)^{1/2} = (3 + 1)^{1/2}$ 와 같다.

## 5. Application

Norm 은 vector 나 행렬의 크기, 또는 두 vector 나 두 행렬의 차이를 숫자 하나로 잰 값이며, 쓰는 곳은 아래 Table 3 이다.

Table 3. Uses of norms

| Use                           | Norm                          | Role                                                                                                                        |
| :---------------------------: | :---------------------------: | :-------------------------------------------------------------------------------------------------------------------------: |
| Error measurement             | Frobenius norm                | 예측 행렬과 실제 행렬의 차이를 숫자 하나로 만든다. 원소가 $N$ 개이면 RMSE 는 이 값을 $\sqrt{N}$ 으로 나눈 값이다            |
| Regularization                | L2 norm, L1 norm              | weight 가 너무 커지지 않게 손실에 penalty 를 더한다. Ridge 는 L2 norm 의 제곱, lasso 는 L1 norm 을 쓴다                     |
| Numerical stability           | Condition number              | 입력이 조금 변할 때 결과가 얼마나 흔들리는지 잰다. 값이 크면 feature 가 서로 겹쳐 회귀 계수가 불안정하다                    |
| Approximation and convergence | Frobenius norm, spectral norm | 특이값 분해로 줄인 행렬과 원래 행렬의 차이를 재고, 반복 계산에서 새 값과 옛 값의 차이의 norm 이 허용 오차보다 작으면 멈춘다 |

Table 3 의 쓰는 곳은 오차 크기 재기, weight 의 penalty, 계산이 안정한지 확인, 반복을 멈출 시점 판정의 넷이다. 어느 norm 을 쓰느냐에 따라 큰 오차를 더 크게 벌주는지 (L2) 와 작은 값을 0 으로 만드는지 (L1) 가 달라진다. Condition number 는 행렬의 norm 과 역행렬의 norm 의 곱 $\lVert \mathbf{A} \rVert \, \lVert \mathbf{A}^{-1} \rVert$ 이며, spectral norm 으로 재면 가장 큰 특이값을 가장 작은 특이값으로 나눈 값이다. 식 (10) 의 $\mathbf{A}$ 에서는 $\sqrt{3} / 1 \approx 1.732$ 이다.

Norm 마다의 쓰임은 아래이다.

- **L2 의 제곱**: 최소제곱 손실 $\lVert \mathbf{A}\mathbf{x} - \mathbf{b} \rVert_2^2$ 과 ridge regularization 이 쓴다. 해가 무수히 많을 때 L2 norm 이 가장 작은 해를 고르면 최소 노름 해이며, [Minimum Norm Solution Primer](../Minimum-Norm-Solution/minimum-norm-solution-primer-ko.md) 에 있다. 제곱하면 미분이 선형이 되어 해를 식으로 구할 수 있다.
- **L1**: Lasso regularization 이 쓴다. Unit ball 이 마름모여서 손실의 등고선이 꼭짓점에서 먼저 닿기 쉽고, 꼭짓점에서는 일부 원소가 0 이므로 일부 weight 가 0 이 되어 쓰는 feature 가 줄어든다. 손실로 쓰면 MAE 이며, 이상치 하나가 손실에 더하는 몫이 L2 보다 작다.
- **L∞**: 원소 가운데 가장 큰 편차를 제한할 때 쓴다. 모든 원소가 spec 한계 안에 들어야 하는 판정이 이 경우이다.
- **Frobenius**: 특이값 분해로 만든 저차원 근사의 오차를 잰다. 자세한 내용은 [Singular Value Decomposition Primer](../Singular-Value-Decomposition/singular-value-decomposition-primer-ko.md) 에 있다.
- **Spectral**: 행렬이 vector 를 가장 크게 늘이는 배율이다. 가장 작은 특이값과의 비가 크면 역행렬과 pseudo-inverse 의 계산이 입력의 작은 변화에 크게 흔들린다.
- **Nuclear**: rank 가 낮은 행렬을 찾는 regularization 에 쓴다. 특이값의 합을 줄이면 작은 특이값이 0 이 되어 rank 가 줄어든다.

"Norm 을 구하라" 를 "길이를 구하라" 로 읽는 것은 L2 norm 일 때만 맞다. L1 regularization 이나 원소별 한계를 다룰 때는 어느 norm 인지를 가려야 결과가 달라지는 이유를 설명할 수 있다.

---

## Appendix A. Terminology

- **condition number**: 행렬의 norm 과 역행렬의 norm 의 곱. 입력의 작은 변화가 결과를 얼마나 크게 바꾸는지를 나타낸다.
- **lasso**: 손실에 weight 의 L1 norm 을 더해 일부 weight 를 0 으로 만드는 선형 회귀.
- **MAE (mean absolute error)**: 오차 절댓값의 평균. 오차 vector 의 L1 norm 을 원소 수로 나눈 값이다.
- **rank**: 행렬에서 서로 일차독립인 열의 최대 개수.
- **ridge**: 손실에 weight 의 L2 norm 의 제곱을 더해 weight 를 작게 만드는 선형 회귀.
- **RMSE (root mean square error)**: 오차 제곱의 평균의 제곱근. 오차 vector 의 L2 norm 을 원소 수의 제곱근으로 나눈 값이다.
- **unit ball**: norm 이 1 이하인 vector 의 집합. Table 1 의 Unit ball 열은 2 차원에서 그 집합의 경계가 이루는 모양이다.
- **동차성 (homogeneity)**: vector 에 상수를 곱하면 norm 에 그 상수의 절댓값이 곱해진다는 조건. 식 (2) 이다.
- **삼각부등식 (triangle inequality)**: 두 vector 의 합의 norm 이 두 norm 의 합을 넘지 않는다는 조건. 식 (3) 이다.
- **양수성 (positivity)**: norm 이 0 이상이고, 0 이 되는 것은 zero vector 뿐이라는 조건. 식 (1) 이다.
- **최소 노름 해 (minimum norm solution)**: 해가 무수히 많은 선형방정식의 해 가운데 norm 이 가장 작은 해.
- **최소제곱 (least squares)**: 잔차의 L2 norm 의 제곱을 가장 작게 하는 방법.
- **특이값 (singular value)**: 특이값 분해 $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ 에서 $\boldsymbol{\Sigma}$ 의 대각 원소.
- **특이값 분해 (singular value decomposition)**: 행렬을 $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ 로 나누는 분해. $\mathbf{U}$ 와 $\mathbf{V}$ 의 열은 단위 vector 이다.

## Appendix B. Derivation

식 (6) 은 두 부등식을 제곱하여 견주면 나온다.

```math
\begin{aligned}
\lVert \mathbf{x} \rVert_\infty^2
&= \max_i x_i^2 \le \sum_{i} x_i^2 = \lVert \mathbf{x} \rVert_2^2 \\
\lVert \mathbf{x} \rVert_1^2
&= \Big( \sum_{i} \lvert x_i \rvert \Big)^2 = \sum_{i} x_i^2 + \sum_{i \ne j} \lvert x_i \rvert \lvert x_j \rvert \ge \lVert \mathbf{x} \rVert_2^2
\end{aligned}
\hspace{19em} (11)
```

- 첫째 줄: 가장 큰 제곱 하나는 0 이상인 제곱을 모두 더한 값을 넘지 않는다.
- 둘째 줄: 합의 제곱을 전개하면 제곱의 합에 0 이상인 교차항이 더해진다.

세 norm 은 모두 0 이상이므로 제곱의 순서가 norm 의 순서와 같다. 따라서 식 (6) 이 성립한다.

```math
\lVert \mathbf{x} \rVert_\infty \le \lVert \mathbf{x} \rVert_2 \le \lVert \mathbf{x} \rVert_1
\hspace{19em} (6)
```

## Appendix C. Triangle Inequality

삼각부등식 (식 (3)) 은 삼각형의 한 변이 나머지 두 변의 합보다 길 수 없다는 성질에서 온 이름이다. $\mathbf{x}$ 를 한 변, $\mathbf{y}$ 를 그 끝에서 이어 붙인 변으로 두면 $\mathbf{x} + \mathbf{y}$ 는 처음과 끝을 잇는 셋째 변이고, 식 (3) 은 그 셋째 변의 norm 이 두 변의 norm 의 합을 넘지 않는다는 뜻이다. 등호는 두 vector 가 같은 방향을 가리켜 삼각형이 한 직선으로 납작해질 때 성립한다.

식 (10) 의 $\mathbf{A}$ 의 두 열 $\mathbf{x} = [1, 0, 1]^\top$, $\mathbf{y} = [0, 1, 1]^\top$ 에서 세 norm 으로 식 (3) 을 확인하면 아래 Table 4 이다. 두 vector 의 합은 $\mathbf{x} + \mathbf{y} = [1, 1, 2]^\top$ 이다.

Table 4. Triangle inequality for the two columns of A in equation (10)

| Norm | $\lVert \mathbf{x} \rVert$ | $\lVert \mathbf{y} \rVert$ | $\lVert \mathbf{x} \rVert + \lVert \mathbf{y} \rVert$ | $\lVert \mathbf{x} + \mathbf{y} \rVert$ |
| :--: | :------------------------: | :------------------------: | :---------------------------------------------------: | :-------------------------------------: |
| L1   | 2                          | 2                          | 4                                                     | 4                                       |
| L2   | $\sqrt{2} \approx 1.414$   | $\sqrt{2} \approx 1.414$   | $2\sqrt{2} \approx 2.828$                             | $\sqrt{6} \approx 2.449$                |
| L∞   | 1                          | 1                          | 2                                                     | 2                                       |

세 norm 모두에서 $\lVert \mathbf{x} + \mathbf{y} \rVert$ 가 $\lVert \mathbf{x} \rVert + \lVert \mathbf{y} \rVert$ 를 넘지 않는다. L1 과 L∞ 에서는 등호가 성립하고, L2 에서는 $2.449 \lt 2.828$ 로 등호가 성립하지 않는다. L1 은 두 vector 의 원소가 모두 0 이상이어서 절댓값의 합이 그대로 더해지고, L∞ 는 두 vector 의 가장 큰 원소 1 이 셋째 원소로 겹쳐 그대로 더해진다. L2 는 $\mathbf{x}$ 와 $\mathbf{y}$ 가 같은 방향이 아니어서 등호가 성립하지 않는다.
