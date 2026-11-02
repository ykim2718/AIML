# Types of Linear Systems
Rev. 0 | Created: 2026-11-02 | Updated: 2026-11-02 21:30 UTC

## 1. Purpose

- **Problem Statement**: 선형방정식은 식과 미지수의 수, 해의 개수, 우변이라는 서로 다른 기준으로 나뉘는데, 기준을 섞으면 어느 풀이를 써야 하는지 가릴 수 없다.
- **Goal**: 독자가 주어진 선형방정식의 종류를 세 기준으로 가리고, 해의 개수를 rank 로 판정하여 쓸 풀이를 고를 수 있게 한다.
- **Non-Goal**: 각 풀이의 유도와 숫자 예시, 그리고 연립방정식을 푸는 algorithm 은 다루지 않는다.

## 2. Summary

선형방정식 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 는 식의 수와 미지수의 수, 해의 개수, 우변이 zero vector 인지의 세 기준으로 나뉜다.
해의 개수는 식과 미지수의 수가 아니라 $\mathbf{A}$ 와 첨가 행렬 $[\mathbf{A} \mid \mathbf{b}]$ 의 rank 로 정해진다.
해가 하나이면 역행렬, 해가 없으면 최소제곱 해, 해가 무수히 많으면 최소 노름 해를 쓴다.

## 3. Taxonomy and its Hierarchy

$\mathbf{A}$ 는 $m \times n$ 계수 행렬, $\mathbf{x}$ 는 미지수 $n$ 개로 이루어진 vector, $\mathbf{b}$ 는 우변 $m$ 개로 이루어진 vector 이다. $m$ 은 식의 수, $n$ 은 미지수의 수이고, rank 는 $\mathbf{A}$ 에서 서로 일차독립인 열의 최대 개수이다. 첨가 행렬 $[\mathbf{A} \mid \mathbf{b}]$ 는 $\mathbf{A}$ 의 오른쪽에 $\mathbf{b}$ 를 열 하나로 붙인 $m \times (n + 1)$ 행렬이다.

세 기준은 서로 독립이며, 축마다의 값은 아래 [Fig 1](#fig-1) 이다.

```text
Axis 1. Shape (m vs n)
  square               m = n
  overdetermined       m > n   (more equations than unknowns)
  underdetermined      m < n   (more unknowns than equations)

Axis 2. Number of solutions (rank test)
  none                 rank(A) < rank([A | b])
  exactly one          rank(A) = rank([A | b]) = n
  infinitely many      rank(A) = rank([A | b]) < n

Axis 3. Right-hand side
  homogeneous          b = 0
  nonhomogeneous       b != 0
```

<a id="fig-1"></a>
Fig 1. Three axes of linear systems

- Axis 1 은 행렬의 모양만 본다. 식이 겹치거나 모순되는지는 보지 않는다.
- Axis 2 는 rank 로 해의 개수를 정한다. 이 축이 풀이를 고르는 기준이다.
- Axis 3 은 우변만 본다. 동차 선형방정식은 zero vector 가 언제나 해이므로 Axis 2 의 none 이 나오지 않는다.

해의 개수를 가리는 판정은 아래 [Fig 2](#fig-2) 의 차례로 내려간다.

```text
Ax = b
 |
 +-- rank(A) < rank([A | b]) ...... no solution
 |     b has a nonzero left null space component
 |     -> least squares solution   x = (A^T A)^-1 A^T b      (rank(A) = n)
 |
 +-- rank(A) = rank([A | b])
       |
       +-- rank(A) = n ............ exactly one solution
       |     null space holds only the zero vector
       |     -> inverse              x = A^-1 b               (m = n)
       |
       +-- rank(A) < n ............ infinitely many solutions
             any null space vector added to a solution is a solution
             -> minimum norm solution  x = A^T (A A^T)^-1 b   (rank(A) = m)
```

<a id="fig-2"></a>
Fig 2. Hierarchy of the rank test

- 첫 갈림은 해가 있는지이다. $\mathbf{b}$ 가 $\mathbf{A}$ 의 열들의 일차결합이면 붙인 열이 새 방향을 더하지 않아 두 rank 가 같다.
- 둘째 갈림은 해가 하나인지이다. $\mathbf{A}$ 의 rank 가 $n$ 보다 작으면 null space 에 zero vector 가 아닌 vector 가 있다.
- 한 단계 내려갈 때마다 해의 조건이 약해진다. 해가 없으면 식을 모두 만족하는 것을 포기하고 잔차를 가장 작게 하며, 해가 무수히 많으면 해 가운데 norm 이 가장 작은 하나를 고른다.
- 세 풀이는 모두 pseudo-inverse $\mathbf{A}^{+}\mathbf{b}$ 로 하나로 적힌다.

### 3.1 Comparison

Axis 1 의 모양마다 흔히 나오는 해의 개수와 쓰는 풀이는 아래 Table 1 이다.

Table 1. Shape of a linear system

| Type                                  | Condition | Usual number of solutions | Solution used                                                            | Rank condition of the formula |
| :-----------------------------------: | :-------: | :-----------------------: | :----------------------------------------------------------------------: | :---------------------------: |
| 정방 시스템 (square system)           | $m = n$   | 하나                      | $\mathbf{A}^{-1}\mathbf{b}$                                              | $\det(\mathbf{A}) \ne 0$      |
| 과결정 시스템 (overdetermined system) | $m \gt n$ | 없음                      | 최소제곱 해 $(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}$  | rank $= n$                    |
| 부정 방정식 (underdetermined system)  | $m \lt n$ | 있으면 무수히 많음        | 최소 노름 해 $\mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top)^{-1}\mathbf{b}$ | rank $= m$                    |

Usual number of solutions 열은 식이 겹치지도 모순되지도 않을 때의 경우이다. 과결정 시스템이라도 식이 겹치면 해가 하나 있을 수 있고, 부정 방정식이라도 식끼리 모순되면 해가 없을 수 있다. 그래서 실제 해의 개수는 아래 Table 2 의 rank 판정으로 정한다.

Table 2. Number of solutions by the rank test

| Number of solutions | Rank condition                                                                  | Null space                         | Left null space component of b |
| :-----------------: | :-----------------------------------------------------------------------------: | :--------------------------------: | :----------------------------: |
| 없음                | $\mathrm{rank}(\mathbf{A}) \lt \mathrm{rank}([\mathbf{A} \mid \mathbf{b}])$     | 판정에 쓰지 않음                   | zero vector 가 아님            |
| 하나                | $\mathrm{rank}(\mathbf{A}) = \mathrm{rank}([\mathbf{A} \mid \mathbf{b}]) = n$   | zero vector 만 있음                | zero vector                    |
| 무수히 많음         | $\mathrm{rank}(\mathbf{A}) = \mathrm{rank}([\mathbf{A} \mid \mathbf{b}]) \lt n$ | zero vector 가 아닌 vector 가 있음 | zero vector                    |

동차 선형방정식 $\mathbf{A}\mathbf{x} = \mathbf{0}$ 은 $\mathbf{b} = \mathbf{0}$ 이어서 두 rank 가 언제나 같으므로 Table 2 의 아래 두 행만 나온다. 비동차 선형방정식은 세 행이 모두 나올 수 있다.

## 4. Application

Table 2 로 해의 개수를 정한 뒤 아래 풀이를 쓴다.

- **역행렬**: 해가 하나이고 $m = n$ 일 때 $\mathbf{A}^{-1}\mathbf{b}$ 로 구한다. $\det(\mathbf{A})$ 가 0 에 가까우면 역행렬의 원소가 매우 커져 결과를 믿을 수 없다. 자세한 내용은 [Matrix Operations Primer](../Matrix-Operations/matrix-operations-primer-ko.md) 에 있다.
- **최소제곱 해**: 해가 없을 때 잔차 $\lVert \mathbf{A}\mathbf{x} - \mathbf{b} \rVert$ 를 가장 작게 하는 $\mathbf{x}$ 를 구한다. 측정값 여러 개로 직선 하나를 맞추는 선형 회귀가 이 경우이다. 잔차는 $\mathbf{b}$ 의 left null space 성분이며, [Left Null Space Primer](../Left-Null-Space/left-null-space-primer-ko.md) 에 있다.
- **최소 노름 해**: 해가 무수히 많을 때 해 가운데 norm 이 가장 작은 하나를 고른다. Sample 수보다 feature 수가 많은 선형 회귀가 이 경우이다. [Minimum Norm Solution Primer](../Minimum-Norm-Solution/minimum-norm-solution-primer-ko.md) 에 있다.
- **pseudo-inverse**: 위 세 풀이를 $\mathbf{A}^{+}\mathbf{b}$ 하나로 적는다. Rank 가 $m$ 과 $n$ 보다 모두 작아 Table 1 의 식을 쓸 수 없으면 특이값 분해로 $\mathbf{A}^{+}$ 를 구한다. [Pseudo-Inverse Primer](../Pseudo-Inverse/pseudo-inverse-primer-ko.md) 에 있다.

해의 개수를 첨가 행렬로 세는 숫자 예시는 [Null Space Primer](../Null-Space/null-space-primer-ko.md) 의 Appendix C 에 있다.

---

## Appendix A. Terminology

- **left null space**: $\mathbf{A}^\top\mathbf{y} = \mathbf{0}$ 인 $\mathbf{y}$ 의 집합. Column space 와 직교한다.
- **norm (Euclidean norm)**: vector 원소의 제곱을 모두 더한 값의 제곱근. Vector 의 길이이며, $\lVert \mathbf{x} \rVert = (x_1^2 + \cdots + x_n^2)^{1/2}$ 로 적는다.
- **null space**: $\mathbf{A}\mathbf{x} = \mathbf{0}$ 인 $\mathbf{x}$ 의 집합.
- **pseudo-inverse**: 정사각이 아니거나 역행렬이 없는 행렬에도 정의되는 역행렬의 일반화. Moore–Penrose pseudo-inverse 라고도 한다.
- **계수 (rank)**: 행렬에서 서로 일차독립인 열의 최대 개수.
- **과결정 시스템 (overdetermined system)**: 식의 수가 미지수의 수보다 많은 선형방정식.
- **동차 선형방정식 (homogeneous linear equation)**: 우변이 zero vector 인 선형방정식 $\mathbf{A}\mathbf{x} = \mathbf{0}$.
- **부정 방정식 (underdetermined system)**: 미지수의 수가 식의 수보다 많은 선형방정식.
- **비동차 선형방정식 (nonhomogeneous linear equation)**: 우변이 zero vector 가 아닌 선형방정식.
- **역행렬 (inverse matrix)**: 정사각 행렬 $\mathbf{A}$ 에 대해 $\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$ 인 행렬. $\mathbf{I}$ 는 단위 행렬이다.
- **일차결합 (linear combination)**: vector 여러 개에 각각 상수를 곱해 모두 더한 vector.
- **잔차 (residual)**: 우변과 근사한 값의 차이 $\mathbf{b} - \mathbf{A}\mathbf{x}$.
- **정방 시스템 (square system)**: 식의 수와 미지수의 수가 같은 선형방정식.
- **첨가 행렬 (augmented matrix)**: 계수 행렬 $\mathbf{A}$ 의 오른쪽에 우변 $\mathbf{b}$ 를 열 하나로 붙인 행렬 $[\mathbf{A} \mid \mathbf{b}]$.
- **최소 노름 해 (minimum norm solution)**: 해가 무수히 많은 선형방정식의 해 가운데 norm 이 가장 작은 해.
- **최소제곱 해 (least squares solution)**: $\lVert \mathbf{A}\mathbf{x} - \mathbf{b} \rVert$ 를 가장 작게 하는 $\mathbf{x}$.
- **특이값 분해 (singular value decomposition)**: 행렬을 $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ 로 나누는 분해. $\mathbf{U}$ 와 $\mathbf{V}$ 의 열은 단위 vector 이고 $\boldsymbol{\Sigma}$ 의 대각 원소가 특이값이다.
