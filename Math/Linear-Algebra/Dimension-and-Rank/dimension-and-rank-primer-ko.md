# Dimension and Rank Primer
Rev. 17 | Created: 2026-11-03 | Updated: 2026-11-07 08:10 UTC

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Notation](#3-notation)
- [4. Definitions](#4-definitions)
- [5. Interpretation](#5-interpretation)
  - [5.1 Dimension of a Space and Rank of a Matrix](#51-dimension-of-a-space-and-rank-of-a-matrix)
  - [5.2 Row Rank and Column Rank](#52-row-rank-and-column-rank)
- [6. Worked Example](#6-worked-example)
- [7. Application](#7-application)
  - [7.1 Rank Conditions of a Linear System](#71-rank-conditions-of-a-linear-system)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Finding a Basis](#appendix-b-finding-a-basis)

## 1. Purpose

- **Problem Statement**: $m \ne n$ 인 $m \times n$ 행렬에서 행은 $\mathbb{R}^n$ 에, 열은 $\mathbb{R}^m$ 에 있어, 공간의 차원과 행렬의 rank 를 섞어 읽으면 row space 와 column space 의 rank 가 다를 수 있다고 잘못 읽는다.
- **Goal**: 독자가 vector space 의 차원과 행렬의 rank 를 구별하고, $m \ne n$ 인 행렬에서도 row space 의 차원과 column space 의 차원이 같은 값임을 숫자로 확인할 수 있게 한다.
- **Non-Goal**: 무한 차원 vector space 와, 부동소수점 계산에서 rank 를 정하는 문턱값은 다루지 않는다.

## 2. Summary

차원 (dimension) 은 vector space 의 기저에 든 vector 의 개수이고, rank 는 행렬 $\mathbf{A}$ 의 column space 의 차원이다.
$\mathbf{A}$ 가 $m \times n$ 이고 $m \ne n$ 이어도 row space 의 차원과 column space 의 차원은 언제나 같은 $\mathrm{rank}(\mathbf{A})$ 이다.
$m$ 과 $n$ 은 열과 행이 놓인 공간 $\mathbb{R}^m$ 과 $\mathbb{R}^n$ 의 차원이고, rank 는 그 안에서 서로 일차독립인 열이나 행의 최대 개수이다.

## 3. Notation

이 문서가 쓰는 기호는 아래 Table 1 이다.

Table 1. Symbols

| Symbol                                         | Meaning                                                                           | Shape                      | Assumption                                               |
| :--------------------------------------------: | :-------------------------------------------------------------------------------: | :------------------------: | :------------------------------------------------------: |
| $\mathbf{A}$                                   | 행렬                                                                              | $m \times n$               | 실수 원소                                                |
| $m$, $n$                                       | $\mathbf{A}$ 의 행의 수와 열의 수                                                 | scalar                     | 양의 정수                                                |
| $\mathbb{R}^n$                                 | 원소가 실수 $n$ 개인 열 vector 전체의 집합                                        | vector space               | 차원 $n$                                                 |
| $V$                                            | 차원을 재는 vector space                                                          | vector space               | 유한 차원                                                |
| $\dim V$                                       | $V$ 의 차원                                                                       | scalar                     | 0 이상의 정수                                            |
| $\mathrm{Col}(\mathbf{A})$                     | $\mathbf{A}$ 의 column space                                                      | $\mathbb{R}^m$ 의 부분공간 | 열들의 일차결합 전체                                     |
| $\mathrm{Row}(\mathbf{A})$                     | $\mathbf{A}$ 의 row space                                                         | $\mathbb{R}^n$ 의 부분공간 | 행들의 일차결합 전체                                     |
| $\mathrm{Null}(\mathbf{A})$                    | $\mathbf{A}$ 의 null space                                                        | $\mathbb{R}^n$ 의 부분공간 | $\mathbf{A}\mathbf{x} = \mathbf{0}$ 인 $\mathbf{x}$ 전체 |
| $\mathrm{rank}(\mathbf{A})$, $r$               | $\mathbf{A}$ 의 계수                                                              | scalar                     | $0 \le r \le \min(m, n)$                                 |
| $\mathbf{x}$                                   | 입력 vector                                                                       | $n \times 1$               | 실수 원소                                                |
| $\mathbf{y}$                                   | left null space 의 vector                                                         | $m \times 1$               | $\mathbf{A}^\top\mathbf{y} = \mathbf{0}$                 |
| $\mathbf{r}_1$, $\mathbf{r}_2$                 | Worked Example 에서 $\mathbf{A}$ 의 첫째 행과 둘째 행                             | $3 \times 1$               | 열 vector 로 적음                                        |
| $\mathbf{c}_1$, $\mathbf{c}_2$, $\mathbf{c}_3$ | Worked Example 에서 $\mathbf{A}$ 의 첫째, 둘째, 셋째 열                           | $2 \times 1$               | 실수 원소                                                |
| $x_1$, $x_2$                                   | $\mathbf{c}_3$ 를 $\mathbf{c}_1$ 과 $\mathbf{c}_2$ 의 일차결합으로 적을 때의 계수 | scalar                     | 실수                                                     |

특수 기호는 아래처럼 읽는다.

- $\mathbb{R}^n$: 영어로 "R n" 또는 "real n-space", 한국어로 "n 차원 실수 공간" 이라 읽는다.
- $\dim V$: 영어로 "the dimension of V", 한국어로 "V 의 차원" 이라 읽는다.
- $[1, 0, 0]^\top$ 의 $\top$: 영어로 "transpose", 한국어로 "전치" 라 읽는다. 가로로 적은 원소를 열 vector 로 바꾼다는 뜻이다.
- $\det$: 영어로 "the determinant of", 한국어로 "행렬식" 이라 읽는다.

## 4. Definitions

> **Definition: Dimension**<br>
> Vector space $V$ 의 기저 (basis) 는 서로 일차독립이면서 일차결합으로 $V$ 의 모든 vector 를 만드는 vector 의 집합이고, $V$ 의 차원 $\dim V$ 는 그 기저에 든 vector 의 개수이다. $V$ 의 기저는 여럿이지만 그 개수는 모두 같다.

> **Definition: Rank**<br>
> $m \times n$ 행렬 $\mathbf{A}$ 의 rank $\mathrm{rank}(\mathbf{A})$ 는 column space $\mathrm{Col}(\mathbf{A})$ 의 차원, 곧 $\mathbf{A}$ 의 열 가운데 서로 일차독립인 열의 최대 개수이다.

```math
\mathrm{rank}(\mathbf{A}) = \dim \mathrm{Col}(\mathbf{A})
\hspace{19em} (1)
```

$\mathbb{R}^3$ 의 차원은 3 이고, $[1, 0, 0]^\top$, $[0, 1, 0]^\top$, $[0, 0, 1]^\top$ 이 그 기저의 하나이다. $\mathbb{R}^3$ 안에서 원점을 지나는 평면은 일차독립인 vector 두 개로 생성되므로 차원이 2 이고, 원점을 지나는 직선은 vector 하나로 생성되므로 차원이 1 이다. 행렬에서 기저를 구하는 방법은 [Appendix B](#appendix-b-finding-a-basis) 에 있다.

## 5. Interpretation

차원은 vector space 에 매기는 수이고, rank 는 행렬에 매기는 수이다. 행렬의 rank 는 그 행렬이 만드는 부분공간인 column space 의 차원이므로, rank 는 차원의 한 경우이다.

### 5.1 Dimension of a Space and Rank of a Matrix

두 수를 견주면 아래 Table 2 와 같다.

Table 2. Dimension and rank

| Aspect     | Dimension                | Rank                                                                                        |
| :--------: | :----------------------: | :-----------------------------------------------------------------------------------------: |
| Applies to | Vector space             | 행렬                                                                                        |
| Counts     | 기저에 든 vector 의 개수 | 서로 일차독립인 열 (또는 행) 의 최대 개수                                                   |
| Example    | $\dim \mathbb{R}^3 = 3$  | $\mathrm{rank}(\mathbf{A}) = \dim \mathrm{Col}(\mathbf{A}) = \dim \mathrm{Row}(\mathbf{A})$ |

$m \times n$ 행렬 $\mathbf{A}$ 에는 차원이 다른 공간 다섯이 함께 나온다. 열이 놓인 $\mathbb{R}^m$ 의 차원은 $m$, 행이 놓인 $\mathbb{R}^n$ 의 차원은 $n$ 이고, $\mathrm{Col}(\mathbf{A})$ 와 $\mathrm{Row}(\mathbf{A})$ 의 차원은 $r$, $\mathrm{Null}(\mathbf{A})$ 의 차원은 $n - r$ 이다. Null space 의 차원 $n - r$ 은 rank-nullity 정리에서 나오며, 그 유도는 [Null Space Primer](../Null-Space/null-space-primer-ko.md) 에 있다. 정리의 해석과 숫자 예시는 [Rank-Nullity Theorem Primer](../Rank-Nullity-Theorem/rank-nullity-theorem-primer-ko.md) 에 있다.

### 5.2 Row Rank and Column Rank

$\mathbf{A}$ 의 row space 의 차원은 column space 의 차원과 언제나 같다.

```math
\dim \mathrm{Row}(\mathbf{A}) = \dim \mathrm{Col}(\mathbf{A}) = \mathrm{rank}(\mathbf{A})
\hspace{19em} (2)
```

$m \ne n$ 이면 행은 $\mathbb{R}^n$ 의 vector 이고 열은 $\mathbb{R}^m$ 의 vector 여서, 행과 열이 놓인 공간의 차원이 다르다. 식 (2) 는 그 공간의 차원이 아니라 그 안에서 서로 일차독립인 행의 최대 개수와 열의 최대 개수가 같다는 식이다.

$\mathbf{A}$ 를 기본 행 연산으로 기약 행사다리꼴로 바꾸어도 행의 일차결합만 만들어지므로 row space 는 바뀌지 않고, 열 사이의 일차독립 관계도 바뀌지 않는다. 기약 행사다리꼴에서 0 이 아닌 행은 각각 pivot 하나를 갖고 서로 일차독립이어서, 그 개수가 row space 의 차원이다. Pivot column 은 서로 일차독립이고 나머지 열은 pivot column 의 일차결합이어서, pivot column 의 개수가 column space 의 차원이다. 0 이 아닌 행의 개수와 pivot column 의 개수는 pivot 의 개수로 같으므로 식 (2) 가 성립한다. Row space 와 null space 의 관계는 [Row Space Primer](../Row-Space/row-space-primer-ko.md) 에 있다.

## 6. Worked Example

행이 두 개이고 열이 세 개인 행렬을 둔다 ($m = 2$, $n = 3$).

```math
\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 6 & 9 \end{bmatrix}
\hspace{19em} (3)
```

행은 $\mathbb{R}^3$ 의 vector 두 개이다.

```math
\mathbf{r}_1 = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \qquad
\mathbf{r}_2 = \begin{bmatrix} 4 \\ 6 \\ 9 \end{bmatrix}
\hspace{19em} (4)
```

$\mathbf{r}_2$ 를 $\mathbf{r}_1$ 의 상수배로 적으려면 첫 원소에서 $4 / 1 = 4$ 배, 둘째 원소에서 $6 / 2 = 3$ 배여서 배수가 하나로 정해지지 않는다. 두 행은 서로 일차독립이므로 row space 는 $\mathbb{R}^3$ 안의 평면이고 차원은 2 이다.

열은 $\mathbb{R}^2$ 의 vector 세 개이다.

```math
\mathbf{c}_1 = \begin{bmatrix} 1 \\ 4 \end{bmatrix}, \qquad
\mathbf{c}_2 = \begin{bmatrix} 2 \\ 6 \end{bmatrix}, \qquad
\mathbf{c}_3 = \begin{bmatrix} 3 \\ 9 \end{bmatrix}
\hspace{19em} (5)
```

$\mathbf{c}_1$ 과 $\mathbf{c}_2$ 를 열로 하는 $2 \times 2$ 행렬의 행렬식은 0 이 아니므로, 두 열은 서로 일차독립이다.

```math
\det \begin{bmatrix} 1 & 2 \\ 4 & 6 \end{bmatrix} = 1 \cdot 6 - 2 \cdot 4 = -2
\hspace{19em} (6)
```

$\mathbb{R}^2$ 에서 서로 일차독립인 vector 는 많아야 두 개이므로, $\mathbf{c}_3$ 는 $\mathbf{c}_1$ 과 $\mathbf{c}_2$ 의 일차결합 $x_1 \mathbf{c}_1 + x_2 \mathbf{c}_2$ 이다. 계수는 아래 두 식을 풀어 구한다.

```math
x_1 + 2 x_2 = 3
\hspace{19em} (7)
```

```math
4 x_1 + 6 x_2 = 9
\hspace{19em} (8)
```

식 (7) 에서 $x_1 = 3 - 2 x_2$ 이고, 이를 식 (8) 에 넣으면 $12 - 2 x_2 = 9$ 이어서 $x_2 = 1.5$, $x_1 = 0$ 이다.

```math
\mathbf{c}_3 = 0 \cdot \mathbf{c}_1 + 1.5 \cdot \mathbf{c}_2 = \begin{bmatrix} 0 + 3 \\ 0 + 9 \end{bmatrix} = \begin{bmatrix} 3 \\ 9 \end{bmatrix}
\hspace{19em} (9)
```

$\mathbf{c}_3$ 는 $\mathbf{c}_2$ 의 1.5 배여서 새 방향을 더하지 않으므로, 서로 일차독립인 열은 최대 두 개이다. 서로 일차독립인 $\mathbf{c}_1$, $\mathbf{c}_2$ 두 개가 차원이 2 인 $\mathbb{R}^2$ 의 기저가 되므로, column space 는 $\mathbb{R}^2$ 전체이며 차원은 2 이다. 행과 열을 견주면 아래 Table 3 과 같다.

Table 3. Row space and column space of the matrix in equation (3)

| Space        | Vectors | Lives in       | Linearly independent | Dimension |
| :----------: | :-----: | :------------: | :------------------: | :-------: |
| Row space    | 2       | $\mathbb{R}^3$ | 2                    | 2         |
| Column space | 3       | $\mathbb{R}^2$ | 2                    | 2         |

행과 열이 놓인 공간의 차원은 3 과 2 로 다르지만, row space 와 column space 의 차원은 모두 2 여서 식 (2) 대로 $\mathrm{rank}(\mathbf{A}) = 2$ 이다. Null space 의 차원은 $n - r = 3 - 2 = 1$ 이며, $\mathbf{A}\mathbf{x}$ 는 $\mathbf{A}$ 의 열을 $\mathbf{x}$ 의 원소로 일차결합한 값이고, 식 (9) 에서 $0 \cdot \mathbf{c}_1 + 1.5\,\mathbf{c}_2 - \mathbf{c}_3 = \mathbf{0}$ 이므로 $\mathbf{x} = [0, 1.5, -1]^\top$ 이 $\mathbf{A}\mathbf{x} = \mathbf{0}$ 을 만족한다.

## 7. Application

Rank 를 행의 수 $m$ 과 열의 수 $n$ 에 견주면 선형방정식 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 의 해가 어떻게 되는지가 정해진다.

### 7.1 Rank Conditions of a Linear System

- $r = n$: $\mathrm{Null}(\mathbf{A})$ 의 차원이 $n - r = 0$ 이어서 null space 에 zero vector 만 있고, 해가 있으면 하나뿐이다.
- $r = m$: $\mathrm{Col}(\mathbf{A})$ 의 차원이 $\mathbb{R}^m$ 의 차원과 같아 column space 가 $\mathbb{R}^m$ 전체이고, 모든 우변 $\mathbf{b}$ 에 대해 해가 있다.
- $r \lt n$: null space 의 차원이 1 이상이어서, 해가 있으면 무수히 많다. 식 (3) 의 $\mathbf{A}$ 는 $r = m = 2 \lt n = 3$ 이어서 모든 $\mathbf{b}$ 에 해가 있고 그 해는 무수히 많다.

해의 개수를 가리는 절차는 [Types of Linear Systems](../Linear-System-Types/linear-system-types-ko.md) 에, 해가 무수히 많을 때 norm 이 가장 작은 해는 [Minimum Norm Solution Primer](../Minimum-Norm-Solution/minimum-norm-solution-primer-ko.md) 에 있다.

---

## Appendix A. Terminology

- **column space**: 행렬의 열들의 일차결합 전체. $\mathbb{R}^m$ 의 부분공간이다.
- **free column**: 기약 행사다리꼴에서 pivot 이 없는 열. 그 열의 변수 (free variable) 는 아무 값이나 가질 수 있고, pivot column 의 변수 (pivot 변수) 는 free variable 의 값으로 정해진다.<br>
  예: Worked Example 의 $\mathbf{A}$ 를 기약 행사다리꼴로 바꾼 아래 행렬에서는 pivot 이 없는 셋째 열 $[0, 1.5]^\top$ 이 free column 이고, 그 열의 변수가 free variable 이다.
  ```math
  \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 1.5 \end{bmatrix}
  ```
- **left null space**: $\mathbf{A}^\top\mathbf{y} = \mathbf{0}$ 인 $\mathbf{y}$ 전체. $\mathbb{R}^m$ 의 부분공간이며 차원은 $m - r$ 이다.
- **null space**: $\mathbf{A}\mathbf{x} = \mathbf{0}$ 인 $\mathbf{x}$ 전체. $\mathbb{R}^n$ 의 부분공간이다.
- **pivot column**: 기약 행사다리꼴에서 각 행의 첫 0 이 아닌 원소 (pivot) 가 있는 열.<br>
  예: 같은 행렬에서는 첫 행의 pivot 1 이 있는 첫째 열과 둘째 행의 pivot 1 이 있는 둘째 열이 pivot column 이다.
  ```math
  \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 1.5 \end{bmatrix}
  ```
- **rank-nullity 정리 (rank-nullity theorem)**: $\mathbf{A}$ 의 rank 와 null space 의 차원을 더하면 열의 수 $n$ 이라는 정리.
- **row space**: 행렬의 행들의 일차결합 전체. $\mathbb{R}^n$ 의 부분공간이다.
- **vector space**: zero vector 를 포함하고, 원소끼리의 덧셈과 상수배에 대해 닫혀 있는 vector 집합.
- **계수 (rank)**: 행렬에서 서로 일차독립인 열의 최대 개수. Column space 의 차원과 같다.
- **기본 행 연산 (elementary row operation)**: 두 행을 맞바꾸기, 한 행에 0 이 아닌 수를 곱하기, 한 행에 다른 행의 상수배를 더하기의 세 연산.
- **기약 행사다리꼴 (reduced row echelon form)**: 기본 행 연산으로 얻는 행렬로, 각 행의 pivot 이 1 이고 pivot 이 있는 열의 다른 원소가 모두 0 인 꼴.
- **기저 (basis)**: Vector space $V$ 의 vector 가운데 서로 일차독립이면서, 일차결합으로 $V$ 의 모든 vector 를 만드는 vector 의 집합. $\mathbb{R}^3$ 에서는 $[1, 0, 0]^\top$, $[0, 1, 0]^\top$, $[0, 0, 1]^\top$ 이 그 하나이다. 공간이 여럿이면 공간마다 기저가 하나씩 있으며, 그 여럿을 복수형 bases 로 부른다. Table 4 에는 세 공간의 기저가 한 행에 하나씩 있다.
- **기저 vector (basis vector)**: 기저에 든 vector 하나하나. Table 4 에서 column space 의 기저 $\{\mathbf{c}_1, \mathbf{c}_2\}$ 에서는 $\mathbf{c}_1$ 과 $\mathbf{c}_2$ 가 각각 기저 vector 이며, 기저 vector 의 개수가 그 공간의 차원이다.
- **부분공간 (subspace)**: Vector space 의 부분집합이면서 그 자체로 vector space 인 집합.
- **일차결합 (linear combination)**: vector 들에 각각 상수를 곱해 더한 것.
- **일차독립 (linear independence)**: 어느 vector 도 나머지 vector 들의 일차결합으로 적히지 않는 것.
- **행렬식 (determinant)**: 정사각 행렬에 매기는 수. $2 \times 2$ 행렬에서는 대각선 두 원소의 곱에서 나머지 두 원소의 곱을 뺀 값이고, 0 이 아니면 두 열이 서로 일차독립이다.

## Appendix B. Finding a Basis

> **Definition: Basis**<br>
> 기저 (basis) 는 vector space $V$ 의 vector 가운데 서로 일차독립이면서, 일차결합으로 $V$ 의 모든 vector 를 만드는 vector 의 집합이다.

> **Definition: Basis vector**<br>
> 기저 vector (basis vector) 는 기저에 든 vector 하나하나이다.

기저가 있으면 $V$ 의 모든 vector 를 기저 vector 의 일차결합으로 적을 수 있고, 기저 vector 가 서로 일차독립이므로 그 계수는 하나로 정해진다. $\mathbb{R}^3$ 의 기저 $[1, 0, 0]^\top$, $[0, 1, 0]^\top$, $[0, 0, 1]^\top$ 에서 $[2, 3, 0]^\top$ 은 $2 \cdot [1, 0, 0]^\top + 3 \cdot [0, 1, 0]^\top + 0 \cdot [0, 0, 1]^\top$ 으로만 적힌다. 같은 $V$ 에 기저는 여럿이지만 기저 vector 의 개수는 모두 같으며, 그 개수가 $V$ 의 차원이다.

$\mathbf{A}$ 를 기본 행 연산으로 기약 행사다리꼴 (reduced row echelon form) 로 바꾸면 column space, row space, null space 의 기저를 모두 구할 수 있다. 기약 행사다리꼴은 각 행의 첫 0 이 아닌 원소 (pivot) 가 1 이고, pivot 이 있는 열의 다른 원소가 모두 0 이며, 아래 행의 pivot 이 위 행의 pivot 보다 오른쪽에 있는 꼴이다.

식 (3) 의 $\mathbf{A}$ 와 식 (4) 의 두 행은 아래이다.

```math
\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 6 & 9 \end{bmatrix}
\hspace{19em} (3)
```

```math
\mathbf{r}_1 = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}, \qquad
\mathbf{r}_2 = \begin{bmatrix} 4 \\ 6 \\ 9 \end{bmatrix}
\hspace{19em} (4)
```

$\mathbf{A}$ 의 둘째 행에서 첫째 행의 4 배를 빼면 $[0, -2, -3]$, 이를 $-2$ 로 나누면 $[0, 1, 1.5]$ 이고, 첫째 행에서 이 행의 2 배를 빼면 $[1, 0, 0]$ 이다.

```math
\begin{bmatrix} 1 & 2 & 3 \\ 4 & 6 & 9 \end{bmatrix}
\;\longrightarrow\;
\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 1.5 \end{bmatrix}
\hspace{19em} (10)
```

식 (10) 의 오른쪽 행렬에서 pivot column 은 첫째와 둘째 열이고, free column 은 셋째 열이다. 세 공간의 기저는 아래처럼 구한다.

- Column space: $\mathbf{A}$ 에서 pivot column 과 번호가 같은 열이 기저이다. 기본 행 연산은 열 사이의 일차독립 관계를 바꾸지 않으므로, 기약 행사다리꼴에서 서로 일차독립인 pivot column 과 번호가 같은 $\mathbf{A}$ 의 열도 서로 일차독립이고, 나머지 열은 그 열들의 일차결합이다. 다만 기본 행 연산은 열이 놓인 공간 $\mathbb{R}^m$ 의 vector 를 바꾸므로, 기약 행사다리꼴의 열이 아니라 원래 $\mathbf{A}$ 의 열을 쓴다. 식 (3) 에서는 첫째와 둘째 열 $\mathbf{c}_1 = [1, 4]^\top$ 과 $\mathbf{c}_2 = [2, 6]^\top$ 이다.
- Row space: 기약 행사다리꼴의 0 이 아닌 행이 기저이다. 기본 행 연산은 행의 일차결합만 만들어 row space 를 바꾸지 않는다. 식 (10) 에서는 $[1, 0, 0]^\top$ 과 $[0, 1, 1.5]^\top$ 이다.
- Null space: $\mathbf{A}\mathbf{x} = \mathbf{0}$ 에서 free variable 하나만 1, 나머지 free variable 은 0 으로 두고, 기약 행사다리꼴의 각 행에서 pivot 변수를 구한다. free variable 마다 기저 vector 하나를 얻는다. 기본 행 연산은 해를 바꾸지 않으므로 $\mathbf{A}\mathbf{x} = \mathbf{0}$ 대신 식 (10) 의 오른쪽 행렬을 $\mathbf{x}$ 에 곱한 값이 zero vector 가 되는 식을 푼다. 그 두 행은 "$\mathbf{x}$ 의 첫째 원소 $= 0$" 과 "둘째 원소 $+ 1.5 \times$ 셋째 원소 $= 0$" 이므로, 셋째 원소를 1 로 두면 첫째 원소가 0, 둘째 원소가 $-1.5$ 여서 $[0, -1.5, 1]^\top$ 이다.

행렬 $\mathbf{A}$ 하나에는 부분공간이 네 개 있다. Column space, row space, null space 에 left null space 가 더해지고, 네 공간마다 기저가 있다. 네 부분공간의 기저를 모으면 아래 Table 4 와 같다.

Table 4. Bases of the four subspaces of the matrix in equation (3)

| Space           | Basis                                                      | Dimension |
| :-------------: | :--------------------------------------------------------: | :-------: |
| Column space    | $\mathbf{c}_1 = [1, 4]^\top$, $\mathbf{c}_2 = [2, 6]^\top$ | 2         |
| Row space       | $[1, 0, 0]^\top$, $[0, 1, 1.5]^\top$                       | 2         |
| Null space      | $[0, -1.5, 1]^\top$                                        | 1         |
| Left null space | None                                                       | 0         |

Table 4 에서 left null space 의 기저가 None 인 것은 식 (3) 의 $\mathbf{A}$ 가 $m = 2$, $r = 2$ 이기 때문이다. 이때 left null space 의 차원이 $m - r = 0$ 이어서 zero vector 만 있고, 기저 vector 가 하나도 없다. Left null space 는 $\mathbf{A}^\top\mathbf{y} = \mathbf{0}$ 인 $\mathbf{y}$ 전체이며, 그 정의와 성질은 [Left Null Space Primer](../Left-Null-Space/left-null-space-primer-ko.md) 에 있다.

Row space 의 기저 $[1, 0, 0]^\top$ 과 $[0, 1, 1.5]^\top$ 은 식 (4) 의 $\mathbf{r}_1$, $\mathbf{r}_2$ 와 다르지만, $[1, 0, 0]^\top = -3\,\mathbf{r}_1 + \mathbf{r}_2$ 와 $[0, 1, 1.5]^\top = 2\,\mathbf{r}_1 - 0.5\,\mathbf{r}_2$ 로 같은 평면을 생성하고 개수도 2 로 같다. Null space 의 기저 $[0, -1.5, 1]^\top$ 은 [꼭지 6](#6-worked-example) 의 $[0, 1.5, -1]^\top$ 에 $-1$ 을 곱한 것이다. 기약 행사다리꼴, pivot column, free variable 로 해를 구하는 순서는 [Null Space Primer](../Null-Space/null-space-primer-ko.md) 에도 있다.

