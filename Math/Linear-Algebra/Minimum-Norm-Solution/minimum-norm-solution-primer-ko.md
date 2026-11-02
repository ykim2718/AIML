# Minimum Norm Solution Primer
Rev. 9 | Created: 2026-11-02 | Updated: 2026-11-02 21:46 UTC

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Notation](#3-notation)
- [4. Interpretation](#4-interpretation)
  - [4.1 Solution Set](#41-solution-set)
  - [4.2 Orthogonal Decomposition of a Solution](#42-orthogonal-decomposition-of-a-solution)
  - [4.3 Closed Form](#43-closed-form)
- [5. Worked Example](#5-worked-example)
- [6. Application](#6-application)
  - [6.1 Pseudo-Inverse](#61-pseudo-inverse)
  - [6.2 Breaking Conditions](#62-breaking-conditions)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Derivation](#appendix-b-derivation)

## 1. Purpose

- **Problem Statement**: $\mathbf{A}\mathbf{x} = \mathbf{b}$ 의 해가 무수히 많으면, 그 가운데 어느 해를 고를지 정하는 기준이 필요하다.
- **Goal**: 독자가 해 가운데 Euclidean norm 이 가장 작은 해를 식으로 구하고, 그 해가 row space 에 있다는 것을 숫자로 확인할 수 있게 한다.
- **Non-Goal**: 해가 없는 경우의 최소제곱 해와 복소 행렬은 다루지 않는다.

## 2. Summary

최소 노름 해 (minimum norm solution) 는 해가 무수히 많은 선형방정식 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 의 해 가운데 Euclidean norm $\lVert \mathbf{x} \rVert$ 가 가장 작은 하나의 해이다.
이 해는 null space 성분이 zero vector 이고 row space 에만 있는 유일한 해이다.
$\mathbf{A}$ 의 rank 가 행의 수 $m$ 과 같으면 $\mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top)^{-1}\mathbf{b}$ 로 구한다.

## 3. Notation

이 문서의 vector 는 모두 열 vector 이다. 이 문서가 쓰는 기호는 아래 Table 1 이다.

Table 1. Symbols

| Symbol                                                     | Meaning                                                                                            | Shape                      | Assumption                                               |
| :--------------------------------------------------------: | :------------------------------------------------------------------------------------------------: | :------------------------: | :------------------------------------------------------: |
| $\mathbb{R}$                                               | 실수 전체의 집합. $\mathbb{R}^n$ 은 원소가 실수 $n$ 개인 열 vector 전체이다                        | 집합                       | 실수                                                     |
| $\mathbf{A}$                                               | 선형방정식의 계수 행렬                                                                             | $m \times n$               | 실수 원소                                                |
| $m$, $n$                                                   | $\mathbf{A}$ 의 행의 수 (식의 수) 와 열의 수 (미지수의 수)                                         | scalar                     | 양의 정수                                                |
| $\mathbf{x}$                                               | 미지수 vector, $\mathbf{A}\mathbf{x} = \mathbf{b}$ 의 해                                           | $n \times 1$               | 실수 원소                                                |
| $\lVert \mathbf{x} \rVert$                                 | $\mathbf{x}$ 의 norm (Euclidean norm), $\lVert \mathbf{x} \rVert = (x_1^2 + \cdots + x_n^2)^{1/2}$ | scalar                     | 0 이상                                                   |
| $\mathbf{b}$                                               | 선형방정식의 우변                                                                                  | $m \times 1$               | $\mathrm{Col}(\mathbf{A})$ 에 속함                       |
| $\mathbf{y}$                                               | $\mathbf{A}^\top$ 에 곱하는 vector                                                                 | $m \times 1$               | 실수 원소                                                |
| $\mathbf{0}$                                               | zero vector                                                                                        | $n \times 1$               | 모든 원소가 0                                            |
| $r$                                                        | $\mathbf{A}$ 의 계수 (rank)                                                                        | scalar                     | $r \le \min(m, n)$                                       |
| $\mathrm{Row}(\mathbf{A})$                                 | $\mathbf{A}$ 의 row space                                                                          | $\mathbb{R}^n$ 의 부분공간 | $\mathbf{A}^\top\mathbf{y}$ 전체                         |
| $\mathrm{Null}(\mathbf{A})$                                | $\mathbf{A}$ 의 null space                                                                         | $\mathbb{R}^n$ 의 부분공간 | $\mathbf{A}\mathbf{x} = \mathbf{0}$ 인 $\mathbf{x}$ 전체 |
| $\mathrm{Col}(\mathbf{A})$                                 | $\mathbf{A}$ 의 column space                                                                       | $\mathbb{R}^m$ 의 부분공간 | $\mathbf{A}\mathbf{x}$ 전체                              |
| $\mathbf{x}_{\mathrm{row}}$                                | 해 $\mathbf{x}$ 의 row space 성분                                                                  | $n \times 1$               | $\mathrm{Row}(\mathbf{A})$ 에 속함                       |
| $\mathbf{x}_{\mathrm{null}}$                               | 해 $\mathbf{x}$ 의 null space 성분                                                                 | $n \times 1$               | $\mathrm{Null}(\mathbf{A})$ 에 속함                      |
| $\mathbf{x}^{\ast}$                                        | 최소 노름 해                                                                                       | $n \times 1$               | $\mathbf{x}_{\mathrm{null}} = \mathbf{0}$                |
| $\mathbf{x}^{(1)}$, $\mathbf{x}^{(2)}$, $\mathbf{x}^{(3)}$ | Worked Example 의 해 세 개                                                                         | $3 \times 1$               | $\mathbf{A}\mathbf{x} = \mathbf{b}$ 를 만족              |

특수 기호는 아래처럼 읽는다.

- $\mathbf{A}^\top$: 영어로 "A transpose", 한국어로 "A 의 전치" 라 읽는다.
- $(\mathbf{A}\mathbf{A}^\top)^{-1}$: 영어로 "the inverse of A A transpose", 한국어로 "A A 전치의 역행렬" 이라 읽는다.
- $\mathbf{A}^{+}$: 영어로 "A plus" 또는 "the pseudo-inverse of A", 한국어로 "A 의 pseudo-inverse" 라 읽는다.
- $\lVert \mathbf{x} \rVert$: 영어로 "the norm of x", 한국어로 "x 의 노름" 이라 읽는다.
- $\mathbf{x}^{\ast}$: 영어로 "x star", 한국어로 "엑스 스타" 라 읽는다.
- $\mathbf{x}^{(1)}$: 영어로 "x superscript one", 한국어로 "x 위첨자 1" 이라 읽는다. 괄호 안의 수는 거듭제곱이 아니라 해의 번호이다.
- $\mathrm{arg\,min}$: 영어로 "argument of the minimum", 한국어로 "최소가 되게 하는 인자" 라 읽는다.
- $\in$, $\mid$: 영어로 "in" 과 "such that", 한국어로 "에 속하는" 과 "인" 이라 읽는다.
- $\perp$: 영어로 "is orthogonal to", 한국어로 "직교한다" 라 읽는다.

## 4. Interpretation

최소 노름 해는 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 를 만족하는 $\mathbf{x}$ 가운데 $\lVert \mathbf{x} \rVert$ 가 가장 작은 해이며, rank 가 $m$ 이면 아래 식으로 구한다.

```math
\mathbf{x}^{\ast}
= \mathrm{arg\,min}_{\mathbf{x} \,\mid\, \mathbf{A}\mathbf{x} = \mathbf{b}} \lVert \mathbf{x} \rVert
= \mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top)^{-1}\mathbf{b}
\hspace{19em} (1)
```

### 4.1 Solution Set

미지수가 식보다 많은 부정 방정식 (underdetermined system) 처럼 $\mathbf{b}$ 가 column space 에 속하고 null space 에 zero vector 가 아닌 vector 가 있으면, $\mathbf{A}\mathbf{x} = \mathbf{b}$ 의 해는 무수히 많다. 해 하나에 null space 의 vector 를 더해도 $\mathbf{A}$ 를 곱한 값은 그대로 $\mathbf{b}$ 이다. 해의 크기를 Euclidean norm 으로 재고, 그 값이 가장 작은 해를 고르는 것이 최소 노름 해 문제이다.

### 4.2 Orthogonal Decomposition of a Solution

입력 공간 $\mathbb{R}^n$ 은 row space 와 null space 로 나뉘고 두 공간은 서로 직교하므로, 해 $\mathbf{x}$ 는 row space 성분과 null space 성분의 합으로 하나씩만 나뉜다. 두 성분이 직교하므로 피타고라스 정리에 따라 norm 의 제곱은 두 성분의 norm 의 제곱을 더한 값이다.

```math
\mathbf{x} = \mathbf{x}_{\mathrm{row}} + \mathbf{x}_{\mathrm{null}}, \qquad
\lVert \mathbf{x} \rVert^2 = \lVert \mathbf{x}_{\mathrm{row}} \rVert^2 + \lVert \mathbf{x}_{\mathrm{null}} \rVert^2
\hspace{19em} (2)
```

두 해의 차이는 null space 에 속하므로, 모든 해는 같은 row space 성분을 갖고 null space 성분만 다르다. 따라서 식 (2) 의 norm 은 null space 성분이 zero vector 일 때 가장 작고, 최소 노름 해는 row space 에만 있는 유일한 해이다. Row space 와 null space 의 직교는 [Row Space Primer](../Row-Space/row-space-primer-ko.md) 에, 해의 구조는 [Null Space Primer](../Null-Space/null-space-primer-ko.md) 에 있다.

### 4.3 Closed Form

식 (1) 의 오른쪽 변은 rank 가 $m$ 인 full row rank 일 때만 성립한다. 이때 $m \times m$ 행렬 $\mathbf{A}\mathbf{A}^\top$ 의 역행렬이 있다. $\mathbf{A}^\top(\cdot)$ 꼴이므로 이 해는 row space 에 속하고, $\mathbf{A}$ 를 곱하면 $\mathbf{A}\mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top)^{-1}\mathbf{b} = \mathbf{b}$ 이다. 유도는 [Appendix B](#appendix-b-derivation) 에 있다.

## 5. Worked Example

미지수가 3 개이고 식이 1 개인 부정 방정식 $x_1 + x_2 + x_3 = 3$ 을 행렬로 적으면 아래이다.

```math
\mathbf{A} = \begin{bmatrix} 1 & 1 & 1 \end{bmatrix}, \qquad
\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}, \qquad
\mathbf{b} = \begin{bmatrix} 3 \end{bmatrix}
\hspace{19em} (3)
```

원소의 합이 3 인 vector 는 모두 해이므로 해는 무수히 많다. 그 가운데 세 개는 아래이다.

```math
\mathbf{x}^{(1)} = \begin{bmatrix} 3 \\ 0 \\ 0 \end{bmatrix}, \qquad
\mathbf{x}^{(2)} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}, \qquad
\mathbf{x}^{(3)} = \begin{bmatrix} 5 \\ -1 \\ -1 \end{bmatrix}
\hspace{19em} (4)
```

식 (1) 에 식 (3) 을 넣으면 아래이다.

```math
\begin{aligned}
\mathbf{A}\mathbf{A}^\top &= \begin{bmatrix} 1 & 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \cdot 1 + 1 \cdot 1 + 1 \cdot 1 \end{bmatrix} = \begin{bmatrix} 3 \end{bmatrix} \\
(\mathbf{A}\mathbf{A}^\top)^{-1} &= \begin{bmatrix} 1/3 \end{bmatrix} \\
\mathbf{x}^{\ast} &= \mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top)^{-1}\mathbf{b} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} \begin{bmatrix} 1/3 \end{bmatrix} \begin{bmatrix} 3 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}
\end{aligned}
\hspace{19em} (5)
```

최소 노름 해는 $\mathbf{x}^{(2)}$ 이다. 세 해를 식 (2) 로 나누면 아래 Table 2 이다.

Table 2. Norm of each solution

| Solution           | $\mathbf{x}$  | $\lVert \mathbf{x} \rVert^2$ | $\mathbf{x}_{\mathrm{null}}$ | $\lVert \mathbf{x}_{\mathrm{null}} \rVert^2$ |
| :----------------: | :-----------: | :--------------------------: | :--------------------------: | :------------------------------------------: |
| $\mathbf{x}^{(1)}$ | $[3, 0, 0]$   | $3^2 + 0^2 + 0^2 = 9$        | $[2, -1, -1]$                | 6                                            |
| $\mathbf{x}^{(2)}$ | $[1, 1, 1]$   | $1^2 + 1^2 + 1^2 = 3$        | $[0, 0, 0]$                  | 0                                            |
| $\mathbf{x}^{(3)}$ | $[5, -1, -1]$ | $5^2 + (-1)^2 + (-1)^2 = 27$ | $[4, -2, -2]$                | 24                                           |

세 해의 row space 성분은 모두 $[1, 1, 1]^\top$ 이고 그 norm 의 제곱은 3 이다. $\lVert \mathbf{x} \rVert^2$ 는 3 에 null space 성분의 norm 의 제곱을 더한 값이어서, null space 성분이 zero vector 인 $\mathbf{x}^{(2)}$ 의 값 3 이 가장 작다.

$\mathbf{A}$ 의 row space 는 $[1, 1, 1]^\top$ 의 상수배 전체인 직선이고, $\mathbf{x}^{(2)}$ 는 이 직선 위에 있다. Null space 는 원소의 합이 0 인 vector 전체인 평면이며 $[1, -1, 0]^\top$ 이 그 예이다. Table 2 의 null space 성분 $[2, -1, -1]^\top$ 과 $[4, -2, -2]^\top$ 은 원소의 합이 0 이고 $[1, 1, 1]^\top$ 과의 내적이 0 이다.

## 6. Application

최소 노름 해는 해가 하나로 정해지지 않는 선형 계산에서 해를 하나로 고르는 기준이다.

### 6.1 Pseudo-Inverse

Pseudo-inverse $\mathbf{A}^{+}$ 로 구한 $\mathbf{A}^{+}\mathbf{b}$ 가 최소 노름 해이고, full row rank 이면 $\mathbf{A}^{+} = \mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top)^{-1}$ 이다. 식 (3) 에서 $\mathbf{A}^{+} = [1/3, 1/3, 1/3]^\top$ 이어서 $\mathbf{A}^{+}\mathbf{b} = [1, 1, 1]^\top$ 로 식 (5) 와 같다. 선형 회귀에서 sample 수보다 feature 수가 많으면 weight 가 하나로 정해지지 않고, pseudo-inverse 는 그 가운데 norm 이 가장 작은 weight 를 고른다. 식이 미지수보다 많은 경우의 pseudo-inverse 와 숫자 예시는 [Pseudo-Inverse Primer](../Pseudo-Inverse/pseudo-inverse-primer-ko.md) 에 있다.

### 6.2 Breaking Conditions

Rank 가 $m$ 보다 작으면 $\mathbf{A}\mathbf{A}^\top$ 의 역행렬이 없어 식 (1) 의 오른쪽 변을 쓸 수 없다. 이때도 $\mathbf{b}$ 가 column space 에 속하면 최소 노름 해는 있으며, 특이값 분해로 구한 $\mathbf{A}^{+}\mathbf{b}$ 가 그 해이다. $\mathbf{b}$ 가 column space 에 속하지 않으면 해가 없으므로, 최소제곱 해 가운데 norm 이 가장 작은 해를 대신 쓴다. 이 경우는 [Left Null Space Primer](../Left-Null-Space/left-null-space-primer-ko.md) 에 있다.

---

## Appendix A. Terminology

- **column space**: $\mathbf{x} \in \mathbb{R}^n$ 인 모든 $\mathbf{x}$ 에 대한 $\mathbf{A}\mathbf{x}$ 의 집합. $\mathbf{A}$ 의 열들의 일차결합 전체이다.
- **full row rank**: 행렬의 rank 가 행의 수 $m$ 과 같은 것.
- **norm (Euclidean norm)**: vector 원소의 제곱을 모두 더한 값의 제곱근. Vector 의 길이이며, $\lVert \mathbf{x} \rVert = (x_1^2 + \cdots + x_n^2)^{1/2}$ 로 적는다. L1, L∞ 같은 다른 vector norm 과 matrix norm 은 [Types of Norms](../Norm-Types/norm-types-ko.md) 에 있다.
- **null space**: $\mathbf{A}\mathbf{x} = \mathbf{0}$ 인 $\mathbf{x}$ 의 집합.
- **pseudo-inverse**: 정사각이 아니거나 역행렬이 없는 행렬에도 정의되며, 해 가운데 norm 이 가장 작은 해를 주는 역행렬의 일반화.
- **row space**: $\mathbf{y} \in \mathbb{R}^m$ 인 모든 $\mathbf{y}$ 에 대한 $\mathbf{A}^\top\mathbf{y}$ 의 집합. $\mathbf{A}$ 의 행들의 일차결합 전체이다.
- **계수 (rank)**: 행렬에서 서로 일차독립인 열의 최대 개수.
- **부분공간 (subspace)**: Vector space 의 부분집합이면서 그 자체로 vector space 인 집합.
- **부정 방정식 (underdetermined system)**: 미지수의 수가 식의 수보다 많은 선형방정식.
- **직교 (orthogonal)**: 두 vector 의 내적이 0 인 것.
- **최소제곱 해 (least squares solution)**: $\lVert \mathbf{A}\mathbf{x} - \mathbf{b} \rVert$ 를 가장 작게 하는 $\mathbf{x}$.
- **특이값 분해 (singular value decomposition)**: 행렬을 $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ 로 나누는 분해. $\mathbf{U}$ 와 $\mathbf{V}$ 의 열은 단위 vector 이고 $\boldsymbol{\Sigma}$ 의 대각 원소가 특이값이다.
- **피타고라스 정리 (Pythagorean theorem)**: 직교하는 두 vector 의 합의 norm 의 제곱은 두 vector 의 norm 의 제곱을 더한 값이다.

## Appendix B. Derivation

최소 노름 해는 row space 에 속하므로 어떤 $\mathbf{y} \in \mathbb{R}^m$ 에 대해 $\mathbf{x}^{\ast} = \mathbf{A}^\top\mathbf{y}$ 이다. 이를 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 에 넣어 $\mathbf{y}$ 를 구한다.

```math
\begin{aligned}
\mathbf{A}\mathbf{x}^{\ast} &= \mathbf{b} \\
\mathbf{A}\mathbf{A}^\top\mathbf{y} &= \mathbf{b} \\
\mathbf{y} &= (\mathbf{A}\mathbf{A}^\top)^{-1}\mathbf{b} \\
\mathbf{x}^{\ast} &= \mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top)^{-1}\mathbf{b}
\end{aligned}
\hspace{19em} (6)
```

- 첫째 줄: $\mathbf{x}^{\ast}$ 는 해이다.
- 둘째 줄: $\mathbf{x}^{\ast} = \mathbf{A}^\top\mathbf{y}$ 를 넣는다.
- 셋째 줄: full row rank 이면 $\mathbf{A}\mathbf{A}^\top$ 의 역행렬이 있으므로 양변의 왼쪽에 곱한다.
- 넷째 줄: 셋째 줄의 $\mathbf{y}$ 를 $\mathbf{x}^{\ast} = \mathbf{A}^\top\mathbf{y}$ 에 넣는다.

따라서 식 (1) 이 성립한다.

```math
\mathbf{x}^{\ast}
= \mathrm{arg\,min}_{\mathbf{x} \,\mid\, \mathbf{A}\mathbf{x} = \mathbf{b}} \lVert \mathbf{x} \rVert
= \mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top)^{-1}\mathbf{b}
\hspace{19em} (1)
```

Row space 에 속하는 해가 하나뿐인 것은 식 (2) 에서 나온다. Row space 에 속하는 두 해의 차이는 row space 에도, null space 에도 속하므로 자기 자신과 직교하고, 자기 자신과의 내적이 0 인 vector 는 zero vector 이다.
