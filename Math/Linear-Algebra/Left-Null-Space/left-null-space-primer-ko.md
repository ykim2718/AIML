# Left Null Space Primer
Rev. 0 | Created: 2026-11-01 | Updated: 2026-11-01 23:37 UTC

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Notation](#3-notation)
- [4. Interpretation](#4-interpretation)
  - [4.1 Transpose Form](#41-transpose-form)
  - [4.2 Orthogonality to the Column Space](#42-orthogonality-to-the-column-space)
  - [4.3 Orthogonal Decomposition of the Output Space](#43-orthogonal-decomposition-of-the-output-space)
  - [4.4 Four Fundamental Subspaces](#44-four-fundamental-subspaces)
- [5. Worked Example](#5-worked-example)
- [6. Application](#6-application)
  - [6.1 Least Squares Residual](#61-least-squares-residual)
  - [6.2 Solvability of Ax = b](#62-solvability-of-ax--b)
  - [6.3 Breaking Conditions](#63-breaking-conditions)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Derivation](#appendix-b-derivation)

## 1. Purpose

- **Problem Statement**: $\mathbf{A}\mathbf{x} = \mathbf{b}$ 에 해가 없을 때, $\mathbf{b}$ 의 어느 성분이 $\mathbf{A}$ 의 출력으로 만들어질 수 없는지는 column space 만으로 드러나지 않는다.
- **Goal**: 독자가 left null space 를 구하고, $\mathbf{b}$ 를 column space 성분과 left null space 성분으로 나누어 해가 있는지와 최소제곱 잔차를 가려낼 수 있게 한다.
- **Non-Goal**: 최소제곱 해를 구하는 algorithm 과 복소 행렬은 다루지 않는다.

## 2. Summary

Left null space 는 $m \times n$ 행렬 $\mathbf{A}$ 의 전치 $\mathbf{A}^\top$ 의 null space 이고, $\mathbf{y}^\top\mathbf{A} = \mathbf{0}^\top$ 인 $\mathbf{y}$ 의 집합이며 출력 공간 $\mathbb{R}^m$ 의 부분공간이다.
Left null space 는 column space 와 직교하고, 두 공간의 직합이 $\mathbb{R}^m$ 전체이다.
$\mathbf{A}\mathbf{x} = \mathbf{b}$ 의 해가 없으면 $\mathbf{b}$ 의 left null space 성분이 0 이 아니고, 그 성분이 최소제곱 잔차이다.

## 3. Notation

이 문서의 vector 는 모두 열 vector 이다. 이 문서가 쓰는 기호는 아래 Table 1 이다.

Table 1. Symbols

| Symbol                           | Meaning                                                                     | Shape                          | Assumption                                                    |
| :------------------------------: | :-------------------------------------------------------------------------: | :----------------------------: | :-----------------------------------------------------------: |
| $\mathbb{R}$                     | 실수 전체의 집합. $\mathbb{R}^m$ 은 원소가 실수 $m$ 개인 열 vector 전체이다 | 집합                           | 실수                                                          |
| $\mathbf{A}$                     | 선형 변환을 나타내는 행렬                                                   | $m \times n$                   | 실수 원소                                                     |
| $m$, $n$                         | $\mathbf{A}$ 의 행의 수와 열의 수                                           | scalar                         | 양의 정수                                                     |
| $\mathbf{x}$                     | 입력 vector                                                                 | $n \times 1$                   | 실수 원소                                                     |
| $\mathbf{y}$                     | 출력 공간의 vector, $\mathbf{A}^\top$ 에 곱하는 vector                      | $m \times 1$                   | 실수 원소                                                     |
| $\mathbf{b}$                     | 선형방정식 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 의 우변                      | $m \times 1$                   | 실수 원소                                                     |
| $\mathbf{0}$                     | zero vector                                                                 | $m \times 1$ 또는 $n \times 1$ | 모든 원소가 0                                                 |
| $\mathrm{Col}(\mathbf{A})$       | $\mathbf{A}$ 의 column space                                                | $\mathbb{R}^m$ 의 부분공간     | $\mathbf{A}\mathbf{x}$ 전체                                   |
| $\mathrm{Row}(\mathbf{A})$       | $\mathbf{A}$ 의 row space                                                   | $\mathbb{R}^n$ 의 부분공간     | $\mathbf{A}^\top\mathbf{y}$ 전체                              |
| $\mathrm{Null}(\mathbf{A})$      | $\mathbf{A}$ 의 null space                                                  | $\mathbb{R}^n$ 의 부분공간     | $\mathbf{A}\mathbf{x} = \mathbf{0}$ 인 $\mathbf{x}$ 전체      |
| $\mathrm{Null}(\mathbf{A}^\top)$ | $\mathbf{A}$ 의 left null space                                             | $\mathbb{R}^m$ 의 부분공간     | $\mathbf{A}^\top\mathbf{y} = \mathbf{0}$ 인 $\mathbf{y}$ 전체 |
| $r$                              | $\mathbf{A}$ 의 계수 (rank)                                                 | scalar                         | $r \le \min(m, n)$                                            |
| $\mathbf{b}_{\mathrm{col}}$      | $\mathbf{b}$ 의 column space 성분                                           | $m \times 1$                   | $\mathrm{Col}(\mathbf{A})$ 에 속함                            |
| $\mathbf{b}_{\mathrm{left}}$     | $\mathbf{b}$ 의 left null space 성분                                        | $m \times 1$                   | $\mathrm{Null}(\mathbf{A}^\top)$ 에 속함                      |
| $\hat{\mathbf{x}}$               | $\mathbf{A}\mathbf{x} = \mathbf{b}$ 의 최소제곱 해                          | $n \times 1$                   | $\mathbf{A}\hat{\mathbf{x}} = \mathbf{b}_{\mathrm{col}}$      |
| $\mathbf{n}$                     | Worked Example 에서 left null space 를 생성하는 vector                      | $3 \times 1$                   | zero vector 가 아님                                           |

특수 기호는 아래처럼 읽는다.

- $\mathbf{A}^\top$, $\mathbf{y}^\top$: 영어로 "A transpose", "y transpose", 한국어로 "A 의 전치", "y 의 전치" 라 읽는다.
- $\{ \mathbf{y} \in \mathbb{R}^m \mid \mathbf{A}^\top\mathbf{y} = \mathbf{0} \}$ 의 $\in$ 와 $\mid$: 영어로 "in" 과 "such that", 한국어로 "에 속하는" 과 "인" 이라 읽는다.
- $\perp$: 영어로 "is orthogonal to", 한국어로 "직교한다" 라 읽는다.
- $\oplus$: 영어로 "direct sum", 한국어로 "직합" 이라 읽는다.
- $\hat{\mathbf{x}}$: 영어로 "x hat", 한국어로 "엑스 햇" 이라 읽는다. 기호 위의 hat 은 그 값이 최소제곱으로 구한 해임을 나타낸다.
- $\lVert \mathbf{v} \rVert$: 영어로 "the norm of v", 한국어로 "v 의 노름" 이라 읽는다.

## 4. Interpretation

Left null space 는 $\mathbf{A}^\top$ 의 null space 이며, 같은 조건을 $\mathbf{y}^\top$ 이 $\mathbf{A}$ 의 왼쪽에 곱해지는 꼴로도 적는다.

```math
\mathrm{Null}(\mathbf{A}^\top)
= \{ \mathbf{y} \in \mathbb{R}^m \mid \mathbf{A}^\top\mathbf{y} = \mathbf{0} \}
= \{ \mathbf{y} \in \mathbb{R}^m \mid \mathbf{y}^\top\mathbf{A} = \mathbf{0}^\top \}
\hspace{19em} (1)
```

### 4.1 Transpose Form

$\mathbf{A}^\top\mathbf{y} = \mathbf{0}$ 의 양변을 전치하면 $\mathbf{y}^\top\mathbf{A} = \mathbf{0}^\top$ 이다. $\mathbf{y}^\top$ 이 $\mathbf{A}$ 의 왼쪽에 곱해져 zero vector 를 만들기 때문에 left null space 라 부른다. $\mathbf{A}^\top$ 는 열이 $m$ 개이므로 $\mathbf{y}$ 는 $\mathbb{R}^m$ 의 vector 이고, left null space 는 출력 공간 $\mathbb{R}^m$ 의 부분공간이다. Null space $\mathrm{Null}(\mathbf{A})$ 가 입력 공간 $\mathbb{R}^n$ 의 부분공간인 것과 대비된다.

### 4.2 Orthogonality to the Column Space

Left null space 의 모든 vector 는 column space 의 모든 vector 와 직교한다. $\mathbf{y}^\top\mathbf{A} = \mathbf{0}^\top$ 은 $\mathbf{y}$ 와 $\mathbf{A}$ 의 각 열의 내적이 0 이라는 뜻이고, column space 의 vector 는 열들의 일차결합이기 때문이다. 이를 $\mathrm{Null}(\mathbf{A}^\top) \perp \mathrm{Col}(\mathbf{A})$ 로 적으며, 유도는 [Appendix B](#appendix-b-derivation) 에 있다.

### 4.3 Orthogonal Decomposition of the Output Space

출력 공간 $\mathbb{R}^m$ 은 column space 와 left null space 의 직합이다. 모든 $\mathbf{b}$ 는 두 성분의 합으로 하나씩만 나뉜다.

```math
\mathbb{R}^m = \mathrm{Col}(\mathbf{A}) \oplus \mathrm{Null}(\mathbf{A}^\top), \qquad
\mathbf{b} = \mathbf{b}_{\mathrm{col}} + \mathbf{b}_{\mathrm{left}}
\hspace{19em} (2)
```

두 부분공간의 차원은 $r$ 과 $m - r$ 이어서 더하면 $m$ 이다. Column space 성분은 $\mathbf{A}\mathbf{x}$ 로 만들 수 있는 성분이고, left null space 성분은 어떤 $\mathbf{x}$ 로도 만들 수 없는 성분이다.

### 4.4 Four Fundamental Subspaces

행렬 $\mathbf{A}$ 의 네 부분공간은 아래 Table 2 이다.

Table 2. Four fundamental subspaces of A

| Subspace        | Symbol                           | Ambient space  | Dimension | Orthogonal to                    |
| :-------------: | :------------------------------: | :------------: | :-------: | :------------------------------: |
| Column space    | $\mathrm{Col}(\mathbf{A})$       | $\mathbb{R}^m$ | $r$       | $\mathrm{Null}(\mathbf{A}^\top)$ |
| Left null space | $\mathrm{Null}(\mathbf{A}^\top)$ | $\mathbb{R}^m$ | $m - r$   | $\mathrm{Col}(\mathbf{A})$       |
| Row space       | $\mathrm{Row}(\mathbf{A})$       | $\mathbb{R}^n$ | $r$       | $\mathrm{Null}(\mathbf{A})$      |
| Null space      | $\mathrm{Null}(\mathbf{A})$      | $\mathbb{R}^n$ | $n - r$   | $\mathrm{Row}(\mathbf{A})$       |

출력 공간 $\mathbb{R}^m$ 은 column space 와 left null space 로, 입력 공간 $\mathbb{R}^n$ 은 row space 와 null space 로 나뉜다. Null space 는 row space 와 직교하고 $\mathbf{A}$ 가 zero vector 로 보내는 입력 vector 의 집합이다. Left null space 는 column space 와 직교하고 $\mathbf{A}$ 의 출력으로 만들 수 없는 방향의 집합이다. Row space 와 null space 는 [Row Space Primer](../Row-Space/row-space-primer-ko.md) 와 [Null Space Primer](../Null-Space/null-space-primer-ko.md) 에 있다.

## 5. Worked Example

행이 세 개이고 열이 두 개인 행렬을 둔다.

```math
\mathbf{A} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \qquad
\mathbf{n} = \begin{bmatrix} -1 \\ -1 \\ 1 \end{bmatrix}, \qquad
\mathbf{A}^\top\mathbf{n} = \begin{bmatrix} -1 + 0 + 1 \\ 0 - 1 + 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
\hspace{19em} (3)
```

$\mathbf{A}$ 의 두 열은 일차독립이어서 rank 는 2 이고, column space 는 $\mathbb{R}^3$ 안의 평면이다. Left null space 는 $m - r = 1$ 차원이며 $\mathbf{n}$ 의 상수배 전체인 직선이다. $\mathbf{n}$ 은 두 열 $[1, 0, 1]^\top$, $[0, 1, 1]^\top$ 과 모두 내적이 0 이다.

$\mathbf{b} = [1, 0, 0]^\top$ 을 두 성분으로 나누면 아래이다.

```math
\mathbf{b}_{\mathrm{left}} = \frac{\mathbf{n}^\top\mathbf{b}}{\mathbf{n}^\top\mathbf{n}}\, \mathbf{n} = \frac{-1}{3} \begin{bmatrix} -1 \\ -1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1/3 \\ 1/3 \\ -1/3 \end{bmatrix}, \qquad
\mathbf{b}_{\mathrm{col}} = \mathbf{b} - \mathbf{b}_{\mathrm{left}} = \begin{bmatrix} 2/3 \\ -1/3 \\ 1/3 \end{bmatrix}
\hspace{19em} (4)
```

Left null space 성분이 zero vector 가 아니므로 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 의 해는 없다. Column space 성분은 $\hat{\mathbf{x}} = [2/3, -1/3]^\top$ 으로 $\mathbf{A}\hat{\mathbf{x}} = [2/3, -1/3, 1/3]^\top$ 이 되어 column space 에 속한다.

## 6. Application

Left null space 는 해가 없는 선형방정식의 잔차와 해의 존재 조건을 정한다.

### 6.1 Least Squares Residual

$\lVert \mathbf{A}\mathbf{x} - \mathbf{b} \rVert$ 를 가장 작게 하는 최소제곱 해 $\hat{\mathbf{x}}$ 에 $\mathbf{A}$ 를 곱하면 $\mathbf{b}$ 의 column space 성분이 되고, 잔차 $\mathbf{b} - \mathbf{A}\hat{\mathbf{x}}$ 는 $\mathbf{b}$ 의 left null space 성분이다. 정규방정식 $\mathbf{A}^\top(\mathbf{b} - \mathbf{A}\hat{\mathbf{x}}) = \mathbf{0}$ 은 잔차가 left null space 에 속한다는 식이다. 식 (3) 의 $\mathbf{A}$ 와 $\mathbf{b} = [1, 0, 0]^\top$ 에서 잔차는 $[1/3, 1/3, -1/3]^\top$ 으로 식 (4) 의 left null space 성분과 같고, $\mathbf{A}^\top$ 을 곱하면 $[0, 0]^\top$ 이다.

### 6.2 Solvability of Ax = b

$\mathbf{A}\mathbf{x} = \mathbf{b}$ 는 $\mathbf{b}$ 가 left null space 의 모든 vector 와 직교할 때만 해가 있다. Left null space 의 기저 vector 와 $\mathbf{b}$ 의 내적이 모두 0 인지로 확인한다. 식 (3) 에서 $\mathbf{n}^\top[1, 0, 0]^\top = -1$ 이어서 해가 없고, $\mathbf{n}^\top[2, 3, 5]^\top = -2 - 3 + 5 = 0$ 이어서 $\mathbf{b} = [2, 3, 5]^\top$ 은 해 $[2, 3]^\top$ 을 가진다.

### 6.3 Breaking Conditions

부동소수점 계산에서는 $\mathbf{n}^\top\mathbf{b}$ 가 정확히 0 이 되지 않으므로, 해가 있는지를 문턱값으로 판정한다. Left null space 는 특이값 분해에서 특이값 0 에 대응하는 좌특이벡터들이 생성하므로, 문턱값보다 작은 특이값을 0 으로 볼지에 따라 차원이 달라진다. $m \le n$ 이고 rank 가 $m$ 이면 left null space 에는 zero vector 만 있어 모든 $\mathbf{b}$ 에 해가 있다.

---

## Appendix A. Terminology

- **column space**: $\mathbf{x} \in \mathbb{R}^n$ 인 모든 $\mathbf{x}$ 에 대한 $\mathbf{A}\mathbf{x}$ 의 집합. $\mathbf{A}$ 의 열들의 일차결합 전체이다.
- **left null space**: $\mathbf{A}^\top\mathbf{y} = \mathbf{0}$ 인 $\mathbf{y}$ 의 집합. $\mathbf{A}^\top$ 의 null space 이다.
- **null space**: $\mathbf{A}\mathbf{x} = \mathbf{0}$ 인 $\mathbf{x}$ 의 집합.
- **row space**: $\mathbf{y} \in \mathbb{R}^m$ 인 모든 $\mathbf{y}$ 에 대한 $\mathbf{A}^\top\mathbf{y}$ 의 집합. $\mathbf{A}$ 의 행들의 일차결합 전체이다.
- **계수 (rank)**: 행렬에서 서로 일차독립인 열의 최대 개수.
- **부분공간 (subspace)**: Vector space 의 부분집합이면서 그 자체로 vector space 인 집합.
- **일차결합 (linear combination)**: vector 여러 개에 각각 상수를 곱해 모두 더한 vector.
- **정규방정식 (normal equation)**: 최소제곱 해 $\hat{\mathbf{x}}$ 를 구하는 식 $\mathbf{A}^\top\mathbf{A}\hat{\mathbf{x}} = \mathbf{A}^\top\mathbf{b}$.
- **좌특이벡터 (left singular vector)**: 특이값 분해 $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ 에서 $\mathbf{U}$ 의 열. 출력 공간 $\mathbb{R}^m$ 의 단위 vector 이다.
- **직교 (orthogonal)**: 두 vector 의 내적이 0 인 것.
- **최소제곱 해 (least squares solution)**: $\lVert \mathbf{A}\mathbf{x} - \mathbf{b} \rVert$ 를 가장 작게 하는 $\mathbf{x}$.

## Appendix B. Derivation

$\mathrm{Null}(\mathbf{A}^\top) \perp \mathrm{Col}(\mathbf{A})$ 는 column space 의 vector $\mathbf{A}\mathbf{x}$ 와 left null space 의 vector $\mathbf{y}$ 의 내적이 0 임을 보이면 된다.

```math
\begin{aligned}
(\mathbf{A}\mathbf{x})^\top \mathbf{y}
&= \mathbf{x}^\top \mathbf{A}^\top \mathbf{y} \\
&= \mathbf{x}^\top \mathbf{0} \\
&= 0
\end{aligned}
\hspace{19em} (5)
```

- 첫째 등호: 전치의 성질 $(\mathbf{A}\mathbf{x})^\top = \mathbf{x}^\top\mathbf{A}^\top$ 이다.
- 둘째 등호: $\mathbf{y}$ 가 left null space 에 속하므로 $\mathbf{A}^\top\mathbf{y} = \mathbf{0}$ 이다.
- 셋째 등호: zero vector 와의 내적은 0 이다.
