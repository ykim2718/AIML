# Pseudo-Inverse Primer
Rev. 10 | Created: 2026-11-02 | Updated: 2026-11-03 03:20 UTC

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Notation](#3-notation)
- [4. Interpretation](#4-interpretation)
  - [4.1 Condition for the Inverse](#41-condition-for-the-inverse)
  - [4.2 Overdetermined System](#42-overdetermined-system)
  - [4.3 Underdetermined System](#43-underdetermined-system)
- [5. Worked Example](#5-worked-example)
  - [5.1 Underdetermined System](#51-underdetermined-system)
  - [5.2 Overdetermined System](#52-overdetermined-system)
- [6. Application](#6-application)
  - [6.1 Least Squares Regression](#61-least-squares-regression)
  - [6.2 Minimum Norm Solution](#62-minimum-norm-solution)
  - [6.3 Breaking Conditions](#63-breaking-conditions)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Derivation](#appendix-b-derivation)

## 1. Purpose

- **Problem Statement**: 정사각이 아니거나 행렬식이 0 인 행렬에는 역행렬이 없어서, $\mathbf{A}\mathbf{x} = \mathbf{b}$ 를 $\mathbf{A}^{-1}\mathbf{b}$ 로 풀 수 없다.
- **Goal**: 독자가 행렬의 모양에 맞는 pseudo-inverse 식을 골라 계산하고, 그 결과가 최소제곱 해인지 최소 노름 해인지 가릴 수 있게 한다.
- **Non-Goal**: rank 가 모자라는 행렬의 pseudo-inverse 를 특이값 분해로 구하는 계산과 복소 행렬은 다루지 않는다.

## 2. Summary

Pseudo-inverse $\mathbf{A}^{+}$ 는 정식 명칭이 Moore–Penrose pseudo-inverse 이며, 역행렬이 없는 $m \times n$ 행렬에도 정의되는 역행렬의 일반화이다.

식이 미지수보다 많은 과결정 시스템에서는 $(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top$ 이고 $\mathbf{A}^{+}\mathbf{b}$ 가 최소제곱 해이다.
미지수가 식보다 많은 부정 방정식에서는 $\mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top)^{-1}$ 이고 $\mathbf{A}^{+}\mathbf{b}$ 가 최소 노름 해이다.

## 3. Notation

이 문서의 vector 는 모두 열 vector 이다. 이 문서가 쓰는 기호는 아래 Table 1 이다.

Table 1. Symbols

| Symbol                         | Meaning                                                                                            | Shape                          | Assumption                |
| :----------------------------: | :------------------------------------------------------------------------------------------------: | :----------------------------: | :-----------------------: |
| $\mathbb{R}$                   | 실수 전체의 집합. $\mathbb{R}^n$ 은 원소가 실수 $n$ 개인 열 vector 전체이다                        | 집합                           | 실수                      |
| $\mathbf{A}$                   | 선형방정식의 계수 행렬                                                                             | $m \times n$                   | 실수 원소                 |
| $m$, $n$                       | $\mathbf{A}$ 의 행의 수 (식의 수) 와 열의 수 (미지수의 수)                                         | scalar                         | 양의 정수                 |
| $r$                            | $\mathbf{A}$ 의 계수 (rank)                                                                        | scalar                         | $r \le \min(m, n)$        |
| $\mathbf{x}$                   | 미지수 vector                                                                                      | $n \times 1$                   | 실수 원소                 |
| $\lVert \mathbf{x} \rVert$     | $\mathbf{x}$ 의 norm (Euclidean norm), $\lVert \mathbf{x} \rVert = (x_1^2 + \cdots + x_n^2)^{1/2}$ | scalar                         | 0 이상                    |
| $\mathbf{b}$                   | 선형방정식의 우변                                                                                  | $m \times 1$                   | 실수 원소                 |
| $[\mathbf{A} \mid \mathbf{b}]$ | $\mathbf{A}$ 의 오른쪽에 $\mathbf{b}$ 를 열로 붙인 첨가 행렬                                       | $m \times (n + 1)$             | 실수 원소                 |
| $\mathbf{y}$                   | $\mathbf{A}^\top$ 에 곱하는 vector                                                                 | $m \times 1$                   | 실수 원소                 |
| $\mathbf{I}$                   | 단위 행렬                                                                                          | $m \times m$ 또는 $n \times n$ | 대각 원소가 1, 나머지가 0 |
| $\hat{\mathbf{x}}$             | 과결정 시스템의 최소제곱 해                                                                        | $n \times 1$                   | $r = n$                   |
| $\mathbf{x}^{\ast}$            | 부정 방정식의 최소 노름 해                                                                         | $n \times 1$                   | $r = m$                   |

특수 기호는 아래처럼 읽는다.

- $\mathbf{A}^\top$: 영어로 "A transpose", 한국어로 "A 의 전치" 라 읽는다.
- $\mathbf{A}^{-1}$: 영어로 "A inverse", 한국어로 "A 의 역행렬" 이라 읽는다.
- $\mathbf{A}^{+}$: 영어로 "A plus" 또는 "the pseudo-inverse of A", 한국어로 "A 의 pseudo-inverse" 라 읽는다.
- $\det(\mathbf{A})$: 영어로 "the determinant of A", 한국어로 "A 의 행렬식" 이라 읽는다.
- $[\mathbf{A} \mid \mathbf{b}]$: 영어로 "the augmented matrix A b", 한국어로 "A 와 b 의 첨가 행렬" 이라 읽는다. 여기의 $\mid$ 는 열의 경계를 나타낸다.
- $\lVert \mathbf{x} \rVert$: 영어로 "the norm of x", 한국어로 "x 의 노름" 이라 읽는다.
- $\hat{\mathbf{x}}$: 영어로 "x hat", 한국어로 "엑스 햇" 이라 읽는다. 기호 위의 hat 은 그 값이 최소제곱으로 구한 해임을 나타낸다.
- $\mathbf{x}^{\ast}$: 영어로 "x star", 한국어로 "엑스 스타" 라 읽는다.

## 4. Interpretation

$\mathbf{A}\mathbf{x} = \mathbf{b}$ 의 해는 $r = m = n$ 이면 $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$ 이고, 일반적으로는 $\mathbf{x} = \mathbf{A}^{+}\mathbf{b}$ 로 적는다. $\mathbf{A}^{+}$ 는 rank $r$ 이 열의 수 $n$ 과 같은지, 행의 수 $m$ 과 같은지에 따라 아래처럼 구한다.

```math
\mathbf{A}^{+} =
\begin{cases}
(\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top & r = n \ \ (\text{full column rank}) \\
\mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top)^{-1} & r = m \ \ (\text{full row rank})
\end{cases}
\hspace{19em} (1)
```

$\mathbf{A}^{+}$ 는 rank 와 관계없이 모든 $m \times n$ 행렬에 있고, rank 는 $\mathbf{A}^{+}$ 를 구하는 식만 정한다. 최소제곱 해, 역행렬, 최소 노름 해의 세 풀이에서 $\mathbf{b}$ 앞에 곱한 행렬이 각 rank 조건에서의 $\mathbf{A}^{+}$ 이다.

- $r = n$: $\mathbf{A}^{+} = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top$ 이고, 최소제곱 해를 준다. 식 (1) 의 첫 줄이다.
- $r = m$: $\mathbf{A}^{+} = \mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top)^{-1}$ 이고, 최소 노름 해를 준다. 식 (1) 의 둘째 줄이다.
- $r = m = n$: $\mathbf{A}^{+} = \mathbf{A}^{-1}$ 이고, 하나뿐인 해를 준다 ([4.1](#41-condition-for-the-inverse)).
- $r \lt m$ 이고 $r \lt n$: $\mathbf{A}^\top\mathbf{A}$ 와 $\mathbf{A}\mathbf{A}^\top$ 모두 역행렬이 없어 식 (1) 의 두 식을 쓸 수 없으므로, $\mathbf{A}^{+}$ 는 특이값 분해로 구한다 ([6.3](#63-breaking-conditions)).

$\mathbf{A}^\top\mathbf{A}$ 는 $n \times n$, $\mathbf{A}\mathbf{A}^\top$ 는 $m \times m$ 의 정사각 행렬이지만, 정사각 행렬은 rank 가 그 크기와 같을 때만 역행렬이 있다. 두 행렬의 rank 는 모두 $\mathbf{A}$ 의 rank $r$ 과 같으므로, $r = n$ 이면 $\mathbf{A}^\top\mathbf{A}$ 만, $r = m$ 이면 $\mathbf{A}\mathbf{A}^\top$ 만 역행렬이 있고, 역행렬이 있는 쪽 식만 쓸 수 있다. $r = m = n$ 이면 두 식이 같은 $\mathbf{A}^{-1}$ 을 주므로, rank 는 $\mathbf{A}^{+}$ 를 바꾸는 것이 아니라 쓸 수 있는 식을 정한다.

$\mathbf{A}^{+}$ 는 $\mathbf{A}$ 만으로 정해지고 $\mathbf{b}$ 를 쓰지 않으므로, 첨가 행렬 $[\mathbf{A} \mid \mathbf{b}]$ 의 rank 와는 무관하다. $\mathbf{A}^{+}\mathbf{b}$ 는 언제나 $\lVert \mathbf{A}\mathbf{x} - \mathbf{b} \rVert$ 를 가장 작게 하는 $\mathbf{x}$ 가운데 norm 이 가장 작은 하나이다. 다만 norm 으로 고르는 일은 그런 $\mathbf{x}$ 가 여럿일 때만 일어나며, 아래처럼 $[\mathbf{A} \mid \mathbf{b}]$ 의 rank 와 $r$ 에 따라 갈린다.

- $r = \mathrm{rank}([\mathbf{A} \mid \mathbf{b}])$ 이고 $r = n$: 해가 하나뿐이어서 고를 것이 없고, $\mathbf{A}^{+}\mathbf{b}$ 는 그 해 자체이다. 정사각이면 $\mathbf{A}^{-1}\mathbf{b}$ 이다.
- $r = \mathrm{rank}([\mathbf{A} \mid \mathbf{b}])$ 이고 $r \lt n$: 해가 무수히 많고, $\mathbf{A}^{+}\mathbf{b}$ 는 해 가운데 최소 노름 해이다.
- $r \lt \mathrm{rank}([\mathbf{A} \mid \mathbf{b}])$: 해가 없고, $\mathbf{A}^{+}\mathbf{b}$ 는 최소제곱 해이다. $r = n$ 이면 최소제곱 해가 하나뿐이어서 그 해이고, $r \lt n$ 이면 최소제곱 해 가운데 norm 이 가장 작은 해이다.

### 4.1 Condition for the Inverse

역행렬 $\mathbf{A}^{-1}$ 은 $\mathbf{A}$ 가 정사각 행렬 ($m = n$) 이고 $\det(\mathbf{A}) \ne 0$ 일 때만 있다. 실제 data 의 선형방정식은 식의 수와 미지수의 수가 다르거나, 한 열이 다른 열들의 일차결합이어서 이 조건을 만족하지 않는 경우가 많다. 정사각이고 $\det(\mathbf{A}) \ne 0$ 이면 식 (1) 의 두 식은 모두 $\mathbf{A}^{-1}$ 과 같다.

### 4.2 Overdetermined System

과결정 시스템 (overdetermined system) 은 식의 수가 미지수의 수보다 많은 ($m \gt n$) 선형방정식이다. 측정값 여러 개로 직선 하나를 맞추는 회귀처럼 모든 식을 동시에 만족하는 해가 대개 없다. $r = n$ 이면 $\mathbf{A}^\top\mathbf{A}$ 의 역행렬이 있고, $\mathbf{A}^{+}\mathbf{b}$ 는 잔차의 제곱합 $\lVert \mathbf{A}\mathbf{x} - \mathbf{b} \rVert^2$ 을 가장 작게 하는 최소제곱 해이다.

```math
\hat{\mathbf{x}} = \mathbf{A}^{+}\mathbf{b} = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}, \qquad
\mathbf{A}^{+}\mathbf{A} = \mathbf{I}
\hspace{19em} (2)
```

$\mathbf{A}^{+}$ 를 $\mathbf{A}$ 의 왼쪽에 곱하면 $n \times n$ 단위 행렬이 되므로, 이 식을 left inverse 라고도 부른다. 유도는 [Appendix B](#appendix-b-derivation) 에 있다.

### 4.3 Underdetermined System

부정 방정식 (underdetermined system) 은 미지수의 수가 식의 수보다 많은 ($m \lt n$) 선형방정식이다. 해가 있으면 무수히 많다. $r = m$ 이면 $\mathbf{A}\mathbf{A}^\top$ 의 역행렬이 있고, $\mathbf{A}^{+}\mathbf{b}$ 는 해 가운데 norm 이 가장 작은 최소 노름 해이다.

```math
\mathbf{x}^{\ast} = \mathbf{A}^{+}\mathbf{b} = \mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top)^{-1}\mathbf{b}, \qquad
\mathbf{A}\mathbf{A}^{+} = \mathbf{I}
\hspace{19em} (3)
```

$\mathbf{A}^{+}$ 를 $\mathbf{A}$ 의 오른쪽에 곱하면 $m \times m$ 단위 행렬이 되므로, 이 식을 right inverse 라고도 부른다. 최소 노름 해가 row space 에 있는 이유와 유도는 [Minimum Norm Solution Primer](../Minimum-Norm-Solution/minimum-norm-solution-primer-ko.md) 에 있다.

## 5. Worked Example

### 5.1 Underdetermined System

미지수가 3 개이고 식이 1 개인 부정 방정식 $x_1 + x_2 + x_3 = 3$ 을 행렬로 적으면 $m = 1$, $n = 3$, $r = 1 = m$ 이다.

```math
\mathbf{A} = \begin{bmatrix} 1 & 1 & 1 \end{bmatrix}, \qquad
\mathbf{b} = \begin{bmatrix} 3 \end{bmatrix}
\hspace{19em} (4)
```

식 (3) 의 $\mathbf{A}^{+}$ 를 차례로 계산하면 아래이다.

```math
\mathbf{A}\mathbf{A}^\top = \begin{bmatrix} 1 & 1 & 1 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \cdot 1 + 1 \cdot 1 + 1 \cdot 1 \end{bmatrix} = \begin{bmatrix} 3 \end{bmatrix}
\hspace{19em} (5)
```

```math
(\mathbf{A}\mathbf{A}^\top)^{-1} = \begin{bmatrix} 1/3 \end{bmatrix}
\hspace{19em} (6)
```

```math
\mathbf{A}^{+} = \mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top)^{-1} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} \begin{bmatrix} 1/3 \end{bmatrix} = \begin{bmatrix} 1/3 \\ 1/3 \\ 1/3 \end{bmatrix}
\hspace{19em} (7)
```

```math
\mathbf{x}^{\ast} = \mathbf{A}^{+}\mathbf{b} = \begin{bmatrix} 1/3 \\ 1/3 \\ 1/3 \end{bmatrix} \begin{bmatrix} 3 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}
\hspace{19em} (8)
```

$\mathbf{A}^{+}$ 는 $3 \times 1$ 행렬이고 $\mathbf{A}\mathbf{A}^{+} = [1/3 + 1/3 + 1/3] = [1]$ 이다. $\mathbf{x}^{\ast} = [1, 1, 1]^\top$ 은 원소의 합이 3 인 해 가운데 norm 이 가장 작은 해이며, Minimum Norm Solution Primer 의 Worked Example 에서 구한 값과 같다.

### 5.2 Overdetermined System

식이 3 개이고 미지수가 2 개인 과결정 시스템을 둔다. 두 열이 일차독립이어서 $m = 3$, $n = 2$, $r = 2 = n$ 이다.

```math
\mathbf{A} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix}, \qquad
\mathbf{b} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
\hspace{19em} (9)
```

세 식 $x_1 = 1$, $x_2 = 0$, $x_1 + x_2 = 0$ 을 동시에 만족하는 해는 없다. 식 (2) 의 $\mathbf{A}^{+}$ 를 차례로 계산하면 아래이다.

```math
\mathbf{A}^\top\mathbf{A} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix} \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{bmatrix} = \begin{bmatrix} 2 & 1 \\ 1 & 2 \end{bmatrix}
\hspace{19em} (10)
```

```math
(\mathbf{A}^\top\mathbf{A})^{-1} = \frac{1}{3} \begin{bmatrix} 2 & -1 \\ -1 & 2 \end{bmatrix}
\hspace{19em} (11)
```

```math
\mathbf{A}^{+} = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top = \frac{1}{3} \begin{bmatrix} 2 & -1 & 1 \\ -1 & 2 & 1 \end{bmatrix}
\hspace{19em} (12)
```

```math
\hat{\mathbf{x}} = \mathbf{A}^{+}\mathbf{b} = \begin{bmatrix} 2/3 \\ -1/3 \end{bmatrix}
\hspace{19em} (13)
```

$\mathbf{A}^{+}$ 는 $2 \times 3$ 행렬이고 $\mathbf{A}^{+}\mathbf{A}$ 는 $2 \times 2$ 단위 행렬이다. 잔차는 $\mathbf{b} - \mathbf{A}\hat{\mathbf{x}} = [1/3, 1/3, -1/3]^\top$ 이고 $\mathbf{A}^\top$ 을 곱하면 $[0, 0]^\top$ 이어서, 잔차가 $\mathbf{A}$ 의 두 열과 모두 직교한다. 같은 $\mathbf{A}$ 와 $\mathbf{b}$ 의 잔차 분해는 [Left Null Space Primer](../Left-Null-Space/left-null-space-primer-ko.md) 에 있다.

## 6. Application

Pseudo-inverse 는 역행렬이 없는 선형방정식에서 해를 하나로 정한다.

### 6.1 Least Squares Regression

선형 회귀에서 sample 수가 feature 수보다 많으면 자료 행렬은 과결정 시스템의 $\mathbf{A}$ 이다. 식 (2) 의 $\hat{\mathbf{x}}$ 가 잔차의 제곱합을 가장 작게 하는 weight 이며, 정규방정식 $\mathbf{A}^\top\mathbf{A}\hat{\mathbf{x}} = \mathbf{A}^\top\mathbf{b}$ 의 해와 같다.

### 6.2 Minimum Norm Solution

선형 회귀에서 feature 수가 sample 수보다 많으면 자료 행렬은 부정 방정식의 $\mathbf{A}$ 이고, 자료를 정확히 맞추는 weight 가 무수히 많다. 식 (3) 의 $\mathbf{x}^{\ast}$ 는 그 가운데 norm 이 가장 작은 weight 이다.

### 6.3 Breaking Conditions

$r \lt n$ 이면 $\mathbf{A}^\top\mathbf{A}$ 의 역행렬이 없고, $r \lt m$ 이면 $\mathbf{A}\mathbf{A}^\top$ 의 역행렬이 없어 식 (1) 의 해당 식을 쓸 수 없다. Feature 사이의 공선성이 강하면 $r = n$ 이어도 $\mathbf{A}^\top\mathbf{A}$ 의 행렬식이 0 에 가까워져 역행렬의 원소가 매우 커진다. 이때는 특이값 분해로 $\mathbf{A}^{+}$ 를 구하며, 그 계산은 [Singular Value Decomposition Primer](../Singular-Value-Decomposition/singular-value-decomposition-primer-ko.md) 에 있다.

---

## Appendix A. Terminology

- **column space**: $\mathbf{x} \in \mathbb{R}^n$ 인 모든 $\mathbf{x}$ 에 대한 $\mathbf{A}\mathbf{x}$ 의 집합. $\mathbf{A}$ 의 열들의 일차결합 전체이다.
- **full column rank**: 행렬의 rank 가 열의 수 $n$ 과 같은 것.
- **full row rank**: 행렬의 rank 가 행의 수 $m$ 과 같은 것.
- **left inverse**: $\mathbf{A}$ 의 왼쪽에 곱하여 단위 행렬을 만드는 행렬.
- **norm (Euclidean norm)**: vector 원소의 제곱을 모두 더한 값의 제곱근. Vector 의 길이이며, $\lVert \mathbf{x} \rVert = (x_1^2 + \cdots + x_n^2)^{1/2}$ 로 적는다. L1, L∞ 같은 다른 vector norm 과 matrix norm 은 [Types of Norms](../Norm-Types/norm-types-ko.md) 에 있다.
- **pseudo-inverse**: 정사각이 아니거나 역행렬이 없는 행렬에도 정의되는 역행렬의 일반화. Moore–Penrose pseudo-inverse 라고도 한다.
- **right inverse**: $\mathbf{A}$ 의 오른쪽에 곱하여 단위 행렬을 만드는 행렬.
- **row space**: $\mathbf{y} \in \mathbb{R}^m$ 인 모든 $\mathbf{y}$ 에 대한 $\mathbf{A}^\top\mathbf{y}$ 의 집합. $\mathbf{A}$ 의 행들의 일차결합 전체이다.
- **계수 (rank)**: 행렬에서 서로 일차독립인 열의 최대 개수.
- **공선성 (collinearity)**: 한 feature 가 다른 feature 들의 일차결합에 가까운 것.
- **과결정 시스템 (overdetermined system)**: 식의 수가 미지수의 수보다 많은 선형방정식.
- **부정 방정식 (underdetermined system)**: 미지수의 수가 식의 수보다 많은 선형방정식.
- **역행렬 (inverse matrix)**: 정사각 행렬 $\mathbf{A}$ 에 대해 $\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I}$ 인 행렬.
- **일차결합 (linear combination)**: vector 여러 개에 각각 상수를 곱해 모두 더한 vector.
- **정규방정식 (normal equation)**: 최소제곱 해 $\hat{\mathbf{x}}$ 를 구하는 식 $\mathbf{A}^\top\mathbf{A}\hat{\mathbf{x}} = \mathbf{A}^\top\mathbf{b}$.
- **직교 (orthogonal)**: 두 vector 의 내적이 0 인 것.
- **첨가 행렬 (augmented matrix)**: 계수 행렬 $\mathbf{A}$ 의 오른쪽에 우변 $\mathbf{b}$ 를 열 하나로 붙인 행렬 $[\mathbf{A} \mid \mathbf{b}]$.
- **최소 노름 해 (minimum norm solution)**: 해가 무수히 많은 선형방정식의 해 가운데 norm 이 가장 작은 해.
- **최소제곱 해 (least squares solution)**: $\lVert \mathbf{A}\mathbf{x} - \mathbf{b} \rVert$ 를 가장 작게 하는 $\mathbf{x}$.
- **특이값 분해 (singular value decomposition)**: 행렬을 $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ 로 나누는 분해. $\mathbf{U}$ 와 $\mathbf{V}$ 의 열은 단위 vector 이고 $\boldsymbol{\Sigma}$ 의 대각 원소가 특이값이다.
- **행렬식 (determinant)**: 정사각 행렬에서 계산하는 scalar. 0 이 아닐 때만 역행렬이 있다.

## Appendix B. Derivation

식 (2) 는 잔차 $\mathbf{b} - \mathbf{A}\hat{\mathbf{x}}$ 가 $\mathbf{A}$ 의 모든 열과 직교한다는 조건에서 나온다.

```math
\begin{aligned}
\mathbf{A}^\top(\mathbf{b} - \mathbf{A}\hat{\mathbf{x}}) &= \mathbf{0} \\
\mathbf{A}^\top\mathbf{A}\hat{\mathbf{x}} &= \mathbf{A}^\top\mathbf{b} \\
\hat{\mathbf{x}} &= (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}
\end{aligned}
\hspace{19em} (14)
```

- 첫째 줄: $\mathbf{A}\hat{\mathbf{x}}$ 가 column space 에서 $\mathbf{b}$ 와 가장 가까운 점이면 잔차는 column space 와 직교하고, 이는 $\mathbf{A}$ 의 각 열과 잔차의 내적이 0 이라는 뜻이다.
- 둘째 줄: 괄호를 풀고 $\mathbf{A}^\top\mathbf{A}\hat{\mathbf{x}}$ 를 우변으로 옮긴 정규방정식이다.
- 셋째 줄: $r = n$ 이면 $\mathbf{A}^\top\mathbf{A}$ 의 역행렬이 있으므로 양변의 왼쪽에 곱한다.

따라서 식 (2) 가 성립한다.

```math
\hat{\mathbf{x}} = \mathbf{A}^{+}\mathbf{b} = (\mathbf{A}^\top\mathbf{A})^{-1}\mathbf{A}^\top\mathbf{b}, \qquad
\mathbf{A}^{+}\mathbf{A} = \mathbf{I}
\hspace{19em} (2)
```

식 (3) 은 최소 노름 해가 row space 에 있어 $\mathbf{x}^{\ast} = \mathbf{A}^\top\mathbf{y}$ 로 적힌다는 조건에서 나온다.

```math
\begin{aligned}
\mathbf{A}\mathbf{A}^\top\mathbf{y} &= \mathbf{b} \\
\mathbf{y} &= (\mathbf{A}\mathbf{A}^\top)^{-1}\mathbf{b} \\
\mathbf{x}^{\ast} &= \mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top)^{-1}\mathbf{b}
\end{aligned}
\hspace{19em} (15)
```

- 첫째 줄: $\mathbf{x}^{\ast} = \mathbf{A}^\top\mathbf{y}$ 를 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 에 넣는다. $\mathbf{y}$ 는 $m \times 1$ vector 이다.
- 둘째 줄: $r = m$ 이면 $\mathbf{A}\mathbf{A}^\top$ 의 역행렬이 있으므로 양변의 왼쪽에 곱한다.
- 셋째 줄: 둘째 줄의 $\mathbf{y}$ 를 $\mathbf{x}^{\ast} = \mathbf{A}^\top\mathbf{y}$ 에 넣는다.

따라서 식 (3) 이 성립한다.

```math
\mathbf{x}^{\ast} = \mathbf{A}^{+}\mathbf{b} = \mathbf{A}^\top(\mathbf{A}\mathbf{A}^\top)^{-1}\mathbf{b}, \qquad
\mathbf{A}\mathbf{A}^{+} = \mathbf{I}
\hspace{19em} (3)
```
