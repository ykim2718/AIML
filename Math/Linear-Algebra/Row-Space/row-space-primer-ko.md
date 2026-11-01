# Row Space Primer
Rev. 0 | Created: 2026-11-01 | Updated: 2026-11-01 14:48 UTC

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Notation](#3-notation)
- [4. Interpretation](#4-interpretation)
  - [4.1 Set Definition](#41-set-definition)
  - [4.2 Orthogonality to the Null Space](#42-orthogonality-to-the-null-space)
  - [4.3 Dimension and Rank](#43-dimension-and-rank)
  - [4.4 Orthogonal Decomposition of the Input Space](#44-orthogonal-decomposition-of-the-input-space)
- [5. Worked Example](#5-worked-example)
- [6. Application](#6-application)
  - [6.1 Minimum-Norm Solution](#61-minimum-norm-solution)
  - [6.2 Right Singular Vectors](#62-right-singular-vectors)
  - [6.3 Breaking Conditions](#63-breaking-conditions)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Derivation](#appendix-b-derivation)

## 1. Purpose

- **Problem Statement**: 행렬 $\mathbf{A}$ 의 column space 는 출력이 놓이는 공간을 알려 주지만, 입력 vector 의 어느 성분이 출력에 쓰이고 어느 성분이 버려지는지는 column space 만으로 드러나지 않는다.
- **Goal**: 독자가 row space 를 구하고, 입력 vector 를 row space 성분과 null space 성분으로 나누어 출력을 정하는 성분을 가려낼 수 있게 한다.
- **Non-Goal**: 복소 행렬의 row space 와 무한 차원 공간의 선형 연산자는 다루지 않는다.

## 2. Summary

Row space 는 $m \times n$ 행렬 $\mathbf{A}$ 의 행 vector 들의 일차결합 전체이며, $\mathbf{A}^\top$ 의 column space 와 같은 $\mathbb{R}^n$ 의 부분공간이다.
Row space 는 null space 와 직교하고, 그 차원은 column space 의 차원과 같은 rank 이다.
입력 공간 $\mathbb{R}^n$ 의 모든 vector 는 row space 성분과 null space 성분의 합으로 하나씩만 나뉘며, $\mathbf{A}$ 의 출력은 row space 성분만으로 정해진다.

## 3. Notation

이 문서가 쓰는 기호는 아래 Table 1 과 같다.

Table 1. Symbols

| Symbol                       | Meaning                                                | Shape                          | Assumption                                               |
| :--------------------------: | :----------------------------------------------------: | :----------------------------: | :------------------------------------------------------: |
| $\mathbf{A}$                 | 선형 변환을 나타내는 행렬                              | $m \times n$                   | 실수 원소                                                |
| $m$, $n$                     | $\mathbf{A}$ 의 행의 수와 열의 수                      | scalar                         | 양의 정수                                                |
| $\mathbf{r}_i^\top$          | $\mathbf{A}$ 의 $i$ 번째 행                            | $1 \times n$                   | $i = 1, \dots, m$                                        |
| $\mathbf{x}$                 | 입력 vector                                            | $n \times 1$                   | 실수 원소                                                |
| $\mathbf{y}$                 | $\mathbf{A}^\top$ 에 곱하는 계수 vector                | $m \times 1$                   | 실수 원소                                                |
| $\mathbf{b}$                 | 선형방정식 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 의 우변 | $m \times 1$                   | 실수 원소                                                |
| $\mathbf{0}$                 | zero vector                                            | $m \times 1$ 또는 $n \times 1$ | 모든 원소가 0                                            |
| $\mathrm{Row}(\mathbf{A})$   | $\mathbf{A}$ 의 row space                              | $\mathbb{R}^n$ 의 부분공간     | zero vector 를 포함                                      |
| $\mathrm{Col}(\mathbf{A})$   | $\mathbf{A}$ 의 column space                           | $\mathbb{R}^m$ 의 부분공간     | zero vector 를 포함                                      |
| $\mathrm{Null}(\mathbf{A})$  | $\mathbf{A}$ 의 null space                             | $\mathbb{R}^n$ 의 부분공간     | $\mathbf{A}\mathbf{x} = \mathbf{0}$ 인 $\mathbf{x}$ 전체 |
| $\mathrm{rank}(\mathbf{A})$  | $\mathbf{A}$ 의 계수                                   | scalar                         | $0 \le \mathrm{rank}(\mathbf{A}) \le \min(m, n)$         |
| $r$                          | $\mathrm{rank}(\mathbf{A})$ 를 줄여 쓴 것              | scalar                         | $r = \mathrm{rank}(\mathbf{A})$                          |
| $\mathbf{x}_{\mathrm{row}}$  | $\mathbf{x}$ 의 row space 성분                         | $n \times 1$                   | $\mathrm{Row}(\mathbf{A})$ 에 속함                       |
| $\mathbf{x}_{\mathrm{null}}$ | $\mathbf{x}$ 의 null space 성분                        | $n \times 1$                   | $\mathrm{Null}(\mathbf{A})$ 에 속함                      |
| $\mathbf{n}$                 | Worked Example 에서 null space 를 생성하는 vector      | $3 \times 1$                   | zero vector 가 아님                                      |
| $\sigma_k$, $\mathbf{v}_k$   | $\mathbf{A}$ 의 $k$ 번째 특이값과 우특이벡터           | scalar, $n \times 1$           | $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$                 |

특수 기호는 아래와 같이 읽는다.

- $\mathbb{R}^n$: 영어로 "R n" 또는 "real n-space", 한국어로 "n 차원 실수 공간" 이라 읽는다.
- $\mathbf{A}^\top$: 영어로 "A transpose", 한국어로 "A 의 전치" 또는 "A 트랜스포즈" 라 읽는다.
- $\{ \mathbf{A}^\top\mathbf{y} \mid \mathbf{y} \in \mathbb{R}^m \}$ 의 $\mid$ 와 $\in$: 영어로 "such that" 과 "in", 한국어로 "인" 과 "에 속하는" 이라 읽는다.
- $\perp$: 영어로 "is orthogonal to", 한국어로 "직교한다" 라 읽는다.
- $\oplus$: 영어로 "direct sum", 한국어로 "직합" 이라 읽는다. 두 부분공간이 zero vector 만 공유하고, 두 공간의 vector 를 더해 전체 공간을 만든다는 뜻이다.
- $\Longleftrightarrow$: 영어로 "if and only if", 한국어로 "필요충분조건" 또는 "동치" 라 읽는다.
- $\propto$: 영어로 "is proportional to", 한국어로 "비례한다" 라 읽는다.
- $\dim$: 영어로 "dimension of", 한국어로 "차원" 이라 읽는다.
- $\mathbf{A}^{+}$: 영어로 "A plus" 또는 "the pseudo-inverse of A", 한국어로 "A 의 pseudo-inverse" 라 읽는다.
- $\lVert \mathbf{x} \rVert$: 영어로 "the norm of x", 한국어로 "x 의 노름" 이라 읽는다.

## 4. Interpretation

Row space 는 $\mathbf{A}$ 의 행 vector 들을 일차결합해 얻는 vector 전체이며, $\mathbf{A}^\top$ 에 계수 vector 를 곱한 결과 전체로 적을 수 있다.

```math
\mathrm{Row}(\mathbf{A}) = \{ y_1 \mathbf{r}_1 + \cdots + y_m \mathbf{r}_m \} = \{ \mathbf{A}^\top\mathbf{y} \mid \mathbf{y} \in \mathbb{R}^m \} = \mathrm{Col}(\mathbf{A}^\top) \hspace{19em} (1)
```

### 4.1 Set Definition

$\mathbf{A}^\top$ 의 열은 $\mathbf{A}$ 의 행을 전치한 것이므로, $\mathbf{A}^\top\mathbf{y}$ 는 $\mathbf{A}$ 의 행들을 $\mathbf{y}$ 의 원소로 일차결합한 vector 이다. 행 vector 는 원소가 $n$ 개이므로 row space 는 입력 공간 $\mathbb{R}^n$ 의 부분공간이고, column space 는 출력 공간 $\mathbb{R}^m$ 의 부분공간이다. Column space 와 row space 를 견주면 아래 Table 2 와 같다.

Table 2. Column space and row space

| Aspect        | Column space                                     | Row space                                  |
| :-----------: | :----------------------------------------------: | :----------------------------------------: |
| Spanned by    | $\mathbf{A}$ 의 열 vector                        | $\mathbf{A}$ 의 행 vector                  |
| Ambient space | 출력 공간 $\mathbb{R}^m$                         | 입력 공간 $\mathbb{R}^n$                   |
| Meaning       | $\mathbf{A}\mathbf{x}$ 가 될 수 있는 vector 전체 | $\mathbf{A}$ 가 출력에 쓰는 입력 성분 전체 |
| Dimension     | $r$                                              | $r$                                        |
| Orthogonal to | $\mathrm{Null}(\mathbf{A}^\top)$                 | $\mathrm{Null}(\mathbf{A})$                |

Column space 의 직교 여공간은 $\mathbf{A}^\top\mathbf{y} = \mathbf{0}$ 인 $\mathbf{y}$ 전체인 left null space $\mathrm{Null}(\mathbf{A}^\top)$ 이다. 네 부분공간의 차원은 column space 와 row space 가 $r$, null space 가 $n - r$, left null space 가 $m - r$ 이다.

### 4.2 Orthogonality to the Null Space

$\mathbf{A}\mathbf{x}$ 의 $i$ 번째 원소는 $i$ 번째 행과 $\mathbf{x}$ 의 내적이다. 따라서 $\mathbf{A}\mathbf{x} = \mathbf{0}$ 은 $\mathbf{x}$ 가 모든 행과 직교한다는 뜻이다.

```math
\mathbf{A}\mathbf{x} = \begin{bmatrix} \mathbf{r}_1^\top \mathbf{x} \\ \vdots \\ \mathbf{r}_m^\top \mathbf{x} \end{bmatrix} = \mathbf{0}
\quad \Longleftrightarrow \quad
\mathbf{r}_i^\top \mathbf{x} = 0 \ \ (i = 1, \dots, m)
\hspace{19em} (2)
```

$\mathbf{x}$ 가 행 하나하나와 직교하면 행들의 일차결합과도 직교하므로, null space 의 모든 vector 는 row space 의 모든 vector 와 직교한다. 이를 $\mathrm{Row}(\mathbf{A}) \perp \mathrm{Null}(\mathbf{A})$ 로 적으며, 유도는 [Appendix B](#appendix-b-derivation) 에 있다.

### 4.3 Dimension and Rank

Row space 의 차원은 column space 의 차원과 언제나 같고, 그 값이 $\mathbf{A}$ 의 rank 이다.

```math
\dim \mathrm{Row}(\mathbf{A}) = \dim \mathrm{Col}(\mathbf{A}) = \mathrm{rank}(\mathbf{A}) = r \hspace{19em} (3)
```

$\mathbf{A}$ 를 기약 행사다리꼴 $\mathbf{R}$ 로 바꾸면 기본 행 연산은 행의 일차결합만 만들므로 row space 가 바뀌지 않는다. $\mathbf{R}$ 의 0 이 아닌 행의 수와 pivot 열의 수가 같고, pivot 열의 수가 일차독립인 열의 최대 개수이므로 식 (3) 이 성립한다.

### 4.4 Orthogonal Decomposition of the Input Space

입력 공간 $\mathbb{R}^n$ 은 row space 와 null space 의 직합이다. 모든 $\mathbf{x}$ 는 두 성분의 합으로 하나씩만 나뉘고, $\mathbf{A}$ 의 출력은 row space 성분만으로 정해진다.

```math
\mathbb{R}^n = \mathrm{Row}(\mathbf{A}) \oplus \mathrm{Null}(\mathbf{A}), \qquad
\mathbf{x} = \mathbf{x}_{\mathrm{row}} + \mathbf{x}_{\mathrm{null}}, \qquad
\mathbf{A}\mathbf{x} = \mathbf{A}\mathbf{x}_{\mathrm{row}}
\hspace{19em} (4)
```

두 부분공간의 차원은 $r$ 과 $n - r$ 이어서 더하면 $n$ 이고, 서로 직교하므로 둘 다에 속하는 vector 는 zero vector 하나뿐이다. $\mathbf{A}\mathbf{x}_{\mathrm{null}} = \mathbf{0}$ 이므로 null space 성분은 출력에 아무것도 더하지 않는다.

## 5. Worked Example

행이 두 개이고 열이 세 개인 행렬을 둔다.

```math
\mathbf{A} = \begin{bmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \end{bmatrix}, \quad
\mathbf{r}_1 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, \quad
\mathbf{r}_2 = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}
\hspace{19em} (5)
```

두 행은 일차독립이므로 row space 는 $\mathbb{R}^3$ 안의 평면이고 rank 는 2 이다. Column space 는 $\mathbb{R}^2$ 전체이고, null space 는 아래 $\mathbf{n}$ 의 상수배 전체인 직선이다.

```math
\mathbf{n} = \begin{bmatrix} -1 \\ -1 \\ 1 \end{bmatrix}, \quad
\mathbf{r}_1^\top \mathbf{n} = -1 + 0 + 1 = 0, \quad
\mathbf{r}_2^\top \mathbf{n} = 0 - 1 + 1 = 0
\hspace{19em} (6)
```

식 (6) 에서 $\mathbf{n}$ 은 두 행과 모두 직교한다. 차원은 row space 가 2, null space 가 1 로 더하면 $n = 3$ 이고, left null space 는 $m - r = 0$ 차원이어서 zero vector 만 있다.

입력 $\mathbf{x} = [3, 0, 0]^\top$ 을 두 성분으로 나누면 아래와 같다.

```math
\begin{gathered}
\mathbf{x}_{\mathrm{null}} = \frac{\mathbf{x}^\top \mathbf{n}}{\mathbf{n}^\top \mathbf{n}}\, \mathbf{n} = \frac{-3}{3} \begin{bmatrix} -1 \\ -1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \\ 1 \\ -1 \end{bmatrix}, \qquad
\mathbf{x}_{\mathrm{row}} = \mathbf{x} - \mathbf{x}_{\mathrm{null}} = \begin{bmatrix} 2 \\ -1 \\ 1 \end{bmatrix} = 2\,\mathbf{r}_1 - \mathbf{r}_2 \\[2ex]
\mathbf{A}\mathbf{x} = \begin{bmatrix} 3 \\ 0 \end{bmatrix}, \qquad
\mathbf{A}\mathbf{x}_{\mathrm{row}} = \begin{bmatrix} 2 + 1 \\ -1 + 1 \end{bmatrix} = \begin{bmatrix} 3 \\ 0 \end{bmatrix}, \qquad
\mathbf{A}\mathbf{x}_{\mathrm{null}} = \begin{bmatrix} 1 - 1 \\ 1 - 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
\end{gathered}
\hspace{19em} (7)
```

Row space 성분 $[2, -1, 1]^\top$ 은 식 (7) 의 첫 줄과 같이 두 행의 일차결합이고, $\mathbf{n}$ 과의 내적이 $2 \cdot (-1) + (-1) \cdot (-1) + 1 \cdot 1 = 0$ 이다. $\mathbf{x}$ 와 row space 성분에 $\mathbf{A}$ 를 곱한 결과가 같은 $[3, 0]^\top$ 이므로, 출력은 row space 성분만으로 정해진다.

## 6. Application

Row space 는 해가 무수히 많은 선형방정식에서 하나를 고르거나, 특이값 분해에서 입력 쪽 기저를 나눌 때 쓰인다.

### 6.1 Minimum-Norm Solution

$\mathbf{A}\mathbf{x} = \mathbf{b}$ 의 해가 무수히 많으면, 해들은 같은 row space 성분에 서로 다른 null space 성분을 더한 것이다. 식 (4) 의 두 성분이 직교하므로 해의 norm 의 제곱은 두 성분의 norm 의 제곱을 더한 값이고, null space 성분이 zero vector 인 해의 norm 이 가장 작다. Pseudo-inverse 로 구한 해 $\mathbf{A}^{+}\mathbf{b}$ 가 이 해이다.

식 (5) 의 $\mathbf{A}$ 와 $\mathbf{b} = [3, 0]^\top$ 에서 $\mathbf{A}^{+}\mathbf{b} = [2, -1, 1]^\top$ 이며, 식 (7) 의 row space 성분과 같다. 선형 회귀에서 feature 가 collinear 하면 weight 가 하나로 정해지지 않고, pseudo-inverse 는 그 가운데 row space 에 있는 weight 를 고른다.

### 6.2 Right Singular Vectors

특이값 분해에서 0 이 아닌 특이값에 대응하는 우특이벡터 $r$ 개는 row space 의 정규직교 기저이고, 특이값 0 에 대응하는 나머지 $n - r$ 개는 null space 의 정규직교 기저이다. 식 (5) 의 $\mathbf{A}$ 는 특이값이 $\sigma_1 = \sqrt{3} \approx 1.7321$, $\sigma_2 = 1$ 이다. $\mathbf{v}_1 \propto [1, 1, 2]^\top = \mathbf{r}_1 + \mathbf{r}_2$ 와 $\mathbf{v}_2 \propto [-1, 1, 0]^\top = \mathbf{r}_2 - \mathbf{r}_1$ 이 row space 를 생성한다. 셋째 우특이벡터 $\mathbf{v}_3 \propto [-1, -1, 1]^\top$ 은 식 (6) 의 $\mathbf{n}$ 방향이다.

### 6.3 Breaking Conditions

부동소수점 계산에서는 특이값이 정확히 0 이 되지 않으므로, row space 와 null space 의 경계를 문턱값보다 작은 특이값으로 정한다. 문턱값을 크게 잡으면 출력에 작게나마 쓰이는 방향이 null space 로 넘어가고, 작게 잡으면 측정 noise 때문에 null space 에 zero vector 만 남는다. 축마다 단위가 다른 입력은 단위를 바꾸면 내적이 바뀌어 식 (4) 의 분해도 달라지므로, scale 을 맞춘 뒤에 나눈다.

---

## Appendix A. Terminology

- **column space**: 행렬의 열들의 일차결합으로 나타나는 vector 전체.
- **left null space**: $\mathbf{A}^\top\mathbf{y} = \mathbf{0}$ 인 $\mathbf{y}$ 전체. $\mathbb{R}^m$ 의 부분공간이며 column space 와 직교한다.
- **null space**: $\mathbf{A}\mathbf{x} = \mathbf{0}$ 인 $\mathbf{x}$ 전체.
- **pseudo-inverse**: 정사각이 아니거나 역행렬이 없는 행렬에도 정의되며, 해 가운데 norm 이 가장 작은 해를 주는 역행렬의 일반화.
- **row space**: 행렬의 행들의 일차결합으로 나타나는 vector 전체.
- **계수 (rank)**: 행렬에서 서로 일차독립인 열의 최대 개수.
- **기약 행사다리꼴 (reduced row echelon form)**: 기본 행 연산으로 얻는 행렬로, 각 행의 pivot 이 1 이고 pivot 이 있는 열의 다른 원소가 모두 0 인 꼴.
- **부분공간 (subspace)**: Vector space 의 부분집합이면서 그 자체로 vector space 인 집합.
- **우특이벡터 (right singular vector)**: 특이값 분해 $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ 에서 $\mathbf{V}$ 의 열. 입력 공간 $\mathbb{R}^n$ 의 단위 vector 이다.
- **정규직교 기저 (orthonormal basis)**: 서로 직교하는 단위 vector 로 이루어진 기저.
- **직교 (orthogonal)**: 두 vector 의 내적이 0 인 것.
- **직교 여공간 (orthogonal complement)**: 한 부분공간의 모든 vector 와 직교하는 vector 전체.

## Appendix B. Derivation

$\mathrm{Row}(\mathbf{A}) \perp \mathrm{Null}(\mathbf{A})$ 는 row space 의 vector $\mathbf{A}^\top\mathbf{y}$ 와 null space 의 vector $\mathbf{x}$ 의 내적이 0 임을 보이면 된다.

```math
\begin{aligned}
(\mathbf{A}^\top\mathbf{y})^\top \mathbf{x}
&= \mathbf{y}^\top \mathbf{A} \mathbf{x} \\
&= \mathbf{y}^\top \mathbf{0} \\
&= 0
\end{aligned}
\hspace{19em} (8)
```

- 첫째 등호: 전치의 성질 $(\mathbf{A}^\top\mathbf{y})^\top = \mathbf{y}^\top\mathbf{A}$ 이다.
- 둘째 등호: $\mathbf{x}$ 가 null space 에 속하므로 $\mathbf{A}\mathbf{x} = \mathbf{0}$ 이다.
- 셋째 등호: zero vector 와의 내적은 0 이다.

식 (4) 의 분해는 하나뿐이다. 두 분해의 차이는 두 부분공간에 함께 속하므로 식 (8) 에 따라 자기 자신과의 내적이 0 이고, 따라서 zero vector 이다.
