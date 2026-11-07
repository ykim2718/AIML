# Rank-Nullity Theorem Primer
Rev. 25 | Created: 2026-11-04 | Updated: 2026-11-07 08:55 UTC

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Notation](#3-notation)
- [4. Definitions](#4-definitions)
- [5. Interpretation](#5-interpretation)
  - [5.1 Rank Term](#51-rank-term)
  - [5.2 Nullity Term](#52-nullity-term)
  - [5.3 Split of the Input Space](#53-split-of-the-input-space)
- [6. Worked Example](#6-worked-example)
- [7. Application](#7-application)
  - [7.1 Number of Solutions](#71-number-of-solutions)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Derivation](#appendix-b-derivation)
- [Appendix C. Intuitive Meaning](#appendix-c-intuitive-meaning)

## 1. Purpose

- **Problem Statement**: 선형방정식 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 의 해가 무수히 많은지를 가리려면 null space 의 차원이 필요한데, null space 를 직접 구하지 않고는 그 차원을 알기 어렵다.
- **Goal**: 독자가 행렬 $\mathbf{A}$ 의 rank 와 열의 수 $n$ 만으로 nullity 를 구하고, 그 값으로 해가 하나인지 무수히 많은지를 가릴 수 있게 한다.
- **Non-Goal**: 무한 차원 vector space 와 선형 변환 일반에 대한 정리, 그리고 부동소수점 계산에서 rank 를 정하는 문턱값은 다루지 않는다.

## 2. Summary

Rank-nullity 정리 (rank-nullity theorem) 는 $m \times n$ 행렬 $\mathbf{A}$ 의 rank 와 nullity 를 더하면 열의 수 $n$ 이 된다는 정리이다.
Rank 는 column space 의 차원이고 nullity 는 null space 의 차원이며, 행의 수 $m$ 은 식에 들지 않는다.
그래서 rank 만 알면 null space 를 구하지 않고도 nullity 가 $n - \mathrm{rank}(\mathbf{A})$ 로 정해진다.

## 3. Notation

이 문서가 쓰는 기호는 아래 Table 1 이다.

Table 1. Symbols

| Symbol                                    | Meaning                                                                   | Shape                          | Assumption                                               |
| :---------------------------------------: | :-----------------------------------------------------------------------: | :----------------------------: | :------------------------------------------------------: |
| $\mathbf{A}$                              | 행렬                                                                      | $m \times n$                   | 실수 원소                                                |
| $m$, $n$                                  | $\mathbf{A}$ 의 행의 수와 열의 수                                         | scalar                         | 양의 정수                                                |
| $\mathbb{R}^n$                            | 원소가 실수 $n$ 개인 열 vector 전체의 집합, $\mathbf{A}$ 의 입력 공간     | vector space                   | 차원 $n$                                                 |
| $\mathbf{x}$                              | 입력 vector                                                               | $n \times 1$                   | 실수 원소                                                |
| $x_1$, $x_2$, $x_3$                       | Worked Example 에서 $\mathbf{x}$ 의 첫째, 둘째, 셋째 원소                 | scalar                         | 실수                                                     |
| $\mathbf{b}$                              | 선형방정식의 우변                                                         | $m \times 1$                   | 실수 원소                                                |
| $\mathbf{0}$                              | zero vector                                                               | $m \times 1$ 또는 $n \times 1$ | 모든 원소가 0                                            |
| $\mathrm{Col}(\mathbf{A})$                | $\mathbf{A}$ 의 column space                                              | $\mathbb{R}^m$ 의 부분공간     | 열들의 일차결합 전체                                     |
| $\mathrm{Row}(\mathbf{A})$                | $\mathbf{A}$ 의 row space                                                 | $\mathbb{R}^n$ 의 부분공간     | 행들의 일차결합 전체                                     |
| $\mathrm{Null}(\mathbf{A})$               | $\mathbf{A}$ 의 null space                                                | $\mathbb{R}^n$ 의 부분공간     | $\mathbf{A}\mathbf{x} = \mathbf{0}$ 인 $\mathbf{x}$ 전체 |
| $\mathrm{rank}(\mathbf{A})$               | $\mathbf{A}$ 의 rank                                                      | scalar                         | $0 \le \mathrm{rank}(\mathbf{A}) \le \min(m, n)$         |
| $\mathrm{nullity}(\mathbf{A})$            | $\mathbf{A}$ 의 nullity                                                   | scalar                         | $0 \le \mathrm{nullity}(\mathbf{A}) \le n$               |
| $\dim V$                                  | vector space $V$ 의 차원                                                  | scalar                         | 0 이상의 정수                                            |
| $\mathbf{R}$                              | $\mathbf{A}$ 의 기약 행사다리꼴                                           | $m \times n$                   | 기본 행 연산으로 얻음                                    |
| $n_{\mathrm{pivot}}$, $n_{\mathrm{free}}$ | $\mathbf{R}$ 의 pivot column 개수와 free column 개수                      | scalar                         | $n_{\mathrm{pivot}} + n_{\mathrm{free}} = n$             |
| $\mathbf{v}$                              | null space 의 vector, Worked Example 에서는 null space 를 생성하는 vector | $n \times 1$                   | zero vector 가 아님                                      |
| $c$                                       | 임의의 상수                                                               | scalar                         | 실수                                                     |

특수 기호는 아래처럼 읽는다.

- $\mathbb{R}^n$: 영어로 "R n" 또는 "real n-space", 한국어로 "n 차원 실수 공간" 이라 읽는다.
- $\dim V$: 영어로 "the dimension of V", 한국어로 "V 의 차원" 이라 읽는다.
- $\{ \mathbf{x} \in \mathbb{R}^n \mid \mathbf{A}\mathbf{x} = \mathbf{0} \}$ 의 $\mid$: 영어로 "such that", 한국어로 "인" 이라 읽는다.
- $[0, -1.5, 1]^\top$ 의 $\top$: 영어로 "transpose", 한국어로 "전치" 라 읽는다. 가로로 적은 원소를 열 vector 로 바꾼다는 뜻이다.

## 4. Definitions

> **Definition: Rank**<br>
> $m \times n$ 행렬 $\mathbf{A}$ 의 rank $\mathrm{rank}(\mathbf{A})$ 는 column space $\mathrm{Col}(\mathbf{A})$ 의 차원, 곧 $\mathbf{A}$ 의 열 가운데 서로 일차독립인 열의 최대 개수이다.

> **Definition: Nullity**<br>
> $m \times n$ 행렬 $\mathbf{A}$ 의 nullity $\mathrm{nullity}(\mathbf{A})$ 는 null space $\mathrm{Null}(\mathbf{A}) = \{ \mathbf{x} \in \mathbb{R}^n \mid \mathbf{A}\mathbf{x} = \mathbf{0} \}$ 의 차원, 곧 $\mathrm{Null}(\mathbf{A})$ 안에서 서로 일차독립인 vector 의 최대 개수이다.

> **Definition: Rank-nullity theorem**<br>
> 실수 원소를 갖는 모든 $m \times n$ 행렬 $\mathbf{A}$ 에서 rank 와 nullity 를 더한 값은 열의 수 $n$ 과 같다. Rank-nullity 정리는 dimension theorem 이라고도 부른다.

Column space 는 출력 공간 $\mathbb{R}^m$ 의 부분공간이고 null space 는 입력 공간 $\mathbb{R}^n$ 의 부분공간이어서, 두 차원은 서로 다른 공간에서 잰 값이다. 차원과 rank 의 구별은 [Dimension and Rank Primer](../Dimension-and-Rank/dimension-and-rank-primer-ko.md) 에 있다.

## 5. Interpretation

Rank-nullity 정리는 실수 원소를 갖는 모든 $m \times n$ 행렬 $\mathbf{A}$ 에서 성립한다.

```math
\mathrm{rank}(\mathbf{A}) + \mathrm{nullity}(\mathbf{A}) = n
\hspace{19em} (1)
```

식 (1) 의 오른쪽 변은 열의 수 $n$, 곧 입력 공간 $\mathbb{R}^n$ 의 차원이다. 유도는 [Appendix B](#appendix-b-derivation) 에, 비유로 풀어 쓴 직관적 의미는 [Appendix C](#appendix-c-intuitive-meaning) 에 있다.

### 5.1 Rank Term

$\mathrm{rank}(\mathbf{A})$ 는 입력 $\mathbf{x}$ 를 $\mathbf{A}\mathbf{x}$ 로 옮겼을 때 결과가 놓이는 column space 의 차원이다. $\mathbf{A}\mathbf{x}$ 는 $\mathbf{A}$ 의 열을 $\mathbf{x}$ 의 원소로 일차결합한 것이므로, 모든 $\mathbf{x}$ 에 대한 결과는 열이 생성하는 column space 를 넘지 않는다. Rank 는 $\mathbb{R}^m$ 안의 차원이므로 $m$ 을 넘지 않고, 열이 $n$ 개이므로 $n$ 도 넘지 않는다.

### 5.2 Nullity Term

$\mathrm{nullity}(\mathbf{A})$ 는 $\mathbf{A}$ 를 곱했을 때 zero vector 가 되는 입력 전체의 차원이다. Nullity 가 1 이상이면 $\mathbf{A}\mathbf{v} = \mathbf{0}$ 을 만족하고 zero vector 가 아닌 null space 의 vector $\mathbf{v}$ 가 있고, 입력 $\mathbf{x}$ 에 $\mathbf{v}$ 를 더하면 $\mathbf{A}(\mathbf{x} + \mathbf{v}) = \mathbf{A}\mathbf{x} + \mathbf{A}\mathbf{v} = \mathbf{A}\mathbf{x} + \mathbf{0} = \mathbf{A}\mathbf{x}$ 이어서 출력이 바뀌지 않는다. 식 (1) 에 따라 nullity 는 $n - \mathrm{rank}(\mathbf{A})$ 이며, 행의 수 $m$ 은 nullity 를 직접 정하지 않고 rank 의 상한 $\min(m, n)$ 을 통해서만 영향을 준다. $\mathrm{rank}(\mathbf{A}) \le \min(m, n)$ 이므로 $\mathrm{nullity}(\mathbf{A}) = n - \mathrm{rank}(\mathbf{A}) \ge n - \min(m, n)$ 이다.

### 5.3 Split of the Input Space

$\mathrm{Row}(\mathbf{A})$ 의 차원은 $\mathrm{Col}(\mathbf{A})$ 의 차원과 같은 $\mathrm{rank}(\mathbf{A})$ 이며, 두 차원이 같은 까닭은 [Dimension and Rank Primer](../Dimension-and-Rank/dimension-and-rank-primer-ko.md) 의 5.2 Row Rank and Column Rank 에 있다. $\mathbf{A}\mathbf{x} = \mathbf{0}$ 의 각 행은 $\mathbf{A}$ 의 한 행과 $\mathbf{x}$ 의 내적이 0 이라는 식이므로, null space 의 vector 는 $\mathbf{A}$ 의 모든 행과, 따라서 행들의 일차결합인 row space 의 모든 vector 와 직교한다. 식 (1) 은 입력 공간 $\mathbb{R}^n$ 의 차원 $n$ 이 row space 의 차원과 null space 의 차원으로 나뉜다는 식이기도 하다. 입력 vector 하나를 두 성분으로 나누는 계산은 [Null Space Primer](../Null-Space/null-space-primer-ko.md) 의 5.5 Orthogonality to the Row Space 에 있다.

## 6. Worked Example

행이 두 개이고 열이 세 개인 행렬을 둔다 ($m = 2$, $n = 3$).

```math
\mathbf{A} = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 6 & 9 \end{bmatrix}
\hspace{19em} (2)
```

$\mathbf{A}$ 의 둘째 행에서 첫째 행의 4 배를 빼면 $[0, -2, -3]$, 이를 $-2$ 로 나누면 $[0, 1, 1.5]$ 이고, 첫째 행에서 이 행의 2 배를 빼면 $[1, 0, 0]$ 이다.

```math
\mathbf{R} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 1.5 \end{bmatrix}
\hspace{19em} (3)
```

식 (3) 에서 pivot column 은 첫째와 둘째 열이고, pivot column 의 개수가 rank 이므로 ([Appendix B](#appendix-b-derivation) 의 둘째 등호) $\mathrm{rank}(\mathbf{A}) = 2$ 이다. Rank 가 행의 수 $m = 2$ 와 같으므로 column space 는 $\mathbb{R}^2$ 전체이다. Free column 은 셋째 열 하나이다. 식 (1) 에 넣으면 null space 를 구하지 않고도 $\mathrm{nullity}(\mathbf{A}) = 3 - 2 = 1$ 이다.

Null space 를 직접 구해 이 값을 확인한다. Null space 는 $\mathbf{A}\mathbf{x} = \mathbf{0}$ 의 해이고, 기본 행 연산은 해를 바꾸지 않으므로 식 (3) 의 $\mathbf{R}$ 로 $\mathbf{R}\mathbf{x} = \mathbf{0}$ 을 푼다. $\mathbf{R}$ 의 두 행은 $x_1 = 0$ 과 $x_2 + 1.5\,x_3 = 0$ 이므로, free variable $x_3$ 를 1 로 두면 $x_1 = 0$, $x_2 = -1.5$ 이다.

```math
\mathbf{v} = \begin{bmatrix} 0 \\ -1.5 \\ 1 \end{bmatrix}, \qquad
\mathbf{A}\mathbf{v} = \begin{bmatrix} 0 - 3 + 3 \\ 0 - 9 + 9 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
\hspace{19em} (4)
```

Free variable $x_3$ 를 1 대신 임의의 실수 $c$ 로 두면 해는 $c\,\mathbf{v}$ 이므로, null space 는 $\mathbf{v}$ 의 상수배 전체인 $\mathbb{R}^3$ 안의 직선이고, 차원은 1 이어서 식 (1) 로 구한 값과 같다.

$n$, $\mathrm{rank}(\mathbf{A})$, $\mathrm{nullity}(\mathbf{A})$ 의 세 값을 모으면 아래 Table 2 와 같다.

Table 2. Rank and nullity of the matrix in equation (2)

| Quantity                       | Space                                     | Value |
| :----------------------------: | :---------------------------------------: | :---: |
| $n$                            | $\mathbb{R}^3$                            | 3     |
| $\mathrm{rank}(\mathbf{A})$    | $\mathrm{Col}(\mathbf{A}) = \mathbb{R}^2$ | 2     |
| $\mathrm{nullity}(\mathbf{A})$ | $\mathbf{v}$ 의 상수배 전체인 직선        | 1     |

입력에 $\mathbf{v}$ 를 더해도 출력이 바뀌지 않는 것 ([5.2](#52-nullity-term)) 도 이 $\mathbf{A}$ 로 확인한다. $\mathbf{x} = [1, 0, 0]^\top$ 일 때 $\mathbf{A}\mathbf{x} = [1, 4]^\top$ 이고, $\mathbf{x} + \mathbf{v} = [1, -1.5, 1]^\top$ 일 때도 $\mathbf{A}(\mathbf{x} + \mathbf{v}) = [1 - 3 + 3, 4 - 9 + 9]^\top = [1, 4]^\top$ 이다.

$\mathbf{A}$ 는 입력 공간 $\mathbb{R}^3$ 가운데 row space 성분 2 차원을 $\mathbb{R}^2$ 전체로 옮기고, $\mathbf{v}$ 방향의 직선 1 차원을 원점으로 보낸다. 같은 행렬의 네 부분공간과 그 기저는 [Dimension and Rank Primer](../Dimension-and-Rank/dimension-and-rank-primer-ko.md) 의 Appendix B 에 있다.

## 7. Application

식 (1) 은 null space 를 구하지 않고 rank 만으로 선형방정식의 해의 개수를 가리는 데 쓴다.

### 7.1 Number of Solutions

$\mathbf{A}\mathbf{x} = \mathbf{b}$ 에 해가 하나라도 있으면, 해 전체는 그 해에 null space 의 vector 를 더한 것이다. 두 해를 빼면 $\mathbf{A}$ 를 곱한 값이 $\mathbf{b} - \mathbf{b} = \mathbf{0}$ 이어서 그 차이가 null space 에 속하고, 거꾸로 해에 null space 의 vector 를 더해도 $\mathbf{A}$ 를 곱한 값은 그대로 $\mathbf{b}$ 이다. 그래서 nullity 로 해의 개수가 갈린다.

- $\mathrm{nullity}(\mathbf{A}) = 0$, 곧 $\mathrm{rank}(\mathbf{A}) = n$: null space 에 zero vector 만 있어 해가 있으면 하나뿐이다.
- $\mathrm{nullity}(\mathbf{A}) \ge 1$, 곧 $\mathrm{rank}(\mathbf{A}) \lt n$: 해가 있으면 무수히 많다.
- $m \lt n$: $\mathrm{rank}(\mathbf{A}) \le m \lt n$ 이어서 nullity 가 $n - m$ 이상이고, 해가 있으면 언제나 무수히 많다. 식 (2) 의 $\mathbf{A}$ 가 이 경우이다.

해가 있는지는 nullity 가 아니라 첨가 행렬 $[\mathbf{A} \mid \mathbf{b}]$ 의 rank 로 가리며, 그 절차는 [Types of Linear Systems](../Linear-System-Types/linear-system-types-ko.md) 와 [Null Space Primer](../Null-Space/null-space-primer-ko.md) 의 Appendix C 에 있다.

---

## Appendix A. Terminology

- **column space**: 행렬의 열들의 일차결합 전체. $\mathbb{R}^m$ 의 부분공간이다.
- **free column**: 기약 행사다리꼴에서 pivot 이 없는 열. 그 열의 변수 (free variable) 는 아무 값이나 가질 수 있고, pivot column 의 변수 (pivot 변수) 는 free variable 의 값으로 정해진다.<br>
  예: Worked Example 의 $\mathbf{A}$ 를 기약 행사다리꼴로 바꾼 아래 행렬에서는 pivot 이 없는 셋째 열 $[0, 1.5]^\top$ 이 free column 이고, 그 열의 변수가 free variable 이다.

```math
\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 1.5 \end{bmatrix}
```

- **pivot column**: 기약 행사다리꼴에서 각 행을 왼쪽부터 읽어 처음 나오는 0 이 아닌 원소 (pivot) 가 있는 열.<br>
  예: 같은 행렬에서는 첫 행의 pivot 1 이 있는 첫째 열과 둘째 행의 pivot 1 이 있는 둘째 열이 pivot column 이다.

```math
\begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 1.5 \end{bmatrix}
```

- **vector space**: zero vector 를 포함하고, 원소끼리의 덧셈과 상수배에 대해 닫혀 있는 vector 집합.
- **기본 행 연산 (elementary row operation)**: 두 행을 맞바꾸기, 한 행에 0 이 아닌 수를 곱하기, 한 행에 다른 행의 상수배를 더하기의 세 연산.
- **기약 행사다리꼴 (reduced row echelon form)**: 기본 행 연산으로 얻는 행렬로, 각 행의 pivot 이 1 이고 pivot 이 있는 열의 다른 원소가 모두 0 이며, 아래 행의 pivot 이 위 행의 pivot 보다 오른쪽에 있는 꼴.<br>
  예: Worked Example 의 $\mathbf{A}$ 는 기본 행 연산으로 아래 오른쪽 행렬이 된다. 두 행의 pivot 1 이 첫째와 둘째 열에 있고, 그 두 열의 다른 원소는 0 이다.

```math
\begin{bmatrix} 1 & 2 & 3 \\ 4 & 6 & 9 \end{bmatrix} \;\longrightarrow\; \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 1.5 \end{bmatrix}
```

- **기저 (basis)**: Vector space 의 vector 가운데 서로 일차독립이면서, 일차결합으로 그 공간의 모든 vector 를 만드는 vector 의 집합.
- **부분공간 (subspace)**: Vector space 의 부분집합이면서 그 자체로 vector space 인 집합.
- **일차결합 (linear combination)**: vector 들에 각각 상수를 곱해 더한 것.
- **일차독립 (linear independence)**: 어느 vector 도 나머지 vector 들의 일차결합으로 적히지 않는 것.
- **직교 (orthogonal)**: 두 vector 의 내적이 0 인 것.
- **차원 (dimension)**: Vector space 의 기저에 든 vector 의 개수.
- **첨가 행렬 (augmented matrix)**: $\mathbf{A}$ 의 오른쪽에 $\mathbf{b}$ 를 열 하나로 붙인 행렬 $[\mathbf{A} \mid \mathbf{b}]$.

## Appendix B. Derivation

$\mathbf{A}$ 를 기본 행 연산으로 기약 행사다리꼴 $\mathbf{R}$ 로 바꾸고, $\mathbf{R}$ 의 열을 pivot column 과 free column 으로 나누어 센다.

```math
\begin{aligned}
n &= n_{\mathrm{pivot}} + n_{\mathrm{free}} \\
  &= \mathrm{rank}(\mathbf{A}) + n_{\mathrm{free}} \\
  &= \mathrm{rank}(\mathbf{A}) + \mathrm{nullity}(\mathbf{A})
\end{aligned}
\hspace{19em} (5)
```

- 첫째 등호: $\mathbf{R}$ 의 각 열은 pivot 이 있으면 pivot column, 없으면 free column 이므로 두 개수를 더하면 $n$ 이다. $\mathbf{R} = [[1, 0, 0], [0, 1, 1.5]]$ 에서는 $2 + 1 = 3$ 이다.
- 둘째 등호: 기본 행 연산은 열 사이의 일차독립 관계를 바꾸지 않는다. $\mathbf{R}$ 에서 pivot column 은 서로 일차독립이고 free column 은 pivot column 의 일차결합이므로 (위 예의 셋째 열은 둘째 열의 1.5 배), 일차독립인 열의 최대 개수인 rank 는 $n_{\mathrm{pivot}}$ 이다.
- 셋째 등호: $\mathbf{A}\mathbf{x} = \mathbf{0}$ 과 $\mathbf{R}\mathbf{x} = \mathbf{0}$ 은 해가 같다. Free variable 하나만 1, 나머지는 0 으로 둔 해가 free column 마다 하나씩 나오고 (위 예에서는 $[0, -1.5, 1]^\top$), 이 해들은 서로 일차독립이면서 null space 의 모든 vector 를 만드는 기저이므로 그 개수 $n_{\mathrm{free}}$ 가 nullity 이다.

따라서 식 (1) 이 성립한다.

```math
\mathrm{rank}(\mathbf{A}) + \mathrm{nullity}(\mathbf{A}) = n
\hspace{19em} (1)
```

## Appendix C. Intuitive Meaning

**직관적인 의미**: "에너지(정보) 보존 법칙"
입력 공간 $\mathbb{R}^n$에 존재하는 모든 벡터들은 선형 변환 $T(x) = Ax$를 통과하면서 단 두 가지 중 하나의 운명을 맞이하게 됩니다.

1. 변환을 견디고 살아남는 성분: 결과물(출력 공간)로 전달되어 열공간(Column Space)을 형성합니다. 이 차원의 크기를 $\text{Rank}(A)$라고 부릅니다.
2. 0으로 찌그러져 소멸하는 성분: 완전히 무(Zero)로 사라져 영공간(Null Space)으로 들어갑니다. 이 차원의 크기를 $\text{Nullity}(A)$라고 부릅니다.

따라서 "살아남은 차원($\text{Rank}$)과 0으로 사라진 차원($\text{Nullity}$)을 더하면, 원래 입력 공간의 전체 차원($n$)이 된다"는 지극히 당연하고도 강력한 보존 법칙 (식 (1)) 을 나타냅니다.

```math
\mathrm{rank}(\mathbf{A}) + \mathrm{nullity}(\mathbf{A}) = n
\hspace{19em} (1)
```
