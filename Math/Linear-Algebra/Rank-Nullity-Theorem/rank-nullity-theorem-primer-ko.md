# Rank-Nullity Theorem Primer
Rev. 4 | Created: 2026-11-04 | Updated: 2026-11-07 03:22 UTC

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

| Symbol                                    | Meaning                                                               | Shape                          | Assumption                                               |
| :---------------------------------------: | :-------------------------------------------------------------------: | :----------------------------: | :------------------------------------------------------: |
| $\mathbf{A}$                              | 행렬                                                                  | $m \times n$                   | 실수 원소                                                |
| $m$, $n$                                  | $\mathbf{A}$ 의 행의 수와 열의 수                                     | scalar                         | 양의 정수                                                |
| $\mathbb{R}^n$                            | 원소가 실수 $n$ 개인 열 vector 전체의 집합, $\mathbf{A}$ 의 입력 공간 | vector space                   | 차원 $n$                                                 |
| $\mathbf{x}$                              | 입력 vector                                                           | $n \times 1$                   | 실수 원소                                                |
| $\mathbf{b}$                              | 선형방정식의 우변                                                     | $m \times 1$                   | 실수 원소                                                |
| $\mathbf{0}$                              | zero vector                                                           | $m \times 1$ 또는 $n \times 1$ | 모든 원소가 0                                            |
| $\mathrm{Col}(\mathbf{A})$                | $\mathbf{A}$ 의 column space                                          | $\mathbb{R}^m$ 의 부분공간     | 열들의 일차결합 전체                                     |
| $\mathrm{Row}(\mathbf{A})$                | $\mathbf{A}$ 의 row space                                             | $\mathbb{R}^n$ 의 부분공간     | 행들의 일차결합 전체                                     |
| $\mathrm{Null}(\mathbf{A})$               | $\mathbf{A}$ 의 null space                                            | $\mathbb{R}^n$ 의 부분공간     | $\mathbf{A}\mathbf{x} = \mathbf{0}$ 인 $\mathbf{x}$ 전체 |
| $\mathrm{rank}(\mathbf{A})$               | $\mathbf{A}$ 의 rank                                                  | scalar                         | $0 \le \mathrm{rank}(\mathbf{A}) \le \min(m, n)$         |
| $\mathrm{nullity}(\mathbf{A})$            | $\mathbf{A}$ 의 nullity                                               | scalar                         | $0 \le \mathrm{nullity}(\mathbf{A}) \le n$               |
| $\dim V$                                  | vector space $V$ 의 차원                                              | scalar                         | 0 이상의 정수                                            |
| $\mathbf{R}$                              | $\mathbf{A}$ 의 기약 행사다리꼴                                       | $m \times n$                   | 기본 행 연산으로 얻음                                    |
| $n_{\mathrm{pivot}}$, $n_{\mathrm{free}}$ | $\mathbf{R}$ 의 pivot column 개수와 free column 개수                  | scalar                         | $n_{\mathrm{pivot}} + n_{\mathrm{free}} = n$             |
| $\mathbf{v}$                              | Worked Example 에서 null space 를 생성하는 vector                     | $3 \times 1$                   | zero vector 가 아님                                      |

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

$\mathrm{nullity}(\mathbf{A})$ 는 $\mathbf{A}$ 를 곱했을 때 zero vector 가 되는 입력 전체의 차원이다. Null space 에 든 vector 는 $\mathbf{A}\mathbf{x}$ 에 아무것도 더하지 않으므로, nullity 가 1 이상이면 서로 다른 입력이 같은 출력을 갖는다. 식 (1) 에 따라 nullity 는 $n - \mathrm{rank}(\mathbf{A})$ 이며, 행의 수 $m$ 은 nullity 를 직접 정하지 않고 rank 의 상한 $\min(m, n)$ 을 통해서만 영향을 준다.

### 5.3 Split of the Input Space

$\mathrm{Row}(\mathbf{A})$ 의 차원은 $\mathrm{Col}(\mathbf{A})$ 의 차원과 같은 $\mathrm{rank}(\mathbf{A})$ 이고, row space 와 null space 는 서로 직교하는 $\mathbb{R}^n$ 의 부분공간이다. 식 (1) 은 입력 공간 $\mathbb{R}^n$ 의 차원 $n$ 이 row space 의 차원과 null space 의 차원으로 나뉜다는 식이기도 하다. 입력 vector 하나를 두 성분으로 나누는 계산은 [Null Space Primer](../Null-Space/null-space-primer-ko.md) 의 5.5 Orthogonality to the Row Space 에 있다.

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

식 (3) 에서 pivot column 은 첫째와 둘째 열이어서 $\mathrm{rank}(\mathbf{A}) = 2$ 이고, free column 은 셋째 열 하나이다. 식 (1) 에 넣으면 null space 를 구하지 않고도 $\mathrm{nullity}(\mathbf{A}) = 3 - 2 = 1$ 이다.

Null space 를 직접 구해 이 값을 확인한다. $\mathbf{x}$ 의 셋째 원소를 1 로 두면 식 (3) 의 첫 행에서 첫째 원소가 0, 둘째 행에서 둘째 원소가 $-1.5$ 이다.

```math
\mathbf{v} = \begin{bmatrix} 0 \\ -1.5 \\ 1 \end{bmatrix}, \qquad
\mathbf{A}\mathbf{v} = \begin{bmatrix} 0 - 3 + 3 \\ 0 - 9 + 9 \end{bmatrix} = \begin{bmatrix} 0 \\ 0 \end{bmatrix}
\hspace{19em} (4)
```

Null space 는 $\mathbf{v}$ 의 상수배 전체인 $\mathbb{R}^3$ 안의 직선이고, 차원은 1 이어서 식 (1) 로 구한 값과 같다. 세 값을 모으면 아래 Table 2 와 같다.

Table 2. Rank and nullity of the matrix in equation (2)

| Quantity                       | Space                                     | Value |
| :----------------------------: | :---------------------------------------: | :---: |
| $n$                            | $\mathbb{R}^3$                            | 3     |
| $\mathrm{rank}(\mathbf{A})$    | $\mathrm{Col}(\mathbf{A}) = \mathbb{R}^2$ | 2     |
| $\mathrm{nullity}(\mathbf{A})$ | $\mathbf{v}$ 의 상수배 전체인 직선        | 1     |

$\mathbf{A}$ 는 입력 공간 $\mathbb{R}^3$ 가운데 row space 성분 2 차원을 $\mathbb{R}^2$ 전체로 옮기고, $\mathbf{v}$ 방향의 직선 1 차원을 원점으로 보낸다. 같은 행렬의 네 부분공간과 그 기저는 [Dimension and Rank Primer](../Dimension-and-Rank/dimension-and-rank-primer-ko.md) 의 Appendix B 에 있다.

## 7. Application

식 (1) 은 null space 를 구하지 않고 rank 만으로 선형방정식의 해의 개수를 가리는 데 쓴다.

### 7.1 Number of Solutions

$\mathbf{A}\mathbf{x} = \mathbf{b}$ 에 해가 하나라도 있으면, 해 전체는 그 해에 null space 의 vector 를 더한 것이다. 그래서 nullity 로 해의 개수가 갈린다.

- $\mathrm{nullity}(\mathbf{A}) = 0$, 곧 $\mathrm{rank}(\mathbf{A}) = n$: null space 에 zero vector 만 있어 해가 있으면 하나뿐이다.
- $\mathrm{nullity}(\mathbf{A}) \ge 1$, 곧 $\mathrm{rank}(\mathbf{A}) \lt n$: 해가 있으면 무수히 많다.
- $m \lt n$: $\mathrm{rank}(\mathbf{A}) \le m \lt n$ 이어서 nullity 가 $n - m$ 이상이고, 해가 있으면 언제나 무수히 많다. 식 (2) 의 $\mathbf{A}$ 가 이 경우이다.

해가 있는지는 nullity 가 아니라 첨가 행렬 $[\mathbf{A} \mid \mathbf{b}]$ 의 rank 로 가리며, 그 절차는 [Types of Linear Systems](../Linear-System-Types/linear-system-types-ko.md) 와 [Null Space Primer](../Null-Space/null-space-primer-ko.md) 의 Appendix C 에 있다.

---

## Appendix A. Terminology

- **column space**: 행렬의 열들의 일차결합 전체. $\mathbb{R}^m$ 의 부분공간이다.
- **free column**: 기약 행사다리꼴에서 pivot 이 없는 열. 그 열의 변수 (free variable) 는 아무 값이나 가질 수 있고, pivot column 의 변수 (pivot 변수) 는 free variable 의 값으로 정해진다.
- **null space**: $\mathbf{A}\mathbf{x} = \mathbf{0}$ 인 $\mathbf{x}$ 전체. $\mathbb{R}^n$ 의 부분공간이다.
- **pivot column**: 기약 행사다리꼴에서 각 행의 첫 0 이 아닌 원소 (pivot) 가 있는 열.
- **row space**: 행렬의 행들의 일차결합 전체. $\mathbb{R}^n$ 의 부분공간이다.
- **vector space**: zero vector 를 포함하고, 원소끼리의 덧셈과 상수배에 대해 닫혀 있는 vector 집합.
- **가역 행렬 (invertible matrix)**: 역행렬이 있는 정사각 행렬. 왼쪽에 곱해도 vector 가 zero vector 인지 아닌지가 바뀌지 않는다.
- **기본 행 연산 (elementary row operation)**: 두 행을 맞바꾸기, 한 행에 0 이 아닌 수를 곱하기, 한 행에 다른 행의 상수배를 더하기의 세 연산.
- **기약 행사다리꼴 (reduced row echelon form)**: 기본 행 연산으로 얻는 행렬로, 각 행의 pivot 이 1 이고 pivot 이 있는 열의 다른 원소가 모두 0 이며, 아래 행의 pivot 이 위 행의 pivot 보다 오른쪽에 있는 꼴.
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

- 첫째 등호: $\mathbf{R}$ 의 열은 pivot column 과 free column 의 두 가지로만 나뉜다.
  - 기약 행사다리꼴의 각 열은 어떤 행의 pivot 을 담거나 담지 않거나 둘 중 하나이다. Pivot 을 담으면 pivot column 이고, 담지 않으면 free column 이다.
  - 한 열이 두 가지에 함께 들 수 없고 어느 쪽에도 들지 않을 수도 없으므로, 두 개수를 더하면 열의 수 $n$ 이다.
  - 예를 들어 $\mathbf{R} = [[1, 0, 0], [0, 1, 1.5]]$ 에서는 첫째와 둘째 열이 pivot column, 셋째 열이 free column 이어서 $2 + 1 = 3$ 이다.
- 둘째 등호: pivot column 의 개수 $n_{\mathrm{pivot}}$ 은 $\mathbf{A}$ 의 rank 와 같다.
  - 기본 행 연산 하나는 가역 행렬 하나를 왼쪽에 곱하는 것과 같으므로, $\mathbf{R}$ 은 가역 행렬을 $\mathbf{A}$ 의 왼쪽에 곱한 행렬이다.
  - $\mathbf{A}$ 의 열들의 일차결합이 zero vector 이면, 같은 계수로 만든 $\mathbf{R}$ 의 열들의 일차결합도 zero vector 이다. 가역 행렬을 곱한 결과가 zero vector 인 것은 곱하기 전의 vector 가 zero vector 일 때뿐이므로 반대 방향도 성립한다. 그래서 $\mathbf{A}$ 와 $\mathbf{R}$ 은 같은 번호의 열 사이에 같은 일차독립 관계를 갖는다.
  - $\mathbf{R}$ 의 pivot column 은 pivot 원소만 1 이고 나머지가 0 이며, pivot 이 놓인 행이 열마다 달라 서로 일차독립이다.
  - $\mathbf{R}$ 의 free column 은 0 이 아닌 원소가 pivot 이 있는 행에만 있으므로, 그 원소를 계수로 한 pivot column 의 일차결합이다. 위 예의 셋째 열 $[0, 1.5]^\top$ 은 둘째 열 $[0, 1]^\top$ 의 1.5 배이다.
  - 따라서 $\mathbf{R}$ 에서 서로 일차독립인 열의 최대 개수는 $n_{\mathrm{pivot}}$ 이고, 일차독립 관계가 같은 $\mathbf{A}$ 에서도 그 개수, 곧 $\mathrm{rank}(\mathbf{A})$ 가 $n_{\mathrm{pivot}}$ 이다.
- 셋째 등호: free column 의 개수 $n_{\mathrm{free}}$ 는 nullity 와 같다.
  - $\mathbf{R}$ 은 가역 행렬을 $\mathbf{A}$ 의 왼쪽에 곱한 행렬이므로, $\mathbf{A}\mathbf{x} = \mathbf{0}$ 과 $\mathbf{R}\mathbf{x} = \mathbf{0}$ 은 해가 같고 두 행렬의 null space 도 같다.
  - $\mathbf{R}\mathbf{x} = \mathbf{0}$ 의 0 이 아닌 행 하나에는 pivot 변수가 하나만 계수 1 로 들어 있고, 나머지 항은 free variable 이다. 그래서 free variable 의 값을 정하면 pivot 변수가 하나씩 정해진다.
  - free column 마다 그 free variable 만 1, 나머지 free variable 은 0 으로 두고 pivot 변수를 구하면 해가 하나 나온다. 위 예에서는 셋째 원소를 1 로 두어 $[0, -1.5, 1]^\top$ 을 얻는다.
  - 이렇게 얻은 해들은 서로 일차독립이다. 각 해는 자기 free variable 의 원소만 1 이고 다른 free variable 의 원소는 0 이므로, 일차결합이 zero vector 이려면 그 원소들에서 계수가 모두 0 이어야 한다.
  - Null space 의 vector 하나를 잡아 그 vector 의 free variable 값을 계수로 이 해들을 일차결합하면, free variable 원소가 같고 pivot 변수는 free variable 로 정해지므로 처음 vector 와 같아진다. 그래서 이 해들은 null space 의 모든 vector 를 만든다.
  - 서로 일차독립이면서 null space 의 모든 vector 를 만드는 이 해들이 null space 의 기저이므로, 그 개수 $n_{\mathrm{free}}$ 가 nullity 이다.

따라서 식 (1) 이 성립한다.

```math
\mathrm{rank}(\mathbf{A}) + \mathrm{nullity}(\mathbf{A}) = n
\hspace{19em} (1)
```

## Appendix C. Intuitive Meaning

식 (1) 은 입력 공간의 차원이 어디로 가는지를 세는 "에너지 (정보) 보존 법칙" 처럼 읽을 수 있다.

```math
\mathrm{rank}(\mathbf{A}) + \mathrm{nullity}(\mathbf{A}) = n
\hspace{19em} (1)
```

입력 공간 $\mathbb{R}^n$ 의 모든 vector 는 $\mathbf{A}$ 를 곱하는 선형 변환 $\mathbf{x} \mapsto \mathbf{A}\mathbf{x}$ 를 지나면서 두 가지 운명 가운데 하나를 맞는다.

1. 변환을 견디고 살아남는 성분: 출력 공간으로 전달되어 column space 를 이룬다. 이 성분이 차지하는 차원이 $\mathrm{rank}(\mathbf{A})$ 이다.
2. 0 으로 찌그러져 사라지는 성분: 완전히 zero vector 가 되어 null space 로 들어간다. 이 성분이 차지하는 차원이 $\mathrm{nullity}(\mathbf{A})$ 이다.

그래서 살아남은 차원 (rank) 과 0 으로 사라진 차원 (nullity) 을 더하면 원래 입력 공간의 차원 $n$ 이 된다. 어떤 차원도 두 몫 밖으로 새어 나가지 않는다는 점에서 보존 법칙과 닮았다.

$\mathbf{A} = [[1, 2, 3], [4, 6, 9]]$ 이면 입력 공간 $\mathbb{R}^3$ 의 3 차원 가운데 2 차원 분량은 살아남아 출력 공간 $\mathbb{R}^2$ 전체를 채우고, 나머지 1 차원 분량인 $[0, -1.5, 1]^\top$ 방향의 직선은 0 으로 완전히 압축되어 사라진다. 계산해 보지 않아도 $2 + \mathrm{nullity}(\mathbf{A}) = 3$ 이므로 사라지는 차원은 반드시 1 이다.
