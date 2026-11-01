# Outer Product Expansion Primer
Rev. 0 | Created: 2026-11-01 | Updated: 2026-11-01 10:04 UTC

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Notation](#3-notation)
- [4. Interpretation](#4-interpretation)
  - [4.1 Inner Product Form](#41-inner-product-form)
  - [4.2 Outer Product Form](#42-outer-product-form)
  - [4.3 Rank of Each Term](#43-rank-of-each-term)
- [5. Worked Example](#5-worked-example)
- [6. Application](#6-application)
  - [6.1 Singular Value Decomposition](#61-singular-value-decomposition)
  - [6.2 Gram Matrix of Samples](#62-gram-matrix-of-samples)
  - [6.3 Breaking Conditions](#63-breaking-conditions)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Derivation](#appendix-b-derivation)

## 1. Purpose

- **Problem Statement**: 행렬 곱 $\mathbf{A}\mathbf{B}$ 를 원소마다 행과 열의 내적으로만 계산하면, 곱 전체가 계수 (rank) 1 인 행렬 몇 개의 합인지가 드러나지 않는다.
- **Goal**: 독자가 행렬 곱을 외적 (outer product) 의 합으로 펼쳐 쓰고, 그 합에서 특이값 분해와 sample 별 누적 계산을 읽어 낼 수 있게 한다.
- **Non-Goal**: 행렬 곱의 계산 속도를 높이는 algorithm 과 block 행렬 곱은 다루지 않는다.

## 2. Summary

$m \times n$ 행렬 $\mathbf{A}$ 와 $n \times p$ 행렬 $\mathbf{B}$ 의 곱은, $\mathbf{A}$ 의 $k$ 번째 열과 $\mathbf{B}$ 의 $k$ 번째 행의 외적 $\mathbf{a}_k \mathbf{b}_k^\top$ 을 $k = 1, \dots, n$ 에 대해 더한 것과 같다.
외적 하나하나는 계수가 1 이하인 $m \times p$ 행렬이므로, $\mathbf{A}\mathbf{B}$ 의 계수는 $n$ 을 넘지 않는다.
특이값 분해 $\mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ 와 sample 별 $\mathbf{X}^\top\mathbf{X}$ 누적이 이 합으로 계산된다.

## 3. Notation

이 문서가 쓰는 기호는 아래 Table 1 과 같다.

Table 1. Symbols

| Symbol                                     | Meaning                                                                | Shape                              | Assumption                               |
| :----------------------------------------: | :--------------------------------------------------------------------: | :--------------------------------: | :--------------------------------------: |
| $\mathbf{A}$                               | 앞에 곱하는 행렬                                                       | $m \times n$                       | 실수 원소                                |
| $\mathbf{B}$                               | 뒤에 곱하는 행렬                                                       | $n \times p$                       | 실수 원소                                |
| $m$, $n$, $p$                              | $\mathbf{A}$ 의 행의 수, 안쪽 크기, $\mathbf{B}$ 의 열의 수            | scalar                             | 양의 정수                                |
| $a_{ik}$, $b_{kj}$                         | $\mathbf{A}$ 의 $i$ 행 $k$ 열 원소, $\mathbf{B}$ 의 $k$ 행 $j$ 열 원소 | scalar                             | 실수                                     |
| $i$, $j$                                   | 결과 행렬의 행 번호와 열 번호                                          | scalar                             | $i = 1, \dots, m$, $j = 1, \dots, p$     |
| $k$                                        | 안쪽 크기를 따라 더하는 번호                                           | scalar                             | $k = 1, \dots, n$                        |
| $\mathbf{a}_k$                             | $\mathbf{A}$ 의 $k$ 번째 열                                            | $m \times 1$                       | 실수 원소                                |
| $\mathbf{b}_k^\top$                        | $\mathbf{B}$ 의 $k$ 번째 행                                            | $1 \times p$                       | 실수 원소                                |
| $\mathbf{X}$                               | 행이 sample 이고 열이 feature 인 자료 행렬                             | $N \times d$                       | $N$ 은 sample 수, $d$ 는 feature 수      |
| $\mathbf{x}_i^\top$                        | $\mathbf{X}$ 의 $i$ 번째 행, sample 하나                               | $1 \times d$                       | 실수 원소                                |
| $\mathbf{U}$, $\mathbf{V}$                 | 특이값 분해의 좌특이벡터 행렬과 우특이벡터 행렬                        | 정사각                             | 직교 행렬                                |
| $\boldsymbol{\Sigma}$                      | 특이값을 대각에 둔 행렬                                                | $\mathbf{A}$ 와 같은 크기          | 대각 원소는 0 이상                       |
| $\sigma_k$, $\mathbf{u}_k$, $\mathbf{v}_k$ | $k$ 번째 특이값, 좌특이벡터, 우특이벡터                                | scalar, $m \times 1$, $n \times 1$ | $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ |
| $r$                                        | $\mathbf{A}$ 의 계수                                                   | scalar                             | $r \le \min(m, n)$                       |

특수 기호는 아래와 같이 읽는다.

- $\mathbf{b}_k^\top$: 영어로 "b k transpose", 한국어로 "b k 의 전치" 라 읽는다. 열 vector $\mathbf{b}_k$ 를 눕혀 행 vector 로 쓴 것이다.
- $\mathbf{a}_k \mathbf{b}_k^\top$: 영어로 "the outer product of a k and b k", 한국어로 "a k 와 b k 의 외적" 이라 읽는다.
- $\sum$: 영어로 "sum" 또는 "sigma", 한국어로 "합" 또는 "시그마" 라 읽는다. 아래위에 $k = 1$ 과 $n$ 이 붙으면 영어로 "the sum over k from 1 to n", 한국어로 "k 가 1 부터 n 까지의 합" 이라 읽는다.

## 4. Interpretation

행렬 곱 $\mathbf{A}\mathbf{B}$ 는 원소마다 내적으로 적을 수도 있고, 행렬 전체를 외적 $n$ 개의 합으로 적을 수도 있으며, 두 식은 같은 행렬을 준다.

```math
\mathbf{A}\mathbf{B} = \left[ \sum_{k=1}^{n} a_{ik} b_{kj} \right]_{ij} = \sum_{k=1}^{n} \mathbf{a}_k \mathbf{b}_k^\top \hspace{19em} (1)
```

가운데 변의 $\left[ \cdot \right]_{ij}$ 는 영어로 "the matrix whose i j entry is", 한국어로 "i 행 j 열 원소가 괄호 안의 값인 행렬" 이라 읽는다. 두 식이 같다는 것은 원소 하나씩 견주면 나오며, 유도는 [Appendix B](#appendix-b-derivation) 에 있다.

### 4.1 Inner Product Form

식 (1) 의 가운데 변은 결과의 $i$ 행 $j$ 열 원소 하나를 $\mathbf{A}$ 의 $i$ 번째 행과 $\mathbf{B}$ 의 $j$ 번째 열의 내적으로 구한다. 원소 $m \times p$ 개를 하나씩 채우는 계산이며, 원소마다 곱 $n$ 개를 더한다.

### 4.2 Outer Product Form

식 (1) 의 오른쪽 변은 안쪽 번호 $k$ 마다 $\mathbf{A}$ 의 $k$ 번째 열과 $\mathbf{B}$ 의 $k$ 번째 행을 곱해 $m \times p$ 행렬 하나를 만들고, 그 행렬 $n$ 개를 더한다. 외적 하나의 원소는 아래 식 (2) 와 같이 $\mathbf{A}$ 의 원소 하나와 $\mathbf{B}$ 의 원소 하나를 곱한 값이다.

```math
\mathbf{a}_k \mathbf{b}_k^\top =
\begin{bmatrix} a_{1k} \\ \vdots \\ a_{mk} \end{bmatrix}
\begin{bmatrix} b_{k1} & \cdots & b_{kp} \end{bmatrix}
=
\begin{bmatrix} a_{1k} b_{k1} & \cdots & a_{1k} b_{kp} \\ \vdots & \ddots & \vdots \\ a_{mk} b_{k1} & \cdots & a_{mk} b_{kp} \end{bmatrix}
\hspace{19em} (2)
```

식 (1) 의 두 변은 같은 곱 $a_{ik} b_{kj}$ 를 모두 더하며, 더하는 차례만 다르다. 가운데 변은 원소 $(i, j)$ 를 고정하고 $k$ 에 대해 더하고, 오른쪽 변은 $k$ 를 고정하고 행렬 전체를 만든 뒤 $k$ 에 대해 더한다. 식 (1) 은 $\mathbf{A}$ 의 열의 수와 $\mathbf{B}$ 의 행의 수가 같은 $n$ 일 때만 성립한다.

### 4.3 Rank of Each Term

식 (2) 의 외적은 모든 행이 $\mathbf{b}_k^\top$ 의 상수배이므로, $\mathbf{a}_k$ 와 $\mathbf{b}_k$ 가 모두 zero vector 가 아니면 계수가 1 이고, 하나라도 zero vector 이면 계수가 0 이다. 계수가 1 이하인 행렬 $n$ 개를 더하므로 $\mathrm{rank}(\mathbf{A}\mathbf{B}) \le n$ 이다.

## 5. Worked Example

$2 \times 2$ 행렬 두 개를 둔다.

```math
\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad
\mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}
\hspace{19em} (3)
```

내적 형태로 원소마다 계산하면 아래와 같다.

```math
\mathbf{A}\mathbf{B} =
\begin{bmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{bmatrix}
=
\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
\hspace{19em} (4)
```

외적 형태로는 $\mathbf{A}$ 의 첫 열과 $\mathbf{B}$ 의 첫 행, $\mathbf{A}$ 의 둘째 열과 $\mathbf{B}$ 의 둘째 행을 각각 곱해 더한다.

```math
\begin{gathered}
\mathbf{a}_1 \mathbf{b}_1^\top = \begin{bmatrix} 1 \\ 3 \end{bmatrix} \begin{bmatrix} 5 & 6 \end{bmatrix} = \begin{bmatrix} 5 & 6 \\ 15 & 18 \end{bmatrix},
\qquad
\mathbf{a}_2 \mathbf{b}_2^\top = \begin{bmatrix} 2 \\ 4 \end{bmatrix} \begin{bmatrix} 7 & 8 \end{bmatrix} = \begin{bmatrix} 14 & 16 \\ 28 & 32 \end{bmatrix} \\[2ex]
\mathbf{a}_1 \mathbf{b}_1^\top + \mathbf{a}_2 \mathbf{b}_2^\top = \begin{bmatrix} 5 + 14 & 6 + 16 \\ 15 + 28 & 18 + 32 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
\end{gathered}
\hspace{19em} (5)
```

식 (4) 와 식 (5) 의 결과가 같으므로 식 (1) 의 두 변이 같다. 식 (5) 의 두 외적은 둘째 행이 첫째 행의 3 배, 2 배여서 각각 계수가 1 이고, 두 외적의 합인 $\mathbf{A}\mathbf{B}$ 는 계수가 2 이다.

## 6. Application

외적의 합은 행렬을 계수 1 인 성분으로 나누어 보거나, 자료를 sample 하나씩 더해 가며 계산할 때 쓰인다.

### 6.1 Singular Value Decomposition

특이값 분해 $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ 에 식 (1) 을 쓰면, $\mathbf{U}\boldsymbol{\Sigma}$ 의 $k$ 번째 열 $\sigma_k \mathbf{u}_k$ 와 $\mathbf{V}^\top$ 의 $k$ 번째 행 $\mathbf{v}_k^\top$ 의 외적을 더한 것이 된다.

```math
\mathbf{A} = \sum_{k=1}^{r} \sigma_k\, \mathbf{u}_k \mathbf{v}_k^\top \hspace{19em} (6)
```

특이값이 0 인 항은 zero 행렬이어서 합에서 빠지므로, 항의 수는 $\mathbf{A}$ 의 계수 $r$ 이다. 큰 특이값의 항 몇 개만 남기면 계수가 낮은 근사 행렬이 된다. 식 (3) 의 곱 $\mathbf{A}\mathbf{B}$ 의 특이값은 $72.07$ 과 $0.0555$ 이고, 첫 항 하나만 남긴 근사 행렬과 $\mathbf{A}\mathbf{B}$ 의 차이는 Frobenius norm 으로 둘째 특이값과 같은 $0.0555$ 이다.

### 6.2 Gram Matrix of Samples

자료 행렬 $\mathbf{X}$ 의 Gram 행렬 $\mathbf{X}^\top\mathbf{X}$ 는 $\mathbf{X}^\top$ 의 열과 $\mathbf{X}$ 의 행이 모두 sample 이므로, 식 (1) 에 따라 sample 마다의 외적을 더한 것이다.

```math
\mathbf{X}^\top\mathbf{X} = \sum_{i=1}^{N} \mathbf{x}_i \mathbf{x}_i^\top \hspace{19em} (7)
```

식 (7) 은 sample 을 하나 받을 때마다 $d \times d$ 행렬 $\mathbf{x}_i \mathbf{x}_i^\top$ 을 더해 두면 되므로, 자료 전체를 memory 에 올리지 않고 $\mathbf{X}^\top\mathbf{X}$ 를 누적할 수 있다. 식 (3) 의 $\mathbf{A}$ 를 sample 두 개의 자료 행렬로 보면 $[1, 2]$ 와 $[3, 4]$ 의 외적 $\begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}$ 와 $\begin{bmatrix} 9 & 12 \\ 12 & 16 \end{bmatrix}$ 를 더해 $\mathbf{A}^\top\mathbf{A} = \begin{bmatrix} 10 & 14 \\ 14 & 20 \end{bmatrix}$ 이 나온다. 선형 회귀의 정규방정식과 공분산 행렬이 이 누적으로 계산된다.

### 6.3 Breaking Conditions

$\mathbf{A}$ 의 열의 수와 $\mathbf{B}$ 의 행의 수가 다르면 $\mathbf{a}_k$ 와 $\mathbf{b}_k^\top$ 의 짝이 맞지 않아 식 (1) 이 정의되지 않는다. 식 (1) 의 두 변은 곱셈 $m n p$ 번을 똑같이 하므로, 외적 형태로 바꾸어도 계산량은 줄지 않는다. 외적 형태는 $m \times p$ 행렬을 $n$ 번 더하므로 결과 크기의 memory 를 계속 쓰고, 부동소수점에서는 더하는 차례가 달라 마지막 자릿수가 내적 형태와 다를 수 있다.

---

## Appendix A. Terminology

- **Frobenius norm**: 행렬의 모든 원소의 제곱합의 제곱근.
- **Gram 행렬 (Gram matrix)**: 자료 행렬 $\mathbf{X}$ 의 열끼리의 내적을 모은 행렬 $\mathbf{X}^\top\mathbf{X}$.
- **계수 (rank)**: 행렬에서 서로 일차독립인 열의 최대 개수.
- **공분산 행렬 (covariance matrix)**: 열마다 평균을 뺀 자료 행렬의 Gram 행렬을 sample 수로 나눈 행렬.
- **내적 (dot product)**: 같은 크기의 두 vector 에서 같은 위치의 원소를 곱해 모두 더한 scalar.
- **외적 (outer product)**: 열 vector $\mathbf{a}$ 와 행 vector $\mathbf{b}^\top$ 를 곱한 행렬 $\mathbf{a}\mathbf{b}^\top$. 원소 $(i, j)$ 는 $a_i b_j$ 이다.
- **정규방정식 (normal equation)**: 선형 회귀의 최소제곱 weight $\mathbf{w}$ 를 구하는 식 $\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}$. $\mathbf{y}$ 는 관측값 vector 이다.
- **특이값 분해 (Singular Value Decomposition, SVD)**: 실수 행렬을 직교 행렬, 0 이상의 대각 원소를 가진 대각 행렬, 직교 행렬의 곱으로 나타내는 분해. 대각 원소가 특이값이다.

## Appendix B. Derivation

식 (1) 의 오른쪽 변의 $i$ 행 $j$ 열 원소가 가운데 변의 원소와 같음을 보이면 된다. 단계마다 쓴 성질은 아래 Table 2 와 같다.

Table 2. Derivation of equation (1)

| Step  | Expression                                                                                                                 | Property                  |
| :---: | :------------------------------------------------------------------------------------------------------------------------: | :-----------------------: |
| 1     | $\left( \sum_{k=1}^{n} \mathbf{a}_k \mathbf{b}_k^\top \right)_{ij} = \sum_{k=1}^{n} (\mathbf{a}_k \mathbf{b}_k^\top)_{ij}$ | 행렬의 합은 원소마다의 합 |
| 2     | $(\mathbf{a}_k \mathbf{b}_k^\top)_{ij} = a_{ik} b_{kj}$                                                                    | 식 (2) 의 외적의 원소     |
| 3     | $\sum_{k=1}^{n} a_{ik} b_{kj} = (\mathbf{A}\mathbf{B})_{ij}$                                                               | 행렬 곱의 원소의 정의     |
