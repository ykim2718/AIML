# Matrix Operations Primer
Rev. 0 | Created: 2026-10-26 | Updated: 2026-10-26 01:30 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Addition and Subtraction](#3-addition-and-subtraction)
  - [3.1 Notation](#31-notation)
  - [3.2 Interpretation](#32-interpretation)
  - [3.3 Worked Example](#33-worked-example)
  - [3.4 Application](#34-application)
- [4. Multiplication](#4-multiplication)
  - [4.1 Notation](#41-notation)
  - [4.2 Interpretation](#42-interpretation)
  - [4.3 Worked Example](#43-worked-example)
  - [4.4 Application](#44-application)
- [5. Inverse and Division](#5-inverse-and-division)
  - [5.1 Notation](#51-notation)
  - [5.2 Interpretation](#52-interpretation)
  - [5.3 Worked Example](#53-worked-example)
  - [5.4 Application](#54-application)
- [6. Differentiation](#6-differentiation)
  - [6.1 Notation](#61-notation)
  - [6.2 Interpretation](#62-interpretation)
  - [6.3 Worked Example](#63-worked-example)
  - [6.4 Application](#64-application)
- [7. Integration](#7-integration)
  - [7.1 Notation](#71-notation)
  - [7.2 Interpretation](#72-interpretation)
  - [7.3 Worked Example](#73-worked-example)
  - [7.4 Application](#74-application)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Derivation](#appendix-b-derivation)

## 1. Purpose

- **Problem Statement**: 행렬의 사칙연산과 미분·적분은 scalar 의 규칙과 모양이 같아 보이지만, 크기 조건과 곱의 순서가 달라 scalar 규칙을 그대로 옮기면 틀린 식이 나온다.
- **Goal**: 독자가 행렬 식 하나를 보고 그 연산이 정의되는 크기 조건, 결과의 크기, 그리고 곱의 순서를 바꿔도 되는지를 가릴 수 있게 한다.
- **Non-Goal**: 고윳값 분해의 계산법, $\exp(\mathbf{A})$ 같은 행렬 함수, tensor 미분은 다루지 않는다.

## 2. Summary

덧셈·뺄셈·scalar 곱은 같은 크기의 행렬에서 원소별로 계산하고, 행렬 곱은 앞 행렬의 열 수와 뒤 행렬의 행 수가 같을 때만 정의되며 순서를 바꾸면 값이 달라진다.
행렬에는 나눗셈이 없고 역행렬을 곱하며, 오른쪽에서 곱한 $\mathbf{B}\mathbf{A}^{-1}$ 와 왼쪽에서 곱한 $\mathbf{A}^{-1}\mathbf{B}$ 는 서로 다르다.
미분과 적분은 원소별로 하며, scalar 함수의 gradient 는 최소제곱 회귀의 정규방정식을, 이차형식의 Gaussian 적분은 다변량 정규분포의 정규화 상수를 준다.

## 3. Addition and Subtraction

같은 크기의 두 행렬을 더하거나 빼면 같은 위치의 원소끼리 더하거나 뺀 행렬이 된다.

### 3.1 Notation

이 꼭지가 쓰는 기호는 아래 Table 1 과 같으며, 뒤 꼭지들도 이 기호를 그대로 쓴다.

Table 1. Symbols for addition and subtraction

| Symbol                     | Meaning              | Shape        | Assumption |
| :------------------------: | :------------------: | :----------: | :--------: |
| $\mathbf{A}$, $\mathbf{B}$ | 행렬                 | $m \times n$ | 실수 원소  |
| $a_{ij}$, $b_{ij}$         | $i$ 행 $j$ 열의 원소 | scalar       | 실수       |
| $c$                        | 행렬에 곱하는 수     | scalar       | 실수       |
| $m$, $n$                   | 행의 수와 열의 수    | scalar       | 양의 정수  |

이 꼭지의 기호는 모두 글자 이름대로 읽으므로 따로 적을 읽는 법이 없다.

### 3.2 Interpretation

덧셈과 뺄셈, scalar 곱은 모두 원소마다 따로 계산한다.

```math
(\mathbf{A} \pm \mathbf{B})_{ij} = a_{ij} \pm b_{ij} \hspace{19em} (1)
```

```math
(c\mathbf{A})_{ij} = c\, a_{ij} \hspace{19em} (2)
```

식 (1) 은 $\mathbf{A}$ 와 $\mathbf{B}$ 의 크기가 같을 때만 성립한다. 원소마다 scalar 덧셈을 하므로 교환법칙 $\mathbf{A} + \mathbf{B} = \mathbf{B} + \mathbf{A}$ 와 결합법칙이 scalar 와 똑같이 성립한다.

### 3.3 Worked Example

두 $2 \times 2$ 행렬을 아래와 같이 둔다. 꼭지 4 와 꼭지 5 도 같은 두 행렬을 쓴다.

```math
\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}, \quad \mathbf{B} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} \hspace{19em} (3)
```

식 (1) 과 식 (2) 에 넣으면 아래와 같다.

```math
\mathbf{A} + \mathbf{B} = \begin{bmatrix} 6 & 8 \\ 10 & 12 \end{bmatrix}, \quad \mathbf{A} - \mathbf{B} = \begin{bmatrix} -4 & -4 \\ -4 & -4 \end{bmatrix}, \quad 2\mathbf{A} = \begin{bmatrix} 2 & 4 \\ 6 & 8 \end{bmatrix} \hspace{19em} (4)
```

### 3.4 Application

- **잔차 행렬**: 관측 행렬에서 예측 행렬을 빼면 원소마다의 오차가 남는다.
- **Centering**: 자료 행렬의 각 행에서 열 평균을 빼면 열마다 평균이 0 이 된다.
- **깨지는 조건**: 크기가 다르면 식 (1) 이 정의되지 않는다. NumPy 는 broadcasting 으로 크기가 1 인 축을 늘려 계산하므로, centering 처럼 의도한 경우가 아니면 크기를 먼저 확인한다.

## 4. Multiplication

행렬 곱 $\mathbf{A}\mathbf{B}$ 의 $i$ 행 $j$ 열 원소는 $\mathbf{A}$ 의 $i$ 번째 행과 $\mathbf{B}$ 의 $j$ 번째 열의 내적이다.

### 4.1 Notation

꼭지 3 의 기호에 더해 이 꼭지가 쓰는 기호는 아래 Table 2 와 같다.

Table 2. Symbols for multiplication

| Symbol                        | Meaning                      | Shape        | Assumption                                                 |
| :---------------------------: | :--------------------------: | :----------: | :--------------------------------------------------------: |
| $\mathbf{A}\mathbf{B}$        | 행렬 곱                      | $m \times p$ | $\mathbf{A}$ 는 $m \times n$, $\mathbf{B}$ 는 $n \times p$ |
| $\mathbf{A} \odot \mathbf{B}$ | 원소별 곱 (Hadamard product) | $m \times n$ | 두 행렬의 크기가 같음                                      |
| $p$                           | 뒤 행렬의 열의 수            | scalar       | 양의 정수                                                  |
| $k$                           | 곱한 뒤 더하는 원소의 번호   | scalar       | $k = 1, \dots, n$                                          |

특수 기호는 아래와 같이 읽는다.

- $\sum$: 영어로 "sum" 또는 "sigma", 한국어로 "합" 또는 "시그마" 라 읽는다.
- $\mathbf{A} \odot \mathbf{B}$: 영어로 "A Hadamard B" 또는 "the element-wise product of A and B", 한국어로 "A 와 B 의 원소별 곱" 이라 읽는다.

### 4.2 Interpretation

행렬 곱은 앞 행렬의 행과 뒤 행렬의 열을 짝지어 원소를 곱한 뒤 더한다.

```math
(\mathbf{A}\mathbf{B})_{ij} = \sum_{k=1}^{n} a_{ik} b_{kj} \hspace{19em} (5)
```

```math
(\mathbf{A} \odot \mathbf{B})_{ij} = a_{ij} b_{ij} \hspace{19em} (6)
```

식 (5) 는 $\mathbf{A}$ 의 열 수와 $\mathbf{B}$ 의 행 수가 같을 때만 성립하며, 결과의 크기는 $\mathbf{A}$ 의 행 수와 $\mathbf{B}$ 의 열 수로 정해진다. 교환법칙은 성립하지 않아 대개 $\mathbf{A}\mathbf{B} \ne \mathbf{B}\mathbf{A}$ 이고, 결합법칙 $(\mathbf{A}\mathbf{B})\mathbf{C} = \mathbf{A}(\mathbf{B}\mathbf{C})$ 와 분배법칙은 성립한다. 전치는 곱의 순서를 뒤집는다.

```math
(\mathbf{A}\mathbf{B})^\top = \mathbf{B}^\top \mathbf{A}^\top \hspace{19em} (7)
```

식 (6) 의 원소별 곱은 식 (1) 과 같이 같은 크기에서만 정의되며 교환법칙이 성립한다.

### 4.3 Worked Example

식 (3) 의 두 행렬로 세 곱을 계산하면 아래 Table 3 과 같다.

Table 3. Three products of A and B

| Product                       | Element (1, 1)                 | Result                                             |
| :---------------------------: | :----------------------------: | :------------------------------------------------: |
| $\mathbf{A}\mathbf{B}$        | $1 \times 5 + 2 \times 7 = 19$ | $\begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$ |
| $\mathbf{B}\mathbf{A}$        | $5 \times 1 + 6 \times 3 = 23$ | $\begin{bmatrix} 23 & 34 \\ 31 & 46 \end{bmatrix}$ |
| $\mathbf{A} \odot \mathbf{B}$ | $1 \times 5 = 5$               | $\begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}$  |

$\mathbf{A}\mathbf{B}$ 와 $\mathbf{B}\mathbf{A}$ 는 네 원소가 모두 다르다. 원소별 곱은 두 행렬 곱 어느 쪽과도 다르다.

### 4.4 Application

- **선형 model**: $n$ 개 sample 과 $p$ 개 feature 의 자료 행렬 $\mathbf{X}$ ($n \times p$) 에 weight vector $\mathbf{w}$ ($p \times 1$) 를 곱하면 예측값 vector $\mathbf{X}\mathbf{w}$ ($n \times 1$) 가 된다.
- **변환의 합성**: $\mathbf{B}$ 로 변환한 뒤 $\mathbf{A}$ 로 변환하는 것은 $\mathbf{A}\mathbf{B}$ 하나를 곱하는 것과 같다. 오른쪽 행렬이 먼저 적용된다.
- **깨지는 조건**: 안쪽 크기가 다르면 식 (5) 가 정의되지 않는다. NumPy 에서 `@` 는 식 (5) 의 행렬 곱이고 `*` 는 식 (6) 의 원소별 곱이므로, 두 연산자를 바꿔 쓰면 같은 크기의 정사각 행렬에서는 오류 없이 다른 값이 나온다.

## 5. Inverse and Division

행렬에는 나눗셈이 없고, 정사각 행렬 $\mathbf{A}$ 의 역행렬 $\mathbf{A}^{-1}$ 을 곱하는 것이 나눗셈을 대신한다.

### 5.1 Notation

꼭지 3 과 꼭지 4 의 기호에 더해 이 꼭지가 쓰는 기호는 아래 Table 4 와 같다.

Table 4. Symbols for the inverse

| Symbol             | Meaning                | Shape        | Assumption                                       |
| :----------------: | :--------------------: | :----------: | :----------------------------------------------: |
| $\mathbf{A}^{-1}$  | $\mathbf{A}$ 의 역행렬 | $n \times n$ | $\mathbf{A}$ 는 정사각, $\det(\mathbf{A}) \ne 0$ |
| $\mathbf{I}$       | 단위 행렬              | $n \times n$ | 대각 원소 1, 나머지 0                            |
| $\det(\mathbf{A})$ | $\mathbf{A}$ 의 행렬식 | scalar       | $\mathbf{A}$ 는 정사각                           |

특수 기호는 아래와 같이 읽는다.

- $\mathbf{A}^{-1}$: 영어로 "A inverse", 한국어로 "A 의 역행렬" 또는 "A 인버스" 라 읽는다.
- $\det(\mathbf{A})$: 영어로 "the determinant of A" 또는 "det A", 한국어로 "A 의 행렬식" 이라 읽는다.

### 5.2 Interpretation

역행렬은 곱해서 단위 행렬이 되는 행렬이며, 어느 쪽에서 곱해도 같다.

```math
\mathbf{A}\mathbf{A}^{-1} = \mathbf{A}^{-1}\mathbf{A} = \mathbf{I} \hspace{19em} (8)
```

$2 \times 2$ 행렬의 역행렬은 아래 식으로 바로 구한다.

```math
\mathbf{A} = \begin{bmatrix} a & b \\ c & d \end{bmatrix}, \quad \det(\mathbf{A}) = ad - bc, \quad \mathbf{A}^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix} \hspace{19em} (9)
```

식 (8) 과 식 (9) 는 $\mathbf{A}$ 가 정사각이고 $\det(\mathbf{A}) \ne 0$ 일 때만 성립한다. 행렬 곱에 교환법칙이 없으므로 "B 를 A 로 나눈다" 는 두 가지로 갈린다. 오른쪽에서 곱한 $\mathbf{B}\mathbf{A}^{-1}$ 은 $\mathbf{X}\mathbf{A} = \mathbf{B}$ 의 해이고, 왼쪽에서 곱한 $\mathbf{A}^{-1}\mathbf{B}$ 는 $\mathbf{A}\mathbf{X} = \mathbf{B}$ 의 해이다.

### 5.3 Worked Example

식 (3) 의 $\mathbf{A}$ 는 $\det(\mathbf{A}) = 1 \times 4 - 2 \times 3 = -2$ 이므로 역행렬이 있다. 식 (9) 에 넣으면 아래와 같고, $\mathbf{A}\mathbf{A}^{-1}$ 을 계산하면 $\mathbf{I}$ 가 된다.

```math
\mathbf{A}^{-1} = \frac{1}{-2} \begin{bmatrix} 4 & -2 \\ -3 & 1 \end{bmatrix} = \begin{bmatrix} -2 & 1 \\ 1.5 & -0.5 \end{bmatrix} \hspace{19em} (10)
```

두 가지 나눗셈은 아래 Table 5 와 같이 다른 값을 준다.

Table 5. Right and left division of B by A

| Division                    | Solves                              | Result                                           |
| :-------------------------: | :---------------------------------: | :----------------------------------------------: |
| $\mathbf{B}\mathbf{A}^{-1}$ | $\mathbf{X}\mathbf{A} = \mathbf{B}$ | $\begin{bmatrix} -1 & 2 \\ -2 & 3 \end{bmatrix}$ |
| $\mathbf{A}^{-1}\mathbf{B}$ | $\mathbf{A}\mathbf{X} = \mathbf{B}$ | $\begin{bmatrix} -3 & -4 \\ 4 & 5 \end{bmatrix}$ |

### 5.4 Application

- **최소제곱 해**: 선형 회귀의 weight 는 $\mathbf{w} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$ 이다. 식은 꼭지 6 의 식 (13) 에서 나온다.
- **계산 방법**: 역행렬을 구해 곱하는 대신 연립방정식 $\mathbf{A}\mathbf{x} = \mathbf{b}$ 를 직접 푼다. NumPy 의 `numpy.linalg.solve` 가 이 계산을 하며, `numpy.linalg.inv` 로 역행렬을 구해 곱하는 것보다 반올림 오차가 작고 빠르다.
- **깨지는 조건**: $\det(\mathbf{A}) = 0$ 인 특이 행렬과 정사각이 아닌 행렬에는 역행렬이 없다. Feature 사이의 공선성이 강하면 $\mathbf{X}^\top\mathbf{X}$ 의 행렬식이 0 에 가까워져 역행렬의 원소가 매우 커진다. 이때는 pseudo-inverse 나 regularization 을 쓴다.

## 6. Differentiation

행렬과 vector 의 미분은 원소마다 미분한 값을 원래 배치대로 모은 것이다.

### 6.1 Notation

이 문서는 scalar 함수를 vector 로 미분한 결과를 열 vector 로 적는다 (denominator layout). 꼭지 3 부터 꼭지 5 까지의 기호에 더해 이 꼭지가 쓰는 기호는 아래 Table 6 과 같다.

Table 6. Symbols for differentiation

| Symbol                             | Meaning                            | Shape        | Assumption                                  |
| :--------------------------------: | :--------------------------------: | :----------: | :-----------------------------------------: |
| $\mathbf{x}$, $\mathbf{a}$         | vector                             | $n \times 1$ | 실수 원소                                   |
| $f(\mathbf{x})$                    | vector 를 받아 scalar 를 주는 함수 | scalar       | 각 원소로 미분 가능                         |
| $\partial f / \partial \mathbf{x}$ | $f$ 의 gradient                    | $n \times 1$ | $i$ 번째 원소는 $\partial f / \partial x_i$ |
| $\mathbf{X}$                       | 자료 행렬                          | $n \times p$ | 실수 원소                                   |
| $\mathbf{y}$                       | 관측값 vector                      | $n \times 1$ | 실수 원소                                   |
| $\mathbf{w}$                       | weight vector                      | $p \times 1$ | 실수 원소                                   |
| $\mathbf{A}(t)$                    | 원소가 $t$ 의 함수인 행렬          | $m \times n$ | 각 원소가 $t$ 로 미분 가능                  |

특수 기호는 아래와 같이 읽는다.

- $\mathbf{A}^\top$: 영어로 "A transpose", 한국어로 "A 의 전치" 또는 "A 트랜스포즈" 라 읽는다.
- $\partial f / \partial \mathbf{x}$: 영어로 "partial f partial x" 또는 "the gradient of f with respect to x", 한국어로 "f 를 x 로 편미분" 이라 읽는다.
- $\lVert \mathbf{v} \rVert$: 영어로 "the norm of v", 한국어로 "v 의 노름" 이라 읽는다.
- $d\mathbf{A}/dt$: 영어로 "d A d t" 또는 "the derivative of A with respect to t", 한국어로 "A 를 t 로 미분" 이라 읽는다.

### 6.2 Interpretation

선형 함수와 이차형식의 gradient 는 아래 두 식이 기본이다.

```math
\frac{\partial (\mathbf{a}^\top \mathbf{x})}{\partial \mathbf{x}} = \mathbf{a} \hspace{19em} (11)
```

```math
\frac{\partial (\mathbf{x}^\top \mathbf{A} \mathbf{x})}{\partial \mathbf{x}} = (\mathbf{A} + \mathbf{A}^\top)\,\mathbf{x} \hspace{19em} (12)
```

식 (12) 는 $\mathbf{A}$ 가 $n \times n$ 정사각일 때 성립하고, $\mathbf{A}$ 가 대칭 ($\mathbf{A}^\top = \mathbf{A}$) 이면 $2\mathbf{A}\mathbf{x}$ 로 줄어든다. 식 (11) 과 식 (12) 를 묶으면 제곱오차의 gradient 가 나온다.

```math
\frac{\partial \lVert \mathbf{X}\mathbf{w} - \mathbf{y} \rVert^2}{\partial \mathbf{w}} = 2\,\mathbf{X}^\top (\mathbf{X}\mathbf{w} - \mathbf{y}) \hspace{19em} (13)
```

원소가 $t$ 의 함수인 행렬은 원소마다 미분하고, 곱의 미분은 순서를 지킨다.

```math
\left( \frac{d\mathbf{A}}{dt} \right)_{ij} = \frac{d a_{ij}(t)}{dt}, \quad \frac{d(\mathbf{A}\mathbf{B})}{dt} = \frac{d\mathbf{A}}{dt}\mathbf{B} + \mathbf{A}\frac{d\mathbf{B}}{dt} \hspace{19em} (14)
```

식 (12) 와 식 (13) 의 유도는 [Appendix B](#appendix-b-derivation) 에 있다.

### 6.3 Worked Example

$\mathbf{a} = [3, -1]^\top$, $\mathbf{x} = [1, 2]^\top$ 이면 $\mathbf{a}^\top\mathbf{x} = 3 - 2 = 1$ 이고, 식 (11) 의 gradient 는 $\mathbf{x}$ 와 무관하게 $[3, -1]^\top$ 이다.

이차형식에는 대칭 행렬 $\mathbf{S}$ 를 쓴다.

```math
\mathbf{S} = \begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}, \quad \mathbf{x}^\top \mathbf{S} \mathbf{x} = 18, \quad 2\mathbf{S}\mathbf{x} = \begin{bmatrix} 8 \\ 14 \end{bmatrix} \hspace{19em} (15)
```

각 원소를 $10^{-6}$ 만큼 움직여 중앙 차분으로 미분하면 8.000 과 14.000 이 나와 식 (12) 의 값과 같다.

제곱오차에는 sample 세 개, feature 두 개의 자료를 쓴다.

```math
\mathbf{X} = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}, \quad \mathbf{y} = \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}, \quad \mathbf{X}^\top\mathbf{X} = \begin{bmatrix} 3 & 6 \\ 6 & 14 \end{bmatrix}, \quad \mathbf{X}^\top\mathbf{y} = \begin{bmatrix} 5 \\ 11 \end{bmatrix} \hspace{19em} (16)
```

$\mathbf{w} = \mathbf{0}$ 에서 식 (13) 의 gradient 는 $-2\,\mathbf{X}^\top\mathbf{y} = [-10, -22]^\top$ 이다. Gradient 를 0 으로 두면 $\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}$ 가 되고, 식 (9) 로 풀면 $\mathbf{w} = [2/3, 1/2]^\top$ 이다. 이 $\mathbf{w}$ 를 식 (13) 에 넣으면 gradient 가 $[0, 0]^\top$ 이 된다.

원소가 $t$ 의 함수인 행렬은 식 (14) 대로 원소마다 미분한다.

```math
\mathbf{A}(t) = \begin{bmatrix} t & t^2 \\ \sin t & 1 \end{bmatrix}, \quad \frac{d\mathbf{A}}{dt} = \begin{bmatrix} 1 & 2t \\ \cos t & 0 \end{bmatrix} \hspace{19em} (17)
```

$t = 0.5$ 에서 식 (17) 의 미분은 $[[1, 1], [0.878, 0]]$ 이다.

### 6.4 Application

- **정규방정식**: 식 (13) 을 0 으로 둔 $\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}$ 가 최소제곱 회귀의 정규방정식이며, 꼭지 5.4 의 최소제곱 해가 여기서 나온다.
- **Gradient descent**: 해를 식으로 풀 수 없는 model 은 식 (13) 같은 gradient 를 구해 $\mathbf{w}$ 를 $-\eta\,\partial f / \partial \mathbf{w}$ 만큼 옮기기를 되풀이한다. $\eta$ 는 한 번에 옮기는 크기 (learning rate) 이다.
- **깨지는 조건**: $\mathbf{A}$ 가 대칭이 아니면 식 (12) 를 $2\mathbf{A}\mathbf{x}$ 로 줄일 수 없다. Gradient 를 행 vector 로 적는 numerator layout 의 문헌을 함께 읽으면 모든 결과가 전치되어 나오므로, 문헌마다 어느 배치를 쓰는지 먼저 확인한다.

## 7. Integration

행렬의 적분은 원소마다 적분하고, vector 위의 적분에서는 이차형식의 Gaussian 적분이 가장 자주 쓰인다.

### 7.1 Notation

꼭지 3 부터 꼭지 6 까지의 기호에 더해 이 꼭지가 쓰는 기호는 아래 Table 7 과 같다.

Table 7. Symbols for integration

| Symbol                | Meaning                               | Shape        | Assumption                 |
| :-------------------: | :-----------------------------------: | :----------: | :------------------------: |
| $\mathbf{S}$          | 이차형식의 계수 행렬                  | $n \times n$ | 대칭, 양의 정부호          |
| $\mathbb{R}^n$        | $n$ 차원 실수 공간                    | 집합         | 적분 범위                  |
| $d\mathbf{x}$         | $\mathbf{x}$ 의 모든 원소에 대한 적분 | 없음         | $dx_1 \cdots dx_n$ 과 같음 |
| $\boldsymbol{\Sigma}$ | 공분산 행렬                           | $n \times n$ | 대칭, 양의 정부호          |
| $\boldsymbol{\mu}$    | 평균 vector                           | $n \times 1$ | 실수 원소                  |

특수 기호는 아래와 같이 읽는다.

- $\int$: 영어로 "integral", 한국어로 "적분" 또는 "인테그랄" 이라 읽는다. $\int_0^1$ 은 영어로 "the integral from 0 to 1", 한국어로 "0 부터 1 까지의 적분" 이라 읽는다.
- $\mathbb{R}^n$: 영어로 "R n" 또는 "real n-space", 한국어로 "n 차원 실수 공간" 이라 읽는다.

### 7.2 Interpretation

원소가 $t$ 의 함수인 행렬은 원소마다 적분한다.

```math
\left( \int_{a}^{b} \mathbf{A}(t)\, dt \right)_{ij} = \int_{a}^{b} a_{ij}(t)\, dt \hspace{19em} (18)
```

양의 정부호 대칭 행렬 $\mathbf{S}$ 의 이차형식을 지수에 둔 함수는 $\mathbb{R}^n$ 전체에서 적분하면 행렬식 하나로 정리된다.

```math
\int_{\mathbb{R}^n} \exp\!\left( -\tfrac{1}{2}\, \mathbf{x}^\top \mathbf{S} \mathbf{x} \right) d\mathbf{x} = \sqrt{\frac{(2\pi)^n}{\det(\mathbf{S})}} \hspace{19em} (19)
```

식 (19) 는 $\mathbf{S}$ 가 양의 정부호일 때만 성립한다. 유도는 [Appendix B](#appendix-b-derivation) 에 있다.

### 7.3 Worked Example

식 (18) 에 아래 행렬을 넣으면 원소마다 적분한 값이 된다.

```math
\int_0^1 \begin{bmatrix} 1 & t \\ t^2 & 2t \end{bmatrix} dt = \begin{bmatrix} 1 & 1/2 \\ 1/3 & 1 \end{bmatrix} \hspace{19em} (20)
```

식 (15) 의 $\mathbf{S}$ 는 $\det(\mathbf{S}) = 2 \times 3 - 1 \times 1 = 5$ 이므로, $n = 2$ 일 때 식 (19) 의 값은 $\sqrt{(2\pi)^2 / 5} = 2\pi / \sqrt{5} \approx 2.80993$ 이다. 두 축을 각각 -12 에서 12 까지 0.01 간격으로 나누어 합으로 적분하면 2.80993 이 나와 식 (19) 의 값과 같다.

### 7.4 Application

- **다변량 정규분포**: 평균 $\boldsymbol{\mu}$ 와 공분산 $\boldsymbol{\Sigma}$ 의 정규분포 밀도는 식 (19) 에 $\mathbf{S} = \boldsymbol{\Sigma}^{-1}$ 을 넣은 값으로 나누어 전체 적분이 1 이 되게 한 것이다. $\det(\boldsymbol{\Sigma}^{-1}) = 1 / \det(\boldsymbol{\Sigma})$ 이므로 정규화 상수는 $\sqrt{(2\pi)^n \det(\boldsymbol{\Sigma})}$ 이다.

```math
p(\mathbf{x}) = \frac{1}{\sqrt{(2\pi)^n \det(\boldsymbol{\Sigma})}} \exp\!\left( -\tfrac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right) \hspace{19em} (21)
```

- **누적량**: 시간에 따라 변하는 행렬, 예를 들어 시점마다의 공분산을 식 (18) 로 적분하면 구간 전체의 누적값이 된다.
- **깨지는 조건**: $\mathbf{S}$ 가 양의 정부호가 아니면 어떤 방향에서 지수가 줄지 않아 식 (19) 의 적분이 발산한다. 공분산 행렬이 특이하면 $\boldsymbol{\Sigma}^{-1}$ 이 없어 식 (21) 을 쓸 수 없다.

---

## Appendix A. Terminology

- **broadcasting**: 크기가 1 인 축을 다른 배열의 크기만큼 늘려 원소별 연산을 하는 NumPy 의 규칙.
- **denominator layout**: Scalar 를 열 vector 로 미분한 결과를 열 vector 로 적는 배치.
- **gradient**: Scalar 함수를 vector 의 각 원소로 편미분한 값을 모은 vector.
- **numerator layout**: Scalar 를 열 vector 로 미분한 결과를 행 vector 로 적는 배치.
- **pseudo-inverse**: 정사각이 아니거나 특이한 행렬에도 정의되며, 최소제곱 해를 주는 역행렬의 일반화.
- **regularization**: 계수의 크기에 벌점을 더해 $\mathbf{X}^\top\mathbf{X}$ 가 특이에 가까워도 해가 안정되게 하는 방법.
- **고윳값 분해**: 대칭 행렬을 $\mathbf{Q}\boldsymbol{\Lambda}\mathbf{Q}^\top$ 로 나타내는 분해. $\mathbf{Q}$ 의 열은 서로 직교하는 단위 vector 이고 $\boldsymbol{\Lambda}$ 는 고윳값을 대각에 둔 행렬이다.
- **양의 정부호**: 영 vector 가 아닌 모든 $\mathbf{x}$ 에서 $\mathbf{x}^\top\mathbf{S}\mathbf{x} \gt 0$ 인 대칭 행렬의 성질. 고윳값이 모두 양수인 것과 같다.
- **이차형식**: $\mathbf{x}^\top\mathbf{A}\mathbf{x}$ 처럼 vector 의 원소를 두 개씩 곱해 더한 scalar.
- **정규방정식**: 최소제곱 회귀의 gradient 를 0 으로 둔 연립방정식 $\mathbf{X}^\top\mathbf{X}\mathbf{w} = \mathbf{X}^\top\mathbf{y}$.
- **특이 행렬**: 행렬식이 0 이어서 역행렬이 없는 정사각 행렬.

## Appendix B. Derivation

식 (12) 는 이차형식을 원소로 풀어 $x_k$ 로 미분하면 나온다. 단계마다 쓴 성질은 아래 Table 8 과 같다.

Table 8. Derivation of equation (12)

| Step  | Expression                                                                                                            | Property             |
| :---: | :-------------------------------------------------------------------------------------------------------------------: | :------------------: |
| 1     | $\mathbf{x}^\top \mathbf{A} \mathbf{x} = \sum_{i} \sum_{j} x_i\, a_{ij}\, x_j$                                        | 식 (5) 를 두 번 적용 |
| 2     | $\partial (\mathbf{x}^\top \mathbf{A} \mathbf{x}) / \partial x_k = \sum_{j} a_{kj} x_j + \sum_{i} x_i\, a_{ik}$       | 곱의 미분            |
| 3     | $\sum_{j} a_{kj} x_j = (\mathbf{A}\mathbf{x})_k, \quad \sum_{i} a_{ik} x_i = (\mathbf{A}^\top\mathbf{x})_k$           | 식 (5), 전치의 정의  |
| 4     | $\partial (\mathbf{x}^\top \mathbf{A} \mathbf{x}) / \partial \mathbf{x} = (\mathbf{A} + \mathbf{A}^\top)\,\mathbf{x}$ | $k$ 번째 원소를 모음 |

식 (13) 은 제곱오차를 전개한 뒤 식 (11) 과 식 (12) 를 적용하면 나온다. 단계마다 쓴 성질은 아래 Table 9 와 같다.

Table 9. Derivation of equation (13)

| Step  | Expression                                                                                                                                                                          | Property                                          |
| :---: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------: |
| 1     | $\lVert \mathbf{X}\mathbf{w} - \mathbf{y} \rVert^2 = \mathbf{w}^\top \mathbf{X}^\top \mathbf{X} \mathbf{w} - 2\,\mathbf{y}^\top \mathbf{X} \mathbf{w} + \mathbf{y}^\top \mathbf{y}$ | 분배법칙, 식 (7)                                  |
| 2     | $\partial (\mathbf{w}^\top \mathbf{X}^\top \mathbf{X} \mathbf{w}) / \partial \mathbf{w} = 2\,\mathbf{X}^\top \mathbf{X} \mathbf{w}$                                                 | 식 (12), $\mathbf{X}^\top\mathbf{X}$ 는 대칭      |
| 3     | $\partial (2\,\mathbf{y}^\top \mathbf{X} \mathbf{w}) / \partial \mathbf{w} = 2\,\mathbf{X}^\top \mathbf{y}$                                                                         | 식 (11), $\mathbf{a} = \mathbf{X}^\top\mathbf{y}$ |
| 4     | $\partial \lVert \mathbf{X}\mathbf{w} - \mathbf{y} \rVert^2 / \partial \mathbf{w} = 2\,\mathbf{X}^\top (\mathbf{X}\mathbf{w} - \mathbf{y})$                                         | 단계 2 에서 단계 3 을 뺌                          |

식 (19) 는 고윳값 분해로 좌표를 돌려 1 차원 Gaussian 적분 $n$ 개의 곱으로 만들면 나온다. 단계마다 쓴 성질은 아래 Table 10 과 같다.

Table 10. Derivation of equation (19)

| Step  | Expression                                                                                                                                                | Property                                                   |
| :---: | :-------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------: |
| 1     | $\mathbf{S} = \mathbf{Q} \boldsymbol{\Lambda} \mathbf{Q}^\top, \quad \boldsymbol{\Lambda} = \mathrm{diag}(\lambda_1, \dots, \lambda_n)$                   | 대칭 행렬의 고윳값 분해                                    |
| 2     | $\mathbf{z} = \mathbf{Q}^\top \mathbf{x}, \quad \mathbf{x}^\top \mathbf{S} \mathbf{x} = \sum_{i} \lambda_i z_i^2, \quad d\mathbf{x} = d\mathbf{z}$        | $\mathbf{Q}$ 는 직교, $\lvert \det(\mathbf{Q}) \rvert = 1$ |
| 3     | $\int_{\mathbb{R}^n} \exp(-\tfrac{1}{2} \sum_{i} \lambda_i z_i^2)\, d\mathbf{z} = \prod_{i} \int_{\mathbb{R}} \exp(-\tfrac{1}{2} \lambda_i z_i^2)\, dz_i$ | 지수의 합은 곱으로 나뉨                                    |
| 4     | $\int_{\mathbb{R}} \exp(-\tfrac{1}{2} \lambda_i z_i^2)\, dz_i = \sqrt{2\pi / \lambda_i}$                                                                  | 1 차원 Gaussian 적분, $\lambda_i \gt 0$                    |
| 5     | $\prod_{i} \sqrt{2\pi / \lambda_i} = \sqrt{(2\pi)^n / \det(\mathbf{S})}$                                                                                  | $\det(\mathbf{S}) = \prod_{i} \lambda_i$                   |
