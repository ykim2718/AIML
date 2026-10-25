# Lagrange Multiplier Method
Rev. 1 | Created: 2026-10-25 | Updated: 2026-10-25 16:40 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Notation](#3-notation)
- [4. Interpretation](#4-interpretation)
  - [4.1 Gradient Condition](#41-gradient-condition)
  - [4.2 Lagrangian Function](#42-lagrangian-function)
  - [4.3 Multiplier](#43-multiplier)
- [5. Worked Example](#5-worked-example)
- [6. Application](#6-application)
  - [6.1 Engineering Constraints](#61-engineering-constraints)
  - [6.2 Support Vector Machine](#62-support-vector-machine)
  - [6.3 Regularization](#63-regularization)
  - [6.4 Semiconductor Process Optimization](#64-semiconductor-process-optimization)
  - [6.5 Breaking Conditions](#65-breaking-conditions)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Derivation](#appendix-b-derivation)

## 1. Purpose

- **Problem Statement**: 제약 조건 (constraint) 이 걸린 최적화 문제는 목적 함수 (objective function) 의 gradient 가 0 인 점을 찾는 방법으로 풀 수 없다.
- **Goal**: 등식 제약 조건을 만족하면서 목적 함수의 최댓값이나 최솟값을 주는 점을 연립방정식으로 구한다.
- **Non-Goal**: SVM 의 dual 문제 유도와 KKT 조건의 상세 증명은 다루지 않는다.

## 2. Summary

라그랑주 승수법 (Lagrange multiplier method) 에서는 제약 조건 $g(x, y) = C$ 위의 극값에서 $\nabla f$ 가 $\nabla g$ 의 $\lambda$ 배와 같다.
비례 상수 $\lambda$ 가 라그랑주 승수 (Lagrange multiplier) 이며, 그 값은 $C$ 를 바꿀 때 최적값 $f^\ast$ 가 변하는 비율이다.
라그랑주 함수 $\mathcal{L}$ 를 $x, y, \lambda$ 로 편미분한 값을 0 으로 두고 연립방정식을 풀면 최적점이 나온다.

## 3. Notation

Table 1. Symbols

| Symbol                     | Meaning                                  | Shape             | Assumption                          |
| :------------------------: | :--------------------------------------: | :---------------: | :---------------------------------: |
| $x, y$                     | 최적화 변수                              | scalar            | 실수                                |
| $f(x, y)$                  | 목적 함수                                | scalar            | 미분 가능                           |
| $g(x, y)$                  | 제약 함수                                | scalar            | 미분 가능                           |
| $C$                        | 제약 조건의 상수                         | scalar            | $g(x, y) = C$ 를 만족하는 점이 있음 |
| $\lambda$                  | 라그랑주 승수                            | scalar            | 실수                                |
| $\mathcal{L}(x, y, \lambda)$ | 라그랑주 함수                        | scalar            | 식 (2) 로 정의                      |
| $\nabla f, \nabla g$       | $f$, $g$ 의 gradient                     | 2 차원 vector     | 최적점에서 $\nabla g \ne 0$         |
| $f^\ast$                   | 제약 조건 아래에서 $f$ 의 최적값         | scalar            | $C$ 의 함수                         |
| $\mathbf{t}$               | 제약 곡선의 접선 방향                    | 2 차원 vector     | $\mathbf{t} \ne 0$                  |
| $\mathbf{w}$               | model 의 가중치                          | $p$ 차원 vector   | 꼭지 6.3 에서만 씀                  |

- $\nabla$: 영어로 "nabla" 또는 "del", 한국어로 "나블라" 라 읽는다.
- $\partial$: 영어로 "partial", 한국어로 "편미분" 또는 "라운드" 라 읽는다.
- $\mathcal{L}$: 영어로 "Lagrangian", 한국어로 "라그랑지안" 이라 읽는다.
- $f^\ast$: 영어로 "f star", 한국어로 "f 스타" 라 읽는다.
- $\cdot$: 영어로 "dot", 한국어로 "내적" 이라 읽는다.
- $\parallel$: 영어로 "is parallel to", 한국어로 "평행하다" 라 읽는다.
- $\lVert \mathbf{w} \rVert$: 영어로 "norm of w", 한국어로 "w 의 노름" 이라 읽는다.

## 4. Interpretation

제약 조건 $g(x, y) = C$ 위에서 $f$ 가 극값을 갖는 점에서는 두 gradient 가 평행하다.

```math
\nabla f = \lambda \nabla g, \quad g(x, y) = C \hspace{19em} (1)
```

식 (1) 은 $f$ 와 $g$ 가 미분 가능하고, 그 점에서 $\nabla g \ne 0$ 이며, 제약 조건이 등식일 때 성립한다.
식 (1) 을 만족하는 점은 극값의 후보이며, 최댓값인지 최솟값인지는 후보마다 $f$ 의 값을 견주어 정한다.

### 4.1 Gradient Condition

최적점은 목적 함수의 등고선과 제약 조건의 곡선이 접하는 점이다. 2 차원 공간에서 세 요소는 아래와 같다.

1. **목적 함수 $f(x, y)$ 의 등고선**: 지도에서 산의 높이를 나타내는 동심원이며, 올라가야 할 정상이 목표다.
2. **제약 조건 $g(x, y) = C$**: 반드시 따라 걸어야 하는 등산로다.
3. **접점 (tangency point)**: 등산로를 따라 걸을 때 $f$ 의 등고선과 등산로가 접하는 점이며, 제약 조건 아래에서 도달할 수 있는 가장 높은 점이다.

접점에서 $\nabla f$ 와 $\nabla g$ 는 모두 제약 곡선의 접선 방향 $\mathbf{t}$ 에 수직이므로, 2 차원에서 두 gradient 는 평행하다.
두 gradient 가 평행하다는 조건이 식 (1) 의 앞 식이며, 유도는 [Appendix B](#appendix-b-derivation) 에 있다.
조건 $\nabla g \ne 0$ 은 이 변에서 필요하다. $\nabla g = 0$ 이면 접선 방향이 정해지지 않아 식 (1) 이 극값을 놓친다.

### 4.2 Lagrangian Function

목적 함수에 제약 조건을 결합해 라그랑주 함수 $\mathcal{L}$ 를 정의한다.

```math
\mathcal{L}(x, y, \lambda) = f(x, y) - \lambda \left( g(x, y) - C \right) \hspace{19em} (2)
```

- $f(x, y)$: 최적화할 목적 함수이며, 성능이나 오차가 여기에 해당한다.
- $g(x, y) = C$: 반드시 만족해야 하는 제약 조건이다.
- $\lambda \left( g(x, y) - C \right)$: 제약 조건을 어긴 정도에 $\lambda$ 를 곱한 항이다.

식 (2) 를 $x, y, \lambda$ 로 각각 편미분하여 0 으로 둔 연립방정식은 아래와 같다.

```math
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial x} &= \frac{\partial f}{\partial x} - \lambda \frac{\partial g}{\partial x} = 0 \\
\frac{\partial \mathcal{L}}{\partial y} &= \frac{\partial f}{\partial y} - \lambda \frac{\partial g}{\partial y} = 0 \\
\frac{\partial \mathcal{L}}{\partial \lambda} &= -\left( g(x, y) - C \right) = 0
\end{aligned}
\hspace{15em} (3)
```

식 (3) 의 앞 두 식은 $\nabla f = \lambda \nabla g$ 와 같고, 마지막 식은 $g(x, y) = C$ 와 같다.
따라서 식 (3) 을 푸는 것과 식 (1) 을 푸는 것은 같은 점을 구한다.

### 4.3 Multiplier

라그랑주 승수 $\lambda$ 는 제약 조건의 상수 $C$ 를 조금 바꿀 때 최적값 $f^\ast$ 가 변하는 비율이다.

```math
\lambda = \frac{d f^\ast}{d C} \hspace{19em} (4)
```

$\lambda$ 가 크면 제약 조건을 조금 늦출 때 목적 함수가 많이 좋아지고, $\lambda = 0$ 이면 그 제약 조건이 최적값에 영향을 주지 않는다.
식 (4) 는 최적점과 $\lambda$ 가 $C$ 에 대해 미분 가능하게 변할 때 성립한다.

## 5. Worked Example

$x + y = 10$ 을 만족하면서 $f(x, y) = xy$ 를 최대화한다. $g(x, y) = x + y$, $C = 10$ 이다.

식 (1) 로 풀면 $\nabla f = (y, x)$, $\nabla g = (1, 1)$ 이므로 $y = \lambda$, $x = \lambda$ 이다.
제약 조건 $x + y = 10$ 에 넣으면 $2\lambda = 10$ 이 되어 $\lambda = 5$, $x = y = 5$, $f^\ast = 25$ 이다.

식 (3) 으로 풀면 $\mathcal{L} = xy - \lambda (x + y - 10)$ 이고, 세 편미분은 아래와 같다.

```math
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial x} &= y - \lambda = 0 \\
\frac{\partial \mathcal{L}}{\partial y} &= x - \lambda = 0 \\
\frac{\partial \mathcal{L}}{\partial \lambda} &= -(x + y - 10) = 0
\end{aligned}
\hspace{15em} (5)
```

식 (5) 의 해도 $x = y = 5$, $\lambda = 5$ 로, 식 (1) 로 구한 값과 같다.
제약 조건 위의 다른 점과 견준 값은 아래 Table 2 와 같다.

Table 2. Objective along the constraint x + y = 10

| $x$ | $y$ | $f = xy$ |
| :-: | :-: | :------: |
| 3   | 7   | 21       |
| 4   | 6   | 24       |
| 5   | 5   | 25       |
| 6   | 4   | 24       |
| 7   | 3   | 21       |

$x = y = 5$ 에서 $f$ 가 가장 크므로 식 (1) 의 후보는 최댓값이다.
$C$ 를 10 에서 10.1 로 늘리면 $x = y = 5.05$ 이고 $f^\ast = 25.5025$ 이다.
$f^\ast$ 의 변화율은 $(25.5025 - 25) / 0.1 = 5.025$ 로 $\lambda = 5$ 에 가깝고, 식 (4) 와 맞는다.

## 6. Application

라그랑주 승수법은 머신러닝, 딥러닝, 반도체 공정 최적화 같은 엔지니어링 전반에서 조건이 걸린 최적화 문제를 풀 때 쓰인다.

### 6.1 Engineering Constraints

실무에는 반드시 지켜야 하는 조건이 걸린 최적화 문제가 많다.

- 원가 예산 1000 만 원을 정확히 맞추면서 성능을 최대화한다. 예산이 $g(x, y) = C$ 의 $C$ 이며, 식 (4) 의 $\lambda$ 는 예산 1 원을 늘릴 때 성능이 늘어나는 양이다.
- 센서 작동 범위 $x^2 + y^2 \le 1$ 을 벗어나지 않으면서 model 오차를 최소화한다. 이 제약 조건은 부등식이므로 식 (1) 대신 KKT 조건으로 푼다.

### 6.2 Support Vector Machine

SVM (support vector machine) 은 마진 (margin) 을 최대화하며, 데이터가 각 class 영역을 침범하지 않아야 한다는 제약 조건을 둔다.
이 제약 조건은 부등식이어서 KKT 조건을 쓰며, 라그랑주 함수를 $\lambda$ 에 대해 다시 쓴 dual 문제로 바꾸어 푼다.

### 6.3 Regularization

손실 함수 (loss function) 에 L2 regularization (Ridge) 항 $\lambda \lVert \mathbf{w} \rVert^2$ 를 더하는 것은, 가중치 vector $\mathbf{w}$ 의 크기를 제한한 최적화의 라그랑주 함수와 같은 꼴이다.
이때 regularization 계수가 식 (2) 의 $\lambda$ 에 해당한다.

### 6.4 Semiconductor Process Optimization

공정 비용이나 시간 제약 안에서 wafer 품질이 최적인 recipe 를 수학적으로 설계할 때 쓴다.
비용이나 시간의 상한이 $C$ 이고, $\lambda$ 는 그 상한을 늘릴 때 품질이 좋아지는 양이다.

### 6.5 Breaking Conditions

아래 조건에서는 식 (1) 만으로 최적점을 구할 수 없다.

- **부등식 제약 조건**: $g(x, y) \le C$ 이면 최적점이 경계 안에 있을 수 있으므로 KKT 조건을 쓴다.
- **$\nabla g = 0$ 인 점**: 접선 방향이 정해지지 않아 식 (1) 이 그 점의 극값을 놓친다. $\nabla g = 0$ 인 점은 따로 $f$ 의 값을 확인한다.
- **미분할 수 없는 $f$ 나 $g$**: gradient 가 없으므로 식 (1) 을 쓸 수 없다.
- **여러 후보**: 식 (1) 의 해는 최댓값, 최솟값, saddle point 를 모두 포함하므로 후보마다 $f$ 의 값을 견준다.

---

## Appendix A. Terminology

- **Dual problem**: 원래 최적화 문제를 라그랑주 승수에 대한 최적화 문제로 바꾼 것.
- **Gradient**: 각 변수에 대한 편미분을 모은 vector 이며, 함수값이 가장 빠르게 증가하는 방향을 가리킨다.
- **KKT conditions**: Karush-Kuhn-Tucker 조건. 부등식 제약 조건이 있는 최적화 문제의 최적점이 만족해야 하는 조건.
- **Ridge**: 가중치 제곱합에 비례하는 항을 손실 함수에 더하는 L2 regularization.
- **Saddle point**: gradient 가 0 이지만 극댓값도 극솟값도 아닌 점.

## Appendix B. Derivation

제약 곡선 $g(x, y) = C$ 위의 점 $(x, y)$ 에서 $\nabla g \ne 0$ 이라 하고, 그 점의 접선 방향을 $\mathbf{t}$ 라 한다.

```math
\begin{aligned}
&\nabla g \cdot \mathbf{t} = 0 && \text{g is constant along the curve} \\
&\nabla f \cdot \mathbf{t} = 0 && \text{directional derivative of f along the curve is 0 at an extremum} \\
&\nabla f \parallel \nabla g && \text{in 2D, two vectors orthogonal to the same nonzero t are parallel} \\
&\nabla f = \lambda \nabla g && \text{nabla g is nonzero, so f's gradient is a multiple of it}
\end{aligned}
\hspace{5em} (6)
```

식 (4) 는 최적점 $(x^\ast(C), y^\ast(C))$ 에서 $f^\ast(C) = f(x^\ast, y^\ast)$ 를 $C$ 로 미분하여 얻는다.

```math
\begin{aligned}
\frac{d f^\ast}{d C} &= \nabla f \cdot \frac{d (x^\ast, y^\ast)}{d C} && \text{chain rule} \\
&= \lambda \nabla g \cdot \frac{d (x^\ast, y^\ast)}{d C} && \text{eq. (1)} \\
&= \lambda \frac{d}{d C} g(x^\ast, y^\ast) = \lambda \frac{d C}{d C} = \lambda && \text{chain rule, constraint holds at the optimum}
\end{aligned}
\hspace{5em} (7)
```
