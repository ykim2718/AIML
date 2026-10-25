# Lagrange Multiplier Method
Rev. 0 | Created: 2026-10-25 | Updated: 2026-10-25 16:28 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Comparison](#31-comparison)
- [4. Principle](#4-principle)
  - [4.1 Geometric Intuition](#41-geometric-intuition)
  - [4.2 Lagrangian Function](#42-lagrangian-function)
- [5. Application](#5-application)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

- **Problem Statement**: 제약 조건 (constraint) 이 걸린 최적화 문제는 목적 함수 (objective function) 의 gradient 가 0 인 점을 찾는 방법으로 풀 수 없다.
- **Goal**: 등식 제약 조건을 만족하면서 목적 함수의 최댓값이나 최솟값을 주는 점을 연립방정식으로 구한다.
- **Non-Goal**: SVM 의 dual 문제 유도와 KKT 조건의 상세 증명은 다루지 않는다.

## 2. Summary

라그랑주 승수법 (Lagrange multiplier method) 은 제약 조건 $g(x, y) = C$ 를 만족하면서 $f(x, y)$ 의 극값을 구하는 최적화 기법이다.
최적점에서는 $\nabla f$ 와 $\nabla g$ 가 평행하며, 그 비례 상수가 라그랑주 승수 $\lambda$ 이다.
라그랑주 함수 $\mathcal{L}$ 를 $x, y, \lambda$ 로 편미분한 값을 0 으로 두고 연립방정식을 풀면 최적점이 나온다.

## 3. Taxonomy and its Hierarchy

최적화 문제는 제약 조건의 유무와 형태에 따라 세 가지로 갈리며, 각각을 푸는 방법이 정해져 있다.
제약 조건의 축과 그 값은 아래 [Fig 1](#fig-1) 과 같다.

```text
Constraint presence : none | present
Constraint form     : equality g(x, y) = C | inequality g(x, y) <= C
```

<a id="fig-1"></a>
Fig 1. Taxonomy of optimization problems

- Constraint presence 는 제약 조건이 있는지를 가른다.
- Constraint form 은 제약 조건이 있을 때만 의미가 있으며, 등식과 부등식으로 갈린다.

방법이 한 단계 내려갈 때마다 다룰 수 있는 제약 조건이 넓어지고 풀어야 할 조건이 늘어난다.
그 계층은 아래 [Fig 2](#fig-2) 와 같다.

```text
Unconstrained optimization
  condition : grad f = 0
  |
  v  adds equality constraint g = C
Lagrange multiplier method
  condition : grad f = lambda * grad g,  g = C
  |
  v  adds inequality constraint g <= C
KKT conditions
  condition : Lagrange condition + lambda >= 0 + complementary slackness
```

<a id="fig-2"></a>
Fig 2. Hierarchy of optimization methods

- 위에서 아래로 갈수록 제약 조건의 종류가 넓어진다.
- 아래 방법은 위 방법의 조건을 포함한다. 제약 조건이 없으면 라그랑주 조건은 $\nabla f = 0$ 으로 줄어든다.

### 3.1 Comparison

Table 1. Optimization methods by constraint

| Method                     | Constraint       | Condition to solve                                      |
| :------------------------: | :--------------: | :-----------------------------------------------------: |
| Gradient = 0               | 없음             | $\nabla f = 0$                                          |
| Lagrange multiplier method | 등식 $g = C$     | $\nabla f = \lambda \nabla g$, $g = C$                  |
| KKT conditions             | 부등식 $g \le C$ | 라그랑주 조건, $\lambda \ge 0$, complementary slackness |

센서 작동 범위 $x^2 + y^2 \le 1$ 처럼 부등식인 제약 조건은 KKT 조건으로 확장해 푼다.
원가 예산을 정확히 맞추는 조건처럼 등식인 제약 조건은 라그랑주 승수법으로 푼다.

## 4. Principle

제약 조건이 없으면 미분해서 gradient 가 0 이 되는 점을 찾으면 된다.
실무에서는 아래처럼 반드시 지켜야 하는 조건이 걸린 문제가 많으며, 라그랑주 승수법은 제약 조건이 걸린 문제의 최적점을 구하려고 고안되었다.

- 원가 예산 1000 만 원을 정확히 맞추면서 성능을 최대화한다.
- 센서 작동 범위 $x^2 + y^2 \le 1$ 을 벗어나지 않으면서 model 오차를 최소화한다.

### 4.1 Geometric Intuition

최적점은 목적 함수의 등고선과 제약 조건의 곡선이 접하는 점이다. 2 차원 공간에서 세 요소는 아래와 같다.

1. **목적 함수 $f(x, y)$ 의 등고선**: 지도에서 산의 높이를 나타내는 동심원이며, 올라가야 할 정상이 목표다.
2. **제약 조건 $g(x, y) = C$**: 반드시 따라 걸어야 하는 등산로다.
3. **접점 (tangency point)**: 등산로를 따라 걸을 때 $f$ 의 등고선과 등산로가 접하는 점이며, 제약 조건 아래에서 도달할 수 있는 가장 높은 점이다.

접점에서는 두 함수의 gradient 가 평행하다.

```math
\nabla f = \lambda \nabla g \hspace{19em} (1)
```

비례 상수 $\lambda$ 가 라그랑주 승수 (Lagrange multiplier) 이다.

### 4.2 Lagrangian Function

목적 함수에 제약 조건을 결합해 라그랑주 함수 $\mathcal{L}$ 를 정의한다.

```math
\mathcal{L}(x, y, \lambda) = f(x, y) - \lambda \left( g(x, y) - C \right) \hspace{19em} (2)
```

- $f(x, y)$: 최적화할 목적 함수이며, 성능이나 오차가 여기에 해당한다.
- $g(x, y) = C$: 반드시 만족해야 하는 제약 조건이다.
- $\lambda$: 제약 조건이 최적화 결과에 미치는 영향의 크기를 나타내는 parameter 이다.

식 (2) 를 $x, y, \lambda$ 로 각각 편미분하여 0 으로 둔 연립방정식은 아래와 같다.

```math
\begin{aligned}
\frac{\partial \mathcal{L}}{\partial x} &= \frac{\partial f}{\partial x} - \lambda \frac{\partial g}{\partial x} = 0 \\
\frac{\partial \mathcal{L}}{\partial y} &= \frac{\partial f}{\partial y} - \lambda \frac{\partial g}{\partial y} = 0 \\
\frac{\partial \mathcal{L}}{\partial \lambda} &= -\left( g(x, y) - C \right) = 0
\end{aligned}
\hspace{15em} (3)
```

식 (3) 의 앞 두 식은 식 (1) 과 같고, 마지막 식은 제약 조건 $g(x, y) = C$ 와 같다.
이 연립방정식을 풀면 제약 조건을 만족하는 최적의 $x, y$ 가 나온다.

## 5. Application

라그랑주 승수법은 머신러닝, 딥러닝, 반도체 공정 최적화 같은 엔지니어링 전반에서 조건이 걸린 최적화 문제를 풀 때 쓰인다.

- **SVM (Support Vector Machine)**: 마진 (margin) 을 최대화할 때, 데이터가 각 class 영역을 침범하지 않아야 한다는 제약 조건을 라그랑주 승수법과 dual 문제 변환으로 푼다.
- **Regularization**: 손실 함수 (loss function) 에 L2 regularization (Ridge) 을 더해 가중치의 크기를 제한하는 방법의 바탕에 같은 최적화 원리가 있다.
- **반도체 공정 최적화**: 공정 비용이나 시간 제약 안에서 wafer 품질이 최적인 recipe 를 수학적으로 설계할 때 쓴다.

---

## Appendix A. Terminology

- **Complementary slackness**: 부등식 제약 조건이 등식으로 성립하지 않으면 그 승수가 0 이어야 한다는 KKT 조건.
- **Dual problem**: 원래 최적화 문제를 라그랑주 승수에 대한 최적화 문제로 바꾼 것.
- **Gradient**: 각 변수에 대한 편미분을 모은 vector 이며, 함수값이 가장 빠르게 증가하는 방향을 가리킨다.
- **KKT conditions**: Karush-Kuhn-Tucker 조건. 부등식 제약 조건이 있는 최적화 문제의 최적점이 만족해야 하는 조건.
- **Ridge**: 가중치 제곱합에 비례하는 항을 손실 함수에 더하는 L2 regularization.
