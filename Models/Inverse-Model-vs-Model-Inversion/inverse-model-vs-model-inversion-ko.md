# Inverse Model and Model Inversion Quick Start
Rev. 3 | Created: 2026-10-22 | Updated: 2026-10-22 15:51 CDT

## 1. Purpose

- **Problem Statement**: Inverse model 과 model inversion (inversion model) 은 실무 문서에서 구별 없이 쓰여, 목표 출력에서 입력을 되찾는 방식 가운데 어느 것을 뜻하는지 가릴 수 없다.
- **Goal**: 두 용어가 가리키는 구축 방식을 구별하고, 추론 속도와 multi-solution 에서의 안정성을 기준으로 둘 중 하나를 고를 수 있게 한다.
- **Non-Goal**: 최적화 algorithm 의 세부, ill-posedness 이론, model 종류별 inversion 방법은 다루지 않는다.

## 2. Summary

Inverse model 은 출력에서 입력으로 방향을 뒤집어 처음부터 학습한 독립 model 이고, model inversion 은 이미 가진 forward model 을 최적화로 거꾸로 푸는 과정이다. 추론 속도가 우선이면 inverse model 을, multi-solution 이 있어 안정성이 우선이면 model inversion 을 쓴다.

## 3. Inverse Model and Model Inversion

Problem 은 forward problem 과 inverse problem 으로 나뉘고, inverse problem 을 푸는 방법은 inverse model 과 model inversion 두 가지다. 계층은 아래 Fig 1 과 같다.

```text
Problem
  +-- Forward problem          y = f(x)            given x, find y
  +-- Inverse problem                              given y, find x
        +-- Inverse model      x = g(y)            learn g directly
        +-- Model inversion    y = f(x), for x     solve with the forward model f
```
Fig 1. Forward problem, inverse problem and the two inverse problem methods

y 는 두께, x 는 recipe 이다. Inverse model 은 x = g(y) 의 g 를 데이터로 직접 학습해 x 를 바로 얻는다. Model inversion 은 forward model f 를 그대로 두고, 최적화로 y = f(x) 를 x 에 대해 푼다.

### 3.1 Inverse Model

- **Definition**: 출력을 입력으로, 입력을 출력으로 받도록 처음부터 방향을 뒤집어 학습한 독립 data-driven model.
- **Inference**: Forward pass 한 번으로 recipe 를 역산.
- **Strength**: 데이터가 충분하면 빠른 추론.
- **Weakness**: 같은 두께를 만드는 recipe 조합이 여럿이면 (multi-solution) 불안정한 학습.

### 3.2 Model Inversion

- **Definition**: Forward model 을 가진 상태에서 gradient descent 같은 최적화나 탐색 algorithm 으로 입력 조건을 거꾸로 찾는 과정, 또는 그 system.
- **Inference**: 목표 두께가 나올 때까지 입력 recipe 를 바꿔 가며 최적값을 찾는 inverse problem solving.
- **Strength**: 물리 법칙과 forward model 의 예측 정밀도를 그대로 쓰므로 안정적.
- **Weakness**: 추론마다 최적화 연산을 거치므로 inverse model 보다 느림.

### 3.3 Placement

두 방식의 차이는 Table 1 과 같다.

Table 1. Inverse model and model inversion

|      Aspect       |            Inverse model            |                    Model inversion                     |
| :---------------: | :---------------------------------: | :----------------------------------------------------: |
| Trained direction |             출력 → 입력             |              입력 → 출력 (forward model)               |
|     Inference     |         Forward pass 한 번          |                      반복 최적화                       |
|       Speed       |                빠름                 |                          느림                          |
|  Multi-solution   |            학습이 불안정            |          forward model 의 정밀도를 써서 안정           |
|     Use when      | 추론 속도가 우선이고 해가 하나일 때 | multi-solution 이 있거나 forward model 이 이미 있을 때 |

Use when 행은 Speed 행과 Multi-solution 행에서 나온다. Multi-solution 이 있으면 inverse model 의 학습이 불안정하므로 model inversion 을 쓰고, 해가 하나이고 추론을 반복해야 하면 inverse model 의 속도를 쓴다.

---

## Appendix A. Terminology

- **forward model**: 입력 조건 (recipe) 을 받아 출력 (두께) 을 예측하는 model.
- **forward pass**: 학습된 model 에 입력을 한 번 넣어 출력을 계산하는 것.
- **forward problem**: 입력 x 가 주어졌을 때 출력 y = f(x) 를 구하는 문제.
- **inverse model**: 출력을 입력으로, 입력을 출력으로 받도록 뒤집어 학습한 model. 역모델이라고도 부른다.
- **inverse problem**: 목표 출력 y 가 주어졌을 때 그것을 만드는 입력 x 를 찾는 문제.
- **inverse problem solving**: 정해진 출력을 내는 입력을 forward model 로 찾는 것.
- **inversion model**: Model inversion 을 수행하는 system 을 부르는 이름. Model inversion 과 같은 것을 가리킨다.
- **model inversion**: 이미 가진 forward model 을 최적화나 탐색으로 거꾸로 풀어 입력을 찾는 과정.
- **multi-solution**: 같은 출력을 만드는 입력 조합이 여럿 있는 상태.
- **recipe**: 공정 장비에 주는 입력 조건의 묶음.
