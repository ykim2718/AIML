# Inverse Model Quick Start
Rev. 12 | Created: 2026-10-22 | Updated: 2026-10-23 00:13 UTC

## 1. Purpose

- **Problem Statement**: Inverse model 과 model inversion 은 실무 문서에서 구별 없이 쓰여, 목표 출력에서 입력을 되찾는 방식 가운데 어느 것을 뜻하는지 가릴 수 없다.
- **Goal**: 두 용어가 가리키는 구축 방식을 구별하고, 추론 속도와 multi-solution 에서의 안정성을 기준으로 둘 중 하나를 고를 수 있게 한다.
- **Non-Goal**: 최적화 algorithm 의 세부, ill-posedness 이론, model 종류별 inversion 방법의 세부는 다루지 않는다.

## 2. Taxonomy

Problem 과 그 해법의 계층은 아래 Fig 1 과 같다.

```text
Problem
  +-- Forward problem                              given x, find y
  |     +-- Physical model                         computes y from physical laws
  |     +-- Surrogate model    y = f(x)            data-driven model that replaces the physical model
  +-- Inverse problem                              given y, find x
        +-- Inverse model      x = g(y)            learn g directly
        |     aka direct (learned) inversion, learned inverse map
        +-- Model inversion    y = f(x), for x     solve with the surrogate model f
              aka optimization-based (iterative) inversion
```
Fig 1. Forward problem, inverse problem and the two inverse problem methods

y 는 두께, x 는 recipe 이다. Forward problem 을 푸는 model f 는 physical model 을 대신해 (x, y) 쌍으로 학습한 surrogate model 이다. Inverse model 은 x = g(y) 의 g 를 데이터로 직접 학습해 x 를 바로 얻는다. Model inversion 은 surrogate model f 를 그대로 두고, 최적화로 y = f(x) 를 x 에 대해 푼다. Inverse model 은 문헌에서 direct (learned) inversion 또는 learned inverse map 으로, model inversion 은 optimization-based (iterative) inversion 으로도 부른다 [[1](#ref-1)]. 추론 속도가 우선이면 inverse model 을, multi-solution 이 있어 안정성이 우선이면 model inversion 을 쓴다.

## 3. Inverse Model and Model Inversion

두 방법은 학습하는 model 의 방향과 추론 때 반복하는 계산으로 갈린다.

### 3.1 Inverse Model

- **Definition**: 출력을 입력으로, 입력을 출력으로 받도록 처음부터 방향을 뒤집어 학습한 독립 data-driven model.
- **Inference**: Forward pass 한 번으로 recipe 를 역산.
- **Strength**: 데이터가 충분하면 빠른 추론.
- **Weakness**: 같은 두께를 만드는 recipe 조합이 여럿이면 (multi-solution) 불안정한 학습.

### 3.2 Model Inversion

- **Definition**: Forward model 을 가진 상태에서 gradient descent 같은 최적화나 탐색 algorithm 으로 입력 조건을 거꾸로 찾는 과정, 또는 그 system.
- **Inference**: 목표 두께가 나올 때까지 입력 recipe 를 바꿔 가며 최적값을 찾음.
- **Strength**: 물리 법칙과 forward model 의 예측 정밀도를 그대로 쓰므로 안정적.
- **Weakness**: 추론마다 최적화 연산을 거치므로 inverse model 보다 느림.

### 3.3 Comparison

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

## 4. Models and Libraries

Inverse model 은 y → x 방향의 model 하나와 그것을 학습할 library 를 쓰고, model inversion 은 surrogate model 과 그것을 거꾸로 풀 solver library 를 함께 쓴다. 방식별 model 과 library 는 Table 2 와 같으며, model 종류별 inversion 방법의 세부는 [[1](#ref-1)] 에 있다.

Table 2. Models and libraries by method

<table>
  <tr><th align="center">Method</th><th align="center">Model</th><th align="center">Library</th><th align="center">Note</th></tr>
  <tr><td align="center" rowspan="3">Inverse model</td><td align="center">PLS</td><td align="center"><code>sklearn.cross_decomposition.PLSRegression</code></td><td align="center">y 를 입력, x 를 출력으로 두고 학습</td></tr>
  <tr><td align="center">Neural network</td><td align="center"><code>sklearn.neural_network.MLPRegressor</code>, <code>torch.nn.Sequential</code></td><td align="center">y 를 입력, x 를 출력으로 두고 network 를 학습</td></tr>
  <tr><td align="center">cINN</td><td align="center"><code>FrEIA.framework.SequenceINN</code>, <code>FrEIA.modules.AllInOneBlock</code></td><td align="center">목표 y 를 조건으로 x 의 사후분포에서 표본을 뽑음</td></tr>
  <tr><td align="center" rowspan="7">Model inversion</td><td align="center">OLS / Ridge</td><td align="center"><code>sklearn.linear_model.LinearRegression</code>, <code>sklearn.linear_model.Ridge</code>, <code>numpy.linalg.pinv</code></td><td align="center">계수 행렬의 pseudo-inverse 로 닫힌 해를 구함</td></tr>
  <tr><td align="center">PLS / PCR</td><td align="center"><code>sklearn.cross_decomposition.PLSRegression</code>, <code>sklearn.decomposition.PCA</code>, <code>scipy.linalg.null_space</code></td><td align="center">score 공간에서 역해를 구하고 null space 로 해의 집합을 얻음</td></tr>
  <tr><td align="center">Random forest / GBM</td><td align="center"><code>sklearn.ensemble.RandomForestRegressor</code>, <code>sklearn.ensemble.GradientBoostingRegressor</code>, <code>scipy.optimize.differential_evolution</code></td><td align="center">미분 없이 입력 범위 안에서 전역 탐색</td></tr>
  <tr><td align="center">Neural network</td><td align="center"><code>torch.Tensor.requires_grad_</code>, <code>torch.optim.Adam</code></td><td align="center">입력 tensor 의 gradient 를 켜고 x 를 갱신</td></tr>
  <tr><td align="center">GP</td><td align="center"><code>botorch.models.SingleTaskGP</code>, <code>botorch.fit.fit_gpytorch_mll</code>, <code>botorch.optim.optimize_acqf</code></td><td align="center">GP 를 학습하고 acquisition function 을 최적화</td></tr>
  <tr><td align="center">임의의 surrogate model</td><td align="center"><code>scipy.optimize.minimize</code></td><td align="center">제약을 건 numerical optimization</td></tr>
  <tr><td align="center">임의의 surrogate model</td><td align="center"><code>pymc.Model</code>, <code>pymc.sample</code>, <code>emcee.EnsembleSampler</code></td><td align="center">x 의 사후분포를 sampling</td></tr>
</table>

Neural network 를 inverse model 로 쓰면 입력이 y 이고, model inversion 의 surrogate model 로 쓰면 입력이 x 이며 x 에 대한 gradient 로 해를 찾는다.

## References

<a id="ref-1"></a>
[1] yRocket. (2026). [Inverse Problem and Model Inversion](https://github.com/ykim2718/AIML/blob/main/Models/Inverse-Problem/inverse-problem-ko.md). *AIML repository*, GitHub.

---

## Appendix A. Terminology

- **acquisition function**: GP 의 예측값과 불확실성으로 다음에 시험할 입력을 고르는 함수.
- **cINN**: 목표 출력을 조건으로 받아 입력의 사후분포에서 표본을 뽑는 invertible neural network. Conditional invertible neural network 의 줄임말.
- **forward model**: 입력 조건 (recipe) 을 받아 출력 (두께) 을 예측하는 model.
- **forward pass**: 학습된 model 에 입력을 한 번 넣어 출력을 계산하는 것.
- **forward problem**: 입력 x 가 주어졌을 때 출력 y = f(x) 를 구하는 문제.
- **GBM**: 잔차를 차례로 학습하는 tree 를 더해 만드는 model. Gradient boosting machine 의 줄임말.
- **GP**: 함수에 사전분포를 두어 예측과 분산을 함께 주는 model. Gaussian process 의 줄임말.
- **inverse model**: 출력에서 입력으로 가는 사상을 데이터로 직접 학습한 model 이며, learned inverse map 또는 역모델이라고도 부른다.
- **inverse problem**: 목표 출력 y 가 주어졌을 때 그것을 만드는 입력 x 를 찾는 문제.
- **model inversion**: 이미 가진 forward model 을 거꾸로 풀어 목표 출력을 내는 입력을 찾는 방법.
- **multi-solution**: 같은 출력을 만드는 입력 조합이 여럿 있는 상태.
- **null space**: 입력을 그 방향으로 움직여도 예측이 바뀌지 않는 방향들의 집합.
- **OLS**: 잔차 제곱합을 최소화하는 선형 회귀. Ordinary least squares 의 줄임말.
- **PCR**: 주성분으로 입력을 줄인 뒤 회귀하는 model. Principal component regression 의 줄임말.
- **physical model**: 물리 법칙으로 입력에서 출력을 계산하는 model.
- **PLS**: 입력과 출력의 공분산이 큰 방향으로 latent 변수를 잡아 회귀하는 model. Partial least squares 의 줄임말.
- **pseudo-inverse**: 정방이 아니거나 특이한 행렬에 대해 최소 norm 해를 주는 역행렬.
- **recipe**: 공정 장비에 주는 입력 조건의 묶음.
- **Ridge**: 계수의 제곱합에 벌점을 주어 계수를 줄이는 선형 회귀.
- **score**: 입력을 PLS 나 PCA 의 latent 좌표계로 투영한 값.
- **surrogate model**: 원래 model (physical model 이나 직접 부를 수 없는 model) 을 대신해 (x, y) 쌍으로 학습한 data-driven forward model.
