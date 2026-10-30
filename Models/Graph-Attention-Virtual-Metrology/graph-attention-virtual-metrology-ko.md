# Graph Attention-Based Virtual Metrology
Rev. 0 | Created: 2026-10-29 | Updated: 2026-10-29 23:02 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Comparison](#31-comparison)
- [4. Principle](#4-principle)
  - [4.1 Problem Formulation](#41-problem-formulation)
  - [4.2 Parameter Node Embedding](#42-parameter-node-embedding)
  - [4.3 Parameter-to-Layer Graph Attention](#43-parameter-to-layer-graph-attention)
  - [4.4 Thickness Prediction](#44-thickness-prediction)
- [5. Application](#5-application)
  - [5.1 Assumptions](#51-assumptions)
  - [5.2 Settings](#52-settings)
  - [5.3 Result](#53-result)
  - [5.4 Attention Interpretation](#54-attention-interpretation)
  - [5.5 Failure Conditions](#55-failure-conditions)
- [6. Further Work](#6-further-work)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

- **Problem Statement**: 기존 virtual metrology (VM) model 은 공정 변수 사이의 구조적 의존 관계를 표현하지 못하고, 어느 변수가 예측을 움직였는지 보여 주지 못한다.
- **Goal**: Han, Lee, Ko 의 graph attention 기반 VM [[1](#ref-1)] 이 graph 를 어떻게 구성하고, 그 graph 가 신경망 layer 의 directed acyclic graph (DAG) 와 무엇이 다른지를 가릴 수 있게 한다.
- **Non-Goal**: Graph neural network 일반론, 논문이 공개하지 않은 공정 조건과 두께 값은 다루지 않는다.

## 2. Summary

논문의 graph 는 데이터 개체 사이의 관계를 나타내는 directed bipartite graph 이다. Process step–parameter pair 가 parameter node, film layer 가 layer node 이고, parameter node 에서 layer node 로 가는 edge 의 가중치를 wafer 마다 attention 으로 구한다 [[1](#ref-1)].
신경망 layer 의 DAG 는 연산의 순서를 나타내므로, 두 graph 는 node 와 edge 가 가리키는 대상이 다르다.
산업 증착 공정 자료에서 제안 model 의 test coefficient of determination ($R^2$) 는 0.7227 로 MLP baseline 의 0.7106 보다 높고, attention 가중치로 film layer 마다 영향이 큰 step 과 parameter 의 순위를 매길 수 있다.

## 3. Taxonomy and its Hierarchy

Deep learning model 에 나오는 graph 는 node, edge, edge 가중치, 가중치의 의미의 네 축에서 data graph 와 computation graph 로 나뉜다. 네 축과 축마다의 값은 아래 [Fig 1](#fig-1) 과 같다.

```text
Graph in a deep learning model
|
+-- Node
|   +-- Data entity ...................... step-parameter pair, film layer
|   +-- Operation ........................ Conv1D, temporal mean, MLP
|
+-- Edge
|   +-- Dependency between entities ...... parameter -> film layer
|   +-- Tensor flow between operations ... output of one layer -> next layer
|
+-- Edge weight
|   +-- Computed per sample .............. attention weight
|   +-- Fixed by design .................. connection exists or not
|
+-- Meaning of edge weight
    +-- Importance of a parameter ........ read as a heatmap
    +-- None beyond execution order
```

<a id="fig-1"></a>Fig 1. Four axes of a graph in a deep learning model

각 축의 첫 값이 data graph, 둘째 값이 computation graph 이다. 논문의 graph 는 네 축에서 모두 첫 값을 갖는다.

VM model 의 계층은 statistical learning 에서 출발하며, 한 단계 내려갈 때마다 표현할 수 있는 관계가 넓어진다 [[1](#ref-1)]. 계층은 아래 [Fig 2](#fig-2) 와 같다.

```text
Statistical learning ...................... linear regression, PLS, SVM
|   limit: complex nonlinear relations
|
+-- Ensemble learning ..................... random forest, gradient boosting
    |   adds: high-dimensional, nonlinear relations
    |
    +-- Deep learning ..................... CNN, RNN, attention
        |   adds: temporal patterns in sensor traces
        |   loses: which parameter drove the prediction
        |
        +-- Graph attention ............... this paper
                adds: step-parameter nodes and parameter-to-layer weights
```

<a id="fig-2"></a>Fig 2. Virtual metrology model families

- Statistical learning 에서 ensemble learning 으로 내려가면 고차원 sensor 자료와 비선형 관계를 다룰 수 있다.
- Ensemble learning 에서 deep learning 으로 내려가면 sensor trace 의 시간 패턴을 학습하지만, 어느 parameter 가 예측을 움직였는지 알 수 없는 black-box predictor 가 된다.
- Deep learning 에서 graph attention 으로 내려가면 입력을 step–parameter node 단위로 나누고, layer 마다 node 의 가중치를 얻어 해석에 쓴다.

### 3.1 Comparison

논문의 graph 와 신경망 layer 의 DAG 는 아래 Table 1 과 같이 다르다. 논문은 DAG 를 언급하지 않으므로, 오른쪽 열은 논문의 model 을 이루는 Conv1D, 평균, attention, MLP 의 연산 순서를 기준으로 적었다.

Table 1. Graph in the paper versus layer DAG

| Aspect            | Graph in the paper                                 | Layer DAG                      |
| :---------------: | :------------------------------------------------: | :----------------------------: |
| Node              | Step–parameter pair $(s, p)$, film layer $l$       | 연산 (Conv1D, 평균, MLP)       |
| Edge              | Parameter node 에서 layer node 로 가는 의존 관계   | 한 연산의 출력이 다음 연산으로 |
| Structure         | Directed bipartite graph                           | 입력에서 출력으로 가는 DAG     |
| Connection        | 모든 parameter node 와 모든 layer node             | 설계 단계에서 고정             |
| Edge weight       | Wafer 마다 식 (5) 의 attention 으로 계산           | 없음                           |
| Meaning of weight | Parameter $(s, p)$ 가 layer $l$ 두께에 갖는 중요도 | 없음                           |

- 논문 model 의 연산은 Conv1D, 시간 평균, attention, MLP 의 순서로 이어지며, 이 순서 자체도 DAG 이다. 논문이 "graph" 라고 부르는 것은 attention 단계 안의 parameter node 와 layer node 의 관계다.
- Edge 의 가중치는 미리 정한 adjacency matrix 없이 학습으로 정해진다. 모든 parameter node 가 모든 layer node 와 연결되고, 연결의 강도만 wafer 마다 달라진다.
- Parameter node 끼리 정보를 주고받는 message passing 은 없다. 정보는 parameter node 에서 layer node 로 한 번만 간다.
- 식 (5) 는 learnable layer query $\mathbf{q}_l$ 이 parameter node 에 거는 softmax attention 이다. 연산만 보면 transformer 의 cross-attention 과 같은 꼴이고, 논문은 이것을 parameter–layer 관계로 해석하여 graph attention 이라 부른다.

## 4. Principle

Model 은 parameter node embedding, parameter-to-layer graph attention, thickness prediction 의 세 단계로 wafer 하나의 sensor trace 에서 film layer $L$ 개의 두께를 예측한다 [[1](#ref-1)].

### 4.1 Problem Formulation

증착 공정은 process step $S$ 개로 이루어지고, step 마다 chamber temperature, pressure, precursor gas flow 같은 equipment parameter $P$ 개를 기록한다. Step $s$ 에서 parameter $p$ 의 sensor trace 는 길이 $T_s$ 의 time series $\mathbf{x}_{s,p} \in \mathbb{R}^{T_s}$ 이다.

```math
X = \{\mathbf{x}_{s,p} \in \mathbb{R}^{T_s} \mid s = 1, \dots, S,\ p = 1, \dots, P\} \hspace{19em} (1)
```

Wafer 하나의 입력 $X$ 에서 film layer $L$ 개의 두께 벡터 $`\hat{\mathbf{y}} = [\hat{y}_1, \dots, \hat{y}_L]^\top \in \mathbb{R}^L`$ 을 구하는 함수 $`\hat{\mathbf{y}} = f_\theta(X)`$ 를 학습한다. $\theta$ 는 학습할 model parameter 이다. Step 마다 $T_s$ 가 달라 입력 길이가 고르지 않고, 두께는 서로 다른 step 의 parameter 가 함께 정한다.

### 4.2 Parameter Node Embedding

Step–parameter pair $(s, p)$ 하나가 parameter node 하나이며, node 의 embedding 은 Conv1D 와 시간 평균으로 구한다.

```math
\mathbf{z}_{s,p} = \mathrm{Conv1D}(\mathbf{x}_{s,p}) \in \mathbb{R}^{T_s \times d_h} \hspace{19em} (2)
```

```math
\mathbf{h}_{s,p} = \frac{1}{T_s} \sum_{t=1}^{T_s} \mathbf{z}_{s,p}^{(t)} \in \mathbb{R}^{d_h} \hspace{19em} (3)
```

```math
N = \{\mathbf{h}_{s,p} \mid s = 1, \dots, S,\ p = 1, \dots, P\} \in \mathbb{R}^{(S \cdot P) \times d_h} \hspace{19em} (4)
```

- $d_h$ 는 embedding 차원이다.
- Conv1D 는 상승, 하강, 요동, 안정 같은 국소 시간 패턴을 잡는다. 논문은 압력이 서서히 오르는 패턴이 deposition kinetics 의 변화를 나타낼 수 있다고 예를 든다.
- 식 (3) 의 평균은 길이 $T_s$ 가 다른 step 의 feature 를 같은 길이 $d_h$ 의 벡터로 맞춘다.
- 모든 parameter 를 한 벡터로 합쳐 모델링하는 대신 step–parameter pair 를 따로 떨어진 node 로 두어, attention 이 다룰 입력 단위를 정한다.

### 4.3 Parameter-to-Layer Graph Attention

Parameter node 와 layer node 로 이루어진 directed bipartite graph 에서, layer node 는 parameter node 의 embedding 을 attention 가중치로 합쳐 자기 표현을 만든다. Layer node $l$ 은 learnable embedding $\mathbf{q}_l \in \mathbb{R}^{d_h}$ 이다.

```math
\alpha_{l,(s,p)} = \frac{\exp\left(\mathbf{q}_l^\top \mathbf{h}_{s,p} / \sqrt{d_h}\right)}{\sum_{s'=1}^{S} \sum_{p'=1}^{P} \exp\left(\mathbf{q}_l^\top \mathbf{h}_{s',p'} / \sqrt{d_h}\right)} \hspace{19em} (5)
```

```math
\mathbf{e}_l = \sum_{s=1}^{S} \sum_{p=1}^{P} \alpha_{l,(s,p)} \mathbf{h}_{s,p} \in \mathbb{R}^{d_h} \hspace{19em} (6)
```

```math
E = \{\mathbf{e}_l \mid l = 1, \dots, L\} \in \mathbb{R}^{L \times d_h} \hspace{19em} (7)
```

- $\alpha_{l,(s,p)}$ 는 film layer $l$ 의 두께에 parameter $(s, p)$ 가 갖는 상대 중요도이며, layer 하나에 대해 모든 parameter node 의 합이 1 이다.
- 논문은 chamber temperature 가 dielectric layer 에, gas flow rate 가 metal layer 에 더 중요할 수 있다고 예를 든다.
- 논문은 $\alpha_{l,(s,p)}$ 를 Granger causality 에서 착안한 parameter 에서 layer 로 가는 방향의 의존 관계로 해석한다. 앞 step 의 parameter 변화가 뒤 step 의 film 두께에 기여하는 정도라는 뜻이다.
- 식 (5) 는 attention head 하나로 적혀 있고, 실험 설정의 attention head 수는 4 이다 (꼭지 5.2).

### 4.4 Thickness Prediction

Layer 표현 $`\mathbf{e}_l`$ 을 multilayer perceptron (MLP) $`f_\theta`$ 에 넣어 layer $l$ 의 두께 $`\hat{y}_l = f_\theta(\mathbf{e}_l)`$ 를 예측한다. Loss 는 layer 마다 표준화한 오차의 평균 제곱이다.

```math
\mathcal{L} = \frac{1}{L} \sum_{l=1}^{L} \left( \frac{\hat{y}_l - y_l}{\sigma_l} \right)^2 \hspace{19em} (8)
```

$y_l$ 은 계측한 두께, $\sigma_l$ 은 학습 자료에서 layer $l$ 두께의 표준편차다. Film layer 마다 두께의 크기가 달라, $\sigma_l$ 로 나누어 layer 사이의 크기를 맞춘다. Layer embedding $\{\mathbf{q}_l\}$ 과 신경망 parameter $\theta$ 는 식 (8) 로 함께 학습한다.

## 5. Application

논문은 Intel 생산 wafer 의 다층 chemical vapor deposition (CVD) 자료로 model 을 검증했다 [[1](#ref-1)].

### 5.1 Assumptions

- 공정이 step 으로 나뉘고, step 마다 같은 parameter 의 sensor trace 가 기록된다. 논문 자료는 10 Hz 로 기록했다.
- Parameter 를 미리 골라 둔다. 논문은 equipment 변수 1,000 개 이상에서 domain 지식으로 30 개를 골랐다.
- Wafer 마다 film layer 두께를 계측한다. 논문은 spectroscopic ellipsometry 로 wafer 당 10 곳 이상을 계측했다.
- 같은 lot 의 wafer 가 학습 자료와 평가 자료에 나뉘지 않도록 lot 단위로 자료를 나눈다.

### 5.2 Settings

Table 2. Dataset and training settings

| Item               | Value                                              |
| :----------------: | :------------------------------------------------: |
| Wafer              | 약 1,000 장 중 804 장 사용                         |
| Split              | Train 558, validation 120, test 126 (lot 단위)     |
| Film type          | Film A 와 Film B 가 layer 의 절반씩                |
| Parameter node     | Parameter 약 30 개 × sub-step 35 개, 1,000 개 이상 |
| Embedding $d_h$    | 64                                                 |
| Attention head     | 4                                                  |
| Conv1D kernel size | 5                                                  |
| Optimizer          | Adam, learning rate $1 \times 10^{-4}$             |
| Epoch, batch size  | 100, 16                                            |
| Model 선택         | Validation loss 가 가장 낮은 checkpoint            |

### 5.3 Result

Table 3. Performance comparison of model configurations

| Configuration                                   | $R^2$  |
| :---------------------------------------------: | :----: |
| Proposed model (full)                           | 0.7227 |
| Proposed model w/o parameter-to-layer attention | 0.7061 |
| MLP (baseline)                                  | 0.7106 |

- $R^2$ 는 Film B 두께 네 개의 test $R^2$ 를 평균한 값이다.
- Ablation 은 parameter-to-layer attention 을 빼서 parameter 와 layer 사이의 명시적 연결을 없앤 model 이다.
- MLP baseline 은 같은 입력 feature 에 hidden layer 128, 64 개와 ReLU 를 쓴 model 이다. 학습 자료가 작아 구조를 단순하게 두었다.
- 논문은 학습 loss 와 validation loss 의 차이를 lot 단위 분할에서 생긴 분포 차이로 본다.

### 5.4 Attention Interpretation

Test 자료 전체에서 평균한 $\alpha_{l,(s,p)}$ 를 heatmap 으로 그리면 layer 마다 영향을 주는 step 과 parameter 가 드러난다.

- 처음 step 과 마지막 step, 특히 temperature 와 pressure 의 ramping 구간에 attention 이 높다.
- 중간 step 에서는 모든 layer 가 같은 주기적 패턴을 보인다. 온도와 압력이 안정된 뒤에는 reactive gas flow 같은 변수가 attention 의 차이를 만든다.
- Layer 마다 상위 $k$ 개만 남기면 소수의 parameter 가 두께를 정한다.
- Film B layer 는 처음 ramping 구간에만 반응하고, Film A3 와 top film 은 마지막 ramping 구간에도 반응한다.
- 상위 10 개 parameter 를 보면 모든 layer 가 temperature 관련 parameter 와 연관이 크고, film A2 를 뺀 모든 layer 에서 처음 step 의 영향이 가장 크다.

### 5.5 Failure Conditions

- 제안 model 과 MLP baseline 의 $R^2$ 차이는 0.0121 이고, 결과는 test 자료 126 장의 Film B 네 layer 에서만 보고되었다.
- Attention 가중치가 물리적 원인이라는 것은 검증되지 않았다. 논문은 이 검증을 Further Work 로 남겼다.
- 식 (3) 의 평균 때문에 Conv1D 가 보는 구간보다 긴 step 안의 시간 순서는 embedding 에 남지 않는다.
- Layer 마다 두께 하나를 예측하며, wafer 당 10 곳 이상 계측한 두께의 분포는 예측하지 않는다.
- Process drift 와 변동에 대한 강건성은 시험하지 않았다.

## 6. Further Work

논문이 제시한 다음 방향은 세 가지다 [[1](#ref-1)].

- **Domain 전문가와 attention 검증**: 학습한 parameter–layer 관계를 물리, 화학 원리와 대조한다. Attention 가중치를 근본 원인 분석에 쓰려면 이 검증이 먼저 필요하다. 공정 전문가의 검토와 layer 마다의 알려진 증착 기구가 필요하다.
- **Wafer 안 두께 분포 예측**: Layer 두께 하나 대신 wafer 안의 두께 분포와 spatial non-uniformity 를 예측한다. Wafer 당 10 곳 이상의 계측값이 이미 있어 target 을 바꾸면 된다.
- **Physics-informed 제약 추가**: 공정 지식이나 physics-informed 제약을 model 에 넣어 process drift 아래에서 강건성과 일반화를 높인다. Drift 가 있는 기간의 자료가 필요하다.

## References

<a id="ref-1"></a>[1] Han, T., Lee, S. K. and Ko, H. (2026). [Graph Attention-Based Virtual Metrology for Film Deposition Processes in Semiconductor Manufacturing](https://doi.org/10.48550/arXiv.2606.00923). *arXiv preprint*, arXiv:2606.00923.

---

## Appendix A. Terminology

- **ablation**: Model 의 한 구성 요소를 빼고 다시 학습하여 그 요소의 기여를 재는 실험.
- **attention**: Query 와 key 의 내적을 softmax 로 정규화한 가중치로 value 를 합치는 연산.
- **bipartite graph**: Node 가 두 집합으로 나뉘고 edge 가 서로 다른 집합의 node 사이에만 있는 graph.
- **coefficient of determination ($R^2$)**: 1 에서 잔차 제곱합을 총 제곱합으로 나눈 값을 뺀 것.
- **computation graph**: 연산을 node, 연산 사이의 tensor 흐름을 edge 로 둔 graph.
- **Conv1D**: 1 차원 convolution. 시간 축을 따라 kernel 을 움직여 국소 패턴을 추출한다.
- **cross-attention**: Query 와 key·value 가 서로 다른 집합에서 오는 attention.
- **CVD**: Chemical vapor deposition. 기체 원료의 화학 반응으로 wafer 위에 film 을 쌓는 공정.
- **DAG**: Directed acyclic graph. 방향이 있고 순환이 없는 graph.
- **data graph**: 데이터 개체를 node, 개체 사이의 관계를 edge 로 둔 graph.
- **Granger causality**: 한 time series 의 과거 값이 다른 time series 의 예측을 개선하면 앞의 것이 뒤의 것의 원인이라고 보는 통계적 정의.
- **layer node**: Film layer 하나를 나타내는 node. Learnable embedding $`\mathbf{q}_l`$ 로 표현한다.
- **message passing**: Graph neural network 에서 node 가 이웃 node 의 표현을 받아 자기 표현을 갱신하는 과정.
- **MLP**: Multilayer perceptron. Fully connected layer 를 쌓은 신경망.
- **parameter node**: Process step–parameter pair $(s, p)$ 하나를 나타내는 node. Embedding $`\mathbf{h}_{s,p}`$ 로 표현한다.
- **spectroscopic ellipsometry**: 반사광의 편광 변화로 film 두께와 광학 특성을 재는 비파괴 계측.
- **VM**: Virtual metrology. 장비 sensor 자료로 공정 후 wafer 특성을 예측하는 방법.
