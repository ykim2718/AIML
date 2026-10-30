# Graph Attention-Based Virtual Metrology for Film Deposition Processes in Semiconductor Manufacturing
Rev. 10 | Created: 2026-10-29 | Updated: 2026-10-30 06:30 UTC

- [1. Problem and Motivation](#1-problem-and-motivation)
- [2. Architecture and Key Mechanisms](#2-architecture-and-key-mechanisms)
  - [2.1 Model Design](#21-model-design)
  - [2.2 Key Equations](#22-key-equations)
  - [2.3 Pipeline / Flow](#23-pipeline--flow)
- [3. Experimental Setup](#3-experimental-setup)
- [4. Main Results and Ablation Studies](#4-main-results-and-ablation-studies)
- [5. Limitations and Critical Review](#5-limitations-and-critical-review)
- [6. Takeaways and Engineering Insights](#6-takeaways-and-engineering-insights)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Mathematical Derivation](#appendix-b-mathematical-derivation)
  - [B.1 Temporal Convolution (Eq. 1)](#b1-temporal-convolution-eq-1)
  - [B.2 Temporal Mean (Eq. 2)](#b2-temporal-mean-eq-2)
  - [B.3 Parameter-to-Layer Attention Weight (Eq. 3)](#b3-parameter-to-layer-attention-weight-eq-3)
  - [B.4 Layer Representation (Eq. 4)](#b4-layer-representation-eq-4)
  - [B.5 Standardized Loss (Eq. 5)](#b5-standardized-loss-eq-5)
  - [B.6 Coefficient of Determination (Eq. 6)](#b6-coefficient-of-determination-eq-6)
- [Appendix C. Application to [Wafer, Feature, Trace] Tensor](#appendix-c-application-to-wafer-feature-trace-tensor)
  - [C.1 Time Series Covered by the Paper](#c1-time-series-covered-by-the-paper)
  - [C.2 Cautions](#c2-cautions)

<br>

- **Authors**: Tao Han (Arizona State University, Intel Foundry), Suk Ki Lee (Arizona State University), Hyunwoong Ko (Arizona State University)
- **Venue / Year**: arXiv preprint, cs.CE (2026)
- **Link**: [arXiv:2606.00923](https://doi.org/10.48550/arXiv.2606.00923)
- **Code**: Not released
- **Read Date**: 2026-10-29

## 1. Problem and Motivation

- **What is the gap?**: 기존 virtual metrology (VM) model 은 공정 변수 사이의 상관만 학습하고, 서로 다른 process step 의 parameter 가 film 두께에 미치는 구조적 의존 관계를 표현하지 못한다. Deep learning 기반 VM 은 black-box predictor 여서 어느 parameter 가 예측을 움직였는지 공정 엔지니어가 알 수 없다 [[1](#ref-1)].
- **Core Hypothesis**: Process step–parameter pair 를 node 로 두고 film layer 마다 이 node 들에 attention 을 걸면, 예측 정확도가 오르고 attention 가중치가 parameter–layer 의존 관계를 드러낸다.

## 2. Architecture and Key Mechanisms

### 2.1 Model Design

Model 은 parameter node embedding, parameter-to-layer graph attention, thickness prediction 의 세 구성 요소로 wafer 하나의 sensor trace 에서 film layer $L$ 개의 두께를 예측한다.

- **Parameter node**: Process step $s$ 에서 equipment parameter $p$ 의 sensor trace 하나가 node 하나다. Step $S$ 개와 parameter $P$ 개에서 node $S \cdot P$ 개가 나오며, 논문 자료에서는 1,000 개를 넘는다.
- **Layer node**: Film layer $l$ 하나가 node 하나이며, learnable embedding $`\mathbf{q}_l \in \mathbb{R}^{d_h}`$ 로 표현한다.
- **Directed bipartite graph**: Edge 는 모든 parameter node 에서 모든 layer node 로 간다. Edge 의 가중치는 미리 정하지 않고 wafer 마다 식 (3) 의 attention 으로 구한다.
- **Thickness head**: Layer 표현 $`\mathbf{e}_l`$ 을 multilayer perceptron (MLP) 에 넣어 두께 하나를 예측한다.

논문은 film 두께에 대한 영향을 공정 조건과 film 형성 결과 사이의 의존 관계로 보고, 이 관계를 directed bipartite graph 로 나타냈다 (논문 3.3절). 공정 조건은 parameter node, 결과는 layer node 이며, edge 는 parameter node 에서 layer node 로만 간다. 이 구조에서 layer 마다 필요한 공정 정보만 골라 모으고, 가중치 $`\alpha_{l,(s,p)}`$ 를 parameter $(s, p)$ 가 layer $l$ 두께에 갖는 중요도로 읽는다. 논문은 이 해석 가능성으로 기존 deep learning VM 의 black-box 한계를 푸는 것을 주 목표로 밝혔다. Parameter node 끼리 연결하는 graph 같은 다른 구조와 견주어 bipartite graph 를 고른 이유는 논문에 적혀 있지 않다.

논문의 graph 와 신경망 layer 의 directed acyclic graph (DAG) 는 아래 Table 1 과 같이 다르다. 논문은 DAG 를 언급하지 않으며, 이 표는 리뷰어의 분석이다.

Table 1. Graph in the paper versus layer DAG

| Aspect            | Graph in the paper                                 | Layer DAG                      |
| :---------------: | :------------------------------------------------: | :----------------------------: |
| Node              | Step–parameter pair $(s, p)$, film layer $l$       | 연산 (Conv1D, 평균, MLP)       |
| Edge              | Parameter node 에서 layer node 로 가는 의존 관계   | 한 연산의 출력이 다음 연산으로 |
| Structure         | Directed bipartite graph                           | 입력에서 출력으로 가는 DAG     |
| Connection        | 모든 parameter node 와 모든 layer node             | 설계 단계에서 고정             |
| Edge weight       | Wafer 마다 식 (3) 의 attention 으로 계산           | 없음                           |
| Meaning of weight | Parameter $(s, p)$ 가 layer $l$ 두께에 갖는 중요도 | 없음                           |

- 논문 model 의 연산도 Conv1D, 시간 평균, attention, MLP 의 순서로 이어지는 DAG 이다. 논문이 "graph" 라고 부르는 것은 attention 단계 안의 parameter node 와 layer node 의 관계다.
- Parameter node 끼리 정보를 주고받는 message passing 은 없다. 정보는 parameter node 에서 layer node 로 한 번만 간다.
- 식 (3) 은 learnable query $`\mathbf{q}_l`$ 이 parameter node 에 거는 softmax attention 이어서, 연산만 보면 cross-attention 과 같은 꼴이다.

### 2.2 Key Equations

Step $s$ 에서 parameter $p$ 의 sensor trace 는 길이 $`T_s`$ 의 time series $`\mathbf{x}_{s,p} \in \mathbb{R}^{T_s}`$ 이고, wafer 하나의 입력은 그 모음 $`X = \{\mathbf{x}_{s,p}\}`$ 이다. Model 은 $X$ 에서 두께 벡터 $\hat{\mathbf{y}} \in \mathbb{R}^{L}$ 을 예측한다. 식 번호는 이 문서의 번호이고, 괄호 안에 논문의 식 번호를 함께 적었다.

Conv1D 로 trace 의 국소 시간 패턴을 추출한다 (논문 Eq. 5). 유도는 [B.1](#b1-temporal-convolution-eq-1) 에 있다.

```math
\mathbf{z}_{s,p} = \mathrm{Conv1D}(\mathbf{x}_{s,p}) \in \mathbb{R}^{T_s \times d_h} \hspace{19em} (1)
```

$`d_h`$ 는 embedding 차원이고, $`\mathbf{z}_{s,p}`$ 의 행 하나가 시점 하나의 feature 다.

Step 마다 다른 길이 $`T_s`$ 를 시간 평균으로 없애 parameter node 의 embedding 을 만든다 (논문 Eq. 6). 유도는 [B.2](#b2-temporal-mean-eq-2) 에 있다.

```math
\mathbf{h}_{s,p} = \frac{1}{T_s} \sum_{t=1}^{T_s} \mathbf{z}_{s,p}^{(t)} \in \mathbb{R}^{d_h} \hspace{19em} (2)
```

$`\mathbf{z}_{s,p}^{(t)}`$ 는 $`\mathbf{z}_{s,p}`$ 의 $t$ 번째 행이다.

Layer node $l$ 이 parameter node $(s, p)$ 에 거는 attention 가중치는 아래와 같다 (논문 Eq. 8). 유도는 [B.3](#b3-parameter-to-layer-attention-weight-eq-3) 에 있다.

```math
\alpha_{l,(s,p)} = \frac{\exp\left(\mathbf{q}_l^\top \mathbf{h}_{s,p} / \sqrt{d_h}\right)}{\sum_{s'=1}^{S} \sum_{p'=1}^{P} \exp\left(\mathbf{q}_l^\top \mathbf{h}_{s',p'} / \sqrt{d_h}\right)} \hspace{19em} (3)
```

$`\alpha_{l,(s,p)}`$ 는 film layer $l$ 의 두께에 parameter $(s, p)$ 가 갖는 상대 중요도이며, layer 하나에 대해 모든 parameter node 의 합이 1 이다.

Layer 표현은 parameter node embedding 을 attention 가중치로 합친 것이다 (논문 Eq. 9). 유도는 [B.4](#b4-layer-representation-eq-4) 에 있다.

```math
\mathbf{e}_l = \sum_{s=1}^{S} \sum_{p=1}^{P} \alpha_{l,(s,p)} \mathbf{h}_{s,p} \in \mathbb{R}^{d_h} \hspace{19em} (4)
```

두께는 MLP $`f_\theta`$ 로 $`\hat{y}_l = f_\theta(\mathbf{e}_l)`$ 와 같이 예측하고 (논문 Eq. 11), loss 는 layer 마다 표준화한 오차의 평균 제곱이다 (논문 Eq. 12). 유도는 [B.5](#b5-standardized-loss-eq-5) 에 있다.

```math
\mathcal{L} = \frac{1}{L} \sum_{l=1}^{L} \left( \frac{\hat{y}_l - y_l}{\sigma_l} \right)^2 \hspace{19em} (5)
```

$`y_l`$ 은 계측한 두께, $`\sigma_l`$ 은 학습 자료에서 layer $l$ 두께의 표준편차다. Layer embedding $`\mathbf{q}_l`$ 과 신경망 parameter $\theta$ 를 식 (5) 로 함께 학습한다.

평가지표는 layer 마다 원래 단위로 계산한 coefficient of determination ($R^2$) 이다 (논문 Eq. 13). 유도는 [B.6](#b6-coefficient-of-determination-eq-6) 에 있다.

```math
R_l^2 = 1 - \frac{\sum_{i=1}^{N} (y_{i,l} - \hat{y}_{i,l})^2}{\sum_{i=1}^{N} (y_{i,l} - \bar{y}_l)^2} \hspace{19em} (6)
```

$N$ 은 평가 wafer 수, $`y_{i,l}`$ 은 wafer $i$ 의 layer $l$ 두께, $`\bar{y}_l`$ 은 그 평균이다.

### 2.3 Pipeline / Flow

Wafer 하나의 입력이 두께 예측이 되기까지의 흐름은 아래 [Fig 1](#fig-1) 과 같다.

```text
sensor traces X = { x_(s,p) },  S steps x P parameters
        |
        |  Conv1D per trace .................. Eq. (1), T_s x d_h
        v
temporal features z_(s,p)
        |
        |  mean over time .................... Eq. (2), d_h
        v
parameter nodes h_(s,p),  S*P nodes
        |
        |  attention from layer query q_l .... Eq. (3), (4)
        v
layer representations e_l,  L nodes
        |
        |  MLP per layer
        v
predicted thickness y_hat_l,  trained with Eq. (5)
```

<a id="fig-1"></a>Fig 1. Forward pass of the graph attention VM model

- Parameter node 까지는 trace 마다 따로 계산하며, 같은 Conv1D 가중치를 모든 trace 에 쓰는지는 논문에 적혀 있지 않다.
- Attention 단계에서 parameter node $S \cdot P$ 개가 layer node $L$ 개로 모인다.

## 3. Experimental Setup

- **Datasets**: Intel 생산 wafer 의 다층 chemical vapor deposition (CVD) 공정 자료다. Sensor trace 는 10 Hz 로 기록했고, equipment 변수 1,000 개 이상에서 domain 지식으로 parameter 30 개를 골랐다. 두께는 spectroscopic ellipsometry 로 wafer 당 10 곳 이상을 계측했다. 같은 lot 의 wafer 가 학습과 평가에 나뉘지 않도록 lot 단위로 자료를 나눴다.
- **Baselines**: MLP baseline 은 같은 입력 feature 에 hidden layer 128, 64 개와 ReLU 를 쓴다. 학습 자료가 작아 구조를 단순하게 두었다. 제안 model 에서 parameter-to-layer attention 을 뺀 ablation model 도 비교한다.
- **Evaluation Metrics**: 식 (6) 의 $R^2$ 를 layer 마다 원래 단위로 계산하고, Film B 두께 네 개의 값을 평균한다.

학습 설정은 아래 Table 2 와 같다.

Table 2. Dataset and training settings

| Item               | Value                                              |
| :----------------: | :------------------------------------------------: |
| Wafer              | 약 1,000 장 중 804 장 사용                         |
| Split              | Train 558, validation 120, test 126 (lot 단위)     |
| Film type          | Film A 와 Film B 가 layer 의 절반씩                |
| Parameter node     | Parameter 약 30 개 × sub-step 35 개, 1,000 개 이상 |
| Embedding $`d_h`$  | 64                                                 |
| Attention head     | 4                                                  |
| Conv1D kernel size | 5                                                  |
| Optimizer          | Adam, learning rate $1 \times 10^{-4}$             |
| Epoch, batch size  | 100, 16                                            |
| Model 선택         | Validation loss 가 가장 낮은 checkpoint            |
| Hardware           | NVIDIA A100 80 GB, PyTorch                         |

## 4. Main Results and Ablation Studies

- **Performance**: 제안 model 의 $R^2$ 는 0.7227 로 MLP baseline 의 0.7106 보다 0.0121 높다 (논문 Table 1). 논문 Fig. 3 의 예측값 대 계측값 산점도에서 점이 대각선 가까이 모이며, 두께 값은 기밀 유지로 가려져 있다.
- **Ablation Study**: Parameter-to-layer attention 을 빼면 $R^2$ 가 0.7061 로 떨어져, 제안 model 보다 0.0166, MLP baseline 보다 0.0045 낮다 (논문 Table 1). 논문은 다른 구성 요소 (Conv1D, 시간 평균, attention head 수, kernel 크기) 를 바꾼 ablation 은 보고하지 않았다.
- **Qualitative / Quantitative Analysis**: Test 자료 전체에서 평균한 $`\alpha_{l,(s,p)}`$ 의 heatmap (논문 Fig. 4, 5, 6) 에서 아래가 드러난다.
  - 처음 step 과 마지막 step, 특히 temperature 와 pressure 의 ramping 구간에 attention 이 높다. 중간 step 에서는 모든 layer 가 같은 주기적 패턴을 보인다.
  - Layer 마다 상위 $k$ 개만 남기면 소수의 parameter 가 두께를 정한다. Film B layer 는 처음 ramping 구간에만, Film A3 와 top film 은 마지막 ramping 구간에도 반응한다.
  - 상위 10 개 parameter 에서 모든 layer 가 temperature 관련 parameter 와 연관이 크고, film A2 를 뺀 모든 layer 에서 처음 step 의 영향이 가장 크다.

Table 3. Performance comparison of model configurations

| Configuration                                   | $R^2$  |
| :---------------------------------------------: | :----: |
| Proposed model (full)                           | 0.7227 |
| Proposed model w/o parameter-to-layer attention | 0.7061 |
| MLP (baseline)                                  | 0.7106 |

값은 논문 Table 1 의 값이며, Film B 두께 네 개의 test $R^2$ 를 평균한 것이다.

## 5. Limitations and Critical Review

- **Computational Cost**: 논문은 학습에 쓴 GPU (NVIDIA A100 80 GB) 만 밝히고, FLOPs, 메모리 사용량, 학습 시간, 추론 시간은 보고하지 않았다. 리뷰어의 분석으로는, 식 (3) 의 계산량이 layer 수 $L$ 과 node 수 $S \cdot P$ 의 곱에 비례하므로 node 1,000 여 개에서는 부담이 작다.
- **Assumptions and Constraints**:
  - 논문이 밝힌 한계: 학습한 parameter–layer 관계는 물리, 화학 원리와 대조해 검증하지 않았다. Wafer 안의 두께 분포는 예측하지 않는다. Process drift 와 변동에 대한 강건성은 시험하지 않았다.
  - 논문이 밝힌 한계: 학습 loss 와 validation loss 의 차이를 lot 단위 분할에서 생긴 분포 차이로 본다.
  - 리뷰어의 분석: 제안 model 과 MLP baseline 의 차이 0.0121 은 test 자료 126 장의 Film B 네 layer 에서 한 번 잰 값이며, 반복 실험의 분산이 없어 차이가 우연인지 가릴 수 없다.
  - 리뷰어의 분석: Film A layer 의 $R^2$ 는 보고되지 않았다.
  - 리뷰어의 분석: 식 (2) 의 평균 때문에 Conv1D 가 보는 구간보다 긴 step 안의 시간 순서는 embedding 에 남지 않는다.
  - 리뷰어의 분석: 식 (3) 은 attention head 하나로 적혀 있지만 실험 설정의 head 수는 4 이며, head 를 어떻게 합쳐 $`\mathbf{e}_l`$ 과 heatmap 을 만드는지는 논문에 없다.
  - 리뷰어의 분석: Parameter 30 개를 domain 지식으로 미리 골랐으므로, attention 이 고른 parameter 는 그 30 개 안에서의 순위다.

## 6. Takeaways and Engineering Insights

이 꼭지는 리뷰어의 판단이다. 어느 project 에 차용할지는 정하지 않았으므로, 논문에서 떼어 쓸 수 있는 구성 요소만 적는다.

- **What can I use?**:
  - Step–parameter node embedding: 길이가 다른 step 별 trace 를 Conv1D 와 시간 평균으로 같은 길이 $`d_h`$ 의 벡터로 바꾼다 (식 (1), (2)).
  - Target 별 learnable query attention: Target 마다 query $`\mathbf{q}_l`$ 하나를 두면 target 별 입력 중요도 $`\alpha_{l,(s,p)}`$ 를 예측과 함께 얻는다 (식 (3), (4)).
  - 표준화한 다중 target loss: 크기가 다른 target 을 학습 자료의 $`\sigma_l`$ 로 나누어 한 loss 로 학습한다 (식 (5)).
  - Lot 단위 자료 분할: 같은 lot 의 wafer 가 학습과 평가에 나뉘지 않게 한다.
- **Implementation Notes**:
  - 식 (1) 의 출력 길이가 $`T_s`$ 이므로 Conv1D 에 same padding 을 써야 한다 ([B.1](#b1-temporal-convolution-eq-1)).
  - Step 마다 $`T_s`$ 가 다르면 batch 로 묶을 때 padding 한 시점을 식 (2) 의 평균에서 빼야 한다.
  - $`\sigma_l`$ 과 평균은 학습 자료에서만 구하고, $R^2$ 는 예측을 원래 단위로 되돌린 뒤 계산한다.
  - Heatmap 은 test wafer 마다 구한 $`\alpha_{l,(s,p)}`$ 를 평균한 것이고, 상위 $k$ 개만 남긴 그림은 나머지 값을 가린 것이다.
  - [Wafer, Feature, Trace] 모양의 tensor 자료에 적용하는 조건과 주의점은 [Appendix C](#appendix-c-application-to-wafer-feature-trace-tensor) 에 있다.

## References

<a id="ref-1"></a>[1] Han, T., Lee, S. K. and Ko, H. (2026). [Graph Attention-Based Virtual Metrology for Film Deposition Processes in Semiconductor Manufacturing](https://doi.org/10.48550/arXiv.2606.00923). *arXiv preprint*, arXiv:2606.00923.

---

## Appendix A. Terminology

- **ablation study**: Model 의 구성 요소를 하나씩 빼거나 바꾸어 다시 학습하고, 성능 변화로 그 요소의 기여를 재는 실험.
- **attention**: Query 와 key 의 내적을 softmax 로 정규화한 가중치로 value 를 합치는 연산.
- **baseline**: 제안 방법의 성능을 견주는 기준 model.
- **bipartite graph**: Node 가 두 집합으로 나뉘고 edge 가 서로 다른 집합의 node 사이에만 있는 graph.
- **coefficient of determination**: $R^2$. 1 에서 잔차 제곱합을 총 제곱합으로 나눈 값을 뺀 것.
- **Conv1D**: 1 차원 convolution. 시간 축을 따라 kernel 을 움직여 국소 패턴을 추출한다.
- **convex combination**: 합이 1 이고 모두 0 이상인 가중치로 벡터를 더한 것.
- **cross-attention**: Query 와 key·value 가 서로 다른 집합에서 오는 attention.
- **CVD**: Chemical vapor deposition. 기체 원료의 화학 반응으로 wafer 위에 film 을 쌓는 공정.
- **DAG**: Directed acyclic graph. 방향이 있고 순환이 없는 graph.
- **layer node**: 논문의 정의로 film layer 하나를 나타내는 node 이며, learnable embedding 으로 표현한다. 표준 용어로는 attention 의 learnable query 다.
- **message passing**: Graph neural network 에서 node 가 이웃 node 의 표현을 받아 자기 표현을 갱신하는 과정.
- **MLP**: Multilayer perceptron. Fully connected layer 를 쌓은 신경망.
- **parameter node**: 논문의 정의로 process step–parameter pair 하나를 나타내는 node 다. 표준 용어로는 attention 의 key 이자 value 다.
- **same padding**: Convolution 의 출력 길이가 입력 길이와 같도록 입력 양 끝에 값을 채우는 방식.
- **softmax**: 실수 벡터의 각 성분에 지수 함수를 취하고 그 합으로 나누어, 합이 1 인 양수 벡터로 만드는 함수.
- **spectroscopic ellipsometry**: 반사광의 편광 변화로 film 두께와 광학 특성을 재는 비파괴 계측.
- **VM**: Virtual metrology. 장비 sensor 자료로 공정 후 wafer 특성을 예측하는 방법.

## Appendix B. Mathematical Derivation

논문은 식의 유도를 싣지 않았다. 아래 유도는 모두 리뷰어의 유도이며, 논문의 정의와 식 번호에서 출발한다.

### B.1 Temporal Convolution (Eq. 1)

리뷰어의 유도이며, 식 (1) 이 출력 $`T_s \times d_h`$ 를 주는 조건을 보인다. 출발점은 kernel 크기 $K$, 입력 channel 하나, 출력 channel $`d_h`$ 개인 1 차원 convolution 의 정의다.

```math
z_{s,p}^{(t,c)} = b_c + \sum_{k=0}^{K-1} w_{c,k}\, x_{s,p}^{(t + k - \lfloor K/2 \rfloor)}, \quad c = 1, \dots, d_h \hspace{19em} (7)
```

- $`w_{c,k}`$ 와 $`b_c`$ 는 출력 channel $c$ 의 kernel 가중치와 bias 다.
- 시점 $t$ 의 출력은 $t - \lfloor K/2 \rfloor$ 부터 $t + \lfloor K/2 \rfloor$ 까지 $K$ 개 시점의 입력에 기대며, 논문 설정 $K = 5$ 에서는 앞뒤 두 시점씩이다.
- 범위 밖 시점의 $x$ 를 0 으로 채우면 (same padding) $`t = 1, \dots, T_s`$ 모두에서 출력이 정의되어 출력 길이가 $`T_s`$ 가 된다.
- Channel $`d_h`$ 개를 열로 쌓으면 $`\mathbf{z}_{s,p} \in \mathbb{R}^{T_s \times d_h}`$ 이 되어 식 (1) 과 같다. 이 식은 same padding 일 때 성립하며, padding 이 없으면 출력 길이는 $`T_s - K + 1`$ 이다.

### B.2 Temporal Mean (Eq. 2)

리뷰어의 유도이며, 식 (2) 가 $`T_s`$ 와 무관한 길이 $`d_h`$ 의 벡터를 주는 것을 보인다. 출발점은 식 (1) 의 $`\mathbf{z}_{s,p}`$ 다.

- 행 $`T_s`$ 개를 더하면 $`\sum_{t} \mathbf{z}_{s,p}^{(t)} \in \mathbb{R}^{d_h}`$ 이다. 행렬의 열 수 $`d_h`$ 는 그대로이고 행 차원만 없어진다.
- $`T_s`$ 로 나누면 그 값은 시점 수에 따라 커지지 않는다. 길이가 다른 step 의 embedding 을 같은 크기로 견줄 수 있다.
- 결과는 식 (2) 와 같다. 덧셈은 순서를 바꿔도 값이 같으므로, Conv1D 가 보는 $K$ 개 시점보다 긴 순서 정보는 $`\mathbf{h}_{s,p}`$ 에 남지 않는다.

### B.3 Parameter-to-Layer Attention Weight (Eq. 3)

리뷰어의 유도이며, 식 (3) 이 정의로 도입한 식이므로 그 정의에서 따라 나오는 성질을 보인다. 출발점은 점수 $`u_{l,(s,p)} = \mathbf{q}_l^\top \mathbf{h}_{s,p} / \sqrt{d_h}`$ 에 softmax 를 건 것이 식 (3) 이라는 사실이다.

합이 1 이다. 분모가 모든 node 의 분자를 더한 것이므로 아래가 성립한다.

```math
\sum_{s=1}^{S} \sum_{p=1}^{P} \alpha_{l,(s,p)} = \frac{\sum_{s,p} \exp(u_{l,(s,p)})}{\sum_{s',p'} \exp(u_{l,(s',p')})} = 1 \hspace{19em} (8)
```

$\exp$ 는 언제나 양수이므로 $`0 \lt \alpha_{l,(s,p)} \lt 1`$ 이다.

$`\sqrt{d_h}`$ 로 나누면 점수의 분산이 1 이 된다. $`\mathbf{q}_l`$ 과 $`\mathbf{h}_{s,p}`$ 의 성분이 서로 독립이고 평균 0, 분산 1 이라고 두면 아래와 같다.

```math
\mathrm{Var}\left(\mathbf{q}_l^\top \mathbf{h}_{s,p}\right) = \sum_{c=1}^{d_h} \mathrm{Var}(q_{l,c} h_{s,p,c}) = \sum_{c=1}^{d_h} 1 = d_h \hspace{19em} (9)
```

```math
\mathrm{Var}\left(\frac{\mathbf{q}_l^\top \mathbf{h}_{s,p}}{\sqrt{d_h}}\right) = \frac{d_h}{d_h} = 1 \hspace{19em} (10)
```

- 식 (9) 의 첫 등호는 독립인 항의 분산이 더해진다는 성질을, 둘째 등호는 독립이고 평균 0 인 두 변수의 곱의 분산이 두 분산의 곱이라는 성질을 썼다.
- 나누지 않으면 $`d_h = 64`$ 에서 점수의 표준편차가 8 이 되어, softmax 가 node 하나에 가중치 거의 1 을 주고 나머지 node 의 gradient 가 0 에 가까워진다.
- 이 성질은 성분이 독립이고 분산이 1 이라는 가정에서 성립하며, 학습 중 embedding 이 그 가정에서 벗어나면 점수의 분산도 1 에서 벗어난다.

### B.4 Layer Representation (Eq. 4)

리뷰어의 유도이며, 식 (4) 의 $`\mathbf{e}_l`$ 이 parameter node embedding 의 convex combination 이라는 것과 가중치가 학습되는 경로를 보인다. 출발점은 식 (3) 과 식 (8) 이다.

- 식 (8) 에 따라 가중치 $`\alpha_{l,(s,p)}`$ 는 모두 양수이고 합이 1 이므로, $`\mathbf{e}_l`$ 은 $`\mathbf{h}_{s,p}`$ 들의 convex combination 이다. 성분마다 $`\min_{s,p} h_{s,p,c} \le e_{l,c} \le \max_{s,p} h_{s,p,c}`$ 가 성립한다.
- 한 node 의 가중치가 1 에 가까우면 $`\mathbf{e}_l`$ 은 그 node 의 embedding 에 가깝다. 식 (5) 를 줄이려면 두께 예측에 쓸모 있는 node 의 가중치가 커져야 하므로, $`\alpha_{l,(s,p)}`$ 를 중요도로 읽을 수 있다.

$`\mathbf{q}_l`$ 은 softmax 를 거쳐 학습된다. Node 를 $j, k$ 로 줄여 적으면 softmax 의 미분은 아래와 같다.

```math
\frac{\partial \alpha_{l,j}}{\partial u_{l,k}} = \alpha_{l,j} \left( \delta_{jk} - \alpha_{l,k} \right) \hspace{19em} (11)
```

```math
\frac{\partial \mathbf{e}_l}{\partial u_{l,k}} = \sum_{j} \alpha_{l,j} (\delta_{jk} - \alpha_{l,k}) \mathbf{h}_j = \alpha_{l,k} \left( \mathbf{h}_k - \mathbf{e}_l \right) \hspace{19em} (12)
```

- $`\delta_{jk}`$ 는 $j = k$ 이면 1, 아니면 0 이다.
- 식 (12) 의 마지막 등호는 $`\sum_j \alpha_{l,j} \mathbf{h}_j = \mathbf{e}_l`$ 을 대입했다.
- Node $k$ 의 점수를 올리면 $`\mathbf{e}_l`$ 은 $`\mathbf{h}_k`$ 쪽으로 움직이며, 그 크기는 $`\alpha_{l,k}`$ 에 비례한다. 가중치가 이미 0 에 가까운 node 는 gradient 도 작아 다시 커지기 어렵다.

### B.5 Standardized Loss (Eq. 5)

리뷰어의 유도이며, 식 (5) 가 표준화한 두께의 mean squared error 와 같다는 것을 보인다. 출발점은 학습 자료에서 구한 layer $l$ 두께의 평균 $`\mu_l`$ 과 표준편차 $`\sigma_l`$ 로 정의한 표준화다.

```math
z_l = \frac{y_l - \mu_l}{\sigma_l}, \qquad \hat{z}_l = \frac{\hat{y}_l - \mu_l}{\sigma_l} \hspace{19em} (13)
```

```math
\hat{z}_l - z_l = \frac{(\hat{y}_l - \mu_l) - (y_l - \mu_l)}{\sigma_l} = \frac{\hat{y}_l - y_l}{\sigma_l} \hspace{19em} (14)
```

- 식 (14) 를 식 (5) 에 대입하면 $`\mathcal{L} = \frac{1}{L} \sum_l (\hat{z}_l - z_l)^2`$ 이 되어, 표준화한 두께의 mean squared error 와 같다.
- $`\sigma_l`$ 로 나누지 않으면 두께가 큰 layer 의 오차가 loss 를 차지한다. 나누면 layer 마다 오차를 자기 표준편차 단위로 재므로 모든 layer 가 같은 비중으로 학습된다.
- Layer $l$ 의 예측에 대한 gradient 는 $`\partial \mathcal{L} / \partial \hat{y}_l = 2(\hat{y}_l - y_l) / (L \sigma_l^2)`$ 이다.

### B.6 Coefficient of Determination (Eq. 6)

리뷰어의 유도이며, 식 (6) 의 $R^2$ 가 평균 예측 대비 줄어든 오차의 비율임을 보인다. 출발점은 평가 자료 $N$ 장의 잔차 제곱합과 총 제곱합이다.

```math
\mathrm{SSE}_l = \sum_{i=1}^{N} (y_{i,l} - \hat{y}_{i,l})^2, \qquad \mathrm{SST}_l = \sum_{i=1}^{N} (y_{i,l} - \bar{y}_l)^2 \hspace{19em} (15)
```

```math
R_l^2 = 1 - \frac{\mathrm{SSE}_l}{\mathrm{SST}_l} = 1 - \frac{\mathrm{SSE}_l / N}{\mathrm{SST}_l / N} = 1 - \frac{\mathrm{MSE}_l}{\mathrm{Var}(y_l)} \hspace{19em} (16)
```

- $`\mathrm{MSE}_l`$ 은 평가 자료의 mean squared error, $`\mathrm{Var}(y_l)`$ 은 $N$ 으로 나눈 평가 자료 두께의 분산이다.
- 모든 wafer 에 평균 $`\bar{y}_l`$ 을 예측하면 $`\mathrm{SSE}_l = \mathrm{SST}_l`$ 이어서 $`R_l^2 = 0`$ 이다. 그보다 나쁜 예측에서는 $`R_l^2`$ 가 음수가 된다.
- 논문의 값 0.7227 은 Film B 네 layer 의 평균 $`\frac{1}{4} \sum_{l \in \mathrm{Film\ B}} R_l^2`$ 이다. 식 (16) 의 오른쪽 변에 따라, model 이 Film B layer 마다 평가 자료의 두께 분산을 평균 약 72 % 설명한다는 뜻이다.

## Appendix C. Application to [Wafer, Feature, Trace] Tensor

리뷰어의 분석이며, 순서가 있는 wafer 들의 [Wafer, Feature, Trace] tensor 자료에 이 model 을 적용하는 조건과 주의점을 적는다. 논문은 장비 trace 의 시계열 특성은 반영하지만, wafer 가 만들어진 순서의 시계열 특성은 반영하지 않는다.

### C.1 Time Series Covered by the Paper

Table 4. Time series handled by the paper

| Time axis              | Tensor axis | Handled in the paper | How                                                     |
| :--------------------: | :---------: | :------------------: | :-----------------------------------------------------: |
| Trace 안의 시점        | Trace       | 부분적으로           | Step 안은 Conv1D 와 시간 평균, step 사이는 node 로 구분 |
| Wafer 가 만들어진 순서 | Wafer       | 아니다               | Wafer 를 서로 독립인 sample 로 다룸                     |

- Wafer 하나를 sample 하나로 보면 적용할 수 있다. 논문의 입력은 wafer 마다 parameter 별 trace $`\mathbf{x}_{s,p} \in \mathbb{R}^{T_s}`$ 이므로, tensor 의 [Feature, Trace] 부분이 parameter $p$ 와 시점 $t$ 에 해당한다.
- Trace 축을 process step 으로 나눌 수 있으면 step–parameter node $S \cdot P$ 개를 만들 수 있다. 나눌 수 없으면 node 는 feature $P$ 개뿐이고, attention 가중치는 step–parameter 단위가 아니라 feature 단위의 중요도가 된다.
- 논문은 attention 가중치를 Granger causality 에서 착안해 앞 step 의 parameter 변화가 뒤 step 의 두께에 기여하는 정도로 해석한다. 이 해석은 논문의 해석이며, Granger causality 검정을 한 것은 아니다.
- 논문은 lot 단위로 자료를 나누지만 lot 을 시간 순서로 나눴는지는 밝히지 않았다. 앞 wafer 의 영향이나 drift 는 model 에 들어가지 않는다.

### C.2 Cautions

- **Step 안의 순서**: 식 (2) 의 평균 때문에 Conv1D kernel 폭 (논문 설정 5 시점) 보다 긴 순서는 embedding 에 남지 않는다. Step 안의 긴 추세가 두께에 중요하면 평균 대신 순서를 보존하는 요약이 필요하다.
- **Step 사이의 순서**: Step 의 순서는 node 의 번호 $s$ 로만 구분된다. 식 (3) 에는 node 의 순서를 나타내는 항이 없으므로, step $s$ 와 $s + 1$ 이 이웃이라는 정보는 model 에 들어가지 않는다.
- **Step 정렬**: 모든 wafer 가 같은 step $S$ 개와 parameter $P$ 개를 가져야 node 가 wafer 사이에 같은 뜻을 갖는다. Recipe 에 따라 step 수가 다르면 step 을 맞춘 뒤에 적용한다.
- **Wafer 순서와 자료 분할**: Wafer 축에 시간 순서가 있으면 앞 기간의 lot 으로 학습하고 뒤 기간의 lot 으로 평가해야 drift 아래의 성능을 잴 수 있다. 무작위 lot 분할은 뒤 기간의 정보가 학습에 들어가 성능을 높게 보일 수 있다.
- **표준화 값**: 식 (13) 의 $`\mu_l`$ 과 식 (5) 의 $`\sigma_l`$ 은 학습 기간에서 구한 값이다. Drift 로 두께의 평균과 분산이 바뀌면 다시 구해야 한다.
- **Attention 해석**: 논문의 heatmap 은 test wafer 전체의 평균이다. Wafer 축에 시간 순서가 있으면 기간마다 따로 평균해야 중요도가 시간에 따라 바뀌는지 볼 수 있다.
