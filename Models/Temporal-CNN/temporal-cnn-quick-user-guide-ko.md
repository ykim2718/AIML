# Temporal CNN Quick User Guide
Rev. 1 | Created: 2026-10-23 | Updated: 2026-10-23 22:47 UTC

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Placement](#31-placement)
- [4. TCN Structure](#4-tcn-structure)
  - [4.1 Causal Convolution](#41-causal-convolution)
  - [4.2 Dilated Convolution](#42-dilated-convolution)
  - [4.3 Residual Connection](#43-residual-connection)
  - [4.4 Advantages over RNN](#44-advantages-over-rnn)
- [5. TCN and TCN-based Encoder](#5-tcn-and-tcn-based-encoder)
- [6. Hybrid Architecture for Wafer Sensor Data](#6-hybrid-architecture-for-wafer-sensor-data)
  - [6.1 Causal Masking](#61-causal-masking)
  - [6.2 Positional Encoding](#62-positional-encoding)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

- **Problem Statement**: 반도체 공정의 wafer 센서 시계열에 쓸 Temporal CNN (TCN) 의 quick user guide 가 없다.
- **Goal**: TCN 으로 예측하는 방법과 embedding vector 를 추출하는 방법을 구별해 쓸 수 있게 한다.
- **Non-Goal**: TCN 과 Transformer 의 구현 code 와 hyperparameter 조정은 다루지 않는다.

## 2. Summary

Standard TCN 은 time step 마다 출력을 하나씩 주어 예측과 분류에 쓰고, TCN 기반 encoder 는 TCN 뒤에 pooling 을 두어 시계열을 embedding vector 하나로 압축한다. 반도체 공정에서는 wafer 내 (intra-wafer) 센서 패턴을 TCN 기반 encoder 로 wafer 한 장당 embedding vector 하나로 압축하고, wafer 간 (inter-wafer) 장기 의존성은 그 vector 의 sequence 를 Transformer 기반 attention 으로 학습한다.

## 3. Taxonomy and its Hierarchy

Wafer 센서 model 은 시간 범위와 출력 형태의 두 축으로 갈리고, 시계열 model 은 시간 의존성을 계산하는 방식에 따라 세 family 로 나뉜다. 두 축과 축마다의 값은 [Fig 1](#fig-1) 과 같다.

```text
Axis          Value           Meaning
Time scope    Intra-wafer     one wafer, tens of seconds to minutes, Hz-level samples
              Inter-wafer     tens to hundreds of wafers in process order
Output form   Seq-to-Seq      one output per time step
              Seq-to-Vector   one embedding vector per sequence
```
<a id="fig-1"></a>
Fig 1. Axes of the wafer sensor model

Intra-wafer 는 압력, RF power, gas flow 같은 센서가 wafer 한 장을 가공하는 동안 남긴 시계열이다. Inter-wafer 는 R2R (Run-to-Run), chamber aging, target drift 처럼 wafer 수십~수백 장에 걸쳐 나타나는 변화다.

시계열 model 의 계층은 [Fig 2](#fig-2) 와 같다.

```text
Sequence model
  +-- Recurrent       computes steps one after another        RNN, LSTM, GRU
  +-- Convolutional   1D convolution over past steps          TCN, TCN-based encoder
  |                   parallel; receptive field set by design
  +-- Attention       relates every pair of positions         Transformer
                      global context; needs positional encoding
```
<a id="fig-2"></a>
Fig 2. Hierarchy of sequence models

Recurrent family 는 이전 step 의 출력을 기다려 다음 step 을 계산한다. Convolutional family 는 그 순차 계산을 버리고 convolution 으로 모든 step 을 함께 계산하며, 참조하는 과거 길이는 kernel 크기, dilation rate, layer 수로 정해진다. Attention family 는 거리와 관계없이 모든 위치 쌍의 관련성을 직접 계산하므로, 순서 정보를 positional encoding 으로 따로 넣어야 한다.

### 3.1 Placement

시간 범위마다 쓸 model 은 Table 1 과 같다.

Table 1. Model by time scope

| Time scope  | Model            | Input → output                                                        | Reason                                                                      |
| :---------: | :--------------: | :-------------------------------------------------------------------: | :-------------------------------------------------------------------------: |
| Intra-wafer | TCN 기반 encoder | (Batch, Wafer_Length, Sensor_Channels) → (Batch, Wafer_Embedding_Dim) | spike, step, 미세 파동 같은 고주파 국소 패턴을 병렬 연산으로 빠르게 압축    |
| Inter-wafer | Transformer      | (Batch, Wafer_Sequence_Length, Wafer_Embedding_Dim) → wafer 간 trend  | PM 직후, 세정 cycle 같은 특정 시점 wafer 와의 관련성을 위치와 관계없이 계산 |

두 model 을 이어 쓰는 방법은 section 6 에 있다.

## 4. TCN Structure

TCN 은 causal convolution, dilated convolution, residual connection 의 세 구조로 1D convolution 을 시계열에 맞춘 network 이다. RNN, LSTM, GRU 가 푸는 시계열 문제를 순환 대신 convolution 으로 푼다.

### 4.1 Causal Convolution

- **Definition**: 시점 $t$ 의 출력이 $x_t, x_{t-1}, \ldots$ 만 참조하는 convolution.
- **Effect**: 미래 정보가 과거와 현재의 예측에 들어가지 않음.

### 4.2 Dilated Convolution

- **Definition**: kernel 의 원소 사이에 dilation rate 만큼 간격을 둔 convolution.
- **Setting**: layer 가 깊어질수록 dilation rate 를 1, 2, 4, 8, ... 로 넓힘.
- **Effect**: 연산량을 크게 늘리지 않고 receptive field 를 layer 수에 대해 지수적으로 넓혀 먼 과거까지 참조.

### 4.3 Residual Connection

- **Definition**: ResNet 처럼 block 의 입력을 출력에 그대로 더하는 skip connection.
- **Effect**: network 가 깊어져도 vanishing gradient 없이 안정적으로 학습.

### 4.4 Advantages over RNN

TCN 은 RNN, LSTM, GRU 보다 아래 세 가지에서 유리하다.

- **Parallelism**: 이전 cell 의 출력을 기다리지 않으므로 GPU 에서 모든 step 을 함께 계산.
- **Variable length**: 1D convolution 이므로 길이가 다른 입력 sequence 에도 그대로 적용.
- **Receptive field control**: kernel 크기, dilation rate, layer 수로 참조할 과거 길이를 명시적으로 설계.

## 5. TCN and TCN-based Encoder

TCN 기반 encoder 는 TCN 을 특징 추출과 압축에 쓰도록 출력 쪽에 pooling 을 더한 구성이다. 입출력 흐름은 [Fig 3](#fig-3) 과 같다.

```text
Standard TCN
  X = (x_1, ..., x_T)  --dilated causal convs-->  Y = (y_1, ..., y_T)

TCN-based encoder
  X = (x_1, ..., x_T)  --TCN blocks-->  (h_1, ..., h_T)  --GAP or last step-->  z
```
<a id="fig-3"></a>
Fig 3. Input and output of standard TCN and TCN-based encoder

Standard TCN 은 시간 축의 길이 $T$ 를 유지하고 step 마다 출력을 하나씩 준다. TCN 기반 encoder 는 TCN block 의 출력 $(h_1, \ldots, h_T)$ 를 GAP (Global Average Pooling) 나 마지막 step 으로 시간 축에 대해 요약하여, wafer 한 장을 대표하는 embedding vector $z$ 하나를 만든다.

두 구성의 차이는 Table 2 와 같다.

Table 2. Standard TCN and TCN-based encoder

| Aspect         | Standard TCN                        | TCN-based encoder                                           |
| :------------: | :---------------------------------: | :---------------------------------------------------------: |
| Purpose        | 시계열 예측 (forecasting) 또는 분류 | 고차원 시계열의 특징 추출과 차원 축소                       |
| Input → output | Seq-to-Seq, 입력 길이 = 출력 길이   | Seq-to-Vector, 시계열 → embedding vector                    |
| Operation      | time step 마다 출력 하나            | TCN block 뒤 pooling 또는 dense layer 로 latent vector 하나 |
| Next stage     | 그 출력이 최종 예측값               | Transformer, FC layer, decoder 의 입력                      |

## 6. Hybrid Architecture for Wafer Sensor Data

TCN 기반 encoder 가 만든 wafer 별 embedding vector 를 공정 순서대로 쌓아 Transformer 의 입력으로 쓴다. 두 단계의 tensor 형태는 [Fig 4](#fig-4) 와 같다.

```text
Stage 1  TCN-based encoder, per wafer
  (Batch, Wafer_Length, Sensor_Channels)  ->  (Batch, Wafer_Embedding_Dim)

Stage 2  Transformer, across wafers
  (Batch, Wafer_Sequence_Length, Wafer_Embedding_Dim)  ->  inter-wafer trend
```
<a id="fig-4"></a>
Fig 4. Tensor shapes of the TCN and Transformer hybrid

Stage 1 은 wafer 한 장의 Hz 단위 고밀도 센서 시계열을 vector 하나로 압축하여 메모리를 줄인다. Stage 2 는 그 vector 를 Wafer_Sequence_Length 장만큼 이어 받아 wafer 간 trend 를 학습한다.

### 6.1 Causal Masking

가상 계측 (Virtual Metrology, VM) 이나 실시간 R2R 제어처럼 미래 wafer 정보를 참조하면 안 되는 model 은 TCN 과 Transformer 모두에 미래 시점을 차단하는 causal masking 을 적용한다. TCN 은 causal convolution (section 4.1) 으로, Transformer 는 attention mask 로 미래를 차단한다.

### 6.2 Positional Encoding

Transformer 에 wafer 진행 순서 (wafer index), lot 내 slot 번호, PM (Preventive Maintenance) 후 누적 순번을 positional encoding 이나 meta feature 로 결합하면 wafer 간 장기 의존성을 학습하는 성능이 높아진다.

---

## Appendix A. Terminology

- **Attention mask**: attention 계산에서 지정한 위치의 관련성을 0 으로 만들어 참조하지 못하게 하는 행렬.
- **Chamber aging**: 가공 횟수가 쌓이면서 chamber 상태가 바뀌어 공정 결과가 서서히 달라지는 현상.
- **Embedding vector**: 시계열 같은 고차원 입력을 고정 길이로 압축한 vector.
- **GAP (Global Average Pooling)**: 시간 축의 모든 step 출력을 평균하여 vector 하나로 만드는 연산.
- **Meta feature**: 센서값이 아닌 wafer 의 부가 정보로, slot 번호나 PM 후 누적 순번 같은 값.
- **PM (Preventive Maintenance)**: 장비를 정해진 period 마다 세정하고 부품을 교체하는 예방 정비.
- **Positional encoding**: Transformer 입력에 위치 정보를 더하는 vector.
- **R2R (Run-to-Run)**: 이전 run 의 계측 결과로 다음 run 의 공정 조건을 보정하는 제어.
- **Receptive field**: 한 출력이 참조하는 입력 구간의 길이.
- **Slot**: lot 안에서 wafer 가 놓이는 위치의 번호.
- **Target drift**: 공정 결과의 목표값 대비 편차가 시간에 따라 한 방향으로 움직이는 현상.
- **Vanishing gradient**: 깊은 network 에서 역전파 gradient 가 층을 지날수록 작아져 앞쪽 layer 가 학습되지 않는 현상.
- **VM (Virtual Metrology)**: 계측 대신 공정 센서 데이터로 계측값을 추정하는 model.
