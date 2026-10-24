# Temporal Convolutional Network Quick Start
Rev. 7 | Created: 2026-10-23 | Updated: 2026-10-24 10:35 UTC

- [1. Purpose](#1-purpose)
- [2. Taxonomy](#2-taxonomy)
- [3. TCN Structure](#3-tcn-structure)
  - [3.1 Causal Convolution](#31-causal-convolution)
  - [3.2 Dilated Convolution](#32-dilated-convolution)
  - [3.3 Residual Connection](#33-residual-connection)
  - [3.4 Advantages over RNN](#34-advantages-over-rnn)
- [4. TCN and TCN-based Encoder](#4-tcn-and-tcn-based-encoder)
- [5. Hybrid Architecture for Wafer Sensor Data](#5-hybrid-architecture-for-wafer-sensor-data)
  - [5.1 Causal Masking](#51-causal-masking)
  - [5.2 Positional Encoding](#52-positional-encoding)
  - [5.3 Comparison](#53-comparison)
- [6. Models and Libraries](#6-models-and-libraries)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Python Example](#appendix-b-python-example)

## 1. Purpose

- **Problem Statement**: Temporal Convolutional Network (TCN) 에 대한 quick start guide 가 없다.
- **Goal**: TCN 으로 예측하는 방법과 embedding vector 를 추출하는 방법을 구별해 쓸 수 있게 한다.
- **Non-Goal**: Temporal Convolutional Network 의 세부를 다루지 않는다.

## 2. Taxonomy

시계열 model 과 그 안에서 TCN 이 속한 계층은 [Fig 1](#fig-1) 과 같다.

```text
Sequence model
  +-- Recurrent                                             computes steps one after another
  |     +-- RNN, LSTM, GRU     h_t = f(h_{t-1}, x_t)        waits for the previous step
  +-- Convolutional                                         1D convolution over past steps
  |     +-- Standard TCN       Y = TCN(X)                   one output per time step (Seq-to-Seq)
  |     |     aka Temporal Convolutional Network
  |     +-- TCN-based encoder  z = Pool(TCN(X))             one embedding vector per sequence (Seq-to-Vector)
  +-- Attention                                             relates every pair of positions
        +-- Transformer        softmax(Q K^T / sqrt(d)) V   global context, needs positional encoding
```
<a id="fig-1"></a>
Fig 1. Sequence models and the place of TCN

Recurrent family 는 이전 step 의 출력을 기다려 다음 step 을 계산한다. RNN, LSTM, GRU 는 시점 $t$ 의 상태 $h_t$ 를 직전 상태 $h_{t-1}$ 와 입력 $x_t$ 로 구한다.

Convolutional family 는 순차 계산 대신 convolution 으로 모든 step 을 함께 계산하며, 참조하는 과거 길이는 kernel 크기, dilation rate, layer 수로 정해진다. 이 family 안에서 standard TCN 은 step 마다 출력을 하나씩 주어 미래 센서값을 예측하고, TCN 기반 encoder 는 같은 TCN 뒤에 pooling 을 두어 시계열 하나를 embedding vector 하나로 압축한다.

TCN 기반 encoder 의 식 $z = \mathrm{Pool}(\mathrm{TCN}(X))$ 에서 $X = (x_1, \ldots, x_T)$ 는 wafer 한 장의 센서 시계열이고, $x_t$ 는 시점 $t$ 의 센서 $C$ 개 값이다. $\mathrm{TCN}(X)$ 는 dilated causal convolution 을 거쳐 같은 길이 $T$ 의 출력 $(h_1, \ldots, h_T)$ 를 만들며, $h_t$ 는 시점 $t$ 까지의 입력을 요약한 $D$ 차원 vector 이다. Pool 은 시간 축을 하나로 줄여, GAP 를 쓰면 $z = \frac{1}{T}\sum_{t=1}^{T} h_t$ 이고 마지막 step 을 쓰면 $z = h_T$ 이다. 결과 $z$ 는 길이 $T$ 와 관계없는 $D$ 차원 vector 하나이며 wafer 한 장을 대표한다.

Attention family 는 거리와 관계없이 모든 위치 쌍의 관련성을 직접 계산하므로, 순서 정보를 positional encoding 으로 따로 넣어야 한다.

## 3. TCN Structure

TCN 은 causal convolution, dilated convolution, residual connection 의 세 구조로 1D convolution 을 시계열에 맞춘 network 이다. RNN, LSTM, GRU 가 푸는 시계열 문제를 순환 대신 convolution 으로 푼다.

### 3.1 Causal Convolution

- **Definition**: 시점 $t$ 의 출력이 $x_t, x_{t-1}, \ldots$ 만 참조하는 convolution.
- **Effect**: 미래 정보가 과거와 현재의 예측에 들어가지 않음.

### 3.2 Dilated Convolution

- **Definition**: kernel 의 원소 사이에 dilation rate 만큼 간격을 둔 convolution. Layer 가 깊어질수록 dilation rate 를 1, 2, 4, 8, ... 로 넓힘.
- **Effect**: 연산량을 크게 늘리지 않고 receptive field 를 layer 수에 대해 지수적으로 넓혀 먼 과거까지 참조.

### 3.3 Residual Connection

- **Definition**: ResNet 처럼 block 의 입력을 출력에 그대로 더하는 skip connection.
- **Effect**: network 가 깊어져도 vanishing gradient 없이 안정적으로 학습.

### 3.4 Advantages over RNN

TCN 은 RNN, LSTM, GRU 보다 아래 세 가지에서 유리하다.

- **Parallelism**: 이전 cell 의 출력을 기다리지 않으므로 GPU 에서 모든 step 을 함께 계산.
- **Variable length**: 1D convolution 이므로 길이가 다른 입력 sequence 에도 그대로 적용.
- **Receptive field control**: kernel 크기, dilation rate, layer 수로 참조할 과거 길이를 명시적으로 설계.

## 4. TCN and TCN-based Encoder

TCN 기반 encoder 는 TCN 을 특징 추출과 압축에 쓰도록 출력 쪽에 pooling 을 더한 구성이다. 입출력 흐름은 [Fig 2](#fig-2) 와 같다.

```text
Standard TCN
  X = (x_1, ..., x_T)  --dilated causal convs-->  Y = (y_1, ..., y_T)

TCN-based encoder
  X = (x_1, ..., x_T)  --TCN blocks-->  (h_1, ..., h_T)  --GAP or last step-->  z
```
<a id="fig-2"></a>
Fig 2. Input and output of standard TCN and TCN-based encoder

Standard TCN 은 시간 축의 길이 $T$ 를 유지하고 step 마다 출력을 하나씩 준다. TCN 기반 encoder 는 TCN block 의 출력 $(h_1, \ldots, h_T)$ 를 GAP (Global Average Pooling) 나 마지막 step 으로 시간 축에 대해 요약하여, wafer 한 장을 대표하는 embedding vector $z$ 하나를 만든다.

두 구성의 차이는 Table 1 과 같다.

Table 1. Standard TCN and TCN-based encoder

| Aspect         | Standard TCN                        | TCN-based encoder                                           |
| :------------: | :---------------------------------: | :---------------------------------------------------------: |
| Purpose        | 시계열 예측 (forecasting) 또는 분류 | 고차원 시계열의 특징 추출과 차원 축소                       |
| Input → output | Seq-to-Seq, 입력 길이 = 출력 길이   | Seq-to-Vector, 시계열 → embedding vector                    |
| Operation      | time step 마다 출력 하나            | TCN block 뒤 pooling 또는 dense layer 로 latent vector 하나 |
| Next stage     | 그 출력이 최종 예측값               | Transformer, FC layer, decoder 의 입력                      |

## 5. Hybrid Architecture for Wafer Sensor Data

TCN 기반 encoder 가 만든 wafer 별 embedding vector 를 공정 순서대로 쌓아 Transformer 의 입력으로 쓴다. 두 단계의 tensor 형태는 [Fig 3](#fig-3) 과 같다.

```text
Stage 1  TCN-based encoder, per wafer
  (Batch, Wafer_Length, Sensor_Channels)  ->  (Batch, Wafer_Embedding_Dim)

Stage 2  Transformer, across wafers
  (Batch, Wafer_Sequence_Length, Wafer_Embedding_Dim)  ->  inter-wafer trend
```
<a id="fig-3"></a>
Fig 3. Tensor shapes of the TCN and Transformer hybrid

Stage 1 은 wafer 한 장의 Hz 단위 고밀도 센서 시계열을 vector 하나로 압축하여 메모리를 줄인다. Stage 2 는 그 vector 를 Wafer_Sequence_Length 장만큼 이어 받아 wafer 간 trend 를 학습한다.

Stage 1 이 다루는 intra-wafer 시계열은 압력, RF power, gas flow 같은 센서가 wafer 한 장을 가공하는 수십 초에서 수분 동안 남긴 기록이다. Stage 2 가 다루는 inter-wafer 변화는 R2R (Run-to-Run), chamber aging, target drift 처럼 wafer 수십에서 수백 장에 걸쳐 나타난다.

### 5.1 Causal Masking

가상 계측 (Virtual Metrology, VM) 이나 실시간 R2R 제어처럼 미래 wafer 정보를 참조하면 안 되는 model 은 TCN 과 Transformer 모두에 미래 시점을 차단하는 causal masking 을 적용한다. TCN 은 causal convolution (section 3.1) 으로, Transformer 는 attention mask 로 미래를 차단한다.

### 5.2 Positional Encoding

Transformer 에 wafer 진행 순서 (wafer index), lot 내 slot 번호, PM (Preventive Maintenance) 후 누적 순번을 positional encoding 이나 meta feature 로 결합하면 wafer 간 장기 의존성을 학습하는 성능이 높아진다.

### 5.3 Comparison

시간 범위마다 쓸 model 은 Table 2 와 같다.

Table 2. Model by time scope

| Time scope  | Model            | Input → output                                                        | Reason                                                                      |
| :---------: | :--------------: | :-------------------------------------------------------------------: | :-------------------------------------------------------------------------: |
| Intra-wafer | TCN 기반 encoder | (Batch, Wafer_Length, Sensor_Channels) → (Batch, Wafer_Embedding_Dim) | spike, step, 미세 파동 같은 고주파 국소 패턴을 병렬 연산으로 빠르게 압축    |
| Inter-wafer | Transformer      | (Batch, Wafer_Sequence_Length, Wafer_Embedding_Dim) → wafer 간 trend  | PM 직후, 세정 cycle 같은 특정 시점 wafer 와의 관련성을 위치와 관계없이 계산 |

## 6. Models and Libraries

TCN 은 `torch.nn` 의 layer 를 조합해 만들고, Transformer 는 `torch.nn` 의 encoder class 를 그대로 쓴다. 방법별 model 과 library 는 Table 3 과 같다.

Table 3. Models and libraries by method

<table>
  <tr><th align="center">Method</th><th align="center">Model</th><th align="center">Library</th><th align="center">Note</th></tr>
  <tr><td align="center" rowspan="2">Standard TCN</td><td align="center">Dilated causal convolution</td><td align="center"><code>torch.nn.Conv1d</code>,<br><code>torch.nn.functional.pad</code></td><td align="center"><code>dilation</code> 을 주고 입력 왼쪽에만 (kernel 크기 − 1) × dilation 만큼 padding 하여 미래 step 을 참조하지 않게 함</td></tr>
  <tr><td align="center">Residual block</td><td align="center"><code>torch.nn.Conv1d</code></td><td align="center">block 입력을 출력에 더하고, 채널 수가 다르면 kernel 크기 1 의 convolution 으로 맞춤</td></tr>
  <tr><td align="center">TCN-based encoder</td><td align="center">GAP</td><td align="center"><code>torch.nn.AdaptiveAvgPool1d</code></td><td align="center"><code>output_size=1</code> 로 시간 축을 평균하여 embedding vector 를 만듦</td></tr>
  <tr><td align="center" rowspan="2">Transformer</td><td align="center">Transformer encoder</td><td align="center"><code>torch.nn.TransformerEncoderLayer</code>,<br><code>torch.nn.TransformerEncoder</code></td><td align="center">wafer 별 embedding vector 의 sequence 로 wafer 간 trend 를 학습</td></tr>
  <tr><td align="center">Causal mask</td><td align="center"><code>torch.nn.Transformer.generate_square_subsequent_mask</code></td><td align="center">미래 wafer 를 가리는 attention mask 를 만듦</td></tr>
</table>

TCN 기반 encoder 는 Standard TCN 의 두 행으로 TCN block 을 만든 뒤 그 뒤에 GAP 를 둔다. `torch.nn.Conv1d` 는 Standard TCN 의 두 행에 모두 나오며, dilated causal convolution 에서는 dilation 과 왼쪽 padding 을 주고 residual block 에서는 kernel 크기 1 로 채널 수만 맞춘다.

Table 3 의 class 로 세 model 을 만든 예시 code 는 [Appendix B](#appendix-b-python-example) 에 있다.

---

## Appendix A. Terminology

- **aka**: also known as. 같은 것을 가리키는 다른 이름 앞에 붙인다.
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

## Appendix B. Python Example

아래 code 는 Table 3 의 `torch.nn` class 로 standard TCN, TCN 기반 encoder, TCN 과 Transformer 의 hybrid 를 만든 예시다. `TCNForecaster` 는 시점 $t$ 까지의 입력으로 $t+1$ 의 센서값을 예측하고, `TCNEncoder` 는 $z = \mathrm{Pool}(\mathrm{TCN}(X))$ 로 wafer 한 장의 embedding vector 를 만든다. `WaferSequenceModel` 은 wafer 별 embedding vector 를 causal Transformer 에 넣어 wafer 마다 박막 두께 같은 값 하나를 예측한다. 세 model 의 출력 head 는 `torch.nn.Linear` 이다.

```python
# Python
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__author__ = 'yRocket'
__version__ = "0.0.0+20261024"  # Semantic Versioning: Major.Minor.Patch+YYYYMMDD

KERNEL_SIZE: int = 3
DILATIONS: tuple = (1, 2, 4, 8)  # receptive field grows exponentially with depth


class CausalConv1d(nn.Module):
    """Dilated 1D convolution that pads only the left side, so step t sees x_t, x_{t-1}, ... only."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.left_pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (batch, channels, time)
        return self.conv(F.pad(x, pad=(self.left_pad, 0)))


class ResidualBlock(nn.Module):
    """Causal convolution with a residual connection; a kernel-1 Conv1d matches channels when they differ."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = CausalConv1d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, dilation=dilation)
        self.skip = (nn.Identity() if in_channels == out_channels
                     else nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv(x) + self.skip(x))


class TCN(nn.Module):
    """Standard TCN: (batch, time, sensors) -> (batch, time, hidden), one output per time step."""

    def __init__(self, n_sensors: int, hidden: int, kernel_size: int = KERNEL_SIZE,
                 dilations: Sequence[int] = DILATIONS):
        super().__init__()
        if not dilations:
            raise ValueError("dilations must not be empty.")
        channels = [n_sensors] + [hidden] * len(dilations)
        self.blocks = nn.Sequential(*[
            ResidualBlock(in_channels=channels[i], out_channels=channels[i + 1],
                          kernel_size=kernel_size, dilation=d)
            for i, d in enumerate(dilations)])
        self.n_sensors = n_sensors
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[-1] != self.n_sensors:
            raise ValueError(f"expected (batch, time, {self.n_sensors}), got {tuple(x.shape)}")
        return self.blocks(x.transpose(1, 2)).transpose(1, 2)


class TCNForecaster(nn.Module):
    """Prediction: forecasts the sensor values of step t+1 from steps up to t."""

    def __init__(self, tcn: TCN):
        super().__init__()
        self.tcn = tcn
        self.head = nn.Linear(in_features=tcn.hidden, out_features=tcn.n_sensors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, time, sensors) -> same shape
        return self.head(self.tcn(x))


class TCNEncoder(nn.Module):
    """Embedding: z = Pool(TCN(X)), (batch, time, sensors) -> (batch, hidden)."""

    def __init__(self, tcn: TCN):
        super().__init__()
        self.tcn = tcn
        self.pool = nn.AdaptiveAvgPool1d(output_size=1)  # GAP over the time axis

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x)  # (batch, time, hidden)
        return self.pool(h.transpose(1, 2)).squeeze(-1)


class WaferSequenceModel(nn.Module):
    """Hybrid: TCN encoder per wafer, causal Transformer across wafers, one value (e.g. thickness) per wafer."""

    def __init__(self, encoder: TCNEncoder, n_heads: int, n_layers: int):
        super().__init__()
        d_model = encoder.tcn.hidden
        self.encoder = encoder
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=layer, num_layers=n_layers)
        self.head = nn.Linear(in_features=d_model, out_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, n_wafers, time, sensors) -> (batch, n_wafers)."""
        if x.dim() != 4:
            raise ValueError(f"expected (batch, n_wafers, time, sensors), got {tuple(x.shape)}")
        batch, n_wafers = x.shape[:2]
        z = self.encoder(x.flatten(0, 1)).view(batch, n_wafers, -1)  # (batch, n_wafers, hidden)
        mask = nn.Transformer.generate_square_subsequent_mask(n_wafers)  # hides future wafers
        return self.head(self.transformer(z, mask=mask, is_causal=True)).squeeze(-1)


if __name__ == '__main__':
    torch.manual_seed(0)
    x = torch.randn(4, 200, 6)  # 4 wafers, 200 time steps, 6 sensors
    tcn = TCN(n_sensors=6, hidden=32)
    print(TCNForecaster(tcn=tcn)(x).shape)  # torch.Size([4, 200, 6])
    print(TCNEncoder(tcn=tcn)(x).shape)  # torch.Size([4, 32])
    model = WaferSequenceModel(encoder=TCNEncoder(tcn=tcn), n_heads=4, n_layers=2)
    print(model(torch.randn(2, 25, 200, 6)).shape)  # torch.Size([2, 25])
```

PyTorch 2.14.1 에서 실행하면 `__main__` block 의 주석과 같은 shape 가 출력된다.
