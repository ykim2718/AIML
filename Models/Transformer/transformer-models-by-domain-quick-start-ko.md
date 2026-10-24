# Transformer Models by Domain Quick Start
Rev. 3 | Created: 2026-10-24 | Updated: 2026-10-24 10:57 UTC

- [1. Purpose](#1-purpose)
- [2. Taxonomy](#2-taxonomy)
- [3. Transformer and LLM](#3-transformer-and-llm)
  - [3.1 Transformer](#31-transformer)
  - [3.2 LLM](#32-llm)
  - [3.3 Time-Series Transformer](#33-time-series-transformer)
- [4. Domain Models](#4-domain-models)
  - [4.1 Comparison](#41-comparison)
- [5. Models and Libraries](#5-models-and-libraries)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

- **Problem Statement**: Transformer Models by Domain 에 대한 quick start guide 가 없다.
- **Goal**: Transformer 와 LLM 을 구별하고, 도메인마다 대표 Transformer model 을 고를 수 있게 한다.
- **Non-Goal**: Transformer Models by Domain 의 세부를 다루지 않는다.

## 2. Taxonomy

Transformer 가 입력 형태와 도메인에 따라 갈리는 계층은 [Fig 1](#fig-1) 과 같다.

```text
Transformer                  softmax(Q K^T / sqrt(d)) V  relates every pair of positions in a sequence
  +-- Discrete token input   x_t in {1, ..., V}          token IDs from a fixed vocabulary
  |     +-- Text                                         word or subword tokens
  |     |     +-- LLM        p(x_t | x_1, ..., x_{t-1})  decoder-only, trained on a large text corpus
  |     |           aka Large Language Model
  |     +-- Protein                                      amino acid tokens
  +-- Continuous input       x_t in R^d                  real-valued vectors, embedded by a linear layer
        +-- Vision                                       image patches
        +-- Time series                                  sensor values, patches or embedding vectors
        +-- Audio                                        mel spectrogram frames
```
<a id="fig-1"></a>
Fig 1. Transformer by input type and domain

모든 Transformer 는 self-attention 으로 sequence 안의 모든 위치 쌍의 관련성을 계산한다. 갈래는 sequence 의 한 원소가 무엇이냐로 정해진다. Discrete token input 은 크기 $V$ 인 vocabulary 의 token 번호를 embedding table 에서 vector 로 바꾸고, continuous input 은 실수 vector 를 linear layer 로 바로 embedding 한다.

LLM 은 Text 아래의 한 갈래이므로 모든 LLM 은 Transformer 이지만, Vision, Time series, Audio, Protein 의 Transformer 는 LLM 이 아니다.

## 3. Transformer and LLM

Transformer 는 architecture 이고, LLM 은 그 architecture 로 자연어를 학습한 model 의 한 종류다. 시계열 센서 데이터에는 LLM 대신 time-series Transformer 를 쓴다.

### 3.1 Transformer

- **Definition**: 2017 년 논문 "Attention Is All You Need" [[1](#ref-1)] 가 제안한 architecture. Self-attention 으로 sequence 구성 요소 사이의 관련성을 계산.
- **Input**: token, image patch, 센서값처럼 sequence 로 놓을 수 있는 모든 데이터.
- **Example**: BERT, ViT, Whisper, PatchTST.

### 3.2 LLM

- **Definition**: 주로 decoder-only Transformer 를 대규모 text 로 학습한 언어 model.
- **Input**: 단어나 subword 단위의 discrete token sequence.
- **Example**: GPT-4, LLaMA, Claude.

### 3.3 Time-Series Transformer

- **Definition**: 연속값 sequence 의 장기 trend 와 drift 를 학습하도록 만든 Transformer.
- **Input**: 센서값, 또는 TCN (Temporal Convolutional Network) 이 wafer 마다 추출한 embedding vector 의 sequence.
- **Example**: 반도체 공정에서 wafer 간 (inter-wafer) 장기 의존성을 학습하는 TCN + Transformer hybrid. TCN 이 wafer 한 장의 센서 시계열을 embedding vector 하나로 압축하고, Transformer 가 공정 순서대로 쌓은 embedding vector 의 sequence 에서 wafer 간 drift 를 학습.

## 4. Domain Models

Transformer 는 자연어뿐 아니라 이미지, 시계열, 음성, 단백질 서열로 확장되어 쓰인다.

### 4.1 Comparison

도메인별 대표 Transformer model 은 Table 1 과 같다.

Table 1. Representative Transformer models by domain

| Domain      | Task                        | Models                                                                  |
| :---------: | :-------------------------: | :---------------------------------------------------------------------: |
| Text        | 대화, 번역, 요약            | LLM (GPT series, LLaMA), BERT, T5                                       |
| Vision      | 객체 인식, 이미지 생성      | ViT (Vision Transformer), Swin Transformer, DiT (Diffusion Transformer) |
| Time series | 센서 데이터 예측, 이상 탐지 | TimeGPT, Lag-Llama, PatchTST, Informer, Autoformer                      |
| Audio       | 음성 인식, 음성 합성        | Whisper, SpeechT5                                                       |
| Protein     | 단백질 구조 예측            | AlphaFold2, ESMFold                                                     |

## 5. Models and Libraries

Table 1 의 model 은 대부분 `transformers` 의 class 로 불러 쓴다. 도메인별 model 과 library 는 Table 2 와 같다.

Table 2. Models and libraries by domain

<table>
  <tr><th align="center">Method</th><th align="center">Model</th><th align="center">Library</th><th align="center">Note</th></tr>
  <tr><td align="center" rowspan="4">Text</td><td align="center">GPT-2</td><td align="center"><code>transformers.GPT2LMHeadModel</code></td><td align="center">공개된 GPT series model 로 다음 token 을 생성</td></tr>
  <tr><td align="center">LLaMA</td><td align="center"><code>transformers.LlamaForCausalLM</code></td><td align="center">decoder-only LLM 으로 대화와 요약 text 를 생성</td></tr>
  <tr><td align="center">BERT</td><td align="center"><code>transformers.BertModel</code></td><td align="center">문장의 embedding vector 를 추출</td></tr>
  <tr><td align="center">T5</td><td align="center"><code>transformers.T5ForConditionalGeneration</code></td><td align="center">입력 text 를 번역문이나 요약문으로 변환</td></tr>
  <tr><td align="center" rowspan="3">Vision</td><td align="center">ViT</td><td align="center"><code>transformers.ViTForImageClassification</code></td><td align="center">image patch sequence 로 이미지를 분류</td></tr>
  <tr><td align="center">Swin Transformer</td><td align="center"><code>transformers.SwinForImageClassification</code></td><td align="center">이동하는 window 안에서 attention 을 계산해 이미지를 분류</td></tr>
  <tr><td align="center">DiT</td><td align="center"><code>diffusers.DiTPipeline</code></td><td align="center">class label 을 받아 이미지를 생성</td></tr>
  <tr><td align="center" rowspan="4">Time series</td><td align="center">TimeGPT</td><td align="center"><code>nixtla.NixtlaClient.forecast</code>,<br><code>nixtla.NixtlaClient.detect_anomalies</code></td><td align="center">Nixtla API 로 예측과 이상 탐지를 요청</td></tr>
  <tr><td align="center">PatchTST</td><td align="center"><code>transformers.PatchTSTForPrediction</code></td><td align="center">시계열을 patch 로 잘라 미래 구간을 예측</td></tr>
  <tr><td align="center">Informer</td><td align="center"><code>transformers.InformerForPrediction</code></td><td align="center">긴 입력 구간으로 장기 예측</td></tr>
  <tr><td align="center">Autoformer</td><td align="center"><code>transformers.AutoformerForPrediction</code></td><td align="center">trend 와 계절 성분을 분해해 장기 예측</td></tr>
  <tr><td align="center" rowspan="2">Audio</td><td align="center">Whisper</td><td align="center"><code>transformers.WhisperForConditionalGeneration</code></td><td align="center">음성을 text 로 변환</td></tr>
  <tr><td align="center">SpeechT5</td><td align="center"><code>transformers.SpeechT5ForSpeechToText</code>,<br><code>transformers.SpeechT5ForTextToSpeech</code></td><td align="center">음성 인식과 음성 합성</td></tr>
  <tr><td align="center">Protein</td><td align="center">ESMFold</td><td align="center"><code>transformers.EsmForProteinFolding</code></td><td align="center">아미노산 서열로 단백질 3D 구조를 예측</td></tr>
</table>

GPT-4 와 Claude 는 API 로만 제공되고, Lag-Llama 와 AlphaFold2 는 개발처의 GitHub source code 로 설치하므로 Table 2 에서 뺐다.

## References

<a id="ref-1"></a>
[1] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30, 5998–6008.

---

## Appendix A. Terminology

- **aka**: also known as. 같은 것을 가리키는 다른 이름 앞에 붙인다.
- **Decoder-only**: Transformer 의 encoder 없이 decoder block 만 쌓아, 앞 token 으로 다음 token 을 생성하는 구조.
- **Drift**: 공정 결과가 시간에 따라 한 방향으로 서서히 움직이는 현상.
- **Embedding vector**: token 이나 시계열 같은 입력을 고정 길이로 바꾼 실수 vector.
- **LLM (Large Language Model)**: 대규모 text 로 학습한 Transformer 기반 언어 model.
- **Mel spectrogram**: 음성 신호를 짧은 구간마다 주파수 성분으로 바꾸고 사람 청각에 맞춘 mel 척도로 나타낸 2D 표현.
- **Patch**: 이미지나 시계열을 일정 크기로 자른 조각. Transformer 는 patch 하나를 sequence 의 원소 하나로 씀.
- **Self-attention**: sequence 안의 각 위치가 다른 모든 위치와의 관련성을 계산해 자기 표현을 갱신하는 연산.
- **TCN (Temporal Convolutional Network)**: causal, dilated 1D convolution 으로 시계열을 처리하는 network.
- **Token**: discrete sequence 의 한 원소로, vocabulary 안의 번호로 나타낸 단어, subword, 아미노산.
- **Vocabulary**: model 이 쓰는 token 전체의 집합.
