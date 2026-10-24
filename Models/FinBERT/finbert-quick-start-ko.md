# FinBERT Quick Start
Rev. 0 | Created: 2026-10-24 | Updated: 2026-10-24 05:45 CDT

- [1. Purpose](#1-purpose)
- [2. Taxonomy](#2-taxonomy)
- [3. Features](#3-features)
  - [3.1 Financial Sentiment Analysis](#31-financial-sentiment-analysis)
  - [3.2 Financial Terms](#32-financial-terms)
  - [3.3 Probability Score](#33-probability-score)
  - [3.4 KR-FinBERT](#34-kr-finbert)
- [4. Applications](#4-applications)
  - [4.1 Comparison](#41-comparison)
- [5. Models and Libraries](#5-models-and-libraries)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

- **Problem Statement**: FinBERT 에 대한 quick start guide 가 없다.
- **Goal**: FinBERT 로 금융 text 의 감성을 분류하는 방법과 활용 분야를 고를 수 있게 한다.
- **Non-Goal**: FinBERT 의 세부를 다루지 않는다.

## 2. Taxonomy

FinBERT 가 BERT 에서 갈라져 나온 계층은 [Fig 1](#fig-1) 과 같다.

```text
BERT                h = Enc(X)                   bidirectional Transformer encoder trained on general text
  +-- FinBERT       p(y | X) = softmax(W h_CLS)  adapted to English financial text, y in {positive, negative, neutral}
  +-- KR-FinBERT    p(y | X) = softmax(W h_CLS)  adapted to Korean financial text, same three labels
```
<a id="fig-1"></a>
Fig 1. FinBERT and KR-FinBERT derived from BERT

BERT 는 일반 text 로 학습한 Transformer encoder 로, 입력 문장 $X$ 의 token 마다 문맥을 반영한 vector $h$ 를 만든다. FinBERT 와 KR-FinBERT 는 BERT 를 금융 corpus 로 추가 pre-training 한 뒤 감성 분류로 fine-tuning 한 model 이다. 두 model 은 `[CLS]` token 의 vector $h_{\mathrm{CLS}}$ 에 분류 layer $W$ 와 softmax 를 적용하여 세 감성의 확률을 준다. FinBERT 는 영어 금융 text 를, KR-FinBERT 는 한국어 금융 text 를 다룬다.

## 3. Features

FinBERT 의 기능은 금융 감성 분석, 금융 용어 학습, 확률 score, 한국어 model 의 네 가지다.

### 3.1 Financial Sentiment Analysis

- **Definition**: 금융 뉴스, 기업 공시, analyst 보고서, social media 글을 긍정 (positive), 부정 (negative), 중립 (neutral) 으로 분류.
- **Effect**: 사람이 읽어야 하던 대량의 금융 text 를 감성 label 로 바꾸어 집계.

### 3.2 Financial Terms

- **Definition**: 금융권의 미묘한 표현, 전문 용어, 시장 은어를 담은 금융 corpus 로 추가 학습.
- **Effect**: 일반 text 로만 학습한 model 보다 금융 문장의 감성을 정확히 분류.

### 3.3 Probability Score

- **Definition**: softmax 로 구한 세 감성의 확률을 score 로 출력.
- **Effect**: label 하나 대신 확률값을 지표나 model 입력으로 사용.

### 3.4 KR-FinBERT

- **Definition**: 한국어 BERT 인 KR-BERT-MEDIUM 을 한국어 경제 뉴스와 증권사 analyst 보고서로 추가 학습한 model.
- **Effect**: 한국어 금융 text 를 FinBERT 와 같은 세 감성으로 분류.

## 4. Applications

FinBERT 의 감성 label 과 score 는 주식 거래, risk 관리, 기업 IR 에 쓰인다.

### 4.1 Comparison

분야마다 분석하는 text 와 쓰임은 Table 1 과 같다.

Table 1. Applications of FinBERT

| Field                   | Input text                          | Use                                        |
| :---------------------: | :---------------------------------: | :----------------------------------------: |
| Stock trading           | 뉴스 headline, 공시 자료            | 실시간 분석으로 주가 방향 예측, quant 전략 |
| Risk and credit         | 기업 보고서, 신용평가 자료          | 문서의 tone 으로 부실 위험과 신용도 평가   |
| Investor relations (IR) | 주주총회 기록, 투자자 communication | 투자자 여론과 심리 파악                    |

## 5. Models and Libraries

두 model 모두 Hugging Face Hub 에 공개되어 있어 `transformers` 로 불러 쓴다. Model 과 library 는 Table 2 와 같다.

Table 2. Models and libraries for FinBERT

<table>
  <tr><th align="center">Method</th><th align="center">Model</th><th align="center">Library</th><th align="center">Note</th></tr>
  <tr><td align="center">FinBERT</td><td align="center"><code>ProsusAI/finbert</code></td><td align="center"><code>transformers.pipeline</code></td><td align="center"><code>task="text-classification"</code> 로 영어 문장의 감성 label 과 score 를 구함</td></tr>
  <tr><td align="center">KR-FinBERT</td><td align="center"><code>snunlp/KR-FinBert-SC</code></td><td align="center"><code>transformers.AutoTokenizer</code>,<br><code>transformers.AutoModelForSequenceClassification</code></td><td align="center">한국어 문장을 token 으로 나누고 세 감성의 logit 을 구함</td></tr>
</table>

Model 열은 Hugging Face Hub 의 model ID 다. 두 model 은 같은 BERT 분류 구조이므로 `transformers.pipeline` 과 `transformers.AutoModelForSequenceClassification` 중 어느 쪽으로도 불러 쓸 수 있다.

---

## Appendix A. Terminology

- **[CLS] token**: BERT 가 입력 맨 앞에 두는 token. 이 token 의 vector 를 문장 전체의 표현으로 써서 분류.
- **Corpus**: model 학습에 쓰는 text 모음.
- **Fine-tuning**: pre-training 한 model 을 특정 task 의 label 이 있는 데이터로 다시 학습하는 것.
- **Hugging Face Hub**: model 과 dataset 을 ID 로 공개하고 내려받는 저장소. `transformers` 가 model ID 로 Hub 에서 model 을 받음.
- **Logit**: softmax 를 적용하기 전의 분류 점수.
- **Pre-training**: label 없는 대량의 text 로 언어 model 을 먼저 학습하는 것.
- **Quant**: 가격, 재무 지표 같은 정량 데이터와 model 로 매매를 정하는 계량 투자.
- **Softmax**: 점수 vector 를 합이 1 인 확률 vector 로 바꾸는 함수.
- **Tone**: 문서 전체에 드러나는 긍정, 부정, 중립의 기조.
- **Transformer encoder**: self-attention 으로 입력 token 사이의 관련성을 계산하는 Transformer 의 encoder block.
