# Time Series Inference Server
Rev. 32 | Created: 2026-08-28 | Updated: 2026-09-04 20:10 UTC

시계열 model 은 적합이 끝난 뒤에도 그 자체로는 아무 답도 내놓지 못한다. 관측이 계속 도착하는 환경에서 model 을 실제 답으로 잇는 장치가 inference server 이며, 이 리포트는 그 장치를 구성, model 의 몫, server 의 몫의 순서로 정리한다. 읽는 이는 시계열 model 을 적합해 본 적은 있으나 그것을 운영에 올려 본 적은 없는 사람을 상정한다.

## 1. Scope

- 다루는 범위: 적합이 끝난 model 또는 pre-trained checkpoint 를 운영에서 사용하는 단계.
- 다루지 않는 범위: 학습 알고리즘, model 선택, feature 설계, 자료 수집 설계.
- 전제: 계열이 수천 개이고 관측이 끊이지 않는 환경. 계열 하나면 scheduled job 하나로 충분하다.

## 2. Structure

Inference server 는 하나의 process 가 아니라 몇 개의 구성 요소가 정해진 순서로 맞물린 구조이다. Fig 1 은 관측이 들어와 답이 되고, 그 답이 다시 학습으로 돌아가기까지의 경로이며, Table 1 은 각 구성 요소가 맡는 일이다.

```text
   producers             ingest              stores                 serving runtime
 +-----------+       +------------+     +----------------+     +----------------------+
 | sensors   |       | broker /   |     | online store   |     | 1 resolve version    |
 | equipment | ----> | stream     | --> | model registry | --> | 2 assemble context   |
 | apps      |       |            |     | segment index  |     | 3 validate or refuse |
 +-----------+       +------------+     +----------------+     | 4 batch keys         |
                                                               | 5 model call         |
                                                               | 6 shape and stamp    |
                                                               | 7 persist            |
                                                               +----------------------+
                                                                          |
                                          consumers                       v
                                     +-----------------+      +----------------------+
                                     | dashboard       | <--- | answer + identifiers |
                                     | alarm           |      +----------------------+
                                     | control action  |                 |
                                     +-----------------+                 v
                                              |             +------------------------+
                                              +-----------> | prediction store       |
                                     actuals, verdicts,     | feedback store         |
                                     event marks            +------------------------+
                                                                          |
                                                                          v
                                                             governed job: score,
                                                             detect, fit, promote
```

Fig 1. Inference server 의 구성과 답이 만들어지는 경로

Table 1. 구성 요소와 그 역할

| # | Component | Role |
|---|-----------|------|
| 1 | Ingest path | 관측을 producer 에서 받아 store 와 server 양쪽으로 나눠 보냄. 유실과 중복은 여기서 정리. |
| 2 | Online store | key 마다 최근 관측과 static attribute 를 보관. 요청 때 밀리초 안에 읽히는 유일한 자료원. |
| 3 | Model registry | model 의 version 과, 어느 version 이 어느 key 를 맡는지에 대한 규칙. |
| 4 | Segment index | 과거 구간의 embedding 과 그 속성. 비교 질문을 forward pass 없이 답하는 자리. |
| 5 | Serving runtime | 위의 store 에서 입력을 모아 model 을 부르고, 그 숫자를 답으로 만드는 process. Fig 1 의 1 부터 7. |
| 6 | Prediction store | 만들어진 답을 그 context 와 version 과 함께 보관. 나중의 scoring 과 재현이 여기에 걸림. |
| 7 | Feedback store | caller 가 되돌려 보낸 actuals, verdict, correction, event mark 를 append 만으로 쌓음. |
| 8 | Governed job | 요청 밖에서 도는 일. scoring, drift 감지, 재적합, 승격이 여기 속함. |

## 3. Model Capability

Model 은 정해진 한 가지 모양의 array 를 받아 숫자를 돌려주는 함수이다. 따라서 model 의 capability 는 "어떤 질문에 대해 숫자를 낼 수 있는가" 로 정리되며, Table 2 가 그 목록이다. 반대로 Table 3 은 model 이 원리상 감당할 수 없는 것이고, 이것이 다음 꼭지가 필요한 이유이다.

Table 2. Model 이 낼 수 있는 답과 그 조건

| # | Question | Model output | Condition on the model |
|---|----------|--------------|------------------------|
| 1 | Forecast | 앞으로 `H` step 의 값. quantile 이면 분포의 여러 지점. | 그 horizon 까지 적합되었을 것. quantile 은 전용 head 나 sampling 이 있을 것. |
| 2 | Anomaly detection | 관측이 기대에서 얼마나 벗어났는지의 deviation score. | 정상 거동의 baseline 을 학습했을 것. |
| 3 | Classification | segment 별 class 와 그 확률. | label 이 붙은 과거 segment 로 적합되었을 것. |
| 4 | Retrieval | segment 의 embedding. 거리 비교는 index 가 수행. | encoder 계열이거나, 중간 표현을 꺼낼 수 있을 것. |
| 5 | Imputation | 결측 구간의 값. | 결측을 포함한 입력으로 적합되었거나 다변량일 것. |
| 6 | Change point detection | 거동이 바뀐 시점과 그 세기. | segment 전체를 한 번에 받는 입력 형태일 것. |
| 7 | Virtual metrology | 나중에 측정될 값의 추정치. | 과거의 기록과 측정을 짝지은 자료로 적합되었을 것. |
| 8 | Remaining useful life | 한계까지 남은 run 수나 시간. | 고장과 교체가 표시된 자료로 적합되었을 것. |
| 9 | Attribution | 답을 움직인 channel 이나 lag 의 기여. | 기여를 계산할 수 있는 구조이거나 그 절차가 붙어 있을 것. |

Table 3. Model 이 스스로 하지 못하는 것

| # | Limitation | Reason |
|---|------------|--------|
| 1 | 지금 다루는 계열이 무엇인지 모름. | 입력이 array 뿐이라 key 가 전달되지 않음. |
| 2 | 지금이 언제인지 모름. | 시각과 cut-off 는 입력 밖에서 정해짐. |
| 3 | 직전에 무엇을 답했는지 모름. | 호출 사이에 상태를 남기지 않음. |
| 4 | 자기 답이 맞았는지 모름. | 실측은 `H` step 뒤에 도착하고, 그 자리에 model 은 없음. |
| 5 | 무엇이 원인인지 말하지 못함. | 관측만으로는 개입 없이 인과를 가릴 수 없음. |
| 6 | 요청 도중 스스로 다시 적합하지 못함. | 새 적합은 다른 caller 의 답까지 바꾸는 일. |

## 4. Server Capability

Server 의 capability 는 Table 3 의 여섯 가지를 메워 model 의 숫자를 답으로 만드는 데 있다. 이 꼭지는 그 몫을 요구사항으로 적고, 이어서 caller 가 그 몫에 개입할 수 있는 통로를 적는다.

### 4.1 Server Requirements

Table 4. Server 가 갖춰야 할 요구사항

| # | Requirement | Content | If unmet |
|---|-------------|---------|----------|
| 1 | Context assembly | cut-off 기준으로 그 key 의 최근 `L` 관측을 모으고, 너무 성긴 window 는 거절. | model 이 본 적 없는 모양의 입력에 답이 나옴. |
| 2 | Point-in-time correctness | 그 시점에 실제로 관측 가능했던 값만 포함. | 학습에서는 좋고 운영에서는 나쁜 결과가 재현됨. |
| 3 | Timestamp discipline | 하나의 frequency, timezone, 결측과 중복 규약. 학습 때와 동일. | 두 경로가 각자 일관된 채로 어긋남. |
| 4 | Covariate handling | past covariate 와 future known covariate 를 구분해 받고, 미리 알 수 없는 값은 거절. | backtest 만 좋고 실제 예측은 나쁜 model 이 만들어짐. |
| 5 | Series fan-out | 한 번의 호출로 여러 key 를 처리하고, 각 key 를 자기 version 으로 routing. | key 수에 비례해 비용과 운영 부담이 늘어남. |
| 6 | State and recovery | recursive model 의 계열별 state 를 들고 checkpoint. | 재시작마다 계열을 처음부터 재생해야 함. |
| 7 | Cold start | 이력 없는 key 는 global 이나 pre-trained model 로 routing, 아니면 거절. | 짧은 window 를 채워 넣은 값이 실제 거동으로 읽힘. |
| 8 | Delayed evaluation | 답을 context 와 version 과 함께 보관하고, 실측이 오면 결합. | 정확도를 사후에 계산할 방법이 없음. |
| 9 | Freshness and drift | 답 뒤에 놓인 최신 관측의 나이를 함께 알리고, 잔차가 움직이면 신호를 냄. | 낡은 입력 위의 답이 최신 답과 구분되지 않음. |
| 10 | Throughput and latency | batching 과 compile 된 runtime 과 autoscaling 으로 예산을 지키고, 초과 시 더 싼 model 로 강등. | 부하가 몰리면 답 대신 timeout 이 돌아감. |
| 11 | Reproducibility | version, context, code path 를 함께 고정해 지난 요청을 같은 답으로 재생. | 사고 조사와 model 비교가 불가능해짐. |
| 12 | Governance of writes | serving 중인 model 을 바꾸는 일은 governed job 에만 허용. | 한 caller 의 호출이 다른 caller 의 답을 바꿈. |

### 4.2 Caller Interface

Caller 는 server 가 스스로 알 수 없는 두 가지를 쥐고 있다. 요청 뒤에 있는 의도와, 그 답이 맞았는지에 대한 사실이다. 앞쪽은 요청에 얹는 option 으로, 뒤쪽은 사후에 되돌려 보내는 feedback 으로 전달된다.

Table 5. 요청에 얹는 option

| # | Option | Content |
|---|--------|---------|
| 1 | Horizon | 행동에 옮길 수 있는 step 수. model 이 적합된 최대치로 잘림. |
| 2 | Quantile levels | 필요한 quantile 의 목록. |
| 3 | Context length | 최근 국면만 따르도록 줄인 window. |
| 4 | Covariate path | 앞으로의 covariate 값. 가정을 바꿔 물을 때 쓰며, 답과 함께 되돌아옴. |
| 5 | Model version | 지난 답의 재현이나 비교를 위해 고정한 version. |
| 6 | Adaptation depth | 호출 자체에 붙는 fine-tuning 인자. TimeGPT 가 받는 방식 [[2](#ref-2)]. |
| 7 | Fallback policy | context 가 짧거나 낡았을 때 원하는 처리. 거절, fallback, degraded 표시 중 하나. |
| 8 | Level of detail | 점, quantile, 성분, attribution 중 무엇까지 받을지. |

Table 6. 사후에 되돌려 보내는 feedback

| # | Feedback | Content |
|---|----------|---------|
| 1 | Actuals | 이미 답한 cut-off 의 실제 관측값. 계열과 timestamp 로 keying. |
| 2 | Verdict | anomaly flag 에 대한 사람의 오탐·진탐 판정. |
| 3 | Correction | serving 된 답 대신 사용한 값. 관측과 섞지 않고 별도 계열로 보관. |
| 4 | Event marks | 정비, 세정, recipe 변경처럼 계열을 뛰게 만든 사건. |
| 5 | Measurement request | 다음에 무엇을 측정할지에 대한 물음. server 는 자기 불확실성으로 답함. |
| 6 | Retrain or promote | 재적합이나 승격 요청. governed job 의 대기열로 감. |

## 5. Deployment Patterns

Fig 1 의 경로에서 답을 만드는 자리를 어디에 두느냐가 pattern 을 가른다.

Table 7. Deployment pattern 과 각각이 맞는 자리

| # | Pattern | Description |
|---|---------|-------------|
| 1 | Scheduled batch | 정해진 주기로 모든 key 를 미리 계산해 table 에 기록. 소비자가 드물게 읽을 때 가장 싼 방식. |
| 2 | Online request-response | 요청 하나에 지연 예산 안에서 답함. 답이 caller 의 인자에 달렸을 때 필요. |
| 3 | Streaming push | 사건마다 state 를 갱신하고 묻지 않아도 답을 냄. recursive model 을 높은 사건율에서 유지하는 방식. |
| 4 | Edge | compile 된 model 을 발생지 옆에서 실행. 왕복이 제어 주기를 넘거나 자료가 현장을 떠날 수 없을 때. |

## 6. Platforms

이 꼭지는 위의 요구사항을 어디까지 이미 제공하는 제품이 있는지를 정리한다. 범용 model server 는 batching 과 version 관리와 autoscaling 을 주지만, Table 4 의 1 부터 3 까지는 사용자 몫으로 남긴다.

Table 8. 범용 model server

| # | Platform | Description |
|---|----------|-------------|
| 1 | NVIDIA Triton Inference Server | 여러 framework runtime 과 dynamic batching. sequence batcher 가 한 sequence 를 한 model instance 로 routing [[3](#ref-3)]. |
| 2 | KServe | InferenceService 를 정의하는 CNCF incubating project. 요청 기반 autoscaling 과 scale-to-zero [[4](#ref-4)]. |
| 3 | BentoML | Python 추론 code 를 adaptive batching 과 함께 container 로 포장. |
| 4 | Ray Serve | 여러 model 을 공유 replica 위에 multiplexing. key 별 version 배분에 맞음. |
| 5 | MLflow model serving | model 을 `pyfunc` 으로 serving. 추론이 library 호출인 통계 model 의 통상 경로. |
| 6 | TorchServe | 제한 유지보수. 갱신도 보안 patch 도 예정 없음 [[5](#ref-5)]. |

Table 9. Pre-trained 시계열 model

| # | Model | Description |
|---|-------|-------------|
| 1 | TimeGPT (Nixtla) | API key 로 접근하는 hosted endpoint. forecasting 과 anomaly detection [[2](#ref-2)]. |
| 2 | Chronos-2 (Amazon) | Open-weight, 약 120M parameter. 단변량·다변량·covariate 입력에 zero-shot, quantile 직접 생성 [[6](#ref-6)]. |
| 3 | TimesFM (Google) | Open-weight decoder-only. 2.5 release 는 200M parameter 와 16k context [[7](#ref-7)]. |
| 4 | Moirai (Salesforce) | 2.0 에서 open-weight 이며 `uni2ts` library 로 serving [[8](#ref-8)]. |
| 5 | Granite TTM (IBM) | TinyTimeMixer 계열. stream processor 안에 넣을 만큼 작음 [[9](#ref-9)]. |
| 6 | Toto (Datadog) | 관측 지표용. quantile 출력과 시간·변량 교대 attention [[10](#ref-10)]. |

Table 10. Framework 와 store

| # | Component | Description |
|---|-----------|-------------|
| 1 | AutoGluon-TimeSeries | 고전·machine learning·pre-trained model 을 탐색하고 ensemble. 확률 예측이 목표 [[11](#ref-11)]. |
| 2 | Nixtla `statsforecast`, `mlforecast`, `neuralforecast` | 고전·feature 기반·neural 경로를 하나의 data contract 위에서. |
| 3 | Apache Flink | checkpoint 와 watermark 를 갖춘 keyed state [[1](#ref-1)]. recursive model 이 있을 자리. |
| 4 | Online store (Redis, DynamoDB) | 최근 `L` 점과 static attribute 를 밀리초 예산 안에 제공. |
| 5 | Feature store (Feast, Tecton) | 하나의 feature 정의를 학습과 serving 양쪽에 materialize. |
| 6 | Time series database (InfluxDB, TimescaleDB, ClickHouse) | 재계산과 backtest 가 읽는 history. |
| 7 | Segment index (FAISS, pgvector, Milvus) | 과거 segment 의 embedding 과 그 속성. 비교 질문의 답. |

Table 11. Managed service

| # | Service | Description |
|---|---------|-------------|
| 1 | Amazon SageMaker AI | real-time, serverless, asynchronous, batch endpoint 와 시계열 AutoML 경로. |
| 2 | Google BigQuery ML | 내장 TimesFM model 에 SQL 로 묻는 `AI.FORECAST`. warehouse 안에서 끝나는 경우 server 가 필요 없음. |
| 3 | Vertex AI, Azure Machine Learning, Databricks | endpoint 와 registry 와 monitoring. Table 4 의 요구사항은 그 위에 따로 지어야 함. |

## 7. Selection

Table 12. 제약과 그것이 강제하는 선택

| # | Constraint | Choice |
|---|------------|--------|
| 1 | 소비자가 자료 갱신보다 훨씬 드물게 읽음. | scheduled batch 로 table 에 기록. |
| 2 | Recursive model 과 끊이지 않는 사건. | keyed state 를 지닌 streaming engine. |
| 3 | 계열은 많고 key 마다 이력은 적음. | global 이나 pre-trained model 하나로 통일. |
| 4 | 과거 구간과의 비교가 주된 질문. | model 이 아니라 segment index 를 짓고 일정에 올림. |
| 5 | 사람의 판단이 있어야 값이 생기는 답. | feedback store 와 그 수집 절차를 먼저 마련. |
| 6 | 자료가 현장을 떠날 수 없음. | self-hosted 또는 open-weight 경로. |
| 7 | 소비 loop 가 망 왕복보다 빠름. | edge 로 내보내고 좁아진 model 선택을 받아들임. |

## 8. Operational Pitfalls

- serving window 와 학습 window 를 서로 다른 code 가 만들어, 각자 일관된 채로 어긋남.
- 상류에서 channel 의 단위가 바뀌었는데 값이 schema 범위 안이라 걸러지지 않음.
- 재학습 trigger 를 horizon 보다 짧은 잔차 window 로 걸어, scoring 이 끝나지 않은 답 위에서 발동.
- Correction 을 관측과 함께 쌓아, model 이 자기 보정된 출력을 다시 학습.
- 호출당 adaptation 인자에 예산과 quota 를 두지 않아, endpoint 가 학습 부하를 떠안음.

## References

<a id="ref-1"></a>
[1] Apache Flink. [Working with state](https://github.com/apache/flink/blob/master/docs/content/docs/dev/datastream/fault-tolerance/state.md). The Apache Software Foundation.<br>
<a id="ref-2"></a>
[2] Nixtla. [TimeGPT](https://github.com/Nixtla/nixtla). Nixtla.<br>
<a id="ref-3"></a>
[3] NVIDIA. [Batchers](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/batcher.md). NVIDIA Corporation.<br>
<a id="ref-4"></a>
[4] KServe authors. [KServe](https://github.com/kserve/kserve). Cloud Native Computing Foundation.<br>
<a id="ref-5"></a>
[5] PyTorch. [TorchServe](https://github.com/pytorch/serve). The Linux Foundation.<br>
<a id="ref-6"></a>
[6] Amazon Science. [Chronos: pretrained models for time series forecasting](https://github.com/amazon-science/chronos-forecasting). Amazon.<br>
<a id="ref-7"></a>
[7] Google Research. [TimesFM](https://github.com/google-research/timesfm). Google.<br>
<a id="ref-8"></a>
[8] Salesforce AI Research. [uni2ts: unified training of universal time series forecasting transformers](https://github.com/SalesforceAIResearch/uni2ts). Salesforce.<br>
<a id="ref-9"></a>
[9] IBM Granite. [granite-tsfm](https://github.com/ibm-granite/granite-tsfm). IBM.<br>
<a id="ref-10"></a>
[10] Datadog. [Toto: time-series-optimized transformer for observability](https://github.com/DataDog/toto). Datadog.<br>
<a id="ref-11"></a>
[11] AutoGluon. [AutoGluon](https://github.com/autogluon/autogluon). The AutoGluon community.

---

## Appendix A. Terminology

- **Attribution**: 어느 channel, step, lag 이 그 답을 움직였는지에 대한 설명.
- **Backfill**: 과거 cut-off 에 대해 serving 경로로 답을 다시 계산하는 일.
- **Baseline**: 정상 거동으로 삼는 기준. deviation score 는 이것에 대해 잼.
- **Chamber**: 한 wafer 나 batch 가 처리되는 process tool 의 내부 공간.
- **Checkpoint**: state 를 나중에 복구할 수 있도록 저장한 사본.
- **Context**: model 이 소비하는 과거 관측의 window. 길이는 model 이 정함.
- **Covariate**: target 이 아니면서 model 이 읽는 변수. 관측만 되면 past, 미래 값이 미리 정해지면 future known.
- **Cut-off**: model 이 볼 수 있는 것과 예측해야 할 것을 가르는 timestamp.
- **Deviation score**: 관측이 model 의 기대에서 얼마나 떨어져 있는지. threshold 가 자르기 전까지 판정을 담지 않음.
- **Drift**: model 이 담은 관계의 변화. 한때 옳던 model 을 나중에 그르게 만듦.
- **Embedding**: segment 를 비교하거나 index 할 때 그것을 대신하는 고정 길이 vector.
- **Fault detection and classification (FDC)**: run 의 sensor trace 로부터 장비가 의도대로 거동했는지 판단하는 일.
- **Governed job**: 요청 밖에서 도는 job. serving 되는 model 을 바꿀 수 있는 유일한 주체.
- **Horizon**: 답이 덮는 앞으로의 step 수. `H` 로 씀.
- **Key**: 하나의 계열을 가리키는 식별자.
- **Keyed state**: stream processor 가 key 별로 들고 checkpoint 에서 복구하는 state.
- **L**: context 의 길이. 최근 관측 몇 개를 model 에 넣을지를 정함.
- **Lot**: 공정을 함께 지나가는 wafer 무리.
- **Metrology**: 공정이 무엇을 만들었는지 보고하는 측정 단계. 공정 뒤에 수행.
- **Model registry**: version 의 catalog 와, 어느 version 이 어느 key 를 맡을지 정하는 규칙.
- **Quantile**: 예측 분포의 한 지점. 값 하나 대신 여러 지점을 보고할 때 씀.
- **Recipe and step**: tool 이 한 제품을 위해 실행하는 program 과 그 한 구간.
- **Recursive model**: 관측마다 자기 state 를 갱신하는 model. Kalman filter, exponential smoother, online learner 가 여기 속함.
- **Run**: 한 wafer 나 batch 에 대한 recipe 한 번의 실행.
- **Scale-to-zero**: 놀고 있는 endpoint 의 replica 를 모두 없애는 것.
- **Scoring**: 나중에 도착한 실측과 답을 맞춰 보는 일.
- **Segment**: 한 계열의 경계 지어진 구간. 길이나 사건으로 경계를 정함.
- **Segment index**: 과거 segment 의 embedding 을 담아, forward pass 없이 비교 질문에 답하는 store.
- **Trace**: 한 run 동안 한 sensor 를 표본으로 기록한 것.
- **Virtual metrology**: metrology 가 잴 값을, 그 측정이 존재하기 전의 자료로 추정하는 일.
- **Wafer**: 공정을 지나가며 metrology 가 측정하는 기판.
- **Watermark**: 그보다 이른 사건은 오지 않는다고 보는 event-time 경계.
- **Window model**: 주어진 context 만의 함수인 model. 호출 사이에 state 를 남기지 않음.
- **Zero-shot**: 한 번도 적합된 적 없는 model 로 그 계열에 답하는 것.

## Appendix B. Case: Fault Detection And Classification

FDC 는 앞의 구성이 한 줄기 stream 위에서 한꺼번에 요구되는 사례이다. Process tool 의 sensor 가 run 내내 표본을 남기고, 그 run 이 recipe 대로 거동했는지를 다음 run 이 끝나기 전에 판정해야 한다.

Table 13. FDC 에서 각 구성 요소가 맡는 일

| # | Component | Role in FDC |
|---|-----------|-------------|
| 1 | Ingest path | run 을 recipe step 으로 분절. 이후의 모든 질문이 이 segment 를 대상으로 함. |
| 2 | Online store | tool, chamber, recipe, step, sensor 의 조합을 key 로 최근 run 을 보관. |
| 3 | Model registry | 이력 없는 chamber 나 recipe 를 global model 로 routing. |
| 4 | Segment index | 지금 step 과 닮은 과거 run 을 거리순으로 반환. 사고를 과거 사례에 연결. |
| 5 | Inference server | step 별 deviation score 와 channel 기여를 다음 run 전에 게시. |
| 6 | Prediction store | 게시된 판정을 보관. metrology 가 도착하면 결합해 scoring. |
| 7 | Feedback store | engineer 의 판정. run 단위에서 얻을 수 있는 유일한 label. |
| 8 | Governed job | 측정된 wafer 로 재적합하고 후보를 승격. |

이 사례에서 특히 무거운 요구사항은 셋이다. 첫째, context 의 경계가 시계가 아니라 recipe step 이므로 분절이 ingest 경로의 일이 된다. 둘째, label 이 되는 metrology 는 표본으로만 도착하므로 scoring 된 부분집합이 serving 된 모집단과 다르다. 셋째, 정비와 세정은 의도된 불연속이므로 drift 로 오인하지 않도록 event mark 로 알려야 한다.
