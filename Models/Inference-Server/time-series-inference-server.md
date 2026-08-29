# Time Series Inference Server
Rev. 19 | Created: 2026-08-28 | Updated: 2026-08-29 09:20 CDT

적합된 model 은 숫자를 돌려준다. 계열이 계속 도착하는 동안 그 숫자를 무언가가 행동으로 옮길 수 있는 답으로 바꾸는 일이 inference server 의 몫이다. 이 문서는 그 일이 무엇인지, caller 가 무엇을 요구할 수 있는지, 그리고 어떤 제품이 이미 그 일을 하고 있는지를 정한다.

## 1. Scope

- 서빙만 다룸. 적합된 model 이나 pre-trained checkpoint 에서 시작해 답이 소비되는 지점까지.
- 범위 밖: 학습 절차, model family, feature 구성.
- 계열이 수천 개인 경우를 전제. 계열 하나면 scheduled job 하나로 끝나고 이 문서의 어느 것도 필요 없음.

### 1.1 The Test For A Servable Question

- Model: 적합된 함수. 정해진 한 가지 모양의 array 를 받아 숫자를 냄. key 도 시계도 없고, 직전에 무엇을 답했는지도 모름.
- Server: 그 함수를 둘러싼 전부. key 와 cut-off 를 array 로 바꾸고, model 이 들지 않는 history 와 state 와 지난 답을 들고, 돌려주는 것에 식별자를 찍음.
- Servable question: 숫자는 model 이 대고, 나머지 전부는 server 가 댈 수 있는 질문.
- Table 1 의 `Servable` 열이 적용하는 판정이자, 그 마지막 열이 재는 것.
- [Appendix B](#appendix-b-the-model-and-the-server): 같은 구분을 operation 이름과 함께 그린 그림.

### 1.2 Questions A Served Series Can Be Asked

Table 1. 질문, deliverable, 그리고 각 질문이 deployment 에 더하는 것

| Question | Servable | Deliverable | What the server must add |
|----------|----------|-------------|--------------------------|
| Forecast | Yes | 요청한 quantile 마다 timestamp 와 값 `H` 행. cut-off 와 model version 을 달고. | 없음. |
| Anomaly detection | Yes | point 나 segment 마다 한 행: timestamp, deviation score, threshold, flag. | score 를 재는 기준인 baseline. 그리고 server 가 소유하는 threshold 규칙. |
| Retrieval | Yes | segment 식별자, 거리, key, 시간 범위를 담은 `k` 행. 거리순. | segment index, 그리고 그것을 다시 만드는 job. |
| Classification | Yes | segment 마다 한 행: class, 그리고 모든 class 의 확률. | feedback store 에 쌓인, label 이 붙은 segment. |
| Change point detection | Yes | timestamp 목록. 각각 거동이 얼마나 급하게 바뀌었는지의 척도와 함께. | 없음. 다만 요청이 segment 전체를 실음. |
| Imputation | Yes | 메운 값. 각각 관측이 아니라 imputed 로 표시. | 같은 key 의 형제 channel. |
| Virtual metrology | Yes | run 마다 구간이 딸린 추정치 하나. 측정이 존재하기 전에. | 나중에 도착하는 측정과의 결합. |
| Sequence target from a trace tensor | Yes | forecast 또는 추정 deliverable. `[sequence, feature, trace]` 입력으로부터 key 와 sequence step 마다 한 행. | 들어오는 길에 trace 축을 요약값으로 줄이거나, tensor 를 통째로 받는 model. 어느 쪽이든 segment 경계는 ingest 경로가 정함. |
| Remaining useful life | Yes | 남은 run 수나 시간. 구간과 함께, 부품 단위로. | 고장과 교체의 event history. |
| What-if | Yes | forecast deliverable, 그리고 그것이 조건으로 삼은 covariate path. | 없음. 다만 미리 계산해 둘 수 없음. |
| Attribution | Yes | channel, lag, step 별 기여. 설명 대상 편차에 합이 맞도록. | 답과 함께 저장되는 설명. |
| Cause | No | 대신 attribution 이 돌아옴. 답과 함께 움직인 것이지, 답을 움직인 것이 아님. | 없음. 개입이나 설계된 실험의 몫이고, 둘 다 요청이 아님. |
| A model fitted on demand | No | 대신 호출 한정 adaptation, recursive model 이면 section 2.2 의 state 갱신. | 없음. 새 적합은 모든 caller 의 답을 바꾸므로 governed job 의 몫. |

algorithm 이 아니라 마지막 열이 deployment 의 모양을 정한다.

- model 과 context 만으로 충분: forecast, change point detection, what-if.
- segment index 와 그것을 유지하는 job: retrieval.
- label, 그래서 section 4.2 의 feedback path 가 먼저: classification, virtual metrology, remaining useful life.
- baseline 과 threshold 규칙: anomaly detection.
- 더 넓은 요청이나 더 넓은 저장: imputation, attribution, trace tensor 로부터의 sequence target.
- 경계를 긋는 No 두 행. 원인을 규명하라거나 model 을 적합하라고 요구받은 server 는 서빙을 그만둔 것.

## 2. Core Capabilities

가운데 열은 범용 model server 가 전제하는 것이며, section 5.1 의 제품들이 멈추는 지점이다.

Table 2. server 가 caller 에게 지는 의무와, stateless model server 의 전제

| Capability | Stateless assumption | Owed by a time series server |
|------------|----------------------|------------------------------|
| Context assembly | payload 가 model 이 쓸 feature 를 전부 담고 있음. | cut-off 기준으로 그 key 의 마지막 `L` 관측. 쓰기에 너무 성긴 window 는 거절. |
| Horizon and uncertainty | 요청마다 값 하나. | horizon 의 모든 step 을 quantile 이나 구간과 함께. 각각 미래 timestamp 에 묶어서. |
| Covariate handling | 모든 입력이 그저 같은 feature. | past covariate, future known covariate, static attribute. 미리 알 수 없는 future covariate 는 거절. |
| Timestamp discipline | 시계 자체가 없음. | 하나의 frequency, 하나의 timezone, 결측과 중복에 대한 하나의 규약. 학습 때와 서빙 때가 동일. |
| Series fan-out | 요청 하나에 행 하나. | 한 번의 호출로 여러 key. accelerator 를 채우도록 batch 로 묶고, 각각 자기 version 으로 routing. |
| State and recovery | 무상태이므로 어느 replica 나 어느 요청이나 답함. | 계열별 state 를 들고 checkpoint. history 재생 없이 복구. |
| Cold start | 처음 보는 행은 흔한 일. | 이력 없는 key 는 global 이나 pre-trained model 로 routing, 아니면 거절. |
| Freshness and drift | 입력의 신선도는 caller 가 만든 만큼. | 답 뒤에 놓인 최신 관측의 나이, 그리고 재학습을 부르는 신호. |
| Delayed evaluation | label 이 사건과 함께 도착. | 모든 답을 context 와 version 과 함께 보존. label 은 `H` step 뒤에 도착하므로. |
| Throughput and latency | 독립된 요청을 마음대로 batch. | batching, 컴파일된 runtime, autoscaling 으로 예산을 지킴. timeout 이 아니라 더 싼 model 로 강등. |
| Reproducibility | 같은 입력이면 같은 답. | 지난 요청을 같은 답으로 재생. version, context, code path 를 함께 고정. |

일반적인 model serving 에 대응물이 없는 세 가지를 아래에 적는다.

### 2.1 Context Assembly

- forward pass 이전에 끝나는, 이 일의 핵심: key 와 시각을 model 이 학습한 window 로 바꾸는 것.
- payload 에 window 를 싣기: server 는 무상태가 되지만, payload 가 커지고 조립 부담이 모든 caller 에게 감.
- online store 에서 window 를 읽기: payload 는 작아지지만, 임계 경로에 read 가 하나 늘고 낡은 데이터를 줄 의존이 생김.
- Point-in-time correctness: cut-off 시점에 관측 가능했던 것만. timestamp 가 그보다 이르다는 것과 같은 말이 아님.
- 제자리에서 덮어쓰는 store: 학습 집합과 서빙 경로가 어긋나고, 그 어긋남은 언제나 학습 집합에 유리함.
- 결측은 보간하거나, 앞 값을 끌고 가거나, 결측인 채로 둠. model 이 적합될 때 본 것과 맞춰야 하므로 server 의 선택.
- context 보다 짧은 window: 짧아도 되는 model 로 넘기거나 거절. 0 으로 채우는 일은 없음.

### 2.2 State And Recovery

- Window model, 예를 들어 lag feature 위의 boosted regressor 나 pre-trained transformer: context 의 순수 함수이므로 replica 가 서로 대체 가능.
- Recursive model, 예를 들어 Kalman filter, exponential smoother, online learner: 관측마다 순서대로 갱신되는 state 를 지님.
- 뒤쪽은 server 를 stateful 하게 만듦. 한 key 의 요청이 그 state 를 든 replica 에 닿거나, state 가 외부 store 에 있어야 함.
- state 는 checkpoint. 다시 만드는 일은 계열을 처음부터 재생하는 일이므로.
- 순서를 벗어난 도착은 거절하거나 watermark 까지 되감음. 어제 표본을 오늘 표본 뒤에 적용하면 state 가 영구히 망가짐.
- 장애를 견디는 keyed state [\[1\]](#ref-1), 거기에 watermark 와 event-time 순서: 서빙 이야기에 streaming engine 이 등장하는 이유.

### 2.3 Delayed Evaluation

- `t+H` 의 forecast 는 `t+H` 가 지나기 전에는 채점 불가. 그때쯤이면 server 는 그것을 잊었음.
- version 과 context 가 달라지는 순간, 입력만으로는 복원 불가.
- 그래서 모든 답을 cut-off, horizon step, quantile, model version, context 와 함께 기록.
- scheduled job 이 그 행들을 실측과 결합해 key 별·horizon step 별 오차 계열을 만듦.
- 그 오차 계열이 상시 두 결정의 유일하게 정직한 입력: 재학습 여부와, 어느 후보를 서빙할지.

## 3. Deployment Patterns

```text
Producers                Ingest / Transport          Context
+-----------+            +------------------+        +---------------------+
| sensors   |            |                  |        | online store        |  <- last L points, per key
| equipment | ---------> | broker / stream  | -----> | key-value or a time |
| apps      |            |                  |        | series database     |
+-----------+            +------------------+        +---------------------+
                                  |                           |
                                  |                           v
                                  |                  +---------------------+
                                  |                  | inference server    |     +-----------------+
                                  +----------------> | - context assembly  | <-> | segment index   |
                                   (push path)       | - batching          |     | (retrieval)     |
                                                     | - model resolution  |     +-----------------+
                                                     +---------------------+
                                                          |          |
                                     model registry <-----+          v
                                     (versions,               +---------------------+
                                      per-key routing)        | prediction store    |  <- scored later
                                                              +---------------------+
                                                                       |
                                                                       v
                                                              dashboards, alarms,
                                                              control actions
```

Fig 1. 수집에서 소비까지의 서빙 경로

Table 3. 배포 pattern 과 각각이 맞는 자리

| Pattern | Description |
|---------|-------------|
| Scheduled batch | 정해진 주기로 모든 key 를 예측해 table 에 기록. 가장 싸고, 소비자가 드물게 읽을 때 맞는 방식. |
| Online request-response | 지연 예산 안에서 key 하나에 답함. 답이 caller 가 주는 인자에 달렸을 때 필수. |
| Streaming push | 사건마다 keyed state 를 갱신하고 묻지 않아도 답을 냄. 높은 사건율에서 recursive model 을 온전히 유지하는 유일한 방식. |
| Edge or on-premises | 컴파일된 model 을 발생지 옆에서 실행. 왕복이 제어 주기를 넘거나 raw trace 가 현장을 떠날 수 없을 때. |

## 4. The Caller's Interface

Caller 는 deployment 가 스스로 얻을 수 없는 두 가지를 쥐고 있는데, 요청 뒤에 있는 의도와 답에 대한 진실이다.

```text
(a) options in, answer out, inside one round trip

    caller --- request + options ---> server === model ---> answer + ids ---> caller

(b) truth back, and the model changed later, never inside that round trip

    caller --- actuals, verdicts, ---> feedback store ---> governed job ---> a new
               corrections,                                                  version
               event marks                                                     |
                                                                               v
                                                              the version (a) answers from,
                                                              from that point onward
```

Fig 2. Interface 의 두 방향

### 4.1 Options That Change The Answer

요청마다 지정. adaptation depth 하나를 빼면 model 을 건드리지 않는다.

Table 4. Caller 가 요청에 지정할 수 있는 option

| Option | Description |
|--------|-------------|
| Horizon | caller 가 행동에 옮길 수 있는 step 수. model 이 적합된 최대 horizon 으로 잘림. |
| Quantile levels | caller 가 필요한 quantile. 직접 내놓는 model 에는 공짜, 아닌 model 에는 sampling 비용. |
| Context length | 최근 국면을 따라가도록 줄인 window. 학습 길이를 넘는 부분은 잘림. |
| Covariate path | caller 가 주는 future covariate 값. what-if 를 묻는 방법이며, 답과 함께 되돌려 줌. |
| Model version | 지난 답을 재현하거나 비교를 돌리려고 고정한 version. 생략하면 registry 가 routing 하는 것. |
| Adaptation depth | 호출 자체에 붙는 fine-tuning 인자. TimeGPT 가 받는 방식 [\[2\]](#ref-2). 지연과 비용을 주고 더 맞는 fit 을 얻음. |
| Fallback policy | context 가 짧거나 낡았을 때 caller 가 원하는 것: 거절, fallback, 또는 degraded 표시가 붙은 답. |
| Level of detail | 점, quantile, 성분 분해, attribution 중 무엇을 받을지. 그리고 언제나 채점을 가능하게 하는 식별자. |

### 4.2 The Feedback Path

사후에 제출된다. Classification, virtual metrology, remaining useful life 는 이 경로 없이는 서빙 자체가 성립하지 않는데, 그 label 이 여기로 오거나 아예 오지 않기 때문이다.

Table 5. Caller 가 되돌려 보낼 수 있는 feedback

| Feedback | Description |
|----------|-------------|
| Actuals | 이미 답한 cut-off 의 관측값. 계열과 timestamp 로 keying 하여 중복 제출이 두 번 세어지지 않게. |
| Verdict on a flag | 사람이 매긴 오탐·진탐 판정. deployment 가 받게 될 유일한 label 인 경우가 많음. |
| Correction | caller 가 서빙된 답 대신 쓴 값. 별도 계열로 보관하여, model 이 자기가 보정된 출력 위에서 적합되지 않게. |
| Event marks | 계열을 뛰게 만든 정비, 세정, recipe 변경, 제품 전환. drift detector 가 의도된 불연속으로 읽음. |
| Measurement request | 다음에 무엇을 측정할지. server 자신의 불확실성으로 답함. sampling plan 에 적용한 active learning. |
| Retrain or promote | 재적합이나 후보 승격 요청. governed job 의 대기열로 감. |

### 4.3 Rules For The Write Path

- Feedback 은 append 만. 제자리 적용은 없음.
- 한 caller 가 서빙 중인 model 을 바꾸게 하는 endpoint 는 다른 모든 caller 의 답을 바꾸고, 재현성을 끝냄.
- 살아남는 형태: 계열·timestamp·출처로 keying 된 feedback store, 그리고 무엇을 바꿀지 정하는 governed job.
- 모든 제출은 보낸 주체와 어느 답에 대한 것인지를 지님. correction 은 쓰이는 순간 label 이 되고, label 은 철회해야 할 수 있으므로.
- 호출당 adaptation 인자는 서빙 요금으로 청구되는 학습 비용. 예산과 cache 와 caller 별 quota 가 필요.

## 5. Key Solutions and Platforms

### 5.1 General Purpose Model Servers

Endpoint 뒤의 어떤 model 이든 서빙하되 시간에 대해서는 아무것도 모른다. Context assembly 는 caller 나 그 위에 쓴 wrapper 의 몫으로 남는다.

Table 6. 범용 model server

| Platform | Description |
|----------|-------------|
| NVIDIA Triton Inference Server | 여러 framework runtime 을 dynamic batching 과 함께. sequence batcher 가 한 sequence 를 한 model instance 로 routing [\[3\]](#ref-3). 이 부류에서 recursive model 을 직접 받쳐 주는 유일한 기능. |
| KServe | InferenceService 를 정의하는 CNCF incubating project. 요청 기반 autoscaling 과 scale-to-zero [\[4\]](#ref-4). 다른 runtime 을 앞에서 감쌀 수 있음. |
| BentoML | Python 추론 code 를 adaptive batching 과 함께 container 로 포장. 평범한 객체 형태의 forecasting code 에 맞음. |
| Ray Serve | model 을 graph 로 조합하고, 여러 model 을 공유 replica 위에 multiplexing. key 별 model 해소와 맞음. |
| MLflow model serving | model 을 `pyfunc` 으로 서빙. 추론이 library 호출인 통계 model 의 통상 경로. |
| TensorFlow Serving | 저장된 TensorFlow graph 에는 안정적이고, 그 밖에는 별 것 없음. |
| TorchServe | 제한 유지보수. 예정된 갱신도, 수정도, 보안 patch 도 없음 [\[5\]](#ref-5). 새 작업의 선택지가 아님. |

어느 것도 section 4.2 의 feedback path 를 주지 않는다. 그쪽으로 내놓는 것은 payload logging 뿐인데, 나중에 복원할 수 없으므로 첫날부터 켜 둘 값어치가 있다.

### 5.2 Time Series Foundation Models

Key 마다 하나씩 두던 model 무리를 checkpoint 하나가 대신한다. 계열 간 차이는 받아들이는 입력의 모양, quantile 이 한 번의 pass 에서 나오는지 sampling 에서 나오는지, 그리고 forward pass 의 크기다.

Table 7. Pre-trained model 과 서빙 방식

| Model | Description |
|-------|-------------|
| TimeGPT (Nixtla) | API key 로 접근하는 hosted endpoint. forecasting 과 함께 anomaly detection 을 다룸 [\[2\]](#ref-2). 데이터를 내보낼 수 없는 곳을 위한 self-hosted 배포도 있음. |
| Chronos-2 (Amazon) | Open-weight, 약 120M parameter. 단변량·다변량·covariate 입력에 zero-shot 이며 quantile 을 직접 생성 [\[6\]](#ref-6). |
| TimesFM (Google) | Open-weight, decoder-only. 2.5 release 는 200M parameter, 16k context, 선택적 quantile head [\[7\]](#ref-7). |
| Moirai (Salesforce) | 2.0 에서 open-weight 이며 `uni2ts` library 를 통해 서빙 [\[8\]](#ref-8). 공급사 의존 대신 library 를 배포에 싣는 형태. |
| Granite TTM (IBM) | IBM 의 TinyTimeMixer 계열. 자체 library 와 benchmark 를 갖춤 [\[9\]](#ref-9). stream processor 안에 넣을 만큼 작음. |
| Toto (Datadog) | 관측 지표를 위해 학습. 수백만에서 수십억 parameter 까지의 계열이며, quantile 출력과 시간·변량 교대 attention [\[10\]](#ref-10). |

- retrieval 에도 같은 checkpoint. forecast 로 가는 길에 만들어지는 표현이 곧 segment index 가 저장하는 것이므로.
- open weight 로 retrieval 을 서빙하는 비용: 새 segment 당 쓰기 시점 forward pass 한 번, 읽기 시점은 없음.
- leaderboard 순위는 후보를 추리는 근거이지 채택의 근거가 아님. 결정하는 평가는 대상 계열 위에서 naive baseline 과 겨루는 것.

### 5.3 Forecasting Frameworks

Table 6 의 server 가 싣는 model 을 만들어 낸다. 대부분 batch 진입점도 함께 내놓는데, scheduled pattern 에는 그것만으로 충분하다.

Table 8. Model 과 batch inference 를 공급하는 framework

| Framework | Description |
|-----------|-------------|
| AutoGluon-TimeSeries | 고전 model, machine learning model, pre-trained model 을 탐색하고 이긴 것을 ensemble. 확률 예측을 목표로 명시 [\[11\]](#ref-11). |
| Nixtla `statsforecast`, `mlforecast`, `neuralforecast` | 고전·feature 기반·neural 경로를 하나의 data contract 위에서. 큰 계열 무리를 한 번의 호출로 다루도록 설계. |
| GluonTS, Darts, sktime | 각각 확률 model 구현과 평가 harness, 하나의 API 뒤의 통계·deep model 과 backtesting, scikit-learn 호환 interface. |
| STUMPY, tslearn | matrix profile 과 elastic distance. retrieval 과 change point 질문을 모양만으로 답함. |

### 5.4 Streaming Engines And Stores

Section 2 의 state 와 context 요구를 받는 층이다. Push pattern 에서는 이 층 자체가 서빙 층이다.

Table 9. Streaming 과 저장 구성 요소

| Component | Description |
|-----------|-------------|
| Apache Kafka | 생산자와 server 를 분리하고, 같은 사건으로 online 경로와 offline store 를 먹이며, 답을 다시 실어 보냄. |
| Apache Flink | checkpoint 와 watermark 를 갖춘 keyed state. 원격 model 을 호출하거나 작은 model 을 직접 실행. recursive model 이 있을 자리. |
| Spark Structured Streaming | micro-batch 위의 batch code. 밀리초가 아니라 분 단위 주기에 맞음. |
| Online store (Redis, DynamoDB) | 마지막 `L` 점과 key 의 static attribute 를 밀리초 예산 안에. section 2.1 이 임계 경로에 두는 read. |
| Feature store (Feast, Tecton) | feature 정의 하나를 offline 과 online 양쪽에 materialize. 학습 window 와 서빙 window 를 같게 유지. |
| Time series database (InfluxDB, TimescaleDB, ClickHouse, Prometheus) | backfill 과 backtest 가 읽는 history. 보존 규칙이 긴 context 에 무엇이 남는지를 정함. |
| Segment index (FAISS, pgvector, Milvus) | 과거 segment 마다 embedding 하나와, 비교가 걸러야 할 attribute. 그 재구축 일정이 서빙 설계의 일부. |

### 5.5 Managed And Warehouse-Native Services

Table 10. Managed service 와 in-database forecasting

| Service | Description |
|---------|-------------|
| Amazon SageMaker AI | real-time, serverless, asynchronous, batch endpoint 와 시계열을 다루는 AutoML 경로. Amazon Forecast 는 2024-07-29 부터 신규 고객에게 닫힘. |
| Google BigQuery ML | 내장 TimesFM model 에 대해 SQL 로 예측하는 `AI.FORECAST`. horizon 과 context window 를 인자로 받음. scheduled pattern 에서는 server 자체가 사라짐. |
| Vertex AI, Azure Machine Learning, Databricks | 사용자가 학습한 model 주위의 endpoint, registry, monitoring. section 2 의 일은 그 위에 따로 지어야 함. |

### 5.6 Edge Runtimes

- ONNX Runtime, OpenVINO, TensorFlow Lite: 내보낸 model 을 발생지 옆에서 실행.
- 내보내기가 model 선택을 제한. 추론이 Python 제어 흐름으로 쓰였다면 살아남지 못함.
- 운영 비용이 확장에서 배포로 옮겨감. 이제 host 마다 굴리고 되돌려야 할 version 이 하나씩.

## 6. Selection

Table 11. 제약과 그것이 강제하는 선택

| Constraint | Choice |
|------------|--------|
| 데이터가 갱신되는 빈도보다 소비자가 훨씬 드물게 읽음. | scheduled batch 로 table 에 기록. caller 입력에 달린 답에만 endpoint 를 더함. |
| Recursive model 과 끊이지 않는 사건. | request-response endpoint 가 아니라 keyed state 를 지닌 streaming engine. |
| 계열은 많고 key 마다 이력은 적음. | global 이나 pre-trained model 하나. 그 수에서는 key 별 model 을 유지할 수 없으므로. |
| 지금 이것이 어느 과거 segment 를 닮았는가 하는 질문. | segment index 를 짓고 일정에 올림. 어떤 forecasting endpoint 도 답하지 않으므로. |
| 사람이 판단하기 전에는 값이 없는 답. | feedback path 를 먼저. 그 판정이 앞으로 도착할 유일한 label 이므로. |
| 현장을 떠날 수 없는 raw data. | hosted API 의 정확도가 어떻든 self-hosted 나 open-weight 경로. |
| 망 왕복보다 빠른 소비 loop. | edge 로 내보내고, 그에 따르는 좁은 model 선택을 받아들임. |
| 이미 warehouse 에 있는 데이터와 일 단위 주기. | 무엇을 짓기 전에 in-database forecasting 함수. |

## 7. Operational Pitfalls

Section 2 가 이미 금지하지 않는, 그리고 어떤 test 에도 걸리지 않는 실패들.

- 서빙 window 와 학습 window 를 서로 다른 code 가 만들어, 각자 일관된 채로 resampling·결측·timezone 에서 어긋남.
- 상류에서 channel 의 단위가 바뀌었는데, 값이 schema 범위 안이라 아무도 거르지 않음.
- 재학습 trigger 를 horizon 보다 짧은 잔차 window 로 걸어, 아직 다 채점되지 않은 답 위에서 발동.

## References

<a id="ref-1"></a>
[1] Apache Flink. [Working with state](https://github.com/apache/flink/blob/master/docs/content/docs/dev/datastream/fault-tolerance/state.md). The Apache Software Foundation.

<a id="ref-2"></a>
[2] Nixtla. [TimeGPT](https://github.com/Nixtla/nixtla). Nixtla.

<a id="ref-3"></a>
[3] NVIDIA. [Batchers](https://github.com/triton-inference-server/server/blob/main/docs/user_guide/batcher.md). NVIDIA Corporation.

<a id="ref-4"></a>
[4] KServe authors. [KServe](https://github.com/kserve/kserve). Cloud Native Computing Foundation.

<a id="ref-5"></a>
[5] PyTorch. [TorchServe](https://github.com/pytorch/serve). The Linux Foundation.

<a id="ref-6"></a>
[6] Amazon Science. [Chronos: pretrained models for time series forecasting](https://github.com/amazon-science/chronos-forecasting). Amazon.

<a id="ref-7"></a>
[7] Google Research. [TimesFM](https://github.com/google-research/timesfm). Google.

<a id="ref-8"></a>
[8] Salesforce AI Research. [uni2ts: unified training of universal time series forecasting transformers](https://github.com/SalesforceAIResearch/uni2ts). Salesforce.

<a id="ref-9"></a>
[9] IBM Granite. [granite-tsfm](https://github.com/ibm-granite/granite-tsfm). IBM.

<a id="ref-10"></a>
[10] Datadog. [Toto: time-series-optimized transformer for observability](https://github.com/DataDog/toto). Datadog.

<a id="ref-11"></a>
[11] AutoGluon. [AutoGluon](https://github.com/autogluon/autogluon). The AutoGluon community.

---

## Appendix A. Terminology

- Active learning: model 이 가장 자신 없어 하는 것을 기준으로 다음에 label 할 대상을 고르는 방식.
- Attribution: 어느 channel, step, lag 이 그 답을 움직였는지에 대한 설명. 답을 만드는 동안 함께 계산됨.
- Backfill: 과거 cut-off 에 대해 서빙 경로로 답을 다시 계산하는 일. 새 model 이 만들었을 history 를 얻기 위함.
- Chamber: 한 wafer 나 batch 가 처리되는 process tool 의 내부 공간. 대부분의 FDC 계열이 keying 되는 단위.
- Change point: 계열의 거동이 바뀌는 timestamp.
- Context: model 이 소비하는 과거 관측의 window. 그 길이는 model 이 정함.
- Covariate: target 이 아니면서 model 이 읽는 변수. 관측될 뿐이면 past, 미래 시점의 값이 미리 정해지면 future known.
- Cut-off: model 이 볼 수 있는 것과 예측해야 할 것을 가르는 timestamp.
- Deviation score: 관측이 model 의 기대에서 얼마나 떨어져 있는지. baseline 에 대해 재며, threshold 가 자르기 전까지는 어떤 판정도 담지 않음.
- Drift: model 이 담은 관계의 변화. 한때 옳던 model 을 나중에 그르게 만듦.
- Dynamic batching: 따로 도착한 요청을 한 번의 forward pass 로 모으는 것. 유계의 대기 지연을 값으로 치름.
- Embedding: segment 를 비교하거나 index 할 때 그 segment 를 대신하는 고정 길이 vector.
- Fault detection and classification (FDC): run 의 sensor trace 로부터 장비가 의도대로 거동했는지 판단하는 일.
- Foundation model: 여러 계열로 pre-train 되어, 한 번도 적합된 적 없는 계열을 서빙하는 model.
- Governed job: 어떤 요청 밖에서 도는 job. server 가 서빙하는 것을 바꿀 수 있는 유일한 주체.
- Horizon: 답이 덮는 앞으로의 step 수. `H` 로 씀.
- Keyed state: stream processor 가 key 별로 들고 checkpoint 에서 복구하는 state.
- Lot: 공정을 함께 지나가는 wafer 무리.
- Metrology: 공정이 무엇을 만들었는지 보고하는 측정 단계. 공정 중이 아니라 공정 뒤에 수행.
- Model registry: version 의 catalog, 그리고 어느 version 이 어느 key 를 서빙할지 정하는 규칙.
- Naive baseline: 마지막 관측이나 한 계절 전 관측을 그대로 되풀이하는 예측. 어떤 model 이든 이겨야 하는 기준.
- Point-in-time correctness: context 안의 모든 값이 cut-off 시점에 관측 가능했다는 성질. timestamp 가 그보다 이르다는 것만으로는 부족함.
- Production line: lot 이 지나가는 공정 단계와 tool 의 연쇄. 하나의 FDC deployment 가 맡는 범위.
- Quantiles: horizon step 마다 예측 분포의 여러 지점. 값 하나가 아님.
- Recipe and step: tool 이 한 제품을 위해 실행하는 program, 그리고 그 program 의 한 구간.
- Remaining useful life: 부품이 한계를 넘기까지 남은 시간이나 run 수.
- Retrieval: 지금 것과 닮은 과거 segment 가 무엇인가 하는 질문. segment index 로 답함.
- Run: 한 wafer 나 batch 에 대한 recipe 한 번의 실행.
- Scale-to-zero: 놀고 있는 endpoint 의 replica 를 모두 없애는 것. 비용 대신 cold start 를 치름.
- Scoring: 나중에 도착한 실측과 답을 맞춰 보는 일.
- Segment: 한 계열의 경계 지어진 구간. 길이로 정하거나 step transition 같은 사건으로 정함.
- Segment index: 과거 segment 마다 embedding 하나를 담아, forward pass 없이 retrieval 에 답하는 store.
- Trace: 한 run 동안 한 sensor 를 표본으로 기록한 것.
- Trace tensor: `[sequence, feature, trace]` 모양의 array. 한 축은 run 의 순서, 다른 한 축은 run 안의 시간.
- Virtual metrology: metrology 가 재는 값을, 그 측정이 존재하기 전의 데이터로 추정하는 일.
- Wafer: 공정을 지나가며 metrology 가 측정하는 기판.
- Watermark: 그보다 이른 사건은 오지 않는다고 보는 event-time 경계.
- Zero-shot: 한 번도 적합된 적 없는 model 로 그 계열에 답하는 것.

## Appendix B. The Model And The Server

```text
+---------------------------------------------------------------------------+
| SERVER   it knows the key, the clock, the history, the versions           |
|                                                                           |
|   request (key, cut-off, options)                                         |
|        |                                                                  |
|        v                                                                  |
|   (1) resolve    key + cut-off        -> the version that answers         |
|   (2) assemble   history + covariates -> one array of shape [L x C]       |
|   (3) validate   gap, staleness, short window -> answer, degrade, refuse  |
|   (4) batch      many keys                    -> one call                 |
|        |                                                                  |
|        v                                                                  |
|      +----------------------------------------------------+               |
|      |  MODEL   a fitted function, and nothing else       |               |
|      |     in   an array of the one shape it was trained  |               |
|      |          on                                        |               |
|      |    out   numbers                                   |               |
|      |    not   which key, what time it is, what it       |               |
|      |          answered before, whether it was right     |               |
|      +----------------------------------------------------+               |
|        |                                                                  |
|        v                                                                  |
|   (5) shape      numbers -> rows of (timestamp, quantile, value)          |
|   (6) stamp      + model version, cut-off, answer id                      |
|   (7) persist    the answer -> prediction store                           |
|        |                                                                  |
|        v                                                                  |
|   answer + identifiers                                                    |
+---------------------------------------------------------------------------+
        ^ reads                                    | writes
        |                                          v
 +-------------------------------+   +-------------------------------------+
 | online store   last L points  |   | prediction store  every past answer |
 | model registry versions       |   | feedback store    actuals, verdicts,|
 | segment index  past segments  |   |                   corrections,      |
 | feedback store past labels    |   |                   event marks       |
 +-------------------------------+   +-------------------------------------+
        ^                                          |
        |                                          v
        |    +--------------------------------------------------+
        +--- | OUT OF BAND   a governed job, never in a request  |
             |   (8)  score     answer + actual -> error series  |
             |   (9)  detect    drift -> retrain trigger         |
             |   (10) fit       a new candidate model            |
             |   (11) promote   candidate -> the served version  |
             +--------------------------------------------------+
```

Fig 3. Server 안의 model 과 그 둘레의 operation

- Operation (1) 부터 (7) 까지: 요청 경로이자 section 2 의 core capability.
- (2) 는 context assembly, (3) 은 그에 따르는 거절.
- (5) 와 (6) 이 답을 채점 가능하게 만들고, (7) 이 delayed evaluation 이 나중에 읽는 것.
- Recursive model 의 state 는 이 operation 중 어느 것도 아니라 model 호출 자체에서 나아감. server 가 그 state 에 지는 의무는 section 2.2.
- 어느 store 를 그 질문이 더 요구하는지는 Table 1 의 마지막 열.
- Operation (8) 부터 (11) 은 요청 안에서 돌지 않음. 요청 안의 적합에 대한 No 가 뜻하는 바.
- 시간에 관한 것은 아무것도 model 로 넘어가지 않음. timestamp, 결측, cut-off 에 대한 결정은 forward pass 전에 이미 끝났고 그 pass 가 고칠 수 없음.
- Model 에서 나오는 화살표는 답이 아님. 숫자와, 누구든 채점할 수 있는 것 사이에 shaping 과 stamping 과 write 가 있음.

## Appendix C. Case: What A Server Makes Possible On A Semiconductor Production Line

FDC (Fault detection and classification) 는 Table 1 의 질문 대부분이 한 stream 에 한꺼번에 던져지는 경우다.

Table 12. Inference server 가 반도체 production line 에 올리는 기능

| Class | Function | Description |
|-------|----------|-------------|
| Detection | Step deviation score | 끝난 run 의 각 step 을 그 recipe 와 step 에 대한 model 의 기대에 비추어 잼. 그 score 를 다음 run 이 끝나기 전에 게시. |
| Detection | Hold decision | model 밖의 규칙이 score 를 자름. 중요한 layer 는 좁은 한계로, 너그러운 layer 는 넓은 한계로. 두 번째 model 없이. |
| Diagnosis | Channel attribution | alarm 이 순위 매긴 channel 별 기여를 달고 옴. 먼저 볼 곳이 이름으로 지목됨. |
| Diagnosis | Nearest run search | 지금 step 을 같은 recipe 의 저장된 step 들과 비교. 가장 가까운 run 들이 거리와 그때의 처분과 함께 돌아옴. |
| Diagnosis | Fault classification | 과거 run 에 판정이 붙어 있는 곳에서는, flag 가 확률을 지닌 이름 있는 class 로 도착. |
| Estimation | Virtual metrology | metrology 가 나중에 표본으로 잴 값을, 모든 run 에 구간과 함께 추정. |
| Estimation | Remaining useful life | 부품이나 소모품을 그것이 넘을 한계까지 투영. run 수나 시간으로, trace 옆의 event history 로부터. |
| Planning | Setpoint evaluation | 제안된 setpoint 를 적용 전에 조건부 forecast 로 답함. 그 제안을 답과 함께 저장. |
| Planning | Sampling nomination | 이미 준 답의 불확실성으로 key 를 순위 매김. metrology 계획이 그 순위를 따를 수 있음. |
| Coverage | Cold-start routing | 이력이 없는 chamber 나 recipe 를 global 이나 pre-trained model 로 routing. 첫 run 부터 덮임. |
| Coverage | Fleet comparison | 모든 tool 에 같은 key 구조를 서빙. 한 chamber 를 자기 과거뿐 아니라 동료 집단과 비교. |

어느 한 행이 아니라 이 묶음에서 나오는 효과가 셋 있다.

- 기능들이 하나의 조립을 공유. 나중에 fault class 나 측정 추정을 더하는 일은 두 번째 pipeline 이 아니라 소비자 하나를 더하는 일.
- 답이 기록이 됨. 각각 context 와 version 과 함께 저장되므로 지난 분기의 정확도를 지금 계산할 수 있음.
- Loop 가 닫힘. 판정과 뒤늦은 측정이 같은 경로로 돌아와 다음 달에 서빙할 model 을 적합함.

이 모든 것 뒤에 조건 하나가 서 있는데, run 이 들어오는 길에 recipe step 으로 분절되어 있어야 한다는 것이다.
