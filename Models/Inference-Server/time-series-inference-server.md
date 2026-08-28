# Time Series Inference Server
Rev. 10 | Created: 2026-08-28 | Updated: 2026-08-28 12:20 CDT

A fitted model returns numbers. Turning those numbers into an answer that something can act on, while the series keeps arriving, is the work of an inference server. This document fixes what that work is, what a caller may ask of it, and which products already do it.

## 1. Scope

- The document covers serving only, from a fitted model or a pre-trained checkpoint to a consumed answer.
- Training procedures, model families, and feature construction are outside it.
- It assumes thousands of series, because a single series needs a scheduled job and none of what follows.

### 1.1 Model And Server

- A model is a fitted function that takes an array of one shape and returns numbers.
- It knows no key, no clock, and nothing it answered before.
- A server turns a key and a cut-off into that array, and refuses when what it assembles is too poor to answer from.
- It keeps what the model does not, which is the history, the per-series state, the past answers, and any index or labels a question needs.
- It stamps every answer with the identifiers that let the answer be scored later.
- A question is servable when the model gives the numbers and the server can give that remainder.
- [Appendix B](#appendix-b-the-model-and-the-server) draws the split with the operations named.

### 1.2 Questions A Served Series Can Be Asked

Table 1. Questions, deliverables, and what each one adds to the deployment

| Question | Servable | Deliverable | What the server must add |
|----------|----------|-------------|--------------------------|
| Forecast | Yes | `H` rows of timestamp and value per requested quantile, carrying the cut-off and the model version. | Nothing. |
| Anomaly detection | Yes | One row per point or segment: timestamp, deviation score, threshold, flag. | A baseline for the score, and the threshold rule, which the server owns. |
| Retrieval | Yes | `k` rows of segment identifier, distance, key, and time range, ordered by distance. | A segment index and the job that rebuilds it. |
| Classification | Yes | One row per segment, with the class and a probability for every class. | Labeled segments in the feedback store. |
| Change point detection | Yes | Timestamps, each with a measure of how sharply the behavior changed. | Nothing, but the request carries a whole segment. |
| Imputation | Yes | Filled values, each marked as imputed rather than observed. | The sibling channels of the same key. |
| Virtual metrology | Yes | One estimate with an interval per run, before the measurement exists. | The join to the measurement that arrives later. |
| Remaining useful life | Yes | Remaining runs or hours, with an interval, per component. | An event history of failures and replacements. |
| What-if | Yes | The forecast deliverable, plus the covariate path it is conditional on. | Nothing, but the answer cannot be precomputed. |
| Attribution | Yes | A contribution per channel, lag, or step, summing to the deviation explained. | The explanation stored with the answer. |
| Cause | No | An attribution comes back instead, which names what moved with the answer. | Nothing. A cause needs an intervention or a designed experiment. |
| A model fitted on demand | No | A per-call adaptation comes back instead, and for a recursive model the state update of section 3.2. | Nothing. A new fit changes every caller's answer, so it belongs to a governed job. |

The last column, not the algorithm, decides the shape of the deployment.

- Forecast, change point detection, and what-if need a model and a context.
- Retrieval needs a segment index, which adds a job that maintains it.
- Classification, virtual metrology, and remaining useful life need labels, so the feedback path of section 5.2 is a precondition.
- Anomaly detection needs a baseline and a threshold rule, while imputation and attribution widen the request or the stored answer.
- The two rows answered No fix the boundary: a server asked to establish a cause or to fit a model has stopped serving.

## 2. Divergence From A Stateless Model Server

A general model server assumes a self-contained request, and every one of those assumptions breaks for a series.

Table 2. Serving a row against serving a series

| Aspect | Stateless model server | Time series inference server |
|--------|------------------------|------------------------------|
| Request payload | It holds every feature the model consumes. | It holds a key and a timestamp, and the history has to be found. |
| Output shape | It is one value per request. | It is horizon steps by quantiles, each tied to a future timestamp. |
| State | Any replica answers any request. | A recursive model's state has to be held or restored by the replica that answers. |
| Correctness | It is a property of the code path. | It also depends on the cut-off, since a context holding what the model could not have seen is silently wrong. |
| Evaluation | The label arrives with the event. | The label arrives `H` steps later, so the answer has to be stored to be scored. |
| Cold start | An unseen row is ordinary. | A new series has no history and needs a global model or a fallback. |

## 3. Core Capabilities

Table 3. What the server owes its caller

| Capability | Requirement |
|------------|-------------|
| Context assembly | It resolves the last `L` observations of the key as of the cut-off, and refuses a window too sparse to use. |
| Horizon and uncertainty | It returns every horizon step with quantiles or an interval, never a point alone. |
| Covariate handling | It accepts past covariates, future known covariates, and static attributes, and rejects a future covariate that is not known in advance. |
| Timestamp discipline | It fixes one frequency, one timezone, and one convention for gaps and duplicates, identical at fit time and at serving time. |
| Series fan-out | It answers many keys in one call, batches them to fill the accelerator, and resolves which version serves which key. |
| State and recovery | It holds the per-series state a recursive model needs, checkpoints it, and restores it without replaying the history. |
| Freshness and drift | It exposes the age of the newest observation behind each answer, and emits the signal that triggers retraining. |
| Delayed evaluation | It persists every answer with its context and model version, so accuracy can be computed when the actuals land. |
| Throughput and latency | It meets its budget by batching, a compiled runtime, and autoscaling, and degrades to a cheaper model rather than to a timeout. |
| Reproducibility | It replays a past request to the same answer, which needs the version, the context, and the code path pinned together. |

The three capabilities with no counterpart in ordinary model serving follow.

### 3.1 Context Assembly

- The defining work happens before the forward pass, turning a key and a time into the window the model was trained on.
- Carrying the window in the payload keeps the server stateless, at the cost of a large payload and of assembly work in every caller.
- Fetching it from an online store keeps the payload small, at the cost of a read on the critical path and a dependency that can serve stale data.
- The window must hold only what was observable at the cut-off, which is not the same as what carries an earlier timestamp.
- A store that overwrites in place therefore makes the training set and the serving path disagree, and the disagreement flatters the training set.
- A gap is interpolated, carried forward, or left as a gap, and the choice is the server's, since it has to match what the model saw while it was fitted.
- A window shorter than the model's context is answered by a model that tolerates it or declined, never padded with zeros.

### 3.2 State And Recovery

- A window model, such as a boosted regressor over lags or a pre-trained transformer, is a pure function of the context, so replicas are interchangeable.
- A recursive model, such as a Kalman filter, an exponential smoother, or an online learner, carries a state that each observation updates in order.
- Serving the second kind makes the server stateful: one key's requests reach the replica holding its state, or that state lives in an external store.
- The state is checkpointed, because rebuilding it means replaying the series from its beginning.
- Out-of-order arrivals are rejected or rolled back to a watermark, since applying yesterday's sample after today's corrupts the state permanently.
- Keyed state that survives a failure [\[1\]](#ref-1), with watermarks and event-time ordering, is why streaming engines appear in a serving discussion at all.

### 3.3 Delayed Evaluation

- A forecast for `t+H` cannot be scored until `t+H` has passed, by which time the server has forgotten it.
- Nothing else recovers it, because the input alone does not determine the answer once versions and contexts vary.
- Each answer is therefore written with its cut-off, its horizon step, its quantiles, the model version, and its context.
- A scheduled job joins those rows to the actuals and produces the error series per key and per horizon step.
- That error series is the only honest input to the two standing decisions, which are whether to retrain and which candidate should serve.

## 4. Deployment Patterns

Fig 1. Serving path from ingestion to consumption

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

Table 4. Deployment patterns and what each one fits

| Pattern | Description |
|---------|-------------|
| Scheduled batch | A job forecasts every key on a cadence and writes a table, which is the cheapest pattern and the right one when consumers read rarely. |
| Online request-response | An endpoint answers one key inside a latency budget, which is required when the answer depends on a parameter the caller supplies. |
| Streaming push | The engine holds keyed state and emits an answer on each event, which is the only pattern that keeps recursive models correct at high rates. |
| Edge or on-premises | A compiled model runs next to the source, which is chosen when the round trip exceeds the control loop or the raw trace may not leave the site. |

## 5. The Caller's Interface

The caller holds two things the deployment cannot obtain on its own, which are the intent behind a request and the truth about an answer.

Fig 2. The two directions of the interface

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

### 5.1 Options That Change The Answer

Each is set per request. All but the adaptation depth leave the model untouched.

Table 5. Options a caller may set on a request

| Option | Description |
|--------|-------------|
| Horizon | The caller asks for the steps it can act on, and the server clamps to the longest horizon the model was fitted for. |
| Quantile levels | The caller names the quantiles it needs, which is free for a model that emits them directly and costs sampling passes for one that does not. |
| Context length | The caller shortens the window to follow a recent regime, and the server truncates anything beyond the trained length. |
| Covariate path | The caller supplies future covariate values, which is how a what-if is asked, and the server returns that path with the answer. |
| Model version | The caller pins a version to reproduce an answer or to run a comparison, and omits it to take whatever the registry routes. |
| Adaptation depth | Some hosted models accept fine-tuning arguments on the call, as TimeGPT does [\[2\]](#ref-2), trading latency and cost for a closer fit. |
| Fallback policy | The caller states what it wants when the context is short or stale, which is to decline, to fall back, or to answer with a degraded flag. |
| Level of detail | The caller asks for a point, quantiles, components, or an attribution, and always for the identifiers that let the answer be scored. |

### 5.2 The Feedback Path

Each is submitted after the fact. Classification, virtual metrology, and remaining useful life cannot be served without this path, since their labels arrive through it or not at all.

Table 6. Feedback a caller may send back

| Feedback | Description |
|----------|-------------|
| Actuals | The observed values for a cut-off already answered, keyed by series and timestamp so that a repeat cannot be counted twice. |
| Verdict on a flag | A person marks a flag as a true or a false alarm, which is often the only label the deployment ever receives. |
| Correction | The number a caller substituted for the served answer, kept as its own series so the model is never fitted on its own adjusted output. |
| Event marks | The maintenance, cleaning, recipe change, or product switch that made the series jump, which the drift detector reads as deliberate. |
| Measurement request | The caller asks what to measure next, and the server answers from its own uncertainty, which is active learning applied to a sampling plan. |
| Retrain or promote | The caller asks that a model be refitted or a candidate promoted, and the request is queued for a governed job. |

### 5.3 Rules For The Write Path

- Feedback is appended, never applied in place.
- An endpoint that lets one caller change the serving model changes every other caller's answers, and ends reproducibility.
- The shape that survives is a feedback store keyed by series, timestamp, and source, plus a governed job that decides what changes.
- Every submission carries who sent it and which answer it refers to, since a correction becomes a label and a label may have to be withdrawn.
- Corrections are kept apart from observations, or the model learns to reproduce the adjustment made to its own output.
- A per-call adaptation argument is a training cost billed as a serving call, so it needs a budget, a cache, and a quota per caller.

## 6. Key Solutions and Platforms

### 6.1 General Purpose Model Servers

These serve any model behind an endpoint and know nothing about time. Context assembly is left to the caller or to a wrapper.

Table 7. General purpose model servers

| Platform | Description |
|----------|-------------|
| NVIDIA Triton Inference Server | It runs several framework runtimes with dynamic batching, and its sequence batcher routes one sequence to one model instance [\[3\]](#ref-3), which is the only support for a recursive model in this class. |
| KServe | It is a CNCF incubating project defining an InferenceService with request-based autoscaling and scale-to-zero [\[4\]](#ref-4), and it can front another runtime. |
| BentoML | It packages Python inference code into a container with adaptive batching, which suits forecasting code that is an ordinary object. |
| Ray Serve | It composes models into a graph and multiplexes many across shared replicas, which matches per-key model resolution. |
| MLflow model serving | It serves a model as a `pyfunc`, which is the usual path for statistical models whose inference is a library call. |
| Seldon Core | It deploys on Kubernetes behind an inference protocol compatible with the KServe one. |
| TensorFlow Serving | It remains stable for saved TensorFlow graphs and offers little else. |
| TorchServe | It is under limited maintenance with no planned updates, fixes, or security patches [\[5\]](#ref-5), so it is not a choice for new work. |

None of them supplies the feedback path of section 5.2. What they offer toward it is payload logging, which is worth turning on from the first day because it cannot be reconstructed later.

### 6.2 Time Series Foundation Models

One checkpoint replaces a population of per-key models. The families differ in the input they accept, in whether quantiles come from one pass or from sampling, and in the size of a forward pass.

Table 8. Pre-trained models and how they are served

| Model | Description |
|-------|-------------|
| TimeGPT (Nixtla) | It is a hosted endpoint reached with an API key and covers anomaly detection as well as forecasting [\[2\]](#ref-2), with a self-hosted deployment for sites that cannot send data out. |
| Chronos-2 (Amazon) | It is open-weight, about 120M parameters, zero-shot for univariate, multivariate, and covariate-informed inputs, and it generates quantiles directly [\[6\]](#ref-6). |
| TimesFM (Google) | It is open-weight and decoder-only, and its 2.5 release carries 200M parameters, a 16k context, and an optional quantile head [\[7\]](#ref-7). |
| Moirai (Salesforce) | Its 2.0 release is open-weight and served through the `uni2ts` library [\[8\]](#ref-8), so the deployment carries a library rather than a vendor. |
| Granite TTM (IBM) | It is IBM's TinyTimeMixer family with its own library and benchmarks [\[9\]](#ref-9), small enough to embed in a stream processor. |
| Toto (Datadog) | It is trained for observability metrics, sized from a few million to a few billion parameters, with quantile output and alternating time and variate attention [\[10\]](#ref-10). |

- The same checkpoints answer retrieval, because the representation formed on the way to a forecast is what a segment index stores.
- Retrieval from open weights therefore costs one forward pass per new segment at write time and none at read time.
- A leaderboard rank is a reason to shortlist a model, not to adopt one, since the deciding evaluation is the one run on the target series against a naive baseline.

### 6.3 Forecasting Frameworks

These produce the models the servers of Table 7 carry, and most expose a batch entry point that is enough for the scheduled pattern.

Table 9. Frameworks that supply models and batch inference

| Framework | Description |
|-----------|-------------|
| AutoGluon-TimeSeries | It searches classical, machine learning, and pre-trained models and ensembles what wins, targeting probabilistic forecasts [\[11\]](#ref-11). |
| Nixtla `statsforecast`, `mlforecast`, `neuralforecast` | They cover the classical, feature-based, and neural routes on one data contract, built for large populations in one call. |
| GluonTS | It supplies probabilistic model implementations and the evaluation harness for them. |
| Darts | It presents statistical and deep models behind one API with backtesting. |
| sktime | It supplies a scikit-learn compatible interface for forecasting and classification over series. |
| STUMPY, tslearn | They compute matrix profiles and elastic distances, answering retrieval and change point questions by shape alone. |

### 6.4 Streaming Engines And Stores

This layer answers the state and context requirements of section 3, and in the push pattern it is the serving layer itself.

Table 10. Streaming and storage components

| Component | Description |
|-----------|-------------|
| Apache Kafka | It decouples producers from the server, feeds the online and offline paths from the same events, and carries the published answers. |
| Apache Flink | It holds keyed state with checkpointing and watermarks and can call a remote model or run a small one inline, which is where a recursive model belongs. |
| Spark Structured Streaming | It runs batch code over micro-batches, which suits minute-scale cadences rather than millisecond ones. |
| Online store (Redis, DynamoDB) | It serves the last `L` points and the static attributes within a millisecond budget, which is the read section 3.1 puts on the critical path. |
| Feature store (Feast, Tecton) | It defines a feature once and materializes it offline and online, which keeps the training window and the serving window identical. |
| Time series database (InfluxDB, TimescaleDB, ClickHouse, Prometheus) | It holds the history backfills and backtests read, and its retention rules decide what a long context can contain. |
| Segment index (FAISS, pgvector, Milvus) | It holds one embedding per past segment with the attributes a comparison filters on, and its rebuild schedule is part of the serving design. |

### 6.5 Managed And Warehouse-Native Services

Table 11. Managed services and in-database forecasting

| Service | Description |
|---------|-------------|
| Amazon SageMaker AI | It offers real-time, serverless, asynchronous, and batch endpoints, and its AutoML path covers time series. Amazon Forecast has been closed to new customers since 2024-07-29. |
| Google BigQuery ML | Its `AI.FORECAST` function forecasts in SQL against a built-in TimesFM model, taking the horizon and the context window as arguments, which removes the server for the scheduled pattern. |
| Vertex AI, Azure Machine Learning, Databricks | They provide endpoints, registries, and monitoring around a model the user trains, leaving the work of section 3 to be built on top. |

### 6.6 Edge Runtimes

- ONNX Runtime, OpenVINO, and TensorFlow Lite execute an exported model next to the source.
- The export limits model choice, since inference written as Python control flow does not survive it.
- The operational cost moves from scaling to distribution, because every host now carries a version to roll forward and back.

## 7. Selection

Table 12. Constraint and the choice it forces

| Constraint | Choice |
|------------|--------|
| Consumers read far less often than the data updates. | Run scheduled batch and write a table, adding an endpoint only for answers that depend on caller input. |
| The model is recursive and events arrive continuously. | Put it in a streaming engine with keyed state, not behind a request-response endpoint. |
| The population is large and each key has little history. | Serve one global or pre-trained model, since per-key models cannot be maintained at that count. |
| The question is which past segment this one resembles. | Build and schedule a segment index, since no forecasting endpoint answers it. |
| The answer is worth nothing until a person judges it. | Build the feedback path first, since that verdict is the only label that will arrive. |
| Raw data may not leave the site. | Take the self-hosted or open-weight route, whatever a hosted API's accuracy. |
| The consuming loop is faster than a network round trip. | Export the model to the edge and accept the narrower model choice. |
| The data is in a warehouse and the cadence is daily. | Use the in-database forecasting function before building anything. |

## 8. Operational Pitfalls

Each is a capability of section 3 left to the caller.

- A covariate that is not known in advance is fed as a future covariate, so the backtest is excellent and the live forecast is not.
- The serving window and the training window are built by different code, and disagree about resampling, gaps, or timezone.
- A late sample overwrites a slot after an answer for it was issued, so the stored answer is scored against a history it never saw.
- A new key with a short history is padded, and the model reads the padding as a regime.
- A channel's unit changes upstream, and nothing rejects it because the value stays inside the schema's range.
- Retraining is triggered on a residual window shorter than the horizon, so it fires on answers not yet fully scored.

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

- Active learning: choosing what to label next by what the model is least certain about.
- Attribution: the account of which channel, step, or lag moved an answer, computed while the answer is produced.
- Backfill: recomputing past answers through the serving path, to get the history a new model would have produced.
- Chamber: the enclosure of a process tool in which one wafer or batch is processed, and the grain most FDC series are keyed at.
- Change point: a timestamp at which the behavior of a series changes.
- Context: the window of past observations a model consumes, whose length the model fixes.
- Covariate: a variable other than the target that the model reads, past when observed and future known when decided in advance.
- Cut-off: the timestamp separating what the model may see from what it is asked to predict.
- Deviation score: how far an observation lies from what the model expected, measured against a baseline and carrying no decision until a threshold cuts it.
- Drift: a change in the relationship the model encodes, which makes a correct model incorrect later.
- Dynamic batching: collecting independent requests into one forward pass, at the cost of a bounded queueing delay.
- Embedding: the fixed-length vector standing in for a segment when segments are compared or indexed.
- Fault detection and classification (FDC): judging from a run's sensor trace whether the equipment behaved as intended.
- Foundation model: a model pre-trained on many series that serves a series it was never fitted to.
- Governed job: a job outside any request, and the only thing allowed to change what the server serves.
- Horizon: the number of steps ahead an answer covers, written `H`.
- Idempotency: the property that a repeated submission leaves the store as one submission would have.
- Keyed state: state a stream processor holds per key and restores from a checkpoint.
- Lot: the group of wafers that moves through the process together.
- Metrology: the measurement step that reports what a process produced, performed after it rather than during it.
- Model registry: the catalog of versions and the rule deciding which version serves which key.
- Naive baseline: repeating the last observation, or the one a season earlier, which any model has to beat.
- Point-in-time correctness: every value in a context was observable at the cut-off, not merely stamped before it.
- Quantile forecast: several quantiles of the predictive distribution per horizon step rather than one value.
- Recipe and step: the program a tool executes for a product, and one segment of that program.
- Remaining useful life: the time or number of runs before a component crosses a limit.
- Retrieval: which past segments resemble the one in hand, answered from a segment index.
- Run: one execution of a recipe on one wafer or batch.
- Scale-to-zero: removing every replica of an idle endpoint, which trades its cost for a cold start.
- Scoring: comparing an answer with the actual that arrived later.
- Segment: a bounded stretch of one series, delimited by a length or by an event such as a step transition.
- Segment index: the store holding one embedding per past segment, answering retrieval without a forward pass.
- Trace: the sampled record of one sensor through one run.
- Virtual metrology: estimating what metrology measures, from data available before that measurement exists.
- Wafer: the substrate carried through the process and measured by metrology.
- Watermark: the event-time bound after which no earlier event is expected.
- Zero-shot: answering for a series with a model never fitted to it.

## Appendix B. The Model And The Server

Fig 3. The model inside the server, and the operations around it

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

- Operations (1) to (7) are the request path, and they are the core capabilities of section 3.
- (2) is context assembly and (3) is the refusal that goes with it.
- A recursive model's state advances at the model call itself rather than at one of these operations, and section 3.2 says what the server owes that state.
- (5) and (6) are what make an answer scorable, and (7) is what delayed evaluation later reads.
- The stores hold what the model does not, and the last column of Table 1 says which of them a question adds.
- Operations (8) to (11) never run inside a request, which is what the No against a model fitted on demand means.
- Nothing about time crosses into the model, so every decision about timestamps, gaps, and the cut-off is already made before the forward pass and cannot be repaired by it.
- The arrow leaving the model is not the answer, because the shaping, the stamping, and the write stand between the numbers and something anyone can score.

## Appendix C. Case: What A Server Makes Possible On A Line

Fault detection and classification is the case where most of the questions of Table 1 are asked of one stream at once.

Table 13. Functions an inference server puts on an FDC line

| Class | Function | Description |
|-------|----------|-------------|
| Detection | Step deviation score | Every step of a finished run is measured against what the model expects for that recipe and step, and the score is published before the next run ends. |
| Detection | Hold decision | A rule outside the model cuts the score, so a critical layer runs a tight limit and a tolerant one a loose limit without a second model. |
| Diagnosis | Channel attribution | The alarm carries the ranked per-channel contribution behind it, so the first place to look is named. |
| Diagnosis | Nearest run search | The step in hand is compared with the stored steps of the same recipe, returning the closest runs with their distances and what was decided about them. |
| Diagnosis | Fault classification | Where past runs carry a verdict, a flag arrives as a named class with a probability. |
| Estimation | Virtual metrology | Every run receives an estimate, with an interval, of what metrology will later measure on a sample. |
| Estimation | Remaining useful life | A part or consumable is projected to the limit it will cross, in runs or hours, from the event history beside the traces. |
| Planning | Setpoint evaluation | A proposed setpoint is answered as a conditional forecast before it is applied, and the proposal is stored with the answer. |
| Planning | Sampling nomination | Keys are ranked by the uncertainty of the answers already given, and the metrology plan can follow that ranking. |
| Coverage | Cold-start routing | A chamber or recipe without history is routed to a global or pre-trained model, so it is covered from its first run. |
| Coverage | Fleet comparison | One key structure is served for every tool, so a chamber is compared with its peers rather than only with its own past. |

Three effects come from the set rather than from any row.

- The functions share one assembly, so adding the fault class or the measurement estimate later adds a consumer, not a second pipeline.
- The answers become a record, since each is stored with its context and version, and last quarter's accuracy can be computed now.
- The loop closes, because the verdict and the later measurement return by the same path and fit the model that serves next month.

One condition stands behind all of it, which is that the run is segmented into recipe steps on the way in.
