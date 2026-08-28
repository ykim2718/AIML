# Time Series Inference Server
Rev. 14 | Created: 2026-08-28 | Updated: 2026-08-28 14:40 CDT

A fitted model returns numbers. Turning those numbers into an answer that something can act on, while the series keeps arriving, is the work of an inference server. This document fixes what that work is, what a caller may ask of it, and which products already do it.

## 1. Scope

- Serving only, from a fitted model or a pre-trained checkpoint to a consumed answer.
- Outside it: training procedures, model families, feature construction.
- Thousands of series assumed. One series needs a scheduled job and none of what follows.

### 1.1 Model And Server

- Model: a fitted function. One input shape in, numbers out.
- No key, no clock, no memory of what it answered before.
- Server: key and cut-off turned into that array. A window too poor to answer from, refused.
- Server holds what the model does not — the history, the per-series state, the past answers, any index or labels a question needs.
- Every answer stamped with the identifiers that make it scorable later.
- Servable question: numbers from the model, remainder from the server.
- [Appendix B](#appendix-b-the-model-and-the-server): the same split, with the operations named.

### 1.2 Questions A Served Series Can Be Asked

Table 1. Questions, deliverables, and what each one adds to the deployment

| Question | Servable | Deliverable | What the server must add |
|----------|----------|-------------|--------------------------|
| Forecast | Yes | `H` rows of timestamp and value per requested quantile, with the cut-off and the model version. | Nothing. |
| Anomaly detection | Yes | One row per point or segment: timestamp, deviation score, threshold, flag. | A baseline for the score. A threshold rule, owned by the server. |
| Retrieval | Yes | `k` rows of segment identifier, distance, key, and time range, ordered by distance. | A segment index, and the job that rebuilds it. |
| Classification | Yes | One row per segment: the class, and a probability for every class. | Labeled segments in the feedback store. |
| Change point detection | Yes | Timestamps, each with a measure of how sharply behavior changed. | Nothing, but a whole segment in the request. |
| Imputation | Yes | Filled values, each marked imputed rather than observed. | The sibling channels of the same key. |
| Virtual metrology | Yes | One estimate with an interval per run, before the measurement exists. | The join to the measurement that arrives later. |
| Remaining useful life | Yes | Remaining runs or hours, with an interval, per component. | An event history of failures and replacements. |
| What-if | Yes | The forecast deliverable, plus the covariate path it is conditional on. | Nothing, but no precomputing it. |
| Attribution | Yes | A contribution per channel, lag, or step, summing to the deviation explained. | The explanation, stored with the answer. |
| Cause | No | An attribution instead: what moved with the answer, not what moved it. | Nothing. An intervention or a designed experiment, neither of them a request. |
| A model fitted on demand | No | A per-call adaptation instead, and for a recursive model the state update of section 3.2. | Nothing. A new fit changes every caller's answer, so a governed job owns it. |

The last column, not the algorithm, decides the shape of the deployment.

- A model and a context, nothing more: forecast, change point detection, what-if.
- A segment index and the job that maintains it: retrieval.
- Labels, so the feedback path of section 5.2 comes first: classification, virtual metrology, remaining useful life.
- A baseline and a threshold rule: anomaly detection.
- A wider request or a wider stored answer: imputation, attribution.
- The two No rows as the boundary. A server asked to establish a cause or to fit a model has stopped serving.

## 2. Divergence From A Stateless Model Server

A general model server assumes a self-contained request, and every one of those assumptions breaks for a series.

Table 2. Serving a row against serving a series

| Aspect | Stateless model server | Time series inference server |
|--------|------------------------|------------------------------|
| Request payload | Every feature the model consumes. | A key and a timestamp. The history, to be found. |
| Output shape | One value per request. | Horizon steps by quantiles, each tied to a future timestamp. |
| State | None. Any replica answers any request. | Per-series state, held or restored by the replica that answers. |
| Correctness | A property of the code path. | Also of the cut-off. A context holding what the model could not have seen is silently wrong. |
| Evaluation | Label with the event. | Label `H` steps later, so the answer has to be stored to be scored. |
| Cold start | An unseen row, ordinary. | A new series with no history, needing a global model or a fallback. |

## 3. Core Capabilities

Table 3. What the server owes its caller

| Capability | Requirement |
|------------|-------------|
| Context assembly | The last `L` observations of the key, as of the cut-off. A window too sparse to use, refused. |
| Horizon and uncertainty | Every horizon step with quantiles or an interval. Never a point alone. |
| Covariate handling | Past covariates, future known covariates, static attributes. A future covariate not known in advance, rejected. |
| Timestamp discipline | One frequency, one timezone, one convention for gaps and duplicates. Identical at fit time and at serving time. |
| Series fan-out | Many keys in one call, batched to fill the accelerator, each routed to its version. |
| State and recovery | Per-series state held and checkpointed. Recovery without replaying the history. |
| Freshness and drift | The age of the newest observation behind each answer, and the signal that triggers retraining. |
| Delayed evaluation | Every answer persisted with its context and version, so accuracy can be computed when the actuals land. |
| Throughput and latency | The budget met by batching, a compiled runtime, and autoscaling. Degradation to a cheaper model, not to a timeout. |
| Reproducibility | A past request replayed to the same answer, with version, context, and code path pinned together. |

The three capabilities with no counterpart in ordinary model serving follow.

### 3.1 Context Assembly

- The defining work, done before the forward pass: a key and a time turned into the window the model was trained on.
- Window in the payload: a stateless server, at the cost of a large payload and assembly work in every caller.
- Window from an online store: a small payload, at the cost of a read on the critical path and a dependency that can serve stale data.
- Point-in-time correctness: only what was observable at the cut-off, which is not the same as what carries an earlier timestamp.
- A store that overwrites in place: training set and serving path in disagreement, and the disagreement flatters the training set.
- A gap interpolated, carried forward, or left as a gap. The server's choice, since it has to match what the model saw while it was fitted.
- A window shorter than the context: a model that tolerates it, or a refusal. Never padding with zeros.

### 3.2 State And Recovery

- Window model, such as a boosted regressor over lags or a pre-trained transformer: a pure function of the context, replicas interchangeable.
- Recursive model, such as a Kalman filter, an exponential smoother, or an online learner: a state that each observation updates in order.
- The second kind makes the server stateful. One key's requests reach the replica holding its state, or that state lives in an external store.
- The state checkpointed, since rebuilding it means replaying the series from its beginning.
- Out-of-order arrivals rejected, or rolled back to a watermark. Yesterday's sample applied after today's corrupts the state permanently.
- Keyed state that survives a failure [\[1\]](#ref-1), with watermarks and event-time ordering: why streaming engines appear in a serving discussion at all.

### 3.3 Delayed Evaluation

- A forecast for `t+H`, unscorable until `t+H` has passed, by which time the server has forgotten it.
- No recovery from the input alone, once versions and contexts vary.
- Each answer written with its cut-off, horizon step, quantiles, model version, and context.
- A scheduled job joining those rows to the actuals, producing the error series per key and per horizon step.
- That error series as the only honest input to the two standing decisions: whether to retrain, and which candidate should serve.

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
| Scheduled batch | Every key forecast on a cadence into a table. The cheapest pattern, and the right one when consumers read rarely. |
| Online request-response | One key answered inside a latency budget. Required when the answer depends on a parameter the caller supplies. |
| Streaming push | Keyed state updated on each event, an answer emitted unasked. The only pattern that keeps recursive models correct at high rates. |
| Edge or on-premises | A compiled model next to the source. Chosen when the round trip exceeds the control loop, or the raw trace may not leave the site. |

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

Set per request. All but the adaptation depth leave the model untouched.

Table 5. Options a caller may set on a request

| Option | Description |
|--------|-------------|
| Horizon | The steps the caller can act on, clamped to the longest horizon the model was fitted for. |
| Quantile levels | The quantiles the caller needs. Free from a model that emits them directly, sampling passes from one that does not. |
| Context length | A shortened window, to follow a recent regime. Anything beyond the trained length, truncated. |
| Covariate path | Future covariate values from the caller, which is how a what-if is asked, returned with the answer. |
| Model version | A pinned version, to reproduce an answer or to run a comparison. Omitted, whatever the registry routes. |
| Adaptation depth | Fine-tuning arguments on the call itself, as TimeGPT accepts [\[2\]](#ref-2). Latency and cost traded for a closer fit. |
| Fallback policy | What the caller wants from a short or stale context: a refusal, a fallback, or an answer flagged degraded. |
| Level of detail | A point, quantiles, components, or an attribution. Always the identifiers that make the answer scorable. |

### 5.2 The Feedback Path

Submitted after the fact. Classification, virtual metrology, and remaining useful life cannot be served without this path, since their labels arrive through it or not at all.

Table 6. Feedback a caller may send back

| Feedback | Description |
|----------|-------------|
| Actuals | The observed values for a cut-off already answered. Keyed by series and timestamp, so a repeat cannot be counted twice. |
| Verdict on a flag | A person's true or false alarm. Often the only label the deployment ever receives. |
| Correction | The number a caller substituted for the served answer. Kept as its own series, so the model is never fitted on its own adjusted output. |
| Event marks | The maintenance, cleaning, recipe change, or product switch that made the series jump. Read by the drift detector as deliberate. |
| Measurement request | What to measure next, answered from the server's own uncertainty. Active learning applied to a sampling plan. |
| Retrain or promote | A request to refit or to promote a candidate. Queued for a governed job. |

### 5.3 Rules For The Write Path

- Feedback appended, never applied in place.
- An endpoint that lets one caller change the serving model changes every other caller's answers, and ends reproducibility.
- The shape that survives: a feedback store keyed by series, timestamp, and source, plus a governed job that decides what changes.
- Every submission carrying who sent it and which answer it refers to, since a correction becomes a label and a label may have to be withdrawn.
- A per-call adaptation argument as a training cost billed as a serving call, needing a budget, a cache, and a quota per caller.

## 6. Key Solutions and Platforms

### 6.1 General Purpose Model Servers

Any model behind an endpoint, and no knowledge of time. Context assembly left to the caller or to a wrapper.

Table 7. General purpose model servers

| Platform | Description |
|----------|-------------|
| NVIDIA Triton Inference Server | Several framework runtimes with dynamic batching. Its sequence batcher routes one sequence to one model instance [\[3\]](#ref-3), the only support for a recursive model in this class. |
| KServe | A CNCF incubating project defining an InferenceService, with request-based autoscaling and scale-to-zero [\[4\]](#ref-4). Able to front another runtime. |
| BentoML | Python inference code packaged into a container with adaptive batching. Suits forecasting code that is an ordinary object. |
| Ray Serve | Models composed into a graph, many multiplexed across shared replicas. Matches per-key model resolution. |
| MLflow model serving | A model served as a `pyfunc`. The usual path for statistical models whose inference is a library call. |
| TensorFlow Serving | Stable for saved TensorFlow graphs, and little else. |
| TorchServe | Limited maintenance, with no planned updates, fixes, or security patches [\[5\]](#ref-5). Not a choice for new work. |

None of them supplies the feedback path of section 5.2. What they offer toward it is payload logging, worth turning on from the first day because it cannot be reconstructed later.

### 6.2 Time Series Foundation Models

One checkpoint in place of a population of per-key models. The families differ in the input they accept, in whether quantiles come from one pass or from sampling, and in the size of a forward pass.

Table 8. Pre-trained models and how they are served

| Model | Description |
|-------|-------------|
| TimeGPT (Nixtla) | A hosted endpoint reached with an API key, covering anomaly detection as well as forecasting [\[2\]](#ref-2). A self-hosted deployment for sites that cannot send data out. |
| Chronos-2 (Amazon) | Open-weight, about 120M parameters. Zero-shot for univariate, multivariate, and covariate-informed inputs, with quantiles generated directly [\[6\]](#ref-6). |
| TimesFM (Google) | Open-weight and decoder-only. Its 2.5 release: 200M parameters, a 16k context, an optional quantile head [\[7\]](#ref-7). |
| Moirai (Salesforce) | Open-weight at 2.0, served through the `uni2ts` library [\[8\]](#ref-8). A library in the deployment rather than a vendor. |
| Granite TTM (IBM) | IBM's TinyTimeMixer family, with its own library and benchmarks [\[9\]](#ref-9). Small enough to embed in a stream processor. |
| Toto (Datadog) | Trained for observability metrics, from a few million to a few billion parameters, with quantile output and alternating time and variate attention [\[10\]](#ref-10). |

- The same checkpoints for retrieval, since the representation formed on the way to a forecast is what a segment index stores.
- Retrieval from open weights: one forward pass per new segment at write time, none at read time.
- A leaderboard rank as a reason to shortlist, not to adopt. The deciding evaluation is the one run on the target series against a naive baseline.

### 6.3 Forecasting Frameworks

The models the servers of Table 7 carry. Most also expose a batch entry point, enough on its own for the scheduled pattern.

Table 9. Frameworks that supply models and batch inference

| Framework | Description |
|-----------|-------------|
| AutoGluon-TimeSeries | Classical, machine learning, and pre-trained models searched and ensembled, targeting probabilistic forecasts [\[11\]](#ref-11). |
| Nixtla `statsforecast`, `mlforecast`, `neuralforecast` | The classical, feature-based, and neural routes on one data contract. Built for large populations in one call. |
| GluonTS, Darts, sktime | Probabilistic implementations with an evaluation harness, statistical and deep models behind one API with backtesting, and a scikit-learn compatible interface, in that order. |
| STUMPY, tslearn | Matrix profiles and elastic distances. Retrieval and change point questions answered by shape alone. |

### 6.4 Streaming Engines And Stores

The state and context requirements of section 3. In the push pattern, the serving layer itself.

Table 10. Streaming and storage components

| Component | Description |
|-----------|-------------|
| Apache Kafka | Producers decoupled from the server, online and offline paths fed from the same events, answers published back. |
| Apache Flink | Keyed state with checkpointing and watermarks, and a remote model called or a small one run inline. Where a recursive model belongs. |
| Spark Structured Streaming | Batch code over micro-batches. Minute-scale cadences rather than millisecond ones. |
| Online store (Redis, DynamoDB) | The last `L` points and the static attributes within a millisecond budget. The read section 3.1 puts on the critical path. |
| Feature store (Feast, Tecton) | One feature definition materialized offline and online. Keeps the training window and the serving window identical. |
| Time series database (InfluxDB, TimescaleDB, ClickHouse, Prometheus) | The history backfills and backtests read. Its retention rules decide what a long context can contain. |
| Segment index (FAISS, pgvector, Milvus) | One embedding per past segment, with the attributes a comparison filters on. Its rebuild schedule is part of the serving design. |

### 6.5 Managed And Warehouse-Native Services

Table 11. Managed services and in-database forecasting

| Service | Description |
|---------|-------------|
| Amazon SageMaker AI | Real-time, serverless, asynchronous, and batch endpoints, with an AutoML path over time series. Amazon Forecast closed to new customers since 2024-07-29. |
| Google BigQuery ML | `AI.FORECAST` in SQL against a built-in TimesFM model, taking the horizon and the context window as arguments. No server at all for the scheduled pattern. |
| Vertex AI, Azure Machine Learning, Databricks | Endpoints, registries, and monitoring around a model the user trains. The work of section 3 still to be built on top. |

### 6.6 Edge Runtimes

- ONNX Runtime, OpenVINO, TensorFlow Lite: an exported model executed next to the source.
- Model choice limited by the export, since inference written as Python control flow does not survive it.
- Operational cost moved from scaling to distribution, with a version on every host to roll forward and back.

## 7. Selection

Table 12. Constraint and the choice it forces

| Constraint | Choice |
|------------|--------|
| Consumers reading far less often than the data updates. | Scheduled batch into a table. An endpoint only for answers that depend on caller input. |
| A recursive model and continuous events. | A streaming engine with keyed state, not a request-response endpoint. |
| A large population, little history per key. | One global or pre-trained model, since per-key models cannot be maintained at that count. |
| The question of which past segment this one resembles. | A segment index, built and scheduled, since no forecasting endpoint answers it. |
| An answer worth nothing until a person judges it. | The feedback path first, since that verdict is the only label that will arrive. |
| Raw data that may not leave the site. | The self-hosted or open-weight route, whatever a hosted API's accuracy. |
| A consuming loop faster than a network round trip. | Export to the edge, and the narrower model choice that comes with it. |
| Data already in a warehouse, at a daily cadence. | The in-database forecasting function, before building anything. |

## 8. Operational Pitfalls

Failures that section 3 does not already forbid, and that no test catches.

- Serving window and training window built by different code, each self-consistent, disagreeing about resampling, gaps, or timezone.
- A channel's unit changed upstream, and nothing rejecting it because the value stays inside the schema's range.
- Retraining triggered on a residual window shorter than the horizon, firing on answers not yet fully scored.

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
- Attribution: which channel, step, or lag moved an answer, computed while the answer is produced.
- Backfill: past answers recomputed through the serving path, to get the history a new model would have produced.
- Chamber: the enclosure of a process tool in which one wafer or batch is processed, and the grain most FDC series are keyed at.
- Change point: a timestamp at which the behavior of a series changes.
- Context: the window of past observations a model consumes, its length fixed by the model.
- Covariate: a variable other than the target that the model reads. Past when observed, future known when decided in advance.
- Cut-off: the timestamp separating what the model may see from what it is asked to predict.
- Deviation score: how far an observation lies from what the model expected, measured against a baseline, carrying no decision until a threshold cuts it.
- Drift: a change in the relationship the model encodes, making a once-correct model incorrect later.
- Dynamic batching: independent requests collected into one forward pass, at the cost of a bounded queueing delay.
- Embedding: the fixed-length vector standing in for a segment when segments are compared or indexed.
- Fault detection and classification (FDC): judging from a run's sensor trace whether the equipment behaved as intended.
- Foundation model: a model pre-trained on many series, serving a series it was never fitted to.
- Governed job: a job outside any request, and the only thing allowed to change what the server serves.
- Horizon: the number of steps ahead an answer covers, written `H`.
- Keyed state: state a stream processor holds per key and restores from a checkpoint.
- Lot: the group of wafers moving through the process together.
- Metrology: the measurement step reporting what a process produced, performed after it rather than during it.
- Model registry: the catalog of versions, and the rule deciding which version serves which key.
- Naive baseline: the last observation repeated, or the one a season earlier, which any model has to beat.
- Point-in-time correctness: every value in a context observable at the cut-off, not merely stamped before it.
- Production line: the sequence of process steps and tools a lot moves through, and the scope one FDC deployment serves.
- Quantiles: several points of the predictive distribution per horizon step, rather than one value.
- Recipe and step: the program a tool executes for a product, and one segment of that program.
- Remaining useful life: the time or number of runs before a component crosses a limit.
- Retrieval: which past segments resemble the one in hand, answered from a segment index.
- Run: one execution of a recipe on one wafer or batch.
- Scale-to-zero: every replica of an idle endpoint removed, its cost traded for a cold start.
- Scoring: an answer compared with the actual that arrived later.
- Segment: a bounded stretch of one series, delimited by a length or by an event such as a step transition.
- Segment index: the store holding one embedding per past segment, answering retrieval without a forward pass.
- Trace: the sampled record of one sensor through one run.
- Virtual metrology: what metrology measures, estimated from data available before that measurement exists.
- Wafer: the substrate carried through the process and measured by metrology.
- Watermark: the event-time bound after which no earlier event is expected.
- Zero-shot: a series answered for by a model never fitted to it.

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

- Operations (1) to (7): the request path, and the core capabilities of section 3.
- (2) context assembly, (3) the refusal that goes with it.
- (5) and (6) what makes an answer scorable, (7) what delayed evaluation later reads.
- A recursive model's state advancing at the model call itself, not at one of these operations. Section 3.2 says what the server owes that state.
- Which of the stores a question adds: the last column of Table 1.
- Operations (8) to (11) never inside a request, which is what the No against a model fitted on demand means.
- Nothing about time crossing into the model. Every decision about timestamps, gaps, and the cut-off already made before the forward pass, and beyond its repair.
- The arrow leaving the model, not the answer. Shaping, stamping, and the write stand between the numbers and something anyone can score.

## Appendix C. Case: What A Server Makes Possible On A Semiconductor Production Line

FDC (Fault detection and classification) is the case where most of the questions of Table 1 are asked of one stream at once.

Table 13. Functions an inference server puts on a semiconductor production line

| Class | Function | Description |
|-------|----------|-------------|
| Detection | Step deviation score | Every step of a finished run measured against what the model expects for that recipe and step, the score published before the next run ends. |
| Detection | Hold decision | The score cut by a rule outside the model, so a critical layer runs a tight limit and a tolerant one a loose limit, with no second model. |
| Diagnosis | Channel attribution | The ranked per-channel contribution carried on the alarm, so the first place to look is named. |
| Diagnosis | Nearest run search | The step in hand compared with the stored steps of the same recipe, the closest runs returned with their distances and what was decided about them. |
| Diagnosis | Fault classification | A flag arriving as a named class with a probability, wherever past runs carry a verdict. |
| Estimation | Virtual metrology | An estimate with an interval on every run, of what metrology will later measure on a sample. |
| Estimation | Remaining useful life | A part or consumable projected to the limit it will cross, in runs or hours, from the event history beside the traces. |
| Planning | Setpoint evaluation | A proposed setpoint answered as a conditional forecast before it is applied, the proposal stored with the answer. |
| Planning | Sampling nomination | Keys ranked by the uncertainty of the answers already given, a ranking the metrology plan can follow. |
| Coverage | Cold-start routing | A chamber or recipe without history routed to a global or pre-trained model, covered from its first run. |
| Coverage | Fleet comparison | One key structure served for every tool, so a chamber is compared with its peers rather than only with its own past. |

Three effects come from the set rather than from any row.

- One assembly shared by the functions, so adding the fault class or the measurement estimate later adds a consumer, not a second pipeline.
- The answers as a record, each stored with its context and version, so last quarter's accuracy can be computed now.
- The loop closed, since the verdict and the later measurement return by the same path and fit the model that serves next month.

One condition stands behind all of it, which is that the run is segmented into recipe steps on the way in.
