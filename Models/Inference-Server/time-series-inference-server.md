# Time Series Inference Server
Rev. 4 | Created: 2026-08-28 | Updated: 2026-08-28 08:25 CDT

A model that has been fitted on a time series is not finished until something answers questions about the series while it keeps arriving. That something is an inference server, and serving a series is not the same job as serving a table row. The request rarely carries everything the model needs, the answer is a horizon rather than a number, and the truth that would score the answer does not exist yet. This document fixes what such a server has to do, what a caller may ask of it, and which products already do that work.

## 1. Scope

The document covers the serving side only, from the moment a trained model or a pre-trained checkpoint exists to the moment an answer is consumed. Training procedures, model families, and feature construction are outside it.

Everything that follows rests on the difference between a model and a server. A model is a fitted function. It takes an array of the shape it was trained on, returns numbers, and knows nothing about which key was asked, what time it is, or what it answered an hour ago. A server is what surrounds that function to make the numbers usable. It turns a key and a cut-off into the array the model expects, and it decides what to do when what it assembles is not good enough to answer from. It also keeps everything the model does not keep, which is the history, the per-series state, the answers already given, and whatever index or labels the question needs, and it attaches to each answer the identifiers that let the answer be scored later. A question is therefore servable when the model supplies the numbers and the server can supply that remainder, and the last column of Table 1 names the remainder for each question. [Appendix B](#appendix-b-the-model-and-the-server) draws the same division with the operations on each side named.

Table 1. Questions a served series can be asked

| Question | Servable | Deliverable | What the server must add |
|----------|----------|-------------|--------------------------|
| Forecast | Yes | `H` rows of timestamp and value at each requested quantile, per key, carrying the cut-off and the model version. | Nothing. This is the case every platform of section 6 is built for. |
| Anomaly detection | Yes | One row per point or segment, holding the timestamp, the score, the threshold applied, and the resulting flag. | The baseline the score is measured against, and the threshold, which is a rule the server owns rather than a part of the model. |
| Retrieval | Yes | `k` rows of segment identifier, distance, key, and time range, ordered by distance. | A segment index and the job that rebuilds it, since no forward pass answers this. |
| Classification | Yes | One row per segment, holding the class and a probability for every class. | Labeled segments in the feedback store, because the label never comes from the series itself. |
| Change point detection | Yes | A list of timestamps, each with a score for how sharply the behavior changed there. | Nothing, but the request carries a whole segment rather than a fixed-length window. |
| Imputation | Yes | The filled values for the gap, each marked as imputed rather than observed. | The sibling channels of the same key, so the request is multivariate even when the answer is not. |
| Virtual metrology | Yes | One estimate with an interval per run, available before the measurement exists. | The join between the record served now and the measurement that arrives later, which is the delayed path of section 3.3. |
| Remaining useful life | Yes | The remaining runs or hours with an interval, per component. | An event history of failures and replacements, which is far scarcer than the series itself. |
| What-if | Yes | The forecast deliverable, plus the covariate path it is conditional on, echoed back so that the two cannot be separated. | Nothing, but the answer exists only for that proposal, so it cannot be precomputed. |
| Attribution | Yes | A contribution per channel, lag, or step, summing to the deviation being explained. | The explanation computed while the answer is produced and stored with it, because it cannot be recovered afterward. |
| Cause | No | An attribution is what comes back instead, and it names what moved together with the answer rather than what made it move. | Nothing supplies it. A cause is established by an intervention or a designed experiment, neither of which is a request. |
| Fitting on demand | No | A bounded per-call adaptation is what comes back instead, as section 5.1 describes. | Nothing supplies it inside a request. A new fit changes what every other caller receives, so it belongs to a governed job. |

The ten servable rows differ by what that last column asks for, and the difference decides the shape of the deployment rather than the choice of algorithm. Forecast, change point detection, and what-if ask for nothing, so a model and a context are the whole deployment. Retrieval asks for a segment index, which adds a job that maintains it as history accumulates. Classification, virtual metrology, and remaining useful life ask for labels, which cannot be manufactured and which make the feedback path of section 5.2 a precondition rather than an improvement. Anomaly detection asks for a baseline and for the threshold rule that cuts the score. Imputation and attribution ask for neither, but each widens what one request carries or what one answer stores. The two rows answered No mark the boundary of the whole document, because a server that is asked to establish a cause or to fit a model has stopped serving and started training with nobody governing it.

The capabilities of section 3 are written for all ten servable questions, though several are stated in forecasting terms because that is where each one bites hardest. The exposition also assumes many series rather than one. A single series can be served by a scheduled job that writes a table, and it needs none of what follows. The cost of these capabilities is paid when the count of series reaches the thousands and the answers are read by something that acts on them.

## 2. Divergence From A Stateless Model Server

A general model server treats a request as self-contained. It receives a feature vector, runs a forward pass, and returns a scalar or a class probability. Each of those assumptions breaks for a series, and the breakage is what every time-series-specific capability exists to repair.

Table 2. Serving a row against serving a series

| Aspect | Stateless model server | Time series inference server |
|--------|------------------------|------------------------------|
| Request payload | The payload holds every feature the model consumes. | The payload holds a key and a timestamp, and the recent history has to be found or carried alongside it. |
| Output shape | The output is one value per request. | The output is a matrix of horizon steps by quantiles, each step tied to a future timestamp. |
| State | The server is stateless, so any replica can answer any request. | Recursive models carry per-series state, so the replica that answers has to hold or restore that state. |
| Correctness | Correctness is a property of the code path. | Correctness also depends on the cut-off, because a context assembled with a value the model could not have seen is silently wrong. |
| Evaluation | The label arrives with the event. | The label arrives `H` steps later, so the forecast has to be stored to be scored at all. |
| Cold start | An unseen row is ordinary. | A new series has no history, so it needs either a global model or a fallback. |

## 3. Core Capabilities

Table 3 lists what the server owes its caller. The three capabilities that have no counterpart in ordinary model serving are then explained in their own subsections.

Table 3. Core capabilities and the requirement each one carries

| Capability | Requirement |
|------------|-------------|
| Context assembly | The server resolves the last `L` observations of the requested key as of the request cut-off, from the payload or from a store, and refuses the request when the window is too sparse to be used. |
| Horizon and uncertainty | The server returns every step of the horizon with quantiles or an interval, not a point alone, because a forecast without a spread cannot be acted on. |
| Covariate handling | The server accepts past covariates, future known covariates such as a calendar or a planned setpoint, and static attributes, and it rejects a future covariate that is not actually known in advance. |
| Timestamp discipline | The server fixes one frequency, one timezone, and one convention for gaps and for duplicate timestamps, and it applies them identically at training time and at serving time. |
| Series fan-out | The server answers for many keys in one call, groups them into batches that fill the accelerator, and resolves which model version serves which key. |
| State and recovery | The server keeps the per-series state that recursive models need, checkpoints it, and restores it after a restart without replaying the whole history. |
| Freshness and drift | The server exposes the age of the newest observation behind each answer, and it emits the signal that triggers retraining when residuals move. |
| Delayed evaluation | The server persists every answer with the context and the model version that produced it, so that accuracy can be computed when the actuals land. |
| Throughput and latency | The server meets its latency budget through dynamic batching, a compiled or quantized runtime, and autoscaling, and it degrades to a cheaper model rather than to a timeout. |
| Reproducibility | The server can replay any past request and return the same answer, which requires the model version, the context, and the code path to be pinned together. |

### 3.1 Context Assembly

The defining work of the server happens before the forward pass. A forecast request names a series and a time, and the server has to turn that into the window the model was trained to consume. There are two ways to do it, and each one buys what the other gives up. Carrying the window in the request payload keeps the server stateless and makes the answer trivially reproducible, at the cost of a large payload and of pushing the assembly work onto every caller. Fetching the window from an online store keeps the payload small and centralizes the assembly rule, at the cost of a read on the critical path and of a dependency that can serve stale or partially written data.

Whichever path is taken, the assembly has to be point-in-time correct. The window must contain only what was observable at the cut-off, which is not the same as what carries a timestamp earlier than the cut-off, because a late-arriving measurement can be written into a past slot after a forecast for that slot has already been produced. A store that overwrites in place therefore makes the training set and the serving path disagree, and the disagreement favors the training set, so the model looks better offline than it can ever be online.

The assembly also decides what happens when the window is imperfect. A missing sample can be interpolated, carried forward, or left as a gap for a model that accepts one, and the choice belongs to the server rather than to the caller, because it has to match what the model saw while it was fitted. A window that is shorter than the model's context is the common case for a new key, and the honest responses are to fall back to a model that tolerates a short context or to decline, never to pad the window with zeros.

### 3.2 State And Recovery

Models split into two groups by what they do with the past. A window model such as a gradient boosted regressor over lag features, or a pre-trained transformer, is a pure function of the assembled context, so replicas are interchangeable and scaling is ordinary. A recursive model such as a Kalman filter, an exponential smoother, or an online learner carries a state that summarizes everything it has seen, and that state is updated by each observation in order.

Serving the second group turns the server into a stateful system. Requests for one key have to reach the replica that holds that key's state, or the state has to live in an external store that is read and written on every update. The state has to be checkpointed, because rebuilding it means replaying the series from its beginning. Out-of-order arrivals have to be either rejected or handled by a state that can be rolled back to a watermark, since applying yesterday's sample after today's corrupts the state permanently rather than for one request.

This is why streaming engines appear in a serving discussion at all. Keyed state, checkpointing, watermarks, and event-time ordering are exactly the machinery that a recursive model needs [1](#ref-1), and a request-response server has none of it.

### 3.3 Delayed Evaluation

A classifier can be scored as soon as the outcome is known, which is often within the same session. A forecast for `t+H` cannot be scored until `t+H` has passed, so the server that produced it has usually forgotten it. Nothing else in the monitoring stack will recover it either, because the input alone does not determine the answer once models are versioned and contexts are assembled dynamically.

The consequence is that persisting predictions is a serving requirement rather than an analytics convenience. Each answer is written with its cut-off, its horizon step, its quantiles, the model version, and a hash or a copy of the context. A scheduled job then joins those rows against the actuals as they arrive and produces the error series per key and per horizon step. That error series is the only honest input to the two decisions the deployment has to make continuously, which are whether a model should be retrained and which of several candidate models should be the one that serves.

## 4. Deployment Patterns

Fig 1 shows the path a value takes from the equipment or the application that produced it to the consumer of an answer. Not every deployment contains every box, and the pattern is chosen by which boxes can be dropped.

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
| Scheduled batch | A job forecasts every key on a fixed cadence and writes the horizon into a table that consumers read, which is the cheapest pattern and the right one whenever the consumer reads far less often than the model could run. |
| Online request-response | An endpoint answers one key at a time inside a latency budget, which is required when the horizon depends on a parameter supplied by the caller, such as a proposed setpoint or a what-if quantity. |
| Streaming push | The engine holds keyed state, updates it on each event, and emits a forecast or an anomaly flag without being asked, which is the only pattern that keeps recursive models correct at high event rates. |
| Edge or on-premises | A compiled model runs next to the source, on the tool controller or a nearby host, which is chosen when the round trip to a data center exceeds the control loop or when the raw trace may not leave the site. |

## 5. The Caller's Interface

A server that only hands back answers is easier to build and worth less than it looks. The caller holds two things the deployment cannot obtain on its own, which are the intent behind the request and the truth about the answer, and the interface exists to collect both. Fig 2 shows the two directions. An option is a read that is answered inside one round trip, while a piece of feedback is a write whose effect, if it has one at all, shows up only in a later version.

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

These are set per request, and all but one of them change the answer without touching the model. The exception is the adaptation depth, which fits the model a little further for that one call, and section 5.3 says what that costs. A server that hides these options forces every caller onto one operating point, and the callers of a forecast rarely agree on one.

Table 5. Options a caller may set on a request

| Option | Description |
|--------|-------------|
| Horizon | The caller asks for the number of steps it can act on, and the server clamps the request to the longest horizon the model was fitted for rather than extrapolating past it. |
| Quantile levels | The caller names the quantiles it needs, which costs nothing for a model that emits quantiles directly and costs sampling passes for one that does not. |
| Context length | The caller may shorten the window to make the answer follow a recent regime, and the server ignores or truncates anything beyond the length the model was trained on. |
| Covariate path | The caller supplies future values of the known covariates, which is how a what-if is asked, and the server echoes that path back with the answer so the two cannot be separated later. |
| Model version | The caller pins a version to reproduce an earlier answer or to take part in a comparison, and omits it to get whatever the registry currently routes to that key. |
| Adaptation depth | Some hosted models accept fine-tuning arguments on the call itself, as TimeGPT does with `finetune_steps`, `finetune_depth`, and `finetune_loss` [2](#ref-2), which trades latency and cost for a closer fit to the caller's own series. |
| Fallback policy | The caller states what it prefers when the context is short or stale, which is to decline, to fall back to a global model, or to answer with a flag that says the answer is degraded. |
| Level of detail | The caller asks for a point, for quantiles, for decomposed components, or for an attribution, and always for the identifiers that let the answer be scored later. |

### 5.2 The Feedback Path

These are submitted after the fact and are the only way the deployment learns whether it was right. Classification, virtual metrology, and remaining useful life cannot be served at all without them, because the labels those questions need arrive through this path or not at all.

Table 6. Feedback a caller may send back

| Feedback | Description |
|----------|-------------|
| Actuals | The caller submits the observed values for a cut-off that was already answered, which is the input to scoring, and the submission is keyed by series and timestamp so that a repeat cannot be counted twice. |
| Verdict on a flag | A person marks a flag as a true or a false alarm, which is often the only label the deployment will ever receive, since nothing else says what the anomaly score should have been. |
| Correction | The caller records the number it substituted for the served answer, kept as a series of its own rather than mixed into the observations, so that the model is never trained on its own adjusted output. |
| Event marks | The caller reports the maintenance, the cleaning, the recipe change, or the product switch that made the series jump, which the drift detector reads as a deliberate discontinuity rather than as drift. |
| Measurement request | The caller asks which keys to measure next and the server answers from the uncertainty of its own predictions, which is active learning applied to a sampling plan and is what stops the plan from being independent of what the model does not know. |
| Retrain or promote | The caller asks that a model be refitted or that a candidate be promoted, and the request is queued for a governed job rather than executed by the endpoint. |

### 5.3 Rules For The Write Path

Feedback is appended, never applied in place. An endpoint that lets a caller change the serving model directly gives every caller the power to change every other caller's answers, and it ends reproducibility, because the answer to an identical request now depends on when it was asked and on who called before. The shape that survives is a feedback store keyed by series, timestamp, and source, plus a governed job that decides what any of it changes. Every submission carries who sent it and which answer it refers to, since a correction is a label the moment it is used for fitting, and an unattributable label cannot be withdrawn when it turns out to be wrong.

Two failure modes follow the path rather than the model. The first is the closed loop, in which corrections are fed back as if they were observations, so that the model learns to reproduce the adjustment a person made to its own output and the two drift together with nothing outside to check them. Keeping the corrected series separate from the observed series is what prevents it. The second is the cost that hides in a per-call adaptation argument, which is a training cost billed as a serving call, and which lets one caller turn an endpoint into a training cluster unless the server holds a budget, a cache of what was already fitted, and a quota per caller.

## 6. Key Solutions and Platforms

### 6.1 General Purpose Model Servers

These serve any model behind an HTTP or gRPC endpoint and know nothing about time. They supply batching, versioning, and autoscaling, and leave context assembly to the caller or to a wrapper written around them.

Table 7. General purpose model servers

| Platform | Description |
|----------|-------------|
| NVIDIA Triton Inference Server | It serves several framework runtimes in one process with dynamic batching and model ensembles, and its sequence batcher routes every request carrying the same correlation identifier to the same model instance [3](#ref-3), which is the one feature in this class that directly supports a recursive model. |
| KServe | It is the Cloud Native Computing Foundation (CNCF) standard for Kubernetes serving, defining an InferenceService resource with Knative autoscaling and scale-to-zero [4](#ref-4), and it can front another runtime rather than replacing it. |
| BentoML | It packages Python inference code and its dependencies into a container with adaptive batching, which suits classical forecasting code that is an ordinary Python object rather than a tensor graph. |
| Ray Serve | It composes several models into one deployment graph and multiplexes many models across a shared pool of replicas, which matches the per-key model resolution that a large series population produces. |
| MLflow model serving | It wraps a model as a `pyfunc` and serves it, which is the usual path for statistical models whose inference is a library call rather than a forward pass. |
| Seldon Core | It deploys models on Kubernetes through a runtime that also exposes an inference protocol compatible with the KServe one. |
| TensorFlow Serving | It remains a stable choice for saved TensorFlow graphs and offers little for anything else. |
| TorchServe | It is under limited maintenance with no planned updates, bug fixes, or security patches [5](#ref-5), so it should not be chosen for a new deployment. |

None of them supplies the feedback path of section 5.2. What they offer toward it is payload logging, which captures requests and answers so that a store can be built behind them, and that capture is worth turning on from the first day because it cannot be reconstructed later.

### 6.2 Time Series Foundation Models

A foundation model forecasts an unseen series without ever being fitted to it, which changes what the server has to hold, because one checkpoint replaces a population of per-key models. The practical differences between the families are the shape of the input they accept, whether quantiles come out of one pass or out of repeated sampling, and how large a forward pass is.

Table 8. Pre-trained time series models and how they are served

| Model | Description |
|-------|-------------|
| TimeGPT (Nixtla) | It is offered as a hosted REST endpoint with an API key, and also as a self-hosted container or Python wheel for sites that cannot send data out, and it covers anomaly detection in addition to forecasting [6](#ref-6). |
| Chronos-2 (Amazon) | It is an open-weight encoder-only model of about 120M parameters that handles univariate, multivariate, and covariate-informed inputs in one architecture and emits multi-step quantile forecasts in a single pass [7](#ref-7), which removes the sampling loop that made earlier Chronos versions expensive to serve. |
| TimesFM (Google) | It is an open-weight decoder-only model whose 2.5 release carries 200M parameters and a context of up to 16k points, with quantiles from an optional head [8](#ref-8), and it is the family that has been embedded into a warehouse rather than only into a server. |
| Moirai (Salesforce) | Its 2.0 release is an open-weight decoder-only model that emits quantiles and predicts several tokens at a time, replacing the masked encoder of the 1.x line [9](#ref-9), and it is served through the `uni2ts` library rather than through an endpoint of its own. |
| Granite TTM (IBM) | It is a deliberately tiny family whose smallest members start near a million parameters and which avoids self-attention altogether [10](#ref-10), so it can be served on CPU and embedded inside a stream processor instead of behind an accelerator. |
| Toto (Datadog) | It is trained for observability metrics and released as a family of sizes with a quantile head and attention that alternates between time and variates [11](#ref-11), which targets the high-cardinality monitoring case rather than the business forecasting one. |

The same checkpoints also answer the retrieval question, because the encoder that produces a forecast produces an embedding of the segment on the way, and that embedding is what a segment index stores. Serving retrieval from an open-weight encoder therefore costs one forward pass per new segment at write time and no forward pass at all at read time.

Published leaderboards move between releases, so a benchmark rank is a reason to shortlist a model rather than to adopt one. The evaluation that decides is the one run on the target series, against the naive baseline that the horizon and the frequency imply.

### 6.3 Forecasting Frameworks

The frameworks below produce the models that the servers in Table 7 carry, and several of them also expose a batch inference entry point that is enough on its own for the scheduled pattern.

Table 9. Frameworks that supply models and batch inference

| Framework | Description |
|-----------|-------------|
| AutoGluon-TimeSeries | It fits and ensembles statistical models, gradient boosted trees, deep models, and pre-trained checkpoints under one interface, and returns quantiles by default [12](#ref-12). |
| Nixtla `statsforecast`, `mlforecast`, `neuralforecast` | They cover the classical, the feature-based, and the neural routes with a common data contract, and they are built to fit and predict a large series population in one call. |
| GluonTS | It provides probabilistic model implementations and the evaluation harness that goes with them. |
| Darts | It presents statistical and deep models behind one API with backtesting utilities. |
| sktime | It supplies a scikit-learn compatible interface for forecasting and classification pipelines over series. |
| STUMPY, tslearn | They compute the matrix profile and the elastic distances that answer retrieval and change point questions by comparing shapes directly, without an embedding and without a fitted model. |

### 6.4 Streaming Engines And Stores

This layer answers the state and the context requirements of section 3, and in the push pattern it is the serving layer rather than a dependency of one.

Table 10. Streaming and storage components

| Component | Description |
|-----------|-------------|
| Apache Kafka | It is the transport that decouples producers from the server and lets the same events feed the online path and the offline store, and it is also where predictions are published for downstream consumers. |
| Apache Flink | It holds keyed state with checkpointing and event-time watermarks, and it can either call a remote model or run a small one inline. Confluent has moved built-in forecasting and anomaly detection into this layer using time series models including Granite and TimesFM [13](#ref-13). |
| Spark Structured Streaming | It applies the same batch code to a stream in micro-batches, which suits minute-scale cadences rather than millisecond ones. |
| Online store (Redis, DynamoDB, and equivalents) | It serves the last `L` points and the static attributes of a key within a millisecond budget, which is the read that section 3.1 puts on the critical path. |
| Feature store (Feast, Tecton) | It defines a feature once and materializes it into both an offline table and an online store, which is the standard mechanism for keeping the training window and the serving window identical. |
| Time series database (InfluxDB, TimescaleDB, ClickHouse, Prometheus) | It stores the history that backfills and backtests read, and its downsampling and retention rules decide what a long context can still contain. |
| Segment index (FAISS, pgvector, Milvus) | It holds one embedding per past segment together with the attributes a comparison has to filter on, and it answers the retrieval question in place of a model, which makes its rebuild schedule part of the serving design. |

### 6.5 Managed And Warehouse-Native Services

Table 11. Managed services and in-database forecasting

| Service | Description |
|---------|-------------|
| Amazon SageMaker AI | It offers real-time, serverless, asynchronous, and batch transform endpoints for a model that the user brings, and its AutoML path covers time series directly. Amazon Forecast, the earlier dedicated service, has been closed to new customers since 2024-07-29 [14](#ref-14), so new work starts on SageMaker instead. |
| Google BigQuery ML | Its `AI.FORECAST` function forecasts directly in SQL against a built-in TimesFM model with `HORIZON` and `CONTEXT_WINDOW` as arguments [15](#ref-15), which removes the server entirely for the scheduled pattern when the data already lives in the warehouse. |
| Vertex AI, Azure Machine Learning, Databricks | They provide managed endpoints, model registries, and monitoring around a model the user trains, and the time-series-specific work in section 3 still has to be built on top. |

### 6.6 Edge Runtimes

When the answer has to be produced next to the source, the model is exported to a portable format and executed by a small runtime rather than by a server. ONNX Runtime, OpenVINO, and TensorFlow Lite fill that role, and the export step is what limits model choice, since a model whose inference is a Python control flow rather than a graph does not survive it. The operational cost moves from scaling to distribution, because every host now carries a model version that has to be rolled forward and rolled back.

## 7. Selection

Table 12. Constraint and the choice it forces

| Constraint | Choice |
|------------|--------|
| Consumers read forecasts far less often than the data updates. | Run the scheduled batch pattern and write a table, and add an endpoint only when a caller needs an answer that depends on its own input. |
| The model is recursive and events arrive continuously. | Put the model in a streaming engine with keyed state rather than behind a request-response endpoint. |
| The series population is large and each key has little history. | Serve one pre-trained or global model for all keys, since per-key models cannot be fitted or maintained at that count. |
| The question is which past segment this one resembles. | Build and schedule a segment index rather than a model, because no forecasting endpoint answers it and the index, not the forward pass, is what has to be maintained. |
| The answer has to be judged by a person before it is worth anything. | Build the feedback path first, since the verdict is the only label that will ever arrive and it is lost if nothing collects it. |
| Raw data may not leave the site. | Take the self-hosted or open-weight route, which rules out a hosted forecasting API regardless of its accuracy. |
| The loop that consumes the answer is faster than a network round trip. | Export the model and run it at the edge, and accept the narrower model choice that the export imposes. |
| The data is already in a warehouse and the cadence is daily. | Use the in-database forecasting function before building anything. |

## 8. Operational Pitfalls

The failures below are the ones that recur, and each is a capability from section 3 that was left to the caller.

- A covariate that is not known in advance is fed as a future covariate, so the backtest is excellent and the live forecast is not. Only a value that is decided rather than observed, such as a calendar field or a planned setpoint, belongs in that slot.
- The serving window is built by a different code path than the training window, and the two disagree about resampling, gaps, or timezone. The disagreement is invisible in both test suites, because each path is self-consistent.
- A late-arriving sample overwrites a slot after a forecast for that slot was issued, and the stored prediction is then scored against a history that no longer matches the one it was produced from.
- A new key with a short history is padded to the model's context length, and the model reads the padding as a real regime.
- The unit or the scaling of a channel changes upstream, and nothing rejects it because the value stays inside the range the schema allows.
- Retraining is triggered on a residual window shorter than the horizon, so the trigger fires on forecasts that have not yet been fully scored.

## References

<a id="ref-1"></a>
[1] Apache Flink. [Working with state](https://nightlies.apache.org/flink/flink-docs-stable/docs/dev/datastream/fault-tolerance/state/). The Apache Software Foundation.

<a id="ref-2"></a>
[2] Nixtla. [Fine-tuning](https://nixtlaverse.nixtla.io/nixtla/docs/capabilities/forecast/finetuning.html). Nixtla.

<a id="ref-3"></a>
[3] NVIDIA. [Batchers](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/batcher.html). NVIDIA Corporation.

<a id="ref-4"></a>
[4] KServe authors. [KServe documentation](https://kserve.github.io/website/). Cloud Native Computing Foundation.

<a id="ref-5"></a>
[5] PyTorch. [Notice: limited maintenance](https://docs.pytorch.org/serve/). The Linux Foundation.

<a id="ref-6"></a>
[6] Nixtla. [TimeGPT](https://www.nixtla.io/). Nixtla.

<a id="ref-7"></a>
[7] Amazon Science. [Introducing Chronos-2: from univariate to universal forecasting](https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting). Amazon.

<a id="ref-8"></a>
[8] Google Research. [TimesFM](https://github.com/google-research/timesfm). Google.

<a id="ref-9"></a>
[9] Salesforce AI Research. [uni2ts: unified training of universal time series forecasting transformers](https://github.com/SalesforceAIResearch/uni2ts). Salesforce.

<a id="ref-10"></a>
[10] IBM Granite. [granite-timeseries-ttm-r3](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3). IBM.

<a id="ref-11"></a>
[11] Datadog. [Toto: time-series-optimized transformer for observability](https://github.com/DataDog/toto). Datadog.

<a id="ref-12"></a>
[12] AutoGluon. [AutoGluon-TimeSeries documentation](https://auto.gluon.ai/stable/tutorials/timeseries/index.html). The AutoGluon community.

<a id="ref-13"></a>
[13] Confluent. [Evolving the data streaming platform for AI, scale, and control](https://www.confluent.io/blog/2026-q3-confluent-intelligence-ai-update/). Confluent.

<a id="ref-14"></a>
[14] Amazon Web Services. [Transition your Amazon Forecast usage to Amazon SageMaker Canvas](https://aws.amazon.com/blogs/machine-learning/transition-your-amazon-forecast-usage-to-amazon-sagemaker-canvas/). Amazon.

<a id="ref-15"></a>
[15] Google Cloud. [The AI.FORECAST function](https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-forecast). Google.

---

## Appendix A. Terminology

- Active learning: the practice of choosing which samples to label next by what the model is least certain about, rather than by a fixed plan.
- Attribution: the account of which channel, step, or lag moved a particular answer, computed while that answer is produced.
- Backfill: the recomputation of forecasts for past cut-offs through the serving path, used to produce a history that a newly deployed model would have generated.
- Chamber: the enclosure of a process tool in which one wafer or one batch is processed, and the grain at which most FDC series are keyed, since two chambers of one tool do not behave identically.
- Change point: a timestamp at which the behavior of a series changes, separating one regime from the next.
- Concept drift: a change in the relationship the model encodes, which makes a model that was correct at fit time incorrect later even though the input schema is unchanged.
- Context: the window of past observations a model consumes to produce an answer, whose length is fixed by the model.
- Covariate: a variable other than the target that the model reads, past when it is only observed and future known when its value at a future timestamp is decided in advance.
- Cut-off: the timestamp that separates what the model is allowed to see from what it is asked to predict.
- Dynamic batching: the server-side collection of independently arriving requests into one forward pass, which raises accelerator utilization at the cost of a bounded queueing delay.
- Embedding: the fixed-length vector an encoder produces for a segment, which stands in for the segment when segments are compared or indexed.
- Fab: the factory in which wafers are processed, and the boundary that decides what data may leave and what has to be answered on site.
- Fault detection and classification (FDC): the practice of judging, from the sensor trace of a process run, whether the equipment behaved as intended, so that a suspect run is held before more material is committed to it.
- Feedback store: the append-only store that holds what callers send back, which is the actuals, the verdicts, the corrections, and the event marks, and which is where labels come from.
- Foundation model: a model pre-trained on many series that forecasts a series it was never fitted to, so that one checkpoint serves a whole population of keys.
- Governed job: a job that runs outside any request and is the only thing allowed to change what the server serves, whether by refitting a model or by promoting a candidate.
- Horizon: the number of steps ahead a forecast covers, written `H`.
- Idempotency: the property that submitting the same feedback twice leaves the store in the state one submission would have left it in.
- Keyed state: state that a stream processor holds separately for each key and restores from a checkpoint after a failure.
- Lot: the group of wafers that moves through the process together and that metrology reports against.
- Metrology: the measurement step that reports what a process actually produced, performed after the process rather than during it.
- Model registry: the catalog that holds model versions and the rule that decides which version serves which key.
- Naive baseline: the reference forecast that repeats the last observation, or the observation one season earlier, against which any model has to be shown to be better.
- Parameter row: the fixed-width record of summary values reduced from a trace, which is what a server consumes when the trace itself is too large or too restricted to ship.
- Point-in-time correctness: the property that every value in an assembled context was observable at the cut-off, not merely stamped before it.
- Preventive maintenance: the scheduled servicing of a tool, which resets the condition the model had learned and so marks a deliberate discontinuity in every series that tool produces.
- Quantile forecast: an answer that reports several quantiles of the predictive distribution per horizon step rather than a single value.
- Recipe and step: the ordered program a tool executes for a product, and one segment of that program, which together bound the window an FDC model reads.
- Remaining useful life: the time or the number of runs a component is expected to last before it crosses a limit.
- Retrieval: the question of which past segments most resemble the one in hand, answered from a segment index rather than from a forward pass.
- Run: one execution of a recipe on one wafer or one batch, which is the unit an FDC answer is produced for.
- Scale-to-zero: the removal of every replica of an idle endpoint, which eliminates its cost and adds a cold start to the next request.
- Segment: a bounded stretch of one series, delimited either by a fixed length or by an event such as a step transition.
- Segment index: the store that holds one embedding per past segment, together with the attributes a comparison filters on, and that answers a retrieval question without a forward pass.
- Trace: the sampled record of one sensor through one run, which is the raw form the parameter row is reduced from.
- Virtual metrology: the estimation of a quantity that metrology measures, from data available before that measurement exists.
- Wafer: the substrate that carries the product through the process and that metrology measures.
- Watermark: the event-time bound a stream processor uses to decide that no earlier event will arrive, after which a window can be closed.
- Zero-shot forecasting: producing a forecast for a series with a pre-trained model that was never fitted to that series.

## Appendix B. The Model And The Server

Section 1 draws the line between the two in prose. Fig 3 draws it as a boundary and names the operations that sit on each side of it.

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

The figure separates three things the document treats apart. Operations (1) to (7) are the request path, and they are what section 3 lists as the core capabilities: (2) is context assembly, (3) is the refusal that goes with it, (5) and (6) are what make an answer scorable at all, and (7) is what delayed evaluation later reads. The stores are what the server holds so that the model does not have to, and the last column of Table 1 says which of them a given question adds. Operations (8) to (11) never run inside a request, which is what the No against `Fitting on demand` in Table 1 means.

Two boundaries are worth reading off the figure. The first is that nothing about time crosses into the model. The model receives an array, so every decision about which timestamps belong in it, what to do about a gap, and where the cut-off was has already been made before the forward pass starts and cannot be repaired by it. The second is that the arrow leaving the model is not the answer. Between the numbers and the answer sit the shaping, the stamping, and the write, and an implementation that skips those three returns something a caller can plot but nobody can score.

## Appendix C. Case: Fault Detection And Classification

Fault detection and classification, abbreviated FDC, is the semiconductor line's own instance of the streaming push pattern, and every capability of section 3 binds tighter there than in a business forecasting deployment. Sensors on a process tool are sampled through a run, a model decides whether the run behaved as the recipe says it should, and the answer is useful only if it arrives before the next lot is processed. The case is worked through here because its constraints change which platform from section 6 can be used at all, not merely how it is configured.

Table 13. What the FDC case forces on each capability

| Capability | What the case forces |
|------------|----------------------|
| Context assembly | The window is a recipe step rather than a count of samples, so the serving path has to segment the run before it can slice a context, and a step that ran longer or shorter than nominal still has to yield a comparable window. |
| Series fan-out | The key is the tuple of tool, chamber, recipe, step, and sensor, which multiplies into tens of thousands of series in one fab, so model resolution has to be a lookup inside one deployment rather than a deployment per key. |
| Cold start | A new recipe, a converted chamber, or a tool returning from maintenance has no history under its key, so serving falls back to a global or pre-trained model until enough runs accumulate. |
| State and drift | Preventive maintenance, a chamber clean, and a part swap are deliberate discontinuities, so a recursive model's state is reset at those events and the drift detector is told they happened; otherwise every maintenance action raises an alarm of its own. |
| Retrieval | Comparison against a reference run is the working method on the line, so the segment index that holds past segments per recipe and step is not an extension but part of the first deployment. |
| Feedback | The engineer's verdict on a held run is the only label the deployment receives at run grain, so an FDC deployment without the write path of section 5.3 never learns anything from its own alarms. |
| Delayed evaluation | The label from metrology returns hours or days later at lot or site grain while the answer was produced at run grain, so scoring is a grain conversion rather than a join on a timestamp. |
| Residency and volume | The raw trace is sampled too fast to ship and often may not leave the site, so what reaches the server is the parameter row reduced from the trace, and the reduction runs upstream of the server. |
| Latency | The budget is the interval between runs, because an answer that arrives after the next lot has started cannot hold it, and that budget rather than the model decides between an edge runtime and a remote endpoint. |

Three consequences deserve more than a table row. The first is that the window comes from the process and not from the clock. A server designed for a regularly sampled series expects a key and a timestamp and returns the last `L` points behind it, but an FDC context is the segment of one run between two step transitions, whose length varies with the recipe and with the run itself. Segmentation therefore belongs on the ingest path rather than inside the server, and the identity of the segment, which is the tool, the recipe, the step, and the run, becomes the request key. What the server receives is then a per-run object rather than a slice of a continuous stream, which is why the scheduled batch pattern survives for the slower loops such as chamber health, while the per-run decision goes to the push path.

The second consequence concerns evaluation, because the scored subset is not the served population. Metrology is applied under a sampling plan, so only a fraction of runs ever acquires a label, and that fraction is chosen by a rule rather than at random. The error series computed from those labels describes the measured wafers, and a retraining trigger driven by it inherits the same bias. The honest reading is to treat the accuracy of an FDC model as an estimate over the sampled population and to keep the served population's drift under a separate unlabeled statistic, such as a shift in the residual distribution. The measurement request of Table 6 is the lever that acts on this, since letting the model nominate what to measure next is what makes the sampled population depend on where the uncertainty actually is.

The third puts the threshold outside the model. Holding a lot that was fine and releasing a lot that was not are failures of very different cost, and the ratio between them changes with the product, the layer, and how full the line is. The server should therefore emit a quantile or a score and leave the cut to a rule outside the model, so that the operating point can be moved without redeploying anything and so that the same served answer can support a tight rule for one product and a loose one for another.
