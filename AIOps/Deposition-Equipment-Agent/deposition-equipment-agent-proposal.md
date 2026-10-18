# Semiconductor Deposition Equipment AI Agent Proposal
Rev. 0 | Created: 2026-10-18 | Updated: 2026-10-18 02:47 CDT

- [1. Purpose](#1-purpose)
- [2. Quick Start and Architecture](#2-quick-start-and-architecture)
- [3. Solution Items](#3-solution-items)
- [4. Equipment Agent Loop](#4-equipment-agent-loop)
- [5. Autonomy Levels](#5-autonomy-levels)
- [6. New Equipment Onboarding](#6-new-equipment-onboarding)
- [7. Operating Rules and Roadmap](#7-operating-rules-and-roadmap)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Single vs Multi-Equipment](#appendix-b-single-vs-multi-equipment)
- [Appendix C. Model Architecture](#appendix-c-model-architecture)
- [Appendix D. Orchestrator Functions](#appendix-d-orchestrator-functions)
- [Appendix E. Onboarding Gate Metrics](#appendix-e-onboarding-gate-metrics)

## 1. Purpose

- **Problem Statement**: An engineer has to read the output of a deposition thickness model and set the next wafer's conditions by hand, and every added tool repeats model development, recalibration and maintenance decisions on its own.
- **Goal**: An AI agent that calls validated ML models as functions makes the per-wafer prediction, correction and maintenance decisions, so that new-tool setup time and engineer interventions do not grow in proportion to the number of tools.
- **Non-Goal**: An LLM that predicts THK or computes process conditions itself, and the build and operation procedures of the feature store, the model registry and the fab systems.

## 2. Quick Start and Architecture

- **Structure**: Two tiers, one equipment agent per tool and an AI orchestrator that coordinates the agents.
- **Division of work**: ML models compute the VM prediction, the recipe inversion and the drift statistics. The LLM decides which function to call and when (function calling) and explains the result; it never computes a THK or recipe value itself. An LLM copilot answers engineers' questions by analysing the data.
- **Human-in-the-loop**: When an approval or a confirmation is needed, the agent pauses its run and only raises a request. The orchestrator routes that request to an engineer with the authority to answer it and records the approval.
- **Autonomy**: Raised from L1 advisory to L3 closed-loop R2R as the operating record builds up ([5. Autonomy Levels](#5-autonomy-levels)).
- **New tools**: Four gates decide whether an existing model can be reused ([6. New Equipment Onboarding](#6-new-equipment-onboarding)).
- **Multi-tool effect**: Each added tool contributes training data and excursion cases, which shortens setup and lightens management per tool ([Appendix B](#appendix-b-single-vs-multi-equipment)).

The whole system is drawn in [Fig 1](#fig-1).

```text
                   +-------------------------------------------+     +-------------------+
                   |              AI Orchestrator              |<--->| Engineer 👩‍🔧       |
                   |  event workflow / policy / permission     |     | approves actions  |
                   |  approval routing / audit record          |     | by autonomy level |
                   +---------------------+---------------------+     | answers requests  |
                                         |                           | reads reports     |
                   predict / recommend / health / calibrate          +-------------------+
                                         |
       +---------------------------------+-------------------+----------------------+
       |                                                     |                      |
+------+----------------------------------------+    +-------+--------+     +-------+--------+
| Equipment Agent #1                            |    | Equipment      |     | Equipment      |
|                                               |    | Agent #2       | ... | Agent #N       |
|  LLM       which function to call, and when   |    | same as #1     |     | same as #1     |
|            explanation of the result          |    +-------+--------+     +-------+--------+
|               |  function call ^  output      |            |                      |
|               v                |              |            |                      |
|  ML model  predict THK and its uncertainty    |            |                      |
|            invert process condition           |            |                      |
|            detect drift, retrain the adapter  |            |                      |
|  Guardrail uncertainty threshold before APC   |            |                      |
|  Engineer 👩‍🔧 approves and answers             |            |                      |
|            via orchestrator                   |            |                      |
+------+----------------------------------------+            |                      |
       |                                                     |                      |
       +------------+-----------------------------------+----+----------------------+
                    |                                   |
       Feature Store (common schema)          Model Registry (Global backbone + Local adapters)
```

<a id="fig-1"></a>
Fig 1. Two-tier agent architecture with the engineer, and the LLM and the ML model inside one equipment agent

`Equipment Agent #1` shows the inside of one agent, and agents `#2` to `#N` are built the same way. The orchestrator and the agents exchange data only through four function interfaces, predict, recommend, health and calibrate, and the orchestrator's functions are listed in [Appendix D](#appendix-d-orchestrator-functions).

- **Feature store**: Turns trace data into features of the common schema and serves the same features to training and inference.
- **Model registry**: Keeps the versions of the Global backbone and of each chamber's Local adapter; an agent loads only versions promoted after validation.

## 3. Solution Items

The solution covers nine items in five categories.

Table 1. Solution items

<table>
  <tr><th align="center">Category</th><th align="center">Item</th><th align="center">Description</th><th align="center">LLM role</th><th align="center">Section</th></tr>
  <tr><td align="center" rowspan="2">A. APC</td><td align="center">Virtual metrology</td><td align="center">Predicts the deposited thickness right after each wafer without measuring it, cutting metrology time and cost</td><td align="center">Function call (VM model)</td><td align="center">4</td></tr>
  <tr><td align="center">R2R control</td><td align="center">Adjusts the next wafer's deposition conditions when the predicted thickness leaves the target</td><td align="center">Function call (inverse model), explains the adjustment</td><td align="center">4</td></tr>
  <tr><td align="center" rowspan="2">B. EES</td><td align="center">FDC</td><td align="center">Watches tool sensor signals in real time and alarms before a defect occurs</td><td align="center">Function call (FDC monitoring)</td><td align="center">4</td></tr>
  <tr><td align="center">FDC root-cause analysis</td><td align="center">Names the suspected parts and conditions behind a fault in plain language</td><td align="center">Explains the cause</td><td align="center">4</td></tr>
  <tr><td align="center" rowspan="2">C. PHM</td><td align="center">Consumable RUL</td><td align="center">Tracks the wear of consumables such as targets and advises the best replacement time</td><td align="center">Function call (RUL model)</td><td align="center">4, Appendix D</td></tr>
  <tr><td align="center">PdM</td><td align="center">Recommends PM timing from tool condition to cut unnecessary down time</td><td align="center">Explains the PM recommendation</td><td align="center">Appendix D</td></tr>
  <tr><td align="center">D. TTTM</td><td align="center">Chamber matching</td><td align="center">Compares results across tools and chambers to hold one quality level</td><td align="center">None (the orchestrator handles it)</td><td align="center">Appendix C, D</td></tr>
  <tr><td align="center" rowspan="2">E. MLOps</td><td align="center">Model retraining</td><td align="center">Keeps accuracy after PM or part replacement by retraining</td><td align="center">Decides when to retrain, function call</td><td align="center">4</td></tr>
  <tr><td align="center">LLM copilot</td><td align="center">Answers questions such as "why did tool 2 stop yesterday" by analysing the data</td><td align="center">Answers questions</td><td align="center">2</td></tr>
</table>

The Section column names the section that covers the item. The expected benefits are lower metrology cost, higher yield, higher tool availability and a lighter engineer workload.

## 4. Equipment Agent Loop

The loop in [Fig 2](#fig-2) runs once each time a wafer finishes.

```text
  +------------------------------------------------------------------+
  | LLM       reads the output of each step, decides which step to   |
  |           call next, and writes the [7] Report                   |
  +---------------------------------+--------------------------------+
                                    |  function calls to [1] - [6]
                                    v
  Trace data in
       |
       v
  [1] Detect  ----->  [2] Predict  ----->  [3] Guardrail (rule)
      (code)              (ML)
                                                |
                          high confidence  +----+----+  low confidence
                                           |         |
                                           v         v
                          [4] Recommend (ML)       Metrology sampling
                                           |       request
                                           v         |
                          Engineer approval          |
                          by autonomy level          |
                                           |         |
                                           v         v
                                       APC (R2R)   Metrology data in
                                                     |
                                                     v
  [7] Report (LLM)  <--  [6] Adapt  <--  [5] Drift monitor
                             (ML)              (ML)
       |
       v
  +------------------------------------------------------------------+
  | Engineer  reached only through the orchestrator: answers the     |
  |           mapping questions of [1], approves [4] by autonomy     |
  |           level, and reads the [7] Report                        |
  +------------------------------------------------------------------+
```

<a id="fig-2"></a>
Fig 2. Per-wafer loop of the equipment agent and what carries each step

The tags (ML), (rule), (code) and (LLM) name what carries each step. The LLM reads each step's output and decides which function to call next, and steps [5] and [6] run only when metrology data arrives.

- **[1] Detect (FDC), code**: Converts the trace into standard features and checks it for missing values, outliers and sensor faults. An unclear sensor mapping goes to the orchestrator for confirmation.
- **[2] Predict (VM), ML**: Picks or weights the Global, Local and Ensemble models and outputs the THK and its confidence interval before metrology. The model structure is in [Appendix C](#appendix-c-model-architecture).
- **[3] Guardrail (VM reliability), rule**: Passes the result to [4] when the uncertainty is within the threshold; otherwise blocks the R2R hand-off and requests adaptive metrology sampling.
- **[4] Recommend (R2R), ML**: Derives the next wafer's recipe with the inverse model and proposes only conditions inside SOA 95%.
- **Engineer approval**: A change that the autonomy level in Table 2 reserves for approval goes to an engineer through the orchestrator before it reaches APC.
- **[5] Drift monitor (EES), ML**: Detects residual trends, SOA shifts with consumable wear, and covariate and concept shifts after PM and seasoning.
- **[6] Adapt (recalibration), ML**: Proposes a recalibration DOE (for example 5 monitor wafers, CCD with 4 axial points and a center point) and retrains only that chamber's adapter. Only candidates that pass shadow validation and time-ordered validation are put forward for promotion.
- **[7] Report (RCA), LLM**: Summarises the root-cause sensor and recipe step of a VM deviation and the recommended action in a plain-language report for the engineer (for example "increase deposition time by 0.4 s, 95% confidence").

## 5. Autonomy Levels

Autonomy rises in three levels as the operating record builds up, and each level narrows what an engineer has to approve.

Table 2. Autonomy levels

| Level                     | Automatic action                               | Human approval                      |
| :---:                     | :---:                                          | :---:                               |
| L1 Advisory, open-loop    | None, recipe recommendation only               | Every recipe change                 |
| L2 Supervised closed-loop | Recipe adjustment inside SOA 95%               | Recipe change outside SOA 95%       |
| L3 Closed-loop R2R        | Recipe adjustment inside SOA, model retraining | Model promotion, recipe outside SOA |

A level is raised only after the hit rate of the L1 recommendations has been confirmed, so the case for each step up comes from the operating record.

## 6. New Equipment Onboarding

A new tool is compared with a reference tool through the four gates of Table 3 in order, and the result at each gate sets the next action.

Table 3. Onboarding gates, results and actions

<table>
  <tr><th align="center">Gate</th><th align="center">Check</th><th align="center">Result</th><th align="center">Action</th></tr>
  <tr><td align="center" rowspan="2">Schema gate</td><td align="center" rowspan="2">Key sensor mapping, sampling period and units</td><td align="center">Key sensors present, period and units match</td><td align="center">Next gate</td></tr>
  <tr><td align="center">Key sensor missing</td><td align="center">New development</td></tr>
  <tr><td align="center" rowspan="4">Covariate shift</td><td align="center" rowspan="4">PSI / domain classifier AUC</td><td align="center">&lt; 0.1 / &lt; 0.6</td><td align="center">Next gate</td></tr>
  <tr><td align="center">Only specific control variables (pressure, temperature, power offset) deviate</td><td align="center">Tool tuning (chamber matching), then re-evaluate</td></tr>
  <tr><td align="center">0.1 ~ 0.25 / 0.6 ~ 0.8</td><td align="center">Next gate as a model correction candidate</td></tr>
  <tr><td align="center">&gt; 0.25 / &gt; 0.8</td><td align="center">New development</td></tr>
  <tr><td align="center" rowspan="3">Concept shift</td><td align="center" rowspan="3">Spearman correlation of SHAP importance ranks</td><td align="center">&gt; 0.8</td><td align="center">Next gate</td></tr>
  <tr><td align="center">0.5 ~ 0.8</td><td align="center">Next gate as a model correction candidate</td></tr>
  <tr><td align="center">&lt; 0.5</td><td align="center">New development</td></tr>
  <tr><td align="center" rowspan="4">Pilot validation</td><td align="center" rowspan="4">RMSE, R², bias on 20~30 measured wafers</td><td align="center">In spec, no bias</td><td align="center">Reuse the existing model</td></tr>
  <tr><td align="center">Offset only</td><td align="center">Bias correction</td></tr>
  <tr><td align="center">Offset and gain</td><td align="center">Adapter fine-tune</td></tr>
  <tr><td align="center">Out of spec after correction</td><td align="center">New development</td></tr>
</table>

The thresholds are initial recommendations and are tuned per fab. Tool tuning is tried before model correction, because a model that absorbs a hardware difference makes later drift hard to tell apart from a hardware change. [Appendix E](#appendix-e-onboarding-gate-metrics) lists the metrics checked alongside each gate.

## 7. Operating Rules and Roadmap

Four things are settled before operation, and the equipment interface usually takes the longest.

- **Safety**: No run outside the SOA, no R2R hand-off of a VM value above the threshold, a validation gate before model promotion, and a log of every decision.
- **Security**: An on-premise LLM, or a design that handles only features, so that customer data never leaves the fab.
- **Equipment interface**: Interfaces to SECS/GEM, FDC/EES, MES and APC (R2R).
- **KPI**: Spec hit rate, correction cost per wafer, number of engineer interventions, and performance near consumable end of life.

The roadmap durations are estimates.

- **Phase 1 (3~6 months, L1)**: An agent that wraps the existing VM model and recipe inversion as functions, Global + Local models across chambers of one tool type, and verification of chamber matching.
- **Phase 2 (6~12 months, L2)**: The [3] Guardrail, [5] Drift monitor and [6] Adapt steps, with automated recalibration DOE and retraining (MLOps).
- **Phase 3 (12 months~, L3)**: Few-shot onboarding of new and different tool types, an LLM-based orchestrator, and extension to other processes.

---

## Appendix A. Terminology

- **APC (R2R)**: Process control that adjusts the next run's recipe from measured or predicted values (advanced process control, run-to-run).
- **CCD**: An experimental design built from axial points and a center point (central composite design).
- **Chamber fingerprint**: The offset, gain and sensor behaviour particular to one chamber on the same recipe.
- **Concept shift**: Drift in which the relation between inputs and output changes.
- **Conformal prediction**: A method that builds prediction intervals with guaranteed coverage from the residual distribution of validation data.
- **Consumable**: A part such as a target that wears with use in a deposition tool and is replaced periodically.
- **Covariate shift**: Drift in which only the input distribution changes and the input-output relation stays the same.
- **Dispatching**: Deciding which tool processes a waiting lot or wafer.
- **DOE**: Design of experiments.
- **Domain classifier AUC**: The AUC of a classifier that separates reference-tool data from new-tool data; values near 0.5 mean the two are alike.
- **DTW**: A distance between two traces that allows stretching along the time axis (dynamic time warping).
- **Excursion**: A state in which a process or tool leaves its control range and may produce bad wafers.
- **Fab systems**: The existing fab systems an equipment agent receives data from and sends control values to: SECS/GEM, FDC/EES, MES and APC (R2R).
- **FDC/EES**: Fault detection and classification / equipment engineering system, which collect and watch tool traces.
- **Function calling**: An LLM picks a function, sets its inputs, asks for it to run and reads the result. This document says function rather than tool because a tool means a piece of equipment in semiconductor manufacturing.
- **Golden trace envelope**: The per-step tolerance band built from the traces of normal runs.
- **Hotelling T²**: A statistic for how far multivariate data lies from the centre of a reference distribution.
- **Human-in-the-loop**: Operation in which a person approves or decides inside an automated flow. Here, an equipment agent pauses its run and raises a request, and the orchestrator passes it to an engineer and returns the answer.
- **KS test**: The Kolmogorov-Smirnov test, which tests whether two samples share a distribution from the gap between their cumulative distributions.
- **LoRA**: A lightweight adapter that freezes a large model's weights and trains only low-rank matrices (low-rank adaptation).
- **Mahalanobis distance**: A multivariate distance that accounts for the covariance between variables.
- **MAML**: A meta-learning method that learns initial weights able to adapt quickly from little data (model-agnostic meta-learning).
- **MES**: Manufacturing execution system.
- **Mixed-effects model**: A statistical model that estimates shared effects together with tool- and chamber-level random effects.
- **MLOps**: The practice of automating model training, validation, deployment and monitoring.
- **Monitor wafer**: A wafer run to check or recalibrate the process rather than to make product.
- **PCA**: Principal component analysis, which reduces correlated variables to a few orthogonal components.
- **PdM**: Predictive maintenance, which predicts failure from tool condition data and services the tool before it.
- **PHM**: Prognostics and health management, covering anomaly detection, remaining-life prediction and maintenance decisions together.
- **PM**: Preventive maintenance.
- **PSI**: Population stability index, which measures a distribution shift from bin-by-bin proportion differences.
- **RCA**: Root-cause analysis, which finds the sensor, part or recipe step behind an anomaly or deviation.
- **RMSE**: Root mean square error.
- **RUL**: Remaining useful life of a part or consumable before it reaches its replacement limit.
- **Seasoning**: Conditioning runs that stabilise a chamber after PM.
- **SECS/GEM**: The standard communication protocol between semiconductor equipment and the host.
- **Shadow validation**: Validation that runs a candidate model beside the production model and compares results without using it for control.
- **SHAP**: SHapley additive explanations, which split a prediction into per-feature contributions.
- **SOA**: Safe operating area, the range of process conditions in which recommendations are allowed. SOA 95% is 95% of that range.
- **SPC**: Statistical process control, which compares process measurements with control limits to catch anomalies.
- **THK**: Thickness of the deposited film.
- **Tool qualification**: The procedure that confirms a new tool meets production conditions and releases it to production.
- **TTTM**: Tool-to-tool matching, which reduces result differences between tools and chambers on the same recipe.
- **Virtual metrology (VM)**: Predicting a measurement from process data before metrology.
- **Wasserstein distance**: A distance between distributions defined as the least cost of moving one into the other.

## Appendix B. Single vs Multi-Equipment

Sharing one AI across many tools makes the prediction more accurate as tools are added and cuts setup time and management load per tool.

Table 4. Single vs multi-equipment deployment

<table>
  <tr><th align="center">Aspect</th><th align="center">Item</th><th align="center">Single equipment</th><th align="center">Multi-equipment</th></tr>
  <tr><td align="center" rowspan="2">Training data</td><td align="center">Data volume</td><td align="center">One tool's data only</td><td align="center">All tools' data trained together</td></tr>
  <tr><td align="center">Domain coverage</td><td align="center">Predicts only situations already seen</td><td align="center">Uses other tools' experience as well</td></tr>
  <tr><td align="center" rowspan="2">Tool qualification</td><td align="center">Model setup</td><td align="center">Developed from scratch per tool</td><td align="center">Existing AI reused</td></tr>
  <tr><td align="center">Lead time</td><td align="center">Months</td><td align="center">Days to weeks</td></tr>
  <tr><td align="center" rowspan="2">SPC</td><td align="center">Control limit</td><td align="center">Managed per tool</td><td align="center">One limit shared by all tools</td></tr>
  <tr><td align="center">TTTM</td><td align="center">Hard to see</td><td align="center">Compared and corrected automatically</td></tr>
  <tr><td align="center" rowspan="2">Excursion control</td><td align="center">Early detection</td><td align="center">Reacts after the problem</td><td align="center">Warned in advance by other tools' cases</td></tr>
  <tr><td align="center">Containment</td><td align="center">The same problem recurs</td><td align="center">Spread to similar tools and blocked in advance</td></tr>
  <tr><td align="center" rowspan="3">Fab productivity</td><td align="center">Dispatching</td><td align="center">Fixed</td><td align="center">Assigned automatically to the best tool</td></tr>
  <tr><td align="center">Sampling·PM scheduling</td><td align="center">Planned per tool</td><td align="center">Optimised as one schedule</td></tr>
  <tr><td align="center">Engineer workload</td><td align="center">Grows with the number of tools</td><td align="center">Managed together by the AI agents</td></tr>
</table>

## Appendix C. Model Architecture

The prediction model is a hierarchy of a Global backbone shared by all tools and a lightweight Local adapter per chamber.

Table 5. Hierarchical model layers

| Layer                              | Role                                                                         | Technique                                               |
| :---:                              | :---:                                                                        | :---:                                                   |
| Global backbone (common model)     | Learns the process physics shared by all tools (recipe to THK)               | Data from all tools, tool/chamber embedding as input    |
| Local adapter (chamber model)      | Corrects the chamber fingerprint (chamber offset and gain)                   | Residual model, LoRA-style adapter, mixed-effects model |
| Few-shot extension (tool start-up) | Brings a new tool into qualification with a few measured monitor wafers      | Transfer learning, meta-learning (MAML family)          |
| Uncertainty (VM reliability)       | Computes the reliability of a VM value, the input to the guardrail threshold | Conformal prediction, ensemble variance                 |

The Local adapter learns only the chamber differences that the Global backbone leaves unexplained, so a drifting chamber retrains only its adapter. To give the Global backbone the same features from every tool, each tool's sensor data is mapped to a common schema.

- Sensor names, units and sampling periods that differ by tool, mapped to standard features through a sensor taxonomy
- RF power, pressure, gas flow and temperature reduced to the same per-step summary statistics
- Context such as recipe, step, RF hours since PM and seasoning state used as common inputs

## Appendix D. Orchestrator Functions

The orchestrator runs the event-driven workflow, holds the policies and permissions, and makes the decisions that span tools.

Table 6. Orchestrator functions

| Function                 | Description                                                                                                                                              |
| :---:                    | :---:                                                                                                                                                    |
| Monitor wafer allocation | Gives recalibration monitor wafers first to the chambers with the largest drift                                                                          |
| Tool start-up            | Transfers the Global backbone and adapters of the same tool type to cut the monitor wafers needed for qualification; the gates of Table 3 decide release |
| Chamber matching         | Corrects the THK offset between chambers on the same recipe (TTTM)                                                                                       |
| PM scheduling            | Coordinates consumable replacement and PM timing per tool from consumable RUL and the SOA narrowing with tool condition                                  |
| Excursion propagation    | Warns tools of the same type in advance about an excursion detected on one tool                                                                          |
| Dispatching              | Assigns lots and wafers to the best tool from tool condition and VM reliability                                                                          |
| Knowledge base           | Reuses feature extraction methods, failure cases and validation criteria as templates                                                                    |

An example stack is given in Table 7.

Table 7. Example technology stack

| Purpose                                                             | Software              |
| :---:                                                               | :---:                 |
| Orchestration (fab-wide event workflow, approval routing)           | LangGraph or Temporal |
| Model registry (versions of VM and R2R models and chamber adapters) | MLflow                |
| Feature store (common features extracted from traces)               | Feast                 |
| Event bus (process end, FDC alarm, metrology complete events)       | Kafka                 |

## Appendix E. Onboarding Gate Metrics

The metrics checked at each gate of Table 3 are listed in Table 8.

Table 8. Gate metrics

| Gate             | Metric                                                                                                                                                                                 |
| :---:            | :---:                                                                                                                                                                                  |
| Schema gate      | Feature mapping rate, presence of key sensors, match of sampling period and units                                                                                                      |
| Covariate shift  | Per-feature PSI, KS test, Wasserstein distance / Hotelling T² in PCA space, Mahalanobis distance / domain classifier AUC / per-step DTW distance, golden trace envelope violation rate |
| Concept shift    | Spearman correlation of SHAP importance ranks, comparison of key sensors' THK sensitivity (slope)                                                                                      |
| Pilot validation | RMSE, R², bias on 20~30 measured monitor wafers                                                                                                                                        |
