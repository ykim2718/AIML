# Agile AI/ML Development
Rev. 0 | Created: 2026-09-13 | Updated: 2026-09-13 13:20 CDT

## 1. Purpose

- **Problem Statement**: An AI/ML project run under a software team's sprint rules cannot declare a task done, because the same code does not return the same result, because code is no longer the only thing under version control, and because a deployed model loses performance while nothing about it changes.
- **Goal**: Fix the three points at which an AI/ML cycle parts from a software cycle, and give both the rules that keep exploration inside one sprint and the bar that closes it, as checks a team can run.
- **Non-Goal**: Installing or configuring a particular MLOps tool (MLflow, W&B, DVC) is not covered.

## 2. Summary

Agile on an AI/ML project builds the simplest baseline model, carries it end to end into the operating environment within one or two weeks, and then improves it a little at a time from the data and the experiments that arrive afterwards. The practice of running that loop is called MLOps (Machine Learning Operations), or agile for AI.

The philosophy is the one every agile team shares — short iterations, fast feedback. What deforms it here is the exploratory nature of the work: the outcome of a sprint is discovered rather than built to specification. The rest of this document is that deformation — two more axes of things to version, a clock on the experiment, a definition of done that asks for a metric and for reproducibility, and a deployment that opens a monitoring period instead of closing the work.

## 3. Taxonomy and its Hierarchy

Three things are under management in an AI/ML project rather than one, and they are ordered: code behaves as written, data behaves as it arrived, and a model behaves as the two of them together allow. A step along that order buys generality and gives up determinism, so the proof each step takes is weaker in kind and costlier to obtain than the one before it.

The three axes, the question each one answers, and the proof each one takes are drawn in [Fig 1](#fig-1).

```text
Code    (deterministic)     "Does it run as written?"
  |
  |   proved by   unit test, code review, schema and type check
  |
  +-- Data   (distributional)   "Is what arrived what we assumed?"
        |
        |   proved by   missing value, outlier and schema validation,
        |               labelling cross-check, dataset hash
        |
        +-- Model   (probabilistic)   "Is it good enough, this time?"
              |
              |   proved by   a metric against a baseline, a bias test,
              |               the same result under a fixed seed

Determinism falls        Code  >  Data  >  Model
Cost of proving done     Code  <  Data  <  Model
```

<a id="fig-1"></a>
Fig 1. The three axes under version control and the proof each one takes

That ordering is what the three differences below come from. Each of them is a place where a software team's habit gives the wrong answer on an AI/ML project.

Table 1. Where an AI/ML cycle parts from a software cycle

| Difference | What it means | What it forces |
| --- | --- | --- |
| Non-determinism | The same code and the same model give a different score once the data or a hyperparameter moves | The work is run as research, with a clock on the experiment |
| Three axes | Software versions code; AI/ML versions code, data and model together | The bar for done asks about all three, not about the code alone |
| Drift | A model that worked loses performance as the real data trend moves away from the training set | Deployment opens a monitoring period instead of closing the work |

## 4. Lifecycle

The software flow of code, build and deploy gains two stages here — a data pipeline and an experiment — and the five together form a loop rather than a line. The loop closes on retraining, which is why the last stage feeds the first.

The stages and the terms met at each of them are drawn in [Fig 2](#fig-2).

```text
[ 1. Problem Framing ]
        |
        +--> Business KPI / Metric ... Success bar agreed (e.g. accuracy 95 % or better)
        +--> Feasibility Check ....... Data availability and technical reach judged
        |
        v
[ 2. Data Pipeline Sprint ]
        |
        +--> Ingestion / Labeling .... Collection and labelling, the longest step
        +--> Data Versioning (DVC) ... Dataset change history kept under a hash
        |
        v
[ 3. Exploratory Sprint ]   <-- the stage carrying the most uncertainty
        |
        +--> Baseline Model .......... Simplest rule or model, for a reference score
        +--> Experiment Tracking ..... Parameters, metrics and plots recorded per run
        +--> Spike / Timeboxing ...... "Closed in two days" agreed before it starts
        |
        v
[ 4. MLOps Pipeline ]
        |
        +--> Model Registry .......... Chosen model registered and versioned
        +--> Model Serving ........... Served as a REST API or on an edge device
        +--> Shadow / A/B Testing .... Compared behind the running system first
        |
        v
[ 5. Monitoring & Retraining ]
        |
        +--> Drift Detection ......... Data and concept drift caught as it appears
        +--> Continuous Training (CT)  Retraining pipeline running on its own
        +--> Retrospective ........... Findings carried into the next sprint
```

<a id="fig-2"></a>
Fig 2. The five stages of an AI/ML sprint and the terms met at each

Stage 3 is where the schedule breaks if it is going to. An experiment has no upper bound on how long it can be continued, since there is always one more thing to try, so the stage is given its limit in advance rather than judged afterwards. Stage 5 is where a software project would already be finished; here it is the stage that decides when the loop runs again.

## 5. Practices

Three rules keep the exploratory half of the work inside a sprint, each one answering a different way in which exploration escapes.

### 5.1 Timeboxing And Spike

An experiment is given a deadline before it is started, because chasing the last percent of a metric routinely costs weeks. Timeboxing states the limit as part of the ticket — "this experiment closes in two days, whatever it has found" — and a spike ticket carries the research-shaped subtask out of the delivery board entirely, so that an unfinished investigation does not hold an otherwise finished sprint open.

### 5.2 Data-Centric AI

Repairing the data moves the metric further than rebuilding the model, so the short iteration is spent there first. Fixing labelling errors and removing noise raises the score more reliably than a change of architecture, and the improvement survives the next model, which the architecture change does not.

### 5.3 Shadow Deployment

A new model answers in the background before it answers a customer. The live request is served by the existing system or the previous model, while the new one computes its answer beside it on the same input; the comparison measures the new model in the real environment, and the release follows only once that comparison holds.

## 6. Definition Of Done

Done on an AI/ML ticket means the data is trustworthy, the model clears its metric, the system answers fast enough, and the monitoring is attached. The software bar is a subset of it rather than a lighter version of it.

Table 2. The bar for done on each kind of project

| Kind | What it requires |
| --- | --- |
| Software DoD | Feature built, test code passing |
| AI/ML DoD | Feature built, target metric reached (for example F1-score above 0.88), data and model versioned, inference latency within its limit |

The seventeen checks below are the working form of that bar, grouped by what each one protects.

Table 3. The AI/ML definition of done, area by area

| Area | Check | What must be true |
| --- | --- | --- |
| Data & Feature | Data validation | Missing values, outliers and schema errors, caught automatically in the pipeline |
| Data & Feature | Label review | A labelling cross-check passed against the agreed consistency bar |
| Data & Feature | Data versioning | Train, validation and test sets versioned and stored under a unique hash |
| Data & Feature | Feature store entry | New cleaned features registered in the shared store, reusable by the team |
| Model & Experimentation | Target metric | The agreed metric above the baseline or above the model now in operation |
| Model & Experimentation | Fairness and bias | A bias test passed, with no group or class carrying a skewed prediction |
| Model & Experimentation | Experiment record | Hyperparameters, dataset version, code commit, metrics and plots logged automatically |
| Model & Experimentation | Reproducibility | The same result under the same random seed and the same parameters |
| Code & Testing | Code review | A pull request approved by at least one peer engineer |
| Code & Testing | Unit and integration tests | Pipeline and pre/post-processing code at the agreed coverage |
| Code & Testing | Model registry entry | The verified model versioned under a staging or candidate tag |
| Serving & MLOps | Inference performance | The serving SLA met, for example P95 latency under 100 ms |
| Serving & MLOps | Serving API test | Container and REST or gRPC endpoint passing an integration test |
| Serving & MLOps | Shadow and A/B readiness | A deployment path that takes part of the traffic or runs behind the live system |
| Serving & MLOps | Monitoring hookup | Inference data stored, drift detection and infrastructure metrics on a dashboard |
| Documentation | Model card | Summary, input and output format, limits, measured performance and dataset, updated |
| Documentation | Failure rule | The fallback for a low-confidence result written down |

### 6.1 Operating The Checklist

Applying all seventeen checks to every ticket slows the sprint down, so the list is split by the kind of sprint the ticket belongs to — a data sprint answers the first group, a research sprint the second, a serving sprint the fourth. A ticket then meets the checks that protect what it actually changed.

The checks that a machine can run are run by the machine. Data validation, code tests, the experiment record and the latency measurement all belong in the CI/CD and continuous training pipeline, where passing or failing is decided on every merge rather than remembered by a person at the end of the sprint.

---

## Appendix A. Terminology

- **A/B testing**: serving two versions to separate slices of live traffic and comparing the result.
- **Baseline model**: the simplest rule or model, kept as the reference score every later model must beat.
- **Concept drift**: a change in the relationship between input and target, which lowers performance without the input distribution moving.
- **CT (Continuous Training)**: the pipeline that retrains and redeploys a model without a person starting it.
- **Data drift**: a change in the distribution of the input data away from the training set.
- **Data versioning**: recording each dataset state under a unique hash, so that a result can be traced to the data that produced it.
- **Data-Centric AI**: the practice of improving the data rather than the model architecture to raise performance.
- **DoD (Definition of Done)**: the explicit bar a team agrees on, which a task must clear to be called done.
- **DVC (Data Version Control)**: a tool that versions datasets alongside the code repository.
- **Experiment tracking**: the automatic record of hyperparameters, dataset version, code commit and result for each run.
- **Feature store**: the shared repository of cleaned features, so that one team's feature is reusable by another.
- **Hyperparameter**: a training setting chosen before fitting, which is not learned from the data.
- **MLOps (Machine Learning Operations)**: the practice that carries a model from experiment to operation and keeps it there.
- **Model card**: the document that records a model's summary, input and output format, limits, measured performance and training data.
- **Model registry**: the store that holds trained models under a version and a stage tag.
- **Model serving**: answering requests from a fitted model, through an API or on a device.
- **Reproducibility**: the property that the same seed and parameters return the same result.
- **Shadow deployment**: running a new model beside the live one on the same input, without its answer reaching the user.
- **SLA (Service Level Agreement)**: the agreed limit a service must stay within, such as a latency percentile.
- **Spike**: a separately ticketed investigation, carried out of the delivery board because its outcome is unknown.
- **Sprint**: one iteration of the agile cycle, normally one to four weeks.
- **Timeboxing**: fixing in advance how long a task may run, and closing it at that limit whatever it has reached.
