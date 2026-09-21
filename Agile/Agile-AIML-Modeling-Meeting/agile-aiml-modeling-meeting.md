# Agile AI/ML Modeling Meeting
Rev. 13 | Created: 2026-09-21 | Updated: 2026-09-21 15:53 CDT

## 1. Purpose

- **Problem Statement**: The examples of agile development applied to AI/ML are vague, so a practitioner is left without a worked account of what the framework asks of a modeling team.
- **Goal**: Apply the Scrum Guide [[5](#ref-5)] stage by stage to AI/ML modeling, so that a practitioner has a method for running agile model development.
- **Non-Goal**: The document gives the process flow; it does not do the modeling.

## 2. Summary

Agile AI/ML model development runs as four stages that repeat once a sprint, and a stage closes only when the items it settles have each passed their definition of done. Nine items are settled in total, and the meeting on the sprint boundary is where the last stage closes one sprint and opens the next.

The four stages carry the Scrum Guide's events, artifacts and commitments rather than replacing them [[5](#ref-5)]: the Hypothesis is the sprint backlog item, the done-when of each item is the commitment attached to the Increment, and a promotable model is the Increment itself. The habits that end a sprint without a decision are each an item nobody supplied.

## 3. Taxonomy and its Hierarchy

A stage opens only once the stage before it has closed, and a metric produced while an earlier stage is still open settles nothing. The nine items are grouped by the stage that settles them, and the four stages run in that order.

The four stages, the items each one settles, and what each item fixes are drawn in [Fig 1](#fig-1).

```text
STAGE                    ITEM          WHAT IT FIXES

1  Premise Check    >    Target        What counts as the answer: Y and its threshold
                         Provenance    Where the rows came from, and what was done to them
                         Baseline      The score to beat, with the run that produced it
      |   every item done before
      v
2  Claim Setting    >    Hypothesis    One change, its physical reason, the movement expected
      |   every item done before
      v
3  Product Review   >    Run           One tracked execution
                         Insight       The explained cause of a metric move
                         Readiness     Latency, train/serve skew, fallback, monitoring
      |   every item done before
      v
4  Decision         >    Verdict       Accepted, rework or stop, issued on the Hypothesis
                         Handoff       Owner, due date and backlog item for the next sprint
```

<a id="fig-1"></a>
Fig 1. The nine items a modeling meeting settles, and the stage that settles each

What closes an item is its definition of done, and that bar differs from item to item.

### 3.1 Placement

<a id="table-1"></a>
Table 1. The nine items, the stage that settles each, and its definition of done

| Item       | Settled at       | Done when                                                                          | What carries it                                            |
| :--------: | :--------------: | :--------------------------------------------------------------------------------: | :--------------------------------------------------------: |
| Target     | 1 Premise Check  | Written as one quantity with its threshold, agreed before the sprint opens         | One line in the backlog item                               |
| Provenance | 1 Premise Check  | Split rule, missing value and outlier handling, and dataset hash recorded          | The tracker run and the data version tool                  |
| Baseline   | 1 Premise Check  | A run tagged as the baseline, carrying the score later claims are compared against | A tracker run with the baseline tag                        |
| Hypothesis | 2 Claim Setting  | One change, its physical reason and the expected movement, all stated              | The sprint backlog item                                    |
| Run        | 3 Product Review | Parameters, dataset version, code commit and metric all tracked                    | The tracker entry                                          |
| Insight    | 3 Product Review | The cause of the metric move reproduced from a clean checkout                      | Feature importance or error analysis, exported as a figure |
| Readiness  | 3 Product Review | Latency, skew, fallback and monitoring compared against the model now serving      | The model registry entry and the monitoring dashboard      |
| Verdict    | 4 Decision       | Said aloud by the product owner, on the Hypothesis                                 | One line in the minutes                                    |
| Handoff    | 4 Decision       | Owner, due date and item id issued for the next sprint                             | A backlog item with an id in the tracker                   |

## 4. Items

Each stage is worked through below in the order of [Fig 1](#fig-1), since a practitioner meets the items in that order.

### 4.1 Premise Check

Premise Check settles what must already be true for this sprint's metric to mean anything, and its three items are checked rather than debated. Together they are the definition of ready for a modeling backlog item: a sprint that opens while one of them is unsettled produces a number that decides nothing.

**Target** is the definition of Y with its threshold, written in one line. Yield below 98 %, or a sensor value crossing an EVT-based threshold, are targets; "catch defects with AI" is not, and a sprint opened on it measures a quantity the team never defined. The domain expert supplies it, and it is fixed before the first meeting rather than during one.

**Provenance** is where the rows came from and what was done to them: the split rule, the handling of missing values and outliers, the leakage barrier, and the dataset version hash. Sliding window augmentation is the usual place a barrier is lost, since overlapping windows share rows across the split. Leakage is recorded in eight distinct forms across 294 papers in seventeen fields, which is why it is a standing item rather than an occasional one [[3](#ref-3)].

**Baseline** is the score of the simplest model — a linear regression or a classical statistic — carried by a tracker run tagged as such. Keeping the first model simple is the established starting point [[6](#ref-6)], and without the tag the comparison every later claim rests on cannot be found again.

### 4.2 Claim Setting

Claim Setting settles the single assertion this sprint tests, and its one item is the Hypothesis. It is built on domain knowledge rather than on a list of untried algorithms, and it has three parts: one change, the physical reason for it, and what the metric does if that reason holds. One Hypothesis per modeler is the work-in-progress limit of the sprint, it carries a timebox fixed before the run starts, and research-shaped work that cannot state its three parts is moved off the board as a spike.

Two examples show the form. Multicollinearity among sensors is severe, so the run uses Elastic Net instead of Lasso to carry the group effect. Time warping distorts the signal, so a 1D-CNN autoencoder reduces the dimension through representation learning rather than through a fixed transform.

A Hypothesis without a physical reason cannot produce an Insight, because there is nothing for the result to confirm or contradict.

### 4.3 Product Review

Product Review settles what the sprint made, and its three items are read from artifacts on the screen rather than from memory. Each carries a different bar, which is why [Table 1](#table-1) states them one by one.

**Run** is one tracked execution carrying its parameters, dataset version, code commit and metric. A metric quoted without its run id is one the room cannot return to, so it closes nothing.

**Insight** is the explained cause of a metric move. "XGBoost comes out better" is a Run with no Insight; "feature importance puts the chamber 3 pressure sensor at the top, and removing it returns the score to the Baseline" is an Insight, and it is done once a clean checkout reproduces it. The loss curve, the confusion matrix and the latent space of the reduced dimensions go on the screen, since a result described in speech cannot be checked by the room.

**Readiness** is latency against its budget, train/serve skew, the fallback for a low-confidence answer, and the monitoring hookup. Promotion compares the candidate against the model already serving, and the rubric for that comparison is a checklist of specific tests rather than a judgement [[2](#ref-2)].

### 4.4 Decision

Decision settles what leaves the room, and its two items are always issued together. A meeting that produces one without the other returns to the queue unchanged.

**Verdict** is accepted, rework or stop, issued on the Hypothesis and said aloud by the product owner, since accepting an increment belongs to the role that owns the order of the backlog. Accepted requires a reproduced Insight; promoting a model to serving requires a compared Readiness; rework names the item that fell short of its bar; stop names the reason and is kept where the next team will read it.

**Handoff** is the owner, the due date and the item id for the next sprint, together with the engineering work the Verdict implies. It is the next sprint backlog item, written as a Hypothesis rather than as a task, and the hypotheses it outranks stay in the product backlog in the order the next Insight would decide. Code review assignments and pipeline integration are named here and nowhere else, so that the modeling discussion is not interrupted by scheduling.

## 5. Agenda

The agenda is the four stages of [Fig 1](#fig-1) in order, and each stage ends when its items can be said in one sentence. The meeting is timeboxed like every other ceremony, and stage 4 is protected: an earlier stage that overruns loses depth rather than taking the time the Verdict needs.

The four sentences below are what a practitioner reads out to close each stage. A stage whose sentence cannot be completed has not finished, whatever else was discussed.

```text
Stage 1   "Target is <TARGET>. Provenance: split by <RULE>, data <HASH>.
           Baseline is <METRIC> = <VALUE>, run <RUN_ID>."

Stage 2   "We change <ONE_THING> because <PHYSICAL_REASON>.
           If it holds, <METRIC> moves <DIRECTION> by at least <AMOUNT>."

Stage 3   "<METRIC> moved <FROM> to <TO>, run <RUN_ID>, spread <SPREAD> over
           <N> seeds. Insight: <ANALYSIS>, reproduced from a clean checkout."

Stage 4   "Verdict <ACCEPTED|REWORK|STOP>. Handoff: <OWNER> runs <EXPERIMENT>
           by <DATE>, backlog item <ITEM_ID>."
```

The meeting itself sits on the sprint boundary, so stage 4 opens the next sprint as it closes this one. Between two meetings the blockers on the running experiment are raised at the daily standup rather than held for the boundary.

## 6. Agile Practice

Each practice below keeps the name the Scrum Guide gives it and changes only what it holds [[5](#ref-5)]. A team adopting this meeting adds vocabulary to the events, artifacts and commitments it already runs rather than new ceremonies. Story points and velocity are left out of the table, since how long an experiment runs is unknown until it has run.

Table 2. Where each agile practice lands in the modeling sprint

| Practice        | What it carries here                                          | What changes for modeling                                                                    |
| :-------------: | :-----------------------------------------------------------: | :------------------------------------------------------------------------------------------: |
| Sprint          | The Hypothesis under review, from Premise Check to Verdict    | Length set by how long one experiment takes to reproduce                                     |
| Product backlog | The hypotheses not yet taken into a sprint, in order          | Ordered by what the next Insight would decide                                                |
| Sprint backlog  | The Hypothesis, one per sprint                                | A backlog item is a claim to test, not a feature to build                                    |
| WIP limit       | One Hypothesis in flight per modeler                          | Two changes at once leave the Insight unattributable                                         |
| Timeboxing      | The clock on the experiment and on the meeting                | The experiment closes at its limit, whatever it has found                                    |
| Spike           | Research-shaped work moved off the delivery board             | Its output is a decision, not a model                                                        |
| Daily standup   | Blockers on the running experiment                            | Raised the day they appear, not at the sprint boundary                                       |
| Sprint review   | Stage 3, Product Review                                       | The demo is the tracked run and the analysis plot                                            |
| Retrospective   | The process finding of stage 4                                | Recorded apart from the Insight, which is a finding about the model                          |
| DoR             | Stage 1 done: Target written, Provenance and Baseline tracked | Outside the Scrum Guide, which calls an item ready for selection when one sprint can Done it |
| DoD             | The done-when column of [Table 1](#table-1)                   | The Scrum Guide's commitment for the Increment, written per item rather than as one bar      |
| Increment       | Readiness compared against the model now serving              | The increment is a model that can be promoted, or nothing                                    |
| BKM             | Where a stop Verdict and its reason are kept                  | A direction closed is knowledge the next team reads                                          |

Two of the practices decide whether the meeting can close at all. Without the WIP limit no Verdict can be issued on the Hypothesis, since the sprint moved more than one thing and the room cannot say which one it is judging. Without a done-when written per item, a metric is refused by argument rather than by rule, and the argument outlasts the meeting.

## 7. Anti-patterns

Three habits end a modeling meeting without a Verdict, and each one is an item of [Fig 1](#fig-1) that nobody supplied. Naming the missing item is faster than debating the habit.

Table 3. What ends a meeting without a Verdict

| Habit                                  | Missing item | What to bring instead                                      |
| :------------------------------------: | :----------: | :--------------------------------------------------------: |
| "Let us put all the data in and train" | Hypothesis   | One change with its physical reason, stated before the run |
| "Let us catch defects with AI"         | Target       | Y and its threshold, written, before the meeting opens     |
| "It came out roughly fine"             | Run          | The tracked run and the analysis plot on the screen        |

## 8. Roles

Four roles supply the items, and no role supplies all of them. A meeting missing one role is missing the items that role carries.

Table 4. Which role supplies which items

| Role           | Supplies                                                                       | Stage   |
| :------------: | :----------------------------------------------------------------------------: | :-----: |
| Product owner  | The order of the hypotheses in the backlog, and the Verdict                    | 2, 4    |
| Domain expert  | Target, the physical reason inside Hypothesis, the physical reading of Insight | 1, 2, 3 |
| Data scientist | Provenance, Baseline, Run, Insight                                             | 1, 3    |
| MLOps engineer | Readiness, and the engineering work inside Handoff                             | 3, 4    |

The domain expert says that two sensors are symmetric and must be grouped by topology; the data scientist answers with the 1D-CNN filter size that carries that topology into the model.

## 9. Record

The minutes carry three of the nine items — Verdict, Insight and Handoff — and a meeting that cannot fill them has not finished. The other six live in the tracker and the backlog item, which the three lines point at. The process finding of the same meeting is recorded apart, in the retrospective note that updates the BKM, since a record mixing the two reads as neither.

```text
Verdict : <ACCEPTED|REWORK|STOP> on <HYPOTHESIS>
Insight : <what moved the metric, and what it was read from>
Handoff : <OWNER> runs <EXPERIMENT> by <DATE>, backlog item <ITEM_ID>
```

Filled from one sprint, the three lines read as below. Each names a quantity, so that a reader three months later can tell what was established rather than what was attempted.

```text
Verdict : ACCEPTED on "1D-CNN autoencoder reduces the trace to 200 dimensions"
Insight : Gas flow variation over the first 2,000 rows moves the final yield
          prediction most, by XGBoost feature importance. Reconstruction error 0.02
Handoff : <OWNER> runs Elastic Net and supervised 1D-CNN on the 200 compressed
          features, target 95 % yield classification accuracy, by 06-28, backlog item <ITEM_ID>
```

The same three lines fill the model card that ships with the model, which records its summary, its measured performance and the data it was trained on [[4](#ref-4)]. A team that has folded this workflow into its agile process still runs it as its own sequence of stages [[1](#ref-1)].

## References

<a id="ref-1"></a>
[1] Amershi, S., Begel, A., Bird, C., DeLine, R., Gall, H., Kamar, E., Nagappan, N., Nushi, B., & Zimmermann, T. (2019). [Software Engineering for Machine Learning: A Case Study](https://doi.org/10.1109/ICSE-SEIP.2019.00042). *2019 IEEE/ACM 41st International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP)*.<br>
<a id="ref-2"></a>
[2] Breck, E., Cai, S., Nielsen, E., Salib, M., & Sculley, D. (2017). [The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction](https://doi.org/10.1109/BigData.2017.8258038). *2017 IEEE International Conference on Big Data (Big Data)*.<br>
<a id="ref-3"></a>
[3] Kapoor, S., & Narayanan, A. (2023). [Leakage and the reproducibility crisis in machine-learning-based science](https://doi.org/10.1016/j.patter.2023.100804). *Patterns*, 4(9), 100804.<br>
<a id="ref-4"></a>
[4] Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). [Model Cards for Model Reporting](https://doi.org/10.1145/3287560.3287596). *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT\* '19)*.<br>
<a id="ref-5"></a>
[5] Schwaber, K., & Sutherland, J. (2020). [The Scrum Guide](https://scrumguides.org/scrum-guide.html). November 2020.<br>
<a id="ref-6"></a>
[6] Zinkevich, M. [Rules of Machine Learning: Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml). Google for Developers.

---

## Appendix A. Terminology

- **1D-CNN autoencoder**: a convolutional network over a one-dimensional signal, trained to rebuild its own input, used here to reduce the dimension of a process trace.
- **Backlog item**: one unit of work in a backlog, held in the team's tracker under an id.
- **BKM (Best Known Method)**: the team document holding the best method known so far for a task, updated from retrospectives.
- **Blocker**: a technical or administrative obstacle that stops an experiment from moving to its next step.
- **Confusion matrix**: the table of predicted against actual classes, read to see which class a classifier confuses with which.
- **Daily standup**: the short daily meeting at which blockers on the running work are raised.
- **Data leakage**: information reaching the model that would not be available when it serves, which raises the offline score without raising the online one.
- **DoD (Definition of Done)**: the explicit bar a work item must clear to be called done.
- **DoR (Definition of Ready)**: the bar a work item must clear before a team takes it into a sprint, used in practice but absent from the Scrum Guide.
- **Elastic Net**: a linear model penalised by both the L1 and the L2 norm, which keeps correlated variables together rather than selecting one of them.
- **EVT (Extreme Value Theory)**: the statistics of the tail of a distribution, used here to set a threshold from how extreme a sensor value is.
- **Feature importance**: the score a fitted model attaches to each input, read to see which input moved the prediction.
- **Foundation model**: a large model pre-trained on broad data, adapted to a task by fine-tuning rather than trained from scratch.
- **ICE score**: an ordering score for a backlog item, from impact, confidence and ease.
- **Increment**: the working product one sprint produces.
- **Latent space**: the reduced coordinates an encoder maps its input onto.
- **Loss curve**: the training and validation loss plotted against training step.
- **Metric**: the value a run reports for the target quantity, such as accuracy, F1-score or RMSE.
- **MLOps**: the practice that carries a model from experiment into operation and keeps it there.
- **Model card**: the document recording a model's summary, measured performance and training data.
- **Multicollinearity**: a near-linear dependence among input variables, which makes individual coefficients unstable.
- **Product backlog**: the ordered list of work not yet taken into a sprint.
- **Product owner**: the role that owns the order of the backlog and accepts the increment.
- **Representation learning**: learning the features themselves from the data rather than specifying them by hand.
- **Retrospective**: the meeting at the end of a sprint that reviews the process and fixes what to change.
- **Sliding window augmentation**: cutting overlapping windows out of a continuous record to make more training samples, which shares rows between windows.
- **Spike**: an investigation carried as its own backlog item, off the delivery board because its outcome is unknown.
- **Sprint**: the fixed-length span that carries one Hypothesis, from the meeting that opens it to the meeting that issues its Verdict.
- **Sprint backlog**: the work a team commits to finish in one sprint.
- **Story point**: a relative estimate of the size of a backlog item.
- **Time warping**: a distortion of the time axis that shifts or stretches a signal between records of the same process.
- **Timeboxing**: fixing in advance how long a task may run, and closing it at that limit whatever it has reached.
- **Train/serve skew**: a difference between the preprocessing applied during training and the preprocessing applied when serving.
- **Velocity**: the number of story points a team completes in one sprint.
- **WIP (Work In Progress)**: work currently under way, limited in number so that each item can be finished before the next starts.
- **Yield**: the fraction of produced units that meet specification.

## Appendix B. Experiment Priority

Brute force — "let us just run everything and see" — is the habit that spends a sprint of compute and returns no Insight, since a run nobody framed answers no question. An idea reaches the product backlog only after passing three filters, and its order inside the backlog comes from an ICE score.

### B.1 Three Filters

A proposal is screened before any code is written, and a filter that rejects it costs minutes where the experiment would have cost weeks.

Table 5. The three filters and what each one rejects

| Filter                        | Question                                                                                  | Rejected when                                              |
| :---------------------------: | :---------------------------------------------------------------------------------------: | :--------------------------------------------------------: |
| Domain alignment              | Do the mathematical properties of this algorithm match the physics of the process data?   | The choice contradicts what the process is known to do     |
| Cost-benefit                  | Is the metric return certain against the time and compute the model costs?                | Three weeks of work buys one point over a cheap baseline   |
| Explainability and deployment | Does the output convince a process engineer, and does the model fit the serving pipeline? | The model is a black box, or too heavy for the line to run |

```text
[ Proposed idea ]
       |
       v
[ 1. Domain alignment ]   --> rejected
       | passed
       v
[ 2. Cost-benefit ]       --> rejected
       | passed
       v
[ 3. Explainability ]     --> rejected
       | passed
       v
[ Product backlog, ordered by ICE score ]
```

<a id="fig-2"></a>
Fig 2. The three filters an idea passes before it reaches the product backlog

**Domain alignment** asks the question first, before the library is chosen. Lasso on sensors with deep mutual correlation drops all but one of a correlated group, so Ridge or Elastic Net comes first. A random forest that shuffles row order is the wrong main model for a continuous process whose causality runs along the time axis, where a 1D-CNN or a time-series model belongs.

**Cost-benefit** compares the return against the cheapest model already standing. A light XGBoost reaching 91 % accuracy outranks a large Transformer designed to reach 92 %, and raising the cheap baseline is the first call rather than the last.

**Explainability and deployment** is a constraint that holds whatever the score says. An ensemble of ensembles cannot answer inside a real-time virtual simulation, so it leaves the top of the backlog however well it scores offline.

### B.2 ICE Score

When the room diverges, each member scores the idea from 1 to 5 on three letters and the average orders the backlog. The score is a way to make disagreement explicit rather than a measurement.

Table 6. The three letters of an ICE score

| Letter | Name       | Question                                                                           |
| :----: | :--------: | :--------------------------------------------------------------------------------: |
| I      | Impact     | How far does the metric move if this Hypothesis holds?                             |
| C      | Confidence | How likely is it to hold, judged from published work or from earlier process data? |
| E      | Ease       | How little preprocessing and implementation does it take?                          |

```math
\mathrm{ICE} = I \times C \times E \hspace{19em} (1)
```

Equation (1) holds when the third letter is scored as Ease, where 5 is easiest. A team that prefers to score Effort, where 5 is hardest, divides by it instead and reads the same ordering.

A **quick win** scores high on all three: adjusting the stride of an existing sliding window to augment the data is implemented in an afternoon and its effect is known in advance. A **long-term** item scores low on Ease and Confidence together: fine-tuning a large time-series foundation model is weeks of work whose outcome nobody can predict, so it waits behind the quick wins rather than opening the sprint.

### B.3 Hypothesis Rule

No Hypothesis, no experiment. A proposal that cannot be said as a Hypothesis — one change, its physical reason, the expected movement — stays out of the next sprint, whatever its novelty.

> Rejected: "LightGBM is popular at the moment, let us give it a run too."

> Accepted: "The sensor data carries noise the linear model appears to be overfitting. A tree-based LightGBM is robust to missing values and noise, so the hypothesis is that accuracy rises by three points or more over the current Baseline. We will test that."
