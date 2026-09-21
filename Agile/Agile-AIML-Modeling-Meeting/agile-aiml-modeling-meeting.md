# Agile AI/ML Modeling Meeting
Rev. 2 | Created: 2026-09-21 | Updated: 2026-09-21 10:38 CDT

## 1. Purpose

- **Problem Statement**: A modeling meeting run like an ordinary development meeting ends on "I will give it a try", because the room has no name for the things it is exchanging and therefore no way to say which of them is missing.
- **Goal**: Give the meeting ten named terms in four classes and place them on the agile practices the cycle already runs, so that a practitioner can say in one sentence what is on the table, what is missing, what grade of checking the missing part has to reach, and which sprint artifact it becomes.
- **Non-Goal**: Configuring an experiment tracker (MLflow, Weights & Biases) or running a ticket system is not covered.

## 2. Summary

An AI/ML modeling meeting exchanges ten things, and naming them is what keeps the meeting from ending without a decision. The ten fall into four classes — Premise, Claim, Product, Decision — and the classes run in that order, which is also the agenda. The meeting sits on a sprint boundary: it closes the cycle whose Product is on the table and opens the next one.

Each class assumes the class before it is settled. A room discussing a Product while its Premise is open is measuring a quantity nobody defined, and the three habits that most often end a modeling meeting without a decision are each a missing term from one of the four classes.

## 3. Taxonomy and its Hierarchy

The ten terms are classified by what they fix, and they are ordered by what they assume. Premise fixes what is true before this cycle, Claim fixes what the cycle asserts, Product fixes what the cycle produced, and Decision fixes what leaves the room.

The four classes, the ten terms, and the grade ladder that says how far a number has been checked are drawn in [Fig 1](#fig-1).

```text
CLASS             TERM          WHAT IT FIXES

Premise       >   Target        What counts as the answer: Y and its threshold
                  Provenance    Where the rows came from, and what was done to them
                  Baseline      The score to beat, with the run that produced it
    |   assumed by
    v
Claim         >   Hypothesis    One change, its physical reason, the movement expected
    |   tested by
    v
Product       >   Run           One tracked execution
                  Evidence      How far a Run has been checked, on the grade ladder
                  Insight       The explained cause of a metric move
                  Readiness     Latency, train/serve skew, fallback, monitoring
    |   judged into
    v
Decision      >   Verdict       Accepted, rework or stop, issued on the Hypothesis
                  Handoff       Owner, due date and ticket, issued for the next cycle

GRADE LADDER      E0 assertion < E1 number < E2 run-backed < E3 reproduced < E4 compared
```

<a id="fig-1"></a>
Fig 1. The ten terms of a modeling meeting, their four classes, and the evidence grade ladder

The grade ladder applies to every term that carries a number. E0 is spoken only and is not admissible, E1 is a value with no run behind it, E2 carries the run id with its parameters and data version, E3 is an E2 reproduced from a clean checkout inside the stated spread, and E4 is an E3 set beside the Baseline or the live model on the same split.

### 3.1 Placement

<a id="table-1"></a>
Table 1. The ten terms, and what each one takes to be settled

| Term       | Class    | Required grade                                        | What carries it                                            |
| :--------: | :------: | :---------------------------------------------------: | :--------------------------------------------------------: |
| Target     | Premise  | E1, written before the first meeting                  | One line in the ticket: the quantity and its threshold     |
| Provenance | Premise  | E2                                                    | The tracker run and the dataset version hash               |
| Baseline   | Premise  | E2                                                    | A tracker run tagged as the baseline                       |
| Hypothesis | Claim    | Not a number; one change plus its reason              | The ticket opened for this cycle                           |
| Run        | Product  | E2                                                    | The tracker entry with parameters, data version and commit |
| Evidence   | Product  | The grade itself, said aloud with the number          | The grade stated beside every metric                       |
| Insight    | Product  | E3                                                    | Feature importance or error analysis, exported as a figure |
| Readiness  | Product  | E4                                                    | The model registry entry and the monitoring dashboard      |
| Verdict    | Decision | Requires Insight at E3, or Readiness at E4 to promote | One line in the minutes                                    |
| Handoff    | Decision | Owner and date, no grade                              | A ticket with an id                                        |

## 4. Terms

Each class is worked through below in the order of [Fig 1](#fig-1), since a practitioner reads them in the order the meeting needs them.

### 4.1 Premise

Premise is what must already be true for this cycle's number to mean anything, and it is checked rather than debated. The three terms are Target, Provenance and Baseline.

**Target** is the definition of Y with its threshold, written in one line. Yield below 98 %, or a sensor value crossing an EVT-based threshold, are targets; "catch defects with AI" is not, and a cycle opened on it measures a quantity the team never defined. The domain expert supplies it, and it is fixed before the first meeting rather than during one.

**Provenance** is where the rows came from and what was done to them: the split rule, the handling of missing values and outliers, the leakage barrier, and the dataset version hash. Sliding window augmentation is the usual place a barrier is lost, since overlapping windows share rows across the split. Leakage is recorded in eight distinct forms across 294 papers in seventeen fields, which is why it is a standing item rather than an occasional one [[3](#ref-3)].

**Baseline** is the score of the simplest model — a linear regression or a classical statistic — carried by a tracker run tagged as such. Keeping the first model simple is the established starting point [[5](#ref-5)], and without the tag the comparison every later claim rests on cannot be found again.

### 4.2 Claim

Claim is the single assertion this cycle tests, and it has exactly one term, Hypothesis. It is built on domain knowledge rather than on a list of untried algorithms, and it has three parts: one change, the physical reason for it, and what the metric does if that reason holds. One Hypothesis per modeler is the work-in-progress limit of the cycle, it carries a timebox fixed before the run starts, and research-shaped work that cannot state its three parts is moved off the board as a spike.

Two examples show the form. Multicollinearity among sensors is severe, so the run uses Elastic Net instead of Lasso to carry the group effect. Time warping distorts the signal, so a 1D-CNN autoencoder reduces the dimension through representation learning rather than through a fixed transform.

A Hypothesis without a physical reason cannot produce an Insight, because there is nothing for the result to confirm or contradict.

### 4.3 Product

Product is what the cycle actually made, and the four terms are graded rather than described. Run, Evidence, Insight and Readiness are read from artifacts on the screen, not from memory.

**Run** is one tracked execution carrying its parameters, dataset version, code commit and metric. **Evidence** is the grade that Run has reached, said aloud beside the number, so that the room knows whether it is hearing E1 or E3. The required grade of each term in [Table 1](#table-1) is the definition of done for a modeling ticket.

**Insight** is the explained cause of a metric move. "XGBoost comes out better" is a Run with no Insight; "feature importance puts the chamber 3 pressure sensor at the top, and removing it returns the score to the Baseline" is an Insight at E3. The loss curve, the confusion matrix and the latent space of the reduced dimensions go on the screen, since a result described in speech stays at E0.

**Readiness** is latency against its budget, train/serve skew, the fallback for a low-confidence answer, and the monitoring hookup. Promotion compares the candidate against the model already serving, and the rubric for that comparison is a checklist of specific tests rather than a judgement [[2](#ref-2)].

### 4.4 Decision

Decision is what leaves the room, and it has two terms that are always issued together. A meeting that produces one without the other returns to the queue unchanged.

**Verdict** is accepted, rework or stop, issued on the Hypothesis and said aloud. Accepted requires Insight at E3; promoting a model to serving requires Readiness at E4; rework names the evidence that was missing; stop names the reason and is kept where the next team will read it.

**Handoff** is the owner, the due date and the ticket id for the next cycle, together with the engineering work the Verdict implies. It is the next sprint backlog item, written as a Hypothesis rather than as a task. Code review assignments and pipeline integration are named here and nowhere else, so that the modeling discussion is not interrupted by scheduling.

## 5. Agenda

The agenda is the four classes in order, one stage per class, and each stage ends when its terms can be said in one sentence. Running the stages out of order returns the room to a class of [Fig 1](#fig-1) it has already passed.

The stages, the class each one settles, and the items each one puts on the table are drawn in [Fig 2](#fig-2).

```text
[ Sprint N ends ]           the cycle whose Product is on the table
        |
        v
[ 1. Premise Check ]        settles   Target, Provenance, Baseline
        |
        +--> Missing / Outlier ....... How NaN and outliers in the raw rows were handled
        +--> Leakage Check ........... Sliding window augmentation checked for leakage
        +--> Baseline Score .......... The simplest model's score, with its run id
        |
        v
[ 2. Claim Setting ]        settles   Hypothesis
        |
        +--> One Change .............. The single thing that moves in this cycle
        +--> Domain Reason ........... The physical ground the change rests on
        +--> Expected Effect ......... What the metric does if the reason holds
        |
        v
[ 3. Product Review ]       settles   Run, Evidence, Insight, Readiness
        |
        +--> Tracked Run ............. Metric read from the tracker, not from memory
        +--> Evidence Grade .......... How far the number has been checked, E0 to E4
        +--> Why It Moved ............ Feature importance or error analysis behind it
        +--> Shown On Screen ......... Loss curve, confusion matrix, latent space plot
        |
        v
[ 4. Decision ]             settles   Verdict, Handoff
        |
        +--> Verdict ................. Accepted, rework or stop, said aloud
        +--> Handoff ................. Owner, due date and ticket id
        +--> Engineering Sync ........ Code review and pipeline work named here
        |
        v
[ Sprint N+1 begins ]       Handoff becomes the next sprint backlog item
```

<a id="fig-2"></a>
Fig 2. The four stages of the meeting and the class each one settles

Each stage closes on one sentence, and the four templates below are what a practitioner reads out to close it. A stage whose sentence cannot be completed has not finished, whatever else was discussed.

```text
Stage 1   "Target is <TARGET>. Provenance: split by <RULE>, data <HASH>.
           Baseline is <METRIC> = <VALUE>, run <RUN_ID>."

Stage 2   "We change <ONE_THING> because <PHYSICAL_REASON>.
           If it holds, <METRIC> moves <DIRECTION> by at least <AMOUNT>."

Stage 3   "<METRIC> moved <FROM> to <TO>, run <RUN_ID>, spread <SPREAD> over
           <N> seeds. Insight: <ANALYSIS>. Evidence grade E<GRADE>."

Stage 4   "Verdict <ACCEPTED|REWORK|STOP>. Handoff: <OWNER> runs <EXPERIMENT>
           by <DATE>, ticket <TICKET_ID>."
```

## 6. Agile Practice

Each agile practice below keeps the name a software team already uses and changes only what it holds. A team adopting this meeting adds vocabulary to the ceremonies it already runs rather than new ceremonies.

Table 2. Where each agile practice lands in the modeling cycle

| Practice       | What it carries here                                        | What changes for modeling                                       |
| :------------: | :---------------------------------------------------------: | :-------------------------------------------------------------: |
| Sprint         | The cycle one meeting closes and the next opens             | Length set by how long one experiment takes to reproduce        |
| Sprint backlog | The Hypothesis, one per cycle                               | A backlog item is a claim to test, not a feature to build       |
| WIP limit      | One Hypothesis in flight per modeler                        | Two changes at once leave the Insight unattributable            |
| Timeboxing     | The clock on the experiment                                 | The experiment closes at its limit, whatever it has found       |
| Spike          | Research-shaped work moved off the delivery board           | Its output is a decision, not a model                           |
| Sprint review  | Stage 3, Product Review                                     | The demo is the tracked run and the analysis plot               |
| Retrospective  | The Insight, and the reason behind a rework or stop Verdict | The process finding and the modeling finding are recorded apart |
| DoD            | The required Evidence grade of every term                   | Done is a grade on the ladder, not a checkbox                   |
| Increment      | Readiness at E4                                             | The increment is a model that can be promoted, or nothing       |
| BKM            | Where a stop Verdict and its reason are kept                | A direction closed is knowledge the next team reads             |

Two of the practices decide whether the meeting can close at all. Without the WIP limit no Verdict can be issued on the Hypothesis, since the cycle moved more than one thing and the room cannot say which one it is judging. Without the DoD written as a grade, a number is refused by argument rather than by rule, and the argument outlasts the meeting.

## 7. Anti-patterns

Three habits end a modeling meeting without a Verdict, and each one is a term of [Fig 1](#fig-1) that nobody supplied. Naming the missing term is faster than debating the habit.

Table 3. What ends a meeting without a Verdict

| Habit                                  | Missing term         | What to bring instead                                      |
| :------------------------------------: | :------------------: | :--------------------------------------------------------: |
| "Let us put all the data in and train" | Hypothesis           | One change with its physical reason, stated before the run |
| "Let us catch defects with AI"         | Target               | Y and its threshold, written, before the meeting opens     |
| "It came out roughly fine"             | Evidence, left at E0 | The tracked run at E2 and the analysis plot on the screen  |

## 8. Roles

Three roles supply the classes, and no role supplies all of them. A meeting missing one role is missing the class that role carries.

Table 4. Which role supplies which class

| Role           | Supplies                                                                       | Class             |
| :------------: | :----------------------------------------------------------------------------: | :---------------: |
| Domain expert  | Target, the physical reason inside Hypothesis, the physical reading of Insight | Premise, Claim    |
| Data scientist | Provenance, Baseline, Run, Evidence, Insight                                   | Premise, Product  |
| MLOps engineer | Readiness, and the engineering work inside Handoff                             | Product, Decision |

The domain expert says that two sensors are symmetric and must be grouped by topology; the data scientist answers with the 1D-CNN filter size that carries that topology into the model.

## 9. Record

The minutes carry three of the ten terms — Verdict, Insight and Handoff — and a meeting that cannot fill them has not finished. The other seven live in the tracker and the ticket, which the three lines point at.

```text
Verdict : <ACCEPTED|REWORK|STOP> on <HYPOTHESIS>
Insight : <what moved the metric, and what it was read from>
Handoff : <OWNER> runs <EXPERIMENT> by <DATE>, ticket <TICKET_ID>
```

Filled from one cycle, the three lines read as below. Each names a quantity, so that a reader three months later can tell what was established rather than what was attempted.

```text
Verdict : ACCEPTED on "1D-CNN autoencoder reduces the trace to 200 dimensions"
Insight : Gas flow variation over the first 2,000 rows moves the final yield
          prediction most, by XGBoost feature importance. Reconstruction error 0.02
Handoff : <OWNER> runs Elastic Net and supervised 1D-CNN on the 200 compressed
          features, target 95 % yield classification accuracy, by 06-28, ticket <TICKET_ID>
```

The same three lines fill the model card that ships with the model, which records its summary, its measured performance and the data it was trained on [[4](#ref-4)]. A team that has folded this workflow into an agile cycle still runs it as its own sequence of stages [[1](#ref-1)].

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
[5] Zinkevich, M. [Rules of Machine Learning: Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml). Google for Developers.

---

## Appendix A. Terminology

- **1D-CNN autoencoder**: a convolutional network over a one-dimensional signal, trained to rebuild its own input, used here to reduce the dimension of a process trace.
- **BKM (Best Known Method)**: the team document holding the best method known so far for a task, updated from retrospectives.
- **Confusion matrix**: the table of predicted against actual classes, read to see which class a classifier confuses with which.
- **Data leakage**: information reaching the model that would not be available when it serves, which raises the offline score without raising the online one.
- **DoD (Definition of Done)**: the explicit bar a team agrees on, which a task must clear to be called done.
- **Elastic Net**: a linear model penalised by both the L1 and the L2 norm, which keeps correlated variables together rather than selecting one of them.
- **EVT (Extreme Value Theory)**: the statistics of the tail of a distribution, used here to set a threshold from how extreme a sensor value is.
- **Feature importance**: the score a fitted model attaches to each input, read to see which input moved the prediction.
- **Increment**: the working product one sprint produces.
- **Latent space**: the reduced coordinates an encoder maps its input onto.
- **Loss curve**: the training and validation loss plotted against training step.
- **MLOps**: the practice that carries a model from experiment into operation and keeps it there.
- **Model card**: the document recording a model's summary, measured performance and training data.
- **Multicollinearity**: a near-linear dependence among input variables, which makes individual coefficients unstable.
- **Representation learning**: learning the features themselves from the data rather than specifying them by hand.
- **Retrospective**: the meeting at the end of a sprint that reviews the process and fixes what to change.
- **Sliding window augmentation**: cutting overlapping windows out of a continuous record to make more training samples, which shares rows between windows.
- **Spike**: a separately ticketed investigation, carried off the delivery board because its outcome is unknown.
- **Sprint**: one iteration of the agile cycle.
- **Sprint backlog**: the work a team commits to finish in one sprint.
- **Time warping**: a distortion of the time axis that shifts or stretches a signal between records of the same process.
- **Timeboxing**: fixing in advance how long a task may run, and closing it at that limit whatever it has reached.
- **Train/serve skew**: a difference between the preprocessing applied during training and the preprocessing applied when serving.
- **WIP (Work In Progress)**: work currently under way, limited in number so that each item can be finished before the next starts.
- **Yield**: the fraction of produced units that meet specification.
