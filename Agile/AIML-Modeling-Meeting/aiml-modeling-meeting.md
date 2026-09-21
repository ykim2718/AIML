# AI/ML Modeling Meeting
Rev. 0 | Created: 2026-09-21 | Updated: 2026-09-21 10:20 CDT

## 1. Purpose

- **Problem Statement**: A modeling meeting run like an ordinary development meeting ends on "I will give it a try", so the room never records what was assumed, what was verified and what the next run is meant to decide, and the same discussion returns the following week.
- **Goal**: Hold the meeting as the joint between two experiment cycles, so that every meeting closes one experiment with its number, names the insight that number carries, and opens one hypothesis with an owner and a due date.
- **Non-Goal**: Configuring an experiment tracker (MLflow, Weights & Biases) or running a ticket system is not covered.

## 2. Summary

An AI/ML modeling meeting manages a chain of experiments, and it runs in the order the modeling pipeline itself runs: data, then hypothesis, then experiment result, then next action. The meeting closes on three recorded lines — the experiment completed, the insight it produced, and the hypothesis that runs next.

The order differs from an ordinary development meeting because the deliverable is discovered rather than built to specification. A software meeting asks whether the feature was built; a modeling meeting asks whether the hypothesis was verified, and a team that has folded its modeling workflow into an agile cycle still runs that workflow as its own sequence of stages [[1](#ref-1)].

## 3. Taxonomy and its Hierarchy

The items a modeling meeting decides stack in five layers, and each layer assumes the layer beneath it is already settled. A layer settled late invalidates everything built on it, so the cost of an error falls as the discussion moves up the stack.

The five layers, the question each one answers, and the order they are settled in are drawn in [Fig 1](#fig-1).

```text
[ 5 ] Next Action        "What runs next, and who owns it?"
         ^   assumes
[ 4 ] Experiment Result  "Why did the number move?"
         ^   assumes
[ 3 ] Hypothesis         "What one change, on what domain reason?"
         ^   assumes
[ 2 ] Data & Baseline    "What do we learn from, and what score must we beat?"
         ^   assumes
[ 1 ] Target (Y)         "What counts as the answer?"

Settled first      Target  ->  Data  ->  Hypothesis  ->  Result  ->  Action
Cost of an error   Target  >   Data  >   Hypothesis  >   Result  >   Action
```

<a id="fig-1"></a>
Fig 1. The five layers a modeling meeting decides, and the order they are settled in

A meeting opened above an unsettled layer produces an action nobody can evaluate. "Let us catch defects with AI" opens at layer 5 while layer 1 is empty, and every metric reported afterwards measures a quantity the team never defined.

### 3.1 Placement

Table 1. Where each layer is settled

| Layer             | Settled at               | What is fixed                                                                   |
| :---------------: | :----------------------: | :-----------------------------------------------------------------------------: |
| Target (Y)        | Before the first meeting | Yield below a stated bound, or a sensor crossing an EVT-based threshold         |
| Data & Baseline   | Agenda stage 1           | Missing value and outlier handling, the leakage barrier, the baseline score     |
| Hypothesis        | Agenda stage 2           | The one change of this cycle and the domain reason behind it                    |
| Experiment Result | Agenda stage 3           | The metric from the tracker, and the feature or error analysis that explains it |
| Next Action       | Agenda stage 4           | The next hypothesis, its owner, its due date and the engineering work it needs  |

## 4. Agenda

The meeting runs in four stages, and each stage fixes the one value the next stage consumes. Running them out of order returns the room to a layer of [Fig 1](#fig-1) it has already passed.

The stages and what each one puts on the table are drawn in [Fig 2](#fig-2).

```text
[ 1. Data & Baseline ]
        |
        +--> Missing / Outlier ....... How NaN and outliers in the raw rows were handled
        +--> Leakage Check ........... Sliding window augmentation checked for leakage
        +--> Baseline Score .......... The simplest model's score, stated as a number
        |
        v
[ 2. Hypothesis Setup ]
        |
        +--> Domain Reason ........... The physical ground the change rests on
        +--> One Change .............. The single thing that moves in this cycle
        +--> Expected Effect ......... What the metric does if the reason holds
        |
        v
[ 3. Experiment Review ]
        |
        +--> Tracked Run ............. Metric read from the tracker, not from memory
        +--> Why It Moved ............ Feature importance or error analysis behind it
        +--> Shown On Screen ......... Loss curve, confusion matrix, latent space plot
        |
        v
[ 4. Next Actions ]
        |
        +--> Next Hypothesis ......... The experiment this review has earned
        +--> Ticket .................. Owner and due date, issued in the tracker
        +--> Engineering Sync ........ Code review and pipeline work named here
        |
        v
[ Minutes ]   Done experiment   |   Insight   |   Next hypothesis
```

<a id="fig-2"></a>
Fig 2. The four stages of an AI/ML modeling meeting and what each one fixes

### 4.1 Data And Baseline

The first stage states the condition of the dataset the modeling will use, in numbers rather than in adjectives. Three questions carry it: how missing values and outliers among the raw rows were handled, whether sliding window augmentation left a path for data leakage, and what score the simplest model — a linear regression or a classical statistic — already reaches.

The baseline is the stage's output, since every later claim is a comparison against it. A first model kept simple is the established starting point for exactly this reason [[5](#ref-5)], and leakage is the failure that makes the comparison meaningless, with eight distinct forms recorded across 294 papers in seventeen fields [[3](#ref-3)].

### 4.2 Hypothesis Setup

The second stage sets the hypothesis this cycle tests, and it is built on domain knowledge rather than on a list of untried algorithms. The form is fixed: one change, the physical reason for it, and what the metric does if that reason holds.

Two examples show the form. Multicollinearity among sensors is severe, so the next run uses Elastic Net instead of Lasso to carry the group effect. Time warping distorts the signal, so a 1D-CNN autoencoder reduces the dimension through representation learning rather than a fixed transform.

### 4.3 Experiment Review

The third stage compares the runs on screen, from the tracker, and it answers why a number moved rather than which name won. "XGBoost comes out better" closes no layer of [Fig 1](#fig-1); "feature importance puts the chamber 3 pressure sensor at the top, and removing it returns the score to baseline" closes layer 4.

The screen carries the evidence: the loss curve, the confusion matrix, and the latent space of the reduced dimensions. A result described in speech cannot be checked by the room or reproduced afterwards.

### 4.4 Next Actions

The fourth stage turns the verified hypothesis into the next cycle, with an owner and a due date issued as a ticket. Production readiness items belong here as tickets rather than as intentions, since the rubric for them is a checklist of specific tests rather than a judgement [[2](#ref-2)].

Engineering work is named in this stage and nowhere else. Code review assignments and pipeline integration are synchronised here, so that the modeling discussion above them is not interrupted by scheduling.

## 5. Anti-patterns

Three habits end a modeling meeting without a decision, and each has a replacement that costs the same time.

Table 2. What ends a meeting without a decision

| Anti-pattern                           | Why it fails                                                    | What replaces it                                                                       |
| :------------------------------------: | :-------------------------------------------------------------: | :------------------------------------------------------------------------------------: |
| "Let us put all the data in and train" | Compute spent on a run whose outcome answers no stated question | Variables filtered by domain knowledge first, and the hypothesis the run tests, stated |
| Opening before the target is defined   | Every later metric measures a quantity the team never defined   | Y fixed first: yield below a stated bound, or an EVT-based threshold crossing          |
| Sharing a result in speech             | A claim the room cannot check and nobody can reproduce          | Loss curve, confusion matrix and latent space plot on the screen                       |

## 6. Roles

Three roles carry a modeling meeting: one names the physical constraint, the second turns it into an algorithm, and the third carries the result to the process pipeline.

Table 3. What each role brings

| Role           | Brings to the room                                                                         | Owns afterwards                            |
| :------------: | :----------------------------------------------------------------------------------------: | :----------------------------------------: |
| Domain expert  | Physical structure, such as two sensors that are symmetric and must be grouped by topology | The physical reading of the result         |
| Data scientist | The algorithm that carries that structure, such as a 1D-CNN filter size set to match it    | The experiment and its record              |
| MLOps engineer | The path from a trained model to the process pipeline                                      | Deployment and infrastructure verification |

## 7. Minutes Template

The minutes close on three lines, and a meeting that cannot fill them has not finished. The lines are the experiment completed, the insight read from it, and the hypothesis that runs next with its owner and date.

```text
Done this cycle : <experiment closed, with the number it produced>
Insight         : <what the result showed, and what it was read from>
Next hypothesis : <the change, the target metric, the owner, the due date>
```

Filled from one cycle, the three lines read as below. Each names a quantity, so that a reader three months later can tell what was actually established.

```text
Done this cycle : 1D-CNN autoencoder dimensionality reduction, 200 dimensions, reconstruction error 0.02
Insight         : Gas flow variation over the first 2,000 rows moves the final yield prediction most, by XGBoost feature importance
Next hypothesis : Elastic Net and supervised 1D-CNN on the 200 compressed features, for 95 % yield classification accuracy, owner <OWNER>, due 06-28
```

The same three lines fill the model card that ships with the model, which records its summary, its measured performance and the data it was trained on [[4](#ref-4)].

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
- **Baseline**: the score of the simplest model or rule, kept as the reference every later model is compared against.
- **Confusion matrix**: the table of predicted against actual classes, read to see which class a classifier confuses with which.
- **Data leakage**: information reaching the model that would not be available when it serves, which raises the offline score without raising the online one.
- **Elastic Net**: a linear model penalised by both the L1 and the L2 norm, which keeps correlated variables together rather than selecting one of them.
- **EVT (Extreme Value Theory)**: the statistics of the tail of a distribution, used here to set a threshold from how extreme a sensor value is.
- **Experiment tracking**: the automatic record of parameters, dataset version, code commit and result for each run.
- **Feature importance**: the score a fitted model attaches to each input, read to see which input moved the prediction.
- **Latent space**: the reduced coordinates an encoder maps its input onto.
- **Loss curve**: the training and validation loss plotted against training step.
- **MLOps**: the practice that carries a model from experiment into operation and keeps it there.
- **Multicollinearity**: a near-linear dependence among input variables, which makes individual coefficients unstable.
- **Representation learning**: learning the features themselves from the data rather than specifying them by hand.
- **Sliding window augmentation**: cutting overlapping windows out of a continuous record to make more training samples, which shares rows between windows.
- **Target (Y)**: the quantity the model predicts, whose definition fixes what counts as a correct answer.
- **Time warping**: a distortion of the time axis that shifts or stretches a signal between records of the same process.
- **Yield**: the fraction of produced units that meet specification.
