# AI/ML Code Review Meeting
Rev. 0 | Created: 2026-09-21 | Updated: 2026-09-21 10:07 CDT

## 1. Purpose

- **Problem Statement**: A review meeting run on modeling work reads the diff line by line, which leaves the split, the baseline and the run-to-run spread unexamined, so a model is approved on a number nobody in the room could reproduce.
- **Goal**: Fix what a meeting settles that a pull request cannot, the packet the author brings so the room can settle it, and the decision the meeting must end on.
- **Non-Goal**: General code style review — naming, structure, test coverage — is not covered here, since a single reviewer settles it asynchronously.

## 2. Summary

A modeling review meeting judges one claim: that a model trained on a stated dataset beats a stated baseline by more than the noise between runs. The room decides whether that claim holds and what happens next.

Everything a machine or one reader can settle is settled before the meeting opens. What remains is the judgement that needs several people at once: whether the split was honest, whether the comparison was fair, whether the gain survives a change of seed, and whether the model is fit to serve. The meeting ends on one of four decisions, each with an owner and a date.

## 3. Taxonomy and its Hierarchy

Review happens at three levels, told apart by how many people a finding needs, and each level passes upward only what it cannot settle. A step up buys judgement and gives up throughput, so the number of items reaching each level falls as the cost of each item rises.

The three levels, the question each one answers, and what each one settles are drawn in [Fig 1](#fig-1).

```text
Review Meeting   (several people)   "Is the result worth building on?"
  |
  |   settles   experiment validity, metric against baseline, model fitness,
  |             the decision to ship, rework or stop
  |
  +-- Async Review   (one reviewer)   "Is the change correct and readable?"
        |
        |   settles   local correctness, naming, structure, test coverage,
        |             leakage visible in the diff
        |
        +-- Automated Check   (no reader)   "Does it pass what a machine decides?"
              |
              |   settles   lint, type, unit test, data schema validation,
              |             experiment logged, metric recorded

Cost per item rises        Automated  <  Async  <  Meeting
Items reaching the level   Automated  >  Async  >  Meeting
```

<a id="fig-1"></a>
Fig 1. The three review levels and what each one settles

An item placed at the wrong level costs twice. A lint finding raised in the meeting spends several people on what a hook decides, and an experiment whose baseline is missing, approved in a pull request, is discovered only when the model reaches serving.

### 3.1 Placement

Table 1. Where each item is settled

| Item                               | Settled at      | Why there                                             |
| :--------------------------------: | :-------------: | :---------------------------------------------------: |
| Lint, format, type                 | Automated check | A fixed rule, decidable without a reader              |
| Unit test, schema validation       | Automated check | A pass or fail the pipeline already computes          |
| Experiment logged, metric recorded | Automated check | A record whose absence is mechanical to detect        |
| Naming, structure, test coverage   | Async review    | One reader's judgement, no second opinion needed      |
| Leakage visible in the diff        | Async review    | Readable from the code that builds the split          |
| Split design and its honesty       | Review meeting  | Several people needed to find what the author assumed |
| Baseline choice and the comparison | Review meeting  | A judgement about fairness, not about correctness     |
| Gain against run-to-run spread     | Review meeting  | A claim the room accepts or rejects together          |
| Serving fitness                    | Review meeting  | Crosses modeling, engineering and operation at once   |
| Ship, rework or stop               | Review meeting  | A decision with an owner, recorded once               |

## 4. Preparation

The meeting opens on a packet the author circulates one working day ahead, and a reviewer who has not read it attends as an observer rather than a voter, which keeps the room from spending its first half on what the packet already answers.

Table 2. What the review packet holds

| Part     | Content                                                                  |
| :------: | :----------------------------------------------------------------------: |
| Claim    | The question the experiment answered, in one sentence                    |
| Ask      | The decision wanted from the room                                        |
| Data     | Dataset version hash, row counts per split, the rule that made the split |
| Baseline | The score to beat, and where that number came from                       |
| Result   | The metric per run, with the spread across seeds                         |
| Diff     | The pull request link, already green and already reviewed asynchronously |
| Artifact | The model registry entry, or the reason there is none yet                |

A packet missing any part is returned rather than discussed. The parts are what the agenda in section 5 walks through, so a gap in the packet becomes a gap in the meeting.

## 5. Agenda

The running order follows the path a number takes to become a claim — data, then experiment, then code, then serving — and the decision is the last item rather than a conclusion reached on the way. The minutes below are one allocation of a 60-minute slot; a team scales them but keeps the order and the closing decision.

The stages and what each one puts on the table are drawn in [Fig 2](#fig-2).

```text
[ 0. Before the Meeting ]
        |
        +--> Review Packet ........... Circulated one working day ahead
        +--> CI Green ................ Lint, unit test and schema check passing
        |
        v
[ 1. Framing (5 min) ]
        |
        +--> Question ................ What the experiment was run to decide
        +--> Ask ..................... The decision the author wants from the room
        |
        v
[ 2. Data And Split (10 min) ]
        |
        +--> Split Rule .............. How train, validation and test were separated
        +--> Leakage Check ........... What could have reached the model from the target
        +--> Data Version ............ The dataset hash the run read
        |
        v
[ 3. Experiment And Metric (15 min) ]
        |
        +--> Baseline ................ The score the new model must beat
        +--> One Change .............. What moved between the compared runs
        +--> Run Spread .............. Variation across seeds, against the claimed gain
        |
        v
[ 4. Code And Reproducibility (10 min) ]
        |
        +--> Clean Rerun ............. The training script from a fresh checkout
        +--> Config .................. Paths and parameters out of the source
        +--> Notebook State .......... Exploration converted into a script or dropped
        |
        v
[ 5. Serving Readiness (10 min) ]
        |
        +--> Train/Serve Skew ........ The same preprocessing on both sides
        +--> Latency ................. The inference budget measured, not estimated
        +--> Fallback ................ The answer returned when confidence is low
        |
        v
[ 6. Decision (10 min) ]
        |
        +--> Outcome ................. Approve, approve with conditions, rework or stop
        +--> Actions ................. An owner and a date for every item raised
```

<a id="fig-2"></a>
Fig 2. The agenda of an AI/ML code review meeting

Four roles hold the meeting to the agenda of [Fig 2](#fig-2), each taken by a different person, since an author who also moderates ends the debate the moderator exists to bound.

Table 3. Roles in the room

| Role      | Holds                                               | Stays out of                                               |
| :-------: | :-------------------------------------------------: | :--------------------------------------------------------: |
| Author    | The claim, the packet, the answers to questions     | Defending a choice the room has already accepted as rework |
| Reviewer  | The packet read in advance, a judgement on validity | Style findings already settled asynchronously              |
| Moderator | The clock, the scope, the order of items            | The technical judgement itself                             |
| Scribe    | The decision and the action list                    | The discussion                                             |

One rule carries most of the timekeeping: debugging is out of scope in the room. An item nobody can resolve from the packet becomes an action with an owner, and the meeting moves on.

## 6. Review Points

Each stage asks one question of the work, and each has a failure that looks like success until someone names it. The checks below are what the room actually looks at, grouped by the stage that raises them.

Table 4. What each stage checks and how it fails

| Stage                    | Check                                                      | What a failure looks like                                                     |
| :----------------------: | :--------------------------------------------------------: | :---------------------------------------------------------------------------: |
| Data and split           | Split made before any fitting, scaler and encoder included | A test score that beats every later attempt, from a scaler fitted on all rows |
| Data and split           | Time order respected where rows carry time                 | A model that reads the future, scoring well until it meets live data          |
| Data and split           | Target absent from the features, directly and by proxy     | A near-perfect metric traced to a column derived from the label               |
| Data and split           | Test set touched once                                      | A test score tuned against, by repeated selection on the same holdout         |
| Experiment and metric    | Baseline present and reachable                             | A gain reported against nothing, or against a model nobody tuned              |
| Experiment and metric    | One variable moved per compared pair                       | A gain credited to the architecture while the data also changed               |
| Experiment and metric    | Gain larger than the spread across seeds                   | A claimed improvement inside the noise of a rerun                             |
| Experiment and metric    | Metric matching the decision the model serves              | A high accuracy on a class balance the operating environment never sees       |
| Code and reproducibility | Training runs from a clean checkout                        | A result reproducible only in the author's working directory                  |
| Code and reproducibility | Data version and parameters pinned, not typed in           | A rerun that reads a moved file and returns another number                    |
| Serving readiness        | Preprocessing identical in training and serving            | A model correct offline and wrong online, from a skewed transform             |
| Serving readiness        | Latency measured against its budget                        | A model within budget on a laptop and outside it under load                   |
| Serving readiness        | Behaviour fixed for a low-confidence answer                | An unhandled request answered with whatever the model emitted                 |

## 7. Outcome

The meeting closes on one of four decisions, said aloud and written down before the room breaks up. A meeting that ends without one of them returns to the queue unchanged, and the next meeting starts from the same packet.

Table 5. The four outcomes

| Outcome                 | What it means                                            | What is recorded                                   |
| :---------------------: | :------------------------------------------------------: | :------------------------------------------------: |
| Approve                 | The claim holds and the model may proceed                | The registry tag and the reviewer names            |
| Approve with conditions | The claim holds and named items must land before serving | Each condition with an owner and a date            |
| Rework                  | The claim is not yet supported by the evidence shown     | What evidence is missing, and the next review date |
| Stop                    | The direction is not worth another iteration             | The reason, kept where the next team will read it  |

The record belongs in the pull request and the experiment tracker rather than in a private note. An approval that lives in someone's inbox cannot be found when the model is questioned three months later.

---

## Appendix A. Terminology

- **Baseline**: the simplest model or rule kept as the reference score a new model must beat.
- **Data leakage**: information reaching the model that would not be available when it serves, which raises the offline score without raising the online one.
- **Data version hash**: the identifier that names one exact state of a dataset, so that a result can be traced to the rows it came from.
- **Holdout**: the split reserved for a single final measurement, kept out of every tuning decision.
- **Latency budget**: the limit an inference must answer within, stated as a percentile rather than an average.
- **Moderator**: the person who holds the clock and the scope of a meeting, and who does not judge the technical content.
- **Review packet**: the set of documents an author circulates before the meeting, holding the claim, the data, the baseline, the result and the artifact.
- **Run spread**: the variation of a metric across repeated runs that differ only in random seed.
- **Scribe**: the person who records the decision and the action list.
- **Seed**: the value that fixes the random draws of a run, so that the run can be repeated exactly.
- **Slice metric**: the metric computed on one subgroup of the data rather than on all of it.
- **Train/serve skew**: a difference between the preprocessing applied during training and the preprocessing applied when serving.
