# Trace Feature Selection
Rev. 8 | Created: 2026-09-10 | Updated: 2026-10-29 03:08 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Comparison](#31-comparison)
- [4. Mechanism Axis](#4-mechanism-axis)
  - [4.1 Preprocessing](#41-preprocessing)
  - [4.2 Filter](#42-filter)
  - [4.3 Wrapper](#43-wrapper)
  - [4.4 Embedded](#44-embedded)
  - [4.5 Post-Hoc Ranking](#45-post-hoc-ranking)
  - [4.6 Error-Controlled Selection](#46-error-controlled-selection)
- [5. Selection Unit Axis](#5-selection-unit-axis)
- [6. Stability Axis](#6-stability-axis)
- [7. Failure Modes](#7-failure-modes)
- [8. Further Work](#8-further-work)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

- **Problem Statement**: One recipe's trace yields thousands of feature columns from hundreds of sensors while the wafer count stays in the hundreds, so a model fitted on all of them learns chance correlation and names nothing a process engineer can adjust.
- **Goal**: Place the selection methods on three axes — mechanism, selection unit, stability — so that a reader can name which axis position a candidate method takes and what its answer therefore does not cover.
- **Non-Goal**: Scoring a single feature, the subject of [Feature Importance](../../Feature-Importance/feature-importance.md), is not covered here.

## 2. Summary

A method that selects trace features has to be placed on three axes — mechanism, selection unit and stability — before it is clear what its answer does not cover. The mechanism axis fixes what the answer is a property of and what it costs, the selection unit axis decides whether the selected set can be acted on in the process, and the stability axis decides whether the set reappears when the wafers change. With wafers in the hundreds against columns in the thousands, the last two axes usually decide whether a selected set on trace data can be used.

## 3. Taxonomy and its Hierarchy

A method that selects trace features takes one value on each of three independent axes: mechanism, selection unit and stability. The mechanism axis extends the filter, wrapper and embedded division of the selection literature [[1](#ref-1)], [[2](#ref-2)] with preprocessing, post-hoc ranking and error-controlled selection. The other two axes come from the shape of trace data: one column names a sensor, a recipe step and a statistic at once, and the wafer count is far below the column count.

The three axes and the values on each are as in [Fig 1](#fig-1).

```text
Trace feature selection
|
+-- Mechanism ........ when the model is consulted
|     Preprocessing | Filter | Wrapper | Embedded | Post-hoc ranking | Error-controlled
|
+-- Selection unit ... what one selected item stands for
|     Column | Sensor | Step window | Statistic family
|
+-- Stability ........ how the selection is checked against a change of wafers
      Single fit | Resampled | Error-bounded
```

<a id="fig-1"></a>
Fig 1. Three independent axes of trace feature selection

A method takes one value on each axis, and its value on one axis does not fix its value on another. Group lasso with sensor groups, fitted once, is Embedded, Sensor and Single fit; the same group lasso rerun over subsamples changes only its stability value, to Resampled. The mechanism axis is treated in §4, the selection unit axis in §5 and the stability axis in §6.

The literature also classifies feature selection on four other axes, as in Table 1.

Table 1. Other classification axes in the literature

| Axis               | Values                                                      | Source        | Relation to Fig 1                                                                                                                                                          |
| :----------------: | :---------------------------------------------------------: | :-----------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| Supervision        | Supervised, unsupervised, semi-supervised                   | [[3](#ref-3)] | Follows from the mechanism axis: branch 0 unsupervised, branches 1 to 5 supervised, semi-supervised with no branch                                                         |
| Search strategy    | Exhaustive, sequential, randomized                          | [[4](#ref-4)] | Divides the wrapper branch: sequential as branches 2.1 and 2.2, randomized as 2.3, exhaustive with no branch                                                               |
| Output form        | Ranking, subset                                             | [[1](#ref-1)] | Follows from the branch: ranking from branches 1, 3.3 and 4, subset from branches 2, 3.1, 3.2 and 5                                                                        |
| Criterion property | Similarity, information theory, sparse learning, statistics | [[3](#ref-3)] | Divides branches by the score used: Relief by similarity, mutual information and mRMR by information theory, lasso by sparse learning, Pearson r and ANOVA F by statistics |

Each of the four either follows from the mechanism axis or subdivides one of its branches, so Fig 1 keeps only the three independent axes.

The mechanism axis divides further by when the model is consulted, running from the branch that consults neither the model nor the target to the branch reached only once a ranking exists, as in [Fig 2](#fig-2).

```text
Trace feature selection
|
+-- 0. Preprocessing (the target is not consulted)
|   +-- Constancy ................... zero-variance drop, dead-sensor drop
|   +-- Redundancy .................. correlation clustering, near-duplicate drop
|
+-- 1. Filter (scored before any model is fitted)
|   +-- 1.1 Univariate .............. Pearson r, Spearman rho, ANOVA F, mutual information
|   +-- 1.2 Multivariate ............ mRMR, correlation-filtered ranking
|   +-- 1.3 Neighbour contrast ...... Relief, ReliefF, RReliefF
|
+-- 2. Wrapper (a model is refitted for each candidate subset)
|   +-- 2.1 Backward ................ RFE, backward elimination
|   +-- 2.2 Forward ................. forward selection, stepwise
|   +-- 2.3 Stochastic search ....... genetic search, simulated annealing
|
+-- 3. Embedded (selection is a term of the training objective)
|   +-- 3.1 Column penalty .......... lasso, elastic net
|   +-- 3.2 Group penalty ........... group lasso, sparse group lasso
|   +-- 3.3 Split structure ......... MDI, split count, gain
|
+-- 4. Post-hoc ranking (the fit is finished, the model is then probed)
|   +-- 4.1 Removal ................. permutation importance, drop-column
|   +-- 4.2 Attribution ............. SHAP, LIME
|   +-- 4.3 Attention ............... attention weight, diagnostic only
|
+-- 5. Error-controlled (a selected set with a stated error bound)
    +-- 5.1 Subsampling ............. stability selection
    +-- 5.2 Knockoffs ............... fixed-X knockoffs, model-X knockoffs
```

<a id="fig-2"></a>
Fig 2. Hierarchy of the mechanism axis by when the model is consulted

Branch 0 runs first and consults no target, so it can leave a column that should have gone but cannot select a wrong one. Branches 1 to 3 differ by whether the model is consulted before the fit, around it, or inside it. Branch 4 probes a model that is already finished. Branch 5 alone answers with an error rate instead of a ranking, and it takes its input from any of branches 1 to 4. The branch boundaries that are easiest to cross by mistake are fixed in §4.1, §4.2, §4.4 and §4.5.

### 3.1 Comparison

The branch of Fig 2 that answers each question asked of a trace feature list is as in Table 2.

Table 2. Question and the branch that answers it

| Question                                                         | Branch                  | Note                                                         |
| :--------------------------------------------------------------: | :---------------------: | :----------------------------------------------------------: |
| Which columns are dead, constant, or duplicated                  | 0                       | Runs before any target is consulted                          |
| Which of thousands of columns are worth carrying further         | 1.1                     | Cut generously; a combination-only feature is invisible here |
| Which columns carry information the already-selected ones do not | 1.2                     | Redundancy as the criterion, beside relevance                |
| Which columns matter only in combination                         | 1.3                     | Neighbour contrast, seeing what a marginal score cannot      |
| Which subset this particular model actually needs                | 2.1                     | One refit per step, so run on the filtered shortlist         |
| Which sensors can be dropped from collection                     | 3.2, groups by sensor   | Sensor as the unit, or the answer is not actionable          |
| Which recipe step the target responds to                         | 3.2, groups by step     | Step as the unit                                             |
| Which sensor to keep but with fewer of its statistics            | 3.2, sparse group lasso | Between-group and within-group sparsity at once              |
| Which columns a fitted black-box model relies on                 | 4.1, on held-out rows   | Reliance of that fit, apart from information in the data     |
| Why this one wafer's prediction came out where it did            | 4.2                     | An explanation of one row, apart from selection              |
| Which features survive a change of wafers                        | 5.1                     | The selection frequency reported, in place of the subset     |
| Which features can be reported as real                           | 5.2                     | The only branch that states an error rate                    |

The number in the second column is the branch number of Fig 2, and a row that names a grouping takes its selection unit from §5.

## 4. Mechanism Axis

The mechanism axis places a branch by when the model is consulted, which fixes what the answer is a property of and what it costs, as in Table 3.

Table 3. Branches of the mechanism axis

| Branch              | When the model is consulted                   | The answer is a property of                 | Cost                                  |
| :-----------------: | :-------------------------------------------: | :-----------------------------------------: | :-----------------------------------: |
| 0. Preprocessing    | Never, and the target is not consulted either | The feature by itself                       | One pass                              |
| 1. Filter           | Before any fit                                | The data                                    | One pass per feature                  |
| 2. Wrapper          | Once per candidate subset                     | That model and that search                  | One refit per step                    |
| 3. Embedded         | During the fit                                | The fitted model                            | The fit itself                        |
| 4. Post-hoc ranking | After the fit is finished                     | That one fit                                | One probe per feature                 |
| 5. Error-controlled | After a ranking already exists                | The distribution, under a stated assumption | Many runs, or a knockoff construction |

Within each branch that selects, the members differ on one property, and the table of that subsection gives the property a column. Branch 0 has no table, because its members remove columns rather than choose among them.

### 4.1 Preprocessing

Branch 0 holds the methods that consult neither the model nor the target. Pearson correlation, Spearman correlation, ANOVA F and mutual information all score a feature against the target and can therefore be compared with one another and with any other supervised criterion. Variance thresholding scores a feature against itself, so placing it beside the correlation tests implies a comparison that cannot be made.

The Constancy group runs first. A trace collection carries columns that are constant by construction, such as a setpoint that never moves within a recipe, and columns from sensors that stopped reporting. Both must go before any supervised criterion runs: a zero-variance column has an undefined correlation, and a near-constant one has a correlation dominated by its quantization.

The Redundancy group keeps one column out of several that repeat one signal. A trace table produces such columns by construction: the mean, the median and the integral of one sensor within one recipe step follow the same physical quantity and rise and fall together from wafer to wafer. The columns are standardized, the absolute Pearson correlation |r| is computed for every pair, and the columns whose |r| exceeds a threshold such as 0.95 are grouped, either by hierarchical clustering on the distance 1 − |r| or by a greedy pass down the correlation matrix; one column is kept per group. On standardized columns the cosine similarity equals Pearson r, so a cosine criterion is the same method. The kept column is an original column rather than an average of the group, so it still names a sensor, a recipe step and a statistic (§5).

The Redundancy group does not consult the target either, so which member of a group survives is independent of the target. The survivor is fixed by a rule set before the target is seen, such as the column with the fewest missing values or the statistic an engineer already reads. The supervised counterpart, which weighs relevance to the target against redundancy with the selected set, is mRMR in branch 1.2 (§4.2).

### 4.2 Filter

The filter branch is cheap enough to run on the full trace feature list, and it is used as one pass that cuts thousands of columns to hundreds before anything expensive is attempted.

The members of the branch divide into univariate members, which score a feature alone, and multivariate members, which score it against the features already chosen. Pearson correlation, Spearman correlation, ANOVA F and mutual information score one column at a time against the target, and a sensor whose several summary statistics all track one physical quantity therefore contributes several near-equal scores. A univariate filter keeps all of those columns, and the column count falls without the redundancy falling with it. mRMR scores relevance to the target against redundancy with the already-selected set, and is the member that handles redundancy [[5](#ref-5)].

The third kind is neighbour contrast. Relief and its variants score a feature by contrasting each row with its nearest neighbours of the same and of the opposite class, so a feature that matters only in combination with another still receives a score, which no marginal criterion can produce [[6](#ref-6)]. The output is one number per feature, as with the univariate members, but the computation is not marginal, so it is kept as branch 1.3 apart from branch 1.1.

The members compare as in Table 4.

Table 4. Filter methods

| Method               | Criterion                                           | Sees redundancy | Sees interaction |
| :------------------: | :-------------------------------------------------: | :-------------: | :--------------: |
| Pearson correlation  | Linear association, signed                          | No              | No               |
| Spearman correlation | Monotone association, signed                        | No              | No               |
| ANOVA F              | Group separation for a categorical factor           | No              | No               |
| Mutual information   | Any dependence, at the cost of density estimation   | No              | No               |
| mRMR                 | Relevance minus redundancy against the selected set | Yes             | No               |
| Relief, ReliefF      | Neighbour contrast in feature space                 | Partly          | Yes              |

The threshold is set loosely. None of the univariate members can see a feature that matters only in combination with another, and a column dropped here does not return at a later step.

### 4.3 Wrapper

The wrapper branch scores a subset by the performance of a model refitted on it, so its answer is a property of that model and its cost is one refit per step [[7](#ref-7)].

The search direction and the cost of each member are as in Table 5.

Table 5. Wrapper methods

| Method               | Search direction                           | Refits per step                   | Failure on a wide table                                                             |
| :------------------: | :----------------------------------------: | :-------------------------------: | :---------------------------------------------------------------------------------: |
| RFE                  | Backward, dropping the lowest-ranked block | One per elimination step          | The initial fit is on all columns, where the ranking is least trustworthy           |
| Backward elimination | Backward, one feature at a time            | One per remaining feature         | Not defined when the column count exceeds the row count                             |
| Forward selection    | Forward, one feature at a time             | One per candidate per step        | An early wrong pick is never revisited                                              |
| Stepwise             | Forward with backward steps                | Both of the above                 | Search cost without a guarantee of the optimum                                      |
| Genetic search       | Stochastic over subsets                    | One per individual per generation | Cost grows with the population, and the result is not reproducible without the seed |

RFE was introduced on gene expression data, where the column count exceeds the row count by orders of magnitude, and it handles that case by eliminating a block of features per step instead of one [[8](#ref-8)]. A trace table has the same shape, so RFE is the member of this branch most often used on it. The cost of refitting at every step remains, so a wrapper runs on the hundreds of columns that survive the filter, not on the thousands that enter it.

### 4.4 Embedded

The embedded branch costs one fit: the selection is read from the objective being minimized, without an outer search over subsets.

The penalty and the selection unit of each member are as in Table 6.

Table 6. Embedded methods

| Method             | Penalty                          | Selection unit     | Effect of the unit                                                           |
| :----------------: | :------------------------------: | :----------------: | :--------------------------------------------------------------------------: |
| Lasso              | L1 on each coefficient           | One column         | The smallest set of columns                                                  |
| Elastic net        | L1 and L2 mixed                  | One column         | Correlated columns kept or dropped together rather than one picked at random |
| Group lasso        | L2 within a group, summed as L1  | One group          | A whole sensor or a whole step, in or out                                    |
| Sparse group lasso | Group and column penalties mixed | Group, then column | A sensor kept with only some of its statistics                               |
| MDI                | None; read off the splits        | One column         | Nothing beyond the fit already performed                                     |

Lasso is unstable on the correlated columns of a trace table, and the elastic net corrects it. When two columns are strongly correlated, lasso selects one of them and zeroes the other, and which one it selects is not determined by the data in any stable way; the L2 term added by the elastic net makes correlated columns enter and leave together [[9](#ref-9)]. The correction answers the stability problem of §6 only in part, since it groups columns by correlation and not by the physical object the engineer acts on.

Tree importance is two different methods under one name, and only MDI is embedded. MDI, the impurity decrease accumulated over the splits a feature was used in, is read off the tree structure the training procedure built, so it is embedded and sits in branch 3.3. Permutation importance shuffles a column of a finished model's input and measures the change in loss, so it is post-hoc and belongs in branch 4.1 (§4.5).

On a trace table the distinction changes the ranking. MDI is biased toward features with many distinct values, and inflates continuous columns over low-cardinality ones [[10](#ref-10)]. A trace feature table holds continuous summary statistics such as the mean and the slope beside low-cardinality counts such as the cycle count and the step index, and MDI ranks the first group above the second for reasons unrelated to the target.

### 4.5 Post-Hoc Ranking

The post-hoc ranking branch is a filter applied to a fitted model instead of to raw data, and it has the filter's defect: it gives a ranking and no cut-off.

What each member disturbs and whether it needs labels are as in Table 7.

Table 7. Post-hoc ranking methods

| Method                 | What is disturbed                          | Labels required | Scope of the result                                     |
| :--------------------: | :----------------------------------------: | :-------------: | :-----------------------------------------------------: |
| Permutation importance | One column, shuffled                       | Yes             | A reliance measure on held-out rows                     |
| Drop-column            | One column, removed and the model refitted | Yes             | The cost of one refit per column                        |
| SHAP                   | Feature subsets, marginalized              | No              | An attribution per prediction, aggregated for a ranking |
| LIME                   | A local neighbourhood, resampled           | No              | A local surrogate model, apart from a global statement  |
| Attention weight       | Nothing; the weight is read                | No              | Diagnostic only                                         |

Without a cut-off the branch is used to order a shortlist, not to produce a set. Permutation and drop-column additionally require a decision that the other members do not: whether the rows are the training rows or held-out rows. On held-out rows the value measures reliance that persists out of sample, and for a noise column with enough distinct values the two choices differ by the whole width of that column's training-data score.

SHAP, LIME and attention weights belong to post-hoc ranking, since embedded means that the selection is a term of the objective minimized during training [[1](#ref-1)]. Lasso selects because the L1 penalty sits inside the loss and drives coefficients to exactly zero [[11](#ref-11)]; SHAP [[12](#ref-12)] and LIME [[13](#ref-13)] have no such term, are computed after the fit is finished, treat the model as a black box, and leave the trained weights untouched. Their output is a ranking, and a cut-off still has to be chosen outside the method.

Attention weights face a second objection, apart from the branch they belong to. Attention over the inputs does not reliably identify the inputs a prediction depends on: alternative weight assignments yield the same prediction, and the weights correlate poorly with gradient-based measures [[14](#ref-14)]. The rebuttal narrows that result without removing it, showing that attention cannot be reassigned freely once the rest of the model is held fixed [[15](#ref-15)]. Attention weights are therefore usable for a diagnostic and insufficient for a selection criterion, and branch 4.3 of Fig 2 is marked diagnostic only on that ground.

### 4.6 Error-Controlled Selection

The error-controlled branch costs more and in return bounds how many of the selected features are there by chance, which no other branch does.

How each member is constructed and what it bounds are as in Table 8.

Table 8. Error-controlled methods

| Method              | Construction                                                        | What is bounded                    | Price                                                          |
| :-----------------: | :-----------------------------------------------------------------: | :--------------------------------: | :------------------------------------------------------------: |
| Stability selection | The base selector rerun on subsamples                               | Expected number of false positives | One run of the base selector per subsample                     |
| Fixed-X knockoffs   | Synthetic columns built from the design matrix                      | False discovery rate               | Requires more rows than columns                                |
| Model-X knockoffs   | Synthetic columns drawn from the joint distribution of the features | False discovery rate               | Requires that joint distribution to be known or well estimated |

Of all the branches, this one alone answers "which features can be reported as real". Model-X knockoffs construct a synthetic copy of each feature that carries its correlation structure but is independent of the target given the rest, then keep the features whose score exceeds that of their own copies by a threshold set to hold the false discovery rate below a chosen level [[16](#ref-16)]. The price is the joint distribution of the features, which for a trace table is estimable only under an assumed structure, and the guarantee is no stronger than that assumption.

## 5. Selection Unit Axis

The unit a method selects at decides whether its answer can be acted on, and on a trace table the actionable unit is the sensor or the recipe step rather than the column. A trace feature table is built by applying a set of summary statistics to each sensor over each recipe step, so one column names three things at once — which sensor, which step, which statistic — and only the sensor and the step correspond to something a line can change. A column-unit selection can keep the maximum of the forward RF power, drop its mean, and keep the maximum of the reflected RF power; that set is a valid model input and an instruction no one can carry out.

What one selected item is at each unit, and the action it leads to, are as in Table 9.

Table 9. Selection units

| Unit             | One selected item is                      | The action it maps to                    | Method that selects at it             |
| :--------------: | :---------------------------------------: | :--------------------------------------: | :-----------------------------------: |
| Column           | One statistic of one sensor over one step | None by itself                           | Lasso, elastic net, univariate filter |
| Sensor           | Every statistic of one sensor             | Keep or drop that sensor from collection | Group lasso with sensor groups        |
| Step window      | Every statistic over one recipe step      | Change or hold that step                 | Group lasso with step groups          |
| Statistic family | One statistic across every sensor         | Change what is reduced from the trace    | Group lasso with statistic groups     |

The group is declared before the fit and cannot be recovered after it, so the selection unit is an axis and not a post-processing choice. Group lasso applies an L2 norm to each group's coefficient vector and sums those norms as an L1, so the whole group is driven to zero together or kept together [[17](#ref-17)]. Sparse group lasso adds a column penalty on top, which is the form that answers "use this sensor but only two of its statistics" — a decision that is actionable at the collection level and economical at the model level [[18](#ref-18)].

The column unit answers a different question — which numbers a model needs — and that question is the right one when the collection set is already fixed and only the model is being trimmed. The error is to answer at the column unit and report the answer as a sensor list.

## 6. Stability Axis

On trace data the value that matters on this axis is Resampled: a selection from one fit changes when the wafers change. With wafers in the hundreds against features in the thousands, physically paired sensors — forward and reflected RF power, adjacent thermocouples, a flow setpoint and its readback — correlate so strongly that a penalized fit's choice between two columns is settled by noise rather than by signal. Refitting on a different subset of the wafers therefore changes which member of each pair is selected.

The procedure and the reported result at each value of the axis are as in Table 10.

Table 10. Stability positions

| Position      | Procedure                                        | What is reported                                  | Cost                                |
| :-----------: | :----------------------------------------------: | :-----------------------------------------------: | :---------------------------------: |
| Single fit    | The selector run once on all rows                | One subset, with no spread around it              | One run                             |
| Resampled     | The selector run on each of B subsamples         | A selection frequency per feature                 | B runs                              |
| Error-bounded | Resampled, with the threshold set from the bound | A subset with a bound on expected false positives | B runs, plus the bound's conditions |

Stability selection is the Resampled value made into a method: the base selector is rerun over subsamples, the selection frequency of each feature is counted, and the features above a threshold are kept, with a finite-sample bound on the expected number of falsely selected features under stated conditions [[19](#ref-19)]. The bound makes it branch 5.1 of Fig 2, and since the resampling wraps any selector from branches 1 to 4 without changing it, it is also a value of the stability axis.

The frequency curve is itself the finding, and reading only the thresholded set loses it. A feature selected in nearly every subsample and a feature selected in just over half of them are two different results, reported identically by a single fit. Two conditions have to hold for the frequency to mean anything: the subsamples must respect the lot, since wafers processed together are not independent draws, and they must respect the processing order, since a split that mixes early and late wafers hides the drift that the selection is supposed to survive.

## 7. Failure Modes

Each defect below belongs to a method rather than to an implementation, and each is reached by a procedure that looks correct at every step.

- **Univariate filter over collinear sensors.** Every copy of one physical signal kept, the column count reduced without the redundancy reduced.
- **Wrapper score computed outside the resampling fold.** Selection bias, with the reported cross-validation error optimistic by a wide margin [[20](#ref-20)].
- **Random split over lot-grouped wafers.** Wafers of one lot on both sides of the split, so the held-out rows are not held out.
- **A drift proxy selected.** Chamber age or consumable life tracking the target's own drift, predictive in the fit and inert under intervention.
- **MDI read as importance.** Continuous summary statistics ranked above low-cardinality counts for their cardinality [[10](#ref-10)].
- **A zero coefficient read as irrelevance.** An exactly duplicated pair scored at zero on both of its members.
- **Permutation over correlated columns.** The model evaluated on rows the process cannot produce, with its behaviour there entering the score [[21](#ref-21)].
- **A set selected on one chamber.** Local chamber state instead of process physics, not transferring to the next chamber.

## 8. Further Work

- **Selection at the trace-segment unit.** A unit finer than the recipe step and coarser than the column, choosing the time window within a step that carries the response. It is reachable now because the feature-based framing of virtual metrology has moved from whole-trace summaries to segment features with a stated physical meaning [[22](#ref-22)]. It needs traces aligned to a common time base within a step, which step boundaries alone do not supply when the step length varies by wafer.
- **Group knockoffs for the sensor unit.** The error-controlled branch applied at the sensor unit of §5, so that a reported sensor list carries a false discovery rate instead of a ranking. It is reachable now because model-X knockoffs moved the requirement from more rows than columns to a known joint distribution of the features [[16](#ref-16)], which a wide trace table can meet in principle. It needs a validated covariance model for the sensor set, since the guarantee is no stronger than the distribution assumed for the synthetic copies.

## References

<a id="ref-1"></a>
[1] Guyon, I. and Elisseeff, A. (2003). [An Introduction to Variable and Feature Selection](https://www.jmlr.org/papers/v3/guyon03a.html). *Journal of Machine Learning Research*, 3, 1157–1182.<br>
<a id="ref-2"></a>
[2] Chandrashekar, G. and Sahin, F. (2014). [A survey on feature selection methods](https://doi.org/10.1016/j.compeleceng.2013.11.024). *Computers & Electrical Engineering*, 40(1), 16–28.<br>
<a id="ref-3"></a>
[3] Li, J., Cheng, K., Wang, S., Morstatter, F., Trevino, R. P., Tang, J. and Liu, H. (2017). [Feature Selection: A Data Perspective](https://doi.org/10.1145/3136625). *ACM Computing Surveys*, 50(6), 94.<br>
<a id="ref-4"></a>
[4] Liu, H. and Yu, L. (2005). [Toward Integrating Feature Selection Algorithms for Classification and Clustering](https://doi.org/10.1109/TKDE.2005.66). *IEEE Transactions on Knowledge and Data Engineering*, 17(4), 491–502.<br>
<a id="ref-5"></a>
[5] Peng, H., Long, F. and Ding, C. (2005). [Feature Selection Based on Mutual Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy](https://doi.org/10.1109/TPAMI.2005.159). *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 27(8), 1226–1238.<br>
<a id="ref-6"></a>
[6] Urbanowicz, R. J., Meeker, M., La Cava, W., Olson, R. S. and Moore, J. H. (2018). [Relief-based feature selection: Introduction and review](https://doi.org/10.1016/j.jbi.2018.07.014). *Journal of Biomedical Informatics*, 85, 189–203.<br>
<a id="ref-7"></a>
[7] Kohavi, R. and John, G. H. (1997). [Wrappers for feature subset selection](<https://doi.org/10.1016/S0004-3702(97)00043-X>). *Artificial Intelligence*, 97(1–2), 273–324.<br>
<a id="ref-8"></a>
[8] Guyon, I., Weston, J., Barnhill, S. and Vapnik, V. (2002). [Gene Selection for Cancer Classification using Support Vector Machines](https://doi.org/10.1023/A:1012487302797). *Machine Learning*, 46, 389–422.<br>
<a id="ref-9"></a>
[9] Zou, H. and Hastie, T. (2005). [Regularization and variable selection via the elastic net](https://doi.org/10.1111/j.1467-9868.2005.00503.x). *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.<br>
<a id="ref-10"></a>
[10] Strobl, C., Boulesteix, A.-L., Zeileis, A. and Hothorn, T. (2007). [Bias in random forest variable importance measures: illustrations, sources and a solution](https://doi.org/10.1186/1471-2105-8-25). *BMC Bioinformatics*, 8, 25.<br>
<a id="ref-11"></a>
[11] Tibshirani, R. (1996). [Regression Shrinkage and Selection via the Lasso](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x). *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.<br>
<a id="ref-12"></a>
[12] Lundberg, S. M. and Lee, S.-I. (2017). [A Unified Approach to Interpreting Model Predictions](https://papers.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html). *Advances in Neural Information Processing Systems*, 30, 4765–4774.<br>
<a id="ref-13"></a>
[13] Ribeiro, M. T., Singh, S. and Guestrin, C. (2016). ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://doi.org/10.1145/2939672.2939778). *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144.<br>
<a id="ref-14"></a>
[14] Jain, S. and Wallace, B. C. (2019). [Attention is not Explanation](https://doi.org/10.18653/v1/N19-1357). *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 3543–3556.<br>
<a id="ref-15"></a>
[15] Wiegreffe, S. and Pinter, Y. (2019). [Attention is not not Explanation](https://doi.org/10.18653/v1/D19-1002). *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing*, 11–20.<br>
<a id="ref-16"></a>
[16] Candès, E., Fan, Y., Janson, L. and Lv, J. (2018). [Panning for gold: 'model-X' knockoffs for high dimensional controlled variable selection](https://doi.org/10.1111/rssb.12265). *Journal of the Royal Statistical Society: Series B*, 80(3), 551–577.<br>
<a id="ref-17"></a>
[17] Yuan, M. and Lin, Y. (2006). [Model selection and estimation in regression with grouped variables](https://doi.org/10.1111/j.1467-9868.2005.00532.x). *Journal of the Royal Statistical Society: Series B*, 68(1), 49–67.<br>
<a id="ref-18"></a>
[18] Simon, N., Friedman, J., Hastie, T. and Tibshirani, R. (2013). [A Sparse-Group Lasso](https://doi.org/10.1080/10618600.2012.681250). *Journal of Computational and Graphical Statistics*, 22(2), 231–245.<br>
<a id="ref-19"></a>
[19] Meinshausen, N. and Bühlmann, P. (2010). [Stability selection](https://doi.org/10.1111/j.1467-9868.2010.00740.x). *Journal of the Royal Statistical Society: Series B*, 72(4), 417–473.<br>
<a id="ref-20"></a>
[20] Ambroise, C. and McLachlan, G. J. (2002). [Selection bias in gene extraction on the basis of microarray gene-expression data](https://doi.org/10.1073/pnas.102102699). *Proceedings of the National Academy of Sciences*, 99(10), 6562–6566.<br>
<a id="ref-21"></a>
[21] Hooker, G., Mentch, L. and Zhou, S. (2021). [Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance](https://doi.org/10.1007/s11222-021-10057-z). *Statistics and Computing*, 31, 82.<br>
<a id="ref-22"></a>
[22] Suthar, K., Shah, D., Wang, J. and He, Q. P. (2019). [Next-generation virtual metrology for semiconductor manufacturing: A feature-based framework](https://doi.org/10.1016/j.compchemeng.2019.05.016). *Computers & Chemical Engineering*, 127, 140–149.

---

## Appendix A. Terminology

- **base selector**: The selection method that stability selection reruns on each subsample.
- **embedded method**: A selection method in which the selection is a term of the objective minimized during training.
- **filter method**: A selection method that scores features before any model is fitted.
- **group lasso**: A penalty that applies an L2 norm within each declared group and sums those norms as an L1, so that a group is kept or dropped whole.
- **held-out row**: A row kept out of fitting and used only to evaluate the fitted model.
- **knockoff**: A synthetic copy of a feature that carries its correlation structure but is independent of the target given the remaining features.
- **MDI**: Mean decrease in impurity, the impurity reduction accumulated over the splits a feature was used in.
- **mRMR**: Minimum redundancy maximum relevance, a filter that scores relevance to the target against redundancy with the already-selected set.
- **post-hoc ranking**: A ranking computed after training is finished, with the fitted model treated as a black box.
- **recipe step**: One named phase of a process recipe, over which a trace is summarized separately.
- **RFE**: Recursive feature elimination, a wrapper that refits the model and drops the lowest-ranked features at each step.
- **selection unit**: The object that one selected item stands for — a column, a sensor, a recipe step, or a statistic family.
- **stability selection**: A procedure that reruns a base selector over subsamples and keeps the features whose selection frequency exceeds a threshold.
- **subsample**: A subset of the rows drawn without replacement, on which a selector is rerun.
- **trace**: The time series a sensor records over one wafer's pass through one recipe.
- **virtual metrology**: The prediction of a metrology measurement from equipment data, in place of measuring the wafer.
- **wrapper method**: A selection method that refits a model for each candidate subset and scores the subset by that model's performance.
