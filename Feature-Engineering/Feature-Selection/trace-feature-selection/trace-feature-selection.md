# Trace Feature Selection
Rev. 18 | Created: 2026-09-10 | Updated: 2026-10-30 01:29 UTC

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Comparison](#31-comparison)
- [4. Evaluation Criterion Axis](#4-evaluation-criterion-axis)
  - [4.1 Preprocessing](#41-preprocessing)
  - [4.2 Filter](#42-filter)
  - [4.3 Wrapper](#43-wrapper)
  - [4.4 Embedded](#44-embedded)
  - [4.5 Post-Hoc Ranking](#45-post-hoc-ranking)
  - [4.6 Error-Controlled Selection](#46-error-controlled-selection)
- [5. Label Availability Axis](#5-label-availability-axis)
- [6. Search Strategy Axis](#6-search-strategy-axis)
- [7. Selection Unit Axis](#7-selection-unit-axis)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

- **Problem Statement**: In a high-dimensional setting where thousands of trace sensor columns are produced for a few hundred wafers, training on all of them learns spurious correlation and does not connect to a process engineer's actual control actions (sensor or step adjustment).
- **Goal**: Systematically diagnose the structural limits and the field applicability of a given feature selection method through the four axes (Evaluation Criterion, Label Availability, Search Strategy, Selection Unit).
- **Non-Goal**: The structure and principles of featurization methods are not covered.

## 2. Summary

A trace feature selection method is defined by a combination of the following four axes:
1. **Evaluation Criterion**: how the model is used and the cost of scoring (Preprocessing ~ Error-Controlled)
2. **Label Availability**: whether target labels are used (Supervised / Unsupervised)
3. **Search Strategy**: how candidate subsets are searched (Exhaustive / Sequential / Randomized)
4. **Selection Unit**: the physical unit of action of a selected item (Column / Sensor / Step Window / Statistic Family)

## 3. Taxonomy and its Hierarchy

A method that selects trace features takes one value on each of four axes: evaluation criterion, label availability, search strategy and selection unit. The evaluation criterion axis, named after the evaluation criteria of Liu and Yu [[1](#ref-1)], extends the filter, wrapper and embedded division of the selection literature [[2](#ref-2)], [[3](#ref-3)] with preprocessing, post-hoc ranking and error-controlled selection. The selection unit axis comes from the shape of trace data: one column names a sensor, a recipe step and a statistic at once.

The four axes and the values on each are as in [Fig 1](#fig-1).

```text
Trace feature selection
|
+-- Evaluation criterion ... dependence on the learning algorithm
|     Preprocessing | Filter | Wrapper | Embedded | Post-hoc ranking | Error-controlled
|
+-- Label availability ..... whether target labels are used
|     Supervised | Unsupervised | Semi-supervised
|
+-- Search strategy ........ how candidate subsets are generated
|     Exhaustive | Sequential | Randomized
|
+-- Selection unit ......... what one selected item stands for
      Column | Sensor | Step window | Statistic family
```

<a id="fig-1"></a>
Fig 1. Feature selection taxonomy for trace data

Evaluation criterion and search strategy are two of the dimensions of Liu and Yu [[1](#ref-1)], label availability is the division of Li et al. [[4](#ref-4)], and selection unit is added by this document for trace data; no single source combines the four. A method takes one value on each axis. Group lasso with sensor groups is Embedded, Supervised, without a subset search, and Sensor; grouped by recipe step instead, it changes only its selection unit, to Step window. The evaluation criterion axis is treated in §4, the label availability axis in §5, the search strategy axis in §6 and the selection unit axis in §7.

The evaluation criterion axis divides further by how the selection uses the learning algorithm, running from the branch that uses neither the learning algorithm nor the target to the branch reached only once a ranking exists, as in [Fig 2](#fig-2), where the right-hand columns give the label availability, search strategy and selection unit of each line.

```text
Trace feature selection                                                                     Label availability | Search strategy | Selection unit
|
+-- 0. Preprocessing (the target is not consulted)
|   +-- Constancy ................... zero-variance drop, dead-sensor drop                  Unsupervised       | -               | Column, sensor
|   +-- Redundancy .................. correlation clustering, near-duplicate drop           Unsupervised       | -               | Column
|
+-- 1. Filter (scored before any model is fitted)
|   +-- 1.1 Univariate .............. Pearson r, Spearman rho, ANOVA F, mutual information  Supervised         | -               | Column
|   +-- 1.2 Multivariate ............ mRMR, correlation-filtered ranking                    Supervised         | Sequential      | Column
|   +-- 1.3 Neighbour contrast ...... Relief, ReliefF, RReliefF                             Supervised         | -               | Column
|
+-- 2. Wrapper (a model is refitted for each candidate subset)
|   +-- 2.1 Backward ................ RFE, backward elimination                             Supervised         | Sequential      | Column
|   +-- 2.2 Forward ................. forward selection, stepwise                           Supervised         | Sequential      | Column
|   +-- 2.3 Stochastic search ....... genetic search, simulated annealing                   Supervised         | Randomized      | Column
|
+-- 3. Embedded (selection is a term of the training objective)
|   +-- 3.1 Column penalty .......... lasso, elastic net                                    Supervised         | -               | Column
|   +-- 3.2 Group penalty ........... group lasso, sparse group lasso                       Supervised         | -               | Sensor, step window, statistic family
|   +-- 3.3 Split structure ......... MDI, split count, gain                                Supervised         | -               | Column
|
+-- 4. Post-hoc ranking (the fit is finished, the model is then probed)
|   +-- 4.1 Removal ................. permutation importance, drop-column                   Supervised         | -               | Column
|   +-- 4.2 Attribution ............. SHAP, LIME                                            Supervised         | -               | Column
|   +-- 4.3 Attention ............... attention weight, diagnostic only                     Supervised         | -               | Column
|
+-- 5. Error-controlled (a selected set with a stated error bound)
    +-- 5.1 Subsampling ............. stability selection                                   Supervised         | -               | Unit of the base selector
    +-- 5.2 Knockoffs ............... fixed-X knockoffs, model-X knockoffs                  Supervised         | -               | Column
```

<a id="fig-2"></a>
Fig 2. Hierarchy of the evaluation criterion axis by use of the learning algorithm

Branch 0 runs first and consults no target, so it can leave a column that should have gone but cannot select a wrong one. Branches 1 to 3 differ by whether the model is consulted before the fit, around it, or inside it. Branch 4 probes a model that is already finished. Branch 5 alone answers with an error rate instead of a ranking, and it takes its input from any of branches 1 to 4. The branch boundaries that are easiest to cross by mistake are fixed in §4.1, §4.2, §4.4 and §4.5. In the right-hand columns, - marks a line whose members do not search over subsets; no line is semi-supervised or exhaustive, and branch 5.1 takes its selection unit from its base selector.

### 3.1 Comparison

The branch of Fig 2 for each use case is as in Table 1.

Table 1. Use case and the branch that answers it

| Use case                                          | Branch                  | Note                                      |
| :-----------------------------------------------: | :---------------------: | :---------------------------------------: |
| Remove dead, constant or duplicated columns       | 0                       | Runs before the target is used            |
| Cut thousands of columns to a shortlist           | 1.1                     | Loose cut; combination-only features lost |
| Keep columns that add information to those chosen | 1.2                     | Relevance scored against redundancy       |
| Find columns that matter only in combination      | 1.3                     | Neighbour contrast catches interactions   |
| Find the subset one model needs                   | 2.1                     | Runs on the filtered shortlist            |
| Drop sensors from collection                      | 3.2, groups by sensor   | Sensor as the unit                        |
| Find the recipe step the target responds to       | 3.2, groups by step     | Step as the unit                          |
| Keep a sensor with fewer of its statistics        | 3.2, sparse group lasso | Sparse between and within groups          |
| Rank the columns a fitted model relies on         | 4.1, on held-out rows   | Reliance of that fit, apart from the data |
| Explain one wafer's prediction                    | 4.2                     | Explains one row; selects nothing         |
| Keep features that survive a change of wafers     | 5.1                     | Selection frequency reported              |
| Report features with an error rate                | 5.2                     | False discovery rate bounded              |

The number in the second column is the branch number of Fig 2, and a row that names a grouping takes its selection unit from §7.

## 4. Evaluation Criterion Axis

The evaluation criterion axis places a branch by how it uses the learning algorithm, which fixes what the answer is a property of and what it costs, as in Table 2.

Table 2. Branches of the evaluation criterion axis

| Branch              | Use of the learning algorithm                 | The answer is a property of                 | Cost                                  |
| :-----------------: | :-------------------------------------------: | :-----------------------------------------: | :-----------------------------------: |
| 0. Preprocessing    | Never, and the target is not consulted either | The feature by itself                       | One pass                              |
| 1. Filter           | Before any fit                                | The data                                    | One pass per feature                  |
| 2. Wrapper          | Once per candidate subset                     | That model and that search                  | One refit per step                    |
| 3. Embedded         | During the fit                                | The fitted model                            | The fit itself                        |
| 4. Post-hoc ranking | After the fit is finished                     | That one fit                                | One probe per feature                 |
| 5. Error-controlled | After a ranking already exists                | The distribution, under a stated assumption | Many runs, or a knockoff construction |

Within each branch that selects, the members differ on one property, and the table of that subsection gives the property a column. Branch 0 has no table, because its members remove columns rather than choose among them.

### 4.1 Preprocessing

**Key features.** Branch 0 removes columns using neither the learning algorithm nor the target. Constancy removes zero-variance and near-constant columns. Redundancy keeps one column out of several whose absolute Pearson correlation |r| exceeds a threshold such as 0.95, grouping them by hierarchical clustering on the distance 1 − |r| or by a greedy pass down the correlation matrix. On standardized columns the cosine similarity equals Pearson r, so a cosine criterion is the same method. Variance thresholding scores a feature against itself, so it cannot be compared with the supervised scores of branch 1.

**On trace data.** A trace collection carries columns constant by construction, such as a setpoint that never moves within a recipe, and columns from sensors that stopped reporting; both go first, since a zero-variance column has no defined correlation and a near-constant one has a correlation dominated by its quantization. The mean, the median and the integral of one sensor within one recipe step rise and fall together, so Redundancy reduces them to one original column that still names a sensor, a recipe step and a statistic (§7). The kept column is fixed by a rule set before the target is seen, such as the fewest missing values; the supervised counterpart is mRMR in branch 1.2 (§4.2).

### 4.2 Filter

**Key features.** The filter branch scores features before any model is fitted, in one pass cheap enough for the full trace feature list. Pearson correlation, Spearman correlation, ANOVA F and mutual information score a column against the target. mRMR scores a column against the target and against the columns already chosen [[5](#ref-5)]. Relief and its variants score a column against the nearest neighbours of each row in the same and the opposite class, so a feature that matters only in combination still receives a score [[6](#ref-6)].

The members compare as in Table 3.

Table 3. Filter methods

| Method               | Criterion                                           | Sees redundancy | Sees interaction |
| :------------------: | :-------------------------------------------------: | :-------------: | :--------------: |
| Pearson correlation  | Linear association, signed                          | No              | No               |
| Spearman correlation | Monotone association, signed                        | No              | No               |
| ANOVA F              | Group separation for a categorical factor           | No              | No               |
| Mutual information   | Any dependence, at the cost of density estimation   | No              | No               |
| mRMR                 | Relevance minus redundancy against the selected set | Yes             | No               |
| Relief, ReliefF      | Neighbour contrast in feature space                 | Partly          | Yes              |

**On trace data.** A sensor whose several summary statistics track one physical quantity gives several near-equal univariate scores, so a univariate filter reduces the column count without reducing the redundancy. The threshold is set loosely: no univariate member sees a feature that matters only in combination, and a column dropped here does not return at a later step.

### 4.3 Wrapper

**Key features.** The wrapper branch scores a subset by the performance of a model refitted on it, so its answer is a property of that model and its cost is one refit per step [[7](#ref-7)].

The search direction and the cost of each member are as in Table 4.

Table 4. Wrapper methods

| Method               | Search direction                           | Refits per step                   | Failure on a wide table                                                             |
| :------------------: | :----------------------------------------: | :-------------------------------: | :---------------------------------------------------------------------------------: |
| RFE                  | Backward, dropping the lowest-ranked block | One per elimination step          | The initial fit is on all columns, where the ranking is least trustworthy           |
| Backward elimination | Backward, one feature at a time            | One per remaining feature         | Not defined when the column count exceeds the row count                             |
| Forward selection    | Forward, one feature at a time             | One per candidate per step        | An early wrong pick is never revisited                                              |
| Stepwise             | Forward with backward steps                | Both of the above                 | Search cost without a guarantee of the optimum                                      |
| Genetic search       | Stochastic over subsets                    | One per individual per generation | Cost grows with the population, and the result is not reproducible without the seed |

**On trace data.** RFE was introduced on gene expression data, whose column count exceeds the row count by orders of magnitude, and it eliminates a block of features per step [[8](#ref-8)]. A trace table has the same shape, so RFE is the member of this branch most often used on it. The refit cost remains, so a wrapper runs on the hundreds of columns that pass the filter, not on the thousands that enter it.

### 4.4 Embedded

**Key features.** The embedded branch costs one fit: the selection is read from the objective being minimized, without an outer search over subsets.

The penalty and the selection unit of each member are as in Table 5.

Table 5. Embedded methods

| Method                             | Penalty                          | Selection unit     | Effect of the unit                                                           |
| :--------------------------------: | :------------------------------: | :----------------: | :--------------------------------------------------------------------------: |
| Lasso                              | L1 on each coefficient           | One column         | The smallest set of columns                                                  |
| Elastic net                        | L1 and L2 mixed                  | One column         | Correlated columns kept or dropped together rather than one picked at random |
| Group lasso [[9](#ref-9)]          | L2 within a group, summed as L1  | One group          | A whole sensor or a whole step, in or out                                    |
| Sparse group lasso [[10](#ref-10)] | Group and column penalties mixed | Group, then column | A sensor kept with only some of its statistics                               |
| MDI                                | None; read off the splits        | One column         | Nothing beyond the fit already performed                                     |

**Lasso and elastic net.** Lasso keeps one of two strongly correlated columns and sets the other to zero, and the data do not fix in any stable way which one it keeps [[11](#ref-11)]. Elastic net adds an L2 term and keeps or drops two strongly correlated columns together [[12](#ref-12)].

**MDI and permutation importance.** MDI reads the impurity decrease from the tree structure the training built, so it is embedded and sits in branch 3.3. Permutation importance reads the loss change after shuffling one column of a finished model's input, so it is post-hoc and sits in branch 4.1 (§4.5).

**On trace data.** Elastic net groups columns by correlation and not by the sensor or step the engineer acts on, so it addresses the instability across wafers of §4.6 only in part. MDI is biased toward features with many distinct values [[13](#ref-13)]; a trace feature table holds continuous summary statistics such as the mean and the slope beside low-cardinality counts such as the cycle count and the step index, and MDI ranks the first above the second for reasons unrelated to the target.

### 4.5 Post-Hoc Ranking

**Key features.** The post-hoc ranking branch is a filter applied to a fitted model, and it gives a ranking with no cut-off. SHAP, LIME and attention weights belong here: embedded means that the selection is a term of the training objective [[2](#ref-2)], while SHAP [[14](#ref-14)] and LIME [[15](#ref-15)] are computed after the fit, treat the model as a black box and leave the trained weights untouched.

What each member disturbs and whether it needs labels are as in Table 6.

Table 6. Post-hoc ranking methods

| Method                 | What is disturbed                          | Labels required | Scope of the result                                     |
| :--------------------: | :----------------------------------------: | :-------------: | :-----------------------------------------------------: |
| Permutation importance | One column, shuffled                       | Yes             | A reliance measure on held-out rows                     |
| Drop-column            | One column, removed and the model refitted | Yes             | The cost of one refit per column                        |
| SHAP                   | Feature subsets, marginalized              | No              | An attribution per prediction, aggregated for a ranking |
| LIME                   | A local neighbourhood, resampled           | No              | A local surrogate model, apart from a global statement  |
| Attention weight       | Nothing; the weight is read                | No              | Diagnostic only                                         |

**On trace data.** The branch orders a shortlist instead of producing a set. Permutation and drop-column also require a choice between training rows and held-out rows, and on held-out rows the value measures reliance that persists out of sample. Attention weights serve as a diagnostic only: other weight assignments yield the same prediction and the weights correlate poorly with gradient-based measures [[16](#ref-16)], and the rebuttal narrows that result without removing it [[17](#ref-17)].

### 4.6 Error-Controlled Selection

**Key features.** The error-controlled branch costs more and in return bounds how many of the selected features are there by chance, which no other branch does. Model-X knockoffs build a synthetic copy of each feature that keeps its correlation structure but is independent of the target given the rest, and keep the features whose score exceeds that of their own copies by a threshold that holds the false discovery rate below a chosen level [[18](#ref-18)]. Stability selection reruns a base selector over subsamples, counts how often each feature is selected, and keeps the features above a threshold, with a finite-sample bound on the expected number of false positives [[19](#ref-19)].

How each member is constructed and what it bounds are as in Table 7.

Table 7. Error-controlled methods

| Method              | Construction                                                        | What is bounded                    | Price                                                          |
| :-----------------: | :-----------------------------------------------------------------: | :--------------------------------: | :------------------------------------------------------------: |
| Stability selection | The base selector rerun on subsamples                               | Expected number of false positives | One run of the base selector per subsample                     |
| Fixed-X knockoffs   | Synthetic columns built from the design matrix                      | False discovery rate               | Requires more rows than columns                                |
| Model-X knockoffs   | Synthetic columns drawn from the joint distribution of the features | False discovery rate               | Requires that joint distribution to be known or well estimated |

**On trace data.** Physically paired sensors — forward and reflected RF power, adjacent thermocouples, a flow setpoint and its readback — correlate so strongly that one penalized fit picks between them by noise, and the selected set changes when the wafers change. The selection frequency of stability selection is the result: a feature selected in nearly every subsample and one selected in just over half are different findings that a single fit reports alike. The frequency has meaning only when the subsamples respect the lot, since wafers processed together are not independent draws, and the processing order, since mixing early and late wafers hides drift. Model-X knockoffs need the joint distribution of the features, which for a trace table is estimable only under an assumed structure, and the guarantee is no stronger than that assumption.

## 5. Label Availability Axis

The label availability axis divides methods by whether the selection uses target labels, and its values are supervised, unsupervised and semi-supervised [[4](#ref-4)]. In Fig 2 only the two groups of branch 0 are unsupervised, branches 1 to 5 are all supervised, and no line is semi-supervised.

An unsupervised selection removes columns without reference to the target, and a supervised selection keeps columns by their relation to the target. Constancy and Redundancy therefore run before any supervised branch, and the column they keep from a redundant group is fixed by a rule set before the target is seen (§4.1). Semi-supervised selection uses unlabeled rows together with the labeled ones when labels exist for only part of the rows [[4](#ref-4)].

## 6. Search Strategy Axis

The search strategy axis divides methods by how candidate subsets are generated, and its values are exhaustive, sequential and randomized [[1](#ref-1)]. The axis has a value only for a method that searches over subsets, which in Fig 2 is branch 1.2 and the three lines of branch 2; the other lines are marked -.

Sequential search adds or removes features one at a time or one block at a time: mRMR in branch 1.2 adds features incrementally [[5](#ref-5)], and RFE, backward elimination, forward selection and stepwise in branches 2.1 and 2.2 do the same within a wrapper. Randomized search moves stochastically over subsets, as genetic search and simulated annealing do in branch 2.3. Exhaustive search evaluates every subset, whose number doubles with each added column, so on a trace table of thousands of columns no line of Fig 2 takes that value. The search direction and the refit cost of each wrapper are as in Table 4.

## 7. Selection Unit Axis

Column-unit selection gives a valid model input but no action a line can take. One trace column names a sensor, a recipe step and a statistic at once, and a line can change only the sensor and the step. A column-unit set can keep the maximum of the forward RF power, drop its mean, and keep the maximum of the reflected RF power, and no one can carry that set out.

The line action each unit maps to is as in Table 8.

Table 8. Selection units

| Unit             | Line action                              | Selected item                             | Method                                |
| :--------------: | :--------------------------------------: | :---------------------------------------: | :-----------------------------------: |
| Column           | None                                     | One statistic of one sensor over one step | Lasso, elastic net, univariate filter |
| Sensor           | Keep or drop that sensor from collection | Every statistic of one sensor             | Group lasso with sensor groups        |
| Step window      | Change or hold that step                 | Every statistic over one recipe step      | Group lasso with step groups          |
| Statistic family | Change what is reduced from the trace    | One statistic across every sensor         | Group lasso with statistic groups     |

The group is declared before the fit and cannot be recovered after it, so the selection unit is an axis and not a post-processing choice. The column unit is right when the collection set is fixed and only the model is trimmed; the error is to report a column-unit answer as a sensor list.

## References

<a id="ref-1"></a>
[1] Liu, H. and Yu, L. (2005). [Toward Integrating Feature Selection Algorithms for Classification and Clustering](https://doi.org/10.1109/TKDE.2005.66). *IEEE Transactions on Knowledge and Data Engineering*, 17(4), 491–502.<br>
<a id="ref-2"></a>
[2] Guyon, I. and Elisseeff, A. (2003). [An Introduction to Variable and Feature Selection](https://www.jmlr.org/papers/v3/guyon03a.html). *Journal of Machine Learning Research*, 3, 1157–1182.<br>
<a id="ref-3"></a>
[3] Chandrashekar, G. and Sahin, F. (2014). [A survey on feature selection methods](https://doi.org/10.1016/j.compeleceng.2013.11.024). *Computers & Electrical Engineering*, 40(1), 16–28.<br>
<a id="ref-4"></a>
[4] Li, J., Cheng, K., Wang, S., Morstatter, F., Trevino, R. P., Tang, J. and Liu, H. (2017). [Feature Selection: A Data Perspective](https://doi.org/10.1145/3136625). *ACM Computing Surveys*, 50(6), 94.<br>
<a id="ref-5"></a>
[5] Peng, H., Long, F. and Ding, C. (2005). [Feature Selection Based on Mutual Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy](https://doi.org/10.1109/TPAMI.2005.159). *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 27(8), 1226–1238.<br>
<a id="ref-6"></a>
[6] Urbanowicz, R. J., Meeker, M., La Cava, W., Olson, R. S. and Moore, J. H. (2018). [Relief-based feature selection: Introduction and review](https://doi.org/10.1016/j.jbi.2018.07.014). *Journal of Biomedical Informatics*, 85, 189–203.<br>
<a id="ref-7"></a>
[7] Kohavi, R. and John, G. H. (1997). [Wrappers for feature subset selection](<https://doi.org/10.1016/S0004-3702(97)00043-X>). *Artificial Intelligence*, 97(1–2), 273–324.<br>
<a id="ref-8"></a>
[8] Guyon, I., Weston, J., Barnhill, S. and Vapnik, V. (2002). [Gene Selection for Cancer Classification using Support Vector Machines](https://doi.org/10.1023/A:1012487302797). *Machine Learning*, 46, 389–422.<br>
<a id="ref-9"></a>
[9] Yuan, M. and Lin, Y. (2006). [Model selection and estimation in regression with grouped variables](https://doi.org/10.1111/j.1467-9868.2005.00532.x). *Journal of the Royal Statistical Society: Series B*, 68(1), 49–67.<br>
<a id="ref-10"></a>
[10] Simon, N., Friedman, J., Hastie, T. and Tibshirani, R. (2013). [A Sparse-Group Lasso](https://doi.org/10.1080/10618600.2012.681250). *Journal of Computational and Graphical Statistics*, 22(2), 231–245.<br>
<a id="ref-11"></a>
[11] Tibshirani, R. (1996). [Regression Shrinkage and Selection via the Lasso](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x). *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.<br>
<a id="ref-12"></a>
[12] Zou, H. and Hastie, T. (2005). [Regularization and variable selection via the elastic net](https://doi.org/10.1111/j.1467-9868.2005.00503.x). *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.<br>
<a id="ref-13"></a>
[13] Strobl, C., Boulesteix, A.-L., Zeileis, A. and Hothorn, T. (2007). [Bias in random forest variable importance measures: illustrations, sources and a solution](https://doi.org/10.1186/1471-2105-8-25). *BMC Bioinformatics*, 8, 25.<br>
<a id="ref-14"></a>
[14] Lundberg, S. M. and Lee, S.-I. (2017). [A Unified Approach to Interpreting Model Predictions](https://papers.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html). *Advances in Neural Information Processing Systems*, 30, 4765–4774.<br>
<a id="ref-15"></a>
[15] Ribeiro, M. T., Singh, S. and Guestrin, C. (2016). ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://doi.org/10.1145/2939672.2939778). *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144.<br>
<a id="ref-16"></a>
[16] Jain, S. and Wallace, B. C. (2019). [Attention is not Explanation](https://doi.org/10.18653/v1/N19-1357). *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies*, 3543–3556.<br>
<a id="ref-17"></a>
[17] Wiegreffe, S. and Pinter, Y. (2019). [Attention is not not Explanation](https://doi.org/10.18653/v1/D19-1002). *Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing*, 11–20.<br>
<a id="ref-18"></a>
[18] Candès, E., Fan, Y., Janson, L. and Lv, J. (2018). [Panning for gold: 'model-X' knockoffs for high dimensional controlled variable selection](https://doi.org/10.1111/rssb.12265). *Journal of the Royal Statistical Society: Series B*, 80(3), 551–577.<br>
<a id="ref-19"></a>
[19] Meinshausen, N. and Bühlmann, P. (2010). [Stability selection](https://doi.org/10.1111/j.1467-9868.2010.00740.x). *Journal of the Royal Statistical Society: Series B*, 72(4), 417–473.

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
- **wrapper method**: A selection method that refits a model for each candidate subset and scores the subset by that model's performance.
