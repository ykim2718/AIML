# Multivariate Feature Selection
Rev. 7 | Created: 2026-09-14 | Updated: 2026-09-17 09:16 CDT

## 1. Purpose

- **Problem Statement**: Scoring features one at a time handles neither the joint information nor the duplicated information between features, so the feature set is never optimized.
- **Goal**: Build and order a taxonomy and a hierarchy of the methods that read interaction and redundancy together, so that they can be applied to the situation at hand or used as a rule of thumb.
- **Non-Goal**: Testing features one at a time belongs to [Univariate Feature Selection](../univariate-feature-selection/univariate-feature-selection-ko.md).

### 1.1 Motivation

A combination of features sometimes explains the target better than the features alone, so selection has to work on combinations. $X_1$ and $X_2$ may each correlate weakly with the target $Y$ while their combination is a strong signal for $Y$, the XOR problem being the standard example.

Multivariate analysis has three purposes.

1️⃣ Removing the multicollinearity and the redundancy between features<br>
2️⃣ Finding the synergy between features<br>
3️⃣ Raising model performance and holding off overfitting

## 2. Taxonomy

Methods split along two lines: when the model is consulted (approach) and how interaction is handled (interaction).

```text
Multivariate feature selection taxonomy
|
+-- 1. Approach-based hierarchy
|   |
|   +-- Filter methods
|   |   +-- Correlation matrix and VIF ...... multicollinearity removal
|   |   +-- mRMR ........................... minimum redundancy maximum relevance
|   |   +-- ReliefF ........................ neighbour contrast
|   |
|   +-- Wrapper methods
|   |   +-- Forward selection / backward elimination
|   |   +-- RFE ........................... recursive feature elimination
|   |   +-- Genetic algorithm search
|   |
|   +-- Embedded methods
|       +-- Lasso (L1) / ElasticNet
|       +-- Tree-based importance ......... random forest, XGBoost, LightGBM
|
+-- 2. Interaction-based hierarchy
    +-- Redundancy reduction ............... removing duplicated information
    +-- Feature synergy .................... keeping features that matter together
    +-- Dimensionality tradeoff ............ trading dimension against signal
```

Fig 1. Two hierarchies of multivariate feature selection

Alongside the two hierarchies runs one more line, whether y is read. The correlation filter and VIF compute the correlation among X only and run without y, while mRMR, ReliefF, the embedded methods and the wrappers all score against y.

## 3. Approach-based Methods

- 3.1 Filter: statistics of X and y computed without a model
- 3.2 Wrapper: a model held as an external scorer and refitted for every candidate subset, the scores compared (fits = candidates)
- 3.3 Embedded: the selection carried out by the penalty or the split gain inside a single fit (fits = 1)

### 3.1 Multivariate Filter Methods

Feature combinations are screened from the statistical properties of the data alone, without fitting a model. Unlike a univariate filter, the correlation between features is computed as well.

mRMR (Minimum Redundancy Maximum Relevance) is solved as an optimization that maximizes the mutual information with the target and minimizes the mutual information among the selected features.

```math
\max_{S} \left[ \frac{1}{|S|} \sum_{i \in S} I(x_i; y)
- \frac{1}{|S|^2} \sum_{i, j \in S} I(x_i; x_j) \right]
\hspace{10em} (1)
```

VIF (Variance Inflation Factor) regresses one feature on the remaining features and measures how well they explain it. Features with $\mathrm{VIF} \gt 10$ are removed one at a time. The definition and the removal procedure are in [Appendix C](#appendix-c-variance-inflation-factor).

### 3.2 Wrapper Methods

A chosen model serves as the judge, and a search algorithm looks for the feature subset that scores best.

The procedure of RFE (Recursive Feature Elimination) runs as follows.

- Fit the model on every feature
- Drop the feature of the smallest coefficient or importance
- Repeat until the target number of features is reached

A greedy search adds (forward) or removes (backward) one feature at a time and follows the change in the cross validation score. A genetic algorithm holds several subsets as one generation and builds the next generation by mixing the high scoring ones and swapping parts, reaching combinations a sequential search never visits.

### 3.3 Embedded Methods

Feature selection sits inside the learning algorithm of the model.

Lasso adds the sum of the absolute coefficients $\lambda \sum |\beta_i|$ to the loss as a penalty, driving the coefficients of unneeded features exactly to zero. ElasticNet mixes in the sum of the squared coefficients, so a group of correlated features is kept together where lasso leaves one member of it.

Tree-based importance reads a multivariate importance from the node split contribution of a tree model (MDI) or from the drop in performance when the values are shuffled (permutation importance). Random forest and the gradient boosting family XGBoost and LightGBM implement it, and all three report split contributions, so they go straight into `SelectFromModel`.

### 3.4 Comparison

Multivariate filter, wrapper and embedded move in opposite directions on computational cost and on how much interaction they carry.

Table 1. Comparison of the three approaches

| # | Aspect              | Multivariate filter             | Wrapper                  | Embedded              |
| :: | :-----------------: | :-----------------------------: | :----------------------: | :-------------------: |
| 1 | Computational cost  | Low                             | Very high                | Medium                |
| 2 | Overfitting risk    | Low                             | High                     | Medium                |
| 3 | Model dependence    | None (model-agnostic)           | Tied to the chosen model | Built into that model |
| 4 | Interaction carried | Limited (mostly 1:1 redundancy) | Very well                | Well                  |

## 4. Interaction-based Methods

Grouping the same methods by how they handle interaction gives the second hierarchy of section 2. A method can sit in two branches, and then what it does in each branch is written separately.

### 4.1 Redundancy Reduction

The branch that erases duplicated signal, which reads the correlation between features and may leave the target unread.

- Correlation filter: one feature of every pair above the limit removed
- VIF: the feature best explained by the rest removed one at a time
- The min-redundancy term of mRMR: the mutual information among the selected features charged as a penalty

### 4.2 Feature Synergy

The branch that keeps joint signal, which shows only when features are scored as a combination.

- Wrapper (RFE, forward and backward search): a candidate subset scored by the model, so the effect of the combination enters the score
- Tree-based importance: a split made inside a node already separated, so a contribution that depends on the values of other features is carried
- ReliefF: the nearest hit and the nearest miss of every sample compared over the distance of the whole feature vector, so a difference that shows only alongside other features enters the score. Class labels only, with RReliefF as the separate algorithm for a regression target

### 4.3 Dimensionality Tradeoff

The branch that trades the number of kept dimensions against signal, a cut-off that places the cut on a ranking or a penalty path already produced. It therefore needs a criterion that produces a ranking first, and it usually sits on the ranking of section 4.1 and section 4.2. The $\lambda$ of lasso is the exception, where one penalty fixes the ranking and the cut-off together.

- $\lambda$ of lasso: the larger the value, the more coefficients reach zero and the fewer dimensions remain
- Target feature count of RFE: the kept dimension given directly
- Threshold of tree-based importance: the cut placed by a rule such as the mean importance

## 5. Target Kind

Classification and regression are properties of the target, orthogonal to the approach hierarchy of section 3 and to the interaction hierarchy of section 4. One method keeps its criterion and substitutes only the estimator that matches the target.

Table 2. What each method changes when the target is regression instead of classification

| Section                  | Method                         | Uses y | Classification target           | Regression target                       |
| :----------------------: | :----------------------------: | :----: | :-----------------------------: | :-------------------------------------: |
| 3.1 Filter               | corr, VIF                      | No     | Correlation among X only        | Correlation among X only                |
| 3.1 Filter               | mRMR                           | Yes    | `mutual_info_classif`           | `mutual_info_regression`                |
| 3.1 Filter / 4.2 Synergy | ReliefF                        | Yes    | hit/miss contrast               | None (RReliefF is a separate algorithm) |
| 3.3 Embedded             | random forest, LightGBM        | Yes    | Classifier                      | Regressor                               |
| 3.3 Embedded             | lasso, elastic net             | Yes    | Regression fit on the 0/1 label | Regression fit on y                     |
| 3.2 Wrapper              | RFE, forward/backward, genetic | Yes    | Scored by `LogisticRegression`  | Scored by `LinearRegression`            |

- Rows whose `Uses y` is No: the same computation under both targets
- Rows whose `Uses y` is Yes: the estimator and the model swapped for their regression form
- ReliefF: no counterpart to use in regression, so the algorithm itself splits

## 6. Selection Instability

Selection instability is the property that the kept columns change when the data or the method changes a little, while the score stays where it was and only the names differ. Two causes produce it.

- Interchangeable features: an equivalence class of columns so correlated that swapping them leaves the performance where it was
- Sampling noise: correlation and importance ranks wobbling from one resample to the next, which flips the features that sat near the threshold

The remedies replace one run of the selection with a selection frequency over repeats or with a treatment applied per correlated group.

- Stability selection: the selection repeated on every bootstrap sample, keeping the features whose selection frequency passes a threshold (0.6, say)
- Cluster representative: the features clustered by correlation and one member kept per cluster, which removes the swapping inside a cluster
- Group-wise selection: a correlated group kept together by ElasticNet or group lasso, which stops one member from standing for the group
- Selection frequency reporting: the selection frequency of every feature written beside the chosen set, which shows the columns that can stand in for one another

## 7. Workflow

Cheap methods cut the candidates down before the expensive ones run. The cost of a wrapper grows with the number of features left, so three steps come before it.

- Step 1 (constant removal): features holding one value across every sample removed first. Neither correlation nor importance is defined for them, and no later step has anything to weigh
- Step 2 (pre-filtering): the duplicated features above a correlation of 0.95 removed by a univariate statistic or by VIF
- Step 3 (embedded selection): the candidates narrowed by lasso or by random forest, XGBoost, LightGBM
- Step 4 (fine-tuning via wrapper): the final subset decided by RFE or sequential feature selection once the candidates are few

---

## Appendix A. Terminology

- **Cross Validation**: the procedure that splits the data into folds and uses each in turn for validation, measuring how the model generalizes.
- **MDI (Mean Decrease in Impurity)**: the sum of the impurity one feature removes over the node splits of a tree.
- **Multicollinearity**: the state where the input variables hold a strong linear relation, so a coefficient is not assigned uniquely to a single variable.
- **Mutual Information**: the amount by which the entropy of one variable falls once the other is known.
- **Overfitting**: the state where a model has learned the noise of the training data and performs worse on new data.
- **Permutation Importance**: the importance measured as the drop in performance when the values of one feature are shuffled.
- **XOR Problem**: the relation that is 1 only when two binary inputs differ. Each input has zero correlation with the output while the two together determine it completely.

## Appendix B. Implementation

The class below runs the four steps of section 7 with scikit-learn. `run` drops the constant features first and passes only the remaining columns through the other three steps. Each step takes its threshold from the constructor and returns the original column indices, so the names of the chosen features can be recovered at the end. Each step takes its method name as the `Literal` alias of that branch, whose members are declared on the class once and whose tuple of names is derived beside it with `get_args`. The filter step implements four names (`corr`, `vif`, `mrmr`, `relieff`), the embedded step four (`random_forest`, `lightgbm`, `lasso`, `elasticnet`) and the wrapper step four (`rfe`, `forward`, `backward`, `genetic`); a name outside a list raises `ValueError`. The `task` the constructor takes picks the model behind every step as Table 2 has it, and calling `relieff` under regression raises `ValueError` together with the filters that task can use.

The input is the breast cancer dataset shipped with scikit-learn, with one column holding 1.0 throughout appended so that the constant removal step is visible, giving 569 samples and 31 features. The original 30 features overlap heavily and all of them are standardized with `StandardScaler`. The label is binary, so the example is written with classification models; for a regression target the right column of Table 2 takes over.

The class is in [src/multivariate_feature_selection.py](src/multivariate_feature_selection.py), and `python src/multivariate_feature_selection.py` redraws Fig 2.

The constant column drops at the first step and reaches no filter. Of the 30 that remain the four filters keep 23, 17, 10 and 10, and the four embedded methods keep 9, 6, 12 and 18, so their answers differ; `elasticnet`, which mixes in L2, keeps a correlated group together and holds 6 more than `lasso`. The four wrappers each pick 5 out of the 9 the random forest left, where `forward` and `backward` reach the same combination while `rfe` and `genetic` each take their own. Following the `corr` → `random_forest` → `rfe` that `run` takes by default, the feature count falls 31, 30, 23, 6, 5, and the costliest step runs where only 6 are left.

Which method kept which feature is in Fig 2.

<img src="multivariate-feature-selection-ko_fig/fig2.png" width="800" style="max-width: 100%;" alt="Fig 2">

Fig 2. Which features each selection method keeps

- The rows are the 30 features left after constant removal, in name order, and the columns are the 12 methods in filter, embedded and wrapper order, with a wider gap between the three branches. A filled cell means that method kept that feature.
- The number in brackets under a column name is how many features that method kept, and the wrapper columns ran on the 9 the random forest left.
- `mean concave points` is filled in nine of the twelve columns and `worst concave points` in eleven. `worst compactness`, in contrast, survives in the corr column alone.

### B.1 Choosing Among Answers 🥑

Each method maximizes a different quantity, so the columns that survive differ. corr drops the later column of every pair above the limit, VIF drops the one the rest explain well, mRMR reads the overlap with what is already chosen, ReliefF the distance to the neighbours, Lasso the penalty that leaves one member of a group, ElasticNet the penalty that keeps the group, the tree family the split gain, and a wrapper the cross validation score of that model. What to use is settled by the validation score on a split that was not used for the selection.

A split answer rarely means a different performance. Where the data holds many interchangeable features several subsets score almost the same, and which of them is kept is settled by the tie-break rule of each criterion rather than by signal. The breast cancer data of this example has 21 pairs correlated above 0.9 among its 30 features, and `mean radius` and `mean perimeter` are the same column at 0.998.

Table 3. Cross validation score of each wrapper subset of the breast cancer example

| Wrapper  | Features it keeps<br>(input) | 10-fold accuracy<br>(output) | Selected features<br>(output)                                                                |
| :------: | :--------------------------: | :--------------------------: | :------------------------------------------------------------------------------------------: |
| rfe      | 5                            | 0.949 ± 0.025                | area error, worst area, <ins>worst concave points</ins>, worst perimeter, worst radius       |
| forward  | 5                            | 0.954 ± 0.037                | mean concavity, worst area, <ins>worst concave points</ins>, worst perimeter, worst radius   |
| backward | 5                            | 0.954 ± 0.037                | mean concavity, worst area, <ins>worst concave points</ins>, worst perimeter, worst radius   |
| genetic  | 5                            | 0.953 ± 0.039                | area error, mean concave points, mean concavity, worst area, <ins>worst concave points</ins> |

The four wrappers take the target count as an argument, and this example gives `final_count=5`. The four subsets are therefore the same size and the score differs only through what was chosen. That difference falls inside the standard deviation, so the score alone cannot pick one on this data. The order below takes over then.

1️⃣ Step 1 (agreement): keep the features several methods chose in common first. `worst concave points` is in all four subsets of Table 3<br>
2️⃣ Step 2 (stability): take the side that returns the same subset when the data is resampled. How it is measured is in section 6<br>
3️⃣ Step 3 (actionability): where a tie is left, take the feature the process can act on or a reader can make sense of

## Appendix C. Variance Inflation Factor

VIF is defined by the coefficient of determination $R_i^2$ of the regression of feature $x_i$ on all the remaining features, and measures how far that feature is reproduced by a linear combination of the rest.

```math
\mathrm{VIF}_i = \frac{1}{1 - R_i^2} \hspace{19em} (2)
```

- $R_i^2$: the coefficient of determination of $x_i$ regressed on the remaining features
- Range: 1 where $R_i^2 = 0$, growing as $R_i^2$ approaches 1 and infinite under perfect collinearity
- Origin of the name: the variance of the coefficient $\hat{\beta}_i$ is VIF times what it would be without collinearity

The removal repeats one feature at a time.

- Compute the value of equation (2) for every remaining feature
- Remove the single feature whose value is the largest, where that value passes the limit
- Recompute the values of the remaining features and repeat until all of them fall under the limit

One at a time is what keeps usable features: where a pair inflated each other, dropping one brings the value of the other down as well, and removing several at once loses features that would have stayed.

Table 4. Reading of a VIF value

| VIF     | Reading                                                   |
| :-----: | :-------------------------------------------------------: |
| 1       | Not explained by the remaining features                   |
| 1–5     | Weak collinearity, usually left in place                  |
| 5–10    | Moderate collinearity, judged by the data and the purpose |
| &gt; 10 | Strong collinearity, a candidate for removal              |
| ∞       | Perfect collinearity, a linear combination of the rest    |

The unit each one reads is where VIF parts from the correlation filter. The correlation filter reads the correlation of a pair and passes over the collinearity three or more features build together, while VIF regresses on all the rest and catches that case. It pays for this with one regression per feature, solved again after every removal.

- Categorical dummies: dummies from one variable are collinear with each other and always score high, so the reference category is dropped and the reading is made per variable
- Perfect collinearity: the value is infinite where $R_i^2 = 1$, and `_vif_of` of Appendix B returns `float("inf")` there so that the feature becomes the next removal
- Independence from the target: $y$ does not enter equation (2), which puts VIF among the `Uses y` No rows of Table 2
