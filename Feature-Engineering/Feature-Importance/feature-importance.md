# Feature Importance
Rev. 2 | Created: 2026-08-31 | Updated: 2026-08-31 18:01 CDT

> A survey of the methods that put a number on a feature, arranged by what the number is computed
> from, so that a method is chosen from the question being asked rather than from whichever library
> is already imported.

## 1. Scope

Feature importance is not one quantity. Every method in this document returns one number per
feature, and a reader who does not know which method produced the number cannot say what it means.
Two numbers for the same feature from two methods can differ by an order of magnitude and both be
correct, because they answer different questions.

The first split is between two questions that the same word covers.

- **Reliance of a fitted model.** How much this particular trained model uses the feature. It changes when the model is refitted on the same data with a different seed.
- **Information in the data.** How much the feature carries about the target, whatever model is used. A property of the joint distribution, not of any one fit.

Most published methods answer the first while being read as the second, and section 12 collects
what follows from that. Section 2 sets out the axes on which a method takes a position, section 3
puts the families in one hierarchy, sections 4 to 10 take the seven families in turn, and section
11 puts the choice in one place.

Every method carries a status, so that a settled technique is not confused with one whose practice
is still moving.

- **Standard.** Available in a mainstream library, described in textbooks, with documented failure modes.
- **Recent.** An active research line, usually implemented in a research package, with practice not yet settled.

Status records how settled a method is, not how good it is. A standard method is very often the
right one, and a recent method is chosen because the question needs what it adds, not because it
is newer.

## 2. Axes Of The Question

The seven subsections below are seven axes, not seven categories. A method takes a position on
every one of them at once, and two methods are comparable only where their positions agree.

### 2.1. Object

What the importance is a property of.

- **A fitted model.** One trained model, and the reliance that this fit happens to have.
- **A model class.** Every model that fits the data nearly as well, which turns one number into a range [[4](#ref-4)].
- **The distribution.** A population quantity, with the model used only as a nuisance estimate [[18](#ref-18)].

A number computed on one fit and reported as a property of the problem is the most common
overstatement in this area, and the model class object exists to name the gap.

### 2.2. Scope

The region over which the number holds.

- **Global.** One number per feature for the whole dataset.
- **Cohort.** One number per feature within a subgroup, where the model behaves differently in different regions.
- **Local.** One number per feature per prediction.

A global number can be built by aggregating local ones, but signed local values cancel when
averaged, so the mean absolute value is used instead. That aggregate is a different quantity from
a directly computed global measure, and the two need not agree in rank.

### 2.3. Effect Measured

What is watched when the feature is disturbed.

- **Loss.** How much worse the predictions get. Labels required.
- **Output.** How far the prediction moves, in either direction. No labels required.
- **Structure.** How the model was built, read off its parameters or its splits. Nothing is disturbed at all.

The first two come apart under redundancy. A loss-based measure gives near zero to a feature the
model leans on heavily whenever a correlated copy can cover the loss, and an output-based measure
gives it full weight. The data-only family of section 4 sits outside all three, because it scores
association with the target directly and there is no model to disturb.

### 2.4. Conditioning

What the feature is replaced by when it is disturbed.

- **Marginal.** A draw from the feature's own marginal distribution, independent of the rest of the row. Input combinations that never occur in the data are produced, and the model is evaluated where it was never fitted [[5](#ref-5)].
- **Conditional.** A draw from the feature's conditional distribution given the other features. The row stays plausible, at the price of estimating that conditional.

Marginal asks what the model does with an input. Conditional asks what the feature adds given the
rest. Correlated features are where the two diverge, and that divergence is the largest single
source of disagreement between methods that otherwise look alike.

### 2.5. Data

Which rows the number is computed on.

- **Training data.** Split statistics and coefficients, which reward whatever the model fitted, including what it overfitted.
- **Held-out data.** Permutation and retraining measures on rows the model has not seen, which report reliance that survives out of sample.

A feature of pure noise with enough distinct values earns a high training-data score and a
near-zero held-out score, and the gap between the two is itself a diagnostic.

### 2.6. Unit

What one number is attached to.

- **A single column.** The default everywhere.
- **A group of columns.** The one-hot block of one categorical variable, or the channels of one sensor, scored together.
- **A pair or a set.** Interaction strength rather than main effect.

One-hot columns scored one at a time understate the categorical variable they came from, because
the variable's contribution is split across its levels. Grouping before scoring is the fix, and it
is available in every family except the structure-based one.

### 2.7. Guarantee

What is claimed about the number.

- **A ranking only.** No statement about error, which is what most methods give.
- **A confidence interval.** A statement about sampling variability of the estimate.
- **A controlled error rate.** A selected set whose false discovery rate is held below a chosen level.

The family of section 10 is the only one that gives the third, and so the only one that answers
"which features can I report as real" rather than "which features scored highest".

Table 1. Axes and the choice each one forces

| Axis | Positions | What is lost by not choosing |
|------|-----------|------------------------------|
| Object | Fitted model, model class, distribution | One fit's accident read as a property of the problem |
| Scope | Global, cohort, local | A model that behaves differently in different regions, averaged flat |
| Effect | Loss, output, structure | Redundant features scored at zero, or noise scored high |
| Conditioning | Marginal, conditional | A number produced where the model was never fitted |
| Data | Training, held-out | Overfitting reported as importance |
| Unit | Column, group, set | A categorical variable split across its own levels |
| Guarantee | Ranking, interval, error rate | A cut-off with no error statement behind it |

## 3. Hierarchy

The families divide by where the number comes from, and that is the only division that stays
stable. A division by model type does not, because the same family reappears for every model, and
a division by global against local does not, because most families supply both.

```
Feature importance
|
+-- A. Data-only association (no model is fitted)
|   +-- A1. Linear and monotone ............. Pearson r, Spearman rho, ANOVA F, chi-square
|   +-- A2. General dependence .............. mutual information, distance correlation, HSIC
|   +-- A3. Redundancy-aware ................ mRMR, correlation-filtered ranking
|
+-- B. Model-internal structure (read off the fitted model)
|   +-- B1. Coefficients .................... standardized beta, lasso path, elastic net
|   +-- B2. Split statistics ................ MDI, split count, cover
|   +-- B3. Component and attention weights . PLS VIP, PCA loading, attention weight
|
+-- C. Perturbation and removal (change an input, watch the model)
|   +-- C1. Marginal permutation ............ MDA, drop-and-shuffle variants
|   +-- C2. Conditional permutation ......... conditional MDA, grid-conditioned permutation
|   +-- C3. Retraining ...................... drop-column, LOCO, model class reliance
|   +-- C4. Curve summaries ................. PDP spread, ALE spread, H-statistic
|
+-- D. Gradient attribution (differentiate the model)
|   +-- D1. Plain gradient .................. saliency, gradient times input
|   +-- D2. Path integral ................... integrated gradients, DeepLIFT
|   +-- D3. Layer propagation ............... LRP, Grad-CAM
|
+-- E. Game-theoretic attribution (share the payoff among features)
|   +-- E1. Local additive .................. LIME, KernelSHAP, TreeSHAP, DeepSHAP
|   +-- E2. Global additive ................. SAGE, Shapley effects
|   +-- E3. Amortized ....................... FastSHAP and learned explainers
|
+-- F. Variance-based sensitivity (decompose output variance)
|   +-- F1. Screening ....................... Morris elementary effects
|   +-- F2. Variance decomposition .......... Sobol indices, FAST, functional ANOVA
|
+-- G. Error-controlled selection (attach a guarantee)
    +-- G1. Knockoffs ....................... linear-model knockoffs, model-X knockoffs
    +-- G2. Nonparametric inference ......... LOCO inference, VIM, decorrelated LOCO
```

Fig 1. Hierarchy of feature importance methods by the source of the number

The order of the families is the order in which they were needed. A is the oldest and needs no
model. B costs nothing once a model is fitted. C is the first family that is model-agnostic and
loss-based. D exists because C is too slow for a network with a million inputs. E exists because
C and D disagree and a set of axioms was wanted to settle the disagreement. F comes from a
different literature, that of computer experiments, and arrives at the same place. G is the only
one that answers with an error rate.

## 4. Data-Only Association

The family scores a feature against the target with no model in between. Nothing is fitted, so
nothing can be overfitted, and the number is a property of the data alone. Every member is
marginal by construction: the feature is compared against the target one at a time, and the other
features are not in the calculation.

Table 2. Data-only association measures

| Method | What it scores | Status |
|--------|----------------|--------|
| Pearson correlation | Linear dependence, signed | Standard |
| Spearman, Kendall | Monotone dependence, signed, rank-based | Standard |
| ANOVA F, chi-square | Group separation for a categorical target or feature | Standard |
| Mutual information | Any dependence, at the cost of density estimation | Standard |
| Distance correlation | Any dependence, zero only under independence | Standard |
| HSIC | Any dependence in a kernel feature space | Standard |
| mRMR [[21](#ref-21)] | Relevance to the target minus redundancy with the already selected | Standard |

The limitation is the same one for all of them. A feature that carries nothing on its own and
everything in combination scores zero, and the exclusive-or is the smallest example: each input
is independent of the target, and the pair determines it. A pure filter can therefore drop the
only two features that mattered, which is why this family is used to shrink a very wide table
before modeling rather than to explain a model afterwards.

mRMR is the one member that looks past a single feature, by penalizing a candidate for redundancy
with what has already been chosen. That makes the output a set rather than a ranking, and the set
depends on the order in which it was built.

## 5. Model-Internal Structure

Here the number is read off the fitted model itself. Nothing is perturbed and nothing is
re-evaluated, so the cost is nil once training is done, and that is the reason for the family's
popularity. It is also the reason for its weakness: the number describes how the model was built,
on the data it was built from, and section 2.5 applies in full.

Table 3. Model-internal structure measures

| Method | What it scores | Status |
|--------|----------------|--------|
| Standardized coefficient | Response change per standard deviation of the feature, signed | Standard |
| Lasso path [[22](#ref-22)] | Order of entry and coefficient size along the regularization path | Standard |
| MDI | Total impurity reduction over the splits on the feature | Standard |
| Split count, cover | How often the feature is split on, and how many rows those splits touch | Standard |
| PLS VIP | Contribution of the feature to the latent components that predict the response | Standard |
| PCA loading | Contribution to reconstruction variance, with no reference to a target | Standard |
| Attention weight | Weight the model places on an input position | Standard |

Two cautions belong to specific members. A coefficient is comparable across features only after
the features are put on one scale, and an unstandardized coefficient is a statement about units,
not about importance. MDI is computed on the training data and is biased toward features with
many distinct values, which is documented for random forests together with the mechanism that
produces it [[2](#ref-2)]; a continuous noise column can outrank a real binary predictor for this
reason alone.

PCA loading is in the table to be excluded from the rest. It scores a feature for its part in
reconstructing the inputs, and a feature can dominate the leading component while carrying nothing
about the target.

## 6. Perturbation And Removal

The family disturbs the input and measures what happens to the model. It is the first family that
is model-agnostic, because it needs only the ability to call the model on new rows, and it is the
first that can be loss-based, because the disturbed rows can be scored against held-out labels.

Table 4. Perturbation and removal measures

| Method | What it scores | Status |
|--------|----------------|--------|
| Permutation importance, MDA [[1](#ref-1)] | Loss increase when one column is shuffled, breaking its link to the target | Standard |
| Conditional permutation [[3](#ref-3)] | The same, with the shuffle restricted within cells of the correlated features | Standard |
| Drop-column, LOCO | Loss increase when the model is refitted without the feature | Standard |
| Model class reliance [[4](#ref-4)] | Range of permutation importance over every model that fits nearly as well | Recent |
| PDP spread | Standard deviation of the partial dependence curve of the feature | Standard |
| ALE spread [[15](#ref-15)] | The same, from accumulated local effects, which use local differences instead of marginal averages | Recent |
| H-statistic [[14](#ref-14)] | Share of prediction variance not explained by the additive parts, one pair at a time | Standard |

Permutation importance is the default in this family and the one whose defects are best described.
Shuffling one column independently of the others is the marginal position of section 2.4, so
correlated features are evaluated at combinations the data never contains, and the resulting number
mixes the model's reliance with its behavior off the data manifold [[5](#ref-5)]. The same
criticism reaches partial dependence, and accumulated local effects were built to avoid it by
averaging local differences within a neighborhood rather than averaging the model over the whole
marginal [[15](#ref-15)].

Retraining answers a different question from permutation, and the difference is worth keeping in
view. Permutation asks how much this model needs the feature. Retraining asks how much a model
fitted without the feature loses, which two correlated features can answer as "nothing" each,
separately and truthfully. Model class reliance closes that gap from the other side, by reporting
the range of reliance over the whole set of well-fitting models rather than the reliance of the
one that happened to be trained [[4](#ref-4)].

## 7. Gradient Attribution

Differentiating the model with respect to its inputs applies only where the model is
differentiable, and it is local by construction, since a gradient is taken at one point. The
reason the family exists is cost. A network with a hundred thousand inputs cannot be permuted
column by column, and a backward pass gives every input a number at once.

Table 5. Gradient attribution measures

| Method | What it scores | Status |
|--------|----------------|--------|
| Saliency | Magnitude of the output gradient at the input | Standard |
| Gradient times input | The same, scaled by the input value, which makes the sum comparable to the output | Standard |
| Integrated gradients [[10](#ref-10)] | Gradient integrated along a path from a baseline to the input | Standard |
| DeepLIFT | Difference from a reference, propagated backwards by per-layer rules | Standard |
| LRP [[11](#ref-11)] | Output relevance redistributed layer by layer to the inputs | Standard |
| Grad-CAM [[12](#ref-12)] | Gradient-weighted activation map of the last convolutional layer, as a coarse spatial mask | Standard |

Integrated gradients was introduced axiomatically, as the attribution that satisfies a stated set
of properties, and the choice of baseline is the free parameter that decides the answer
[[10](#ref-10)]. A black image, a blurred image and an average image are three different questions
about the same prediction.

The family's position on the axes is worth stating plainly. It is output-based, never loss-based,
so it says where the prediction came from and never whether the prediction was any good. A
confidently wrong prediction produces a clean attribution map.

## 8. Game-Theoretic Attribution

The family treats the features as players and the prediction as a payoff to be divided among them.
The Shapley value is the unique division satisfying a small set of fairness conditions, which is
what recommends it and what makes its cost exponential in the number of features, since the value
is an average over all subsets.

Table 6. Game-theoretic attribution measures

| Method | What it scores | Status |
|--------|----------------|--------|
| LIME [[6](#ref-6)] | Coefficients of a sparse linear model fitted to the black box in a neighborhood of one row | Standard |
| KernelSHAP [[7](#ref-7)] | Shapley values of the prediction, estimated by weighted least squares over sampled subsets | Standard |
| TreeSHAP [[8](#ref-8)] | The same for tree ensembles, exactly and in polynomial time | Standard |
| DeepSHAP | The same for networks, by composing per-layer attributions | Standard |
| SAGE [[9](#ref-9)] | Shapley values of the loss rather than the prediction, giving one global number per feature | Recent |
| Shapley effects | Shapley values of the output variance, which links this family to section 9 | Recent |
| FastSHAP [[20](#ref-20)] | Shapley values predicted in one forward pass by an explainer trained for the purpose | Recent |

The unification is the family's contribution: LIME, DeepLIFT and layer-wise relevance propagation
were shown to be members of one class of additive attribution methods, with the Shapley value as
its distinguished element [[7](#ref-7)]. The additive-importance framing that SAGE belongs to
covers a further set of global measures in the same way [[9](#ref-9)].

Three practical points decide use. First, the subset average requires a value for a prediction
made with some features missing, and the two ways of supplying it, marginal and conditional,
reproduce exactly the split of section 2.4 and give different attributions to correlated features.
Second, the estimation algorithms are many and their accuracy differs, which has been surveyed and
compared in one place [[24](#ref-24)]. Third, the axioms constrain how the payoff is divided and
say nothing about whether the division answers a human question, which is the substance of the
standing critique of the family [[19](#ref-19)].

## 9. Variance-Based Sensitivity

Variance-based sensitivity comes from the study of computer models rather than from statistics,
and it decomposes the variance of the output into the contribution of each input and of each
interaction. Because what is decomposed is the model's output, no labels are involved, and the
inputs are usually sampled from a design rather than observed.

Table 7. Variance-based sensitivity measures

| Method | What it scores | Status |
|--------|----------------|--------|
| Morris elementary effects | Mean and spread of one-at-a-time changes over a coarse grid, as a screening pass | Standard |
| Sobol first-order index [[13](#ref-13)] | Share of output variance explained by the feature alone | Standard |
| Sobol total-effect index [[13](#ref-13)] | Share explained by the feature and every interaction it takes part in | Standard |
| FAST | The same indices, estimated by frequency analysis rather than by sampling | Standard |
| Functional ANOVA | Full decomposition into main effects and interaction terms of increasing order | Standard |

The pair of Sobol indices is the family's most useful export. The first-order index and the
total-effect index bound the feature from below and above, and their gap is exactly the part of
its influence that lives in interactions. A feature with a first-order index near zero and a large
total-effect index matters only in combination, which is exactly the case the data-only family of
section 4 cannot see at all.

The assumption to check is independence of the inputs. The variance decomposition is derived for
independent inputs, which a designed experiment supplies and observational data does not, so on
correlated process data these indices carry the same off-manifold problem as marginal permutation.

## 10. Error-Controlled Selection

What follows answers a different question from the rest of this document. These methods do not
rank features; they return a set, with a bound on the fraction of the set that is spurious. That
is the third guarantee of section 2.7, and it is bought with an explicit model of what a null
feature would look like.

Table 8. Error-controlled selection methods

| Method | What it scores | Status |
|--------|----------------|--------|
| Knockoffs [[16](#ref-16)] | A selected set with a controlled false discovery rate, by contrast against synthetic negative-control copies | Recent |
| LOCO inference [[17](#ref-17)] | A confidence interval for the prediction-error increase caused by dropping the feature | Recent |
| VIM [[18](#ref-18)] | A confidence interval for a population importance parameter, with the fitted model as a nuisance | Recent |
| Decorrelated LOCO [[23](#ref-23)] | The same as LOCO, with the effect of correlation between covariates removed from the parameter | Recent |

The knockoff construction makes, for each feature, a synthetic copy that keeps the correlation
structure and carries no information about the target. Any feature that does not beat its own copy
is evidence about the null, and counting those gives the error rate without any distributional
assumption on the response. The procedure began in the linear model and was extended to arbitrary
models by the model-X construction, which shifts the assumption onto the distribution of the
features [[16](#ref-16)].

The inference members define importance as a population contrast — the predictiveness available
with every feature against the predictiveness available without the one in question — and then
estimate that contrast with any algorithm at all [[18](#ref-18)]. The dependence on correlation
that makes a leave-one-out parameter hard to read is what the decorrelated version modifies
[[23](#ref-23)].

## 11. Selection Guide

Table 9 places the seven families on the axes of section 2, and Table 10 goes the other way, from
a question to the family that answers it.

Table 9. Families on the axes

| Family | Object | Scope | Effect | Conditioning | Labels | Cost |
|--------|--------|-------|--------|--------------|--------|------|
| A. Data-only association | Distribution | Global | Association | Marginal | Required | Lowest |
| B. Model-internal structure | Fitted model | Global | Structure | Not applicable | Not used | None beyond training |
| C. Perturbation and removal | Fitted model, or model class | Global | Loss or output | Either | Required for loss | One pass per feature, or one refit per feature |
| D. Gradient attribution | Fitted model | Local | Output | Marginal | Not used | One backward pass |
| E. Game-theoretic attribution | Fitted model | Local, or global for SAGE | Output, or loss for SAGE | Either | Required for SAGE | Exponential in principle, sampled in practice |
| F. Variance-based sensitivity | Fitted model | Global | Output variance | Marginal, independence assumed | Not used | Large designed sample |
| G. Error-controlled selection | Distribution | Global | Loss | Conditional | Required | Highest |

Table 10. Question and the family that answers it

| Question | Family | Note |
|----------|--------|------|
| Which columns of a very wide table are worth keeping before modeling | A | Interactions invisible, so keep the cut generous |
| Which features did this tree ensemble split on | B | Training data only, biased toward many-valued columns |
| Which features does the deployed model actually need | C1 or C3, on held-out rows | Permutation for reliance, retraining for necessity |
| Which features matter once correlation is accounted for | C2 or G2 | The conditional question, which needs the conditional distribution |
| Why did the model produce this one prediction | D or E1 | Gradient for a network with many inputs, Shapley for a table |
| Which features matter over the whole dataset, with interactions included | E2 or F2 | SAGE for observed data, Sobol for a designed sample |
| Which features can be reported as real, with an error rate | G1 | The only family that answers this at all |
| How much does the answer depend on which model was fitted | C3, model class reliance | A range instead of one number |

## 12. Failure Modes

The defects below are properties of the methods, not of any implementation, and they recur across
families.

- **Correlation splits the credit.** Two features carrying the same information receive half each under one method and full weight each under another, and neither answer is wrong. The choice is the marginal against conditional choice of section 2.4, and reporting a number without stating which was taken leaves the reader unable to read it.
- **Structure-based scores reward overfitting.** MDI and unpenalized coefficients are computed on the data the model was fitted to, so a high-cardinality noise column can outrank a real predictor [[2](#ref-2)].
- **Permutation evaluates the model where it was never fitted.** Independent shuffling of a correlated column produces rows outside the data manifold, and the model's behavior there enters the score [[5](#ref-5)].
- **Zero importance does not mean irrelevant.** Under drop-one retraining, a feature with an exact duplicate in the table scores zero, and so does its duplicate.
- **Importance is not causation.** Every method in this document reports association or model reliance. A feature can be the model's strongest input and have no effect on the target when intervened on, and no reweighting of the numbers changes that.
- **Rankings are unstable.** Refitting with a different seed, or resampling the rows, reorders the middle of the ranking. A single ranking with no spread around it overstates what the data supports, which is what the interval methods of section 10 supply.
- **Aggregating local values changes the quantity.** Signed local attributions cancel when averaged, so the mean absolute value is used, and that is a different measure from a global method's output rather than a summary of it.
- **One-hot columns understate their variable.** Scoring the levels separately splits one variable's contribution across its levels, in every family except the structure-based one, where grouping is not available.
- **The axioms do not carry over to the question.** The Shapley value's uniqueness is a statement about how a payoff is divided, and the standing critique is that this does not make the division an answer to what a person wanted to know [[19](#ref-19)]. For model classes as rich as ordinary networks, any attribution method that is complete and linear — integrated gradients and the Shapley methods among them — has been shown to be no better than random guessing at several natural end tasks [[25](#ref-25)].

## References

<a id="ref-1"></a>
[1] Breiman, L. (2001). [Random Forests](https://doi.org/10.1023/A:1010933404324). *Machine Learning*, 45(1), 5–32.

<a id="ref-2"></a>
[2] Strobl, C., Boulesteix, A.-L., Zeileis, A. and Hothorn, T. (2007). [Bias in random forest variable importance measures: illustrations, sources and a solution](https://doi.org/10.1186/1471-2105-8-25). *BMC Bioinformatics*, 8, 25.

<a id="ref-3"></a>
[3] Strobl, C., Boulesteix, A.-L., Kneib, T., Augustin, T. and Zeileis, A. (2008). [Conditional variable importance for random forests](https://doi.org/10.1186/1471-2105-9-307). *BMC Bioinformatics*, 9, 307.

<a id="ref-4"></a>
[4] Fisher, A., Rudin, C. and Dominici, F. (2019). [All Models are Wrong, but Many are Useful: Learning a Variable's Importance by Studying an Entire Class of Prediction Models Simultaneously](https://jmlr.org/papers/v20/18-760.html). *Journal of Machine Learning Research*, 20(177), 1–81.

<a id="ref-5"></a>
[5] Hooker, G., Mentch, L. and Zhou, S. (2021). [Unrestricted permutation forces extrapolation: variable importance requires at least one more model, or there is no free variable importance](https://doi.org/10.1007/s11222-021-10057-z). *Statistics and Computing*, 31, 82.

<a id="ref-6"></a>
[6] Ribeiro, M. T., Singh, S. and Guestrin, C. (2016). ["Why Should I Trust You?": Explaining the Predictions of Any Classifier](https://doi.org/10.1145/2939672.2939778). *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 1135–1144.

<a id="ref-7"></a>
[7] Lundberg, S. M. and Lee, S.-I. (2017). [A Unified Approach to Interpreting Model Predictions](https://papers.neurips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html). *Advances in Neural Information Processing Systems*, 30, 4765–4774.

<a id="ref-8"></a>
[8] Lundberg, S. M., Erion, G., Chen, H., DeGrave, A., Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N. and Lee, S.-I. (2020). [From local explanations to global understanding with explainable AI for trees](https://doi.org/10.1038/s42256-019-0138-9). *Nature Machine Intelligence*, 2, 56–67.

<a id="ref-9"></a>
[9] Covert, I., Lundberg, S. M. and Lee, S.-I. (2020). [Understanding Global Feature Contributions With Additive Importance Measures](https://papers.neurips.cc/paper/2020/hash/c7bf0b7c1a86d5eb3be2c722cf2cf746-Abstract.html). *Advances in Neural Information Processing Systems*, 33.

<a id="ref-10"></a>
[10] Sundararajan, M., Taly, A. and Yan, Q. (2017). [Axiomatic Attribution for Deep Networks](https://proceedings.mlr.press/v70/sundararajan17a.html). *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70, 3319–3328.

<a id="ref-11"></a>
[11] Bach, S., Binder, A., Montavon, G., Klauschen, F., Müller, K.-R. and Samek, W. (2015). [On Pixel-Wise Explanations for Non-Linear Classifier Decisions by Layer-Wise Relevance Propagation](https://doi.org/10.1371/journal.pone.0130140). *PLoS ONE*, 10(7), e0130140.

<a id="ref-12"></a>
[12] Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D. and Batra, D. (2020). [Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization](https://doi.org/10.1007/s11263-019-01228-7). *International Journal of Computer Vision*, 128, 336–359.

<a id="ref-13"></a>
[13] Sobol', I. M. (2001). [Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates](https://doi.org/10.1016/S0378-4754(00)00270-6). *Mathematics and Computers in Simulation*, 55(1–3), 271–280.

<a id="ref-14"></a>
[14] Friedman, J. H. and Popescu, B. E. (2008). [Predictive learning via rule ensembles](https://doi.org/10.1214/07-AOAS148). *The Annals of Applied Statistics*, 2(3), 916–954.

<a id="ref-15"></a>
[15] Apley, D. W. and Zhu, J. (2020). [Visualizing the effects of predictor variables in black box supervised learning models](https://doi.org/10.1111/rssb.12377). *Journal of the Royal Statistical Society: Series B*, 82(4), 1059–1086.

<a id="ref-16"></a>
[16] Candès, E., Fan, Y., Janson, L. and Lv, J. (2018). [Panning for gold: 'model-X' knockoffs for high dimensional controlled variable selection](https://doi.org/10.1111/rssb.12265). *Journal of the Royal Statistical Society: Series B*, 80(3), 551–577.

<a id="ref-17"></a>
[17] Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. and Wasserman, L. (2018). [Distribution-Free Predictive Inference for Regression](https://doi.org/10.1080/01621459.2017.1307116). *Journal of the American Statistical Association*, 113(523), 1094–1111.

<a id="ref-18"></a>
[18] Williamson, B. D., Gilbert, P. B., Simon, N. R. and Carone, M. (2023). [A General Framework for Inference on Algorithm-Agnostic Variable Importance](https://doi.org/10.1080/01621459.2021.2003200). *Journal of the American Statistical Association*, 118(543), 1645–1658.

<a id="ref-19"></a>
[19] Kumar, I. E., Venkatasubramanian, S., Scheidegger, C. and Friedler, S. (2020). [Problems with Shapley-value-based explanations as feature importance measures](https://proceedings.mlr.press/v119/kumar20e.html). *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119, 5491–5500.

<a id="ref-20"></a>
[20] Jethani, N., Sudarshan, M., Covert, I., Lee, S.-I. and Ranganath, R. (2022). [FastSHAP: Real-Time Shapley Value Estimation](https://arxiv.org/abs/2107.07436). *International Conference on Learning Representations*.

<a id="ref-21"></a>
[21] Peng, H., Long, F. and Ding, C. (2005). [Feature Selection Based on Mutual Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy](https://doi.org/10.1109/TPAMI.2005.159). *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 27(8), 1226–1238.

<a id="ref-22"></a>
[22] Tibshirani, R. (1996). [Regression Shrinkage and Selection via the Lasso](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x). *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.

<a id="ref-23"></a>
[23] Verdinelli, I. and Wasserman, L. (2024). [Decorrelated Variable Importance](https://www.jmlr.org/papers/volume25/22-0801/22-0801.pdf). *Journal of Machine Learning Research*, 25(7), 1–27.

<a id="ref-24"></a>
[24] Chen, H., Covert, I. C., Lundberg, S. M. and Lee, S.-I. (2023). [Algorithms to estimate Shapley value feature attributions](https://doi.org/10.1038/s42256-023-00657-x). *Nature Machine Intelligence*, 5, 590–601.

<a id="ref-25"></a>
[25] Bilodeau, B., Jaques, N., Koh, P. W. and Kim, B. (2024). [Impossibility theorems for feature attribution](https://doi.org/10.1073/pnas.2304406120). *Proceedings of the National Academy of Sciences*, 121(2), e2304406120.

---

## Appendix A. Terminology

- **ALE**: Accumulated local effects, a curve built from local differences within a neighborhood instead of an average of the model over the marginal distribution.
- **Family**: One of the seven top-level groups of Fig 1, holding the methods whose number is computed from the same source. Used here because the literature names the axes rather than the groups, and has no settled word for this level.
- **FAST**: Fourier amplitude sensitivity test, an estimator of the Sobol indices based on frequency analysis of the model output.
- **FDR**: False discovery rate, the expected fraction of the selected set that is spurious.
- **HSIC**: Hilbert-Schmidt independence criterion, a dependence measure computed in a kernel feature space.
- **LIME**: Local interpretable model-agnostic explanations, a sparse linear model fitted to the black box in the neighborhood of one row.
- **LOCO**: Leave out covariates, the increase in prediction error when a model is refitted without the feature.
- **LRP**: Layer-wise relevance propagation, the redistribution of output relevance back to the inputs by per-layer rules.
- **MDA**: Mean decrease in accuracy, the loss increase when one column is randomly shuffled.
- **MDI**: Mean decrease in impurity, the total impurity reduction over the splits a tree ensemble makes on the feature.
- **mRMR**: Minimum redundancy maximum relevance, a greedy selection criterion that subtracts redundancy with the already selected features from relevance to the target.
- **PDP**: Partial dependence plot, the model output averaged over the marginal distribution of the other features.
- **SAGE**: Shapley additive global importance, the Shapley value of the model's loss rather than of one prediction.
- **SHAP**: Shapley additive explanations, the family of local attributions that estimate Shapley values of a prediction.
- **Shapley value**: The unique division of a payoff among players that satisfies efficiency, symmetry, the null player property and linearity.
- **Total-effect index**: The share of output variance attributable to a feature together with every interaction it takes part in.
- **VIM**: Variable importance measure, used here for a population contrast in predictiveness estimated with the model as a nuisance.
- **VIP**: Variable importance in projection, the contribution of a feature to the latent components of a partial least squares model.
