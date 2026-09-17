# Feature Selection
Rev. 6 | Created: 2026-09-12 | Updated: 2026-09-17 09:26 CDT

> This folder collects the methods that choose which input columns a model is given.
> The documents divide by what the selection looks at: one feature at a time, several features together, or the trace columns a semiconductor equipment produces.

## 1. Scope

A selection method is fixed by two decisions — whether a feature is scored alone or as part of a subset, and when the model is consulted. The documents here are organized around those two decisions rather than around a ranking of methods, because the same dataset admits several methods and the cost is what separates them.

## 2. Documents

Table 1. Documents in this folder

| Document | Description |
|---|---|
| [univariate-feature-selection/univariate-feature-selection-ko.md](univariate-feature-selection/univariate-feature-selection-ko.md) | It scores each feature alone against the target, and gives the statistical test for each combination of continuous and categorical data, the cut-off rules, and the two blind spots that follow from looking at one feature at a time; an appendix links the `SelectKBest` selector class kept in `src/`, which runs over the iris dataset under each of the four metrics, named by a Literal, and prints every feature score beside the two features that metric keeps. |
| [multivariate-feature-selection/multivariate-feature-selection.md](multivariate-feature-selection/multivariate-feature-selection.md) | It scores features as a subset, splits the methods into multivariate filter, wrapper and embedded, sets that split against the interaction hierarchy, the kind of target and the instability of a single run, and closes on a four-step workflow that puts the expensive method last; an appendix links that workflow as one selector class kept in `src/`, run over the breast cancer dataset, where every step names its method with a Literal declared on the class — four filters, five embedded models, four wrapper searches, all implemented and all reachable from run — and charts which features each of the thirteen keeps, cutting 31 features to 5. |
| [multivariate-feature-selection/multivariate-feature-selection-ko.md](multivariate-feature-selection/multivariate-feature-selection-ko.md) | It is the Korean edition of the multivariate feature selection document, carrying the same hierarchies, tables, workflow and selector class. |
| [trace-feature-selection/trace-feature-selection.md](trace-feature-selection/trace-feature-selection.md) | It places the methods that choose which of a semiconductor equipment trace's features move the target on three axes — when the model is consulted, the unit the selection is made at, and whether the selection survives a change of wafers — and closes on a question-to-branch guide. |
| [trace-feature-selection/trace-feature-selection-ko.md](trace-feature-selection/trace-feature-selection-ko.md) | It is the Korean edition of the trace feature selection document, carrying the same hierarchy, axes, tables and references. |

## 3. Order Of Use

1. If the features are still to be screened for the first time, or the dataset is wide enough that a per-feature pass is all that fits, read the univariate document.
2. If a screened set still carries duplicated columns, or a pair of features is suspected to matter only together, read the multivariate document.
3. If the columns come from equipment trace and the selected set has to name a sensor or a step an engineer can act on, read the trace document.

The documents are cumulative rather than exclusive. The workflow of the multivariate document starts from a univariate pass, and the trace document places both of them as branches of a wider map.
