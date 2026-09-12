# Feature Selection
Rev. 0 | Created: 2026-09-12 | Updated: 2026-09-12 17:55 CDT

> This folder collects the methods that choose which input columns a model is given.
> The documents divide by what the selection looks at: one feature at a time, several features together, or the trace columns a semiconductor equipment produces.

## 1. Scope

A selection method is fixed by two decisions — whether a feature is scored alone or as part of a subset, and when the model is consulted. The documents here are organized around those two decisions rather than around a ranking of methods, because the same dataset admits several methods and the cost is what separates them.

## 2. Documents

Table 1. Documents in this folder

| Document | Description |
|---|---|
| [univariate-feature-selection/univariate-feature-selection-ko.md](univariate-feature-selection/univariate-feature-selection-ko.md) | It scores each feature alone against the target, and gives the statistical test for each combination of continuous and categorical data, the cut-off rules, a scikit-learn example, and the two blind spots that follow from looking at one feature at a time. |
| [multivariate-feature-selection/multivariate-feature-selection-ko.md](multivariate-feature-selection/multivariate-feature-selection-ko.md) | It scores features as a subset, splits the methods into multivariate filter, wrapper and embedded, compares their cost and their grip on interaction, and closes on a three-step workflow that puts the expensive method last. |
| [trace-feature-selection/trace-feature-selection.md](trace-feature-selection/trace-feature-selection.md) | It places the methods that choose which of a semiconductor equipment trace's features move the target on three axes — when the model is consulted, the unit the selection is made at, and whether the selection survives a change of wafers — and closes on a question-to-branch guide. |
| [trace-feature-selection/trace-feature-selection-ko.md](trace-feature-selection/trace-feature-selection-ko.md) | It is the Korean edition of the trace feature selection document, carrying the same hierarchy, axes, tables and references. |

## 3. Order Of Use

1. If the features are still to be screened for the first time, or the dataset is wide enough that a per-feature pass is all that fits, read the univariate document.
2. If a screened set still carries duplicated columns, or a pair of features is suspected to matter only together, read the multivariate document.
3. If the columns come from equipment trace and the selected set has to name a sensor or a step an engineer can act on, read the trace document.

The documents are cumulative rather than exclusive. The workflow of the multivariate document starts from a univariate pass, and the trace document places both of them as branches of a wider map.
