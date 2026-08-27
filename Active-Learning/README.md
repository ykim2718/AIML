# Active-Learning
Rev. 1 | Created: 2026-08-27 | Updated: 2026-08-27 20:08 UTC

> This folder covers how to decide which experiment to run next when one label costs a wafer, a tool hour or a destructive test.
> Its documents answer what range those experiments should cover and what a model trained on that range is allowed to predict.

## 1. Scope

Where a single label is expensive, the choice of what data to collect moves the result more than the choice of model does. Semiconductor work is such a place. A thickness reading may cost the wafer it was taken from, and holding a tool at an off-target condition costs the runs it could have made instead. This folder holds the methods that pick the next experiment under that constraint, and the reasoning about what the resulting coverage does to the model.

## 2. Documents

Table 1. Documents in this folder

| Document | Description |
|----------|-------------|
| [DOE/wide-and-narrow-doe-for-semiconductor.md](DOE/wide-and-narrow-doe-for-semiconductor.md) | It separates what a wide experiment range and a narrow one around the production condition each contribute to training and to inference, works out why the production data never leaves the narrow range on its own, and places the method within the machine learning workflow. |
| [Bayesian-Optimization/bayesian-optimization-for-semiconductor.md](Bayesian-Optimization/bayesian-optimization-for-semiconductor.md) | It covers the loop that picks the next process condition from a surrogate rather than from a fixed design, compares it with classical DOE, and sets out the five constraints a fab adds to it, of which the batch and the safety limit depart furthest from the textbook. |

## 3. Order Of Use

1. Fix what the model is for before the range is chosen. A model that predicts inside the production window and a model that must recognize a departure from it do not need the same coverage.
2. Cover the wide range first, so that the production window sits inside the learned range rather than at its edge.
3. Fill the production window densely and fine-tune on it, since the wide range is too sparse to resolve the variation that happens there.
4. Choose each further point by what is still missing. Reduce the model error with active learning, or find the best condition with Bayesian optimization, but not both with the same criterion.
