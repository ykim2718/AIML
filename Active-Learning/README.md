# Active-Learning
Rev. 2 | Created: 2026-08-27 | Updated: 2026-08-27 15:12 CDT
> This folder covers how to decide which experiment to run next when one label costs a wafer, a tool hour or a destructive test.
> Its documents answer what range those experiments should cover and what a model trained on that range is allowed to predict.

## 1. Scope

Where a single label is expensive, the choice of what data to collect moves the result more than the choice of model does. Semiconductor work is such a place. A thickness reading may cost the wafer it was taken from, and holding a tool at an off-target condition costs the runs it could have made instead. This folder holds the methods that pick the next experiment under that constraint, and the reasoning about what the resulting coverage does to the model.

## 2. Position in Machine Learning

Design of experiments is not an algorithm or a model structure inside machine learning. It is the method that designs and collects the data a model is given, and in the workflow it belongs to the data acquisition strategy rather than to the modelling. Where the statistical method meets machine learning it splits three ways.

Table 1. DOE in the machine learning workflow

| Field | What it chooses | Relation to DOE |
|-------|-----------------|-----------------|
| Active learning | The sample to label next | It picks the point that reduces the model error fastest |
| Bayesian optimization | The condition to try next | It finds the optimum with a surrogate and an acquisition function |
| Data-centric AI | The composition of the data itself | It covers the space evenly so that the model generalizes |

The first two are the ones that decide a single next point rather than a whole design up front, and this folder is organized around them: active learning names the folder itself, Bayesian optimization one of its subfolders, and the DOE subfolder holds the fixed design both of them start from. Neither of the two ranks above the other. Both run a surrogate and an acquisition function, and they differ only in what they are trying to get: active learning picks the point that improves the model, Bayesian optimization picks the point that improves the process. The third is a standing constraint on both rather than a method that picks points, so it runs through the documents instead of holding a folder of its own.

## 3. Documents

Table 2. Documents in this folder

| Document | Description |
|----------|-------------|
| [DOE/wide-and-narrow-doe-for-semiconductor.md](DOE/wide-and-narrow-doe-for-semiconductor.md) | It separates what a wide experiment range and a narrow one around the production condition each contribute to training and to inference, and works out why the production data never leaves the narrow range on its own. |
| [Bayesian-Optimization/bayesian-optimization-for-semiconductor.md](Bayesian-Optimization/bayesian-optimization-for-semiconductor.md) | It covers the loop that picks the next process condition from a surrogate rather than from a fixed design, compares it with classical DOE, and sets out the five constraints a fab adds to it, of which the batch and the safety limit depart furthest from the textbook. |

## 4. Order Of Use

1. Fix what the model is for before the range is chosen. A model that predicts inside the production window and a model that must recognize a departure from it do not need the same coverage.
2. Cover the wide range first, so that the production window sits inside the learned range rather than at its edge.
3. Fill the production window densely and fine-tune on it, since the wide range is too sparse to resolve the variation that happens there.
4. Choose each further point by what is still missing, taking the method from Table 1 that matches the goal. Do not run both criteria at once.
