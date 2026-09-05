# R² (Coefficient of Determination)
Rev. 2 | Created: 2026-09-04 | Updated: 2026-09-04 23:02 CDT

> This folder covers the one metric that reports a fit as a fraction of variance, and what that
> fraction is actually a fraction of.
> Its documents work outward from the algebra of the ratio to the cases where the denominator has
> to be replaced or the whole value has to be reported as a distribution.

## 1. Scope

R² is a ratio of two sums of squares, and everything difficult about it follows from the
denominator being computed from the data rather than stated. The numerator is the error the model
left, which is a property of the model; the denominator is the error a baseline left, which is a
property of the dataset. A single value therefore mixes two things, and the same model reads high
on a wide dataset and low on a narrow one.

Every document here follows from that split. Two of them stay inside the standard definition and
ask what moves it: what the variance components do to the ratio, and where the metric sits among
the alternatives that answer a different question. The other three change something. One replaces
the denominator with a stated baseline so the yardstick stops moving. One computes the ratio once
per posterior draw so the answer arrives with an interval. One measures the relationship between
R² and a percentage error metric and finds no fixed one.

The material is written for regression models evaluated on engineering data, so the worked
examples are virtual metrology, sensor traces and process lots.

## 2. Documents

Table 1. Documents in this folder

| Document | Language | Description |
|---|---|---|
| [the-impact-of-variance-components-on-the-coefficient-of-determination-r2.md](the-impact-of-variance-components-on-the-coefficient-of-determination-r2.md) | English | It partitions the total variation into the explained and the residual part, then shows what each one does to the ratio: raising the residual variance drives R² down, and widening the predictor variance drives it up with the relationship unchanged. It closes by sweeping the coefficient of variation of a sample on the 1-to-1 line, which shows the same dependence from the data side. |
| [mean-variance-and-agreement-metrics-for-regression-in-ai-ml.md](mean-variance-and-agreement-metrics-for-regression-in-ai-ml.md) | English | It places R² in a taxonomy of regression metrics divided into mean-based, variance-based and agreement-based families, reads each against the $y=x$ line, and sets out the low variance effect that makes the variance-based family collapse on a nearly constant signal. |
| [R2-Denominator/r2-denominator.md](R2-Denominator/r2-denominator.md) | English, [Korean](R2-Denominator/r2-denominator-ko.md) | It reads the denominator as the error of a baseline, so that stating it is choosing that baseline. It gives three ways to state it — the training mean, a fixed reference dispersion such as a spec spread, and a baseline model answering per sample — shows that a fixed dispersion reduces to one minus the squared ratio of the root mean squared error to the spec, and sets out what has to be reported alongside the value. |
| [Bayesian-R2/bayesian-r2.md](Bayesian-R2/bayesian-r2.md) | English, [Korean](Bayesian-R2/bayesian-r2-ko.md) | It computes one R² per posterior draw to obtain R² as a distribution, uses the Gelman form whose denominator is the sum of the explained and the residual variance so no draw leaves [0, 1], and reads the resulting credible interval as the confidence in the explanatory power. A worked example on eight points carries every number. |
| [R2-MAPE/r2-vs-mape.md](R2-MAPE/r2-vs-mape.md) | Korean | It asks whether R² and the mean absolute percentage error convert into each other and finds that they do not. The monotone trend always holds, but the curve itself depends on what the data generation held fixed, and three designs give three different expressions. |

The three documents that carry figures keep the script that produced them alongside, so those
numbers can be traced to a run rather than to a value copied from elsewhere.

## 3. Order Of Use

1. Read what the ratio is made of first. The explained and the residual sums of squares move
   independently, and a value that fell may mean a noisier process rather than a worse model.
2. Check the spread of the data before quoting the number. A sample with little variation gives a
   low R² to a prediction that is physically close, which is the low variance effect rather than a
   finding about the model.
3. Choose the metric family by the question. R² answers how much of the variation was explained;
   an error in the unit of the measurement answers how far the prediction is from the truth, and
   neither substitutes for the other.
4. State the denominator when comparing across datasets. The standard denominator is recomputed
   for every dataset, so lots and periods ranked by it are ranked by their own spread.
5. Report a distribution rather than a point where posterior draws exist. The interval is what
   says whether the value would survive a different sample.

Steps 1 and 2 come before any use of the metric. Steps 3 to 5 are independent of each other and
are adopted as the evaluation requires.
