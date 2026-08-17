# Referenced R² — Choosing the Baseline in the R² Denominator
Rev. 10 | Created: 2026-08-15 | Updated: 2026-08-17 00:01 CDT

> This document asks whether the denominator of the standard R² can be replaced
> by a reference stated from outside rather than computed from the data, and
> what the number comes to mean once it is.

## 1. Question

The standard R² is defined as follows.

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

The denominator `SS_tot` is the total sum of squares taken about the mean
`y_bar` of the data itself, and it is the quantity usually called the total
spread of the data. Whether that denominator can be replaced by a value stated
from outside instead of computed from the data is the question of this document.

The answer is that it can. Not only is it natural in principle, several
established forms already exist. But the meaning of the metric changes the
moment the denominator does, so what it was replaced by must always be reported
with it.

A metric built by stating the denominator this way is called a **Referenced R²**
in this document and written `Ref_R2`. Section 3 divides the individual variants
by what goes into the denominator: `R2_oos` (out-of-sample), `R2_frd` (fixed
reference dispersion), and `R2_base` (baseline model).

## 2. Structure

Why the denominator may be changed becomes clear on rereading R². `SS_tot` is
exactly the sum of squared errors of a model that always predicts `y_bar`. That
is, R² had the form below from the start.

```text
                error left by the model
R2 = 1 -  ───────────────────────────────────
            error left by a chosen baseline
```

The standard R² is the special case that fixes the baseline to a model which
only ever returns the mean, and this structure is widely used under the name
skill score. **Stating the denominator is changing the baseline, no more and no
less.** This fact governs the rest of the document.

`Ref_R2` is therefore not a new calculation but an R² whose baseline was chosen
explicitly, and the standard R² is the case that left that baseline implicitly
at the mean of the data.

Numerator and denominator are both in units of y squared, so the ratio is
dimensionless, and `Ref_R2` reads as what percentage of the error the baseline
had to accept was removed by the model. That reading holds whatever the baseline
is.

## 3. Variants

**Table 1. Ways to choose the baseline**

| Symbol | Denominator | Baseline it encodes | Typical use | Reference |
|---|---|---|---|---|
| `R²` | `Σ(y_i − y_bar)²` | The mean of this dataset | Goodness of fit within one dataset | [1](#ref-1) |
| `R2_oos` | `Σ(y_i − y_train_bar)²` | The mean known at training time | Test set evaluation | [2](#ref-2) |
| `R2_frd` | `N · sigma_ref²` | The spread the spec allows | Comparison across lots, batches, periods | [3](#ref-3), [4](#ref-4) |
| `R2_base` | `Σ(y_i − y_base_i)²` | A reference model answering per sample | Time series, comparison against a model in production | [5](#ref-5) |

The Reference column points at where each form is actually in use, and the
bibliography is in [References](#references). None of the four is a calculation
this document invented; each is already standard in its own field. What this
document added is notation rather than arithmetic, and section 4 returns to
that.

### 3.1. Fixed Reference Point

This is the most common case. When a test set is evaluated, the reference in the
denominator is anchored to the mean of the training data, `y_train_bar`, rather
than to the mean of the test data.

$$R^2_{oos} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y}_{train})^2}$$

Two reasons make this the right form. First, the mean of the test data is not
knowable at evaluation time, so putting it in the baseline amounts to borrowing
information from the future. Second, the denominator moves with how the test set
was cut, which leaves the yardstick itself unsteady. Anchoring to the training
mean turns the metric into the question of how much the model improved on one
that keeps returning the mean it knew during training, and the yardstick then
holds however the data was split.

This form is known outside as the out-of-sample R², so external reports use that
name.

### 3.2. Fixed Reference Dispersion

The denominator is replaced by a known reference variance instead of being
computed from the data.

$$R^2_{frd} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{N \cdot \sigma_{ref}^2}$$

For `sigma_ref` one uses a domain reference such as the spread a process spec
allows, the variance of accumulated historical data, or the reference variance
of a metrology system. For a thickness prediction, putting the control spec
spread of the layer in question into the denominator turns the metric into a
measure of the model against a fixed yardstick.

Because the denominator is a constant, the expression reduces one step further.
The square root of the numerator `SS_res` divided by the sample count `N` is the
root mean squared error, RMSE.

$$\mathrm{RMSE} = \sqrt{\frac{SS_{res}}{N}}$$

RMSE returns the error to the same unit as y, so it can be divided by
`sigma_ref` directly. Substituting `N · RMSE²` for `SS_res` cancels `N` and leaves only the
following.

$$R^2_{frd} = 1 - \left(\frac{\mathrm{RMSE}}{\sigma_{ref}}\right)^2$$

This is the physical meaning of the variant. `R2_frd` **measures how many times
the spec spread the model error is** and subtracts that from 1. Half the spec
gives 0.75, error equal to the spec gives 0, and error beyond the spec gives a
negative value. Put in the language of baselines, the reference is an imaginary
model whose error is exactly what the spec allows, and the question is whether
the model beat it.

Metrology has a practice of reporting the ratio of error to spec as it is rather
than dressing it up as an R². When the reader is a process or metrology
engineer, handing over `RMSE / sigma_ref` unchanged communicates better; when
the number has to sit in one table beside other metrics, converting it to
`R2_frd` puts it on the 0-to-1 scale. The two convert to each other by the
expression above, so they carry the same information.

### 3.3. Baseline Model

The baseline in section 3.1 answers with one number, `y_train_bar`, whatever
sample it meets, and the baseline in section 3.2 makes an error of `sigma_ref`
whatever sample it meets. Both are constant baselines whose reference does not
vary with the sample.

A case where the reference itself has to vary per sample cannot be written in
that form. There the baseline holds its own value `y_base_i` for each i.

$$R^2_{base} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - y_{base,i})^2}$$

The persistence baseline of a time series is the representative case. It uses
the value at the previous step as the prediction for the next, so `y_base_i`
differs for each i and no constant can produce the same denominator. The same
holds when a seasonal mean serves as the baseline, and when a model currently in
production serves as the baseline to measure the value of replacing it.

Setting `y_base_i` to the constant `y_train_bar` reduces this to section 3.1, so
that section is a special case of this one. They are kept apart because the two
answer different questions. Section 3.1 asks whether the model beats the mean;
section 3.3 asks whether it beats what is already in use. The first question
decides whether a model is worth anything, the second whether it is worth a
replacement.

## 4. Meaning and Reporting

💡 The real reason for fixing the denominator is not convenience but that **the
standard R² cannot be used to compare across datasets**. The denominator is
recomputed for every dataset, so the yardstick changes with the dataset.

The problem is plain in a case where one model applied to three lots gives the
same RMSE of 0.5 nm in all three.

**Table 2. The same model on three lots, RMSE fixed at 0.5 nm**

| Lot | Lot dispersion (nm) | Standard R² | `R2_frd` against a spec of 1.0 nm |
|---|---|---|---|
| A | 2.0 | 0.9375 | 0.7500 |
| B | 1.2 | 0.8264 | 0.7500 |
| C | 0.6 | 0.3056 | 0.7500 |

The prediction error of the model is identical across the three lots, yet the
standard R² spreads from 0.94 down to 0.31. What changed is the denominator, not
the model. A lot with a large spread is one the baseline was already failing on,
so the same performance looks good; a lot with a small spread is one the
baseline was already handling, so the same performance looks bad. The standard
R² is only reporting this honestly, and ranking lots by this value ranks the
spread of the lots rather than the model.

Fixing the denominator at 1.0 nm gives 0.75 for all three, and the metric then
reflects the performance of the model alone. Put differently, the two metrics
answer different questions.

- Standard R² — how much better than the mean the model is within this dataset. The yardstick differs per dataset.
- `R2_frd` — how much better than a stated yardstick the model is. It can be compared across datasets.

Which one is right depends on the question. Asking how much of this lot was
explained calls for the standard R²; asking whether the model can be deployed
across the line calls for `R2_frd`.

A single value handed over on its own is therefore not enough. The answers to
both questions land in the same place in the same shape, so the recipient cannot
tell which one was seen. Reporting follows the rules below.

- Use the symbol that matches the variant rather than `R²` itself. The same symbol invites the reader to take it for the standard R².
- The symbol alone does not carry enough, so record the reference of the denominator together with its value and its source. Write `R2_frd = 0.75 (reference: spec tolerance sigma = 1.0 nm)` rather than the number alone.
- Report the standard R² alongside. The gap between the two is what tells the reader which way the spread of that dataset leans against the spec.
- Leave the grounds for the choice of `sigma_ref`. A denominator chosen without grounds makes the whole metric arbitrary.

`Ref_R2`, `R2_frd`, and `R2_base` are names set by this document and do not
carry outside it. When sending them out, attach a one-line definition after the
symbol, or use the established name where one exists, as in section 3.1.

## 5. Cautions

- **Negative values occur.** When the model is worse than the baseline, `SS_res` exceeds the denominator and the value drops below 0. Against a spec of 1.0 nm, an RMSE of 1.3 nm gives −0.69. This is not an error but the normal signal that the baseline is the better thing to use, and it is common in out-of-sample evaluation.
- **A small denominator makes the value explode.** Applying this metric to an item whose spec spread is very small produces a large negative value from a small error. Such items are better managed by the absolute size of the error than by anything in the R² family.
- **The upper bound is still 1.** Changing the denominator cannot push `SS_res` below 0, so the value never exceeds 1. A value near 1 does not mean the model is perfect; it means the stated baseline was overwhelmed.
- **The denominator is not chosen after the fact.** Picking a flattering `sigma_ref` after seeing the results turns the metric into rhetoric. Fix the reference before the evaluation and record it.

## 6. Summary

The denominator of R² is the error of a baseline, and the standard R² is the
special case that fixes that baseline to the mean of the data. Replacing the
denominator with the training mean, with a spec spread, or with the error of a
model in production are all the same operation of changing the baseline, and the
metric shifts in meaning from explanatory power within this dataset to
performance relative to a stated reference. This document calls the forms whose
baseline is chosen explicitly `Ref_R2` and divides them into `R2_oos`, `R2_frd`,
and `R2_base`. What the exchange buys is a yardstick that stops moving with the
dataset, so lots and periods become comparable; what it costs is the
interpretation the standard R² carried and the responsibility for choosing the
reference.

## References

<a id="ref-1"></a>[1] Kvålseth, T. O. (1985). [Cautionary Note about R²](https://doi.org/10.1080/00031305.1985.10479448). *The American Statistician*, 39(4), 279–285. Sets out the interpretive limits the standard R² inherits from having its denominator tied to the mean of the data, which is what makes the variants in section 3 necessary.

<a id="ref-2"></a>[2] Campbell, J. Y., & Thompson, S. B. (2008). [Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?](https://doi.org/10.1093/rfs/hhm055) *The Review of Financial Studies*, 21(4), 1509–1531. Uses an out-of-sample R² whose denominator is fixed to the mean of the training window as its evaluation metric. It is the form section 3.1 describes, used as is.

<a id="ref-3"></a>[3] Murphy, A. H. (1988). [Skill Scores Based on the Mean Square Error and Their Relationships to the Correlation Coefficient](https://doi.org/10.1175/1520-0493%281988%29116%3C2417%3ASSBOTM%3E2.0.CO%3B2). *Monthly Weather Review*, 116(12), 2417–2424. Formalizes the skill score as mean squared error divided by a reference error, and shows the reference may be an external value such as a historical mean. It is the basis for the structure section 2 describes.

<a id="ref-4"></a>[4] Automotive Industry Action Group (2010). [Measurement Systems Analysis Reference Manual](https://www.aiag.org/training-and-resources/manuals/details/MSA-4), 4th ed. ISBN 978-1-60534-211-5. Defines the precision-to-tolerance ratio, measurement error divided by the spec tolerance, as the acceptance criterion for a measurement system. It is the `RMSE / sigma_ref` of section 3.2 as metrology uses it.

<a id="ref-5"></a>[5] Hyndman, R. J., & Koehler, A. B. (2006). [Another Look at Measures of Forecast Accuracy](https://doi.org/10.1016/j.ijforecast.2006.03.001). *International Journal of Forecasting*, 22(4), 679–688. Sets out how dividing forecast error by the error of a persistence baseline puts different series on one scale. It is the standard example of the per-sample baseline section 3.3 covers.

---

## Appendix A. Terminology

- **baseline model** — the model that serves as the reference for comparison. The denominator of R² is its sum of squared errors.
- **out-of-sample** — evaluation on data that was not used for training.
- **persistence baseline** — a time series baseline that uses the value at the previous step as the prediction for the next.
- **Referenced R² (`Ref_R2`)** — the collective name, set by this document, for an R² computed with the baseline in its denominator stated from outside.
- **RMSE** — root mean squared error, computed as `sqrt(SS_res / N)`. It returns the error to the same unit as y, so it can be set against a spec spread directly.
- **skill score** — the collective name for metrics of the form `1 − model error / baseline error`. R² is the case whose baseline is the mean.
- **spec tolerance** — the spread a process spec allows. It is often used as a fixed denominator.
- **SS_res** — the residual sum of squares, `Σ(y_i − y_hat_i)²`.
- **SS_tot** — the total sum of squares, `Σ(y_i − y_bar)²`, which is the denominator of the standard R².

## Appendix B. Reference Implementation

Each of the three metrics defined in section 3 is carried into one function.
`r2_oos` takes `R2_oos` from 3.1, `r2_frd` takes `R2_frd` from 3.2, and
`r2_base` takes `R2_base` from 3.3.

All three receive the observations `y_true` and the predictions `y_pred` and
return one skill value; the single place they differ is where the denominator
comes from. `r2_oos` receives the training mean `y_train_bar`, `r2_frd` the
reference dispersion `sigma_ref`, and `r2_base` the baseline prediction vector
`y_base`, each as an argument, and each builds its denominator from it. None of
the three derives the reference from `y_true`; all take it from outside, which
is what section 2 called choosing the baseline explicitly, carried into code. The
numerator `SS_res` is identical for all three, so it is pulled out into
`_ss_res`, which keeps the shape check and the empty-array check in one place.

Defining `r2_base` first and having `r2_oos` call it carries the last paragraph
of section 3.3 into code. The denominator of `r2_frd`, by contrast, is not the
residual of any prediction vector and so cannot be expressed through `r2_base`;
it is computed separately.

```python
# Python
import numpy as np


def _ss_res(y_true: np.ndarray = None, y_pred: np.ndarray = None) -> float:
    """Residual sum of squares, shared by the numerator and by baseline denominators."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shapes must match, got y_true {y_true.shape} and y_pred {y_pred.shape}.")
    if y_true.size == 0:
        raise ValueError("y_true is empty; a skill score needs at least one observation.")

    return float(np.sum((y_true - y_pred) ** 2))


def r2_base(y_true: np.ndarray = None, y_pred: np.ndarray = None, y_base: np.ndarray = None) -> float:
    """Section 3.3 — skill against a baseline that answers per sample.

    Args:
        y_true: observations, shape (n,).
        y_pred: predictions, shape (n,).
        y_base: baseline predictions, shape (n,), one value per sample.

    Returns:
        The skill relative to y_base. Negative when the baseline is the better predictor.
    """
    ss_base = _ss_res(y_true=y_true, y_pred=y_base)
    if ss_base == 0.0:
        raise ValueError("the baseline reproduces y_true exactly, so there is no error to improve on.")

    return 1.0 - _ss_res(y_true=y_true, y_pred=y_pred) / ss_base


def r2_oos(y_true: np.ndarray = None, y_pred: np.ndarray = None, y_train_bar: float = None) -> float:
    """Section 3.1 — skill against the mean that was known at training time.

    Args:
        y_true: observations, shape (n,).
        y_pred: predictions, shape (n,).
        y_train_bar: mean of the training targets. Never derived from y_true, which would
            put the test mean into the baseline and leak the evaluation set.

    Returns:
        The skill relative to always predicting y_train_bar.
    """
    if y_train_bar is None:
        raise ValueError("y_train_bar is required; taking the mean of y_true would leak the test set.")

    y_base = np.full(shape=y_true.shape, fill_value=float(y_train_bar))
    return r2_base(y_true=y_true, y_pred=y_pred, y_base=y_base)


def r2_frd(y_true: np.ndarray = None, y_pred: np.ndarray = None, sigma_ref: float = None) -> float:
    """Section 3.2 — skill against a fixed reference dispersion instead of the data spread.

    Args:
        y_true: observations, shape (n,).
        y_pred: predictions, shape (n,).
        sigma_ref: reference dispersion in the unit of y, strictly positive.

    Returns:
        The skill relative to a baseline whose error scale is sigma_ref.
        Equal to 1 - (RMSE / sigma_ref) ** 2.
    """
    if sigma_ref is None or sigma_ref <= 0.0:
        raise ValueError(f"sigma_ref must be a positive dispersion, got {sigma_ref}.")

    # The denominator is a stated error scale, not the residual of any prediction vector,
    # so it cannot be routed through r2_base.
    ss_ref = float(y_true.size) * sigma_ref ** 2
    return 1.0 - _ss_res(y_true=y_true, y_pred=y_pred) / ss_ref
```
