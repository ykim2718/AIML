# Single Predicted Value From Two Models
Rev. 22 | Created: 2026-09-11 | Updated: 2026-09-11 18:30 CDT

## 1. Purpose

- **Problem Statement**: No simple way to combine the predictions computed by several models into one.
- **Goal**: A single prediction that accounts for both probabilities, where two models each emit a prediction and a probability.
- **Non-Goal**: Fitting and improving the models themselves.

## 2. Summary

The single prediction comes from the two probabilities: `y_pred` is read off their weighted average, and a predicted value of any other kind is the two values averaged with those same probabilities as weights. Where both models T and S provide predicted probabilities ($p_T$, $p_S$), soft voting, a weighted average of the probabilities that reflects the confidence of each model precisely, is the most effective method. The weight w comes from a grid search on a validation dataset (section 4.1) rather than from a guess, and the value found is carried unchanged into the weighted sum on the test dataset.

## 3. Principle

### 3.1 Weighted Soft Voting

A weight w ($0 \le w \le 1$) applied to the probabilities $p_T$ and $p_S$ of the two models gives the hybrid probability $p_{\mathrm{hybrid}}$.

```math
p_{\mathrm{hybrid}} = w \cdot p_T + (1 - w) \cdot p_S \hspace{19em} (1)
```

The prediction `y_pred` of a binary classification is decided at the threshold 0.5.

```math
y_{\mathrm{pred}} =
\begin{cases}
1 & \mathrm{if}\ p_{\mathrm{hybrid}} \ge 0.5 \\
0 & \mathrm{otherwise}
\end{cases}
\hspace{15em} (2)
```

Each model's own predicted value takes no part in this: `y_pred` is read off $p_{\mathrm{hybrid}}$, and the share a model holds in it is what the weight w sets. Section 3.3 is the rule for the predicted values themselves.

### 3.2 Multi-Class Extension

In a multi-class classification with three or more classes, the per-class probability vectors $p_T$ and $p_S$ are summed with the weight, and `y_pred` is the class holding the highest probability.

```math
\mathbf{p}_{\mathrm{hybrid}} = w \cdot \mathbf{p}_T + (1 - w) \cdot \mathbf{p}_S \hspace{19em} (3)
```

```math
y_{\mathrm{pred}} = \arg\max \left( \mathbf{p}_{\mathrm{hybrid}} \right) \hspace{19em} (4)
```

Both inputs are arrays of shape (`N_samples`, `N_classes`).

### 3.3 Single Predicted Value

Where each model emits a predicted value of its own, the single value is the two values averaged with the probabilities as weights, so the model surer of its answer pulls the result toward it.

```math
v_{\mathrm{hybrid}} = \frac{w \cdot p_T \cdot v_T + (1 - w) \cdot p_S \cdot v_S}{w \cdot p_T + (1 - w) \cdot p_S} \hspace{15em} (5)
```

The denominator is the hybrid probability of equation (1), so the weighted soft voting of section 3.1 is what carries the two values here as well. The implementation is [Appendix B.1](#b1-single-predicted-value). It checks that both probabilities lie within 0~1 before weighing them, and it raises on a row where both are zero instead of returning a number.

## 4. Optimal Weight Search

Rather than fixing w arbitrarily, a search (grid search) for the w that raises a performance metric (R-squared, ROC-AUC, F1-score and the like) the most on the validation dataset is recommended.

### 4.1 Grid Search

A grid search finds the optimal weight w on the validation dataset against a metric the user defines, such as R-squared, F1-score, ROC-AUC or log loss. Scoring on `r2` needs the predicted value of each model, `v_T` and `v_S`, and every other metric refuses them. The search runs from 0.0 to 1.0, at a step whose default of 0.01 covers 100 intervals. The return is the optimal weight `best_w` to give model T and the metric score at that weight, the weight of model S being `1 - best_w`. The search function is [Appendix B.2](#b2-optimal-weight-search), and the run on synthetic data is [Appendix B.3](#b3-execution-example).

The optimal `best_w` found on the validation data is carried unchanged into the weighted sum on the test dataset for the final evaluation.

```math
p_{\mathrm{hybrid,test}} = w_{\mathrm{best}} \cdot p_{T,\mathrm{test}} + (1 - w_{\mathrm{best}}) \cdot p_{S,\mathrm{test}} \hspace{19em} (6)
```

### 4.2 Metric Selection

Metrics divide on what they hold against `y_true`. `r2` scores the single predicted value of equation (5) and is the one that aims at the deliverable directly; `F1-Score` and `Accuracy` score `y_pred`, the class the `threshold` produces, so the weight they choose moves `y_pred`; `ROC-AUC` and `Log Loss` score the hybrid probability itself and leave the chosen value alone.

Table 1. Metrics for the weight search

| Metric | Direction | Threshold | Compared with y_true |
| --- | --- | --- | --- |
| `r2` | Higher is better | Not used | `v_hybrid` of equation (5) |
| `f1` | Higher is better | Used | `y_pred`, the class from the threshold |
| `accuracy` | Higher is better | Used | `y_pred`, the class from the threshold |
| `roc_auc` | Higher is better | Not used | `p_hybrid` of equation (1) |
| `log_loss` | Lower is better | Not used | `p_hybrid` of equation (1) |

#### `r2`

Scores `v_hybrid` of equation (5) against `y_true`, which is the deliverable of this document.

- Strength: the only metric that moves the value the reader takes away; no threshold to fix; the same rule serves a continuous value and a class.
- Weakness: needs `v_T` and `v_S`; undefined on a row where both probabilities are zero, since that is the denominator of equation (5); squares its errors, so one wild row can decide w.

#### `f1`

Scores `y_pred`, the class the `threshold` produces, as the harmonic mean of precision and recall on the positive class.

- Strength: shows the positive class under an imbalanced label, where a majority guess cannot hide; reads as one number at the operating point actually shipped.
- Weakness: tied to the `threshold`, so a new threshold makes the chosen w stale; the negative class enters only through the errors it causes.

#### `accuracy`

Scores `y_pred`, the class the `threshold` produces, as the fraction of rows it gets right.

- Strength: the plainest reading of the result, and it counts both classes on the same footing.
- Weakness: an imbalanced label lifts it on the majority class alone; tied to the `threshold` in the same way `f1` is.

#### `roc_auc`

Scores `p_hybrid` of equation (1) as the ranking it induces, over every threshold at once.

- Strength: no threshold to fix, so it serves a search run before the operating point is chosen; steady under an imbalanced label.
- Weakness: unchanged by any monotone rescaling of the probability, so a badly calibrated model passes it; the single predicted value is not what it measures.

#### `log_loss`

Scores `p_hybrid` of equation (1) against `y_true`, penalizing the probability by how far it sits from the label.

- Strength: the only metric here that grades the scale of the probability, so it exposes the calibration section 5 warns about.
- Weakness: a confident mistake costs it enormously, so one row can decide w; a probability at 0 or 1 sends it to infinity unless it is clipped.

## 5. Cautions

The probability distributions of the two models have to be checked for a fine match. Where one model is too confident (for example, mostly near 0.05 or 0.95) and the other is cautious (for example, between 0.4 and 0.6), a plain weighted combination can give the confident model too much influence.

## 6. Further Work

- **Applying probability calibration**
  - What: calibration through `IsotonicRegression` or `Platt Scaling`, applied to both probabilities before the weighted sum.
  - Why now: under the condition of section 5, one model confident and the other cautious, a weight alone does not correct the imbalance in influence.
  - What is needed: data for the calibration, and a comparison of the optimal w and the metric scores before and after it.

## References

N/A — no external source is cited.

---

## Appendix A. Terminology

- **Grid Search**: A search that sweeps a fixed interval at a fixed step to find the optimal value.
- **Isotonic Regression**: A calibration method fitting predicted probabilities to the observed frequency under a monotone increasing constraint.
- **Log Loss**: A metric of the disagreement between the predicted probability and the label. Lower is better.
- **Platt Scaling**: A calibration method passing a model output through a logistic function to put it on a probability scale.
- **Probability Calibration**: The procedure of matching the probabilities a model emits to the observed frequency.
- **R-squared**: A metric of how much of the variance of the true value the prediction accounts for. Higher is better.
- **ROC-AUC**: A metric summarizing classification performance over every threshold.
- **Soft Voting**: A combination that averages the predicted probabilities of the models with weights to decide the final class.
- **Threshold**: The classification threshold that turns a probability into a final class.

## Appendix B. Python Implementation

### B.1 Single Predicted Value

```python
from typing import Union

import numpy as np

Probability = Union[float, np.ndarray]


def check_probability(p: Probability) -> np.ndarray:
    """Check that the probability lies within 0~1.

    p: predicted probability of a model

    >>> check_probability(np.array([0.9, 0.5, 0.2]))
    array([0.9, 0.5, 0.2])
    >>> check_probability(np.array([0.9, 1.4]))
    Traceback (most recent call last):
    ValueError: probability outside 0~1: min=0.9, max=1.4
    """
    p = np.asarray(p, dtype=float)
    if p.min() < 0 or p.max() > 1:
        raise ValueError(f"probability outside 0~1: min={p.min()}, max={p.max()}")
    return p


def hybrid_predict_value(v_T: np.ndarray, v_S: np.ndarray, p_T: Probability, p_S: Probability,
                         w: float = 0.5) -> np.ndarray:
    """
    v_T: predicted value of model T
    v_S: predicted value of model S
    p_T: predicted probability of model T (0~1)
    p_S: predicted probability of model S (0~1)
    w: weight given to model T (0~1)

    >>> y_pred_by_t = np.array([10.0, 20.0, 30.0])
    >>> y_pred_by_s = np.array([12.0, 22.0, 36.0])
    >>> y_prob_by_t = np.array([0.9, 0.5, 0.2])
    >>> y_prob_by_s = np.array([0.3, 0.6, 0.9])
    >>> np.round(hybrid_predict_value(y_pred_by_t, y_pred_by_s, y_prob_by_t, y_prob_by_s, w=0.5), 4)
    array([10.5   , 21.0909, 34.9091])
    >>> hybrid_predict_value(y_pred_by_t, y_pred_by_s, np.array([0.0, 0.5, 0.2]), np.array([0.0, 0.6, 0.9]))
    Traceback (most recent call last):
    ValueError: both models report zero probability, so the weighted value is undefined.
    """
    # both probabilities have to be on the 0~1 scale before they are weighed against each other
    p_T = check_probability(p_T)
    p_S = check_probability(p_S)

    # confidence each model carries into the combination
    weight_T = w * p_T
    weight_S = (1 - w) * p_S

    denominator = weight_T + weight_S
    if np.any(denominator == 0):
        raise ValueError("both models report zero probability, so the weighted value is undefined.")
    return (weight_T * v_T + weight_S * v_S) / denominator
```

### B.2 Optimal Weight Search

```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, r2_score, roc_auc_score


def find_optimal_weight(y_true: np.ndarray, p_T: np.ndarray, p_S: np.ndarray, metric: str = "f1",
                        threshold: float = 0.5, step: float = 0.01,
                        v_T: np.ndarray = None, v_S: np.ndarray = None) -> tuple[float, float]:
    """Search the weight w of models T and S that suits the validation dataset best, by grid search.

    Parameters:
    - y_true: true labels (N,)
    - p_T: predicted probability of model T (N,)
    - p_S: predicted probability of model S (N,)
    - metric: metric to optimize ('f1', 'accuracy', 'r2', 'roc_auc', 'log_loss')
    - threshold: classification threshold (used by f1 and accuracy)
    - step: weight step of the grid search (default: 0.01 -> 100 intervals)
    - v_T: predicted value of model T (N,), required by 'r2' and refused by every other metric
    - v_S: predicted value of model S (N,), required by 'r2' and refused by every other metric

    Returns:
    - best_w: optimal weight for model T (model S takes 1 - best_w)
    - best_score: metric score at that weight

    >>> y_true = np.array([0, 0, 1, 1])
    >>> p_T = np.array([0.2, 0.3, 0.7, 0.8])
    >>> p_S = np.array([0.6, 0.4, 0.5, 0.3])
    >>> best_w, best_score = find_optimal_weight(y_true, p_T, p_S, metric="roc_auc", step=0.25)
    >>> float(best_w), round(float(best_score), 4)
    (0.5, 1.0)
    >>> best_w, best_score = find_optimal_weight(y_true, p_T, p_S, metric="log_loss", step=0.25)
    >>> float(best_w), round(float(best_score), 4)
    (1.0, 0.2899)
    >>> y_value = np.array([10.0, 20.0, 30.0, 40.0])
    >>> v_T = np.array([11.0, 19.0, 33.0, 37.0])
    >>> v_S = np.array([14.0, 25.0, 26.0, 44.0])
    >>> best_w, best_score = find_optimal_weight(
    ...     y_value, p_T, p_S, metric="r2", step=0.25, v_T=v_T, v_S=v_S
    ... )
    >>> float(best_w), round(float(best_score), 4)
    (0.75, 0.9707)
    """
    if metric == "r2":
        if v_T is None or v_S is None:
            raise ValueError("metric 'r2' scores the single predicted value, so v_T and v_S are required.")
    elif v_T is not None or v_S is not None:
        raise ValueError(f"v_T and v_S belong to metric 'r2', not to '{metric}'.")

    weights = np.arange(0.0, 1.0 + step, step)
    best_w = None

    # log loss is lower-better, every other metric is higher-better
    is_lower_better = metric == "log_loss"
    best_score = float("inf") if is_lower_better else -float("inf")

    for w in weights:
        # weighted average probability
        p_hybrid = w * p_T + (1 - w) * p_S

        # score of the chosen metric
        if metric == "f1":
            y_pred = (p_hybrid >= threshold).astype(int)
            score = f1_score(y_true, y_pred)
        elif metric == "accuracy":
            y_pred = (p_hybrid >= threshold).astype(int)
            score = accuracy_score(y_true, y_pred)
        elif metric == "roc_auc":
            score = roc_auc_score(y_true, p_hybrid)
        elif metric == "log_loss":
            score = log_loss(y_true, p_hybrid)
        elif metric == "r2":
            score = r2_score(y_true, hybrid_predict_value(v_T, v_S, p_T, p_S, w))
        else:
            raise ValueError(f"unsupported metric: {metric}")

        # keep the best so far
        if is_lower_better:
            if score < best_score:
                best_score = score
                best_w = round(w, 4)
        else:
            if score > best_score:
                best_score = score
                best_w = round(w, 4)

    return best_w, best_score
```

### B.3 Execution Example

```python
# --- synthetic validation data ---
np.random.seed(42)
N_samples = 1000

# ground truth
y_true = np.random.randint(0, 2, size=N_samples)

# probability from model T, relatively close to the truth
y_prob_by_t = y_true * 0.7 + np.random.normal(0, 0.2, size=N_samples)
y_prob_by_t = np.clip(y_prob_by_t, 0.01, 1)

# probability from model S
y_prob_by_s = y_true * 0.5 + np.random.normal(0, 0.3, size=N_samples)
y_prob_by_s = np.clip(y_prob_by_s, 0.01, 1)

# the class each model predicts on its own, at the 0.5 threshold
y_pred_by_t = (y_prob_by_t >= 0.5).astype(int)
y_pred_by_s = (y_prob_by_s >= 0.5).astype(int)

# --- the first ten rows of the sample ---
print("y_true     :", y_true[:10])
print("y_pred_by_t:", y_pred_by_t[:10])
print("y_pred_by_s:", y_pred_by_s[:10])
print("y_prob_by_t:", np.round(y_prob_by_t[:10], 4))
print("y_prob_by_s:", np.round(y_prob_by_s[:10], 4))

# --- run the grid search ---
# 1. optimize on F1-score
best_w_f1, best_score_f1 = find_optimal_weight(
    y_true, y_prob_by_t, y_prob_by_s, metric="f1"
)
print(f"[F1-Score] best w: {best_w_f1} | score: {best_score_f1:.4f}")

# 2. optimize on ROC-AUC
best_w_auc, best_score_auc = find_optimal_weight(
    y_true, y_prob_by_t, y_prob_by_s, metric="roc_auc"
)
print(
    f"[ROC-AUC]  best w: {best_w_auc} | score: {best_score_auc:.4f}"
)

# 3. optimize on log loss, lower is better
best_w_loss, best_score_loss = find_optimal_weight(
    y_true, y_prob_by_t, y_prob_by_s, metric="log_loss"
)
print(
    f"[Log Loss] best w: {best_w_loss} | score: {best_score_loss:.4f}"
)

# 4. optimize on R-squared, which scores the single predicted value itself
best_w_r2, best_score_r2 = find_optimal_weight(
    y_true, y_prob_by_t, y_prob_by_s, metric="r2",
    v_T=y_pred_by_t, v_S=y_pred_by_s
)
print(f"[R-squared] best w: {best_w_r2} | score: {best_score_r2:.4f}")

# --- the single predicted value at the weight R-squared chose ---
v_hybrid = hybrid_predict_value(
    y_pred_by_t, y_pred_by_s, y_prob_by_t, y_prob_by_s, w=best_w_r2
)
print("v_hybrid   :", np.round(v_hybrid[:10], 4))
```

The five arrays printed first are the head of the sample, ten rows of it, the four lines after them are the search result, and the last line is the single predicted value at the weight R-squared chose.

```text
y_true     : [0 1 0 0 0 1 0 0 0 1]
y_pred_by_t: [0 1 0 0 0 1 0 0 0 1]
y_pred_by_s: [0 1 0 0 0 0 1 0 0 1]
y_prob_by_t: [0.0684 1.     0.1901 0.01   0.01   0.7984 0.01   0.3663 0.2359 0.6062]
y_prob_by_s: [0.3905 0.9685 0.01   0.01   0.138  0.2967 0.604  0.041  0.01   0.5554]
[F1-Score] best w: 0.78 | score: 0.9174
[ROC-AUC]  best w: 0.73 | score: 0.9968
[Log Loss] best w: 1.0 | score: 0.2542
[R-squared] best w: 0.7 | score: 0.7127
v_hybrid   : [0.     1.     0.     0.     0.     0.8626 0.9628 0.     0.     1.    ]
```
