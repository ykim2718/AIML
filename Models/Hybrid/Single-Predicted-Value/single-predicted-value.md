# Single Predicted Value From Two Models
Rev. 14 | Created: 2026-09-11 | Updated: 2026-09-11 14:20 CDT

## 1. Purpose

- **Problem Statement**: No simple way to combine the predictions computed by several models into one.
- **Goal**: A single prediction that accounts for both probabilities, where two models each emit a prediction and a probability.
- **Non-Goal**: Fitting and improving the models themselves.

## 2. Summary

The single prediction comes from the two probabilities: a class is read off their weighted average, and a predicted value of any other kind is the two values averaged with those same probabilities as weights. Where both models M and S provide predicted probabilities ($p_M$, $p_S$), soft voting, a weighted average of the probabilities that reflects the confidence of each model precisely, is the most effective method. The weight w comes from a grid search on a validation dataset (section 4.2) rather than from a guess, and the value found is carried unchanged into the weighted sum on the test dataset.

## 3. Principle

### 3.1 Weighted Soft Voting

A weight w ($0 \le w \le 1$) applied to the probabilities $p_M$ and $p_S$ of the two models gives the hybrid probability $p_{\mathrm{hybrid}}$.

```math
p_{\mathrm{hybrid}} = w \cdot p_M + (1 - w) \cdot p_S \hspace{19em} (1)
```

The final class of a binary classification is decided at the threshold 0.5.

```math
\mathrm{Final\ Class} =
\begin{cases}
1 & \mathrm{if}\ p_{\mathrm{hybrid}} \ge 0.5 \\
0 & \mathrm{otherwise}
\end{cases}
\hspace{15em} (2)
```

Each model's own predicted value takes no part in this: the class is read off $p_{\mathrm{hybrid}}$, and the share a model holds in it is what the weight w sets. Section 3.3 is the rule for the predicted values themselves.

### 3.2 Multi-Class Extension

In a multi-class classification with three or more classes, the per-class probability vectors $p_M$ and $p_S$ are summed with the weight, and the class holding the highest probability is selected.

```math
\mathbf{p}_{\mathrm{hybrid}} = w \cdot \mathbf{p}_M + (1 - w) \cdot \mathbf{p}_S \hspace{19em} (3)
```

```math
\mathrm{Final\ Class} = \arg\max \left( \mathbf{p}_{\mathrm{hybrid}} \right) \hspace{19em} (4)
```

Both inputs are arrays of shape (`N_samples`, `N_classes`).

### 3.3 Single Predicted Value

Where each model emits a predicted value of its own, the single value is the two values averaged with the probabilities as weights, so the model surer of its answer pulls the result toward it.

```math
v_{\mathrm{hybrid}} = \frac{w \cdot p_M \cdot v_M + (1 - w) \cdot p_S \cdot v_S}{w \cdot p_M + (1 - w) \cdot p_S} \hspace{15em} (5)
```

The denominator is the hybrid probability of equation (1), so the weighted soft voting of section 3.1 is what carries the two values here as well. The implementation is [Appendix B.1](#b1-single-predicted-value). It checks that both probabilities lie within 0~1 before weighing them, and it raises on a row where both are zero instead of returning a number.

## 4. Application

### 4.1 Cautions

Two things call for care when the combination is built on probabilities.

- **Probability Calibration**: The probability distributions of the two models have to be checked for a fine match. Where one model is too confident (for example, mostly near 0.05 or 0.95) and the other is cautious (for example, between 0.4 and 0.6), a plain weighted combination can give the confident model too much influence.
- **Optimization of the weight w**: Rather than fixing w arbitrarily, a search (grid search) for the w that raises a performance metric (ROC-AUC, F1-score and the like) the most on the validation dataset is recommended.

### 4.2 Optimal Weight Search

A grid search finds the optimal weight w on the validation dataset against a metric the user defines, such as F1-score, ROC-AUC or log loss. The search runs from 0.0 to 1.0, at a step whose default of 0.01 covers 100 intervals. The return is the optimal weight `best_w` to give model M and the metric score at that weight, the weight of model S being `1 - best_w`. The search function is [Appendix B.2](#b2-optimal-weight-search), and the run on synthetic data is [Appendix B.3](#b3-execution-example).

The optimal `best_w` found on the validation data is carried unchanged into the weighted sum on the test dataset for the final evaluation.

```math
p_{\mathrm{hybrid,test}} = w_{\mathrm{best}} \cdot p_{M,\mathrm{test}} + (1 - w_{\mathrm{best}}) \cdot p_{S,\mathrm{test}} \hspace{19em} (6)
```

### 4.3 Metric Selection

Metrics divide on whether they use the classification threshold. `ROC-AUC` evaluates the whole probability dimension and is therefore untouched by the classification threshold, while `F1-Score` and `Accuracy` work against the `threshold` setting that turns a probability into a final class.

Table 1. Metrics for the weight search

| Metric | Direction | Threshold | Basis |
| --- | --- | --- | --- |
| `roc_auc` | Higher is better | Not used | The whole probability dimension |
| `log_loss` | Lower is better | Not used | Disagreement between the probability and the label |
| `f1` | Higher is better | Used | Final class produced by the threshold |
| `accuracy` | Higher is better | Used | Final class produced by the threshold |

## 5. Comparison

N/A — no alternative combination rule is covered.

## 6. Further Work

- **Applying probability calibration**
  - What: calibration through `IsotonicRegression` or `Platt Scaling`, applied to both probabilities before the weighted sum.
  - Why now: under the condition of section 4.1, one model confident and the other cautious, a weight alone does not correct the imbalance in influence.
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


def hybrid_predict_value(v_M: np.ndarray, v_S: np.ndarray, p_M: Probability, p_S: Probability,
                         w: float = 0.5) -> np.ndarray:
    """
    v_M: predicted value of model M
    v_S: predicted value of model S
    p_M: predicted probability of model M (0~1)
    p_S: predicted probability of model S (0~1)
    w: weight given to model M (0~1)

    >>> v_M = np.array([10.0, 20.0, 30.0])
    >>> v_S = np.array([12.0, 22.0, 36.0])
    >>> p_M = np.array([0.9, 0.5, 0.2])
    >>> p_S = np.array([0.3, 0.6, 0.9])
    >>> np.round(hybrid_predict_value(v_M, v_S, p_M, p_S, w=0.5), 4)
    array([10.5   , 21.0909, 34.9091])
    >>> hybrid_predict_value(v_M, v_S, np.array([0.0, 0.5, 0.2]), np.array([0.0, 0.6, 0.9]))
    Traceback (most recent call last):
    ValueError: both models report zero probability, so the weighted value is undefined.
    """
    # both probabilities have to be on the 0~1 scale before they are weighed against each other
    p_M = check_probability(p_M)
    p_S = check_probability(p_S)

    # confidence each model carries into the combination
    weight_M = w * p_M
    weight_S = (1 - w) * p_S

    denominator = weight_M + weight_S
    if np.any(denominator == 0):
        raise ValueError("both models report zero probability, so the weighted value is undefined.")
    return (weight_M * v_M + weight_S * v_S) / denominator
```

### B.2 Optimal Weight Search

```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


def find_optimal_weight(y_true: np.ndarray, p_M: np.ndarray, p_S: np.ndarray, metric: str = "f1",
                        threshold: float = 0.5, step: float = 0.01) -> tuple[float, float]:
    """Search the weight w of models M and S that suits the validation dataset best, by grid search.

    Parameters:
    - y_true: true labels (N,)
    - p_M: predicted probability of model M (N,)
    - p_S: predicted probability of model S (N,)
    - metric: metric to optimize ('f1', 'roc_auc', 'log_loss', 'accuracy')
    - threshold: classification threshold (used by f1 and accuracy)
    - step: weight step of the grid search (default: 0.01 -> 100 intervals)

    Returns:
    - best_w: optimal weight for model M (model S takes 1 - best_w)
    - best_score: metric score at that weight

    >>> y_true = np.array([0, 0, 1, 1])
    >>> p_M = np.array([0.2, 0.3, 0.7, 0.8])
    >>> p_S = np.array([0.6, 0.4, 0.5, 0.3])
    >>> best_w, best_score = find_optimal_weight(y_true, p_M, p_S, metric="roc_auc", step=0.25)
    >>> float(best_w), round(float(best_score), 4)
    (0.5, 1.0)
    >>> best_w, best_score = find_optimal_weight(y_true, p_M, p_S, metric="log_loss", step=0.25)
    >>> float(best_w), round(float(best_score), 4)
    (1.0, 0.2899)
    """
    weights = np.arange(0.0, 1.0 + step, step)
    best_w = None

    # log loss is lower-better, every other metric is higher-better
    is_lower_better = metric == "log_loss"
    best_score = float("inf") if is_lower_better else -float("inf")

    for w in weights:
        # weighted average probability
        p_hybrid = w * p_M + (1 - w) * p_S

        # score of the chosen metric
        if metric == "f1":
            preds = (p_hybrid >= threshold).astype(int)
            score = f1_score(y_true, preds)
        elif metric == "accuracy":
            preds = (p_hybrid >= threshold).astype(int)
            score = accuracy_score(y_true, preds)
        elif metric == "roc_auc":
            score = roc_auc_score(y_true, p_hybrid)
        elif metric == "log_loss":
            score = log_loss(y_true, p_hybrid)
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
y_val = np.random.randint(0, 2, size=N_samples)

# probability from model M, relatively close to the truth
p_M_val = y_val * 0.7 + np.random.normal(0, 0.2, size=N_samples)
p_M_val = np.clip(p_M_val, 0, 1)

# probability from model S
p_S_val = y_val * 0.5 + np.random.normal(0, 0.3, size=N_samples)
p_S_val = np.clip(p_S_val, 0, 1)

# --- run the grid search ---
# 1. optimize on F1-score
best_w_f1, best_score_f1 = find_optimal_weight(
    y_val, p_M_val, p_S_val, metric="f1"
)
print(f"[F1-Score] best w: {best_w_f1} | score: {best_score_f1:.4f}")

# 2. optimize on ROC-AUC
best_w_auc, best_score_auc = find_optimal_weight(
    y_val, p_M_val, p_S_val, metric="roc_auc"
)
print(
    f"[ROC-AUC]  best w: {best_w_auc} | score: {best_score_auc:.4f}"
)

# 3. optimize on log loss, lower is better
best_w_loss, best_score_loss = find_optimal_weight(
    y_val, p_M_val, p_S_val, metric="log_loss"
)
print(
    f"[Log Loss] best w: {best_w_loss} | score: {best_score_loss:.4f}"
)
```
