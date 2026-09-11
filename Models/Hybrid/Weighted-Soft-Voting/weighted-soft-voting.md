# Weighted Soft Voting
Rev. 4 | Created: 2026-09-11 | Updated: 2026-09-11 09:40 CDT

## 1. Purpose

- **Problem Statement**: Two models M and S each emit a predicted probability, with no rule for combining the two into one prediction.
- **Goal**: The definition of the hybrid probability that joins the two probabilities through a weight w, and the procedure for fixing w on a validation dataset.
- **Non-Goal**: Fitting and improving the models M and S themselves.

## 2. Summary

Where both models M and S provide predicted probabilities ($p_M$, $p_S$), soft voting, a weighted average of the probabilities that reflects the confidence of each model precisely, is the most effective method. The weight w comes from a grid search on a validation dataset (section 4.2) rather than from a guess, and the value found is carried unchanged into the weighted sum on the test dataset.

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

The implementation of equation (1) and equation (2) is [Appendix B.1](#b1-binary-classification).

### 3.2 Multi-Class Extension

In a multi-class classification with three or more classes, the per-class probability vectors $p_M$ and $p_S$ are summed with the weight, and the class holding the highest probability is selected.

```math
\mathbf{p}_{\mathrm{hybrid}} = w \cdot \mathbf{p}_M + (1 - w) \cdot \mathbf{p}_S \hspace{19em} (3)
```

```math
\mathrm{Final\ Class} = \arg\max \left( \mathbf{p}_{\mathrm{hybrid}} \right) \hspace{19em} (4)
```

The implementation is [Appendix B.2](#b2-multi-class-classification), and both inputs are arrays of shape (`N_samples`, `N_classes`).

## 4. Application

### 4.1 Cautions

Two things call for care when the combination is built on probabilities.

- **Probability Calibration**: The probability distributions of the two models have to be checked for a fine match. Where one model is too confident (for example, mostly near 0.05 or 0.95) and the other is cautious (for example, between 0.4 and 0.6), a plain weighted combination can give the confident model too much influence.
- **Optimization of the weight w**: Rather than fixing w arbitrarily, a search (grid search) for the w that raises a performance metric (ROC-AUC, F1-score and the like) the most on the validation dataset is recommended.

### 4.2 Optimal Weight Search

A grid search finds the optimal weight w on the validation dataset against a metric the user defines, such as F1-score, ROC-AUC or log loss. The search runs from 0.0 to 1.0, at a step whose default of 0.01 covers 100 intervals. The return is the optimal weight `best_w` to give model M and the metric score at that weight, the weight of model S being `1 - best_w`. The search function is [Appendix B.3](#b3-optimal-weight-search), and the run on synthetic data is [Appendix B.4](#b4-execution-example).

The optimal `best_w` found on the validation data is carried unchanged into the weighted sum on the test dataset for the final evaluation.

```math
p_{\mathrm{hybrid,test}} = w_{\mathrm{best}} \cdot p_{M,\mathrm{test}} + (1 - w_{\mathrm{best}}) \cdot p_{S,\mathrm{test}} \hspace{19em} (5)
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

### B.1 Binary Classification

```python
import numpy as np


def hybrid_predict_proba(p_M, p_S, w=0.5):
    """
    p_M: 모델 M의 예측 확률 (0~1)
    p_S: 모델 S의 예측 확률 (0~1)
    w: 모델 M에 부여할 가중치 (0~1)
    """
    # weighted average probability
    p_hybrid = w * p_M + (1 - w) * p_S
    return p_hybrid


def hybrid_predict(p_M, p_S, w=0.5, threshold=0.5):
    p_hybrid = hybrid_predict_proba(p_M, p_S, w)
    return (p_hybrid >= threshold).astype(int)
```

### B.2 Multi-Class Classification

```python
import numpy as np


def hybrid_predict_multiclass(p_M_array, p_S_array, w=0.5):
    """
    p_M_array: Shape (N_samples, N_classes)
    p_S_array: Shape (N_samples, N_classes)
    """
    # weighted sum of the per-class probabilities
    p_hybrid = w * p_M_array + (1 - w) * p_S_array

    # index of the class holding the highest probability
    predictions = np.argmax(p_hybrid, axis=1)
    return predictions, p_hybrid
```

### B.3 Optimal Weight Search

```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score


def find_optimal_weight(
    y_true, p_M, p_S, metric="f1", threshold=0.5, step=0.01
):
    """Validation 데이터셋을 활용해 모델 M과 S의 최적 가중치 w를 Grid Search로 탐색합니다.

    Parameters:
    - y_true: 실제 정답 레이블 (N,)
    - p_M: 모델 M의 예측 확률 (N,)
    - p_S: 모델 S의 예측 확률 (N,)
    - metric: 최적화 기준 지표 ('f1', 'roc_auc', 'log_loss', 'accuracy')
    - threshold: 분류 임계값 (f1, accuracy에서 사용)
    - step: Grid Search 가중치 간격 (기본값: 0.01 -> 100개 구간 탐색)

    Returns:
    - best_w: 모델 M에 부여할 최적 가중치 (모델 S의 가중치는 1 - best_w)
    - best_score: 해당 가중치에서의 평가 지표 점수
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
            raise ValueError(f"지원하지 않는 평가 지표입니다: {metric}")

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

### B.4 Execution Example

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
print(f"[F1-Score 기준] 최적 w: {best_w_f1} | 점수: {best_score_f1:.4f}")

# 2. optimize on ROC-AUC
best_w_auc, best_score_auc = find_optimal_weight(
    y_val, p_M_val, p_S_val, metric="roc_auc"
)
print(
    f"[ROC-AUC  기준] 최적 w: {best_w_auc} | 점수: {best_score_auc:.4f}"
)

# 3. optimize on log loss, lower is better
best_w_loss, best_score_loss = find_optimal_weight(
    y_val, p_M_val, p_S_val, metric="log_loss"
)
print(
    f"[Log Loss 기준] 최적 w: {best_w_loss} | 점수: {best_score_loss:.4f}"
)
```
