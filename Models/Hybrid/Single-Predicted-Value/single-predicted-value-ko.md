# Single Predicted Value From Two Models (Korean)
Rev. 16 | Created: 2026-09-11 | Updated: 2026-09-11 14:50 CDT

## 1. Purpose

- **Problem Statement**: 복수의 모델에서 계산한 예측값을 하나로 합치는 간단한 방법이 없다.
- **Goal**: 두 모델에서 각각 예측 값과 확률을 내놓을 때, 두 확률을 고려한 단일 예측값을 제시하고자 한다.
- **Non-Goal**: 모델 자체의 학습과 개선은 하지 않는다.

## 2. Summary

단일 예측값은 두 확률에서 나옵니다. 클래스는 두 확률의 가중 평균에서 다시 읽어내고, 클래스가 아닌 예측 값은 그 두 확률을 가중치로 삼아 평균합니다. 두 모델 M과 S가 모두 예측 확률 ($p_M$, $p_S$) 을 제공한다면, 각 모델의 확신도를 정밀하게 반영하는 Soft Voting (확률 가중 평균) 방식을 사용하는 것이 가장 효과적입니다. 가중치 w 는 임의로 정하지 않고 validation dataset 에서 grid search 로 찾으며 (section 4.2), 찾은 값을 test dataset 의 가중합에 그대로 적용합니다.

## 3. Principle

### 3.1 Weighted Soft Voting

두 모델의 확률값 $p_M$ 과 $p_S$ 에 가중치 w ($0 \le w \le 1$) 를 적용하여 하이브리드 확률 $p_{\mathrm{hybrid}}$ 를 계산합니다.

```math
p_{\mathrm{hybrid}} = w \cdot p_M + (1 - w) \cdot p_S \hspace{19em} (1)
```

이진 분류의 최종 클래스는 임계값 0.5 를 기준으로 갈립니다.

```math
\mathrm{Final\ Class} =
\begin{cases}
1 & \mathrm{if}\ p_{\mathrm{hybrid}} \ge 0.5 \\
0 & \mathrm{otherwise}
\end{cases}
\hspace{15em} (2)
```

각 모델이 내놓은 예측값은 여기에 쓰이지 않습니다. 클래스는 $p_{\mathrm{hybrid}}$ 에서 다시 읽어내며, 각 모델이 그 안에서 차지하는 몫은 가중치 w 가 정합니다. 예측 값 자체를 합치는 규칙은 꼭지 3.3 에 있습니다.

### 3.2 Multi-Class Extension

클래스가 3개 이상인 다중 클래스 분류에서는 각 클래스별 확률 벡터 $p_M$ 과 $p_S$ 를 가중합한 후 가장 높은 확률을 가진 클래스를 선택합니다.

```math
\mathbf{p}_{\mathrm{hybrid}} = w \cdot \mathbf{p}_M + (1 - w) \cdot \mathbf{p}_S \hspace{19em} (3)
```

```math
\mathrm{Final\ Class} = \arg\max \left( \mathbf{p}_{\mathrm{hybrid}} \right) \hspace{19em} (4)
```

입력 두 개는 모두 (`N_samples`, `N_classes`) 형상의 배열입니다.

### 3.3 Single Predicted Value

각 모델이 예측 값을 함께 내놓는 경우, 단일 예측값은 두 값을 확률로 가중 평균한 것입니다. 자기 답을 더 확신하는 모델 쪽으로 결과가 끌립니다.

```math
v_{\mathrm{hybrid}} = \frac{w \cdot p_M \cdot v_M + (1 - w) \cdot p_S \cdot v_S}{w \cdot p_M + (1 - w) \cdot p_S} \hspace{15em} (5)
```

분모는 식 (1) 의 하이브리드 확률이므로, 꼭지 3.1 의 Soft Voting 이 여기서도 두 값을 실어 나릅니다. 구현은 [Appendix B.1](#b1-single-predicted-value) 입니다. 가중합에 앞서 두 확률이 0~1 안에 있는지 확인하며, 두 확률이 모두 0 인 행에서는 값을 돌려주는 대신 오류를 냅니다.

## 4. Application

### 4.1 Cautions

확률 기반 하이브리드 결합 시 주의사항은 두 가지입니다.

- **확률 교정 (Probability Calibration)**: 두 모델의 확률 분포가 정교하게 맞추어져 있는지 확인해야 합니다. 한 모델이 확률을 너무 과신 (예: 대부분 0.05 또는 0.95 근처) 하고 다른 모델은 신중한 경우 (예: 0.4∼0.6 사이), 단순 가중치 조합 시 과신하는 모델의 영향력이 과도하게 커질 수 있습니다.
- **가중치 w 최적화**: w 값을 임의로 정하기보다는, Validation Dataset에서 성능 지표 (ROC-AUC, F1-score 등) 를 가장 높여주는 w를 탐색 (Grid Search) 하여 선정하는 것을 권장합니다.

### 4.2 Optimal Weight Search

Validation 데이터셋에서 F1-score, ROC-AUC, Log-Loss 등 사용자가 정의한 평가 지표를 기준으로 최적의 가중치 w를 Grid Search로 탐색합니다. 탐색 구간은 0.0 에서 1.0 까지이고, 간격은 기본값 0.01 로 100개 구간을 훑습니다. 반환값은 모델 M 에 부여할 최적 가중치 `best_w` 와 그 가중치에서의 평가 지표 점수이며, 모델 S 의 가중치는 `1 - best_w` 입니다. 탐색 함수는 [Appendix B.2](#b2-optimal-weight-search) 이고, 가상 데이터를 활용한 실행 예시는 [Appendix B.3](#b3-execution-example) 입니다.

Validation 데이터로 찾아낸 최적의 `best_w` 를 그대로 Test 데이터셋의 가중합 계산에 적용하여 최종 평가를 수행하면 됩니다.

```math
p_{\mathrm{hybrid,test}} = w_{\mathrm{best}} \cdot p_{M,\mathrm{test}} + (1 - w_{\mathrm{best}}) \cdot p_{S,\mathrm{test}} \hspace{19em} (6)
```

### 4.3 Metric Selection

지표는 분류 임계값 (Threshold) 을 쓰는지로 갈립니다. `ROC-AUC`는 확률값 차원 전체를 평가하므로 분류 임계값에 영향을 받지 않는 반면, `F1-Score`나 `Accuracy`는 확률을 최종 클래스로 변환하는 `threshold` 설정에 맞춰 작동합니다.

Table 1. Metrics for the weight search

| Metric | Direction | Threshold | Basis |
| --- | --- | --- | --- |
| `roc_auc` | 클수록 좋음 | 쓰지 않음 | 확률값 차원 전체 |
| `log_loss` | 작을수록 좋음 | 쓰지 않음 | 확률값과 정답의 불일치 |
| `f1` | 클수록 좋음 | 씀 | 임계값으로 변환한 최종 클래스 |
| `accuracy` | 클수록 좋음 | 씀 | 임계값으로 변환한 최종 클래스 |

## 5. Comparison

N/A — 다른 결합 방식을 다루지 않음.

## 6. Further Work

- **확률 교정 적용**
  - 무엇: `IsotonicRegression`이나 `Platt Scaling`을 통한 교정을 두 모델의 확률에 적용한 뒤 가중합.
  - 왜 지금: section 4.1 의 조건, 즉 한 모델이 과신하고 다른 모델이 신중한 상태에서는 가중치만으로 영향력의 불균형을 바로잡지 못함.
  - 무엇이 필요: 교정에 쓸 데이터와, 교정 전후의 최적 w 및 지표 점수 비교.

## References

N/A — 외부 출처를 인용하지 않음.

---

## Appendix A. Terminology

- **Grid Search**: 정해진 구간을 일정 간격으로 훑어 최적값을 찾는 탐색.
- **Isotonic Regression**: 단조 증가 제약 아래 예측 확률을 실제 빈도에 맞추는 교정 방법.
- **Log Loss**: 예측 확률과 정답의 불일치를 재는 지표. 작을수록 좋음.
- **Platt Scaling**: 모델의 출력을 logistic 함수에 통과시켜 확률로 맞추는 교정 방법.
- **Probability Calibration**: 모델이 내놓는 확률을 실제 빈도에 맞추는 절차.
- **ROC-AUC**: 모든 임계값에 걸친 분류 성능을 하나로 요약한 지표.
- **Soft Voting**: 각 모델의 예측 확률을 가중 평균하여 최종 클래스를 정하는 결합 방식.
- **Threshold**: 확률을 최종 클래스로 변환하는 분류 임계값.

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

# --- the first five rows of the sample ---
print("y_val:", y_val[:5])
print("p_M  :", np.round(p_M_val[:5], 4))
print("p_S  :", np.round(p_S_val[:5], 4))

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

먼저 찍히는 다섯 행이 표본의 머리이고, 그 뒤 세 줄이 탐색 결과입니다.

```text
y_val: [0 1 0 0 0]
p_M  : [0.0684 1.     0.1901 0.     0.    ]
p_S  : [0.3905 0.9685 0.0096 0.     0.138 ]
[F1-Score] best w: 0.78 | score: 0.9174
[ROC-AUC]  best w: 0.73 | score: 0.9968
[Log Loss] best w: 1.0 | score: 0.2517
```
