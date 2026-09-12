# Univariate Feature Selection (Korean)
Rev. 2 | Created: 2026-09-12 | Updated: 2026-09-12 18:08 CDT

## 1. Purpose

- **Problem Statement**: Data 의 차원이 높아질수록 curse of dimensionality 가 나타나고 model 학습 속도가 떨어진다.
- **Goal**: Feature 를 하나씩 통계 검정하여 고르는 절차와 data type 별 검정 지표의 선택 기준을 정리하여, 주어진 data 에 맞는 지표를 독자가 스스로 고를 수 있게 한다.
- **Non-Goal**: Feature 조합의 상호작용을 이용한 선택은 다루지 않는다. 그것은 [Multivariate Feature Selection](../multivariate-feature-selection/multivariate-feature-selection-ko.md) 의 주제다.

## 2. Overview

Univariate feature selection 은 각 feature 를 다른 feature 와의 연관을 고려하지 않고 개별 단위로 통계 검정하여 고르는 기법이다. 점수가 feature 하나에 대한 scalar 값이므로 scalar feature selection 이라고도 부른다.

- 계산 위치: model 학습 이전 단계에 적용하는 filter method
- 계산 비용: feature 하나에 검정 한 번
- 다변량 기법과의 차이: 전체 feature 집합의 상호작용을 부분집합 단위로 보는 방식과 달리, 각 변수를 1차원 scalar 값의 통계적 유의성으로 평가

## 3. Principle

절차는 세 단계이며, 앞 단계의 결과가 다음 단계의 입력이 된다.

- 독립적 평가: 각 입력 변수 $x_i$ 와 target 변수 $y$ 사이의 통계적 점수 계산
- 순위 매기기: 계산된 점수를 기준으로 feature 정렬
- Cut-off: 상위 $k$ 개 feature 선택, 또는 p-value 가 기준치 (예: $0.05$) 이하인 feature 추출

```text
[ Input data X ]
    |
    +-- Feature 1 --( statistical test )--> score S1
    |
    +-- Feature 2 --( statistical test )--> score S2  --[ top-k ]--> [ Selected feature subset ]
    |
    +-- Feature 3 --( statistical test )--> score S3
```

Fig 1. Scoring each feature independently and taking the top k

## 4. Statistical Metrics

지표는 입력과 target 이 각각 연속형인지 범주형인지에 따라 갈린다.

### 4.1 Chi-square Test

- 적용 조건: 범주형 입력 + 범주형 target
- 원리: feature 와 target 의 독립성 검정. $\chi^2$ 값이 클수록 target 과의 관련성이 높음

### 4.2 ANOVA F-Test

- 적용 조건: 연속형 입력 + 범주형 또는 연속형 target
- 원리: 그룹 간 분산과 그룹 내 분산의 비를 측정

```math
F = \frac{\sigma_{\mathrm{between}}^2}{\sigma_{\mathrm{within}}^2} \hspace{19em} (1)
```

### 4.3 Pearson Correlation

- 적용 조건: 연속형 입력 + 연속형 target
- 원리: 두 변수의 선형 상관관계를 $-1$ 에서 $1$ 사이의 scalar 값으로 계산

### 4.4 Mutual Information

- 적용 조건: 비선형 관계를 포함한 모든 data type
- 원리: 한 변수를 알 때 다른 변수의 entropy 가 줄어드는 양을 측정

## 5. Pros And Cons

장점은 비용과 단순성에서 오고, 단점은 feature 를 하나씩만 본다는 전제에서 온다.

Table 1. Pros and cons of univariate feature selection

| Category | Item | Detail |
| --- | --- | --- |
| Pros | 계산 효율성 | $O(N)$ 의 시간 복잡도. 대용량·고차원 dataset 에서 빠름 |
| Pros | 단순성 | Model 의존성 없음. 전처리 단계에 직관적으로 적용 |
| Pros | 오버피팅 방지 | 학습 model 의 특성에 종속되지 않는 범용 filtering |
| Cons | 상호작용 무시 | $X_1$ 과 $X_2$ 가 결합해야 드러나는 pattern 을 감지하지 못함 |
| Cons | 다중공선성 미반영 | 서로 같은 정보를 담은 두 변수가 함께 선택됨 |

---

## Appendix A. Terminology

- **Curse of Dimensionality**: 차원이 늘어날수록 표본 사이의 거리가 고르게 멀어져, 같은 밀도를 유지하는 데 필요한 표본 수가 지수로 커지는 현상.
- **Cut-off**: 순위가 매겨진 feature 가운데 무엇까지 남길지를 정하는 기준. 개수 $k$ 또는 p-value 한계로 준다.
- **Entropy**: 확률 변수의 불확실성을 bit 단위로 잰 양.
- **Filter Method**: Model 을 적합하기 전에 data 의 통계량만으로 feature 를 고르는 방식.
- **p-value**: 귀무가설이 참일 때 관측된 것만큼 극단적인 통계량이 나올 확률.
- **다중공선성 (Multicollinearity)**: 입력 변수들이 서로 강한 선형 관계를 가져, 계수가 개별 변수에 고유하게 배정되지 않는 상태.

## Appendix B. Implementation

scikit-learn 의 `SelectKBest` 로 section 3 의 세 단계를 실행하는 class 다. 검정 함수와 cut-off 개수 $k$ 를 생성자로 받고, feature 마다의 점수와 선택된 column 번호를 함께 돌려주어 무엇이 어떤 점수로 남았는지 볼 수 있게 한다.

입력은 scikit-learn 에 들어 있는 iris dataset 으로, 표본 150 개와 feature 4 개를 가진다. $\chi^2$ 검정은 음수가 아닌 값을 요구하며 iris 의 네 feature 는 모두 그 조건을 만족한다.

```python
from typing import Callable

import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2


class UnivariateFeatureSelector:
    """Score every feature alone against the target and keep the k best ones.

    Args:
        score_func: statistical test applied to one feature at a time.
        k: number of features the cut-off keeps.
    """

    def __init__(self, score_func: Callable = chi2, k: int = 2) -> None:
        if k < 1:
            raise ValueError(f"k must be at least 1: {k=}")
        self.score_func = score_func
        self.k = k

    def run(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Return the score of every feature and the columns left after the cut-off."""
        if self.k > X.shape[1]:
            raise ValueError(f"k asks for more features than the data has: {self.k=}, {X.shape[1]=}")
        selector = SelectKBest(score_func=self.score_func, k=self.k).fit(X, y)
        return {"scores": selector.scores_, "columns": np.where(selector.get_support())[0]}


if __name__ == "__main__":
    data = load_iris()
    print(f"input: {data.data.shape[0]} samples, {data.data.shape[1]} features")

    selector = UnivariateFeatureSelector(score_func=chi2, k=2)
    result = selector.run(X=data.data, y=data.target)
    for name, score in zip(data.feature_names, result["scores"]):
        print(f"{name:>18}: {score:7.2f}")
    print(f"selected: {', '.join(np.asarray(data.feature_names)[result['columns']])}")
```

실행 결과는 다음과 같다.

```text
input: 150 samples, 4 features
 sepal length (cm):   10.82
  sepal width (cm):    3.71
 petal length (cm):  116.31
  petal width (cm):   67.05
selected: petal length (cm), petal width (cm)
```

Petal 의 두 feature 가 sepal 의 두 feature 보다 한 자리 큰 점수를 받아 남는다. 두 petal feature 는 서로 상관계수 0.96 으로 묶여 있지만 이 방법은 그것을 보지 못하고 둘 다 남기며, 그것이 section 5 가 적은 단점이다.
