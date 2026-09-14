# Multivariate Feature Selection
Rev. 43 | Created: 2026-09-12 | Updated: 2026-09-14 10:20 CDT

## 1. Purpose

- **Problem Statement**: Feature 를 하나씩만 평가하면 두 feature 가 결합해야 드러나는 신호를 놓치고, 같은 정보를 담은 feature 가 함께 남는다.
- **Goal**: Feature 사이의 상호작용과 중복성을 함께 보는 선택 기법을 접근 방식별로 갈라 놓아, 주어진 data 크기와 계산 예산에서 어느 기법을 쓸지 독자가 고를 수 있게 한다.
- **Non-Goal**: Feature 를 하나씩 검정하는 기법은 다루지 않는다. 그것은 [Univariate Feature Selection](../univariate-feature-selection/univariate-feature-selection-ko.md) 의 주제다.

### 1.1 Motivation

Feature 조합이 개별 feature 보다 target 을 더 잘 설명하는 경우가 있으므로, 선택은 조합 단위로 이루어져야 한다. $X_1$ 과 $X_2$ 가 각각은 target $Y$ 와 낮은 상관을 보여도 두 feature 의 조합은 $Y$ 를 설명하는 강한 신호가 될 수 있으며, XOR 문제가 그 대표적인 예다.

다변량 분석의 목적은 셋이다.

1️⃣ Feature 사이의 다중공선성 및 중복성 제거<br>
2️⃣ Feature 사이의 시너지 효과 발굴<br>
3️⃣ Model 성능 향상과 과적합 방지

## 2. Taxonomy

기법은 model 을 언제 참조하는가 (approach) 와 상호작용을 어떤 방식으로 다루는가 (interaction) 의 두 갈래로 나뉜다.

```text
Multivariate feature selection taxonomy
|
+-- 1. Approach-based hierarchy
|   |
|   +-- Filter methods
|   |   +-- Correlation matrix and VIF ...... multicollinearity removal
|   |   +-- mRMR ........................... minimum redundancy maximum relevance
|   |   +-- ReliefF ........................ neighbour contrast
|   |
|   +-- Wrapper methods
|   |   +-- Forward selection / backward elimination
|   |   +-- RFE ........................... recursive feature elimination
|   |   +-- Genetic algorithm search
|   |
|   +-- Embedded methods
|       +-- Lasso (L1) / ElasticNet
|       +-- Tree-based importance ......... random forest, XGBoost, LightGBM
|
+-- 2. Interaction-based hierarchy
    +-- Redundancy reduction ............... removing duplicated information
    +-- Feature synergy .................... keeping features that matter together
    +-- Dimensionality tradeoff ............ trading dimension against signal
```

Fig 1. Two hierarchies of multivariate feature selection

## 3. Approach-based Methods

접근 방식은 계산 비용과 답의 성질을 정한다. Filter 는 자료의 성질을, wrapper 는 그 model 과 탐색의 성질을, embedded 는 적합된 model 의 성질을 답으로 내놓는다.

### 3.1 Multivariate Filter Methods

Model 학습 없이 data 의 통계적 특성만으로 feature 조합을 선별한다. Univariate filter 와 달리 feature 사이의 상관성을 함께 계산한다.

mRMR (Minimum Redundancy Maximum Relevance) 는 target 과의 mutual information 을 최대화하고 선택된 feature 사이의 mutual information 을 최소화하는 최적화 문제로 푼다.

```math
\max_{S} \left[ \frac{1}{|S|} \sum_{i \in S} I(x_i; y)
- \frac{1}{|S|^2} \sum_{i, j \in S} I(x_i; x_j) \right]
\hspace{10em} (1)
```

VIF (Variance Inflation Factor) 는 한 feature 를 나머지 feature 로 회귀하여 그 설명력을 측정한다. $\mathrm{VIF} \gt 10$ 인 feature 를 순차적으로 제거한다.

### 3.2 Wrapper Methods

특정 model 을 검증 도구로 삼아, 최적의 성능을 내는 feature subset 을 탐색 algorithm 으로 찾는다.

RFE (Recursive Feature Elimination) 의 절차는 다음과 같다.

- 전체 feature 로 model 을 학습
- 계수나 feature 중요도가 가장 낮은 feature 를 제거
- 목표 feature 개수에 닿을 때까지 반복

Greedy search 는 feature 를 하나씩 추가 (forward) 하거나 제거 (backward) 하며 cross validation 점수의 변화를 추적한다. Genetic algorithm 은 subset 여럿을 한 세대로 두고, 점수가 높은 것들을 섞고 일부를 바꿔 가며 다음 세대를 만들어, 순차 탐색이 닿지 않는 조합까지 훑는다.

### 3.3 Embedded Methods

Model 의 학습 algorithm 안에 feature 선택 과정이 들어 있다.

Lasso 는 손실 함수에 계수 절댓값의 합 $\lambda \sum |\beta_i|$ 을 penalty 로 더하여, 불필요한 feature 의 계수를 정확히 0 으로 보낸다. ElasticNet 은 거기에 계수 제곱합을 섞어, 서로 상관된 feature 가운데 하나만 남기는 lasso 와 달리 그 무리를 함께 남긴다.

Tree-based importance 는 tree model 의 node 분할 기여도 (MDI) 나 값을 무작위로 섞었을 때의 성능 저하 폭 (permutation importance) 으로 다변량 관점의 중요도를 계산한다. 구현으로는 random forest 와 gradient boosting 계열의 XGBoost, LightGBM 이 있으며, 셋 다 분할 기여도를 내놓으므로 `SelectFromModel` 에 그대로 들어간다.

### 3.4 Comparison

Multivariate filter, wrapper, embedded 는 계산 비용과 상호작용 반영 정도가 서로 반대 방향으로 움직인다.

Table 1. Comparison of the three approaches

| #   | Aspect        | Multivariate filter         | Wrapper              | Embedded           |
| :-: | :-----------: | :-------------------------: | :------------------: | :----------------: |
| 1   | 계산 복잡도   | 낮음                        | 매우 높음            | 중간               |
| 2   | 과적합 위험   | 낮음                        | 높음                 | 중간               |
| 3   | Model 의존성  | 없음 (model-agnostic)       | 선택한 model 에 종속 | 해당 model 에 내장 |
| 4   | 상호작용 반영 | 제한적 (주로 1:1 중복 제거) | 매우 잘 반영         | 잘 반영            |

## 4. Interaction-based Methods

같은 기법을 상호작용을 어떻게 다루는가로 다시 묶으면 section 2 의 둘째 갈래가 된다. 한 기법이 두 갈래에 걸치기도 하며, 그때는 그 기법이 각 갈래에서 무엇을 하는지로 갈라 적는다.

### 4.1 Redundancy Reduction

중복 신호를 지우는 갈래이며, feature 사이의 상관만 보고 target 은 보지 않아도 된다.

- 상관 filter: 상관계수가 기준치를 넘는 쌍에서 한쪽을 제거
- VIF: 나머지 feature 로 설명되는 정도가 큰 feature 를 순차적으로 제거
- mRMR 의 min-redundancy 항: 선택된 feature 사이의 mutual information 을 벌점으로 부과

### 4.2 Feature Synergy

결합 신호를 살리는 갈래이며, feature 를 조합 단위로 평가해야 드러난다.

- Wrapper (RFE, forward·backward search): 후보 subset 을 model 에 넣어 점수를 매기므로 조합의 효과가 그대로 점수에 들어감
- Tree-based importance: 분할이 이미 갈라진 node 안에서 이루어져, 다른 feature 의 값에 따라 달라지는 기여가 반영됨
- ReliefF: 표본마다 가까운 같은 class 와 다른 class 의 이웃을 전체 feature vector 의 거리 위에서 비교하므로, 다른 feature 와 함께일 때만 드러나는 차이가 점수에 들어감

### 4.3 Dimensionality Tradeoff

남길 차원 수를 신호와 맞바꾸는 갈래이며, 앞의 두 갈래가 매긴 순위를 어디서 자를지를 정한다.

- Lasso 의 $\lambda$: 값이 클수록 0 이 되는 계수가 늘어 차원이 줄어듦
- RFE 의 목표 feature 개수: 남길 차원을 직접 지정
- Tree-based importance 의 문턱값: 평균 중요도 같은 기준으로 자를 자리를 정함

## 5. Workflow

비용이 낮은 기법으로 후보를 줄인 뒤 비싼 기법을 쓴다. Wrapper 의 비용은 남은 feature 개수에 따라 커지므로, 그 앞에 세 단계를 둔다.

- Step 1 (constant removal): 모든 표본에서 값이 같은 feature 를 먼저 제거. 상관도 중요도도 정의되지 않고, 뒤 단계가 가릴 것이 없음
- Step 2 (pre-filtering): univariate 통계량 또는 VIF 로 상관계수 0.95 이상인 중복 feature 를 1차 제거
- Step 3 (embedded selection): Lasso 또는 random forest, XGBoost, LightGBM 기반으로 중요 feature 후보군 2차 선별
- Step 4 (fine-tuning via wrapper): 후보군이 줄어든 뒤 RFE 나 sequential feature selection 으로 최종 subset 결정
---

## Appendix A. Terminology

- **Cross Validation**: 자료를 여러 조각으로 나누어 번갈아 검증에 써서 model 의 일반화 성능을 재는 절차.
- **MDI (Mean Decrease in Impurity)**: Tree 의 node 분할에서 한 feature 가 줄인 불순도의 합.
- **Mutual Information**: 한 변수를 알 때 다른 변수의 entropy 가 줄어드는 양.
- **Permutation Importance**: 한 feature 의 값을 무작위로 섞었을 때의 성능 저하 폭으로 잰 중요도.
- **XOR 문제**: 두 이진 입력이 서로 다를 때만 1 이 되는 관계. 각 입력은 출력과 상관이 0 이지만 두 입력의 조합은 출력을 완전히 결정한다.
- **과적합 (Overfitting)**: Model 이 학습 자료의 잡음까지 학습하여 새 자료에서 성능이 떨어지는 상태.
- **다중공선성 (Multicollinearity)**: 입력 변수들이 서로 강한 선형 관계를 가져, 계수가 개별 변수에 고유하게 배정되지 않는 상태.

## Appendix B. Implementation

scikit-learn 으로 section 5 의 네 단계를 실행하는 class 다. `run` 은 상수 feature 를 먼저 떨어뜨린 뒤 남은 column 에만 나머지 세 단계를 돌린다. 각 단계의 기준값을 생성자로 받고, 각 단계는 원본 column 번호를 그대로 돌려주어 마지막에 고른 feature 의 이름을 찾을 수 있게 한다. 단계마다 method 이름을 `Literal` 로 받아 그 갈래의 members 를 함께 적어 두므로, 무엇이 적용되었는지 서명에서 읽힌다. Filter 단계는 네 이름 (`corr`, `vif`, `mrmr`, `relieff`) 을, embedded 단계는 네 이름 (`random_forest`, `lightgbm`, `lasso`, `elasticnet`) 을, wrapper 단계는 네 이름 (`rfe`, `forward`, `backward`, `genetic`) 을 모두 구현하며, 목록에 없는 이름은 `ValueError` 로 막는다.

입력은 scikit-learn 에 들어 있는 breast cancer dataset 이며, 상수 제거 단계가 보이도록 값이 늘 1.0 인 column 하나를 덧붙여 표본 569 개와 feature 31 개로 만들었다. 원래의 feature 30 개는 서로 중복이 크고, 모두 `StandardScaler` 로 표준화한다.

```python
__author__ = "yRocket"
__version__ = "0.4.0+20260914"

import pathlib
import textwrap
from typing import Literal

import matplotlib
import numpy as np
from lightgbm import LGBMClassifier
from matplotlib import pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (RFE, SelectFromModel, SequentialFeatureSelector,
                                       mutual_info_classif, mutual_info_regression)
from sklearn.linear_model import ElasticNet, Lasso, LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

FIGURE_PATH = pathlib.Path("multivariate-feature-selection-ko_fig/fig2.png")
FIGSIZE: tuple = (9.0, 9.0)
REFERENCE_WIDTH: float = 9.0     # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 9.0
FILTER_METHODS: tuple = ("corr", "vif", "mrmr", "relieff")
EMBEDDED_METHODS: tuple = ("random_forest", "lightgbm", "lasso", "elasticnet")
WRAPPER_METHODS: tuple = ("rfe", "forward", "backward", "genetic")


class MultivariateFeatureSelector:
    """Run the three steps of the workflow on one dataset, keeping the original column indices.

    Each step takes a method name whose Literal lists the members of that branch, so the code says
    which one is applied. A name outside that list raises ValueError.

    Args:
        correlation_limit: absolute correlation above which one feature of a pair is dropped.
        vif_limit: variance inflation factor above which a feature is dropped, one at a time.
        penalty_alpha: weight of the lasso and elastic net penalty of the embedded step.
        elasticnet_ratio: share of the elastic net penalty that is L1, the rest being L2.
        filter_count: number of features the ranking filters (mrmr, relieff) keep.
        neighbour_count: number of hits and misses relieff compares per sample.
        forest_size: number of trees of the random forest used by the embedded step.
        fold_count: number of cross validation folds the sequential and genetic wrappers score on.
        population_size: number of subsets the genetic search holds in one generation.
        generation_count: number of generations the genetic search runs.
        mutation_rate: probability that a child of the genetic search has one feature swapped.
        final_count: number of features the wrapper step leaves.
        random_state: seed of the random forest and of the mutual information estimates.
    """

    def __init__(self, correlation_limit: float = 0.95, vif_limit: float = 10.0,
                 penalty_alpha: float = 0.01, elasticnet_ratio: float = 0.5,
                 filter_count: int = 10, neighbour_count: int = 10,
                 forest_size: int = 200, fold_count: int = 5, population_size: int = 20,
                 generation_count: int = 10, mutation_rate: float = 0.2,
                 final_count: int = 5, random_state: int = 0) -> None:
        if not 0.0 < correlation_limit < 1.0:
            raise ValueError(f"correlation_limit must lie between 0 and 1: {correlation_limit=}")
        if vif_limit <= 1.0:
            raise ValueError(f"vif_limit must exceed 1: {vif_limit=}")
        if penalty_alpha <= 0.0:
            raise ValueError(f"penalty_alpha must be positive: {penalty_alpha=}")
        if not 0.0 < elasticnet_ratio <= 1.0:
            raise ValueError(f"elasticnet_ratio must lie in (0, 1]: {elasticnet_ratio=}")
        if fold_count < 2:
            raise ValueError(f"fold_count must be at least 2: {fold_count=}")
        if population_size < 4 or generation_count < 1:
            raise ValueError(f"the genetic search needs a population and a generation: "
                             f"{population_size=}, {generation_count=}")
        if not 0.0 <= mutation_rate <= 1.0:
            raise ValueError(f"mutation_rate must lie between 0 and 1: {mutation_rate=}")
        if filter_count < 1 or neighbour_count < 1 or final_count < 1:
            raise ValueError(f"counts must be at least 1: {filter_count=}, {neighbour_count=}, {final_count=}")
        self.correlation_limit = correlation_limit
        self.vif_limit = vif_limit
        self.penalty_alpha = penalty_alpha
        self.elasticnet_ratio = elasticnet_ratio
        self.filter_count = filter_count
        self.neighbour_count = neighbour_count
        self.forest_size = forest_size
        self.fold_count = fold_count
        self.population_size = population_size
        self.generation_count = generation_count
        self.mutation_rate = mutation_rate
        self.final_count = final_count
        self.random_state = random_state

    def drop_constant(self, X: np.ndarray) -> np.ndarray:
        """Return the columns whose value changes across the samples, dropping the constant ones."""
        varying = np.where(np.ptp(X, axis=0) > 0.0)[0]
        if len(varying) == 0:
            raise ValueError(f"every feature holds one value: {X.shape=}")
        return varying

    def filter_step(self, X: np.ndarray, y: np.ndarray, columns: np.ndarray,
                    method: Literal["corr", "vif", "mrmr", "relieff"] = "corr") -> np.ndarray:
        """Return the subset of columns the named filter keeps, as indices into the columns of X."""
        if method == "corr":
            return self._by_correlation(X=X, columns=columns)
        if method == "vif":
            return self._by_vif(X=X, columns=columns)
        if method == "mrmr":
            return self._by_mrmr(X=X, y=y, columns=columns)
        if method == "relieff":
            return self._by_relieff(X=X, y=y, columns=columns)
        raise ValueError(f"unknown filter method: {method=}")

    def _by_correlation(self, X: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Drop the later feature of every pair whose absolute correlation exceeds the limit."""
        corr = np.abs(np.corrcoef(X[:, columns], rowvar=False))
        redundant = np.unique(np.where(np.triu(corr, k=1) > self.correlation_limit)[1])
        return columns[np.setdiff1d(np.arange(len(columns)), redundant)]

    def _by_vif(self, X: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Drop the feature of the largest variance inflation factor until every one is under the limit."""
        held = list(columns)
        while len(held) > 1:
            factors = [self._vif_of(X=X, columns=held, position=position) for position in range(len(held))]
            worst = int(np.argmax(factors))
            if factors[worst] <= self.vif_limit:
                break
            held.pop(worst)
        return np.asarray(held)

    def _vif_of(self, X: np.ndarray, columns: list, position: int) -> float:
        """Return 1 / (1 - R^2) of one feature regressed on the remaining ones."""
        target = X[:, columns[position]]
        others = X[:, [column for index, column in enumerate(columns) if index != position]]
        design = np.column_stack([np.ones(len(others)), others])
        residual = target - design @ np.linalg.lstsq(design, target, rcond=None)[0]
        unexplained = float(np.sum(residual ** 2) / np.sum((target - target.mean()) ** 2))
        return float("inf") if unexplained <= 0.0 else 1.0 / unexplained

    def _by_mrmr(self, X: np.ndarray, y: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Add the feature of the largest relevance minus mean redundancy, until filter_count are held."""
        subset = X[:, columns]
        relevance = mutual_info_classif(subset, y, random_state=self.random_state)
        selected = [int(np.argmax(relevance))]
        while len(selected) < min(self.filter_count, len(columns)):
            rest = [position for position in range(len(columns)) if position not in selected]
            redundancy = np.array([
                np.mean([mutual_info_regression(subset[:, [position]], subset[:, chosen],
                                                random_state=self.random_state)[0] for chosen in selected])
                for position in rest])
            selected.append(rest[int(np.argmax(relevance[rest] - redundancy))])
        return columns[np.sort(np.asarray(selected))]

    def _by_relieff(self, X: np.ndarray, y: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Keep the features whose value separates nearest misses from nearest hits the most."""
        subset = X[:, columns]
        span = np.ptp(subset, axis=0)
        score = np.zeros(len(columns))
        for label in np.unique(y):
            hits = self._neighbours_of(source=subset[y == label], pool=subset[y == label], skip_self=True)
            misses = self._neighbours_of(source=subset[y == label], pool=subset[y != label], skip_self=False)
            score += (misses - hits) / (span * len(subset))
        return columns[np.sort(np.argsort(score)[-min(self.filter_count, len(columns)):])]

    def _neighbours_of(self, source: np.ndarray, pool: np.ndarray, skip_self: bool) -> np.ndarray:
        """Return the mean absolute per-feature distance from each source sample to its nearest pool samples."""
        count = min(self.neighbour_count + int(skip_self), len(pool))
        finder = NearestNeighbors(n_neighbors=count).fit(pool)
        neighbours = finder.kneighbors(source, return_distance=False)[:, int(skip_self):]
        return np.abs(source[:, None, :] - pool[neighbours]).sum(axis=(0, 1))

    def embedded_step(self, X: np.ndarray, y: np.ndarray, columns: np.ndarray,
                      method: Literal["random_forest", "lightgbm", "lasso",
                                      "elasticnet"] = "random_forest") -> np.ndarray:
        """Return the columns the named embedded model keeps, as indices into the columns of X."""
        if method == "random_forest":
            return self._by_forest(X=X, y=y, columns=columns)
        if method == "lightgbm":
            return self._by_lightgbm(X=X, y=y, columns=columns)
        if method == "lasso":
            return self._by_penalty(X=X, y=y, columns=columns, l1_ratio=1.0)
        if method == "elasticnet":
            return self._by_penalty(X=X, y=y, columns=columns, l1_ratio=self.elasticnet_ratio)
        raise ValueError(f"unknown embedded method: {method=}")

    def _by_forest(self, X: np.ndarray, y: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Keep the columns whose random forest importance is above the mean importance."""
        forest = RandomForestClassifier(n_estimators=self.forest_size, random_state=self.random_state)
        selector = SelectFromModel(estimator=forest, threshold="mean").fit(X[:, columns], y)
        return columns[selector.get_support()]

    def _by_lightgbm(self, X: np.ndarray, y: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Keep the columns whose gradient boosting split gain is above the mean gain."""
        booster = LGBMClassifier(n_estimators=self.forest_size, importance_type="gain",
                                 random_state=self.random_state, verbose=-1)
        selector = SelectFromModel(estimator=booster, threshold="mean").fit(X[:, columns], y)
        return columns[selector.get_support()]

    def _by_penalty(self, X: np.ndarray, y: np.ndarray, columns: np.ndarray,
                    l1_ratio: float) -> np.ndarray:
        """Keep the columns whose penalized coefficient stays off zero, reading the class label as 0 or 1.

        An L1 share of 1 is the lasso; a smaller share adds the L2 term of the elastic net, which keeps
        correlated features together instead of picking one of them.
        """
        estimator = (Lasso(alpha=self.penalty_alpha, random_state=self.random_state) if l1_ratio == 1.0
                     else ElasticNet(alpha=self.penalty_alpha, l1_ratio=l1_ratio,
                                     random_state=self.random_state))
        selector = SelectFromModel(estimator=estimator, threshold=1e-10).fit(X[:, columns], y)
        return columns[selector.get_support()]

    def wrapper_step(self, X: np.ndarray, y: np.ndarray, columns: np.ndarray,
                     method: Literal["rfe", "forward", "backward", "genetic"] = "rfe") -> np.ndarray:
        """Return the columns the named search keeps, as indices into the columns of X."""
        if len(columns) < self.final_count:
            raise ValueError(f"the wrapper step got fewer columns than it must keep: "
                             f"{len(columns)=}, {self.final_count=}")
        if method == "rfe":
            return self._by_rfe(X=X, y=y, columns=columns)
        if method in ("forward", "backward"):
            return self._by_sequential(X=X, y=y, columns=columns, direction=method)
        if method == "genetic":
            return self._by_genetic(X=X, y=y, columns=columns)
        raise ValueError(f"unknown wrapper method: {method=}")

    def _by_rfe(self, X: np.ndarray, y: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Keep the columns left after dropping the smallest coefficient one feature at a time."""
        estimator = LogisticRegression(max_iter=5000)
        selector = RFE(estimator=estimator, n_features_to_select=self.final_count).fit(X[:, columns], y)
        return columns[selector.get_support()]

    def _by_sequential(self, X: np.ndarray, y: np.ndarray, columns: np.ndarray,
                       direction: Literal["forward", "backward"]) -> np.ndarray:
        """Keep the columns a greedy search holds, adding or removing one by cross validation score."""
        estimator = LogisticRegression(max_iter=5000)
        selector = SequentialFeatureSelector(estimator=estimator, n_features_to_select=self.final_count,
                                             direction=direction, cv=self.fold_count).fit(X[:, columns], y)
        return columns[selector.get_support()]

    def _by_genetic(self, X: np.ndarray, y: np.ndarray, columns: np.ndarray) -> np.ndarray:
        """Keep the subset of the wanted size that a genetic search scores highest.

        Every individual is a subset of exactly final_count features, so the generations compare
        subsets of one size instead of trading size against score.
        """
        rng = np.random.default_rng(self.random_state)
        parent_count = max(2, self.population_size // 2)
        population = [np.sort(rng.choice(len(columns), size=self.final_count, replace=False))
                      for _ in range(self.population_size)]
        for _ in range(self.generation_count):
            parents = sorted(population, key=lambda held: -self._fitness_of(X=X, y=y, columns=columns[held]))
            parents = parents[:parent_count]
            population = parents + [self._child_of(parents=parents, position_count=len(columns), rng=rng)
                                    for _ in range(self.population_size - parent_count)]
        best = max(population, key=lambda held: self._fitness_of(X=X, y=y, columns=columns[held]))
        return columns[np.sort(best)]

    def _fitness_of(self, X: np.ndarray, y: np.ndarray, columns: np.ndarray) -> float:
        """Return the mean cross validation score of a model fitted on the given columns."""
        estimator = LogisticRegression(max_iter=5000)
        return float(np.mean(cross_val_score(estimator, X[:, columns], y, cv=self.fold_count)))

    def _child_of(self, parents: list, position_count: int, rng: np.random.Generator) -> np.ndarray:
        """Draw a child from the union of two parents, then swap one of its features at random."""
        first, second = rng.choice(len(parents), size=2, replace=False)
        pool = np.union1d(parents[first], parents[second])
        child = rng.choice(pool, size=self.final_count, replace=False)
        outside = np.setdiff1d(np.arange(position_count), child)
        if rng.random() < self.mutation_rate and len(outside) > 0:
            child[rng.integers(len(child))] = rng.choice(outside)
        return np.sort(child)

    def run(self, X: np.ndarray, y: np.ndarray,
            filter_method: Literal["corr", "vif", "mrmr", "relieff"] = "corr",
            embedded_method: Literal["random_forest", "lightgbm", "lasso",
                                     "elasticnet"] = "random_forest",
            wrapper_method: Literal["rfe", "forward", "backward", "genetic"] = "rfe") -> dict:
        """Return the surviving column indices of each step, keyed by step name.

        The constant features go first: they carry nothing any later step can weigh.
        """
        varying = self.drop_constant(X=X)
        filtered = self.filter_step(X=X, y=y, columns=varying, method=filter_method)
        embedded = self.embedded_step(X=X, y=y, columns=filtered, method=embedded_method)
        wrapped = self.wrapper_step(X=X, y=y, columns=embedded, method=wrapper_method)
        return {"varying": varying, "filter": filtered, "embedded": embedded, "wrapper": wrapped}


def draw_matrix(kept: dict, names: np.ndarray, columns: np.ndarray, path: pathlib.Path) -> None:
    """Draw one cell per (feature, method) pair, filled where that method kept the feature.

    Args:
        kept: method name mapped to the column indices that method kept.
        names: feature name of every column of X.
        columns: the columns that reach the methods, drawn as the rows of the chart.
        path: file the chart is written to.
    """
    order = columns[np.argsort(names[columns])]
    methods = list(kept)
    grid = np.array([[1.0 if column in set(kept[method]) else 0.0 for method in methods] for column in order])

    font_size = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
    fig, axes = plt.subplots(figsize=FIGSIZE)
    axes.imshow(grid, aspect="auto", vmin=0.0, vmax=1.0,
                cmap=matplotlib.colors.ListedColormap(["#f2f2f2", matplotlib.colors.TABLEAU_COLORS["tab:blue"]]))
    axes.set_xticks(range(len(methods)),
                    [f"{method}\n({len(kept[method])})" for method in methods],
                    rotation=45, ha="right", fontsize=font_size)
    axes.set_yticks(range(len(order)), names[order], fontsize=font_size)
    axes.set_xticks(np.arange(len(methods) + 1) - 0.5, minor=True)
    axes.set_yticks(np.arange(len(order) + 1) - 0.5, minor=True)
    axes.grid(which="minor", color="white", linewidth=1.5)
    axes.tick_params(which="minor", length=0)
    for boundary in np.cumsum([len(FILTER_METHODS), len(EMBEDDED_METHODS)]) - 0.5:
        axes.axvline(boundary, color="white", linewidth=5.0)
    for spine in axes.spines.values():
        spine.set_visible(False)
    axes.set_xlabel("Selection method, with the number of features it keeps", fontsize=font_size)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    data = load_breast_cancer()
    # Append a column that never changes, to show the first step removing it
    X = np.column_stack([StandardScaler().fit_transform(data.data), np.full(len(data.data), 1.0)])
    names = np.asarray(list(data.feature_names) + ["constant probe"])
    selector = MultivariateFeatureSelector(correlation_limit=0.95, final_count=5)
    varying = selector.drop_constant(X=X)

    kept = {}
    for filter_method in FILTER_METHODS:
        kept[filter_method] = selector.filter_step(X=X, y=data.target, columns=varying, method=filter_method)
    for embedded_method in EMBEDDED_METHODS:
        kept[embedded_method] = selector.embedded_step(X=X, y=data.target, columns=varying,
                                                       method=embedded_method)
    for wrapper_method in WRAPPER_METHODS:
        kept[wrapper_method] = selector.wrapper_step(X=X, y=data.target, columns=kept["random_forest"],
                                                     method=wrapper_method)

    workflow = selector.run(X=X, y=data.target, filter_method="corr",
                            embedded_method="random_forest", wrapper_method="rfe")
    draw_matrix(kept=kept, names=names, columns=varying, path=FIGURE_PATH)
    print(f"{X.shape[0]} samples and {X.shape[1]} features in, "
          f"{len(workflow['wrapper'])} out; chart written to {FIGURE_PATH}")
```

상수 column 은 첫 단계에서 떨어져 어느 filter 에도 닿지 않는다. 남은 30 개에서 네 filter 는 23, 17, 10, 10 개를, 네 embedded 는 9, 6, 12, 18 개를 남겨 서로 다른 답을 내며, L2 를 섞은 `elasticnet` 이 상관된 무리를 함께 남겨 `lasso` 보다 6 개를 더 든다. 네 wrapper 는 random forest 가 남긴 9 개에서 저마다 5 개를 고르는데, `forward` 와 `backward` 는 같은 조합에 닿고 `rfe` 와 `genetic` 은 저마다 다른 조합을 집는다. `run` 이 기본값으로 받는 `corr` → `random_forest` → `rfe` 로 이어 가면 feature 수가 31, 30, 23, 6, 5 로 줄고, 비용이 가장 큰 wrapper 는 6 개만 남은 자리에서 돈다.

어느 method 가 어느 feature 를 남겼는지는 Fig 2 에 있다.

<img src="multivariate-feature-selection-ko_fig/fig2.png" width="800" style="max-width: 100%;" alt="Fig 2">

Fig 2. Which features each selection method keeps

- 행은 상수 제거 뒤 남은 feature 30 개를 이름순으로, 열은 method 12 개를 filter, embedded, wrapper 순으로 두고, 세 갈래 사이는 열 간격을 넓혀 갈랐다. 칸이 채워진 것은 그 method 가 그 feature 를 남겼다는 뜻이다.
- 열 이름 아래 괄호 안 숫자는 그 method 가 남긴 feature 수이며, wrapper 세 열은 random forest 가 남긴 9 개 위에서 돌린 결과다.
- `mean concave points` 는 열두 열 가운데 아홉에서, `worst concave points` 는 열하나에서 채워진다. 반대로 `worst compactness` 는 corr 한 열에만 남는다.

### B.1 Choosing Among Answers 🥑

Method 마다 최대화하려는 양이 달라 남는 열이 갈린다. corr 은 상관이 기준을 넘는 쌍에서 열 순서상 뒤를 버리고, VIF 는 나머지로 잘 설명되는 쪽을 버리며, mRMR 은 이미 고른 것과의 중복을, ReliefF 는 이웃까지의 거리를, Lasso 는 무리에서 하나만 남기는 penalty 를, ElasticNet 은 무리를 함께 남기는 penalty 를, tree 계열은 분할 이득을, wrapper 는 그 model 의 cross validation 점수를 본다. 무엇을 쓸지는 선택에 쓰지 않은 분할에서의 검증 점수로 정한다.

갈린 답이 실제로 다른 성능을 뜻하는 경우는 드물다. 자료에 서로 대체 가능한 feature 가 많으면 여러 집합이 거의 같은 점수를 내고, 그 가운데 누구를 남길지는 신호가 아니라 각 기준의 tie-break 규칙이 정한다. 이 예제의 breast cancer data 는 feature 30 개 가운데 상관 0.9 이상인 쌍이 21 개이고 `mean radius` 와 `mean perimeter` 는 0.998 로 사실상 같은 열이다.

Table 2. Cross validation score of each wrapper subset of the breast cancer example

| Wrapper  | Features it keeps<br>(input) | 10-fold accuracy<br>(output) | Selected features<br>(output)                                                                |
| :------: | :--------------------------: | :--------------------------: | :------------------------------------------------------------------------------------------: |
| rfe      | 5                            | 0.949 ± 0.025                | area error, worst area, <ins>worst concave points</ins>, worst perimeter, worst radius       |
| forward  | 5                            | 0.954 ± 0.037                | mean concavity, worst area, <ins>worst concave points</ins>, worst perimeter, worst radius   |
| backward | 5                            | 0.954 ± 0.037                | mean concavity, worst area, <ins>worst concave points</ins>, worst perimeter, worst radius   |
| genetic  | 5                            | 0.953 ± 0.039                | area error, mean concave points, mean concavity, worst area, <ins>worst concave points</ins> |

네 wrapper 는 목표 개수를 인자로 받으며, 이 예제는 `final_count=5` 다. 그래서 네 집합의 크기가 같지만 다르게 골랐기에 점수 차이가 발생한다. 그 차이가 표준편차 안에 들어오므로, 이 자료에서는 점수만으로 하나를 고를 수 없다. 그럴 때는 아래 순서로 내려간다.

1️⃣ Step 1 (agreement): 여러 method 가 공통으로 고른 feature 를 먼저 믿는다. `worst concave points` 는 Table 2 의 네 집합 모두에 들어 있다<br>
2️⃣ Step 2 (stability): 자료를 재표본해도 같은 집합이 나오는 쪽, 곧 더 안정적인 쪽을 고른다<br>
3️⃣ Step 3 (actionability): 그래도 남으면 공정에서 손댈 수 있거나 뜻이 읽히는 feature 를 고른다
