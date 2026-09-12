# Multivariate Feature Selection (Korean)
Rev. 0 | Created: 2026-09-12 | Updated: 2026-09-12 18:05 CDT

## 1. Purpose

- **Problem Statement**: Feature 를 하나씩만 평가하면 두 feature 가 결합해야 드러나는 신호를 놓치고, 같은 정보를 담은 feature 가 함께 남는다.
- **Goal**: Feature 사이의 상호작용과 중복성을 함께 보는 선택 기법을 접근 방식별로 갈라 놓아, 주어진 data 크기와 계산 예산에서 어느 기법을 쓸지 독자가 고를 수 있게 한다.
- **Non-Goal**: Feature 를 하나씩 검정하는 기법은 다루지 않는다. 그것은 [Univariate Feature Selection](../univariate-feature-selection/univariate-feature-selection-ko.md) 의 주제다.

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
|       +-- Tree-based importance ......... random forest, XGBoost
|
+-- 2. Interaction-based hierarchy
    +-- Redundancy reduction ............... removing duplicated information
    +-- Feature synergy .................... keeping features that matter together
    +-- Dimensionality tradeoff ............ trading dimension against signal
```

Fig 1. Two hierarchies of multivariate feature selection

## 3. Motivation

Feature 조합이 개별 feature 보다 target 을 더 잘 설명하는 경우가 있으므로, 선택은 조합 단위로 이루어져야 한다. $X_1$ 과 $X_2$ 가 각각은 target $Y$ 와 낮은 상관을 보여도 두 feature 의 조합은 $Y$ 를 설명하는 강한 신호가 될 수 있으며, XOR 문제가 그 대표적인 예다.

다변량 분석의 목적은 셋이다.

- Feature 사이의 다중공선성 및 중복성 제거
- Feature 사이의 시너지 효과 발굴
- Model 성능 향상과 과적합 방지

## 4. Approach-based Methods

접근 방식은 계산 비용과 답의 성질을 정한다. Filter 는 자료의 성질을, wrapper 는 그 model 과 탐색의 성질을, embedded 는 적합된 model 의 성질을 답으로 내놓는다.

### 4.1 Multivariate Filter Methods

Model 학습 없이 data 의 통계적 특성만으로 feature 조합을 선별한다. Univariate filter 와 달리 feature 사이의 상관성을 함께 계산한다.

mRMR (Minimum Redundancy Maximum Relevance) 는 target 과의 mutual information 을 최대화하고 선택된 feature 사이의 mutual information 을 최소화하는 최적화 문제로 푼다.

```math
\max_{S} \left[ \frac{1}{|S|} \sum_{i \in S} I(x_i; y)
- \frac{1}{|S|^2} \sum_{i, j \in S} I(x_i; x_j) \right]
\hspace{10em} (1)
```

VIF (Variance Inflation Factor) 는 한 feature 를 나머지 feature 로 회귀하여 그 설명력을 측정한다. $\mathrm{VIF} \gt 10$ 인 feature 를 순차적으로 제거한다.

### 4.2 Wrapper Methods

특정 model 을 검증 도구로 삼아, 최적의 성능을 내는 feature subset 을 탐색 algorithm 으로 찾는다.

RFE (Recursive Feature Elimination) 의 절차는 다음과 같다.

- 전체 feature 로 model 을 학습
- 계수나 feature 중요도가 가장 낮은 feature 를 제거
- 목표 feature 개수에 닿을 때까지 반복

Greedy search 는 feature 를 하나씩 추가 (forward) 하거나 제거 (backward) 하며 cross validation 점수의 변화를 추적한다.

### 4.3 Embedded Methods

Model 의 학습 algorithm 안에 feature 선택 과정이 들어 있다.

Lasso 는 손실 함수에 계수 절댓값의 합 $\lambda \sum |\beta_i|$ 을 penalty 로 더하여, 불필요한 feature 의 계수를 정확히 0 으로 보낸다.

Tree-based importance 는 tree model 의 node 분할 기여도 (MDI) 나 값을 무작위로 섞었을 때의 성능 저하 폭 (permutation importance) 으로 다변량 관점의 중요도를 계산한다.

## 5. Comparison

세 방식은 계산 비용과 상호작용 반영 정도가 서로 반대 방향으로 움직인다.

Table 1. Comparison of the three approaches

| Aspect | Multivariate filter | Wrapper | Embedded |
| --- | --- | --- | --- |
| 계산 복잡도 | 낮음 | 매우 높음 | 중간 |
| 과적합 위험 | 낮음 | 높음 | 중간 |
| Model 의존성 | 없음 (model-agnostic) | 선택한 model 에 종속 | 해당 model 에 내장 |
| 상호작용 반영 | 제한적 (주로 1:1 중복 제거) | 매우 잘 반영 | 잘 반영 |

## 6. Workflow

비용이 낮은 기법으로 후보를 줄인 뒤 비싼 기법을 쓴다. Wrapper 의 비용은 남은 feature 개수에 따라 커지므로, 그 앞에 두 단계를 둔다.

- Step 1 (pre-filtering): univariate 통계량 또는 VIF 로 상관계수 0.95 이상인 중복 feature 와 분산이 0 인 feature 를 1차 제거
- Step 2 (embedded selection): Lasso 또는 random forest, XGBoost 기반으로 중요 feature 후보군 2차 선별
- Step 3 (fine-tuning via wrapper): 후보군이 줄어든 뒤 RFE 나 sequential feature selection 으로 최종 subset 결정

---

## Appendix A. Terminology

- **Cross Validation**: 자료를 여러 조각으로 나누어 번갈아 검증에 써서 model 의 일반화 성능을 재는 절차.
- **MDI (Mean Decrease in Impurity)**: Tree 의 node 분할에서 한 feature 가 줄인 불순도의 합.
- **Mutual Information**: 한 변수를 알 때 다른 변수의 entropy 가 줄어드는 양.
- **Permutation Importance**: 한 feature 의 값을 무작위로 섞었을 때의 성능 저하 폭으로 잰 중요도.
- **XOR 문제**: 두 이진 입력이 서로 다를 때만 1 이 되는 관계. 각 입력은 출력과 상관이 0 이지만 두 입력의 조합은 출력을 완전히 결정한다.
- **과적합 (Overfitting)**: Model 이 학습 자료의 잡음까지 학습하여 새 자료에서 성능이 떨어지는 상태.
- **다중공선성 (Multicollinearity)**: 입력 변수들이 서로 강한 선형 관계를 가져, 계수가 개별 변수에 고유하게 배정되지 않는 상태.
