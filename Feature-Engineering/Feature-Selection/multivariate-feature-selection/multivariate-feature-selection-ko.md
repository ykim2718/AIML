# Multivariate Feature Selection
Rev. 65 | Created: 2026-09-12 | Updated: 2026-09-17 09:26 CDT

## 1. Purpose

- **Problem Statement**: Feature 의 개별 평가는 feature 간의 결합 정보와 중복 정보를 처리하지 못해 feature 의 최적화를 하지 못한다.
- **Goal**: Feature 사이의 상호작용과 중복성을 함께 보는 선택 기법에 대한 taxonomy 와 hierarchy 를 세우고 정리하여, 상황에 맞춰 적용하거나 빠른 지침 (rule of thumb) 으로 쓰게 한다.
- **Non-Goal**: Feature 의 개별 검증은 [Univariate Feature Selection](../univariate-feature-selection/univariate-feature-selection-ko.md) 에서 다룬다.

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

두 갈래와 나란히, y 를 보는지로도 갈린다. 상관 filter 와 VIF 는 X 안의 상관만 계산하여 y 없이 돌아가고, mRMR 과 ReliefF 와 embedded 와 wrapper 는 모두 y 와의 관계를 재어 고른다.

## 3. Approach-based Methods

- 3.1 Filter: model 없이 X 와 y 의 통계량만 계산
- 3.2 Wrapper: model 을 외부 채점기로 두고, 후보 subset 마다 다시 적합하여 점수를 비교 (적합 횟수 = 후보 수)
- 3.3 Embedded: 한 번의 적합 안에서 penalty 나 split gain 이 선택을 수행 (적합 횟수 = 1)

### 3.1 Multivariate Filter Methods

Model 학습 없이 data 의 통계적 특성만으로 feature 조합을 선별한다. Univariate filter 와 달리 feature 사이의 상관성을 함께 계산한다.

mRMR (Minimum Redundancy Maximum Relevance) 는 target 과의 mutual information 을 최대화하고 선택된 feature 사이의 mutual information 을 최소화하는 최적화 문제로 푼다.

```math
\max_{S} \left[ \frac{1}{|S|} \sum_{i \in S} I(x_i; y)
- \frac{1}{|S|^2} \sum_{i, j \in S} I(x_i; x_j) \right]
\hspace{10em} (1)
```

VIF (Variance Inflation Factor) 는 한 feature 를 나머지 feature 로 회귀하여 그 설명력을 측정한다. $\mathrm{VIF} \gt 10$ 인 feature 를 순차적으로 제거한다. 정의와 제거 절차는 [Appendix C](#appendix-c-variance-inflation-factor) 에 있다.

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
- ReliefF: 표본마다 가까운 같은 class 와 다른 class 의 이웃을 전체 feature vector 의 거리 위에서 비교하므로, 다른 feature 와 함께일 때만 드러나는 차이가 점수에 들어감. Class label 전용이며, 회귀 target 에는 RReliefF 가 따로 있음

### 4.3 Dimensionality Tradeoff

남길 차원 수를 신호와 맞바꾸는 갈래이며, 이미 매겨진 순위나 penalty path 위에서 자를 자리를 정하는 cut-off 다. 그래서 순위를 내는 기준이 먼저 있어야 적용되며, 보통 section 4.1 과 section 4.2 의 순위 위에 붙는다. Lasso 의 $\lambda$ 는 예외로, 같은 penalty 가 순위와 cut-off 를 함께 정한다.

- Lasso 의 $\lambda$: 값이 클수록 0 이 되는 계수가 늘어 차원이 줄어듦
- RFE 의 목표 feature 개수: 남길 차원을 직접 지정
- Tree-based importance 의 문턱값: 평균 중요도 같은 기준으로 자를 자리를 정함

## 5. Target Kind

분류와 회귀는 target 의 성질이고, section 3 의 approach 갈래와 section 4 의 interaction 갈래 어느 쪽과도 직교한다. 같은 method 가 criterion 을 유지한 채 estimator 만 target 에 맞는 것으로 교체한다.

Table 2. What each method changes when the target is regression instead of classification

| Section                  | Method                                     | Uses y | Classification target           | Regression target                       |
| :----------------------: | :----------------------------------------: | :----: | :-----------------------------: | :-------------------------------------: |
| 3.1 Filter               | corr, VIF                                  | No     | Correlation among X only        | Correlation among X only                |
| 3.1 Filter               | mRMR                                       | Yes    | `mutual_info_classif`           | `mutual_info_regression`                |
| 3.1 Filter / 4.2 Synergy | ReliefF                                    | Yes    | hit/miss contrast               | None (RReliefF is a separate algorithm) |
| 3.3 Embedded             | random forest, LightGBM, gradient boosting | Yes    | Classifier                      | Regressor                               |
| 3.3 Embedded             | lasso, elastic net                         | Yes    | Regression fit on the 0/1 label | Regression fit on y                     |
| 3.2 Wrapper              | RFE, forward/backward, genetic             | Yes    | Scored by `LogisticRegression`  | Scored by `LinearRegression`            |

- `Uses y` 가 No 인 행: 두 target 에서 같은 계산
- `Uses y` 가 Yes 인 행: 추정량과 model 만 회귀용으로 교체
- ReliefF: 회귀에서 쓸 수 있는 대응물이 없어 알고리즘 자체가 갈림

## 6. Selection Instability

Selection instability 는 자료나 method 를 조금만 바꿔도 고른 열이 바뀌는 성질이며, 점수는 거의 그대로인 채 이름만 갈린다. 원인은 둘이다.

- Interchangeable features: 상관이 높아 서로 바꿔 써도 성능이 같은 열들의 equivalence class
- Sampling noise: 재표본마다 상관과 중요도 순위가 흔들려, 문턱 가까이 있던 feature 의 당락이 뒤집힘

대책은 한 번의 선택 대신 반복 선택의 빈도나 상관 무리 단위의 처리를 쓴다.

- Stability selection: bootstrap 표본마다 선택을 되풀이하고, 선택 빈도가 문턱 (예: 0.6) 을 넘는 feature 만 남김
- Cluster representative: 상관으로 feature 를 clustering 한 뒤 무리마다 하나를 대표로 남겨, 무리 안의 교체를 없앰
- Group-wise selection: ElasticNet 이나 group lasso 로 상관된 무리를 함께 남겨, 하나만 뽑히는 것을 막음
- Selection frequency reporting: 고른 집합 하나가 아니라 feature 마다의 선택 빈도를 함께 적어, 바꿔 쓸 수 있는 열을 드러냄

## 7. Workflow

비용이 낮은 기법으로 후보를 줄인 뒤 비싼 기법을 쓴다. Wrapper 의 비용은 남은 feature 개수에 따라 커지므로, 그 앞에 세 단계를 둔다.

- Step 1 (constant removal): 모든 표본에서 값이 같은 feature 를 먼저 제거. 상관도 중요도도 정의되지 않고, 뒤 단계가 가릴 것이 없음
- Step 2 (pre-filtering): univariate 통계량 또는 VIF 로 상관계수 0.95 이상인 중복 feature 를 1차 제거
- Step 3 (embedded selection): Lasso 또는 random forest, XGBoost, LightGBM 기반으로 중요 feature 후보군 2차 선별
- Step 4 (fine-tuning via wrapper): 후보군이 줄어든 뒤 RFE 나 sequential feature selection 으로 최종 subset 결정
---

## Appendix A. Terminology

- **Cross Validation**: 자료를 여러 조각으로 나누어 번갈아 검증에 써서 model 의 일반화 성능을 재는 절차.
- **MDI (Mean Decrease in Impurity)**: Tree 의 node 분할에서 한 feature 가 줄인 불순도의 합.
- **Multicollinearity**: 입력 변수들이 서로 강한 선형 관계를 가져, 계수가 개별 변수에 고유하게 배정되지 않는 상태.
- **Mutual Information**: 한 변수를 알 때 다른 변수의 entropy 가 줄어드는 양.
- **Overfitting**: Model 이 학습 자료의 잡음까지 학습하여 새 자료에서 성능이 떨어지는 상태.
- **Permutation Importance**: 한 feature 의 값을 무작위로 섞었을 때의 성능 저하 폭으로 잰 중요도.
- **XOR Problem**: 두 이진 입력이 서로 다를 때만 1 이 되는 관계. 각 입력은 출력과 상관이 0 이지만 두 입력의 조합은 출력을 완전히 결정한다.

## Appendix B. Implementation

scikit-learn 으로 section 7 의 네 단계를 실행하는 class 다. `run` 은 상수 feature 를 먼저 떨어뜨린 뒤 남은 column 에만 나머지 세 단계를 돌린다. 각 단계의 기준값을 생성자로 받고, 각 단계는 원본 column 번호를 그대로 돌려주어 마지막에 고른 feature 의 이름을 찾을 수 있게 한다. 단계마다 method 이름을 그 갈래의 `Literal` 별칭으로 받으며, members 는 class 안의 그 별칭 한 곳에만 적고, 곁에 둔 목록 tuple 은 `get_args` 로 파생시킨다. Filter 단계는 네 이름 (`corr`, `vif`, `mrmr`, `relieff`) 을, embedded 단계는 다섯 이름 (`random_forest`, `lightgbm`, `gradient_boosting`, `lasso`, `elasticnet`) 을, wrapper 단계는 네 이름 (`rfe`, `forward`, `backward`, `genetic`) 을 모두 구현하며, 목록에 없는 이름은 `ValueError` 로 막는다. 생성자가 받는 `task` 는 각 단계 뒤에 설 model 을 Table 2 대로 고르며, 회귀에서 `relieff` 를 부르면 `ValueError` 와 함께 그 task 가 쓸 수 있는 filter 목록을 돌려준다.

입력은 scikit-learn 에 들어 있는 breast cancer dataset 이며, 상수 제거 단계가 보이도록 값이 늘 1.0 인 column 하나를 덧붙여 표본 569 개와 feature 31 개로 만들었다. 원래의 feature 30 개는 서로 중복이 크고, 모두 `StandardScaler` 로 표준화한다. Label 이 이진이므로 예제는 분류 model 로 짰고, 회귀 target 이면 Table 2 의 오른쪽 열로 바꾼다.

Class 는 [src/multivariate_feature_selection.py](src/multivariate_feature_selection.py) 에 있으며, `python src/multivariate_feature_selection.py` 로 돌리면 Fig 2 를 다시 그린다.

상수 column 은 첫 단계에서 떨어져 어느 filter 에도 닿지 않는다. 남은 30 개에서 네 filter 는 23, 17, 10, 10 개를, 다섯 embedded 는 9, 6, 5, 12, 18 개를 남겨 서로 다른 답을 내며, L2 를 섞은 `elasticnet` 이 상관된 무리를 함께 남겨 `lasso` 보다 6 개를 더 든다. 네 wrapper 는 random forest 가 남긴 9 개에서 저마다 5 개를 고르는데, `forward` 와 `backward` 는 같은 조합에 닿고 `rfe` 와 `genetic` 은 저마다 다른 조합을 집는다. `run` 이 기본값으로 받는 `corr` → `random_forest` → `rfe` 로 이어 가면 feature 수가 31, 30, 23, 6, 5 로 줄고, 비용이 가장 큰 wrapper 는 6 개만 남은 자리에서 돈다.

어느 method 가 어느 feature 를 남겼는지는 Fig 2 에 있다.

<img src="multivariate-feature-selection-ko_fig/fig2.png" width="800" style="max-width: 100%;" alt="Fig 2">

Fig 2. Which features each selection method keeps

- 행은 상수 제거 뒤 남은 feature 30 개를 이름순으로, 열은 method 13 개를 filter, embedded, wrapper 순으로 두고, 세 갈래 사이는 열 간격을 넓혀 갈랐다. 칸이 채워진 것은 그 method 가 그 feature 를 남겼다는 뜻이다.
- 열 이름 아래 괄호 안 숫자는 그 method 가 남긴 feature 수이며, wrapper 네 열은 random forest 가 남긴 9 개 위에서 돌린 결과다.
- `mean concave points` 는 열세 열 가운데 열에서, `worst concave points` 는 열둘에서 채워진다. 반대로 `worst compactness` 는 corr 한 열에만 남는다.

### B.1 Choosing Among Answers 🥑

Method 마다 최대화하려는 양이 달라 남는 열이 갈린다. corr 은 상관이 기준을 넘는 쌍에서 열 순서상 뒤를 버리고, VIF 는 나머지로 잘 설명되는 쪽을 버리며, mRMR 은 이미 고른 것과의 중복을, ReliefF 는 이웃까지의 거리를, Lasso 는 무리에서 하나만 남기는 penalty 를, ElasticNet 은 무리를 함께 남기는 penalty 를, tree 계열은 분할 이득을, wrapper 는 그 model 의 cross validation 점수를 본다. 무엇을 쓸지는 선택에 쓰지 않은 분할에서의 검증 점수로 정한다.

갈린 답이 실제로 다른 성능을 뜻하는 경우는 드물다. 자료에 서로 대체 가능한 feature 가 많으면 여러 집합이 거의 같은 점수를 내고, 그 가운데 누구를 남길지는 신호가 아니라 각 기준의 tie-break 규칙이 정한다. 이 예제의 breast cancer data 는 feature 30 개 가운데 상관 0.9 이상인 쌍이 21 개이고 `mean radius` 와 `mean perimeter` 는 0.998 로 사실상 같은 열이다.

Table 3. Cross validation score of each wrapper subset of the breast cancer example

| Wrapper  | Features it keeps<br>(input) | 10-fold accuracy<br>(output) | Selected features<br>(output)                                                                |
| :------: | :--------------------------: | :--------------------------: | :------------------------------------------------------------------------------------------: |
| rfe      | 5                            | 0.949 ± 0.025                | area error, worst area, <ins>worst concave points</ins>, worst perimeter, worst radius       |
| forward  | 5                            | 0.954 ± 0.037                | mean concavity, worst area, <ins>worst concave points</ins>, worst perimeter, worst radius   |
| backward | 5                            | 0.954 ± 0.037                | mean concavity, worst area, <ins>worst concave points</ins>, worst perimeter, worst radius   |
| genetic  | 5                            | 0.953 ± 0.039                | area error, mean concave points, mean concavity, worst area, <ins>worst concave points</ins> |

네 wrapper 는 목표 개수를 인자로 받으며, 이 예제는 `final_count=5` 다. 그래서 네 집합의 크기가 같지만 다르게 골랐기에 점수 차이가 발생한다. 그 차이가 표준편차 안에 들어오므로, 이 자료에서는 점수만으로 하나를 고를 수 없다. 그럴 때는 아래 순서로 내려간다.

1️⃣ Step 1 (agreement): 여러 method 가 공통으로 고른 feature 를 먼저 남긴다. `worst concave points` 는 Table 3 의 네 집합 모두에 들어 있다<br>
2️⃣ Step 2 (stability): 자료를 재표본해도 같은 집합이 나오는 쪽을 고른다. 재는 방법은 section 6 에 있다<br>
3️⃣ Step 3 (actionability): 그래도 남으면 공정에서 손댈 수 있거나 뜻이 읽히는 feature 를 고른다

## Appendix C. Variance Inflation Factor

VIF 는 feature $x_i$ 를 나머지 feature 전체로 회귀했을 때의 결정계수 $R_i^2$ 로 정의되며, 그 feature 가 나머지의 선형 결합으로 얼마나 재현되는지를 잰다.

```math
\mathrm{VIF}_i = \frac{1}{1 - R_i^2} \hspace{19em} (2)
```

- $R_i^2$: $x_i$ 를 나머지 feature 로 회귀한 결정계수
- 값의 범위: $R_i^2 = 0$ 이면 1, $R_i^2$ 가 1 에 가까울수록 커지고 완전 공선성에서 무한대
- 이름의 유래: 회귀 계수 $\hat{\beta}_i$ 의 분산이 공선성 없는 경우의 VIF 배가 됨

제거는 한 번에 하나씩 되풀이한다.

- 남은 feature 마다 식 (2) 의 값을 계산
- 가장 큰 값이 한계를 넘으면 그 feature 하나를 제거
- 남은 feature 의 값을 다시 계산하여, 모두 한계 아래로 내려올 때까지 되풀이

한 번에 하나만 빼는 이유는 서로를 부풀리던 쌍에서 하나가 빠지면 남은 쪽의 값도 함께 내려가기 때문이며, 여럿을 한꺼번에 빼면 남겨도 될 feature 까지 잃는다.

Table 4. Reading of a VIF value

| VIF     | Reading                              |
| :-----: | :----------------------------------: |
| 1       | 나머지 feature 로 설명되지 않음      |
| 1–5     | 약한 공선성, 보통 그대로 둠          |
| 5–10    | 중간 공선성, 자료와 목적에 따라 판단 |
| &gt; 10 | 강한 공선성, 제거 대상               |
| ∞       | 완전 공선성, 나머지의 선형 결합      |

상관 filter 와 갈리는 자리는 보는 단위다. 상관 filter 는 feature 쌍의 상관만 보므로 셋 이상이 합쳐 만드는 공선성을 지나치고, VIF 는 나머지 전체에 대한 다중 회귀이므로 그 경우를 잡는다. 대신 feature 마다 회귀를 한 번씩 풀고 제거할 때마다 다시 풀어야 하므로 비용이 크다.

- 범주형 dummy: 한 변수에서 나온 dummy 들은 서로 공선이므로 값이 늘 높게 나오며, 기준 범주를 뺀 뒤 변수 단위로 읽음
- 완전 공선성: $R_i^2 = 1$ 이면 값이 무한대이고, Appendix B 의 `_vif_of` 는 그 자리를 `float("inf")` 로 돌려주어 다음 제거 대상이 되게 함
- Target 과의 무관: 식 (2) 에 $y$ 가 들어가지 않으므로 Table 2 의 `Uses y` 가 No 인 자리에 있음
