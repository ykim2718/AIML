# Nonlinearity in Linear Models
Rev. 8 | Created: 2026-09-23 | Updated: 2026-09-23 09:40 CDT

## 1. Purpose

- **Problem Statement**: 수리통계와 machine learning model 의 선형성 (linearity) 과 비선형성 (non-linearity), 그리고 data 의 선형 특성 (linear property) 과 비선형 특성 (non-linear property) 의 차이에 대한 이해가 부족하여 modeling 전략이 혼란하다.
- **Goal**: 수리통계학적 선형성 (linearity) 을 바탕으로, machine learning 에서 변수 사이의 상호작용 (feature interaction) 과 비선형 data 특성을 다루는 modeling 전략을 비교 분석한다.
- **Non-Goal**: Model 상세는 다루지 않는다.

## 2. Summary

Model 의 선형성과 data 의 비선형 특성은 서로 다른 대상입니다. 선형성은 가중치 $\beta$ 에 대한 1차성이고, 비선형 특성은 $x$ 와 $y$ 사이 관계의 곡률과 변수 사이의 interaction 입니다. 그래서 입력을 비선형으로 변환해도 $\beta$ 에 대한 1차 구조는 남고, 최소제곱법과 Ridge, Lasso, PLS 의 해법을 그대로 사용합니다 (4.2 절).

비선형 data 특성을 누가 담당하는지가 두 modeling 전략을 가릅니다.

- **Feature-Intensive Model**: 선형 모델 + 비선형 Feature (특성 공학 접근법). "모델은 단순 (선형) 하게 두고, 데이터 (Feature) 를 복잡하게 만든다." 분석가가 $x^2$, $x_1 x_2$ 같은 열을 만들어 넣는 선형 모델입니다.
- **Algorithm-Intensive Model**: 비선형 모델 + 선형 Feature (알고리즘 접근법). "데이터 (Feature) 는 있는 그대로 (선형/원본) 두고, 모델을 복잡 (비선형) 하게 만든다." 원본 열을 그대로 넣고 tree ensemble 이나 neural network 가 내부에서 학습하는 접근입니다.

## 3. Taxonomy and its Hierarchy

비선형성은 model 의 선형성, feature 의 선형성, 읽어 내는 값의 세 축으로 갈립니다. Feature-Intensive Model 은 선형 model 에 비선형 feature 를 넣어 항마다의 계수를 읽고, Algorithm-Intensive Model 은 비선형 model 에 선형 feature 를 넣어 변수 중요도를 읽습니다. [Fig 1](#fig-1) 이 그 세 축과 각 model 에 속한 방법입니다.

```text
Nonlinearity in a model
|
+-- Feature-Intensive Model ........................ linear model + non-linear feature
|     +-- Power term: x^2, x^3 ..................... curvature of one variable
|     +-- Interaction term: x1 * x2 ................ joint effect of two variables
|     +-- Basis expansion: spline, RBF ............. local shape without a global degree
|
+-- Algorithm-Intensive Model ...................... non-linear model + linear feature
      +-- Tree ensemble ............................ split points cut the input space
      +-- Neural network ........................... activation function bends the response
      +-- Kernel method ............................ inner product in an implicit feature space
```

<a id="fig-1"></a>
Fig 1. The composition of the two models and the methods on each side

두 model 의 계층은 가정의 강도로 내려갑니다. Feature-Intensive Model 은 비선형의 형태를 항으로 미리 적어 두는 대신 그 항의 계수를 그대로 읽습니다. Algorithm-Intensive Model 은 형태를 적지 않아도 되는 대신, 어느 변수의 어느 구간이 예측을 움직였는지를 계수 하나로 읽지 못합니다.

### 3.1 Placement

Table 1. Where each model sits on the three axes

| Model                     | Linearity | Feature            | What you read out     | Breaks when                        |
| :-----------------------: | :-------: | :----------------: | :-------------------: | :--------------------------------: |
| Feature-Intensive Model   | 선형      | 비선형 (변환된 열) | 항마다의 계수 $\beta$ | 비선형의 형태를 미리 알 수 없을 때 |
| Algorithm-Intensive Model | 비선형    | 선형 (원본 열)     | 변수 중요도           | 외삽 구간과 적은 표본에서          |

Feature-Intensive Model 은 어떤 항을 만들지를 분석가가 정하므로, 자료에 어떤 곡선과 어떤 interaction 이 있는지 짐작할 근거가 있어야 합니다. Algorithm-Intensive Model 은 그 근거 없이도 적합하지만, tree ensemble 은 train data 밖의 값을 외삽하지 못하고 neural network 는 표본이 적으면 과적합합니다.

## 4. Principle

### 4.1 Linear Model Mechanics and Limits

선형 회귀, Ridge, Lasso, PLS (Partial Least Squares) 등의 모델은 입력 변수 $X$ 와 타겟 변수 $Y$ 의 관계를 선형 결합 (Linear Combination) 으로 가정합니다.

```math
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon \hspace{19em} (1)
```

- 장점
  - 모델이 단순하여 과적합 (Overfitting) 위험이 적고, 파라미터 수렴 속도가 매우 빠릅니다.
  - 계수 $\beta_i$ 를 통해 어떤 변수가 타겟 변화에 얼마큼 기여했는지 직관적으로 해석 가능합니다.
- 한계
  - 데이터에 곡선 (Non-linear) 관계가 존재하거나 변수 간 상호작용 (Interaction) 이 있을 경우, 1차 평면 형태의 예측 경계 (Decision Boundary) 로는 이를 적합할 수 없어 언더피팅 (Underfitting) 이 발생합니다.

### 4.2 Nonlinear Features in a Linear Model

선형 모델에서 '선형 (Linearity)' 의 수학적 정의는 입력 변수 $x$ 에 대한 1차식이 아니라, 최적화 대상인 가중치 파라미터 $\beta$ 에 대해 1차식임을 의미합니다.

따라서 입력 공간을 비선형 변환하여 확장하더라도, $\beta$ 에 대한 1차 구조는 유지되므로 선형 모델의 해법 (Closed-form solution) 을 그대로 사용할 수 있습니다.

원래 입력 데이터가 $x_1$, $x_2$ 일 때, 비선형 차수 항을 추가하여 새로운 기저 (Basis) 로 매핑합니다.

```math
\phi(x_1, x_2) = [1, x_1, x_2, x_1^2, x_2^2, x_1 x_2]^T \hspace{19em} (2)
```

이후 이 확장된 공간에서 선형 모델을 적합합니다.

```math
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_2^2 + \beta_5 (x_1 x_2) \hspace{14em} (3)
```

- 변수 $x$ 기준: 비선형 모델 (곡선 및 상호작용 곡면 표현 가능)
- 가중치 $\beta$ 기준: 선형 모델 (최소제곱법, Ridge/Lasso 규제 등 기존 알고리즘 그대로 적용)

선형대수학의 선형성과 공학에서 쓰는 선형성이 갈리는 지점은 [Appendix C](#appendix-c-two-views-of-linearity) 에 있습니다.

식 (3) 의 계수는 확장 전의 계수와 같은 방식으로 읽습니다. $\beta_3$ 은 $x_1$ 의 곡률이고 $\beta_5$ 는 두 변수가 함께 움직일 때의 기여이며, 둘 다 최소제곱법이 정합니다.

## 5. Application

비선형성 처리 주체에 따라 접근법이 갈립니다. Feature-Intensive Model 은 분석가가 직접 `PolynomialFeatures` 등을 활용해 비선형/상호작용 항을 추가한 뒤 선형 모델 (Ridge, PLS 등) 에 학습시킵니다. Algorithm-Intensive Model 은 원본 데이터 ($x_1$, $x_2$) 를 그대로 입력하고, 트리 기반 앙상블 (XGBoost, Random Forest) 이나 신경망 모델 내부에서 분기 (Split) 및 활성화 함수를 통해 비선형 패턴을 자동 학습하도록 합니다.

Feature-Intensive Model 은 $\beta$ 에 대해서만 선형입니다. 절편 $\beta_0$ 를 가지므로 선형대수의 정의로는 affine 변환이고, 공학에서 쓰는 선형은 그 affine 까지 포함합니다 ([Appendix C](#appendix-c-two-views-of-linearity)). Algorithm-Intensive Model 은 $x$ 로도 $\beta$ 로도 선형이 아니므로, 계수 하나로 기여를 읽는 방식 자체가 성립하지 않습니다.

### 5.1 Feature-Intensive Model

- **가정**: 담을 비선형의 형태를 항으로 적을 수 있습니다. Degree 2 이면 한 변수의 제곱과 두 변수의 곱까지입니다.
- **설정값**: `PolynomialFeatures(degree=2, include_bias=False)` 와 regularization 강도 `alpha`. 확장한 열은 규모가 달라지므로 Ridge 나 Lasso 로 계수를 제한합니다.
- **깨지는 조건**: 참된 관계가 적어 둔 항 밖에 있으면 확장 후에도 underfitting 이 남습니다. Degree 를 높여 맞추면 열 수가 급히 늘어 계수가 흔들립니다.
- **만나는 자리**: 공정 변수처럼 물리적 근거로 곡률과 interaction 을 짐작할 수 있고, 계수를 보고해야 하는 자리입니다.

### 5.2 Algorithm-Intensive Model

- **가정**: 표본이 분기 구조를 정할 만큼 많습니다. Tree ensemble 은 구간마다 상수를 적합하므로 구간 안의 표본 수가 정확도를 정합니다.
- **설정값**: Tree ensemble 의 `max_depth` 와 learning rate, neural network 의 층 수와 활성화 함수.
- **깨지는 조건**: Train data 밖의 입력에서 tree ensemble 은 마지막 구간의 상수를 그대로 내놓아 외삽하지 못합니다. 표본이 적으면 neural network 가 과적합합니다.
- **만나는 자리**: 변수 수가 많아 항을 일일이 적기 어렵고, 예측 정확도가 계수 해석보다 앞서는 자리입니다.

두 model 의 정확도를 같은 data 에서 비교한 실행이 [Appendix B](#appendix-b-python-implementation) 에 있습니다.

## 6. Further Work

- **Basis expansion 의 비교**
  - 무엇을 하는가: Spline 과 RBF basis 를 degree 2 확장과 같은 data 에서 비교하여, 국소적인 곡선을 담을 때의 열 수와 정확도를 잰다.
  - 왜 지금인가: `SplineTransformer` 가 scikit-learn 1.0 부터 pipeline 안에서 `PolynomialFeatures` 와 같은 자리에 들어간다.
  - 무엇이 필요한가: 곡선의 국소성이 다른 data set 두 개와, 같은 조건의 validation split.

---

## Appendix A. Terminology

- **basis expansion**: 입력 변수를 미리 정한 함수의 값으로 바꾸어 열을 늘리는 변환. Power term, spline, RBF 가 여기에 속한다.
- **closed-form solution**: 반복 없이 식 하나로 얻는 해. 최소제곱법의 정규방정식이 그 예이다.
- **interaction**: 두 변수가 함께 움직일 때만 나타나는 기여. 곱한 열 $x_1 x_2$ 로 담는다.
- **kernel method**: 입력을 직접 변환하지 않고 두 표본 사이의 내적으로 비선형 관계를 담는 방법.
- **RBF**: Radial basis function. 중심에서의 거리로 값이 정해지는 기저 함수.
- **tree ensemble**: 여러 결정 tree 의 예측을 모아 쓰는 model. Random Forest 와 gradient boosting 이 여기에 속한다.
- **underfitting**: Model 의 표현력이 모자라 training data 에서도 오차가 큰 상태.

## Appendix B. Python Implementation

두 model 을 같은 data 에 적용해 정확도를 비교합니다.

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# --- synthetic data with one curvature term and one interaction term ---
rng = np.random.default_rng(0)
n_samples = 400
x1 = rng.uniform(-3, 3, size=n_samples)
x2 = rng.uniform(-3, 3, size=n_samples)
y = 3 + 2 * x1 - x2 + 1.5 * x1 ** 2 + 0.8 * x1 * x2 + rng.normal(0, 1.0, size=n_samples)

X = np.column_stack([x1, x2])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# the baseline: a linear model on the original columns
plain = Ridge(alpha=1.0).fit(X_train, y_train)

# the feature-intensive model: the analyst adds the power and interaction columns
feature_intensive = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=1.0)
).fit(X_train, y_train)

# the algorithm-intensive model: the algorithm splits the original columns on its own
algorithm_intensive = HistGradientBoostingRegressor(max_depth=3, random_state=0).fit(X_train, y_train)

for name, model in (("linear model, original columns", plain),
                    ("feature-intensive: Ridge on degree-2 columns", feature_intensive),
                    ("algorithm-intensive: HistGradientBoostingRegressor", algorithm_intensive)):
    print(f"{name:50s} R2 = {r2_score(y_test, model.predict(X_test)):.4f}")

# the coefficients the feature-intensive model reads out, one per term of equation (3)
term_names = feature_intensive[0].get_feature_names_out(["x1", "x2"])
print("feature-intensive coefficients:",
      dict(zip(term_names, np.round(feature_intensive[1].coef_, 3))))
```

첫 세 줄이 세 model 의 held-out $R^2$ 이고, 마지막 줄이 Feature-Intensive Model 이 읽어 내는 항마다의 계수입니다.

```text
linear model, original columns                     R2 = 0.4735
feature-intensive: Ridge on degree-2 columns       R2 = 0.9782
algorithm-intensive: HistGradientBoostingRegressor R2 = 0.9689
feature-intensive coefficients: {'x1': np.float64(1.963), 'x2': np.float64(-1.015), 'x1^2': np.float64(1.516), 'x1 x2': np.float64(0.76), 'x2^2': np.float64(0.005)}
```

계수 네 개는 data 를 만든 식의 계수 2, -1, 1.5, 0.8 을 되찾았고, 식에 없던 $x_2^2$ 의 계수는 0.005 로 남았습니다.

## Appendix C. Two Views of Linearity

- 선형대수학의 선형성: 가산성 ($f(x+y)=f(x)+f(y)$) 과 동차성 ($f(cx)=cf(x)$) 을 만족해야 하며, 반드시 원점을 지나야 함 ($y=ax+b$ 에서 $b \neq 0$ 이면 아핀 변환).
- 도메인/공학에서의 선형성: 직선 및 비례 관계를 의미하며, 선형대수의 엄밀한 정의보다 확장된 의미로 사용됨.
