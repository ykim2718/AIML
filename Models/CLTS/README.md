# CLTS (Continuous Learning for Time Series)
Rev. 5 | Created: 2026-08-12 | Updated: 2026-08-12 17:58 CDT

CLTS는 CL for TS, 즉 Continuous Learning for Time Series의 약어이다. 시계열 데이터에 새로운 샘플이 추가될 때 전체 모델을 처음부터 다시 학습시키지 않고, 새로운 데이터만 추가로 학습시켜 예측 성능을 지속적으로 개선하는 기법을 다룬다. 이 기법은 적용 방식과 요구 사항에 따라 재귀적 재학습 (Recursive Retraining), 온라인 학습 (Online Learning), 점진적 학습 (Incremental Learning) 등으로 불린다.

## 1. Terminology and Core Concepts

대표적인 명칭과 핵심 개념은 Table 1과 같다.

Table 1. Common names and core concepts

| Name | Core concept |
|------|--------------|
| Recursive Retraining / Rolling Retraining | 시계열의 rolling window나 expanding window 기법을 활용하여, 새로운 데이터가 들어올 때마다 모델을 주기적으로 갱신하는 시계열 특화 방식이다. |
| Online Learning / Streaming Learning | 데이터가 실시간 streaming 형태로 들어올 때, 전체 데이터를 저장하지 않고 새 샘플 단위 (또는 mini-batch) 로 가중치를 즉시 업데이트하는 방식이다. |
| Incremental Learning / Continual Learning | 기존에 학습한 지식을 잊어버리지 않고 (catastrophic forgetting 방지), 새로 들어오는 데이터의 특성을 계속해서 누적 축적하는 학습 방식이다. |

세 명칭은 관점의 차이일 뿐 서로 배타적이지 않으며, 어느 관점에서 부르는지는 6장에서 정리한다.

이들 주변에는 transfer learning, meta-learning, test-time adaptation 같은 인접 개념이 있다. 기법 전체는 모델 갱신 방식 기준과 forgetting 대응 기준의 두 축으로 크게 분류할 수 있으며, 전체 구조는 Fig 1과 같다. 점선 화살표 (`<....`) 는 각 명칭이 주로 속하는 분류를 나타낸다.

Fig 1. CLTS taxonomy of names, adjacent concepts, and classifications

```
CLTS (Continuous Learning for Time Series)
|
+-- Common names
|   +-- Recursive / Rolling Retraining .... periodic refresh on rolling or expanding window
|   +-- Online / Streaming Learning ....... instant update per sample or mini-batch
|   +-- Incremental / Continual Learning .. accumulate knowledge, prevent forgetting
|
+-- Adjacent concepts
|   +-- Transfer Learning / Fine-Tuning
|   +-- Meta-Learning
|   +-- Test-Time Adaptation
|   +-- Concept Drift Detection / Adaptation
|   +-- Adaptive Filtering (Kalman filter, RLS)
|   +-- Data Stream Mining
|   +-- Lifelong Learning
|
+-- Classification by update strategy
|   +-- Periodic retraining (rolling / expanding window) <..... Recursive / Rolling Retraining
|   +-- Online update (SGD, Kalman filter) <................... Online / Streaming Learning
|   +-- Pre-trained + fine-tuning (warm start) <............... Transfer Learning / Fine-Tuning
|
+-- Classification by forgetting mitigation <................... Incremental / Continual Learning
    +-- Replay-based
    +-- Regularization-based (EWC)
    +-- Architecture-based (parameter isolation)
```

## 2. Key Strategies

시계열 예측 성능을 지속적으로 개선하기 위해 실제로 사용하는 주요 기술적 방법은 다음 3가지이다.

### 2.1 Sliding / Rolling Window Retraining

고정된 크기 (예: 최근 30일) 의 window를 유지하면서, 새로운 데이터가 들어오면 가장 오래된 데이터를 밀어내고 최신 데이터로 모델을 재학습시킨다. 데이터의 최신 trend와 계절성 변화 (concept drift) 를 가장 잘 반영한다는 장점이 있다. Window를 늘려 가며 전체 이력을 유지하는 expanding window 방식은 장기 패턴 보존에 유리하지만, 데이터가 커질수록 재학습 비용이 증가한다.

### 2.2 Kalman Filter and State Space Models

새로운 관측값이 들어올 때마다 수학적 상태 방정식을 통해 현재 상태의 확률 분포 (평균, 분산) 를 실시간으로 업데이트하는 전통적이고 강력한 시계열 기법이다. 계산량이 적고 실시간 예측 업데이트에 매우 효율적이라는 장점이 있다. Kalman filter 외에도 recursive least squares 같은 adaptive filter가 streaming 데이터 위에서 선형 모델을 갱신하는 고전적 방법으로 함께 쓰인다.

### 2.3 Fine-Tuning / Warm Start

기존 데이터를 기반으로 사전 학습된 (pre-trained) 딥러닝/머신러닝 모델의 가중치를 파라미터 초기화 없이 새로운 데이터로만 소량 추가 학습 (warm start) 시키는 방식이다. 전체 재학습 대비 계산 비용이 낮고, 기존 모델이 학습한 표현을 재활용할 수 있다는 장점이 있다.

## 3. Challenges

시계열에 대한 continual learning은 일반적인 continual learning과 구별되는 어려움을 가진다.

+ Concept drift: 입력 변수와 목표값 사이의 관계가 시간에 따라 변한다. 정적 과거 데이터로 학습한 offline 모델은 빠르게 낡은 모델이 된다.
+ Catastrophic forgetting: 새 데이터에만 맞춰 갱신하면 과거에 학습한 패턴 (예: 재발하는 계절성) 을 잊어버린다. 많은 online continual learning 방법이 이를 완화하기 위해 과거 샘플을 다시 학습에 섞는 replay를 사용한다.
+ Delayed feedback: 예측 시점에는 실제 미래 값을 알 수 없고, forecast horizon이 지나야 정답을 얻는다. 이 지연 동안 concept drift가 진행되면 모델이 이미 낡은 개념에 적응하는 문제가 생긴다.

## 4. Deep Learning Approaches

딥러닝 기반 online 시계열 예측에서는 concept drift와 forgetting을 동시에 다루기 위한 전용 architecture가 제안되었다. FSNet은 빠른 적응을 위한 보조 구조를 붙인 online 예측 모델이고, OneNet은 복수 모델의 online ensemble로 drift에 대응한다. 시계열 회귀와 예측에 대한 continual learning 연구는 분류 대비 아직 초기 단계이며, 최근에야 첫 survey가 정리되었다.

## 5. Tools and Libraries

Table 2. Python tools for continual time series learning

| Tool | Approach | Note |
|------|----------|------|
| River | Online Learning | Creme와 scikit-multiflow가 병합된 streaming 학습 라이브러리로, 샘플 단위 회귀·분류·이상 탐지와 progressive validation을 지원한다. |
| scikit-learn | Incremental Learning / Warm Start | `partial_fit` 을 제공하는 estimator는 mini-batch 단위 갱신을 지원하고, `warm_start=True` 는 이전 학습 결과에서 이어서 학습한다. |
| statsmodels | State Space Models | Kalman filter 기반 state space 모델로 새 관측값에 대한 상태 갱신을 지원한다. |
| pySmooth | Kalman Filter / Online ARIMA | 이산·확장·unscented Kalman filter와 online ARIMA를 제공한다. |
| LightGBM | Continued Training | `init_model` 에 기존 booster를 전달하면 새 데이터로 tree를 추가하며 이어서 학습한다. |

scikit-learn·LightGBM·statsmodels·River의 구현 예시는 [Appendix B](#appendix-b-python-examples) 에 있다.

## 6. Summary

새 샘플이 추가될 때 전체 재학습 없이 모델을 갱신하는 기법의 명칭은 관점에 따라 다르다. 일반적인 시계열 머신러닝 pipeline 관점의 명칭은 재귀적 재학습 (Recursive Retraining) 또는 롤링 윈도우 재학습 (Rolling Window Retraining) 이고, 데이터가 실시간으로 들어오는 시스템 관점의 명칭은 온라인 학습 (Online Learning) 또는 점진적 학습 (Incremental Learning) 이다. 구현 전략은 rolling window 재학습, Kalman filter 기반 상태 갱신, warm start 기반 fine-tuning의 3가지가 대표적이며, concept drift·catastrophic forgetting·delayed feedback이 공통 난제이다.

## 7. References

+ [Continual Learning for Time Series Forecasting: A First Survey](https://univ-evry.hal.science/INSA-CVL/hal-04836655v1)
+ [Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting](https://arxiv.org/pdf/2412.08435)
+ [Continuous Evolution Pool: Taming Recurring Concept Drift in Online Time Series Forecasting](https://arxiv.org/html/2506.14790)
+ [Online Continual Learning for Time Series: a Natural Score-driven Approach](https://arxiv.org/html/2601.12931)
+ [pySmooth: Kalman filters and online ARIMA in Python](https://github.com/kenluck2001/pySmooth)
+ [Kalman Filter for Time Series Forecasting in Python](https://forecastegy.com/posts/kalman-filter-for-time-series-forecasting-in-python/)

## Appendix A. Terminology

+ adaptive filter: 새 관측값이 들어올 때마다 계수를 실시간으로 갱신하는 filter이다.
+ ARIMA: Autoregressive Integrated Moving Average. 자기회귀와 이동평균을 결합한 고전적 시계열 예측 모델이다.
+ Avalanche: PyTorch 기반의 continual learning 라이브러리로, replay·regularization·architecture 계열 기법의 구현을 제공한다.
+ booster: gradient boosting 모델에서 학습된 tree들의 집합을 담는 객체이다.
+ concept drift: 입력 변수와 목표값 사이의 통계적 관계가 시간에 따라 변하는 현상이다.
+ data stream mining: 끝없이 이어지는 데이터 stream에서 실시간으로 패턴을 추출하는 분야이다.
+ EWC: Elastic Weight Consolidation. 이전 과제에 중요한 가중치의 변화에 벌점을 주어 forgetting을 줄이는 regularization 기법이다.
+ expanding window: 시작점을 고정하고 끝점만 앞으로 늘려 학습 구간을 확장하는 방식이다.
+ forecast horizon: 예측 시점부터 예측 대상 시점까지의 시간 간격이다.
+ FSNet: Fast and Slow learning Network. 빠른 적응용 보조 구조를 가진 online 시계열 예측 딥러닝 모델이다.
+ gradient boosting: 이전 모델의 오차를 보정하는 tree를 순차적으로 추가하는 ensemble 학습 기법이다.
+ lifelong learning: 하나의 모델이 이어지는 여러 과제를 계속 학습하는 패러다임으로, continual learning과 거의 같은 뜻으로 쓰인다.
+ LightGBM: gradient boosting 기반의 오픈소스 머신러닝 framework이다.
+ local level model: 관측값을 서서히 변하는 수준 성분과 관측 노이즈로 분해하는 가장 단순한 state space 모델이다.
+ MAE: Mean Absolute Error. 예측 오차 절대값의 평균이다.
+ meta-learning: 새로운 과제에 빠르게 적응하는 방법 자체를 학습하는 기법이다.
+ OneNet: 복수 예측 모델을 online ensemble로 결합하여 concept drift에 대응하는 시계열 예측 모델이다.
+ parameter isolation: 과제별로 서로 다른 파라미터 부분집합을 할당하여 과제 간 간섭을 막는 continual learning 기법이다.
+ progressive validation: 각 샘플에 대해 먼저 예측하고 그 다음 학습하여, 별도의 평가 데이터 없이 online 모델을 평가하는 방식이다.
+ PyTorch: Meta가 주도하는 오픈소스 딥러닝 framework이다.
+ recursive least squares (RLS): 새 관측값이 들어올 때마다 최소제곱 해를 점진적으로 갱신하는 adaptive filter 알고리즘이다.
+ regularization: 모델의 복잡도나 파라미터 변화에 벌점을 주어 과적합과 forgetting을 억제하는 기법이다.
+ replay: 과거 샘플 일부를 저장해 두었다가 새 데이터와 함께 다시 학습에 사용하는 forgetting 완화 기법이다.
+ rolling window: 고정 길이의 학습 구간을 시간 축을 따라 밀며 최신 데이터만 유지하는 방식이다.
+ SGD: Stochastic Gradient Descent. 샘플 (또는 mini-batch) 단위의 gradient로 파라미터를 갱신하는 최적화 알고리즘이다.
+ TensorFlow: Google이 주도하는 오픈소스 딥러닝 framework이다.
+ test-time adaptation: 배포된 모델이 예측 시점의 입력 분포 변화에 맞춰 스스로를 조정하는 기법이다.
+ transfer learning: 한 과제에서 학습한 지식을 다른 과제의 학습에 재사용하는 기법이다.

## Appendix B. Python Examples

아래 예시는 모두 난수 데이터를 사용한 최소 실행 예시이며, 실제 적용 시 데이터 준비와 hyperparameter만 바꾸면 된다.

#### Incremental learning with scikit-learn partial_fit

`partial_fit` 을 제공하는 estimator (SGDRegressor, SGDClassifier, MLPRegressor 등) 는 새 샘플이 도착할 때마다 파라미터 초기화 없이 모델을 갱신한다.

```python
import numpy as np
from sklearn.linear_model import SGDRegressor

rng = np.random.default_rng(0)
model = SGDRegressor(learning_rate="constant", eta0=0.01, random_state=0)

# initial fit on the first batch
X_init, y_init = rng.random((100, 3)), rng.random(100)
model.partial_fit(X_init, y_init)

# update the model one sample at a time as new data arrives
X_stream, y_stream = rng.random((50, 3)), rng.random(50)
for i in range(len(X_stream)):
    model.partial_fit(X_stream[i:i + 1], y_stream[i:i + 1])
```

#### Warm start with scikit-learn GradientBoostingRegressor

`warm_start=True` 는 이미 학습된 tree를 유지한 채, 늘어난 `n_estimators` 만큼의 tree만 추가로 학습한다.

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

rng = np.random.default_rng(0)
X_old, y_old = rng.random((200, 3)), rng.random(200)
X_new, y_new = rng.random((30, 3)), rng.random(30)

model = GradientBoostingRegressor(n_estimators=100, warm_start=True, random_state=0)
model.fit(X_old, y_old)

# grow 20 more trees on the combined data; existing trees are kept
model.n_estimators += 20
model.fit(np.vstack([X_old, X_new]), np.concatenate([y_old, y_new]))
```

#### Continued training with LightGBM init_model

`lgb.train` 의 `init_model` 에 기존 booster를 전달하면, 기존 tree를 그대로 둔 채 새 데이터에 대한 tree를 이어서 학습한다.

```python
import lightgbm as lgb
import numpy as np

rng = np.random.default_rng(0)
X_old, y_old = rng.random((200, 3)), rng.random(200)
X_new, y_new = rng.random((30, 3)), rng.random(30)
params = {
  "objective": "regression",
  "verbosity": -1,
}

booster = lgb.train(params, lgb.Dataset(X_old, y_old), num_boost_round=100)

# continue training from the existing booster with new data only
booster = lgb.train(
    params,
    lgb.Dataset(X_new, y_new),
    num_boost_round=20,
    init_model=booster,
)
```

#### Kalman filter update with statsmodels

statsmodels의 UnobservedComponents로 local level model을 적합한 뒤, `append` 로 새 관측값을 Kalman filter 상태에 반영한다. 파라미터를 다시 추정하지 않으므로 갱신 비용이 매우 낮다.

```python
import numpy as np
import statsmodels.api as sm

rng = np.random.default_rng(0)
y_old = rng.normal(0, 0.5, 100).cumsum()
y_new = y_old[-1] + rng.normal(0, 0.5, 10).cumsum()

model = sm.tsa.UnobservedComponents(y_old, level="local level")
res = model.fit(disp=False)

# update the Kalman filter state with new observations, keeping the fitted parameters
res = res.append(y_new)
forecast = res.forecast(5)
```

#### Online learning with River

River는 샘플을 dict 형태로 받아 `predict_one` 과 `learn_one` 으로 한 건씩 처리한다. 각 샘플에 대해 예측을 먼저 하고 같은 샘플로 학습하면, 별도의 평가 데이터 없이 online 성능을 측정하는 progressive validation이 된다.

```python
import numpy as np
from river import linear_model, metrics, preprocessing

rng = np.random.default_rng(0)
model = preprocessing.StandardScaler() | linear_model.LinearRegression()
metric = metrics.MAE()

for _ in range(200):
    x = {"x1": rng.random(), "x2": rng.random()}
    y = 2 * x["x1"] - x["x2"]

    # predict first, then learn from the same sample (progressive validation)
    y_pred = model.predict_one(x)
    metric.update(y, y_pred)
    model.learn_one(x, y)
```

#### Rolling window retraining

새 시점마다 최근 `WINDOW` 개의 샘플로만 모델을 다시 학습시켜 concept drift에 대응한다. 재학습 주기를 매 시점 대신 매주·매월로 늘리면 계산 비용을 줄일 수 있다.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(0)
X, y = rng.random((120, 3)), rng.random(120)

WINDOW = 30
preds = []
for t in range(WINDOW, len(X)):
    # keep only the latest WINDOW samples and retrain
    model = LinearRegression().fit(X[t - WINDOW:t], y[t - WINDOW:t])
    preds.append(model.predict(X[t:t + 1])[0])
```

## Appendix C. Taxonomy with Python Libraries

Fig 1의 두 분류 축을 각 분류별 대표 Python library와 연결하면 Fig 2와 같다. 이 중 scikit-learn·LightGBM·statsmodels·River의 실행 예시는 [Appendix B](#appendix-b-python-examples) 에 있다.

Fig 2. Classifications of Fig 1 extended with representative Python libraries

```
Classification by update strategy
|
+-- Periodic retraining (rolling / expanding window)
|   +-- scikit-learn ........... refit any estimator on each window
|   +-- statsmodels ............ refit ARIMA / state space models per window
|   +-- LightGBM ............... periodic retraining of boosting models
|
+-- Online update (per sample or mini-batch)
|   +-- River .................. predict_one / learn_one streaming pipeline
|   +-- scikit-learn ........... partial_fit (SGDRegressor, MLPRegressor)
|   +-- statsmodels ............ Kalman filter state update via append
|   +-- pySmooth ............... online ARIMA, Kalman filter variants
|
+-- Pre-trained + fine-tuning (warm start)
    +-- scikit-learn ........... warm_start=True (GradientBoostingRegressor)
    +-- LightGBM ............... continued training via init_model
    +-- PyTorch / TensorFlow ... load pre-trained weights and fine-tune

Classification by forgetting mitigation
|
+-- Replay-based ............... Avalanche (replay plugin), custom replay buffer
+-- Regularization-based ....... Avalanche (EWC plugin)
+-- Architecture-based ......... Avalanche (parameter isolation strategies)
```
