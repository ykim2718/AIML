# Time-Series ML Model Validation Checklist
Rev. 7 | Created: 2026-09-20 | Updated: 2026-09-23 11:08 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Placement](#31-placement)
- [4. General ML Check](#4-general-ml-check)
- [5. Time-Series Specific Check](#5-time-series-specific-check)
  - [5.1 Validation Strategy](#51-validation-strategy)
  - [5.2 Preprocessing and Feature Engineering Leakage](#52-preprocessing-and-feature-engineering-leakage)
  - [5.3 Data Properties](#53-data-properties)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

- **Problem Statement**: 시계열 data 를 random split 과 전체 segment preprocessing 으로 validation 하면 예측 시점 이후의 정보가 training 에 섞여, validation 성능이 production 성능보다 높게 나온다.
- **Goal**: 시계열 model 을 deploy 하기 전의 check 를 general ML level 과 time-series level 로 나누고, 각 check 가 modeling pipeline 의 어느 stage 에서 무엇을 확인하는지 정한다.
- **Non-Goal**: model family 별 algorithm 선택과 hyperparameter 조정은 다루지 않는다. 어긋난 check 를 고치는 retraining 절차도 다루지 않는다.

## 2. Summary

두 level 은 general ML check 와 time-series specific check 이다. 이 두 level 에서 check 가 나뉘어, general ML check 는 시간 순서와 무관하게 모든 model 이 받고, time-series specific check 는 order, trend, seasonality 가 있는 data 에만 더 붙는다.

Time-series specific check 의 세 axis 는 모두 한 가지를 막는다. 예측 시점 $t$ 이후의 정보가 training 과 preprocessing 에 닿는 것이다.

- **Data split**: random K-Fold 를 쓰지 않고, 과거로 training 하고 미래를 예측하는 Time-Series Split 을 쓴다.
- **Preprocessing**: scaler 와 imputation 은 train segment 의 statistics 만으로 fit 한다.
- **Data properties**: stationarity, concept drift, look-ahead bias 를 확인한다.

## 3. Taxonomy and its Hierarchy

Check 는 두 level 로 갈리고, time-series level 은 다시 세 axis 로 갈린다. General ML level 은 data 의 종류를 가리지 않아 어느 model 에서나 같은 check 이고, time-series level 은 data 에 시간 순서가 있다는 조건에서만 성립한다. [Fig 1](#fig-1) 이 그 두 level 과 세 axis 이다.

```text
Time-Series ML Model Validation
|
+-- General ML Check .............. order-independent, applies to every model
|     +-- Overfitting / Underfitting .. train and validation loss convergence
|     +-- Metric Selection ............ class imbalance, outlier sensitivity
|     +-- Feature Leakage ............. target or unknown value inside a feature
|     +-- Baseline Comparison ......... naive forecast, linear model
|     +-- Reproducibility & Latency ... fixed random seed, inference time budget
|
+-- Time-Series Specific Check .... order-dependent, adds trend and seasonality
      |
      +-- Validation Strategy ......... how the rows are split along time
      |     +-- Time-Series Split: walk-forward, expanding window
      |     +-- Temporal separation of train, validation, and test
      |
      +-- Preprocessing Leakage ....... what each transform is fitted on
      |     +-- Scaling and imputation fitted on the train segment only
      |     +-- Lag and rolling-window features shifted by one step
      |
      +-- Data Properties ............. what the series itself does over time
            +-- Stationarity and differencing
            +-- Concept drift and covariate shift
            +-- Look-ahead bias from data availability time
```

<a id="fig-1"></a>
Fig 1. Two levels of the validation checklist and the three axes of the time-series level

### 3.1 Placement

각 check 는 modeling pipeline 의 한 stage 에서 확인한다. Table 1 이 check 와 그 stage 이며, stage 의 순서는 data split 에서 deployment 까지의 순서다.

Table 1. Placement of each check in the modeling pipeline

| Check                         | Level       | Stage               |
| :---------------------------: | :---------: | :-----------------: |
| Time-Series Split             | Time-series | Data split          |
| Temporal separation of sets   | Time-series | Data split          |
| Scaling and imputation        | Time-series | Preprocessing       |
| Stationarity and differencing | Time-series | Preprocessing       |
| Lag and rolling window        | Time-series | Feature engineering |
| Feature leakage               | General     | Feature engineering |
| Overfitting / underfitting    | General     | Training            |
| Metric selection              | General     | Evaluation          |
| Baseline comparison           | General     | Evaluation          |
| Concept drift                 | Time-series | Evaluation          |
| Look-ahead bias               | Time-series | Deployment          |
| Reproducibility and latency   | General     | Deployment          |

Data split stage 의 두 check 가 Table 1 의 맨 위에 있다. Data split 이 어긋난 채로 얻은 점수는 뒤 stage 의 check 를 모두 통과해도 production 에서 재현되지 않는다.

Fig 2 는 Table 1 의 stage 를 순서대로 놓고 각 stage 의 check 를 그 아래에 붙인 것이다.

<img src="time-series-ml-model-validation-checklist_fig/pipeline-stages.png" width="900" style="max-width: 100%;" alt="Fig 2">

<a id="fig-2"></a>
Fig 2. Placement of the twelve checks on the six pipeline stages

Marker 의 색이 Level 을 가른다. 파란색 일곱 check 가 time-series level 이고, 주황색 다섯 check 가 general level 이다.

## 4. General ML Check

시계열 여부와 상관없이 모든 machine learning model 에서 공통으로 확인하는 check 이다.

- **Overfitting / Underfitting**
  - Train loss 와 validation loss 의 convergence.
  - Parameter 수 대비 data 수. 부족하면 overfitting.
- **Metric Selection**
  - Classification: class imbalance 에서 accuracy 대신 F1-score, PR-AUC.
  - Regression: outlier sensitivity 에 따라 RMSE, MAE, MAPE, Huber loss 중 선택.
- **Feature Leakage**
  - 예측 시점에 알 수 없는 미래 정보가 들어간 feature.
  - Target 값 자체가 들어간 feature.
- **Baseline Comparison**
  - Naive forecast 대비 성능 향상. 지난달 값을 그대로 쓰는 예측, moving average.
  - Linear model 대비 성능 향상.
- **Reproducibility & Latency**
  - Random seed 고정 여부.
  - Batch inference 와 real-time inference 의 latency 만족 여부.

## 5. Time-Series Specific Check

시간 순서와 seasonality, trend 가 있는 data 에서 더 확인하는 check 이다.

### 5.1 Validation Strategy

Data split 은 시간을 따라야 한다. Random K-Fold 는 미래 data 로 training 하고 과거 data 를 예측하게 하므로 쓰지 않는다.

- **No Random K-Fold**
  - 과거 data 로 training 하고 미래 data 를 예측하는 Time-Series Split. Walk-forward validation, expanding window.
- **Temporal Separation of Sets**
  - Train $\rightarrow$ validation $\rightarrow$ test 의 시간 순서.
  - 세 segment 사이의 gap 과 look-ahead 혼입 여부.

### 5.2 Preprocessing and Feature Engineering Leakage

Preprocessing 과 feature engineering 은 train segment 안에서만 계산한다. 전체 segment 의 statistics 와 미래 시점의 값이 preprocessing 과 feature engineering 으로 들어온다.

- **Scaling & Imputation**
  - Scaler (MinMax, Standard) 의 fit segment. 전체 data 가 아닌 train segment 의 mean, standard deviation.
  - Imputation 의 fit segment. Train segment 에서 얻은 값을 validation 과 test 에 적용.
- **Lag Feature & Rolling Window**
  - $t$ 시점 예측에 $t+1$, $t+2$ 등 미래 시점 feature 참조 여부.
  - Rolling window 계산의 `shift(1)`. 현재 시점 이전의 data 만 사용.

### 5.3 Data Properties

Data 자체가 시간에 따라 무엇을 하는지 확인한다. Mean 과 variance 의 이동, 분포의 구조적 변화, data availability time 이 data properties 에 속한다. 세 check 의 용어 정의는 [Appendix A](#appendix-a-terminology) 에 있다.

- **Stationarity & Differencing**
  - Trend 와 seasonality 로 인한 mean 과 variance 의 시간 변화.
  - 필요 시 differencing 또는 log 변환 적용 여부.
  - Check 결과가 정하는 것: differencing order $d$ 와 seasonal differencing order $D$, model family, validation 점수의 해석.
  - Linear model (ARMA, VAR, linear regression): stationarity 를 전제. 서로 관계가 없는 non-stationary series 두 개를 그대로 회귀하면 spurious regression 으로 $R^2$ 와 t 통계량이 부풀려지므로, differencing 이나 cointegration 확인 뒤 fit.
  - Non-linear model (gradient boosting, neural network): stationarity 를 요구하지 않음. Train segment 밖의 값을 외삽하지 못하므로, trend 가 남으면 differencing 이나 detrending 으로 target 범위를 맞춤.
- **Concept Drift & Covariate Shift**
  - 과거 수집 기간과 예측 대상 기간 사이의 구조적 변화. 시장 상황, 규제, exogenous variable.
  - 예시: 코로나19 이전과 이후의 data 분포 변화.
- **Look-ahead Bias**
  - Feature data 의 data availability time 과 model 이 호출되는 시점의 차이.
  - Production 환경에서 그 차이의 반영 여부.

---

## Appendix A. Terminology

- **cointegration**: 두 non-stationary series 의 선형 결합이 stationary 가 되는 관계.
- **concept drift**: 입력과 target 의 관계가 시간에 따라 바뀌는 현상.
- **covariate shift**: Target 과의 관계는 그대로인 채 입력 분포만 바뀌는 현상.
- **data availability time**: 어떤 값이 실제로 조회 가능해지는 시점. 그 값이 가리키는 시점보다 늦다.
- **detrending**: Trend 성분을 추정해 빼는 변환.
- **expanding window**: Train segment 의 시작을 고정하고 끝만 뒤로 미는 split. Fold 가 진행될수록 train data 가 늘어난다.
- **look-ahead bias**: 예측 시점에 아직 조회할 수 없는 값을 feature 로 써서 성능이 높게 나오는 bias.
- **PR-AUC**: Precision-recall curve 아래 면적. 양성 class 가 드문 data 에서 accuracy 대신 쓴다.
- **spurious regression**: 서로 관계가 없는 non-stationary series 끼리의 회귀에서 $R^2$ 와 t 통계량이 높게 나오는 현상.
- **stationarity**: Mean 과 variance 가 시간에 따라 변하지 않는 성질.
- **walk-forward validation**: Train segment 와 예측 segment 를 시간 축을 따라 한 칸씩 밀며 반복하는 validation.
