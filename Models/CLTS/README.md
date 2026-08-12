# CLTS (Continuous Learning for Time Series)
Rev. 0 | Created: 2026-08-12 | Updated: 2026-08-12 15:43 CDT

CLTS는 CL for TS, 즉 Continuous Learning for Time Series의 약어이다. 시계열 데이터에 새로운 샘플이 추가될 때 전체 모델을 처음부터 다시 학습시키지 않고, 새로운 데이터만 추가로 학습시켜 예측 성능을 지속적으로 개선하는 기법을 다룬다. 이 기법은 적용 방식과 요구 사항에 따라 재귀적 재학습 (Recursive Retraining), 온라인 학습 (Online Learning), 점진적 학습 (Incremental Learning) 등으로 불린다.

## 1. Terminology and Core Concepts

대표적인 명칭과 핵심 개념은 Table 1과 같다.

Table 1. Common names and core concepts

| Name | English | Core concept |
|------|---------|--------------|
| 재귀적 재학습 / 롤링 재학습 | Recursive Retraining / Rolling Retraining | 시계열의 rolling window나 expanding window 기법을 활용하여, 새로운 데이터가 들어올 때마다 모델을 주기적으로 갱신하는 시계열 특화 방식이다. |
| 온라인 학습 | Online Learning / Streaming Learning | 데이터가 실시간 streaming 형태로 들어올 때, 전체 데이터를 저장하지 않고 새 샘플 단위 (또는 mini-batch) 로 가중치를 즉시 업데이트하는 방식이다. |
| 점진적 학습 | Incremental Learning / Continual Learning | 기존에 학습한 지식을 잊어버리지 않고 (catastrophic forgetting 방지), 새로 들어오는 데이터의 특성을 계속해서 누적 축적하는 학습 방식이다. |

세 명칭은 관점의 차이일 뿐 서로 배타적이지 않으며, 어느 관점에서 부르는지는 6장에서 정리한다.

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

+ ARIMA: Autoregressive Integrated Moving Average. 자기회귀와 이동평균을 결합한 고전적 시계열 예측 모델이다.
+ concept drift: 입력 변수와 목표값 사이의 통계적 관계가 시간에 따라 변하는 현상이다.
+ expanding window: 시작점을 고정하고 끝점만 앞으로 늘려 학습 구간을 확장하는 방식이다.
+ forecast horizon: 예측 시점부터 예측 대상 시점까지의 시간 간격이다.
+ FSNet: Fast and Slow learning Network. 빠른 적응용 보조 구조를 가진 online 시계열 예측 딥러닝 모델이다.
+ OneNet: 복수 예측 모델을 online ensemble로 결합하여 concept drift에 대응하는 시계열 예측 모델이다.
+ recursive least squares: 새 관측값이 들어올 때마다 최소제곱 해를 점진적으로 갱신하는 adaptive filter 알고리즘이다.
+ replay: 과거 샘플 일부를 저장해 두었다가 새 데이터와 함께 다시 학습에 사용하는 forgetting 완화 기법이다.
+ rolling window: 고정 길이의 학습 구간을 시간 축을 따라 밀며 최신 데이터만 유지하는 방식이다.
