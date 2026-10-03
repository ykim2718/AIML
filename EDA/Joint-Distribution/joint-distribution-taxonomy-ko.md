# Taxonomy of Modeling Elements by P(X), P(Y) and P(Y|X)
Rev. 27 | Created: 2026-05-29 | Updated: 2026-10-03 07:25 CDT

## 1. Purpose

- **Problem Statement**: AI/ML model 개발을 위한 요소를 묶는 체계가 없다.
- **Goal**: P(X), P(Y), P(Y|X) 를 결합분포 P(X,Y) 로 묶어 반도체 공정 AI/ML model 의 taxonomy 와 physical meaning을 밝힌다.
- **Non-Goal**: 결합분포 밖에 있는 Model (추정기·최적화) 축과 특정 과제의 진단 수치는 다루지 않는다.

## 2. Summary

> ### 좋은 예측은 좋은 데이타와 좋은 모델에서 나옵니다.

인용구의 좋은 데이터와 좋은 모델은 예측한 출력분포를 정하는 두 factor 이며, 식 (1) 과 그 적분형 식 (2) 가 그 관계를 적는다.

```math
P(Y)_{\mathrm{pred}} = P(X_{o})_{\mathrm{true}} \cdot P(Y \mid X_{i})_{\mathrm{pred}} \hspace{19em} (1)
```

```math
P(Y)_{\mathrm{pred}} = \int P(Y \mid X_{i})_{\mathrm{pred}} \cdot P(X_{o})_{\mathrm{true}}\, dX_{o} \hspace{19em} (2)
```

식 (1) 은 출력 주변분포 (marginal distribution) 식 (4) 를 예측에 옮겨 곱의 형태로 줄여 적은 것이고, 식 (2) 는 적분까지 적은 형태다. X<sub>i</sub> 는 in-sample, 곧 model 을 학습할 때 쓴 학습 데이터이고, X<sub>o</sub> 는 out-of-sample, 곧 추론 때 새로 들어오는 추론 데이터다. P(Y|X<sub>i</sub>)<sub>pred</sub> 는 X<sub>i</sub> 로 학습한 관계를 추론 데이터 X<sub>o</sub> 의 값에서 읽은 것이며, 식 (2) 는 그 값을 X<sub>o</sub> 에 대해 적분한다.

식 (1) 이 적분 없이 곱만으로 성립하려면 아래 둘 가운데 하나를 가정한다. 어느 가정에서든 식 (1) 이 실제 P(Y) 를 맞히려면 X<sub>o</sub> 가 X<sub>i</sub> 의 범위 안에 있고 실제 P(Y|X) 가 학습 뒤에 바뀌지 않아야 한다.

- **한 값에 모인 추론 데이터**: P(X<sub>o</sub>)<sub>true</sub> 가 한 값 x<sub>o</sub> 에서만 확률 1 이고 다른 모든 값에서 확률 0 인 분포 (point mass) 이면, 식 (2) 의 적분이 그 한 점의 값이 되어 P(Y)<sub>pred</sub> = P(Y|X=x<sub>o</sub>)<sub>pred</sub> 이다.
- **추론 데이터 한 건에 대한 해석**: 추론 데이터 한 건 x<sub>o</sub> 에 대해서는 곱이 그대로 성립하며, 이때 좌변은 주변분포 P(Y) 대신 결합확률 P(Y, X<sub>o</sub>=x<sub>o</sub>) 이다. 모든 x<sub>o</sub> 에 대해 더해야 P(Y)<sub>pred</sub> 가 된다.

인용구의 세 요소는 식 (1) 의 세 항에 아래와 같이 대응한다.

- **좋은 예측**: P(Y)<sub>pred</sub> 는 두 factor 에서 유도되는 값이므로, 두 factor 가운데 하나만 어긋나도 실제 P(Y) 에서 벗어난다.
- **좋은 데이터**: X<sub>i</sub> 는 wafer 계측값과 짝지어 model 학습에 쓰는, 장비 sensor 등에서 측정한 학습 데이터이고, P(X<sub>o</sub>)<sub>true</sub> 는 추론 때 실제로 측정되는 추론 데이터의 분포다. P(X<sub>i</sub>) 가 P(X<sub>o</sub>) 를 덮어야 하며, P(X<sub>o</sub>) 가 P(X<sub>i</sub>) 와 달라지면 covariate shift 다.
- **좋은 모델**: P(Y|X<sub>i</sub>)<sub>pred</sub> 는 model 이 학습 데이터 X<sub>i</sub> 로 추정한 조건부 관계다. 실제 P(Y|X) 에 가까워야 하며, 실제 관계가 학습 뒤에 바뀌면 concept drift 다.

세 항목을 결합분포의 taxonomy 로 정리하면 아래와 같다.

- **Taxonomy**: 반도체 공정 AI/ML model 의 요소는 결합분포 P(X,Y) 의 세 항목, 곧 측정 데이터 주변분포 P(X), 출력 주변분포 P(Y), 조건부 관계 P(Y|X) 로 분류되며, 식 (1) 에서 각각 좋은 데이터, 좋은 예측, 좋은 모델의 자리에 놓인다.
- **Hierarchy**: P(X,Y) = P(Y|X) · P(X) 로 분해되는 두 factor 가 P(X) 와 P(Y|X) 이고, P(Y) 는 두 factor 의 곱을 X 에 대해 적분한 주변분포다 (식 (4)). 반대 방향 분해 P(X|Y) · P(Y) 는 prior shift 를 정의한다 (식 (5)).
- **Change**: 세 항목의 변화는 각각 covariate shift, prior shift, concept drift 이다.
- **Physical meaning**: P(X) 는 장비 sensor 등에서 측정한 데이터 (학습 데이터 X<sub>i</sub>, 추론 데이터 X<sub>o</sub>) 의 분포, P(Y) 는 wafer 계측값의 분포, P(Y|X) 는 측정 데이터가 주어졌을 때 계측값의 조건부 분포로, 공정 물리가 측정 데이터와 계측값 사이에 남긴 관계다.
- **Reading rule**: 장비 교체·recipe 변경은 측정 데이터의 분포와 조건부 관계를 함께 이동시킬 수 있으므로, 세 항목을 배타적 분류 대신 **관측·개입 지점** 으로 읽는다.
- **Premises**: P(Y|X) 안의 불확실성 출처, 학습 데이터의 표본 선택, i.i.d. 가정과 lot·chamber 계층 구조는 세 항목 모두에 걸리는 전제다 (section 3).

## 3. Taxonomy and its Hierarchy

결합분포는 관계 P(Y|X) 와 데이터 분포 P(X) 의 곱으로 분해되고, 출력 P(Y) 는 그 곱을 X 에 대해 적분하여 얻는다 ([Fig 1](#fig-1)).

```text
                 P(X, Y)   joint distribution
                    |
      factorize     |   P(X, Y) = P(Y|X) * P(X)
          +---------+---------+
          |                   |
        P(X)               P(Y|X)
   data marginal       conditional relation
   good data           good model
   covariate shift     concept drift
          |                   |
          +---------+---------+
      marginalize   |   integrate over X
                    |
                  P(Y)
            output marginal
            good prediction
            prior shift

  Model (estimator): orthogonal axis, how P(Y|X) is estimated
  reverse factorization: P(X, Y) = P(X|Y) * P(Y), defines prior shift
  premises: uncertainty sources, sample selection, i.i.d. and lot / chamber hierarchy
```

<a id="fig-1"></a>
Fig 1. Taxonomy and hierarchy of the joint distribution

- 위층의 P(X) 와 P(Y|X) 는 결합분포를 이루는 두 factor 다.
- 아래층의 P(Y) 는 두 factor 에서 유도되는 주변분포이므로, P(X) 나 P(Y|X) 가 바뀌면 P(Y) 도 따라 바뀔 수 있다.
- Model 축은 P(Y|X)<sub>pred</sub> 를 얻는 추정 방법 (algorithm·최적화) 이다. 세 항목은 무엇을 추정하는지를, Model 축은 어떻게 추정하는지를 정하므로 둘은 직교 (orthogonal) 한다.

```math
P(X, Y) = P(Y \mid X) \cdot P(X) \hspace{19em} (3)
```

```math
P(Y) = \int P(Y \mid X)\, P(X)\, dX \hspace{19em} (4)
```

결합분포는 반대 방향으로도 분해된다.

```math
P(X, Y) = P(X \mid Y) \cdot P(Y) \hspace{19em} (5)
```

Prior shift 는 식 (5) 에서 P(X|Y) 가 그대로인 채 P(Y) 만 바뀌는 경우로 정의된다. 식 (3) 의 분해만으로는 prior shift 를 P(X) 와 P(Y|X) 의 변화와 구별할 수 없으므로, 두 분해를 함께 둔다.

세 항목은 아래 세 전제 위에서 성립한다.

- **불확실성의 출처**: 계측값은 Y<sub>obs</sub> = Y + ε<sub>m</sub> 이어서, model 이 학습하는 P(Y<sub>obs</sub>|X) 의 산포에는 공정 고유 산포와 계측 오차 ε<sub>m</sub> (label noise) 가 함께 들어 있다. 둘은 데이터를 늘려도 줄지 않는 aleatoric uncertainty 이고, 학습 데이터가 부족해 생기는 model 의 불확실성 (epistemic uncertainty) 만 데이터로 줄어든다. 좋은 모델이 도달할 수 있는 정확도의 한계는 공정 고유 산포와 계측 오차가 정한다.
- **표본 선택**: 계측은 일부 wafer 만 sampling 하므로, 학습 데이터의 분포 P(X<sub>i</sub>) 는 계측된 wafer 의 분포이며 전체 wafer 의 P(X) 와 다를 수 있다 (selection bias). 추론은 계측하지 않은 wafer 에 하므로, 이 차이는 그대로 covariate shift 가 된다. X 나 Y 의 값에 따라 계측이 누락 (missing) 되면 P(X<sub>i</sub>) 나 학습 데이터의 P(Y) 가 같은 방식으로 치우친다.
- **i.i.d. 가정과 계층 구조**: 식 (1)·(2) 로 학습 데이터에서 추론 데이터의 예측을 얻는 것은 wafer 가 서로 독립이고 같은 분포에서 나온다 (i.i.d.) 는 가정 위에서다. 공정 데이터에는 시간 자기상관과 lot·chamber 의 계층 구조가 있어 같은 lot·chamber 의 wafer 가 서로 닮으며, 이 구조를 무시하고 학습·검증을 나누면 같은 lot 이 양쪽에 들어가 성능이 실제보다 높게 나온다.

### 3.1 Placement

Table 1 은 세 항목을 분해, 변화, 개입, 질문, 관측, 대책의 여섯 축에 놓는다.

Table 1. Six lenses on P(X), P(Y) and P(Y|X)

| Lens              | P(X)                                                                | P(Y)                                         | P(Y\|X)                                               |
| :---------------: | :-----------------------------------------------------------------: | :------------------------------------------: | :---------------------------------------------------: |
| 확률 분해         | 측정 데이터 주변분포                                                | 출력 주변분포                                | 조건부 (관계)                                         |
| 분포 변화 (shift) | Covariate shift                                                     | Prior / label shift                          | Concept drift                                         |
| 개입 지점         | 데이터 공간                                                         | 출력공간 (target 구조화)                     | 관계 학습                                             |
| 질문 형태         | 추론 데이터 X<sub>o</sub> 가 학습 데이터 X<sub>i</sub> 와 달라졌나? | 계측값 분포가 달라졌나?                      | 학습 뒤 측정 데이터와 계측값의 관계가 달라졌나?       |
| 관측 (탐지)       | PSI·KS·KL, domain classifier                                        | 계측값 분포 비교                             | Binning CDT, 시간창별 I(X;Y), 잔차 CUSUM·Page-Hinkley |
| 대책 (lever)      | Feature 선택·증강, importance weighting, domain adaptation          | Target 변환·분해, group 별 scale, prior 보정 | 재학습 간격, 최신성 가중, detrending, drift 적응      |

관측 행은 변화를 재는 방법이고, 대책 행은 변화를 가정하고 model 을 맞추는 방법이다. 변동 시점을 특정하는 것은 관측 행의 방법뿐이다. 4.1 ~ 4.3 의 세 변화를 탐지하는 방법은 [Appendix B](#appendix-b-detection-methods) 에 모은다.

## 4. Physical Meaning

세 항목은 반도체 공정에서 각각 측정 데이터, 계측 결과, 측정 데이터와 계측값 사이의 조건부 관계에 대응하고. Table 2 가 그 대응과 식 (1) 에서의 역할을 모은다.

Table 2. Physical meaning of each term

| Term    | Role in eq. (1)                                                | Physical meaning                                         | Shift between training and inference                                                           |
| :-----: | :------------------------------------------------------------: | :------------------------------------------------------: | :--------------------------------------------------------------------------------------------: |
| P(X)    | 좋은 데이터: P(X<sub>i</sub>), P(X<sub>o</sub>)<sub>true</sub> | 장비 sensor 등에서 측정한 데이터의 분포                  | Covariate shift: P(X<sub>o</sub>) ≠ P(X<sub>i</sub>) (새 장비, sensor drift, 신규 recipe 유입) |
| P(Y)    | 좋은 예측: P(Y)<sub>pred</sub>                                 | Wafer 계측값 (측정 map, 공간 분해 계수) 의 분포          | Prior shift: 실제 P(Y) 이동 (target spec·계수 분포 이동)                                       |
| P(Y\|X) | 좋은 모델: P(Y\|X<sub>i</sub>)<sub>pred</sub>                  | 공정 물리가 측정 데이터와 계측값 사이에 남긴 조건부 관계 | Concept drift: 학습 뒤 실제 P(Y\|X) 변화 (chamber 노화 등)                                     |

### 4.1 P(X) Covariate Shift

- 의미: 학습 데이터 X<sub>i</sub> 와 추론 데이터 X<sub>o</sub> 의 분포가 다르다 (P(X<sub>o</sub>) ≠ P(X<sub>i</sub>)). 관계 P(Y|X) 는 그대로일 수 있다.
- 해석: model 이 관계는 학습했으나, 추론 데이터가 학습 데이터로 드물게 덮인 영역으로 옮겨 가 식 (1) 의 좋은 데이터가 깨진 경우다.
- 범위: 데이터 공간 작업은 shift 대응과 함께 feature 선택·생성 전반을 포함한다.

### 4.2 P(Y) Prior Shift

- 의미: 출력 주변분포가 이동한다 (label shift, prior probability shift). 식 (5) 에서 P(X|Y) 가 그대로인 채 P(Y) 만 바뀌는 경우이며, 식 (3) 의 분해로 보면 P(X) 와 P(Y|X) 가 함께 바뀐 것으로 나타난다.
- 확장: 출력공간을 어떻게 정의하고 구조화하는가, 곧 target 변환과 분해도 이 항목에 든다.
- Spatial decomposition: wafer 측정 map 을 공간 기저로 분해하여 출력공간에 개입하므로 P(Y) 에 속한다. 효과는 P(Y|X) 학습 난이도를 낮추는 쪽으로 전파된다. 출력을 매끄럽고 물리적 의미가 있는 계수로 바꾸면 관계 학습이 쉬워진다.

### 4.3 P(Y|X) Concept Drift

측정 데이터와 계측값 사이의 관계 자체가 학습 뒤에 변하는 경우이며, 식 (1) 의 P(Y|X<sub>i</sub>)<sub>pred</sub> 가 추론 시점의 실제 P(Y|X) 와 어긋나 좋은 모델이 깨진다. 세 항목 가운데 다루기 가장 어렵고, 대응과 관측을 구분한다.

- **대응**: detrending, 최신성 sample 가중, 최근 drift windowing. 관계가 변한다고 가정하고 최근 sample 에 가중치를 더 주며, 그 효과는 temporal CV 로 시간순으로 검증한다. 변동 시점은 특정하지 못한다.
- **관측**: 변화를 측정하고 시점을 특정한다.

관측 방법은 P(Y|X) 를 얼마나 직접 보는지로 갈린다.

- **Binning CDT**: X 를 bin 으로 나눠 P(Y|bin) 을 시간창별로 검정하여 변동 시점을 특정한다. P(Y|X) 를 가장 직접 본다.
- **잔차 CUSUM, Page-Hinkley**: 예측 잔차 통계량이 임계값을 넘는 시점을 출력한다.
- **I(X;Y)**: 의존성 총량 (거시 지표). 단독으로 쓰면 P(X), P(Y), 관계의 변화가 함께 잡히므로, 시간창별로 추적해야 concept drift 에 가까워진다.

---

## Appendix A. Terminology

- **aleatoric uncertainty**: 데이터 자체가 지닌 산포에서 오는 불확실성. 데이터를 늘려도 줄지 않는다.
- **CUSUM (Cumulative Sum)**: 기준값과의 편차를 누적하여 임계값을 넘는 시점을 변화점으로 보는 관리도.
- **epistemic uncertainty**: 학습 데이터가 부족하여 model 이 지니는 불확실성. 데이터를 늘리면 줄어든다.
- **I(X;Y)**: 상호정보량 (mutual information). X 와 Y 의 의존성 총량을 나타내는 거시 지표.
- **i.i.d. (independent and identically distributed)**: 각 표본이 서로 독립이고 같은 분포에서 나온다는 가정.
- **label noise**: 계측 오차처럼 정답값 Y 에 섞인 잡음.
- **point mass**: 한 값에서만 확률 1 이고 다른 모든 값에서 확률 0 인 분포. 그 값 하나만 나온다.
- **selection bias**: 표본을 고르는 방식 때문에 표본의 분포가 모집단의 분포와 달라지는 치우침.
- **Spatial decomposition**: wafer 측정 map 을 공간 기저 (다항식) 로 분해하고, 그 계수 (a1, …) 를 예측하는 방법.
- **temporal CV**: 과거로 학습하고 미래로 검증하는 시간순 교차검증.
- **주변분포 (marginal distribution)**: 결합분포 P(X,Y) 에서 X 를 적분하여 없애고 Y 하나만 남긴 분포. P(Y) = ∫ P(X,Y) dX 이며, X 의 값과 상관없이 Y 가 어떻게 분포하는지를 나타낸다.

## Appendix B. Detection Methods

### B.1 Covariate Shift Detection

- **PSI (Population Stability Index)**: 변수마다 학습 데이터와 추론 데이터의 구간별 비율을 비교하여 P(X) 의 이동 크기를 잰다.
- **KS test (Kolmogorov–Smirnov test)**: 변수마다 두 데이터의 누적분포 최대 차이로 분포가 같은지 검정한다.
- **KL divergence (Kullback–Leibler divergence)**: 학습 데이터 분포에 대한 추론 데이터 분포의 차이를 정보량으로 잰다.
- **MMD (Maximum Mean Discrepancy)**: kernel 공간에서 두 데이터의 평균 차이로 다변량 분포 차이를 검정한다.
- **Domain classifier**: 학습 데이터와 추론 데이터를 가르는 classifier 를 학습하여, AUC 가 0.5 보다 클수록 두 분포가 다르다고 본다.
- **Hotelling T²·SPE (PCA 기반)**: 학습 데이터로 만든 PCA model 에서 추론 데이터의 T² 와 잔차 SPE 가 관리 한계를 넘는지 본다.

### B.2 Prior Shift Detection

- **계측값 분포 비교**: 학습 데이터의 계측값과 최근 계측값의 분포를 KS test 나 PSI 로 비교한다.
- **SPC 관리도 (Shewhart·EWMA)**: 계측값의 평균과 산포가 관리 한계를 벗어나는 시점을 감시한다.
- **예측 분포 감시**: 계측값이 늦게 들어올 때 P(Y)<sub>pred</sub> 의 분포 이동을 먼저 감시하여 P(Y) 이동을 미리 알린다.

### B.3 Concept Drift Detection

- **Binning CDT (Conditional Distribution Test)**: X 를 bin 으로 나눠 bin 마다 P(Y|bin) 을 시간창별로 검정하여 변동 시점을 특정한다.
- **잔차 CUSUM**: 예측 잔차의 누적합이 임계값을 넘는 시점을 관계 변화 시점으로 본다.
- **Page-Hinkley**: 예측 잔차의 누적 편차와 그 최솟값의 차이가 임계값을 넘으면 평균 변화를 알린다.
- **ADWIN (Adaptive Windowing)**: 오차 window 를 두 부분으로 나눠 평균 차이가 유의하면 오래된 부분을 버리고 변화를 알린다.
- **시간창별 성능 감시**: 시간창마다 R²·RMSE 를 계산하여 성능 저하 시점을 찾는다. 계측값이 있어야 한다.
- **시간창별 I(X;Y)**: 시간창마다 상호정보량을 계산하여 측정 데이터와 계측값의 의존성 변화를 추적한다.
