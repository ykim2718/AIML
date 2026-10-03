# P(X) · P(Y) · P(Y|X) Taxonomy of the Joint Distribution for Semiconductor Process AI/ML
Rev. 5 | Created: 2026-05-29 | Updated: 2026-10-03 00:56 CDT

## 1. Purpose

- **Problem Statement**: AI/ML model 개발을 위한 요소를 묶는 체계가 없다.
- **Goal**: P(X), P(Y), P(Y|X) 를 결합분포 P(X,Y) 로 묶어 반도체 공정 AI/ML model 의 taxonomy 와 physical meaning을 밝힌다.
- **Non-Goal**: 결합분포 밖에 있는 Model (추정기·최적화) 축과 특정 과제의 진단 수치는 다루지 않는다.

## 2. Summary

> ### P(Y)<sub>pred</sub> = P(X<sub>o</sub>)<sub>true</sub> · P(Y|X<sub>i</sub>)<sub>pred</sub>
>
> ### X<sub>i</sub> = in-samples, X<sub>o</sub> = out-of-samples
>
> ### 좋은 예측은 좋은 데이타와 좋은 모델에서 나옵니다.

인용한 식은 식 (2) 를 예측에 옮겨 곱의 형태로 줄여 적은 것이며, 적분까지 적으면 P(Y)<sub>pred</sub> = ∫ P(Y|X<sub>i</sub>)<sub>pred</sub> · P(X<sub>o</sub>)<sub>true</sub> dX 이다. Model 은 in-sample 입력 X<sub>i</sub> 와 그 계측값으로 학습하여 조건부 관계 P(Y|X<sub>i</sub>)<sub>pred</sub> 를 추정하고, 추론에서는 그 관계를 out-of-sample 입력의 실제 분포 P(X<sub>o</sub>)<sub>true</sub> 에 적용하여 예측한 출력분포 P(Y)<sub>pred</sub> 를 얻는다.

- **좋은 데이터**: X<sub>i</sub> 는 장비·recipe 의 입력과 wafer 계측값을 짝지은 학습 표본의 입력이고, P(X<sub>o</sub>)<sub>true</sub> 는 추론 때 장비·recipe 가 실제로 내놓는 입력의 분포다. X<sub>o</sub> 가 X<sub>i</sub> 의 범위를 벗어나면 covariate shift 다.
- **좋은 model**: P(Y|X<sub>i</sub>)<sub>pred</sub> 는 model 이 X<sub>i</sub> 로 추정한 공정 물리다. 실제 P(Y|X) 에 가까워야 하며, 실제 관계가 학습 뒤에 바뀌면 concept drift 다.
- **결과**: P(Y)<sub>pred</sub> 는 두 factor 에서 유도되는 값이므로, 두 factor 가운데 하나만 어긋나도 실제 P(Y) 에서 벗어난다.

- **Taxonomy**: 반도체 공정 AI/ML model 의 요소는 결합분포 P(X,Y) 의 세 항목, 곧 입력 주변분포 P(X), 출력 주변분포 P(Y), 조건부 관계 P(Y|X) 로 분류된다.
- **Hierarchy**: P(X,Y) = P(Y|X) · P(X) 로 분해되는 두 factor 가 P(X) 와 P(Y|X) 이고, P(Y) 는 두 factor 의 곱을 X 에 대해 적분한 주변분포다 (식 (2)).
- **Change**: 세 항목의 변화는 각각 covariate shift, prior shift, concept drift 이다.
- **Physical meaning**: P(X) 는 장비·recipe 가 내놓는 입력의 분포, P(Y) 는 wafer 계측값의 분포, P(Y|X) 는 그 입력에서 계측값을 정하는 공정 물리다.
- **Reading rule**: 장비·recipe 를 바꾸면 입력과 관계가 함께 이동하므로, 세 항목을 배타적 분류 대신 **관측·개입 지점** 으로 읽는다.

## 3. Taxonomy and its Hierarchy

결합분포는 관계 P(Y|X) 와 입력 P(X) 의 곱으로 분해되고, 출력 P(Y) 는 그 곱을 X 에 대해 적분하여 얻는다 ([Fig 1](#fig-1)).

```text
                 P(X, Y)   joint distribution
                    |
      factorize     |   P(X, Y) = P(Y|X) * P(X)
          +---------+---------+
          |                   |
        P(X)               P(Y|X)
   input marginal      conditional relation
   covariate shift     concept drift
          |                   |
          +---------+---------+
      marginalize   |   integrate over X
                    |
                  P(Y)
            output marginal
            prior shift

  Model (estimator): orthogonal axis, outside P(X, Y)
```

<a id="fig-1"></a>
Fig 1. Taxonomy and hierarchy of the joint distribution

- 위층의 P(X) 와 P(Y|X) 는 결합분포를 이루는 두 factor 다.
- 아래층의 P(Y) 는 두 factor 에서 유도되는 주변분포이므로, P(X) 나 P(Y|X) 가 바뀌면 함께 바뀐다.
- Model 은 결합분포를 추정하는 쪽이므로 세 항목과 직교 (orthogonal) 한다.

```math
P(X, Y) = P(Y \mid X) \cdot P(X) \hspace{19em} (1)
```

```math
P(Y) = \int P(Y \mid X)\, P(X)\, dX \hspace{19em} (2)
```

### 3.1 Placement

Table 1 은 세 항목을 분해, 변화, 개입, 질문, 관측, 대책의 여섯 축에 놓는다.

Table 1. Six lenses on P(X), P(Y) and P(Y|X)

| Lens              | P(X)                                                       | P(Y)                                         | P(Y\|X)                                               |
| :---------------: | :--------------------------------------------------------: | :------------------------------------------: | :---------------------------------------------------: |
| 확률 분해         | 입력 주변분포                                              | 출력 주변분포                                | 조건부 (관계)                                         |
| 분포 변화 (shift) | Covariate shift                                            | Prior / label shift                          | Concept drift                                         |
| 개입 지점         | 입력공간                                                   | 출력공간 (target 구조화)                     | 관계·mechanism                                        |
| 질문 형태         | 들어오는 데이터가 달라졌나?                                | 정답 분포가 달라졌나?                        | 입력→출력 규칙이 달라졌나?                            |
| 관측 (탐지)       | PSI·KS·KL, domain classifier                               | Target 주변분포 비교                         | Binning CDT, 시간창별 I(X;Y), 잔차 CUSUM·Page-Hinkley |
| 대책 (lever)      | Feature 선택·증강, importance weighting, domain adaptation | Target 변환·분해, group 별 scale, prior 보정 | 재학습 period, 최신성 가중, detrending, drift 적응    |

관측 행은 변화를 재는 방법이고, 대책 행은 변화를 가정하고 model 을 맞추는 방법이다. 변동 시점을 특정하는 것은 관측 행의 방법뿐이다.

## 4. Physical Meaning

세 항목은 반도체 공정에서 각각 입력, 계측 결과, 공정 물리에 대응한다. Table 2 가 그 대응을 모은다.

Table 2. Physical meaning of each term

| Term    | Physical meaning                                | Process event that moves it             |
| :-----: | :---------------------------------------------: | :-------------------------------------: |
| P(X)    | 장비 sensor 값·recipe 설정의 분포               | 새 장비, sensor drift, 신규 recipe 유입 |
| P(Y)    | Wafer 계측값 (측정 map, 공간 분해 계수) 의 분포 | Target spec·계수 분포 이동              |
| P(Y\|X) | 입력에서 계측값을 정하는 공정 물리              | 같은 입력에 다른 결과 (chamber 노화 등) |

### 4.1 P(X) Covariate Shift

- 의미: 학습 때와 운영 때의 입력분포가 다르다. 관계 P(Y|X) 는 그대로일 수 있다.
- 해석: model 이 관계는 학습했으나, 학습 때 없던 입력이 들어온 경우다.
- 범위: 입력공간 작업은 shift 대응과 함께 feature 선택·생성 전반을 포함한다.

### 4.2 P(Y) Prior Shift

- 의미: 출력 주변분포가 이동한다 (label shift, prior probability shift).
- 확장: 출력공간을 어떻게 정의하고 구조화하는가, 곧 target 변환과 분해도 이 항목에 든다.
- Spatial decomposition: wafer 측정 map 을 공간 기저로 분해하여 출력공간에 개입하므로 P(Y) 에 속한다. 효과는 P(Y|X) 학습 난이도를 낮추는 쪽으로 전파된다. 출력을 매끄럽고 물리적인 값으로 바꾸면 관계 학습이 쉬워진다.

### 4.3 P(Y|X) Concept Drift

입력→출력 관계 자체가 변하는 경우이며, 세 항목 가운데 다루기 가장 어렵다. 대응과 관측을 구분한다.

- **대응**: detrending, 최신성 sample 가중, 최근 drift windowing, temporal CV. 관계가 변한다고 가정하고 최근 sample 에 가중치를 더 주며, 변동 시점은 특정하지 못한다.
- **관측**: 변화를 측정하고 시점을 특정한다.

관측 방법은 P(Y|X) 를 얼마나 직접 보는지로 갈린다.

- **Binning CDT**: X 를 bin 으로 나눠 P(Y|bin) 을 시간창별로 검정하여 변동 시점을 특정한다. P(Y|X) 를 가장 직접 본다.
- **잔차 CUSUM, Page-Hinkley**: 예측 잔차 통계량이 임계값을 넘는 시점을 출력한다.
- **I(X;Y)**: 의존성 총량 (거시 지표). 단독으로 쓰면 P(X), P(Y), 관계의 변화가 함께 잡히므로, 시간창별로 추적해야 concept drift 에 가까워진다.

---

## Appendix A. Terminology

- **Binning CDT (Conditional Distribution Test)**: X 를 bin 으로 나눠 P(Y|bin) 을 시간창별로 검정하여 변동 시점을 특정하는 방법.
- **CUSUM (Cumulative Sum)**: 기준값과의 편차를 누적하여 임계값을 넘는 시점을 변화점으로 보는 관리도.
- **I(X;Y)**: 상호정보량 (mutual information). X 와 Y 의 의존성 총량을 나타내는 거시 지표.
- **KL (Kullback–Leibler divergence)**: 두 분포의 차이를 정보량으로 잰 값.
- **KS (Kolmogorov–Smirnov test)**: 두 표본의 누적분포 최대 차이로 분포가 같은지 검정하는 방법.
- **Page-Hinkley**: 누적 편차와 그 최솟값의 차이가 임계값을 넘으면 평균 변화를 알리는 순차 검정.
- **PSI (Population Stability Index)**: 두 시점의 구간별 분포 비율 차이로 분포 이동을 재는 지표.
- **Spatial decomposition**: wafer 측정 map 을 공간 기저 (다항식) 로 분해하고, 그 계수 (a1, …) 를 예측하는 방법.
- **temporal CV**: 과거로 학습하고 미래로 검증하는 시간순 교차검증.
