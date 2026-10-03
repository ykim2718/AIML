# Modeling Elements from Joint Distribution Decomposition for Manufacturing Data
Rev. 71 | Created: 2026-05-29 | Updated: 2026-10-03 11:53 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Placement](#31-placement)
- [4. Prediction from the Joint Distribution](#4-prediction-from-the-joint-distribution)
- [5. Physical Meaning](#5-physical-meaning)
  - [5.1 P(X) Shift and Drift](#51-px-shift-and-drift)
  - [5.2 P(Y) Shift and Drift](#52-py-shift-and-drift)
  - [5.3 P(Y|X) Shift and Drift](#53-pyx-shift-and-drift)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Detection and Implementation by Cell](#appendix-b-detection-and-implementation-by-cell)
  - [B.1 P(X) Shift and Drift](#b1-px-shift-and-drift)
  - [B.2 P(Y) Shift and Drift](#b2-py-shift-and-drift)
  - [B.3 P(Y|X) Shift and Drift](#b3-pyx-shift-and-drift)
  - [B.4 Estimation Error](#b4-estimation-error)
- [Appendix C. Benchmarking](#appendix-c-benchmarking)
- [Appendix D. Prior in Bayes' Theorem](#appendix-d-prior-in-bayes-theorem)
- [Appendix E. Talk Slides](#appendix-e-talk-slides)

## 1. Purpose

- **Problem Statement**: AI/ML model 개발을 위한 요소를 묶는 체계가 없다.
- **Goal**: 결합확률분포 $P(X,Y) = P(Y\vert{}X) \cdot P(X)$의 분해를 통해 제조 공정 데이터 기반 AI/ML의 요소 (데이터, 모델, 예측) 와 그 변화 (shift, drift), 그리고 model 의 추정 오차를 명확하게 연결하는 framework 를 제시한다.
- **Non-Goal**: 특정 과제의 진단 수치는 다루지 않는다.

## 2. Summary

> ### 좋은 예측은 좋은 데이터와 좋은 모델에서 나옵니다.

- **Taxonomy**: 결합분포 P(X,Y) 를 chain rule 로 분해하면 데이터 P(X), 모델 P(Y|X), 예측 P(Y) 의 세 요소가 나오고, 각 요소는 한 번에 바뀌는 shift 와 서서히 바뀌는 drift 를 겪는다 (section 3).
- **Error**: 예측 오차는 model 의 추정 오차와 학습 뒤 X → Y 관계의 변화로 나뉘며, P(X) 의 변화는 추정 오차가 큰 영역으로 가중치를 옮겨 오차를 키운다 (section 4).
- **Reading rule**: X → Y 구조인 제조 공정에서 P(Y) 의 변화는 대부분 P(X) 나 X → Y 관계가 바뀐 결과이며, 세 요소는 불확실성 출처·표본 선택·i.i.d. 가정을 전제로 한다 (section 3, section 5).

## 3. Taxonomy and its Hierarchy

결합분포는 관계 P(Y|X) 와 데이터 분포 P(X) 의 곱으로 분해되고, 출력 P(Y) 는 그 곱을 X 에 대해 적분하여 얻으며, 세 요소는 각각 shift 와 drift 의 두 방식으로 바뀐다 ([Fig 1](#fig-1)).

```text
                 P(X, Y)   joint distribution
                    |
      factorize     |   P(X, Y) = P(Y|X) * P(X)
          +---------+---------+
          |                   |
        P(X)               P(Y|X)
   data marginal       relation X -> Y
   good data           good model
          |                   |
          +---------+---------+
      marginalize   |   integrate over X
                    |
                  P(Y)
            output marginal
            good prediction

  change mode (each element): shift = changes at once, drift = changes gradually
  estimation error: P(Y|X)_model departs from the true P(Y|X), set at training
  reverse factorization: P(X, Y) = P(X|Y) * P(Y), defines P(Y) shift in a Y -> X structure
  premises: uncertainty sources, sample selection, i.i.d. and lot / chamber hierarchy
```

<a id="fig-1"></a>
Fig 1. Taxonomy and hierarchy of the joint distribution

- 위층의 P(X) 와 P(Y|X) 는 결합분포를 이루는 두 factor 다.
- 아래층의 P(Y) 는 두 factor 에서 유도되는 marginal distribution 이므로, P(X) 나 P(Y|X) 가 바뀌면 P(Y) 도 따라 바뀔 수 있다.
- 세 요소의 변화는 빠르기에 따라 한 번에 바뀌는 shift 와 시간에 따라 서서히 바뀌는 drift 로 갈리며, 두 방식은 탐지 방법과 대응 방법이 다르다 (section 3.1).
- 추정 오차는 실제 관계가 그대로여도 P(Y|X)<sub>model</sub> 이 실제 P(Y|X) 와 다른 정도이며, 예측 오차를 이루는 한 항이다 (section 4).

이 문서의 P(·) 는 확률분포 (이산 변수는 확률질량함수, 연속 변수는 확률밀도함수) 를 뜻한다. 조건부 분포의 정의 P(Y|X) = P(X,Y) / P(X) 로부터, 결합분포는 식 (1) 의 chain rule 로 분해된다.

```math
P(X, Y) = P(Y \mid X) \cdot P(X) \hspace{19em} (1)
```

```math
P(Y) = \int P(Y \mid X)\, P(X)\, dX \hspace{19em} (2)
```

식 (1) 의 양변을 X 에 대해 적분하면 식 (2) 를 얻는다. 좌변 ∫ P(X,Y) dX 는 X 를 적분하여 없앤 Y 의 marginal distribution P(Y) 이고, 우변은 조건부 분포 P(Y|X) 를 X 의 분포 P(X) 로 가중하여 평균한 값이다. 이 적분을 marginalization 이라 하며, 식 (1) 이 결합분포를 두 factor 로 나눈다면 식 (2) 는 두 factor 에서 P(Y) 를 다시 얻는다.

꼭지 2 에 있는 인용구의 세 요소는 식 (1) 과 식 (2) 의 세 자리에 놓인다.

- **$`P(Y)`$ (Good Prediction):** 좋은 예측. 두 factor 의 곱인 결합분포 P(X,Y) 를 X 에 대해 적분하여 얻는 marginal distribution 이다 (식 (2)).
- **$`P(X)`$ (Good Data):** 좋은 데이터. 식 (1) 의 factor 로, 측정 데이터의 분포다.
- **$`P(Y \mid X)`$ (Good Model):** 좋은 모델. 식 (1) 의 factor 로, 측정 데이터가 주어졌을 때 계측값이 나오는 X → Y 관계다.

결합분포는 식 (1) 과 반대 방향으로도 분해된다.

```math
P(X, Y) = P(X \mid Y) \cdot P(Y) \hspace{19em} (3)
```

P(Y) shift 는 식 (3) 에서 P(X|Y) 가 그대로인 채 P(Y) 만 바뀌는 경우로 정의되며, 문헌은 이것을 prior shift 라 부른다 ([Appendix D](#appendix-d-prior-in-bayes-theorem)). 이 정의는 Y 가 X 의 원인인 인과 구조 (Y → X) 에서 성립하며, 공정이 진행되어 측정 데이터 X 를 남긴 뒤 계측값 Y 가 나오는 제조 공정은 X → Y 구조다. X → Y 구조에서는 공정 변화가 P(Y|X) 와 P(X) 를 통해 P(X|Y) 까지 바꾸므로, P(X|Y) 가 그대로라는 전제가 깨지는 경우가 많다. 그 경우 그 전제에 기댄 보정 (label shift 보정) 은 근거를 잃는다.

세 요소는 아래 세 전제 위에서 성립한다.

- **불확실성의 출처**: 계측값은 Y<sub>obs</sub> = Y + ε<sub>m</sub>, 측정 데이터는 X<sub>obs</sub> = X + ε<sub>x</sub> 이다. Model 이 학습하는 P(Y<sub>obs</sub>|X<sub>obs</sub>) 의 산포에는 공정 고유 산포, 계측 오차 ε<sub>m</sub> (label noise), sensor 측정 오차 ε<sub>x</sub> 가 함께 들어 있으며, 셋은 데이터를 늘려도 줄지 않는 aleatoric uncertainty 다. ε<sub>x</sub> 는 산포를 키울 뿐 아니라 추정한 관계의 기울기를 0 쪽으로 줄여 (regression dilution) P(Y|X) 추정을 왜곡한다. 학습 데이터가 부족해 생기는 model 의 불확실성 (epistemic uncertainty) 만 데이터로 줄어들며, 좋은 모델이 도달할 수 있는 정확도의 한계는 공정 고유 산포와 두 측정 오차가 정한다.
- **표본 선택**: 계측은 일부 wafer 만 sampling 하므로, 학습 데이터 X<sub>i</sub> 의 분포 P(X<sub>i</sub>) 는 계측된 wafer 의 분포이며 전체 wafer 의 P(X) 와 다를 수 있다 (selection bias). 추론은 계측하지 않은 wafer 에 하므로, 이 차이는 그대로 P(X) shift 가 된다. X 나 Y 의 값에 따라 계측이 누락 (missing) 되면 P(X<sub>i</sub>) 나 학습 데이터의 P(Y) 가 같은 방식으로 치우친다.
- **i.i.d. 가정과 계층 구조**: 식 (4) 로 학습 데이터에서 추론 데이터의 예측을 얻는 것은 wafer 가 서로 독립이고 같은 분포에서 나온다 (i.i.d.) 는 가정 위에서다. 공정 데이터에는 시간 자기상관과 lot·chamber 의 계층 구조가 있어 같은 lot·chamber 의 wafer 가 서로 닮으며, 이 구조를 무시하고 학습·검증을 나누면 같은 lot 이 양쪽에 들어가 성능이 실제보다 높게 나온다.

세 요소의 변화는 Table 1 의 3×2 칸으로 나뉜다.

Table 1. Shift and drift of the three elements

| Cell          | Example                        | Effect                                  |
| :-----------: | :----------------------------: | :-------------------------------------: |
| P(X) shift    | 새 장비, recipe 변경           | X 의 값 범위가 한 번에 옮겨 감          |
| P(X) drift    | sensor drift                   | 측정값 X<sub>obs</sub> 가 서서히 치우침 |
| P(Y) shift    | target spec 변경               | Y 의 계측 기준이 한 번에 바뀜           |
| P(Y) drift    | 제품 구성의 점진적 변화        | 제품별 비율을 따라 Y 가 서서히 옮겨 감  |
| P(Y\|X) shift | PM, 부품 교체                  | X → Y 관계가 한 번에 바뀜               |
| P(Y\|X) drift | chamber 노화, 찌꺼기 누적 Z(t) | X → Y 관계가 서서히 바뀜                |

한 사건이 여러 칸에 함께 영향을 줄 수 있으며, Table 1 은 각 사건을 가장 먼저 바뀌는 요소의 칸에 둔다. Sensor drift 는 P(X) drift 이면서 측정값 기준의 관계 P(Y|X<sub>obs</sub>) 도 바꾸는 예다. 문헌과 업계 도구는 같은 칸을 다른 이름으로 부른다. P(X) shift 는 covariate shift, P(Y) shift 는 prior shift 또는 label shift, P(Y|X) shift 와 drift 는 concept shift 와 concept drift 이며, model monitoring 도구는 P(X) drift 를 data drift 라 부른다 ([Appendix C](#appendix-c-benchmarking)).

### 3.1 Placement

Table 2 는 세 요소를 분해, 변화, 개입, 질문, 관측, 대책의 여섯 관점에 놓는다.

Table 2. Six lenses on P(X), P(Y) and P(Y|X)

| Lens         | P(X)                                                                | P(Y)                                         | P(Y\|X)                                         |
| :----------: | :-----------------------------------------------------------------: | :------------------------------------------: | :---------------------------------------------: |
| 분포 분해    | 측정 데이터의 marginal distribution                                 | 출력의 marginal distribution                 | X → Y 조건부 관계                               |
| 분포 변화    | P(X) shift, P(X) drift                                              | P(Y) shift, P(Y) drift                       | P(Y\|X) shift, P(Y\|X) drift                    |
| 개입 지점    | 데이터 공간                                                         | Target engineering (변환·분해)               | 모델·algorithm 공간 (관계 학습)                 |
| 질문 형태    | 추론 데이터 X<sub>o</sub> 가 학습 데이터 X<sub>i</sub> 와 달라졌나? | 계측값 분포가 달라졌나?                      | 학습 뒤 측정 데이터와 계측값의 관계가 달라졌나? |
| 관측 (탐지)  | PSI·KS, domain classifier, T²·SPE 관리도                            | 계측값 분포 비교, Shewhart·EWMA 관리도       | Binning CDT, 잔차 CUSUM·Page-Hinkley, ADWIN     |
| 대책 (lever) | Feature 선택·증강, importance weighting, domain adaptation          | Target 변환·분해, group 별 scale, prior 보정 | 사건 뒤 재학습, 최신성 가중, adaptive update    |

관측 행은 변화를 재는 방법이고, 대책 행은 변화를 가정하고 model 을 맞추는 방법이다. Shift 는 학습 데이터와 추론 데이터의 두 집합을 비교하여 찾고, drift 는 시간창마다 통계량을 추적하는 관리도와 순차 검정으로 찾는다. 변동 시점을 특정하는 것은 관측 행의 방법뿐이다. 여섯 칸과 추정 오차의 탐지·대응·검증 방법은 [Appendix B](#appendix-b-detection-and-implementation-by-cell) 에 모은다. 이 분류가 학계와 업계에서 쓰이는 사례는 [Appendix C](#appendix-c-benchmarking) 에 모은다.

## 4. Prediction from the Joint Distribution

식 (4) 는 학습한 model 로 추론할 때 세 요소의 관계를 학습 데이터 X<sub>i</sub> 와 추론 데이터 X<sub>o</sub> 로 나누어 적고, 식 (5) 는 그 예측의 오차를 추정 오차와 관계 변화의 두 항으로 나눈다. X<sub>i</sub> 는 in-sample, 곧 model 을 학습할 때 쓴 데이터이고, X<sub>o</sub> 는 out-of-sample, 곧 추론 때 새로 들어오는 데이터다.

```math
P(Y)_{\mathrm{pred}} = \int P(Y \mid X = x;\, X_{i})_{\mathrm{model}} \cdot P(X_{o} = x)_{\mathrm{true}}\, dx \hspace{19em} (4)
```

- **$`P(Y)_{\mathrm{pred}}`$ (Overall Predicted Distribution):** out-of-sample 추론 데이터에서 나오리라 기대하는 target 변수 $`Y`$ 의 최종 분포로, 측정 데이터를 적분하여 없앤 marginal distribution 이다.
- **$`P(Y \mid X = x;\, X_i)_{\mathrm{model}}`$ (Model's Conditional Prediction):** 예측 model 그 자체다. In-sample 학습 데이터 $`X_i`$ 로 학습하며, 측정 데이터의 값 $`x`$ 가 주어지면 $`Y`$ 의 조건부 분포를 내놓는다. 첨자 `model` 은 이것이 추정·학습한 함수이며 실제 분포와 다를 수 있음을 나타낸다. `;` 뒤의 $`X_i`$ 는 model 을 학습한 데이터를 나타내며, `|` 뒤의 조건 변수와 구별된다. 이 문서의 다른 자리에서는 줄여 P(Y|X)<sub>model</sub> 로 적는다.
- **$`P(X_o = x)_{\mathrm{true}}`$ (True Distribution of Out-of-Sample Data):** out-of-sample 추론 데이터 $`X_o`$ 가 값 $`x`$ 를 가질 실제 확률밀도다. Model 을 추론에 쓸 때 $`X_o`$ 가 실제로 어떻게 분포하는지를 나타낸다.
- **$`\int \ldots dx`$ (Marginalization over $`x`$):** 추론 데이터가 가질 수 있는 모든 값 $`x`$ 에 걸쳐 예측을 더한다. 각 값의 가중치는 추론 때 그 값이 나올 확률이므로, 적분 결과는 예측의 가중 평균이다.

식 (4) 는 출력의 marginal distribution 식 (2) 를 예측에 옮긴 것이다. 추론 시점 t 의 실제 P(Y)<sub>t</sub> 는 그 시점의 실제 관계 P(Y|X=x)<sub>t</sub> 를 같은 P(X<sub>o</sub>) 로 적분하여 얻으므로, 식 (4) 에서 P(Y)<sub>t</sub> 를 빼면 예측 오차가 식 (5) 의 두 항으로 나뉜다. t<sub>i</sub> 는 학습 시점이다.

```math
\begin{aligned}
P(Y)_{\mathrm{pred}} - P(Y)_{t}
&= \int \left[ P(Y \mid X = x;\, X_{i})_{\mathrm{model}} - P(Y \mid X = x)_{t_{i}} \right] P(X_{o} = x)_{\mathrm{true}}\, dx \\
&+ \int \left[ P(Y \mid X = x)_{t_{i}} - P(Y \mid X = x)_{t} \right] P(X_{o} = x)_{\mathrm{true}}\, dx
\end{aligned}
\hspace{19em} (5)
```

- **Estimation error**: 첫 항. 실제 관계가 학습 시점 그대로여도 model 이 그 관계를 다르게 추정한 몫이다.
- **Relation change**: 둘째 항. 학습 뒤 실제 X → Y 관계가 바뀐 몫이며, Table 1 의 P(Y|X) shift 와 drift 가 여기에 든다.
- **Weight**: 두 항은 모두 추론 데이터의 분포 P(X<sub>o</sub>) 로 가중된다. P(X) shift 나 drift 는 괄호 안의 차이를 바꾸지 않고 가중치를 옮겨, 차이가 큰 영역의 몫을 키운다.

P(Y|X)<sub>model</sub> 이 실제 관계를 반영하지 못하는 원인은 아래 다섯 가지이며, 앞의 넷은 첫 항에, 마지막 하나는 둘째 항에 든다.

- **Variance**: 표본이 변수에 비해 적으면 추정한 관계가 학습 표본에 따라 크게 흔들린다.
- **Bias**: model 계열이 실제 관계의 형태를 담지 못하면 표본을 늘려도 오차가 남는다. 비선형 관계를 선형 model 로 맞추는 경우다.
- **Measurement error**: 계측 오차 ε<sub>m</sub> 은 산포를 키우고, sensor 오차 ε<sub>x</sub> 는 추정한 기울기를 0 쪽으로 줄인다 (section 3 의 불확실성의 출처).
- **Extrapolation**: P(X<sub>i</sub>) 가 드문 영역에서는 model 을 묶어 줄 학습 표본이 없어 추정이 정해지지 않는다. P(X) shift 나 drift 가 P(X<sub>o</sub>) 를 그 영역으로 옮기면 식 (5) 에서 이 몫이 커진다.
- **Unobserved state**: chamber 상태 Z 가 측정 데이터 X 에 없으면 model 은 학습 시점의 Z 분포로 평균한 관계를 배운다. 학습 뒤 Z 의 분포가 바뀌면 그 차이가 둘째 항이 된다 (section 5.3).

식 (5) 로 보면 세 요소는 아래와 같다.

- **좋은 예측**: P(Y)<sub>pred</sub> 는 식 (5) 의 두 항이 모두 작을 때 실제 P(Y) 에 가깝다.
- **좋은 데이터**: X<sub>i</sub> 는 wafer 계측값과 짝지어 model 학습에 쓰는, 장비 sensor 등에서 측정한 학습 데이터이고, P(X<sub>o</sub>)<sub>true</sub> 는 추론 때 실제로 측정되는 추론 데이터의 분포다. P(X<sub>i</sub>) 가 P(X<sub>o</sub>) 를 덮어야 Extrapolation 몫이 작다.
- **좋은 모델**: P(Y|X)<sub>model</sub> 은 model 이 학습 데이터 X<sub>i</sub> 로 추정한 X → Y 관계다. 첫 항은 추정기 선택과 검증으로 줄이고 ([B.4](#b4-estimation-error)), 둘째 항은 관계 변화를 탐지하여 다시 학습해야 줄어든다 ([B.3](#b3-pyx-shift-and-drift)).

## 5. Physical Meaning

세 요소는 제조 공정에서 각각 측정 데이터, 계측 결과, 측정 데이터와 계측값 사이의 X → Y 관계에 대응한다. Table 3 이 그 대응과 식 (4) 에서의 역할을 모은다.

Table 3. Physical meaning of each term

| Term    | Role in eq. (4)                                                | Physical meaning                                        |
| :-----: | :------------------------------------------------------------: | :-----------------------------------------------------: |
| P(X)    | 좋은 데이터: P(X<sub>i</sub>), P(X<sub>o</sub>)<sub>true</sub> | 장비 sensor 등에서 측정한 데이터의 분포                 |
| P(Y)    | 좋은 예측: P(Y)<sub>pred</sub>                                 | Wafer 계측값 (측정 map, 공간 분해 계수) 의 분포         |
| P(Y\|X) | 좋은 모델: P(Y\|X)<sub>model</sub>                             | 공정 물리가 측정 데이터와 계측값 사이에 남긴 X → Y 관계 |

### 5.1 P(X) Shift and Drift

- 의미: P(X) shift 는 추론 데이터의 분포 P(X<sub>o</sub>) 가 학습 데이터의 P(X<sub>i</sub>) 와 한 번에 달라지는 것이고, P(X) drift 는 P(X<sub>o</sub>) 가 시간에 따라 서서히 옮겨 가는 것이다. X → Y 관계는 그대로일 수 있다.
- 해석: model 이 관계는 학습했으나 추론 데이터가 학습 데이터로 드물게 덮인 영역으로 옮겨 가, 식 (5) 의 Extrapolation 몫이 커진 경우다.
- 범위: 데이터 공간 작업은 변화 대응과 함께 feature 선택·생성 전반을 포함한다.

### 5.2 P(Y) Shift and Drift

- 의미: 출력의 marginal distribution 이 바뀐다. 식 (3) 에서 P(X|Y) 가 그대로인 채 P(Y) 만 바뀌는 경우로 정의되며, 이 정의는 Y → X 인과 구조에서 성립한다.
- 제조 공정에서의 해석: X → Y 구조에서 관측되는 P(Y) 변화는 대부분 P(X) 변화나 X → Y 관계 변화의 결과로 나타난다. P(Y) 자체가 바뀌는 경우는 target spec 변경처럼 Y 의 정의나 기준이 한 번에 바뀌는 P(Y) shift 와, 제품 구분이 X 에 없을 때 제품 구성이 서서히 바뀌어 Y 의 분포가 옮겨 가는 P(Y) drift 다.
- Target engineering: P(Y) 에 대한 개입은 출력의 정의와 구조를 바꾸는 것이며, target 변환 (log, Box-Cox), spatial decomposition, multi-task target 재구성이 여기에 든다.
- Spatial decomposition: wafer 측정 map 을 공간 기저의 계수로 바꾸는 target engineering 이다. 출력이 매끄럽고 물리적 의미가 있는 계수가 되어 P(Y|X) 학습 난이도가 낮아진다.

### 5.3 P(Y|X) Shift and Drift

측정 데이터와 계측값 사이의 X → Y 관계 자체가 학습 뒤에 바뀌는 경우이며, 식 (5) 의 둘째 항이 이것이다. PM 이나 부품 교체처럼 한 번에 바뀌면 P(Y|X) shift, chamber 노화처럼 서서히 바뀌면 P(Y|X) drift 다. 세 요소 가운데 다루기 가장 어렵다.

공정 물리 관점에서 X → Y 관계의 변화는 관측되지 않은 chamber 상태 변수 Z(t) (노화, 찌꺼기 등) 의 변화로 일어나며, 그 관계는 식 (6) 으로 적는다.

```math
P(Y \mid X, t) = \int P(Y \mid X, Z)\, P(Z \mid t)\, dZ \hspace{19em} (6)
```

Z 는 잠재 변수 (latent variable) 이다. 식 (6) 은 주어진 t 에서 Z 가 X 와 독립 (Z ⫫ X | t) 이라는 전제에서 성립하며, 이 전제가 깨지면 P(Z|t) 대신 P(Z|X,t) 로 적분해야 한다. Chamber 상태가 주어졌을 때의 관계 P(Y|X,Z) 는 시간에 따라 바뀌지 않아도, Z 가 측정 데이터 X 에 들어 있지 않으므로 model 은 P(Z|t) 의 변화를 P(Y|X) 의 변화로만 본다. P(Z|t) 가 PM 시점에 계단처럼 바뀌면 P(Y|X) shift 가 되고, 시간에 따라 서서히 옮겨 가면 P(Y|X) drift 가 된다. 관계 변화에는 대응과 관측의 두 가지 길이 있다.

- **대응**: drift 에는 detrending, 최신성 sample 가중, 최근 window 재학습을 쓰고, shift 에는 PM 같은 사건 뒤의 데이터로 다시 학습한다. 효과는 temporal CV 로 시간순으로 검증하며, 변동 시점은 특정하지 못한다.
- **관측**: 변화를 측정하고 시점을 특정한다.

관측 방법은 P(Y|X) 를 얼마나 직접 보는지로 갈린다.

- **Binning CDT**: X 를 bin 으로 나눠 P(Y|bin) 을 시간창별로 검정하여 변동 시점을 특정한다. P(Y|X) 를 가장 직접 본다.
- **잔차 CUSUM, Page-Hinkley**: 예측 잔차 통계량이 임계값을 넘는 시점을 출력한다.
- **I(X;Y)**: 의존성 총량 (거시 지표). 단독으로 쓰면 P(X), P(Y), 관계의 변화가 함께 잡히므로 시간창별로 추적해야 관계 변화에 가까워진다. 고차원 X 에서는 시간창마다 정확히 추정하기 어려우므로, 중요도 상위 K 개 변수 X<sub>k</sub> 에 대한 I(X<sub>k</sub>;Y) 로 좁혀 추적한다.

## References

<a id="ref-1"></a>
[1] Kang, S., & Kang, P. (2017). [An intelligent virtual metrology system with adaptive update for semiconductor manufacturing](https://doi.org/10.1016/j.jprocont.2017.02.002). *Journal of Process Control*, 52, 66–74.<br>
<a id="ref-2"></a>
[2] Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (Eds.). (2009). [Dataset Shift in Machine Learning](https://mitpressbookstore.mit.edu/book/9780262170055). MIT Press. ISBN 978-0-262-17005-5.<br>
<a id="ref-3"></a>
[3] Moreno-Torres, J. G., Raeder, T., Alaiz-Rodríguez, R., Chawla, N. V., & Herrera, F. (2012). [A unifying view on dataset shift in classification](https://doi.org/10.1016/j.patcog.2011.06.019). *Pattern Recognition*, 45(1), 521–530.<br>
<a id="ref-4"></a>
[4] Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). [A survey on concept drift adaptation](https://doi.org/10.1145/2523813). *ACM Computing Surveys*, 46(4), 44.<br>
<a id="ref-5"></a>
[5] Amazon Web Services. [Data and model quality monitoring with Amazon SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html). *Amazon SageMaker AI Developer Guide*.<br>
<a id="ref-6"></a>
[6] Google Cloud. [Introduction to Vertex AI Model Monitoring](https://docs.cloud.google.com/vertex-ai/docs/model-monitoring/overview). *Vertex AI documentation*.<br>
<a id="ref-7"></a>
[7] Evidently AI. [Concept drift in ML](https://www.evidentlyai.com/ml-in-production/concept-drift). *ML in Production guide*.

---

## Appendix A. Terminology

- **aleatoric uncertainty**: 데이터 자체가 지닌 산포에서 오는 불확실성. 데이터를 늘려도 줄지 않는다.
- **covariate**: model 의 입력 변수 X. 확률변수 하나를 가리키는 일반적인 말인 variate 는 X 와 Y 모두에 쓰이고, "co-" 는 주된 관심 변수 Y 와 함께 변하는 (co-vary) 변수, 곧 Y 를 설명하려고 함께 관측하는 변수라는 뜻이다. 이 문서에서는 장비 sensor 등에서 측정한 데이터이며, 문헌은 그 분포 P(X) 가 학습과 추론 사이에 달라지는 것을 covariate shift 라 부른다.
- **CUSUM (Cumulative Sum)**: 기준값과의 편차를 누적하여 임계값을 넘는 시점을 변화점으로 보는 관리도.
- **drift**: 분포나 관계가 시간에 따라 서서히 바뀌는 변화. 시간창마다 통계량을 추적하여 찾는다.
- **epistemic uncertainty**: 학습 데이터가 부족하여 model 이 지니는 불확실성. 데이터를 늘리면 줄어든다.
- **EWMA (Exponentially Weighted Moving Average)**: 최근 값에 더 큰 가중치를 주는 이동평균으로 작고 꾸준한 이동을 잡는 관리도.
- **I(X;Y)**: 상호정보량 (mutual information). X 와 Y 의 의존성 총량을 나타내는 거시 지표.
- **i.i.d. (independent and identically distributed)**: 각 표본이 서로 독립이고 같은 분포에서 나온다는 가정.
- **label noise**: 계측 오차처럼 정답값 Y 에 섞인 잡음.
- **latent variable**: 결과에 영향을 주지만 직접 관측되지 않는 변수. Chamber 의 노화나 찌꺼기 누적 상태가 그 예다.
- **marginal distribution (주변분포)**: 결합분포 P(X,Y) 에서 X 를 적분하여 없애고 Y 하나만 남긴 분포. P(Y) = ∫ P(X,Y) dX 이며, X 의 값과 상관없이 Y 가 어떻게 분포하는지를 나타낸다.
- **regression dilution**: 입력 X 에 측정 오차가 있을 때 추정한 회귀 기울기가 0 쪽으로 줄어드는 현상.
- **selection bias**: 표본을 고르는 방식 때문에 표본의 분포가 모집단의 분포와 달라지는 치우침.
- **shift**: 분포나 관계가 한 번에 바뀌는 변화. 학습 데이터와 추론 데이터의 두 집합을 비교하여 찾는다.
- **spatial decomposition**: wafer 측정 map 을 공간 기저 (다항식) 로 분해하고, 그 계수 (a1, …) 를 예측하는 방법.
- **target engineering**: 예측 대상 Y 를 변환·분해·재구성하여 model 이 학습하기 쉬운 형태로 바꾸는 일.
- **temporal CV**: 과거로 학습하고 미래로 검증하는 시간순 교차검증.

## Appendix B. Detection and Implementation by Cell

Table 1 의 세 요소마다 shift 와 drift 를 무엇으로 탐지하고 (Detection), 어떤 model·기법으로 대응하며 어떻게 검증하는지 (Response) 를 B.1 부터 B.3 에 모으고, 식 (5) 의 추정 오차를 B.4 에 둔다. Shift 는 두 집합의 비교로, drift 는 시간창별 추적으로 찾는다.

### B.1 P(X) Shift and Drift

단변량 방법은 변수마다 분포를 따로 비교하므로 변수 사이의 상관 변화를 잡지 못한다. 수백 개 sensor 변수를 쓰는 제조 데이터에서는 다변량 방법을 함께 쓴다.

#### Shift Detection

- **PSI (Population Stability Index)**: 변수마다 학습 데이터와 추론 데이터의 구간별 비율을 비교하여 P(X) 의 이동 크기를 잰다.
- **KS test (Kolmogorov–Smirnov test)**: 변수마다 두 데이터의 누적분포 최대 차이로 분포가 같은지 검정한다.
- **KL divergence (Kullback–Leibler divergence)**: 학습 데이터 분포에 대한 추론 데이터 분포의 차이를 정보량으로 잰다.
- **MMD (Maximum Mean Discrepancy)**: kernel 공간에서 두 데이터의 평균 차이로 다변량 분포 차이를 검정한다.
- **Domain classifier**: 학습 데이터와 추론 데이터를 가르는 classifier 를 학습하여, AUC 가 0.5 보다 클수록 두 분포가 다르다고 본다.

#### Drift Detection

- **Hotelling T²·SPE control chart (PCA-based)**: 학습 데이터로 만든 PCA model 에서 추론 데이터의 T² 와 잔차 SPE 를 시간순으로 그려 관리 한계를 넘는 시점을 찾는다.
- **Autoencoder reconstruction error**: 학습 데이터로 만든 autoencoder 에서 추론 데이터의 재구성 오차를 시간순으로 추적하여 관리 한계를 넘는 시점을 찾는다.
- **Windowed PSI**: 시간창마다 PSI 를 계산하여 추론 데이터가 학습 데이터에서 멀어지는 추세를 본다.

#### Response

- **Methods**: shift 에는 domain classifier 로 추정한 밀도비로 학습 sample 에 가중치를 주는 importance weighting, 학습·추론 입력 분포를 맞추는 domain adaptation, 학습 데이터가 추론 범위를 덮도록 계측 sampling 계획을 조정. Drift 에는 sensor 교정과 최근 window 재학습.
- **Validation**: 추론 데이터와 닮은 학습 sample 로 검증 set 을 꾸리는 adversarial validation.

### B.2 P(Y) Shift and Drift

#### Shift Detection

- **Metrology value distribution comparison**: 학습 데이터의 계측값과 최근 계측값의 분포를 KS test 나 PSI 로 비교한다.
- **Shewhart control chart**: 계측값의 평균과 산포가 관리 한계를 한 번에 벗어나는 시점을 감시한다.

#### Drift Detection

- **EWMA control chart**: 계측값의 지수가중 이동평균으로 작고 꾸준한 이동을 감시한다.
- **Prediction distribution monitoring**: 계측값이 늦게 들어올 때 P(Y)<sub>pred</sub> 의 분포 이동을 먼저 감시하여 P(Y) 변화를 미리 알린다.

#### Response

- **Methods**: shift 에는 target spec 변경에 맞춘 target 재정의와 재학습. Drift 에는 group 별 scale 정규화와 target engineering (log·Box-Cox 변환, spatial decomposition).
- **Validation**: spec·group 별로 나누어 오차를 따로 평가.

### B.3 P(Y|X) Shift and Drift

#### Shift Detection

- **Binning CDT (Conditional Distribution Test)**: X 를 bin 으로 나눠 bin 마다 P(Y|bin) 을 시간창별로 검정하여 변동 시점을 특정한다.
- **Event-split comparison**: PM 이나 부품 교체 앞뒤로 예측 잔차의 분포를 나누어 비교한다.

#### Drift Detection

- **Residual CUSUM**: 예측 잔차의 누적합이 임계값을 넘는 시점을 관계 변화 시점으로 본다.
- **Page-Hinkley**: 예측 잔차의 누적 편차와 그 최솟값의 차이가 임계값을 넘으면 평균 변화를 알린다.
- **ADWIN (Adaptive Windowing)**: 오차 window 를 두 부분으로 나눠 평균 차이가 유의하면 오래된 부분을 버리고 변화를 알린다.
- **Windowed performance monitoring**: 시간창마다 R²·RMSE 를 계산하여 성능 저하 시점을 찾는다. 계측값이 있어야 한다.
- **Windowed I(X<sub>k</sub>;Y)**: 시간창마다 중요도 상위 K 개 변수 X<sub>k</sub> 와 계측값의 상호정보량을 계산하여 의존성 변화를 추적한다. 고차원 X 전체의 I(X;Y) 는 표본 수에 비해 차원이 커서 시간창마다 정확히 추정하기 어렵다. I(X<sub>k</sub>;Y) 는 P(X) 변화만으로도 바뀔 수 있으므로, B.1 의 탐지 결과와 함께 해석한다.

#### Response

- **Methods**: shift 에는 사건 뒤의 데이터로 다시 학습하고, chamber 상태 Z(t) 의 대리 변수 (PM 이후 경과 시간, RF 누적 시간) 를 feature 로 추가. Drift 에는 최근 window 로 일정 간격 재학습, 최신성 가중, 신뢰도가 낮은 wafer 만 계측해 즉시 갱신하는 adaptive update [[1](#ref-1)].
- **Validation**: 과거로 학습하고 미래로 검증하는 temporal CV, lot 단위로 나눈 group split.

### B.4 Estimation Error

B.3 은 학습 뒤 실제 관계 P(Y|X) 가 바뀌어 생기는 식 (5) 의 둘째 항을 다루고, B.4 는 실제 관계가 그대로여도 model 이 그 관계를 추정하며 생기는 첫 항을 다룬다. 첫 항은 추정기 선택, 과적합, hyperparameter 처럼 학습 시점에 정해진다.

#### Detection

- **Train–validation gap**: 학습 성능과 검증 성능의 차이로 과적합을 본다. 시간에 따른 성능 저하는 B.3 의 Windowed performance monitoring 으로 본다.
- **Out-of-range check**: 추론 데이터가 학습 데이터의 범위 밖에 있는지 보아, 식 (5) 의 Extrapolation 몫이 커지는 wafer 를 가린다.

#### Response

- **Methods**: 표본이 변수보다 적으면 PLS·ridge·lasso 같은 정규화 선형 model, 비선형이면 LightGBM·XGBoost·CatBoost 같은 tree ensemble, 불확실성이 필요하면 Gaussian process 나 quantile regression·conformal prediction. 물리 지식은 monotone constraint 나 물리식 위에 잔차만 학습하는 hybrid 로 넣고, hyperparameter 는 Optuna 같은 Bayesian 최적화로 찾는다.
- **Validation**: lot 단위 group split 과 temporal CV 를 함께 쓰고, R²·RMSE 와 예측 구간의 coverage 를 본다.

## Appendix C. Benchmarking

Table 4 는 이 문서의 분류가 학계의 dataset shift 이론과 업계의 model monitoring 도구에서 쓰이는 사례를 모은다.

Table 4. Use of the shift taxonomy in research and industry

| Source                                       | Kind             | Terms used                                              | Term in this document                                |
| :------------------------------------------: | :--------------: | :-----------------------------------------------------: | :--------------------------------------------------: |
| Quiñonero-Candela et al. [[2](#ref-2)]       | Book             | dataset shift, covariate shift                          | 학습·추론 사이의 P(X,Y) 차이, P(X) shift             |
| Moreno-Torres et al. [[3](#ref-3)]           | Paper            | covariate shift, prior probability shift, concept shift | P(X) shift, P(Y) shift, P(Y\|X) shift                |
| Gama et al. [[4](#ref-4)]                    | Survey           | concept drift detection, adaptation                     | 5.3 의 P(Y\|X) drift 관측과 대응                     |
| Kang & Kang [[1](#ref-1)]                    | Paper            | virtual metrology adaptive update                       | 제조 데이터의 P(Y\|X) drift 대응                     |
| Amazon SageMaker Model Monitor [[5](#ref-5)] | Industry tool    | data quality drift, model quality drift                 | P(X) drift, 좋은 예측의 성능 저하                    |
| Vertex AI Model Monitoring [[6](#ref-6)]     | Industry tool    | training-serving skew, inference drift                  | P(X) shift, P(X) drift                               |
| Evidently AI [[7](#ref-7)]                   | Open-source tool | data drift, prediction drift, concept drift             | P(X) drift, P(Y)<sub>pred</sub> drift, P(Y\|X) drift |

- **Research**: dataset shift 는 학습과 추론 사이에 결합분포 P(X,Y) 가 달라지는 문제로 정의된다 [[2](#ref-2)]. Moreno-Torres et al. 은 결합분포의 어느 항이 바뀌는지로 covariate shift (P(X) 이동, P(Y|X) 유지), prior probability shift (P(Y) 이동, P(X|Y) 유지), concept shift 를 정리하였고 [[3](#ref-3)], 이 문서의 식 (3) 과 세 요소의 shift 가 이를 따른다.
- **Manufacturing data**: 반도체 virtual metrology 에서는 wafer 특성이 시간에 따라 바뀌어 예측 성능이 떨어지므로, 신뢰도가 낮은 wafer 만 계측하고 그 결과로 model 을 즉시 갱신하는 adaptive update 가 제안되었다 [[1](#ref-1)]. Concept drift 의 탐지와 적응 방법은 Gama et al. 이 정리하였다 [[4](#ref-4)].
- **Industry tools**: model monitoring 도구는 정답값 없이 볼 수 있는 P(X) 변화를 data quality drift [[5](#ref-5)], training-serving skew·inference drift [[6](#ref-6)], data drift [[7](#ref-7)] 라는 이름으로 감시한다. 정답값이 들어온 뒤에는 예측과 정답의 차이로 model quality drift [[5](#ref-5)] 나 concept drift [[7](#ref-7)] 를 확인한다.
- **Framework of this document**: 세 요소의 변화 분류는 학계와 업계에서 쓰는 표준 개념이다. 각 요소를 shift 와 drift 의 3×2 칸으로 나누어 중립 이름으로 부르고, 식 (5) 로 예측 오차에 잇는 틀은 이 문서가 정리한 것이며, 위 출처들이 이름 붙여 쓰는 표준 framework 는 아니다.

## Appendix D. Prior in Bayes' Theorem

베이즈 정리 (Bayes' theorem) 를 이 문서의 변수로 적으면 식 (7) 이다. Model 이 추론하는 미지의 양은 계측값 Y 이고, 측정 데이터 X 는 그 추론에 쓰는 증거다.

```math
P(Y \mid X) = \frac{P(X \mid Y) \cdot P(Y)}{P(X)} \hspace{19em} (7)
```

각 항의 뜻은 다음과 같다.

- **Prior**: P(Y). 사전 확률이며, 측정 데이터 X 를 보기 전에 계측값 Y 에 대해 가지고 있던 분포다.
- **Likelihood**: P(X|Y). 가능도이며, 계측값 Y 가 주어졌다는 가정 아래 측정 데이터 X 가 나타날 확률이다.
- **Posterior**: P(Y|X). 사후 확률이며, 측정 데이터 X 를 관측한 뒤 갱신된 Y 의 분포로, model 이 추정하려는 조건부 분포다.
- **Evidence**: P(X). 측정 데이터의 marginal distribution 이며, posterior 를 확률분포로 맞추는 정규화 상수다.

Prior 는 추론하려는 미지의 양인 계측값 Y 에 붙으므로, 문헌의 prior shift 는 이 문서의 P(Y) shift 다. 식 (7) 의 분자 P(X|Y)·P(Y) 는 식 (3) 의 우변과 같으므로, 식 (3) 에서 P(X|Y) 를 그대로 둔 채 P(Y) 만 바뀌는 P(Y) shift 는 이 prior 가 바뀌는 경우다. Label shift 는 같은 P(Y) 변화를 label Y 쪽에서 부르는 이름이다. 두 이름은 Y 가 X 를 만드는 Y → X 구조의 분류 문제에서 왔다. 증상 X 를 보고 그 원인인 병 Y 를 맞히는 문제처럼, 결과를 보고 원인 class 를 고르는 분류 문제를 가리킨다. 제조 공정은 반대로 공정 데이터 X 가 원인이고 계측값 Y 가 결과인 X → Y 구조여서, 이 이름이 그대로 들어맞지 않는다. 그래서 제조 공정에서 관측되는 P(Y) 변화는 대부분 P(X) 나 P(Y|X) 변화의 결과로 나타난다 (section 5.2).

## Appendix E. Talk Slides

이 문서를 발표할 때 쓰는 slide 는 세 장이며, 원본은 [modeling-elements-invited-talk.pptx](talk-slides/modeling-elements-invited-talk.pptx) 이다.

첫 slide 는 section 3 의 taxonomy 를 보인다 ([Fig 2](#fig-2)).

<img src="talk-slides/modeling-elements-invited-talk-1.png" width="800" style="max-width: 100%;" alt="Fig 2">

<a id="fig-2"></a>
Fig 2. Talk slide 1, taxonomy from the joint distribution

결합분포를 P(X), P(Y|X), P(Y) 로 나누고 식 (1) 과 식 (2) 로 세 요소를 이으며, 세 요소가 shift 와 drift 의 두 방식으로 바뀐다는 것을 함께 보인다.

둘째 slide 는 section 4 의 식 (4) 를 보인다 ([Fig 3](#fig-3)).

<img src="talk-slides/modeling-elements-invited-talk-2.png" width="800" style="max-width: 100%;" alt="Fig 3">

<a id="fig-3"></a>
Fig 3. Talk slide 2, prediction from the joint distribution

식 (4) 의 세 항을 좋은 데이터, 좋은 모델, 좋은 예측으로 읽고, 각 요소를 깨는 P(X) shift, P(Y|X) drift, P(Y) shift 를 schematic 으로 보인다. Slide 아래쪽에는 식 (5) 의 두 오차 항을 적었다.

셋째 slide 는 [Appendix B](#appendix-b-detection-and-implementation-by-cell) 를 줄여 보인다 ([Fig 4](#fig-4)).

<img src="talk-slides/modeling-elements-invited-talk-3.png" width="800" style="max-width: 100%;" alt="Fig 4">

<a id="fig-4"></a>
Fig 4. Talk slide 3, detection, response and validation by cell

세 요소의 shift 와 drift, 그리고 추정 오차마다 탐지, 대응, 검증 방법을 한 칸에 모았다.
