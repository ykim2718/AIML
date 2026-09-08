# Medallion architecture in practice: six stages from raw source files to a model-ready dataset (Korean)
Rev. 2 | Created: 2026-09-08 | Updated: 2026-09-08 16:52 CDT

## 1. Overview

이 문서는 raw source file 에서 model-ready dataset 까지 데이터를 나르는 pipeline 을 데이터 성숙도 순으로 정리한다. 계층 구조는 업계 표준인 Databricks 의 Medallion architecture (Bronze → Silver → Gold) 이며, 이 pipeline 의 여섯 stage 가 그 세 layer 를 채운다. 각 stage 는 한 가지 책임만 지고 자기 앞의 stage 만 읽으므로, pipeline 이 재현 가능하고 추적 가능하게 유지된다.

Databricks 는 이 architecture 를 lakehouse 안의 데이터를 조직하는 data design pattern 으로 정의하며, 그 목적은 데이터가 세 layer 를 지나는 동안 구조와 품질을 점진적으로 끌어올리는 데 있다 [[1](#ref-1)]. layer 의 이름은 데이터가 놓인 자리가 아니라 데이터가 무엇이 되었는지를 가리킨다.

- Bronze 는 source system — RDBMS (Relational Database Management System), IoT (Internet of Things) 기기, log, API (Application Programming Interface) — 의 기록을 도착한 그대로 담는다.
- Silver 는 그 기록을 정제하고 결합하고 하나의 규격에 맞춘, source 와 소비자 사이의 중간 상태로 담는다.
- Gold 는 집계와 modeling 을 마친 결과를 그것을 읽는 쪽에 맞춰 담는다. BI (Business Intelligence) 보고를 위한 star schema, model 훈련을 위한 feature 표가 그것이다.

각 layer 는 자기가 담은 데이터에 대해 하나의 보증을 하며, 그 보증이 곧 그 layer 가 존재하는 이유이다.

```text
           BRONZE                         SILVER                          GOLD
  ┌───────────────────────┐      ┌───────────────────────┐      ┌───────────────────────┐
  │        keep it        │      │        make it        │      │        make it        │
  │      as it landed     │ ───▶ │      trustworthy      │ ───▶ │      model-ready      │
  └───────────────────────┘      └───────────────────────┘      └───────────────────────┘
      written once and              nulls, outliers and            features built and
      never edited, the             clocks resolved, so            reduced, and pinned
       only safety net                the data can be             to a version so that
     if a parse is wrong            queried with trust            train and serve agree
```

Fig 1. 각 Medallion layer 가 하는 보증

Bronze 는 기록이 도착한 그대로임을, Silver 는 값을 믿을 수 있음을, Gold 는 열이 model 이 그대로 쓰는 것임을 보증한다. 꼭지 2 는 pipeline 의 여섯 stage 를 이 layer 안에 배치하고, 꼭지 3 은 각 stage 를 차례로 다룬다.

## 2. Medallion Architecture Mapping

여섯 stage 는 데이터를 품질과 성숙도로 가르는 사실상의 표준인 Medallion architecture 의 세 layer 에 나뉘어 들어간다. Fig 2 는 layer 경계마다 그곳을 넘는 transform 의 이름을 붙여 경계를 그 일로 읽을 수 있게 하고, Clean 이 두 Silver 형태로 갈라졌다가 Feature 에서 다시 합쳐지는 것을 보인다.

```text
        BRONZE                               SILVER                       GOLD
                                                  ┌─────────────┐
                                              ┌─▶ │  Structured │─┐
  ┌───────────┬───────────┐    ┌───────────┐  │   └─────────────┘ │   ┌───────────┐
  │  Original │    Raw    │──▶ │   Clean   │ ─┤                   ├──▶│  Feature  │
  └───────────┴───────────┘    └───────────┘  │   ┌─────────────┐ │   └───────────┘
        └── parse ──┘            clean        └─▶ │ Transformed │─┘     features
                                                  └─────────────┘
```

Fig 2. 여섯 stage 와 그 사이의 transform, 그리고 그것들이 속한 layer

Table 1. Medallion layer 와 그 안에 담기는 stage

| Layer | Stages | State | Purpose |
| --- | --- | --- | --- |
| Bronze | Original + Raw | 도착한 그대로. 형식 불일치와 비정형 내용 포함 | 원본 기록 보존 |
| Silver | Clean + Structured + Transformed | 정제·정규화 후 model 입력 형태로 재배치하고 model 이 읽는 척도로 다시 표현 | 신뢰할 수 있고 조회 가능한 데이터 |
| Gold | Feature | 완전히 가공된 최고 성숙도 | model 에 그대로 투입 |

Structured Data 와 Transformed Data 는 과도기적이다. model 에 무관한 작업 — 단순 재배치, 표준 windowing, 표준 scaling — 은 여러 model 이 함께 쓸 수 있으므로 Silver 에 남고, 특정 model 에만 맞춘 재배치나 encoding 은 Gold 쪽으로 기운다. 여러 model 이 같은 산출물을 재사용한다면 Silver 에 고정하는 것이 낫다.

## 3. Pipeline Stages

### 3.1 Original Data (Bronze)

각 source 에서 도착한 그대로의 손대지 않은 file 이다. source 와 version 마다 고유한 형식 — CSV (Comma-Separated Values), JSON (JavaScript Object Notation), XML (Extensible Markup Language) — 과 고유한 열 이름, 고유한 header 관례를 가진다. 이 file 은 받은 그대로 보관하며 제자리에서 고치지 않는다. 원본 기록이자, 나중에 parsing 결함이 드러났을 때의 유일한 안전망이기 때문이다.

### 3.2 Raw Data (Bronze)

같은 데이터를 하나의 schema 로 맞춘 것이다. Original 을 parsing 하여 열 이름과 단위와 timestamp 를 표준화한다. 형태는 일관해졌지만 내용은 아직 거칠어서 null 과 이상치와 중복이 그대로 남아 있다. parsing 은 idempotent 하게 유지하여 Raw 를 언제든 Original 에서 다시 만들 수 있게 한다.

### 3.3 Clean Data (Silver)

믿을 수 있는 데이터이다. 결측값을 처리하고 잡음과 이상치를 제거하며 source 간 timestamp 를 정렬한다. 확신을 갖고 조회할 수 있는 첫 stage 이다. 한 가지 주의할 점이 있다. 일시적인 spike 와 실제 분포 변화 — dataset shift [[2](#ref-2)] — 는 통계적으로 비슷해 보일 수 있으므로, 제거 규칙은 도메인 검토를 거쳐 정해야 실제 신호를 버리지 않는다.

### 3.4 Structured Data (Silver)

같은 값을 model 의 입력 규격에 맞춰 재배치한 것이다. 이차원 (2D) 형태는 XGBoost (eXtreme Gradient Boosting) 같은 고전 model 을 위한 [samples, features] 표이다. 삼차원 (3D) tensor 형태는 Convolutional Neural Network (CNN) 이나 Long Short-Term Memory (LSTM) 같은 deep model 을 위해 시계열 window 를 적용하여 [samples, timesteps, features] 를 만든다. group key 를 함께 넘겨서 나중에 보지 않은 group 으로 model 을 검증할 수 있게 한다.

### 3.5 Transformed Data (Silver)

같은 값을 model 이 읽는 척도로 다시 표현한 것이다. 수치 열은 scaling 하고 범주 열은 encoding 하며 치우친 열은 단조 변환을 거친다. 표의 배치는 건드리지 않으며, 이것이 Structured Data 와 갈리는 지점이다. 한쪽은 값이 놓이는 방식을 바꾸고 다른 쪽은 값 자체를 바꾼다. 두 stage 는 서로를 읽지 않고 둘 다 Clean Data 를 읽으므로 어느 순서로 만들어도 된다. 여기서 적합하는 parameter — scaler 의 평균과 분산, encoder 의 범주 목록 — 는 훈련 행에서만 얻어 dataset 과 함께 저장한다. serving 시점에 다시 적합하는 것은 train/serve skew [[3](#ref-3)] 로 가는 알려진 길이기 때문이다.

### 3.6 Feature Data (Gold)

최적화된 dataset 이다. 도메인 지식이 읽어 들인 열을 model 이 학습하는 변수 — 이동 평균, 주파수 성분, embedding — 로 바꾸고 차원 축소를 함께 적용한다. feature 가 sample 보다 많아지면 ($p \gg n$) 차원 축소는 선택이 아니라 필수이다 [[4](#ref-4)]. feature 정의는 version 을 붙여 train/serve skew [[3](#ref-3)] 를 막는다.

## 4. Key Principles

pipeline 의 값어치는 여섯 개의 이름표가 아니라 그 뒤의 규율에 있다.

- Immutability — 각 layer 는 한 번 쓰고 제자리에서 고치지 않는다.
- Reproducibility — 각 transform 은 결정적이므로 같은 입력이 같은 출력을 낸다.
- Lineage — 모든 열은 그것을 만든 source 기록까지 거슬러 갈 수 있다.

이 셋이 함께 잘못된 예측을 진단 가능하게 만든다. 그 예측을 실어 나른 열을 다시 유도할 수 있고, 그 열 뒤의 기록을 열어 볼 수 있다.

## References

<a id="ref-1"></a>
[1] Databricks. [What is Medallion Architecture?](https://www.databricks.com/blog/what-is-medallion-architecture). Databricks.<br>
<a id="ref-2"></a>
[2] Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (Eds.) (2009). [*Dataset Shift in Machine Learning*](https://doi.org/10.7551/mitpress/9780262170055.001.0001). MIT Press. ISBN 978-0-262-17005-8.<br>
<a id="ref-3"></a>
[3] Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). [Hidden Technical Debt in Machine Learning Systems](https://papers.neurips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems). *Advances in Neural Information Processing Systems*, 28.<br>
<a id="ref-4"></a>
[4] Bühlmann, P., & van de Geer, S. (2011). [*Statistics for High-Dimensional Data: Methods, Theory and Applications*](https://doi.org/10.1007/978-3-642-20192-9). Springer.

---

## Appendix A. Terminology

- **Bronze**: 정제나 검증 없이 도착한 그대로의 데이터를 담는 Medallion layer.
- **dataset shift**: 훈련 시점과 serving 시점 사이에서 입력과 출력의 결합 분포가 달라지는 것.
- **Gold**: 완전히 가공되어 model 에 바로 쓸 수 있는 데이터를 담는 Medallion layer.
- **idempotent**: 여러 번 적용해도 한 번 적용한 것과 같은 결과를 내는 transform 의 성질.
- **lineage**: 한 열을 그 기원까지 잇는, 기록으로 남은 transform 의 사슬.
- **Medallion architecture**: data lakehouse 를 데이터 품질과 성숙도에 따라 Bronze, Silver, Gold 로 나눈 계층 구조.
- **Silver**: 정제·정규화된 뒤 model 을 위해 재배치되고 다시 표현된 데이터를 담는 Medallion layer.
- **star schema**: 측정값을 하나의 fact table 에 두고 그 설명 속성을 둘레의 dimension table 에 두는 표 배치.
- **train/serve skew**: 훈련 시점에 계산한 feature 값과 serving 시점에 계산한 값이 어긋나는 것.
