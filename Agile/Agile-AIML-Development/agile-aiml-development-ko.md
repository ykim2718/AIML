# Agile AI/ML Development
Rev. 0 | Created: 2026-09-13 | Updated: 2026-09-13 13:20 CDT

## 1. Purpose

- **Problem Statement**: Software 팀의 sprint 규칙 아래에서 AI/ML 과제는 완료를 선언하지 못한다. 같은 code 가 같은 결과를 돌려주지 않고, version 관리 대상이 code 하나가 아니며, 배포한 model 은 아무것도 손대지 않아도 성능을 잃기 때문이다.
- **Goal**: AI/ML 주기가 software 주기와 갈라지는 세 지점을 고정하고, 탐색을 한 sprint 안에 가두는 규칙들과 그 sprint 를 닫는 기준을 팀이 그대로 돌려 볼 수 있는 점검 항목으로 준다.
- **Non-Goal**: 특정 MLOps 도구 (MLflow, W&B, DVC) 의 설치와 설정은 다루지 않는다.

## 2. Summary

AI/ML 과제에서의 애자일은 가장 단순한 baseline model 을 만들어 1~2주 안에 운영 환경까지 end-to-end 로 붙여 본 뒤, 그 뒤에 들어오는 데이터와 실험으로 조금씩 다듬어 가는 것이다. 그 loop 를 돌리는 실천을 MLOps (Machine Learning Operations) 또는 agile for AI 라 부른다.

철학은 어느 agile 팀이나 공유하는 그것이다. 짧은 반복과 빠른 피드백이다. 여기서 그것을 변형시키는 것은 일의 탐색적 성격 (exploratory nature) 이며, sprint 의 결과가 명세대로 만들어지는 것이 아니라 밝혀지는 것이라는 점이다. 이 문서의 나머지가 그 변형이다. version 관리할 축이 둘 더 늘고, 실험에 시계가 붙고, 완료의 기준이 지표와 재현성까지 묻고, 배포가 일을 닫는 대신 감시 기간을 여는 것이 그것이다.

## 3. Taxonomy and its Hierarchy

AI/ML 과제에서 관리되는 것은 하나가 아니라 셋이며, 그 셋에는 순서가 있다. Code 는 적힌 대로 움직이고, 데이터는 도착한 대로 움직이며, model 은 그 둘이 함께 허락하는 만큼 움직인다. 그 순서를 한 단계 나아갈 때마다 일반성을 얻고 결정성을 내려놓으므로, 각 단계가 받는 증명은 앞 단계보다 종류가 약하고 얻는 값이 비싸다.

세 축과 각 축이 답하는 질문, 그리고 각 축이 받는 증명은 [Fig 1](#fig-1) 에 그렸다.

```text
Code    (deterministic)     "Does it run as written?"
  |
  |   proved by   unit test, code review, schema and type check
  |
  +-- Data   (distributional)   "Is what arrived what we assumed?"
        |
        |   proved by   missing value, outlier and schema validation,
        |               labelling cross-check, dataset hash
        |
        +-- Model   (probabilistic)   "Is it good enough, this time?"
              |
              |   proved by   a metric against a baseline, a bias test,
              |               the same result under a fixed seed

Determinism falls        Code  >  Data  >  Model
Cost of proving done     Code  <  Data  <  Model
```

<a id="fig-1"></a>
Fig 1. The three axes under version control and the proof each one takes

아래 세 가지 차이가 그 순서에서 나온다. 셋 모두 software 팀의 습관이 AI/ML 과제에서 틀린 답을 내놓는 자리다.

Table 1. Where an AI/ML cycle parts from a software cycle

| Difference | What it means | What it forces |
| --- | --- | --- |
| Non-determinism | 같은 code 와 같은 model 이라도 데이터 상태나 hyperparameter 가 달라지면 성능이 달라짐 | 연구 (R&D) 성격의 일로 다루고, 실험에 시계를 붙임 |
| Three axes | Software 는 code 를, AI/ML 은 code 와 데이터와 model 을 함께 version 관리 | 완료의 기준이 code 하나가 아니라 세 축 모두를 물음 |
| Drift | 현실의 데이터 경향이 학습 집합에서 멀어지면 잘 작동하던 model 의 성능이 떨어짐 | 배포가 일을 닫는 대신 감시 기간을 엶 |

## 4. Lifecycle

Software 의 code, build, 배포 흐름에 여기서는 두 단계 — 데이터 pipeline 과 실험 — 가 더해지고, 다섯이 모여 선이 아니라 순환을 이룬다. 그 순환은 재학습에서 닫히며, 그래서 마지막 단계가 첫 단계로 되돌아간다.

각 단계와 그 단계에서 만나는 용어는 [Fig 2](#fig-2) 에 그렸다.

```text
[ 1. Problem Framing ]
        |
        +--> Business KPI / Metric ... Success bar agreed (e.g. accuracy 95 % or better)
        +--> Feasibility Check ....... Data availability and technical reach judged
        |
        v
[ 2. Data Pipeline Sprint ]
        |
        +--> Ingestion / Labeling .... Collection and labelling, the longest step
        +--> Data Versioning (DVC) ... Dataset change history kept under a hash
        |
        v
[ 3. Exploratory Sprint ]   <-- the stage carrying the most uncertainty
        |
        +--> Baseline Model .......... Simplest rule or model, for a reference score
        +--> Experiment Tracking ..... Parameters, metrics and plots recorded per run
        +--> Spike / Timeboxing ...... "Closed in two days" agreed before it starts
        |
        v
[ 4. MLOps Pipeline ]
        |
        +--> Model Registry .......... Chosen model registered and versioned
        +--> Model Serving ........... Served as a REST API or on an edge device
        +--> Shadow / A/B Testing .... Compared behind the running system first
        |
        v
[ 5. Monitoring & Retraining ]
        |
        +--> Drift Detection ......... Data and concept drift caught as it appears
        +--> Continuous Training (CT)  Retraining pipeline running on its own
        +--> Retrospective ........... Findings carried into the next sprint
```

<a id="fig-2"></a>
Fig 2. The five stages of an AI/ML sprint and the terms met at each

일정이 깨진다면 깨지는 자리는 3단계다. 실험은 더 해 볼 것이 늘 하나 더 남아 있어 얼마나 오래 끌 수 있는지에 상한이 없으므로, 이 단계는 끝난 뒤에 판정하는 대신 시작하기 전에 한계를 받는다. 5단계는 software 과제라면 이미 끝났을 자리이며, 여기서는 loop 를 언제 다시 돌릴지를 정하는 단계다.

## 5. Practices

세 가지 규칙이 일의 탐색적인 쪽을 sprint 안에 가두며, 각각 탐색이 빠져나가는 다른 길목을 막는다.

### 5.1 Timeboxing And Spike

실험은 시작하기 전에 기한을 받는다. 지표의 마지막 1 % 를 쫓는 데 몇 주가 드는 일이 흔하기 때문이다. Timeboxing 은 그 한계를 ticket 의 일부로 적어 "이 실험은 무엇을 찾았든 2일 안에 끝낸다" 로 두고, spike ticket 은 연구 성격의 하위 과제를 배포 board 밖으로 아예 들어내어, 끝나지 않은 조사가 다 끝난 sprint 를 붙들고 있지 않게 한다.

### 5.2 Data-Centric AI

데이터를 고치는 쪽이 model 을 다시 짜는 쪽보다 지표를 더 멀리 옮기므로, 짧은 반복은 거기에 먼저 쓴다. 라벨링 오류를 바로잡고 noise 를 걷어내는 일이 architecture 를 바꾸는 일보다 점수를 더 미덥게 올리며, 그 향상은 다음 model 에서도 살아남는다. Architecture 변경은 그렇지 않다.

### 5.3 Shadow Deployment

새 model 은 고객에게 답하기 전에 배후에서 먼저 답한다. 들어온 요청은 기존 시스템이나 이전 model 이 받아 답하고, 새 model 은 같은 입력 위에서 나란히 제 답을 계산한다. 그 비교가 실제 환경에서의 새 model 을 재며, release 는 그 비교가 버틴 뒤에야 따라온다.

## 6. Definition Of Done

AI/ML ticket 의 완료는 데이터를 믿을 수 있고, model 이 제 지표를 넘겼고, 시스템이 충분히 빨리 답하고, 감시가 붙어 있다는 뜻이다. Software 의 기준은 그것의 부분 집합이지 그것의 가벼운 판이 아니다.

Table 2. The bar for done on each kind of project

| Kind | What it requires |
| --- | --- |
| Software DoD | 기능 개발 완료, test code 통과 |
| AI/ML DoD | 기능 개발 완료, 목표 지표 달성 (예: F1-score 0.88 초과), 데이터와 model 의 versioning, 추론 latency 의 한계 안 |

아래 열일곱 개의 점검 항목이 그 기준을 실제로 돌릴 수 있는 형태로 옮긴 것이며, 각 항목이 무엇을 지키는지에 따라 묶었다.

Table 3. The AI/ML definition of done, area by area

| Area | Check | What must be true |
| --- | --- | --- |
| Data & Feature | Data validation | 결측치, 이상치, schema 오류가 pipeline 에서 자동으로 걸림 |
| Data & Feature | Label review | 합의된 일관성 기준에 대한 라벨링 교차 검수 통과 |
| Data & Feature | Data versioning | 학습·검증·시험 집합이 versioning 되고 고유 hash 로 보관됨 |
| Data & Feature | Feature store entry | 새로 정제한 feature 가 공유 저장소에 등록되어 팀이 재사용 가능 |
| Model & Experimentation | Target metric | 합의된 지표가 baseline 또는 현재 운영 중인 model 보다 높음 |
| Model & Experimentation | Fairness and bias | 편향성 test 통과. 특정 그룹이나 class 에 치우친 예측 없음 |
| Model & Experimentation | Experiment record | Hyperparameter, dataset 버전, code commit, 지표, plot 이 자동 기록됨 |
| Model & Experimentation | Reproducibility | 같은 random seed 와 같은 parameter 에서 같은 결과 |
| Code & Testing | Code review | 최소 한 명의 peer engineer 가 승인한 pull request |
| Code & Testing | Unit and integration tests | Pipeline 과 pre/post-processing code 가 합의된 coverage 를 채움 |
| Code & Testing | Model registry entry | 검증된 model 이 staging 또는 candidate 태그로 versioning 됨 |
| Serving & MLOps | Inference performance | Serving SLA 충족. 예를 들어 P95 latency 100 ms 미만 |
| Serving & MLOps | Serving API test | Container 와 REST 또는 gRPC endpoint 가 통합 test 통과 |
| Serving & MLOps | Shadow and A/B readiness | Traffic 의 일부를 받거나 운영 시스템 뒤에서 도는 배포 경로 |
| Serving & MLOps | Monitoring hookup | 추론 데이터 저장, drift 감지, 인프라 metric 이 dashboard 에 붙음 |
| Documentation | Model card | 개요, 입출력 format, 한계, 측정된 성능, 사용 dataset 이 갱신됨 |
| Documentation | Failure rule | Low confidence 결과에 대한 fallback 이 적혀 있음 |

### 6.1 Operating The Checklist

열일곱 개를 모든 ticket 에 다 대면 sprint 가 느려지므로, 목록은 그 ticket 이 속한 sprint 의 종류로 나눈다. 데이터 sprint 는 첫 묶음에, 연구 sprint 는 둘째 묶음에, serving sprint 는 넷째 묶음에 답한다. 그러면 ticket 은 자기가 실제로 바꾼 것을 지키는 항목만 만난다.

기계가 돌릴 수 있는 항목은 기계가 돌린다. 데이터 검증, code test, 실험 기록, latency 측정은 모두 CI/CD 와 지속적 학습 pipeline 안에 두어, sprint 끝에 사람이 기억해 내는 대신 merge 마다 통과 여부가 판정되게 한다.

---

## Appendix A. Terminology

- **A/B testing**: 살아 있는 traffic 을 나누어 두 판을 서로 다른 몫에 내보내고 결과를 견주는 것.
- **Baseline model**: 가장 단순한 규칙이나 model. 뒤에 오는 모든 model 이 넘어야 할 기준 점수로 남긴다.
- **Concept drift**: 입력과 목표 사이의 관계가 바뀌어, 입력 분포가 움직이지 않았는데도 성능이 떨어지는 것.
- **CT (Continuous Training)**: 사람이 시작하지 않아도 model 을 재학습하고 다시 배포하는 pipeline.
- **Data drift**: 입력 데이터의 분포가 학습 집합에서 멀어지는 것.
- **Data versioning**: dataset 의 각 상태를 고유 hash 로 기록하여, 결과를 그것을 낸 데이터까지 되짚을 수 있게 하는 것.
- **Data-Centric AI**: 성능을 올리기 위해 model architecture 대신 데이터를 손보는 실천.
- **DoD (Definition of Done)**: 팀이 합의한 명시적 기준. 작업이 완료로 불리려면 이것을 넘어야 한다.
- **DVC (Data Version Control)**: code 저장소와 나란히 dataset 을 versioning 하는 도구.
- **Experiment tracking**: 실행마다 hyperparameter, dataset 버전, code commit, 결과를 자동으로 남기는 것.
- **Feature store**: 정제된 feature 의 공유 저장소. 한 팀의 feature 를 다른 팀이 다시 쓰게 한다.
- **Hyperparameter**: 학습 전에 사람이 고르는 설정값. 데이터로부터 학습되지 않는다.
- **MLOps (Machine Learning Operations)**: model 을 실험에서 운영까지 옮기고 거기에 머물게 하는 실천.
- **Model card**: model 의 개요, 입출력 format, 한계, 측정된 성능, 학습 데이터를 기록한 문서.
- **Model registry**: 학습된 model 을 버전과 단계 태그와 함께 보관하는 저장소.
- **Model serving**: 학습된 model 로 요청에 답하는 것. API 로 또는 기기 위에서 한다.
- **Reproducibility**: 같은 seed 와 parameter 가 같은 결과를 돌려주는 성질.
- **Shadow deployment**: 새 model 을 운영 중인 model 곁에서 같은 입력 위에 돌리되, 그 답이 사용자에게 닿지 않게 하는 것.
- **SLA (Service Level Agreement)**: service 가 지켜야 할 합의된 한계. latency 백분위 같은 것이 그것이다.
- **Spike**: 결과를 알 수 없어 배포 board 밖으로 들어낸, 따로 ticket 을 받은 조사.
- **Sprint**: Agile 주기의 한 반복. 보통 1~4주.
- **Timeboxing**: 어떤 일이 얼마나 오래 돌 수 있는지를 미리 정하고, 그 한계에서 무엇에 닿았든 닫는 것.
