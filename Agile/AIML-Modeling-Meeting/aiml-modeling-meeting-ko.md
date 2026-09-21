# AI/ML Modeling Meeting
Rev. 1 | Created: 2026-09-21 | Updated: 2026-09-21 10:29 CDT

## 1. Purpose

- **Problem Statement**: Modeling 회의를 일반적인 개발 회의처럼 진행하면 "한 번 해볼게요" 로 끝난다. 방이 주고받는 것에 이름이 없어, 그 가운데 무엇이 빠졌는지를 말할 방법이 없기 때문이다.
- **Goal**: 회의에 네 갈래로 묶은 열 개의 이름을 주어, 실무자가 지금 상 위에 무엇이 있고 무엇이 빠졌으며 빠진 것이 어느 검증 등급까지 가야 하는지를 한 문장으로 말할 수 있게 한다.
- **Non-Goal**: 실험 추적 도구 (MLflow, Weights & Biases) 의 설정과 ticket system 운영은 다루지 않는다.

## 2. Summary

AI/ML modeling 회의는 열 가지를 주고받으며, 그것에 이름을 붙이는 일이 회의를 결정 없이 끝나지 않게 한다. 열 개는 Premise, Claim, Product, Decision 네 갈래로 묶이고, 그 갈래의 차례가 곧 회의의 차례다.

각 갈래는 앞 갈래가 정해졌다고 가정한다. Premise 가 열린 채로 Product 를 논의하는 방은 아무도 정의하지 않은 양을 재고 있으며, modeling 회의를 결정 없이 끝내는 습관 세 가지는 각각 네 갈래 가운데 하나에서 빠진 이름이다.

## 3. Taxonomy and its Hierarchy

열 개의 이름은 무엇을 고정하는가로 분류되고, 무엇을 가정하는가로 순서가 매겨진다. Premise 는 이번 주기 이전에 참인 것을, Claim 은 주기가 주장하는 것을, Product 는 주기가 만들어 낸 것을, Decision 은 방을 떠나는 것을 고정한다.

네 갈래와 열 개의 이름, 그리고 수치가 어디까지 검증되었는지를 말하는 등급 사다리는 [Fig 1](#fig-1) 에 그렸다.

```text
CLASS             TERM          WHAT IT FIXES

Premise       >   Target        What counts as the answer: Y and its threshold
                  Provenance    Where the rows came from, and what was done to them
                  Baseline      The score to beat, with the run that produced it
    |   assumed by
    v
Claim         >   Hypothesis    One change, its physical reason, the movement expected
    |   tested by
    v
Product       >   Run           One tracked execution
                  Evidence      How far a Run has been checked, on the grade ladder
                  Insight       The explained cause of a metric move
                  Readiness     Latency, train/serve skew, fallback, monitoring
    |   judged into
    v
Decision      >   Verdict       Accepted, rework or stop, issued on the Hypothesis
                  Handoff       Owner, due date and ticket, issued for the next cycle

GRADE LADDER      E0 assertion < E1 number < E2 run-backed < E3 reproduced < E4 compared
```

<a id="fig-1"></a>
Fig 1. The ten terms of a modeling meeting, their four classes, and the evidence grade ladder

등급 사다리는 수치를 담은 모든 이름에 적용된다. E0 은 말로만 있어 받아들이지 않고, E1 은 뒤에 실행이 없는 값이며, E2 는 run id 와 parameter 와 데이터 버전을 함께 단 값이고, E3 은 그 E2 를 새 checkout 에서 다시 돌려 명시된 편차 안에 들어온 것이며, E4 는 그 E3 을 같은 split 위에서 Baseline 이나 운영 중인 model 과 나란히 놓은 것이다.

### 3.1 Placement

Table 1. The ten terms, and what each one takes to be settled

| Term       | Class    | Required grade                          | What carries it                                    |
| :--------: | :------: | :-------------------------------------: | :------------------------------------------------: |
| Target     | Premise  | E1. 첫 회의 이전에 문서로               | Ticket 의 한 줄. 그 양과 임계값                    |
| Provenance | Premise  | E2                                      | 추적 도구의 run 과 dataset 버전 hash               |
| Baseline   | Premise  | E2                                      | Baseline 으로 태그한 추적 도구의 run               |
| Hypothesis | Claim    | 수치가 아님. 바꾸는 하나와 그 근거      | 이번 주기로 연 ticket                              |
| Run        | Product  | E2                                      | Parameter, 데이터 버전, commit 을 단 추적 항목     |
| Evidence   | Product  | 등급 자체. 수치와 함께 소리 내어 말함   | 모든 지표 옆에 적는 등급                           |
| Insight    | Product  | E3                                      | Feature importance 또는 오차 분석. 그림으로 내보냄 |
| Readiness  | Product  | E4                                      | Model registry 항목과 monitoring dashboard         |
| Verdict    | Decision | Insight 가 E3, 승격이면 Readiness 가 E4 | 회의록의 한 줄                                     |
| Handoff    | Decision | 담당자와 날짜. 등급 없음                | Id 가 붙은 ticket                                  |

## 4. Terms

아래는 [Fig 1](#fig-1) 의 차례대로 갈래를 하나씩 짚는다. 실무자가 회의에서 필요한 순서가 그 순서이기 때문이다.

### 4.1 Premise

Premise 는 이번 주기의 수치가 뜻을 가지려면 이미 참이어야 하는 것이며, 논쟁이 아니라 확인의 대상이다. 세 이름은 Target, Provenance, Baseline 이다.

**Target** 은 Y 의 정의와 임계값을 한 줄로 적은 것이다. 수율 98 % 미만, 또는 EVT 기반 임계값을 넘는 센서 값은 target 이고 "AI 로 불량을 잡자" 는 target 이 아니다. 후자 위에서 연 주기는 팀이 정의한 적 없는 양을 잰다. 도메인 엔지니어가 대며, 회의 중이 아니라 첫 회의 이전에 고정한다.

**Provenance** 는 행이 어디서 왔고 거기에 무엇을 했는가이다. Split 규칙, 결측치와 이상치 처리, 누수 차단, dataset 버전 hash 가 그것이다. Sliding window 증강이 차단을 잃는 흔한 자리인데, 겹치는 window 가 split 을 가로질러 행을 나누어 갖기 때문이다. 누수는 열일곱 분야 294편의 논문에서 여덟 가지 형태로 기록되어 있어, 가끔이 아니라 상시 항목이다 [[3](#ref-3)].

**Baseline** 은 가장 단순한 model — 선형 회귀나 고전 통계 — 의 점수이며, 그렇게 태그한 추적 도구의 run 이 담는다. 첫 model 을 단순하게 두는 것은 확립된 출발점이고 [[5](#ref-5)], 태그가 없으면 뒤에 오는 모든 주장이 기대는 그 비교를 다시 찾지 못한다.

### 4.2 Claim

Claim 은 이번 주기가 검증하는 단 하나의 주장이며, 이름도 Hypothesis 하나다. 아직 써 보지 않은 algorithm 의 목록이 아니라 도메인 지식 위에 세우고, 세 부분으로 이루어진다. 바꾸는 하나, 그것의 물리적 근거, 그 근거가 서면 지표가 어떻게 움직이는가이다.

두 가지 예가 그 형식을 보여 준다. 센서 간 multicollinearity 가 심하므로 이번 실행은 Lasso 대신 Elastic Net 을 써서 그룹 효과를 담는다. Time warping 이 신호를 일그러뜨리므로 고정된 변환 대신 1D-CNN autoencoder 가 representation learning 으로 차원을 줄인다.

물리적 근거가 없는 Hypothesis 는 Insight 를 낳지 못한다. 결과가 확인하거나 뒤집을 대상이 없기 때문이다.

### 4.3 Product

Product 는 이번 주기가 실제로 만든 것이며, 네 이름은 서술이 아니라 등급을 받는다. Run, Evidence, Insight, Readiness 는 기억이 아니라 화면 위의 산출물에서 읽는다.

**Run** 은 parameter, dataset 버전, code commit, 지표를 달고 있는 하나의 추적된 실행이다. **Evidence** 는 그 Run 이 닿은 등급이며, 수치 옆에서 소리 내어 말한다. 그래야 방이 지금 듣는 것이 E1 인지 E3 인지 안다.

**Insight** 는 지표가 움직인 까닭을 설명한 것이다. "XGBoost 가 잘 나옵니다" 는 Insight 없는 Run 이고, "Feature importance 가 chamber 3 압력 센서를 맨 위에 두고, 그것을 빼면 점수가 Baseline 으로 돌아온다" 는 E3 의 Insight 다. Loss curve, confusion matrix, 축소된 차원의 latent space 를 화면에 올린다. 말로 서술된 결과는 E0 에 머문다.

**Readiness** 는 제 한계에 대고 잰 latency, train/serve skew, low confidence 결과의 fallback, monitoring 연동이다. 승격은 후보 model 을 이미 서빙 중인 model 과 견주는 일이며, 그 비교를 재는 척도는 판단이 아니라 구체적인 test 의 점검표다 [[2](#ref-2)].

### 4.4 Decision

Decision 은 방을 떠나는 것이며, 언제나 함께 발행되는 두 이름을 가진다. 둘 가운데 하나만 낸 회의는 그대로 대기열로 돌아간다.

**Verdict** 는 accepted, rework, stop 가운데 하나이며, Hypothesis 에 대고 소리 내어 말한다. Accepted 는 Insight 가 E3 이어야 하고, model 을 serving 으로 승격하려면 Readiness 가 E4 여야 한다. Rework 는 빠진 근거를 이름 붙이고, stop 은 그 이유를 적어 다음 팀이 읽을 자리에 남긴다.

**Handoff** 는 다음 주기의 담당자, 기한, ticket id 이며, Verdict 가 함의하는 engineering 작업도 함께 담는다. Code review 배정과 pipeline 연동을 여기서만 이름 붙여, modeling 논의가 일정 조율에 끊기지 않게 한다.

## 5. Agenda

Agenda 는 네 갈래를 순서대로 놓은 것이며 갈래마다 한 단계씩이고, 각 단계는 그 이름들을 한 문장으로 말할 수 있을 때 끝난다. 순서를 어기면 방은 [Fig 1](#fig-1) 에서 이미 지나온 갈래로 되돌아간다.

각 단계와 그 단계가 정하는 갈래, 그리고 상 위에 올리는 항목은 [Fig 2](#fig-2) 에 그렸다.

```text
[ 1. Premise Check ]        settles   Target, Provenance, Baseline
        |
        +--> Missing / Outlier ....... How NaN and outliers in the raw rows were handled
        +--> Leakage Check ........... Sliding window augmentation checked for leakage
        +--> Baseline Score .......... The simplest model's score, with its run id
        |
        v
[ 2. Claim Setting ]        settles   Hypothesis
        |
        +--> One Change .............. The single thing that moves in this cycle
        +--> Domain Reason ........... The physical ground the change rests on
        +--> Expected Effect ......... What the metric does if the reason holds
        |
        v
[ 3. Product Review ]       settles   Run, Evidence, Insight, Readiness
        |
        +--> Tracked Run ............. Metric read from the tracker, not from memory
        +--> Evidence Grade .......... How far the number has been checked, E0 to E4
        +--> Why It Moved ............ Feature importance or error analysis behind it
        +--> Shown On Screen ......... Loss curve, confusion matrix, latent space plot
        |
        v
[ 4. Decision ]             settles   Verdict, Handoff
        |
        +--> Verdict ................. Accepted, rework or stop, said aloud
        +--> Handoff ................. Owner, due date and ticket id
        +--> Engineering Sync ........ Code review and pipeline work named here
```

<a id="fig-2"></a>
Fig 2. The four stages of the meeting and the class each one settles

각 단계는 한 문장으로 닫히며, 아래 네 개의 문형이 실무자가 그 자리에서 읽어 단계를 닫는 문장이다. 그 문장을 채우지 못한 단계는 그 밖에 무엇을 논의했든 끝난 것이 아니다.

```text
Stage 1   "Target is <TARGET>. Provenance: split by <RULE>, data <HASH>.
           Baseline is <METRIC> = <VALUE>, run <RUN_ID>."

Stage 2   "We change <ONE_THING> because <PHYSICAL_REASON>.
           If it holds, <METRIC> moves <DIRECTION> by at least <AMOUNT>."

Stage 3   "<METRIC> moved <FROM> to <TO>, run <RUN_ID>, spread <SPREAD> over
           <N> seeds. Insight: <ANALYSIS>. Evidence grade E<GRADE>."

Stage 4   "Verdict <ACCEPTED|REWORK|STOP>. Handoff: <OWNER> runs <EXPERIMENT>
           by <DATE>, ticket <TICKET_ID>."
```

## 6. Anti-patterns

세 가지 습관이 modeling 회의를 Verdict 없이 끝내며, 각각은 아무도 대지 않은 [Fig 1](#fig-1) 의 이름 하나다. 빠진 이름을 부르는 편이 그 습관을 두고 논쟁하는 것보다 빠르다.

Table 2. What ends a meeting without a Verdict

| Habit                                 | Missing term        | What to bring instead                          |
| :-----------------------------------: | :-----------------: | :--------------------------------------------: |
| "일단 데이터 다 넣고 학습 돌려볼게요" | Hypothesis          | 실행 전에 말한, 물리적 근거가 붙은 하나의 변경 |
| "AI 로 불량 잡아봅시다"               | Target              | 회의를 열기 전에 문서로 적은 Y 와 그 임계값    |
| "대충 잘 나옵니다"                    | E0 에 머문 Evidence | E2 의 추적된 run 과 화면에 올린 분석 그림      |

## 7. Roles

세 역할이 갈래를 대며, 한 역할이 전부를 대지는 않는다. 한 역할이 빠진 회의는 그 역할이 지고 있던 갈래가 빠진 회의다.

Table 3. Which role supplies which class

| Role           | Supplies                                                    | Class             |
| :------------: | :---------------------------------------------------------: | :---------------: |
| Domain expert  | Target, Hypothesis 안의 물리적 근거, Insight 의 물리적 해석 | Premise, Claim    |
| Data scientist | Provenance, Baseline, Run, Evidence, Insight                | Premise, Product  |
| MLOps engineer | Readiness, 그리고 Handoff 안의 engineering 작업             | Product, Decision |

도메인 엔지니어가 두 센서는 대칭이라 topology 상 함께 묶여야 한다고 말하면, 데이터 과학자가 그 topology 를 model 로 옮기는 1D-CNN filter 크기로 답한다.

## 8. Record

회의록은 열 개 가운데 Verdict, Insight, Handoff 셋을 담으며, 그 셋을 채우지 못한 회의는 끝난 것이 아니다. 나머지 일곱은 추적 도구와 ticket 에 남고, 세 줄이 그것을 가리킨다.

```text
Verdict : <ACCEPTED|REWORK|STOP> on <HYPOTHESIS>
Insight : <what moved the metric, and what it was read from>
Handoff : <OWNER> runs <EXPERIMENT> by <DATE>, ticket <TICKET_ID>
```

한 주기를 채워 넣으면 세 줄은 아래와 같이 읽힌다. 각 줄이 양을 하나씩 대고 있어, 석 달 뒤의 독자도 무엇이 시도되었는지가 아니라 무엇이 확립되었는지를 가릴 수 있다.

```text
Verdict : ACCEPTED on "1D-CNN autoencoder reduces the trace to 200 dimensions"
Insight : Gas flow variation over the first 2,000 rows moves the final yield
          prediction most, by XGBoost feature importance. Reconstruction error 0.02
Handoff : <OWNER> runs Elastic Net and supervised 1D-CNN on the 200 compressed
          features, target 95 % yield classification accuracy, by 06-28, ticket <TICKET_ID>
```

같은 세 줄이 model 과 함께 나가는 model card 를 채운다. Model card 는 model 의 개요, 측정된 성능, 학습에 쓴 데이터를 기록한다 [[4](#ref-4)]. 이 workflow 를 agile 주기 안에 접어 넣은 팀도 그것을 여전히 자기 단계의 연속으로 돌린다 [[1](#ref-1)].

## References

<a id="ref-1"></a>
[1] Amershi, S., Begel, A., Bird, C., DeLine, R., Gall, H., Kamar, E., Nagappan, N., Nushi, B., & Zimmermann, T. (2019). [Software Engineering for Machine Learning: A Case Study](https://doi.org/10.1109/ICSE-SEIP.2019.00042). *2019 IEEE/ACM 41st International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP)*.<br>
<a id="ref-2"></a>
[2] Breck, E., Cai, S., Nielsen, E., Salib, M., & Sculley, D. (2017). [The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction](https://doi.org/10.1109/BigData.2017.8258038). *2017 IEEE International Conference on Big Data (Big Data)*.<br>
<a id="ref-3"></a>
[3] Kapoor, S., & Narayanan, A. (2023). [Leakage and the reproducibility crisis in machine-learning-based science](https://doi.org/10.1016/j.patter.2023.100804). *Patterns*, 4(9), 100804.<br>
<a id="ref-4"></a>
[4] Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). [Model Cards for Model Reporting](https://doi.org/10.1145/3287560.3287596). *Proceedings of the Conference on Fairness, Accountability, and Transparency (FAT\* '19)*.<br>
<a id="ref-5"></a>
[5] Zinkevich, M. [Rules of Machine Learning: Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml). Google for Developers.

---

## Appendix A. Terminology

- **1D-CNN autoencoder**: 1차원 신호 위의 convolutional network 로, 자기 입력을 다시 만들도록 학습한다. 여기서는 공정 trace 의 차원을 줄이는 데 쓴다.
- **Confusion matrix**: 예측 class 와 실제 class 를 교차시킨 표. 어느 class 를 어느 class 와 혼동하는지 읽는다.
- **Data leakage**: serving 시점에는 얻을 수 없는 정보가 model 에 닿는 것. offline 점수만 올리고 online 점수는 올리지 않는다.
- **Elastic Net**: L1 과 L2 norm 을 함께 쓰는 선형 model. 상관된 변수 가운데 하나만 고르지 않고 함께 남긴다.
- **EVT (Extreme Value Theory)**: 분포 꼬리의 통계. 여기서는 센서 값이 얼마나 극단인지로 임계값을 정하는 데 쓴다.
- **Feature importance**: 학습된 model 이 각 입력에 붙이는 점수. 어느 입력이 예측을 움직였는지 읽는다.
- **Latent space**: encoder 가 입력을 옮겨 놓은 축소된 좌표.
- **Loss curve**: 학습 단계에 대해 그린 학습 loss 와 검증 loss.
- **MLOps**: model 을 실험에서 운영으로 옮기고 거기에 머물게 하는 실천.
- **Model card**: model 의 개요, 측정된 성능, 학습 데이터를 기록한 문서.
- **Multicollinearity**: 입력 변수 사이의 거의 선형인 종속. 개별 계수를 불안정하게 만든다.
- **Representation learning**: feature 를 사람이 지정하는 대신 데이터에서 학습하는 것.
- **Sliding window augmentation**: 연속 기록에서 겹치는 window 를 잘라 학습 표본을 늘리는 것. window 끼리 행을 나누어 갖는다.
- **Time warping**: 시간 축의 일그러짐. 같은 공정의 기록 사이에서 신호를 밀거나 늘인다.
- **Train/serve skew**: 학습 때 적용한 preprocessing 과 serving 때 적용한 preprocessing 사이의 차이.
- **Yield**: 생산된 단위 가운데 규격을 만족하는 비율.
