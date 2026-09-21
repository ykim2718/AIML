# Agile AI/ML Modeling Meeting
Rev. 13 | Created: 2026-09-21 | Updated: 2026-09-21 15:53 CDT

## 1. Purpose

- **Problem Statement**: Agile development 을 AI/ML 에 적용하는 예시가 모호하여, 실무자는 그 framework 이 modeling 팀에게 무엇을 요구하는지 손에 쥐지 못한다.
- **Goal**: Scrum Guide [[5](#ref-5)] 를 AI/ML modeling 단계별로 적용하여, 실무자에게 agile model development 를 돌리는 방법을 제시한다.
- **Non-Goal**: Process flow 를 제시하며, modeling 자체는 하지 않는다.

## 2. Summary

Agile AI/ML model development 는 sprint 마다 한 번 도는 네 단계로 진행되며, 한 단계는 그 단계가 확정하는 항목이 저마다 definition of done 을 통과했을 때에만 닫힌다. 확정하는 항목은 모두 아홉이고, sprint 경계의 회의가 마지막 단계로 이번 sprint 를 닫고 다음 sprint 를 여는 자리다.

네 단계는 Scrum Guide 의 event, artifact, commitment 를 대체하지 않고 그대로 나른다 [[5](#ref-5)]. Hypothesis 가 sprint backlog 항목이고, 항목마다의 done-when 이 Increment 에 붙은 commitment 이며, 승격할 수 있는 model 이 곧 Increment 다. Sprint 를 결정 없이 끝내는 습관은 각각 아무도 대지 않은 항목 하나다.

## 3. Taxonomy and its Hierarchy

한 단계는 앞 단계가 닫힌 뒤에야 열리며, 앞 단계가 열려 있는 동안 나온 metric 은 아무것도 판정하지 못한다. 아홉 항목은 확정되는 단계에 따라 묶이고, 네 단계는 그 차례로 돈다.

네 단계와 각 단계가 확정하는 항목, 그리고 각 항목이 고정하는 것은 [Fig 1](#fig-1) 에 그렸다.

```text
STAGE                    ITEM          WHAT IT FIXES

1  Premise Check    >    Target        What counts as the answer: Y and its threshold
                         Provenance    Where the rows came from, and what was done to them
                         Baseline      The score to beat, with the run that produced it
      |   every item done before
      v
2  Claim Setting    >    Hypothesis    One change, its physical reason, the movement expected
      |   every item done before
      v
3  Product Review   >    Run           One tracked execution
                         Insight       The explained cause of a metric move
                         Readiness     Latency, train/serve skew, fallback, monitoring
      |   every item done before
      v
4  Decision         >    Verdict       Accepted, rework or stop, issued on the Hypothesis
                         Handoff       Owner, due date and backlog item for the next sprint
```

<a id="fig-1"></a>
Fig 1. The nine items a modeling meeting settles, and the stage that settles each

항목을 닫는 것은 그 항목의 definition of done 이며, 그 기준은 아홉이 저마다 다르다.

### 3.1 Placement

<a id="table-1"></a>
Table 1. The nine items, the stage that settles each, and its definition of done

| Item       | Settled at       | Done when                                                           | What carries it                                    |
| :--------: | :--------------: | :-----------------------------------------------------------------: | :------------------------------------------------: |
| Target     | 1 Premise Check  | Sprint 을 열기 전에 합의된, 임계값이 붙은 하나의 양으로 적힘        | Backlog item 의 한 줄                              |
| Provenance | 1 Premise Check  | Split 규칙, 결측치와 이상치 처리, dataset hash 가 기록됨            | 추적 도구의 run 과 데이터 버전 도구                |
| Baseline   | 1 Premise Check  | Baseline 으로 태그된 run. 이후의 주장이 견주는 점수를 담음          | Baseline tag 가 붙은 추적 도구의 run               |
| Hypothesis | 2 Claim Setting  | 바꾸는 하나, 물리적 근거, 예상되는 움직임이 모두 말해짐             | Sprint backlog item                                |
| Run        | 3 Product Review | Parameter, dataset 버전, code commit, metric 이 모두 추적됨         | 추적 도구의 항목                                   |
| Insight    | 3 Product Review | Metric 을 움직인 원인이 새 checkout 에서 재현됨                     | Feature importance 또는 오차 분석. 그림으로 내보냄 |
| Readiness  | 3 Product Review | Latency, skew, fallback, monitoring 을 지금 서빙 중인 model 과 견줌 | Model registry 항목과 monitoring dashboard         |
| Verdict    | 4 Decision       | Product owner 가 Hypothesis 에 대고 소리 내어 말함                  | 회의록의 한 줄                                     |
| Handoff    | 4 Decision       | 다음 sprint 를 위한 담당자, 기한, item id 가 발행됨                 | Tracker 에서 id 가 붙은 backlog item               |

## 4. Items

아래는 [Fig 1](#fig-1) 의 차례대로 단계를 하나씩 짚는다. 실무자가 항목을 만나는 순서가 그 순서이기 때문이다.

### 4.1 Premise Check

Premise Check 는 이번 sprint 의 metric 이 뜻을 가지려면 이미 참이어야 하는 것을 확정하며, 세 항목은 논쟁이 아니라 확인의 대상이다. 셋이 함께 modeling backlog item 의 definition of ready 다. 그 가운데 하나가 확정되지 않은 채 연 sprint 는 아무것도 판정하지 못하는 metric 을 낸다.

**Target** 은 Y 의 정의와 임계값을 한 줄로 적은 것이다. 수율 98 % 미만, 또는 EVT 기반 임계값을 넘는 센서 값은 target 이고 "AI 로 불량을 잡자" 는 target 이 아니다. 후자 위에서 연 sprint 는 팀이 정의한 적 없는 양을 잰다. 도메인 엔지니어가 대며, 회의 중이 아니라 첫 회의 이전에 고정한다.

**Provenance** 는 행이 어디서 왔고 거기에 무엇을 했는가이다. Split 규칙, 결측치와 이상치 처리, 누수 차단, dataset 버전 hash 가 그것이다. Sliding window augmentation 이 차단을 잃는 흔한 자리인데, 겹치는 window 가 split 을 가로질러 행을 나누어 갖기 때문이다. 누수는 열일곱 분야 294편의 논문에서 여덟 가지 형태로 기록되어 있어, 가끔이 아니라 상시 항목이다 [[3](#ref-3)].

**Baseline** 은 가장 단순한 model — 선형 회귀나 고전 통계 — 의 점수이며, 그렇게 태그한 추적 도구의 run 이 담는다. 첫 model 을 단순하게 두는 것은 확립된 출발점이고 [[6](#ref-6)], 태그가 없으면 뒤에 오는 모든 주장이 기대는 그 비교를 다시 찾지 못한다.

### 4.2 Claim Setting

Claim Setting 은 이번 sprint 가 검증하는 단 하나의 주장을 확정하며, 항목은 Hypothesis 하나다. 아직 써 보지 않은 algorithm 의 목록이 아니라 도메인 지식 위에 세우고, 세 부분으로 이루어진다. 바꾸는 하나, 그것의 물리적 근거, 그 근거가 서면 metric 이 어떻게 움직이는가이다. Modeler 한 사람당 Hypothesis 하나가 이 sprint 의 WIP limit 이고, 실행 전에 고정한 timebox 를 달고 있으며, 세 부분을 말하지 못하는 연구 성격의 작업은 spike 로 board 밖에 둔다.

두 가지 예가 그 형식을 보여 준다. 센서 간 multicollinearity 가 심하므로 이번 실행은 Lasso 대신 Elastic Net 을 써서 그룹 효과를 담는다. Time warping 이 신호를 일그러뜨리므로 고정된 변환 대신 1D-CNN autoencoder 가 representation learning 으로 차원을 줄인다.

물리적 근거가 없는 Hypothesis 는 Insight 를 낳지 못한다. 결과가 확인하거나 뒤집을 대상이 없기 때문이다.

### 4.3 Product Review

Product Review 는 이번 sprint 가 만든 것을 확정하며, 세 항목은 기억이 아니라 화면 위의 산출물에서 읽는다. 셋이 저마다 다른 기준을 달고 있어, [Table 1](#table-1) 이 그것을 하나씩 적는다.

**Run** 은 parameter, dataset 버전, code commit, metric 을 달고 있는 하나의 추적된 실행이다. Run id 없이 말해진 metric 은 방이 되짚어 갈 수 없으므로 아무것도 닫지 못한다.

**Insight** 는 metric 이 움직인 까닭을 설명한 것이다. "XGBoost 가 잘 나옵니다" 는 Insight 없는 Run 이고, "Feature importance 가 chamber 3 압력 센서를 맨 위에 두고, 그것을 빼면 점수가 Baseline 으로 돌아온다" 는 Insight 이며, 새 checkout 이 그것을 재현하면 done 이다. Loss curve, confusion matrix, 축소된 차원의 latent space 를 화면에 올린다. 말로 서술된 결과는 방이 확인할 수 없기 때문이다.

**Readiness** 는 제 한계에 대고 잰 latency, train/serve skew, low confidence 결과의 fallback, monitoring 연동이다. 승격은 후보 model 을 이미 서빙 중인 model 과 견주는 일이며, 그 비교를 재는 척도는 판단이 아니라 구체적인 test 의 점검표다 [[2](#ref-2)].

### 4.4 Decision

Decision 은 방을 떠나는 것을 확정하며, 두 항목은 언제나 함께 발행된다. 둘 가운데 하나만 낸 회의는 그대로 대기열로 돌아간다.

**Verdict** 는 accepted, rework, stop 가운데 하나이며, product owner 가 Hypothesis 에 대고 소리 내어 말한다. Increment 를 받아들이는 일은 backlog 의 순서를 소유한 역할의 몫이기 때문이다. Accepted 는 재현된 Insight 를 요구하고, model 을 serving 으로 승격하려면 견주어진 Readiness 가 있어야 한다. Rework 는 제 기준에 못 미친 항목을 이름 붙이고, stop 은 그 이유를 적어 다음 팀이 읽을 자리에 남긴다.

**Handoff** 는 다음 sprint 의 담당자, 기한, item id 이며, Verdict 가 함의하는 engineering 작업도 함께 담는다. Handoff 가 다음 sprint backlog 항목이며, 할 일이 아니라 Hypothesis 의 형태로 적는다. 그것에 밀린 Hypothesis 들은 다음 Insight 가 무엇을 판정하는가의 순서로 product backlog 에 남는다. Code review 배정과 pipeline 연동을 여기서만 이름 붙여, modeling 논의가 일정 조율에 끊기지 않게 한다.

## 5. Agenda

Agenda 는 [Fig 1](#fig-1) 의 네 단계를 순서대로 놓은 것이며, 각 단계는 그 항목들을 한 문장으로 말할 수 있을 때 끝난다. 회의도 다른 ceremony 와 같이 timebox 를 받으며, 4단계는 보호한다. 앞 단계가 넘치면 Verdict 가 쓸 시간을 가져가는 대신 그 단계의 깊이를 줄인다.

아래 네 문장이 실무자가 각 단계를 닫으려고 읽는 문장이다. 그 문장을 채우지 못한 단계는 그 밖에 무엇을 논의했든 끝난 것이 아니다.

```text
Stage 1   "Target is <TARGET>. Provenance: split by <RULE>, data <HASH>.
           Baseline is <METRIC> = <VALUE>, run <RUN_ID>."

Stage 2   "We change <ONE_THING> because <PHYSICAL_REASON>.
           If it holds, <METRIC> moves <DIRECTION> by at least <AMOUNT>."

Stage 3   "<METRIC> moved <FROM> to <TO>, run <RUN_ID>, spread <SPREAD> over
           <N> seeds. Insight: <ANALYSIS>, reproduced from a clean checkout."

Stage 4   "Verdict <ACCEPTED|REWORK|STOP>. Handoff: <OWNER> runs <EXPERIMENT>
           by <DATE>, backlog item <ITEM_ID>."
```

회의 자체가 sprint 의 경계에 놓이므로, 4단계는 이번 sprint 를 닫으면서 다음 sprint 를 연다. 두 회의 사이에서 진행 중인 실험의 blocker 는 경계까지 쥐고 있지 않고 daily standup 에서 드러낸다.

## 6. Agile Practice

아래의 기법은 Scrum Guide 가 붙인 이름을 그대로 두고 담는 것만 바꾼다 [[5](#ref-5)]. 이 회의를 들이는 팀은 새 ceremony 가 아니라 이미 돌리는 event, artifact, commitment 에 용어를 더한다. Story point 와 velocity 는 표에서 뺐다. 실험이 얼마나 도는지는 돌려 보기 전에는 알 수 없기 때문이다.

Table 2. Where each agile practice lands in the modeling sprint

| Practice        | What it carries here                                                    | What changes for modeling                                                                           |
| :-------------: | :---------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------: |
| Sprint          | 검토 중인 Hypothesis. Premise Check 부터 Verdict 까지                   | 달력이 아니라 실험 하나를 재현하는 데 걸리는 시간으로 길이를 정함                                   |
| Product backlog | 아직 sprint 에 들지 않은 Hypothesis 를 순서대로                         | 전달한 가치가 아니라 다음 Insight 가 무엇을 판정하는가로 순서를 매김                                |
| Sprint backlog  | Hypothesis. Sprint 당 하나                                              | Backlog 항목이 만들 기능이 아니라 검증할 주장                                                       |
| WIP limit       | Modeler 한 사람당 진행 중인 Hypothesis 하나                             | 둘을 한꺼번에 바꾸면 Insight 의 귀속이 불가능해짐                                                   |
| Timeboxing      | 실험에 붙인 시계, 그리고 회의에 붙인 시계                               | 무엇을 찾았든 한계에서 실험을 닫음                                                                  |
| Spike           | 배포 board 밖으로 들어낸 연구 성격의 작업                               | 산출물이 model 이 아니라 결정                                                                       |
| Daily standup   | 진행 중인 실험의 blocker                                                | Sprint 경계가 아니라 생긴 날에 드러냄                                                               |
| Sprint review   | 3단계 Product Review                                                    | 시연 대상이 추적된 run 과 분석 그림                                                                 |
| Retrospective   | 4단계의 process finding                                                 | Model 에 대한 발견인 Insight 와 따로 기록                                                           |
| DoR             | 1단계가 done 인 상태. Target 이 적히고 Provenance 와 Baseline 이 추적됨 | Scrum Guide 밖의 용어. Guide 는 한 sprint 안에 Done 될 수 있는 항목을 ready for selection 이라 부름 |
| DoD             | [Table 1](#table-1) 의 done-when 열                                     | Scrum Guide 가 Increment 에 붙인 commitment. Sprint 전체에 하나로 걸지 않고 항목마다 적음           |
| Increment       | 지금 서빙 중인 model 과 견주어진 Readiness                              | Increment 는 승격할 수 있는 model 이거나 아무것도 아님                                              |
| BKM             | Stop Verdict 와 그 이유를 두는 자리                                     | 닫은 방향이 다음 팀이 읽는 지식이 됨                                                                |

두 기법이 회의를 닫을 수 있는지를 가른다. WIP limit 이 없으면 Hypothesis 에 Verdict 를 낼 수 없다. Sprint 가 하나보다 많이 움직여, 방이 지금 무엇을 판정하는지 말하지 못하기 때문이다. 항목마다 적은 done-when 이 없으면 metric 을 규칙이 아니라 논쟁으로 물리게 되고, 그 논쟁은 회의보다 오래간다.

## 7. Anti-patterns

세 가지 습관이 modeling 회의를 Verdict 없이 끝내며, 각각은 아무도 대지 않은 [Fig 1](#fig-1) 의 항목 하나다. 빠진 항목을 부르는 편이 그 습관을 두고 논쟁하는 것보다 빠르다.

Table 3. What ends a meeting without a Verdict

| Habit                                 | Missing item | What to bring instead                          |
| :-----------------------------------: | :----------: | :--------------------------------------------: |
| "일단 데이터 다 넣고 학습 돌려볼게요" | Hypothesis   | 실행 전에 말한, 물리적 근거가 붙은 하나의 변경 |
| "AI 로 불량 잡아봅시다"               | Target       | 회의를 열기 전에 문서로 적은 Y 와 그 임계값    |
| "대충 잘 나옵니다"                    | Run          | 추적된 run 과 화면에 올린 분석 그림            |

## 8. Roles

네 역할이 항목을 대며, 한 역할이 전부를 대지는 않는다. 한 역할이 빠진 회의는 그 역할이 지고 있던 항목이 빠진 회의다.

Table 4. Which role supplies which items

| Role           | Supplies                                                    | Stage   |
| :------------: | :---------------------------------------------------------: | :-----: |
| Product owner  | Backlog 안 Hypothesis 의 순서, 그리고 Verdict               | 2, 4    |
| Domain expert  | Target, Hypothesis 안의 물리적 근거, Insight 의 물리적 해석 | 1, 2, 3 |
| Data scientist | Provenance, Baseline, Run, Insight                          | 1, 3    |
| MLOps engineer | Readiness, 그리고 Handoff 안의 engineering 작업             | 3, 4    |

도메인 엔지니어가 두 센서는 대칭이라 topology 상 함께 묶여야 한다고 말하면, 데이터 과학자가 그 topology 를 model 로 옮기는 1D-CNN filter 크기로 답한다.

## 9. Record

회의록은 아홉 항목 가운데 Verdict, Insight, Handoff 셋을 담으며, 그 셋을 채우지 못한 회의는 끝난 것이 아니다. 나머지 여섯은 추적 도구와 backlog item 에 남고, 세 줄이 그것을 가리킨다. 같은 회의의 process finding 은 따로, BKM 을 갱신하는 retrospective note 에 적는다. 둘을 섞은 기록은 어느 쪽으로도 읽히지 않기 때문이다.

```text
Verdict : <ACCEPTED|REWORK|STOP> on <HYPOTHESIS>
Insight : <what moved the metric, and what it was read from>
Handoff : <OWNER> runs <EXPERIMENT> by <DATE>, backlog item <ITEM_ID>
```

한 sprint 를 채워 넣으면 세 줄은 아래와 같이 읽힌다. 각 줄이 양을 하나씩 대고 있어, 석 달 뒤의 독자도 무엇이 시도되었는지가 아니라 무엇이 확립되었는지를 가릴 수 있다.

```text
Verdict : ACCEPTED on "1D-CNN autoencoder reduces the trace to 200 dimensions"
Insight : Gas flow variation over the first 2,000 rows moves the final yield
          prediction most, by XGBoost feature importance. Reconstruction error 0.02
Handoff : <OWNER> runs Elastic Net and supervised 1D-CNN on the 200 compressed
          features, target 95 % yield classification accuracy, by 06-28, backlog item <ITEM_ID>
```

같은 세 줄이 model 과 함께 나가는 model card 를 채운다. Model card 는 model 의 개요, 측정된 성능, 학습에 쓴 데이터를 기록한다 [[4](#ref-4)]. 이 workflow 를 agile process 안에 접어 넣은 팀도 그것을 여전히 자기 단계의 연속으로 돌린다 [[1](#ref-1)].

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
[5] Schwaber, K., & Sutherland, J. (2020). [The Scrum Guide](https://scrumguides.org/scrum-guide.html). November 2020.<br>
<a id="ref-6"></a>
[6] Zinkevich, M. [Rules of Machine Learning: Best Practices for ML Engineering](https://developers.google.com/machine-learning/guides/rules-of-ml). Google for Developers.

---

## Appendix A. Terminology

- **1D-CNN autoencoder**: 1차원 신호 위의 convolutional network 로, 자기 입력을 다시 만들도록 학습한다. 여기서는 공정 trace 의 차원을 줄이는 데 쓴다.
- **Backlog item**: Backlog 안의 작업 한 단위. 팀의 tracker 에 id 를 달고 놓인다.
- **BKM (Best Known Method)**: 어떤 작업에 대해 현재까지 알려진 최선의 방법을 담은 팀 문서. 회고에서 갱신된다.
- **Blocker**: 실험이 다음 단계로 나가지 못하게 막는 기술적·행정적 걸림돌.
- **Confusion matrix**: 예측 class 와 실제 class 를 교차시킨 표. 어느 class 를 어느 class 와 혼동하는지 읽는다.
- **Daily standup**: 진행 중인 일의 blocker 를 드러내는 짧은 일일 회의.
- **Data leakage**: serving 시점에는 얻을 수 없는 정보가 model 에 닿는 것. offline 점수만 올리고 online 점수는 올리지 않는다.
- **DoD (Definition of Done)**: 작업이 완료로 불리려면 넘어야 하는 명시적 기준.
- **DoR (Definition of Ready)**: 팀이 어떤 작업을 sprint 안으로 들이기 전에 그 작업이 넘어야 할 기준. 현장에서 쓰이지만 Scrum Guide 에는 없다.
- **Elastic Net**: L1 과 L2 norm 을 함께 쓰는 선형 model. 상관된 변수 가운데 하나만 고르지 않고 함께 남긴다.
- **EVT (Extreme Value Theory)**: 분포 꼬리의 통계. 여기서는 센서 값이 얼마나 극단인지로 임계값을 정하는 데 쓴다.
- **Feature importance**: 학습된 model 이 각 입력에 붙이는 점수. 어느 입력이 예측을 움직였는지 읽는다.
- **Foundation model**: 넓은 데이터로 미리 학습해 두고, 처음부터 학습하는 대신 fine-tuning 으로 과제에 맞추는 큰 model.
- **ICE score**: Impact, Confidence, Ease 로 backlog item 의 순서를 매기는 점수.
- **Increment**: 한 sprint 가 만들어 낸 작동하는 산출물.
- **Latent space**: encoder 가 입력을 옮겨 놓은 축소된 좌표.
- **Loss curve**: 학습 단계에 대해 그린 학습 loss 와 검증 loss.
- **Metric**: Run 이 목표 양에 대해 내놓는 값. accuracy, F1-score, RMSE 같은 것이다.
- **MLOps**: model 을 실험에서 운영으로 옮기고 거기에 머물게 하는 실천.
- **Model card**: model 의 개요, 측정된 성능, 학습 데이터를 기록한 문서.
- **Multicollinearity**: 입력 변수 사이의 거의 선형인 종속. 개별 계수를 불안정하게 만든다.
- **Product backlog**: 아직 sprint 안으로 들이지 않은 작업의 순서 매긴 목록.
- **Product owner**: Backlog 의 순서를 소유하고 increment 를 받아들이는 역할.
- **Representation learning**: feature 를 사람이 지정하는 대신 데이터에서 학습하는 것.
- **Retrospective**: Sprint 끝에 process 를 되짚고 무엇을 고칠지 정하는 회의.
- **Sliding window augmentation**: 연속 기록에서 겹치는 window 를 잘라 학습 표본을 늘리는 것. window 끼리 행을 나누어 갖는다.
- **Spike**: 결과를 알 수 없어 배포 board 밖으로 들어낸, 제 backlog item 으로 다루는 조사.
- **Sprint**: 길이가 고정된 구간. Hypothesis 하나를 여는 회의부터 그 Verdict 를 내는 회의까지 나른다.
- **Sprint backlog**: 한 sprint 안에 끝내기로 한 작업.
- **Story point**: Backlog 항목의 크기를 상대적으로 매긴 추정값.
- **Time warping**: 시간 축의 일그러짐. 같은 공정의 기록 사이에서 신호를 밀거나 늘인다.
- **Timeboxing**: 어떤 일이 얼마나 오래 돌 수 있는지를 미리 정하고, 그 한계에서 무엇에 닿았든 닫는 것.
- **Train/serve skew**: 학습 때 적용한 preprocessing 과 serving 때 적용한 preprocessing 사이의 차이.
- **Velocity**: 한 sprint 에서 팀이 끝낸 story point 의 수.
- **WIP (Work In Progress)**: 현재 진행 중인 작업. 다음 것을 시작하기 전에 끝낼 수 있도록 개수를 제한한다.
- **Yield**: 생산된 단위 가운데 규격을 만족하는 비율.

## Appendix B. Experiment Priority

"일단 다 돌려보죠" 는 sprint 만큼의 연산 자원을 쓰고 Insight 는 남기지 못하는 습관이다. 아무도 틀을 잡아 주지 않은 실행은 어떤 질문에도 답하지 않기 때문이다. 아이디어는 세 filter 를 통과한 뒤에야 product backlog 에 오르고, backlog 안에서의 순서는 ICE score 가 정한다.

### B.1 Three Filters

제안은 code 를 쓰기 전에 걸러지며, 여기서 기각하는 데 드는 몇 분이 실험으로 갔을 때의 몇 주를 대신한다.

Table 5. The three filters and what each one rejects

| Filter                        | Question                                                                 | Rejected when                                 |
| :---------------------------: | :----------------------------------------------------------------------: | :-------------------------------------------: |
| Domain alignment              | 이 algorithm 의 수학적 성질이 공정 데이터의 물리적 성질과 맞는가         | 공정에 대해 알려진 사실과 어긋나는 선택       |
| Cost-benefit                  | 드는 시간과 연산 자원에 대해 metric 의 return 이 확실한가                | 싼 baseline 대비 1점을 얻는 데 3주            |
| Explainability and deployment | 출력이 공정 엔지니어를 납득시키고, model 이 serving pipeline 에 얹히는가 | Black box 이거나, 현장이 돌리기에 너무 무거움 |

```text
[ Proposed idea ]
       |
       v
[ 1. Domain alignment ]   --> rejected
       | passed
       v
[ 2. Cost-benefit ]       --> rejected
       | passed
       v
[ 3. Explainability ]     --> rejected
       | passed
       v
[ Product backlog, ordered by ICE score ]
```

<a id="fig-2"></a>
Fig 2. The three filters an idea passes before it reaches the product backlog

**Domain alignment** 는 library 를 고르기 전에 질문을 먼저 던진다. 센서 간 상관이 깊은데 Lasso 를 쓰면 상관된 무리에서 하나만 남기므로 Ridge 나 Elastic Net 이 먼저다. 시간 축을 따라 인과가 흐르는 연속 공정에서 행 순서를 섞는 random forest 를 주력으로 두는 것은 맞지 않으며, 그 자리는 1D-CNN 이나 시계열 model 의 것이다.

**Cost-benefit** 은 이미 서 있는 가장 싼 model 에 대고 return 을 견준다. 정확도 91 % 의 가벼운 XGBoost 가 92 % 를 겨냥해 설계하는 대형 Transformer 보다 앞서며, 싼 baseline 을 끌어올리는 일이 마지막이 아니라 첫 번째 선택이다.

**Explainability and deployment** 는 점수가 무엇이든 지켜야 하는 제약이다. Ensemble 의 ensemble 은 실시간 가상 simulation 안에서 답하지 못하므로, offline 점수가 아무리 좋아도 backlog 의 윗자리에서 내려온다.

### B.2 ICE Score

방의 의견이 갈릴 때는 각자가 세 글자에 1 부터 5 까지 점수를 매기고 그 평균이 backlog 의 순서를 정한다. 이 점수는 측정이 아니라 견해차를 드러내 놓는 방법이다.

Table 6. The three letters of an ICE score

| Letter | Name       | Question                                                        |
| :----: | :--------: | :-------------------------------------------------------------: |
| I      | Impact     | 이 Hypothesis 가 서면 metric 이 얼마나 움직이는가               |
| C      | Confidence | 발표된 연구나 이전 공정 데이터에 비추어 그것이 설 확률이 높은가 |
| E      | Ease       | 전처리와 구현이 얼마나 적게 드는가                              |

```math
\mathrm{ICE} = I \times C \times E \hspace{19em} (1)
```

식 (1) 은 셋째 글자를 Ease 로, 5 를 가장 쉬움으로 매길 때 성립한다. Effort 로 매겨 5 를 가장 어려움으로 두는 팀은 곱하는 대신 나누며, 같은 순서를 읽는다.

**Quick win** 은 셋 모두에서 높은 점수를 받는다. 이미 있는 sliding window 의 보폭만 조절해 데이터를 늘리는 일은 반나절이면 구현되고 그 효과도 미리 안다. **Long-term** 항목은 Ease 와 Confidence 가 함께 낮다. 대형 시계열 foundation model 을 fine-tuning 하는 일은 몇 주가 들고 결과를 아무도 예측하지 못하므로, sprint 를 여는 대신 quick win 뒤에서 기다린다.

### B.3 Hypothesis Rule

Hypothesis 가 없으면 실험도 없다. Hypothesis 의 형태로 — 바꾸는 하나, 물리적 근거, 예상되는 움직임 — 말하지 못하는 제안은 아무리 새롭더라도 다음 sprint 에 들지 않는다.

> 기각: "요즘 LightGBM 이 유행이라는데 이것도 한번 돌려보죠."

> 채택: "센서 데이터에 noise 가 많아 선형 model 이 overfitting 하는 것으로 보입니다. Tree 기반의 LightGBM 은 결측치와 noise 에 robust 하므로, 현재 Baseline 보다 정확도가 3점 이상 오른다는 가설입니다. 이것을 검증하겠습니다."
