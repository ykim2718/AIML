# AI/ML Code Review Meeting
Rev. 0 | Created: 2026-09-21 | Updated: 2026-09-21 10:07 CDT

## 1. Purpose

- **Problem Statement**: Modeling 작업을 두고 여는 review 회의가 diff 를 한 줄씩 읽는 데 쓰이면 split 과 baseline 과 실행 간 편차가 검토되지 않은 채로 남고, 그 방 안의 누구도 재현하지 못하는 숫자 하나로 model 이 승인된다.
- **Goal**: Pull request 가 판정하지 못하는 것 가운데 회의가 판정할 것, 방이 그것을 판정하도록 저자가 미리 돌리는 packet, 그리고 회의가 반드시 닿아야 할 결정을 고정한다.
- **Non-Goal**: 이름짓기, 구조, test coverage 같은 일반적인 code style review 는 다루지 않는다. 검토자 한 사람이 비동기로 판정하는 것이기 때문이다.

## 2. Summary

Modeling review 회의는 주장 하나를 판정한다. 명시된 dataset 으로 학습한 model 이 명시된 baseline 을 실행 간 잡음보다 큰 폭으로 앞선다는 주장이며, 방은 그것이 서는지와 다음에 무엇을 할지를 정한다.

기계나 검토자 한 사람이 판정할 수 있는 것은 회의가 열리기 전에 모두 끝나 있다. 남는 것은 여러 사람이 한자리에 있어야 내려지는 판단이다. Split 이 정직했는지, 비교가 공정했는지, 얻은 이득이 seed 를 바꾸어도 살아남는지, 그리고 그 model 이 serving 에 맞는지가 그것이다. 회의는 네 가지 결정 가운데 하나로 끝나며, 각각에 담당자와 날짜가 붙는다.

## 3. Taxonomy and its Hierarchy

Review 는 세 층에서 일어나고, 그 층은 한 findings 가 몇 사람을 필요로 하는가로 갈리며, 각 층은 자기가 판정하지 못하는 것만 위로 넘긴다. 한 층 올라갈 때마다 판단을 얻고 처리량을 내려놓으므로, 각 층에 닿는 항목의 수는 줄고 항목 하나의 값은 오른다.

세 층과 각 층이 답하는 질문, 그리고 각 층이 판정하는 것은 [Fig 1](#fig-1) 에 그렸다.

```text
Review Meeting   (several people)   "Is the result worth building on?"
  |
  |   settles   experiment validity, metric against baseline, model fitness,
  |             the decision to ship, rework or stop
  |
  +-- Async Review   (one reviewer)   "Is the change correct and readable?"
        |
        |   settles   local correctness, naming, structure, test coverage,
        |             leakage visible in the diff
        |
        +-- Automated Check   (no reader)   "Does it pass what a machine decides?"
              |
              |   settles   lint, type, unit test, data schema validation,
              |             experiment logged, metric recorded

Cost per item rises        Automated  <  Async  <  Meeting
Items reaching the level   Automated  >  Async  >  Meeting
```

<a id="fig-1"></a>
Fig 1. The three review levels and what each one settles

층을 잘못 잡은 항목은 값을 두 번 치른다. 회의에서 꺼낸 lint findings 는 hook 이 판정할 것에 여러 사람의 시간을 쓰고, baseline 이 빠진 실험을 pull request 에서 승인하면 그 사실은 model 이 serving 에 닿고 나서야 드러난다.

### 3.1 Placement

Table 1. Where each item is settled

| Item                               | Settled at      | Why there                                         |
| :--------------------------------: | :-------------: | :-----------------------------------------------: |
| Lint, format, type                 | Automated check | 읽는 사람 없이 판정되는 고정된 규칙               |
| Unit test, schema validation       | Automated check | Pipeline 이 이미 계산해 둔 통과 여부              |
| Experiment logged, metric recorded | Automated check | 없다는 사실을 기계가 알아내는 기록                |
| Naming, structure, test coverage   | Async review    | 검토자 한 사람의 판단. 두 번째 의견이 필요 없음   |
| Leakage visible in the diff        | Async review    | Split 을 만드는 code 에서 그대로 읽힘             |
| Split design and its honesty       | Review meeting  | 저자가 무엇을 가정했는지 찾는 데 여러 사람이 필요 |
| Baseline choice and the comparison | Review meeting  | 정확성이 아니라 공정성에 대한 판단                |
| Gain against run-to-run spread     | Review meeting  | 방이 함께 받아들이거나 물리치는 주장              |
| Serving fitness                    | Review meeting  | Modeling, engineering, 운영에 한꺼번에 걸침       |
| Ship, rework or stop               | Review meeting  | 담당자가 붙고 한 번 기록되는 결정                 |

## 4. Preparation

회의는 저자가 하루 앞서 돌린 packet 위에서 열리고, 읽지 않고 온 검토자는 표결자가 아니라 참관자로 앉는다. 그래야 방이 앞 절반을 packet 에 이미 적힌 것에 쓰지 않는다.

Table 2. What the review packet holds

| Part     | Content                                                          |
| :------: | :--------------------------------------------------------------: |
| Claim    | 실험이 답한 질문. 한 문장                                        |
| Ask      | 방에서 받고자 하는 결정                                          |
| Data     | Dataset 버전 hash, split 별 행 수, split 을 만든 규칙            |
| Baseline | 넘어야 할 점수와 그 숫자의 출처                                  |
| Result   | 실행별 지표와 seed 간 편차                                       |
| Diff     | Pull request link. 이미 green 이고 이미 비동기 review 를 거친 것 |
| Artifact | Model registry 등록분, 또는 아직 없는 이유                       |

어느 한 부분이 빠진 packet 은 논의 대신 반려한다. 여기의 각 부분이 section 5 의 agenda 가 지나가는 자리이므로, packet 의 빈자리는 그대로 회의의 빈자리가 된다.

## 5. Agenda

진행 순서는 숫자 하나가 주장이 되기까지 지나온 길을 따른다. 데이터, 실험, code, serving 의 차례이며, 결정은 오는 길에 다다른 결론이 아니라 마지막 항목으로 둔다. 아래의 분 배분은 60분 회의를 한 가지로 나눈 것이며, 팀은 그것을 늘리거나 줄이되 순서와 마지막의 결정은 그대로 둔다.

각 단계와 그 단계가 상 위에 올리는 것은 [Fig 2](#fig-2) 에 그렸다.

```text
[ 0. Before the Meeting ]
        |
        +--> Review Packet ........... Circulated one working day ahead
        +--> CI Green ................ Lint, unit test and schema check passing
        |
        v
[ 1. Framing (5 min) ]
        |
        +--> Question ................ What the experiment was run to decide
        +--> Ask ..................... The decision the author wants from the room
        |
        v
[ 2. Data And Split (10 min) ]
        |
        +--> Split Rule .............. How train, validation and test were separated
        +--> Leakage Check ........... What could have reached the model from the target
        +--> Data Version ............ The dataset hash the run read
        |
        v
[ 3. Experiment And Metric (15 min) ]
        |
        +--> Baseline ................ The score the new model must beat
        +--> One Change .............. What moved between the compared runs
        +--> Run Spread .............. Variation across seeds, against the claimed gain
        |
        v
[ 4. Code And Reproducibility (10 min) ]
        |
        +--> Clean Rerun ............. The training script from a fresh checkout
        +--> Config .................. Paths and parameters out of the source
        +--> Notebook State .......... Exploration converted into a script or dropped
        |
        v
[ 5. Serving Readiness (10 min) ]
        |
        +--> Train/Serve Skew ........ The same preprocessing on both sides
        +--> Latency ................. The inference budget measured, not estimated
        +--> Fallback ................ The answer returned when confidence is low
        |
        v
[ 6. Decision (10 min) ]
        |
        +--> Outcome ................. Approve, approve with conditions, rework or stop
        +--> Actions ................. An owner and a date for every item raised
```

<a id="fig-2"></a>
Fig 2. The agenda of an AI/ML code review meeting

네 가지 역할이 회의를 [Fig 2](#fig-2) 의 agenda 에 붙들어 두며, 저자가 사회까지 보면 사회자를 두는 이유인 논쟁을 저자가 끝내므로 각각 다른 사람이 맡는다.

Table 3. Roles in the room

| Role      | Holds                                   | Stays out of                                         |
| :-------: | :-------------------------------------: | :--------------------------------------------------: |
| Author    | 주장, packet, 질문에 대한 답            | 방이 이미 rework 로 받아들인 선택을 다시 변호하는 일 |
| Reviewer  | 미리 읽어 온 packet, 타당성에 대한 판단 | 비동기로 이미 끝난 style findings                    |
| Moderator | 시계, 범위, 항목의 순서                 | 기술적 판단 자체                                     |
| Scribe    | 결정과 action 목록                      | 논의                                                 |

시간 관리의 대부분은 규칙 하나가 진다. 방 안에서의 debugging 은 범위 밖이라는 것이다. Packet 만으로 아무도 풀지 못하는 항목은 담당자가 붙은 action 이 되고, 회의는 다음으로 넘어간다.

## 6. Review Points

각 단계는 작업에 질문 하나씩을 던지고, 각각에는 누군가 이름을 붙이기 전까지 성공처럼 보이는 실패가 있다. 아래는 방이 실제로 들여다보는 항목이며, 그것을 꺼내는 단계별로 묶었다.

Table 4. What each stage checks and how it fails

| Stage                    | Check                                                              | What a failure looks like                                                  |
| :----------------------: | :----------------------------------------------------------------: | :------------------------------------------------------------------------: |
| Data and split           | Scaler 와 encoder 를 포함하여, 어떤 fitting 보다도 먼저 만든 split | 모든 행에 맞춘 scaler 에서 나온, 이후의 어떤 시도보다 높은 test 점수       |
| Data and split           | 행이 시간을 담을 때 지켜진 시간 순서                               | 미래를 읽어 live data 를 만나기 전까지만 잘 맞는 model                     |
| Data and split           | Feature 에 target 이 직접으로도 대리 변수로도 없음                 | Label 에서 파생된 열로 추적되는, 거의 완벽한 지표                          |
| Data and split           | 한 번만 건드린 test set                                            | 같은 holdout 위에서 거듭 고른 끝에 맞춰져 버린 test 점수                   |
| Experiment and metric    | 있고 또 닿을 수 있는 baseline                                      | 아무것도 없는 것에 대고, 또는 아무도 조율하지 않은 model 에 대고 말한 이득 |
| Experiment and metric    | 비교한 쌍마다 하나씩만 움직인 변수                                 | 데이터도 함께 바뀌었는데 architecture 의 공으로 돌린 이득                  |
| Experiment and metric    | Seed 간 편차보다 큰 이득                                           | 다시 돌리면 나오는 잡음 안쪽에 있는 향상                                   |
| Experiment and metric    | Model 이 맡을 결정에 맞는 지표                                     | 운영 환경에는 오지 않는 class 균형 위에서의 높은 accuracy                  |
| Code and reproducibility | 새로 받은 checkout 에서 도는 학습                                  | 저자의 작업 folder 에서만 재현되는 결과                                    |
| Code and reproducibility | 손으로 적지 않고 고정한 데이터 버전과 parameter                    | 옮겨진 파일을 읽어 다른 숫자를 돌려주는 재실행                             |
| Serving readiness        | 학습과 serving 에서 똑같은 preprocessing                           | 어긋난 변환 때문에 offline 에서 맞고 online 에서 틀리는 model              |
| Serving readiness        | 제 한계에 대고 측정한 latency                                      | Laptop 에서는 한계 안, 부하 아래에서는 한계 밖인 model                     |
| Serving readiness        | Low confidence 결과에 대해 정해진 동작                             | Model 이 내놓은 것을 그대로 돌려준 처리되지 않은 요청                      |

## 7. Outcome

회의는 네 가지 결정 가운데 하나로 닫히며, 방이 흩어지기 전에 소리 내어 말하고 적어 둔다. 그 가운데 하나에 닿지 못한 회의는 packet 을 그대로 둔 채 대기열로 돌아가고, 다음 회의가 같은 자리에서 다시 시작한다.

Table 5. The four outcomes

| Outcome                 | What it means                                      | What is recorded                    |
| :---------------------: | :------------------------------------------------: | :---------------------------------: |
| Approve                 | 주장이 서고 model 이 다음으로 나아가도 됨          | Registry tag 와 검토자 이름         |
| Approve with conditions | 주장은 서되 serving 전에 지정한 항목이 들어와야 함 | 조건마다 담당자와 날짜              |
| Rework                  | 보인 근거로는 주장이 아직 서지 않음                | 빠진 근거와 다음 review 날짜        |
| Stop                    | 한 번 더 돌릴 값이 없는 방향                       | 그 이유. 다음 팀이 읽을 자리에 남김 |

기록은 개인 memo 가 아니라 pull request 와 experiment tracker 에 둔다. 누군가의 inbox 에 있는 승인은 석 달 뒤 그 model 이 의심받을 때 찾을 수 없다.

---

## Appendix A. Terminology

- **Baseline**: 새 model 이 넘어야 할 기준 점수로 남겨 두는 가장 단순한 model 또는 규칙.
- **Data leakage**: serving 시점에는 얻을 수 없는 정보가 model 에 닿는 것. offline 점수만 올리고 online 점수는 올리지 않는다.
- **Data version hash**: dataset 의 한 상태를 정확히 가리키는 식별자. 결과를 그것이 나온 행까지 되짚게 한다.
- **Holdout**: 마지막 측정 한 번을 위해 남겨 두고 모든 조율 결정에서 빼 두는 split.
- **Latency budget**: 추론이 그 안에 답해야 하는 한계. 평균이 아니라 백분위로 적는다.
- **Moderator**: 회의의 시계와 범위를 쥐고, 기술적 내용은 판정하지 않는 사람.
- **Review packet**: 회의 전에 저자가 돌리는 문서 묶음. 주장, 데이터, baseline, 결과, artifact 를 담는다.
- **Run spread**: random seed 만 다른 반복 실행 사이에서 지표가 흔들리는 폭.
- **Scribe**: 결정과 action 목록을 기록하는 사람.
- **Seed**: 한 실행의 난수 추출을 고정하는 값. 그 실행을 그대로 되풀이할 수 있게 한다.
- **Slice metric**: 전체가 아니라 데이터의 한 부분 집단 위에서 계산한 지표.
- **Train/serve skew**: 학습 때 적용한 preprocessing 과 serving 때 적용한 preprocessing 사이의 차이.
