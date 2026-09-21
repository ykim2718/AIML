# AI/ML Modeling Meeting
Rev. 0 | Created: 2026-09-21 | Updated: 2026-09-21 10:20 CDT

## 1. Purpose

- **Problem Statement**: Modeling 회의를 일반적인 개발 회의처럼 진행하면 "한 번 해볼게요" 로 끝나, 무엇을 가정했고 무엇이 검증되었으며 다음 실행이 무엇을 판정할 것인지가 남지 않고 같은 논의가 다음 주에 되돌아온다.
- **Goal**: 회의를 두 실험 주기 사이의 이음매로 두어, 매 회의가 실험 하나를 수치와 함께 닫고 그 수치가 담은 insight 를 이름 붙이며 담당자와 기한이 붙은 가설 하나를 여는 자리가 되게 한다.
- **Non-Goal**: 실험 추적 도구 (MLflow, Weights & Biases) 의 설정과 ticket system 운영은 다루지 않는다.

## 2. Summary

AI/ML modeling 회의는 실험의 연속을 관리하는 자리이며, modeling pipeline 이 도는 순서 그대로 진행한다. 데이터, 가설, 실험 결과, 다음 액션의 차례다. 회의는 기록으로 남는 세 줄 — 완료된 실험, 그것이 낸 insight, 다음에 도는 가설 — 로 닫는다.

순서가 일반 개발 회의와 다른 까닭은 산출물이 명세대로 만들어지는 것이 아니라 실험으로 밝혀지는 데 있다. Software 회의는 기능을 만들었는지 묻고 modeling 회의는 가설이 검증되었는지 묻는다. Modeling workflow 를 agile 주기 안에 접어 넣은 팀도 그 workflow 를 여전히 자기 단계의 연속으로 돌린다 [[1](#ref-1)].

## 3. Taxonomy and its Hierarchy

Modeling 회의가 정하는 항목은 다섯 층으로 쌓이고, 각 층은 아래 층이 이미 정해졌다고 가정한다. 늦게 정해진 층은 그 위에 쌓은 것을 모두 무효로 만들므로, 논의가 위로 올라갈수록 오류의 값은 낮아진다.

다섯 층과 각 층이 답하는 질문, 그리고 정해지는 순서는 [Fig 1](#fig-1) 에 그렸다.

```text
[ 5 ] Next Action        "What runs next, and who owns it?"
         ^   assumes
[ 4 ] Experiment Result  "Why did the number move?"
         ^   assumes
[ 3 ] Hypothesis         "What one change, on what domain reason?"
         ^   assumes
[ 2 ] Data & Baseline    "What do we learn from, and what score must we beat?"
         ^   assumes
[ 1 ] Target (Y)         "What counts as the answer?"

Settled first      Target  ->  Data  ->  Hypothesis  ->  Result  ->  Action
Cost of an error   Target  >   Data  >   Hypothesis  >   Result  >   Action
```

<a id="fig-1"></a>
Fig 1. The five layers a modeling meeting decides, and the order they are settled in

정해지지 않은 층 위에서 연 회의는 아무도 평가할 수 없는 액션을 낸다. "AI 로 불량 잡아봅시다" 는 1층이 비어 있는 채로 5층에서 여는 것이며, 그 뒤에 보고되는 모든 지표는 팀이 정의한 적 없는 양을 잰다.

### 3.1 Placement

Table 1. Where each layer is settled

| Layer             | Settled at     | What is fixed                                                    |
| :---------------: | :------------: | :--------------------------------------------------------------: |
| Target (Y)        | 첫 회의 이전   | 명시된 한계 아래의 수율, 또는 EVT 기반 임계값을 넘는 센서 값     |
| Data & Baseline   | Agenda stage 1 | 결측치와 이상치 처리, 누수 차단, baseline 점수                   |
| Hypothesis        | Agenda stage 2 | 이번 주기에 바꾸는 하나와 그 뒤의 도메인 근거                    |
| Experiment Result | Agenda stage 3 | 추적 도구에서 읽은 지표와 그것을 설명하는 feature 또는 오차 분석 |
| Next Action       | Agenda stage 4 | 다음 가설, 담당자, 기한, 그리고 거기에 필요한 engineering 작업   |

## 4. Agenda

회의는 네 단계로 돌고, 각 단계는 다음 단계가 쓰는 값 하나를 확정한다. 순서를 어기면 방은 [Fig 1](#fig-1) 에서 이미 지나온 층으로 되돌아간다.

각 단계와 그 단계가 상 위에 올리는 것은 [Fig 2](#fig-2) 에 그렸다.

```text
[ 1. Data & Baseline ]
        |
        +--> Missing / Outlier ....... How NaN and outliers in the raw rows were handled
        +--> Leakage Check ........... Sliding window augmentation checked for leakage
        +--> Baseline Score .......... The simplest model's score, stated as a number
        |
        v
[ 2. Hypothesis Setup ]
        |
        +--> Domain Reason ........... The physical ground the change rests on
        +--> One Change .............. The single thing that moves in this cycle
        +--> Expected Effect ......... What the metric does if the reason holds
        |
        v
[ 3. Experiment Review ]
        |
        +--> Tracked Run ............. Metric read from the tracker, not from memory
        +--> Why It Moved ............ Feature importance or error analysis behind it
        +--> Shown On Screen ......... Loss curve, confusion matrix, latent space plot
        |
        v
[ 4. Next Actions ]
        |
        +--> Next Hypothesis ......... The experiment this review has earned
        +--> Ticket .................. Owner and due date, issued in the tracker
        +--> Engineering Sync ........ Code review and pipeline work named here
        |
        v
[ Minutes ]   Done experiment   |   Insight   |   Next hypothesis
```

<a id="fig-2"></a>
Fig 2. The four stages of an AI/ML modeling meeting and what each one fixes

### 4.1 Data And Baseline

첫 단계는 modeling 에 쓸 dataset 의 상태를 형용사가 아니라 수치로 말한다. 세 가지 질문이 그것을 지고 간다. 원본 행의 결측치와 이상치를 어떻게 처리했는가, sliding window 증강이 data leakage 의 길을 남기지 않았는가, 그리고 가장 단순한 model — 선형 회귀나 고전 통계 — 이 이미 몇 점에 닿아 있는가이다.

Baseline 이 이 단계의 산출물이다. 뒤에 나오는 모든 주장이 그것에 대고 하는 비교이기 때문이다. 첫 model 을 단순하게 두는 것은 바로 그 이유로 확립된 출발점이며 [[5](#ref-5)], 누수는 그 비교를 무의미하게 만드는 실패다. 열일곱 분야 294편의 논문에서 여덟 가지 형태로 기록되어 있다 [[3](#ref-3)].

### 4.2 Hypothesis Setup

둘째 단계는 이번 주기가 검증할 가설을 세우며, 아직 써 보지 않은 algorithm 의 목록이 아니라 도메인 지식 위에 세운다. 형식은 고정되어 있다. 바꾸는 하나, 그것의 물리적 근거, 그 근거가 서면 지표가 어떻게 움직일지다.

두 가지 예가 그 형식을 보여 준다. 센서 간 multicollinearity 가 심하므로 다음 실행은 Lasso 대신 Elastic Net 을 써서 그룹 효과를 담는다. Time warping 이 신호를 일그러뜨리므로 고정된 변환 대신 1D-CNN autoencoder 가 representation learning 으로 차원을 줄인다.

### 4.3 Experiment Review

셋째 단계는 추적 도구에서 가져온 실행들을 화면에 띄워 견주며, 어느 이름이 이겼는지가 아니라 왜 수치가 움직였는지에 답한다. "XGBoost 가 잘 나옵니다" 는 [Fig 1](#fig-1) 의 어느 층도 닫지 못한다. "Feature importance 가 chamber 3 압력 센서를 맨 위에 두고, 그것을 빼면 점수가 baseline 으로 돌아온다" 는 4층을 닫는다.

화면에는 근거를 올린다. Loss curve, confusion matrix, 그리고 축소된 차원의 latent space 가 그것이다. 말로 서술된 결과는 방이 확인할 수도, 나중에 재현할 수도 없다.

### 4.4 Next Actions

넷째 단계는 검증된 가설을 다음 주기로 옮기며, 담당자와 기한을 붙여 ticket 으로 발행한다. 운영 준비 항목도 의지가 아니라 ticket 으로 여기에 들어온다. 그것을 재는 척도가 판단이 아니라 구체적인 test 의 점검표이기 때문이다 [[2](#ref-2)].

Engineering 작업은 이 단계에서만 이름을 받는다. Code review 배정과 pipeline 연동을 여기서 맞추어, 그 위의 modeling 논의가 일정 조율에 끊기지 않게 한다.

## 5. Anti-patterns

세 가지 습관이 modeling 회의를 결정 없이 끝내며, 각각에는 같은 시간을 쓰는 대체 행동이 있다.

Table 2. What ends a meeting without a decision

| Anti-pattern                          | Why it fails                                           | What replaces it                                                   |
| :-----------------------------------: | :----------------------------------------------------: | :----------------------------------------------------------------: |
| "일단 데이터 다 넣고 학습 돌려볼게요" | 결과가 어떤 질문에도 답하지 않는 실행에 쓰인 연산 자원 | 도메인 지식으로 먼저 거른 변수와, 그 실행이 검증할 가설의 명시     |
| 타깃 정의 전에 회의를 엶              | 뒤에 나오는 모든 지표가 팀이 정의한 적 없는 양을 잼    | Y 를 먼저 고정. 명시된 한계 아래의 수율, 또는 EVT 기반 임계값 이탈 |
| 결과를 구두로 공유                    | 방이 확인할 수 없고 아무도 재현할 수 없는 주장         | Loss curve, confusion matrix, latent space plot 을 화면에          |

## 6. Roles

세 역할이 modeling 회의를 지고 간다. 한 사람이 물리적 제약에 이름을 붙이고, 둘째가 그것을 algorithm 으로 옮기며, 셋째가 그 결과를 공정 pipeline 까지 나른다.

Table 3. What each role brings

| Role           | Brings to the room                                                   | Owns afterwards    |
| :------------: | :------------------------------------------------------------------: | :----------------: |
| Domain expert  | 물리적 구조. 대칭이어서 topology 상 함께 묶여야 하는 두 센서 같은 것 | 결과의 물리적 해석 |
| Data scientist | 그 구조를 담는 algorithm. 거기에 맞춘 1D-CNN filter 크기 같은 것     | 실험과 그 기록     |
| MLOps engineer | 학습된 model 에서 공정 pipeline 까지의 경로                          | 배포와 infra 검증  |

## 7. Minutes Template

회의록은 세 줄로 닫으며, 그 세 줄을 채우지 못한 회의는 끝난 것이 아니다. 완료된 실험, 거기서 읽은 insight, 그리고 담당자와 날짜가 붙은 다음 가설이다.

```text
Done this cycle : <experiment closed, with the number it produced>
Insight         : <what the result showed, and what it was read from>
Next hypothesis : <the change, the target metric, the owner, the due date>
```

한 주기를 채워 넣으면 세 줄은 아래와 같이 읽힌다. 각 줄이 양을 하나씩 대고 있어, 석 달 뒤의 독자도 무엇이 실제로 확립되었는지 가릴 수 있다.

```text
Done this cycle : 1D-CNN autoencoder dimensionality reduction, 200 dimensions, reconstruction error 0.02
Insight         : Gas flow variation over the first 2,000 rows moves the final yield prediction most, by XGBoost feature importance
Next hypothesis : Elastic Net and supervised 1D-CNN on the 200 compressed features, for 95 % yield classification accuracy, owner <OWNER>, due 06-28
```

같은 세 줄이 model 과 함께 나가는 model card 를 채운다. Model card 는 model 의 개요, 측정된 성능, 학습에 쓴 데이터를 기록한다 [[4](#ref-4)].

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
- **Baseline**: 가장 단순한 model 또는 규칙의 점수. 뒤에 오는 모든 model 을 견주는 기준으로 남긴다.
- **Confusion matrix**: 예측 class 와 실제 class 를 교차시킨 표. 어느 class 를 어느 class 와 혼동하는지 읽는다.
- **Data leakage**: serving 시점에는 얻을 수 없는 정보가 model 에 닿는 것. offline 점수만 올리고 online 점수는 올리지 않는다.
- **Elastic Net**: L1 과 L2 norm 을 함께 쓰는 선형 model. 상관된 변수 가운데 하나만 고르지 않고 함께 남긴다.
- **EVT (Extreme Value Theory)**: 분포 꼬리의 통계. 여기서는 센서 값이 얼마나 극단인지로 임계값을 정하는 데 쓴다.
- **Experiment tracking**: 실행마다 parameter, dataset 버전, code commit, 결과를 자동으로 남기는 것.
- **Feature importance**: 학습된 model 이 각 입력에 붙이는 점수. 어느 입력이 예측을 움직였는지 읽는다.
- **Latent space**: encoder 가 입력을 옮겨 놓은 축소된 좌표.
- **Loss curve**: 학습 단계에 대해 그린 학습 loss 와 검증 loss.
- **MLOps**: model 을 실험에서 운영으로 옮기고 거기에 머물게 하는 실천.
- **Multicollinearity**: 입력 변수 사이의 거의 선형인 종속. 개별 계수를 불안정하게 만든다.
- **Representation learning**: feature 를 사람이 지정하는 대신 데이터에서 학습하는 것.
- **Sliding window augmentation**: 연속 기록에서 겹치는 window 를 잘라 학습 표본을 늘리는 것. window 끼리 행을 나누어 갖는다.
- **Target (Y)**: model 이 예측하는 양. 그 정의가 무엇을 정답으로 볼지를 고정한다.
- **Time warping**: 시간 축의 일그러짐. 같은 공정의 기록 사이에서 신호를 밀거나 늘인다.
- **Yield**: 생산된 단위 가운데 규격을 만족하는 비율.
