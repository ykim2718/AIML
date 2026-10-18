# Semiconductor Deposition Equipment AI Agent Proposal
Rev. 34 | Created: 2026-10-14 | Updated: 2026-10-18 04:38 CDT

- [1. Purpose](#1-purpose)
- [2. Quick Start and Architecture](#2-quick-start-and-architecture)
- [3. Solution Items](#3-solution-items)
- [4. Equipment Agent Loop](#4-equipment-agent-loop)
- [5. Autonomy Levels](#5-autonomy-levels)
- [6. New Equipment Onboarding](#6-new-equipment-onboarding)
- [7. Operating Rules and Roadmap](#7-operating-rules-and-roadmap)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Single vs Multi-Equipment](#appendix-b-single-vs-multi-equipment)
- [Appendix C. Model Architecture](#appendix-c-model-architecture)
- [Appendix D. Orchestrator Functions](#appendix-d-orchestrator-functions)
- [Appendix E. Onboarding Gate Metrics](#appendix-e-onboarding-gate-metrics)
- [Appendix F. Model Lifecycle](#appendix-f-model-lifecycle)

## 1. Purpose

- **Problem Statement**: 증착 장비의 두께 예측 model 은 엔지니어가 결과를 읽고 다음 웨이퍼의 조건을 직접 정해야 하며, 장비가 늘면 model 개발, 보정, 정비 판단이 장비마다 따로 반복된다.
- **Goal**: 검증된 ML model 을 함수로 호출하는 AI agent 가 웨이퍼 단위의 예측·보정·정비 판단을 수행하여, 장비가 늘어도 신규 장비 셋업 기간과 엔지니어 개입 횟수가 장비 수에 비례해 늘지 않게 한다.
- **Non-Goal**: LLM 이 THK 를 직접 예측하거나 공정 조건 값을 산출하는 구조와, feature store·model registry·fab systems 의 구축과 운영 절차는 다루지 않는다.

## 2. Quick Start and Architecture

- **구성**: 장비마다 하나인 equipment agent 와, 여러 agent 를 조율하는 AI orchestrator 의 2단 구조.
- **역할 분담**: ML model 은 VM 예측, recipe 역산, drift 계산을 맡는다. LLM 은 어떤 함수를 언제 호출할지 정하고 (function calling) 결과를 설명하며, THK 나 recipe 값을 직접 계산하지 않는다. 엔지니어의 질문에는 LLM copilot 이 data 를 분석해 답한다.
- **Human-in-the-loop**: 승인이나 확인이 필요하면 agent 는 run 을 멈추고 요청만 올린다. Orchestrator 가 그 요청을 권한 있는 엔지니어에게 보내고 승인 기록을 남긴다.
- **자율성**: L1 advisory 에서 L3 closed-loop R2R 까지 운영 실적에 따라 올린다 ([5. Autonomy Levels](#5-autonomy-levels)).
- **신규 장비**: 네 gate 로 기존 model 을 다시 쓸 수 있는지 판정한다 ([6. New Equipment Onboarding](#6-new-equipment-onboarding)).
- **다장비 효과**: 장비가 늘수록 학습 data 와 excursion 사례가 쌓여, 장비당 셋업 기간과 관리 부담이 줄어든다 ([Appendix B](#appendix-b-single-vs-multi-equipment)).

전체 구성은 [Fig 1](#fig-1) 과 같다.

```text
                   +-------------------------------------------+     +-------------------+
                   |              AI Orchestrator              |<--->| Engineer 👩‍🔧       |
                   |  event workflow / policy / permission     |     | approves actions  |
                   |  approval routing / audit record          |     | by autonomy level |
                   +---------------------+---------------------+     | answers requests  |
                                         |                           | reads reports     |
                   predict / recommend / health / calibrate          +-------------------+
                                         |
       +---------------------------------+-------------------+----------------------+
       |                                                     |                      |
+------+----------------------------------------+    +-------+--------+     +-------+--------+
| Equipment Agent #1                            |    | Equipment      |     | Equipment      |
|                                               |    | Agent #2       | ... | Agent #N       |
|  LLM       which function to call, and when   |    | same as #1     |     | same as #1     |
|            explanation of the result          |    +-------+--------+     +-------+--------+
|               |  function call ^  output      |            |                      |
|               v                |              |            |                      |
|  ML model  predict THK and its uncertainty    |            |                      |
|            invert process condition           |            |                      |
|            detect drift, retrain the adapter  |            |                      |
|  Guardrail uncertainty threshold before APC   |            |                      |
+------+----------------------------------------+            |                      |
       |                                                     |                      |
       +------------+-----------------------------------+----+----------------------+
                    |                                   |
       Feature Store          Model Registry (Global backbone + Local adapters)
```

<a id="fig-1"></a>
Fig 1. Two-tier agent architecture with the engineer, and the LLM and the ML model inside one equipment agent

`Equipment Agent #1` 은 agent 하나의 안을 펼친 것이며, `#2` 부터 `#N` 까지도 같은 구성이다. Orchestrator 와 agent 는 predict, recommend, health, calibrate 네 함수 interface 로만 주고받으며, orchestrator 의 기능은 [Appendix D](#appendix-d-orchestrator-functions) 에 있다.

- **Feature store**: 장비 trace 를 step 별 요약 통계와 recipe·PM 이후 RF hours 같은 context 로 가공한 model 입력 (feature) 을 보관하여, 학습과 추론에 같은 feature 를 내준다.
- **Model registry**: Global backbone 과 chamber 별 Local adapter 의 version 을 보관하며, agent 는 그 가운데 운영에 쓰도록 지정된 production version 만 불러 쓴다. Local adapter 는 chamber 마다 하나씩 두는 작은 보정 model 로, Global backbone 의 예측에 남은 그 chamber 고유의 offset·gain 만 학습한다. Version 이 만들어지고 production version 으로 정해지는 흐름은 [Appendix F](#appendix-f-model-lifecycle) 에 있다.

## 3. Solution Items

솔루션은 다섯 분류, 아홉 항목이다.

Table 1. Solution items

<table>
  <tr><th align="center">Category</th><th align="center">Item</th><th align="center">Description</th><th align="center">LLM role</th><th align="center">Section</th></tr>
  <tr><td align="center" rowspan="2">A. APC</td><td align="center">Virtual metrology (두께 실시간 예측)</td><td align="center">웨이퍼마다 계측하지 않고 증착 두께를 즉시 예측하여 계측 시간과 비용 절감</td><td align="center">함수 호출 (VM model)</td><td align="center">4</td></tr>
  <tr><td align="center">R2R control (자동 공정 보정)</td><td align="center">예측 두께가 목표에서 벗어나면 다음 웨이퍼의 증착 조건 자동 조정</td><td align="center">함수 호출 (역산 model), 조정 근거 설명</td><td align="center">4</td></tr>
  <tr><td align="center" rowspan="2">B. EES</td><td align="center">FDC (이상 조기 감지)</td><td align="center">장비 센서 신호를 실시간 감시하여 불량 발생 전 경보</td><td align="center">함수 호출 (FDC 감시)</td><td align="center">4</td></tr>
  <tr><td align="center">FDC root-cause 분석 (고장 원인 안내)</td><td align="center">이상 발생 시 원인으로 의심되는 부품과 조건을 쉬운 말로 안내</td><td align="center">원인 설명</td><td align="center">4</td></tr>
  <tr><td align="center" rowspan="2">C. PHM</td><td align="center">Consumable RUL (소모품 수명 예측)</td><td align="center">타깃 등 소모 부품의 소모량을 추적하여 최적 교체 시점 안내</td><td align="center">함수 호출 (RUL model)</td><td align="center">4, Appendix D</td></tr>
  <tr><td align="center">PdM (예방 정비 최적화)</td><td align="center">장비 상태에 따라 PM 시점을 추천하여 불필요한 down time 감소</td><td align="center">PM 추천 근거 설명</td><td align="center">Appendix D</td></tr>
  <tr><td align="center">D. TTTM</td><td align="center">Chamber matching (장비 간 편차 관리)</td><td align="center">여러 장비와 챔버의 결과 차이를 비교하여 같은 품질 유지</td><td align="center">없음 (orchestrator 가 맡음)</td><td align="center">Appendix C, D</td></tr>
  <tr><td align="center" rowspan="2">E. MLOps</td><td align="center">Model retraining (자동 학습 유지)</td><td align="center">PM 이나 부품 교체 뒤에도 재학습으로 정확도 유지</td><td align="center">재학습 시점 판단, 함수 호출</td><td align="center">4</td></tr>
  <tr><td align="center">LLM copilot (AI 운영 비서)</td><td align="center">"어제 2번 장비가 왜 멈췄나" 같은 질문에 data 를 분석해 답변</td><td align="center">질문에 답변</td><td align="center">2</td></tr>
</table>

Section 열은 그 항목을 다루는 꼭지다. 기대 효과는 계측 비용 절감, 수율 향상, 장비 가동률 증가, 엔지니어 업무 부담 감소이다.

## 4. Equipment Agent Loop

웨이퍼 한 장이 끝날 때마다 [Fig 2](#fig-2) 의 loop 가 한 번 돈다.

```text
  +--------------------------------------------------------------------------+
  | LLM  calls [1] - [6] as functions, reads the output of each step,        |
  |      decides which step to call next, and writes the [7] Report          |
  +--------------------------------------------------------------------------+

  +---------------+
  | Trace data in |
  +-------+-------+
          |
          v
  +-------------------+     +------------------+     +----------------------+
  | [1] Detect (code) |---->| [2] Predict (ML) |---->| [3] Guardrail (rule) |
  +-------------------+     +------------------+     +-----------+----------+
                                                                 |
                    +--------------------------------------------+
    high confidence |                                            | low confidence
                    v                                            v
  +----------------------------------+            +----------------------------+
  | [4] Recommend (ML)               |            | Metrology sampling request |
  +-----------------+----------------+            +--------------+-------------+
                    |                                            |
                    v                                            v
  +----------------------------------+            +----------------------------+
  | Approval request to orchestrator |            | Metrology data in          |
  +-----------------+----------------+            +--------------+-------------+
                    |                                            |
                    v                                            v
  +----------------------------------+            +----------------------------+
  | APC (R2R)                        |            | [5] Drift monitor (ML)     |
  +----------------------------------+            +--------------+-------------+
                                                                 |
                                                                 v
  +----------------------------------+            +----------------------------+
  | [7] Report (LLM)                 |<-----------| [6] Adapt (ML)             |
  +-----------------+----------------+            +----------------------------+
                    |
                    v
  +--------------------------------------------------------------------------+
  | Orchestrator  routes the report and the requests of [1] and [4]          |
  |               to the engineer, and returns the answers                   |
  +--------------------------------------------------------------------------+
```

<a id="fig-2"></a>
Fig 2. Per-wafer loop of the equipment agent and what carries each step

단계 옆의 (ML), (rule), (code), (LLM) 은 그 단계를 맡는 주체다. LLM 은 단계마다 결과를 읽어 다음에 호출할 함수를 정하며, [5] 와 [6] 은 계측값이 들어올 때만 돈다.

- **[1] Detect (FDC), code**: trace 를 표준 feature 로 바꾸고 missing value, outlier, sensor fault 를 점검한다. Sensor mapping 이 애매하면 orchestrator 에 확인을 요청한다.
- **[2] Predict (VM), ML**: Global, Local, Ensemble model 가운데 하나를 고르거나 가중 결합해 계측 전 THK 와 신뢰구간을 낸다. Model 구성은 [Appendix C](#appendix-c-model-architecture) 에 있다.
- **[3] Guardrail (VM reliability), rule**: Uncertainty 가 threshold 안이면 [4] 로 넘기고, 넘으면 R2R 전달을 막고 adaptive metrology sampling 을 요청한다.
- **[4] Recommend (R2R), ML**: Inverse model 로 다음 wafer 의 recipe 를 내고, SOA 95% 안의 조건만 제안한다.
- **Approval request**: Table 2 의 autonomy level 이 승인을 요구하는 변경은 run 을 멈추고 orchestrator 에 승인을 요청하며, 승인을 받은 뒤 APC 로 보낸다.
- **[5] Drift monitor (EES), ML**: Residual trend, consumable 수명에 따른 SOA shift, PM 과 seasoning 뒤의 covariate shift 와 concept shift 를 감지한다.
- **[6] Adapt (recalibration), ML**: Recalibration DOE (예: monitor wafer 5매, CCD axial 4점 + center point) 를 제안하고 해당 chamber 의 adapter 만 재학습한다. Shadow validation 과 시간순 검증을 통과한 후보만 production version 교체를 요청하며, 교체 절차는 [Appendix F](#appendix-f-model-lifecycle) 에 있다.
- **[7] Report (RCA), LLM**: VM 편차의 root-cause sensor 와 recipe step, 권장 조치를 자연어 report 로 요약하여 orchestrator 를 거쳐 엔지니어에게 보낸다 (예: "증착 시간 0.4초 증가 권장, 신뢰도 95%").

## 5. Autonomy Levels

자율성은 운영 실적에 따라 세 단계로 올리며, 단계가 오를수록 엔지니어가 승인하는 범위가 줄어든다.

Table 2. Autonomy levels

| Level                                | Automatic action                       | Human approval                         |
| :---:                                | :---:                                  | :---:                                  |
| L1 Advisory, open-loop (조언형)      | 없음, recipe 추천만                    | 모든 recipe 변경                       |
| L2 Supervised closed-loop (반자동)   | SOA 95% 안의 recipe 조정               | SOA 95% 밖의 recipe 변경               |
| L3 Closed-loop R2R (자동 run-to-run) | SOA 안의 recipe 조정, model retraining | Production version 교체, SOA 밖 recipe |

L1 에서 추천의 적중을 확인한 뒤 다음 단계로 넘어가므로, 단계를 올리는 근거가 운영 실적으로 쌓인다.

## 6. New Equipment Onboarding

신규 장비는 기준 장비와 견주어 Table 3 의 네 gate 를 차례로 거치며, gate 마다 판정 결과에 따라 다음 조치가 정해진다.

Table 3. Onboarding gates, results and actions

<table>
  <tr><th align="center">Gate</th><th align="center">Check</th><th align="center">Result</th><th align="center">Action</th></tr>
  <tr><td align="center" rowspan="2">Schema gate</td><td align="center" rowspan="2">핵심 sensor 의 mapping, sampling period 와 단위</td><td align="center">핵심 sensor 있음, period·단위 일치</td><td align="center">다음 gate</td></tr>
  <tr><td align="center">핵심 sensor 없음</td><td align="center">신규 개발</td></tr>
  <tr><td align="center" rowspan="4">Covariate shift</td><td align="center" rowspan="4">PSI / domain classifier AUC</td><td align="center">&lt; 0.1 / &lt; 0.6</td><td align="center">다음 gate</td></tr>
  <tr><td align="center">특정 제어 변수 (압력, 온도, power offset) 만 이탈</td><td align="center">장비 튜닝 (chamber matching) 후 재평가</td></tr>
  <tr><td align="center">0.1 ~ 0.25 / 0.6 ~ 0.8</td><td align="center">Model 보정 후보로 다음 gate</td></tr>
  <tr><td align="center">&gt; 0.25 / &gt; 0.8</td><td align="center">신규 개발</td></tr>
  <tr><td align="center" rowspan="3">Concept shift</td><td align="center" rowspan="3">SHAP 중요도 순위의 Spearman 상관</td><td align="center">&gt; 0.8</td><td align="center">다음 gate</td></tr>
  <tr><td align="center">0.5 ~ 0.8</td><td align="center">Model 보정 후보로 다음 gate</td></tr>
  <tr><td align="center">&lt; 0.5</td><td align="center">신규 개발</td></tr>
  <tr><td align="center" rowspan="4">Pilot validation</td><td align="center" rowspan="4">20~30매 실계측의 RMSE, R², bias</td><td align="center">스펙 안, bias 없음</td><td align="center">기존 model 재사용</td></tr>
  <tr><td align="center">Offset 만 있음</td><td align="center">Bias 보정</td></tr>
  <tr><td align="center">Offset 과 gain</td><td align="center">Adapter fine-tune</td></tr>
  <tr><td align="center">보정 후에도 스펙 밖</td><td align="center">신규 개발</td></tr>
</table>

경계값은 초기 권장값이며 fab 마다 보정한다. Model 이 하드웨어 차이를 흡수하면 그 뒤의 drift 를 하드웨어 변화와 구별하기 어려우므로, 장비 튜닝을 model 보정보다 먼저 검토한다. Gate 마다 함께 보는 지표는 [Appendix E](#appendix-e-onboarding-gate-metrics) 에 있다.

## 7. Operating Rules and Roadmap

운영 전에 아래 네 가지를 먼저 정하며, 이 가운데 장비 연동이 보통 가장 오래 걸린다.

- **Safety**: SOA 밖 조건 실행 금지, threshold 를 넘는 VM 값의 R2R 전달 차단, production version 교체 전 검증 gate, 모든 결정의 log 기록.
- **Security**: 고객 데이터가 fab 밖으로 나가지 않도록 on-premise LLM 이나 feature 만 다루는 구조.
- **Equipment interface**: SECS/GEM, FDC/EES, MES, APC (R2R) 와의 interface.
- **KPI**: 스펙 적중률, wafer 당 보정 비용, 엔지니어 개입 횟수, consumable 수명 종료 구간 성능.

Roadmap 은 단기·중기·장기의 세 Phase 로 나누며, Phase 마다 걸리는 기간은 장비 구성과 data 상태에 따라 달라진다.

- **Phase 1 (단기, L1)**: 기존 VM model 과 역산을 함수로 감싼 agent, 승인 routing 과 질의응답만 맡는 최소 orchestrator, 동일 기종·다 chamber 의 Global + Local model 적용, chamber matching 검증.
- **Phase 2 (중기, L2)**: [3] Guardrail, [5] Drift monitor, [6] Adapt 단계 도입, recalibration DOE 와 재학습 자동화 (MLOps).
- **Phase 3 (장기, L3)**: 이기종·신규 장비의 few-shot onboarding, LLM 기반 orchestrator, 다른 공정으로 확장.

---

## Appendix A. Terminology

- **APC (R2R)**: 계측값이나 예측값으로 다음 run 의 recipe 를 조정하는 공정 제어 (advanced process control, run-to-run).
- **CCD**: Axial 점과 기준점으로 구성하는 실험 설계 (central composite design).
- **Challenger model**: 재학습으로 새로 만든 후보 model version. 검증을 마치기 전에는 제어에 쓰지 않는다.
- **Chamber fingerprint**: 같은 recipe 에서도 chamber 마다 다르게 나타나는 고유한 offset·gain 과 sensor 특성.
- **Champion model**: 지금 production version 으로 지정되어 운영에 쓰이는 model version. Challenger 와 견주는 기준이 된다.
- **Concept shift**: 입력과 출력의 관계가 바뀌는 drift.
- **Conformal prediction**: 검증 data 의 잔차 분포로 정해진 확률을 보장하는 예측 구간을 만드는 방법.
- **Consumable**: 타깃처럼 증착 장비에서 쓰는 만큼 닳아 주기적으로 교체하는 부품.
- **Covariate shift**: 입출력 관계는 그대로이고 입력 분포만 바뀌는 drift.
- **Dispatching**: 대기 중인 lot 이나 wafer 를 어느 장비에서 처리할지 정하는 일.
- **DOE**: 실험 계획 (design of experiments).
- **Domain classifier AUC**: 기준 장비와 신규 장비의 data 를 분류기가 얼마나 구분하는지를 나타내는 AUC. 0.5 에 가까울수록 두 data 가 비슷하다.
- **DTW**: 시간 축의 늘어남을 허용하여 두 trace 의 형상 차이를 재는 거리 (dynamic time warping).
- **Excursion**: 공정이나 장비가 관리 범위를 벗어나 불량 wafer 가 생길 수 있는 상태.
- **Fab systems**: 장비 에이전트가 데이터를 받고 제어값을 보내는 fab 의 기존 시스템. SECS/GEM, FDC/EES, MES, APC (R2R) 를 묶어 부르는 이름이다.
- **FDC/EES**: 장비의 trace 를 수집·감시하는 fault detection and classification / equipment engineering system.
- **Function calling**: LLM 이 함수를 고르고 입력값을 정해 실행을 요청한 뒤 결과를 받아 읽는 방식. 이 문서에서는 함수 호출이라 적으며, 반도체에서 tool 은 장비를 뜻하므로 tool calling 이라 적지 않는다.
- **Golden trace envelope**: normal run 들의 trace 로 정한 step 별 허용 범위.
- **Hotelling T²**: 다변량 data 가 기준 분포의 중심에서 벗어난 정도를 재는 통계량.
- **Human-in-the-loop**: 자동화된 흐름 안에서 사람이 승인이나 판단을 맡는 운영 방식. 이 문서에서는 장비 에이전트가 run 을 멈추고 요청을 올리면 orchestrator 가 엔지니어에게 전달하고 답을 돌려주는 흐름을 가리킨다.
- **KS test**: 두 표본의 누적 분포 차이로 분포가 같은지 검정하는 Kolmogorov-Smirnov test.
- **LoRA**: 큰 model 의 가중치는 고정하고 저차원 행렬만 학습하는 경량 adapter 방식 (low-rank adaptation).
- **Mahalanobis distance**: 변수 간 공분산을 고려한 다변량 거리.
- **MAML**: 적은 data 로 빠르게 적응하도록 초기 가중치를 학습하는 meta-learning 방법 (model-agnostic meta-learning).
- **MES**: 공장의 생산 실행 시스템 (manufacturing execution system).
- **Mixed-effects model**: 공통 효과와 장비·챔버별 random effect 를 함께 추정하는 통계 model.
- **MLOps**: Model 의 학습·검증·배포·감시를 자동화하는 운영 체계.
- **Monitor wafer**: 제품이 아니라 공정 상태를 확인하거나 보정하려고 흘리는 wafer.
- **PCA**: 상관된 변수를 소수의 직교 성분으로 줄이는 주성분 분석 (principal component analysis).
- **PdM**: 장비 상태 data 로 고장 시점을 예측해 그 전에 정비하는 예지 정비 (predictive maintenance).
- **PHM**: 장비의 이상 감지, 잔여 수명 예측, 정비 판단을 묶어 다루는 분야 (prognostics and health management).
- **PM**: 예방 정비 (preventive maintenance).
- **Production version**: Model registry 에 보관한 model version 가운데 지금 운영에서 예측에 쓰도록 지정된 version.
- **PSI**: 두 분포의 구간별 비율 차이로 분포 이동을 재는 지표 (population stability index).
- **RCA**: 이상이나 편차를 일으킨 원인 sensor·부품·recipe step 을 찾는 분석 (root-cause analysis).
- **RMSE**: 예측 오차 제곱 평균의 제곱근 (root mean square error).
- **Rollback**: 새 production version 에 문제가 생겼을 때 보관해 둔 이전 version 으로 되돌리는 일.
- **RUL**: 부품이나 소모품이 교체 기준에 이를 때까지 남은 사용 기간 (remaining useful life).
- **Seasoning**: PM 뒤 챔버 상태를 안정시키려고 진행하는 사전 run.
- **SECS/GEM**: 반도체 장비와 host 사이의 표준 통신 protocol.
- **Shadow validation**: 후보 model 을 운영 model 과 나란히 돌려 결과만 비교하고 제어에는 쓰지 않는 검증.
- **SHAP**: 예측값을 feature 별 기여도로 나누어 설명하는 방법 (SHapley additive explanations).
- **SOA**: 추천을 허용하는 공정 조건의 운전 영역 (safe operating area). SOA 95% 는 그 영역의 95% 범위이다.
- **SPC**: 공정 측정값을 관리 한계와 비교해 이상을 가려내는 통계적 공정 관리 (statistical process control).
- **THK**: 증착 막의 두께 (thickness).
- **Tool qualification**: 신규 장비가 양산 조건을 만족하는지 확인하고 생산에 투입하는 절차.
- **TTTM**: 같은 recipe 에서 장비·챔버 사이의 결과 차이를 줄이는 matching (tool-to-tool matching).
- **Virtual metrology (VM)**: 계측 전에 공정 data 로 계측값을 예측하는 방법.
- **Wasserstein distance**: 한 분포를 다른 분포로 옮기는 최소 비용으로 정의한 분포 간 거리.

## Appendix B. Single vs Multi-Equipment

같은 AI 를 여러 장비에 함께 쓰면 장비가 늘수록 예측이 정확해지고 장비당 셋업 기간과 관리 부담이 줄어든다.

Table 4. Single vs multi-equipment deployment

<table>
  <tr><th align="center">Aspect</th><th align="center">Item</th><th align="center">Single equipment</th><th align="center">Multi-equipment</th></tr>
  <tr><td align="center" rowspan="2">Training data (학습 데이터)</td><td align="center">Data volume (데이터 양)</td><td align="center">한 장비 데이터만 사용</td><td align="center">모든 장비 데이터를 함께 학습</td></tr>
  <tr><td align="center">Domain coverage (경험 범위)</td><td align="center">겪어 본 상황만 예측</td><td align="center">다른 장비의 경험까지 활용</td></tr>
  <tr><td align="center" rowspan="2">Tool qualification (신규 장비 도입)</td><td align="center">Model setup (셋업)</td><td align="center">장비마다 처음부터 개발</td><td align="center">기존 AI 재활용</td></tr>
  <tr><td align="center">Lead time (소요 기간)</td><td align="center">장비마다 처음부터 개발하는 기간</td><td align="center">대폭 감소</td></tr>
  <tr><td align="center" rowspan="2">SPC (품질 관리)</td><td align="center">Control limit (기준)</td><td align="center">장비별 개별 관리</td><td align="center">전체를 하나의 기준으로 관리</td></tr>
  <tr><td align="center">TTTM (장비 간 편차)</td><td align="center">파악 어려움</td><td align="center">자동 비교와 보정</td></tr>
  <tr><td align="center" rowspan="2">Excursion control (이상 대응)</td><td align="center">Early detection (감지)</td><td align="center">문제 발생 후 대응</td><td align="center">다른 장비의 사례로 사전 경고</td></tr>
  <tr><td align="center">Containment (확산)</td><td align="center">같은 문제 반복 발생</td><td align="center">유사 장비에 미리 전파·차단</td></tr>
  <tr><td align="center" rowspan="3">Fab productivity (운영 효율)</td><td align="center">Dispatching (웨이퍼 배정)</td><td align="center">고정</td><td align="center">최적 장비로 자동 배정</td></tr>
  <tr><td align="center">Sampling·PM scheduling (계측·정비)</td><td align="center">장비별 개별 계획</td><td align="center">전체 일정 통합 최적화</td></tr>
  <tr><td align="center">Engineer workload (관리 인력)</td><td align="center">장비 수에 비례해 증가</td><td align="center">AI 에이전트가 통합 관리</td></tr>
</table>

신규 장비 도입 시, 공통 model 을 재사용하므로 셋업 기간이 대폭 줄어든다.

- **공통 model 재사용**: 모든 장비 data 로 학습한 Global backbone 이 recipe 와 THK 의 관계를 이미 담고 있다.
- **차이만 학습**: 신규 장비는 그 장비 고유의 차이만 소량의 monitor wafer 로 Local adapter 에 학습한다.
- **재사용 판정**: 재사용 여부는 Table 3 의 네 gate 로 판정하며, 통과하면 model 을 처음부터 만들지 않는다.
- **재사용 범위 확대**: 장비가 늘수록 학습 data 와 excursion 사례가 쌓여, 기존 model 을 재사용할 수 있는 장비가 늘어난다.

## Appendix C. Model Architecture

예측 model 은 전체 장비 공통의 Global backbone 과 chamber 별 경량 Local adapter 를 결합한 계층형 구조다.

Table 5. Hierarchical model layers

| Layer                              | Role                                                          | Technique                                            |
| :---:                              | :---:                                                         | :---:                                                |
| Global backbone (common model)     | 전체 장비 공통의 process physics (recipe 와 THK 의 관계) 학습 | 전체 장비 data, tool/chamber embedding 입력          |
| Local adapter (chamber model)      | Chamber fingerprint (chamber 고유 offset·gain) 보정           | Residual model, LoRA 형 adapter, mixed-effects model |
| Few-shot extension (tool start-up) | 신규 tool qualification 때 소량의 monitor wafer 계측으로 투입 | Transfer learning, meta-learning (MAML 계열)         |
| Uncertainty (VM reliability)       | VM 값의 신뢰도 산출, guardrail 의 threshold 입력              | Conformal prediction, ensemble variance              |

Local adapter 는 Global backbone 이 설명하지 못한 chamber 별 차이만 학습하므로, drift 가 생긴 chamber 는 adapter 만 다시 학습한다. Global backbone 이 모든 장비의 입력을 같은 feature 로 받도록, 장비마다 다른 sensor data 를 공통 schema 로 맞춘다.

- 장비마다 다른 sensor 이름, 단위, sampling period 를 sensor taxonomy 기준으로 표준 feature 에 mapping
- RF power, pressure, gas flow, temperature 를 step 별 요약 통계로 통일
- Recipe, step, PM 이후 RF hours, seasoning 상태 같은 context 정보를 공통 입력으로 사용

## Appendix D. Orchestrator Functions

Orchestrator 는 event 기반 workflow 와 정책·권한을 관리하며 장비 사이의 판단을 맡는다.

Table 6. Orchestrator functions

| Function                                         | Description                                                                                                                                         |
| :---:                                            | :---:                                                                                                                                               |
| Monitor wafer allocation (보정 웨이퍼 예산 배분) | Drift 가 큰 chamber 에 recalibration 용 monitor wafer 를 우선 배정                                                                                  |
| Tool start-up (신규 장비 빠른 시작)              | 같은 기종의 Global backbone 과 adapter 를 transfer learning 하여 qualification 에 필요한 monitor wafer 수 축소, 투입 여부는 Table 3 의 gate 로 판정 |
| Chamber matching (장비 간 매칭)                  | 같은 recipe 에서 chamber 별 THK offset 보정 (TTTM)                                                                                                  |
| PM scheduling (정비 일정 최적화)                 | Consumable RUL 과 장비 상태에 따른 SOA 축소를 보고 consumable 교체와 PM 시점을 장비별로 조율                                                        |
| Excursion propagation (이상 사례 전파)           | 한 장비에서 감지한 excursion 을 같은 기종 장비에 미리 알려 사전 경고                                                                                |
| Dispatching (웨이퍼 배정)                        | 장비 상태와 VM reliability 를 보고 lot·wafer 를 최적 장비에 배정                                                                                    |
| Knowledge base (공통 지식 축적)                  | Feature extraction 방식, failure case, 검증 기준을 template 으로 재사용                                                                             |

구성 예는 Table 7 과 같다.

Table 7. Example technology stack

| Purpose                                                     | Software                |
| :---:                                                       | :---:                   |
| Orchestration (fab 전체 event workflow, 승인 routing)       | LangGraph 또는 Temporal |
| Model registry (VM·R2R model 과 chamber adapter 의 version) | MLflow                  |
| Feature store (trace 에서 뽑은 공통 feature)                | Feast                   |
| Event bus (공정 종료, FDC alarm, 계측 완료 event)           | Kafka                   |

## Appendix E. Onboarding Gate Metrics

Table 3 의 gate 마다 함께 보는 지표는 Table 8 과 같다.

Table 8. Gate metrics

| Gate             | Metric                                                                                                                                                                     |
| :---:            | :---:                                                                                                                                                                      |
| Schema gate      | Feature mapping rate, key sensor 유무, sampling period·unit 일치 여부                                                                                                      |
| Covariate shift  | Feature 별 PSI, KS test, Wasserstein distance / PCA 공간의 Hotelling T², Mahalanobis distance / domain classifier AUC / step 별 DTW distance, golden trace envelope 이탈률 |
| Concept shift    | SHAP importance rank 의 Spearman correlation, key sensor 의 THK sensitivity (slope) 비교                                                                                   |
| Pilot validation | Monitor wafer 20~30매 실측의 RMSE, R², bias                                                                                                                                |

## Appendix F. Model Lifecycle

Model version 은 challenger 로 만들어져 검증과 엔지니어 승인을 거친 뒤 production version 이 되며, 그때부터 champion 이 된다. 교체된 이전 champion 은 model registry 에 보관했다가 문제가 생기면 되돌린다.

Version 하나가 거치는 단계는 [Fig 3](#fig-3) 과 같다.

```text
  +----------------------+     +----------------------+       +-------------+
  | Challenger created   |---->| Validation           |-fail->| Discarded   |
  +----------------------+     +-----------+----------+       +-------------+
                                           | pass
                                           v
                               +----------------------+
                               | Engineer approval    |
                               +-----------+----------+
                                           | approved
                                           v
                           +-------------------------------+
                           | Production version (champion) |
                           +---------------+---------------+
                                           | replaced by a new champion
                                           v
                            +----------------------------+
                            | Archived (rollback target) |
                            +----------------------------+
```

<a id="fig-3"></a>
Fig 3. Lifecycle of one model version

Validation 에서 champion 보다 나아지지 않은 challenger 는 버리고, 엔지니어의 승인을 받은 challenger 만 production version 이 된다.

Challenger 가 만들어지는 경우는 Table 9 와 같다.

Table 9. Sources of challenger versions

| Model           | Trigger                         | Created by                       | Validation                                                     |
| :---:           | :---:                           | :---:                            | :---:                                                          |
| Local adapter   | 신규 장비 onboarding (Table 3)  | Bias 보정 또는 adapter fine-tune | Table 3 의 pilot validation                                    |
| Local adapter   | [5] Drift monitor 의 drift 감지 | [6] Adapt 의 재학습              | Shadow validation, 시간순 검증                                 |
| Global backbone | 여러 장비의 data 축적           | 전체 장비 data 로 재학습 (MLOps) | Shadow validation, 시간순 검증, 모든 chamber 의 adapter 재검증 |

Onboarding 에서 만든 첫 version 은 그 chamber 에 견줄 champion 이 없으므로 Table 3 의 판정으로 검증한다.

- **Validation**: Challenger 를 champion 과 나란히 돌리는 shadow validation 과 시간순 검증으로, challenger 의 VM 오차가 champion 보다 작은지 본다. 검증 동안 challenger 의 예측은 제어에 쓰지 않는다.
- **Approval**: Production version 교체는 Table 2 의 모든 autonomy level 에서 엔지니어가 승인하며, 승인 요청은 orchestrator 를 거친다.
- **Global backbone 교체**: 모든 chamber 의 예측이 함께 바뀌므로, 그 위의 Local adapter 를 새 backbone 에 맞춰 함께 검증한 뒤 한 번에 교체한다.
- **Archive and rollback**: 밀려난 champion 은 지우지 않고 보관하며, 새 production version 에서 [5] Drift monitor 가 이상을 감지하면 이전 version 으로 되돌린다.
