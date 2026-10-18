# Semiconductor Deposition Equipment AI Agent Proposal
Rev. 7 | Created: 2026-10-14 | Updated: 2026-10-18 01:02 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Solution Scope](#3-solution-scope)
- [4. Single vs Multi-Equipment](#4-single-vs-multi-equipment)
- [5. Model Architecture](#5-model-architecture)
  - [5.1 Common Feature Schema](#51-common-feature-schema)
- [6. Equipment Agent](#6-equipment-agent)
  - [6.1 Agent Loop](#61-agent-loop)
  - [6.2 Role of LLM](#62-role-of-llm)
  - [6.3 Autonomy Level](#63-autonomy-level)
- [7. Multi-Equipment Orchestrator](#7-multi-equipment-orchestrator)
  - [7.1 Technology Stack](#71-technology-stack)
- [8. New Equipment Onboarding Gate](#8-new-equipment-onboarding-gate)
- [9. Key Issues](#9-key-issues)
- [10. Roadmap](#10-roadmap)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

- **Problem Statement**: 증착 장비의 두께 예측 model 은 엔지니어가 결과를 읽고 다음 웨이퍼의 조건을 직접 정해야 하며, 장비가 늘면 model 개발, 보정, 정비 판단이 장비마다 따로 반복된다.
- **Goal**: 검증된 ML model 을 도구로 쓰는 AI agent 가 웨이퍼 단위의 예측·보정·정비 판단을 수행하여, 장비가 늘어도 신규 장비 셋업 기간과 엔지니어 개입 횟수가 장비 수에 비례해 늘지 않게 한다.
- **Non-Goal**: LLM 이 THK 를 직접 예측하거나 공정 조건 값을 산출하는 구조는 다루지 않는다.

## 2. Summary

증착 장비 1대를 맡는 장비 에이전트를 두고, 그 위에 공통 Global model 과 장비·챔버별 Local adapter 로 여러 장비를 조율하는 AI orchestrator 를 두는 2단 구성을 제안한다.
예측과 역산은 검증된 ML model 이 맡고 LLM 은 도구 호출 판단과 결과 설명만 맡으며, 자율성은 L1 조언형에서 L3 자동 run-to-run 까지 단계적으로 올린다.
신규 장비는 sensor 호환성, 데이터 분포, 입력·출력 관계, 20\~30매 pilot 계측의 네 gate ([8. New Equipment Onboarding Gate](#8-new-equipment-onboarding-gate)) 를 차례로 거쳐 기존 model 재사용·장비 튜닝·model 보정·신규 개발 중 하나로 판정하고, 기존 model 을 다시 쓸 수 있는 장비는 수일\~수주 안에 투입한다.

전체 구성은 [Fig 1](#fig-1) 과 같다.

```text
                   +-------------------------------------------+     +-------------------+
                   |              AI Orchestrator              |<--->| Engineer          |
                   |  event workflow / policy / permission     |     | approves actions  |
                   +---------------------+---------------------+     | by autonomy level |
                                         |                           | reads reports     |
                   predict / recommend / health / calibrate          +-------------------+
                                         |
       +---------------------------------+-------------------+------------------+
       |                                                     |                  |
+------+----------------------------------------+    +-------+--------+ +-------+--------+
| Equipment Agent #1                            |    | Equipment      | | Equipment      |
|                                               |    | Agent #2       | | Agent #N       |
|  LLM       which tool to call, and when       |    | same as #1     | | same as #1     |
|            explanation of the result          |    +-------+--------+ +-------+--------+
|               |  tool call     ^  output      |            |    ...           |
|               v                |              |            |                  |
|  ML model  predict THK and its uncertainty    |            |                  |
|            invert process condition           |            |                  |
|            detect drift, retrain the adapter  |            |                  |
|  Guardrail uncertainty threshold before APC   |            |                  |
+------+----------------------------------------+            |                  |
       |                                                     |                  |
       +------------+-----------------------------------+----+------------------+
                    |                                   |
       Feature Store (common schema)          Model Registry (Global
                    |                         backbone + Local adapters)
       SECS/GEM, FDC/EES, MES, APC (R2R)
```

<a id="fig-1"></a>
Fig 1. Two-tier agent architecture with the engineer, and the LLM and the ML model inside one equipment agent

아래 층은 장비마다 하나인 장비 에이전트이고, 위 층의 orchestrator 는 네 가지 공통 도구 interface 로만 장비 에이전트와 주고받는다. 공통 feature schema 로 정리한 data 는 feature store 에, Global model 과 장비별 adapter 는 model registry 에 둔다.

`Equipment Agent #1` 상자는 장비 에이전트 하나의 안을 펼쳐 그린 것이며, 다른 장비 에이전트도 같은 구성이다. LLM 은 어떤 도구를 언제 부를지 정하고 그 결과를 설명하며, THK 예측과 공정 조건 역산은 ML model 이, 예측값을 APC 로 넘길지는 guardrail 의 불확실성 임계값이 정한다. LLM 이 맡는 범위는 [6.2 Role of LLM](#62-role-of-llm) 에 정리되어 있다.

엔지니어는 orchestrator 의 policy·permission 을 거쳐 [6.3 Autonomy Level](#63-autonomy-level) 의 Table 5 가 정한 범위의 조정을 승인하고, 장비 에이전트가 [7] Report 단계에서 쓴 report 를 읽는다.

## 3. Solution Scope

제안하는 솔루션은 품질 예측·제어, 장비 상태·정비 관리, 다장비 통합 관리, AI 운영 기반의 네 분류, 아홉 항목이다.

Table 1. Solution items

<table>
  <tr><th align="center">Category</th><th align="center">Item</th><th align="center">Description</th><th align="center">Section</th></tr>
  <tr><td align="center" rowspan="2">A. 품질 예측·제어</td><td align="center">두께 실시간 예측</td><td align="center">웨이퍼마다 계측하지 않고 증착 두께를 즉시 예측하여 계측 시간과 비용 절감</td><td align="center">6.1</td></tr>
  <tr><td align="center">자동 공정 보정</td><td align="center">예측 두께가 목표에서 벗어나면 다음 웨이퍼의 증착 조건 자동 조정</td><td align="center">6.1</td></tr>
  <tr><td align="center" rowspan="4">B. 장비 상태·정비 관리</td><td align="center">소모품 수명 예측</td><td align="center">타깃 등 소모 부품의 소모량을 추적하여 최적 교체 시점 안내</td><td align="center">6.1, 7</td></tr>
  <tr><td align="center">이상 조기 감지</td><td align="center">장비 센서 신호를 실시간 감시하여 불량 발생 전 경보</td><td align="center">6.1</td></tr>
  <tr><td align="center">고장 원인 안내</td><td align="center">이상 발생 시 원인으로 의심되는 부품과 조건을 쉬운 말로 안내</td><td align="center">6.1</td></tr>
  <tr><td align="center">예방 정비 최적화</td><td align="center">장비 상태에 따라 PM 시점을 추천하여 불필요한 down time 감소</td><td align="center">7</td></tr>
  <tr><td align="center">C. 다장비 통합 관리</td><td align="center">장비 간 편차 관리</td><td align="center">여러 장비와 챔버의 결과 차이를 비교하여 같은 품질 유지</td><td align="center">5, 7</td></tr>
  <tr><td align="center" rowspan="2">D. AI 운영 기반</td><td align="center">자동 학습 유지</td><td align="center">PM 이나 부품 교체 뒤에도 재학습으로 정확도 유지</td><td align="center">6.1</td></tr>
  <tr><td align="center">AI 운영 비서</td><td align="center">"어제 2번 장비가 왜 멈췄나" 같은 질문에 data 를 분석해 답변</td><td align="center">6.2</td></tr>
</table>

Section 열은 그 항목을 구현하는 본문 절이다. 기대 효과는 계측 비용 절감, 수율 향상, 장비 가동률 증가, 엔지니어 업무 부담 감소이다.

## 4. Single vs Multi-Equipment

같은 AI 를 여러 장비에 함께 쓰면 장비가 늘수록 학습 data 와 이상 사례가 쌓여 예측이 정확해지고, 장비당 셋업 기간과 관리 부담이 줄어든다.

Table 2. Single vs multi-equipment deployment

<table>
  <tr><th align="center">Aspect</th><th align="center">Item</th><th align="center">Single equipment</th><th align="center">Multi-equipment</th></tr>
  <tr><td align="center" rowspan="2">학습 데이터</td><td align="center">데이터 양</td><td align="center">한 장비 데이터만 사용</td><td align="center">모든 장비 데이터를 함께 학습</td></tr>
  <tr><td align="center">경험 범위</td><td align="center">겪어 본 상황만 예측</td><td align="center">다른 장비의 경험까지 활용</td></tr>
  <tr><td align="center" rowspan="2">신규 장비 도입</td><td align="center">셋업</td><td align="center">장비마다 처음부터 개발</td><td align="center">기존 AI 재활용</td></tr>
  <tr><td align="center">소요 기간</td><td align="center">수개월</td><td align="center">수일~수주</td></tr>
  <tr><td align="center" rowspan="2">품질 관리</td><td align="center">기준</td><td align="center">장비별 개별 관리</td><td align="center">전체를 하나의 기준으로 관리</td></tr>
  <tr><td align="center">장비 간 편차</td><td align="center">파악 어려움</td><td align="center">자동 비교와 보정</td></tr>
  <tr><td align="center" rowspan="2">이상 대응</td><td align="center">감지</td><td align="center">문제 발생 후 대응</td><td align="center">다른 장비의 사례로 사전 경고</td></tr>
  <tr><td align="center">확산</td><td align="center">같은 문제 반복 발생</td><td align="center">유사 장비에 미리 전파·차단</td></tr>
  <tr><td align="center" rowspan="3">운영 효율</td><td align="center">웨이퍼 배정</td><td align="center">고정</td><td align="center">최적 장비로 자동 배정</td></tr>
  <tr><td align="center">계측·정비</td><td align="center">장비별 개별 계획</td><td align="center">전체 일정 통합 최적화</td></tr>
  <tr><td align="center">관리 인력</td><td align="center">장비 수에 비례해 증가</td><td align="center">AI 에이전트가 통합 관리</td></tr>
</table>

단일 장비 적용은 그 장비 하나의 개별 최적화이고, 다수 장비 적용은 장비가 늘수록 성능이 오르는 전체 최적화이다.

## 5. Model Architecture

예측 model 은 장비 공통의 Global backbone 과 장비·챔버별 경량 Local adapter 를 결합한 계층형 구조로 하고, orchestrator 가 데이터 정합, model 선택, 드리프트 대응, 재학습을 조율한다.

Table 3. Hierarchical model layers

| Layer              | Role                            | Technique                                            |
| :---:              | :---:                           | :---:                                                |
| Global backbone    | 장비 공통의 물리·공정 관계 학습 | 전체 장비 data, tool/chamber embedding 입력          |
| Local adapter      | 챔버 고유 offset·gain 보정      | Residual model, LoRA 형 adapter, mixed-effects model |
| Few-shot extension | 신규 장비를 소량 계측으로 투입  | Transfer learning, meta-learning (MAML 계열)         |
| Uncertainty        | 예측 신뢰도 산출                | Conformal prediction, ensemble variance              |

Local adapter 는 Global backbone 이 설명하지 못한 챔버별 차이만 학습하므로, 드리프트가 생긴 챔버는 adapter 만 다시 학습한다.

### 5.1 Common Feature Schema

Global backbone 은 모든 장비의 입력을 같은 표준 feature 로 받을 때 성립하므로, 장비마다 다른 센서 data 를 공통 schema 로 맞춘다.

- 장비마다 다른 센서명, 단위, sampling 주기를 센서 taxonomy 기준으로 표준 feature 에 mapping
- RF power, pressure, gas flow, temperature 를 step 별 요약 통계로 통일
- Recipe, step, PM 이후 RF hours, seasoning 상태 같은 context 정보를 공통 입력으로 사용

## 6. Equipment Agent

장비 에이전트는 증착 장비 1대를 맡아, 웨이퍼마다 감지에서 보고까지의 loop 를 역할별 에이전트로 수행한다.

### 6.1 Agent Loop

웨이퍼 한 장이 끝날 때마다 [Fig 2](#fig-2) 의 loop 가 한 번 돈다.

```text
  +------------------------------------------------------------------+
  | LLM   reads the output of each step, decides which step to call  |
  |       next, and writes the [7] Report                            |
  +---------------------------------+--------------------------------+
                                    |  tool calls to [1] - [6]
                                    v
  Trace data in
       |
       v
  [1] Detect  ----->  [2] Predict  ----->  [3] Guardrail (rule)
      (tool)              (ML)
                                                |
                          high confidence  +----+----+  low confidence
                                           |         |
                                           v         v
                          [4] Recommend -> APC     Metrology sampling
                              (ML)         (R2R)   request
                                                     |
                                                     v
  [7] Report (LLM)  <--  [6] Adapt  <--  [5] Drift monitor  <---  Metrology data in
                             (ML)              (ML)
       |
       v
  Engineer
```

<a id="fig-2"></a>
Fig 2. Per-wafer loop of the equipment agent and what carries each step

LLM 은 단계마다 결과를 읽어 다음에 부를 단계를 정하고, [7] Report 는 직접 쓴다. 단계 옆의 (ML) 은 ML model, (rule) 은 불확실성 임계값 규칙, (tool) 은 trace data 를 표준 feature 로 바꾸는 변환 도구가 맡는다는 표시다.

[5] 와 [6] 은 계측값이 들어올 때만 돌고, 나머지 단계는 웨이퍼마다 돈다. 단계마다 맡는 에이전트와 하는 일은 Table 4 와 같다.

Table 4. Steps of the agent loop

| Step          | Role agent               | Action                                                                                                                                          |
| :---:         | :---:                    | :---:                                                                                                                                           |
| Detect        | Data Harmonization Agent | 공정 종료 직후 trace data 를 표준 feature 로 변환, 결측·이상치·센서 이상 점검, mapping 이 애매하면 엔지니어에게 확인 요청                       |
| Predict       | Model Router Agent       | 장비 상태와 data 유사도로 Global, Local, Ensemble 중 선택 또는 가중치 결정, THK 와 신뢰구간 예측 (계측 전 virtual metrology)                    |
| Guardrail     | Guardrail Agent          | 불확실성이 임계값 안이면 예측값을 APC (R2R) 로 전달, 넘으면 전달하지 않고 adaptive metrology sampling 요청                                      |
| Recommend     | Equipment Agent          | 다음 웨이퍼의 공정 조건 역산, SOA 95% 영역 안의 조건만 제안                                                                                     |
| Drift monitor | Drift Monitor Agent      | 잔차 추세, 소모품 수명에 따른 SOA 이동, PM·부품 교체·seasoning 에 따른 covariate shift 와 concept shift 감지                                    |
| Adapt         | Adaptation Agent         | 보정 DOE 제안 (예: 보정 웨이퍼 5매, CCD axial 4점 + 기준점), 해당 챔버의 adapter 만 재학습, shadow 검증과 시간순 검증을 통과한 후보만 승격 요청 |
| Report        | Explanation Agent        | 예측 편차의 원인 센서와 step, 권장 조치를 엔지니어용 자연어 report 로 요약 (예: "증착 시간 0.4초 증가 권장, 신뢰도 95%")                        |

Detect 의 이상 점검과 Report 의 원인 요약이 Table 1 의 이상 조기 감지와 고장 원인 안내를 맡고, Adapt 가 자동 학습 유지를 맡는다.

### 6.2 Role of LLM

예측과 역산은 검증된 ML model 이 도구로서 수행하고, LLM 은 어떤 도구를 언제 부를지 판단하고 그 결과를 설명하는 역할만 맡는다. Explanation Agent 의 report 와 AI 운영 비서의 질의 응답이 LLM 이 맡는 일이다.

에이전트는 feature 이름 대신 데이터 기반 도구의 출력만으로 판단하므로, 암호화된 고객 데이터처럼 도메인 지식을 쓸 수 없는 경우에도 동작한다.

### 6.3 Autonomy Level

자율성은 세 단계로 나누어 올리며, 단계마다 사람이 승인하는 범위가 줄어든다.

Table 5. Autonomy levels

| Level              | Automatic action      | Human approval          |
| :---:              | :---:                 | :---:                   |
| L1 조언형          | 없음 (추천만)         | 모든 조정               |
| L2 반자동          | SOA 95% 안의 조정     | SOA 95% 밖의 조정       |
| L3 자동 run-to-run | SOA 안의 조정, 재학습 | Model 교체, SOA 밖 조건 |

고객은 L1 에서 추천의 적중을 확인한 뒤 다음 단계로 넘어가므로, 자율성을 올리는 근거가 운영 실적으로 쌓인다.

## 7. Multi-Equipment Orchestrator

장비 에이전트들이 같은 도구 interface (predict, recommend, health, calibrate) 를 공유하면, orchestrator 가 event 기반 workflow 와 정책·권한을 관리하며 장비 사이의 판단을 맡는다.

Table 6. Orchestrator functions

| Function              | Description                                                                                                                               |
| :---:                 | :---:                                                                                                                                     |
| 보정 웨이퍼 예산 배분 | 여러 장비 중 드리프트가 큰 장비에 보정 웨이퍼를 우선 배정                                                                                 |
| 신규 장비 빠른 시작   | 같은 기종 장비의 Global backbone 과 adapter 를 전이학습하여 초기 학습에 필요한 계측 웨이퍼 수 축소, 투입 여부는 section 8 의 gate 로 판정 |
| 장비 간 매칭          | 동일 recipe 에서 챔버별 THK 차이 보정                                                                                                     |
| 정비 일정 최적화      | 소모품 수명과 장비 상태에 따른 SOA 축소를 보고 소모품 교체와 PM 시점을 장비별로 조율                                                      |
| 이상 사례 전파        | 한 장비에서 감지한 이상 사례를 유사 장비에 미리 알려 사전 경고                                                                            |
| 웨이퍼 배정           | 장비 상태와 예측 신뢰도를 보고 웨이퍼를 최적 장비에 배정                                                                                  |
| 공통 지식 축적        | Feature 추출 방식, 실패 사례, 검증 기준을 template 으로 재사용                                                                            |

### 7.1 Technology Stack

구성 예는 Table 7 과 같다.

Table 7. Example technology stack

| Purpose        | Tool                    |
| :---:          | :---:                   |
| Orchestration  | LangGraph 또는 Temporal |
| Model registry | MLflow                  |
| Feature store  | Feast                   |
| Event bus      | Kafka                   |

## 8. New Equipment Onboarding Gate

신규 장비는 기준 장비 대비 네 gate 를 [Fig 3](#fig-3) 의 차례로 평가하여 기존 model 재사용, 장비 튜닝, model 보정, 신규 개발 중 하나로 판정한다.

```text
[1] Schema gate  -->  [2] Covariate shift  -->  [3] Concept shift  -->  [4] Pilot validation
    sensor                data distribution         input-output             20~30 wafers
    compatibility         similarity                relation similarity      measured
```

<a id="fig-3"></a>
Fig 3. Four-step onboarding gate

Gate 마다 보는 지표와 판정은 Table 8 과 같다.

Table 8. Gate metrics and decisions

| Gate             | Metric                                                                                                                                                                     | Decision                                                                                  |
| :---:            | :---:                                                                                                                                                                      | :---:                                                                                     |
| Schema gate      | 입력 feature 의 mapping 률, 상위 중요도 센서의 존재 여부, sampling 주기와 단위의 일치 여부                                                                                 | 핵심 센서가 없으면 신규 개발                                                              |
| Covariate shift  | Feature 별 PSI, KS test, Wasserstein distance / PCA 공간의 Hotelling T², Mahalanobis distance / domain classifier AUC / step 별 DTW distance, golden trace envelope 이탈률 | 이탈이 특정 제어 변수 (압력, 온도, power offset) 에 국한되면 장비 튜닝 (chamber matching) |
| Concept shift    | SHAP 중요도 순위의 Spearman 상관, 주요 센서의 두께 민감도 (기울기) 비교                                                                                                    | 관계 구조가 다르면 분포가 비슷해도 신규 개발                                              |
| Pilot validation | 20~30매 실계측의 RMSE, R², bias                                                                                                                                            | Offset 만 있으면 bias 보정, offset 과 gain 이면 adapter fine-tune, 비선형이면 재개발      |

네 gate 의 결과를 모아 Table 9 로 최종 판정한다.

Table 9. Decision matrix

| Decision                    | Distribution (PSI / Domain AUC) | Relation (SHAP rank correlation) | Pilot result                 |
| :---:                       | :---:                           | :---:                            | :---:                        |
| 기존 model 재사용           | < 0.1 / < 0.6                   | > 0.8                            | 스펙 안, bias 없음           |
| 장비 튜닝                   | 특정 제어 변수만 이탈           | 유지                             | 튜닝 후 재평가               |
| Model 보정 (bias / adapter) | 0.1\~0.25 / 0.6\~0.8            | 0.5\~0.8                         | Offset·gain 보정으로 스펙 안 |
| 신규 개발                   | > 0.25 / > 0.8                  | < 0.5                            | 보정 후에도 스펙 밖          |

표의 경계값은 초기 권장값이며 fab 마다 보정한다. 하드웨어 차이를 model 이 흡수하면 그 뒤의 드리프트를 하드웨어 변화와 구별하기 어려워지므로, 장비 튜닝을 model 보정보다 먼저 검토한다.

## 9. Key Issues

실현에는 안전 조건, 보안, 장비 연동, 성과 지표의 네 가지를 먼저 정해야 하며, 이 중 장비 연동이 보통 가장 오래 걸린다.

Table 10. Key issues for deployment

| Issue     | Approach                                                                                                        |
| :---:     | :---:                                                                                                           |
| 안전 조건 | SOA 밖 조건 실행 금지, 불확실성 임계값을 넘는 예측의 R2R 전달 차단, model 교체 전 검증 gate, 모든 결정 log 기록 |
| 보안      | 고객 데이터가 외부로 나가지 않도록 온프레미스 LLM 또는 feature 만 다루는 구조                                   |
| 장비 연동 | SECS/GEM, FDC/EES, MES, APC (R2R) 와의 interface                                                                |
| 성과 지표 | 스펙 적중률, 웨이퍼당 보정 비용, 엔지니어 개입 횟수, 소모품 수명 종료 구간 성능                                 |

## 10. Roadmap

장비 1대의 L1 조언형과 동일 기종 다챔버의 Global + Local model 에서 시작하여, 재학습 자동화와 L2 를 거쳐 이기종·신규 장비와 다른 공정으로 넓힌다.

Table 11. Recommended roadmap

| Phase   | Duration (estimated) | Autonomy | Scope                                                                                                                          |
| :---:   | :---:                | :---:    | :---:                                                                                                                          |
| Phase 1 | 3~6개월              | L1       | 기존 두께 예측 model 과 역산을 도구로 감싼 장비 에이전트, 동일 기종·다챔버의 Global + Local model 적용과 chamber matching 검증 |
| Phase 2 | 6~12개월             | L2       | Drift Monitor·Adaptation·Guardrail Agent 도입, 보정 DOE·재학습 자동화, MLOps                                                   |
| Phase 3 | 12개월~              | L3       | 이기종·신규 장비의 few-shot onboarding 과 4단계 gate, LLM 기반 orchestrator, 다른 공정으로 확장                                |

---

## Appendix A. Terminology

- **APC (R2R)**: 계측값이나 예측값으로 다음 run 의 recipe 를 조정하는 공정 제어 (advanced process control, run-to-run).
- **CCD**: Axial 점과 기준점으로 구성하는 실험 설계 (central composite design).
- **Concept shift**: 입력과 출력의 관계가 바뀌는 drift.
- **Conformal prediction**: 검증 data 의 잔차 분포로 정해진 확률을 보장하는 예측 구간을 만드는 방법.
- **Covariate shift**: 입출력 관계는 그대로이고 입력 분포만 바뀌는 drift.
- **DOE**: 실험 계획 (design of experiments).
- **Domain classifier AUC**: 기준 장비와 신규 장비의 data 를 분류기가 얼마나 구분하는지를 나타내는 AUC. 0.5 에 가까울수록 두 data 가 비슷하다.
- **DTW**: 시간 축의 늘어남을 허용하여 두 trace 의 형상 차이를 재는 거리 (dynamic time warping).
- **FDC/EES**: 장비의 trace 를 수집·감시하는 fault detection and classification / equipment engineering system.
- **Golden trace envelope**: normal run 들의 trace 로 정한 step 별 허용 범위.
- **Hotelling T²**: 다변량 data 가 기준 분포의 중심에서 벗어난 정도를 재는 통계량.
- **KS test**: 두 표본의 누적 분포 차이로 분포가 같은지 검정하는 Kolmogorov-Smirnov test.
- **LoRA**: 큰 model 의 가중치는 고정하고 저차원 행렬만 학습하는 경량 adapter 방식 (low-rank adaptation).
- **Mahalanobis distance**: 변수 간 공분산을 고려한 다변량 거리.
- **MAML**: 적은 data 로 빠르게 적응하도록 초기 가중치를 학습하는 meta-learning 방법 (model-agnostic meta-learning).
- **MES**: 공장의 생산 실행 시스템 (manufacturing execution system).
- **Mixed-effects model**: 공통 효과와 장비·챔버별 random effect 를 함께 추정하는 통계 model.
- **MLOps**: Model 의 학습·검증·배포·감시를 자동화하는 운영 체계.
- **PCA**: 상관된 변수를 소수의 직교 성분으로 줄이는 주성분 분석 (principal component analysis).
- **PM**: 예방 정비 (preventive maintenance).
- **PSI**: 두 분포의 구간별 비율 차이로 분포 이동을 재는 지표 (population stability index).
- **RMSE**: 예측 오차 제곱 평균의 제곱근 (root mean square error).
- **Seasoning**: PM 뒤 챔버 상태를 안정시키려고 진행하는 사전 run.
- **SECS/GEM**: 반도체 장비와 host 사이의 표준 통신 protocol.
- **Shadow 검증**: 후보 model 을 운영 model 과 나란히 돌려 결과만 비교하고 제어에는 쓰지 않는 검증.
- **SHAP**: 예측값을 feature 별 기여도로 나누어 설명하는 방법 (SHapley additive explanations).
- **SOA**: 추천을 허용하는 공정 조건의 운전 영역 (safe operating area). SOA 95% 는 그 영역의 95% 범위이다.
- **THK**: 증착 막의 두께 (thickness).
- **Virtual metrology**: 계측 전에 공정 data 로 계측값을 예측하는 방법.
- **Wasserstein distance**: 한 분포를 다른 분포로 옮기는 최소 비용으로 정의한 분포 간 거리.
- **소모품 수명**: 타깃 등 증착 장비의 소모 부품이 누적 사용으로 닳은 정도.
