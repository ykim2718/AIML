# Semiconductor Deposition Equipment AI Agent Proposal
Rev. 2 | Created: 2026-10-14 | Updated: 2026-10-17 22:44 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Equipment Agent](#3-equipment-agent)
  - [3.1 Agent Loop](#31-agent-loop)
  - [3.2 Role of LLM](#32-role-of-llm)
  - [3.3 Autonomy Level](#33-autonomy-level)
- [4. Multi-Equipment Orchestrator](#4-multi-equipment-orchestrator)
- [5. Key Issues](#5-key-issues)
- [6. Roadmap](#6-roadmap)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

- **Problem Statement**: 당사가 PVD 장비용으로 개발한 THK 예측·역산 model 은 엔지니어가 결과를 읽고 다음 웨이퍼의 조건을 직접 정해야 하므로, 증착 장비마다 웨이퍼 단위의 판단과 보정 작업이 사람에게 남는다.
- **Goal**: 검증된 ML model 을 도구로 쓰는 AI agent 가 감지·예측·추천·드리프트 감시·보정·보고를 웨이퍼 단위로 수행하고, 엔지니어 개입을 자율성 단계에 따라 줄인다.
- **Non-Goal**: LLM 이 THK 를 직접 예측하거나 공정 조건 값을 산출하는 구조는 다루지 않는다.

## 2. Summary

증착 장비 1대를 맡는 장비 에이전트를 먼저 만들고, 같은 도구 interface 를 공유하는 복수 장비 위에 AI orchestrator 를 두는 2단계 구성을 제안한다.
예측과 역산은 검증된 ML model 이 맡고 LLM 은 도구 호출 판단과 결과 설명만 맡으며, 자율성은 L1 조언형에서 L3 자동 run-to-run 까지 단계적으로 올린다.
Phase 1 은 3~6개월 안에 기존 THK 예측 model 과 역산을 도구로 감싼 L1 조언형 에이전트를 장비 1대에 적용한다.

전체 구성은 아래와 같다.

```text
                 +--------------------------------------+
                 |          AI Orchestrator             |
                 |  calibration budget / transfer       |
                 |  learning / chamber matching /       |
                 |  maintenance schedule / templates    |
                 +------------------+-------------------+
                                    |
             predict / recommend / health / calibrate
                                    |
        +---------------------------+---------------------------+
        |                           |                           |
+-------+--------+          +-------+--------+          +-------+--------+
| Equipment      |          | Equipment      |          | Equipment      |
| Agent #1       |          | Agent #2       |   ...    | Agent #N       |
| (Deposition)   |          | (Deposition)   |          | (Deposition)   |
+-------+--------+          +-------+--------+          +-------+--------+
        |                           |                           |
   SECS/GEM, FDC/EES, MES      SECS/GEM, FDC/EES, MES      SECS/GEM, FDC/EES, MES
```

<a id="fig-1"></a>
Fig 1. Two-tier agent architecture

아래 층은 장비마다 한 개의 장비 에이전트이고, 위 층의 orchestrator 는 네 가지 공통 도구 interface 로만 장비 에이전트와 주고받는다.

## 3. Equipment Agent

장비 에이전트는 증착 장비 1대를 맡아 웨이퍼마다 감지에서 보고까지 여섯 단계의 loop 를 돈다.

### 3.1 Agent Loop

웨이퍼 한 장이 끝날 때마다 아래 loop 가 한 번 돈다.

```text
  Trace data in
       |
       v
  [1] Detect  ---->  [2] Predict  ---->  [3] Decide / Recommend
                                                  |
                                                  v
  [6] Report  <----  [5] Calibration plan  <----  [4] Drift monitor
       |                                          ^
       v                                          |
  Engineer                              Metrology data in
```

<a id="fig-2"></a>
Fig 2. Per-wafer loop of the equipment agent

단계마다 하는 일은 Table 1 과 같다.

Table 1. Steps of the agent loop

| Step               | Action                                                                                                         |
| :---:              | :---:                                                                                                          |
| Detect             | 공정 종료 직후 Trace data 수신, 결측·이상치·센서 이상 점검                                                         |
| Predict            | THK 예측 model (PVD 과제의 H3/M8) 로 THK 예측과 예측 불확실성 산출 (계측 전 virtual metrology)                      |
| Decide / Recommend | 다음 웨이퍼의 공정 조건 (PVD 과제의 TM/TS/DT) 역산, SOA 95% 영역 안의 조건만 제안 (보고서 4.3절의 선제적 제어 자동화)   |
| Drift monitor      | 계측값 수신 시 잔차 추세, 소모품 수명 (PVD 의 Target life) 대비 SOA 이동, feature 분포 변화 확인                      |
| Calibration plan   | 드리프트 감지 시 보정 DOE 제안 (예: 보정 웨이퍼 5매, CCD axial 4점 + 기준점), 재학습한 후보 model 이 시간순 검증을 통과할 때만 교체 요청 |
| Report             | 엔지니어에게 자연어로 상황 요약 (PVD 예: "Target life 1,450kWh, DT 0.4초 증가 권장, 신뢰도 95%")                           |

Drift monitor 는 계측값이 들어올 때만 돌고, 나머지 단계는 웨이퍼마다 돈다.

### 3.2 Role of LLM

예측과 역산은 검증된 ML model 이 도구로서 수행하고, LLM 은 어떤 도구를 언제 부를지 판단하고 그 결과를 설명하는 역할만 맡는다.

에이전트는 feature 이름 대신 데이터 기반 도구의 출력만으로 판단하므로, 암호화된 고객 데이터처럼 도메인 지식을 쓸 수 없는 경우에도 동작한다.

### 3.3 Autonomy Level

자율성은 세 단계로 나누어 올리며, 단계마다 사람이 승인하는 범위가 줄어든다.

Table 2. Autonomy levels

| Level                | Automatic action           | Human approval                 |
| :---:                | :---:                      | :---:                          |
| L1 조언형            | 없음 (추천만)              | 모든 조정                      |
| L2 반자동            | SOA 95% 안의 조정          | SOA 95% 밖의 조정              |
| L3 자동 run-to-run   | SOA 안의 조정, 재학습      | Model 교체, SOA 밖 조건        |

고객은 L1 에서 추천의 적중을 확인한 뒤 다음 단계로 넘어가므로, 자율성을 올리는 근거가 운영 실적으로 쌓인다.

## 4. Multi-Equipment Orchestrator

장비 에이전트들이 같은 도구 interface (predict, recommend, health, calibrate) 를 공유하면, 그 위의 orchestrator 가 장비 사이의 판단을 맡는다.

Table 3. Orchestrator functions

| Function                | Description                                                                                         |
| :---:                   | :---:                                                                                               |
| 보정 웨이퍼 예산 배분    | 여러 장비 중 드리프트가 큰 장비에 보정 웨이퍼를 우선 배정                                                   |
| 신규 장비 빠른 시작      | 같은 기종 장비의 model 을 전이학습하여, 현재 model 고정에 필요한 초기 110장을 줄임 (보고서 차기 과제 "초기 학습 데이터 축소" 와 직결) |
| 장비 간 매칭             | 동일 레시피에서 챔버별 THK 차이 보정                                                                       |
| 정비 일정 최적화         | 소모품 수명에 따른 SOA 축소를 보고 소모품 교체 시점 (PVD 의 타깃 교체) 을 장비별로 조율                         |
| 공통 지식 축적           | Feature 추출 방식, 실패 사례, 검증 기준을 template 으로 재사용                                               |

## 5. Key Issues

실현에는 안전 조건, 보안, 장비 연동, 성과 지표의 네 가지를 먼저 정해야 하며, 이 중 장비 연동이 보통 가장 오래 걸린다.

Table 4. Key issues for deployment

| Issue      | Approach                                                               |
| :---:      | :---:                                                                  |
| 안전 조건  | SOA 밖 조건 실행 금지, model 교체 전 검증 게이트, 모든 결정 log 기록        |
| 보안       | 고객 데이터가 외부로 나가지 않도록 온프레미스 LLM 또는 feature 만 다루는 구조 |
| 장비 연동  | SECS/GEM, FDC/EES, MES 와의 interface                                    |
| 성과 지표  | 스펙 적중률, 웨이퍼당 보정 비용, 엔지니어 개입 횟수, 수명종료 구간 성능       |

## 6. Roadmap

장비 1대의 L1 조언형에서 시작하여, 보정 자동화와 L2 를 거쳐 복수 장비와 다른 공정으로 넓힌다.

Table 5. Recommended roadmap

| Phase   | Duration (estimated) | Scope                                                         |
| :---:   | :---:                | :---:                                                         |
| Phase 1 | 3~6개월              | 기존 THK 예측 model 과 역산을 도구로 감싼 L1 조언형 에이전트, 장비 1대 |
| Phase 2 | 6~12개월             | 드리프트 감지·보정 DOE·재학습 자동화, L2 반자동, MLOps          |
| Phase 3 | 12개월~              | 복수 장비 에이전트 + orchestrator, 전이학습, 다른 공정으로 확장  |

---

## Appendix A. Terminology

- **CCD**: 보정 DOE 에서 쓰는 실험 설계 (central composite design). Axial 점과 기준점으로 구성한다.
- **DOE**: 실험 계획 (design of experiments).
- **FDC/EES**: 장비의 trace 를 수집·감시하는 fault detection and classification / equipment engineering system.
- **H3/M8**: 당사 PVD 과제에서 개발·검증한 THK 예측 model.
- **MES**: 공장의 생산 실행 시스템 (manufacturing execution system).
- **MLOps**: Model 의 학습·검증·배포·감시를 자동화하는 운영 체계.
- **PVD**: 물리 기상 증착 (physical vapor deposition). 당사 과제의 대상 장비이다.
- **SECS/GEM**: 반도체 장비와 host 사이의 표준 통신 protocol.
- **SOA**: 추천을 허용하는 공정 조건의 운전 영역 (safe operating area). SOA 95% 는 그 영역의 95% 범위이다.
- **Target life**: PVD 타깃의 누적 사용량 (kWh).
- **THK**: 증착 막의 두께 (thickness).
- **TM/TS/DT**: PVD 과제에서 역산으로 정하는 다음 웨이퍼의 공정 조건 parameter. DT 는 초 단위로 조정한다.
- **Virtual metrology**: 계측 전에 공정 data 로 계측값을 예측하는 방법.
- **소모품 수명**: 증착 장비의 소모 부품 (PVD 의 타깃 등) 이 누적 사용으로 닳는 정도.
