# Model Deployment Plan
Rev. 0 | Created: 2026-10-14 | Updated: 2026-10-14 09:25 CDT

## 1. Purpose

- **Problem Statement**: THK 예측 model 을 현장에 배포하는 방식이 위치, 서빙 방식, 전환 전략의 세 축으로 갈리며, 이번 과제에 맞는 조합이 정해지지 않았다.
- **Goal**: 계측을 기다리지 않고 다음 wafer 의 공정 조건을 바꿀 수 있는 배포 조합과 그 전환 순서를 정한다.
- **Non-Goal**: Model 의 구조와 학습 방법은 다루지 않는다.

## 2. Summary

엣지 + 실시간 API 로 공정 직후 THK 를 예측하고, 재학습은 온프레미스 server 에서 한다.
전환은 Shadow 로 시작해 신뢰가 쌓이면 Canary 로 장비 1 대씩 자동 적용을 넓힌다.
잘못된 제어 한 번이 곧 wafer 손실로 이어지므로 자동 제어로 바로 넘어가지 않는다.

## 3. Taxonomy and its Hierarchy

AI model 배포는 어디에 두는가 (위치), 예측을 어떻게 내주는가 (서빙 방식), 새 model 로 어떻게 바꾸는가 (전환 전략) 의 세 축으로 나눈다.

세 축과 축마다의 값은 Fig 1 과 같다.

```text
Deployment
├── Location   : Cloud | On-premise | Edge | Hybrid
├── Serving    : Batch | Online API | Streaming | Embedded
└── Transition : Shadow | Canary | A/B test | Blue-Green | Rolling
```

<a id="fig-1"></a>
Fig 1. Three axes of model deployment

세 축은 서로 독립이므로, 한 배포는 축마다 값을 하나씩 골라 정한다.

전환 전략은 새 model 이 실제 제어에 미치는 범위에 따라 Fig 2 의 차례로 넓어진다.

```text
Transition (control exposure of the new model)
├── None     : Shadow             - predicts only, no control
├── Partial  : Canary             - one tool or some lots first
│              A/B test           - two models run side by side
└── Full     : Blue-Green         - switch at once, instant rollback
               Rolling            - replace tools or servers in sequence
```

<a id="fig-2"></a>
Fig 2. Hierarchy of transition strategies

한 단계 내려갈 때마다 새 model 의 효과를 실제 공정에서 확인할 수 있는 범위는 넓어지고, 잘못된 예측이 wafer 에 닿는 범위도 함께 넓어진다.

### 3.1 Location

Table 1. Location

| Location   | Description                                | Fab example                          |
| :--------: | :----------------------------------------: | :----------------------------------: |
| Cloud      | AWS, Azure, GCP 같은 외부 server 에서 실행 | 외부 반출이 가능한 데이터 분석       |
| On-premise | 회사 내부 데이터센터 server 에서 실행      | Fab 내부 MES·분석 server             |
| Edge       | 데이터가 생기는 장비 옆 PC 에서 실행       | PVD 장비 옆 산업용 PC                |
| Hybrid     | 학습은 server, 추론은 엣지처럼 역할을 나눔 | 재학습은 사내 server, 예측은 장비 PC |

### 3.2 Serving

Table 2. Serving

| Serving    | Description                                          | Fab example                        |
| :--------: | :--------------------------------------------------: | :--------------------------------: |
| Batch      | 모아 둔 데이터를 정해진 시간에 한꺼번에 예측         | 하루치 wafer THK 를 밤에 일괄 예측 |
| Online API | 요청이 올 때마다 즉시 예측해 돌려줌 (REST, gRPC)     | 공정이 끝나는 순간 THK 예측 요청   |
| Streaming  | 센서 데이터가 들어오는 동안 연속으로 예측 (Kafka 등) | 공정 중 이상 감지                  |
| Embedded   | Model 을 장비 software 나 PLC 안에 직접 넣음         | 장비 제어 로직에 model 내장        |

### 3.3 Transition

Table 3. Transition

| Transition | Description                                                                      |
| :--------: | :------------------------------------------------------------------------------: |
| Shadow     | 새 model 은 예측만 하고 제어에 쓰지 않으며, 기존 방식과 결과를 나란히 비교       |
| Canary     | 장비 1 대나 일부 lot 에 먼저 적용하고, 문제가 없으면 넓힘                        |
| A/B test   | 두 model 을 동시에 운영해 성능을 비교                                            |
| Blue-Green | 기존 환경과 새 환경을 함께 띄워 두고 한 번에 전환하며, 문제가 생기면 즉시 되돌림 |
| Rolling    | 여러 장비나 server 에 차례로 교체                                                |

### 3.4 Placement

Table 4. Deployment choice for this project

| Axis       | Choice                            | Reason                                       |
| :--------: | :-------------------------------: | :------------------------------------------: |
| Location   | Edge (추론) + On-premise (재학습) | 공정 직후 장비 옆에서 바로 예측해야 함       |
| Serving    | Online API                        | 공정이 끝나는 순간 THK 예측과 다음 조건 추천 |
| Transition | Shadow → Canary                   | 잘못된 제어 한 번이 wafer 손실로 이어짐      |

## 4. Plan

배포는 세 단계로 진행한다.

1. **엣지 + 실시간 API**: 장비 옆 PC 가 공정 직후 THK 를 예측하고 다음 공정 조건을 추천한다.
2. **온프레미스 server 에서 재학습**: 보정 wafer 데이터로 model 을 주기적으로 갱신한다.
3. **Shadow 로 시작, Canary 로 확대**: 처음 일정 기간은 추천값만 보여 주고 실제 적용은 엔지니어가 판단한다. 신뢰가 쌓이면 Canary 로 장비 1 대씩 자동 적용을 넓힌다.

반도체 현장은 잘못된 제어 한 번이 곧 wafer 손실로 이어지므로, 자동 제어로 바로 넘어가지 않고 Shadow → Canary 순서로 단계적으로 전환한다.

---

## Appendix A. Terminology

- **gRPC**: Google 이 만든 원격 함수 호출 protocol
- **Kafka**: 연속으로 들어오는 event 를 저장하고 전달하는 streaming platform
- **lot**: 함께 공정을 거치는 wafer 묶음
- **MES**: Manufacturing Execution System. 생산 현장의 작업과 이력을 관리하는 system
- **PLC**: Programmable Logic Controller. 장비를 제어하는 산업용 controller
- **PVD**: Physical Vapor Deposition. 물리적 방법으로 박막을 증착하는 공정
- **REST**: HTTP 위에서 자원을 주고받는 API 방식
- **THK**: Thickness. Wafer 위 박막의 두께
