# AI Deployment Locations For Semiconductor Equipment
Rev. 0 | Created: 2026-10-17 | Updated: 2026-10-17 08:28 CDT

## 1. Purpose

- **Problem Statement**: 반도체 장비 AI 의 배포 방식을 부르는 이름에 실행 위치와 소프트웨어 제공 주체가 섞여 있어, 같은 배포를 두 이름으로 부른다.
- **Goal**: 독자가 반도체 장비 AI 하나를 실행 위치 축의 값 하나로 분류할 수 있다.
- **Non-Goal**: Vendor 제품끼리 비교하지 않는다.

## 2. Summary

반도체 장비 AI 의 배포는 추론 (Inference) 을 실행하는 위치에 따라 edge, on-premise, cloud 로 나뉘고, 이 가운데 둘 이상을 함께 쓰면 hybrid 다.
소프트웨어를 자체 개발했는지 vendor 가 제공했는지는 따로 정해지는 축이어서 실행 위치를 바꾸지 않는다.

## 3. Taxonomy and its Hierarchy

배포는 서로 독립인 두 축으로 나뉘며, 배포 방식을 서로 구별하는 축은 실행 위치 하나다.

두 축과 축마다의 값은 Fig 1 과 같다.

```text
AI deployment for semiconductor equipment
├── Location : Edge | On-premise | Cloud
│              Hybrid = two or more locations in one deployment
└── Provider : In-house | Vendor
               independent of Location
```

<a id="fig-1"></a>
Fig 1. Two axes of AI deployment

한 축의 값은 다른 축의 어느 값과도 짝을 이룬다.
Vendor 의 FDC 제품을 fab 안 server 에 설치하면 on-premise 이고, 같은 제품을 장비 옆 PC 에 설치하면 edge 다.

실행 위치는 장비에서 먼 순서로 Fig 2 와 같이 놓인다.

```text
Location (distance from the tool)
├── Edge       : tool controller or a PC beside the tool
├── On-premise : servers inside the fab
└── Cloud      : servers outside the company
```

<a id="fig-2"></a>
Fig 2. Hierarchy of deployment locations

한 단계 내려갈 때마다 장비와 model 사이의 network 구간이 늘어 응답 지연이 커지고, model 이 쓸 수 있는 연산 자원도 함께 커진다.
데이터가 나가는 범위도 넓어져서, on-premise 에서는 장비 밖으로, cloud 에서는 회사 밖으로 나간다.
Hybrid 는 세 위치에 역할을 나누어 맡기는 배포로, 보통 학습과 추론을 서로 다른 위치에서 하므로 Fig 2 에 따로 단계를 두지 않는다.

### 3.1 Placement

Table 1 은 위치마다 실행 장소, 응답 지연, 데이터가 나가는 범위, 그리고 fab 에서 맡는 기능을 정리한다.

Table 1. Deployment locations

| Location   | Runs on                         | Latency          | Data travel      | Fab use                                      |
| :--------: | :-----------------------------: | :--------------: | :--------------: | :------------------------------------------: |
| Edge       | 장비 controller 또는 장비 옆 PC | 가장 짧음        | 장비 옆에 머묾   | FDC 이상 감지, wafer 단위 R2R                |
| On-premise | Fab 안 server                   | Fab network 왕복 | Fab 안에 머묾    | Virtual metrology, APC, 여러 장비 yield 분석 |
| Cloud      | 회사 밖 server                  | 가장 김          | 회사 밖으로 나감 | 반출이 허가된 데이터로 학습                  |
| Hybrid     | 위의 둘 이상                    | 역할별로 정함    | 역할별로 정함    | Fab server 에서 학습, edge 에서 추론         |

Fab use 열은 기능을 적으며, 기능은 응답 시간이나 데이터 규정이 바뀌면 다른 행으로 옮겨 간다.

## 4. Application

기능을 어느 위치에 둘지는 결과가 장비에 얼마나 빨리 반영되어야 하는지와 데이터가 어디까지 나갈 수 있는지로 정한다.
Table 2 는 위치마다 성립해야 하는 가정과 그 가정이 깨지는 조건을 적는다.

Table 2. Assumptions and breaking conditions

| Location   | Assumes                                                 | Breaks when                                                |
| :--------: | :-----------------------------------------------------: | :--------------------------------------------------------: |
| Edge       | Model 이 장비 PC 의 연산 자원과 memory 안에 들어감      | Model 이 여러 장비의 데이터를 한꺼번에 받아야 함           |
| On-premise | Fab network 왕복이 다음 조치 전에 끝남                  | 조치가 장비 자체의 제어 주기 안에 일어나야 함              |
| Cloud      | 보안 규정이 데이터의 회사 밖 반출을 허용함              | 데이터 반출이 금지되거나 결과가 제어 주기 안에 돌아와야 함 |
| Hybrid     | 학습한 model 이 version 을 유지한 채 추론 위치로 옮겨짐 | 추론하는 model 이 검증한 model 과 다름                     |

Table 2 의 가정과 조건은 소프트웨어를 누가 제공하든 실행 위치에 따라 정해진다.
Vendor 제품을 edge 에 설치하면 Edge 행의 가정과 조건을 따르고, 자체 개발한 model 을 fab server 에 두면 On-premise 행을 따른다.

---

## Appendix A. Terminology

- **APC**: Advanced process control. 측정값이나 예측값을 받아 run 사이에 공정 recipe 를 조정한다.
- **Edge**: 데이터가 생기는 장비 안이나 장비 옆의 실행 위치.
- **FDC**: Fault detection and classification. 공정 중 sensor 데이터로 장비의 abnormal 동작을 감지한다.
- **Inference**: 학습한 model 을 새 데이터에 실행해 예측값을 내는 일.
- **R2R**: Run-to-run control. 이전 run 의 결과로 다음 run 의 recipe 를 보정한다.
- **Virtual metrology**: Wafer 를 측정하는 대신 장비 sensor 데이터로 측정값을 예측하는 일.
