# Model Deployment Quick Start
Rev. 3 | Created: 2026-10-14 | Updated: 2026-10-23 09:44 UTC

## 1. Purpose

- **Problem Statement**: Model deployment 에 대한 quick start guide 가 없다.
- **Goal**: Model deployment 의 quick start guide 를 제공한다.
- **Non-Goal**: Model 의 구조와 학습 방법은 다루지 않는다.

## 2. Taxonomy

Model 배포는 위치, 서빙 방식, 전환 전략의 세 축으로 나뉘며, 세 축과 축마다의 값은 아래 Fig 1 과 같다.

```text
Deployment
├── Location   : Cloud | On-premise | Edge | Hybrid
├── Serving    : Batch | Online API | Streaming | Embedded
└── Transition : Shadow | Canary | A/B test | Blue-Green | Rolling
```

<a id="fig-1"></a>
Fig 1. Three axes of model deployment

세 축은 서로 독립이므로, 한 배포는 축마다 값을 하나씩 골라 정한다. Location 은 model 을 어디에서 실행하는지, Serving 은 예측을 어떻게 내주는지, Transition 은 새 model 로 어떻게 바꾸는지를 정한다.

전환 전략은 새 model 이 실제 제어에 미치는 범위에 따라 아래 Fig 2 의 차례로 넓어진다.

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

한 단계 내려갈 때마다 새 model 의 효과를 실제 운영에서 확인할 수 있는 범위는 넓어지고, 잘못된 예측이 실제 제어에 닿는 범위도 함께 넓어진다.

## 3. Deployment Options

세 축의 값마다 실행 방식과 쓰이는 예는 3.1 부터 3.3 까지와 같고, 조건에 맞는 조합은 3.4 와 같다.

### 3.1 Location

Table 1. Location

|  Location  |                Description                 |               Example                |
| :--------: | :----------------------------------------: | :----------------------------------: |
|   Cloud    | AWS, Azure, GCP 같은 외부 server 에서 실행 |    외부 반출이 가능한 데이터 분석    |
| On-premise |   회사 내부 데이터센터 server 에서 실행    |           사내 분석 server           |
|    Edge    |    데이터가 생기는 장비 옆 PC 에서 실행    |          장비 옆 산업용 PC           |
|   Hybrid   | 학습은 server, 추론은 엣지처럼 역할을 나눔 | 재학습은 사내 server, 예측은 장비 PC |

위치마다의 가정과 깨지는 조건은 [[1](#ref-1)] 에 있다.

### 3.2 Serving

Table 2. Serving

|  Serving   |                     Description                      |            Example             |
| :--------: | :--------------------------------------------------: | :----------------------------: |
|   Batch    |     모아 둔 데이터를 정해진 시간에 한꺼번에 예측     | 하루치 데이터를 밤에 일괄 예측 |
| Online API |   요청이 올 때마다 즉시 예측해 돌려줌 (REST, gRPC)   |  작업이 끝나는 순간 예측 요청  |
| Streaming  | 센서 데이터가 들어오는 동안 연속으로 예측 (Kafka 등) |       작업 중 이상 감지        |
|  Embedded  |     Model 을 장비 software 나 PLC 안에 직접 넣음     |  장비 제어 로직에 model 내장   |

### 3.3 Transition

Table 3. Transition

| Transition |                                   Description                                    |
| :--------: | :------------------------------------------------------------------------------: |
|   Shadow   |    새 model 은 예측만 하고 제어에 쓰지 않으며, 기존 방식과 결과를 나란히 비교    |
|   Canary   |            장비 1 대나 일부 lot 에 먼저 적용하고, 문제가 없으면 넓힘             |
|  A/B test  |                      두 model 을 동시에 운영해 성능을 비교                       |
| Blue-Green | 기존 환경과 새 환경을 함께 띄워 두고 한 번에 전환하며, 문제가 생기면 즉시 되돌림 |
|  Rolling   |                        여러 장비나 server 에 차례로 교체                         |

### 3.4 Comparison

실시간으로 다음 작업 조건을 바꾸는 model 의 배포 조합은 Table 4 와 같다.

Table 4. Deployment choice for real-time control

|    Axis    |              Choice               |                     Choose when                      |
| :--------: | :-------------------------------: | :--------------------------------------------------: |
|  Location  | Edge (추론) + On-premise (재학습) |      작업 직후 장비 옆에서 바로 예측해야 할 때       |
|  Serving   |            Online API             | 작업이 끝나는 순간 예측과 다음 조건 추천이 필요할 때 |
| Transition |          Shadow → Canary          |           잘못된 제어 한 번의 손실이 클 때           |

Shadow 로 시작해 추천값만 보여 주고, 신뢰가 쌓이면 Canary 로 장비 1 대씩 자동 적용을 넓힌다. 잘못된 제어 한 번의 손실이 크면 자동 제어로 바로 넘어가지 않는다.

## 4. Models and Libraries

Serving 방식마다 학습된 model 을 불러 예측을 내주는 library 가 다르며, 방식별 model 형식과 library 는 Table 5 와 같다.

Table 5. Models and libraries by serving method

<table>
  <tr><th align="center">Method</th><th align="center">Model</th><th align="center">Library</th><th align="center">Note</th></tr>
  <tr><td align="center">Batch</td><td align="center">joblib 으로 저장한 model</td><td align="center"><code>joblib.load</code>,<br><code>pandas.read_parquet</code>,<br><code>pandas.DataFrame.to_parquet</code></td><td align="center">저장한 model 을 불러 모아 둔 데이터를 한꺼번에 예측하고 결과를 파일로 씀</td></tr>
  <tr><td align="center" rowspan="2">Online API</td><td align="center">joblib 으로 저장한 model</td><td align="center"><code>fastapi.FastAPI</code>,<br><code>uvicorn.run</code></td><td align="center">요청마다 예측을 돌려주는 REST endpoint 를 띄움</td></tr>
  <tr><td align="center">joblib 으로 저장한 model</td><td align="center"><code>grpc.server</code></td><td align="center">같은 예측을 gRPC endpoint 로 내줌</td></tr>
  <tr><td align="center">Streaming</td><td align="center">joblib 으로 저장한 model</td><td align="center"><code>kafka.KafkaConsumer</code>,<br><code>kafka.KafkaProducer</code></td><td align="center">topic 에서 event 를 읽어 예측하고 결과를 다른 topic 으로 보냄</td></tr>
  <tr><td align="center">Embedded</td><td align="center">ONNX 로 바꾼 model</td><td align="center"><code>skl2onnx.convert_sklearn</code>,<br><code>onnxruntime.InferenceSession</code></td><td align="center">scikit-learn model 을 ONNX 로 바꿔 장비 software 안에서 실행</td></tr>
</table>

Embedded 만 model 을 ONNX 형식으로 바꾼다. 장비 software 안에서는 Python 과 scikit-learn 을 그대로 쓰기 어려워, ONNX Runtime 이 그 model 을 실행한다.

## References

<a id="ref-1"></a>
[1] yRocket. (2026). [AI Deployment Locations For Semiconductor Equipment](https://github.com/ykim2718/AIML/blob/main/AIOps/AI-Deployment-Locations/ai-deployment-locations-ko.md). *AIML repository*, GitHub.

---

## Appendix A. Terminology

- **gRPC**: Google 이 만든 원격 함수 호출 protocol
- **joblib**: Python 객체를 파일로 저장하고 불러오는 library
- **Kafka**: 연속으로 들어오는 event 를 저장하고 전달하는 streaming platform
- **lot**: 함께 처리되는 작업 단위의 묶음
- **ONNX**: Open Neural Network Exchange. 여러 framework 의 model 을 한 형식으로 저장하는 표준
- **PLC**: Programmable Logic Controller. 장비를 제어하는 산업용 controller
- **REST**: HTTP 위에서 자원을 주고받는 API 방식
- **topic**: Kafka 에서 event 를 이름별로 모아 두는 단위
