# Prefect As An AI Agent Backend
Rev. 14 | Created: 2026-09-27 | Updated: 2026-09-27 23:34 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Placement](#31-placement)
- [4. Backend Composition](#4-backend-composition)
- [5. Function Comparison](#5-function-comparison)
- [6. Strength](#6-strength)
- [7. Application](#7-application)
- [8. Benchmarking](#8-benchmarking)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. What Prefect Does In An Agent Backend](#appendix-b-what-prefect-does-in-an-agent-backend)
- [Appendix C. How An Event Trigger Is Done In Prefect](#appendix-c-how-an-event-trigger-is-done-in-prefect)
- [Appendix D. Agent Frameworks In Use As Of September 2026](#appendix-d-agent-frameworks-in-use-as-of-september-2026)

## 1. Purpose

- **Problem Statement**: AI Agent 와 AI Orchestrator 를 구축하는 데 Prefect workflow 를 활용하는 방안과, 그것으로 강점을 만들어 내는 방안이 없다.
- **Goal**: AI Orchestrator 와 AI agent 의 frontend 와 backend 역할을 가르고, Prefect 가 backend 의 역할에 대해 특별히 무엇을 어떻게 할 수 있는지를 적어, manager 또는 designer 가 benchmarking 자료를 만든다.
- **Non-Goal**: Agent 의 prompt 와 tool 을 설계하는 방법은 다루지 않고, 구현 code 도 주지 않으며, Prefect 밖의 orchestrator 에 점수를 매기지 않는다.

## 2. Summary

Orchestrator 는 AI agent backend 의 일곱 책임 가운데 바깥 셋, 곧 한 실행이 아니라 모든 실행을 쥐는 셋이며, Prefect 는 팀이 직접 지어야 할 그 셋을 제품으로 내놓는다. Prefect 가 함께 가져오는 다섯 가지 — suspension, idempotent rerun, 선언으로 두는 rate limiting, per-step observability, one admission path — 와 그에 딸린 제약 셋이 benchmarking 자료가 제품을 견주는 행이다.

Frontend 는 세 역할을 그대로 두고, Prefect 를 쓸 때 할 일 하나를 얻는다. 멈춘 실행이 기다리는 답이 frontend 를 지나 들어온다. 기준선으로 삼은 agent framework 는 LangGraph 다. [Table 2](#table-2) 의 왼쪽 열이 기대는 두 가지, checkpointer 와 node 에 붙인 retry policy 를 vendor 문서로 확인할 수 있어 고른 것이며 [[6](#ref-6)], 지금 쓰이는 다른 framework 는 [Appendix D](#appendix-d-agent-frameworks-in-use-as-of-september-2026) 에 적었다. [Appendix B](#appendix-b-what-prefect-does-in-an-agent-backend) 가 Prefect 가 agent backend 에서 하는 아홉 가지와, agent framework 와 API server 에 남기는 셋을 적고, [Appendix C](#appendix-c-how-an-event-trigger-is-done-in-prefect) 가 그 가운데 이벤트 트리거를 어떻게 붙이는지 보인다.

## 3. Taxonomy and its Hierarchy

Frontend 와 backend 의 경계는 사용자의 뜻이 확정된 뒤, 첫 LLM 호출 앞에 놓이므로 추론 loop 는 backend 의 몫이다. 열 가지 역할이 두 층에 갈려 frontend 에 셋, backend 에 일곱이 놓이며 그 일곱의 바깥 셋이 orchestrator 다. 일곱은 각자가 쥐어야 하는 범위 — 한 단계, 한 실행, 모든 실행 — 의 순서로 늘어선다.

누가 어느 책임을 지는지는 범위가 정한다. Framework 는 한 graph 실행을 보므로 단계 범위와 실행 범위의 책임을 진다. Orchestrator 는 모든 실행을 보는 쪽이며, fleet 범위의 세 책임이 곧 그것의 정의다. 열 역할과 각각의 층, 그리고 각 역할이 정하는 것은 [Fig 1](#fig-1) 에 그렸다.

```text
LAYER          ROLE              WHAT IT DECIDES                            SCOPE

Frontend  >    Intent            What the user asked, in the backend's form   one request
               Approval          The answer a paused run is waiting for       one run
               Presentation      What the user sees of the result             one request
      |   one request crosses the boundary
      v
Backend   >    Reasoning         Which action the LLM picks next              one step
               Tool execution    The action actually carried out              one step
      |   the two above are what the agent is
      v
               State             The point a stopped run resumes from         one run
               Recovery          Which failed step is tried again, once only  one run
      |   the two above are what makes one run survive
      v
               Admission         What starts a run: request, schedule, event  every run
               Throughput        How many calls run at once, and how fast     every run
               Record            What ran, when, and with which result        every run
      |
      +-- the three above are the orchestrator
```

<a id="fig-1"></a>
Fig 1. The three frontend roles, the seven backend responsibilities ordered by scope, and the three of them that are the orchestrator

범위를 한 단계 넓힐 때마다, 앞 범위보다 오래 사는 상태를 둘 곳이 하나 더 필요하다. 단계 범위는 process 밖에 아무것도 필요하지 않고, 실행 범위는 process 가 죽어도 잃지 않는 저장소를 필요로 하며, fleet 범위는 모든 process 보다 오래 살면서 무엇이 있었는지 물을 수 있는 service 를 필요로 한다.

### 3.1 Placement

<a id="table-1"></a>
Table 1. Each role, the layer that holds it, and what fixes it

| Role           | Layer        | Scope       | What fixes it                                  |
| :------------: | :----------: | :---------: | :--------------------------------------------: |
| Intent         | Frontend     | One request | Backend 가 받아들이는 요청 schema              |
| Approval       | Frontend     | One run     | 멈춘 실행에 답하는 양식                        |
| Presentation   | Frontend     | One request | 사용자가 읽는 stream 또는 page                 |
| Reasoning      | Backend      | One step    | LLM 호출과 그것이 돌려주는 tool 선택           |
| Tool execution | Backend      | One step    | 고른 tool 이 가리키는 함수                     |
| State          | Backend      | One run     | 다시 시작한 실행이 읽는 checkpoint             |
| Recovery       | Backend      | One run     | 재시도 횟수, 그리고 재실행이 건너뛸 수 있는 것 |
| Admission      | Orchestrator | Every run   | Deployment 과 그것을 켜는 것                   |
| Throughput     | Orchestrator | Every run   | 동시 실행 상한과 호출 속도                     |
| Record         | Orchestrator | Every run   | 단계마다 남기는 실행 기록                      |

어느 agent framework 도 orchestrator 세 행을 지지 않으므로, design 은 그 세 행에 제품을 적고 나머지 행에 code 를 적는다.

## 4. Backend Composition

두 구성은 같은 추론 loop 를 돌리며, 그 loop 가 어디에 사는지에서 갈린다. Prefect 가 없으면 loop 는 요청에 답한 process 안에서 돌고, Prefect 가 있으면 loop 는 worker 가 제 infrastructure 위에서 시작하는 flow 이며 요청은 그것을 청하기만 한다. 둘은 [Fig 2](#fig-2) 에 그렸다.

```text
WITHOUT PREFECT                        WITH PREFECT (self-hosted)

[ HTTP request ]                       [ HTTP request ]  [ Schedule ]  [ Event ]
       |                                      |               |            |
       v                                      +-------+-------+------------+
[ Web process ]                                       |
       |                                              v
       +--> agent loop, in-process              [ Prefect Server ]
       +--> RetryPolicy on a node                 API, UI, scheduler,
       +--> checkpointer -> DB                    events and automations
       |                                                |
       v                                                v   run queued in a work pool
[ Response ]                                      [ Worker polls the pool ]
                                                        |
written by hand beside it:                              v
   cron or a queue, for schedules             [ Flow: the agent loop ]
   a semaphore, for the call rate                       |
   a log table, for the run history                     +--> Task per tool call
   a guard, so a rerun skips done work                  |      retries, cache, rate limit
                                                        +--> pause or suspend for input
                                                        +--> checkpointer -> DB
                                                        |
                                                        v
                                               [ Response, and a run record ]
```

<a id="fig-2"></a>
Fig 2. The same backend without Prefect and with a self-hosted Prefect Server

Agent framework 는 [Fig 1](#fig-1) 의 안쪽 네 책임을 지고 orchestrator 의 셋을 팀에 남긴다. Checkpointer 는 graph 상태의 snapshot 을 super-step 마다 thread id 아래 저장하여 멈춘 실행을 다시 시작하게 하고 사람이 한 단계를 끊어 들여다보고 승인하게 하며, node 에 붙인 retry policy 는 그 node 를 지수 backoff 로 다시 시도한다 [[6](#ref-6)]. 그러면 scheduler 와 semaphore 와 기록용 표를 직접 쓰게 되고, 그 셋은 팀이 소유하는 구성 요소가 된다.

Prefect 는 그 같은 셋을 server 와 worker 로 채우고, agent loop 는 tool 호출이 task 인 flow 가 된다. Deployment 은 flow 를 어디서·언제·어떻게 돌릴지 적어 loop 를 API 가 관리하는 대상으로 바꾸고 schedule 과 UI 와 automation 과 REST API 가 그것을 켜며, work pool 이 infrastructure 를 가리키고 worker 가 그 pool 을 살펴 실행을 그 위에서 시작한다 [[1](#ref-1)]. Self-hosted server 는 API, UI, scheduling, work pool, 그리고 event 와 automation engine 을 담는다 [[4](#ref-4)]. Pydantic 과 Prefect 가 함께 관리하는 통합이 이 감싸는 일을 해 주므로, backend 개발자가 그 code 를 쓰지 않는다. Tool 이 자동으로 task 가 되어, 호출마다 제 재시도와 제 cache 된 결과와 실행 기록의 제 줄을 갖는다 [[5](#ref-5)].

## 5. Function Comparison

열 기능은 어느 구성에서든 필요하며, [Table 2](#table-2) 가 그것을 무엇이 지는지 양쪽에 나란히 적는다. `Agent framework alone` 열은 agent framework 하나만 쓴 구성이다. Agent framework 는 LLM 호출과 tool 선택 loop 를 팀이 code 로 써 넣는 library 이며, 이 문서는 LangGraph 를 그 예로 든다 [[6](#ref-6)].

<a id="table-2"></a>
Table 2. The same function in each composition

| #   | Function                | Agent framework alone                                      | Self-hosted Prefect added                                           |
| :-: | :---------------------: | :--------------------------------------------------------: | :-----------------------------------------------------------------: |
| 1   | Step retry              | 한 graph 실행 안, node 에 붙인 retry policy                | Task 마다 붙는 `retries` 와 `retry_delay_seconds`                   |
| 2   | Resume after a crash    | Checkpointer 가 thread 를 마지막 super-step 에서 재생      | 같은 checkpoint, 그리고 server 가 쥔 실행 상태                      |
| 3   | Human approval          | Interrupt, 그리고 backend 개발자가 잇는 resume 호출        | `wait_for_input` 을 받는 `pause_flow_run`, API 로 답함              |
| 4   | Where a run executes    | 요청에 답한 web process                                    | Work pool, 그리고 그것을 살피는 worker                              |
| 5   | Suspension              | 아무도 놓아주지 않음. Web process 가 thread 를 쥐고 기다림 | `suspend_flow_run` 이 빠져나가고, 입력이 다시 시작함                |
| 6   | Idempotent rerun        | Backend 개발자가 건너뛰기 조건을 node 안에 직접 씀         | Result caching 이 앞선 결과를 불러 다시 돌지 않음                   |
| 7   | Rate limiting           | Backend 개발자가 semaphore 를 직접 써서 호출 수를 묶음     | Global concurrency limit 과 rate limit                              |
| 8   | Per-step observability  | Backend 개발자가 기록용 표를 만들어 단계마다 직접 남김     | Server 의 UI 에 담긴 모든 flow run 과 task run                      |
| 9   | One admission path      | Backend 개발자가 실행을 시작하는 web 요청을 직접 이어 붙임 | 요청·schedule·automation 이 켜는 deployment                         |
| 10  | ML pipeline integration | Backend 개발자가 재학습과 배포를 따로 둔 scheduler 로 돌림 | 재학습·배포·agent 가 한 server 위의 flow 로 돌고, 서로를 켤 수 있음 |

## 6. Strength

[Table 2](#table-2) 의 #5 부터 #9 까지 다섯 행에서 agent framework 는 그 일을 스스로 하지 못한다. Backend 개발자가 code 를 손으로 써서 채우거나, 그대로 비워 둔다. 그 다섯 행이 benchmarking 자료에서 제품을 가르며, 아래에서 같은 이름으로 하나씩 Prefect 는 대신 무엇을 하는지 적는다.

- **Suspension**: `pause_flow_run` 은 기다리는 동안 flow 를 살려 두고, `suspend_flow_run` 은 빠져나가 infrastructure 를 내릴 수 있게 하며, 입력이 닿으면 실행이 다시 시작된다 [[2](#ref-2)]. 하루를 기다리는 HITL 단계가 process 를 하나도 잡아 두지 않는다.
- **Idempotent rerun**: Prefect 의 transactional orchestration 이 문맥이 같은 재실행을 다시 돌리지 않고 앞선 결과를 불러온다 [[5](#ref-5)]. 그 idempotency 아래에서 다시 시도한 agent 실행이 같은 tool 호출에 LLM 비용을 두 번 치르지 않는다.
- **Rate limiting**: Global concurrency limit 이 띄울 수 있는 호출 수를 묶고, rate limit 이 초당 slot 회복량으로 호출 간격을 벌린다 [[3](#ref-3)]. 둘은 flow 밖의 Python code 에서도 쓰여, task 로 감싸지 않은 tool 도 상한 안에 든다.
- **Per-step observability**: Tool 호출 하나가 task 이므로 하나씩 따로 실행 기록에 보이고 하나씩 따로 다시 시도된다 [[5](#ref-5)]. 보고가 agent 실행이 실패했다는 데서, 어느 tool 호출이 어느 입력에서 실패했는지로 내려간다.
- **One admission path**: Deployment 하나가 대화형 요청과 야간 schedule 과 event 기반 automation 에 함께 답한다 [[1](#ref-1)]. 야간 일괄 실행과 대화 요청이 두 벌이 아니라 같은 code 를 돌린다.

## 7. Application

Prefect 는 실행이 그것을 시작한 요청보다 오래 살 때 제값을 하고, 실행이 요청 안에서 끝날 때는 드는 비용이 얻는 것보다 크다. 아래 세 조건이 design 이 어느 경우에 있는지를 가른다.

**Assumption** 은 backend 가 제 process 를 가질 수 있다는 것이다. Worker 는 work pool 을 살펴 실행을 infrastructure 위에서 시작하는 client 쪽 process 이므로 [[1](#ref-1)], 오래 사는 process 를 금지하는 배포 대상에서는 [Fig 2](#fig-2) 의 구성이 가운데를 잃는다.

**Breaking condition** 은 인증이다. Open source server 에는 사용자도 인증도 없어 UI 나 API 에 닿는 누구나 전체 권한을 갖는다 [[4](#ref-4)]. 그래서 self-hosted server 는 사설망 안에 두거나 인증하는 proxy 뒤에 둔다. Webhook 은 Prefect Cloud 의 기능이므로 [[4](#ref-4)], 외부 system 에서 실행을 시작해야 하는 self-hosted backend 는 그 event 를 API 로 보내는 자리를 스스로 둔다.

**Exclusion** 은 실행이 LLM 호출 하나로 끝나고 그 결과를 뒤에 아무도 찾지 않는 agent 다. Server 와 worker 는 운영할 구성 요소 둘이고, 도착한 요청 안에서 끝나는 실행은 그 둘이 쥘 상태를 남기지 않는다.

## 8. Benchmarking

Benchmarking 자료는 행을 이 문서에서 가져오고 열을 견줄 제품에서 가져온다. [Table 3](#table-3) 이 그 행 목록이며, Prefect 의 답이 이미 채워져 있다.

<a id="table-3"></a>
Table 3. The benchmarking rows, and Prefect's answer on each

| #   | Row                    | What it asks                                      | Prefect's answer                 |
| :-: | :--------------------: | :-----------------------------------------------: | :------------------------------: |
| 1   | Orchestrator scope     | Fleet 범위의 셋 가운데 제품이 어느 것을 지는가    | 셋 모두                          |
| 2   | Suspension             | 사람을 기다리는 실행이 무엇을 붙들고 있는가       | 없음. Process 가 빠져나감        |
| 3   | Idempotent rerun       | 재실행이 이미 끝난 일에 무엇을 치르는가           | 앞선 결과를 불러옴               |
| 4   | Rate limiting          | 호출 속도를 무엇으로 묶는가                       | 어떤 Python code 에서도 선언으로 |
| 5   | Per-step observability | 실패를 어디까지 좁혀 짚는가                       | Tool 호출 하나                   |
| 6   | One admission path     | Trigger 마다 code 경로가 몇 개 드는가             | Deployment 하나                  |
| 7   | Shared platform        | 같은 제품이 재학습·배포 pipeline 도 함께 돌리는가 | 돌림                             |
| 8   | Access control         | API 와 UI 를 무엇이 지키는가                      | Open source server 에는 없음     |
| 9   | Inbound events         | 외부 system 이 실행을 어떻게 시작하는가           | Cloud webhook, 또는 relay        |

한 행에 답하지 못하는 제품은 그 행을 code 에 넘기므로, 자료는 그 code 를 쓰는 비용을 제품 이름 곁에 함께 적는다.

## References

<a id="ref-1"></a>
[1] Prefect. [Deployments](https://docs.prefect.io/v3/concepts/deployments). Prefect 3 documentation.<br>
<a id="ref-2"></a>
[2] Prefect. [How to write interactive workflows](https://docs.prefect.io/v3/advanced/interactive). Prefect 3 documentation.<br>
<a id="ref-3"></a>
[3] Prefect. [How to apply global concurrency and rate limits](https://docs.prefect.io/v3/how-to-guides/workflows/global-concurrency-limits). Prefect 3 documentation.<br>
<a id="ref-4"></a>
[4] Prefect. [Cloud vs OSS feature comparison](https://www.prefect.io/compare/prefect-oss).<br>
<a id="ref-5"></a>
[5] Pydantic. [Durable execution with Prefect](https://pydantic.dev/docs/ai/integrations/durable_execution/prefect/). Pydantic AI documentation.<br>
<a id="ref-6"></a>
[6] LangChain. [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers). LangGraph documentation.<br>
<a id="ref-7"></a>
[7] Prefect. [Define event triggers](https://docs.prefect.io/v3/concepts/event-triggers). Prefect 3 documentation.<br>
<a id="ref-8"></a>
[8] LangChain. [The best AI agent frameworks in 2026](https://www.langchain.com/resources/ai-agent-frameworks).

---

## Appendix A. Terminology

- **Agent framework**: LLM 호출과 tool 선택 loop 를 팀이 써 넣는 library. LangGraph 가 그 예다.
- **Automation**: 맞는 event 가 닿으면 미리 정한 동작을 시작하는 Prefect 의 규칙.
- **Checkpointer**: Graph 상태를 단계마다 저장해, 멈춘 실행이 그 자리에서 다시 시작하게 하는 구성 요소.
- **Deployment**: 어디서·언제·어떻게 돌릴지가 붙은 flow. API 가 관리하는 대상이 된다.
- **FDC (Fault Detection and Classification)**: 장비 센서 trace 를 지켜보다가 한계를 벗어나면 알람을 내는 fab 쪽 시스템.
- **Flow**: Prefect 가 한 실행으로 다루는 함수.
- **HITL (Human In The Loop)**: 사람의 입력을 기다린 뒤에 이어 가는 실행.
- **Idempotency**: 같은 입력으로 다시 돌려도 결과와 부수 효과가 한 번 돌린 것과 같은 성질.
- **Jinja**: Prefect 가 event 값을 flow 의 parameter 에 끼워 넣는 데 쓰는 template 문법.
- **K8s (Kubernetes)**: Work pool 이 flow 실행을 올릴 수 있는 container platform.
- **Orchestrator**: Backend 가운데 한 실행이 아니라 모든 실행을 보는 쪽. 무엇이 실행을 시작하는가, 호출이 몇 개 떠 있는가, 실행 기록이 무엇을 담는가를 진다.
- **Prefect Server**: API, UI, scheduler, 그리고 event 와 automation 을 담은 self-hosted orchestration backend.
- **RAG (Retrieval Augmented Generation)**: 질문 시점에 찾아온 문서를 질문과 함께 LLM 에 건네어 답하는 것.
- **Rate limit**: 정해진 시간 안에 나갈 수 있는 호출 수의 상한.
- **Result caching**: 입력이 같을 때 다시 돌지 않고 앞선 결과를 불러오는 것.
- **Super-step**: Graph 가 상태 snapshot 하나를 저장하는 실행 단위.
- **Task**: Flow 안에서 Prefect 가 따로 재시도하고 cache 하고 기록하는 단위.
- **Thread**: Checkpointer 가 한 대화의 상태를 모아 두는 단위.
- **VM (Virtual Metrology)**: 계측하는 대신 공정 센서 데이터로 계측값을 예측하는 것.
- **Work pool**: Flow 실행이 어느 infrastructure 위에서 돌지 가리키는 Prefect 설정.
- **Worker**: Work pool 을 살펴 예정된 실행을 그 infrastructure 위에서 시작하는 client 쪽 process.

## Appendix B. What Prefect Does In An Agent Backend

각 행의 끝 괄호는 그 항목이 [Table 2](#table-2) 의 어느 function 인지를 가리킨다.

1. 중단된 자리에서 다시 시작: LLM·도구 호출을 task 단위로 캐시하고, 실패하면 그 지점부터 재개합니다 (#2 Resume after a crash, #6 Idempotent rerun).
2. 재시도와 제한 시간: 호출마다 정책을 적용합니다. LLM API 장애와 도구 오류에 대응합니다 (#1 Step retry).
3. 이벤트로 자동 실행: FDC 알람이나 drift 감지 같은 이벤트가 오면 Agent를 자동 실행합니다 (#9 One admission path).
4. 정해진 때마다 실행: 정기 분석과 리포트 Agent를 주기적으로 실행합니다 (#9 One admission path).
5. 여러 기계에 나누어 실행: K8s나 GPU worker에 작업을 분배합니다 (#4 Where a run executes).
6. 실행 기록 보기: 실행 이력, 로그, 상태를 UI로 추적합니다 (#8 Per-step observability).
7. 사람의 승인 기다리기: `pause_flow_run`으로 승인이 날 때까지 멈췄다가 재개합니다 (#3 Human approval, #5 Suspension).
8. ML pipeline 과 한 체계에서 운영: 재학습, 배포, Agent 실행을 함께 돌립니다 (#10 ML pipeline integration).
9. 호출 속도 제한: 동시 호출 수와 초당 호출 속도를 선언으로 묶습니다. LLM API rate limit 에 대응합니다 (#7 Rate limiting).

직접 하지 않는 일: LLM 추론 로직, 메모리와 RAG, 실시간 대화 서빙은 Agent 프레임워크와 API 서버가 담당합니다.

## Appendix C. How An Event Trigger Is Done In Prefect

방법은 두 단계입니다. `emit_event` 로 이벤트를 발행하고, `DeploymentEventTrigger` 또는 Automation 으로 그 이벤트를 받아 flow 를 실행합니다 [[7](#ref-7)].

**1. Emit the event** — FDC 시스템이나 수집기 쪽입니다.

```python
from prefect.events import emit_event

emit_event(
    event="fdc.alarm.raised",
    resource={"prefect.resource.id": "tool.ETCH01.TG1"},
    payload={"wafer_id": "W123", "sensor": "RF_power", "severity": "high"},
)
```

**2. Run the agent flow from a trigger**

```python
from prefect import flow
from prefect.events import DeploymentEventTrigger


@flow
def fdc_agent(tool_id: str, wafer_id: str, sensor: str):
    ...  # agent analysis, then the report and the notification


if __name__ == "__main__":
    fdc_agent.serve(
        name="fdc-agent",
        triggers=[
            DeploymentEventTrigger(
                expect={"fdc.alarm.raised"},
                match={"prefect.resource.id": "tool.*"},
                parameters={  # Jinja injects the event values into the flow parameters
                    "tool_id": "{{ event.resource.id }}",
                    "wafer_id": "{{ event.payload.wafer_id }}",
                    "sensor": "{{ event.payload.sensor }}",
                },
            )
        ],
    )
```

Table 4. Trigger types

| Type                | Use                                                          | Setting                                       |
| :-----------------: | :----------------------------------------------------------: | :-------------------------------------------: |
| Reactive            | 이벤트가 발생하면 즉시 실행합니다                            | 기본값                                        |
| Threshold           | N회 누적 시 실행합니다 (예: 10분 내 알람 3회)                | `threshold=3`, `within=timedelta(minutes=10)` |
| Proactive           | 이벤트가 오지 않을 때 실행합니다 (예: 데이터 수집 중단 감지) | `posture="Proactive"`                         |
| Compound / Sequence | 여러 이벤트의 조합이나 순서를 조건으로 실행합니다            | `CompoundTrigger`, `SequenceTrigger`          |
| Flow state          | 다른 flow 가 완료되거나 실패하면 연쇄 실행합니다             | `expect={"prefect.flow-run.Completed"}` 등    |

**External systems**

- Prefect Cloud: Webhook 으로 외부 HTTP 요청을 바로 이벤트로 받을 수 있습니다.
- Self-hosted (OSS): Webhook 이 없으므로 FastAPI endpoint 나 Kafka/MQ consumer 에서 `emit_event` 를 불러 이어 줍니다.

**Fab application**

- FDC 알람 → Reactive → 원인 분석 Agent
- VM 오차 초과가 10분 내 3회 → Threshold → 재학습 flow
- 센서 데이터 30분 미수신 → Proactive → 장비 점검 알림
- 재학습 완료 → Flow state → 검증 Agent → 리포트 생성

## Appendix D. Agent Frameworks In Use As Of September 2026

이 문서가 기준선으로 쓴 것은 #1 LangGraph 이며, 고른 이유는 꼭지 2 에 적었다. 출시 시점은 각 제품의 공개 발표 시점이다 [[8](#ref-8)].

Table 5. Agent frameworks and when each was first announced

| #   | Framework                 | What it is                                                         | Announced |
| :-: | :-----------------------: | :----------------------------------------------------------------: | :-------: |
| 1   | LangGraph                 | Graph 로 상태를 다룬다. 감사 이력과 사람 승인이 필요한 곳의 기본값 | 2024-01   |
| 2   | CrewAI                    | 역할을 맡은 agent 여럿을 팀으로 묶는다. Prototype 까지 가장 빠름   | 2024-01   |
| 3   | OpenAI Agents SDK         | Model 이 loop 를 끌고 간다. GPT 중심이면 마찰이 가장 적음          | 2025-03   |
| 4   | Google ADK                | Multimodal 에 강하다. Google Cloud NEXT 2025 에서 공개             | 2025-04   |
| 5   | Claude Agent SDK          | Claude Code 의 agent harness. Claude Code SDK 에서 이름을 바꿈     | 2025-09   |
| 6   | Microsoft Agent Framework | Graph 기반. Azure 와 .NET 환경의 선택                              | 2026-04   |
| 7   | Pydantic AI               | Type 안전한 Python. V2 가 durable execution 을 담음                | 2026-06   |

#7 Pydantic AI 는 Pydantic 과 Prefect 가 함께 관리하는 Prefect 통합을 가진 유일한 framework 이며 [[5](#ref-5)], 그 통합이 agent 를 flow 로, tool 을 task 로 감싸는 code 를 backend 개발자가 쓰지 않게 해 준다.
