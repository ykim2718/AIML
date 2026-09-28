# Prefect As An AI Agent Backend
Rev. 1 | Created: 2026-09-27 | Updated: 2026-09-27 21:47 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Placement](#31-placement)
- [4. Backend Composition](#4-backend-composition)
  - [4.1 Without Prefect](#41-without-prefect)
  - [4.2 With Prefect](#42-with-prefect)
- [5. Function Comparison](#5-function-comparison)
- [6. Strength](#6-strength)
- [7. Application](#7-application)
- [8. Design Record](#8-design-record)
- [9. Further Work](#9-further-work)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

- **Problem Statement**: AI Agent 와 AI Orchestrator 를 구축하는 데 Prefect workflow 를 활용하는 방안과, 그것으로 강점을 만들어 내는 방안이 없다.
- **Goal**: AI agent 의 frontend 와 backend 역할을 가르고, backend 의 역할에 대해 Prefect 가 특별히 무엇을 어떻게 할 수 있는지를 적어, manager 또는 designer 가 그 답을 AI Orchestrator design 에 반영하게 한다.
- **Non-Goal**: Agent 의 prompt 와 tool 을 설계하는 방법은 다루지 않고, 구현 code 도 주지 않으며, self-hosted Prefect Server 가 담지 않는 기능도 다루지 않는다.

## 2. Summary

Orchestrator 는 AI agent backend 가운데 fleet 범위를 지는 삼분의 일이며, Prefect 는 design 이 code 로 적어야 할 그 삼분의 일을 제품으로 내놓는다. 그 자리에 Prefect 를 적은 design 은 직접 지은 orchestrator 가 닿지 못하는 다섯 가지 — suspension, idempotent rerun, 선언으로 두는 rate limiting, per-step observability, one admission path — 를 얻고, 제약 셋을 진다. 그 가운데 open source server 가 담지 않는 인증이 design 이 먼저 답하는 것이다.

Frontend 는 세 역할을 그대로 두고, Prefect 를 쓸 때 할 일 하나를 얻는다. 멈춘 실행이 기다리는 답이 frontend 를 지나 들어온다. 이 문서의 나머지가 열 역할과 그 범위, 두 구성, 그 둘에 걸쳐 견준 아홉 기능, 다섯 강점, 그리고 design 이 적는 여섯 결정이다.

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

범위를 한 단계 넓힐 때마다, 앞 범위보다 오래 사는 상태를 둘 자리가 하나 더 든다. 단계 범위는 process 밖에 아무것도 필요하지 않고, 실행 범위는 process 가 죽어도 잃지 않는 저장소를 필요로 하며, fleet 범위는 모든 process 보다 오래 살면서 무엇이 있었는지 물을 수 있는 service 를 필요로 한다.

### 3.1 Placement

<a id="table-1"></a>
Table 1. Each role, the layer that holds it, and what fixes it

| Role           | Layer        | Scope       | What fixes it                                  |
| :------------: | :----------: | :---------: | :--------------------------------------------: |
| Intent         | Frontend     | One request | Backend 이 받아들이는 요청 schema              |
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

두 구성은 같은 추론 loop 를 돌리며, 그 loop 가 어디에 사는지에서 갈린다. Prefect 가 없으면 loop 는 요청에 답한 process 안에서 돌고, Prefect 가 있으면 loop 는 worker 가 제 기반 위에서 시작하는 flow 이며 요청은 그것을 청하기만 한다. 둘은 [Fig 2](#fig-2) 에 그렸다.

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

### 4.1 Without Prefect

Agent framework 는 [Fig 1](#fig-1) 의 안쪽 네 책임을 지고 바깥 셋을 팀에 남긴다. Checkpointer 는 graph 상태의 snapshot 을 super-step 마다 thread id 아래 저장하며, 그것이 멈춘 실행을 다시 시작하게 하고 사람이 한 단계를 끊어 들여다보고 승인하게 한다 [[6](#ref-6)]. Node 에 붙인 retry policy 는 그 node 를 지수 backoff 로 다시 시도한다 [[6](#ref-6)].

Framework 곁에 팀이 직접 쓰는 것이 fleet 범위의 절반이다. 요청 말고 무엇이 실행을 시작하는가, 속도 제한이 걸린 API 에 대해 LLM 호출을 몇 개까지 띄울 수 있는가, 어젯밤에 무엇이 돌았는가를 framework 는 말하지 않으므로 scheduler 와 semaphore 와 기록용 표를 직접 쓰게 되고, 그 셋은 그때부터 팀이 소유하는 구성 요소가 된다.

### 4.2 With Prefect

Prefect 는 server 와 worker 를 더하고, agent loop 는 tool 호출이 task 인 flow 가 된다. Deployment 은 flow 를 어디서·언제·어떻게 돌릴지 적어, 손으로 부르는 함수를 API 가 관리하는 대상으로 바꾸며, schedule 과 UI 와 automation 과 REST API 가 그것을 켠다 [[1](#ref-1)]. Work pool 은 기반을 가리키고, worker 는 그 pool 을 살펴 실행을 그 기반 위에서 시작하고 끝까지 지켜본다 [[1](#ref-1)].

Self-hosted server 는 API, UI, scheduling, work pool, 그리고 event 와 automation engine 을 담는다 [[4](#ref-4)]. Agent 를 이렇게 감싸는 일은 팀이 지어내는 방식이 아니라 이미 나와 있는 통합이다. Agent 의 tool 이 자동으로 task 로 감싸이며, 그러면 tool 호출마다 제 재시도와 제 cache 된 결과와 실행 기록의 제 줄을 갖는다 [[5](#ref-5)].

## 5. Function Comparison

아홉 기능이 두 구성에 모두 있고, 차이는 있느냐가 아니라 무엇이 그것을 지느냐다. [Table 2](#table-2) 는 왼쪽에서 오른쪽으로 같은 요구가 두 번 채워지는 것으로 읽힌다.

<a id="table-2"></a>
Table 2. The same function in each composition

| Function                          | Agent framework alone                                 | Self-hosted Prefect added                              |
| :-------------------------------: | :---------------------------------------------------: | :----------------------------------------------------: |
| Step retry                        | 한 graph 실행 안, node 에 붙인 retry policy           | Task 마다 붙는 `retries` 와 `retry_delay_seconds`      |
| Resume after a crash              | Checkpointer 가 thread 를 마지막 super-step 에서 재생 | 같은 checkpoint, 그리고 server 가 쥔 실행 상태         |
| Skip work already done            | Node 안에 직접 씀                                     | Result caching 이 앞선 결과를 불러 다시 돌지 않음      |
| Human approval                    | Interrupt, 그리고 팀이 잇는 resume 호출               | `wait_for_input` 을 받는 `pause_flow_run`, API 로 답함 |
| Release the process while waiting | Process 가 thread 를 쥐고 있음                        | `suspend_flow_run` 이 빠져나가고, 입력이 다시 시작함   |
| What starts a run                 | 팀이 이어 붙인 web 요청                               | 요청·schedule·automation 이 켜는 deployment            |
| Calls in flight and call rate     | 직접 쓴 semaphore                                     | Global concurrency limit 과 rate limit                 |
| Run history                       | 쥐고 있지 않음                                        | Server 의 UI 에 담긴 모든 flow run 과 task run         |
| Where a run executes              | 요청에 답한 web process                               | Work pool, 그리고 그것을 살피는 worker                 |

## 6. Strength

아홉 가운데 다섯이 정도가 아니라 종류에서 다르며, 각각은 그것을 이미 담은 제품을 이름으로 적어야만 design 이 명세할 수 있는 것이다.

**Suspension** 은 process 를 놓아준다. `pause_flow_run` 은 기다리는 동안 flow 를 살려 두고, `suspend_flow_run` 은 빠져나가 기반을 내릴 수 있게 하며, 입력이 닿으면 실행이 다시 시작된다 [[2](#ref-2)]. 하루를 기다리는 HITL 단계가 기다리는 동안 아무것도 쓰지 않으므로, thread 를 쥐고 있으면 하루 내내 process 하나가 드는 자리와 갈린다.

**Idempotent rerun** 은 다시 돌리는 일을 안전하게 만든다. Idempotency 는 Prefect 의 transactional orchestration 에서 오며, 문맥이 같을 때 재실행이 다시 돌지 않고 앞선 결과를 불러오게 하고, 그래서 다시 시도한 agent 실행이 같은 tool 호출에 LLM 값을 두 번 치르지 않는다 [[5](#ref-5)]. 팀이 직접 쓴 장치는 생각해 낸 경우만 막는다.

**Rate limiting** 은 code 가 아니라 선언이다. Global concurrency limit 이 띄울 수 있는 호출 수를 묶고, rate limit 이 초당 slot 회복량으로 속도를 고르며, 둘은 flow 안이 아니라 어떤 Python code 에서도 쓰이므로 task 로 감싸지 않은 tool 도 상한 안에 든다 [[3](#ref-3)].

**Per-step observability** 는 재시도를 주는 그 감싸기에서 함께 나온다. Tool 호출 하나가 task 이므로 하나씩 따로 실행 기록에 보이고 하나씩 따로 다시 시도된다 [[5](#ref-5)]. Agent 실행이 실패했다는 것을 아는 것과, 어느 tool 호출이 어느 입력에서 실패했는지를 아는 것이 그 차이다.

**One admission path** 는 모든 방아쇠를 한 자리로 받는다. Deployment 하나가 대화형 요청과 야간 schedule 과 event 기반 automation 에 함께 답하므로 [[1](#ref-1)], 야간 일괄 실행과 대화 요청이 서로 갈라지는 두 벌이 아니라 같은 code 를 돌린다.

## 7. Application

Prefect 는 실행이 그것을 시작한 요청보다 오래 살 때 값을 하고, 실행이 요청 안에서 끝날 때는 얻는 것보다 값이 크다. 아래 세 조건이 design 이 어느 경우에 있는지를 가른다.

**Assumption** 은 backend 가 제 process 를 가질 수 있다는 것이다. Worker 는 work pool 을 살펴 실행을 기반 위에서 시작하는 client 쪽 process 이므로 [[1](#ref-1)], 오래 사는 process 를 금지하는 배포 대상에서는 [Fig 2](#fig-2) 의 구성이 가운데를 잃는다.

**Breaking condition** 은 인증이다. Open source server 에는 사용자도 인증도 없어 UI 나 API 에 닿는 누구나 전체 권한을 갖는다 [[4](#ref-4)]. 그래서 self-hosted server 는 사설망 안에 두거나 인증하는 proxy 뒤에 둔다. Webhook 은 Prefect Cloud 의 기능이므로 [[4](#ref-4)], 외부 system 에서 실행을 시작해야 하는 self-hosted backend 는 그 event 를 API 로 보내는 자리를 스스로 둔다.

**Exclusion** 은 실행이 LLM 호출 하나로 끝나고 그 결과를 뒤에 아무도 찾지 않는 agent 다. Server 와 worker 는 운영할 구성 요소 둘이고, 도착한 요청 안에서 끝나는 실행은 그 둘이 쥘 상태를 남기지 않는다.

## 8. Design Record

Design 이 이 문서에서 가져가는 결정은 여섯이며, 각각은 구현이 아니라 이름이 붙은 꼭지가 답한다. [Table 3](#table-3) 이 검토자가 AI Orchestrator design 을 대고 확인하는 목록이다.

<a id="table-3"></a>
Table 3. What an AI orchestrator design records, and where this document answers it

| Decision           | What it fixes                                        | Answered in                         |
| :----------------: | :--------------------------------------------------: | :---------------------------------: |
| Boundary           | 역할마다 frontend 와 backend 의 어느 쪽에 놓이는가   | [Fig 1](#fig-1)                     |
| Orchestrator scope | 어느 책임을 code 가 아니라 제품이 지는가             | [Table 1](#table-1)                 |
| Composition        | Fleet 범위의 셋을 직접 짓는가, 이름으로 적는가       | [Chapter 4](#4-backend-composition) |
| Capability claim   | Design 이 약속해도 되는 강점이 무엇인가              | [Chapter 6](#6-strength)            |
| Access             | 인증이 없는 server 를 어디에 두는가                  | [Chapter 7](#7-application)         |
| Inbound events     | Webhook 없이 외부 system 이 실행을 어떻게 시작하는가 | [Chapter 7](#7-application)         |

한 행을 비워 둔 design 은 그 결정을 그 부분을 구현하는 사람에게 넘기며, 그러면 일괄 경로와 대화 경로에서 답이 갈린다.

## 9. Further Work

- **나와 있는 통합으로 agent 하나를 감싸 보기**. 그 통합은 agent 를 flow 로, tool 을 task 로 감싸고 LLM 호출에 지수 backoff 재시도를 기본으로 주므로 [[5](#ref-5)], 이 문서가 적은 감싸기 code 가 없어진다. 그 framework 로 이미 쓴 agent 하나와, 그것을 가리킬 self-hosted server 가 필요하다.
- **Tool 마다 cache 정책을 정하기**. 재실행이 LLM 을 다시 부르지 않고 cache 된 결과를 불러오는 것은 문맥이 같을 때의 Prefect 기본 동작이므로 [[5](#ref-5)], 남는 판단은 어느 tool 을 cache 해도 되는지다. Tool 마다 같은 입력이 같은 출력을 내야 하는지에 대한 판단이 필요하다.
- **Event relay 를 세우기**. Webhook 이 Cloud 전용이라 [[4](#ref-4)] self-hosted backend 에는 외부 event 가 들어올 길이 없고, automation 은 API 에 닿은 event 에만 반응한다. Agent 실행을 시작할 system 의 목록이 필요하다.

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
[6] LangChain. [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers). LangGraph documentation.

---

## Appendix A. Terminology

- **Agent framework**: LLM 호출과 tool 선택 loop 를 팀이 써 넣는 library.
- **Automation**: 맞는 event 가 닿으면 미리 정한 동작을 시작하는 Prefect 의 규칙.
- **Checkpointer**: Graph 상태를 단계마다 저장해, 멈춘 실행이 그 자리에서 다시 시작하게 하는 구성 요소.
- **Deployment**: 어디서·언제·어떻게 돌릴지가 붙은 flow. API 가 관리하는 대상이 된다.
- **Flow**: Prefect 가 한 실행으로 다루는 함수.
- **HITL (Human In The Loop)**: 사람의 입력을 기다린 뒤에 이어 가는 실행.
- **Idempotency**: 같은 입력으로 다시 돌려도 결과와 부수 효과가 한 번 돌린 것과 같은 성질.
- **Orchestrator**: Backend 가운데 한 실행이 아니라 모든 실행을 보는 쪽. 무엇이 실행을 시작하는가, 호출이 몇 개 떠 있는가, 실행 기록이 무엇을 담는가를 진다.
- **Prefect Server**: API, UI, scheduler, 그리고 event 와 automation 을 담은 self-hosted orchestration backend.
- **Rate limit**: 정해진 시간 안에 나갈 수 있는 호출 수의 상한.
- **Result caching**: 입력이 같을 때 다시 돌지 않고 앞선 결과를 불러오는 것.
- **Super-step**: Graph 가 상태 snapshot 하나를 저장하는 실행 단위.
- **Task**: Flow 안에서 Prefect 가 따로 재시도하고 cache 하고 기록하는 단위.
- **Thread**: Checkpointer 가 한 대화의 상태를 모아 두는 단위.
- **Work pool**: Flow 실행이 어느 기반 위에서 돌지 가리키는 Prefect 설정.
- **Worker**: Work pool 을 살펴 예정된 실행을 그 기반 위에서 시작하는 client 쪽 process.
