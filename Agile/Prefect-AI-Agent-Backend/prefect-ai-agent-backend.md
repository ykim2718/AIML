# Prefect As An AI Agent Backend
Rev. 21 | Created: 2026-09-27 | Updated: 2026-09-28 00:07 CDT

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

- **Problem Statement**: No account exists of how a Prefect workflow is used to build an AI agent or an AI orchestrator, or of where using it creates a strength.
- **Goal**: Split the frontend and backend roles of an AI orchestrator and an AI agent, and state what Prefect can specially do for the backend role and how it does it, so that a manager or a designer produces benchmarking material from it.
- **Non-Goal**: Designing an agent's prompt and its tools is not covered, the implementation code is not given, and no comparison is made with orchestrators other than Prefect (Airflow, Temporal, Dagster and the like).

## 2. Summary

Prefect fills, as a product, the places in an AI agent backend that decide what starts a run, how many calls may be in flight, and what ran. Those places together are the orchestrator, and [Fig 1](#fig-1) draws where the orchestrator sits inside the backend. An agent framework does the work inside one run only, so without Prefect the backend developer writes those places by hand. Six of the ten functions in [Table 2](#table-2) — suspension, idempotent rerun, declared rate limiting, per-step observability, one admission path and ML pipeline integration — and three constraints are the rows a benchmarking sheet compares products on.

The frontend keeps three roles and gains one duty when Prefect is used: the answer a paused run waits for arrives through it. The agent framework taken as the baseline is LangGraph, chosen because its checkpointer and its node retry policy, the two entries the left of [Table 2](#table-2) rests on, are stated in vendor documentation [[6](#ref-6)]; the other frameworks in use are listed in [Appendix D](#appendix-d-agent-frameworks-in-use-as-of-september-2026). [Appendix B](#appendix-b-what-prefect-does-in-an-agent-backend) lists the nine things Prefect does inside an agent backend and the three it leaves to the agent framework and the API server, and [Appendix C](#appendix-c-how-an-event-trigger-is-done-in-prefect) shows how the event trigger among them is attached.

## 3. Taxonomy and its Hierarchy

The boundary between frontend and backend falls after the user's intent is fixed and before the first LLM call, so the reasoning loop belongs to the backend. Ten responsibilities split across the two layers, three on the frontend and seven on the backend whose outer three are the orchestrator, and the seven are ordered by the scope each one has to hold: one step, one run, or every run at once.

Scope is what decides who carries a responsibility. A framework sees one graph run and covers the one-step and one-run responsibilities; the orchestrator is the part that sees every run, and the three every-run responsibilities are what define it. The ten roles, the layer of each, and what each one decides are drawn in [Fig 1](#fig-1).

```text
LAYER          ROLE              WHAT IT DECIDES                              SCOPE       PART

Frontend  >    Intent            What the user asked, in the backend's form   one request
               Approval          The answer a paused run is waiting for       one run
               Presentation      What the user sees of the result             one request
      |   one request crosses the boundary
      v
Backend   >    Reasoning         Which action the LLM picks next              one step    --+
               Tool execution    The action actually carried out              one step      |
               State             The point a stopped run resumes from         one run       |  Agent
               Recovery          Which failed step is tried again, once only  one run     --+
               Admission         What starts a run: request, schedule, event  every run   --+
               Throughput        How many calls run at once, and how fast     every run     |  Orchestrator
               Record            What ran, when, and with which result        every run   --+
```

<a id="fig-1"></a>
Fig 1. The three frontend roles and the seven backend responsibilities ordered by scope, four of them the agent and three the orchestrator

Widening the scope by one step needs one more place to keep the state that outlives the previous scope. One-step scope needs nothing beyond the process, one-run scope needs a store the process can die without losing, and every-run scope needs a service that outlives every process and can be asked what happened.

### 3.1 Placement

<a id="table-1"></a>
Table 1. Each role, the part that holds it, and what fixes it

| #   | Role           | Part         | Scope       | What fixes it                               |
| :-: | :------------: | :----------: | :---------: | :-----------------------------------------: |
| 1   | Intent         | Frontend     | One request | The request schema the backend accepts      |
| 2   | Approval       | Frontend     | One run     | The form that answers a paused run          |
| 3   | Presentation   | Frontend     | One request | The stream or page the user reads           |
| 4   | Reasoning      | Agent        | One step    | The LLM call and the tool choice it returns |
| 5   | Tool execution | Agent        | One step    | The function the chosen tool names          |
| 6   | State          | Agent        | One run     | The checkpoint a resumed run reads          |
| 7   | Recovery       | Agent        | One run     | The retry count, and what a rerun may skip  |
| 8   | Admission      | Orchestrator | Every run   | The deployment and what triggers it         |
| 9   | Throughput     | Orchestrator | Every run   | The concurrency limit and the call rate     |
| 10  | Record         | Orchestrator | Every run   | The run history each step writes            |

No agent framework carries the three orchestrator rows, so a design names a product on those rows and writes code on the others.

## 4. Backend Composition

The two compositions run the same reasoning loop and differ in where that loop lives. Without Prefect the loop runs inside the process that answered the request; with Prefect the loop is a flow that a worker starts on its own infrastructure, and the request only asks for it. The two are drawn in [Fig 2](#fig-2).

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

An agent framework covers the four inner responsibilities of [Fig 1](#fig-1) and leaves the orchestrator's three to the team. A checkpointer saves a snapshot of the graph state at every super-step under a thread id, which is what lets a stopped run resume and what lets a human interrupt, inspect and approve a step, and a retry policy attached to a node retries it with exponential backoff [[6](#ref-6)]. A scheduler, a semaphore and a log table are then written by hand, and each of the three is a component the team owns.

Prefect fills the same three with a server and a worker, and the agent loop becomes a flow whose tool calls are tasks. A deployment states where, when and how the flow runs, which turns the loop into an entity the API manages, triggered by a schedule, the UI, an automation or the REST API, while a work pool names the infrastructure and a worker polls the pool and starts the run on it [[1](#ref-1)]. The self-hosted server carries the API, the UI, scheduling, work pools, and the events and automations engine [[4](#ref-4)]. The official integration between Prefect and Pydantic AI does this wrapping, so the backend developer does not write it: tools become tasks automatically, each with its own retries, its own cached result and its own line in the run history [[5](#ref-5)].

## 5. Function Comparison

Ten functions are needed in either composition, and [Table 2](#table-2) sets out what holds each one on each side. The `Agent framework alone` column is the composition that uses only an agent framework, the library a team writes the LLM call and tool selection loop in, which is LangGraph in this document [[6](#ref-6)].

<a id="table-2"></a>
Table 2. The same function in each composition

| #   | Function                | Agent framework alone                                                        | Self-hosted Prefect added                                                                    |
| :-: | :---------------------: | :--------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------: |
| 1   | Step retry              | A retry policy on a node, inside one graph run                               | `retries` and `retry_delay_seconds` on every task                                            |
| 2   | Resume after a crash    | The checkpointer replays the thread from its last super-step                 | The same checkpoint, and the run state the server holds                                      |
| 3   | Human approval          | An interrupt, and a resume call the backend developer routes                 | `pause_flow_run` with `wait_for_input`, answered by API                                      |
| 4   | Where a run executes    | The web process that answered                                                | A work pool, with a worker polling it                                                        |
| 5   | Suspension              | Nobody releases it: the web process holds the thread and waits               | `suspend_flow_run` exits, and input starts the run again                                     |
| 6   | Idempotent rerun        | The backend developer writes the skip condition into the node                | Result caching loads the previous result instead of running again                            |
| 7   | Rate limiting           | The backend developer writes a semaphore to bound the calls                  | A global concurrency limit and a rate limit                                                  |
| 8   | Per-step observability  | The backend developer builds a log table and writes each step to it          | Every flow run and task run, in the server's UI                                              |
| 9   | One admission path      | The backend developer wires the web request that starts it                   | A deployment on a request, a schedule or an automation                                       |
| 10  | ML pipeline integration | The backend developer runs retraining and deployment on a separate scheduler | Retraining, deployment and the agent run as flows on one server, each able to start the next |

## 6. Strength

In rows #5 to #10 of [Table 2](#table-2) the agent framework does not do the work itself. The backend developer fills those rows with code written by hand, or leaves them undone. Those six rows are what separates one product from another on a benchmarking sheet, and each is taken below under the same name, with what Prefect does instead.

- **Suspension**: `pause_flow_run` keeps the flow alive while it waits, and `suspend_flow_run` exits so the infrastructure can be taken down, with the run started again when the input arrives [[2](#ref-2)]. A HITL step that waits a day occupies no process at all.
- **Idempotent rerun**: Prefect's transactional orchestration loads the previous result instead of running again when the context is identical [[5](#ref-5)]. Under that idempotency a retried agent run does not pay the LLM twice for the same tool call.
- **Rate limiting**: a global concurrency limit bounds how many calls are in flight, and a rate limit paces them by a slot decay per second [[3](#ref-3)]. Both work in Python code outside a flow, so a tool never wrapped as a task stays bounded too.
- **Per-step observability**: each tool call is a task, so each one appears in the run history and is retried on its own [[5](#ref-5)]. The report moves from an agent run having failed to which tool call failed on which input.
- **One admission path**: one deployment answers an interactive request, a nightly schedule and an event-driven automation [[1](#ref-1)]. The overnight batch and the chat request run the same code rather than two copies.
- **ML pipeline integration**: retraining, deployment and the agent run as flows on one server, and the state of a finished flow starts the next [[7](#ref-7)]. Retraining and the agent need not sit in two systems with a bridge written between them.

## 7. Application

Prefect earns its place when a run outlives the request that started it, and costs more than it returns when the run ends inside the request. The three conditions below decide which case a design is in.

**Assumption** is that the backend may own a process of its own. A worker is a client-side process that polls a work pool and starts runs on infrastructure [[1](#ref-1)], so a deployment target that forbids a long-lived process leaves the composition of [Fig 2](#fig-2) without its middle.

**Breaking condition** is authentication. The open source server carries no users and no authentication, so anyone who reaches the UI or the API has full access to it [[4](#ref-4)]; a self-hosted server therefore sits inside a private network or behind an authenticating proxy. Webhooks are a Prefect Cloud feature [[4](#ref-4)], so a self-hosted backend that must start runs from an outside system relays those events to the API itself.

**Exclusion** is an agent whose run is one LLM call and whose result nobody looks up later. The server and the worker are two components to operate, and a run that finishes in the request it arrived on has no state for them to hold.

## 8. Benchmarking

A benchmarking sheet takes its rows from this document and its columns from the products being compared. [Table 3](#table-3) is that row list, with Prefect's answer already filled in.

<a id="table-3"></a>
Table 3. The benchmarking rows, and Prefect's answer on each

| #   | Row                     | What it asks                                                               | Prefect's answer                  |
| :-: | :---------------------: | :------------------------------------------------------------------------: | :-------------------------------: |
| 1   | Orchestrator scope      | Which of the every-run three the product carries                           | All three                         |
| 2   | Suspension              | What a run waiting for a person holds open                                 | Nothing, the process exits        |
| 3   | Idempotent rerun        | What a rerun pays for work already done                                    | The previous result, loaded       |
| 4   | Rate limiting           | How the call rate is bounded                                               | Declared, in any Python code      |
| 5   | Per-step observability  | How far down a failure is located                                          | The one tool call                 |
| 6   | One admission path      | How many code paths the triggers need                                      | One deployment                    |
| 7   | ML pipeline integration | Whether the same product also runs the retraining and deployment pipelines | It does                           |
| 8   | Access control          | What guards the API and the UI                                             | Nothing in the open source server |
| 9   | Inbound events          | How an outside system starts a run                                         | A Cloud webhook, or a relay       |

A product that cannot answer a row leaves that row to code, so the sheet carries the cost of that code beside the product's name.

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

- **Agent framework**: the library a team writes the LLM call and tool selection loop in, such as LangGraph.
- **Automation**: the Prefect rule that starts a preset action when a matching event arrives.
- **Checkpointer**: the component that saves graph state at each step so that a stopped run resumes from it.
- **Deployment**: a flow with where, when and how it runs attached, which makes it an entity the API manages.
- **FDC (Fault Detection and Classification)**: the fab system that watches equipment sensor traces and raises an alarm when one leaves its limits.
- **Flow**: the function Prefect treats as one run.
- **HITL (Human In The Loop)**: a run that waits for a person's input before it continues.
- **Idempotency**: the property that running again with the same input leaves the same result and the same side effects as running once.
- **Jinja**: the template syntax Prefect substitutes event values into a flow's parameters with.
- **K8s (Kubernetes)**: the container platform a work pool can run flow runs on.
- **Orchestrator**: the part of a backend that sees every run rather than one, holding what starts a run, how many calls are in flight, and what the run history keeps.
- **Prefect Server**: the self-hosted orchestration backend holding the API, the UI, the scheduler, and events and automations.
- **RAG (Retrieval Augmented Generation)**: answering with documents retrieved at query time and handed to the LLM beside the question.
- **Rate limit**: the ceiling on how many calls may leave within a span of time.
- **Result caching**: loading a previous result for an identical input instead of executing again.
- **Super-step**: the execution unit a graph saves one state snapshot for.
- **Task**: the unit inside a flow that Prefect retries, caches and records on its own.
- **Thread**: the unit a checkpointer collects one conversation's state under.
- **VM (Virtual Metrology)**: predicting a measurement from process sensor data instead of measuring it.
- **Work pool**: the Prefect setting that names the infrastructure flow runs execute on.
- **Worker**: the client-side process that polls a work pool and starts each scheduled run on that infrastructure.

## Appendix B. What Prefect Does In An Agent Backend

The parenthesis at the end of each item names the [Table 2](#table-2) function it is.

1. Durable execution: it caches LLM and tool calls per task, and resumes from that point on a failure (#2 Resume after a crash, #6 Idempotent rerun).
2. Retry and timeout: it applies a policy per call, against an LLM API outage or a tool error (#1 Step retry).
3. Event trigger: it runs an agent automatically when an event such as an FDC alarm or a drift detection arrives (#9 One admission path).
4. Scheduling: it runs the regular analysis and report agents on a schedule (#9 One admission path).
5. Distributed execution: it distributes work to K8s or GPU workers (#4 Where a run executes).
6. Observability: it tracks run history, logs and state in the UI (#8 Per-step observability).
7. Human-in-the-loop waiting: with `pause_flow_run` it stops until the approval arrives, then resumes (#3 Human approval, #5 Suspension).
8. ML pipeline integration: it operates retraining, deployment and agent execution together (#10 ML pipeline integration).
9. Rate limiting: it bounds the calls in flight and the calls per second by declaration, against an LLM API rate limit (#7 Rate limiting).

What it does not do itself: the LLM inference logic, memory and RAG, and real-time conversation serving are carried by the agent framework and the API server.

## Appendix C. How An Event Trigger Is Done In Prefect

It takes two steps. `emit_event` publishes the event, and a `DeploymentEventTrigger` or an automation receives it and runs the flow [[7](#ref-7)].

**1. Emit the event** — on the FDC system or the collector side.

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

| Type                | Use                                                                    | Setting                                              |
| :-----------------: | :--------------------------------------------------------------------: | :--------------------------------------------------: |
| Reactive            | Runs as soon as the event occurs                                       | The default                                          |
| Threshold           | Runs once N have accumulated, such as three alarms within ten minutes  | `threshold=3`, `within=timedelta(minutes=10)`        |
| Proactive           | Runs when the event does not arrive, such as collection having stopped | `posture="Proactive"`                                |
| Compound / Sequence | Runs on a combination or an order of several events                    | `CompoundTrigger`, `SequenceTrigger`                 |
| Flow state          | Runs in sequence after another flow completes or fails                 | `expect={"prefect.flow-run.Completed"}` and the like |

**External systems**

- Prefect Cloud: a webhook takes an outside HTTP request as an event directly.
- Self-hosted (OSS): with no webhook, a FastAPI endpoint or a Kafka or MQ consumer calls `emit_event` to bridge it.

**Fab application**

- An FDC alarm, reactively, to the cause analysis agent
- Three VM error excursions within ten minutes, on threshold, to the retraining flow
- Sensor data unreceived for thirty minutes, proactively, to the equipment check notice
- Retraining completed, on flow state, to the validation agent and the report

## Appendix D. Agent Frameworks In Use As Of September 2026

The baseline this document uses is #1 LangGraph, and chapter 2 says why it was chosen. Each date is the product's public announcement [[8](#ref-8)].

Table 5. Agent frameworks and when each was first announced

| #   | Framework                 | What it is                                                                             | Announced |
| :-: | :-----------------------: | :------------------------------------------------------------------------------------: | :-------: |
| 1   | LangGraph                 | Holds state as a graph. The default where an audit trail and human approval are needed | 2024-01   |
| 2   | CrewAI                    | Binds several role-playing agents into a team. Fastest to a prototype                  | 2024-01   |
| 3   | OpenAI Agents SDK         | The model drives the loop. Least friction if the work is GPT-centred                   | 2025-03   |
| 4   | Google ADK                | Strong on multimodal. Announced at Google Cloud NEXT 2025                              | 2025-04   |
| 5   | Claude Agent SDK          | The agent harness behind Claude Code, renamed from the Claude Code SDK                 | 2025-09   |
| 6   | Microsoft Agent Framework | Graph based. The choice in an Azure and .NET estate                                    | 2026-04   |
| 7   | Pydantic AI               | Type-safe Python. V2 carries durable execution                                         | 2026-06   |

#7 Pydantic AI and Prefect are two products used separately, and a package that joins them is published [[5](#ref-5)]. Installing it runs an agent written with Pydantic AI as a Prefect flow and makes that agent's tools Prefect tasks, so the wrapping code does not have to be written by hand. Neither product has absorbed the other, and neither uses the other inside itself.
