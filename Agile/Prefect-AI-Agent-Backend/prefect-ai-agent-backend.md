# Prefect As An AI Agent Backend
Rev. 2 | Created: 2026-09-27 | Updated: 2026-09-27 22:08 CDT

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

## 1. Purpose

- **Problem Statement**: No account exists of how a Prefect workflow is used to build an AI agent or an AI orchestrator, or of where using it creates a strength.
- **Goal**: Split the frontend and backend roles of an AI orchestrator and an AI agent, and state what Prefect can specially do for the backend role and how it does it, so that a manager or a designer produces benchmarking material from it.
- **Non-Goal**: Designing an agent's prompt and its tools is not covered, the implementation code is not given, and no orchestrator other than Prefect is scored.

## 2. Summary

The orchestrator is the fleet-scoped third of an AI agent's backend, and Prefect supplies that third as a product where a team would otherwise write it. Five of its capabilities — suspension, idempotent rerun, declared rate limiting, per-step observability and one admission path — and the three constraints beside them are what a benchmarking sheet compares one orchestrator against another on.

The frontend keeps three roles and gains one duty when Prefect is used: the answer a paused run waits for arrives through it.

## 3. Taxonomy and its Hierarchy

The boundary between frontend and backend falls after the user's intent is fixed and before the first LLM call, so the reasoning loop belongs to the backend. Ten responsibilities split across the two layers, three on the frontend and seven on the backend whose outer three are the orchestrator, and the seven are ordered by the scope each one has to hold: one step, one run, or every run at once.

Scope is what decides who carries a responsibility. A framework sees one graph run and covers the step-scoped and run-scoped responsibilities; the orchestrator is the part that sees every run, and the three fleet-scoped responsibilities are what define it. The ten roles, the layer of each, and what each one decides are drawn in [Fig 1](#fig-1).

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

Widening the scope by one step costs a place to put the state that outlives the previous scope. Step scope needs nothing beyond the process, run scope needs a store the process can die without losing, and fleet scope needs a service that outlives every process and can be asked what happened.

### 3.1 Placement

<a id="table-1"></a>
Table 1. Each role, the layer that holds it, and what fixes it

| Role           | Layer        | Scope       | What fixes it                               |
| :------------: | :----------: | :---------: | :-----------------------------------------: |
| Intent         | Frontend     | One request | The request schema the backend accepts      |
| Approval       | Frontend     | One run     | The form that answers a paused run          |
| Presentation   | Frontend     | One request | The stream or page the user reads           |
| Reasoning      | Backend      | One step    | The LLM call and the tool choice it returns |
| Tool execution | Backend      | One step    | The function the chosen tool names          |
| State          | Backend      | One run     | The checkpoint a resumed run reads          |
| Recovery       | Backend      | One run     | The retry count, and what a rerun may skip  |
| Admission      | Orchestrator | Every run   | The deployment and what triggers it         |
| Throughput     | Orchestrator | Every run   | The concurrency limit and the call rate     |
| Record         | Orchestrator | Every run   | The run history each step writes            |

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

Prefect fills the same three with a server and a worker, and the agent loop becomes a flow whose tool calls are tasks. A deployment states where, when and how the flow runs, which turns the loop into an entity the API manages, triggered by a schedule, the UI, an automation or the REST API, while a work pool names the infrastructure and a worker polls the pool and starts the run on it [[1](#ref-1)]. The self-hosted server carries the API, the UI, scheduling, work pools, and the events and automations engine [[4](#ref-4)], and wrapping an agent this way is a published integration in which tools become tasks automatically, each with its own retries, its own cached result and its own line in the run history [[5](#ref-5)].

## 5. Function Comparison

Nine functions are carried in both compositions, and the difference is what holds them rather than whether they exist. [Table 2](#table-2) reads left to right as the same requirement met twice.

<a id="table-2"></a>
Table 2. The same function in each composition

| Function                          | Agent framework alone                                        | Self-hosted Prefect added                                         |
| :-------------------------------: | :----------------------------------------------------------: | :---------------------------------------------------------------: |
| Step retry                        | A retry policy on a node, inside one graph run               | `retries` and `retry_delay_seconds` on every task                 |
| Resume after a crash              | The checkpointer replays the thread from its last super-step | The same checkpoint, and the run state the server holds           |
| Skip work already done            | Written by hand in the node                                  | Result caching loads the previous result instead of running again |
| Human approval                    | An interrupt, and a resume call the team routes              | `pause_flow_run` with `wait_for_input`, answered by API           |
| Release the process while waiting | The process holds the thread open                            | `suspend_flow_run` exits, and input starts the run again          |
| What starts a run                 | The web request the team wires                               | A deployment on a request, a schedule or an automation            |
| Calls in flight and call rate     | A semaphore written by hand                                  | A global concurrency limit and a rate limit                       |
| Run history                       | Not held                                                     | Every flow run and task run, in the server's UI                   |
| Where a run executes              | The web process that answered                                | A work pool, with a worker polling it                             |

## 6. Strength

Five of the nine differ in kind rather than in degree, so they are the rows on which one orchestrator is compared against another rather than tuned.

**Suspension** releases the process. `pause_flow_run` keeps the flow running while it waits, and `suspend_flow_run` exits so the infrastructure can be deprovisioned, with the run started again when the input arrives [[2](#ref-2)]. A HITL step that waits a day therefore costs nothing while it waits, where a held thread costs a process for the whole day.

**Idempotent rerun** makes a repeat safe. Idempotency comes from Prefect's transactional orchestration, which makes a rerun load a previous result instead of executing again when the context is identical, so a retried agent run does not pay the LLM twice for the same tool call [[5](#ref-5)].

**Rate limiting** is declared rather than coded. A global concurrency limit bounds how many calls are in flight, and a rate limit paces them by a slot decay per second; both work in any Python code rather than only inside a flow, so a tool that was never wrapped as a task is still bounded [[3](#ref-3)].

**Per-step observability** comes from the same wrapping that gives retries. Because each tool call is a task, each one is separately visible in the run history and separately retryable [[5](#ref-5)], which is the difference between knowing that an agent run failed and knowing which tool call failed on which input.

**One admission path** serves every trigger. A single deployment answers an interactive request, a nightly schedule and an event-driven automation [[1](#ref-1)], so the overnight batch and the chat request run the same code rather than two copies that drift apart.

## 7. Application

Prefect earns its place when a run outlives the request that started it, and costs more than it returns when the run ends inside the request. The three conditions below decide which case a design is in.

**Assumption** is that the backend may own a process of its own. A worker is a client-side process that polls a work pool and starts runs on infrastructure [[1](#ref-1)], so a deployment target that forbids a long-lived process leaves the composition of [Fig 2](#fig-2) without its middle.

**Breaking condition** is authentication. The open source server carries no users and no authentication, so anyone who reaches the UI or the API has full access to it [[4](#ref-4)]; a self-hosted server therefore sits inside a private network or behind an authenticating proxy. Webhooks are a Prefect Cloud feature [[4](#ref-4)], so a self-hosted backend that must start runs from an outside system relays those events to the API itself.

**Exclusion** is an agent whose run is one LLM call and whose result nobody looks up later. The server and the worker are two components to operate, and a run that finishes in the request it arrived on has no state for them to hold.

## 8. Benchmarking

A benchmarking sheet takes its rows from this document and its columns from the products being compared. [Table 3](#table-3) is that row list, with Prefect's answer already filled in.

<a id="table-3"></a>
Table 3. The benchmarking rows, and Prefect's answer on each

| Row                | What it asks                                        | Prefect's answer                   |
| :----------------: | :-------------------------------------------------: | :--------------------------------: |
| Orchestrator scope | Which of the fleet-scoped three the product carries | All three                          |
| Waiting cost       | What a run waiting for a person holds open          | Nothing, the process exits         |
| Repeat cost        | What a rerun pays for work already done             | The previous result, loaded        |
| Call pacing        | How the call rate is bounded                        | Declared, in any Python code       |
| Failure locality   | How far down a failure is located                   | The one tool call                  |
| Trigger count      | How many code paths the triggers need               | One deployment                     |
| Access control     | What guards the API and the UI                      | Nothing, in the open source server |
| Inbound events     | How an outside system starts a run                  | A Cloud webhook, or a relay        |

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
[6] LangChain. [Checkpointers](https://docs.langchain.com/oss/python/langgraph/checkpointers). LangGraph documentation.

---

## Appendix A. Terminology

- **Agent framework**: the library a team writes the LLM call and tool selection loop in.
- **Automation**: the Prefect rule that starts a preset action when a matching event arrives.
- **Checkpointer**: the component that saves graph state at each step so that a stopped run resumes from it.
- **Deployment**: a flow with where, when and how it runs attached, which makes it an entity the API manages.
- **Flow**: the function Prefect treats as one run.
- **HITL (Human In The Loop)**: a run that waits for a person's input before it continues.
- **Idempotency**: the property that running again with the same input leaves the same result and the same side effects as running once.
- **Orchestrator**: the part of a backend that sees every run rather than one, holding what starts a run, how many calls are in flight, and what the run history keeps.
- **Prefect Server**: the self-hosted orchestration backend holding the API, the UI, the scheduler, and events and automations.
- **Rate limit**: the ceiling on how many calls may leave within a span of time.
- **Result caching**: loading a previous result for an identical input instead of executing again.
- **Super-step**: the execution unit a graph saves one state snapshot for.
- **Task**: the unit inside a flow that Prefect retries, caches and records on its own.
- **Thread**: the unit a checkpointer collects one conversation's state under.
- **Work pool**: the Prefect setting that names the infrastructure flow runs execute on.
- **Worker**: the client-side process that polls a work pool and starts each scheduled run on that infrastructure.
