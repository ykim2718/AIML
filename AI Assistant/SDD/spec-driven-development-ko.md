# Spec-Driven Development
Rev. 3 | Created: 2026-09-26 | Updated: 2026-09-26 13:53 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Placement](#31-placement)
  - [3.2 Spec Components](#32-spec-components)
  - [3.3 Requirement Notation](#33-requirement-notation)
- [4. Workflow](#4-workflow)
  - [4.1 Roles](#41-roles)
  - [4.2 Phase 1: Spec Writing](#42-phase-1-spec-writing)
  - [4.3 Phase 2: Plan and Test Gate](#43-phase-2-plan-and-test-gate)
  - [4.4 Phase 3: Implementation](#44-phase-3-implementation)
  - [4.5 Phase 4: Self-Verification Loop](#45-phase-4-self-verification-loop)
  - [4.6 Phase 5: Review](#46-phase-5-review)
- [5. Effects and Limits](#5-effects-and-limits)
- [6. Further Work](#6-further-work)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

- **Problem Statement**: Prompt 만으로 AI coding agent 에 구현을 맡기면 codebase 가 커질수록 기존 설계와의 불일치, side effect, 품질 저하가 생긴다.
- **Goal**: Data science 엔지니어가 specifier 로서 agent 에 맡길 작업을 spec 파일 하나로 정의하고, reviewer 로서 agent 의 결과를 그 spec 에 비추어 받아들일지 돌려보낼지 판정할 수 있게 한다.
- **Non-Goal**: 특정 agent 제품의 설치와 설정, 그리고 제품 사이의 기능 비교는 다루지 않는다.

## 2. Summary

Spec-Driven Development (SDD) 는 사람이 구조와 명세 (spec) 를 정하고, AI coding agent 가 그 spec 에 따라 code 를 구현하고 검증하며, 사람이 그 결과를 다시 검수하는 개발 방식이다. Spec 파일은 네 요소, 곧 scope, data schema, behavior, acceptance criteria 를 담는다. Agent 는 scope 안의 파일만 고치고, acceptance criteria 에 적힌 명령이 모두 통과할 때까지 구현을 되풀이한다.

Specifier 와 reviewer 는 기존 code writer 가 code 를 쓰기 전과 쓴 뒤에 하던 일이, AI 를 사용함으로써 강조된 역할이다. Code writer 는 code 를 쓰기 전에 무엇을 만들지 정하고, 쓴 뒤에 그 code 가 의도대로 동작하는지 검토했다. AI coding agent 가 그 사이의 code 작성을 맡으면서, 사람의 일은 그 앞의 명세 (specifier) 와 그 뒤의 검수 (reviewer) 에 모인다. Specifier 는 spec 으로 agent 의 작업 범위와 통과 기준을 정하고, reviewer 는 구현 전에 agent 의 plan 과 test 를 승인하며, 구현 뒤에는 test 가 가리지 못한 결함을 찾아 agent 나 specifier 에게 돌려보낸다.

GitHub Spec Kit, Kiro, OpenSpec 같은 SDD 도구는 기능마다 spec, plan, tasks 의 세 파일을 두고, 모든 기능에 공통으로 적용할 project 규칙을 constitution 이나 steering 파일에 따로 둔다 [[1](#ref-1)] [[3](#ref-3)] [[4](#ref-4)]. 요구사항은 번호를 붙인 MUST 문장과 Given-When-Then scenario 로 적고, 정하지 못한 자리는 `[NEEDS CLARIFICATION]` 으로 표시하여 agent 가 추측으로 메우지 못하게 한다 [[2](#ref-2)].

Spec 파일은 대화 이력 대신 single source of truth 가 되어, LLM 의 hallucination 과 context drift 를 줄인다. 같은 spec 에서는 실행마다 결과가 같은 기준으로 판정되므로, 대규모 codebase 에서도 agent 의 결과를 예측할 수 있다.

## 3. Taxonomy and its Hierarchy

AI 를 쓰는 개발 방식은 agent 에게 설계 결정까지 맡기는가로 갈린다. Prompt-based development 는 설계와 구현을 모두 agent 에게 맡기고, SDD 는 설계를 spec 으로 고정한 뒤 구현만 맡긴다. [Fig 1](#fig-1) 은 두 방식과 SDD 가 쓰는 파일의 계층, 그리고 spec 의 네 요소를 보여 준다.

```text
AI-assisted coding
├── Prompt-based development   ad-hoc natural-language prompt; agent designs and implements
└── Spec-driven development    structured files; agent implements within them
    ├── Project rules            constitution / steering; apply to every feature
    ├── Spec (spec.md)           what to build; written by the specifier
    │   ├── 1. Scope & Context       target files, tech stack, library versions
    │   ├── 2. Data Schema           input/output schema, DataFrame and table contracts
    │   ├── 3. Behavior              numbered MUST requirements, Given-When-Then scenarios
    │   └── 4. Acceptance Criteria   test and check commands that decide pass or fail
    ├── Plan (plan.md)           how to build it; drafted by the agent, approved by the reviewer
    └── Tasks (tasks.md)         ordered work items the agent executes
```

<a id="fig-1"></a>
Fig 1. Development styles, SDD files and the four components of a spec

SDD 의 파일은 위에서 아래로 적용 범위가 좁아진다. Project rules 는 모든 기능에, spec 은 한 기능에, plan 은 그 기능의 구현 방법에, tasks 는 agent 가 한 번에 실행할 작업 하나에 적용된다. 아래 파일은 위 파일을 어길 수 없으며, plan 이 project rules 와 어긋나면 reviewer 가 Phase 2 의 gate 에서 plan 을 반려한다.

Spec 의 네 요소는 위에서 아래로 agent 가 스스로 정할 수 있는 폭을 좁힌다. Scope 는 손댈 파일과 library 를, data schema 는 입출력의 모양을, behavior 는 경계 조건에서의 동작을 고정한다. Acceptance criteria 는 앞의 세 요소를 실행할 수 있는 명령으로 바꾸어, 통과 여부를 사람의 판단 없이 가린다. 앞의 세 요소만 있고 acceptance criteria 가 없으면 agent 는 구현을 마쳤다고 보고할 뿐 spec 을 지켰는지는 확인하지 않는다.

### 3.1 Placement

아래 표는 두 방식을 같은 항목으로 나란히 놓는다.

Table 1. Prompt-based development and spec-driven development

| Aspect          | Prompt-based development                  | Spec-driven development                      |
| :-------------: | :---------------------------------------: | :------------------------------------------: |
| Anchor          | 자연어 prompt (ad-hoc)                    | 구조화된 spec 문서 (spec as code)            |
| Agent role      | 자율적인 설계와 구현                      | Spec 에 묶인 구현과 자가 검증                |
| Human role      | Prompt 작성과 결과 확인                   | Specifier 와 reviewer                        |
| Context keeping | 대화 이력이 길어질수록 hallucination 증가 | Spec 파일을 single source of truth 로 유지   |
| Reproducibility | 실행마다 결과의 변동 폭 큼                | 같은 spec 기준으로 결과가 일관됨             |
| Assumption      | 작업이 대화 몇 번 안에 끝남               | Acceptance criteria 를 명령으로 적을 수 있음 |
| Breaks when     | 파일이 여럿이고 기존 설계와 얽힘          | Spec 이 모호하거나 test 가 비어 있음         |
| Use at          | 일회성 질의응답, 탐색용 notebook code     | Pipeline, feature 계산, 배포할 model code    |

Prompt-based development 는 결과를 버려도 되는 탐색 단계에 맞고, SDD 는 다른 code 가 그 결과에 의존하는 단계에 맞는다. Notebook 에서 가설을 확인하는 동안은 prompt 로 충분하지만, 그 계산을 pipeline 의 module 로 옮기는 순간부터는 spec 을 쓴다.

### 3.2 Spec Components

네 요소는 data science code 에서 아래 내용을 담는다.

1. **Scope & Context**
    - 새로 만들거나 고칠 파일의 목록과, 고치지 않을 파일
    - Python, pandas, scikit-learn 등 쓸 library 와 그 version 제약
2. **Data Schema**
    - 입력과 출력 DataFrame 의 열 이름, dtype, index, 정렬 조건
    - 읽고 쓰는 table 과 file 의 형식, API 의 request 와 response 형식
3. **Behavior**
    - Pre-condition 과 post-condition
    - 결측값, 빈 입력, 정렬되지 않은 시간 열 같은 edge case 와 그때의 오류 처리
    - Look-ahead leakage 를 막는 시간 기준처럼 결과의 타당성을 정하는 규칙
4. **Acceptance Criteria**
    - 통과 여부를 가르는 test 파일과 실행 명령
    - Type check, lint 명령

Seed 고정, 원본 data 의 불변, 시간 기준 train/test 분할처럼 모든 기능이 지켜야 하는 규칙은 기능마다 spec 에 되풀이하지 않고 project rules 파일에 한 번 적는다. Spec Kit 은 이 파일을 constitution 이라 부르며, test 를 구현보다 먼저 쓰고 그 test 가 실패하는 것을 확인하라는 TDD 의 test-first 규칙을 그 조항의 하나로 둔다 [[1](#ref-1)].

### 3.3 Requirement Notation

Behavior 의 요구사항은 번호를 붙인 MUST 문장으로 적고, 대표 경우를 Given-When-Then scenario 로 덧붙인다. 번호는 요구사항과 test 를 하나씩 짝짓는 key 가 되어, reviewer 가 빠진 test 를 번호로 찾는다. 아래 표는 SDD 도구가 쓰는 표기를 모은다.

Table 2. Requirement notation in SDD tools

| Notation             | Form                             | Purpose                                     | Source                 |
| :------------------: | :------------------------------: | :-----------------------------------------: | :--------------------: |
| Numbered requirement | FR-001: System MUST ...          | 요구사항과 test 의 1 대 1 대응              | Spec Kit [[2](#ref-2)] |
| Acceptance scenario  | Given ..., When ..., Then ...    | 입력 상태, 동작, 기대 결과의 고정           | Spec Kit [[2](#ref-2)] |
| Clarification marker | [NEEDS CLARIFICATION: ...]       | 정하지 못한 자리의 표시, agent 의 추측 금지 | Spec Kit [[1](#ref-1)] |
| Delta spec           | ADDED, MODIFIED, REMOVED         | 기존 spec 에 대한 변경분만 기록             | OpenSpec [[3](#ref-3)] |
| Executable property  | 생성한 입력 전체에서 성립할 조건 | 예시 test 가 놓치는 입력의 검증             | Kiro [[4](#ref-4)]     |

Clarification marker 는 Phase 1 의 spec 에만 남을 수 있고, Phase 2 로 넘어가기 전에 specifier 가 모두 풀어야 한다. Delta spec 은 이미 운영 중인 pipeline 을 고칠 때 쓴다. 변경이 끝나면 delta 를 본래 spec 에 합쳐, spec 이 언제나 현재 code 의 동작을 적은 상태로 남게 한다 [[3](#ref-3)]. Executable property 는 data science code 에서 "출력 행 수는 입력 행 수와 같다" 처럼 모든 입력에서 성립해야 하는 조건이며, property-based test 로 무작위 입력을 만들어 검증한다 [[4](#ref-4)].

## 4. Workflow

SDD 는 specifier 의 spec 작성, agent 가 만든 plan 과 test 의 reviewer 승인, agent 의 구현, agent 의 자가 검증, reviewer 의 검수의 다섯 단계로 진행한다. 자가 검증이 실패하면 agent 가 구현으로 돌아가고, 검수에서 결함이 나오면 reviewer 가 그 원인에 따라 결과를 agent 나 specifier 에게 돌려보낸다. [Fig 2](#fig-2) 는 그 흐름을 보여 준다.

```text
Phase 1  Specifier   [ Spec file ] <── spec gap ────────────────────────────────┐
                           │                                                    │
                           v                                                    │
Phase 2  Agent       [ Plan, tasks, tests ]                                     │
                           │      ^                                             │
                           v      │ rejected                                    │
Phase 2  Reviewer    [ Plan and test gate ]                                     │
                           │ approved                                           │
                           v                                                    │
Phase 3  Agent       [ Implementation ] <── spec violation ─────────────┐       │
                           │      ^                                     │       │
                           v      │ fail                                │       │
Phase 4  Agent       [ Verification ]                                   │       │
                           │ pass                                       │       │
                           v                                            │       │
Phase 5  Reviewer    [ Review ] ────────────────────────────────────────┴───────┘
                           │ accept
                           v
                     [ Merge ]
```

<a id="fig-2"></a>
Fig 2. SDD workflow with the specifier and reviewer loops

Phase 4 의 fail loop 는 agent 가 사람 없이 돌리고, Phase 2 와 Phase 5 의 loop 는 reviewer 가 판정하여 연다. Phase 2 의 gate 는 구현 전에 방향을 확인하여, 틀린 plan 위에 code 가 쌓이는 것을 막는다. Phase 5 의 spec violation 은 spec 을 그대로 두고 agent 를 다시 실행하며, spec gap 은 specifier 가 spec 을 고친 뒤 Phase 1 부터 다시 시작한다.

### 4.1 Roles

SDD 에서 사람은 specifier 와 reviewer 의 두 역할을 맡고, 그 사이의 plan 초안, 구현, 검증 명령 실행은 agent 가 맡는다. Specifier 는 무엇을 만들지를 spec 으로 정하고, reviewer 는 구현 전에는 plan 과 test 를, 구현 뒤에는 결과를 그 spec 에 비추어 받아들일지 정한다. 한 사람이 두 역할을 함께 맡을 수 있으며, 역할은 하는 일로 나눈다.

Table 3. Roles in SDD

| Role      | Actor           | Phase         | Output                                      | Decides                                                    |
| :-------: | :-------------: | :-----------: | :-----------------------------------------: | :--------------------------------------------------------: |
| Specifier | 사람            | Phase 1       | Spec 파일, project rules                    | 범위, data schema, behavior, acceptance criteria           |
| Agent     | AI coding agent | Phase 2, 3, 4 | Plan, tasks, test, code                     | Spec 안에서의 구현 방법                                    |
| Reviewer  | 사람            | Phase 2, 5    | Gate 승인, 수용, agent 반려, specifier 반려 | Plan 과 test 의 승인, 결과의 수용 여부, 결함을 돌려보낼 곳 |

Reviewer 는 찾은 결함을 agent 나 specifier 에게 돌려보내 spec 을 거쳐 고치게 한다. Reviewer 가 code 를 직접 고치면 spec 과 code 가 어긋나 spec 이 single source of truth 로 남지 않는다.

### 4.2 Phase 1: Spec Writing

Specifier 가 구현할 기능의 spec 을 Markdown 으로 `spec.md` 같은 파일에 적는다. 아래는 sensor trace 에 rolling mean feature 를 더하는 함수의 spec 이다.

````markdown
# Spec: Rolling mean feature for sensor trace

## Context & Boundaries
- Target files: `src/features/rolling.py`, `tests/test_rolling.py`
- Tech stack: Python 3.11, pandas 2.x, pytest, hypothesis, mypy, ruff
- Do not modify any other file.

## Input Schema
```python
# Python
import pandas as pd


def add_rolling_mean(*, df: pd.DataFrame, value_col: str, window: int, time_col: str = "timestamp") -> pd.DataFrame:
    """Append the trailing rolling mean of value_col.

    Returns:
        pd.DataFrame: index and columns of df unchanged, plus one float64 column named f"{value_col}_rm{window}".
    """
```

- `time_col`: datetime64[ns], sorted ascending, no duplicates.
- `value_col`: float64, may contain NaN.

## Requirements
- FR-001: The function MUST raise `ValueError` when `window < 1`.
- FR-002: The function MUST raise `KeyError` naming the column when `value_col` or `time_col` is missing.
- FR-003: The function MUST raise `ValueError` when `time_col` is not sorted; it MUST NOT sort silently.
- FR-004: Each output row MUST use only the current and earlier rows (no look-ahead leakage).
- FR-005: The first `window - 1` rows of the new column MUST be NaN.
- FR-006: The function MUST NOT mutate the input `df`.

## Acceptance Scenarios
- Given `value_col` = [1, 2, 3, 4] and `window` = 2, When the function runs, Then the new column is [NaN, 1.5, 2.5, 3.5].
- Given a later value changed from 4 to 100, When the function runs, Then the first three rows of the new column do not change (FR-004).

## Properties
- For any valid input, the output has the same number of rows and the same index as the input.

## Verification / Acceptance Criteria
- [ ] `pytest tests/test_rolling.py` passes; each FR-00N has at least one test named after it.
- [ ] `mypy src/features/rolling.py` reports no error.
- [ ] `ruff check src/features` reports no error.
````

Spec 의 각 FR 번호는 `tests/test_rolling.py` 의 test 하나 이상으로 옮겨 적을 수 있다. 옮겨 적을 수 없는 요구사항은 agent 도 검증할 수 없으므로, 그 요구사항을 test 할 수 있는 문장으로 고친다. 결측값을 건너뛸지 NaN 으로 둘지처럼 아직 정하지 못한 자리는 `[NEEDS CLARIFICATION: skip or propagate NaN?]` 으로 적어 두고, Phase 2 로 넘기기 전에 specifier 가 답을 적는다.

### 4.3 Phase 2: Plan and Test Gate

Specifier 는 spec 파일을 agent 의 입력으로 주고, agent 는 구현에 앞서 plan, tasks, test 를 만든다. Plan 에는 쓸 pandas 연산과 오류를 검사하는 순서를, tasks 에는 실행할 작업 단위 (task) 를, test 에는 FR 번호마다 하나 이상의 test 함수를 적는다.

Reviewer 는 이 셋을 승인한 뒤에만 구현을 허락한다. Plan 이 project rules 나 spec 의 scope 를 벗어나면 반려하고, test 는 FR 번호와 하나씩 대조한 뒤 구현 전에 실행하여 모두 실패하는지 확인한다. 구현 전에 통과하는 test 는 아무 동작도 검사하지 않는 test 이므로 고치게 한다. 여기서 승인한 test 는 Phase 5 에서 test 가 약해졌는지 가리는 기준이 된다.

### 4.4 Phase 3: Implementation

Agent 는 tasks 의 순서대로 code 를 쓰며, Context & Boundaries 에 적힌 파일만 만들거나 고친다. Phase 2 에서 승인한 test 는 고치지 않는다.

### 4.5 Phase 4: Self-Verification Loop

Agent 는 code 를 쓴 뒤 Verification 에 적힌 명령을 스스로 실행한다.

- 실패하면 agent 가 error log 를 읽고 code 를 고친 뒤 명령을 다시 실행한다.
- 모두 통과하면 agent 가 commit 또는 pull request 를 만들어 Phase 5 의 reviewer 에게 넘긴다.

### 4.6 Phase 5: Review

Reviewer 는 acceptance criteria 가 가리지 못하는 결함을 찾는다. Acceptance criteria 는 spec 에 적힌 명령의 통과 여부만 가리므로, agent 가 test 를 약하게 고쳐 통과시킨 경우, scope 밖의 파일을 고친 경우, spec 에 없는 동작을 넣은 경우는 통과 결과에 드러나지 않는다. Data science code 에서는 test 가 통과해도 look-ahead leakage 나 행 누락으로 결과 수치가 틀릴 수 있어, reviewer 는 작은 표본에 code 를 직접 실행해 값을 확인한다.

Table 4. Review checklist

| Check                | Looks for                                           | Evidence                                       |
| :------------------: | :-------------------------------------------------: | :--------------------------------------------: |
| Scope                | Target file 밖의 수정                               | git diff --stat                                |
| Test integrity       | Phase 2 에서 승인한 test 의 삭제, skip, 기대값 변경 | 승인한 test 와의 diff, pytest -rs 의 skip 목록 |
| Requirement coverage | FR 번호마다 대응하는 test                           | FR 번호와 test 함수 이름의 대조                |
| Unspecified behavior | Spec 에 없는 fallback, 암묵적 정렬, 결측 대체       | Code 읽기                                      |
| Data validity        | 행 수, dtype, 값 범위, look-ahead leakage           | 작은 표본에 실행한 결과                        |

Scope 와 test integrity 는 diff 만으로 가려지고, 나머지 셋은 reviewer 가 spec 과 code 를 함께 읽어야 가려진다. Reviewer 는 checklist 의 결과로 아래 셋 중 하나를 판정한다.

Table 5. Review decisions

| Decision            | Condition                       | Returns to | Next step                                              |
| :-----------------: | :-----------------------------: | :--------: | :----------------------------------------------------: |
| Accept              | 모든 check 통과                 | 없음       | Merge                                                  |
| Return to agent     | Spec 에 적힌 항목의 위반        | Agent      | 어긴 FR 번호를 붙여 같은 spec 으로 Phase 3 부터 재실행 |
| Return to specifier | Spec 이 정하지 않은 동작의 결함 | Specifier  | 요구사항과 test 를 spec 에 더한 뒤 Phase 1 부터 재실행 |

Return to specifier 로 spec 에 더한 test 는 다음 실행부터 acceptance criteria 가 되어, 같은 결함을 reviewer 대신 명령이 가린다. Review 를 거칠 때마다 사람이 판단할 항목이 test 로 옮겨 가므로, reviewer 는 test 로 적을 수 없는 판단에 시간을 쓴다.

## 5. Effects and Limits

SDD 는 agent 의 작업 범위와 판정 기준을 고정하는 대가로, spec 을 쓰는 시간과 결과를 검수하는 시간을 사람에게 요구한다.

- **Token 과 비용 절감**: Agent 가 범위 밖을 탐색하지 않고 spec 이 정한 파일 안에서만 작업한다.
- **Code 품질 유지**: Agent 가 구조를 임의로 바꾸거나 spec 에 없는 동작을 넣는 것을 막는다.
- **문서와 code 의 일치**: Spec 파일이 그대로 그 기능의 설계 문서가 된다.
- **초기 spec 작성 비용**: Spec 을 꼼꼼히 쓰는 데 specifier 의 시간이 든다. Agent 에게 spec 초안을 쓰게 하고 사람이 검수하면 이 시간을 줄일 수 있다.
- **Spec 의 명확성 요구**: 모호한 spec 은 모호한 결과를 낸다. 모든 FR 번호를 test 할 수 있는 수준으로 적어야 한다.
- **Reviewer 의 검수 부담**: Agent 가 code 를 빨리 만들수록 reviewer 가 읽을 diff 가 늘어난다. Phase 2 의 gate 는 방향이 틀린 결과를 구현 전에 걸러 이 부담을 줄이고, section 4.6 의 return to specifier 는 결함을 test 로 옮겨 다음 review 의 부담을 줄인다.

## 6. Further Work

- **운영 중인 pipeline 의 spec 화**
    - 무엇을 하는가: 이미 운영 중인 feature pipeline 의 현재 동작을 spec 으로 거꾸로 적고, 이후의 변경은 delta spec 으로만 적는다.
    - 왜 지금인가: OpenSpec 이 기존 codebase 를 대상으로 delta spec 과, delta spec 을 본래 spec 에 합치는 archive 단계를 도구로 제공한다 [[3](#ref-3)].
    - 무엇이 필요한가: 기존 pipeline 의 입출력 schema 기록과, 현재 동작을 고정하는 regression test.

## References

<a id="ref-1"></a>
[1] GitHub. [Specification-Driven Development (SDD)](https://github.com/github/spec-kit/blob/main/spec-driven.md). *github/spec-kit*, GitHub repository.<br>
<a id="ref-2"></a>
[2] GitHub. [Feature Specification Template](https://github.com/github/spec-kit/blob/main/templates/spec-template.md). *github/spec-kit*, GitHub repository.<br>
<a id="ref-3"></a>
[3] Fission AI. [OpenSpec](https://github.com/Fission-AI/OpenSpec). GitHub repository.<br>
<a id="ref-4"></a>
[4] Kiro. [Kiro](https://github.com/kirodotdev/Kiro). GitHub repository.

---

## Appendix A. Terminology

- **acceptance criteria**: 구현이 spec 을 만족하는지 가르는 조건. SDD 에서는 test, type check, lint 의 실행 명령으로 적는다.
- **AI coding agent**: 파일을 읽고 쓰고 명령을 실행하면서 code 를 구현하는 LLM 기반 도구.
- **code writer**: AI coding agent 를 쓰기 전의 개발자 역할. 무엇을 만들지 정하고, code 를 직접 쓰고, 쓴 code 를 검토하는 일을 한 사람이 모두 맡는다.
- **constitution**: Spec Kit 에서 모든 기능에 공통으로 적용하는 project 규칙을 적은 파일.
- **context drift**: 대화가 길어지면서 agent 가 앞서 정한 조건과 설계를 놓치는 현상.
- **delta spec**: 기존 spec 에 대한 변경분만을 ADDED, MODIFIED, REMOVED 로 나누어 적은 spec.
- **edge case**: 입력이 허용 범위의 경계나 예외에 놓인 경우. 빈 DataFrame, 결측값, 정렬되지 않은 시간 열이 그 예다.
- **Given-When-Then**: 입력 상태 (Given), 동작 (When), 기대 결과 (Then) 의 세 부분으로 acceptance scenario 를 적는 형식.
- **hallucination**: LLM 이 존재하지 않는 API, 파일, 사실을 그럴듯하게 만들어 내는 현상.
- **LLM**: Large language model. 대량의 text 로 학습하여 다음 token 을 예측하는 model.
- **look-ahead leakage**: Feature 를 계산할 때 그 시점 이후의 값이 들어가, 학습 결과가 실제 운용보다 좋게 나오는 오류.
- **plan**: Spec 을 구현하는 기술적 방법을 적은 파일. Agent 가 초안을 쓰고 reviewer 가 승인한다.
- **post-condition**: 함수가 끝난 뒤 반드시 성립해야 하는 조건.
- **pre-condition**: 함수를 호출하기 전에 입력이 만족해야 하는 조건.
- **property-based test**: 입력을 무작위로 많이 만들어, 모든 입력에서 성립해야 하는 조건을 검사하는 test.
- **regression test**: 이미 있는 동작이 변경 뒤에도 그대로인지 검사하는 test.
- **reviewer**: 구현 전에 agent 의 plan 과 test 를 승인하고, 구현 뒤에 agent 가 acceptance criteria 를 통과시킨 결과를 spec 에 비추어 검수하여, 수용할지 agent 나 specifier 에게 돌려보낼지 정하는 사람의 역할.
- **single source of truth**: 같은 정보를 한 곳에만 두어, 다른 모든 곳이 그곳을 따르게 하는 원칙.
- **spec**: 구현할 기능의 범위, data schema, 동작, 검증 조건을 적은 문서.
- **specifier**: 구현할 기능의 spec 을 쓰고, reviewer 가 돌려보낸 결함에 맞춰 spec 을 고치는 사람의 역할.
- **steering**: Kiro 에서 agent 가 모든 작업에 따르도록 project 관례를 적은 파일.
- **tasks**: Plan 을 agent 가 차례로 실행할 작업 단위로 나눈 목록 파일.
- **TDD**: Test-driven development. 구현보다 test 를 먼저 쓰고, 그 test 를 통과시키는 code 를 쓰는 개발 방식.
