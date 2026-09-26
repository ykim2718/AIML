# Spec-Driven Development
Rev. 0 | Created: 2026-09-26 | Updated: 2026-09-26 13:32 CDT

## 1. Purpose

- **Problem Statement**: Prompt 만으로 AI coding agent 에 구현을 맡기면 codebase 가 커질수록 기존 설계와의 불일치, side effect, 품질 저하가 생긴다.
- **Goal**: Data science 엔지니어가 agent 에 맡길 작업을 spec 파일 하나로 정의하고, 그 spec 의 acceptance criteria 로 결과의 통과 여부를 판정할 수 있게 한다.
- **Non-Goal**: 특정 agent 제품의 설치와 설정, 그리고 TDD 와 결합하는 절차는 다루지 않는다.

## 2. Summary

Spec-Driven Development (SDD) 는 사람이 구조와 명세 (spec) 를 정하고, AI coding agent 가 그 spec 에 따라 code 를 구현하고 검증하는 개발 방식이다. Spec 파일은 네 요소, 곧 scope, data schema, behavior, acceptance criteria 를 담는다. Agent 는 scope 안의 파일만 고치고, acceptance criteria 에 적힌 명령이 모두 통과할 때까지 구현을 되풀이한다.

Spec 파일은 대화 이력 대신 single source of truth 가 되어, LLM 의 hallucination 과 context drift 를 줄인다. 같은 spec 에서는 실행마다 결과가 같은 기준으로 판정되므로, 대규모 codebase 에서도 agent 의 결과를 예측할 수 있다. 개발자의 역할은 code 를 직접 입력하는 writer 에서 system 을 설계하고 명세하는 specifier 와 reviewer 로 옮겨 간다.

## 3. Taxonomy and its Hierarchy

AI 를 쓰는 개발 방식은 agent 에게 설계 결정까지 맡기는가로 갈린다. Prompt-based development 는 설계와 구현을 모두 agent 에게 맡기고, SDD 는 설계를 spec 으로 고정한 뒤 구현만 맡긴다. [Fig 1](#fig-1) 은 두 방식과 SDD spec 의 네 요소를 보여 준다.

```text
AI-assisted coding
├── Prompt-based development   ad-hoc natural-language prompt; agent designs and implements
└── Spec-driven development    structured spec file; agent implements within the spec
    ├── 1. Scope & Context       target files, tech stack, library versions
    ├── 2. Data Schema           input/output schema, DataFrame and table contracts
    ├── 3. Behavior              pre-/post-conditions, edge cases, error handling
    └── 4. Acceptance Criteria   test and check commands that decide pass or fail
```

<a id="fig-1"></a>
Fig 1. Development styles and the four components of a spec

Spec 의 네 요소는 위에서 아래로 agent 가 스스로 정할 수 있는 폭을 좁힌다. Scope 는 손댈 파일과 library 를, data schema 는 입출력의 모양을, behavior 는 경계 조건에서의 동작을 고정한다. Acceptance criteria 는 앞의 세 요소를 실행할 수 있는 명령으로 바꾸어, 통과 여부를 사람의 판단 없이 가린다. 앞의 세 요소만 있고 acceptance criteria 가 없으면 agent 는 구현을 마쳤다고 보고할 뿐 spec 을 지켰는지는 확인하지 않는다.

### 3.1 Placement

아래 표는 두 방식을 같은 항목으로 나란히 놓는다.

Table 1. Prompt-based development and spec-driven development

| Aspect          | Prompt-based development                  | Spec-driven development                      |
| :-------------: | :---------------------------------------: | :------------------------------------------: |
| Anchor          | 자연어 prompt (ad-hoc)                    | 구조화된 spec 문서 (spec as code)            |
| Agent role      | 자율적인 설계와 구현                      | Spec 에 묶인 구현과 자가 검증                |
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

## 4. Workflow

SDD 는 spec 작성, agent 실행, 자가 검증의 세 단계로 진행하며, 검증이 실패하면 agent 가 구현 단계로 돌아간다. [Fig 2](#fig-2) 는 그 흐름을 보여 준다.

```text
[ Requirements ] ──> [ Spec file (Markdown/Schema) ] ──> [ Agent execution ] ──> [ Spec-based verification ]
                                                                ^                          │
                                                                └──────── fail ────────────┘
```

<a id="fig-2"></a>
Fig 2. SDD workflow

### 4.1 Phase 1: Spec Writing

개발자가 구현할 기능의 spec 을 Markdown 또는 YAML 로 `<feature>.spec.md` 같은 파일에 적는다. 아래는 sensor trace 에 rolling mean feature 를 더하는 함수의 spec 이다.

````markdown
# Spec: Rolling mean feature for sensor trace

## Context & Boundaries
- Target files: `src/features/rolling.py`, `tests/test_rolling.py`
- Tech stack: Python 3.11, pandas 2.x, pytest, mypy, ruff
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

## Behavior & Logic Rules
1. `window < 1` raises `ValueError`.
2. A missing `value_col` or `time_col` raises `KeyError` naming the column.
3. An unsorted `time_col` raises `ValueError`; the function does not sort silently.
4. Each output row uses only the current and earlier rows (no look-ahead leakage).
5. The first `window - 1` rows of the new column are NaN.
6. The input `df` is not mutated.

## Verification / Acceptance Criteria
- [ ] `pytest tests/test_rolling.py` passes.
- [ ] `mypy src/features/rolling.py` reports no error.
- [ ] `ruff check src/features` reports no error.
````

이 spec 에서 behavior 의 각 항목은 `tests/test_rolling.py` 의 test 하나로 옮겨 적을 수 있다. 옮겨 적을 수 없는 항목은 agent 도 검증할 수 없으므로, 그 항목을 test 할 수 있는 문장으로 고친다.

### 4.2 Phase 2: Agent Execution

개발자는 작성한 spec 파일을 coding agent 의 입력으로 준다. Agent 는 spec 을 읽어 작업 단위 (task) 로 나누고, Context & Boundaries 에 적힌 파일만 만들거나 고친다.

### 4.3 Phase 3: Self-Verification Loop

Agent 는 code 를 쓴 뒤 Verification 에 적힌 명령을 스스로 실행한다.

- 실패하면 agent 가 error log 를 읽고 code 를 고친 뒤 명령을 다시 실행한다.
- 모두 통과하면 agent 가 commit 또는 pull request 를 만들고 개발자에게 돌려준다.

개발자는 통과한 결과를 검수하며, spec 에 없던 동작을 찾으면 spec 을 먼저 고친 뒤 agent 를 다시 실행한다.

## 5. Effects and Limits

SDD 는 agent 의 작업 범위와 판정 기준을 고정하는 대가로, spec 을 쓰는 시간을 개발자에게 요구한다.

- **Token 과 비용 절감**: Agent 가 범위 밖을 탐색하지 않고 spec 이 정한 파일 안에서만 작업한다.
- **Code 품질 유지**: Agent 가 구조를 임의로 바꾸거나 spec 에 없는 동작을 넣는 것을 막는다.
- **문서와 code 의 일치**: Spec 파일이 그대로 그 기능의 설계 문서가 된다.
- **초기 spec 작성 비용**: Spec 을 꼼꼼히 쓰는 데 개발자의 시간이 든다. Agent 에게 spec 초안을 쓰게 하고 사람이 검수하면 이 시간을 줄일 수 있다.
- **Spec 의 명확성 요구**: 모호한 spec 은 모호한 결과를 낸다. Behavior 의 모든 항목을 test 할 수 있는 수준으로 적어야 한다.

## 6. Further Work

- **TDD 와의 결합**
    - 무엇을 하는가: Acceptance criteria 의 test 를 spec 과 함께 사람이 먼저 쓰고, agent 에게는 그 test 를 통과시키는 구현만 맡긴다.
    - 왜 지금인가: SDD 의 acceptance criteria 가 이미 test 실행 명령이므로, test 를 먼저 쓰면 그 명령이 그대로 agent 의 종료 조건이 된다.
    - 무엇이 필요한가: Behavior 항목과 test 함수를 하나씩 짝짓는 팀의 test 작성 관례.

---

## Appendix A. Terminology

- **acceptance criteria**: 구현이 spec 을 만족하는지 가르는 조건. SDD 에서는 test, type check, lint 의 실행 명령으로 적는다.
- **AI coding agent**: 파일을 읽고 쓰고 명령을 실행하면서 code 를 구현하는 LLM 기반 도구.
- **context drift**: 대화가 길어지면서 agent 가 앞서 정한 조건과 설계를 놓치는 현상.
- **edge case**: 입력이 허용 범위의 경계나 예외에 놓인 경우. 빈 DataFrame, 결측값, 정렬되지 않은 시간 열이 그 예다.
- **hallucination**: LLM 이 존재하지 않는 API, 파일, 사실을 그럴듯하게 만들어 내는 현상.
- **LLM**: Large language model. 대량의 text 로 학습하여 다음 token 을 예측하는 model.
- **look-ahead leakage**: Feature 를 계산할 때 그 시점 이후의 값이 들어가, 학습 결과가 실제 운용보다 좋게 나오는 오류.
- **post-condition**: 함수가 끝난 뒤 반드시 성립해야 하는 조건.
- **pre-condition**: 함수를 호출하기 전에 입력이 만족해야 하는 조건.
- **single source of truth**: 같은 정보를 한 곳에만 두어, 다른 모든 곳이 그곳을 따르게 하는 원칙.
- **spec**: 구현할 기능의 범위, data schema, 동작, 검증 조건을 적은 문서.
- **TDD**: Test-driven development. 구현보다 test 를 먼저 쓰고, 그 test 를 통과시키는 code 를 쓰는 개발 방식.
