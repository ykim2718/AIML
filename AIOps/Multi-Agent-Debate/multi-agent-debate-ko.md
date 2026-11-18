# Multi-Agent Debate (MAD) for Model Specification
Version 3 | Created: 2026-11-18 | Updated: 2026-11-18 10:34 UTC

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Process Flow](#3-process-flow)
- [4. Roles and Prompts](#4-roles-and-prompts)
    - [4.1 Spec Prompt](#41-spec-prompt)
    - [4.2 Roles](#42-roles)
    - [4.3 Defect Checklist](#43-defect-checklist)
    - [4.4 Prompts](#44-prompts)
    - [4.5 Code-Settled and Human-Settled Defects](#45-code-settled-and-human-settled-defects)
- [5. Worked Example: pls_rsm.py](#5-worked-example-pls_rsmpy)
    - [5.1 Source Model](#51-source-model)
    - [5.2 Draft Spec](#52-draft-spec)
    - [5.3 Debate Rounds](#53-debate-rounds)
    - [5.4 Human Clarification](#54-human-clarification)
    - [5.5 Final Spec](#55-final-spec)
    - [5.6 Rebuild Verification](#56-rebuild-verification)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

**Problem Statement**:<br>
이미 있는 model code 를 읽고 spec 을 쓰면, 한 agent 가 code 에 없는 의도를 지어 넣거나 code 에 있는 동작을 빠뜨려, 그 spec 으로 다시 build 한 model 이 기존 model 과 다르게 동작한다.

**Goal**:

1. 독자가 기존 model code 를 읽어 markdown spec prompt 를 쓰는 process flow 를 단계마다 입력, 출력, 맡는 역할로 따라갈 수 있다.
2. 독자가 실제 model 의 spec 예에서 비판자가 code 와 spec 을 견주어 빠진 것, 모호한 것, 어긋난 것을 찾고 작성자가 고치는 과정을 확인할 수 있다.
3. 독자가 code 로 확정할 수 있는 항목과 사람이 확정해야 할 항목 (`[NEEDS CLARIFICATION]`) 을 가를 수 있다.

**Non-Goal**:

- 기존 model 없이 새 model 의 spec 을 처음부터 쓰는 방법은 다루지 않는다.
- Spec 으로 model 을 다시 build 하는 code 의 작성 방법은 다루지 않는다.
- MAD 의 갈래 비교와 문헌 조사는 다루지 않는다.

## 2. Summary

- 기존 model 의 spec 은 작성자가 code 를 읽어 쓴 초안을 비판자가 code 와 견주어 검사하고, 작성자가 고치거나 반박하며, judge 가 종료를 정하는 debate 로 다듬는다.
- Process flow 는 code 읽기, 초안 작성, debate round, 사람의 확정, spec 만으로 다시 build, 원본과의 비교 검증의 여섯 단계이다.
- 기존 code 가 동작의 기준이므로, code 에 있는 동작에 관한 결함은 agent 가 code 를 근거로 확정하고, 허용 오차나 library version 처럼 code 에 없는 사실이 필요한 결함만 `[NEEDS CLARIFICATION]` 으로 사람에게 넘긴다.
- 검증은 같은 data 를 원본 model 과 다시 build 한 model 에 넣어 acceptance criteria 를 돌리는 것이며, 실패하면 원인이 spec 에 있으면 debate 로, code 에 있으면 다시 build 로 돌아간다.

## 3. Process Flow

기존 model 에서 spec 을 얻는 일은 여섯 단계이며, debate 와 검증 두 곳에서 앞 단계로 돌아가는 loop 가 있다. 단계마다의 역할, 입력, 출력은 아래 Fig 1 이다.

```text
STEP           ROLE           INPUT                                 OUTPUT

1. Read        Author         existing model code                   function list, in-scope / out-of-scope
2. Draft       Author         code, scope                           draft spec (v0)
3. Debate      Critics        code, spec                            defect list
               Author         code, defect list                     revised spec, rebuttals, [NEEDS CLARIFICATION]
               Judge          code, defect list, author response    CONTINUE -> 3 | STOP -> 4
4. Clarify     Human          [NEEDS CLARIFICATION] list            settled values written into the spec
5. Rebuild     Coding agent   final spec only                       rebuilt model code
6. Verify      Script         original code, rebuilt code, data     all AC pass -> done
                                                                    AC fail, defect in spec -> 3
                                                                    AC fail, defect in rebuilt code -> 5
```

<a id="fig-1"></a>
Fig 1. Process flow from an existing model to a verified spec

- 1 Read: 작성자가 code 의 함수와 상수를 훑어, spec 이 다룰 model 부분과 다루지 않을 부분 (data 생성, 그림, CLI 등) 을 가른다.
- 2 Draft: 작성자가 code 를 읽어 spec 의 네 요소를 채운 초안 v0 을 쓴다.
- 3 Debate: 비판자는 spec 의 문장마다 code 의 해당 줄과 견주어 결함을 찾고, 작성자는 결함마다 고치거나 반박하며, judge 는 반박이 맞는지 판정한 뒤 계속과 종료를 정한다.
- 4 Clarify: 사람이 `[NEEDS CLARIFICATION]` 마다 값을 정해 spec 에 적는다.
- 5 Rebuild: Coding agent 가 기존 code 를 보지 않고 최종 spec 만으로 model code 를 다시 쓴다.
- 6 Verify: Script 가 acceptance criteria 를 돌리고, 같은 data 로 원본과 다시 build 한 model 의 결과를 견준다.

단계 5 와 6 은 SDD 와 겹치며, 구현과 자가 검증의 진행은 [Spec-Driven Development (SDD) 4.4 Phase 3: Implementation](../SDD/spec-driven-development-ko.md#44-phase-3-implementation) 과 [4.5 Phase 4: Self-Verification Loop](../SDD/spec-driven-development-ko.md#45-phase-4-self-verification-loop) 에 있다. 이 문서의 단계 6 은 SDD 와 달리 기존 model 이 있으므로, 같은 data 에서 원본과 결과가 같은지를 acceptance criteria 에 넣는다.

## 4. Roles and Prompts

Spec debate 는 작성자, 비판자, judge 의 세 agent 와, `[NEEDS CLARIFICATION]` 을 정하는 사람이 맡으며, 모든 agent 는 기존 code 를 함께 받는다.

### 4.1 Spec Prompt

Spec 은 coding agent 에 model build 를 맡길 때 주는 markdown 형식의 prompt 이며, scope, data schema, behavior, acceptance criteria 의 네 요소로 이루어진다. 이 네 요소는 SDD 와 겹치며, 각 요소의 정의는 [Spec-Driven Development (SDD) 3.2 Spec Components](../SDD/spec-driven-development-ko.md#32-spec-components) 에 있다.

- Scope: 만들 파일, 쓸 library 와 그 version, 다루지 않는 부분.
- Data schema: 함수의 인자와 반환값, 그 shape 와 type, 중간 행렬의 열 순서.
- Behavior: 번호를 붙인 요구사항 (FR) 과 예외에서의 동작.
- Acceptance criteria: 통과 여부를 가르는 test 와 원본과의 비교 기준 (AC).

### 4.2 Roles

Table 1. Roles in a spec debate on an existing model

| Role         | Input                          | Output                                   | Prompt focus                                        |
| :----------: | :----------------------------: | :--------------------------------------: | :-------------------------------------------------: |
| Author       | code, 앞 round 의 결함 목록    | 고친 spec, 결함마다 고침 또는 반박       | code 에 없는 동작은 쓰지 않음                       |
| Critic       | code, spec                     | 결함 목록 (요소, 유형, 위치, code 근거)  | 비판자마다 다른 관점, spec 은 고쳐 쓰지 않음        |
| Judge        | code, 결함 목록, 작성자의 응답 | 반박된 결함마다 판정, CONTINUE 또는 STOP | code 를 근거로 판정, 마지막 round 면 남은 결함 표시 |
| Human        | `[NEEDS CLARIFICATION]` 목록   | 확정한 값                                | -                                                   |
| Coding agent | 최종 spec                      | 다시 build 한 model code                 | 기존 code 를 보지 않음                              |

작성자와 비판자의 loop 는 MARE [[1](#ref-1)] 를 따른다. MARE 에서는 checker 가 요구사항 초안에서 결함 (requirements smell) 을 찾으면 collector 와 modeler 가 초안을 고치고, 결함이 없을 때 documenter 가 요구사항 명세를 쓴다. Judge 는 Liang 등 [[2](#ref-2)] 의 debate 에서 종료를 정하는 역할을 따른다. 같은 문헌에서 debater 와 judge 가 서로 다른 LLM 을 쓰면 judge 가 자기와 같은 LLM 을 쓰는 쪽을 편들었으므로, 세 agent 는 같은 LLM 으로 둔다. 비판자에게는 서로 다른 관점의 역할 prompt 를 준다. ChatEval [[3](#ref-3)] 에서 두 agent 에게 같은 역할 설명을 주면 정확도가 단일 agent 와 같았다.

### 4.3 Defect Checklist

비판자가 찾는 결함은 spec 의 네 요소와 세 결함 유형의 조합이며, 기존 model 이 있으면 결함의 근거는 언제나 code 의 줄이다. Spec Kit [[4](#ref-4)] 은 spec 에서 모호한 것 (ambiguity), 어긋난 것 (contradiction), 빠진 것 (gap) 을 찾으라고 하고, MARE [[1](#ref-1)] 의 checker 는 요구사항의 correctness, completeness, consistency 를 검사한다.

Table 2. Defect checklist against existing code

| Element             | Missing                                  | Ambiguous                                     | Contradictory                                    |
| :-----------------: | :--------------------------------------: | :-------------------------------------------: | :----------------------------------------------: |
| Scope               | code 가 쓰는 library 가 빠짐             | library 의 version 을 정하지 않음             | scope 밖이라 한 함수의 동작을 behavior 가 요구함 |
| Data schema         | 인자, 반환값, 중간 행렬의 열 순서가 빠짐 | shape 나 type 을 정하지 않음                  | schema 의 출력을 만드는 FR 이 없음               |
| Behavior            | code 의 분기, 예외, 경고가 빠짐          | 난수 seed, fold, 동점 처리 규칙을 정하지 않음 | 식이나 상수가 code 와 다름                       |
| Acceptance criteria | FR 번호에 짝지은 test 가 없음            | 허용 오차나 비교할 data 가 없음               | 기대값이 behavior 로 계산한 값과 다름            |

### 4.4 Prompts

세 agent 의 prompt 는 아래이다. 비판자 prompt 의 "You need not agree with the author" 는 Liang 등 [[2](#ref-2)] 의 meta prompt 가 debater 에게 서로 완전히 동의할 필요가 없다고 한 것을 따른다.

```markdown
You are the author of a spec for the existing model code below.
Write the spec in four sections: Scope, Data Schema, Behavior (numbered FR-n), Acceptance Criteria (numbered AC-n).
State only what the code does. Do not add checks, defaults or intentions that the code does not contain.
In later rounds, for each defect in the critics' list, either revise the spec or rebut the defect in one sentence
that cites the code line. If a revision needs a fact the code does not contain, write
[NEEDS CLARIFICATION: <QUESTION>] instead of guessing.

<CODE>
<SPEC>
<DEFECT_LIST>
```

```markdown
You are a critic of a spec written for the existing model code below. Your view: <CRITIC_VIEW>.
Compare every spec sentence with the code and list every defect, one item per defect:
- element: Scope | Data Schema | Behavior | Acceptance Criteria
- type: missing | ambiguous | contradictory
- location: <SPEC_ITEM>
- evidence: <CODE_LINE>
Do not rewrite the spec. You need not agree with the author; the goal is a spec that rebuilds the same model.

<CODE>
<SPEC>
```

```markdown
You are the judge of a spec debate on the existing model code below.
For each defect the author rebutted, decide "defect stands" or "rebuttal stands" from the code, in one sentence.
Answer CONTINUE if the spec changed in this round or any defect stands; otherwise answer STOP.
If this is round <MAX_ROUND>, answer STOP and write each defect that still stands as [NEEDS CLARIFICATION: <QUESTION>].

<CODE>
<DEFECT_LIST>
<AUTHOR_RESPONSE>
```

Round 상한은 Liang 등 [[2](#ref-2)] 의 실험 설정인 3 회를 쓴다.

### 4.5 Code-Settled and Human-Settled Defects

기존 model 이 있으면 code 가 동작의 기준이므로, code 에 있는 동작에 관한 결함은 agent 가 code 를 근거로 확정하고, code 에 없는 사실이 필요한 결함과 round 상한까지 판정되지 않은 결함만 사람이 확정한다. `[NEEDS CLARIFICATION]` 표시는 SDD 와 겹치며, 그 표기는 [Spec-Driven Development (SDD) 3.3 Requirement Notation](../SDD/spec-driven-development-ko.md#33-requirement-notation) 에 있다.

Table 3. Who settles a defect

<table>
  <thead>
    <tr>
      <th align="center">Settled by</th>
      <th align="center">Defect</th>
      <th align="center">Example in section 5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Agent</td>
      <td align="center">code 에 있는 동작을 빠뜨리거나 다르게 적은 결함</td>
      <td align="center">fold 의 seed, 열 순서, 성분 수의 상한</td>
    </tr>
    <tr>
      <td align="center" rowspan="2">Human</td>
      <td align="center">code 에 없는 사실이 필요한 결함</td>
      <td align="center">원본과 견줄 허용 오차, library version</td>
    </tr>
    <tr>
      <td align="center">round 상한까지 판정되지 않은 결함</td>
      <td align="center">judge 가 마지막 round 에 남긴 <code>[NEEDS CLARIFICATION]</code></td>
    </tr>
  </tbody>
</table>

Code 에 없는 사실을 agent 끼리 합의로 채우지 않는다. Spec Kit [[4](#ref-4)] 은 정하지 않은 것을 추측하지 말고 `[NEEDS CLARIFICATION: <QUESTION>]` 으로 표시하라고 하며, 이 표시가 LLM 이 그럴듯하지만 틀릴 수 있는 가정을 세우는 것을 막는다고 적는다. 작성자는 code 에 없는 검사나 의도를 spec 에 넣자는 결함을 반박하고, judge 는 code 에 그 동작이 없으면 반박을 받아들인다.

## 5. Worked Example: pls_rsm.py

이 예는 이 문서와 같은 folder 의 [`src/pls_rsm.py`](src/pls_rsm.py) 를 기존 model 로 삼아 Fig 1 의 여섯 단계를 실제로 진행한 기록이다. 단계 1 에서 4 의 작성자, 비판자, judge, 사람 역할과 단계 5 의 coding agent 역할은 이 문서의 작성자가 차례로 맡았다.

### 5.1 Source Model

`pls_rsm.py` 는 coded factor 설정 x 와 response y 로 2차 response surface 를 partial least squares (PLS) 로 맞추고, 그 surface 의 stationary point 를 구하는 script 이다. Script 는 같은 참 surface 에서 rotatable central composite design 과 factor 끼리 상관된 operating data 를 만들어 PLS 와 OLS 의 결과를 견주며, 아래 명령으로 실행한다.

```bash
python3 src/pls_rsm.py --output-folder <OUTPUT_FOLDER>
```

단계 1 에서 작성자는 함수를 아래처럼 갈랐다.

Table 4. Functions of pls_rsm.py in and out of the spec scope

| Function                        | Scope | Role in the script                           |
| :-----------------------------: | :---: | :------------------------------------------: |
| `expand_quadratic`              | In    | x 에 제곱 열과 두 factor 곱 열을 더함        |
| `fit_pls_surface`               | In    | 교차 검증으로 PLS 성분 수를 고르고 다시 맞춤 |
| `quadratic_from_coefficients`   | In    | 계수를 1차 vector b 와 대칭 행렬 B 로 나눔   |
| `stationary_point`              | In    | B 의 고유값과 stationary point 를 구함       |
| `true_response`                 | Out   | 참 surface 로 y 를 만듦                      |
| `central_composite_design`      | Out   | rotatable central composite design 을 만듦   |
| `correlated_operating_data`     | Out   | 상관된 operating data 를 만듦                |
| `fit_ols_surface`               | Out   | 같은 확장 행렬로 OLS 를 풂                   |
| `evaluate_design`,<br>`run`     | Out   | 두 방법의 결과를 모으고 표와 그림을 씀       |
| `plot_results`,<br>`parse_args` | Out   | 그림과 CLI                                   |

### 5.2 Draft Spec

단계 2 에서 작성자가 code 를 한 번 읽고 쓴 초안 v0 은 아래이다.

```markdown
# Spec: PLS response surface

## Scope
- File: pls_rsm_rebuilt.py
- Python, numpy, scikit-learn

## Data Schema
- Input x: array (n, k), coded factor settings
- Input y: array (n,), response
- Output: stationary point (k,), eigenvalues (k,), component count

## Behavior
- FR-1: Expand x with squares and pairwise products.
- FR-2: Fit PLSRegression, choosing the number of components by cross-validation.
- FR-3: Convert the coefficients into a linear vector b and a matrix B.
- FR-4: Return the stationary point -0.5 B^-1 b and the eigenvalues of B.

## Acceptance Criteria
- Results should be close to the original model.
```

### 5.3 Debate Rounds

단계 3 은 세 round 로 끝났다. 비판자 A 의 관점은 "data schema 와 behavior 가 code 와 같은가" 이고, 비판자 B 의 관점은 "scope 와 acceptance criteria 로 같은 model 을 다시 build 하고 검증할 수 있는가" 이다.

Table 5. Defects found in round 1

| ID    | Critic | Element             | Type          | Code evidence                                                     | Author     | Judge           |
| :---: | :----: | :-----------------: | :-----------: | :---------------------------------------------------------------: | :--------: | :-------------: |
| D1    | A      | Data schema         | missing       | 확장 행렬의 열을 1차, 제곱, i < j 인 곱의 차례로 쌓음             | 고침       | -               |
| D2    | A      | Behavior            | ambiguous     | `KFold(min(n_splits, n), shuffle=True, random_state=0)` 과 RMSE   | 고침       | -               |
| D3    | A      | Behavior            | missing       | 성분 수 상한을 rank 와 fold 크기로 제한하고 1 미만이면 오류       | 고침       | -               |
| D4    | A      | Behavior            | missing       | 예측이 유한하지 않으면 그 성분 수에서 탐색을 멈추고 stderr 에 씀  | 고침       | -               |
| D5    | A      | Behavior            | ambiguous     | B 의 비대각 원소는 곱 계수의 절반                                 | 고침       | -               |
| D6    | A      | Behavior            | missing       | B 가 singular 이면 `LinAlgError`                                  | 고침       | -               |
| D7    | A      | Data schema         | missing       | x 가 coded 범위 안에 있어야 한다는 조건이 없음                    | 반박       | rebuttal stands |
| D8    | A      | Behavior            | ambiguous     | `PLSRegression` 기본값 (scale=True), intercept 없이 `coef_` 만 씀 | 고침       | -               |
| D9    | B      | Data schema         | missing       | 인자 `max_components`, `n_splits` 와 반환값 cv RMSE, b, B         | 고침       | -               |
| D10   | B      | Acceptance criteria | ambiguous     | 비교할 data 와 허용 오차가 없음                                   | 고침, 질문 | -               |
| D11   | B      | Scope               | ambiguous     | code 가 library version 을 정하지 않음                            | 질문       | -               |
| D12   | B      | Acceptance criteria | missing       | FR 번호에 짝지은 test 가 없음                                     | 고침       | -               |
| D13   | B      | Data schema         | contradictory | 출력의 component count 를 만드는 FR 이 없음                       | 고침       | -               |

Author 열의 "질문" 은 작성자가 그 결함을 `[NEEDS CLARIFICATION]` 으로 적었다는 뜻이다.

- D7: 작성자는 code 에 x 의 범위를 검사하는 줄이 없고 `correlated_operating_data` 가 x 를 coded 범위가 아니라 표준편차 1 로 맞춘다는 것을 근거로 반박했고, judge 는 code 에 없는 검사를 spec 에 넣지 않는다고 보아 반박을 받아들였다.
- D10: 비교할 data 는 code 의 두 design 과 CLI 기본값으로 정했고, 허용 오차는 code 에 없어 `[NEEDS CLARIFICATION]` 으로 남겼다.
- Judge 는 spec 이 바뀌었으므로 CONTINUE 로 답했다.

Table 6. Defects found in round 2

| ID    | Critic | Element             | Type          | Code evidence                                                                          | Author | Judge |
| :---: | :----: | :-----------------: | :-----------: | :------------------------------------------------------------------------------------: | :----: | :---: |
| D14   | A      | Behavior            | ambiguous     | 상한의 fold 크기는 `ceil(n / folds.get_n_splits())` 이고, fold 수는 `min(n_splits, n)` | 고침   | -     |
| D15   | B      | Acceptance criteria | contradictory | v1 의 AC-2 는 4 행에서 오류를 기대하지만, FR-3 대로면 4 행의 상한은 2                  | 고침   | -     |

- D14: v1 의 FR-3 은 fold 크기를 `ceil(n / n_splits)` 로 적어, 행이 `n_splits` 보다 적을 때 code 와 다른 상한을 준다.
- D15: 4 행이면 fold 4 개, fold 크기 1 이므로 상한은 min(9, rank, 4 - 1 - 1) = 2 이고 오류가 나지 않는다. 작성자는 상한이 1 미만이 되는 2 행으로 AC-2 를 고쳤다.
- Judge 는 spec 이 바뀌었으므로 CONTINUE 로 답했다.

Round 3 에서 두 비판자는 새 결함을 찾지 못했고, judge 는 STOP 으로 답하며 D10 과 D11 의 `[NEEDS CLARIFICATION]` 두 개를 단계 4 로 넘겼다.

### 5.4 Human Clarification

단계 4 에서 사람 역할이 두 항목을 아래처럼 정했다.

Table 7. Values settled by the human

| ID    | Question                                | Settled value                                                   |
| :---: | :-------------------------------------: | :-------------------------------------------------------------: |
| C1    | 원본과 다시 build 한 model 의 허용 오차 | 성분 수는 같고, 실수 결과는 `np.allclose(rtol=1e-9, atol=1e-9)` |
| C2    | library version                         | scikit-learn 1.9.1, numpy 2.5.3                                 |

C2 는 이 예를 실행한 환경에 설치된 version 이다.

### 5.5 Final Spec

단계 4 까지 마친 최종 spec v3 은 아래이며, 단계 5 의 coding agent 는 이 spec 만 받는다.

```markdown
# Spec: PLS second-order response surface and its stationary point

## Scope
- New module: pls_rsm_rebuilt.py. No CLI, no plotting.
- Python 3, numpy 2.5.3, scikit-learn 1.9.1 (PLSRegression, KFold, cross_val_predict).
- Out of scope: generating designs, the true surface, the OLS comparison, figures.

## Data Schema
- x: float array of shape (n, k), coded factor settings, one run per row. No range check.
- y: float array of shape (n,), response.
- max_components: int >= 1, default 9.
- n_splits: int >= 2, default 5.
- Expanded matrix: shape (n, 2k + k(k-1)/2), columns in this order:
  x1..xk, then x1^2..xk^2, then xi*xj for i < j in lexicographic order, named "xixj".
- Public functions:
  - expand_quadratic(x) -> (expanded matrix, column names)
  - stationary_point(b, B) -> (stationary point of shape (k,), eigenvalues of shape (k,), ascending)
  - fit_surface(x, y, max_components=9, n_splits=5) -> dict with keys
    coefficients (shape (2k + k(k-1)/2,)), n_components (int), cv_rmse (list of float, one per tried count),
    b (shape (k,)), B (shape (k, k), symmetric), stationary_point, eigenvalues

## Behavior
- FR-1: Build the expanded matrix in the column order above.
- FR-2: folds = KFold(n_splits=min(n_splits, n), shuffle=True, random_state=0).
- FR-3: upper = min(max_components, rank of the column-centered expanded matrix,
  n - ceil(n / number of folds) - 1). If upper < 1, raise ValueError.
- FR-4: For count = 1..upper, predict y with cross_val_predict(PLSRegression(n_components=count), cv=folds),
  using PLSRegression defaults (scale=True), and record RMSE = sqrt(mean((y - prediction)^2)).
  If any prediction is not finite, print one line to stderr and stop before that count.
  If no count is usable, raise ValueError.
- FR-5: n_components = the count with the smallest RMSE; on a tie, the smallest count.
  Refit PLSRegression(n_components) on all rows; coefficients = coef_ flattened. The intercept is not used.
- FR-6: b_i = coefficient of xi; B_ii = coefficient of xi^2; B_ij = B_ji = half the coefficient of xixj.
- FR-7: eigenvalues = eigvalsh(B). If min |eigenvalue| < machine epsilon * max(1, max |eigenvalue|),
  raise numpy.linalg.LinAlgError. Otherwise stationary point = -0.5 * solve(B, b).

## Acceptance Criteria
- AC-1 (FR-1): for k = 3 the column names are x1 x2 x3 x1^2 x2^2 x3^2 x1x2 x1x3 x2x3.
- AC-2 (FR-3): fit_surface on 2 rows raises ValueError.
- AC-3 (FR-7): stationary_point(b=[1, 1], B=diag(1, 0)) raises numpy.linalg.LinAlgError.
- AC-4 (FR-1 to FR-7): on the central composite design and the correlated operating data of the original
  pls_rsm.py (n_runs 20, correlation 0.85, noise_sd 0.35, seed 20260906 + replicate, 200 replicates),
  n_components equals the original, and coefficients, cv_rmse, stationary point and eigenvalues satisfy
  np.allclose(rtol=1e-9, atol=1e-9) against the original.
```

### 5.6 Rebuild Verification

단계 5 에서 coding agent 가 최종 spec 으로 [`src/pls_rsm_rebuilt.py`](src/pls_rsm_rebuilt.py) 를 썼고, 단계 6 의 [`src/verify_rebuild.py`](src/verify_rebuild.py) 가 AC-1 에서 AC-4 를 돌렸다. `verify_rebuild.py` 는 AC-1 에서 AC-3 을 다시 build 한 module 로 검사하고, AC-4 는 원본 `pls_rsm.py` 의 함수와 다시 build 한 `fit_surface` 에 같은 data 를 넣어 결과를 견준 뒤, replicate 와 design 마다 한 행씩 `rebuild_comparison.csv` 에 쓴다.

```bash
cd src
python3 verify_rebuild.py --output-folder <OUTPUT_FOLDER>
```

Table 8. Acceptance criteria results of the rebuild

| AC    | Check                                      | Result                        |
| :---: | :----------------------------------------: | :---------------------------: |
| AC-1  | k = 3 의 확장 열 이름과 차례               | pass                          |
| AC-2  | 2 행에서 ValueError                        | pass                          |
| AC-3  | singular B 에서 LinAlgError                | pass                          |
| AC-4  | 200 replicate × 2 design 의 원본 대비 결과 | pass, 400 fit 가운데 400 일치 |

AC-4 의 400 fit 에서 계수와 stationary point 의 최대 절대 차이는 모두 0 이었다. 이 예의 coding agent 는 단계 1 에서 원본 code 를 읽은 작성자와 같은 agent 이므로, 이 결과는 spec 만으로 같은 model 을 다시 build 할 수 있음을 보여 주지만, 원본을 본 적 없는 coding agent 로도 같은 결과가 나오는지는 확인하지 않았다.

## References

<a id="ref-1"></a>
[1] Jin, D., Jin, Z., Chen, X., & Wang, C. (2024). [MARE: Multi-Agents Collaboration Framework for Requirements Engineering](https://doi.org/10.48550/arXiv.2405.03256). *arXiv preprint* arXiv:2405.03256.<br>
<a id="ref-2"></a>
[2] Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., Yang, Y., Shi, S., & Tu, Z. (2024). [Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate](https://doi.org/10.48550/arXiv.2305.19118). *Proceedings of EMNLP 2024*.<br>
<a id="ref-3"></a>
[3] Chan, C.-M., Chen, W., Su, Y., Yu, J., Xue, W., Zhang, S., Fu, J., & Liu, Z. (2023). [ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate](https://doi.org/10.48550/arXiv.2308.07201). *arXiv preprint* arXiv:2308.07201.<br>
<a id="ref-4"></a>
[4] GitHub. (n.d.). [Specification-Driven Development (SDD)](https://github.com/github/spec-kit/blob/main/spec-driven.md). *Spec Kit repository*.

---

## Appendix A. Terminology

- **Acceptance criteria**: spec 의 네 요소 가운데 하나로, 통과 여부를 가르는 test 와 원본과의 비교 기준. 항목마다 AC 번호를 붙인다.
- **Author**: code 를 읽어 spec 을 쓰고, 비판자의 결함마다 고치거나 반박하는 agent.
- **Behavior**: spec 의 네 요소 가운데 하나로, 번호를 붙인 요구사항 (FR) 과 예외에서의 동작.
- **Central composite design**: factorial 점, 축 위의 star 점, 중심점을 합친 실험 설계. Rotatable 이면 중심에서 같은 거리의 점에서 예측 분산이 같다.
- **Coding agent**: 최종 spec 만 받아 model code 를 다시 쓰는 agent.
- **Critic**: code 와 spec 을 견주어 결함 목록만 돌려주고 spec 은 고쳐 쓰지 않는 agent.
- **Data schema**: spec 의 네 요소 가운데 하나로, 함수의 인자와 반환값, 그 shape 와 type, 중간 행렬의 열 순서.
- **Judge**: 반박된 결함을 code 를 근거로 판정하고 debate 의 계속 (CONTINUE) 과 종료 (STOP) 를 정하는 agent.
- **Multi-Agent Debate (MAD)**: 여러 LLM agent 가 prompt 로 서로의 답을 받아 여러 round 에 걸쳐 고쳐 쓰게 하는 방법.
- **NEEDS CLARIFICATION**: spec 에서 정하지 못한 것을 추측으로 채우지 않고 사람에게 묻도록 남기는 표시.
- **Partial least squares (PLS)**: 입력과 출력의 공분산이 큰 방향으로 성분을 뽑아 회귀하는 방법. 성분 수가 model 의 hyperparameter 이다.
- **Round**: debate 에서 비판자, 작성자, judge 가 차례로 한 번씩 답하는 단위.
- **Scope**: spec 의 네 요소 가운데 하나로, 만들 파일, 쓸 library 와 그 version, 다루지 않는 부분.
- **Spec**: coding agent 에 model build 를 맡길 때 주는 markdown 형식의 prompt.
- **Stationary point**: response surface 의 gradient 가 0 인 점. B 의 고유값 부호로 최대, 최소, saddle 을 가른다.
