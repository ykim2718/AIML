# Multi-Agent Debate (MAD) for Model Specification
Version 0 | Created: 2026-11-18 | Updated: 2026-11-18 02:05 CST

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
    - [3.1 Comparison](#31-comparison)
- [4. Principle](#4-principle)
- [5. Spec Debate](#5-spec-debate)
    - [5.1 Spec Prompt](#51-spec-prompt)
    - [5.2 Roles](#52-roles)
    - [5.3 Round Procedure](#53-round-procedure)
    - [5.4 Human-Settled Defects](#54-human-settled-defects)
- [6. Application](#6-application)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

**Problem Statement**: 한 agent 나 한 사람이 쓴 model spec 에는 빠진 요구사항, 모호한 문장, 서로 어긋나는 조건이 남아, 그 spec 으로 구현한 model 이 의도와 다르게 동작한다.

**Goal**:

1. 독자가 MAD 를 역할 구조, 통신 구조, 종료와 결론 집계의 세 축으로 나누고, 그 가운데 spec 작성에 맞는 구성을 고를 수 있다.
2. 독자가 agent 끼리 prompt 로 주고받는 debate 로 spec 의 네 요소 (scope, data schema, behavior, acceptance criteria) 에서 빠진 것, 모호한 것, 어긋난 것을 찾아 고칠 수 있다.
3. 독자가 debate 가 끝난 뒤 agent 가 합의해서 확정할 항목과 사람이 확정해야 할 항목 (`[NEEDS CLARIFICATION]`) 을 가를 수 있다.

**Non-Goal**:

- Debate 로 model 을 학습시키거나 학습 데이터와 보상을 만드는 방법은 다루지 않는다.
- Spec 을 받은 뒤의 구현과 검증은 다루지 않는다.
- 구현 code 와 특정 framework (AutoGen 등) 사용법은 다루지 않는다.

## 2. Summary

- MAD 는 여러 LLM agent 가 prompt 로 서로의 답을 받아 여러 round 에 걸쳐 고쳐 쓰게 하는 방법이며, 이 문서는 MAD 로 model build 에 쓰는 markdown prompt 형식의 spec 을 다듬는다.
- Spec 작성에는 작성자 (author) 가 쓴 spec 한 판을 비판자 (critic) 가 검사하고 작성자가 고치는 구성에, 합의하지 못한 결함을 판정하고 종료를 정하는 judge 를 더해 쓴다.
- 비판자는 spec 의 네 요소마다 빠진 것, 모호한 것, 어긋난 것을 찾고, 이 round 는 judge 가 남은 결함이 없다고 판정하거나 round 상한에 닿을 때까지 되풀이된다.
- 요구사항 안에서 근거를 찾을 수 있는 결함은 agent 가 확정하고, 요구사항에 없는 사실이 필요한 결함과 round 상한까지 판정되지 않은 결함은 `[NEEDS CLARIFICATION]` 으로 남겨 사람이 확정한다.
- MAD 는 단일 agent 방법보다 늘 낫지는 않고 역할 prompt 와 반박 강도에 따라 결과가 크게 달라지며, 호출 수가 agent 수와 round 수를 곱한 만큼 는다.

## 3. Taxonomy and its Hierarchy

MAD 구성은 역할 구조, 통신 구조, 종료와 결론 집계의 세 축으로 나뉘며, spec 작성에는 작성자와 비판자가 spec 한 판을 두고 결함 목록을 주고받고 judge 가 종료와 결론을 정하는 구성을 쓴다.

세 축과 축마다의 값은 아래 Fig 1 이다.

```text
AXIS                         VALUES

Role structure           >   Symmetric peers | Author + critic | Affirmative + negative + judge
Communication            >   One-by-one | Simultaneous-talk | Simultaneous-talk with summarizer
Termination/aggregation  >   Fixed rounds, converged answer | Checker pass, final revision | Judge break, judge extraction
```

<a id="fig-1"></a>
Fig 1. Three axes of a MAD configuration

세 축은 서로 독립이어서, 한 구성은 축마다 값 하나를 골라 정한다. Du 등 [[1](#ref-1)] 은 대등한 동료 (symmetric peers) 세 agent 가 매 round 다른 agent 의 앞 round 답을 함께 받아 고쳐 쓰고, 정한 round 수가 끝나면 수렴한 답을 결론으로 삼는다. Liang 등 [[2](#ref-2)] 은 찬성 (affirmative) 과 반대 (negative) 두 debater 가 정한 차례로 말하고, judge 가 round 마다 종료를 판정하며 상한까지 결론이 없으면 debate 기록에서 답을 뽑는다.

세 역할 구조가 무엇을 가정하고, 한 단계 내려갈 때 무엇을 버리고 무엇을 더하는지는 아래 Fig 2 이다.

```text
Multi-agent debate (MAD)
├── Symmetric peers                  each agent writes a full answer; answers converge          Du et al.
│     assumes   one short answer that every agent can produce alone
├── Author + critic                  one draft; critics report defects; author revises          MARE
│     drops     independent full answers
│     adds      one artifact under revision, a defect checklist
└── Affirmative + negative + judge   two sides argue; a judge stops the debate and decides      Liang et al.
      drops     agreement as the stop rule
      adds      a judge that decides when to stop, with its own bias
```

<a id="fig-2"></a>
Fig 2. Role structures of MAD and what each one assumes, drops and adds

대등한 동료 구성은 agent 마다 답 전체를 따로 쓰게 하므로, 답이 짧아 같은지 다른지를 바로 가릴 수 있어야 수렴을 판정할 수 있다. 작성자 + 비판자 구성은 따로 쓴 답을 버리고 고쳐 갈 판 하나를 두며, MARE [[3](#ref-3)] 에서는 checker 가 요구사항 초안의 결함 (requirements smell) 을 찾으면 collector 와 modeler 가 초안과 요구사항 model 을 차례로 고치고, 결함이 없을 때 documenter 가 요구사항 명세 (SRS) 를 쓴다. 찬성 + 반대 + judge 구성은 합의를 종료 조건에서 빼고, 그 대신 judge 가 종료와 결론을 정한다.

Spec 은 여러 꼭지로 이루어진 markdown 문서이므로, agent 마다 spec 전체를 따로 쓰게 하면 어느 판으로 수렴했는지를 가릴 수 없다. 그래서 spec 작성에는 작성자 + 비판자 구성으로 spec 한 판을 고쳐 가고, 작성자가 반박해 합의하지 못한 결함만 judge 가 판정한다.

### 3.1 Comparison

Table 1. Role structures compared for spec writing

| Role structure                 | Use for spec              | Assumption                               | Setting in source                              | Breaks when                                            | Source        |
| :----------------------------: | :-----------------------: | :--------------------------------------: | :--------------------------------------------: | :----------------------------------------------------: | :-----------: |
| Symmetric peers                | 쓰지 않음                 | 각 agent 가 혼자 쓸 수 있는 짧은 답 하나 | agent 3 개, round 2 회                         | 답이 긴 문서여서 수렴을 판정할 수 없음                 | [[1](#ref-1)] |
| Author + critic                | spec 작성과 결함 탐지     | 고쳐 갈 판 하나와 결함 기준              | agent 5 개, checker 가 결함을 찾지 못할 때까지 | 결함 기준에 없는 결함은 찾지 못함                      | [[3](#ref-3)] |
| Affirmative + negative + judge | 합의하지 못한 결함의 판정 | judge 가 두 쪽을 공정하게 판정           | debater 2 개, judge 1 개, round 상한 3 회      | debater 와 judge 의 LLM 이 다르면 judge 가 한쪽을 편듦 | [[2](#ref-2)] |

Setting in source 열은 각 문헌의 실험 설정이다. Liang 등 [[2](#ref-2)] 에서 debater 와 judge 가 서로 다른 LLM 을 쓰면 judge 는 자기와 같은 LLM 을 쓰는 쪽을 편들었다.

Table 2. Communication strategies compared for spec writing

| Communication                     | Use for spec                                       | Message flow                                                           | Source                       |
| :-------------------------------: | :------------------------------------------------: | :--------------------------------------------------------------------: | :--------------------------: |
| One-by-one                        | 작성자 다음에 비판자가 말하는 차례                 | 정한 차례로 말하고, 말한 답을 뒤 agent 의 대화 이력에 더함             | [[4](#ref-4)]                |
| Simultaneous-talk                 | 여러 비판자가 서로의 결함 목록을 보지 않고 찾을 때 | 각자 자기 이력으로 답하고, round 끝에 모든 답을 모든 agent 이력에 더함 | [[4](#ref-4)]                |
| Simultaneous-talk with summarizer | 결함 목록이 길어 context 길이를 넘을 때            | round 끝에 다른 LLM 이 답을 요약하고, 요약만 이력에 더함               | [[4](#ref-4)], [[1](#ref-1)] |

Du 등 [[1](#ref-1)] 은 agent 가 많아 context 길이 오류가 날 때 다른 agent 의 답을 이어 붙이는 대신 요약해 넘겼고, 요약했을 때 성능이 이어 붙였을 때보다 높았으며 agent 를 다섯 개 이상 쓸 수 있었다.

## 4. Principle

MAD 의 한 round 는 각 agent 가 역할 prompt 와 앞 round 까지의 debate 기록을 받아 답하는 호출이며, 결과는 역할 prompt 의 다양성, 반박 강도, judge 의 종료 판정에 따라 달라진다.

한 LLM 이 자기 답을 스스로 고치는 self-reflection 에서는, LLM 이 자기 답에 확신을 가지면 처음 답이 틀렸어도 새 생각을 하지 못하는 Degeneration-of-Thought (DoT) 가 생긴다 [[2](#ref-2)]. Debate 에서는 다른 agent 의 반박이 들어오므로, 한 agent 가 처음 답에 머물러도 다른 관점의 결함이 기록에 남는다.

Round 마다 agent 가 받는 prompt 는 다른 agent 의 앞 round 답과 고쳐 쓰라는 요청으로 이루어진다. Du 등 [[1](#ref-1)] 이 쓴 짧은 형식의 prompt 는 아래이다.

```text
These are the solutions to the problem from other agents: [other answers]
Based off the opinion of other agents, can you give an updated response
```

같은 문헌에서 요청을 "Using the opinion of other agents as additional advice" 로 바꾼 긴 형식은 agent 가 답을 덜 바꾸게 해 수렴이 늦었다 [[1](#ref-1)]. Round 를 늘리면 성능이 올랐지만, 4 round 를 넘긴 성능은 4 round 와 거의 같았다 [[1](#ref-1)].

역할 prompt 는 agent 마다 달라야 한다. ChatEval [[4](#ref-4)] 에서 두 agent 에게 같은 역할 설명을 주면 정확도가 53.8 % 로 단일 agent 의 53.8 % 와 같았고, 서로 다른 역할을 주면 60 % 였다.

반박 강도 (tit for tat) 는 중간에 둔다. Liang 등 [[2](#ref-2)] 은 모두 합의하라는 수준 0 부터 모든 점에서 반대하라는 수준 3 까지 네 수준을 견주었고, 기본 설정은 반대를 요구하되 완전한 합의는 요구하지 않는 수준이었다. 반대를 더 요구하면 결과가 나아졌지만 모든 점에서 반대하라는 수준 3 은 가장 좋은 결과를 주지 않았으며, 문헌은 끝없는 반대가 양극화를 낳는다고 보았다. Smit 등 [[5](#ref-5)] 도 agent 가 서로에게 동의하는 정도를 조절하면 성능이 크게 나아진다고 보고했다.

Judge 는 두 방식으로 결론을 정한다 [[2](#ref-2)].

- Discriminative mode: round 마다 모든 debater 가 말한 뒤, 올바른 답에 이르렀는지 판정해 종료하거나 이어 간다.
- Extractive mode: round 상한까지 올바른 답이 나오지 않으면, debate 기록 전체에서 결론을 뽑는다.

Judge 가 최선의 답에 이른 round 에서 debate 를 끝내는 adaptive break 가 성능에 중요했고, 그 round 를 지나서도 debate 를 이어 가게 하면 번역 품질이 떨어졌다 [[2](#ref-2)].

## 5. Spec Debate

Spec debate 는 작성자가 쓴 markdown spec prompt 를 비판자가 네 요소와 세 결함 유형으로 검사하고 작성자가 고치는 round 를, judge 가 남은 결함이 없다고 판정하거나 round 상한에 닿을 때까지 되풀이한다.

### 5.1 Spec Prompt

Spec 은 coding agent 에 model build 를 맡길 때 주는 markdown 형식의 prompt 이며, scope, data schema, behavior, acceptance criteria 의 네 요소로 이루어진다. 이 네 요소는 SDD 와 겹치며, 각 요소의 정의는 [Spec-Driven Development (SDD) 3.2 Spec Components](../SDD/spec-driven-development-ko.md#32-spec-components) 에 있다.

- Scope: 손댈 파일, 쓸 library 와 그 version.
- Data schema: 입력과 출력의 열, type, 단위.
- Behavior: 번호를 붙인 MUST 요구사항과 경계 조건에서의 동작.
- Acceptance criteria: 통과 여부를 가르는 test 와 명령.

Debate 가 다루는 spec 의 꼴은 아래이다.

```markdown
# Spec: <MODEL_NAME>
## Scope
## Data Schema
## Behavior
## Acceptance Criteria
```

### 5.2 Roles

Spec debate 에는 작성자, 비판자, judge 의 세 agent 역할과, `[NEEDS CLARIFICATION]` 을 확정하는 사람이 있다.

Table 3. Roles in a spec debate

| Role   | Input                           | Output                               | Prompt focus                                                   |
| :----: | :-----------------------------: | :----------------------------------: | :------------------------------------------------------------: |
| Author | 요구사항, 앞 round 의 결함 목록 | 고친 spec, 결함마다 고침 또는 반박   | 모르는 사실은 지어 쓰지 않고 `[NEEDS CLARIFICATION]` 으로 표시 |
| Critic | spec                            | 결함 목록 (요소, 유형, 위치, 이유)   | 비판자마다 다른 관점, spec 은 고쳐 쓰지 않음                   |
| Judge  | spec, 결함 목록, 작성자의 반박  | 반박된 결함마다 판정, 계속 또는 종료 | 결함이 남으면 계속, 마지막 round 면 남은 결함 표시             |
| Human  | `[NEEDS CLARIFICATION]` 목록    | 확정한 값                            | -                                                              |

비판자에게는 서로 다른 역할 prompt 를 준다. 예를 들어 한 비판자는 data schema 와 behavior 를, 다른 비판자는 acceptance criteria 가 behavior 의 요구사항을 모두 검사하는지를 본다. 같은 역할 설명을 준 두 agent 의 정확도는 단일 agent 와 같았다 [[4](#ref-4)]. 작성자, 비판자, judge 는 같은 LLM 으로 둔다. Debater 와 judge 가 서로 다른 LLM 을 쓰면 judge 가 자기와 같은 LLM 을 쓰는 쪽을 편든다 [[2](#ref-2)].

비판자가 찾는 결함은 spec 의 네 요소와 세 결함 유형의 조합이다. Spec Kit [[6](#ref-6)] 은 spec 에서 모호한 것 (ambiguity), 어긋난 것 (contradiction), 빠진 것 (gap) 을 찾으라고 하고, MARE [[3](#ref-3)] 의 checker 는 요구사항의 correctness, completeness, consistency 를 검사한다.

Table 4. Defect checklist for a model spec

| Element             | Missing                                         | Ambiguous                                        | Contradictory                                 |
| :-----------------: | :---------------------------------------------: | :----------------------------------------------: | :-------------------------------------------: |
| Scope               | 손댈 파일이나 library 의 version 이 없음        | "최신 library" 처럼 version 을 정하지 않은 문장  | Behavior 가 scope 밖의 파일을 고치라고 요구함 |
| Data schema         | target 열이나 입력 열의 type 이 없음            | 결측값 처리나 단위를 정하지 않은 열              | Behavior 가 schema 에 없는 열을 씀            |
| Behavior            | 빈 입력, 결측값 같은 경계 조건의 동작이 없음    | MUST 대신 "가능하면" 으로 적은 요구사항          | 두 요구사항이 같은 입력에 다른 동작을 요구함  |
| Acceptance criteria | Behavior 의 요구사항 번호에 짝지은 test 가 없음 | metric 이름만 있고 문턱값이나 data split 이 없음 | 기준이 behavior 와 다른 metric 을 씀          |

### 5.3 Round Procedure

Spec debate 의 round 는 아래 차례로 진행하며, 통신 구조는 작성자 다음에 비판자와 judge 가 말하는 one-by-one 이다.

1. 작성자가 사람이 준 요구사항으로 spec 초안을 쓴다.
2. 비판자가 spec 을 받아 Table 4 의 결함을 찾고, 결함마다 요소, 유형, 위치, 이유를 적은 목록을 돌려준다. 비판자가 여럿이면 simultaneous-talk 로 서로의 목록을 보지 않고 찾게 한다.
3. 작성자가 결함마다 spec 을 고치거나 한 문장의 근거로 반박한다. 고치는 데 요구사항에 없는 사실이 필요하면 `[NEEDS CLARIFICATION: <QUESTION>]` 으로 적는다.
4. Judge 가 반박된 결함마다 결함이 맞는지 반박이 맞는지 판정한다. 남은 결함이 있으면 2 로 돌아가고, 없으면 종료한다.
5. Round 상한에 닿으면 judge 가 종료하고, 남은 결함을 `[NEEDS CLARIFICATION]` 으로 적는다.
6. 사람이 `[NEEDS CLARIFICATION]` 마다 값을 적고, 그 값이 다른 요소와 어긋나는지 보도록 2 부터 한 round 를 더 돌린다.

Round 상한은 문헌의 설정을 따른다. Liang 등 [[2](#ref-2)] 은 상한을 3 회로 두었고, Du 등 [[1](#ref-1)] 에서는 4 round 를 넘긴 성능이 4 round 와 같았다.

세 역할의 prompt 는 아래이다. 비판자 prompt 의 "You need not agree with the author" 는 Liang 등 [[2](#ref-2)] 의 meta prompt 가 debater 에게 서로 완전히 동의할 필요가 없다고 한 것을 따른다.

```markdown
You are a critic of the model spec below. Your view: <CRITIC_VIEW>.
List every defect in the spec, one item per defect, in this format:
- element: Scope | Data Schema | Behavior | Acceptance Criteria
- type: missing | ambiguous | contradictory
- location: <HEADING_AND_LINE>
- reason: one sentence
Do not rewrite the spec. Do not assume facts the spec does not state.
You need not agree with the author; the goal is a spec with no defects.

<SPEC>
```

```markdown
You are the author of the model spec below.
For each defect in the critic's list, either revise the spec or rebut the defect in one sentence.
If a revision needs a fact the requirement does not give, write [NEEDS CLARIFICATION: <QUESTION>] instead of guessing.
Return the full revised spec, then one line per defect: defect, action (revised | rebutted | clarification), reason.

<REQUIREMENT>
<SPEC>
<DEFECT_LIST>
```

```markdown
You are the judge of a spec debate.
For each defect the author rebutted, decide "defect stands" or "rebuttal stands", with one sentence of reason.
Answer CONTINUE if any defect stands or was not handled; otherwise answer STOP.
If this is round <MAX_ROUND>, answer STOP and write each defect that still stands as [NEEDS CLARIFICATION: <QUESTION>].

<SPEC>
<DEFECT_LIST>
<AUTHOR_RESPONSE>
```

### 5.4 Human-Settled Defects

Agent 가 확정하는 결함은 요구사항과 spec 의 문장끼리 견주어 가릴 수 있는 결함이고, 사람이 확정하는 결함은 요구사항에 없는 사실이 필요한 결함과 round 상한까지 판정되지 않은 결함이다. `[NEEDS CLARIFICATION]` 표시는 SDD 와 겹치며, 그 표기는 [Spec-Driven Development (SDD) 3.3 Requirement Notation](../SDD/spec-driven-development-ko.md#33-requirement-notation) 에 있다.

Table 5. Who settles a defect

<table>
  <thead>
    <tr>
      <th align="center">Settled by</th>
      <th align="center">Defect</th>
      <th align="center">Example</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Agent</td>
      <td align="center">요구사항 안에서 근거를 찾을 수 있는 결함</td>
      <td align="center">두 요구사항의 충돌, 요구사항 번호와 test 의 짝 누락</td>
    </tr>
    <tr>
      <td align="center" rowspan="2">Human</td>
      <td align="center">요구사항에 없는 사실이 필요한 결함</td>
      <td align="center">metric 의 문턱값, 학습 data 의 기간</td>
    </tr>
    <tr>
      <td align="center">round 상한까지 판정되지 않은 결함</td>
      <td align="center">judge 가 마지막 round 에 남긴 <code>[NEEDS CLARIFICATION]</code></td>
    </tr>
  </tbody>
</table>

요구사항에 없는 사실을 agent 끼리 합의로 채우지 않는다. Spec Kit [[6](#ref-6)] 은 정하지 않은 것을 추측하지 말고 `[NEEDS CLARIFICATION: <QUESTION>]` 으로 표시하라고 하며, 이 표시가 LLM 이 그럴듯하지만 틀릴 수 있는 가정을 세우는 것을 막는다고 적는다. 같은 문서는 spec 을 완성으로 보기 전에 `[NEEDS CLARIFICATION]` 이 하나도 남지 않아야 한다고 정한다.

## 6. Application

MAD 는 spec 의 결함을 찾는 데 쓸 수 있지만 단일 agent 방법보다 늘 낫지는 않으므로, 같은 spec 을 한 agent 의 자기 검토로 다듬은 결과와 견준 뒤에 쓴다. Smit 등 [[5](#ref-5)] 은 현재 형태의 MAD 가 self-consistency 나 여러 추론 경로의 ensemble 보다 늘 낫지는 않았고, hyperparameter 에 더 민감해 최적화가 어렵다고 보고했다.

Table 6. Failure cases of a spec debate

| Failure                       | Symptom                                         | Countermeasure                                        | Source        |
| :---------------------------: | :---------------------------------------------: | :---------------------------------------------------: | :-----------: |
| Same role prompt              | 비판자들이 같은 결함만 되풀이해 찾음            | 비판자마다 다른 관점의 역할 prompt                    | [[4](#ref-4)] |
| Full agreement                | 작성자의 첫 판을 비판 없이 받아들임             | 완전한 합의를 요구하지 않는 meta prompt               | [[2](#ref-2)] |
| Full disagreement             | 고친 문장까지 다시 반박해 debate 가 끝나지 않음 | 모든 점에서 반대하라고 요구하지 않는 meta prompt      | [[2](#ref-2)] |
| Mixed LLM backbones           | judge 가 자기와 같은 LLM 쪽을 편듦              | 작성자, 비판자, judge 를 같은 LLM 으로 둠             | [[2](#ref-2)] |
| Debate past the best round    | 고친 spec 이 이전 round 보다 나빠짐             | judge 의 adaptive break, round 상한                   | [[2](#ref-2)] |
| Defects outside the checklist | Table 4 에 없는 결함이 그대로 남음              | 사람이 확정한 결함을 다음 debate 의 checklist 에 더함 | -             |

비용은 호출 수로 정해진다. 한 round 에 agent 마다 한 번 호출하므로 호출 수는 agent 수와 round 수를 곱한 값이며, Du 등 [[1](#ref-1)] 도 계산 비용 때문에 agent 3 개와 round 2 회를 주로 썼다.

## References

<a id="ref-1"></a>
[1] Du, Y., Li, S., Torralba, A., Tenenbaum, J. B., & Mordatch, I. (2023). [Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://doi.org/10.48550/arXiv.2305.14325). *arXiv preprint* arXiv:2305.14325.<br>
<a id="ref-2"></a>
[2] Liang, T., He, Z., Jiao, W., Wang, X., Wang, Y., Wang, R., Yang, Y., Shi, S., & Tu, Z. (2024). [Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate](https://doi.org/10.48550/arXiv.2305.19118). *Proceedings of EMNLP 2024*.<br>
<a id="ref-3"></a>
[3] Jin, D., Jin, Z., Chen, X., & Wang, C. (2024). [MARE: Multi-Agents Collaboration Framework for Requirements Engineering](https://doi.org/10.48550/arXiv.2405.03256). *arXiv preprint* arXiv:2405.03256.<br>
<a id="ref-4"></a>
[4] Chan, C.-M., Chen, W., Su, Y., Yu, J., Xue, W., Zhang, S., Fu, J., & Liu, Z. (2023). [ChatEval: Towards Better LLM-based Evaluators through Multi-Agent Debate](https://doi.org/10.48550/arXiv.2308.07201). *arXiv preprint* arXiv:2308.07201.<br>
<a id="ref-5"></a>
[5] Smit, A., Duckworth, P., Grinsztajn, N., Barrett, T. D., & Pretorius, A. (2023). [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://doi.org/10.48550/arXiv.2311.17371). *arXiv preprint* arXiv:2311.17371.<br>
<a id="ref-6"></a>
[6] GitHub. (n.d.). [Specification-Driven Development (SDD)](https://github.com/github/spec-kit/blob/main/spec-driven.md). *Spec Kit repository*.

---

## Appendix A. Terminology

- **Acceptance criteria**: spec 의 네 요소 가운데 하나로, 통과 여부를 가르는 test 와 명령.
- **Adaptive break**: judge 가 최선의 답에 이른 round 에서 debate 를 끝내는 종료 방식.
- **Author**: spec 을 쓰고 비판자의 결함마다 고치거나 반박하는 agent.
- **Behavior**: spec 의 네 요소 가운데 하나로, 번호를 붙인 MUST 요구사항과 경계 조건에서의 동작.
- **Critic**: spec 을 고쳐 쓰지 않고 결함 목록만 돌려주는 agent.
- **Data schema**: spec 의 네 요소 가운데 하나로, 입력과 출력의 열, type, 단위.
- **Degeneration-of-Thought (DoT)**: LLM 이 자기 답에 확신을 가진 뒤에는 처음 답이 틀렸어도 self-reflection 으로 새 생각을 하지 못하는 현상.
- **Judge**: 반박된 결함을 판정하고 debate 의 계속과 종료를 정하는 agent.
- **Multi-Agent Debate (MAD)**: 여러 LLM agent 가 prompt 로 서로의 답을 받아 여러 round 에 걸쳐 고쳐 쓰게 하는 방법. Liang 등은 자기 framework 의 이름으로도 MAD 를 쓴다.
- **NEEDS CLARIFICATION**: spec 에서 정하지 못한 것을 추측으로 채우지 않고 사람에게 묻도록 남기는 표시.
- **Round**: debate 에 참여한 agent 가 차례대로 또는 동시에 한 번씩 답하는 단위.
- **Scope**: spec 의 네 요소 가운데 하나로, 손댈 파일, 쓸 library 와 그 version.
- **Self-consistency**: 한 LLM 이 여러 추론 경로로 답을 여러 번 쓰고 가장 많이 나온 답을 고르는 방법.
- **Self-reflection**: 한 LLM 이 자기 답에 스스로 feedback 을 주어 고치는 방법.
- **Spec**: coding agent 에 model build 를 맡길 때 주는 markdown 형식의 prompt.
- **Tit for tat**: debater 가 상대의 주장에 반박하는 정도. 이 문서는 반박 강도로 적는다.
