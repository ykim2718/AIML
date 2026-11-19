---
name: rnd-management
description: R&D 과제의 단계와 관문 (gate) 을 설명하는 문서 (R&D management document) 를 쓰거나 고치기 전에 반드시 로드할 것. AI/ML model 개발 과제를 Plan, Develop, Review 세 단계로 나누고, 단계마다 무엇을 하고 무엇을 내놓는지 적는 문서의 목차를 정한다. Plan 은 PSM (목표 설정, 문제 정의, 해결 방향 합의), Develop 은 EVD (실험, 검증, model 선정) 와 Deployment, Review 는 결과 평가, 피드백, 개선 방향과 다음 계획 결정을 담는다. "R&D 관리 문서", "개발 workflow 설명해줘", "PSM EVD 정리해줘", "과제 단계 문서 써줘" 같은 요청에 해당한다. 서식은 상위 규칙인 md-rules 를, 구성 판단은 tech-doc-rules 를 따르고, mirror 문서는 md-mirror 를 함께 따른다.
---

# R&D Management Document
Version 0 | Created: 2026-11-19 | Updated: 2026-11-19 17:50 CST

이 skill 은 **R&D 과제의 단계를 설명하는 문서의 목차와 단계 구분** 을 정한다. 이 skill 은 `md-rules` 의 하위 skill 이며, 서식은 `md-rules` 가, 구성 판단은 `tech-doc-rules` 가, 두 판의 짝은 `md-mirror` 가 정한다. `tech-doc-rules` 의 골격과 목차가 어긋나면 이 skill 을 따른다.

## 1. Scope

- 대상은 AI/ML model 개발 과제의 단계, 관문 (gate), 단계별 산출물을 설명하는 문서다.
- 발표 자료의 한 page 를 설명하는 문서도 대상이며, 그 page 는 그림으로 문서에 싣는다 (section 4).
- 특정 과제의 실측값과 일정은 다루지 않는다. 그런 문서는 `tech-doc-rules` 의 project doc 골격을 따른다.

## 2. Stages

R&D 과제는 아래 세 단계를 차례로 거치고, Review 의 결정이 다음 Plan 의 입력이 된다.

Table 1. R&D management stages

| Stage   | Main content                                     |
| :-----: | :----------------------------------------------: |
| Plan    | 목표 설정, 문제 정의, 해결 방향 합의 (PSM)       |
| Develop | 실험, 검증, 모델 선정 (EVD), Deployment          |
| Review  | 결과 평가, 피드백, 개선 방향 및 다음 계획 결정   |

- 문서는 위 세 단계의 이름과 순서를 바꾸지 않는다. 단계를 더하거나 합치지 않는다.
- PSM 은 Problem Statement, Solutions, Model Candidates 의 머리글자이고, EVD 는 Evidence-Based Decision 의 줄임말이다. 두 약어는 문서에서 처음 쓸 때 풀어 적는다 (`md-rules` section 1).
- 단계 안의 작업 (예: PSM 의 Problem Statement) 은 원 자료가 쓰는 영어 이름을 그대로 쓴다.
- 단계마다 아래 세 가지를 같은 형식으로 적는다.
    - **Activity**: 그 단계에서 하는 작업
    - **Output**: 그 단계가 끝날 때 내놓는 산출물
    - **Gate**: 다음 단계로 넘어가기 위한 승인 조건
- 원 자료에 없는 Activity, Output, Gate 는 지어 적지 않는다. 비어 있으면 사용자에게 묻는다 (`tech-doc-rules` section 9).

## 3. Contents

꼭지의 차례는 아래이다. H2 제목은 영어로 적는다 (`md-rules` section 2).

```markdown
# <Subject>
Version <n> | Created: <YYYY-MM-DD> | Updated: <YYYY-MM-DD HH:MM TZ>

## 1. Purpose

**Problem Statement**: ...

**Goal**: ...

**Non-Goal**: ...

## 2. Summary
## 3. Workflow
## 4. R&D Management Cycle

---

## Appendix A. Terminology
```

Table 2. Chapter scope

| Chapter                | Contains                                                    | Excludes                       |
| :--------------------: | :---------------------------------------------------------: | :----------------------------: |
| Purpose                | Problem Statement, Goal, Non-Goal                           | 근거, 배경                     |
| Summary                | 세 단계와 그 산출물의 요약 (서술형 목록, 5 문장 이내)       | 단계별 작업의 상세             |
| Workflow               | 원 자료의 그림, 단계별 Activity, Output, Gate 의 표         | 원 자료에 없는 작업            |
| R&D Management Cycle   | Workflow 의 단계를 Plan, Develop, Review 에 대응시킨 표     | 특정 과제의 일정과 실측값      |
| Appendix A             | 본문에서 정의하지 않은 용어                                 | 본문에서 정의한 용어           |

- 원 자료가 없는 문서는 Workflow 꼭지를 빼고, 단계별 표를 R&D Management Cycle 꼭지에 둔다.
- References 는 원 자료가 공개 문헌일 때만 둔다 (`md-rules` section 14).

## 4. Source Page

- 발표 자료의 page 를 실을 때는 그 page 하나를 image 로 바꾸어 `<md file stem>_fig/` 에 둔다 (`md-rules` section 12).
- 파일 이름은 `page<N>.png` 또는 `page<N>.svg` 로 하고, SVG 와 PNG 가운데 작은 쪽 하나만 남긴다.
- 그림은 `<img src="<md file stem>_fig/page<N>.png" width="800" style="max-width: 100%;" alt="Fig 1">` 로 싣고, 아래에 `Fig 1. <Page title>` 을 둔다.
- 그림 아래에는 page 의 내용을 표로 옮겨, 그림을 열지 않고도 글자로 읽을 수 있게 한다.

## 5. Before Finishing

- [ ] 세 단계의 이름과 순서가 Table 1 과 같은가
- [ ] 단계마다 Activity, Output, Gate 가 같은 형식으로 있는가
- [ ] 원 자료에 없는 작업이나 값을 지어 적지 않았는가
- [ ] PSM 과 EVD 를 처음 쓸 때 풀어 적었는가
- [ ] `md-rules` 를 적용하고 `check-terms.py` 를 돌렸는가
