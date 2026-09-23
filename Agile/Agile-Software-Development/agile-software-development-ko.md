# Agile Software Development
Rev. 7 | Created: 2026-09-13 | Updated: 2026-09-23 11:33 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Values](#31-values)
  - [3.2 Placement](#32-placement)
- [4. Iteration Cycle](#4-iteration-cycle)
- [5. Completion And Release](#5-completion-and-release)
  - [5.1 Definition Of Done](#51-definition-of-done)
  - [5.2 Deployment And Release](#52-deployment-and-release)
  - [5.3 Retrospective And BKM](#53-retrospective-and-bkm)
- [6. Team Vocabulary](#6-team-vocabulary)
  - [6.1 Knowledge And Standard](#61-knowledge-and-standard)
  - [6.2 Feature Control](#62-feature-control)
  - [6.3 Work State](#63-work-state)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

- **Problem Statement**: Agile, DevOps, CI/CD 가 같은 층위의 선택지처럼 나란히 불려, 무엇을 도입하는 것인지가 끝내 합의되지 않는다.
- **Goal**: 세 낱말을 각자가 답하는 질문으로 갈라 놓고 그 주기를 돌리는 용어를 고정하여, 팀이 스스로 완료의 기준과 release 단위와 개선 loop 를 정하고 어느 작업의 상태든 한 낱말로 말할 수 있게 한다.
- **Non-Goal**: 특정 도구 (Jira, Jenkins, GitHub Actions) 의 설정 방법은 다루지 않는다.

## 2. Summary

세 낱말은 한 체계의 세 층에 놓이며, 바깥 층이 안쪽 층을 담는다. Agile 은 개발 철학이고, DevOps 는 그 철학을 개발과 운영에 걸쳐 실어 나르는 문화이며, CI/CD 는 그 둘을 구현하는 자동화다. 팀은 셋 중 하나를 고르는 것이 아니라, 그 층을 어디까지 내려왔는지를 정할 뿐이다.

Agile 자체는 말하기는 짧고 확인하기는 어렵다. 거대한 계획 하나 대신 짧은 주기로 제품을 만들고, 주기마다 끝에서 피드백을 받고, 그 피드백이 드러낸 것으로 process 를 고친다. 이 문서의 나머지는 그 문장을 확인할 수 있게 만드는 장치다. 작업을 완료라고 선언하는 기준, 되돌릴 수 있는 release, 그 결과가 다음 주기를 바꾸는 회고, 그리고 팀이 그 모두를 말할 때 쓰는 용어가 그것이다.

## 3. Taxonomy and its Hierarchy

세 층은 각자가 답하는 질문으로 갈리고, 서로의 안에 포개진다. 한 단계 내려가면 정하는 대상이 좁아지고 구체적이 되며, 한 단계 올라가면 아래 층이 빠뜨린 것이 더해진다. Agile 은 팀이 어떻게 일하고 협업할 것인가에 답하고, DevOps 는 개발과 운영이 어떻게 두 개의 조직이기를 그만두는가에 답하며, CI/CD 는 그 가운데 무엇을 기계가 수행하는가에 답한다.

세 층과 각 층이 답하는 질문, 그리고 각 층이 아래 층에 더하는 것은 [Fig 1](#fig-1) 에 그렸다.

```text
Agile   (philosophy)   "How will we work and collaborate?"
  |
  |   adds   sprint planning, backlog management, customer feedback,
  |          product planning (PO/PM)
  |
  +-- DevOps   (culture)   "How do Dev and Ops stop being two organizations?"
        |
        |   adds   monitoring, organizational culture, feedback system,
        |          the way Dev and Ops teams communicate
        |
        +-- CI/CD   (practice)   "Which part of that runs by machine?"
              |
              |   holds   the build, test and deployment pipeline

Agile (superset)  >  DevOps (superset)  >  CI/CD (subset)
```

<a id="fig-1"></a>
Fig 1. The three layers, what each one answers, and what each one adds

Table 1. The three layers and what each one decides

| Concept | Role | Perspective |
| --- | --- | --- |
| Agile | 개발 철학 및 진행 방식 | 빠른 피드백과 변화 유연성 (way of working) |
| DevOps | 조직 문화 및 협업 방식 | 개발 (Dev) 과 운영 (Ops) 의 통합 및 자동화 (culture) |
| CI/CD | 기술적 실천 및 자동화 도구 | build, test, 배포 절차의 자동화 (technical practice) |

Agile 은 요구사항을 작게 쪼개어 sprint 라 부르는 짧은 cycle 단위로 개발하고 개선하고 배포한다. DevOps 는 두 조직 사이의 경계가 전달된 가치 앞에 놓는 지체를 걷어낸다. CI/CD 는 code 를 검증하고 build 하여 service 환경까지 안전하게 옮기며, 앞의 둘을 현장에서 눈에 보이게 만드는 것이 그것이다.

CI/CD 가 그 포함 관계의 맨 아래에 놓이는 까닭은 그것이 거대한 체계의 기술적 구성 요소 하나이지 체계 자체가 아니기 때문이다.

### 3.1 Values

계층 맨 위의 철학은 2001 년 발표된 애자일 소프트웨어 개발 선언 (Manifesto for Agile Software Development) 의 네 가지 가치로 고정되어 있다. 각 가치는 둘 다 실재하는 것 사이의 우선순위를 말하는 것이지, 오른쪽 항목을 버린다는 뜻이 아니다.

Table 2. The four values of the Agile Manifesto

| Valued | Over |
| --- | --- |
| 개인과 상호작용 | 공정과 도구 |
| 작동하는 소프트웨어 | 포괄적인 문서 |
| 고객과의 협력 | 계약 협상 |
| 변화에 대응하기 | 계획을 따르기 |

### 3.2 Placement

폭포수 (waterfall) 모델과 갈리는 지점은 계획을 언제 바꿀 수 있고 피드백이 언제 도착하는가에 있으며, 기술적 단계의 목록에 있지 않다. Waterfall 은 각 단계를 순서대로 한 번씩 지나므로 피드백이 끝에 닿고, Agile 은 그 전 과정을 짧은 단위로 되풀이하므로 피드백이 단위마다 닿는다.

Table 3. Waterfall and Agile compared

| Aspect | Waterfall | Agile |
| --- | --- | --- |
| 기본 철학 | "처음에 완벽한 계획을 세운다" | "계획은 항상 수정될 수 있다" |
| 진행 방식 | 기획, 디자인, 개발, 테스트, 배포를 순차적으로 한 번씩 | [기획-개발-테스트-배포] 를 2~4주 단위로 반복 (iteration) |
| 피드백 시점 | 프로젝트 최후반, 배포 직전 또는 직후 | 매 반복 cycle (sprint) 마다 지속적 피드백 |
| 장점 | 예측 가능성이 높고 관리가 체계적 | 시장과 고객의 변화에 매우 신속하게 대응 |

## 4. Iteration Cycle

Sprint 안의 기술적 단계는 다른 곳과 같다. Commit, merge, build, 배포가 그것이다. 달라지는 것은 업무를 진행하는 방식, 완료의 정의, 그리고 배포 주기다. 몇 달 동안 coding 만 하고 마지막에 한 번 배포하는 대신, 기획부터 회고까지의 전 과정이 sprint 마다 되풀이되며 그 길이는 보통 1~4주다.

그 주기의 단계와 각 단계에서 쓰이는 용어는 [Fig 2](#fig-2) 에 그렸다.

```text
[ 1. Sprint Planning ]
        |
        +--> Product Backlog ......... Full list of requirements and features to build
        +--> Sprint Backlog .......... Work to be finished in this sprint (1-2 weeks)
        +--> POC (Proof of Concept) .. Feasibility checked before the work is committed
        |
        v
[ 2. Iterative Development & Daily Check ]
        |
        +--> Daily Standup / Scrum ... 15 minutes a day on progress and blockers
        +--> Ticket / User Story ..... Unit of work written from the user's viewpoint
        +--> WIP (Work In Progress) .. Card or PR marked as not yet ready for review
        +--> Blocker ................. What stops the next step until it is cleared
        +--> Commit -> PR -> Merge ... Continuous coding and integration
        |
        v
[ 3. Continuous Integration (CI) ]
        |
        +--> CI/CD Pipeline .......... Automatic build and test on every merge
        +--> DoD (Definition of Done)  Explicit bar for "done" agreed by the team
        |
        v
[ 4. Internal Demo & Customer Feedback (CD) ]
        |
        +--> Sprint Review / Demo .... Increment shown to the stakeholders
        +--> Dogfooding .............. Staff using the build before any customer does
        +--> Feature Flag / Canary ... New feature opened to a subset of users first
        +--> Continuous Deployment ... Verified code deployed daily or hourly
        +--> SOP ..................... Fixed procedure an incident response follows
        +--> Hotfix .................. Urgent deployment outside the regular schedule
        |
        v
[ 5. Retrospective ]
        |
        +--> Sprint Retrospective .... What went well, what did not, what to improve
        +--> Post-mortem ............. Cause and prevention fixed after an outage
        +--> BKM / Playbook Update ... Improvements written into the team standard
```

<a id="fig-2"></a>
Fig 2. Sprint cycle and the terms used at each stage

## 5. Completion And Release

Agile 주기와 계획 주도 주기의 차이는 대부분 아래 세 결정이 지고 있다. 셋 모두 프로젝트 끝에 한 번 내려질 판단을 sprint 안쪽으로 당겨 놓기 때문이다.

### 5.1 Definition Of Done

작업은 code 를 다 썼을 때가 아니라 DoD (Definition of Done) 를 통과했을 때 완료다. 일반적인 개발에서는 "code 를 다 짰다" 를 완료라 부르기도 하지만, agile 팀은 기준을 미리 정해 두고 그 기준을 넘긴 작업만 완료라 부른다. DoD 는 code 작성, 단위 test 통과, code review 완료, 문서화, staging server 배포 완료를 요구할 수 있으며, 한 작업이 Done 이 되려면 그 다섯이 모두 충족되어야 한다.

### 5.2 Deployment And Release

배포와 release 는 두 개의 기술적 사건으로 분리된다. CD (지속적 배포) 아래에서 검증된 code 는 server 에 자동으로 배포되며, merge 가 그만큼 나온다면 하루에도 수십 번 배포된다. Feature flag 또는 canary release 는 그 배포가 곧 공개가 되지 않게 막는다. Code 는 배포되지만 switch 는 기능을 먼저 5 % 의 사용자에게만 열고, 반응을 읽은 뒤에 대상을 넓힌다.

### 5.3 Retrospective And BKM

Sprint 회고는 process 가 process 를 바꾸는 자리다. 팀은 이번 sprint 에서 process 상 무엇이 문제였는지를 묻고, 그 답을 팀의 BKM (Best Known Method) 문서나 개발 규칙에 곧바로 적는다. 그래서 다음 sprint 는 적용된 적 없는 memo 가 아니라 고쳐진 규칙 아래에서 돌아간다.

## 6. Team Vocabulary

개발 process 용어는 팀 안에서나 밖에서나 같은 뜻으로 쓰이며, 배포, merge, release 가 모두 그렇다. 팀이 그 위에 더하는 것은 짧은 자기 용어이다. 알아낸 것을 나누려고, 변경을 누가 먼저 만날지 정하려고, 작업이 어디에 서 있는지 말하려고 쓴다. 그 세 가지 목적이 아래 세 묶음을 만들고, 여기 든 모든 용어는 [Fig 2](#fig-2) 의 어느 단계에 등장한다.

### 6.1 Knowledge And Standard

첫 묶음은 팀이 이미 알고 있는 것을 담아, 한 문제를 두 번 풀지 않게 한다. BKM 을 갱신하는 loop 는 section 5.3 이 다루었고, 여기서는 셋을 나란히 놓아 고정한다.

Table 4. Terms that hold what the team has learned

| Term | What it names | Where it is used |
| --- | --- | --- |
| BKM (Best Known Method) | 특정 작업에 대해 현재까지 밝혀진 가장 좋고 효율적인 방법 | "이 이슈는 BKM 문서 참고해서 처리해 주세요" |
| SOP (Standard Operating Procedure) | 반복적인 작업을 위한 공식 매뉴얼 | Server 점검, 장애 대응 |
| Post-mortem | 프로젝트 종료 뒤나 큰 장애 뒤에 원인과 재발 방지책을 고정하는 부검 | 장애 뒤에 남기는 문서 또는 회고 미팅 |

팀의 문화와 쓰는 system 에 따라 BKM 문서를 playbook 이나 runbook 이라 부르기도 한다. 집마다 이름은 달라지지만 담기는 것은 같다.

### 6.2 Feature Control

둘째 묶음은 변경을 누가 언제 만나는지를 정한다. Section 5.2 는 feature flag 를 고객의 일부에게 겨누었고, 같은 switch 와 그 곁의 두 용어는 팀 자신과 일정을 기다릴 수 없는 수리에도 겨눈다.

Table 5. Terms that control who meets a change

| Term | What it names | Where it is used |
| --- | --- | --- |
| Feature flag | 배포된 기능을 특정 사용자나 내부 팀원에게만 켜고 끄는 switch | 배포는 끝났고 공개는 아직인 자리 |
| Dogfooding | 고객에게 공개하기 전에 내부 임직원이 먼저 써 보며 bug 를 찾는 과정 | "이번 release 전 내부 dogfooding 먼저 진행합니다" |
| Hotfix | 운영 환경에서 발생한 심각한 bug 를 고치는 긴급 배포 | 정기 배포 일정과 무관한 자리 |

### 6.3 Work State

셋째 묶음은 작업이 어디에 서 있는지를 말하여, 한 상태를 모두가 같은 뜻으로 읽게 한다.

Table 6. Terms that state where work stands

| Term | What it names | Where it is used |
| --- | --- | --- |
| WIP (Work In Progress) | 현재 진행 중인 작업 | PR 이나 업무 card 의 `[WIP]` 표시. 아직 검토할 때가 아니라는 뜻 |
| Blocker | 다음 단계로 나가지 못하게 막는 기술적·행정적 걸림돌 | "현재 DB 권한 문제로 작업이 Blocker 상태입니다" |
| POC (Proof of Concept) | 아이디어를 기술적으로 구현할 수 있는지 검증하는 시범 프로젝트 | 새 기술을 도입하기 전 |

팀의 평범한 소통 한 문장이 이 용어의 대부분을 한꺼번에 실어 나르며, 매번 설명하는 대신 낱말을 한 번 고정해 두는 이유가 여기에 있다.

> 이번 신규 기능은 POC 를 끝내고 main branch 에 merge 를 마쳤습니다. Feature flag 를 켜서 내부 팀원이 먼저 dogfooding 해 보고, 배포에서 문제가 생기면 wiki 에 적힌 BKM 을 참고해 hotfix 로 대응해 주세요.

---

## Appendix A. Terminology

- **Agile**: 일상에서는 민첩함을 뜻하는 형용사이나, IT 와 비즈니스에서는 section 3 이 서술한 개발 철학을 가리키는 고유명사로 쓰인다.
- **BKM (Best Known Method)**: 어떤 작업에 대해 현재까지 알려진 최선의 방법을 담은 팀 문서. 회고에서 갱신된다.
- **Blocker**: 다음 단계로 나가지 못하게 막는 기술적·행정적 걸림돌.
- **Canary release**: 배포된 기능을 전체 사용자에 앞서 일부에게만 여는 것.
- **CD (Continuous Deployment)**: 검증된 code 를 service 환경에 자동으로 배포하는 것.
- **CI (Continuous Integration)**: Code 가 merge 될 때마다 자동으로 build 하고 test 하는 것.
- **DevOps**: 개발 (Dev) 과 운영 (Ops) 을 하나로 묶는 문화이자 실천.
- **DoD (Definition of Done)**: 팀이 합의한 명시적 기준. 작업이 완료로 불리려면 이것을 넘어야 한다.
- **Dogfooding**: 고객에게 닿기 전에 임직원이 제품을 먼저 써 보며 bug 를 찾는 것.
- **Feature flag**: 배포된 기능을 새 배포 없이 켜고 끄는 switch.
- **Hotfix**: 정기 일정과 무관하게, 운영 환경의 심각한 bug 를 고치는 긴급 배포.
- **Increment**: 한 sprint 가 만들어 낸 작동하는 산출물.
- **Playbook**: BKM 문서를 부르는 다른 이름. 팀 문화에 따라 이렇게 쓴다.
- **PO/PM (Product Owner / Product Manager)**: 제품 기획과 backlog 의 순서를 책임지는 역할.
- **POC (Proof of Concept)**: 아이디어나 기술을 구현할 수 있는지 검증하는 시범 프로젝트.
- **Post-mortem**: 프로젝트 종료 뒤나 큰 장애 뒤에 열어 원인과 재발 방지책을 고정하는 회고.
- **Product backlog**: 제품에 필요한 요구사항과 기능의 전체 목록. 순서가 매겨져 있다.
- **Runbook**: BKM 문서를 부르는 다른 이름. 팀 문화에 따라 이렇게 쓴다.
- **Scrum**: 매일의 standup 회의, 나아가 그것이 속한 framework.
- **SOP (Standard Operating Procedure)**: server 점검, 장애 대응처럼 반복되는 작업을 위한 공식 매뉴얼.
- **Sprint**: Agile 주기의 한 반복. 보통 1~4주.
- **Sprint backlog**: 한 sprint 안에 끝내기로 한 product backlog 의 부분 집합.
- **Sprint retrospective**: Sprint 끝에 process 를 되짚고 무엇을 고칠지 정하는 회의.
- **Sprint review**: 이해관계자에게 increment 를 시연하는 자리.
- **Staging server**: 운영 환경을 본뜬 환경. Release 전에 build 를 검증하는 데 쓴다.
- **User story**: 사용자의 관점에서 작성된 작업 단위.
- **Waterfall**: 기획, 디자인, 개발, 테스트, 배포를 순서대로 한 번씩 지나는 모델.
- **WIP (Work In Progress)**: 현재 진행 중인 작업. 아직 검토 대상이 아님을 나타내려고 표시한다.
