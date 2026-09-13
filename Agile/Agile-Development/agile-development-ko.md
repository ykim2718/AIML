# Agile Development
Rev. 0 | Created: 2026-09-13 | Updated: 2026-09-13 12:20 CDT

## 1. Purpose

- **Problem Statement**: Agile, DevOps, CI/CD 가 같은 층위의 선택지처럼 나란히 불려, 무엇을 도입하는 것인지가 끝내 합의되지 않는다.
- **Goal**: 세 낱말을 각자가 답하는 질문으로 갈라, 팀이 스스로 완료의 기준과 release 단위와 개선 loop 를 정할 수 있게 한다.
- **Non-Goal**: 특정 도구 (Jira, Jenkins, GitHub Actions) 의 설정 방법은 다루지 않는다.

## 2. Summary

세 낱말은 한 체계의 세 층에 놓이며, 바깥 층이 안쪽 층을 담는다. Agile 은 개발 철학이고, DevOps 는 그 철학을 개발과 운영에 걸쳐 실어 나르는 문화이며, CI/CD 는 그 둘을 구현하는 자동화다. 팀은 셋 중 하나를 고르는 것이 아니라, 그 층을 어디까지 내려왔는지를 정할 뿐이다.

Agile 자체는 말하기는 짧고 확인하기는 어렵다. 거대한 계획 하나 대신 짧은 주기로 제품을 만들고, 주기마다 끝에서 피드백을 받고, 그 피드백이 드러낸 것으로 process 를 고친다. 이 문서의 나머지는 그 문장을 확인할 수 있게 만드는 장치다. 작업을 완료라고 선언하는 기준, 되돌릴 수 있는 release, 그리고 그 결과가 다음 주기를 바꾸는 회고가 그것이다.

## 3. Taxonomy

세 층은 각자가 답하는 질문으로 갈린다. Agile 은 팀이 어떻게 일하고 협업할 것인가에 답하고, DevOps 는 개발과 운영이 어떻게 두 개의 조직이기를 그만두는가에 답하며, CI/CD 는 그 가운데 무엇을 기계가 수행하는가에 답한다.

Table 1. The three layers and what each one decides

| Concept | Role | Perspective |
| --- | --- | --- |
| Agile | 개발 철학 및 진행 방식 | 빠른 피드백과 변화 유연성 (way of working) |
| DevOps | 조직 문화 및 협업 방식 | 개발 (Dev) 과 운영 (Ops) 의 통합 및 자동화 (culture) |
| CI/CD | 기술적 실천 및 자동화 도구 | build, test, 배포 절차의 자동화 (technical practice) |

Agile 은 "어떻게 일하고 협업할 것인가" 에 대한 framework 이며, 요구사항을 작게 쪼개어 sprint 라 부르는 짧은 주기 단위로 개발하고 개선하고 배포한다. DevOps 는 "개발 (Dev) 과 운영 (Ops) 의 경계를 없애고 어떻게 지체 없이 가치를 전달할 것인가" 에 대한 문화다. CI/CD 는 code 를 검증하고 build 하여 service 환경까지 안전하게 옮기는 pipeline 이며, 앞의 둘을 현장에서 눈에 보이게 만드는 것이 그것이다.

세 층의 포함 관계와 각 층이 아래 층에 없는 무엇을 더 가지는지는 [Fig 1](#fig-1) 에 그렸다.

```text
Agile  (philosophy: how to work and collaborate)
  |
  +-- Sprint planning, backlog management, customer feedback, product planning (PO/PM)
  |
  +-- DevOps  (culture: remove the Dev/Ops boundary)
        |
        +-- Monitoring, organizational culture, feedback system, Dev-Ops communication
        |
        +-- CI/CD  (technical practice: automate the delivery path)
              |
              +-- Build, test and deployment pipeline

Agile (superset)  >  DevOps (superset)  >  CI/CD (subset)
```

<a id="fig-1"></a>
Fig 1. Containment of Agile, DevOps and CI/CD

## 4. Hierarchy

층을 한 단계 내려갈 때마다 정하는 대상이 좁아지고 구체적이 되며, 한 단계 올라갈 때마다 아래 층이 빠뜨린 것이 더해진다. 각 단계가 무엇을 더하는지는 아래에 적었고, 그렇게 생기는 포함 관계는 [Fig 1](#fig-1) 에 그렸다.

- DevOps 는 CI/CD 의 상위 집합 — pipeline 외에 monitoring, 조직 문화, 피드백 체계, 개발팀과 운영팀의 소통 방식을 담는다.
- Agile 은 DevOps 의 상위 집합 — 그 기술·운영 요소 외에 sprint 계획, backlog 관리, 고객 피드백 수용, 제품 기획 (PO/PM) 을 담는다.

CI/CD 가 그 포함 관계의 맨 아래에 놓이는 까닭은 그것이 거대한 체계의 기술적 구성 요소 하나이지 체계 자체가 아니기 때문이다.

### 4.1 Values

계층 맨 위의 철학은 2001 년 발표된 애자일 소프트웨어 개발 선언 (Manifesto for Agile Software Development) 의 네 가지 가치로 고정되어 있다. 각 가치는 둘 다 실재하는 것 사이의 우선순위를 말하는 것이지, 오른쪽 항목을 버린다는 뜻이 아니다.

Table 2. The four values of the Agile Manifesto

| Valued | Over |
| --- | --- |
| 개인과 상호작용 | 공정과 도구 |
| 작동하는 소프트웨어 | 포괄적인 문서 |
| 고객과의 협력 | 계약 협상 |
| 변화에 대응하기 | 계획을 따르기 |

### 4.2 Placement

폭포수 (waterfall) 모델과 갈리는 지점은 기술적 단계의 목록이 아니라, 계획을 언제 바꿀 수 있고 피드백이 언제 도착하는가이다. Waterfall 은 각 단계를 순서대로 한 번씩 지나므로 피드백이 끝에 닿고, Agile 은 그 전 과정을 짧은 단위로 되풀이하므로 피드백이 단위마다 닿는다.

Table 3. Waterfall and Agile compared

| Aspect | Waterfall | Agile |
| --- | --- | --- |
| 기본 철학 | "처음에 완벽한 계획을 세운다" | "계획은 항상 수정될 수 있다" |
| 진행 방식 | 기획, 디자인, 개발, 테스트, 배포를 순차적으로 한 번씩 | [기획-개발-테스트-배포] 를 2~4주 단위로 반복 (iteration) |
| 피드백 시점 | 프로젝트 최후반, 배포 직전 또는 직후 | 매 반복 주기 (sprint) 마다 지속적 피드백 |
| 장점 | 예측 가능성이 높고 관리가 체계적 | 시장과 고객의 변화에 매우 신속하게 대응 |

## 5. Iteration Cycle

Sprint 안의 기술적 단계는 다른 곳과 같다. Commit, merge, build, 배포가 그것이다. 달라지는 것은 업무를 진행하는 방식, 완료의 정의, 그리고 배포 주기다. 몇 달 동안 coding 만 하고 마지막에 한 번 배포하는 대신, 기획부터 회고까지의 전 과정이 sprint 마다 되풀이되며 그 길이는 보통 1~4주다.

그 주기의 단계와 각 단계에서 쓰이는 용어는 [Fig 2](#fig-2) 에 그렸다.

```text
[ 1. Sprint Planning ]
        |
        +--> Product Backlog ......... Full list of requirements and features to build
        +--> Sprint Backlog .......... Work to be finished in this sprint (1-2 weeks)
        |
        v
[ 2. Iterative Development & Daily Check ]
        |
        +--> Daily Standup / Scrum ... 15 minutes a day on progress and blockers
        +--> Ticket / User Story ..... Unit of work written from the user's viewpoint
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
        +--> Feature Flag / Canary ... New feature opened to a subset of users first
        +--> Continuous Deployment ... Verified code deployed daily or hourly
        |
        v
[ 5. Retrospective ]
        |
        +--> Sprint Retrospective .... What went well, what did not, what to improve
        +--> BKM / Playbook Update ... Improvements written into the team standard
```

<a id="fig-2"></a>
Fig 2. Sprint cycle and the terms used at each stage

## 6. Completion And Release

Agile 주기와 계획 주도 주기의 차이는 대부분 아래 세 용어가 지고 있다. 셋 모두 프로젝트 끝에 한 번 내려질 결정을 앞으로 당겨 놓기 때문이다.

### 6.1 Definition Of Done

작업은 code 를 다 썼을 때가 아니라 DoD (Definition of Done) 를 통과했을 때 완료다. 일반적인 개발에서는 "code 를 다 짰다" 를 완료라 부르기도 하지만, agile 팀은 기준을 미리 정해 두고 그 기준을 넘긴 작업만 완료라 부른다. DoD 는 code 작성, 단위 test 통과, code review 완료, 문서화, staging server 배포 완료를 요구할 수 있으며, 한 작업이 Done 이 되려면 그 다섯이 모두 충족되어야 한다.

### 6.2 Deployment And Release

배포와 release 는 두 개의 기술적 사건으로 분리된다. CD (지속적 배포) 아래에서 검증된 code 는 server 에 자동으로 배포되며, merge 가 그만큼 나온다면 하루에도 수십 번 배포된다. Feature flag 또는 canary release 는 그 배포가 곧 공개가 되지 않게 막는다. Code 는 배포되지만 switch 는 기능을 먼저 5 % 의 사용자에게만 열고, 반응을 읽은 뒤에 대상을 넓힌다.

### 6.3 Retrospective And BKM

Sprint 회고는 process 가 process 를 바꾸는 자리다. 팀은 이번 sprint 에서 process 상 무엇이 문제였는지를 묻고, 그 답을 팀의 BKM (Best Known Method) 문서나 개발 규칙에 곧바로 적는다. 그래서 다음 sprint 는 적용된 적 없는 memo 가 아니라 고쳐진 규칙 아래에서 돌아간다.

---

## Appendix A. Terminology

- **Agile**: 일상에서는 민첩함을 뜻하는 형용사이나, IT 와 비즈니스에서는 section 3 이 서술한 개발 철학을 가리키는 고유명사로 쓰인다.
- **BKM (Best Known Method)**: 어떤 작업에 대해 현재까지 알려진 최선의 방법을 담은 팀 문서. 회고에서 갱신된다.
- **Canary release**: 배포된 기능을 전체 사용자에 앞서 일부에게만 여는 것.
- **CD (Continuous Deployment)**: 검증된 code 를 service 환경에 자동으로 배포하는 것.
- **CI (Continuous Integration)**: Code 가 merge 될 때마다 자동으로 build 하고 test 하는 것.
- **DevOps**: 개발 (Dev) 과 운영 (Ops) 을 하나로 묶는 문화이자 실천.
- **DoD (Definition of Done)**: 팀이 합의한 명시적 기준. 작업이 완료로 불리려면 이것을 넘어야 한다.
- **Feature flag**: 배포된 기능을 새 배포 없이 켜고 끄는 switch.
- **Increment**: 한 sprint 가 만들어 낸 작동하는 산출물.
- **PO/PM (Product Owner / Product Manager)**: 제품 기획과 backlog 의 순서를 책임지는 역할.
- **Product backlog**: 제품에 필요한 요구사항과 기능의 전체 목록. 순서가 매겨져 있다.
- **Scrum**: 매일의 standup 회의, 나아가 그것이 속한 framework.
- **Sprint**: Agile 주기의 한 반복. 보통 1~4주.
- **Sprint backlog**: 한 sprint 안에 끝내기로 한 product backlog 의 부분 집합.
- **Sprint retrospective**: Sprint 끝에 process 를 되짚고 무엇을 고칠지 정하는 회의.
- **Sprint review**: 이해관계자에게 increment 를 시연하는 자리.
- **Staging server**: 운영 환경을 본뜬 환경. Release 전에 build 를 검증하는 데 쓴다.
- **User story**: 사용자의 관점에서 작성된 작업 단위.
- **Waterfall**: 기획, 디자인, 개발, 테스트, 배포를 순서대로 한 번씩 지나는 모델.
