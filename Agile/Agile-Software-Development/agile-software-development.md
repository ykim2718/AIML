# Agile Software Development
Rev. 4 | Created: 2026-09-13 | Updated: 2026-09-13 13:07 CDT

## 1. Purpose

- **Problem Statement**: Agile, DevOps and CI/CD are named side by side as if a team picked one of them, so what is actually being adopted is never settled.
- **Goal**: Separate the three by the question each one answers and fix the vocabulary the cycle is run with, so that a team can state its own bar for done, its own release unit, its own improvement loop, and the standing of any piece of work in one word.
- **Non-Goal**: Configuring a particular tool (Jira, Jenkins, GitHub Actions) is not covered.

## 2. Summary

The three names sit on three layers of one system, and the outer layer contains the inner one. Agile is the development philosophy, DevOps is the culture that carries that philosophy across development and operation, and CI/CD is the automation that implements both. A team does not choose among them; it decides how far down that stack it has gone.

Agile itself is short to state and hard to check. It builds a product in short cycles instead of one long plan, takes feedback at the end of every cycle, and improves the process from what that feedback showed. The rest of this document is the machinery that makes the statement checkable — a bar that declares a task done, a release that can be withdrawn, a retrospective whose output changes the next cycle, and the vocabulary a team says all of that in.

## 3. Taxonomy and its Hierarchy

The three layers are told apart by the question each one answers, and they nest inside one another. A step down narrows what is being decided and makes it concrete; a step up adds what the layer below leaves out. Agile answers how the team works and collaborates, DevOps answers how development and operation stop being two separate organizations, and CI/CD answers which part of that is carried out by machines.

The three layers, the question each one answers, and what each one adds to the layer below are drawn in [Fig 1](#fig-1).

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
| Agile | Development philosophy and way of running the work | Fast feedback and flexibility toward change (way of working) |
| DevOps | Organizational culture and mode of collaboration | Integration and automation of development (Dev) and operation (Ops) (culture) |
| CI/CD | Technical practice and automation tooling | Automation of build, test and deployment (technical practice) |

Agile is a framework for the question "how will we work and collaborate", and it cuts requirements into small pieces that are developed, improved and deployed in short cycles called sprints. DevOps is a culture for the question "how do we remove the boundary between development (Dev) and operation (Ops) and deliver value without delay". CI/CD is the pipeline that verifies code, builds it, and carries it safely into the service environment, which is what makes the other two visible in practice.

CI/CD sits at the bottom of that containment because it is one technical component of the larger system rather than the system itself.

### 3.1 Values

The philosophy at the top of the hierarchy is fixed by the four values published in the Manifesto for Agile Software Development in 2001. Each value states a preference between two things that are both real, not a rejection of the item on the right.

Table 2. The four values of the Agile Manifesto

| Valued | Over |
| --- | --- |
| Individuals and interactions | Processes and tools |
| Working software | Comprehensive documentation |
| Customer collaboration | Contract negotiation |
| Responding to change | Following a plan |

### 3.2 Placement

Against the waterfall model the difference is not the set of technical steps but where the plan is allowed to change and when feedback arrives. Waterfall passes each stage once in order, so the feedback lands at the end; Agile repeats the whole path in short units, so the feedback lands every unit.

Table 3. Waterfall and Agile compared

| Aspect | Waterfall | Agile |
| --- | --- | --- |
| Philosophy | "A complete plan is made at the start" | "A plan is always open to revision" |
| Flow | Plan, design, develop, test, deploy, in order and once | [plan-develop-test-deploy] repeated as an iteration in units of 2 to 4 weeks |
| Feedback point | Late in the project, just before or after deployment | Continuous feedback at every iteration (sprint) |
| Strength | Predictable, and systematic to manage | Very fast to follow a change in market or customer |

## 4. Iteration Cycle

The technical steps of a sprint are the same as anywhere else — commit, merge, build, deploy. What changes is the way the work is run, the definition of done, and the deployment interval. Instead of months of coding closed by a single deployment, the whole path from planning through retrospective repeats every sprint, normally one to four weeks.

The stages of that cycle and the terms used at each of them are drawn in [Fig 2](#fig-2).

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

Three terms carry most of the difference between an agile cycle and a plan-driven one, because each of them moves a decision that would otherwise be made once at the end of the project.

### 5.1 Definition Of Done

Work is done when it passes the Definition of Done (DoD), not when the code is written. Ordinary development often calls "the code is finished" done; an agile team fixes a bar in advance and calls a task done only once it clears that bar. A DoD may require code written, unit tests passed, code review closed, documentation updated, and deployment to the staging server completed — all five, for one task to be Done.

### 5.2 Deployment And Release

Deployment and release are separated as two technical events. Under continuous deployment (CD) verified code is deployed to the server automatically, dozens of times a day if that is what the merges produce. A feature flag or canary release then keeps the deployment from being an exposure: the code is deployed, but the switch opens the feature to 5 % of users first, and the audience is widened once the response is read.

### 5.3 Retrospective And BKM

The sprint retrospective is the part of the process that changes the process. The team asks what went wrong procedurally in this sprint, and the answer is written straight into the team's Best Known Method (BKM) document or development rules, so that the next sprint runs under the revised rule rather than under a note that was never applied.

## 6. Team Vocabulary

The process terms carry the same meaning inside a team as outside it, deployment, merge and release included. What a team adds on top of them is a short vocabulary of its own, used to share what it has learned, to decide who meets a change first, and to say where a piece of work stands. Those three purposes give the three groups below, and every term in them appears at one of the stages of [Fig 2](#fig-2).

### 6.1 Knowledge And Standard

The first group holds what the team already knows, so that one problem is not solved twice. Section 5.3 covered the loop that keeps the first of them current; here the three are fixed side by side.

Table 4. Terms that hold what the team has learned

| Term | What it names | Where it is used |
| --- | --- | --- |
| BKM (Best Known Method) | The best and most efficient method known so far for a task | "Handle this issue by the BKM document" |
| SOP (Standard Operating Procedure) | The official manual for repeated work | Server checks, incident response |
| Post-mortem | The review after a project closes or a large outage, fixing cause and prevention | The document or the meeting that follows an outage |

Depending on the culture and the systems a team works in, the BKM document is also called a playbook or a runbook. The name changes with the house; what it holds does not.

### 6.2 Feature Control

The second group decides who meets a change and when. Section 5.2 pointed the feature flag at a fraction of customers, and the same switch, with the two terms beside it, also points at the team's own members and at the repair that cannot wait for the schedule.

Table 5. Terms that control who meets a change

| Term | What it names | Where it is used |
| --- | --- | --- |
| Feature flag | The switch that turns a deployed feature on or off for a chosen user or staff member | Deployment finished, exposure withheld |
| Dogfooding | Staff using the product before any customer does, to find the bugs first | "Dogfood this internally before the release" |
| Hotfix | The urgent deployment that repairs a serious bug in the operating environment | Outside the regular deployment schedule |

### 6.3 Work State

The third group states where a piece of work stands, so that one status is read the same way by everyone.

Table 6. Terms that state where work stands

| Term | What it names | Where it is used |
| --- | --- | --- |
| WIP (Work In Progress) | Work currently under way | `[WIP]` on a PR or a card, meaning not ready for review |
| Blocker | The technical or administrative obstacle stopping the next step | "Blocked on the DB permission problem" |
| POC (Proof of Concept) | The trial that checks whether an idea can be built at all | Before a new technology is adopted |

One sentence of ordinary team traffic carries most of this vocabulary at once, which is the reason to fix each term once rather than explain it every time.

> This new feature is past its POC and merged into the main branch. Turn the feature flag on so the team can dogfood it first, and if the deployment goes wrong, follow the BKM in the wiki and handle it as a hotfix.

---

## Appendix A. Terminology

- **Agile**: an adjective meaning nimble in ordinary use, taken in IT and business as a proper noun for the development philosophy described in section 3.
- **BKM (Best Known Method)**: the team document holding the best method known so far for a task, updated from retrospectives.
- **Blocker**: a technical or administrative obstacle that stops work from moving to the next step.
- **Canary release**: opening a deployed feature to a small fraction of users before the whole audience.
- **CD (Continuous Deployment)**: automatic deployment of verified code to the service environment.
- **CI (Continuous Integration)**: automatic build and test triggered whenever code is merged.
- **DevOps**: the culture and practice that unite development (Dev) and operation (Ops).
- **DoD (Definition of Done)**: the explicit bar a team agrees on, which a task must clear to be called done.
- **Dogfooding**: staff using the product themselves before it reaches a customer, to find bugs first.
- **Feature flag**: a switch that turns a deployed feature on or off without a new deployment.
- **Hotfix**: an urgent deployment, outside the regular schedule, that repairs a serious bug in the operating environment.
- **Increment**: the working product produced by one sprint.
- **Playbook**: another name for the BKM document, used where the team's culture prefers it.
- **PO/PM (Product Owner / Product Manager)**: the role that owns the product plan and the order of the backlog.
- **POC (Proof of Concept)**: a trial project that checks whether an idea or a technology can be built at all.
- **Post-mortem**: the review held after a project closes or a large outage, which fixes the cause and the prevention.
- **Product backlog**: the full ordered list of requirements and features for the product.
- **Runbook**: another name for the BKM document, used where the team's culture prefers it.
- **Scrum**: the daily standup meeting, and by extension the framework it belongs to.
- **SOP (Standard Operating Procedure)**: the official manual for repeated work such as server checks and incident response.
- **Sprint**: one iteration of the agile cycle, normally one to four weeks.
- **Sprint backlog**: the subset of the product backlog a team commits to finish in one sprint.
- **Sprint retrospective**: the meeting at the end of a sprint that reviews the process and fixes what to change.
- **Sprint review**: the demonstration of the increment to the stakeholders.
- **Staging server**: the environment that mirrors production, used to verify a build before release.
- **User story**: a unit of work written from the user's viewpoint.
- **Waterfall**: the model that passes plan, design, development, test and deployment once each, in order.
- **WIP (Work In Progress)**: work currently under way, marked so that it is not taken as ready for review.
