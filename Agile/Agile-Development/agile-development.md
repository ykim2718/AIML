# Agile Development
Rev. 0 | Created: 2026-09-13 | Updated: 2026-09-13 12:20 CDT

## 1. Purpose

- **Problem Statement**: Agile, DevOps and CI/CD are named side by side as if a team picked one of them, so what is actually being adopted is never settled.
- **Goal**: Separate the three by the question each one answers, so that a team can state its own bar for done, its own release unit, and its own improvement loop.
- **Non-Goal**: Configuring a particular tool (Jira, Jenkins, GitHub Actions) is not covered.

## 2. Summary

The three names sit on three layers of one system, and the outer layer contains the inner one. Agile is the development philosophy, DevOps is the culture that carries that philosophy across development and operation, and CI/CD is the automation that implements both. A team does not choose among them; it decides how far down that stack it has gone.

Agile itself is short to state and hard to check. It builds a product in short cycles instead of one long plan, takes feedback at the end of every cycle, and improves the process from what that feedback showed. The rest of this document is the machinery that makes the statement checkable — a bar that declares a task done, a release that can be withdrawn, and a retrospective whose output changes the next cycle.

## 3. Taxonomy

The three layers are told apart by the question each one answers. Agile answers how the team works and collaborates, DevOps answers how development and operation stop being two separate organizations, and CI/CD answers which part of that is carried out by machines.

Table 1. The three layers and what each one decides

| Concept | Role | Perspective |
| --- | --- | --- |
| Agile | Development philosophy and way of running the work | Fast feedback and flexibility toward change (way of working) |
| DevOps | Organizational culture and mode of collaboration | Integration and automation of development (Dev) and operation (Ops) (culture) |
| CI/CD | Technical practice and automation tooling | Automation of build, test and deployment (technical practice) |

Agile is a framework for the question "how will we work and collaborate", and it cuts requirements into small pieces that are developed, improved and deployed in short cycles called sprints. DevOps is a culture for the question "how do we remove the boundary between development (Dev) and operation (Ops) and deliver value without delay". CI/CD is the pipeline that verifies code, builds it, and carries it safely into the service environment, which is what makes the other two visible in practice.

The containment among the three layers, and what each layer holds that the layer below does not, are drawn in [Fig 1](#fig-1).

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

Each step down the layers narrows what is being decided and makes it concrete, and each step up adds what the layer below leaves out. What each step adds is named below, and the containment it produces is drawn in [Fig 1](#fig-1).

- DevOps as a superset of CI/CD — beyond the pipeline it holds monitoring, organizational culture, the feedback system, and the way development and operation teams communicate.
- Agile as a superset of DevOps — beyond those technical and operational elements it holds sprint planning, backlog management, customer feedback intake, and product planning (PO/PM).

CI/CD sits at the bottom of that containment because it is one technical component of the larger system rather than the system itself.

### 4.1 Values

The philosophy at the top of the hierarchy is fixed by the four values published in the Manifesto for Agile Software Development in 2001. Each value states a preference between two things that are both real, not a rejection of the item on the right.

Table 2. The four values of the Agile Manifesto

| Valued | Over |
| --- | --- |
| Individuals and interactions | Processes and tools |
| Working software | Comprehensive documentation |
| Customer collaboration | Contract negotiation |
| Responding to change | Following a plan |

### 4.2 Placement

Against the waterfall model the difference is not the set of technical steps but where the plan is allowed to change and when feedback arrives. Waterfall passes each stage once in order, so the feedback lands at the end; Agile repeats the whole path in short units, so the feedback lands every unit.

Table 3. Waterfall and Agile compared

| Aspect | Waterfall | Agile |
| --- | --- | --- |
| Philosophy | "A complete plan is made at the start" | "A plan is always open to revision" |
| Flow | Plan, design, develop, test, deploy, in order and once | [plan-develop-test-deploy] repeated as an iteration in units of 2 to 4 weeks |
| Feedback point | Late in the project, just before or after deployment | Continuous feedback at every iteration (sprint) |
| Strength | Predictable, and systematic to manage | Very fast to follow a change in market or customer |

## 5. Iteration Cycle

The technical steps of a sprint are the same as anywhere else — commit, merge, build, deploy. What changes is the way the work is run, the definition of done, and the deployment interval. Instead of months of coding closed by a single deployment, the whole path from planning through retrospective repeats every sprint, normally one to four weeks.

The stages of that cycle and the terms used at each of them are drawn in [Fig 2](#fig-2).

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

Three terms carry most of the difference between an agile cycle and a plan-driven one, because each of them moves a decision that would otherwise be made once at the end of the project.

### 6.1 Definition Of Done

Work is done when it passes the Definition of Done (DoD), not when the code is written. Ordinary development often calls "the code is finished" done; an agile team fixes a bar in advance and calls a task done only once it clears that bar. A DoD may require code written, unit tests passed, code review closed, documentation updated, and deployment to the staging server completed — all five, for one task to be Done.

### 6.2 Deployment And Release

Deployment and release are separated as two technical events. Under continuous deployment (CD) verified code is deployed to the server automatically, dozens of times a day if that is what the merges produce. A feature flag or canary release then keeps the deployment from being an exposure: the code is deployed, but the switch opens the feature to 5 % of users first, and the audience is widened once the response is read.

### 6.3 Retrospective And BKM

The sprint retrospective is the part of the process that changes the process. The team asks what went wrong procedurally in this sprint, and the answer is written straight into the team's Best Known Method (BKM) document or development rules, so that the next sprint runs under the revised rule rather than under a note that was never applied.

---

## Appendix A. Terminology

- **Agile**: an adjective meaning nimble in ordinary use, taken in IT and business as a proper noun for the development philosophy described in section 3.
- **BKM (Best Known Method)**: the team document holding the best method known so far for a task, updated from retrospectives.
- **Canary release**: opening a deployed feature to a small fraction of users before the whole audience.
- **CD (Continuous Deployment)**: automatic deployment of verified code to the service environment.
- **CI (Continuous Integration)**: automatic build and test triggered whenever code is merged.
- **DevOps**: the culture and practice that unite development (Dev) and operation (Ops).
- **DoD (Definition of Done)**: the explicit bar a team agrees on, which a task must clear to be called done.
- **Feature flag**: a switch that turns a deployed feature on or off without a new deployment.
- **Increment**: the working product produced by one sprint.
- **PO/PM (Product Owner / Product Manager)**: the role that owns the product plan and the order of the backlog.
- **Product backlog**: the full ordered list of requirements and features for the product.
- **Scrum**: the daily standup meeting, and by extension the framework it belongs to.
- **Sprint**: one iteration of the agile cycle, normally one to four weeks.
- **Sprint backlog**: the subset of the product backlog a team commits to finish in one sprint.
- **Sprint retrospective**: the meeting at the end of a sprint that reviews the process and fixes what to change.
- **Sprint review**: the demonstration of the increment to the stakeholders.
- **Staging server**: the environment that mirrors production, used to verify a build before release.
- **User story**: a unit of work written from the user's viewpoint.
- **Waterfall**: the model that passes plan, design, development, test and deployment once each, in order.
