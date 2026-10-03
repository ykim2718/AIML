# Model Developer Agent
Rev. 2 | Created: 2026-10-03 | Updated: 2026-10-03 17:46 CDT

- [1. Executive Summary](#1-executive-summary)
- [2. Proposed Solutions](#2-proposed-solutions)
- [3. Evaluation Criteria](#3-evaluation-criteria)
- [4. Technical Analysis and Comparison](#4-technical-analysis-and-comparison)
- [5. Risks and Constraints](#5-risks-and-constraints)
- [6. Cost and Resource Estimation](#6-cost-and-resource-estimation)
- [7. Recommendations and Next Steps](#7-recommendations-and-next-steps)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Executive Summary

A model is chosen for the data by hand today, so the algorithms tried are the ones the analyst remembers and nothing records what was tried. The conclusion is to proceed (Go) on a modular stack of PyCaret and MLflow: no single open source tool covers all four capabilities a model developer agent needs, and a managed platform hands the candidate list to the vendor.

- **Problem Statement**: A team that wants a model chosen for the data it has, rather than by hand, has no account of which capabilities that takes and which product holds each one.
- **Goal**: Split the work into the four capabilities that register, search, validate and freeze a model, score the candidate solutions against them, and recommend one, so that a manager or a designer can take the recommendation or the condition that overturns it.
- **Non-Goal**: The mathematics of each algorithm and the implementation Python code are not covered, and the price and the contract terms of a commercial platform are not compared.

## 2. Proposed Solutions

Three solutions were investigated, and they differ in how much of the candidate list and the glue code the engineers keep: a managed platform that sells all four capabilities as one product, a modular stack assembled one tool per capability, and a single library that runs all four inside one process. [Fig 1](#fig-1) names the four capabilities and places the three solutions against them.

```text
CAPABILITY            COMPONENT                      WHAT IT SETTLES                           SCOPE

Register          >   Model Registry                 Which models and versions exist           every run
Search            >   AutoML, model search           Which candidates this dataset gets        one dataset
Validate          >   Automated validation           Which candidate wins, on which metric     one candidate
Freeze            >   HPO, lineage tracking          Which hyperparameters and artifact stay   one winner

TIER                  EXAMPLE                        WHAT THE TIER ASSUMES

Managed platform  >   DataRobot, SageMaker, Vertex   The vendor's candidate list is enough
Modular stack     >   MLflow, Optuna, Kubeflow       The engineers run the glue between the four
Single library    >   PyCaret, auto-sklearn          One process is enough for all four
```

<a id="fig-1"></a>
Fig 1. The four capabilities of a model developer agent and the three solutions that hold them

Whichever solution is adopted, the four capabilities mesh into the existing system as one pipeline whose stages pass a narrowing set forward, and each stage is where a product is swapped. [Fig 2](#fig-2) draws it.

```text
[ Dataset ]
     |
     v
[ Feature engineering and data profiling ]  --> the dataset's own features
     |
     v
[ Model bank ]                              --> the candidate models the bank holds
     |
     v
[ Validation and HPO ]                      --> a score per candidate, a parameter set per score
     |
     v
[ Winner and its parameters ]               --> the lineage and the performance report
     |
     v
[ Registered version ]                      --> approved for production
```

<a id="fig-2"></a>
Fig 2. The pipeline the four capabilities run as, from a dataset to a registered version

The products behind the three solutions are Amazon SageMaker and Google Vertex AI for the managed platform [[1](#ref-1)] [[2](#ref-2)], MLflow with Optuna and Kubeflow Pipelines for the modular stack [[3](#ref-3)] [[4](#ref-4)] [[5](#ref-5)], and PyCaret, auto-sklearn or H2O-3 for the single library [[6](#ref-6)] [[7](#ref-7)] [[8](#ref-8)]. DataRobot and H2O Driverless AI sell the same managed shape, and this document carries no vendor page for either.

## 3. Evaluation Criteria

The criteria are the four capabilities plus the three properties that decide who owns what, and each one is read from a product's own documentation rather than from a benchmark of ours.

<a id="table-1"></a>
Table 1. The criteria and what each one is measured by

| #   | Criterion                | Measured by                                                                    | Passing                       |
| :-: | :----------------------: | :----------------------------------------------------------------------------: | :---------------------------: |
| 1   | Register                 | Whether a model version outlives the process that produced it                  | A store outside the run       |
| 2   | Search                   | Whether candidates are trained and ranked without being named one by one       | Several algorithms per run    |
| 3   | Validate                 | Whether the ranking comes from a validation split rather than the training fit | Cross-validation or a holdout |
| 4   | Freeze                   | Whether the winning hyperparameters and artifact are kept with their lineage   | Both, recorded together       |
| 5   | Pipeline integration     | Whether retraining and deployment run in the same system                       | One scheduler for all of it   |
| 6   | Candidate list ownership | Who decides which algorithms the data is tried against                         | The adopting team             |
| 7   | Operating surface        | How many components have to be operated                                        | As few as the work needs      |

Criteria 1 to 4 are the capabilities themselves, so a solution that fails one of them leaves that capability to Python code the engineers write. Criteria 5 to 7 separate the solutions that pass all four.

## 4. Technical Analysis and Comparison

No PoC was run for this investigation, so [Table 2](#table-2) carries what each product's own repository states and no latency or throughput of ours.

<a id="table-2"></a>
Table 2. The candidate solutions against the criteria of Table 1

| #   | Criterion                | SageMaker, Vertex AI              | MLflow, Optuna, Kubeflow              | PyCaret                                | auto-sklearn, H2O-3                  | Note                                                    |
| :-: | :----------------------: | :-------------------------------: | :-----------------------------------: | :------------------------------------: | :----------------------------------: | :-----------------------------------------------------: |
| 1   | Register                 | Each service's own Model Registry | MLflow Model Registry                 | Its control plane's registry           | Nothing                              | [[1](#ref-1)] [[2](#ref-2)] [[3](#ref-3)] [[6](#ref-6)] |
| 2   | Search                   | Autopilot, AutoML per data type   | An AutoML library, added              | About twelve algorithms per experiment | Algorithm selection over the library | [[1](#ref-1)] [[2](#ref-2)] [[6](#ref-6)] [[7](#ref-7)] |
| 3   | Validate                 | The service's candidate metrics   | A pipeline step the engineers write   | Its leaderboard                        | Its leaderboard or score             | [[2](#ref-2)] [[6](#ref-6)] [[8](#ref-8)]               |
| 4   | Freeze                   | A registered model version        | Optuna, recorded in MLflow            | The returned estimator, promoted       | The returned estimator               | [[3](#ref-3)] [[4](#ref-4)] [[6](#ref-6)]               |
| 5   | Pipeline integration     | The service's own pipelines       | Kubeflow Pipelines on Kubernetes      | Deployments from its control plane     | Nothing                              | [[5](#ref-5)] [[6](#ref-6)]                             |
| 6   | Candidate list ownership | The vendor                        | The adopting team                     | The adopting team                      | The adopting team                    | Read from each product's scope                          |
| 7   | Operating surface        | The cloud account only            | Tracking server, Kubernetes, the glue | One process and one control plane      | One process                          | Count of components to operate                          |

PyCaret reaches criteria 1 to 5 on its own, which no other open source entry in the table does [[6](#ref-6)], and its registry is the one piece a team may prefer to keep in MLflow so that the record outlives the PyCaret version in use [[3](#ref-3)]. The managed column passes every capability but fails criterion 6 by definition, and the auto-sklearn and H2O-3 column fails criteria 1 and 5 because neither tool keeps anything once the process ends [[7](#ref-7)] [[8](#ref-8)].

## 5. Risks and Constraints

Each risk below is paired with what reduces it, and the one risk without a mitigation is stated as accepted.

- **A library's winner lives only as long as its process** (technical): auto-sklearn and H2O-3 return an estimator and keep no registry, so a winner not exported before the process ends is lost [[7](#ref-7)] [[8](#ref-8)]. Mitigation: log the run and the artifact to MLflow from inside the script [[3](#ref-3)].
- **The glue between four tools is code the team owns** (technical): the modular stack needs the search tool, the registry and the pipeline wired together, and that wiring breaks on either side's upgrade. Mitigation: let PyCaret cover criteria 1 to 5 and keep the glue down to the MLflow logging call [[6](#ref-6)].
- **Kubernetes is a prerequisite, not a detail** (operational): Kubeflow Pipelines runs its workflows on Kubernetes [[5](#ref-5)], so a team without that platform takes on the cluster before the agent. Mitigation: start without Kubeflow and add it once more than one pipeline has to be scheduled.
- **A managed platform decides what the data is tried against** (operational): the candidate list is the vendor's, which is why criterion 6 fails. This risk is accepted when the platform is chosen, since that list is what the platform sells.
- **Vendor lock-in follows the registry** (operational): a model registered in a cloud's own registry is addressed by that cloud's API [[1](#ref-1)] [[2](#ref-2)]. Mitigation: keep the lineage in MLflow as well, so the record survives a move.

## 6. Cost and Resource Estimation

The cost of each solution is given as what is paid for rather than as an amount, since comparing prices and contract terms is a Non-Goal and no figure of ours was measured.

<a id="table-3"></a>
Table 3. What each solution is paid for, and what has to be built

| #   | Solution                 | Paid for                             | Engineering effort                                                 |
| :-: | :----------------------: | :----------------------------------: | :----------------------------------------------------------------: |
| 1   | SageMaker, Vertex AI     | The service's own usage-based charge | None beyond the API calls                                          |
| 2   | MLflow, Optuna, Kubeflow | The infrastructure the three run on  | The glue between the four capabilities, and the Kubernetes cluster |
| 3   | PyCaret                  | The machine the experiment runs on   | The MLflow logging call                                            |
| 4   | auto-sklearn, H2O-3      | The machine the experiment runs on   | The export of the winner, and whatever replaces the registry       |

The three open source rows carry no licence fee, so their cost is the machine and the engineering effort beside it. Effort in man-months belongs to the environment it is built in, so a team sizes rows 2 to 4 against its own stack before the roadmap of section 7 is scheduled.

## 7. Recommendations and Next Steps

PyCaret with MLflow is the recommendation: it is the only combination in [Table 2](#table-2) that passes criteria 1 to 5 while leaving criterion 6 with the adopting team and holding criterion 7 to one process and one tracking server [[6](#ref-6)] [[3](#ref-3)]. The managed platforms are not recommended because they fail criterion 6, and auto-sklearn and H2O-3 are not recommended because they fail criteria 1 and 5.

1. **Run one dataset through PyCaret and log it to MLflow.** The stage ends when the leaderboard of a real dataset sits in the MLflow registry with its parameters.
2. **Put the run behind a schedule and a trigger.** The stage ends when the same Python code runs from a request and from a schedule, with every run in the registry.
3. **Add Kubeflow Pipelines, or an orchestrator already in use, once more than one pipeline is scheduled.** The stage ends when retraining and the agent run as flows in one system, which is criterion 5 met outside PyCaret.

## References

<a id="ref-1"></a>
[1] Amazon Web Services. [aws/sagemaker-python-sdk](https://github.com/aws/sagemaker-python-sdk). GitHub repository.<br>
<a id="ref-2"></a>
[2] Google. [googleapis/python-aiplatform](https://github.com/googleapis/python-aiplatform). GitHub repository.<br>
<a id="ref-3"></a>
[3] MLflow. [mlflow/mlflow](https://github.com/mlflow/mlflow). GitHub repository.<br>
<a id="ref-4"></a>
[4] Optuna. [optuna/optuna](https://github.com/optuna/optuna). GitHub repository.<br>
<a id="ref-5"></a>
[5] Kubeflow. [kubeflow/pipelines](https://github.com/kubeflow/pipelines). GitHub repository.<br>
<a id="ref-6"></a>
[6] PyCaret. [pycaret/pycaret](https://github.com/pycaret/pycaret). GitHub repository.<br>
<a id="ref-7"></a>
[7] AutoML Freiburg-Hannover. [automl/auto-sklearn](https://github.com/automl/auto-sklearn). GitHub repository.<br>
<a id="ref-8"></a>
[8] H2O.ai. [h2oai/h2o-3](https://github.com/h2oai/h2o-3). GitHub repository.

---

## Appendix A. Terminology

- **AutoML**: the automated search for the algorithm and the hyperparameters a dataset is best served by.
- **HPO (Hyperparameter Optimization)**: the search for the parameter values that are set before training rather than learned by it.
- **Leaderboard**: the candidates of one search, ranked by the validation metric.
- **Lineage**: the record of which data, code and parameters produced a given model version.
- **Model bank**: the set of algorithms and already fitted models a search draws its candidates from.
- **Model Registry**: the store that holds model versions and what each version was promoted to.
- **PoC (Proof of Concept)**: a small implementation built before adoption to measure what the real one would do.
- **Vendor lock-in**: the state in which moving off one vendor's product has become expensive.
