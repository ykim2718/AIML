# Model Developer Agent
Rev. 0 | Created: 2026-10-03 | Updated: 2026-10-03 15:52 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Placement](#31-placement)
- [4. Pipeline](#4-pipeline)
- [5. Products](#5-products)
  - [5.1 Managed Platforms](#51-managed-platforms)
  - [5.2 Open Source Stack](#52-open-source-stack)
- [6. Application](#6-application)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

- **Problem Statement**: A team that wants a model chosen for the data it has, rather than by hand, has no account of which capabilities that takes and which product holds each one.
- **Goal**: Split the work of a model developer agent into the four capabilities that register, search, validate and freeze a model, and name the products that hold each, so that a manager or a designer picks a managed platform, a modular stack or a single library for their own case.
- **Non-Goal**: The mathematics of each algorithm and the implementation Python code are not covered, and the price and the contract terms of a commercial platform are not compared.

## 2. Summary

A model developer agent carries four capabilities: it registers algorithms and pre-trained models in a bank, searches the bank for the candidates a dataset deserves, validates and ranks those candidates, and freezes the hyperparameters and the artifact of the winner. Those four are what MLOps calls a Model Registry, AutoML or model search, automated validation, and hyperparameter optimization with lineage tracking; [Fig 1](#fig-1) places them and the three tiers a team can buy or build them at.

A managed platform holds all four as one product, a modular stack assembles them from one tool per capability, and a single library runs all four inside one process without a registry. [Table 1](#table-1) is what a design picks from, [Fig 2](#fig-2) draws the pipeline the four capabilities run as, and [Table 2](#table-2) and [Table 3](#table-3) name the products of each tier.

## 3. Taxonomy and its Hierarchy

The four capabilities split by what each one settles, and they run in that order: nothing can be searched that was never registered, and nothing can be frozen that was never validated. The tiers below them split by how much of the candidate list and the glue code a team keeps. [Fig 1](#fig-1) draws both.

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
Fig 1. The four capabilities of a model developer agent and the three tiers that hold them

Widening a capability's scope moves where its state has to live. Validation and freezing are settled per candidate and per winner, so a run holds them; registration spans every run, so it needs a store that outlives the run that wrote it, which is the one capability a single library leaves out.

### 3.1 Placement

<a id="table-1"></a>
Table 1. What holds each capability at each tier

| #   | Tier             | Register                              | Search                   | Validate                            | Freeze                           |
| :-: | :--------------: | :-----------------------------------: | :----------------------: | :---------------------------------: | :------------------------------: |
| 1   | Managed platform | The platform's registry               | The platform's AutoML    | The platform's validation           | The platform, as a promotion     |
| 2   | Modular stack    | MLflow Model Registry                 | An AutoML library        | A pipeline step the engineers write | Optuna, recorded in MLflow       |
| 3   | Single library   | Nothing, the leaderboard is in memory | The library's own search | The library's own scoring           | The library's returned estimator |

The row a design picks fixes how much Python code the engineers write. Row 1 writes none and takes the vendor's candidate list; row 2 writes the glue between four tools and keeps the list; row 3 writes nothing and keeps no registry, so the winner has to be exported before the process ends.

## 4. Pipeline

The four capabilities run as one pipeline whose stages pass a narrowing set forward: the dataset becomes features, the features select candidates, the candidates become one winner, and the winner becomes a registered version. [Fig 2](#fig-2) draws it.

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

Each arrow is where a stage hands something to the next one, and each stage is a place a product can be swapped. A design that cannot name what crosses one of these arrows has not yet chosen a product for the stage above it.

## 5. Products

Products divide by the tier of [Fig 1](#fig-1): a managed platform sells all four capabilities together, while an open source stack is assembled per capability.

### 5.1 Managed Platforms

<a id="table-2"></a>
Table 2. Managed platforms and the capabilities they cover

| #   | Product           | Register                               | Search                                                               | Validate                                      | Freeze                                   |
| :-: | :---------------: | :------------------------------------: | :------------------------------------------------------------------: | :-------------------------------------------: | :--------------------------------------: |
| 1   | DataRobot         | Its own repository                     | Automated modeling over its algorithm set                            | A leaderboard scored by cross-validation      | The recommended model, deployed          |
| 2   | H2O Driverless AI | Its own repository                     | Automated feature engineering and model search                       | A leaderboard                                 | The winning pipeline, exported           |
| 3   | Amazon SageMaker  | Model Registry, by model package group | Autopilot [[1](#ref-1)]                                              | Autopilot's candidate metrics                 | A registered model version [[1](#ref-1)] |
| 4   | Google Vertex AI  | Model Registry                         | AutoML for tabular, image, text, video and forecasting [[2](#ref-2)] | Model evaluations the SDK lists [[2](#ref-2)] | An uploaded model version [[2](#ref-2)]  |

Rows 1 and 2 state what the platform sells as one product; rows 3 and 4 name the service's own component per capability, so a design on either cloud reaches the four through separate APIs rather than one.

### 5.2 Open Source Stack

<a id="table-3"></a>
Table 3. Open source tools and the capability each one holds

| #   | Tool               | Capability                 | What it does                                                                                                                               |
| :-: | :----------------: | :------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: |
| 1   | MLflow             | Register, Freeze           | Model Registry over the model lifecycle, with experiment tracking of parameters and metrics [[3](#ref-3)]                                  |
| 2   | Optuna             | Freeze                     | Hyperparameter optimization by TPE, CMA-ES and NSGA-II samplers, pruning the trials that lag [[4](#ref-4)]                                 |
| 3   | Kubeflow Pipelines | Pipeline                   | Reusable components composed into one ML workflow on Kubernetes [[5](#ref-5)]                                                              |
| 4   | auto-sklearn       | Search, Validate           | Algorithm selection and hyperparameter tuning over scikit-learn estimators, by Bayesian optimization and meta-learning [[6](#ref-6)]       |
| 5   | PyCaret            | Search, Validate, Register | Roughly twelve algorithms trained per experiment into a leaderboard, with a control plane that holds projects and a registry [[7](#ref-7)] |
| 6   | H2O-3              | Search, Validate           | AutoML that trains and tunes many models and ranks them on a leaderboard [[8](#ref-8)]                                                     |

No single tool covers all four, so the stack of [Table 1](#table-1) row 2 is at least two: a search tool for the candidates and MLflow for what is kept. PyCaret is the exception that reaches three, which is why it pairs with MLflow into the shortest stack to run [[7](#ref-7)].

## 6. Application

The tier is fixed by what a team has to keep rather than by what it wants to automate.

1. **Managed platform**: the data can leave for the vendor's environment and nobody on the team is to maintain the candidate list. The vendor's algorithm set then decides which models the data is ever tried against.
2. **Modular stack**: the candidate list is the team's own, or the pipeline has to run beside existing training jobs. The cost is the glue between the four capabilities, which is Python code the engineers write and keep.
3. **Single library**: the work is one dataset at one sitting and no later run will ask what was tried. Without a registry, the winner lives only as long as the process, so it is exported before the process ends.

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
[6] AutoML Freiburg-Hannover. [automl/auto-sklearn](https://github.com/automl/auto-sklearn). GitHub repository.<br>
<a id="ref-7"></a>
[7] PyCaret. [pycaret/pycaret](https://github.com/pycaret/pycaret). GitHub repository.<br>
<a id="ref-8"></a>
[8] H2O.ai. [h2oai/h2o-3](https://github.com/h2oai/h2o-3). GitHub repository.

---

## Appendix A. Terminology

- **AutoML**: the automated search for the algorithm and the hyperparameters a dataset is best served by.
- **HPO (Hyperparameter Optimization)**: the search for the parameter values that are set before training rather than learned by it.
- **Leaderboard**: the candidates of one search, ranked by the validation metric.
- **Lineage**: the record of which data, code and parameters produced a given model version.
- **MLOps (Machine Learning Operations)**: the practice of running the training, validation, registration and deployment of a model as one operated system.
- **Model bank**: the set of algorithms and pre-trained models a search draws its candidates from.
- **Model Registry**: the store that holds model versions and what each version was promoted to.
- **Pre-trained model**: a model whose weights were fitted on earlier data and are reused rather than fitted again.
