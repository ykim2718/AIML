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

- **Problem Statement**: 손으로 고르는 대신 가진 데이터에 맞는 model 을 골라 주기를 바라는 팀에게, 그 일이 어떤 기능으로 갈리고 각 기능을 어느 제품이 맡는지가 정리되어 있지 않다.
- **Goal**: Model developer agent 의 일을 model 을 등록하고 탐색하고 검증하고 확정하는 네 기능으로 가르고 각 기능을 맡는 제품을 적어, manager 또는 designer 가 managed platform 과 modular stack 과 single library 가운데 자기 경우에 맞는 것을 고르게 한다.
- **Non-Goal**: 각 algorithm 의 수학과 구현 Python code 는 다루지 않고, 상용 platform 의 가격과 계약 조건은 견주지 않는다.

## 2. Summary

Model developer agent 는 네 기능을 맡는다. Algorithm 과 pre-trained model 을 bank 에 등록하고, 데이터에 맞는 후보를 그 bank 에서 찾고, 후보를 검증해 순위를 내고, 이긴 후보의 hyperparameter 와 artifact 를 확정한다. MLOps 는 이 넷을 Model Registry, AutoML 또는 model search, automated validation, lineage tracking 을 곁들인 hyperparameter optimization 이라 부르며, [Fig 1](#fig-1) 이 그 넷과 그것을 사거나 만드는 세 tier 를 함께 그린다.

Managed platform 은 넷을 한 제품으로 팔고, modular stack 은 기능마다 도구를 하나씩 모아 쓰며, single library 는 registry 없이 한 process 안에서 넷을 돌린다. 고르는 자리는 [Table 1](#table-1) 이고, 네 기능이 어떤 pipeline 으로 도는지는 [Fig 2](#fig-2) 에 있으며, tier 마다의 제품은 [Table 2](#table-2) 와 [Table 3](#table-3) 에 적었다.

## 3. Taxonomy and its Hierarchy

네 기능은 무엇을 정하는가로 갈리고 그 순서대로 돈다. 등록하지 않은 model 은 탐색에 들지 못하고, 검증하지 않은 후보는 확정할 수 없다. 그 아래의 tier 는 후보 목록과 네 기능을 잇는 Python code 를 엔지니어가 얼마나 맡는가로 갈린다. 둘은 [Fig 1](#fig-1) 에 그렸다.

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

기능의 범위가 넓어지면 그 상태를 둘 자리도 달라진다. 검증과 확정은 후보 하나와 이긴 후보 하나에서 끝나므로 한 실행이 맡으면 되고, 등록은 모든 실행에 걸치므로 그 실행이 끝난 뒤에도 남는 저장소를 쓴다. 그 저장소가 single library 가 빼놓는 한 기능이다.

### 3.1 Placement

<a id="table-1"></a>
Table 1. What holds each capability at each tier

| #   | Tier             | Register                              | Search              | Validate                      | Freeze                        |
| :-: | :--------------: | :-----------------------------------: | :-----------------: | :---------------------------: | :---------------------------: |
| 1   | Managed platform | Platform 의 registry                  | Platform 의 AutoML  | Platform 의 validation        | Platform 의 승격 절차         |
| 2   | Modular stack    | MLflow Model Registry                 | AutoML library 하나 | 엔지니어가 쓰는 pipeline 단계 | Optuna, 기록은 MLflow         |
| 3   | Single library   | 없음. Leaderboard 가 memory 에만 있음 | Library 자체 탐색   | Library 자체 평가             | Library 가 돌려주는 estimator |

고른 행이 엔지니어가 쓸 Python code 의 양을 정한다. 1 행은 code 를 쓰지 않고 vendor 의 후보 목록을 받고, 2 행은 네 기능 사이를 잇는 code 를 쓰고 목록을 제 것으로 두고, 3 행은 code 도 registry 도 없어 이긴 model 을 process 가 끝나기 전에 내보내야 한다.

## 4. Pipeline

네 기능은 단계마다 범위를 좁혀 넘기는 하나의 pipeline 으로 돈다. 데이터는 feature 가 되고, feature 는 후보를 고르고, 후보는 하나로 줄고, 이긴 하나는 등록된 version 이 된다. [Fig 2](#fig-2) 가 그 흐름이다.

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

화살표마다 앞 단계가 뒤 단계로 무엇을 넘기는지가 적혀 있고, 단계마다 제품을 갈아 끼울 수 있다. 어느 화살표에 무엇이 지나가는지 대지 못하는 design 은 그 위 단계의 제품을 아직 고르지 못한 것이다.

## 5. Products

제품은 [Fig 1](#fig-1) 의 tier 로 갈린다. Managed platform 은 네 기능을 묶어 팔고, open source 는 기능마다 하나씩 골라 모은다.

### 5.1 Managed Platforms

<a id="table-2"></a>
Table 2. Managed platforms and the capabilities they cover

| #   | Product           | Register                                     | Search                                                  | Validate                                       | Freeze                              |
| :-: | :---------------: | :------------------------------------------: | :-----------------------------------------------------: | :--------------------------------------------: | :---------------------------------: |
| 1   | DataRobot         | 자체 repository                              | 보유 algorithm 집합을 돌리는 자동 modeling              | 교차 검증 점수로 매긴 leaderboard              | 추천 model 을 배포                  |
| 2   | H2O Driverless AI | 자체 repository                              | 자동 feature engineering 과 model 탐색                  | Leaderboard                                    | 이긴 pipeline 을 내보냄             |
| 3   | Amazon SageMaker  | Model package group 으로 묶는 Model Registry | Autopilot [[1](#ref-1)]                                 | Autopilot 이 내는 후보별 지표                  | 등록된 model version [[1](#ref-1)]  |
| 4   | Google Vertex AI  | Model Registry                               | 표·image·text·video·forecasting 의 AutoML [[2](#ref-2)] | SDK 가 나열하는 model evaluation [[2](#ref-2)] | 올려 둔 model version [[2](#ref-2)] |

1 행과 2 행은 platform 이 한 제품으로 파는 것을 적었고, 3 행과 4 행은 기능마다 그 service 의 구성 요소 이름을 적었다. 두 cloud 에서는 네 기능에 한 자리가 아니라 따로 난 API 로 닿는다.

### 5.2 Open Source Stack

<a id="table-3"></a>
Table 3. Open source tools and the capability each one holds

| #   | Tool               | Capability                 | What it does                                                                                                              |
| :-: | :----------------: | :------------------------: | :-----------------------------------------------------------------------------------------------------------------------: |
| 1   | MLflow             | Register, Freeze           | Model 수명 전체를 다루는 Model Registry 와, parameter 와 지표를 남기는 experiment tracking [[3](#ref-3)]                  |
| 2   | Optuna             | Freeze                     | TPE·CMA-ES·NSGA-II sampler 로 hyperparameter 를 찾고, 뒤처지는 trial 을 잘라 냄 [[4](#ref-4)]                             |
| 3   | Kubeflow Pipelines | Pipeline                   | 다시 쓰는 component 를 모아 Kubernetes 위의 ML workflow 하나로 만듦 [[5](#ref-5)]                                         |
| 4   | auto-sklearn       | Search, Validate           | Bayesian optimization 과 meta-learning 으로 scikit-learn estimator 의 algorithm 과 hyperparameter 를 고름 [[6](#ref-6)]   |
| 5   | PyCaret            | Search, Validate, Register | 실험마다 열두 개쯤의 algorithm 을 돌려 leaderboard 를 내고, project 와 registry 를 담은 control plane 을 둠 [[7](#ref-7)] |
| 6   | H2O-3              | Search, Validate           | 여러 model 을 자동으로 학습·조정해 leaderboard 로 줄 세우는 AutoML [[8](#ref-8)]                                          |

네 기능을 혼자 다 덮는 도구는 없으므로 [Table 1](#table-1) 2 행의 stack 은 적어도 둘이다. 후보를 찾는 도구 하나와, 남길 것을 쥐는 MLflow 다. PyCaret 은 셋까지 닿는 예외여서 MLflow 와 짝지으면 가장 짧은 stack 이 된다 [[7](#ref-7)].

## 6. Application

Tier 는 무엇을 자동화하고 싶은가가 아니라 팀이 무엇을 쥐어야 하는가로 정해진다.

1. **Managed platform**: 데이터를 vendor 환경으로 보낼 수 있고, 후보 목록을 유지할 사람을 팀에 두지 않는 경우다. 그때부터 데이터가 어떤 model 과 겨루는지는 vendor 의 algorithm 집합이 정한다.
2. **Modular stack**: 후보 목록이 팀의 것이거나, pipeline 이 이미 돌고 있는 학습 job 곁에서 돌아야 하는 경우다. 값은 네 기능 사이를 잇는 Python code 이며, 그것을 엔지니어가 쓰고 유지한다.
3. **Single library**: 한 데이터셋을 한자리에서 끝내고, 뒤에 무엇을 돌렸는지 아무도 묻지 않는 경우다. Registry 가 없어 이긴 model 이 process 와 함께 사라지므로, process 가 끝나기 전에 내보낸다.

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

- **AutoML**: 데이터에 맞는 algorithm 과 hyperparameter 를 자동으로 찾는 것.
- **HPO (Hyperparameter Optimization)**: 학습으로 얻는 것이 아니라 학습 전에 정하는 parameter 값을 찾는 것.
- **Leaderboard**: 한 번의 탐색이 낸 후보를 검증 지표로 줄 세운 것.
- **Lineage**: 어떤 데이터와 code 와 parameter 가 그 model version 을 만들었는지의 기록.
- **MLOps (Machine Learning Operations)**: model 의 학습·검증·등록·배포를 하나의 운영 체계로 돌리는 방식.
- **Model bank**: 탐색이 후보를 꺼내 오는 algorithm 과 pre-trained model 의 집합.
- **Model Registry**: Model version 과 그 version 이 어디까지 승격되었는지를 담는 저장소.
- **Pre-trained model**: 앞선 데이터로 이미 weight 를 맞춰 두어 다시 학습하지 않고 쓰는 model.
