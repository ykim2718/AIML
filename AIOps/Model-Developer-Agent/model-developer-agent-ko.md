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

지금은 데이터에 맞는 model 을 손으로 고르므로, 겨루는 algorithm 이 분석자가 기억하는 것에 머물고 무엇을 돌렸는지가 남지 않는다. 조사 결론은 도입 진행 (Go) 이며, 그 대상은 PyCaret 과 MLflow 를 짝지은 modular stack 이다. Model developer agent 에 필요한 네 기능을 혼자 다 덮는 open source 도구가 없고, managed platform 은 후보 목록을 vendor 에 넘기기 때문이다.

- **Problem Statement**: 손으로 고르는 대신 가진 데이터에 맞는 model 을 골라 주기를 바라는 팀에게, 그 일이 어떤 기능으로 갈리고 각 기능을 어느 제품이 맡는지가 정리되어 있지 않다.
- **Goal**: 그 일을 model 을 등록하고 탐색하고 검증하고 확정하는 네 기능으로 가르고, 후보 solution 을 그 기준으로 채점해 하나를 권고하여, manager 또는 designer 가 그 권고나 그것을 뒤집는 조건을 받아 가게 한다.
- **Non-Goal**: 각 algorithm 의 수학과 구현 Python code 는 다루지 않고, 상용 platform 의 가격과 계약 조건은 견주지 않는다.

## 2. Proposed Solutions

검토한 solution 은 셋이고, 후보 목록과 네 기능을 잇는 Python code 를 엔지니어가 얼마나 맡는가로 갈린다. 네 기능을 한 제품으로 파는 managed platform, 기능마다 도구를 하나씩 모은 modular stack, 한 process 안에서 넷을 돌리는 single library 다. [Fig 1](#fig-1) 이 네 기능과 그 자리에 놓인 세 solution 을 함께 그린다.

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

어느 solution 을 쓰든 네 기능은 단계마다 범위를 좁혀 넘기는 하나의 pipeline 으로 기존 system 에 맞물리고, 단계마다 제품을 갈아 끼울 수 있다. [Fig 2](#fig-2) 가 그 흐름이다.

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

세 solution 의 제품은 managed platform 쪽이 Amazon SageMaker 와 Google Vertex AI [[1](#ref-1)] [[2](#ref-2)], modular stack 쪽이 MLflow 와 Optuna 와 Kubeflow Pipelines [[3](#ref-3)] [[4](#ref-4)] [[5](#ref-5)], single library 쪽이 PyCaret 과 auto-sklearn 과 H2O-3 이다 [[6](#ref-6)] [[7](#ref-7)] [[8](#ref-8)]. DataRobot 과 H2O Driverless AI 도 managed platform 과 같은 모양으로 파는 제품이며, 이 문서는 그 둘의 vendor page 를 인용하지 않는다.

## 3. Evaluation Criteria

평가 항목은 네 기능에, 무엇을 누가 쥐는지를 정하는 세 항목을 더한 일곱이다. 항목마다 우리가 측정한 benchmark 가 아니라 제품의 공개 문서에 적힌 것으로 판정한다.

<a id="table-1"></a>
Table 1. The criteria and what each one is measured by

| #   | Criterion                | Measured by                                                 | Passing                     |
| :-: | :----------------------: | :---------------------------------------------------------: | :-------------------------: |
| 1   | Register                 | Model version 이 그것을 만든 process 보다 오래 남는가       | 실행 밖의 저장소            |
| 2   | Search                   | 후보를 하나씩 지목하지 않아도 학습·순위가 되는가            | 한 실행에 algorithm 여러 개 |
| 3   | Validate                 | 순위가 학습 적합도가 아니라 검증 분할에서 나오는가          | 교차 검증 또는 holdout      |
| 4   | Freeze                   | 이긴 hyperparameter 와 artifact 를 lineage 와 함께 남기는가 | 둘을 함께 기록              |
| 5   | Pipeline integration     | 재학습과 배포가 같은 체계에서 도는가                        | 하나의 scheduler            |
| 6   | Candidate list ownership | 데이터가 어떤 algorithm 과 겨루는지 누가 정하는가           | 도입하는 팀                 |
| 7   | Operating surface        | 운영할 구성 요소가 몇인가                                   | 일에 필요한 만큼만          |

1 부터 4 는 네 기능 그대로여서, 하나를 못 하는 solution 은 그 기능을 엔지니어가 쓰는 Python code 에 넘긴다. 5 부터 7 은 네 기능을 모두 통과한 solution 을 가르는 항목이다.

## 4. Technical Analysis and Comparison

이 조사에서 PoC 를 돌리지 않았으므로 [Table 2](#table-2) 에는 제품의 공개 repository 에 적힌 것만 있고 우리가 잰 latency 나 throughput 은 없다.

<a id="table-2"></a>
Table 2. The candidate solutions against the criteria of Table 1

| #   | Criterion                | SageMaker, Vertex AI          | MLflow, Optuna, Kubeflow               | PyCaret                           | auto-sklearn, H2O-3           | Note                                                    |
| :-: | :----------------------: | :---------------------------: | :------------------------------------: | :-------------------------------: | :---------------------------: | :-----------------------------------------------------: |
| 1   | Register                 | 각 service 의 Model Registry  | MLflow Model Registry                  | Control plane 의 registry         | 없음                          | [[1](#ref-1)] [[2](#ref-2)] [[3](#ref-3)] [[6](#ref-6)] |
| 2   | Search                   | Autopilot, 자료 종류별 AutoML | AutoML library 를 더해야 함            | 실험마다 열두 개쯤의 algorithm    | Library 범위의 algorithm 선택 | [[1](#ref-1)] [[2](#ref-2)] [[6](#ref-6)] [[7](#ref-7)] |
| 3   | Validate                 | Service 가 내는 후보별 지표   | 엔지니어가 쓰는 pipeline 단계          | 자체 leaderboard                  | 자체 leaderboard 또는 점수    | [[2](#ref-2)] [[6](#ref-6)] [[8](#ref-8)]               |
| 4   | Freeze                   | 등록된 model version          | Optuna, 기록은 MLflow                  | 돌려받은 estimator 를 승격        | 돌려받은 estimator            | [[3](#ref-3)] [[4](#ref-4)] [[6](#ref-6)]               |
| 5   | Pipeline integration     | Service 자체 pipeline         | Kubernetes 위의 Kubeflow Pipelines     | Control plane 의 deployment       | 없음                          | [[5](#ref-5)] [[6](#ref-6)]                             |
| 6   | Candidate list ownership | Vendor                        | 도입하는 팀                            | 도입하는 팀                       | 도입하는 팀                   | 제품의 범위에서 읽음                                    |
| 7   | Operating surface        | Cloud 계정 하나               | Tracking server, Kubernetes, 이음 code | Process 하나와 control plane 하나 | Process 하나                  | 운영할 구성 요소의 수                                   |

PyCaret 은 1 부터 5 까지를 혼자 통과하며, 표의 다른 open source 항목에는 그런 것이 없다 [[6](#ref-6)]. 다만 그 registry 는 MLflow 에 두는 쪽이 나을 수 있다. 기록이 쓰고 있는 PyCaret version 보다 오래 남기 때문이다 [[3](#ref-3)]. Managed 열은 네 기능을 모두 통과하지만 6 을 정의상 통과하지 못하고, auto-sklearn 과 H2O-3 열은 process 가 끝나면 아무것도 남기지 않아 1 과 5 를 통과하지 못한다 [[7](#ref-7)] [[8](#ref-8)].

## 5. Risks and Constraints

아래 risk 마다 그것을 줄이는 방법을 함께 적었고, 줄일 방법이 없는 하나는 받아들이는 risk 로 적었다.

- **Library 가 고른 model 이 process 와 함께 사라진다** (기술): auto-sklearn 과 H2O-3 는 estimator 를 돌려주고 registry 를 두지 않으므로, process 가 끝나기 전에 내보내지 않은 model 은 없어진다 [[7](#ref-7)] [[8](#ref-8)]. 완화: script 안에서 실행과 artifact 를 MLflow 에 기록한다 [[3](#ref-3)].
- **네 도구를 잇는 code 가 팀의 몫으로 남는다** (기술): modular stack 은 탐색 도구와 registry 와 pipeline 을 이어 붙여야 하고, 그 이음은 어느 한쪽을 올릴 때 깨진다. 완화: 1 부터 5 를 PyCaret 이 덮게 하고 이음을 MLflow 기록 호출 하나로 줄인다 [[6](#ref-6)].
- **Kubernetes 가 곁가지가 아니라 선행 조건이다** (운영): Kubeflow Pipelines 는 workflow 를 Kubernetes 위에서 돌리므로 [[5](#ref-5)], 그 platform 이 없는 팀은 agent 보다 cluster 를 먼저 안게 된다. 완화: Kubeflow 없이 시작하고, 예약할 pipeline 이 둘을 넘을 때 더한다.
- **Managed platform 은 데이터가 무엇과 겨루는지를 vendor 가 정한다** (운영): 후보 목록이 vendor 의 것이어서 6 을 통과하지 못한다. 그 목록이 곧 platform 이 파는 것이므로, platform 을 고르는 순간 받아들이는 risk 다.
- **Lock-in 은 registry 를 따라온다** (운영): Cloud 의 registry 에 등록한 model 은 그 cloud 의 API 로만 불린다 [[1](#ref-1)] [[2](#ref-2)]. 완화: lineage 를 MLflow 에도 남겨 옮길 때 기록이 따라오게 한다.

## 6. Cost and Resource Estimation

비용은 금액이 아니라 무엇에 돈을 내는지로 적는다. 가격과 계약 조건 비교는 Non-Goal 이고, 우리가 잰 값도 없다.

<a id="table-3"></a>
Table 3. What each solution is paid for, and what has to be built

| #   | Solution                 | Paid for                    | Engineering effort                             |
| :-: | :----------------------: | :-------------------------: | :--------------------------------------------: |
| 1   | SageMaker, Vertex AI     | Service 의 사용량 기반 요금 | API 호출 밖에는 없음                           |
| 2   | MLflow, Optuna, Kubeflow | 셋이 도는 infrastructure    | 네 기능을 잇는 code 와 Kubernetes cluster      |
| 3   | PyCaret                  | 실험이 도는 machine         | MLflow 기록 호출                               |
| 4   | auto-sklearn, H2O-3      | 실험이 도는 machine         | 이긴 model 의 내보내기와 registry 를 대신할 것 |

Open source 세 행은 license 비용이 없어, 값이 machine 과 그 곁의 공수다. 공수의 M/M 은 그것을 만드는 환경의 것이므로, 꼭지 7 의 roadmap 을 일정에 넣기 전에 2 부터 4 행을 자기 구성에 대고 산정한다.

## 7. Recommendations and Next Steps

권고는 PyCaret 과 MLflow 다. [Table 2](#table-2) 에서 1 부터 5 를 통과하면서 6 을 도입하는 팀에 남기고 7 을 process 하나와 tracking server 하나로 묶는 조합이 이것뿐이다 [[6](#ref-6)] [[3](#ref-3)]. Managed platform 은 6 을 통과하지 못해서, auto-sklearn 과 H2O-3 는 1 과 5 를 통과하지 못해서 권고하지 않는다.

1. **데이터셋 하나를 PyCaret 으로 돌리고 MLflow 에 기록한다.** 실제 데이터셋의 leaderboard 가 parameter 와 함께 MLflow registry 에 들어가면 이 단계가 끝난다.
2. **그 실행을 schedule 과 trigger 뒤에 둔다.** 같은 Python code 가 요청에서도 schedule 에서도 돌고 모든 실행이 registry 에 남으면 이 단계가 끝난다.
3. **예약할 pipeline 이 둘을 넘으면 Kubeflow Pipelines 나 이미 쓰는 orchestrator 를 더한다.** 재학습과 agent 가 한 체계의 flow 로 돌면 이 단계가 끝나며, 그것이 PyCaret 밖에서 5 를 통과한 것이다.

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

- **AutoML**: 데이터에 맞는 algorithm 과 hyperparameter 를 자동으로 찾는 것.
- **HPO (Hyperparameter Optimization)**: 학습으로 얻는 것이 아니라 학습 전에 정하는 parameter 값을 찾는 것.
- **Leaderboard**: 한 번의 탐색이 낸 후보를 검증 지표로 줄 세운 것.
- **Lineage**: 어떤 데이터와 code 와 parameter 가 그 model version 을 만들었는지의 기록.
- **Model bank**: 탐색이 후보를 꺼내 오는 algorithm 과 이미 학습해 둔 model 의 집합.
- **Model Registry**: Model version 과 그 version 이 어디까지 승격되었는지를 담는 저장소.
- **PoC (Proof of Concept)**: 도입 전에 작게 구현해 실제로 어떻게 될지를 재어 보는 검증.
- **Vendor lock-in**: 한 vendor 의 제품에서 다른 것으로 옮기는 비용이 커진 상태.
