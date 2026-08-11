# PCA
Rev. 1 | Created: 2026-08-11 | Updated: 2026-08-11 13:16 CDT

> 이 폴더는 PCA 를 하나의 기법이 아니라 계통으로 다룬다.
> 문서는 계통을 세우는 것, 그 계통을 현장 데이터에 대는 것, 그리고 표현학습 시대의 자리를 적는 것으로 나뉜다.

## 1. Scope

PCA 는 원래 여러 가정을 한꺼번에 깔고 있고, 변형들은 그 중 하나씩을 풀어 준 결과다. 그래서 이 폴더의 문서는 기법을 성능순으로 늘어놓지 않고 **어떤 가정이 깨졌을 때 어느 가지로 가는가** 를 축으로 삼는다.

## 2. Documents

Table 1. Documents in this folder

| Document | Description |
|---|---|
| [pca-classical-lineage.md](pca-classical-lineage.md) | 고전 계통을 열 갈래로 세운다. 계산, 확률 모형, 온라인, 강건성, 희소성, 비선형, 고차원 점근, 데이터 구조, 감독, 분산이 각각 하나의 가지이며, 문서 머리의 요약표가 그 지도다. |
| [pca-applications.md](pca-applications.md) | 반도체 계측 데이터에서 어느 가지가 실제로 쓰이는지를 데이터 종류별로 적고, 데이터 조건에서 기법으로 가는 결정표를 둔다. |
| [pca-modern-lineage.md](pca-modern-lineage.md) | 자기지도 표현학습에서 PCA 가 남아 있는 자리와, 함수형 데이터·딥러닝·목표 변수를 결합한 최근 갈래를 정리한다. |

## 3. Code

Table 2. Scripts in this folder

| File | Description |
|---|---|
| `ccipca.py` | Candid Covariance-free Incremental PCA 구현이다. 표본 하나마다 성분을 갱신하며 학습률 대신 표본 수로 평균을 낸다. 온라인 계통의 예이다. |

## 4. Order Of Use

1. 어떤 가정이 깨졌는지 모르는 상태라면 고전 계통 문서의 요약표부터 읽는다.
2. 데이터가 무엇인지 이미 알고 있다면 응용 문서의 결정표에서 시작해 필요한 가지만 거슬러 올라간다.
3. 학습된 표현을 다루거나 신경망과 결합할 계획이면 현대 계통 문서를 본다.

가지는 서로 배타적이지 않다. 한 데이터에 두 가지 이상이 걸리는 것이 보통이며, 그때는 한 번에 하나씩 적용해 각각의 효과를 확인한다.
