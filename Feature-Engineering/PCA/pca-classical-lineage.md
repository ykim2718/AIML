# PCA Classical Lineage
Rev. 2 | Created: 2026-08-11 | Updated: 2026-08-11 14:40 CDT

> PCA 는 하나의 기법이 아니라 같은 뿌리에서 갈라져 나온 계통이다.
> 이 문서는 그 가지를 "원래 PCA 의 어떤 가정을 풀었는가" 라는 한 축으로 세우고, 가지마다 무엇을 얻고 무엇을 잃는지를 적는다.

원래 PCA 는 여러 가정을 한꺼번에 전제한다. 데이터가 행렬이고, 한 번에 메모리에 들어가며, 이상치가 없고, 성분이 모든 변수를 조금씩 쓰고, 구조가 선형이며, 표본이 변수보다 많고, 목표 변수를 모른다는 가정이다. 각 가지는 이 중 하나를 푸는 대신 다른 것을 내준다. 그래서 어느 가지를 고를지는 성능 비교가 아니라 **어떤 가정이 내 데이터에서 깨졌는가** 로 정해진다.

## 1. Reading The Map

Table 1. The lineage at a glance

| Branch | Assumption it relaxes | Representative methods | Cost |
|---|---|---|---|
| Computation (§3) | 전체 공분산을 만들 수 있다 | Randomized SVD, Lanczos, Frequent Directions | 근사 오차를 받아들인다 |
| Probabilistic (§4) | 생성 모형이 없어도 된다 | PPCA, Factor Analysis, Bayesian PCA | 잡음 구조를 가정해야 한다 |
| Online (§5) | 데이터가 한 번에 다 있다 | Oja, GHA, CCIPCA, Incremental PCA | 수렴 속도와 학습률에 매인다 |
| Robustness (§6) | 이상치가 없다 | L1-PCA, Robust PCA | 계산이 비싸고 해가 유일하지 않을 수 있다 |
| Sparsity (§7) | 성분이 모든 변수를 써도 된다 | SCoTLASS, Sparse PCA | 직교성과 설명 분산을 일부 포기한다 |
| Non-linear (§8) | 구조가 선형 부분공간이다 | Kernel PCA, Autoencoder | 역변환과 해석이 어려워진다 |
| Asymptotics (§9) | 표본이 변수보다 많다 | Shrinkage, spiked model | 추정량이 모형 가정에 의존한다 |
| Data structure (§10) | 데이터가 행렬이다 | FPCA, Tensor PCA, Dynamic PCA | 구조를 사람이 선언해야 한다 |
| Supervised (§11) | 목표 변수를 모른다 | Supervised PCA, PLS, Contrastive PCA | 목표가 바뀌면 축도 바뀐다 |
| Distribution (§12) | 데이터를 한곳에 모을 수 있다 | Distributed PCA, DP-PCA | 통신량이나 정확도를 내준다 |

가지는 서로 배타적이지 않다. 센서를 시각마다 펼친 데이터처럼 변수가 많고 이상치도 있으면 §6 과 §9 를 함께 쓰는 것이 보통이다. 다만 한 번에 두 가정을 풀면 대가도 함께 붙으므로, 깨진 가정을 먼저 특정하고 그 가지부터 적용한다.

## 2. Root

PCA 는 두 가지 서로 다른 목적이 같은 답에 도달한다는 사실 위에 서 있다. 하나는 사영된 데이터의 분산을 최대로 만드는 방향을 찾는 것이고, 다른 하나는 원 데이터를 가장 작은 오차로 근사하는 저계수 부분공간을 찾는 것이다. Eckart–Young 정리가 두 번째 문제의 답이 절단 SVD 임을 말하고, 그 답이 첫 번째 문제의 답과 같다. 여기에 선형 autoencoder 를 더하면 세 가지가 같은 부분공간을 가리킨다.

Table 2. Three equivalent formulations

| Formulation | Object | Solution |
|---|---|---|
| Variance maximization | 중심화한 데이터의 사영 분산 | 공분산 행렬의 상위 고유벡터 |
| Best low-rank approximation | Frobenius 노름 근사 오차 | 절단 SVD 의 오른쪽 특이벡터 |
| Linear autoencoder | 재현 오차를 내는 선형 encoder-decoder | 주성분이 span 하는 부분공간 |

세 정식화가 같은 부분공간을 주지만 같은 좌표를 주지는 않는다. 선형 autoencoder 는 부분공간만 맞추고 그 안의 회전은 정하지 않으므로, 개별 성분을 해석하려면 세 번째가 아니라 첫 번째 정식화가 필요하다.

전처리가 결과를 바꾼다. 중심화하지 않으면 첫 성분이 평균 방향을 향하고, 표준화하지 않으면 단위가 큰 변수가 분산을 독차지한다. 압력을 Pa 로 적을지 kPa 로 적을지가 성분을 바꾼다는 뜻이므로, 단위가 섞인 계측 데이터에서는 상관행렬 기반, 곧 표준화가 기본이다.

## 3. Computation Branch

같은 부분공간을 구하는 방법이 여럿이고, 데이터의 크기와 데이터를 몇 번 읽을 수 있는지가 어느 것을 쓸지 정한다.

Table 3. Ways to reach the same subspace

| Method | Idea | When it fits |
|---|---|---|
| Covariance eigendecomposition | `p × p` 공분산을 만들어 고유분해한다 | `p` 가 작을 때 |
| Direct SVD | 데이터 행렬을 그대로 분해한다 | 수치적으로 가장 안정적이며 기본값이다 |
| Power iteration, Lanczos | 상위 몇 개 성분만 반복으로 뽑는다 | `k ≪ p` 이고 행렬 곱만 가능할 때 |
| Randomized SVD | 무작위 사영으로 부분공간을 먼저 좁힌다 | 큰 행렬에서 상위 `k` 개를 빠르게 얻을 때 |
| Frequent Directions | 한 번 읽으며 고정 크기 sketch 를 유지한다 | 데이터를 두 번 읽을 수 없을 때 |

`p` 가 크면 `p × p` 공분산을 만드는 것 자체가 병목이 된다. 센서 200 개를 시각 10000 개로 펼치면 `p` 가 이백만이 되므로, 공분산은 만들지 않고 SVD 나 randomized SVD 로 바로 간다.

## 4. Probabilistic Branch

PCA 를 생성 모형으로 다시 쓰면 결측치와 성분 개수를 원리적으로 다룰 수 있게 된다. 관측을 저차원 잠재변수의 선형 사상에 잡음을 더한 것으로 보는 것이 출발점이다.

Table 4. Generative reformulations

| Method | Noise assumption | What it buys |
|---|---|---|
| Probabilistic PCA | 등방 잡음 `σ²I` | 우도, 결측치 처리, EM 알고리즘 |
| Factor Analysis | 대각이지만 등방은 아닌 잡음 | 변수마다 다른 잡음 크기를 인정한다 |
| EM for PCA | — | 공분산을 만들지 않고 반복으로 푼다 |
| Bayesian PCA | 성분별 사전분포 | 성분 개수를 자동으로 줄인다 |

PPCA 와 Factor Analysis 가 실무에서 갈리는 자리는 센서마다 잡음 크기가 다를 때다. 잡음이 등방이라는 가정을 유지하면 잡음이 큰 센서가 주성분 방향을 끌어당긴다.

## 5. Online And Incremental Branch

데이터가 흘러 들어오거나 한 번에 담기지 않을 때는 성분을 갱신해 나간다.

Table 5. Updating instead of recomputing

| Method | Update unit | Note |
|---|---|---|
| Oja, GHA | 표본 하나 | 학습률 설계가 수렴을 좌우한다 |
| CCIPCA | 표본 하나 | 학습률 대신 표본 수로 평균을 내어 조정 항이 없다 |
| Incremental PCA | 블록 | 블록 SVD 를 이어 붙인다 |
| Streaming SVD | 블록 | 오래된 데이터에 망각 계수를 둘 수 있다 |

흐르는 데이터에서는 부분공간 자체가 시간에 따라 변할 수 있다. 그 변화를 추정 오차로 볼지 공정 변화의 신호로 볼지가 이 가지의 핵심 질문이며, FDC 에서는 대개 후자다.

## 6. Robustness Branch

이상치 하나가 최소제곱 기준의 성분을 통째로 끌어당긴다. 계측 데이터에서 이상치는 예외가 아니라 상시 조건이다.

Table 6. Resisting outliers

| Method | Idea | Cost |
|---|---|---|
| L1-PCA | 제곱 대신 절댓값 오차를 최소화한다 | 최적화가 볼록하지 않다 |
| M-estimation | 잔차가 큰 표본의 가중치를 낮춘다 | 가중 함수와 조율값이 필요하다 |
| Robust PCA | 행렬을 저계수와 희소의 합으로 분해한다 | 볼록 완화로 풀지만 계산이 비싸다 |

Robust PCA 는 이상치를 버리는 대신 희소 성분에 따로 담는다. 그 희소 성분이 곧 이상 표본의 목록이므로, 축소와 이상 검출을 한 번에 얻는다.

## 7. Sparsity And Interpretability Branch

주성분은 보통 모든 변수에 0 이 아닌 하중을 준다. 변수가 수백 개면 그 성분이 무엇을 뜻하는지 말할 수 없다.

Table 7. Making loadings readable

| Method | Idea | What it gives up |
|---|---|---|
| Varimax rotation | 부분공간을 유지한 채 축을 회전한다 | 성분별 분산의 순서 |
| SCoTLASS | 하중에 L1 제약을 건다 | 최적화가 어렵다 |
| Sparse PCA | 회귀 문제로 바꾸고 elastic net 을 쓴다 | 성분 간 직교성 |
| Structured sparsity | 변수 그룹 단위로 0 을 만든다 | 그룹 정의를 사람이 준다 |

회전은 부분공간을 바꾸지 않으므로 재현 오차가 그대로다. 반면 희소화는 부분공간 자체를 바꾸므로 설명 분산이 줄어든다. 해석성을 위해 얼마를 낼지 정하는 자리다.

## 8. Non-linear Branch

구조가 선형 부분공간이 아닐 때 쓰는 가지다.

Table 8. Leaving the linear subspace

| Method | Idea | Note |
|---|---|---|
| Kernel PCA | 특징 공간에서 PCA 를 하되 내적만 쓴다 | 표본 수 제곱의 커널 행렬이 병목이며 Nyström 으로 근사한다 |
| Autoencoder | 비선형 encoder-decoder 를 학습한다 | 은닉층이 선형이면 PCA 부분공간과 같아진다 |
| Manifold learning | 이웃 관계를 보존한다 | PCA 의 후손이 아니라 이웃이다 |

t-SNE 와 UMAP 을 이 가지에 놓는 것은 오해를 부른다. 둘은 사영 함수를 주지 않아 새 표본을 같은 좌표로 보낼 수 없고 거리도 보존하지 않으므로, 시각화 도구이지 차원 축소 단계의 대체가 아니다.

## 9. High-Dimensional Asymptotics Branch

`p` 가 `n` 에 비해 클 때는 표본 고유값이 모집단 고유값에서 체계적으로 벗어난다. 잡음만 있는 데이터에서도 상위 고유값이 크게 나오므로, 크다는 사실만으로 신호라고 부를 수 없다.

Table 9. What large `p` does

| Topic | Statement |
|---|---|
| Marchenko–Pastur | 순수 잡음의 고유값이 특정 구간에 퍼져 나타난다 |
| Spiked model | 신호가 임계값을 넘어야 잡음 구간 밖으로 분리된다 |
| Eigenvalue shrinkage | 표본 고유값을 줄여 편향을 보정한다 |
| Component count | scree, 평행분석, 교차검증으로 정한다 |

임계값 아래의 신호는 표본 고유벡터가 모집단 고유벡터와 무관해지는 영역이다. 성분 개수를 고르기 전에 이 경계를 확인하지 않으면, 잡음 방향을 주성분으로 채택하게 된다.

## 10. Data Structure Branch

데이터가 행렬이 아닐 때 구조를 펼쳐 버리면 그 구조가 담고 있던 정보를 잃는다.

Table 10. When the data is not a matrix

| Method | Structure it keeps | Typical data |
|---|---|---|
| Functional PCA | 곡선의 매끄러움 | 시간에 따라 이어지는 계측 곡선 |
| Tensor, Multilinear PCA | 여러 축의 곱 구조 | wafer × sensor × time |
| 2DPCA | 이미지의 행과 열 | 웨이퍼 맵 |
| Dynamic PCA | 시차 상관 | 자기상관이 있는 공정 트레이스 |
| Multi-block | 블록 사이의 관계 | 장비와 계측을 함께 두는 경우 |

트레이스를 시각마다 하나의 변수로 펼치면 이웃한 시각이 서로 무관한 변수가 된다. Functional PCA 는 그 인접성을 유지하고, Dynamic PCA 는 시차를 변수로 명시해 자기상관을 모형 안으로 넣는다.

## 11. Supervised And Contrastive Branch

목표 변수를 알고 있으면 분산이 큰 방향이 아니라 목표와 관계있는 방향을 찾는 편이 낫다.

Table 11. Reduction that knows the target

| Method | Uses | Note |
|---|---|---|
| Supervised PCA | `y` 와의 상관으로 변수를 먼저 거른다 | PCA 자체는 그대로 쓴다 |
| PLS | `X` 와 `y` 의 공분산을 최대화한다 | 회귀와 축소를 함께 한다 |
| CCA | 두 블록의 상관을 최대화한다 | 목표가 벡터일 때 |
| LDA | 클래스 사이 분산 대 안쪽 분산 | 분류 전용이며 성분 수가 클래스 수에 매인다 |
| Contrastive PCA | 배경 데이터 대비 특이한 방향 | 정상군이 따로 있을 때 |

분산이 큰 방향과 목표를 설명하는 방향은 자주 어긋난다. 장비 사이의 큰 차이가 첫 성분을 차지하고 수율과 관계있는 미세한 변동은 뒤로 밀리는 상황이 그 예다.

## 12. Distributed And Private Branch

데이터를 한곳에 모을 수 없을 때의 가지다.

Table 12. Computing without gathering the data

| Method | Constraint | Idea |
|---|---|---|
| Distributed PCA | 데이터가 여러 노드에 있다 | 노드별 요약을 모아 합친다 |
| Federated PCA | 원 데이터를 내보낼 수 없다 | 갱신량만 주고받는다 |
| Differentially Private PCA | 개별 표본이 드러나면 안 된다 | 공분산이나 성분에 잡음을 더한다 |

공장 사이, 고객 사이에 데이터를 옮길 수 없는 상황이 이 가지가 필요한 자리다. 정확도를 얼마나 내주어야 하는지가 그 제약의 대가다.

## Appendix A. Terminology

본문에서 정의하지 않고 쓴 용어를 정리한다.

- **Eckart–Young** 은 주어진 계수의 행렬 근사 중 절단 SVD 가 오차를 최소로 한다는 정리이다.
- **Elastic net** 은 L1 과 L2 벌점을 함께 쓰는 회귀 정칙화이다.
- **EM** 은 잠재변수가 있는 모형을 기댓값 단계와 최대화 단계의 반복으로 적합하는 알고리즘이다.
- **FDC** 는 Fault Detection and Classification 이고, 공정 장비의 센서 기록으로 이상을 찾는 체계이다.
- **Frobenius norm** 은 행렬 원소의 제곱합의 제곱근이다.
- **Lanczos** 는 대칭 행렬의 상위 고유쌍을 행렬 곱만으로 구하는 반복법이다.
- **Loading** 은 주성분을 원 변수의 선형결합으로 적을 때의 계수이다.
- **Marchenko–Pastur** 는 표본 공분산 고유값이 따르는 극한 분포이다.
- **Nyström** 은 커널 행렬의 일부 열만으로 전체를 근사하는 방법이다.
- **Scree** 는 고유값을 크기순으로 그린 그림이고, 꺾이는 자리를 성분 개수로 삼는다.
- **Sketch** 는 원 데이터보다 작은 크기로 유지하는 요약 행렬이다.
- **Spiked model** 은 잡음 공분산에 소수의 큰 고유값을 더한 모형이다.
- **SVD** 는 행렬을 좌특이벡터, 특이값, 우특이벡터의 곱으로 분해하는 것이다.
- **Varimax** 는 하중의 분산을 키워 해석을 쉽게 하는 직교 회전이다.
