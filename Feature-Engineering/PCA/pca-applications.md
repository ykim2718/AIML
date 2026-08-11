# PCA Applications
Rev. 3 | Created: 2026-08-11 | Updated: 2026-08-11 14:46 CDT

> 계통을 아는 것과 무엇을 쓸지 정하는 것은 다른 일이다.
> 이 문서는 확장이 어느 방향으로 일어났는지를 먼저 묶고, 반도체 계측 데이터에서 어느 가지가 실제로 쓰이는지를 데이터 종류별로 적은 뒤, 데이터 조건에서 기법으로 가는 결정표를 둔다.

PCA 의 변형은 열 갈래가 넘지만 한 현장에서 동시에 쓰이는 것은 두셋이다. 무엇이 쓰이는지는 데이터가 어떤 모양으로 오는지와 그 데이터에서 어떤 가정이 깨져 있는지로 정해진다. 반도체 데이터는 그 조건이 비교적 뚜렷해서, 데이터 종류만 알아도 후보가 크게 좁혀진다.

## 1. Four Directions Of Extension

확장은 데이터의 한계와 분석의 목적을 따라 네 방향으로 일어났다. 아래 넷은 배타적이지 않으며, 한 데이터가 둘 이상에 걸리는 것이 보통이다.

Table 1. Four directions and their methods

| Direction | Limitation or purpose | Methods |
|---|---|---|
| Nonlinear and kernel | 선형 부분공간으로 나뉘지 않는다 | Kernel PCA |
| Robustness and missing data | 이상치가 섞이고 결측이 많다 | Robust PCA, Probabilistic PCA |
| Complex data structure | 행렬로 펼치면 구조를 잃는다 | Sparse PCA, Tensor PCA, Functional PCA |
| High-dimensional and large data | 메모리와 계산 시간이 한계다 | Randomized PCA, Incremental PCA |

**Nonlinear and kernel.** Kernel PCA 는 커널 트릭으로 특징 공간에서 성분을 구해, 선형으로 나뉘지 않는 구조를 뽑는다. 이웃 관계를 보존하는 매니폴드 학습과 맞닿아 있으나, 사영 함수를 주는지가 둘을 가른다.

**Robustness and missing data.** Robust PCA 는 행렬을 정상 차원과 이상치의 합으로 분해해 둘을 갈라 놓는다. Probabilistic PCA 는 PCA 를 확률 모형으로 다시 써서, 결측이 많은 데이터에서도 주성분을 안정적으로 추정한다.

**Complex data structure.** Sparse PCA 는 성분이 쓰는 변수를 줄여 읽을 수 있게 만들고, Tensor PCA 는 다차원 구조를 펼치지 않은 채 축소하며, Functional PCA 는 이어진 곡선 자체를 하나의 단위로 다룬다. Functional PCA 는 배경 대비 특이한 곡선 변동을 찾는 contrastive 형태로도 이어졌다.

**High-dimensional and large data.** Randomized PCA 는 무작위 사영으로 상위 성분을 빠르게 근사하고, Incremental PCA 는 데이터를 조각으로 나누어 갱신해 메모리 한계를 넘는다.

네 방향은 서로 다른 것을 고친다. 첫째는 모양을, 둘째는 자료의 질을, 셋째는 구조를, 넷째는 규모를 고친다. 그래서 규모 문제를 강건성 기법으로 풀거나 그 반대로 하는 선택은 성립하지 않는다.

## 2. Data Types And Their Conditions

Table 2. What each data type breaks

| Data type | Shape | Broken assumption |
|---|---|---|
| Wafer metrology | 웨이퍼 × 계측 항목 | 표본이 적고 항목 사이 상관이 강하다 |
| FDC trace | 웨이퍼 × 센서 × 시각 | 행렬이 아니고 `p` 가 `n` 을 크게 넘는다 |
| Wafer map | 웨이퍼 × die 격자 | 이웃 관계가 있는 2 차원 배열이다 |
| DOE result | 조건 × 응답 | 표본이 설계로 배치되어 분산이 정보가 아니다 |
| Marathon test | 시간 × 센서 | 부분공간이 시간에 따라 변한다 |

여기서 `n` 은 표본 수, 곧 행의 개수이고 `p` 는 변수 수, 곧 열의 개수이다. 웨이퍼 하나가 한 행이면 `n` 은 웨이퍼 장수이고, 계측 항목 하나가 한 열이면 `p` 는 항목 수이다. `p` 가 `n` 을 넘는 상태를 wide 라 부르며, 이때는 표본보다 추정할 값이 많아 표본 공분산이 온전한 계수를 갖지 못한다.

깨진 가정이 다르므로 같은 공장 안에서도 데이터마다 다른 가지를 쓴다. 계측값 표에는 고전 PCA 로 충분한 경우가 많고, 트레이스에는 구조를 지키는 가지가 필요하다.

## 3. Wafer Metrology

계측값은 웨이퍼 하나가 한 행이고 두께·저항·임계치수 같은 항목이 열이 되는 표다. 항목 수가 수십 개 수준이라 계산은 문제가 되지 않고, 대신 두 가지가 걸린다.

첫째는 단위다. 두께는 나노미터, 저항은 옴, 각도는 도로 적히므로 표준화하지 않으면 숫자가 큰 항목이 첫 성분을 가져간다. 상관행렬 기반이 기본이다.

둘째는 표본 수다. 로트 하나에 웨이퍼 25 장이면 성분을 안정적으로 추정하기에 부족하다. 고유값 축소나 성분 개수를 보수적으로 잡는 처리가 필요하며, 성분을 몇 개 쓸지는 scree 만으로 정하지 않고 교차검증으로 확인한다.

## 4. FDC Trace

트레이스는 웨이퍼 × 센서 × 시각의 3 차 구조다. 이것을 웨이퍼 × (센서·시각) 행렬로 펼치는 것이 가장 흔한 처리이고, 그 순간 `p` 가 수십만을 넘는다.

Table 3. Three ways to handle a trace

| Approach | Idea | Trade-off |
|---|---|---|
| Parameterize, then PCA | Reduce the trace to a few shape parameters and run PCA on them | Variation the parameters do not carry is lost |
| Functional PCA | Take the components while keeping the smoothness of the curve | The curves must be registered onto a common domain first |
| Tensor decomposition | Decompose the three axes without unfolding them | Harder to read and expensive to compute |

**펼치기 전에 파라미터화하는 쪽이 대개 낫다.** 시각마다 하나의 변수로 펼치면 이웃한 시각이 서로 무관한 변수가 되어, 트레이스가 곡선이라는 사실이 모형에서 사라진다. 그리고 step 길이가 웨이퍼마다 조금씩 달라 같은 열이 같은 시점을 가리키지 않게 되는 문제가 남는다.

Functional PCA 가 요구하는 상태는 **모든 곡선이 같은 시간축 위에서 같은 시점끼리 견줄 수 있는 상태**이다. 세 가지가 갖추어져야 한다. 첫째, 표본 시각이 곡선마다 달라도 공통 격자 위의 값으로 다시 뽑을 수 있어야 한다. 둘째, 시작과 끝의 기준점이 같아야 한다. Step 이 시작되는 순간을 0 으로 맞추지 않으면 같은 시각이 서로 다른 공정 단계를 가리킨다. 셋째, 남은 위상 차이를 registration 으로 없애야 한다. Recipe 가 같아도 step 길이가 웨이퍼마다 흔들리므로, 정점이나 전이 같은 표식을 맞추어 시간축을 늘이거나 줄인다.

**정렬하지 않으면 위상 차이가 진폭 변동으로 둔갑한다.** 같은 곡선이 조금 늦게 시작했을 뿐인데도 성분은 그것을 크기 변화로 읽고, 첫 성분이 공정 변동이 아니라 시각 어긋남을 담게 된다.

자기상관도 고려해야 한다. 시각을 그대로 변수로 두면 인접 변수가 거의 같은 값이라 첫 성분이 그 중복을 반영하게 되므로, 시차를 명시하는 Dynamic PCA 나 파라미터화가 그 중복을 줄인다.

## 5. Wafer Map

Die 마다의 값을 격자로 담은 자료는 이웃 관계가 정보다. 격자를 한 줄로 늘어놓고 PCA 를 걸면 인접한 die 가 서로 무관한 변수가 되어 공간 패턴을 잃는다. 2DPCA 처럼 행과 열 구조를 유지하는 가지나, 공간 상관을 명시하는 처리가 필요하다.

## 6. Multiple Tools And Chambers

여러 장비의 데이터를 한 표에 모으면 장비 사이의 차이가 가장 큰 분산이 되어 첫 성분을 차지한다. 그 성분은 공정 변동이 아니라 장비 식별자와 같으므로, 그대로 두면 뒤 성분이 밀려난다.

Table 4. Removing the tool effect

| Approach | Idea | Note |
|---|---|---|
| 장비별 정규화 | 장비 안에서 중심화하고 표준화한다 | 장비 사이의 실제 차이도 함께 지운다 |
| Contrastive PCA | 정상군 대비 특이한 방향을 찾는다 | 비교할 배경 데이터가 필요하다 |
| Multi-block | 장비 효과와 공정 효과를 따로 둔다 | 블록 구조를 사람이 선언한다 |

장비 차이를 지울지 남길지는 목적이 정한다. 수율 예측이면 지우는 편이 낫고, 장비 간 정합성 감시가 목적이면 그 성분이 바로 보려는 대상이다.

## 7. Drift Over Time

장기 운전 데이터에서는 부분공간 자체가 천천히 변한다. 고정된 성분으로 계속 사영하면 잔차가 서서히 커지는데, 이것을 모형 열화로 볼지 공정 변화로 볼지가 갈린다.

증분 계통을 쓰면 성분을 갱신해 잔차를 낮게 유지할 수 있지만, 그 갱신이 감시하려던 변화를 흡수해 버린다. 그래서 감시용 기준 성분은 고정하고, 갱신형 성분은 따로 두어 둘의 차이를 보는 구성을 쓴다.

## 8. Selection Map

데이터 조건에서 기법으로 가는 결정표다. 위에서부터 차례로 확인하고, 처음 걸리는 줄이 답이다.

Table 5. From condition to method

| Condition | Method | Direction | Section |
|---|---|---|---|
| 데이터가 곡선이고 매끄러움이 정보다 | Functional PCA 또는 파라미터화 후 PCA | Structure | §4 |
| 데이터가 세 축 이상의 구조다 | Tensor 또는 Multilinear PCA | Structure | §4 |
| 격자 위의 공간 패턴이 정보다 | 2DPCA 또는 공간 상관을 명시한 처리 | Structure | §5 |
| 이상치가 상시 존재한다 | Robust PCA 또는 L1-PCA | Robustness | §1 |
| 결측이 많다 | Probabilistic PCA | Robustness | §1 |
| `p` 가 `n` 을 크게 넘는다 | 고유값 축소와 보수적 성분 수 | Scale | §3 |
| 데이터가 메모리에 들어가지 않는다 | Randomized 또는 Incremental PCA | Scale | §1 |
| 데이터가 흘러 들어온다 | 증분 또는 스트리밍 계통 | Scale | §7 |
| 성분을 사람이 읽어야 한다 | Sparse PCA 또는 varimax 회전 | Structure | §1 |
| 구조가 선형이 아니다 | Kernel PCA | Nonlinear | §1 |
| 예측 목표가 정해져 있다 | PLS 또는 supervised 계통 | — | — |
| 장비 효과가 첫 성분을 차지한다 | 장비별 정규화 또는 contrastive | — | §6 |
| 위 어느 것도 아니다 | 표준화 후 고전 PCA | — | — |

```text
Is the column a trace?
├── yes → keep the curve structure?
│          ├── yes → functional PCA / parameterize        [4]
│          └── no  → unfold, then shrink eigenvalues      [3]
└── no  → is there a target variable?
           ├── yes → PLS / supervised branch
           └── no  → are outliers always present?
                      ├── yes → robust PCA
                      └── no  → standardize, then classical PCA
```

**결정표가 하나를 고르지 못하는 경우가 정상이다.** 트레이스이면서 이상치가 있고 장비도 여럿인 데이터가 흔하다. 그럴 때는 위에서부터 순서대로 적용하되, 한 번에 하나씩 넣고 그때마다 재현 오차와 성분 해석이 나아졌는지 확인한다. 두 가지를 동시에 넣으면 어느 쪽이 효과를 냈는지 알 수 없게 된다.

## Appendix A. Terminology

본문에서 정의하지 않고 쓴 용어를 정리한다.

- **Contrastive PCA** 는 배경 데이터에 비해 대상 데이터에서 특히 큰 분산을 갖는 방향을 찾는 방법이다.
- **DOE** 는 Design of Experiments 이고, 조건을 설계해 배치한 실험이다.
- **Dynamic PCA** 는 시차를 변수로 덧붙여 자기상관을 모형에 넣는 PCA 이다.
- **FDC** 는 Fault Detection and Classification 이고, 공정 장비의 센서 기록으로 이상을 찾는 체계이다.
- **FPCA** 는 Functional PCA 이고, 관측을 벡터가 아니라 곡선으로 보고 주성분을 구한다.
- **Incremental PCA** 는 데이터를 블록으로 나누어 성분을 갱신해 나가는 PCA 이다.
- **Kernel trick** 은 특징 공간의 좌표를 만들지 않고 내적만으로 계산을 마치는 기법이다.
- **Manifold learning** 은 고차원 데이터가 저차원 곡면 위에 있다고 보고 이웃 관계를 보존해 좌표를 찾는 방법이다.
- **Multi-block** 은 변수를 블록으로 나누고 블록 사이의 관계를 따로 모형화하는 방식이다.
- **PLS** 는 Partial Least Squares 이고, 입력과 목표의 공분산이 큰 방향을 찾는다.
- **Probabilistic PCA** 는 관측을 저차원 잠재변수의 선형 사상에 등방 잡음을 더한 것으로 보는 확률 모형이다.
- **Randomized PCA** 는 무작위 사영으로 부분공간을 좁힌 뒤 상위 성분을 근사하는 PCA 이다.
- **Registration** 은 곡선마다 어긋난 시간축을 늘이거나 줄여 같은 시점끼리 맞추는 처리이다.
- **Robust PCA** 는 행렬을 저계수 성분과 희소 성분의 합으로 분해해 이상치를 분리하는 방법이다.
- **Scree** 는 고유값을 크기순으로 그린 그림이고, 꺾이는 자리를 성분 개수로 삼는다.
- **Sparse PCA** 는 하중의 상당수를 0 으로 만들어 성분을 읽을 수 있게 하는 PCA 이다.
- **Tensor PCA** 는 세 축 이상의 배열을 펼치지 않은 채 축소하는 방법이다.
- **Trace** 는 한 대상을 시간에 따라 이어서 기록한 값의 열이다.
- **Varimax** 는 하중의 분산을 키워 해석을 쉽게 하는 직교 회전이다.
- **2DPCA** 는 이미지를 벡터로 펼치지 않고 행과 열 구조를 유지한 채 성분을 구하는 방법이다.
