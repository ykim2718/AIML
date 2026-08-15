# Semiconductor Sensor Trace Wide-to-Narrow Conversion
Rev. 2 | Created: 2026-07-29 | Updated: 2026-08-15 08:26 CDT

반도체 장비의 sensor 시계열 data는 [wafer, feature, trace] 구조의 3-way 배열이며, wafer 하나에 딸린 열의 수가 표본 수를 압도하는 wide data ($p \gg n$) 다. 이 문서는 정보를 최대한 유지하면서 이 wide data를 narrow data ($p \lesssim n$) 로 바꾸는 방법을 쉬운 방법부터 어려운 방법, 최신 방법 순으로 정리한다.

## 1. Problem Definition

### 1.1 Data Structure and Scale

대상 data를 tensor $X \in \mathbb{R}^{n \times s \times T}$ 로 둔다. 각 축의 규모는 다음과 같다.

- $n$ (wafer): 200 수준이다.
- $s$ (feature): sensor 수로, 200 수준이다.
- $T$ (trace): trace 길이로, $10^3$ 또는 $10^4$ 수준이다.

Wafer 하나를 평탄화하면 열의 수는 $p = s \times T = 2 \times 10^5 \sim 2 \times 10^6$ 이고, 표본 수는 $n = 200$ 에 불과하다. $p/n$ 이 $10^3 \sim 10^4$ 에 이르는 극단적 wide data이므로, 회귀·분류·이상탐지 어느 목적이든 먼저 wafer당 열 수를 $n$ 이하로 줄인 narrow table을 만들어야 한다.

### 1.2 Definition of Narrow Data

이 문서에서 narrow data는 wafer당 열 수 $k$ 가 $k \lesssim n$ 인 (wafer × k) 표를 뜻한다. Data 형식 관점의 wide-to-long 변환 (melt), 즉 (wafer, sensor, time, value) 의 4열 long format으로 재배열하는 것은 정보 손실이 전혀 없지만 열 수를 줄이는 것이 아니므로 이 문서의 대상이 아니다. 여기서 다루는 것은 정보를 압축하여 차원을 줄이는 변환이다.

### 1.3 Information Preservation Criteria

"정보를 최대한 유지한다"는 목적에 따라 다르게 측정한다.

- 복원 관점: 축소된 표현에서 원 trace를 되살렸을 때의 reconstruction error 또는 설명 분산 비율로 잰다.
- 예측 관점: 축소된 표현으로 downstream task (metrology 예측, 불량 분류, 이상탐지) 를 수행한 성능으로 잰다.
- 해석 관점: 각 열이 물리적 의미 (ramp 속도, 정착 지연, 총 dose) 를 유지하는지로 잰다.

세 기준은 서로 상충할 수 있다. 예를 들어 deep embedding은 예측 관점에서 우수하지만 해석 관점에서 불리하다.

## 2. Method Map

Table 1. Wide-to-narrow methods ordered by difficulty

| Level | Method group | Output per wafer | Supervision | Interpretability |
|---|---|---|---|---|
| 1 | Summary statistics, segmentation, downsampling | sensor당 수 개 통계량 | 불필요 | 높음 |
| 2 | Linear projection (PCA, PLS, FPCA, wavelet) | 주성분 점수 수십 개 | PCA 계열은 불필요, PLS는 필요 | 중간 |
| 3 | Automated feature library + selection | 선별된 특징 수십~수백 개 | 선택 단계에서 활용 가능 | 중간~높음 |
| 4 | Tensor decomposition (CP, Tucker) | wafer-mode factor 수십 개 | 불필요 | 중간 |
| 5 | Random convolution kernel (ROCKET 계열) | kernel 특징 수천 개 → 사후 축소 | 불필요 | 낮음 |
| 6 | Deep representation learning (autoencoder, contrastive, masked) | embedding 수십~수백 차원 | 불필요 (self-supervised) | 낮음 |
| 7 | Foundation model embedding | pretrained embedding 수백 차원 | 불필요 (zero-shot) | 낮음 |

모든 level에 공통되는 전처리가 있다. Trace 길이가 wafer마다 다르면 공통 시간축으로 resampling하거나, recipe step 경계 기준으로 자르거나, DTW로 정렬한 뒤에 변환을 적용한다. Sensor별 단위 차이는 sensor 단위 표준화로 제거한다.

## 3. Summary Statistics and Segmentation

가장 쉽고, 현업 FDC 시스템의 표준 관행이다.

### 3.1 Per-Sensor Summary Statistics

각 (wafer, sensor) trace를 평균, 표준편차, 최소, 최대, 범위, 기울기, AUC 변형 같은 통계량 몇 개로 요약한다. Sensor 200개 × 통계량 10개면 wafer당 2,000열이 되어 아직 $n$ 을 넘으므로, 분산 filter나 상관 filter, 모형 기반 선택으로 한 번 더 줄인다.

- 장점: 계산이 싸고, 각 열의 물리적 의미가 명확하며, 이상 원인 추적이 쉽다.
- 단점: trace의 형상 (파형, 순서, 국소 이벤트) 정보가 사라진다. 같은 평균·분산을 갖는 전혀 다른 파형을 구분하지 못한다.

### 3.2 Recipe Step Segmentation

Trace를 recipe step 경계로 잘라 step별로 통계량을 낸다. 형상 정보의 상당 부분이 step 구조에 있으므로, 전체 구간 통계보다 정보 유지가 훨씬 좋다. Step 경계는 장비 log의 step 번호를 쓰거나, 변화점 탐지로 추정한다. 단점은 열 수가 step 수만큼 배가되어 선택 단계의 부담이 커진다는 점이다.

### 3.3 Downsampling and PAA

Trace를 등간격 구간으로 나눠 구간 평균만 남기는 PAA (Piecewise Aggregate Approximation) 는 길이 $T$ 를 임의의 $K$ 로 줄이는 가장 단순한 형상 보존 압축이다. $K$ 를 크게 잡으면 정보 유지가 좋아지지만 열 수가 늘어나는 trade-off가 있으며, 보통 PAA 결과를 다시 level 2 이상의 방법에 입력하는 중간 단계로 쓴다.

## 4. Linear Projection

### 4.1 Unfolding and PCA

Tensor를 행렬로 펼친 뒤 (wafer-wise unfolding: $n \times sT$ 행렬) PCA를 적용해 wafer당 주성분 점수 수십 개만 남긴다. Batch process 감시에서 MPCA (Multiway PCA) 로 불리는 고전적 방법이며, 설명 분산 비율로 정보 유지량을 정량화할 수 있다는 것이 장점이다.

- $n \lt sT$ 이므로 주성분은 최대 $n - 1$ 개다. 극단적 wide data에서도 주성분은 $n \times n$ 크기의 행렬 분해로 저렴하게 계산할 수 있다.
- 단점은 성분이 sensor와 시간에 걸쳐 섞여 있어 해석이 어렵고, 평균 중심의 선형 구조만 잡는다는 점이다.

### 4.2 Supervised Projection with PLS

Metrology 값 같은 응답변수 $y$ 가 있으면 PLS로 $y$ 와 공분산이 큰 방향만 남긴다. 같은 차원 수라면 PCA보다 예측 관점의 정보 유지가 좋다. 대신 응답변수가 바뀌면 표현을 다시 만들어야 하고, $n = 200$ 에서 성분 수를 교차검증으로 엄격히 통제하지 않으면 과적합한다.

### 4.3 Basis Expansion and FPCA

Trace를 함수 $x(t)$ 로 보고 B-spline, Fourier, wavelet 기저의 계수로 표현한 뒤, 계수 공간에서 PCA를 하는 것이 FPCA (Functional PCA) 다. 매끄러운 trace에는 소수의 기저 계수로 충분하여 압축 효율이 좋고, 도함수 정보 (ramp 속도) 를 자연스럽게 다룬다. Wavelet 계수는 국소 이벤트 (spike, glitch) 를 보존한다는 점에서 Fourier보다 FDC에 적합하다. 길이가 다르거나 위상이 밀린 trace는 landmark registration으로 정렬한 뒤 적용한다.

## 5. Automated Feature Library

### 5.1 Feature Libraries

tsfresh는 trace당 수백 개의 통계·주파수·엔트로피 특징을 자동 추출하고 가설검정 기반 선별까지 해 준다. catch22는 방대한 특징 후보에서 중복을 제거해 추린 22개의 정예 특징으로, trace당 22개 × sensor 200개 = 4,400열이 나온다. 수작업 통계량 (level 1) 보다 넓은 특징 공간을 체계적으로 탐색한다는 점이 가치다.

### 5.2 Feature Selection

Library 출력은 여전히 $p \gt n$ 이므로 선택이 필수다. $n = 200$ 에서는 단일 선택 결과가 불안정하므로, resampling을 반복하며 자주 뽑히는 특징만 남기는 stability selection이나 mRMR, lasso를 쓴다. 선택까지 끝나면 각 열이 "어느 sensor의 어떤 특징"인지 명시적이어서 해석 관점의 정보 유지가 좋다.

## 6. Tensor Decomposition

펼치지 않고 3-way 구조를 그대로 두고 분해한다. Unfolding PCA가 버리는 "같은 sensor의 시간 pattern은 wafer마다 공유된다"는 구조적 정보를 활용하므로, 같은 차원 수에서 더 많은 정보를 유지한다.

- CP (CANDECOMP/PARAFAC) 분해는 tensor를 rank-1 tensor $R$ 개의 합 $X \approx \sum_{r=1}^{R} a_r \circ b_r \circ c_r$ 로 근사한다. Wafer-mode factor $A \in \mathbb{R}^{n \times R}$ 의 각 행이 wafer의 narrow 표현이 되고, sensor-mode·time-mode factor가 "어떤 sensor 조합의 어떤 시간 pattern"인지 알려 주므로 해석도 가능하다.
- Tucker 분해는 mode마다 다른 rank를 허용해 CP보다 유연하며, time mode의 rank를 크게 잡아 형상 정보를 더 보존할 수 있다.
- 각 mode의 요인 수는 재구성 오차가 더 이상 크게 줄지 않는 지점으로 고른다. 결측이 있어도 분해가 가능하다는 실무적 장점이 있다.

$200 \times 200 \times 10^4$ 규모는 표준 라이브러리 (tensorly 등) 로 무리 없이 분해된다.

## 7. Random Convolution Kernels

ROCKET은 무작위로 생성한 1차원 convolution kernel 수천 개를 trace에 통과시키고 kernel당 max와 PPV (Proportion of Positive Values) 만 남긴다. MiniROCKET·MultiROCKET은 이를 결정적이고 더 빠르게 만든 후속이다.

- 학습이 없어서 $n = 200$ 의 소표본에서도 과적합 없이 쓸 수 있고, 시계열 분류 benchmark에서 deep learning과 대등한 성능을 낸다.
- 출력이 wafer당 수천~수만 열로 오히려 넓어지므로, ridge 회귀처럼 wide에 강한 모형을 바로 붙이거나 PCA로 한 번 더 눌러 narrow table을 만든다.
- Kernel이 무작위라 개별 특징의 해석은 불가능하다. Sensor 200개에는 sensor별로 독립 적용한 뒤 합치거나 다변량 확장을 쓴다.

## 8. Deep Representation Learning

Encoder로 trace를 저차원 embedding으로 압축하는 방법이며, 비선형 구조까지 잡는다는 것이 선형 계열과의 차이다.

- Autoencoder: 1D-CNN 또는 LSTM encoder-decoder를 reconstruction 손실로 학습하고 병목의 embedding을 쓴다. VAE는 잠재 공간을 정칙화해 보간·생성이 가능한 표현을 준다.
- Contrastive learning: TS2Vec, TS-TCC처럼 같은 trace의 두 augmentation을 가깝게, 다른 trace를 멀게 학습한다. Label 없이 downstream 성능이 좋은 표현을 얻는다.
- Masked modeling: trace의 일부 구간을 가리고 복원하도록 학습한다. Transformer 기반 시계열 모형의 표준 pretraining 방식이 되었다.

주의할 점은 표본 규모다. Wafer 단위로는 $n = 200$ 뿐이라 wafer 하나를 입력으로 하는 대형 모형은 학습이 불가능하다. 실무 요령은 (wafer, sensor) trace 하나를 표본으로 삼아 $200 \times 200 = 4 \times 10^4$ 개의 단변량 trace로 encoder를 학습하고, wafer 표현은 sensor embedding들의 결합 (연결 또는 pooling) 으로 만드는 것이다. 그래도 augmentation과 강한 정칙화가 필요하며, 해석은 포기해야 한다.

## 9. Foundation Model Embeddings

2024년 이후의 최신 흐름은 대규모 이종 시계열로 pretraining된 foundation model에서 zero-shot embedding을 뽑는 것이다.

- MOMENT는 masked reconstruction으로 pretraining된 범용 시계열 모형으로, 학습 없이 trace를 넣으면 고정 차원 embedding을 돌려준다.
- Chronos, TimesFM, Moirai는 예측용으로 pretraining되었지만 encoder 출력을 표현으로 쓸 수 있다. TOTEM은 VQ-VAE 방식으로 trace를 이산 token으로 양자화한다.
- LETS-C처럼 text embedding 모형을 시계열에 전용하는 시도, channel 설명문을 함께 넣어 다변량을 다루는 시도도 있다.

소표본 문제를 pretraining이 대신 흡수해 주므로 $n = 200$ 상황과 잘 맞는다. 다만 대부분 단변량·고정 patch 길이 전제라 sensor별로 embedding을 뽑아 결합해야 하고, 반도체 trace처럼 계단형 setpoint 신호가 pretraining 분포와 멀면 품질이 떨어질 수 있으므로 downstream 성능으로 반드시 검증한다. Embedding 차원 (수백) × sensor 200개는 다시 wide가 되므로 pooling이나 PCA로 마무리한다.

## 10. Method Selection for This Scale

$n = 200$, $s = 200$, $T = 10^3 \sim 10^4$ 규모에 대한 권고는 다음과 같다.

1. Baseline은 recipe step segmentation + step별 summary statistics + stability selection으로 잡는다. 싸고 해석되며, 이후 모든 방법의 비교 기준이 된다.
2. 형상 정보가 성능을 좌우하면 Tucker 분해와 MiniROCKET을 추가한다. 둘 다 학습 부담 없이 소표본에서 안정적이다.
3. 응답변수가 고정되어 있으면 PLS 성분을 병행해 예측 관점의 상한을 확인한다.
4. 여력이 있으면 foundation model embedding을 뽑아 baseline과 성능을 비교한다. 이기면 채택하되, 해석이 필요한 용도에는 baseline을 유지한다.
5. 어느 방법이든 평가에는 wafer 단위 교차검증을 쓴다. 표준화·선택·투영의 모든 적합은 훈련 fold 안에서만 수행해 leakage를 막는다.

```python
# Python — baseline pipeline sketch
import numpy as np

X = np.load("traces.npy")            # shape: (n_wafer, n_sensor, T)
steps = segment_by_recipe_step(X)    # list of (n_wafer, n_sensor, T_step)
feats = np.concatenate(
    [summary_stats(s) for s in steps], axis=1
)                                    # (n_wafer, n_sensor * n_step * n_stat)
selected = stability_select(feats, y, n_resample=100)   # fit inside cross-validation folds
```

## 11. References

- Nomikos, P. and MacGregor, J. F., "Monitoring Batch Processes Using Multiway Principal Component Analysis", AIChE Journal, 1994.
- Ramsay, J. O. and Silverman, B. W., Functional Data Analysis, Springer, 2005.
- Christ, M. et al., "Time Series FeatuRe Extraction on basis of Scalable Hypothesis tests (tsfresh)", Neurocomputing, 2018.
- Lubba, C. H. et al., "catch22: CAnonical Time-series CHaracteristics", Data Mining and Knowledge Discovery, 2019.
- Kolda, T. G. and Bader, B. W., "Tensor Decompositions and Applications", SIAM Review, 2009.
- Dempster, A. et al., "ROCKET: Exceptionally Fast and Accurate Time Series Classification Using Random Convolutional Kernels", Data Mining and Knowledge Discovery, 2020.
- Yue, Z. et al., "TS2Vec: Towards Universal Representation of Time Series", AAAI, 2022.
- Goswami, M. et al., "MOMENT: A Family of Open Time-series Foundation Models", ICML, 2024. <https://arxiv.org/abs/2402.03885>
- Kaufman, R. et al., "LETS-C: Leveraging Text Embedding for Time Series Classification", 2024. <https://arxiv.org/pdf/2407.06533>
- "A Survey of Deep Learning and Foundation Models for Time Series Forecasting", 2024. <https://arxiv.org/pdf/2401.13912>
- "Non-Stationarity in the Embedding Space of Time Series Foundation Models", 2026. <https://arxiv.org/pdf/2604.16428>

---

## Appendix A. Terminology

- 1D-CNN: 1-Dimensional Convolutional Neural Network. 시간축 방향 convolution으로 국소 pattern을 학습하는 신경망.
- AUC: Area Under the Curve. Trace를 시간에 대해 적분한 값과 그 변형들.
- CP (CANDECOMP/PARAFAC): tensor를 rank-1 tensor들의 합으로 근사하는 분해.
- Downstream task: 축소된 표현을 입력으로 수행하는 후속 과제. 예측, 분류, 이상탐지가 해당한다.
- DTW: Dynamic Time Warping. 길이와 위상이 다른 두 시계열을 비선형 시간축 왜곡으로 정렬하는 방법.
- FDC: Fault Detection and Classification. 장비 sensor trace로 공정 이상을 탐지·분류하는 반도체 제조 시스템.
- FDR: False Discovery Rate. 다중검정에서 허위 발견의 기대 비율. Benjamini-Yekutieli 절차가 이를 통제한다.
- Foundation model: 대규모 이종 자료로 pretraining되어 추가 학습 없이 여러 과제에 전용되는 범용 모형.
- FPCA: Functional PCA. Trace를 함수로 보고 기저 전개 후 수행하는 PCA.
- FRESH: FeatuRe Extraction based on Scalable Hypothesis tests. tsfresh의 가설검정 기반 특징 선별 절차.
- GBM: Gradient Boosting Machine. 얕은 tree를 순차적으로 더해 가는 boosting 예측 모형. LightGBM이 대표 구현이다.
- Landmark registration: 함수 자료 분석에서 특징적 시점 (peak, 변곡점) 을 기준으로 시간축을 정렬하는 방법.
- Lasso: L1 정칙화로 일부 계수를 0으로 만들어 특징 선택을 겸하는 선형 회귀.
- LSTM: Long Short-Term Memory. 장기 의존성을 다루는 순환 신경망 구조.
- MPCA: Multiway PCA. Tensor를 행렬로 펼친 뒤 적용하는 PCA. Batch process 감시의 고전적 방법.
- mRMR: minimum Redundancy Maximum Relevance. 응답과의 관련성은 크고 특징 간 중복은 작게 뽑는 특징 선택 기준.
- PAA: Piecewise Aggregate Approximation. 시계열을 등간격 구간 평균으로 압축하는 방법.
- PCA: Principal Component Analysis. 분산이 큰 직교 방향으로 투영하는 선형 차원 축소.
- PLS: Partial Least Squares. 응답변수와의 공분산이 큰 방향으로 투영하는 지도 학습형 차원 축소.
- PPV: Proportion of Positive Values. Convolution 출력에서 양수의 비율을 취하는 ROCKET의 pooling 통계량.
- Ridge 회귀: L2 정칙화로 wide data에서도 안정적으로 적합되는 선형 회귀.
- ROCKET: RandOm Convolutional KErnel Transform. 무작위 convolution kernel로 시계열 특징을 만드는 방법.
- Stability selection: resampling을 반복하며 선택 빈도가 높은 특징만 남기는 안정화된 특징 선택 절차.
- TSFM: Time Series Foundation Model. 시계열용 foundation model.
- Tucker 분해: mode별로 다른 rank를 허용하는 tensor 분해. core tensor와 mode별 factor 행렬로 구성된다.
- VAE: Variational Autoencoder. 잠재 공간에 확률적 정칙화를 가한 autoencoder.
- VM: Virtual Metrology. 장비 sensor data로 metrology 측정값을 예측하는 기술.
- VQ-VAE: Vector Quantized VAE. 잠재 표현을 이산 codebook token으로 양자화하는 autoencoder.
- Zero-shot: 추가 학습 없이 pretrained 모형을 그대로 새 자료에 적용하는 사용 방식.

## Appendix B. Further Development

본문의 방법들을 실제로 조합할 때의 심화 지침이다. 라벨 (metrology 측정값) 은 sampling 측정이라 희소하므로, 어느 계열이 정확한지는 라벨된 wafer 수 $n$ 이 결정한다. $n \lesssim 1000$ 이면 고전 계열이, $n \gtrsim 5000$ 이면 encoder 계열이 유리하며, 대부분의 fab 사례는 전자에 속한다. 경험적으로 $n$ 이 수백 규모일 때는 고전 계열이 deep learning 계열을 이긴다. 목적함수가 train 성능을 보상하도록 잘못 설계된 hyperparameter 탐색이 과적합을 유도하는 함정은 deep encoder에서 훨씬 심하게 나타난다.

#### FPCA-PLS Pipeline

$n \lesssim 1000$ 에서의 첫 번째 권장 조합이다.

- 각 sensor trace를 B-spline 기저 (knot 8~12개) 로 평활화해 함수형 자료로 취급한다.
- Sensor별로 FPC (Functional Principal Component) score 상위 3~4개를 추출한다. 보통 누적분산 95% 이상을 기준으로 삼는다.
- Trace 하나가 score 3~4개로 압축되며, 펼친 PCA와 달리 곡선의 형상 자체를 보존한다. 노이즈에 강하고, 첫 두 FPC가 "전체 level 이동 / 기울기 변화"처럼 물리적으로 해석된다.
- 이후 PLS로 응답변수 — 두께 (THK) 나 면저항 (Rs) 의 uniformity — 를 향해 지도적으로 축약한다. 반도체 VM에서 오랫동안 표준인 이유는 $p \gg n$ 상황에서 PCA와 달리 응답변수와의 공분산을 최대화하기 때문이다.
- Trace에 위상 변동 (step 시작 시점의 흔들림) 이 있으면 FPCA 전에 curve registration 또는 DTW 정렬을 반드시 넣는다. 정렬 없이 FPCA를 하면 진폭 정보와 위상 정보가 뒤섞여 score가 오염된다.

#### Feature Library Pipeline

$n \lesssim 1000$ 에서의 두 번째 권장 조합이다.

- catch22는 22개의 canonical feature로 압축이 이미 내장되어 있어, 수백 개를 쏟아내는 tsfresh보다 이 규모에 적합하다.
- FRESH 절차로 유의한 특징만 선별하되 FDR을 Benjamini-Yekutieli로 통제해, sensor 200개에 대한 다중검정을 통계적으로 관리한다.
- 선별된 특징은 GBM (LightGBM) 에 입력한다.

#### Channel-Independent Encoder Embedding

$n \gtrsim 5000$ 에서의 구성이다.

- 1D-CNN (InceptionTime, dilated convolution) 또는 PatchTST encoder를 sensor 200개에 가중치 공유로 적용해 sensor당 16차원 embedding을 만든다.
- Sensor축을 attention pooling으로 집계해 wafer당 64~128차원 vector 하나로 만든다.
- Channel별 독립 parameter를 쓰면 parameter 수가 200배로 늘어 즉시 과적합한다. 가중치 공유가 wide 구조를 막는 핵심 장치다.
- 라벨이 부족하면 미측정 wafer의 trace로 TS2Vec이나 SimMTM 자기지도 사전학습을 하고, 소량의 라벨로 예측 head만 fine-tuning한다.

#### Recent Methods

Table 2. Recent methods (2024~2026) and their fit to this scale

| Method | Core idea | Fit |
|---|---|---|
| PatchTST | 시간축을 patch로 묶어 token화하고 channel을 독립으로 다룬다. | $T = 10^3 \sim 10^4$ 인 이 규모에서는 patch화가 sequence 길이를 유효하게 줄여 준다. 수백 점 이하의 짧은 trace에서는 이득이 제한적이다. |
| TOTEM | Trace를 VQ-VAE로 이산 token으로 양자화하고 channel 간 codebook을 공유한다. | Channel 200개가 codebook 하나를 공유해 차원 폭증이 없어 유망하다. |
| TSFM frozen embedding | Chronos-2나 MOMENT의 encoder를 동결하고 mean pooling으로 embedding을 뽑는다. | 사전학습 corpus에 fab trace가 없어 domain gap이 크다. PCA로 32차원 이하로 줄여 baseline과 병렬 비교하는 용도만 권한다. |
| iTransformer | 시간축이 아니라 변수 (sensor) 를 token으로 취급해 200개 token을 만든다. | Sensor 간 상호작용을 직접 modeling하며 구조적으로 wide 구조를 회피한다. |
| Graph attention VM | (step, parameter) 쌍을 그래프의 마디로 두고 방향성 attention을 준다. | 2026년 박막 증착 두께 예측에 실증되었다. 공정 지식을 그래프 구조로 그대로 투입할 수 있어 활용도가 높다. |
| Path signature | 경로적분의 대수적 불변량을 level 2~3에서 절단해 쓴다. | 가변 길이 trace를 수학적으로 원리적인 고정 길이 vector로 만든다. log-signature로 더 압축할 수 있다. |

TSFM embedding 공간은 입력 분포 변화에 대해 비정상성을 보인다는 보고가 있다. Chamber drift가 있는 환경에서는 embedding 자체가 시간에 따라 이동하므로, 적용할 때 embedding 공간의 shift 모니터링을 병행해야 한다.
