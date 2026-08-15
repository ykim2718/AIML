# Wide-to-Narrow Practice
Rev. 2 | Created: 2026-07-29 | Updated: 2026-08-15 11:20 CDT

반도체 장비 sensor trace의 wide data를 narrow data로 바꾸는 방법 가운데 세 가지 — 가중치 공유 encoding, sparsity 기반 sensor 선택, PLS 지도적 축약 — 의 구현 세부를 다룬다. 각 방법이 왜 동작하는지, 구현에서 무엇이 필수인지, 어떤 함정이 있는지를 적는다.

## 1. Scope and Premise

대상 data는 wafer × sensor × trace의 3-way tensor다. Wafer 수 $n$ 은 수백 ($10^2 \sim 10^3$), sensor 수 $s$ 는 200 수준, trace 길이 $T$ 는 $10^3 \sim 10^4$ 수준으로 둔다. 응답변수는 metrology 측정값으로, 두께 (THK) 와 면저항 (Rs) 을 예시로 쓴다. 목적은 세 가지다.

- VM: trace로 metrology 값을 예측한다.
- Sensor 감축: 수집·유지할 sensor 부분집합을 결정한다.
- 모니터링: chamber 상태의 drift를 감시한다.

운영 data에는 chamber 간 계통 차이, recipe step 구조, PM 주기에 따른 장기 drift와 계단형 shift가 섞여 있다고 전제한다. 이 전제는 세 방법 모두의 검증 설계를 좌우한다.

## 2. Weight-Shared Channel-Independent Encoding

Deep encoder로 trace를 압축할 때, sensor마다 별도 encoder를 두지 않고 encoder 하나를 모든 sensor에 공유하는 구성이다.

### 2.1 Parameter Arithmetic

Encoder 하나의 parameter 수를 $P$ 라 하면 두 구성은 다음과 같이 갈린다.

Table 1. Independent versus weight-shared encoders

| Design | Encoder parameters | Samples the encoder sees |
|---|---|---|
| Sensor별 독립 encoder | $P \times s$ | sensor당 $n$ 개 |
| 가중치 공유 encoder | $P$ | $n \times s$ 개 |

두 번째 열이 본질이다. Wafer가 수백 장뿐이어도, (wafer, sensor) 쌍을 표본으로 삼으면 encoder는 수만 개의 1차원 sequence를 학습 data로 받는다. Sensor 200개를 하나의 다변량 입력으로 붙여 넣으면 표본이 $n$ 개에 머물지만, sensor축을 batch 차원으로 접으면 $n \times s$ 개가 된다. 이것이 $n$ 이 작은 fab data에서 deep encoder를 성립시키는 유일한 장치다.

### 2.2 Why One Encoder Suffices

RF power, chamber 압력, 가스 유량, heater 온도는 물리량이 전부 다르지만 trace의 문법은 공유한다. 대부분의 sensor가 ramp-up, overshoot, settle, plateau, ramp-down이라는 동일한 형태소를 갖는다. Convolution kernel이 학습하는 것은 "압력이란 무엇인가"가 아니라 "기울기·overshoot·정착 진동·plateau drift를 어떻게 검출하는가"이므로, 같은 kernel을 sensor 전체에 재사용할 수 있다. 자연어 처리에서 tokenizer 하나를 모든 문서에 적용하는 것과 같은 구조다.

### 2.3 Implementation

Tensor 재배치가 구현의 전부다. Embedding 차원을 $d$ 로 둔다 (예: 16).

```python
# Python — weight-shared encoding sketch (PyTorch style)
B, C, T = x.shape                          # (batch, sensors, trace length)

# 1. Per-channel normalization (RevIN / instance norm) — required
mu = x.mean(dim=2, keepdim=True)           # (B, C, 1)
sd = x.std(dim=2, keepdim=True) + 1e-5
xn = (x - mu) / sd                         # (B, C, T)

# 2. Fold channels into the batch axis — the encoder sees univariate series
z = xn.reshape(B * C, 1, T)
z = encoder(z)                             # shared weights, output (B*C, d)
z = z.reshape(B, C, -1)                    # (B, C, d)

# 3. Restore sensor identity and re-inject the level statistics
z = z + channel_emb                        # learnable (C, d), broadcast over B
z = torch.cat([z, mu, sd], dim=2)          # mu, sd are (B, C, 1) -> (B, C, d + 2)

# 4. Aggregate over the sensor axis into one wafer vector
w = torch.softmax(attn(z), dim=1)          # (B, C, 1)
wafer_vec = (w * z).sum(dim=1)             # (B, d + 2)
```

### 2.4 Stepwise Requirements

- Channel별 정규화 (단계 1): 압력 (mTorr, $10^0$ 규모) 과 RF power (W, $10^3$ 규모) 를 같은 encoder에 넣으려면 규모를 반드시 제거해야 한다. 제거하지 않으면 kernel이 규모 큰 sensor에만 반응한다. 단, 제거한 `mu` 와 `sd` 는 버리지 말고 단계 3에서 다시 붙인다. VM에서 "평균 압력이 얼마였는가"는 두께를 결정하는 1차 정보이기 때문이다. 정규화로 형상 정보만 encoding하고, level 정보는 scalar로 우회 전달하는 구조다.
- Channel embedding (단계 3): batch 차원으로 접는 순간 "몇 번 sensor였는지"가 사라진다. 학습 가능한 $(s, d)$ embedding을 더해 정체성을 복원한다. Sensor를 기능 그룹으로 묶는 taxonomy가 있으면 무작위 초기화 대신 그룹 one-hot을 초기값으로 주는데, 수렴이 빨라지고 해석이 붙는다.
- 집계 (단계 4): channel 독립 encoding은 sensor 간 상호작용 — 예를 들어 RF reflected power 상승과 chamber 압력 이상의 동시 발생 — 을 볼 수 없다. 이 상호작용은 전부 집계 층에서 회복해야 한다. 단순 mean pooling은 정보 손실이 크고, attention pooling이나 taxonomy 그룹별 pooling 후 그룹 간 소형 MLP가 낫다.

### 2.5 Limitations

Encoder의 parameter는 줄지만, 예측 head가 보는 표본은 여전히 wafer 수 $n$ 이다. Head는 반드시 얕게 (선형 또는 은닉층 1개) 두고 dropout과 weight decay를 강하게 건다. $n$ 이 수백 규모이면 이 구성이 FPCA와 PLS의 조합을 이기지 못하는 경우가 흔하므로, baseline 대비 test 성능 개선이 없으면 폐기한다는 기준을 미리 정해 둔다.

## 3. Sparse Sensor Selection

Sensor 부분집합을 고르는 두 기법은 목적이 다르므로 혼용하면 안 된다.

Table 2. Sparse PCA versus group lasso

| Aspect | Sparse PCA | Group lasso |
|---|---|---|
| Supervision | 비지도 ($X$ 만) | 지도 ($X$, $y$) |
| Objective | 소수 sensor로 $X$ 의 분산 설명 | 소수 sensor로 $y$ 예측 |
| Output | 희소 loading vector | 희소 회귀계수 |
| Use | Sensor 감축 결정, 모니터링 | VM 특징 선택 |

### 3.1 Group Lasso

Sensor별 특징들을 그룹으로 묶고, 그룹 단위로 희소성을 거는 회귀다. 그룹 $g$ 의 특징 수를 $p_g$, 그 계수 vector를 $\beta_g$ 로 두면 목적함수는 다음과 같다.

$$\min_{\beta}\ \frac{1}{2n}\lVert y - X\beta \rVert_2^2 + \lambda \sum_{g=1}^{G} \sqrt{p_g}\,\lVert \beta_g \rVert_2$$

핵심은 그룹 계수 vector에 제곱 없는 L2 norm을 씌우고 그것들을 L1처럼 합산한다는 점이다. L2 norm은 원점에서 미분 불가능하므로 $\lVert \beta_g \rVert_2 = 0$, 즉 그룹 전체가 통째로 0이 되는 해가 발생한다. 반면 그룹 내부에서는 미분 가능하므로 선택된 그룹 안의 계수는 모두 살아남는다. 전부 아니면 전무다. $\sqrt{p_g}$ 는 그룹 크기 보정으로, 이것이 없으면 특징을 많이 가진 sensor가 자동으로 유리해진다.

그룹 정의는 두 가지가 가능하다.

- 정의 A (sensor 단위): 그룹 하나가 sensor 하나의 모든 특징 (FPC score, step별 AUC, step별 기울기 등 10개 내외) 이 되고, $G = s$ 다. "이 sensor를 쓸 것인가"를 결정한다.
- 정의 B (기능 그룹 단위): 그룹 하나가 taxonomy 그룹 하나에 속한 모든 sensor의 모든 특징이 되고, $G$ 는 그룹 수 (10~20개 수준) 다. "이 계열 전체가 응답과 무관한가"를 결정한다.

정의 A를 권장한다. Sensor 감축은 물리적으로 sensor 단위 (수집 항목 제거) 로 일어나고, 정의 B는 그룹이 너무 커서 하나만 선택돼도 차원 감축 효과가 거의 없다.

### 3.2 Sparse Group Lasso

혼합 비율 $\alpha$ 를 도입해 그룹 간 희소성과 그룹 내 희소성을 동시에 건다.

$$\min_{\beta}\ \frac{1}{2n}\lVert y - X\beta \rVert_2^2 + \lambda \left[ (1-\alpha) \sum_{g=1}^{G} \sqrt{p_g}\,\lVert \beta_g \rVert_2 + \alpha \lVert \beta \rVert_1 \right]$$

실무에서 유용한 이유는 "RF power sensor는 쓰되 그 sensor의 특징 중 일부만 쓴다"는 결정이 가능해지기 때문이다. $\alpha$ 는 0.05~0.3 범위를 탐색한다.

필수 전처리는 세 가지다.

- 모든 특징을 표준화한다. Penalty가 규모에 직접 반응하므로 표준화 없이는 결과가 무의미하다.
- 그룹을 직교화한다. 그룹별 QR 분해로 whitening해야 이론적 보장이 성립하며, 생략하면 그룹 내 공선성이 심한 sensor가 부당하게 penalty를 덜 받는다.
- 표준화 통계량과 $\lambda$ 선택 모두 train fold 안에서만 적합한다.

### 3.3 Stability Selection

Lasso 계열의 변수 선택은 표본이 작을 때 극도로 불안정하다. Wafer 수십 장만 바뀌어도 선택된 sensor 집합이 절반 이상 달라진다. 게다가 sensor 200개에는 RF forward와 reflected, 인접 지점의 온도처럼 상관이 0.95를 넘는 쌍이 다수 존재하고, lasso는 그중 하나를 임의로 고른다. 따라서 resampling으로 선택 빈도를 세는 stability selection이 사실상 필수다.

```python
# Python — stability selection over bootstrap resamples
import numpy as np
from sklearn.utils import resample

sel_count = np.zeros(n_groups)
for b in range(200):                       # 200 bootstrap rounds
    Xb, yb = resample(X, y, n_samples=int(0.6 * len(y)))
    beta = sparse_group_lasso(Xb, yb, lam, alpha)
    sel_count += (group_norm(beta) > 0)
stability = sel_count / 200
final_sensors = np.where(stability >= 0.6)[0]   # threshold 0.6 to 0.8
```

선택 빈도 60% 미만인 sensor는 버린다. 단일 적합 결과를 "이 sensor가 중요하다"고 보고하면 재현되지 않는다.

### 3.4 Sparse PCA

일반 PCA의 주성분은 sensor 전부에 0이 아닌 loading을 가지므로, "첫 주성분이 중요하다"는 결론이 나와도 sensor를 하나도 뺄 수 없다. Sparse PCA는 loading에 elastic-net penalty를 걸어 각 성분이 소수 sensor만 쓰게 만든다. 성분 행렬을 $A$, 희소 loading 행렬을 $B$ (열 vector $b_j$) 로 두면 다음과 같다.

$$\min_{A, B}\ \lVert X - X B A^{\mathsf T} \rVert_F^2 + \lambda_2 \lVert B \rVert_F^2 + \sum_{j} \lambda_{1,j} \lVert b_j \rVert_1 \quad \mathrm{s.t.}\ A^{\mathsf T} A = I$$

함정이 하나 있다. 성분별로 희소해도 성분들의 합집합은 여전히 클 수 있다. 성분 5개가 각각 다른 sensor 20개를 쓰면 총 100개가 필요하다. Sensor 대수 자체를 줄이는 것이 목적이면, 성분 전체에 걸친 전역 sensor 개수 제약을 거는 변형을 써야 한다.

### 3.5 Selection Workflow

1. Sparse PCA (비지도) 로 chamber 상태 모니터링용 소수 sensor 집합을 확보한다.
2. Sparse group lasso (지도) 로 응답 예측에 기여하는 sensor 집합을 확보한다.
3. 두 집합의 합집합이 실제 유지할 sensor 목록이 된다.
4. Chamber별로 1~3을 따로 수행하고 교집합을 본다. 한쪽 chamber에서만 선택되는 sensor는 그 chamber의 국소 이상을 반영할 뿐 공정 물리를 반영하지 않을 가능성이 높다.

## 4. PLS Supervised Reduction

### 4.1 Objective Contrast with PCA

두 방법의 목적함수는 한 항 차이다.

$$\mathrm{PCA:}\ \max_{\lVert w \rVert = 1} \mathrm{Var}(Xw) \qquad \mathrm{PLS:}\ \max_{\lVert w \rVert = 1} \mathrm{Cov}(Xw,\, y)^2$$

PLS의 목적함수를 분해하면 다음과 같다.

$$\mathrm{Cov}(Xw, y)^2 = \mathrm{Var}(Xw) \times \mathrm{Corr}(Xw, y)^2 \times \mathrm{Var}(y)$$

PCA는 첫 항만 최대화하고 둘째 항을 무시한다. 문제는 fab trace data에서 $X$ 의 지배적 분산이 $y$ 와 무관한 경우가 대부분이라는 점이다. 구체적으로 상위 주성분은 거의 항상 다음 중 하나를 잡는다.

- Chamber 간 계통 차이
- Recipe step 구조 (모든 sensor가 step 경계에서 동시에 움직인다)
- PM 주기에 따른 장기 drift

이들은 분산이 크지만 응답 예측력은 낮다. PCR로 상위 성분 몇 개만 쓰면 응답과 상관된 신호가 하위 성분에 묻혀 잘려 나간다. 반도체 VM에서 PCR이 아니라 PLS가 표준인 이유다.

### 4.2 Multi-Response PLS2

두께와 면저항을 동시에 예측한다면 개별 PLS1 두 개보다 잠재공간을 공유하는 PLS2를 권한다. 두 응답이 동일한 물리적 원인 (막 두께 profile) 을 공유하므로, 잠재공간을 공유하면 상호 정칙화 효과가 생겨 $n$ 이 작을 때 유리하다. 단, 두 응답의 규모와 noise 수준이 크게 다르면 PLS2가 분산 큰 쪽에 끌려가므로 응답변수도 반드시 autoscaling한다.

### 4.3 Choosing the Component Count

성분 수 $A$ 는 유일한 hyperparameter이자 유일한 정칙화 강도다. $A$ 를 키우면 PLS는 OLS로 수렴한다. $A$ 선택이 곧 모형 복잡도 선택이므로, 이 결정에 test data가 조금이라도 개입하면 test 성능이 즉시 낙관 편향된다.

무작위 KFold가 안 되는 이유는 누수 경로가 두 가지 있기 때문이다.

- 시간 누수: 장기 drift와 계단형 shift가 있는 data에서 무작위 분할은 미래 wafer로 학습해 과거를 예측하게 만든다. 실제 운영에서는 불가능한 조건이며, drift 아래에서 이 편향은 결정계수 0.1~0.2 수준으로 크다.
- Lot 누수 (더 치명적): 같은 lot, 같은 batch의 wafer는 trace가 거의 동일하다. 무작위 분할은 같은 lot의 wafer를 train과 test에 나눠 넣어, 사실상 동일 표본을 양쪽에 두는 것이 된다. 반드시 lot 또는 일자 단위로 그룹을 묶어 분할한다.

```python
# Python — nested, time-ordered, lot-grouped CV for the component count
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

# Sort wafers by processing time, block them by lot,
# and leave a purge gap of one or two lots between train and test.
outer = time_ordered_lot_splits(lots, n_splits=5, gap=2)

for tr, te in outer:                       # outer loop: performance estimate
    best_A, best_score = None, -np.inf
    inner = time_ordered_lot_splits(lots[tr], n_splits=4, gap=2)
    for A in range(1, 21):                 # inner loop: choose A
        scores = []
        for itr, ite in inner:
            sc = StandardScaler().fit(X[tr][itr])       # fit inside the fold
            m = PLSRegression(n_components=A, scale=False)
            m.fit(sc.transform(X[tr][itr]), y[tr][itr])
            scores.append(r2(y[tr][ite],
                             m.predict(sc.transform(X[tr][ite]))))
        if np.mean(scores) > best_score:
            best_score, best_A = np.mean(scores), A
    # refit with best_A, then evaluate on the outer test block
```

- 중첩 구조가 필수인 이유: 단일 CV로 $A$ 를 고르고 같은 CV 점수를 성능으로 보고하면 selection bias가 들어간다. 후보 20개에서 고른 최댓값은 이미 CV noise의 상위 극단을 취한 값이다. 목적함수가 train 성능을 보상하도록 잘못 설계된 hyperparameter 탐색이 과적합을 유도하는 것과 동일 계열의 오류이며, 여기서는 CV 점수 자체가 오염된다.
- 1-SE 규칙: 최고 점수의 $A$ 대신, 최고 점수에서 1 표준오차 이내를 달성하는 최소 $A$ 를 고른다. $n$ 이 작을 때 CV 곡선은 평탄하고 noise가 크므로 최댓값의 $A$ 는 거의 항상 과대추정이다. 이 규칙 하나로 $A$ 가 절반 이하로 줄어드는 경우가 흔하다.
- Purge gap: train의 마지막 lot과 test의 첫 lot 사이에 공백을 두는 이유는 인접 lot 간 자기상관 때문이다. Chamber 상태는 연속적으로 변하므로, 바로 다음 lot을 예측하는 것은 실제 배포 조건 (주 단위 재학습) 보다 쉽다.

### 4.4 Diagnostics

- VIP: 특징별 기여도다. 1을 넘으면 중요하다는 것이 관례적 임계이나, channel 200개 상황에서는 임계를 1.2~1.5로 올려 상위 30개 정도만 본다. Group lasso의 선택 결과와 교차 확인하는 신호로 쓰며, 두 방법이 동의하는 sensor가 신뢰도가 높다.
- Hotelling T² 와 SPE: PLS 잠재공간에서의 T² 와 잔차 SPE는 그대로 drift 모니터링 지표가 된다. 운영 data의 계단형 shift가 T² 에서 계단 형태로 나타나는지 검증하면, dataset shift 분석과 VM 모형이 하나의 틀로 연결된다.
- 비선형성 대응: PLS는 선형 사영이다. 잔차에 체계적 곡률이 보이면 두 선택지가 있다. (a) PLS score $A$ 개를 특징으로 GBM에 투입한다. 차원이 이미 5~10개이므로 안전하다. (b) kernel PLS를 쓴다. $n$ 이 수백 규모이면 (a) 가 훨씬 안정적이다.

## 5. References

- Yuan, M. and Lin, Y., "Model Selection and Estimation in Regression with Grouped Variables", Journal of the Royal Statistical Society Series B, 2006.
- Simon, N. et al., "A Sparse-Group Lasso", Journal of Computational and Graphical Statistics, 2013.
- Zou, H., Hastie, T. and Tibshirani, R., "Sparse Principal Component Analysis", Journal of Computational and Graphical Statistics, 2006.
- Meinshausen, N. and Bühlmann, P., "Stability Selection", Journal of the Royal Statistical Society Series B, 2010.
- Wold, S., Sjöström, M. and Eriksson, L., "PLS-regression: a Basic Tool of Chemometrics", Chemometrics and Intelligent Laboratory Systems, 2001.
- Kim, T. et al., "Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift", ICLR, 2022.

---

## Appendix A. Terminology

- 1-SE 규칙: CV에서 최고 점수 대신 최고 점수의 1 표준오차 이내를 달성하는 가장 단순한 모형을 고르는 선택 규칙.
- Attention pooling: 집계 가중치를 학습해 가중합으로 축을 접는 pooling.
- Autoscaling: 변수별로 평균 0, 분산 1이 되도록 표준화하는 전처리.
- Bootstrap: 자료에서 복원 추출로 표본을 반복 생성하는 resampling 방법.
- CV: Cross-Validation. 자료를 fold로 나눠 남긴 fold로 성능을 추정하는 교차검증.
- Dropout: 학습 중 신경망의 unit을 무작위로 끄는 정칙화.
- Elastic-net: L1과 L2 정칙화를 함께 거는 penalty.
- FPC / FPCA: Functional Principal Component (Analysis). Trace를 함수로 보고 기저 전개 후 수행하는 PCA와 그 성분 점수.
- GBM: Gradient Boosting Machine. 얕은 tree를 순차적으로 더해 가는 boosting 예측 모형.
- Hotelling T²: 잠재공간에서 중심으로부터의 거리를 공분산으로 보정한 통계량. 공정 감시의 표준 지표.
- KFold: 자료를 무작위로 K개 fold로 나누는 기본 CV 분할 방식.
- Lot: 같은 회차로 함께 처리되는 wafer 묶음.
- MLP: Multi-Layer Perceptron. 완전연결 신경망.
- OLS: Ordinary Least Squares. 최소제곱 선형 회귀.
- One-hot: 소속 범주 하나만 1이고 나머지는 0인 vector 표현.
- PCA: Principal Component Analysis. 분산이 큰 직교 방향으로 투영하는 선형 차원 축소.
- PCR: Principal Component Regression. PCA 점수를 설명변수로 쓰는 회귀.
- PLS: Partial Least Squares. 응답변수와의 공분산이 큰 방향으로 투영하는 지도 학습형 차원 축소. 응답이 하나면 PLS1, 여럿이면 PLS2로 부른다.
- PM: Preventive Maintenance. 장비의 주기적 정비. 전후로 chamber 상태가 계단형으로 변한다.
- QR 분해: 행렬을 직교 행렬과 상삼각 행렬의 곱으로 나누는 분해. 그룹 whitening에 쓴다.
- RevIN: Reversible Instance Normalization. 표본별 평균·분산을 제거했다가 출력에서 되돌리는 정규화.
- Rs: 면저항. Sheet resistance.
- Selection bias: 여러 후보 중 점수 최댓값을 고르는 절차 자체가 만드는 낙관 편향.
- SPE: Squared Prediction Error. 잠재공간 밖 잔차의 제곱합. Q 통계량으로도 부른다.
- THK: 두께. Thickness.
- VIP: Variable Importance in Projection. PLS에서 특징별 기여도를 요약한 지표.
- VM: Virtual Metrology. 장비 sensor data로 metrology 측정값을 예측하는 기술.
- Weight decay: 가중치의 L2 크기에 비례하는 penalty를 손실에 더하는 정칙화.
- Whitening: 변수 간 상관을 제거해 공분산을 단위 행렬로 만드는 변환.
