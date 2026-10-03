# P(X) · P(Y) · P(Y|X) Multi-Lens Note
Rev. 0 | Created: 2026-10-02 | Updated: 2026-10-02 23:32 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Joint Distribution Decomposition](#3-joint-distribution-decomposition)
- [4. Lens Matrix](#4-lens-matrix)
- [5. Per-Term Detail](#5-per-term-detail)
  - [5.1 P(X) Covariate Shift](#51-px-covariate-shift)
  - [5.2 P(Y) Prior Shift](#52-py-prior-shift)
  - [5.3 P(Y|X) Concept Drift](#53-pyx-concept-drift)
- [6. Model Axis](#6-model-axis)
- [7. Project Diagnosis](#7-project-diagnosis)
- [8. Applied vs Roadmap Matrix](#8-applied-vs-roadmap-matrix)
- [9. Further Work](#9-further-work)
- [Appendix A. Terminology](#appendix-a-terminology)

## 1. Purpose

- **Problem Statement**: wafer virtual metrology model 의 test R² 가 0.36 에서 정체하며, 개선 방향을 고를 공통 기준이 없다.
- **Goal**: P(X), P(Y), P(Y|X) 를 여러 각도로 이해하고, 각 개선 작업을 결합분포 분해의 어느 항목에 대한 관측·개입인지로 분류한다.
- **Non-Goal**: 개선 실험의 구현 code 는 다루지 않는다.

## 2. Summary

- **P(X,Y) = P(Y|X) · P(X)**. 세 항목은 결합분포의 분해이고, **P(Y)** 는 그 주변분포다.
- 분포 변화 관점의 표준 3분류: **P(X) = Covariate shift**, **P(Y) = Prior shift**, **P(Y|X) = Concept drift**.
- **Model** 은 분포축과 직교 (orthogonal) 하는 추정기축이다.
- 반도체 공정에서는 세 변화가 동시에 온다. 장비·레시피를 바꾸면 입력과 관계가 함께 이동하므로, 세 항목을 배타적 분류 대신 **관측·개입 지점** 으로 읽는다.
- 이 프로젝트의 headline 은 ① 목적함수 오정의와 ② p≫n 과적합이고, shift 축은 보조다 (section 7).

## 3. Joint Distribution Decomposition

결합분포는 관계와 입력의 곱으로 분해되고, 출력 주변분포는 관계를 입력분포로 적분한 값이다.

```math
P(X, Y) = P(Y \mid X) \cdot P(X) \hspace{19em} (1)
```

```math
P(Y) = \int P(Y \mid X)\, P(X)\, dX \hspace{19em} (2)
```

- **P(X)**: 입력 (feature) 의 분포.
- **P(Y)**: 출력 (target) 의 분포.
- **P(Y|X)**: 입력이 주어졌을 때 출력이 결정되는 관계 (mechanism).

## 4. Lens Matrix

Table 1 은 같은 세 기호를 분해, 변화, 개입, 질문, 공정, 관측, 대책의 7개 축으로 교차하여 본다.

Table 1. Seven lenses on P(X), P(Y) and P(Y|X)

| Lens              | P(X)                                                       | P(Y)                                         | P(Y\|X)                                               |
| :---------------: | :--------------------------------------------------------: | :------------------------------------------: | :---------------------------------------------------: |
| 확률 분해         | 입력 주변분포                                              | 출력 주변분포                                | 조건부 (관계)                                         |
| 분포 변화 (shift) | Covariate shift                                            | Prior / label shift                          | Concept drift                                         |
| 개입 지점         | 입력공간                                                   | 출력공간 (target 구조화)                     | 관계·mechanism                                        |
| 질문 형태         | 들어오는 데이터가 달라졌나?                                | 정답 분포가 달라졌나?                        | 입력→출력 규칙이 달라졌나?                            |
| 공정 비유         | 새 장비·센서 drift·신규 레시피 유입                        | Target spec·계수 분포 이동                   | 같은 입력에 다른 결과 (chamber 노화 등)               |
| 관측 (탐지)       | PSI·KS·KL, domain classifier                               | Target 주변분포 비교                         | Binning CDT, 시간창별 I(X;Y), 잔차 CUSUM·Page-Hinkley |
| 대책 (lever)      | Feature 선택·증강, importance weighting, domain adaptation | Target 변환·분해, group 별 scale, prior 보정 | 재학습 period, 최신성 가중, detrending, drift 적응    |

## 5. Per-Term Detail

### 5.1 P(X) Covariate Shift

- 의미: 학습 때와 운영 때의 입력분포가 다르다. 관계 P(Y|X) 는 그대로일 수 있다.
- 해석: model 이 관계는 학습했으나, 학습 때 없던 입력이 들어온 경우다.
- 범위: 입력공간 작업은 shift 대응과 함께 feature 선택·생성 전반을 포함한다.

### 5.2 P(Y) Prior Shift

- 의미: 출력 주변분포가 이동한다 (label shift, prior probability shift).
- 확장: 출력공간을 어떻게 정의하고 구조화하는가, 곧 target 변환과 분해도 이 항목에 든다.
- Spatial decomposition: 출력공간에 개입하므로 P(Y) 에 속한다. 효과는 P(Y|X) 학습 난이도를 낮추는 쪽으로 전파된다. 출력을 매끄럽고 물리적인 값으로 바꾸면 관계 학습이 쉬워진다.

### 5.3 P(Y|X) Concept Drift

입력→출력 관계 자체가 변하는 경우이며, 세 항목 가운데 다루기 가장 어렵다. 대응과 관측을 구분한다.

- **대응 (현재 적용)**: detrending, 최신성 sample 가중, 최근 drift windowing, temporal CV. 관계가 변한다고 가정하고 최근 sample 에 가중치를 더 준다.
- **관측 (미구현)**: 변화를 측정하고 시점을 특정한다.

변동 시점을 특정할 수 있는지는 방법에 따라 갈린다.

- 현재의 대응 방법 (detrending, 가중) 은 시점을 특정하지 못한다.
- **Binning CDT**: X 를 bin 으로 나눠 P(Y|bin) 을 시간창별로 검정하여 변동 시점을 특정한다. P(Y|X) 를 가장 직접 본다.
- **잔차 CUSUM, Page-Hinkley**: 예측 잔차 통계량이 임계값을 넘는 시점을 출력한다.
- **I(X;Y)**: 의존성 총량 (거시 지표). 단독으로 쓰면 P(X), P(Y), 관계의 변화가 함께 잡히므로, 시간창별로 추적해야 concept drift 에 가까워진다.

## 6. Model Axis

P(X), P(Y), P(Y|X) 가 데이터 (분포) 관점이라면, Model 은 추정기·최적화 관점이다. 발표에서는 "앞 셋은 데이터를 어떻게 보느냐, Model 은 어떻게 학습하느냐" 로 설명한다.

도메인 지식을 Model 에 주입하는 통로는 세 가지다.

- **물리 prior·단조 (monotonic)**: 제약으로 주입. "이 변수가 늘면 출력은 항상 한 방향" 을 LGBM, CatBoost, XGB 의 `monotone_constraints` 로 지정한다. 과적합을 억제하고 외삽을 안전하게 한다.
- **룰·도메인 feature**: feature 로 주입. Engineer 의 관계식, 임계값, 파생량을 입력으로 넣는다. 작은 데이터에서 특히 유효하다.
- **물리잔차 hybrid**: model 구조로 주입. 물리식으로 1차 예측하고, ML 은 잔차 (실제 − 물리) 만 학습하며, 최종 예측은 물리 + ML 보정이다 (gray-box).

## 7. Project Diagnosis

진단의 출처는 Optuna DB `optuna_ps1.db` 의 study `wafer_wlzpoly` 이다.

- 규모: train 222, test 66 (소표본). Feature 단계별 개수 1,444 → 93 → 59.
- Best trial #92: train R² 0.587, test R² 0.336 (gap +0.25). 과적합.
- 전 trial 의 test R² 최고 0.36, 평균 0.11. Trial #92 이후 32회 정체.
- 목적함수 누수: Optuna value 와 train R² 의 상관 0.93, test R² 와의 상관 0.48. 목적이 temporal CV 의 last-fold R² 하나여서 noise 가 크고 train 적합에 끌린다.
- ur2 (≈0.98) ≫ r2 (≈0.34): 추세와 offset 은 잘 맞추지만 평균을 뺀 (centered) 변동을 못 잡는다. 지표 정의를 다시 검토한다.
- Headline: ① 목적함수 오정의, ② p≫n 과적합. 분포 shift 축은 보조다.

## 8. Applied vs Roadmap Matrix

Table 2 는 네 축마다 DB 에서 복원한 적용 방법과 개선 방향을 나란히 둔다.

Table 2. Applied methods and roadmap per axis

| Axis                  | Applied (restored from DB)                                                                 | Roadmap                                                                                                                            |
| :-------------------: | :----------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------: |
| P(X) Covariate shift  | 3단계 feature 선택 (1,444→59), 안정성·상관 filter, human feature (AUC·pct), CORAL (option) | Feature 압축 강화 (후보·상한 축소, θ↑), 물리 feature 확대, CORAL 경량화·대체                                                       |
| P(Y) Prior shift      | 공간 분해 target (decomposed_targets·a1), group 별 z-score, 다항 계수 예측                 | 분해 고도화 (고차·Zernike·radius·zone), 계수 간 상관 동시 modeling                                                                 |
| P(Y\|X) Concept drift | Detrending (MA·23), 최신성 가중, 최근 drift windowing, temporal CV                         | 누수 차단과 다중 fold 평균 (last-fold → repeated), drift 적응 자동화, Binning CDT·시간창별 I(X;Y) monitoring                       |
| Model Estimator       | LGBM + CatBoost (+ XGB), 얕은 tree와 강한 정규화 (depth 3·L2·bagging), Optuna TPE          | 물리·engineer 지식 반영 (단조 제약, 도메인·룰 feature, 물리잔차 hybrid), 목적함수 안정화, UQ·신뢰구간, 데이터 효율 (능동학습·전이) |

## 9. Further Work

1. **목적함수 재정의**: last-fold 단일 R² 를 repeated·averaged temporal CV R² 로 바꾼다. Feature 선택, scale, detrending 은 fold 안에서만 적합하여 누수를 막는다.
2. **p≫n 축소 실험**: `n_features_1` 상한을 크게 낮추고 `stability_threshold` 하한을 올렸을 때의 test R² 변화를 A/B 로 비교한다.
3. **Concept drift 관측 module**: 잔차 CUSUM·Page-Hinkley 와 Binning CDT 로 변동 시점을 보고한다. 소표본이므로 등빈도 bin 을 쓰고 창 길이를 보수적으로 잡는다.
4. **P(Y) 고도화**: 분해 기저를 Zernike 등으로 확장하고, 계수 간 상관 구조를 joint·multi-output 으로 modeling 한다.
5. **Model 물리 반영**: `monotone_constraints` 변수 목록을 확정한 뒤 물리잔차 hybrid prototype 을 만든다.
6. **지표 재정의**: centered R² 와 ur2 가운데 업무 의미에 맞는 목적지표를 확정한다.
7. **CORAL 처리**: 비활성 또는 경량 대체를 결정한다. 현재 상위 trial 에서는 off 가 유리하다.

참고 자료는 발표 slide `domain_informed_aiml.html` 과 source DB `optuna_ps1.db` (study `wafer_wlzpoly`) 이다.

---

## Appendix A. Terminology

- **Binning CDT (Conditional Distribution Test)**: X 를 bin 으로 나눠 P(Y|bin) 을 시간창별로 검정하여 변동 시점을 특정하는 방법.
- **CORAL (CORrelation ALignment)**: 학습과 운영의 입력분포를 공분산 정렬로 맞추는 domain adaptation. 현재 데이터에서는 역효과가 관측되었다.
- **CUSUM (Cumulative Sum)**: 기준값과의 편차를 누적하여 임계값을 넘는 시점을 변화점으로 보는 관리도.
- **I(X;Y)**: 상호정보량 (mutual information). X 와 Y 의 의존성 총량을 나타내는 거시 지표.
- **KL (Kullback–Leibler divergence)**: 두 분포의 차이를 정보량으로 잰 값.
- **KS (Kolmogorov–Smirnov test)**: 두 표본의 누적분포 최대 차이로 분포가 같은지 검정하는 방법.
- **monotonic 제약**: 특정 변수에 대해 출력이 한 방향으로만 변하도록 강제하는 제약.
- **Page-Hinkley**: 누적 편차와 그 최솟값의 차이가 임계값을 넘으면 평균 변화를 알리는 순차 검정.
- **PSI (Population Stability Index)**: 두 시점의 구간별 분포 비율 차이로 분포 이동을 재는 지표.
- **p≫n**: feature 수 (p) 가 표본 수 (n) 보다 훨씬 많은 구조. 과적합 위험이 높다.
- **Spatial decomposition**: wafer 측정 map 을 공간 기저 (다항식) 로 분해하고, 그 계수 (a1, …) 를 예측하는 방법.
- **temporal CV**: 과거로 학습하고 미래로 검증하는 시간순 교차검증.
- **TPE (Tree-structured Parzen Estimator)**: Optuna 의 기본 Bayesian 최적화 sampler.
- **UQ (Uncertainty Quantification)**: 불확실성 정량화. 점추정 대신 예측 범위 [하한, 상한] 을 제공한다.
- **ur2**: 평균을 빼지 않은 (uncentered) R². 추세와 offset 까지 맞춘 것으로 계산되므로 centered R² 보다 크게 나온다.
- **Zernike**: 원판 위에서 정의된 직교 다항식 기저. Wafer map 분해에 쓴다.
- **θ↑**: `stability_threshold` (안정성 선택 임계값) 상향. 반복 표집에서 꾸준히 뽑힌 feature 만 채택한다.
- **물리잔차 hybrid**: 물리 model 로 1차 예측하고 ML 이 잔차를 보정하는 2단 구조.
- **후보·상한 축소**: 1단계 feature 후보 수와 탐색 상한을 낮춰 과적합의 입구를 막는 것.
