# Wafer VM Improvement for Study wafer_wlzpoly
Rev. 1 | Created: 2026-05-29 | Updated: 2026-10-03 09:07 CDT

## 1. Purpose

- **Problem Statement**: wafer virtual metrology model 의 test R² 가 0.36 에서 정체하고, train R² 와의 차이가 크다.
- **Goal**: 적용한 방법과 개선 방향을 결합분포 taxonomy 의 P(X), P(Y), P(Y|X) 와 Model 축에 배치하여 다음 실험의 순서를 정한다.
- **Non-Goal**: taxonomy 의 원리는 다루지 않는다. 원리는 [modeling-elements-from-joint-distribution-decomposition-for-manufacturing-data-ko.md](modeling-elements-from-joint-distribution-decomposition-for-manufacturing-data-ko.md) 에 있다.

## 2. Summary

- Headline 은 ① 목적함수 오정의와 ② p≫n 과적합이다. 분포 shift 축은 보조다 (section 5).
- 따라서 첫 실험은 목적함수 재정의와 p≫n 축소다 (section 6 의 1, 2 번).
- 분류 축은 결합분포의 세 항목 P(X) (covariate shift), P(Y) (prior shift), P(Y|X) (concept drift) 와, 그 밖에 있는 Model (추정기) 축이다.

## 3. Scope

- Source: Optuna DB `optuna_ps1.db` 의 study `wafer_wlzpoly`.
- 데이터: train 222, test 66 (소표본).
- 참고 자료: 발표 slide `domain_informed_aiml.html`.

## 4. Method

Table 1 은 네 축마다 DB 에서 복원한 적용 방법과 개선 방향을 나란히 둔다.

Table 1. Applied methods and roadmap per axis

| Axis                  | Applied (restored from DB)                                                                 | Roadmap                                                                                                                            |
| :-------------------: | :----------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------: |
| P(X) Covariate shift  | 3단계 feature 선택 (1,444→59), 안정성·상관 filter, human feature (AUC·pct), CORAL (option) | Feature 압축 강화 (후보·상한 축소, θ↑), 물리 feature 확대, CORAL 경량화·대체                                                       |
| P(Y) Prior shift      | 공간 분해 target (decomposed_targets·a1), group 별 z-score, 다항 계수 예측                 | 분해 고도화 (고차·Zernike·radius·zone), 계수 간 상관 동시 modeling                                                                 |
| P(Y\|X) Concept drift | Detrending (MA·23), 최신성 가중, 최근 drift windowing, temporal CV                         | 누수 차단과 다중 fold 평균 (last-fold → repeated), drift 적응 자동화, Binning CDT·시간창별 I(X;Y) monitoring                       |
| Model Estimator       | LGBM + CatBoost (+ XGB), 얕은 tree와 강한 정규화 (depth 3·L2·bagging), Optuna TPE          | 물리·engineer 지식 반영 (단조 제약, 도메인·룰 feature, 물리잔차 hybrid), 목적함수 안정화, UQ·신뢰구간, 데이터 효율 (능동학습·전이) |

Model 축은 데이터 (분포) 관점인 세 항목과 달리 추정기·최적화 관점이다. 발표에서는 "앞 셋은 데이터를 어떻게 보느냐, Model 은 어떻게 학습하느냐" 로 설명한다. 도메인 지식을 Model 에 주입하는 통로는 세 가지다.

- **물리 prior·단조 (monotonic)**: 제약으로 주입. "이 변수가 늘면 출력은 항상 한 방향" 을 LGBM, CatBoost, XGB 의 `monotone_constraints` 로 지정한다. 과적합을 억제하고 외삽을 안전하게 한다.
- **룰·도메인 feature**: feature 로 주입. Engineer 의 관계식, 임계값, 파생량을 입력으로 넣는다. 작은 데이터에서 특히 유효하다.
- **물리잔차 hybrid**: model 구조로 주입. 물리식으로 1차 예측하고, ML 은 잔차 (실제 − 물리) 만 학습하며, 최종 예측은 물리 + ML 보정이다 (gray-box).

## 5. Result

- Feature 단계별 개수: 1,444 → 93 → 59.
- Best trial #92: train R² 0.587, test R² 0.336 (gap +0.25). 과적합.
- 전 trial 의 test R² 최고 0.36, 평균 0.11. Trial #92 이후 32회 정체.
- 목적함수 누수: Optuna value 와 train R² 의 상관 0.93, test R² 와의 상관 0.48. 목적이 temporal CV 의 last-fold R² 하나여서 noise 가 크고 train 적합에 끌린다.
- ur2 (≈0.98) ≫ r2 (≈0.34): 추세와 offset 은 잘 맞추지만 평균을 뺀 (centered) 변동을 못 잡는다.
- CORAL: 상위 trial 에서는 off 가 유리하다. 현재 데이터에서는 역효과가 관측되었다.

## 6. Further Work

1. **목적함수 재정의**: last-fold 단일 R² 를 repeated·averaged temporal CV R² 로 바꾼다. Feature 선택, scale, detrending 은 fold 안에서만 적합하여 누수를 막는다.
2. **p≫n 축소 실험**: `n_features_1` 상한을 크게 낮추고 `stability_threshold` 하한을 올렸을 때의 test R² 변화를 A/B 로 비교한다.
3. **Concept drift 관측 module**: 잔차 CUSUM·Page-Hinkley 와 Binning CDT 로 변동 시점을 보고한다. 소표본이므로 등빈도 bin 을 쓰고 창 길이를 보수적으로 잡는다.
4. **P(Y) 고도화**: 분해 기저를 Zernike 등으로 확장하고, 계수 간 상관 구조를 joint·multi-output 으로 modeling 한다.
5. **Model 물리 반영**: `monotone_constraints` 변수 목록을 확정한 뒤 물리잔차 hybrid prototype 을 만든다.
6. **지표 재정의**: centered R² 와 ur2 가운데 업무 의미에 맞는 목적지표를 확정한다.
7. **CORAL 처리**: 비활성 또는 경량 대체를 결정한다.

---

## Appendix A. Terminology

- **Binning CDT (Conditional Distribution Test)**: X 를 bin 으로 나눠 P(Y|bin) 을 시간창별로 검정하여 변동 시점을 특정하는 방법.
- **CORAL (CORrelation ALignment)**: 학습과 운영의 입력분포를 공분산 정렬로 맞추는 domain adaptation.
- **CUSUM (Cumulative Sum)**: 기준값과의 편차를 누적하여 임계값을 넘는 시점을 변화점으로 보는 관리도.
- **I(X;Y)**: 상호정보량 (mutual information). X 와 Y 의 의존성 총량을 나타내는 거시 지표.
- **monotonic 제약**: 특정 변수에 대해 출력이 한 방향으로만 변하도록 강제하는 제약.
- **Page-Hinkley**: 누적 편차와 그 최솟값의 차이가 임계값을 넘으면 평균 변화를 알리는 순차 검정.
- **p≫n**: feature 수 (p) 가 표본 수 (n) 보다 훨씬 많은 구조. 과적합 위험이 높다.
- **temporal CV**: 과거로 학습하고 미래로 검증하는 시간순 교차검증.
- **TPE (Tree-structured Parzen Estimator)**: Optuna 의 기본 Bayesian 최적화 sampler.
- **UQ (Uncertainty Quantification)**: 불확실성 정량화. 점추정 대신 예측 범위 [하한, 상한] 을 제공한다.
- **ur2**: 평균을 빼지 않은 (uncentered) R². 추세와 offset 까지 맞춘 것으로 계산되므로 centered R² 보다 크게 나온다.
- **Zernike**: 원판 위에서 정의된 직교 다항식 기저. Wafer map 분해에 쓴다.
- **θ↑**: `stability_threshold` (안정성 선택 임계값) 상향. 반복 표집에서 꾸준히 뽑힌 feature 만 채택한다.
- **물리잔차 hybrid**: 물리 model 로 1차 예측하고 ML 이 잔차를 보정하는 2단 구조.
- **후보·상한 축소**: 1단계 feature 후보 수와 탐색 상한을 낮춰 과적합의 입구를 막는 것.
