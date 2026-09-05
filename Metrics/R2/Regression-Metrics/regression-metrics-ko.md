# Regression Metrics (Korean)
Rev. 0 | Created: 2026-09-05 | Updated: 2026-09-05 00:02 CDT

> 회귀 평가 지표를 variance-based, mean-based, agreement-based 세 갈래로 나눈 분류이며,
> $y=x$ 선에 견주어, 그리고 그 가운데 여럿을 무너뜨리는 low variance effect 에 견주어
> 읽는다.

## 1. Executive Summary

반도체 제조, virtual metrology, 다중 sensor 시계열 분석 같은 앞선 공학 분야에서 예측 model 을
검증하려면 성능 점수 하나로는 모자란다. 두 물음을 갈라야 한다. Model 이 추세를 얼마나 잘 따라가는가
하는 정밀도와, 예측이 물리적 참값에 얼마나 가까운가 하는 정확도이다.

이 문서는 평가 지표를 variance-based, mean-based, agreement-based 지수로 나누는 분류를 세운다. 그
위계는 이상적인 $y=x$ 선에 견주어 model 성능을 해석하는 틀을 주며, low variance effect 가 부르는
위험에 특히 초점을 둔다.

## 2. Metric Hierarchy

세 갈래는 먼저 지표가 무엇으로 만들어졌는지로 나뉘고, 그다음 값이 측정의 단위를 지니는지로 나뉜다.

```
Regression metrics
├── Mean-based
│   ├── Scale-dependent   → MAE, MSE, RMSE, Huber
│   └── Scale-independent → MPE, MAPE, SMAPE, CV(RMSE)
├── Variance-based
│   └── Scale-independent → R², Adj. R²
└── Mean+Variance-based (Hybrid)
    └── Scale-independent → CCC, KGE
```

### 2.1 Variance Index

이 지표들은 관측값과 예측값 사이 관계의 선형성과 세기를 잰다. 절대적인 크기와 무관하게 model 이
자료의 모양을 붙잡는지에 초점을 둔다.

#### Pearson Correlation Coefficient

$$r = \frac{\sum (y_i - \mu_y)(\hat{y}_i - \mu_{\hat{y}})}{\sqrt{\sum (y_i - \mu_y)^2 \sum (\hat{y}_i - \mu_{\hat{y}})^2}}$$

여기서 $y_i$ 는 관측된 참값이고, $\hat y_i$ 는 예측값이며, $\mu_y$ 와
$\mu_{\hat{y}}$ 는 관측값과 예측값의 평균이다.

- 1:1 선과의 관계: $r$ 은 자료가 어떤 직선 둘레에 얼마나 바짝 모이는지를 잰다. $r=1$ 이 완벽해도
  자료가 $y=x$ 선 위에 있다는 보장은 없으며, $y = 2x + 10$ 위에 있을 수도 있다.
- Low variance effect: 자료의 분산이 아주 작으면, 이를테면 sensor 가 거의 일정한 값을 내면 분모가
  0 에 다가간다. 그러면 $r$ 이 미세한 잡음에도 극도로 민감해져, 예측이 물리적으로 참값에 가까운데도
  상관이 낮거나 정의되지 않는 일이 잦다.
- 응용: 초기 feature selection, 그리고 비슷한 거동을 보이는 sensor 찾기.

#### Coefficient Of Determination

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \mu_y)^2}$$

여기서 $SS_{res}$ 는 잔차제곱합, 곧 설명되지 않은 분산이고, $SS_{tot}$ 는 총제곱합, 곧 자료의 총
분산이다.

- 1:1 선과의 관계: model 이 설명하는 분산의 비율을 나타낸다. $r$ 보다 1:1 선에서 멀어지는 것을 더
  벌하지만, model 이 계통적으로 치우쳐 있으면 여전히 사람을 오도한다.
- Low variance effect: 목표값의 분산이 작을 때 $R^2$ 는 기만적이다. 분모 $SS_{tot}$ 가 작아 아주
  작은 예측 오차로도 음수이거나 0 에 가까운 $R^2$ 가 나오므로, 절대 오차가 공학적 허용 범위 안에
  있어도 나쁜 model 처럼 보인다.
- 응용: 제조 yield 분석에서 회귀 model 의 설명력을 재는 표준 benchmark.

#### Explained Variance Score

$$ExpVar = 1 - \frac{Var(y - \hat{y})}{Var(y)}$$

여기서 $Var(y - \hat{y})$ 는 잔차의 분산이고 $Var(y)$ 는 참값의 분산이다.

- 1:1 선과의 관계: $R^2$ 와 비슷하지만 잔차의 평균을 무시한다. 예측의 흔들림이 참값의 흔들림과
  맞아떨어지는지에만 집중한다.
- Low variance effect: $R^2$ 처럼 $Var(y)$ 가 작으면 이 지표도 무너진다. 일정한 setpoint 를 지키는
  것이 목적인 안정한 공정에서는 뜻있는 점수를 주지 못한다.
- 응용: 절대적인 기준선보다 상대적인 변화가 중요한 신호 처리.

### 2.2 Mean Index

이 지표들은 예측 vector 와 참값 사이의 물리적 거리를 잰다. 오차의 실제 비용을 이해하는 데 꼭
필요하다.

#### Mean Absolute Error

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

여기서 $n$ 은 표본의 개수이다.

- 1:1 선과의 관계: 선까지의 평균 수직 거리.
- 한계: 모든 벗어남을 선형으로 다루므로 크고 드문 오차를 드러내지 못한다. Low variance effect 를
  타지 않아 안정한 공정에서 더 믿을 만하다.
- 응용: 오차의 비용이 오차의 크기에 정확히 비례하는 경우.

#### Mean Squared Error And Root Mean Squared Error

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2, \quad RMSE = \sqrt{MSE}$$

- 1:1 선과의 관계: 선까지 거리의 제곱의 평균. RMSE 는 원래 단위로 잰 전형적인 거리를 나타낸다.
- 한계: 이상값에 크게 휘둘린다. 목표값의 낮은 분산에는 강건하지만, model 이 자료의 추세를 붙잡고
  있는지는 말해 주지 않는다.
- 응용: 학습의 표준 loss function 이며, 큰 벗어남이 wafer 폐기로 이어지는 두께 예측에서 결정적이다.

#### Mean Percentage Error And Mean Absolute Percentage Error

$$MPE = \frac{100}{n} \sum_{i=1}^{n} \frac{y_i - \hat{y}_i}{y_i}, \quad MAPE = \frac{100}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

- 1:1 선과의 관계: 선에서 벗어난 상대적인 정도를 평가한다. MPE 는 평균 백분율 치우침, 곧 model 이
  줄곧 과대평가하는지 과소평가하는지를 재고, MAPE 는 항등선에 견준 백분율 오차의 평균 크기를
  나타낸다.
- 한계: 가장 큰 약점은 0 또는 0 에 가까운 값으로 나누는 것이다. 목표값이 0 이거나 아주 작으면 지표가
  폭발한다. MAPE 는 비대칭이기도 해서 어떤 맥락에서는 과대평가를 과소평가보다 무겁게 벌한다. 그
  약점은 분모 $y_i$ 가 변한다는 데 있다. 모든 점에서 자료의 척도가 한결같으면 분모가 상수처럼
  작용하여 MAPE 가 안정해지고, MSE 나 RMSE 가 그러하듯 절대 오차에 선형으로 비례한다. Section
  2.3 의 agreement 지수와 달리 이 지표들은 척도의 이동과 위치의 이동을 가려내지 못한다.
- 응용: 기술적이지 않은 이해관계자에게 사업의 말로 model 성능을 전할 때 쓰며, 금융 예측과 yield
  관리에서 널리 쓰인다.

#### Coefficient Of Variation Of RMSE

$$CV(RMSE) = \frac{RMSE}{\mu_y}$$

- 1:1 선과의 관계: 오차를 평균으로 정규화한다.
- Low variance effect: 평균 $\mu_y$ 가 0 에 가까우면 이 지표는 폭발한다. 그래도 평균이 0 에 가깝지
  않은 저분산 자료에서는 $R^2$ 보다 안정하다.
- 응용: 척도가 서로 다른 sensor 종류들 사이에서 model 성능을 비교하기.

### 2.3 Agreement Index

이 지표들은 충실도, 곧 model 이 추세를 따라가면서 동시에 절대값도 맞혀야 한다는 요구를 평가한다.

#### Lin's Concordance Correlation Coefficient

$$\rho_c = \frac{2 \rho \sigma_y \sigma_{\hat{y}}}{\sigma_y^2 + \sigma_{\hat{y}}^2 + (\mu_y - \mu_{\hat{y}})^2}$$

여기서 $\rho$ 는 Pearson 상관계수이고 $\sigma_y$ 와 $\sigma_{\hat{y}}$ 는 관측값과 예측값의
표준편차이다. 이 계수를 아래에서는 CCC 로 줄여 적는다.

- 1:1 선과의 관계: 자료가 45 도 선에서 얼마나 벗어나는지를 곧바로 재며, 정밀도인 $r$ 과 정확도인
  치우침 벌점을 합친다.
- Low variance effect: $\rho$ 가 성분으로 들어 있으므로 자료의 분산이 극도로 낮으면 CCC 도 떨어져,
  절대 거리로는 잘 맞히는 model 을 가린다.
- 응용: 새 계측 sensor 를 표준 실험실 측정에 견주어 검증하기.

#### Kling-Gupta Efficiency

$$KGE = 1 - \sqrt{(r-1)^2 + (\alpha-1)^2 + (\beta-1)^2}$$

여기서 $r$ 은 Pearson 상관이고, $\alpha = \sigma_{\hat{y}}/\sigma_y$ 는 변동성 비이며,
$\beta = \mu_{\hat{y}}/\mu_y$ 는 치우침 비이다.

- 1:1 선과의 관계: 총체적인 일치 지표이다. $r$, $\alpha$, $\beta$ 가 모두 1 일 때만 1.0 에 닿는다.
- Low variance effect: 변동성 비 $\alpha$ 에 극도로 민감하다. 참값의 분산이 거의 0 이면 $\alpha$ 가
  정의되지 않거나 불안정해져 KGE 가 무너진다.
- 응용: 복잡한 산업 공정 제어와 높은 충실도의 시계열 simulation.

## 3. Comparative Summary

Table 1. The three metric families against the 1:1 line and the low variance effect

| Category | Primary focus | Best use case | Relation to $y=x$ | Low variance effect |
|----------|---------------|---------------|-------------------|---------------------|
| Variance-based | 추세와 pattern | Feature selection | 선형이면 치우쳐 있어도 높은 점수. | 위험이 크다. 오차가 작아도 점수가 무너지거나 요동친다. |
| Mean-based | 절대 오차 | Loss 로 삼는 model 학습 | 선 위에 정확히 있을 때만 0. | 강건하다. 분산과 무관하게 안정하고 해석된다. |
| Agreement-based | 충실도와 보정 | System 전체 검증 | 선 위에 정확히 있을 때만 1. | 위험이 중간이다. 상관 성분에서 민감도를 물려받는다. |

## 4. Recommendation For Engineering Teams

반도체나 sensor 기반 시설에 model 을 배포할 때 variance 지수에만 기대서는 안 된다. 고정밀 제조에서
sensor 는 흔히 좁고 안정한 범위에서 도는데, 그 영역에서는 Pearson 과 $R^2$ 가 model 이 실패하고
있다고 말한다. 실제로는 sub-micron 정확도로 예측하고 있을 수 있는데도 그렇다.

- 저분산 환경에서는 RMSE 나 MAE 를 참값의 일차 근거로 삼는다.
- Agreement 지수인 CCC 와 KGE 는 자료의 범위가 충분할 때만 system 전체 검증에 쓴다.
- $R^2$ 가 떨어진 것을 해석하기 전에 low variance effect 를 먼저 확인한다. 예측력이 사라진 것이
  아니라 수학적 산물인 경우가 잦다.

---

## Appendix A. Terminology

- Adjusted R²: 조정 결정계수. 예측변수의 개수로 보정한 결정계수이며, 예측변수를 더한다고 해서 그
  자체로 점수가 오르지 않게 한다.
- Huber: 후버 손실. 잔차가 작으면 제곱, 크면 선형인 손실이며, 이상값을 버리지 않으면서 그것이 끄는
  힘을 제한한다.
- Scale-dependent: 척도 종속. 측정의 단위를 지니므로 크기가 다른 양들 사이에서 값을 비교할 수 없다.
- Scale-independent: 척도 독립. 정규화되어 있어 크기가 다른 양들 사이에서 값을 비교할 수 있다.
- SMAPE: 대칭 평균 절대 백분율 오차. 관측값과 예측값의 평균으로 나누는 백분율 오차이며, MAPE 의
  비대칭을 없앤다.
- Virtual metrology: 가상 계측. 직접 재는 대신 공정과 sensor 자료로부터 측정값을 예측하는 것.
