# Ensemble Learning (Korean)
Rev. 0 | Created: 2026-09-10 | Updated: 2026-09-10 21:24 UTC

## 1. Purpose

- **Problem Statement**: 학습을 마친 여러 model 을 합칠 때 대개 손에 잡히는 방법, 곧 label 의 다수결이나 가중 없는 평균을 쓰게 되는데, 이때 각 model 이 이미 계산해 둔 확률이 버려지고, 그 결합이 어느 행에서 확신이 없었는지도 남지 않는다.
- **Goal**: 결합 규칙과 member 를 만들어 내는 framework 를 서로 떼어 두어, 주어진 문제에서 둘을 각각 고를 수 있게 하고, member 가 내놓는 확률이나 confidence 를 결합의 어느 자리에 넣을지 정할 수 있게 한다.
- **Non-Goal**: 개별 base model 의 원리와 그 hyperparameter 조정, streaming 자료에서 member 의 가중치를 갱신하는 방법, snapshot ensemble 과 Monte Carlo dropout 처럼 deep learning 고유의 ensemble 은 다루지 않는다.

## 2. Summary

Meta-learner 에 label 대신 확률을 먹이는 것이 이긴다. 그 차이는 hard voting 과 soft voting 중 무엇을 고르는지보다 크다. Table 5 의 held-out 171 행에서 가장 좋은 단일 member 는 0.9591 이고, hard voting 도 0.9591 로 같으며, 그냥 soft voting 은 0.9415 로 내려가고, stacking 은 label 로는 0.9591 이지만 확률로는 0.9649 이다. 확률은 member 가 어느 행에서 확신이 없었는지를 담고 있지만 label 은 담지 않으며, 잃은 뒤에는 어떤 결합 규칙으로도 되찾지 못한다.

확률을 평균하는 것은 확률을 쓰는 좋은 방법이 아니다. Soft voting 은 모든 행에서 모든 member 에게 정해진 몫을 주므로, 과신하는 member 하나가 평균을 어디서나 끌고 간다. 이 자료의 naive Bayes 는 예측의 95.3% 를 $0.99$ 위나 $0.01$ 아래에 두며, 그 때문에 soft voting 이 hard voting 아래로 내려간다. 고치는 방법 둘은 듣고 하나는 듣지 않는다. Calibration 은 soft voting 을 0.9474 로 올리고, 확률로 학습한 meta-learner 는 그 member 에 0.969, 가장 좋은 member 에 4.392 의 계수를 주어 0.9649 까지 올린다. 반면 log-odds 를 평균하면 0.9298 로 더 나빠지는데, 산술평균이 한 member 에 씌워 두었던 상한이 사라지기 때문이다.

결합은 **평균** member 를 이기는 것이지 **가장 좋은** member 를 저절로 이기는 것이 아니다. Equation (1) 은 근사가 아니라 항등식이며, ensemble 의 오차가 member 오차의 평균에서 member 사이의 흩어짐을 뺀 값임을 말한다. Section 4.2 의 회귀 실행이 이를 소수점까지 재현한다: $3267.2 - 177.9 = 3089.3$. 약한 member 를 더하면 첫 항이 올라가고, 그만큼을 둘째 항이 물어내야 한다.

이득의 바닥은 개수가 아니라 상관이 정한다. Equation (2) 는 쌍별 상관이 $\rho$ 인 $M$ 개의 member 를 평균해도 $M$ 이 아무리 커지든 $\rho\sigma^{2}$ 이 남는다는 것을 보인다. Bagging, random subspace, boosting 이 모두 member 를 더 많이 만드는 대신 서로 다르게 만드는 쪽으로 작동하는 이유가 여기에 있다.

## 3. Principle

### 3.1 Why Combining Helps

Ensemble 의 제곱오차는 member 오차의 평균에서 member 사이 흩어짐의 평균을 뺀 값이며, 이는 근사가 아니라 모든 입력에서 정확히 성립한다. 가중 없는 평균이 $\bar f$ 인 $M$ 개의 member $f_1 \ldots f_M$ 에 대해 ambiguity decomposition 이 점마다 성립한다 [[1](#ref-1)].

$$\left(\bar f - y\right)^{2} = \frac{1}{M}\sum_{m}\left(f_m - y\right)^{2} - \frac{1}{M}\sum_{m}\left(f_m - \bar f\right)^{2} \hspace{19em} (1)$$

여기서 세 가지가 따라 나오며, 이것이 ensemble 을 쓰는 이유의 전부이다.

- 둘째 항이 음수가 되지 않음. 따라서 ensemble 이 평균 member 보다 나빠지지 않음.
- 가장 좋은 member 에 대해서는 아무 보장 없음. 강한 member 를 약한 member 와 평균하면 질 수 있음.
- Member 사이의 흩어짐만이 이득의 원천. 나머지와 같은 답을 내는 member 는 보태는 것이 없음.

분류에서는 같은 생각이 Condorcet 논증으로 나타난다. 각각 $p \gt 0.5$ 의 확률로 맞히는 독립인 member 들의 다수결은 $M$ 이 커질수록 정확해진다. 실제로 깨지는 전제는 독립성이며, 그것이 깨졌을 때 남는 것이 section 3.2 이다.

### 3.2 The Correlation Floor

Member 가 서로 상관되면 개수는 더 이상 값을 하지 않는다. 분산이 모두 $\sigma^{2}$ 이고 쌍별 상관이 $\rho$ 인 $M$ 개의 member 를 두면, 그 평균의 분산은 다음과 같다.

$$\mathrm{Var}\left(\bar f\right) = \rho\sigma^{2} + \frac{1 - \rho}{M}\sigma^{2} \hspace{19em} (2)$$

둘째 항은 $M$ 이 커지면 사라지지만 첫 항은 남는다. $\rho = 0.9$ 이면 member 를 무한히 모아도 단일 member 분산의 90% 가 그대로 남는다. Section 6 의 모든 framework 이 행을 다시 뽑거나, 열을 가리거나, target 을 바꾸어 $\rho$ 를 낮추는 장치인 이유가 이것이며, 같은 gradient boosting 적합을 네 번째로 복사해 붙여도 아무것도 달라지지 않는 이유도 같다.

다양성은 필요하지만 그 자체를 최대화할 양은 아니다. 쌍별과 비쌍별을 아우르는 열 가지 다양성 통계량을 ensemble 정확도와 견주었을 때, 선택 기준으로 쓸 만큼 정확도를 따라가는 것은 하나도 없었다 [[13](#ref-13)]. Member 를 약하게 만들어 산 다양성은 equation (1) 의 첫 항에서 값을 치른다.

### 3.3 Two Independent Axes

두 가지 결정은 따로 내리며, 어느 짝을 골라도 성립한다. Table 1 이 그 둘이다.

Table 1. The two axes of an ensemble

| Axis | Question | Choices |
|------|----------|---------|
| Framework | Member 사이에서 무엇을 다르게 만드는가 | 행 재추출, 열 은닉, target 변경, algorithm 변경 |
| Aggregation | 그 출력을 어떻게 처리하는가 | 투표, 평균, 가중 평균, meta-learner |

Random forest 는 두 축을 한꺼번에 고정하므로 흔히 하나의 method 로 읽히지만, 실은 행과 열을 함께 뽑는 framework 에 평균 규칙을 붙인 것이다. Stacking 은 aggregation 축만 고정하고 framework 은 열어 두며, 그래서 서로 다른 종류의 member 를 받아들인다.

## 4. Aggregation

### 4.1 Voting

Hard voting 은 label 의 다수결을 취하고, soft voting 은 평균 확률의 argmax 를 취한다. Hard voting 은 member 가 확률을 내지 않을 때 남는 방법이며, soft voting 의 약한 판이 아니다. Label 은 과신할 수 없으므로, calibration 되지 않은 member 앞에서 살아남는 쪽은 오히려 이쪽이다.

Table 2 는 각 규칙이 언제 쓸 수 있고 무엇을 버리는지를 적은 것이다.

Table 2. Voting rules

| Rule | Input per member | Ignores | Ties |
|------|------------------|---------|------|
| Hard voting | Label 하나 | Member 가 얼마나 확신했는가 | $M$ 이 짝수면 발생, 미리 정한 규칙으로 해소 |
| Soft voting | 확률 vector 하나 | 버리는 것 없음 | 확률이 정확히 같지 않은 한 발생하지 않음 |
| Weighted hard voting | Label 과 scalar 가중치 | 이 행에서 member 가 얼마나 확신했는가 | 가중치가 서로 같지 않은 한 가중치로 해소 |
| Weighted soft voting | 확률 vector 와 scalar 가중치 | 버리는 것 없음 | 실제로 발생하지 않음 |

Member 수가 짝수이면 hard voting 은 동점 규칙을 요구하며, 첫 class 를 고르는 동점 규칙은 드러나지 않는 편향이다. Member 를 셋으로, 일반적으로는 홀수로 두면 이 물음 자체가 사라진다.

### 4.2 Averaging

회귀에서 투표에 해당하는 것은 member 예측의 평균이다. Diabetes 자료의 held-out 133 행에 대한 Appendix B.2 의 실행에서 RMSE 는 ridge 55.67, forest 59.42, k-nearest neighbours 56.32 이고, 단순 평균은 55.58 로 모든 member 아래에 있다.

같은 실행이 equation (1) 을 재현한다. Member MSE 의 평균이 3267.2, member 사이 흩어짐의 평균이 177.9, ensemble MSE 가 3089.3 이다.

### 4.3 Weights

가중치는 non-negativity 제약 아래에서 out-of-fold 예측으로 적합하며, 학습 적합값으로도 손으로도 정하지 않는다. Non-negativity 를 건 최소제곱은 stacked regression 의 원래 처방이고, 계수를 해석 가능하게 두고 결합을 안정시키는 것이 그 제약이다 [[7](#ref-7)]. Appendix B.2 에서 적합된 가중치는 0.621, 0.162, 0.216 이며, 가중 평균이 RMSE 를 55.58 에서 55.29 로 낮춘다.

Table 3 은 가중치를 정하는 방법들을 각각이 쓰는 근거의 양 순으로 늘어놓은 것이다.

Table 3. Ways to set member weights

| Method | Fitted on | Cost | Failure mode |
|--------|-----------|------|--------------|
| Equal weights | 없음 | 없음 | 약한 member 가 평균을 끌어내림 |
| Inverse validation error | Member 별 held-out 오차 | Held-out 분할 한 번 | Member 사이 상관을 무시 |
| Non-negative least squares | Out-of-fold 예측 | Cross-validation 한 번 | Out-of-fold 집합이 작으면 overfitting |
| Meta-learner | Out-of-fold 예측 | Cross-validation 한 번 | 위와 같고, 유지할 model 이 하나 더 늘어남 |

Member 가 이미 본 행의 예측으로 가중치를 적합하는 것이 전형적인 leak 이다. 학습 행을 외운 model 은 그 행에서 완벽해 보여 가중치를 독차지하고, 그 실패는 배포 뒤에야 드러난다.

### 4.4 Log-Odds And Rank

Log-odds 를 평균하는 것은 calibration 이 어긋난 member 에 대한 잘못된 처방이며, Appendix B.3 의 실행이 그것을 보인다. Log-odds 의 평균을 logistic 함수로 되돌린 값은 odds 의 기하평균이다.

$$\bar z = \frac{1}{M}\sum_{m}\log\frac{p_m}{1 - p_m}, \qquad \bar p = \frac{1}{1 + e^{-\bar z}} \hspace{19em} (3)$$

확률은 $[0, 1]$ 안에 갇히므로 확률의 산술평균은 한 member 의 영향을 $1/M$ 로 묶는다. Log-odds 평균은 그 상한을 없앤다. $0.9999$ 를 내놓는 member 는 $z \approx 9.2$ 를 보태어 0 근처에 있는 두 member 를 이길 수 있다. Table 5 의 자료에서 정확도가 0.9298 로 떨어지고 Brier score 는 0.0572 로 거의 두 배가 되어, 시도한 결합 가운데 가장 나쁘다.

Rank 평균은 어떤 어긋남에도 살아남는 선택지인데, 각 member 가 행을 늘어놓은 순서만 쓰기 때문이다. 대신 순위에 맞춘 목적을 얻고 확률을 내어 준다. 따라서 한 행씩 답하는 것이 아니라 한 묶음을 줄 세우는 일에 쓴다.

## 5. Confidence

### 5.1 What A Model Emits

0 과 1 사이의 숫자는 무엇인가가 확인해 주기 전까지 확률이 아니다. Table 4 는 각 계열이 실제로 무엇을 내놓는지, 그 숫자를 다른 숫자와 평균하기 전에 무엇을 해야 하는지를 적은 것이다.

Table 4. What each model family emits

| Family | Native output | Shape of the error | Repair |
|--------|---------------|--------------------|--------|
| Logistic regression | 적합된 link 에서 나온 확률 | Link 가 맞으면 calibration 에 가까움 | 대개 불필요 |
| Random forest | Member 득표 비율 | 0 과 1 에서 밀려남 | Isotonic regression |
| Gradient boosting | 누적 margin 의 logistic | 과신 | Platt scaling 또는 isotonic |
| Naive Bayes | 독립 가정 아래 likelihood 의 곱 | Feature 가 종속이면 극단적으로 과신 | Isotonic regression |
| SVM | 경계까지의 부호 있는 거리, 확률 아님 | 애초에 확률 척도가 아님 | Platt scaling |
| Neural network | Logit 의 softmax | 과신하며, 망이 커질수록 심해짐 | Temperature scaling |

Table 5 의 자료에서 잰 효과는 미묘하지 않다. Naive Bayes 는 test 예측의 95.3% 를 $0.99$ 위나 $0.01$ 아래에 두는데 logistic regression 은 71.9% 이고, Brier score 는 각각 0.0760 과 0.0265 이다.

### 5.2 Calibration

Calibration 은 내놓은 점수를 확률로 옮기는 단조 사상이며, member 가 학습하지 않은 행에서 적합한다. Platt scaling 은 매개변수 하나짜리 logistic 을 적합하며 어긋남이 sigmoid 모양이라고 전제하고, isotonic regression 은 비감소 계단함수라면 무엇이든 적합하는 대신 더 많은 행을 필요로 한다 [[9](#ref-9)] [[10](#ref-10)]. 둘 다 행의 순위는 건드리지 않으므로, 고정된 threshold 에서의 정확도는 그 threshold 를 넘나드는 자리에서만 움직인다.

이 사상을 학습 행에서 적합하면 사상 자체가 망가진다. Appendix D.1 에서는 학습 집합을 다섯 겹으로 나눈 안쪽에서 적합하며, 표가 보고하는 held-out 행은 적합에 쓰지 않는다.

읽어야 할 점수는 Brier score 이며, 내놓은 확률과 결과의 제곱차 평균이다 [[11](#ref-11)]. Calibration 과 판별력에 함께 반응하므로, 손질이 도움이 되었는지를 말해 주는 유일한 수치이다. Naive Bayes 는 isotonic regression 아래에서 정확도가 그대로인 채 0.0760 에서 0.0614 로 내려간다.

### 5.3 Four Ways To Spend Confidence

확률이 결합에 들어가는 자리는 넷이며, 넷이 서로 대등하지는 않다. 정보를 얼마나 남기는지 순으로 적는다.

- **Soft voting** — 모든 행에서 member 마다 정해진 몫. 가장 값싸며, 과신하는 member 하나에 무너짐.
- **Per-row confidence weighting** — 행마다 달라지는 몫. 그 행에서 그 member 가 낸 출력의 entropy 로 정함.
- **Meta-feature** — 확률을 두 번째 model 의 열로 넣음. 가중치와 member 사이 상관을 함께 학습.
- **Abstention** — Ensemble 확률로 답할지 말지를 결정.

행별 가중은 행 $x$ 에서 member $m$ 이 낸 entropy 를 $K$ 개 class 에서의 최댓값으로 나누어 정규화한다.

$$c_m(x) = 1 - \frac{H\left(p_m(x)\right)}{\log K} \hspace{19em} (4)$$

이 식은 확신에 상을 주는데, 손대지 않은 출력에서는 그것이 정확히 반대로 작동한다. 과신하는 member 가 어디서나 가장 큰 가중치를 받기 때문이다. Appendix D.2 에서 정확도가 0.9240 까지 떨어져 그냥 soft voting 보다 낮다. Calibration 한 확률에 적용하면 0.9532 로, calibration 한 soft voting 의 0.9474 보다 높다. 규칙은 하나다. Equation (4) 는 section 5.2 를 거친 뒤에만 쓴다.

Meta-feature 경로가 가장 강하며, 그 이유는 member 들을 함께 보는 유일한 방법이기 때문이다. Label 이 아니라 confidence 로 stacking 하라는 것은 이 method 의 설계 선택을 다룬 원 연구의 결론이었고 [[8](#ref-8)], Table 5 가 그것을 재현한다. 0.9649 대 0.9591 이다. 적합된 계수 4.392, 2.678, 0.969 는 meta-learner 가 세 번째 member 를 깎아야 한다는 것을 스스로 알아낸 결과이며, 고정된 평균 규칙으로는 내릴 수 없는 판단이다.

Abstention 은 confidence 를 coverage 의 선택으로 바꾼다. 최적 규칙은 사후확률이 threshold 아래인 곳에서 답을 거절하는 것이며, 오차율과 거절률은 그 threshold 가 정하는 곡선 위에서 맞바꿔진다 [[12](#ref-12)]. Appendix D.3 이 이 자료에서 그 곡선을 잰다. Coverage 가 전부일 때 정확도 0.9474, coverage 87.1% 에서 0.9866, coverage 74.3% 에서 1.0000 이다. Ensemble 이 물러선 행이 곧 틀렸을 행이다.

### 5.4 Disagreement

Member 사이의 흩어짐이 불확실성을 재는 것은 member 가 실제로 서로 다를 때뿐이며, 이 문서의 회귀 실행이 그 반례이다. 따로 학습한 member 들은 쓸 만한 불확실성 추정이 되고 더 정교한 Bayesian 처리에 견줄 만하지만 [[14](#ref-14)], 그것은 member 들이 서로 다른 맹점을 가진다는 데에 기댄 결과이다.

Appendix B.4 에서 held-out 133 행을 member 흩어짐의 중앙값에서 둘로 나눈다. 흩어짐이 작은 쪽은 평균 흩어짐 6.14 에 RMSE 55.69 이고, 큰 쪽은 평균 흩어짐 16.92 에 RMSE 55.47 이다. 흩어짐은 세 배 가까이 차이가 나는데 오차는 움직이지 않는다. 같은 예측변수를 쓰고 같은 변수를 빠뜨린 세 member 는 같은 자리에서 나란히 확신하며 틀리고, 그 일치는 행의 어려움이 아니라 그들이 공유하는 맹점을 잰 값이다.

같은 자료에서 분류가 다르게 움직이는 이유는 section 5.3 의 abstention 신호가 member 사이의 불일치가 아니라 ensemble 자신의 사후확률이기 때문이다. 불일치는 재어 볼 값어치가 있지만, 위의 확인 없이 믿을 값어치는 없다.

## 6. Frameworks

### 6.1 Bagging

Bagging 은 행을 bootstrap 으로 다시 뽑아 member 마다 적합하고 그 결과를 평균하거나 투표하며, 분산을 낮추고 편향은 그대로 둔다 [[2](#ref-2)]. 표본이 흔들릴 때 함께 흔들리는 member 에만 값을 하므로, 가지치기하지 않은 깊은 tree 는 이득을 보고 선형 적합은 거의 보지 못한다.

Random forest 는 split 마다 열을 뽑는 단계를 더해, member 들이 지배적인 예측변수 하나를 통해 서로 같아지는 것을 막는다 [[3](#ref-3)]. Table 6 의 차이는 전부 그 한 단계에서 온다. 같은 member 와 같은 자료에서 bagging 한 tree 는 0.9298, forest 는 0.9532 이다.

### 6.2 Boosting

Boosting 은 member 를 차례로 적합하되 각각을 앞의 것들이 틀린 자리에 맞추고, 가중치를 주어 더한다. AdaBoost 는 잘못 분류된 행의 가중치를 올리고 [[4](#ref-4)], gradient boosting 은 새 member 를 손실의 gradient 에 맞추어 손실함수를 자유롭게 고를 수 있게 한다 [[5](#ref-5)]. 널리 쓰이는 구현들은 뒤쪽 형태를 물려받아 2차 정보, 희소성 처리, 메모리에 담기지 않는 자료의 학습을 더한 것이다 [[6](#ref-6)].

값은 순차성으로 치른다. Bagging 의 member 는 서로 독립이어서 나란히 적합하지만 boosting 의 member 는 그렇지 않고, held-out 곡선을 보고 조기 종료하거나 끝까지 간다. Boosting 은 bagging 과 달리 편향도 낮추므로, 어려운 target 에서는 같은 tree 를 bagging 한 ensemble 을 이길 수 있고, bagging 이라면 나지 않았을 overfitting 이 날 수도 있다.

### 6.3 Stacking

Stacking 은 member 들의 out-of-fold 예측 위에 두 번째 model 을 적합한다 [[15](#ref-15)]. Out-of-fold 가 이 방법의 전부이다. Member 가 학습한 행에 대해 내는 예측은 낙관적이며, 그런 행으로 적합한 meta-learner 는 가장 많이 외운 member 를 믿는 법을 배운다.

도움이 되는 stacking 과 leak 하는 stacking 을 가르는 규칙은 넷이다.

- Meta-feature 는 학습 집합에 대한 cross-validation 예측에서. 학습 적합값에서 만들지 않음.
- Held-out 행을 채점할 때 쓸 member 는 학습 집합 전체로 다시 적합.
- Meta-feature 로 label 이 아니라 확률 [[8](#ref-8)].
- 이진 문제에서는 열을 둘이 아니라 하나. 두 열의 합이 1 이라 공선이기 때문.

Meta-learner 는 작게 둔다. 열 셋에 정칙화한 선형 model 은 out-of-fold 행에서 계수 셋만 추정하면 되지만, 같은 열 셋에 forest 를 두면 fold 안에 없는 구조를 찾아낸다.

### 6.4 Blending

Blending 은 stacking 의 cross-validation 을 holdout 하나로 바꾼다. Member 는 학습 집합의 한쪽에서, meta-learner 는 다른 쪽에서 적합한다. Member 당 적합이 $K+1$ 번이 아니라 한 번이면 되고, 두 쪽이 만나지 않으므로 leak 이 생길 수 없다.

그 값은 두 번 치르며, Table 5 가 청구서이다. 같은 member 로 blending 은 0.9298, stacking 은 0.9649 이다. Member 는 학습 행 전부가 아니라 70% 만 보았고, meta-learner 는 stacking 이 주는 out-of-fold 예측 398 개가 아니라 120 행으로 적합되었다.

## 7. Comparison

Table 5 가 실측 비교이다. Logistic regression, tree 200 개의 random forest, Gaussian naive Bayes 세 member 를 breast cancer 자료의 398 행으로 적합하고 held-out 171 행에서 채점했으며, code 는 Appendix B, C, D, E 에 있다.

Table 5. Combination rules on 171 held-out rows

| Method | Combines | Accuracy | Brier |
|--------|----------|----------|-------|
| Stacking on probabilities | Probability | 0.9649 | 0.0275 |
| Logistic regression alone | Nothing | 0.9591 | 0.0265 |
| Hard voting | Label | 0.9591 | N/A |
| Stacking on labels | Label | 0.9591 | 0.0317 |
| Random forest alone | Nothing | 0.9532 | 0.0404 |
| Confidence weighted, calibrated | Calibrated probability | 0.9532 | 0.0360 |
| Soft voting, calibrated | Calibrated probability | 0.9474 | 0.0359 |
| Soft voting, raw | Probability | 0.9415 | 0.0373 |
| Blending | Probability | 0.9298 | 0.0406 |
| Mean of log-odds | Probability | 0.9298 | 0.0572 |
| Naive Bayes alone | Nothing | 0.9240 | 0.0760 |
| Confidence weighted, raw | Probability | 0.9240 | 0.0496 |

Hard voting 에 Brier score 가 없는 것은 확률을 내지 않기 때문이며, 그 빈칸이 label 만 쓰는 경로의 값이다. 뒤에서 threshold 를 걸 수도, abstention 을 둘 수도, 비용행렬을 쓰는 결정 규칙에 넘길 수도 없다.

Table 6 은 framework 을 따로 떼어 본 것이며, 모두 같은 분할이고 모두 그 자체로 완결된 tree ensemble 이다.

Table 6. Frameworks on the same 171 held-out rows

| Framework | Members | Accuracy | Brier |
|-----------|---------|----------|-------|
| Random forest | Tree 200 개, 행과 열을 함께 추출 | 0.9532 | 0.0404 |
| Gradient boosting | Tree 100 개, 순차 | 0.9357 | 0.0422 |
| Bagging | Tree 200 개, 행만 추출 | 0.9298 | 0.0450 |
| Single tree | 1 | 0.9064 | 0.0936 |

Table 7 이 선택 규칙이며, 왼쪽 열부터 읽는다.

Table 7. Which combination to use

| Use | When | Why |
|-----|------|-----|
| Hard voting | Member 가 확률을 내지 않거나, calibration 할 held-out 집합이 없음 | Label 말고는 아무것도 요구하지 않는 유일한 규칙 |
| Soft voting | 세기가 비슷한 member, 확률이 이미 calibration 됨 | 적합하고 유지할 두 번째 model 이 없음 |
| Weighted averaging | 회귀, 세기가 서로 다른 member | 가중치를 out-of-fold 행에서 한 번 적합 |
| Stacking | 종류가 서로 다른 member, cross-validation 을 돌릴 만한 행 수 | 가중치와 상관을 함께 학습 |
| Blending | Member 당 적합 한 번이 예산이거나 fold 가 비쌈 | Leak 없음, 대신 행을 내어 줌 |

## 8. Further Work

- **Threshold 가 아니라 보장으로 정하는 coverage** — Section 5.3 의 abstention 표는 threshold 를 바꿔 가며 결과를 읽어 coverage 를 정하므로, 답한 행에 대한 보장이 없다. Conformal prediction 은 대신 목표 오차율에서 threshold 를 정하고, 그 분포 무관 coverage 보장은 바탕 model 이 무엇이든 성립하므로 section 6 의 ensemble 을 그대로 쓸 수 있다. 배포와 같은 분포에서 떼어 두었고 그와 교환가능한 calibration 분할이 필요하며, drift 하는 공정이 깨뜨리는 조건이 바로 그것이다.
- **공정을 따라가는 가중치** — Section 4.3 의 non-negative least squares 가중치는 한 번 적합한 뒤 고정되므로, 성능이 나빠진 member 도 제 몫을 그대로 가진다. 진행 중인 손실로 member 의 가중치를 갱신하는 방식은 이제 streaming library 에서 표준이며 member 자체를 다시 적합할 필요가 없다. 지연이 제한된 label 과 가중치가 얼마나 빨리 움직일 수 있는지에 대한 규칙이 필요한데, 잡음을 쫓는 가중치는 고정된 가중치보다 나쁘기 때문이다.

## References

<a id="ref-1"></a>[1] Krogh, A. and Vedelsby, J. (1995). [Neural Network Ensembles, Cross Validation, and Active Learning](https://proceedings.neurips.cc/paper/1994/hash/b8c37e33defde51cf91e1e03e51657da-Abstract.html). *Advances in Neural Information Processing Systems*, 7, 231-238.<br>
<a id="ref-2"></a>[2] Breiman, L. (1996). [Bagging Predictors](https://doi.org/10.1007/BF00058655). *Machine Learning*, 24(2), 123-140.<br>
<a id="ref-3"></a>[3] Breiman, L. (2001). [Random Forests](https://doi.org/10.1023/A:1010933404324). *Machine Learning*, 45(1), 5-32.<br>
<a id="ref-4"></a>[4] Freund, Y. and Schapire, R. E. (1997). [A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting](https://doi.org/10.1006/jcss.1997.1504). *Journal of Computer and System Sciences*, 55(1), 119-139.<br>
<a id="ref-5"></a>[5] Friedman, J. H. (2001). [Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451). *The Annals of Statistics*, 29(5), 1189-1232.<br>
<a id="ref-6"></a>[6] Chen, T. and Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785). *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.<br>
<a id="ref-7"></a>[7] Breiman, L. (1996). [Stacked Regressions](https://doi.org/10.1007/BF00117832). *Machine Learning*, 24(1), 49-64.<br>
<a id="ref-8"></a>[8] Ting, K. M. and Witten, I. H. (1999). [Issues in Stacked Generalization](https://doi.org/10.1613/jair.594). *Journal of Artificial Intelligence Research*, 10, 271-289.<br>
<a id="ref-9"></a>[9] Niculescu-Mizil, A. and Caruana, R. (2005). [Predicting Good Probabilities with Supervised Learning](https://doi.org/10.1145/1102351.1102430). *Proceedings of the 22nd International Conference on Machine Learning*, 625-632.<br>
<a id="ref-10"></a>[10] Zadrozny, B. and Elkan, C. (2002). [Transforming Classifier Scores into Accurate Multiclass Probability Estimates](https://doi.org/10.1145/775047.775151). *Proceedings of the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 694-699.<br>
<a id="ref-11"></a>[11] Brier, G. W. (1950). [Verification of Forecasts Expressed in Terms of Probability](https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml). *Monthly Weather Review*, 78(1), 1-3.<br>
<a id="ref-12"></a>[12] Chow, C. K. (1970). [On Optimum Recognition Error and Reject Tradeoff](https://doi.org/10.1109/TIT.1970.1054406). *IEEE Transactions on Information Theory*, 16(1), 41-46.<br>
<a id="ref-13"></a>[13] Kuncheva, L. I. and Whitaker, C. J. (2003). [Measures of Diversity in Classifier Ensembles and Their Relationship with the Ensemble Accuracy](https://doi.org/10.1023/A:1022859003006). *Machine Learning*, 51(2), 181-207.<br>
<a id="ref-14"></a>[14] Lakshminarayanan, B., Pritzel, A. and Blundell, C. (2017). [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html). *Advances in Neural Information Processing Systems*, 30, 6402-6413.<br>
<a id="ref-15"></a>[15] Wolpert, D. H. (1992). [Stacked Generalization](https://doi.org/10.1016/S0893-6080(05)80023-1). *Neural Networks*, 5(2), 241-259.

---

## Appendix A. Terminology

- **abstention**: Ensemble 확률이 threshold 아래인 행에 대해 답하기를 거절하는 것.
- **ambiguity**: Member 들과 그 평균 사이 제곱거리의 평균이며, equation (1) 의 둘째 항.
- **bagging**: 행을 bootstrap 으로 다시 뽑아 member 마다 적합하는 framework.
- **base model**: Ensemble 안의 적합된 model 하나이며 member 라고도 한다.
- **blending**: Meta-feature 를 cross-validation fold 가 아니라 holdout 분할 하나에서 얻는 stacking.
- **boosting**: Member 를 차례로 적합하되 각각을 앞의 것들이 틀린 자리에 맞추는 framework.
- **Brier score**: 내놓은 확률과 결과의 제곱차 평균.
- **calibration**: 내놓은 점수를 확률로 옮기는 단조 사상이며, member 가 학습하지 않은 행에서 적합한다.
- **confidence**: Model 이 label 과 함께 내놓는, 얼마나 확신하는지를 나타내는 수치.
- **conformal prediction**: 어떤 model 의 점수든 coverage 가 미리 정해진 예측 집합으로 바꾸는 절차.
- **coverage**: Abstention 을 두는 ensemble 이 실제로 답하는 행의 비율.
- **ensemble**: 하나의 결합 규칙으로 함께 쓰이는 여러 개의 적합된 model.
- **hard voting**: Member label 의 다수결을 취하는 결합.
- **isotonic regression**: 비감소 계단함수라면 무엇이든 될 수 있는 calibration 사상.
- **leak**: Model 이 이미 학습 자료로 본 값을 적합이나 평가에 쓰는 것.
- **meta-learner**: Stacking 의 두 번째 model 이며, member 들의 out-of-fold 예측으로 적합한다.
- **out-of-fold prediction**: 그 행을 학습하지 않은 member 가 그 행에 대해 내놓은 예측.
- **Platt scaling**: 매개변수 하나짜리 logistic 인 calibration 사상.
- **soft voting**: Member 확률 평균의 argmax 를 취하는 결합.
- **stacking**: Member 들의 out-of-fold 예측 위에 meta-learner 를 적합하는 framework.
- **temperature scaling**: Logit 을 적합된 scalar 하나로 나누는 calibration 사상.

## Appendix B. Voting And Averaging

### B.1 Hard And Soft Voting

종류가 서로 다른 세 member 를 breast cancer 자료에 적합하고, 같은 적합에 두 투표 규칙을 적용한다. 정확도는 Table 5 의 첫 행들이다.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0)

models = {
    'logistic': make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    'forest': RandomForestClassifier(n_estimators=200, random_state=0),
    'naive_bayes': GaussianNB(),
}
for model in models.values():
    model.fit(X_tr, y_tr)

# hard voting: the majority of the labels, nothing else
labels = np.column_stack([m.predict(X_te) for m in models.values()])
hard = (labels.mean(axis=1) > 0.5).astype(int)

# soft voting: the argmax of the mean probability
probs = np.stack([m.predict_proba(X_te) for m in models.values()])
soft = probs.mean(axis=0).argmax(axis=1)

for name, model in models.items():
    print(f'{name:12s} {accuracy_score(y_te, model.predict(X_te)):.4f}')
print(f'hard voting  {accuracy_score(y_te, hard):.4f}')
print(f'soft voting  {accuracy_score(y_te, soft):.4f}')
```

```text
logistic     0.9591
forest       0.9532
naive_bayes  0.9240
hard voting  0.9591
soft voting  0.9415
```

### B.2 Weighted Averaging With Out-Of-Fold Weights

회귀 쪽 대응물이다. 가중치는 out-of-fold 예측에 대한 non-negative least squares 에서 나오며, 마지막 두 줄이 equation (1) 을 수치로 확인한다.

```python
import numpy as np
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X_reg, y_reg = load_diabetes(return_X_y=True)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=0)

members = {
    'ridge': make_pipeline(StandardScaler(), RidgeCV()),
    'forest': RandomForestRegressor(n_estimators=300, random_state=0),
    'knn': make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15)),
}
cv = KFold(n_splits=5, shuffle=True, random_state=0)
oof = np.column_stack([
    cross_val_predict(clone(m), Xr_tr, yr_tr, cv=cv) for m in members.values()])
for model in members.values():
    model.fit(Xr_tr, yr_tr)
pred = np.column_stack([m.predict(Xr_te) for m in members.values()])

def rmse(p):
    return np.sqrt(mean_squared_error(yr_te, p))


simple = pred.mean(axis=1)
weight, _ = nnls(oof, yr_tr)     # weights never see a row the member trained on
weight = weight / weight.sum()

for i, name in enumerate(members):
    print(f'{name:8s} rmse {rmse(pred[:, i]):.2f}')
print(f'simple   rmse {rmse(simple):.2f}')
print(f'weighted rmse {rmse(pred @ weight):.2f}  '
      f'weights {dict(zip(members, weight.round(3).tolist()))}')

# the ambiguity decomposition of equation (1), checked on the same rows
mean_member = np.mean([mean_squared_error(yr_te, pred[:, i])
                       for i in range(pred.shape[1])])
ambiguity = np.mean((pred - simple[:, None]) ** 2)
print(f'{mean_member:.1f} - {ambiguity:.1f} = {mean_member - ambiguity:.1f}, '
      f'ensemble mse {mean_squared_error(yr_te, simple):.1f}')
```

```text
ridge    rmse 55.67
forest   rmse 59.42
knn      rmse 56.32
simple   rmse 55.58
weighted rmse 55.29  weights {'ridge': 0.621, 'forest': 0.162, 'knn': 0.216}
3267.2 - 177.9 = 3089.3, ensemble mse 3089.3
```

### B.3 Averaging Log-Odds

B.1 의 member 에 equation (3) 을 적용한 것이며, member 의 영향에 걸린 상한을 없애면 여기서 정확도를 잃는 이유를 보인다.

```python
from sklearn.metrics import brier_score_loss

EPS = 1e-6
p = np.clip(np.stack([m.predict_proba(X_te)[:, 1] for m in models.values()]),
            EPS, 1 - EPS)

arithmetic = p.mean(axis=0)
geometric = 1 / (1 + np.exp(-np.log(p / (1 - p)).mean(axis=0)))

for name, q in (('arithmetic', arithmetic), ('log-odds', geometric)):
    print(f'{name:11s} {accuracy_score(y_te, (q > 0.5).astype(int)):.4f}  '
          f'brier {brier_score_loss(y_te, q):.4f}')
```

```text
arithmetic  0.9415  brier 0.0373
log-odds    0.9298  brier 0.0572
```

### B.4 Member Spread As An Uncertainty Signal

B.2 의 회귀 member 에 대한 section 5.4 의 확인이다. 흩어짐은 두 쪽을 세 배 가까이 갈라 놓지만 오차는 따라오지 않는다.

```python
spread = pred.std(axis=1)
order = np.argsort(spread)
half = len(order) // 2

for label, idx in (('smallest', order[:half]), ('largest', order[half:])):
    error = np.sqrt(mean_squared_error(yr_te[idx], simple[idx]))
    print(f'{label:9s} n {len(idx)}  rmse {error:.2f}  '
          f'mean spread {spread[idx].mean():.2f}')
```

```text
smallest  n 66  rmse 55.69  mean spread 6.14
largest   n 67  rmse 55.47  mean spread 16.92
```

## Appendix C. Stacking And Blending

### C.1 Stacking On Probabilities Against Labels

Table 5 의 맨 위 행과 가운데 행 뒤에 있는 비교이다. 두 stacking 의 차이는 meta-learner 에 무엇을 먹이는가 하나뿐이다.

```python
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# one column per member, not two: the pair sums to one and is collinear
meta_tr = np.column_stack([
    cross_val_predict(clone(m), X_tr, y_tr, cv=cv, method='predict_proba')[:, 1]
    for m in models.values()])
meta_te = np.column_stack([m.predict_proba(X_te)[:, 1] for m in models.values()])

meta = LogisticRegression(max_iter=5000).fit(meta_tr, y_tr)
print(f'on probabilities {accuracy_score(y_te, meta.predict(meta_te)):.4f}  '
      f'brier {brier_score_loss(y_te, meta.predict_proba(meta_te)[:, 1]):.4f}')
print('coefficients', dict(zip(models, meta.coef_[0].round(3).tolist())))

# the same stack fed only the labels
label_tr = np.column_stack([
    cross_val_predict(clone(m), X_tr, y_tr, cv=cv) for m in models.values()])
label_te = np.column_stack([m.predict(X_te) for m in models.values()])
meta_label = LogisticRegression(max_iter=5000).fit(label_tr, y_tr)
print(f'on labels        {accuracy_score(y_te, meta_label.predict(label_te)):.4f}  '
      f'brier {brier_score_loss(y_te, meta_label.predict_proba(label_te)[:, 1]):.4f}')
```

```text
on probabilities 0.9649  brier 0.0275
coefficients {'logistic': 4.392, 'forest': 2.678, 'naive_bayes': 0.969}
on labels        0.9591  brier 0.0317
```

### C.2 Blending

K 개의 fold 대신 분할 하나를 쓴다. Member 는 학습 행의 70% 를 보고, meta-learner 는 그들이 보지 않은 120 행으로 적합한다.

```python
X_fit, X_bl, y_fit, y_bl = train_test_split(
    X_tr, y_tr, test_size=0.3, stratify=y_tr, random_state=1)

blend_members = [clone(m).fit(X_fit, y_fit) for m in models.values()]
blend_tr = np.column_stack([m.predict_proba(X_bl)[:, 1] for m in blend_members])
blend_te = np.column_stack([m.predict_proba(X_te)[:, 1] for m in blend_members])

blender = LogisticRegression(max_iter=5000).fit(blend_tr, y_bl)
print(f'blending {accuracy_score(y_te, blender.predict(blend_te)):.4f}  '
      f'brier {brier_score_loss(y_te, blender.predict_proba(blend_te)[:, 1]):.4f}  '
      f'meta rows {len(y_bl)}')
```

```text
blending 0.9298  brier 0.0406  meta rows 120
```

## Appendix D. Confidence

### D.1 Calibration Before Averaging

사상은 학습 집합을 다섯 겹으로 나눈 안쪽에서 적합하므로, held-out 행은 그 적합에 끼지 않는다. 마지막 두 줄이 Table 5 의 soft voting 행이다.

```python
from sklearn.calibration import CalibratedClassifierCV

for name, model in models.items():
    p = model.predict_proba(X_te)[:, 1]
    print(f'{name:12s} brier {brier_score_loss(y_te, p):.4f}  '
          f'extreme {np.mean(np.abs(p - 0.5) > 0.49):.3f}')

calibrated = {name: CalibratedClassifierCV(m, method='isotonic', cv=5).fit(X_tr, y_tr)
              for name, m in models.items()}
for name, model in calibrated.items():
    print(f'{name:12s} brier {brier_score_loss(y_te, model.predict_proba(X_te)[:, 1]):.4f}'
          f'  (calibrated)')

raw = np.stack([m.predict_proba(X_te) for m in models.values()]).mean(axis=0)
cal = np.stack([m.predict_proba(X_te) for m in calibrated.values()]).mean(axis=0)
for name, q in (('raw', raw), ('calibrated', cal)):
    print(f'soft voting, {name:11s} {accuracy_score(y_te, q.argmax(axis=1)):.4f}  '
          f'brier {brier_score_loss(y_te, q[:, 1]):.4f}')
```

```text
logistic     brier 0.0265  extreme 0.719
forest       brier 0.0404  extreme 0.532
naive_bayes  brier 0.0760  extreme 0.953
logistic     brier 0.0301  (calibrated)
forest       brier 0.0419  (calibrated)
naive_bayes  brier 0.0614  (calibrated)
soft voting, raw         0.9415  brier 0.0373
soft voting, calibrated  0.9474  brier 0.0359
```

### D.2 Per-Row Confidence Weighting

손대지 않은 확률과 calibration 한 확률에 equation (4) 를 각각 적용한 것이며, 가중이 calibration 뒤에 와야 함을 보인다.

```python
def confidence_weighted(probs):
    """Weight each member on each row by 1 - normalized entropy of its own output."""
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=2)
    confidence = 1 - entropy / np.log(probs.shape[2])
    weight = confidence / confidence.sum(axis=0, keepdims=True)
    return (probs * weight[:, :, None]).sum(axis=0)

raw_stack = np.stack([m.predict_proba(X_te) for m in models.values()])
cal_stack = np.stack([m.predict_proba(X_te) for m in calibrated.values()])

for name, stack in (('raw', raw_stack), ('calibrated', cal_stack)):
    q = confidence_weighted(stack)
    print(f'{name:11s} {accuracy_score(y_te, q.argmax(axis=1)):.4f}  '
          f'brier {brier_score_loss(y_te, q[:, 1]):.4f}')
```

```text
raw         0.9240  brier 0.0496
calibrated  0.9532  brier 0.0360
```

### D.3 Abstention

Section 5.3 의 오차 대 거절 맞바꿈을 calibration 한 ensemble 에서 잰 것이다.

```python
p = cal_stack.mean(axis=0)
top = p.max(axis=1)
pred_label = p.argmax(axis=1)

print('threshold  coverage  accuracy')
for threshold in (0.50, 0.90, 0.99, 0.999):
    keep = top >= threshold
    print(f'{threshold:9.3f}  {keep.mean():8.3f}  '
          f'{accuracy_score(y_te[keep], pred_label[keep]):.4f}')
```

```text
threshold  coverage  accuracy
    0.500     1.000  0.9474
    0.900     0.871  0.9866
    0.990     0.743  1.0000
    0.999     0.152  1.0000
```

## Appendix E. Frameworks

Table 6 을 B.1 의 분할에서 잰 것이며, framework 과 결합 규칙이 같은 171 행에서 채점되도록 했다. 아래의 각각은 그 자체로 완결된 tree ensemble 이어서 따로 aggregation 단계를 두지 않는다.

```python
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

frameworks = {
    'single tree': DecisionTreeClassifier(random_state=0),
    'bagging': BaggingClassifier(DecisionTreeClassifier(random_state=0),
                                 n_estimators=200, random_state=0),
    'random forest': RandomForestClassifier(n_estimators=200, random_state=0),
    'boosting': GradientBoostingClassifier(random_state=0),
}
for name, clf in frameworks.items():
    clf.fit(X_tr, y_tr)
    print(f'{name:14s} acc {accuracy_score(y_te, clf.predict(X_te)):.4f}  '
          f'brier {brier_score_loss(y_te, clf.predict_proba(X_te)[:, 1]):.4f}')
```

```text
single tree    acc 0.9064  brier 0.0936
bagging        acc 0.9298  brier 0.0450
random forest  acc 0.9532  brier 0.0404
boosting       acc 0.9357  brier 0.0422
```
