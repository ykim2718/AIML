# Within-Wafer and Wafer-to-Wafer Variance Decomposition
Rev. 97 | Created: 2026-09-01 | Updated: 2026-09-22 00:53 CDT

> ANOVA (analysis of variance) 는 관측치의 전체 산포를 몇 개의 원인으로 나누어, 어느 원인이 얼마나 기여하는지 수치로 보이는 방법이다.

측정값이 여러 층으로 묶여 있을 때 각 층이 산포에 얼마나 기여하는지는 눈으로 가려낼 수 없다. ANOVA 는 전체 제곱합을 층별 제곱합으로 쪼개어 이 물음에 답한다. 한 층 안에서 값이 흩어진 정도와 층 사이에서 평균이 벌어진 정도를 각각의 자유도로 나누어 평균제곱으로 만들고, 그 비를 F 통계량으로 삼아 층 사이의 차이가 층 안의 산포만으로 설명되는지 판정한다. 이 문서는 wafer 를 층으로 두어 측정값의 산포를 within-wafer 성분과 wafer-to-wafer 성분으로 나눈다.

## 1. Theory

### 1.1 Notation

Wafer 를 장당 여러 site 에서 재어 얻은 표를 아래 기호로 적는다. 항목마다 그 값이 어디서 오는지를 설계값, 측정값, 관측값, 계산값으로 적었다. 측정값은 계측기가 읽은 $`X_{ij}`$ 하나이고, 관측값은 그것을 단순 계산한 것이며, 계산값은 분산성분 모형을 거친 것이다.

- $`K`$: wafer 장수. 이 자료에서는 261. 설계값.
- $`N`$: wafer 한 장에서 재는 site 개수. 이 자료에서는 13. 설계값.
- $`M`$: 전체 관측치 개수이며 $`M = K N`$ 이다. 설계값.
- $`X_{ij}`$: $`i`$ 번째 wafer 의 $`j`$ 번째 site 에서 읽은 값. 측정값.
- $`\bar{X}_i`$: $`i`$ 번째 wafer 의 평균. 관측값.
- $`\bar{X}`$: 전체 $`M`$ 개의 총평균. 관측값.
- $`s_i`$: $`i`$ 번째 wafer 한 장의 site 값 $`N`$ 개로 계산한 표본표준편차. 관측값.
- $`\sigma_{\mu_n}`$: 처음 $`n`$ 장의 wafer 평균을 표본표준편차로 잰 값. 관측값.
- $`\hat{\sigma}_{\mu_K}`$: wafer 평균의 산포를 분산성분에서 얻은 값. 식 (13) 부터 식 (16) 까지의 좌변이다. 계산값.
- $`s_i^2`$: $`i`$ 번째 wafer 안 site 값의 표본분산. within-wafer 성분. 관측값.
- $`S_{\mathrm{total}}^2`$: 전체 $`M`$ 개의 표본분산. 관측값.

### 1.2 Decomposition Identity

전체 제곱합은 wafer 안의 편차와 wafer 평균의 편차로 남김없이 갈라진다. 이것이 ANOVA 가 딛는 항등식이다.

$$\mathrm{SST} = \mathrm{SSW} + \mathrm{SSB} \hspace{19em} (1)$$

- SST: total sum of squares. 전체 변동. 모든 관측치가 총평균에서 벗어난 정도.
- SSW: within-group sum of squares. wafer 내 변동. 각 site 값이 제 wafer 평균에서 벗어난 정도. 모형이 설명하지 못하고 남은 몫이므로 SSE (error sum of squares) 로도 쓴다.
- SSB: between-group sum of squares. wafer 간 변동. 각 wafer 평균이 총평균에서 벗어난 정도. 인자가 설명하는 몫이므로 SSA (factor sum of squares) 로도 쓴다.

세 제곱합을 풀어쓰면 아래와 같다.

$$\sum_{i}\sum_{j} (X_{ij} - \bar{X})^2 = \sum_{i}\sum_{j} (X_{ij} - \bar{X}_i)^2 + N \sum_{i} (\bar{X}_i - \bar{X})^2 \hspace{19em} (2)$$

각 제곱합을 제 자유도로 나누면 평균제곱 (mean square, MS) 이 되고, 그것이 곧 분산이다. 우변의 두 항을 각각 within-wafer 분산의 평균과 wafer 평균의 분산으로 바꾸면 아래와 같다.

$$\overline{S_{\mathrm{within}}^2} = \frac{1}{K} \sum_{i=1}^{K} s_i^2, \qquad S_{\mathrm{between}}^2 = \frac{1}{K-1} \sum_{i=1}^{K} (\bar{X}_i - \bar{X})^2 \hspace{19em} (3)$$

$$S_{\mathrm{total}}^2 = \frac{K(N-1)}{M-1} \overline{S_{\mathrm{within}}^2} + \frac{N(K-1)}{M-1} S_{\mathrm{between}}^2 \hspace{19em} (4)$$

두 계수는 $`K`$ 와 $`N`$ 이 커질수록 1 에 가까워지므로, 흔히 쓰는 형태는 계수를 떼어낸 아래 근사식이다. 계수가 1 로 가는 과정은 [B.1](#b1-the-two-coefficients) 에 적었다.

$$S_{\mathrm{total}} \approx \sqrt{\overline{S_{\mathrm{within}}^2} + S_{\mathrm{between}}^2} \hspace{19em} (5)$$

### 1.3 Interpretation

- $`S_{\mathrm{between}}^2 = 0`$ 일 때: wafer 평균이 모두 같은 경우이며, 전체 표준편차는 wafer 내 표준편차의 제곱평균제곱근으로 줄어든다.
- $`S_{\mathrm{between}}^2 \gt 0`$ 일 때: wafer 내 표준편차가 아무리 작아도 wafer 평균이 서로 벌어져 있으면 전체 표준편차는 개별 wafer 의 표준편차보다 훨씬 커진다.
- 공정 관리에서의 쓰임: 전체 산포를 wafer 내 균일도 문제와 wafer 간 재현성 문제로 갈라 원인을 찾는 것.

## 2. Data

측정 자료는 [example.csv](example.csv) 이며 261 행 14 열이다. 한 행이 한 장의 wafer 이고, 열 `wafer_id` 는 `wf0001` 부터 `wf0261` 까지의 일련번호로 파일의 행 순서, 곧 run order 를 나타낸다. 나머지 열 `S1`~`S13` 은 그 wafer 위의 13 개 site 이다. 결측은 없고 전체 관측치는 3393 개이다.

- 전체 site 값: 평균 622.1, 표준편차 32.45, 최소 435.10, 최대 734.68.
- Wafer 평균: 최소 452.9, 최대 705.0, 표준편차 28.70.
- Within-wafer range: 평균 41.32, 최대 123.46.
- Wafer uniformity $`s_i / \bar{X}_i`$: 중앙값 1.81%, 최소 0.87% (wf0033), 최대 8.62% (wf0011).

Wafer 한 장을 violin 하나로 두고 run order 로 늘어놓으면, 분포의 위치와 폭이 wafer 마다 함께 움직이는 것이 보인다. 앞쪽 wafer 는 610 대에 모여 있다가 뒤쪽에서 650 근처까지 올라가고, 아래로 홀로 처진 wafer 는 그 자리에서 값이 크게 낮았다는 뜻이다. Wafer 당 site 가 13 개뿐이라 violin 의 모양 자체는 거칠어서 site 값 13 점을 그대로 겹쳐 찍었다. 겹쳐 그린 선은 wafer 평균을 이은 것으로, 위치가 wafer 마다 얼마나 튀는지 보여준다.

<img src="wiw-w2w-anova_fig/site_value_violin.png" width="900" style="max-width: 100%;" alt="Fig 1">

Fig 1. Distribution of the site values on each wafer along run order, with the wafer means traced

## 3. Variance Decomposition

Wafer 를 인자로 둔 일원 ANOVA 로 wafer 간 성분과 wafer 내 성분을 나눈다.

Table 1. One-way ANOVA with wafer as the factor

| Source | SS | df | MS | F | p | Sigma component |
|---|---:|---:|---:|---:|---:|---:|
| Between wafer | 2,783,290 | 260 | 10,705.0 | 42.48 | ~0 | $`\sigma_{between} = \sqrt{(10705.0 - 252.0)/13} = 28.36`$ |
| Within wafer | 789,202 | 3132 | 252.0 | | | $`\sigma_{within} = \sqrt{252.0} = 15.87`$ |

표의 각 열이 뜻하는 바는 아래와 같다.

- SS: sum of squares. Between wafer 행이 section 1.2 의 SSB, within wafer 행이 SSW 이며, 둘을 더하면 SST 3,572,492 가 된다.
- df: degrees of freedom. 그 제곱합이 담은 독립한 정보의 개수. Wafer 261 장이므로 between 은 260, wafer 마다 site 13 개에서 평균 하나를 뺀 12 를 261 배 하여 within 은 3132.
- MS: mean square. SS 를 df 로 나눈 값이며 분산의 추정치. Within 의 252.0 은 site 한 점의 산포, between 의 10,705.0 은 wafer 평균의 산포에 site 산포가 얹힌 크기.
- F: 두 MS 의 비. 여기서는 10,705.0 / 252.0 = 42.48. wafer 사이에 차이가 없다면 1 근처에 머무는 값.
- p: wafer 사이에 차이가 없다는 가정 아래 그만큼 큰 F 가 나올 확률. 여기서는 0 에 가까워, 차이가 없다는 가정을 버린다.
- Sigma component: 그 행이 내는 분산성분의 표준편차. Within 은 MS within 의 제곱근이고, between 은 MS between 에서 MS within 을 빼고 site 수 13 으로 나눈 뒤 제곱근을 취한 값이다.

Table 2. Variance components

| Component | Sigma | Variance | Share |
|---|---:|---:|---:|
| Wafer-to-wafer | 28.36 | 804.1 | 76.1% |
| Within-wafer | 15.87 | 252.0 | 23.9% |
| Total | 32.50 | 1056.1 | 100% |

두 성분을 더한 32.50 은 section 2 의 관측 표준편차 32.45 와 0.05 만큼 다르다. Section 1.2 에서 본 대로 두 성분의 단순 합은 근사식이고, 정확한 관계에는 1 보다 작은 계수가 붙기 때문이다.

ICC (intraclass correlation) 는 전체 분산 중 wafer 간 분산이 차지하는 비율로, 804.1 / 1056.1 = 0.761 이다. 값이 1 에 가까울수록 같은 wafer 에서 뽑은 두 site 값이 서로 닮았다는 뜻이고, 0 에 가까울수록 어느 wafer 에서 뽑았는지가 값을 예측하는 데 도움이 되지 않는다는 뜻이다. 0.761 은 site 한 점의 산포 중 76.1% 를 그 점이 놓인 wafer 가 결정한다는 것이므로, 산포를 줄이려면 site 단위 균일도보다 wafer 단위 조건을 먼저 봐야 한다.

Table 2 의 두 성분은 261 장 전체를 한 번에 본 값이다. Wafer 한 장에서는 wafer 간 변동을 잴 수 없으므로, 창의 왼쪽 끝을 첫 wafer 에 고정하고 오른쪽 끝만 한 장씩 늘리며 (expanding window) 창마다 두 성분을 다시 구하면 그 값이 몇 장째에 자리를 잡는지 보인다. 두 성분 모두 앞쪽 몇십 장에서 크게 흔들리다가 (w2w 는 $`n = 14`$ 에서 36.55 까지 치솟는다) 표본이 쌓이면서 잦아들어, $`n = 261`$ 에서 각각 28.36 과 15.87 로 Table 2 의 값에 닿는다. $`n \ge 100`$ 에서 WiW 는 12.78~15.88 안에 머물러 일찍 안정되지만, w2w 는 18.28 에서 28.36 으로 계속 올라간다 — 뒤쪽 wafer 가 앞쪽과 다른 수준에 있었다는 뜻이며, 그래서 wafer 간 산포는 표본을 더 모을수록 커진다.

## 4. Cumulative Standard Deviation and WiW Excursion Detection

### 4.1 Formula and Its Closed Forms

처음 $`n`$ 장의 wafer 평균으로 계산한 표준편차 $`\sigma_{\mu_n}`$ 을 구하려고 한다. One-way random effects model 의 표준 표기로, 총평균과 group 의 몫을 갈라 적는다. 총평균을 $`\mu`$, wafer $`i`$ 의 wafer effect 를 $`\alpha_i`$, within-wafer site 오차를 $`e_{ij}`$ 로 두면 측정값은 세 항의 합이다.

$$X_{ij} = \mu + \alpha_i + e_{ij} \hspace{19em} (6)$$

$`\mu`$ 는 wafer 와 무관한 상수이고, $`\mu + \alpha_i`$ 는 wafer $`i`$ 한 장의 참 평균, 곧 site 오차가 없었다면 그 wafer 의 모든 site 가 가리켰을 값이고, $`\alpha_i`$ 는 그 값이 총평균에서 벗어난 양이다. 장마다 공정 조건이 달라 $`\alpha_i`$ 가 wafer 마다 다르며, one-way random effects model 은 $`\alpha_i`$ 를 고정된 상수가 아니라 평균 0 으로 wafer 마다 새로 뽑히는 확률변수로 둔다. 그래서 $`\mathrm{Var}(\alpha_i)`$ 라는 양이 정의된다.

$`\alpha_i`$ 와 $`e_{ij}`$ 는 각각 평균이 0 이고, $`e_{ij}`$ 는 $`\alpha_i`$ 와도 같은 wafer 의 다른 site 오차와도 독립이다. 두 확률변수의 variance 가 이 문서가 나누려는 두 성분이다.

$$E[\alpha_i] = 0, \quad \mathrm{Var}(\alpha_i) = \sigma_{between}^2, \qquad E[e_{ij}] = 0, \quad \mathrm{Var}(e_{ij}) = \sigma_{within}^2 \hspace{19em} (7)$$

식 (7) 이 붙인 $`\sigma_{between}^2`$ 이라는 이름을 자료에서 재려면 관측되는 양과 이어야 하며, $`\alpha_i`$ 는 관측되지 않으므로 그 연결을 같은 wafer 두 site 값의 covariance 에서 찾는다. 총평균 $`\mu`$ 는 상수라 covariance 에 들어가지 않으므로, 같은 wafer 의 두 site $`j`$ 와 $`j'`$ 가 함께 지니는 항은 $`\alpha_i`$ 뿐이다. Covariance 를 bilinear 로 펼치면 네 항이 나온다. 둘째와 셋째 항은 within-wafer site 오차가 wafer effect 와 독립이라 0 이고, 넷째 항은 같은 wafer 의 서로 다른 두 site 오차가 서로 독립이라 0 이다. 남는 것은 첫째 항 $`\mathrm{Cov}(\alpha_i, \alpha_i) = \mathrm{Var}(\alpha_i)`$ 이다. 두 인자가 같은 covariance 가 variance 가 되는 과정은 [Appendix E](#appendix-e-covariance-with-a-repeated-argument) 에 적었다.

$$\mathrm{Cov}(X_{ij}, X_{ij'}) = \mathrm{Cov}(\alpha_i, \alpha_i) + \mathrm{Cov}(\alpha_i, e_{ij'}) + \mathrm{Cov}(e_{ij}, \alpha_i) + \mathrm{Cov}(e_{ij}, e_{ij'}) = \mathrm{Cov}(\alpha_i, \alpha_i) = \mathrm{Var}(\alpha_i) = \sigma_{between}^2 \hspace{19em} (8)$$

식 (8) 은 같은 wafer 의 두 site 가 얼마나 닮았는지를 재지만, 그 값이 곧 wafer 끼리 얼마나 벌어졌는지를 재는 값이다. $`\mathrm{Var}(\alpha_i)`$ 는 $`i`$ 가 바뀔 때, 곧 wafer 가 바뀔 때 $`\alpha_i`$ 가 흩어지는 양이며, 한 wafer 안에서 $`\alpha_i`$ 는 고정된 한 값이다. 그 wafer 의 $`\alpha_i`$ 가 크면 두 site 값은 둘 다 같은 크기만큼 총평균 위로 올라가고, 작으면 둘 다 같은 크기만큼 내려간다. 두 값을 갈라놓는 것은 각자의 site 오차 $`e_{ij}`$ 와 $`e_{ij'}`$ 뿐이다. 따라서 $`\alpha_i`$ 의 산포가 site 오차보다 클수록 두 값은 공통으로 움직인 몫이 커져 더 닮는다. 그래서 section 3 의 ICC 는 한 wafer 안 두 site 의 상관계수이면서 동시에 전체 분산 중 wafer 간 분산의 비율 $`\sigma_{between}^2 / S_{\mathrm{total}}^2`$ 이다.

$`\alpha_i`$ 와 $`e_{ij}`$ 가 독립이므로 측정값의 variance 는 식 (7) 의 두 variance 의 합이다.

$$S_{\mathrm{total}}^2 = \sigma_{between}^2 + \sigma_{within}^2 \hspace{19em} (9)$$

관측한 wafer 평균은 wafer 의 참 평균 $`\mu + \alpha_i`$ 에 site 오차의 평균 $`\bar{e}_i`$ 가 얹힌 값이다. 그 오차는 site $`N`$ 개를 평균한 것이라 분산이 $`N`$ 분의 1 로 줄어든다.

$$\bar{X}_i = \mu + \alpha_i + \bar{e}_i, \qquad \mathrm{Var}(\bar{e}_i) = \frac{\sigma_{within}^2}{N} \hspace{19em} (10)$$

$`\alpha_i`$ 와 $`\bar{e}_i`$ 는 독립이므로 처음 $`n`$ 장의 wafer 평균의 분산은 두 분산의 합이고, 여기서 $`s_{\mu}(1..n)`$ 은 처음 $`n`$ 장의 wafer effect 의 표준편차이다.

$$\mathrm{Var}(\bar{X}_1, \dots, \bar{X}_n) = s_{\mu}^2(1..n) + \frac{\sigma_{within}^2}{N} \hspace{19em} (11)$$

제곱근을 취하면 관측값을 설명하는 식이 된다.

$$\sigma_{\mu_n} = \sqrt{\frac{\sigma_{within}^2}{N} + s_{\mu}^2(1..n)} \hspace{19em} (12)$$

처음 $`n`$ 장에서 얻은 관측값 $`s_{\mu}(1..n)`$ 이 전체에서 얻은 계산값 $`\sigma_{between}`$ 과 같을 경우, 곧 $`s_{\mu}^2(1..n) = \sigma_{between}^2`$ 일 경우에 식 (12) 의 오른쪽 항을 $`\sigma_{between}^2`$ 으로 바꿔 쓸 수 있다. 이때 이 조건을 만족하는 $`n`$ 을 $`K`$ 로 하여, 식 (13) 은 아래 첨자를 $`\mu_n`$ 이 아니라 $`\mu_K`$ 로 쓴다.

$$\hat{\sigma}_{\mu_K} = \sqrt{\frac{\sigma_{within}^2}{N} + \sigma_{between}^2} \hspace{19em} (13) 🌳$$

$`\sigma_{between}^2 = S_{\mathrm{total}}^2 - \sigma_{within}^2`$ 은 식 (9) 를 옮겨 적은 것이라 $`n`$ 과 무관하게 성립한다. 이 항등식을 식 (13) 에 넣어 $`\sigma_{between}^2`$ 자리를 전체 표준편차로 바꾼 것이 식 (14) 이다. 식 (14) 의 오른쪽 형태는 section 3 의 ICC 를 쓴 것이며, 그 정의는 $`\mathrm{ICC} = \sigma_{between}^2 / S_{\mathrm{total}}^2`$ 이다.

$$\hat{\sigma}_{\mu_K} = \sqrt{S_{\mathrm{total}}^2 - \frac{N-1}{N} \sigma_{within}^2} = S_{\mathrm{total}} \sqrt{\mathrm{ICC} + \frac{1 - \mathrm{ICC}}{N}} \hspace{19em} (14)$$

Table 2 의 wafer-to-wafer 성분 $`\sigma_{between}`$ 에 대해 $`\sigma_{within}^2 = S_{\mathrm{total}}^2 - \sigma_{between}^2`$ 이므로, 같은 식을 within 대신 between 으로도 적을 수 있고, 그 과정은 [Appendix C](#appendix-c-derivation-of-the-between-component-form) 에 적었다.

$$\hat{\sigma}_{\mu_K} = \sqrt{\frac{S_{\mathrm{total}}^2 + (N-1) \sigma_{between}^2}{N}} = S_{\mathrm{total}} \sqrt{\frac{1 + (N-1) \mathrm{ICC}}{N}} \hspace{19em} (15)$$

Wafer effect 가 모두 0 일 경우에, $`\sigma_{between} = 0`$, 곧 ICC = 0 이면 식 (14) 와 식 (15) 에서 wafer 평균의 산포는 표준오차만 남는다. 관측한 wafer 평균은 이때도 site 잡음만큼 흩어지므로 0 이 아니다.

$$\hat{\sigma}_{\mu_K} = \frac{S_{\mathrm{total}}}{\sqrt{N}} \hspace{19em} (16)$$

이것이 흔히 기대하는 $`\sqrt{N}`$ 법칙이다. 이 자료는 ICC = 0.761 이라 식 (16) 이 서지 않는데, 그래도 $`S_{\mathrm{total}}/\sqrt{N}`$ 을 그대로 쓰면 $`32.50/\sqrt{13}`$ = 9.01 로 관측한 28.70 의 3 분의 1 도 되지 않는다.

### 4.2 W2W Detection Point

Fig 2 는 식 (13) 의 두 항을 처음 $`n`$ 장으로 계산해 함께 보인다. 세 곡선을 얻는 방법은 아래와 같다.

- 왼쪽 항 $`\sigma_{within}/\sqrt{N}`$: wafer 마다의 site 분산 $`s_i^2`$ 를 처음 $`n`$ 장까지 평균한 $`\sigma_{within}(1..n) = \sqrt{\frac{1}{n} \sum_{i \le n} s_i^2}`$ 을 $`\sqrt{N}`$ 으로 나눈 값. Site 를 $`N`$ 개 평균해도 wafer 평균에 남는 측정 잡음이며, wafer 가 모두 같아도 사라지지 않는 바닥이다. Wafer 평균을 쓰지 않으므로 자료에서 바로 나온다.
- 관측 곡선 $`\sigma_{\mu_n}`$: 처음 $`n`$ 장의 wafer 평균의 표본표준편차.
- 오른쪽 항 $`\sigma_{between}`$: 식 (13) 을 뒤집은 $`\sqrt{\sigma_{\mu_n}^2 - \sigma_{within}^2(1..n)/N}`$ 이며, $`\sigma_{between}`$ 자리에 드는 $`s_{\mu}(1..n)`$ 이 그 값이다. Wafer 마다 다른 wafer effect 의 산포, 곧 wafer 간의 변동 그 자체이다. 제곱근 안이 음수인 $`n`$ 에서는 정의되지 않아 그리지 않으며, 이 자료에서는 $`n = 3`$ 이 그렇다.

<img src="wiw-w2w-anova_fig/cum_stdev.png" width="900" style="max-width: 100%;" alt="Fig 2">

Fig 2. Cumulative standard deviation of the wafer means with the two terms of equation (13) and the w2w detection point, each computed from the first n wafers only

Fig 2 에서 두 항의 크기가 뒤집히는 곳을 w2w detection point 라 부르며, 오른쪽 항이 관측값의 98% 를 넘는 첫 $`n`$ 으로 잡으면 이 자료에서는 $`n = 5`$ 이다 ($`n = 4`$ 에서 74%, $`n = 5`$ 에서 98%). w2w detection point 이후 관측 곡선은 사실상 wafer effect 의 산포 그 자체이다.

공정 관리로 옮기면 w2w detection point 는 판단에 필요한 최소 표본이다. 그 앞에서 잰 산포는 wafer-to-wafer 를 볼 수 없으므로 그 값으로 관리 한계선을 세우면 산포를 크게 낮춰 잡게 되고, 이 점을 넘어서야 "이 산포는 site 균일도가 아니라 wafer 단위 조건에서 온다" 는 판정이 성립한다. 거꾸로 그 앞 구간에서 산포가 작게 나왔다고 공정이 안정된 것으로 읽으면 안 된다 — 아직 볼 수 있는 것이 측정 잡음뿐이기 때문이다.

### 4.3 WiW Excursion Detection

Wafer 한 장의 산포가 그때까지 본 wafer 내 산포에서 크게 벗어나면 그 wafer 를 WiW excursion 으로 본다. Wafer $`i`$ 를 판정할 때 앞선 wafer 만으로 구한 $`\sigma_{within}(1..i-1)`$ 을 기준선으로 두고, 그 wafer 한 장의 site 표준편차 $`s_i`$ 가 아래 한계를 넘는지 본다. 한계는 표본표준편차의 분포에서 나오며, 유도는 [Appendix D](#appendix-d-derivation-of-the-screening-limit) 에 적었다.

$$s_i \gt \sigma_{within}(1..i-1) \sqrt{\frac{\chi^2_{p, N-1}}{N-1}} \hspace{19em} (17)$$

Fig 3 이 그 판정이다. 회색 점이 wafer 한 장의 $`s_i`$, 초록 선이 기준선, 빨간 선이 식 (17) 의 한계이고, 한계를 넘은 wafer 를 빨간 점으로 표시했다. 세 값 모두 site 값의 표준편차라 단위가 같으므로 오른쪽 축을 따로 두지 않고 한 축에 겹쳐 그렸다.

<img src="wiw-w2w-anova_fig/wafer_screening.png" width="900" style="max-width: 100%;" alt="Fig 3">

Fig 3. Site value spread of each wafer against the running baseline and the screening limit of equation (17)

판정된 wafer 는 기준선 갱신에서 뺀다. 그대로 담으면 excursion 이 기준선을 끌어올려 뒤의 excursion 을 가리므로, excursion 이 잦을수록 판정이 둔해진다. 261 장을 다 담은 pooled `sigma_within` 15.87 과 견주면 이렇게 얻은 기준선은 마지막 wafer 에서 12.07 로 3.8 이 낮은데, 그 차이가 excursion 이 pooled 값에 실어 놓은 몫이다.

처음 20 장은 기준선을 쌓는 데만 쓰고 판정하지 않는다. 표본 몇 장 위에 선 기준선은 그 자체가 크게 흔들려 판정이 우연에 좌우되기 때문이며, 그 대가로 uniformity 가 가장 나빴던 wf0011 이 $`s_i`$ = 55.04 로 이 자료에서 가장 큰 산포인데도 판정 대상에서 빠진다.

---

## Appendix A. Terminology

- **ANOVA**: analysis of variance. 전체 제곱합을 원인별 제곱합으로 나누고, 각각을 자유도로 나눈 평균제곱의 비로 원인의 유의성을 판정하는 방법.
- **bilinear**: 두 인자 각각에 대해 linear 인 성질. Covariance 에서는 첫 인자에 대해 $`\mathrm{Cov}(aX + bY, Z) = a \, \mathrm{Cov}(X, Z) + b \, \mathrm{Cov}(Y, Z)`$ 이고, 둘째 인자에 대해 $`\mathrm{Cov}(X, aZ + bW) = a \, \mathrm{Cov}(X, Z) + b \, \mathrm{Cov}(X, W)`$ 이다.
- **Covariance**: 두 확률변수가 각자의 평균에서 벗어난 양을 곱해 기댓값을 취한 값. 두 인자가 같으면 $`\mathrm{Cov}(Y, Y) = \mathrm{Var}(Y)`$ 이며, 그 과정은 [Appendix E](#appendix-e-covariance-with-a-repeated-argument) 에 적었다.
- **ICC**: intraclass correlation. 전체 분산 중 group 간 분산이 차지하는 비율. 같은 group 에서 뽑은 두 관측치가 얼마나 닮았는지를 0 에서 1 사이로 나타내며, 이 문서의 group 은 wafer 이다. 이 문서가 쓰는 것은 one-way random effects model 의 ICC(1) 이며, two-way model 의 ICC 와는 값이 다르다.
- **run order**: 자료 파일의 행 순서. 측정 순서를 따르므로 시간 축으로 사용.
- **running baseline**: wafer 한 장을 판정할 때 쓰는 기준선. 그 wafer 앞에 있으면서 excursion 으로 판정되지 않은 wafer 만으로 구한 within-wafer 성분이다.
- **sigma_between**: wafer 간 분산성분의 표준편차. Table 2 의 wafer-to-wafer 값이며, wafer 평균의 표본표준편차 $`S_{\mathrm{between}}`$ 과 달리 within-wafer site 오차의 몫을 뺀 값이다.
- **sigma_within**: wafer 내 분산성분의 표준편차. MS within 의 제곱근이다.
- **site**: 한 wafer 위의 측정 지점. 열 `S1`~`S13` 에 해당.
- **Var**: variance. 값이 제 평균에서 벗어난 정도를 제곱하여 평균한 값이며, 표준편차의 제곱이다. 관측 수 $`m`$ 인 표본에서는 $`\mathrm{Var}(Y) = \frac{1}{m-1} \sum_{i=1}^{m} (Y_i - \bar{Y})^2`$ 로 계산한다.
- **variogram**: 두 지점의 값 차이가 갖는 분산을 두 지점 사이 거리의 함수로 나타낸 것. 거리에 따라 값이 얼마나 닮는지를 재는 데 쓴다.
- **w2w**: wafer-to-wafer. wafer 사이의 변동.
- **w2w detection point**: 식 (13) 의 오른쪽 항이 관측된 wafer 평균 산포의 98% 를 넘는 첫 $`n`$. 그 앞에서는 wafer 사이의 차이가 측정 잡음에 묻혀 분리되지 않는다.
- **wafer effect**: wafer $`i`$ 의 참 평균이 총평균에서 벗어난 양 $`\alpha_i`$. One-way random effects model 에서는 평균 0 으로 wafer 마다 새로 뽑히는 확률변수이고 그 variance 가 $`\sigma_{between}^2`$ 이다.
- **WiW**: within-wafer. 한 wafer 안 site 사이의 변동.
- **WiW excursion**: site 표준편차가 running baseline 이 세운 한계를 넘은 wafer.

## Appendix B. Limits of the Decomposition

### B.1 The Two Coefficients

Section 1.2 의 두 계수를 $`a`$ 와 $`b`$ 로 두면 아래와 같다.

$$a = \frac{K(N-1)}{M-1} = \frac{KN-K}{KN-1}, \qquad b = \frac{N(K-1)}{M-1} = \frac{KN-N}{KN-1} \hspace{19em} (18)$$

분자와 분모가 모두 $`KN`$ 에서 시작하므로, 1 에서 얼마나 모자라는지를 보는 편이 빠르다.

$$1 - a = \frac{K-1}{KN-1}, \qquad 1 - b = \frac{N-1}{KN-1} \hspace{19em} (19)$$

두 결손항은 각각 한쪽 크기에만 매인다. $`1-a`$ 의 분자와 분모를 $`K`$ 로, $`1-b`$ 의 분자와 분모를 $`N`$ 으로 나누면 아래 꼴이 된다.

$$1 - a = \frac{1 - 1/K}{N - 1/K}, \qquad 1 - b = \frac{1 - 1/N}{K - 1/N} \hspace{19em} (20)$$

$`K`$ 를 아무리 키워도 $`1-a`$ 는 $`1/N`$ 에서 멈추고, $`N`$ 을 아무리 키워도 $`1-b`$ 는 $`1/K`$ 에서 멈춘다.

$$\lim_{K \to \infty} (1 - a) = \frac{1}{N}, \qquad \lim_{N \to \infty} (1 - b) = \frac{1}{K} \hspace{19em} (21)$$

곧 한쪽만 키운 극한에서 계수는 아래 값에 멈춘다.

$$\lim_{K \to \infty} a = 1 - \frac{1}{N}, \qquad \lim_{N \to \infty} b = 1 - \frac{1}{K} \hspace{19em} (22)$$

따라서 $`a`$ 를 1 로 보내는 것은 wafer 당 site 수 $`N`$ 이고, $`b`$ 를 1 로 보내는 것은 wafer 수 $`K`$ 이며, 둘이 함께 커져야 두 계수가 같이 1 이 된다.

$$\lim_{N \to \infty} a = 1, \qquad \lim_{K \to \infty} b = 1, \qquad \lim_{K, N \to \infty} S_{\mathrm{total}}^2 = \overline{S_{\mathrm{within}}^2} + S_{\mathrm{between}}^2 \hspace{19em} (23)$$

이 문서의 $`K = 261`$, $`N = 13`$ 에서는 $`1 - a = 260/3392 = 0.0767`$ 로 $`1/N = 0.0769`$ 에 거의 같고, $`1 - b = 12/3392 = 0.0035`$ 로 $`1/K = 0.0038`$ 에 거의 같다. 즉 $`b`$ 는 이미 1 로 보아도 되지만 $`a`$ 는 7.7% 모자라며, site 를 13 개만 재는 한 이 결손은 wafer 를 아무리 더 재도 줄지 않는다. 이 자료에서 $`\overline{S_{\mathrm{within}}^2} = 251.98`$ 과 $`S_{\mathrm{between}}^2 = 823.46`$ 을 그냥 더하면 $`S_{\mathrm{total}} = 32.79`$ 가 되어 관측값 32.45 를 넘지만, 두 계수를 붙이면 관측값과 같아진다.

### B.2 Correlated Sites Within a Wafer

식 (8) 은 같은 wafer 의 서로 다른 두 site 오차가 독립이라고 두어 $`\mathrm{Cov}(e_{ij}, e_{ij'})`$ 을 0 으로 지운다. 실제 wafer 는 radial pattern 이나 edge roll-off 처럼 site 위치를 따라 함께 움직이는 성분을 지녀 그 covariance 가 0 이 아니며, 식 (6) 의 모형은 site 를 자리와 무관한 반복으로 보아 그 공간 구조를 $`e_{ij}`$ 안에 묻는다.

두 site 오차의 상관을 $`\rho`$ 로 두면 wafer 평균에 남는 잡음은 $`\mathrm{Var}(\bar{e}_i) = \sigma_{within}^2 [1 + (N-1)\rho] / N`$ 이며, 식 (10) 은 $`\rho = 0`$ 인 경우이다. $`\rho \gt 0`$ 이면 실제 잡음 바닥이 식 (13) 의 왼쪽 항 $`\sigma_{within}/\sqrt{N}`$ 보다 크고, 덜 빼는 만큼 오른쪽 항 $`s_{\mu}(1..n)`$ 이 부풀려져 section 4.2 의 w2w detection point 가 실제보다 이른 $`n`$ 에서 잡힌다.

같은 상관이 section 4.3 의 한계에도 걸린다. 식 (32) 가 자유도 $`N-1`$ 의 $`\chi^2`$ 를 쓰는 것은 한 wafer 의 site $`N`$ 개가 독립한 정보 $`N-1`$ 개를 낸다는 뜻인데, site 끼리 닮으면 실효 자유도가 그보다 작아 한계가 좁게 잡히고 WiW excursion 판정이 실제보다 민감해진다.

$`\rho`$ 를 재려면 site 좌표를 인자로 둔 모형이나 variogram 이 필요하며, 이 문서의 자료로는 그 값을 대지 않았다.

### B.3 Wafers as a Sample of One Process

식 (6) 은 $`\alpha_i`$ 를 평균 0, variance $`\sigma_{between}^2`$ 인 한 분포에서 wafer 마다 독립으로 뽑는다고 둔다. 이 가정 위에서만 261 장이 공정의 표본이 되고, $`\sigma_{between}`$ 이 그 261 장을 넘어 앞으로 나올 wafer 에도 적용된다. 같은 자료를 fixed effects 로 두면 $`\alpha_i`$ 가 저마다 모수라 결론이 그 261 장에 머물고, wafer-to-wafer 성분이라는 하나의 수가 서지 않는다.

이 자료는 그 가정에서 벗어난다. Section 3 의 expanding window 에서 w2w 성분이 $`n \ge 100`$ 에서도 18.28 에서 28.36 으로 계속 오르는데, 한 분포에서 독립으로 뽑은 표본이라면 쌓일수록 한 값에 잦아들어야 한다. 뒤쪽 wafer 의 $`\alpha_i`$ 가 앞쪽과 다른 수준에 있다는 뜻이다.

그래서 이 문서의 $`\sigma_{between}`$ 은 한 공정 수준 둘레의 산포가 아니라 261 장에 걸친 drift 까지 담은 값이다. 그 값으로 세운 관리 한계선은 drift 를 공정이 늘 내는 산포로 받아들이므로 새 wafer 에 적용하면 실제보다 넓다.

둘을 가르려면 run order 를 인자로 둔 모형이 필요하다. 시간 추세항을 뺀 잔차에서 $`\sigma_{between}`$ 을 다시 구하거나, 구간을 나눠 각 구간 안에서 성분을 구하는 방법이 있으며, 이 문서는 그 분리를 하지 않았다.

## Appendix C. Derivation of the Between-Component Form

식 (14) 는 within 성분으로 적혀 있다.

$$\hat{\sigma}_{\mu_K}^2 = S_{\mathrm{total}}^2 - \frac{N-1}{N} \sigma_{within}^2 \hspace{19em} (24)$$

식 (9) 에서 $`S_{\mathrm{total}}^2 = \sigma_{within}^2 + \sigma_{between}^2`$ 이므로 within 성분을 나머지 둘로 바꿀 수 있다.

$$\sigma_{within}^2 = S_{\mathrm{total}}^2 - \sigma_{between}^2 \hspace{19em} (25)$$

이를 대입하고 $`S_{\mathrm{total}}^2`$ 의 계수를 정리하면 아래와 같다.

$$\hat{\sigma}_{\mu_K}^2 = S_{\mathrm{total}}^2 \left(1 - \frac{N-1}{N}\right) + \frac{N-1}{N} \sigma_{between}^2 = \frac{S_{\mathrm{total}}^2 + (N-1) \sigma_{between}^2}{N} \hspace{19em} (26)$$

ICC 의 정의 $`\mathrm{ICC} = \sigma_{between}^2 / S_{\mathrm{total}}^2`$ 를 넣어 $`\sigma_{between}^2`$ 을 지우면 두 번째 형태가 나오고, 제곱근을 취한 것이 식 (15) 이다.

$$\hat{\sigma}_{\mu_K}^2 = S_{\mathrm{total}}^2 \frac{1 + (N-1) \mathrm{ICC}}{N} \hspace{19em} (27)$$

$`N = 1`$ 이면 두 형태 모두 $`\hat{\sigma}_{\mu_K} = S_{\mathrm{total}}`$ 이 되고, $`N`$ 이 커지면 $`\hat{\sigma}_{\mu_K}`$ 는 $`\sigma_{between}`$ 으로 수렴한다. site 를 많이 잴수록 wafer 평균에서 within 성분이 지워진다는 뜻이다.

## Appendix D. Derivation of the Screening Limit

아래에서 $`i`$ 는 wafer 번호, $`j`$ 는 그 wafer 위의 site 번호로 section 1.1 의 표기를 그대로 쓴다. 곧 $`X_{ij}`$ 는 wafer $`i`$ 의 $`j`$ 번째 site 측정값이고, $`\bar{X}_i`$ 는 그 wafer 의 평균, $`s_i^2`$ 은 그 wafer 안 site 값의 표본분산이다. 한 wafer 안의 site 값이 서로 독립이고 같은 정규분포를 따른다고 둔다.

$$X_{ij} \sim \mathcal{N}(\mu + \alpha_i,\ \sigma_{within}^2), \qquad s_i^2 = \frac{1}{N-1} \sum_{j=1}^{N} (X_{ij} - \bar{X}_i)^2 \hspace{19em} (28)$$

한계를 세우려면 $`s_i`$ 가 우연만으로 얼마나 커질 수 있는지 알아야 한다. 같은 공정에서 나온 wafer 라도 site $`N`$ 점을 어디서 뽑느냐에 따라 $`s_i`$ 는 매번 달라지므로, 그 흔들림의 분포를 알아야 어디부터가 우연으로 보기 어려운 값인지 정할 수 있다. 그 분포가 카이제곱이며, 카이제곱 분포는 서로 독립인 표준정규 변수 $`m`$ 개를 제곱해 더한 값의 분포로 $`m`$ 이 그 자유도이다. 그러므로 $`s_i^2`$ 의 분포를 아는 일은 그것을 표준정규 몇 개의 제곱합으로 적을 수 있는지를 세는 일이 된다. 측정값에서 그 wafer 의 참 평균 $`\mu + \alpha_i`$ 를 빼고 표준편차로 나누면 표준정규가 된다.

$$Z_{ij} = \frac{X_{ij} - \mu - \alpha_i}{\sigma_{within}} \sim \mathcal{N}(0, 1) \hspace{19em} (29)$$

$`X_{ij} - \bar{X}_i = \sigma_{within}(Z_{ij} - \bar{Z}_i)`$ 이므로 식 (28) 의 제곱합은 $`Z`$ 의 제곱합으로 바뀐다. 각 항을 $`(Z_{ij} - \bar{Z}_i)^2 = Z_{ij}^2 - 2 Z_{ij} \bar{Z}_i + \bar{Z}_i^2`$ 로 풀고 $`j = 1`$ 부터 $`N`$ 까지 더하면 세 조각이 된다. 첫 조각은 그대로 $`\sum_j Z_{ij}^2`$ 이고, 둘째 조각은 $`\bar{Z}_i`$ 가 $`j`$ 에 따라 변하지 않는 상수라 합 밖으로 빠져 $`-2 \bar{Z}_i \sum_j Z_{ij}`$ 가 되며, 셋째 조각은 그 상수를 $`N`$ 번 더한 $`N \bar{Z}_i^2`$ 이다.

$$\frac{(N-1) s_i^2}{\sigma_{within}^2} = \sum_{j=1}^{N} (Z_{ij} - \bar{Z}_i)^2 = \sum_{j=1}^{N} Z_{ij}^2 - 2 \bar{Z}_i \sum_{j=1}^{N} Z_{ij} + N \bar{Z}_i^2 \hspace{19em} (30)$$

평균의 정의에서 $`\sum_{j} Z_{ij} = N \bar{Z}_i`$ 이므로 가운데 항은 $`2 N \bar{Z}_i^2`$ 이 되고, 마지막 항과 합치면 $`N \bar{Z}_i^2`$ 하나만 남는다. 곧 표준정규 제곱합에서 평균의 몫을 뺀 꼴이다.

$$\frac{(N-1) s_i^2}{\sigma_{within}^2} = \sum_{j=1}^{N} Z_{ij}^2 - 2 N \bar{Z}_i^2 + N \bar{Z}_i^2 = \sum_{j=1}^{N} Z_{ij}^2 - N \bar{Z}_i^2 \hspace{19em} (31)$$

우변의 첫 항은 표준정규 $`N`$ 개의 제곱합이므로 정의에 따라 $`\chi^2_N`$ 이다. $`\bar{Z}_i`$ 는 평균 0, 분산 $`1/N`$ 의 정규분포를 따라 $`\sqrt{N}\,\bar{Z}_i`$ 가 표준정규이므로 둘째 항은 $`\chi^2_1`$ 이다. 정규 표본에서 표본평균과 표본분산은 서로 독립이라 두 몫이 겹치지 않으므로, 자유도는 그대로 빼진다.

$$\frac{(N-1) s_i^2}{\sigma_{within}^2} \sim \chi^2_{N-1} \hspace{19em} (32)$$

편차 $`X_{ij} - \bar{X}_i`$ 는 합이 0 이라는 제약 하나에 묶여 $`N`$ 개 중 $`N-1`$ 개만 자유로우므로, 자유도가 $`N-1`$ 이다.

$`\chi^2_{p,\,N-1}`$ 을 자유도 $`N-1`$ 인 카이제곱 분포의 $`p`$ 분위, 곧 그보다 작을 확률이 $`p`$ 인 점으로 두면, 식 (32) 의 좌변이 그 점을 넘을 확률은 나머지인 $`1-p`$ 이다.

$$P\left( \frac{(N-1) s_i^2}{\sigma_{within}^2} \gt \chi^2_{p, N-1} \right) = 1 - p \hspace{19em} (33)$$

괄호 안을 $`s_i`$ 에 대해 풀고 참값 $`\sigma_{within}`$ 자리에 running baseline 을 놓으면 식 (17) 이 된다. 곧 식 (17) 을 넘은 wafer 는, 그 wafer 의 산포가 기준선과 같았다면 $`1-p`$ 의 확률로만 나올 값을 낸 wafer 이다.

기준선은 참값이 아니라 앞선 wafer 로 추정한 값이므로, 엄밀하게는 두 분산의 비가 F 분포를 따른다. 기준선이 wafer $`m`$ 장 위에 서 있으면 그 자유도는 $`\nu = m(N-1)`$ 이다.

$$\frac{s_i^2}{\sigma_{within}^2(1..i-1)} \sim F(N-1,\ \nu) \hspace{19em} (34)$$

$`\nu`$ 가 커지면 $`F(N-1, \nu)`$ 의 $`p`$ 분위는 $`\chi^2_{p,\,N-1}/(N-1)`$ 로 수렴하므로 식 (17) 을 그대로 쓸 수 있다. 이 자료의 $`N = 13`$, $`p = 0.999`$ 에서 계수는 카이제곱으로 1.656 이고, 판정을 시작하는 wafer 21 에서 $`\nu = 240`$ 을 넣은 F 로는 1.696, 마지막 wafer 에서는 1.660 이다. 곧 판정 초반에 한계를 2.4% 낮게 잡는 것이 카이제곱을 쓰는 대가이다.

## Appendix E. Covariance With a Repeated Argument

Covariance 의 두 인자에 같은 확률변수를 넣으면 곱해지는 두 편차가 같은 값이라 제곱이 되고, 그 기댓값은 variance 의 정의 그대로이다.

$$\mathrm{Cov}(Y, Y) = E[(Y - E[Y])(Y - E[Y])] = E[(Y - E[Y])^2] = \mathrm{Var}(Y) \hspace{19em} (35)$$

식 (8) 의 첫째 항이 그 꼴이므로 $`\mathrm{Cov}(\alpha_i, \alpha_i) = \mathrm{Var}(\alpha_i)`$ 이고, 식 (7) 이 그 값을 $`\sigma_{between}^2`$ 으로 둔다.
