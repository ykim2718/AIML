# Outlier Detection Methods (Korean)
Rev. 0 | Created: 2026-09-09 | Updated: 2026-09-04 20:10 UTC

> 나머지 데이터가 따르는 pattern 에서 벗어난 관측을 찾아내는 방법들을, 각각이 무엇을 가정하는지에
> 따라 정리한 survey 이다. 방법을 습관이 아니라 데이터의 모양에서 고를 수 있게 하려는 것이다.

## 1. Scope

Outlier 는 나머지 표본이 따르는 model 과 어긋나는 관측이다. 하나를 찾았다는 것은 그 model 과의 정합성에 대한 진술이지 그 관측이 틀렸다는 판정이 아니므로, 검출과 처리는 따로 둔다. Flag 는 조사를 닫는 것이 아니라 여는 것이다.

여기 있는 어느 방법도 보편적으로 옳지 않다. 각각이 가정을 치르고 답을 사기 때문이다. 데이터가 그 가정을 어기는 방법도 여전히 flag 를 내놓지만, 그 flag 는 실제 이탈이 아니라 가정 위반을 기록한 것이다. 따라서 선택이란 데이터의 두 성질을 방법이 요구하는 것에 맞추는 일이다.

- **Dimension.** 변수 하나, 몇 개, 아니면 거리가 의미를 잃을 만큼 큰 공간.
- **Distribution.** 모수적 형태를, 무엇보다 정규성을 가정할 수 있는지 여부.

이 둘은 데이터의 성질이다. 선택을 고정하는 나머지는 찾고 있는 outlier 의 성질이며, 꼭지 2 가 어떤 방법을 이름 대기 전에 그것들을 정리한다. 꼭지 3 부터 5 까지는 세 가지 방법 family 를 차례로 다루고, 꼭지 6 이 선택을 표 하나에 담는다. [Appendix C. Semiconductor Practice](#appendix-c-semiconductor-practice) 는 산업 표준 규칙 둘을 그 방법들에 비추어 읽는다.

## 2. Kinds of Outlier

Outlier 라는 말은 정도가 아니라 종류가 다른 관측들을 함께 덮으며, 한 방법은 그 모두가 아니라 한 종류의 질문에 답한다. 아래 여덟 꼭지는 여덟 개의 category 가 아니라 여덟 개의 축이다. 하나의 관측은 그 모든 축 위에 동시에 자리를 가진다. 한 번의 측정이 point outlier 이면서, global 이 아니라 local 이고, 기록 오류에서 비롯되었으며, contaminant 는 아니면서 discordant 이고, 자기가 속한 regression 에서 leverage 가 높은 점일 수 있다.

축을 갈라 두는 것이 값어치 있는 이유는 방법이 그 가운데 하나를 기준으로 선택되고, 한 축에 답하는 방법은 나머지 축에 대해 아무 말도 하지 않기 때문이다. 꼭지 6.2 는 이 문서의 모든 방법을 아래 축 위에 놓는다.

### 2.1. Form

[Chandola, Banerjee and Kumar (2009)](#ref-8) 는 anomaly 를 데이터에서 취하는 형태로 나누며, 이것이 가장 자주 인용되는 구분이다.

- **Point.** 관측 하나가 그 자체로 극단적이다. 평범한 날씨 기록에 섞인 영하 100 도.
- **Contextual.** 값 자체는 표본 안에서 평범하고 맥락에서 극단적이다. 2 도는 한 해의 측정값 가운데서는 눈에 띄지 않지만 8 월의 값으로는 틀렸다.
- **Collective.** 어느 값 하나도 극단적이지 않지만 그것들이 이어진 구간이 함께 극단적이다. 모든 측정값이 정상 범위 안에 있는 심전도의 저전압 구간.

Contextual anomaly 와 collective anomaly 는 값이 스스로 담고 있지 않은 것을 요구한다. 앞은 context 변수를, 뒤는 순서를 요구한다. 값의 주변 분포만 보는 방법은 둘 가운데 어느 것도 찾지 못한다.

### 2.2. Reference Set

관측은 무엇인가에 견주어 극단적이며, 그 무엇은 표본 전체일 수도 있고 이웃일 수도 있다.

- **Global.** 표본 전체에 견주어 극단적.
- **Local.** 표본 전체에 견주면 평범하고, 자기가 속한 group 에 견주면 극단적.

이 축은 꼭지 2.1 과 독립이며, 그래서 point 와 global 을 한 label 로 묶는 것이 잘못이다. Local point outlier 야말로 [Breunig, Kriegel, Ng and Sander (2000)](#ref-17) 이 local outlier factor 를 만들어 찾으려 한 것이고, 꼭지 4.3 이 다루는 경우이다.

### 2.3. Cause

같은 값에 flag 가 붙었더라도 그것이 도착한 경로는 셋이며, 그것을 어떻게 해야 하는지는 경로마다 다르다.

- **Error.** 측정, 전사, 전송에서 생긴 실수. 그 값은 연구 대상 process 가 아니라 그것을 기록한 process 를 기술한다.
- **Foreign population.** 다른 것을 옳게 측정한 값. 예를 들어 batch 에 섞여 들어온 다른 lot 의 부품.
- **Genuine rare event.** 연구 대상 process 를 옳게 측정한 값으로, 그 process 가 실제로 가진 꼬리에 놓여 있다.

어느 통계량도 이 셋을 가르지 못한다. 검출은 후보를 내놓을 뿐이고 원인은 그 뒤의 기록을 들여다보아야 밝혀지며, 이것이 이 문서가 검출과 처리를 따로 두는 이유이다.

### 2.4. Discordancy and Contamination

[Barnett and Lewis (1994)](#ref-2) 는 놓치기 쉬운 구분을 그어 둔다. **Contaminant** 는 다른 분포에서 온 관측이다. **Discordant observation** 은 나머지와 통계적으로 어긋나 보이는 관측이다.

둘은 서로를 함의하지 않는다. Contaminant 가 본체 안에 숨어 discordant 로 보이지 않을 수 있고, 오염되지 않은 heavy-tailed 분포의 표본도 예측 가능한 비율로 discordant 관측을 내놓는다. 이 문서의 모든 검정은 discordancy 를 검정한다. Contamination 은 꼭지 2.3 의 조사가 밝히려는 것이다.

### 2.5. Position in a Regression

분포가 아니라 model 을 적합할 때에는 축 하나가 셋으로 갈라지고, 그 셋은 서로 어긋난다.

- **Residual outlier.** 반응에서 적합된 면으로부터 멀다.
- **Leverage point.** 예측변수에서 극단적이며, 실제로 적합을 움직이든 아니든 움직일 힘을 가진다.
- **Influential observation.** 제거하면 적합이 눈에 띄게 바뀌며, [Cook (1977)](#ref-4) 이 자기 이름이 붙은 거리로 재었다.

영향 없는 높은 leverage 는 흔하다. 멀리 떨어진 예측변수 값이 마침 직선 위에 놓이면 직선을 어디로도 끌지 않는다. 큰 잔차 없는 영향도 마찬가지로, 점이 직선을 자기 쪽으로 끌어다 놓은 경우이다. [Belsley, Kuh and Welsch (1980)](#ref-6) 은 그 둘을 가르는 진단 지표들을 모아 놓았다.

### 2.6. Labels

찾기를 시작하기 전에 무엇을 알고 있는지가 무엇을 할 수 있는지를 정한다.

- **Supervised.** 두 class 모두의 label 이 붙은 예시. 이것은 outlier 문제라기보다 class 불균형이 심한 classification 문제이다.
- **Semi-supervised.** 깨끗하다고 알려진 training set 과, 그것에 견주어 판정할 새 관측. 이것은 novelty detection 이다.
- **Unsupervised.** Label 이 없는 표본 하나이며, 이미 outlier 를 담고 있을 수 있다.

꼭지 3 부터 5 까지의 모든 방법은 unsupervised 이거나 semi-supervised 이다. Label 이 붙은 outlier 가 드물기 때문이다. 뒤의 둘을 가르는 것은 실무에서 중요하다. 깨끗하다고 가정했지만 실제로는 깨끗하지 않은 training set 은 그 안의 outlier 를 정상으로 여기도록 방법을 가르치기 때문이다.

### 2.7. Count

Outlier 를 몇 개 예상하는지는 문턱값만이 아니라 절차 자체를 바꾼다.

- **Single.** 검정 하나, 명시된 오류율 하나.
- **Multiple.** 개수를 모르는 여럿이며, masking 과 swamping 이 나타나는 자리이다.

Masking 은 outlier 하나가 중심이나 척도를 부풀려 두 번째 outlier 가 더는 극단적으로 보이지 않게 하는 것이다. Swamping 은 그 반대로, outlier 가 추정값을 크게 일그러뜨려 깨끗한 관측까지 함께 flag 되는 것이다. [Hawkins (1980)](#ref-5) 이 다수 outlier 문제를 다루었고, 꼭지 3.4 가 그것을 위해 만들어진 절차이다.

### 2.8. Time Series Type

데이터에 순서가 있으면 꼭지 2.1 의 form 축은 이탈이 series 에 들어오는 방식으로 더 잘게 갈라진다. [Fox (1972)](#ref-3) 가 앞의 둘을 내놓았고, [Chen and Liu (1993)](#ref-7) 이 표준이 된 넷을 정리했다.

- **Additive.** 측정값 하나가 밀려나고 series 는 곧바로 돌아온다.
- **Innovational.** 충격이 process 안으로 들어와, 뒤따르는 측정값들로 이탈이 전파된다.
- **Level shift.** Series 가 새 수준으로 옮겨 가 그대로 머문다.
- **Temporary change.** Series 가 움직였다가 여러 측정값에 걸쳐 되돌아온다.

넷 모두 꼭지 2.1 아래에서는 하나의 collective anomaly 이며, 그래서 그 축은 장비 trace 에 쓰기에는 너무 성기다. Chamber 가 영구히 drift 한 것인지 스스로 회복한 것인지는 level shift 와 temporary change 의 차이인데, 더 성긴 어휘에는 그것을 묻는 말이 없다.

## 3. Statistical Methods

이 방법들은 분포의 형태를 가정하고 그로부터의 이탈을 잰다. 계산이 가장 싸고 근거를 대기가 가장 쉬우며, 그 가정이 성립하는 한 옳은 기본값이다.

### 3.1. Z-Score

Z-score 는 관측과 표본 평균의 편차를 표본 표준편차로 나눈다.

$$z_i = \frac{x_i - \bar{x}}{s}$$

- $z_i$ — 관측 $i$ 의 z-score.
- $x_i$ — $n$ 개 값으로 이루어진 표본의 $i$ 번째 관측.
- $\bar{x}$ (x bar) — 그 표본의 평균.
- $s$ — 그 표본의 표준편차로, 편차 제곱의 합을 $n-1$ 로 나누어 만든다.

절대값이 3 을 넘으면 flag 하는 것이 관례이다. 이 규칙은 정규성을 가정하며, 그 아래에서는 우연만으로 3 을 넘는 관측이 약 0.27% 이다.

한 가지 성질이 이 방법을 두 가지로 제약한다. 평균과 표준편차가 모두 검정 대상 표본에서 계산되므로, outlier 는 자기가 견주어지는 척도를 부풀려 스스로를 가린다. 같은 자기참조가 점수에 상한을 씌운다. 크기 $n$ 의 표본에서 어떤 z-score 의 절대값도 $(n-1)/\sqrt{n}$ 을 넘지 못하며, 이는 [Shiffler (1988)](#ref-12) 의 한계이다. 그래서 3 의 규칙은 관측 11 개 아래에서, 3.5 의 규칙은 15 개 아래에서 발동할 수 없다.

### 3.2. Interquartile Range

Interquartile range 는 1 사분위수에서 3 사분위수까지의 거리이다. Tukey 의 규칙은 아래 구간 안에 드는 관측을 남기고 벗어나는 관측에 flag 를 붙인다. 구간의 두 끝은 box plot 의 수염이 그리는 fence 이다.

$$\left[ \ Q_1 - 1.5 \cdot \mathrm{IQR}, \quad Q_3 + 1.5 \cdot \mathrm{IQR} \ \right], \qquad \mathrm{IQR} = Q_3 - Q_1$$

- $Q_1$ — 1 사분위수로, 표본의 4 분의 1 이 그 아래에 놓이는 값.
- $Q_3$ — 3 사분위수로, 표본의 4 분의 3 이 그 아래에 놓이는 값.
- $\mathrm{IQR}$ — 그 둘 사이의 거리로, 가운데 절반의 퍼짐이다.

사분위수는 순서통계량이므로 이 규칙은 분포 가정을 요구하지 않고, z-score 의 0% 에 견주어 25% 의 breakdown point 를 가진다. 표준편차가 $\sigma$ 인 정규 표본에서 이 범위 자체는 $1.349 \sigma$ 이고, 그러면 fence 는 $\pm 2.7 \sigma$ 에 놓여 관측의 약 0.7% 를 그 밖에 남긴다. 따라서 이 규칙은 3 의 z-score 와 엄격함이 비슷하면서, 그 점수라면 무너질 오염을 견딘다. [Appendix B. Tukey's Rule](#appendix-b-tukeys-rule) 이 그 비교를 계산해 보이고, 이 규칙에 보통 함께 그리는 두 번째 fence 를 준다.

### 3.3. Hampel Identifier

Hampel identifier 는 z-score 의 형태를 그대로 두고 두 추정값을 모두 바꾼다. 평균의 자리에 median 이 들어가고, 표준편차의 자리에는 median 으로부터의 편차의 median 을 다시 잰 값이 들어간다.

$$\mathrm{MAD} = \mathrm{median}\left( \left| x_1 - \tilde{x} \right|, \ldots, \left| x_n - \tilde{x} \right| \right)$$

$$M_i = \frac{x_i - \tilde{x}}{\mathrm{MAD} / \Phi^{-1}(0.75)}$$

- $x_1, \ldots, x_n$ — 표본이고 $x_i$ 는 그 $i$ 번째 관측으로, 꼭지 3.1 과 같다.
- $\tilde{x}$ (x tilde) — 표본의 median 으로, 편차를 그로부터 재고 점수를 그에 맞추어 중심에 놓는다.
- $\mathrm{MAD}$ — 그 절대편차들의 median 으로, 다시 재기 전의 raw robust 척도이다.
- $\Phi^{-1}(0.75) = 0.674490$ — 표준정규분포의 3 사분위수로, MAD 를 이 값으로 나눈다.
- $M_i$ — 관측 $i$ 의 modified z-score 로, 꼭지 3.1 의 $z_i$ 와 같은 척도에서 읽는다.

그 제수는 consistency constant 이며, raw MAD 가 $s$ 의 추정값이 아니기 때문에 있다. 정규 표본에서 MAD 는 $\sigma$ 가 아니라 $0.674490 \sigma$ 로 수렴하므로 퍼짐을 3 분의 1 가량 낮추어 말한다. 이 상수로 나누는 것은 1.482602 를 곱하는 것과 같으며, $M_i$ 를 $z_i$ 를 읽는 척도 위에 올려놓는다. 그 단계가 없으면 점수는 자기만의 척도에 놓이고, 어떤 문턱값도 두 규칙 사이를 오갈 수 없으며 false positive rate 으로 읽히지도 않는다.

이 상수는 가정이 아니라 calibration 이고, 이 방법에서 정규성이 들어오는 유일한 자리이다. 이것을 바꾸면 모든 점수가 같은 배율로 다시 재어질 뿐 순서는 그대로이므로, 이것이 정하는 것은 문턱값이 놓이는 자리이지 어느 관측이 극단적인가가 아니다.

절대값이 3.5 를 넘으면 flag 하는 것이 관례이며, [Iglewicz and Hoaglin (1993)](#ref-14) 이 권한 값이다. Median 도 MAD 도 소수가 움직일 수 없으므로, 이 identifier 는 꼭지 3.1 에서 스스로를 가리던 outlier 에 닿고 반복이 필요 없다. 모든 관측을 한 번씩, 오염된 적 없는 추정값에 견주어 채점한다.

무너지는 자리는 분포가 아니라 동점이다. 표본의 절반이 넘게 한 값을 취하면 MAD 는 0 이 되어 어떤 점수도 정의되지 않으며, 같은 50% breakdown point 를 가진 어떤 추정량도 이를 벗어나지 못한다.

### 3.4. Generalized ESD

표본을 outlier 하나에 대해 검정하고 남은 것에 같은 검정을 되풀이하는 방식은 유의수준을 지키지 못한다. Generalized extreme studentized deviate 절차는 outlier 개수의 상한 $r$ 을 먼저 선언하고 같은 통계량을 $r$ 단계에 걸쳐 돌려 그것을 고친다.

$$R_i = \frac{\max_j \left| x_j - \bar{x}_i \right|}{s_i}, \qquad i = 1, \ldots, r$$

- $R_i$ — 단계 $i$ 에서의 extreme studentized deviate.
- $x_j$ — 표본의 관측이며, 단계 번호와 구별하려고 $j$ 로 첨자를 붙였다.
- $\bar{x}_i$ 와 $s_i$ — 앞선 단계에서 제거한 $i-1$ 개 관측이 빠진 뒤 남은 표본의 평균과 표준편차.
- $\max_j$ — 아직 남아 있는 관측에 대한 최대값. 그것을 취한 관측은 단계 $i+1$ 전에 제거된다.
- $r$ — 선언한 outlier 개수의 상한으로, 데이터를 읽기 전에 정한다.

각 $R_i$ 는 그 단계에 맞추어 유도되어 [Rosner (1983)](#ref-13) 이 표로 만든 임계값 $\lambda_i$ 와 견준다. Outlier 의 개수는 $R_i \gt \lambda_i$ 가 되는 **가장 큰** $i$ 이지 첫 번째 $i$ 가 아니다. 그렇게 읽는 것이 masking 을 이기는 방법이다. 한 단계가 실패해도, masking 을 일으킨 관측이 이미 제거된 뒤의 나중 단계는 성공할 수 있기 때문이다. 이 절차는 [ISO 16269-4](#ref-15) 의 다수 outlier 방법이며, 표본에서 오염되지 않은 부분이 근사적으로 정규라고 가정한다.

### 3.5. Mahalanobis Distance

다변량 데이터에서 [Mahalanobis distance](#ref-11) 는 변수 사이의 공분산을 감안한 단위로 관측이 중심에서 얼마나 떨어져 있는지를 잰다.

$$d^2(x) = \left( x - \mu \right)^{T} \Sigma^{-1} \left( x - \mu \right)$$

- $x$ — 관측 하나이며, 변수마다 성분 하나를 가지는 vector 로 쓴다.
- $\mu$ — 표본의 중심으로, 변수별 평균의 vector 이다.
- $\Sigma$ — 변수들의 공분산 행렬이고, $\Sigma^{-1}$ 은 그 역행렬이다.
- $d^2(x)$ — 제곱 거리로, 변수가 하나일 때 꼭지 3.1 의 $z_i^2$ 로 줄어든다.

공분산 항이 이 방법을 변수별 확인 위로 올린다. 변수 하나하나에서는 평범한 관측도 그 조합에서는 있을 법하지 않을 수 있고, 상관 구조를 읽는 방법만이 그것을 본다. 다변량 정규성 아래에서, 그리고 중심과 공분산이 표본에서 추정된 것이 아니라 알려져 있을 때, $d^2$ 는 변수 개수만큼의 자유도를 가진 chi-square 분포를 따르며 cut-off 는 거기에서 나온다.

같은 자기참조가 더 심하게 돌아온다. $\mu$ 와 $\Sigma$ 가 모두 오염된 표본에서 추정되고, outlier 들이 뭉쳐 있으면 그것들을 감추는 바로 그 방향으로 $\Sigma$ 를 부풀린다. [Rousseeuw and Van Driessen (1999)](#ref-16) 의 minimum covariance determinant 같은 robust 추정값을 쓰면, 이미 오염되었을 수 있는 데이터에서도 이 거리를 쓸 수 있다.

## 4. Machine Learning Methods

이 방법들은 분포 가정을 버리고 label 없는 데이터에서 정상 영역의 범위를 배운다. 여러 변수를 한꺼번에 다루고 오류율을 주장하지 않으므로, 그 출력은 통과 여부를 묻는 검정이 아니라 순위를 매길 점수이다.

### 4.1. [Isolation Forest](#ref-19)

Isolation Forest 는 무작위 변수를 무작위 문턱값에서 쪼개어 tree 를 세우고, 한 관측이 홀로 남기까지 몇 번의 분할이 필요한지를 기록한다. 성긴 영역의 관측은 적은 분할로 떨어져 나오므로, forest 전체에 걸친 짧은 평균 경로 길이가 anomaly score 가 된다.

이 방법은 profile 하지 않고 isolate 하며, 속도는 거기에서 온다. 밀도나 거리를 결코 추정하지 않고, 표본 크기에 선형인 시간에 돌며, 부분표본에서도 동작한다. 그래서 데이터가 크거나 변수가 많을 때 보통 첫 번째 선택이 된다.

### 4.2. [One-Class SVM](#ref-18)

One-Class SVM 은 training 데이터가 차지하는 영역을 감싸는 경계를 배우고, 그 경계 밖의 것을 outlier 라고 부른다. Kernel 이 경계가 어떻게 휠 수 있는지를 정하고, parameter $\nu$ 가 경계 밖으로 나가도 되는 training 데이터의 비율에 상한을 준다.

이 방법은 이미 경계로 진술된 문제, 곧 새 관측이 알려진 영역에 속하는지를 묻는 문제에 맞는다. 대가는 다른 선택지보다 비싸다는 것이다. 적합이 표본 크기에 대해 이차 이상이고, 답이 kernel 과 그 bandwidth 와 변수의 척도에 달려 있는데, 그 가운데 어느 것도 데이터가 스스로 고르지 않는다.

### 4.3. LOF (Local Outlier Factor)

Local Outlier Factor 는 한 관측 둘레의 밀도를 그 $k$ 개 최근접 이웃 각각의 둘레 밀도와 견준다. 1 에 가까운 factor 는 그 관측이 이웃들만큼 빽빽이 둘러싸여 있다는 뜻이고, 1 보다 크게 높은 factor 는 이웃들보다 성긴 자리에 있다는 뜻이다.

비교가 local 이기 때문에, 이 방법은 표본 전체에 견주면 눈에 띄지 않으면서 자기가 속한 group 에서는 뚜렷이 떨어져 있는 관측을 찾아낸다. 어떤 global 방법도 닿지 못하는 경우이며, 밀도가 실제로 다른 cluster 들을 데이터가 담고 있을 때 이웃 탐색의 값을 치를 이유가 된다.

### 4.4. [ECOD](#ref-21) (Empirical Cumulative Distribution)

ECOD 는 outlier 를 꼬리의 드문 사건으로 보고, 아무것도 적합하지 않은 채 꼬리의 희소함을 잰다. 변수마다 따로 empirical cumulative distribution 을 만들고, 모든 관측의 왼쪽 꼬리 확률과 오른쪽 꼬리 확률을 읽어 낸 다음, 그 확률들을 변수에 걸쳐 하나의 점수로 모은다.

여기서 정할 hyperparameter 가 없는 유일한 방법이다. 이 꼭지의 다른 모든 방법은 답을 바꾸는 이웃 크기나 kernel 이나 contamination rate 을 가지는데, 그 가운데 어느 것도 존재하지 않는 label 에 맞추어 조율할 수 없다. 또한 표본 크기와 변수 개수 모두에 선형이며, 변수별 꼬리 확률이 어느 변수가 그 관측을 극단적으로 만들었는지 말해 준다.

변수를 하나씩 따로 읽는 대가는, ECOD 가 변수별 확인과 마찬가지로 변수들의 조합에만 있는 이탈을 보지 못한다는 것이다. 꼭지 3.5 가 다루는 경우이다.

## 5. Deep Learning Methods

이 방법들은 정상 데이터의 표현을 배우고, 그것을 재현하지 못하는 정도에서 이탈을 읽어 낸다. 원래 좌표에서의 거리가 통하지 않는 구조의 데이터, 곧 audio, 긴 time series, 장비 trace, 무엇보다 image 를 위한 것이며, training 에 쓸 깨끗한 데이터가 충분히 있어야 한다. Image 에는 자기만의 정착된 처방이 있고, 꼭지 5.3 이 그것을 다룬다.

### 5.1. Autoencoder

Autoencoder 는 입력을 좁은 code 로 압축하고 그로부터 입력을 재구성한다. 정상 데이터만으로 training 하면 자기 용량을 정상 구조에 쓰는 표현을 배우게 되고, 그러면 재구성 오차가 anomaly score 노릇을 한다.

가정은 bottleneck 이 network 가 무엇이든 그대로 복사하는 법을 배우지 못할 만큼 좁다는 것이다. 용량을 너무 주면 본 적 없는 anomaly 도 정상 데이터만큼 충실히 재구성해 내고, 오차는 아무것도 가르지 못한다.

### 5.2. Generative Adversarial Network

Adversarial 방식은 정상 데이터와 구별되지 않는 표본을 만들어 내도록 generator 를 training 한다. 새 관측을 채점한다는 것은 그것에 가장 가까운 생성 표본을 찾아 잔차를 읽는 일이다. 정상 관측은 generator 가 배운 manifold 위에 놓여 가깝게 맞추어지고, anomaly 는 그렇지 않다.

이런 종류의 첫 방법인 [AnoGAN](#ref-20) 은 그 잔차에 discriminator feature 항을 더한다. 그 대가는 관측 하나를 채점하는 데 forward pass 한 번이 아니라 latent space 에서의 반복 탐색이 필요하다는 것이며, 뒤이은 변형들이 없애려 한 것이 바로 그것이다.

Diffusion model 이 그 뒤 같은 자리를 넘겨받아, 정상 데이터에서 잡음을 걷어 내는 법을 배우고 관측을 그 걷어 냄이 얼마나 멀리 옮겨야 하는지로 채점한다. Tabular benchmark 에서 autoencoder baseline 보다 나은 결과를 보고하지만, training 과 추론 비용은 adversarial 방법보다 다시 높다.

### 5.3. Industrial Image Inspection

시각 결함 검사는 하나의 처방으로 모였다. Pretrained network 를 image 위로 돌리고, 결함 없는 예시의 patch feature 를 보관하고, 새 patch 를 그 기억까지의 거리로 채점한다. [PatchCore](#ref-23) 가 이 처방을 세워 MVTec AD benchmark 에서 최고 99.1% 의 검출 AUROC 를 보고했고, [EfficientAD](#ref-24) 는 32 개 dataset 에 걸쳐 95.4% 의 검출 AUROC 에 image 당 2.2 ms 로 닿는데, 이 지연이 inline 검사를 가능하게 하는 값이다.

그 수치들은 공장이 아니라 benchmark 에 속한다. 실제 검사의 조명과 결함 변이를 담으려고 만든 더 어려운 MVTec AD 2 에서는, false positive rate 5% 에서 31% 의 localization AU-PRO 를 넘긴 방법이 아직 발표된 적이 없다. Benchmark 를 깔끔하게 가르는 방법이 그것만으로 line 에 나갈 준비가 된 것은 아니다.

## 6. Selection

### 6.1. By the Shape of the Data

**Table 1. Method by the shape of the data**

| Data | Method | Why |
|---|---|---|
| 변수 하나, 분포 미상 | Interquartile Range | 형태를 가정하지 않고, fence 가 25% 의 breakdown point 를 가진다. |
| 변수 하나, 근사적으로 정규, 깨끗함 | Z-Score | 문턱값이 명시된 오류율을 가진다. 단 꼭지 3.1 의 상한이 그 위에 놓일 만큼 표본이 커야 한다. |
| 변수 하나, 오염이 예상됨 | Hampel Identifier | Median 과 MAD 는 찾고 있는 outlier 가 움직이지 못하므로, 스스로를 가리는 것이 없다. |
| 변수 하나, outlier 여럿, 근사적으로 정규 | Generalized ESD | 검정 하나가 아니라 탐색 전체에 대한 수준을 명시하고, 첫 단계가 아니라 마지막으로 통과한 단계를 읽는다. |
| 변수 몇 개, 서로 상관 | Mahalanobis Distance | 공분산을 읽는 유일한 항목이며, 믿으려면 robust 한 중심과 척도가 필요하다. |
| 변수 다수, 조율할 label 없음 | ECOD | Hyperparameter 가 아예 없는 유일한 항목이며, 어느 변수가 그 관측을 극단적으로 만들었는지 말해 준다. |
| 변수도 많고 관측도 많음 | Isolation Forest | 표본 크기에 선형이고, 부분표본에서 동작하며, 분포를 가정하지 않는다. |
| 밀도가 서로 다른 cluster | Local Outlier Factor | 관측을 표본 전체가 아니라 그 이웃에 견준다. |
| 알려진 영역과 판정할 새 점 | One-Class SVM | 문제가 경계이고, 경계는 이 방법이 적합하는 것이다. |
| Audio, 긴 time series, 장비 trace | Autoencoder | 원래 좌표의 거리가 통하지 않는 곳에서도 재구성 오차는 살아남는다. |
| 되풀이 생산되는 제품의 image | Patch feature memory | 꼭지 5.3 의 pretrained feature 가 결함의 모습을 이미 담고 있고, 채점이 inline 으로 돌릴 만큼 빠르다. |
| 생산 lot 안의 부품 | [Part average testing](#appendix-c-semiconductor-practice) | 표준이 규칙을 이름 지어 두어, 한계값을 다투는 대신 감사할 수 있다. |
| 장비 sensor trace | [Multivariate control chart](#appendix-c-semiconductor-practice) | 점수를 $T^2$ 와 $Q$ 로 나누면 무엇인가 움직였다는 것만이 아니라 어느 sensor 를 보아야 하는지를 말해 준다. |

### 6.2. By the Axis Answered

Table 1 은 데이터에 대한 기술에서 방법을 고른다. Table 2 는 반대 방향으로 읽어, 각 방법이 꼭지 2 의 질문 가운데 실제로 어느 것에 답하는지를 말한다.

**Table 2. Where each method sits on the axes of section 2**

| Method | Form (2.1) | Reference set (2.2) | Labels (2.6) | Count (2.7) |
|---|---|---|---|---|
| Z-Score | Point | Global | Unsupervised | Single |
| Interquartile Range | Point | Global | Unsupervised | 통제하지 않음 |
| Hampel Identifier | Point | Global | Unsupervised | 통제하지 않으나 masking 이 일어날 수 없음 |
| Generalized ESD | Point | Global | Unsupervised | **Multiple, 명시된 수준에서** |
| Mahalanobis Distance | Point | Global | Unsupervised | Single |
| Isolation Forest | Point | Global | Unsupervised | 통제하지 않음 |
| One-Class SVM | Point | Global | Semi-supervised | 통제하지 않음 |
| Local Outlier Factor | Point | **Local** | Unsupervised | 통제하지 않음 |
| ECOD | Point | Global | Unsupervised | 통제하지 않음 |
| Autoencoder | Point 또는 collective | Global | Semi-supervised | 통제하지 않음 |
| Adversarial and diffusion | Point 또는 collective | Global | Semi-supervised | 통제하지 않음 |
| Patch feature memory | Collective, 공간에서 | Global | Semi-supervised | 통제하지 않음 |

굵게 쓴 두 칸이 그 열들이 가진 변화의 전부이다. Local outlier factor 는 reference set 을 바꾸는 유일한 방법이고, generalized ESD 는 outlier 를 몇 개 찾을지를 통제하는 유일한 방법이다. 다른 모든 방법은 이웃이 앉은 자리에 함께 앉으며, 그래서 그 둘이 따로 항목을 얻는다.

Deep 방법들은 방법이 아니라 데이터를 바꾸어 collective anomaly 에 닿는다. Series 위의 window 나 image 의 patch 가 vector 하나가 되고, 꼭지 2.1 의 collective anomaly 가 그 vector 에서 point anomaly 가 된다. 꼭지 5 의 어느 것도 관측의 연속을 직접 채점하지 않는다.

네 개의 축이 표에서 빠져 있는데, 셋은 여기 있는 어느 방법도 답하지 않기 때문이고 하나는 모든 방법이 똑같이 답하기 때문이다.

- **Cause (2.3).** 어느 통계량도 오류와 드문 사건을 가르지 못하며, 그 꼭지가 그것을 그대로 말한다.
- **Discordancy and contamination (2.4).** 여기 있는 모든 방법이 discordancy 를 검정하고 어느 것도 contamination 을 판정하지 않으므로, 이 축은 어느 방법이 결과를 내놓는가가 아니라 결과를 어떻게 읽는가를 가른다.
- **Position in a regression (2.5).** Leverage 와 influence 는 적합된 model 을 요구하는데, 이 문서는 그 대신 분포와 영역을 적합한다.
- **Time series type (2.8).** Window 를 쓰는 방법이 level shift 에 flag 를 붙일 수는 있지만, 여기 있는 어느 것도 level shift 와 temporary change 를 가르지 못한다.

Contextual anomaly 가 form 축에서 손에 닿지 않는 것도 같은 이유이다. 그것은 견줄 context 변수를 요구하는데, 여기 있는 어느 방법도 그것을 받지 않는다.

### 6.3. What the Benchmarks Report

발표된 비교들은 승자를 이름 대지 않는다. [ADBench](#ref-22) 의 알고리즘 30 개와 dataset 57 개에 걸쳐, 어느 unsupervised 방법도 나머지보다 통계적으로 우월하지 않다. Isolation Forest 와 ECOD 가 압도하지는 않으면서 꾸준히 나은 축에 들고, tabular 데이터를 위해 만든 deep 방법 여럿이 그 아래에 놓인다. 따라서 새롭다는 것은 그 자체로 갈아탈 이유가 되지 않으며, tabular 데이터에서는 꼭지 3 과 4 의 값싼 방법들이 정직한 기본값으로 남는다.

예외는 원래 좌표가 쓸 만한 거리를 담고 있지 않은 경우이다. Deep 방법들이 자기 비용을 벌어들이는 자리가 거기이며, image 가 그 가장 뚜렷한 사례이다.

### 6.4. Two Habits

선택 자체보다 더 중요한 습관이 둘 있다. 데이터를 보기 전에 문턱값을 고정하여, 좋아하는 답이 나오도록 그것을 조율하지 않는다. 그다음에는 판정이 아니라 여유를 읽는다. Cut-off 를 간신히 넘긴 통계량과 넉넉한 차이로 넘긴 통계량은 서로 다른 발견이고, 위의 선택들이 바뀌어도 살아남는 것은 뒤의 것뿐이기 때문이다.

## References

<a id="ref-1"></a>
[1] Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley, Reading. [https://www.pearson.com](https://www.pearson.com). ISBN 978-0-201-07616-5.<br>
<a id="ref-2"></a>
[2] Barnett, V., & Lewis, T. (1994). *Outliers in Statistical Data*, 3rd edition. Wiley, Chichester. [https://www.wiley.com/en-us/Outliers+in+Statistical+Data,+3rd+Edition-p-9780471930945](https://www.wiley.com/en-us/Outliers+in+Statistical+Data,+3rd+Edition-p-9780471930945). ISBN 978-0-471-93094-5.<br>
<a id="ref-3"></a>
[3] Fox, A. J. (1972). [Outliers in Time Series](https://doi.org/10.1111/j.2517-6161.1972.tb00912.x). *Journal of the Royal Statistical Society: Series B*, 34(3), 350–363.<br>
<a id="ref-4"></a>
[4] Cook, R. D. (1977). [Detection of Influential Observation in Linear Regression](https://doi.org/10.1080/00401706.1977.10489493). *Technometrics*, 19(1), 15–18.<br>
<a id="ref-5"></a>
[5] Hawkins, D. M. (1980). [Identification of Outliers](https://doi.org/10.1007/978-94-015-3994-4). Monographs on Applied Probability and Statistics. Chapman and Hall, London. ISBN 978-94-015-3996-8.<br>
<a id="ref-6"></a>
[6] Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). [Regression Diagnostics: Identifying Influential Data and Sources of Collinearity](https://doi.org/10.1002/0471725153). Wiley, New York. ISBN 978-0-471-05856-4.<br>
<a id="ref-7"></a>
[7] Chen, C., & Liu, L.-M. (1993). [Joint Estimation of Model Parameters and Outlier Effects in Time Series](https://doi.org/10.1080/01621459.1993.10594321). *Journal of the American Statistical Association*, 88(421), 284–297.<br>
<a id="ref-8"></a>
[8] Chandola, V., Banerjee, A., & Kumar, V. (2009). [Anomaly Detection: A Survey](https://doi.org/10.1145/1541880.1541882). *ACM Computing Surveys*, 41(3), Article 15.<br>
<a id="ref-9"></a>
[9] Brys, G., Hubert, M., & Struyf, A. (2004). [A Robust Measure of Skewness](https://doi.org/10.1198/106186004X12632). *Journal of Computational and Graphical Statistics*, 13(4), 996–1017.<br>
<a id="ref-10"></a>
[10] Hubert, M., & Vandervieren, E. (2008). [An Adjusted Boxplot for Skewed Distributions](https://doi.org/10.1016/j.csda.2007.11.008). *Computational Statistics and Data Analysis*, 52(12), 5186–5201.<br>
<a id="ref-11"></a>
[11] Mahalanobis, P. C. (1936). On the Generalised Distance in Statistics. *Proceedings of the National Institute of Sciences of India*, 2(1), 49–55. [https://www.insa.nic.in](https://www.insa.nic.in).<br>
<a id="ref-12"></a>
[12] Shiffler, R. E. (1988). [Maximum Z Scores and Outliers](https://doi.org/10.1080/00031305.1988.10475530). *The American Statistician*, 42(1), 79–80.<br>
<a id="ref-13"></a>
[13] Rosner, B. (1983). [Percentage Points for a Generalized ESD Many-Outlier Procedure](https://doi.org/10.1080/00401706.1983.10487848). *Technometrics*, 25(2), 165–172.<br>
<a id="ref-14"></a>
[14] Iglewicz, B., & Hoaglin, D. C. (1993). *How to Detect and Handle Outliers*. The ASQC Basic References in Quality Control: Statistical Techniques, Vol. 16. ASQC Quality Press, Milwaukee. [https://asq.org/quality-press](https://asq.org/quality-press). ISBN 978-0-87389-247-6.<br>
<a id="ref-15"></a>
[15] ISO 16269-4:2010, *Statistical interpretation of data — Part 4: Detection and treatment of outliers*. International Organization for Standardization. [https://www.iso.org/standard/44396.html](https://www.iso.org/standard/44396.html)<br>
<a id="ref-16"></a>
[16] Rousseeuw, P. J., & Van Driessen, K. (1999). [A Fast Algorithm for the Minimum Covariance Determinant Estimator](https://doi.org/10.1080/00401706.1999.10485670). *Technometrics*, 41(3), 212–223.<br>
<a id="ref-17"></a>
[17] Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). [LOF: Identifying Density-Based Local Outliers](https://doi.org/10.1145/335191.335388). *ACM SIGMOD Record*, 29(2), 93–104.<br>
<a id="ref-18"></a>
[18] Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001). [Estimating the Support of a High-Dimensional Distribution](https://doi.org/10.1162/089976601750264965). *Neural Computation*, 13(7), 1443–1471.<br>
<a id="ref-19"></a>
[19] Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). [Isolation Forest](https://doi.org/10.1109/ICDM.2008.17). *Proceedings of the Eighth IEEE International Conference on Data Mining*, 413–422.<br>
<a id="ref-20"></a>
[20] Schlegl, T., Seeböck, P., Waldstein, S. M., Schmidt-Erfurth, U., & Langs, G. (2017). [Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://doi.org/10.1007/978-3-319-59050-9_12). *Information Processing in Medical Imaging*, Lecture Notes in Computer Science 10265, 146–157.<br>
<a id="ref-21"></a>
[21] Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C., & Chen, G. H. (2022). [ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions](https://doi.org/10.1109/TKDE.2022.3159580). *IEEE Transactions on Knowledge and Data Engineering*, 35(12), 12181–12193.<br>
<a id="ref-22"></a>
[22] Han, S., Hu, X., Huang, H., Jiang, M., & Zhao, Y. (2022). [ADBench: Anomaly Detection Benchmark](https://arxiv.org/abs/2206.09426). *Advances in Neural Information Processing Systems 35, Datasets and Benchmarks Track*.<br>
<a id="ref-23"></a>
[23] Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P. (2022). [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265). *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 14318–14328.<br>
<a id="ref-24"></a>
[24] Batzner, K., Heckler, L., & König, R. (2024). [EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies](https://arxiv.org/abs/2303.14535). *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, 128–138.<br>
<a id="ref-25"></a>
[25] AEC-Q001 Rev-D (2011), *Guidelines for Part Average Testing*. Automotive Electronics Council. [http://www.aecouncil.com/AECDocuments.html](http://www.aecouncil.com/AECDocuments.html)<br>
<a id="ref-26"></a>
[26] Hsu, C.-Y., Chien, C.-F., & Lin, K.-Y. (2012). [Semiconductor Fault Detection and Classification for Yield Enhancement and Manufacturing Intelligence](https://doi.org/10.1007/s10696-012-9161-4). *Flexible Services and Manufacturing Journal*, 24(3), 358–378.

---

## Appendix A. Terminology

- **adjusted boxplot** — 표본의 왜도에 따라 fence 를 옮긴 box plot 으로, 긴 꼬리가 outlier 의 연속으로 읽히지 않게 한다.
- **anomaly score** — 정상 pattern 에서 얼마나 벗어났는지로 관측의 순위를 매기는 수이며, 그 값 어디에도 명시된 오류율이 붙어 있지 않다.
- **AU-PRO** — 영역별 겹침 곡선 아래 면적으로, 결함을 검출했는지가 아니라 얼마나 잘 국소화했는지를 채점한다.
- **AUROC** — 수신자 조작 특성 곡선 아래 면적으로, 무작위로 고른 anomaly 가 무작위로 고른 정상 관측보다 높은 점수를 받을 확률이다.
- **box plot** — 상자가 interquartile range 를 덮고 수염이라 부르는 선이 상자에서 interquartile range 의 1.5 배 안에 있는 가장 극단적인 관측까지 뻗으며, 그 너머의 것은 따로 점으로 그리는 요약 그림.
- **breakdown point** — 추정값이 나머지 데이터를 더는 기술하지 못하게 되기까지 망가뜨려야 하는 표본의 비율. 평균은 0%, median 은 50% 이다.
- **chi-square distribution** — 독립인 표준정규 변수의 제곱합이 따르는 분포로, 항마다 자유도 하나를 가진다. 제곱 거리를 확률로 바꾸는 것이 이것이다.
- **consistency constant** — Robust 척도 추정값에 곱하거나 나누어, 가정한 분포 아래에서 표준편차로 수렴하게 하는 인자. MAD 에서는 0.674490 이고 interquartile range 에서는 1.349 이며, [AEC-Q001](#ref-25) 은 뒤의 것을 1.35 로 반올림한다.
- **contaminant** — 나머지 표본이 따르는 분포가 아닌 다른 분포에서 온 관측이며, 단지 나머지와 어긋나 보이는 관측과는 다르다.
- **contamination** — 가정한 분포에서 오지 않은 표본의 비율.
- **critical value** — 검정 통계량이 유의하다고 불리려면 넘어야 하는 값. 검정 대상 데이터가 아니라 유의수준과 표본 크기에서 나온다.
- **cumulative distribution function** — 각 값에 대해 그 값 이하로 떨어질 확률을 주는 함수. 표준정규의 것은 $\Phi$ 로 쓰고, 그 역함수는 확률을 다시 표준편차의 개수로 바꾼다.
- **degrees of freedom** — 통계량이 자유롭게 변할 수 있는 독립한 양의 개수. 제곱 거리를 어느 chi-square 분포에 견주어 읽을지를 정하며, 여기서는 변수마다 하나이다.
- **discordant observation** — 나머지 표본과 통계적으로 어긋나 보이는 관측. Discordancy 의 검정이 보고하는 것은 이것이지 contamination 이 아니다.
- **discriminator** — 생성된 표본을 실제 표본과 가르도록 generator 와 나란히 training 하는 network. 그 내부 feature 는 관측을 generator 가 만든 것과 견주는 데 다시 쓸 수 있다.
- **ECOD** — Empirical cumulative distribution 에 기반한 outlier 검출로, 꼭지 4.4 의 방법이다.
- **ESD** — Extreme studentized deviate 의 줄임말로, 꼭지 3.4 의 generalized ESD 절차의 이름에 쓰였다.
- **extreme studentized deviate** — 표본 평균으로부터의 가장 큰 절대편차를 표본 표준편차로 나눈 값. Generalized ESD 절차의 각 단계가 계산하는 통계량이다.
- **false positive rate** — 규칙이 flag 를 붙이는 정상 관측의 비율. 그 규칙이 닿는 검출률에 대해 치르는 값이다.
- **generator** — Discriminator 가 training 데이터와 가르지 못할 표본을 만들어 내도록 training 하는 network. 일단 training 되면 정상 데이터가 나온 분포를 대신한다.
- **Hotelling's T-squared** — 제곱 z-score 의 다변량 대응물로, model 이 적합한 구조 안에서 관측이 중심으로부터 떨어진 거리를 잰다.
- **hyperparameter** — 데이터에서 추정하지 않고 방법을 돌리기 전에 정하는 설정값으로, 이웃 크기나 kernel bandwidth 같은 것이다. Label 이 없으면 그것을 맞추어 볼 대상이 없다.
- **influential observation** — 제거하면 적합된 model 이 눈에 띄게 바뀌는 관측으로, Cook 의 거리로 잰다.
- **interquartile range** — 1 사분위수에서 3 사분위수까지의 거리이며, 표본에서 가운데 절반의 퍼짐이다. 정규 표본에서는 표준편차의 1.349 배이다.
- **IQR** — 이 문서에서 interquartile range 를 가리키는 데 쓰는 줄임말.
- **kernel** — One-class SVM 이 일하는 기하 구조를 정하는 함수이며, 그와 함께 학습된 경계가 취할 수 있는 모양도 정한다.
- **latent space** — 생성 model 이 오가며 대응시키는 압축된 좌표로, 그 안의 한 점이 재구성된 관측 하나를 대신한다.
- **leverage** — 적합된 model 의 예측변수에서 관측이 얼마나 극단적인가이며, 그것이 적합을 실제로 움직이든 아니든 움직일 수 있는 폭을 정한다.
- **loading** — 주성분이 원래 변수 하나에 주는 가중치로, 성분 공간에서 올라온 flag 를 그것을 통해 sensor 까지 되짚는다.
- **lot** — 함께 처리되어 제조 공정을 한 단위로 지나가는 부품의 묶음. Part average testing 이 부품을 견주는 상대가 이 group 이다.
- **manifold** — 데이터가 실제로 차지하는, 전체 공간 안의 더 낮은 차원의 면. 그것을 배운 생성 model 은 그 위의 점은 재현하고 그 밖의 점은 재현하지 못한다.
- **masking** — Outlier 하나가 자기가 견주어지는 중심이나 척도를 충분히 부풀려, 자기 자신이나 두 번째 outlier 가 더는 극단적으로 보이지 않게 되는 효과.
- **medcouple** — 마이너스 1 과 1 사이에 있고 대칭 표본에서 0 이 되는 robust 한 왜도 척도로, median 양쪽 관측들을 견준 비교의 median 으로 만든다.
- **median absolute deviation (MAD)** — 관측들이 표본 median 에서 떨어진 절대편차의 median 으로, 소수의 극단적인 관측이 부풀릴 수 없는 척도 추정값으로 쓴다.
- **minimum covariance determinant** — 다변량 중심과 공분산의 robust 한 추정값으로, 공분산 행렬의 행렬식이 가장 작은 관측 부분집합에서 얻는다.
- **MVTec AD** — 공산품 사진으로 만든 공개 benchmark 로, training 용은 결함이 없고 test 용은 결함이 있으며 결함 영역이 표시되어 있다. MVTec AD 2 는 더 어렵게 만든 뒤의 집합이다.
- **novelty detection** — 새 관측을 outlier 가 없다고 가정한 training set 에 견주어 판정하는 것으로, 이미 outlier 를 담고 있을 수 있는 표본 하나를 뒤지는 것과 다르다.
- **order statistic** — 값이 아니라 정렬된 표본에서의 순위로 지목되는 관측으로, median 이나 사분위수 같은 것이다. 극단적인 관측을 더 멀리 옮겨도 그것은 움직이지 않는다.
- **outlier** — 나머지 표본이 따르는 분포와 어긋나는 관측. 이 label 은 model 과의 정합성에 대한 것이며, 그 자체로 관측이 틀렸음을 밝히지는 않는다.
- **pretrained network** — 크고 일반적인 dataset 에서 적합한 뒤 더 training 하지 않고 쓰는 network 로, 자기 출력이 아니라 중간 층이 내놓는 feature 를 쓰려는 것이다.
- **principal component** — 이미 적합된 방향들과 상관이 없다는 조건 아래, 분산이 가장 큰 쪽으로 데이터에 적합한 방향. 서로 상관된 sensor 들에서는 보통 몇 개가 변동의 대부분을 담는다.
- **reconstruction error** — 입력과, model 이 그 입력을 압축했다가 다시 세워 내놓은 출력 사이의 거리.
- **robust** — 소수의 오염된 관측이 멀리 움직이지 못하는 추정값을 이르는 말. 그 소수가 얼마나 커도 되는지는 breakdown point 가 말한다.
- **significance level** — 표본이 실제로 깨끗한데도 flag 를 붙일 확률로, 데이터를 보기 전에 정한다. 되풀이를 셈에 넣지 않고 검정을 되풀이하면 이것이 고른 값 위로 올라간다.
- **specification limit** — 부품이 팔리려면 측정된 parameter 가 그 안에 머물러야 하는 경계로, 표본이 아니라 설계에서 정한다. 부품이 이것을 통과하고도 자기 lot 안에서는 outlier 일 수 있다.
- **squared prediction error (Q statistic)** — 적합된 model 이 설명하지 못한 관측의 부분으로, 관측에서 그 model 의 공간 안 재구성까지의 제곱 거리로 잰다.
- **SVM** — Support vector machine 으로, kernel 이 정한 기하 구조에서 얻을 수 있는 가장 넓은 여백으로 class 를 가르는 분류기. 꼭지 4.2 의 one-class 변형에는 두 번째 class 가 없어, 가진 하나를 감싼다.
- **swamping** — Outlier 가 중심이나 척도를 충분히 일그러뜨려, 깨끗한 관측까지 그것과 함께 flag 되는 효과.

## Appendix B. Tukey's Rule

꼭지 3.2 는 이 규칙을 한 줄로 진술한다. 이 appendix 는 1.5 라는 배수가 어디에서 왔는지, 그것이 z-score 에 견주어 무엇을 치르는지, 그리고 이 규칙이 어디에서 통하지 않게 되는지를 적는다.

### B.1. Inner and Outer Fences

[Tukey (1977)](#ref-1) 은 fence 를 한 쌍이 아니라 두 쌍 그렸다. 안쪽 쌍이 꼭지 3.2 의 규칙이고, 바깥쪽 쌍은 interquartile range 의 1.5 배가 아니라 3 배에 놓인다.

$$Q_1 - c \cdot \mathrm{IQR} \ \le \ x_i \ \le \ Q_3 + c \cdot \mathrm{IQR}$$

- $c$ — Fence 를 놓는 배수로, 안쪽 쌍에서는 1.5 이고 바깥쪽 쌍에서는 3 이다.
- $Q_1$, $Q_3$, $\mathrm{IQR}$ — 꼭지 3.2 와 같다.

안쪽 fence 를 넘어간 관측을 Tukey 는 **outside** 라 불렀고, 바깥쪽 fence 를 넘어간 관측을 **far out** 이라 불렀다. Box plot 은 안쪽 fence 안의 마지막 관측까지 수염을 그리므로, 그림에서 따로 점으로 보이는 것은 모두 적어도 outside 이다.

두 fence 는 함께 읽으라고 있는 것이다. Outside 는 그 관측을 한 번 보아야 한다고 말하고, far out 은 어떻게 읽어도 극단적이라고 말하는데, 이는 cut-off 하나로는 지을 수 없는 구분이다.

### B.2. What the Multiple Costs

1.5 라는 배수는 유도된 것이 아니라 편의로 고른 것이다. 그것이 정규 표본에서 하는 일은 정확히 적어 둘 값어치가 있다. 꼭지 3.2 가 이 규칙을 3 의 z-score 와 비슷하다고 부르지만 둘은 같지 않기 때문이다.

**Table 3. Where each fence sits on a normal sample**

| Rule | Position | Share of a normal sample flagged |
|---|---|---|
| Inner fence, $c = 1.5$ | 2.6980 $\sigma$ | 0.6977% |
| Outer fence, $c = 3$ | 4.7214 $\sigma$ | 0.0002% |
| Classical rule at 3 | 3.0000 $\sigma$ | 0.2700% |

따라서 안쪽 fence 는 3 의 z-score 보다 2.6 배 느슨하고, 1.5 대신 1.724 의 배수라면 3 의 z-score 가 놓인 자리에 정확히 놓였을 것이다. 바깥쪽 fence 는 그 둘보다 세 자릿수만큼 엄격하며, 그래서 far out 이 강한 진술이 된다.

안쪽 fence 에 맞는 말은 같다가 아니라 비슷하다이다. 두 규칙 모두 표준편차 2 의 규칙이 5% 를 flag 할 자리에서 1% 도 되지 않는 몫에 flag 를 붙이며, 둘 사이의 선택은 그 몫이 아니라 오염에 달려 있다.

### B.3. Skewed Samples

Fence 는 대칭으로 놓인다. $Q_1$ 아래와 $Q_3$ 위에 같은 배수로 놓이는 것이다. 왜도가 있는 표본에서 그것은 틀린 모양이다. 긴 꼬리는 분포의 성질인데 규칙은 그것을 outlier 의 연속으로 읽고, 짧은 쪽에는 아무것도 flag 하지 않는다.

Lognormal 표본이 그 크기를 뚜렷이 보여 준다. 200,000 개를 뽑으면 표준 fence 는 위쪽 fence 너머에서 표본의 6.22% 에 flag 를 붙이고 아래쪽 fence 너머에서는 하나도 붙이지 않는데, 그 데이터에는 오염이 전혀 없다.

[Hubert and Vandervieren (2008)](#ref-10) 의 adjusted boxplot 은 [Brys, Hubert and Struyf (2004)](#ref-9) 의 medcouple 로 잰 표본의 왜도에 따라 각 fence 를 옮겨 이것을 고친다.

$$\left[ \ Q_1 - 1.5 e^{a \cdot \mathrm{MC}} \cdot \mathrm{IQR}, \quad Q_3 + 1.5 e^{b \cdot \mathrm{MC}} \cdot \mathrm{IQR} \ \right]$$

- $\mathrm{MC}$ — Medcouple 로, $-1$ 과 $1$ 사이에 있고 대칭 표본에서 0 이 되는 robust 한 왜도 척도이다.
- $a$, $b$ — $\mathrm{MC} \ge 0$ 일 때 $-4$ 와 $3$ 이고 음수일 때 $-3$ 과 $4$ 여서, 긴 쪽의 fence 는 밖으로 나가고 짧은 쪽의 fence 는 안으로 들어온다.

같은 lognormal 표본에서 medcouple 은 0.3264 이고, adjusted fence 는 6.22% 와 없음 대신 위쪽에서 1.10%, 아래쪽에서 0.42% 에 flag 를 붙인다. 이 규칙도 정규 표본이 줄 것보다는 여전히 많이 flag 하지만, 분포의 모양을 outlier 의 목록으로 보고하지는 않는다.

## Appendix C. Semiconductor Practice

Fab 이 실제로 돌리는 방법은 가장 새로운 것이 아니다. 표준이 이름을 붙였고, 감사자가 확인할 수 있고, 기술자가 그에 따라 움직일 수 있는 것이다. 그 가운데 둘은 꼭지 3 부터 5 까지의 옆에 놓아 볼 값어치가 있는데, 둘 다 거기에서 이미 다룬 구성으로 드러나기 때문이다.

### C.1. Part Average Testing

Part average testing 은 측정값이 모두 specification limit 을 통과하더라도 자기 lot 에서 비정상인 parameter 를 가진 부품을 걸러 낸다. AEC-Q001 이 자동차 부품에 대해 이를 정의하며, 꼭지 3.3 의 설계 위에 세워져 있다. Robust mean 은 median 이고, robust sigma 는 interquartile range 를 1.35 로 나눈 것이다. 부품은 아래 구간 안에 들 때 남는다.

$$\tilde{x} \pm k \cdot \frac{\mathrm{IQR}}{1.35}$$

- $\tilde{x}$ — 판정 대상 부품들에 걸친 그 parameter 의 median 으로, 표준은 이것을 robust mean 이라 부른다.
- $\mathrm{IQR}$ — 그것들의 interquartile range 이며, $\mathrm{IQR}/1.35$ 가 표준이 robust sigma 라 부르는 것이다.
- $k$ — 한계값을 그 sigma 의 몇 배에 둘지를 정하는 배수로, 관례상 6 이다.

그 제수는 꼭지 3.2 의 $1.349\sigma$ 를 반올림한 것이다. 그것으로 나누면 사분위수의 퍼짐이 다시 표준편차가 되며, 이는 $\Phi^{-1}(0.75)$ 가 MAD 에 하는 일과 똑같다. 표준은 MAD 대신 사분위수를 고르고 3.5 대신 6 을 고르지만, 구성은 같다. Robust 한 중심, 정규 단위로 다시 잰 robust 한 척도, 그리고 그 척도의 배수이다.

Static 한 한계값은 과거 데이터에서 한 번 계산하여 모든 lot 에 적용한다. Dynamic 한 한계값은 lot 마다 다시 계산하여, 전체가 고르게 밀려 있으면서 내부는 촘촘한 lot 을 잡아낸다. 다만 사분위수가 뜻을 가지려면 lot 마다 최소 표본이 필요하며, 표준에서는 30 개이다.

### C.2. Fault Detection and Classification

장비 sensor 는 한 공정 단계 내내 압력, 유량, 전력, 온도를 보고한다. [Fault detection and classification](#ref-26) 은 각 trace 를 wafer 마다의 요약 parameter 로 줄인 다음, 그 parameter 들을 하나씩이 아니라 함께 감시한다. 변수들이 함께 움직이므로 변수별 한계값은 조합에서만 드러나는 이탈을 놓치기 때문이다.

표준적인 구성은 축소된 공간에서 돌리는 꼭지 3.5 의 multivariate control chart 이다. 정상 생산에서 주성분을 적합하고, 그 공간 안에서는 Hotelling 의 $T^2$ 로, 성분이 설명하지 못한 부분에 대해서는 squared prediction error 인 $Q$ 통계량으로 관측을 채점한다. 둘은 서로 다른 질문에 답한다. $T^2$ 는 process 가 평소의 구조 안에서 움직였다고 말하고, $Q$ 는 그 구조를 벗어났다고 말한다.

점수를 그렇게 나누는 것이 flag 를 움직일 수 있는 것으로 만든다. $T^2$ 나 $Q$ 에 가장 크게 이바지한 loading 이 들여다볼 sensor 를 이름 대며, 그것이 tool 을 세우기만 하는 chart 와 왜 그런지도 말해 주는 chart 의 차이이다.
