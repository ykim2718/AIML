# Referenced R² — Choosing the Baseline in the R² Denominator (Korean)
Rev. 1 | Created: 2026-09-04 | Updated: 2026-09-04 14:57 CDT

> 이 문서는 표준 R² 의 분모를 자료에서 계산하지 않고 바깥에서 정한 기준으로 바꿀 수 있는지,
> 그리고 그렇게 바꾸고 나면 그 수가 무엇을 뜻하게 되는지를 묻는다.

## 1. Question

표준 R² 는 다음과 같이 정의된다.

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2} \hspace{19em} (1)$$

분모 `SS_tot` 는 자료 자신의 평균 `y_bar` 를 중심으로 잡은 총제곱합이며, 흔히 자료의 총산포라
부르는 양이다. 그 분모를 자료에서 계산하는 대신 바깥에서 정한 값으로 바꿀 수 있는지가 이 문서의
물음이다.

답은 바꿀 수 있다는 것이다. 원리로만 자연스러운 것이 아니라, 이미 자리 잡은 형태가 여럿 있다.
다만 분모가 바뀌는 순간 지표의 뜻도 바뀌므로, 무엇으로 바꾸었는지를 언제나 함께 보고해야 한다.

분모를 이렇게 정해서 만든 지표를 이 문서에서는 **Referenced R²** 라 부르고 `Ref_R2` 로 적는다.
Section 3 은 분모에 무엇이 들어가는지로 개별 변형을 나눈다. `R2_oos` (out-of-sample), `R2_frd`
(fixed reference dispersion), `R2_base` (baseline model) 이다.

## 2. Structure

분모를 바꾸어도 되는 이유는 R² 를 다시 읽으면 드러난다. `SS_tot` 는 언제나 `y_bar` 를 답하는
model 이 남기는 제곱오차합과 정확히 같다. 즉 R² 는 처음부터 아래의 꼴이었다.

```text
                error left by the model
R2 = 1 -  ───────────────────────────────────
            error left by a chosen baseline
```

표준 R² 는 baseline 을 평균만 돌려주는 model 로 고정한 특수한 경우이며, 이 구조는 skill score 라는
이름으로 널리 쓰인다. **분모를 정한다는 것은 baseline 을 바꾸는 것이고, 그 이상도 그 이하도
아니다.** 이 사실이 문서의 나머지를 지배한다.

따라서 `Ref_R2` 는 새로운 계산이 아니라 baseline 을 명시적으로 고른 R² 이며, 표준 R² 는 그
baseline 을 자료의 평균에 암묵적으로 둔 경우이다.

분자와 분모가 모두 y 의 제곱 단위이므로 그 비는 무차원이고, `Ref_R2` 는 baseline 이 감수해야 했던
오차 가운데 model 이 얼마나 덜어냈는지를 백분율로 읽힌다. 그 읽기는 baseline 이 무엇이든 성립한다.

## 3. Variants

**Table 1. Ways to choose the baseline**

| Symbol | Denominator | Baseline it encodes | Typical use | Reference |
|---|---|---|---|---|
| `R²` | `Σ(y_i − y_bar)²` | 이 자료의 평균 | 한 자료 안에서의 적합도 | [[1](#ref-1)] |
| `R2_oos` | `Σ(y_i − y_train_bar)²` | 학습 시점에 알던 평균 | Test set 평가 | [[2](#ref-2)] |
| `R2_frd` | `N · sigma_ref²` | Spec 이 허용하는 산포 | Lot, batch, 기간 사이의 비교 | [[3](#ref-3)], [[4](#ref-4)] |
| `R2_base` | `Σ(y_i − y_base_i)²` | 표본마다 답하는 기준 model | 시계열, 운영 중인 model 과의 비교 | [[5](#ref-5)] |

Reference 열은 각 형태가 실제로 쓰이는 곳을 가리키며, 서지 사항은 [References](#references) 에 있다.
넷 가운데 이 문서가 지어낸 계산은 하나도 없고, 각각은 자기 분야에서 이미 표준이다. 이 문서가 더한
것은 산술이 아니라 표기이며, section 4 가 그 이야기로 돌아온다.

### 3.1. Fixed Reference Point

가장 흔한 경우이다. Test set 을 평가할 때 분모의 기준은 test 자료의 평균이 아니라 학습 자료의 평균
`y_train_bar` 에 묶는다.

$$R^2_{oos} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y}_{train})^2} \hspace{19em} (2)$$

이것이 옳은 꼴인 이유는 둘이다. 첫째, test 자료의 평균은 평가 시점에 알 수 없으므로 그것을 baseline
에 넣는 일은 미래에서 정보를 빌려오는 셈이 된다. 둘째, 분모가 test set 을 어떻게 잘랐는지에 따라
움직여 잣대 자체가 흔들린다. 학습 평균에 묶으면 지표는 학습 중에 알던 평균을 계속 돌려주는 model
보다 얼마나 나아졌는가 하는 물음이 되고, 자료를 어떻게 나누든 잣대가 유지된다.

이 형태는 바깥에서 out-of-sample R² 로 알려져 있으므로, 외부 보고에는 그 이름을 쓴다.

### 3.2. Fixed Reference Dispersion

분모를 자료에서 계산하지 않고 이미 알고 있는 기준 분산으로 바꾼다.

$$R^2_{frd} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{N \cdot \sigma_{ref}^2} \hspace{19em} (3)$$

`sigma_ref` 에는 공정 spec 이 허용하는 산포, 쌓아 온 이력 자료의 분산, 계측 system 의 기준 분산
같은 도메인 기준을 쓴다. 두께 예측이라면 해당 층의 관리 spec 산포를 분모에 넣는 것으로 지표는
고정된 잣대에 견준 model 의 척도가 된다.

분모가 상수이므로 식은 한 단계 더 줄어든다. 분자 `SS_res` 를 표본 수 `N` 으로 나눈 것의 제곱근이
root mean squared error, 곧 RMSE 이다.

$$\mathrm{RMSE} = \sqrt{\frac{SS_{res}}{N}} \hspace{19em} (4)$$

RMSE 는 오차를 y 와 같은 단위로 돌려놓으므로 `sigma_ref` 로 바로 나눌 수 있다. `SS_res` 에
`N · RMSE²` 를 넣으면 `N` 이 약분되고 아래만 남는다.

$$R^2_{frd} = 1 - \left(\frac{\mathrm{RMSE}}{\sigma_{ref}}\right)^2 \hspace{19em} (5)$$

이것이 이 변형의 물리적 의미이다. `R2_frd` 는 **model 의 오차가 spec 산포의 몇 배인지를 재어** 그것을
1 에서 뺀다. Spec 의 절반이면 0.75, spec 과 같은 오차면 0, spec 을 넘는 오차면 음수가 된다. Baseline
의 말로 옮기면 기준은 spec 이 허용하는 만큼의 오차를 내는 가상의 model 이고, 물음은 그것을 model 이
이겼는가이다.

계측 분야에는 오차와 spec 의 비를 R² 로 꾸미지 않고 그대로 보고하는 관행이 있다. 읽는 이가 공정이나
계측 engineer 라면 `RMSE / sigma_ref` 를 그대로 건네는 편이 잘 통하고, 그 수가 다른 지표들과 한 표에
앉아야 한다면 `R2_frd` 로 바꾸어 0 과 1 사이의 척도에 올린다. 둘은 위 식으로 서로 변환되므로 같은
정보를 지닌다.

### 3.3. Baseline Model

Section 3.1 의 baseline 은 어떤 표본을 만나든 `y_train_bar` 라는 수 하나로 답하고, section 3.2 의
baseline 은 어떤 표본을 만나든 `sigma_ref` 만큼의 오차를 낸다. 둘 다 기준이 표본에 따라 변하지 않는
상수 baseline 이다.

기준 자체가 표본마다 달라져야 하는 경우는 그 꼴로 적을 수 없다. 그때 baseline 은 각 i 마다 자기 값
`y_base_i` 를 가진다.

$$R^2_{base} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - y_{base,i})^2} \hspace{19em} (6)$$

시계열의 persistence baseline 이 대표적인 경우이다. 직전 단계의 값을 다음 단계의 예측으로 쓰므로
`y_base_i` 가 i 마다 다르고, 어떤 상수로도 같은 분모를 만들 수 없다. 계절 평균을 baseline 으로 삼을
때도, 운영 중인 model 을 baseline 으로 삼아 교체의 가치를 잴 때도 마찬가지이다.

`y_base_i` 를 상수 `y_train_bar` 로 두면 이것은 section 3.1 로 줄어들므로, 그 절은 이 절의 특수한
경우이다. 그런데도 따로 두는 것은 둘이 다른 물음에 답하기 때문이다. Section 3.1 은 model 이 평균을
이기는지를 묻고, section 3.3 은 이미 쓰고 있는 것을 이기는지를 묻는다. 앞의 물음은 model 이 값어치가
있는지를 정하고, 뒤의 물음은 그것이 교체할 값어치가 있는지를 정한다.

## 4. Meaning and Reporting

💡 분모를 고정하는 진짜 이유는 편의가 아니라 **표준 R² 로는 자료 사이를 비교할 수 없다는** 데 있다.
분모는 자료마다 다시 계산되므로 잣대가 자료를 따라 바뀐다.

한 model 을 세 lot 에 적용해 셋 모두에서 RMSE 가 0.5 nm 로 같은 경우를 보면 문제가 뚜렷하다.

**Table 2. The same model on three lots, RMSE fixed at 0.5 nm**

| Lot | Lot dispersion (nm) | Standard R² | `R2_frd` against a spec of 1.0 nm |
|---|---|---|---|
| A | 2.0 | 0.9375 | 0.7500 |
| B | 1.2 | 0.8264 | 0.7500 |
| C | 0.6 | 0.3056 | 0.7500 |

Model 의 예측 오차는 세 lot 에서 같은데 표준 R² 는 0.94 에서 0.31 까지 벌어진다. 바뀐 것은 분모이지
model 이 아니다. 산포가 큰 lot 은 baseline 이 이미 실패하고 있던 lot 이므로 같은 성능이 좋아 보이고,
산포가 작은 lot 은 baseline 이 이미 감당하던 lot 이므로 같은 성능이 나빠 보인다. 표준 R² 는 이것을
정직하게 보고하고 있을 뿐이며, 이 값으로 lot 을 줄 세우면 model 이 아니라 lot 의 산포를 줄 세우게
된다.

분모를 1.0 nm 로 고정하면 셋 모두 0.75 가 되고, 지표는 그때 model 의 성능만을 비춘다. 달리 말하면 두
지표는 다른 물음에 답한다.

- 표준 R² — 이 자료 안에서 model 이 평균보다 얼마나 나은가. 잣대가 자료마다 다르다.
- `R2_frd` — model 이 정해 둔 잣대보다 얼마나 나은가. 자료 사이를 비교할 수 있다.

어느 쪽이 옳은지는 물음이 정한다. 이 lot 이 얼마나 설명되었는가를 묻는다면 표준 R² 이고, model 을
line 전체에 배포할 수 있는가를 묻는다면 `R2_frd` 이다.

그러므로 값 하나만 건네는 것으로는 모자란다. 두 물음의 답이 같은 자리에 같은 꼴로 내려앉으므로 받는
이는 어느 쪽을 본 것인지 알 수 없다. 보고는 아래 규칙을 따른다.

- `R²` 자체가 아니라 변형에 맞는 기호를 쓴다. 같은 기호는 읽는 이에게 표준 R² 로 받아들이라고 권한다.
- 기호만으로는 충분히 실리지 않으므로, 분모의 기준을 값과 출처와 함께 적는다. 수만 적지 말고 `R2_frd = 0.75 (reference: spec tolerance sigma = 1.0 nm)` 로 적는다.
- 표준 R² 를 나란히 보고한다. 둘 사이의 간격이야말로 그 자료의 산포가 spec 에 견주어 어느 쪽으로 기울어 있는지를 읽는 이에게 말해 준다.
- `sigma_ref` 를 그렇게 고른 근거를 남긴다. 근거 없이 고른 분모는 지표 전체를 자의적으로 만든다.

`Ref_R2`, `R2_frd`, `R2_base` 는 이 문서가 정한 이름이며 문서 밖에서는 통하지 않는다. 밖으로 보낼
때는 기호 뒤에 한 줄짜리 정의를 붙이거나, section 3.1 처럼 이미 자리 잡은 이름이 있으면 그것을 쓴다.

## 5. Cautions

- **음수가 나온다.** Model 이 baseline 보다 나쁘면 `SS_res` 가 분모를 넘어 값이 0 아래로 내려간다. Spec 이 1.0 nm 일 때 RMSE 가 1.3 nm 이면 −0.69 가 된다. 이것은 오류가 아니라 baseline 을 쓰는 편이 낫다는 정상적인 신호이며, out-of-sample 평가에서 흔하다.
- **분모가 작으면 값이 폭발한다.** Spec 산포가 아주 작은 항목에 이 지표를 쓰면 작은 오차에서 큰 음수가 나온다. 그런 항목은 R² 계열의 무엇보다 오차의 절대 크기로 관리하는 편이 낫다.
- **위 한계는 여전히 1 이다.** 분모를 바꾸어도 `SS_res` 를 0 아래로 내릴 수는 없으므로 값이 1 을 넘지 않는다. 1 에 가까운 값은 model 이 완벽하다는 뜻이 아니라 정해 둔 baseline 이 압도되었다는 뜻이다.
- **분모는 사후에 고르지 않는다.** 결과를 본 뒤 유리한 `sigma_ref` 를 고르면 지표는 수사가 된다. 기준은 평가 전에 정하고 기록한다.

## 6. Summary

R² 의 분모는 baseline 의 오차이고, 표준 R² 는 그 baseline 을 자료의 평균으로 고정한 특수한 경우이다.
분모를 학습 평균으로, spec 산포로, 운영 중인 model 의 오차로 바꾸는 일은 모두 baseline 을 바꾸는 같은
조작이며, 지표의 뜻은 이 자료 안에서의 설명력에서 정해 둔 기준에 견준 성능으로 옮겨 간다. 이 문서는
baseline 을 명시적으로 고른 형태들을 `Ref_R2` 라 부르고 `R2_oos`, `R2_frd`, `R2_base` 로 나눈다. 이
교환으로 얻는 것은 자료를 따라 움직이지 않는 잣대여서 lot 과 기간이 비교 가능해지는 것이고, 잃는
것은 표준 R² 가 지니던 해석과 기준을 고르는 책임이다.

## References

<a id="ref-1"></a>[1] Kvålseth, T. O. (1985). [Cautionary Note about R²](https://doi.org/10.1080/00031305.1985.10479448). *The American Statistician*, 39(4), 279–285. 표준 R² 가 분모를 자료의 평균에 묶어 둔 데서 물려받는 해석의 한계를 밝히며, section 3 의 변형들이 필요한 이유가 된다.<br>
<a id="ref-2"></a>[2] Campbell, J. Y., & Thompson, S. B. (2008). [Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?](https://doi.org/10.1093/rfs/hhm055) *The Review of Financial Studies*, 21(4), 1509–1531. 분모를 학습 구간의 평균에 고정한 out-of-sample R² 를 평가 지표로 쓴다. Section 3.1 이 기술하는 형태를 그대로 쓴 것이다.<br>
<a id="ref-3"></a>[3] Murphy, A. H. (1988). [Skill Scores Based on the Mean Square Error and Their Relationships to the Correlation Coefficient](https://doi.org/10.1175/1520-0493%281988%29116%3C2417%3ASSBOTM%3E2.0.CO%3B2). *Monthly Weather Review*, 116(12), 2417–2424. Skill score 를 평균제곱오차를 기준 오차로 나눈 것으로 정식화하고, 그 기준이 과거 평균 같은 외부 값일 수 있음을 보인다. Section 2 가 기술하는 구조의 근거이다.<br>
<a id="ref-4"></a>[4] Automotive Industry Action Group (2010). [Measurement Systems Analysis Reference Manual](https://www.aiag.org/training-and-resources/manuals/details/MSA-4), 4th ed. ISBN 978-1-60534-211-5. 계측 오차를 spec tolerance 로 나눈 precision-to-tolerance ratio 를 계측 system 의 합격 기준으로 정의한다. Section 3.2 의 `RMSE / sigma_ref` 를 계측 분야가 쓰는 방식이다.<br>
<a id="ref-5"></a>[5] Hyndman, R. J., & Koehler, A. B. (2006). [Another Look at Measures of Forecast Accuracy](https://doi.org/10.1016/j.ijforecast.2006.03.001). *International Journal of Forecasting*, 22(4), 679–688. 예측 오차를 persistence baseline 의 오차로 나누면 서로 다른 계열이 한 척도에 놓인다는 것을 밝힌다. Section 3.3 이 다루는 표본별 baseline 의 표준적인 예이다.

---

## Appendix A. Terminology

- **baseline model**: 비교의 기준이 되는 model. R² 의 분모는 그 제곱오차합이다.
- **out-of-sample**: 학습에 쓰지 않은 자료에서의 평가.
- **persistence baseline**: 직전 단계의 값을 다음 단계의 예측으로 쓰는 시계열 baseline.
- **Referenced R² (`Ref_R2`)**: 분모의 baseline 을 바깥에서 정해 계산한 R² 를 이 문서에서 통칭하는 이름.
- **RMSE**: root mean squared error 이며 `sqrt(SS_res / N)` 로 계산한다. 오차를 y 와 같은 단위로 돌려놓으므로 spec 산포에 바로 견줄 수 있다.
- **skill score**: `1 − model error / baseline error` 꼴 지표들의 통칭. R² 는 baseline 이 평균인 경우이다.
- **spec tolerance**: 공정 spec 이 허용하는 산포. 고정 분모로 자주 쓰인다.
- **SS_res**: 잔차제곱합 `Σ(y_i − y_hat_i)²`.
- **SS_tot**: 총제곱합 `Σ(y_i − y_bar)²` 이며, 표준 R² 의 분모이다.

## Appendix B. Reference Implementation

Section 3 에서 정의한 세 지표는 각각 함수 하나로 옮겨진다. `r2_oos` 가 3.1 의 `R2_oos` 를, `r2_frd`
가 3.2 의 `R2_frd` 를, `r2_base` 가 3.3 의 `R2_base` 를 맡는다.

셋 모두 관측값 `y_true` 와 예측값 `y_pred` 를 받아 skill 값 하나를 돌려주며, 다른 곳은 분모가 어디서
오는가 한 군데뿐이다. `r2_oos` 는 학습 평균 `y_train_bar` 를, `r2_frd` 는 기준 산포 `sigma_ref` 를,
`r2_base` 는 baseline 예측 vector `y_base` 를 각각 인자로 받아 거기서 분모를 만든다. 셋 가운데
어느 것도 기준을 `y_true` 에서 끌어내지 않고 모두 바깥에서 받는데, 이것이 section 2 가 baseline 을
명시적으로 고르는 일이라 부른 것을 code 로 옮긴 것이다. 분자 `SS_res` 는 셋이 같으므로 `_ss_res` 로
빼내어, 모양 검사와 빈 배열 검사를 한곳에 둔다.

`r2_base` 를 먼저 정의하고 `r2_oos` 가 그것을 부르게 한 것은 section 3.3 의 마지막 문단을 code 로
옮긴 것이다. 반면 `r2_frd` 의 분모는 어떤 예측 vector 의 잔차도 아니어서 `r2_base` 를 통해 표현할 수
없으므로 따로 계산한다.

```python
# Python
import numpy as np


def _ss_res(y_true: np.ndarray = None, y_pred: np.ndarray = None) -> float:
    """Residual sum of squares, shared by the numerator and by baseline denominators."""
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shapes must match, got y_true {y_true.shape} and y_pred {y_pred.shape}.")
    if y_true.size == 0:
        raise ValueError("y_true is empty; a skill score needs at least one observation.")

    return float(np.sum((y_true - y_pred) ** 2))


def r2_base(y_true: np.ndarray = None, y_pred: np.ndarray = None, y_base: np.ndarray = None) -> float:
    """Section 3.3 — skill against a baseline that answers per sample.

    Args:
        y_true: observations, shape (n,).
        y_pred: predictions, shape (n,).
        y_base: baseline predictions, shape (n,), one value per sample.

    Returns:
        The skill relative to y_base. Negative when the baseline is the better predictor.
    """
    ss_base = _ss_res(y_true=y_true, y_pred=y_base)
    if ss_base == 0.0:
        raise ValueError("the baseline reproduces y_true exactly, so there is no error to improve on.")

    return 1.0 - _ss_res(y_true=y_true, y_pred=y_pred) / ss_base


def r2_oos(y_true: np.ndarray = None, y_pred: np.ndarray = None, y_train_bar: float = None) -> float:
    """Section 3.1 — skill against the mean that was known at training time.

    Args:
        y_true: observations, shape (n,).
        y_pred: predictions, shape (n,).
        y_train_bar: mean of the training targets. Never derived from y_true, which would
            put the test mean into the baseline and leak the evaluation set.

    Returns:
        The skill relative to always predicting y_train_bar.
    """
    if y_train_bar is None:
        raise ValueError("y_train_bar is required; taking the mean of y_true would leak the test set.")

    y_base = np.full(shape=y_true.shape, fill_value=float(y_train_bar))
    return r2_base(y_true=y_true, y_pred=y_pred, y_base=y_base)


def r2_frd(y_true: np.ndarray = None, y_pred: np.ndarray = None, sigma_ref: float = None) -> float:
    """Section 3.2 — skill against a fixed reference dispersion instead of the data spread.

    Args:
        y_true: observations, shape (n,).
        y_pred: predictions, shape (n,).
        sigma_ref: reference dispersion in the unit of y, strictly positive.

    Returns:
        The skill relative to a baseline whose error scale is sigma_ref.
        Equal to 1 - (RMSE / sigma_ref) ** 2.
    """
    if sigma_ref is None or sigma_ref <= 0.0:
        raise ValueError(f"sigma_ref must be a positive dispersion, got {sigma_ref}.")

    # The denominator is a stated error scale, not the residual of any prediction vector,
    # so it cannot be routed through r2_base.
    ss_ref = float(y_true.size) * sigma_ref ** 2
    return 1.0 - _ss_res(y_true=y_true, y_pred=y_pred) / ss_ref
```
