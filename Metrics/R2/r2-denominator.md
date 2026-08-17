# Referenced R² — Choosing the Baseline in the R² Denominator
Rev. 9 | Created: 2026-08-15 | Updated: 2026-08-17 00:20 CDT

> 표준 R² 의 분모를 데이터에서 계산하지 않고 지정한 기준으로 바꿀 수 있는지,
> 바꾸면 그 값이 물리적으로 무엇을 뜻하게 되는지 정리한 문서.

## 1. Question

표준 R² 는 아래와 같이 정의된다.

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

분모 `SS_tot` 는 데이터 자체의 평균 `y_bar` 를 기준으로 한 총제곱합이며, 이것이
"데이터의 전체 산포" 로 불리는 양이다. 이 분모를 데이터에서 계산하지 않고 밖에서
지정한 값으로 바꿀 수 있는가 하는 것이 이 문서의 질문이다.

답은 가능하다는 것이다. 원리상 자연스러울 뿐 아니라 이미 확립된 형태가 여럿 있다.
다만 분모를 바꾸는 순간 지표의 의미가 함께 바뀌므로, 무엇으로 바꿨는지를 반드시
같이 보고해야 한다.

이렇게 분모를 지정해서 만든 지표를 통칭해 이 문서에서는 **Referenced R²** 라 부르고
`Ref_R2` 로 표기한다. 개별 변형은 분모에 무엇을 넣었는지를 따라 3 에서 `R2_oos`
(out-of-sample), `R2_frd` (fixed reference dispersion), `R2_base` (baseline model) 로
나눈다.

## 2. Structure

분모를 왜 바꿔도 되는지는 R² 를 다시 읽으면 드러난다. `SS_tot` 는 "항상 `y_bar` 를
예측하는 모델" 의 오차제곱합과 정확히 같다. 즉 R² 는 처음부터 아래 형태였다.

```text
                error left by the model
R2 = 1 -  ───────────────────────────────────
            error left by a chosen baseline
```

표준 R² 는 baseline 을 "평균만 내놓는 모델" 로 고정한 특수한 경우이고, 이 구조는
skill score 라는 이름으로 널리 쓰인다. **분모를 지정한다는 것은 baseline 을 바꾸는
것이며, 그 이상도 이하도 아니다.** 이것이 이 문서의 나머지 내용을 지배하는 사실이다.

따라서 `Ref_R2` 는 새로운 계산이 아니라 baseline 을 명시적으로 고른 R² 이고, 표준 R² 는
그 baseline 을 데이터의 평균으로 암묵적으로 정해 둔 사례다.

분자와 분모가 모두 y 의 제곱 단위이므로 비율은 무차원이고, `Ref_R2` 는 "baseline 이
감수해야 했던 오차 중 몇 %를 모델이 없앴는가" 로 읽힌다. Baseline 을 무엇으로 잡든 이
해석은 유지된다.

## 3. Variants

**Table 1. Ways to choose the baseline**

| Symbol | Denominator | Baseline it encodes | Typical use | Reference |
|---|---|---|---|---|
| `R²` | `Σ(y_i − y_bar)²` | 이 데이터셋의 평균 | 단일 데이터셋 안에서의 적합도 | [1](#ref-1) |
| `R2_oos` | `Σ(y_i − y_train_bar)²` | 학습 때 알던 평균 | Test set 평가 | [2](#ref-2) |
| `R2_frd` | `N · sigma_ref²` | 규격이 허용하는 산포 | Lot, batch, 기간 간 비교 | [3](#ref-3), [4](#ref-4) |
| `R2_base` | `Σ(y_i − y_base_i)²` | Sample 마다 값이 다른 기준 모델 | 시계열, 기존 운영 모델 대비 | [5](#ref-5) |

Reference 열은 각 형태가 실제로 쓰이고 있는 자리를 가리키며, 서지 사항은
[References](#references) 에 있다. 네 형태 모두 이 문서가 지어낸 계산이 아니라
각 분야에서 이미 표준으로 자리 잡은 지표다. 이 문서가 붙인 것은 계산이 아니라 기호뿐이고,
그 사정은 4 에서 다시 다룬다.

### 3.1. Fixed Reference Point

가장 흔한 경우다. Test set 을 평가할 때 분모의 기준을 test 데이터의 평균이 아니라
학습 데이터의 평균 `y_train_bar` 로 고정한다.

$$R^2_{oos} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y}_{train})^2}$$

두 가지 이유로 이 형태가 옳다. 첫째, test 데이터의 평균은 평가 시점에 알 수 없는
값이므로 그것을 baseline 에 쓰면 미래 정보를 끌어다 쓰는 셈이 된다. 둘째, test set 을
어떻게 자르느냐에 따라 분모가 달라져 잣대 자체가 흔들린다. 학습 평균으로 고정하면
"훈련 때 알던 평균만 계속 내놓는 모델 대비 얼마나 개선했는가" 라는 질문이 되어,
데이터를 어떻게 잘랐든 같은 잣대가 유지된다.

이 형태는 밖에서 out-of-sample R² 로 통하므로, 외부 보고에서는 그 이름을 쓴다.

### 3.2. Fixed Reference Dispersion

분모를 데이터에서 계산하지 않고 알려진 참조 분산으로 대체한다.

$$R^2_{frd} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{N \cdot \sigma_{ref}^2}$$

`sigma_ref` 로는 공정 규격의 허용 산포, 과거 누적 데이터의 분산, 계측 시스템의 기준
분산 같은 도메인 기준을 쓴다. 두께 예측이라면 해당 layer 의 관리 규격 산포를 분모에
넣어, 고정된 잣대 대비 모델의 성능을 재는 지표가 된다.

분모가 상수이므로 식이 한 단계 더 줄어든다. 분자의 `SS_res` 를 표본 수 `N` 으로 나눈 뒤
제곱근을 취한 값이 root mean squared error, 곧 RMSE 다.

$$\mathrm{RMSE} = \sqrt{\frac{SS_{res}}{N}}$$

RMSE 는 오차를 y 와 같은 단위로 되돌린 값이므로 `sigma_ref` 와 직접 나눌 수 있다.
`SS_res` 를 `N · RMSE²` 으로 바꿔 넣으면 `N` 이 약분되어 아래만 남는다.

$$R^2_{frd} = 1 - \left(\frac{\mathrm{RMSE}}{\sigma_{ref}}\right)^2$$

이것이 이 변형의 물리적 의미다. `R2_frd` 는 **모델의 오차가 규격 산포의 몇 배인지**를
재서 1 에서 뺀 값이다. 오차가 규격의 절반이면 0.75, 규격과 같으면 0, 규격을 넘으면
음수가 된다. Baseline 의 말로 옮기면 "규격이 허용하는 만큼의 오차를 정확히 내는 가상의
모델" 을 기준으로 삼은 것이며, 그 가상의 모델을 이겼는지를 묻는 것이다.

계측 분야에는 오차와 규격의 비를 R² 로 포장하지 않고 그대로 보고하는 관행이 있다.
읽는 쪽이 공정·계측 담당이면 `RMSE / sigma_ref` 를 그대로 주는 편이 잘 통하고, 다른
지표와 한 표에 나란히 놓아야 하면 `R2_frd` 로 바꿔 0 과 1 의 잣대에 태우는 편이 낫다.
두 값은 위 식으로 서로 옮겨갈 수 있으므로 정보량은 같다.

### 3.3. Baseline Model

3.1 의 baseline 은 어떤 sample 을 만나든 `y_train_bar` 라는 숫자 하나를 답으로 내놓고,
3.2 의 baseline 은 어떤 sample 을 만나든 `sigma_ref` 만큼의 오차를 낸다. 둘 다 기준이
sample 에 따라 달라지지 않는 상수 baseline 이다.

기준 자체가 sample 마다 달라져야 하는 경우는 이 형태로 적을 수 없다. 이때는 baseline 이
i 마다 자기 값 `y_base_i` 를 갖는다.

$$R^2_{base} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - y_{base,i})^2}$$

시계열의 persistence baseline 이 대표적이다. 직전 시점의 값을 다음 시점의 예측으로
쓰므로 `y_base_i` 가 i 마다 달라지고, 어떤 상수로도 같은 분모를 만들 수 없다. 계절
평균을 baseline 으로 둘 때, 현재 운영 중인 모델을 baseline 으로 두어 교체 가치를 잴
때도 마찬가지다.

`y_base_i` 를 상수 `y_train_bar` 로 두면 3.1 이 되므로 3.1 은 이 형태의 특수한 경우다.
그래도 절을 나눠 두는 이유는 두 형태가 답하는 질문이 다르기 때문이다. 3.1 은 "평균보다
나은가" 를 묻고 3.3 은 "이미 쓰고 있는 것보다 나은가" 를 묻는다. 앞의 질문은 모델이
쓸모가 있는지를, 뒤의 질문은 모델을 교체할 값어치가 있는지를 결정한다.

## 4. Meaning and Reporting

💡 분모를 고정하는 진짜 이유는 편의가 아니라 **표준 R² 가 데이터셋 간 비교에 쓸 수 없는
지표**라는 데 있다. 분모가 데이터마다 다시 계산되므로 잣대가 데이터마다 바뀐다.

같은 모델을 세 lot 에 적용해 RMSE 가 세 곳 모두 0.5 nm 로 동일하게 나온 경우를 보면
문제가 분명해진다.

**Table 2. The same model on three lots, RMSE fixed at 0.5 nm**

| Lot | Lot dispersion (nm) | Standard R² | `R2_frd` against a spec of 1.0 nm |
|---|---|---|---|
| A | 2.0 | 0.9375 | 0.7500 |
| B | 1.2 | 0.8264 | 0.7500 |
| C | 0.6 | 0.3056 | 0.7500 |

모델의 예측 오차는 세 lot 에서 완전히 같은데 표준 R² 는 0.94 에서 0.31 까지 벌어진다.
변한 것은 모델이 아니라 분모다. 산포가 큰 lot 은 baseline 이 원래 못 맞히던 lot 이라
같은 성능이 좋아 보이고, 산포가 작은 lot 은 baseline 도 이미 잘 맞히던 lot 이라 같은
성능이 나빠 보인다. 표준 R² 는 이 사실을 정직하게 반영하고 있을 뿐이며, 이 값으로
lot 을 줄 세우면 모델이 아니라 lot 의 산포를 줄 세우게 된다.

분모를 1.0 nm 로 고정하면 세 lot 모두 0.75 가 되어, 지표가 모델의 성능만 반영한다.
바꿔 말하면 두 지표는 서로 다른 질문에 답한다.

- 표준 R² — 이 데이터셋 안에서 모델이 평균 대비 얼마나 나은가. 데이터셋마다 잣대가 다르다.
- `R2_frd` — 정해진 잣대 대비 모델이 얼마나 나은가. 데이터셋을 가로질러 비교할 수 있다.

어느 쪽이 옳은지는 질문에 달렸다. "이 lot 을 얼마나 설명했는가" 를 묻는다면 표준 R² 가
맞고, "이 모델을 라인 전체에 깔아도 되는가" 를 묻는다면 `R2_frd` 가 맞다.

그래서 값 하나만 건네서는 안 된다. 두 질문의 답이 같은 자리에 같은 모양으로 찍히므로,
받는 쪽은 무엇을 본 것인지 알 수 없다. 보고할 때는 아래를 지킨다.

- 변형에 맞는 기호를 쓰고 `R²` 를 그대로 쓰지 않는다. 같은 기호를 쓰면 읽는 쪽이 표준 R² 로 오해한다.
- 기호만으로는 부족하므로 분모의 기준을 값과 출처까지 함께 적는다. `R2_frd = 0.75 (reference: spec tolerance sigma = 1.0 nm)` 처럼 쓰고, 숫자만 적지 않는다.
- 표준 R² 를 함께 보고한다. 두 값의 차이가 곧 그 데이터셋의 산포가 규격 대비 어느 쪽으로 치우쳤는지를 알려준다.
- `sigma_ref` 를 고른 근거를 남긴다. 근거 없이 고른 분모는 지표 전체를 자의적으로 만든다.

`Ref_R2` 와 `R2_frd`, `R2_base` 는 이 문서에서 정한 이름이므로 문서 밖에서는 통하지
않는다. 외부에 낼 때는 기호 뒤에 정의식을 한 줄 붙이거나, 3.1 처럼 통용되는 이름이 있으면
그쪽을 쓴다.

## 5. Cautions

- **음수가 나온다.** 모델이 baseline 보다 못하면 `SS_res` 가 분모를 넘어 값이 0 아래로 간다. 규격 1.0 nm 에 대해 RMSE 가 1.3 nm 이면 −0.69 다. 이는 오류가 아니라 "이 모델을 쓰느니 baseline 을 쓰는 편이 낫다" 는 정상적인 신호이며, out-of-sample 평가에서는 흔히 나온다.
- **분모가 작으면 값이 폭발한다.** 규격 산포가 매우 작은 항목에 이 지표를 쓰면 작은 오차에도 큰 음수가 나온다. 그런 항목은 R² 계열 대신 오차의 절대 크기로 관리하는 편이 낫다.
- **상한은 여전히 1 이다.** 분모를 바꿔도 `SS_res` 는 0 아래로 갈 수 없으므로 값이 1 을 넘지 않는다. 1 에 가까운 값은 모델이 완벽하다는 뜻이 아니라 지정한 baseline 을 압도했다는 뜻이다.
- **분모를 사후에 고르지 않는다.** 결과를 본 뒤 보기 좋은 `sigma_ref` 를 고르면 지표가 아니라 수사가 된다. 기준은 평가 전에 정하고 문서에 남긴다.

## 6. Summary

R² 의 분모는 baseline 의 오차이고, 표준 R² 는 그 baseline 을 데이터의 평균으로 고정한
특수한 경우다. 분모를 학습 평균, 규격 산포, 기존 운영 모델의 오차로 바꾸는 것은 모두
baseline 을 바꾸는 같은 조작이며, 그 결과 지표는 "이 데이터셋 안에서의 설명력" 에서
"지정한 기준 대비 상대 성능" 으로 의미가 옮겨간다. 이렇게 baseline 을 명시적으로 고른
형태를 이 문서는 `Ref_R2` 로 통칭하고 `R2_oos`, `R2_frd`, `R2_base` 로 나눈다. 이 교체의
실익은 데이터셋마다 달라지던 잣대가 고정되어 lot 과 기간을 가로질러 비교할 수 있게 되는
데 있고, 대가는 표준 R² 로서의 해석을 잃는 것과 기준 선정의 책임을 지는 것이다.

## References

<a id="ref-1"></a>[1] Kvålseth, T. O. (1985). [Cautionary Note about R²](https://doi.org/10.1080/00031305.1985.10479448). *The American Statistician*, 39(4), 279–285. 분모가 데이터의 평균에 묶여 있어서 생기는 표준 R² 의 해석상 한계를 정리한 글이며, 3 의 변형들이 필요한 이유를 준다.

<a id="ref-2"></a>[2] Campbell, J. Y., & Thompson, S. B. (2008). [Predicting Excess Stock Returns Out of Sample: Can Anything Beat the Historical Average?](https://doi.org/10.1093/rfs/hhm055) *The Review of Financial Studies*, 21(4), 1509–1531. 분모를 학습 구간의 평균으로 고정한 out-of-sample R² 를 평가 지표로 쓴다. 3.1 이 말하는 형태가 그대로 쓰인 사례다.

<a id="ref-3"></a>[3] Murphy, A. H. (1988). [Skill Scores Based on the Mean Square Error and Their Relationships to the Correlation Coefficient](https://doi.org/10.1175/1520-0493%281988%29116%3C2417%3ASSBOTM%3E2.0.CO%3B2). *Monthly Weather Review*, 116(12), 2417–2424. 평균제곱오차를 기준 오차로 나눈 skill score 를 정식화하고, 그 기준을 과거 평균 같은 외부 값으로 둘 수 있음을 보인다. 2 가 말하는 구조의 근거다.

<a id="ref-4"></a>[4] Automotive Industry Action Group (2010). [Measurement Systems Analysis Reference Manual](https://www.aiag.org/training-and-resources/manuals/details/MSA-4), 4th ed. ISBN 978-1-60534-211-5. 계측 오차를 규격 공차로 나눈 precision-to-tolerance ratio 를 계측 시스템의 합부 판정 기준으로 규정한다. 3.2 의 `RMSE / sigma_ref` 가 계측 분야에서 쓰이는 형태다.

<a id="ref-5"></a>[5] Hyndman, R. J., & Koehler, A. B. (2006). [Another Look at Measures of Forecast Accuracy](https://doi.org/10.1016/j.ijforecast.2006.03.001). *International Journal of Forecasting*, 22(4), 679–688. 예측 오차를 persistence baseline 의 오차로 나눠 서로 다른 계열을 같은 잣대에 태우는 방법을 정리한다. 3.3 이 다루는 sample 마다 달라지는 baseline 의 표준적인 예다.

---

## Appendix A. Terminology

- **baseline model** — 비교의 기준이 되는 모델이며, R² 의 분모는 이 모델의 오차제곱합이다.
- **out-of-sample** — 학습에 쓰이지 않은 데이터에서의 평가를 뜻한다.
- **persistence baseline** — 직전 시점의 값을 그대로 다음 시점의 예측으로 쓰는 시계열 baseline 이다.
- **Referenced R² (`Ref_R2`)** — 분모의 baseline 을 밖에서 지정해 계산한 R² 를 통칭하며, 이 문서에서 정한 이름이다.
- **RMSE** — root mean squared error 이며 `sqrt(SS_res / N)` 로 계산한다. 오차를 y 와 같은 단위로 되돌린 값이라 규격 산포와 직접 견줄 수 있다.
- **skill score** — `1 − 모델오차 / baseline오차` 형태의 지표를 통칭하며, R² 는 baseline 을 평균으로 둔 사례다.
- **spec tolerance** — 공정 규격이 허용하는 산포이며, 고정 분모로 자주 쓰인다.
- **SS_res** — 잔차제곱합 `Σ(y_i − y_hat_i)²` 이다.
- **SS_tot** — 총제곱합 `Σ(y_i − y_bar)²` 이며, 표준 R² 의 분모다.

## Appendix B. Reference Implementation

3 에서 정의한 세 지표를 함수 하나씩으로 옮긴 것이다. 3.1 의 `R2_oos` 는 `r2_oos` 가,
3.2 의 `R2_frd` 는 `r2_frd` 가, 3.3 의 `R2_base` 는 `r2_base` 가 맡는다.

세 함수는 모두 관측값 `y_true` 와 예측값 `y_pred` 를 받아 skill 값 하나를 돌려주며,
서로 다른 곳은 분모를 어디서 얻느냐 한 군데뿐이다. `r2_oos` 는 학습 평균 `y_train_bar`
를, `r2_frd` 는 참조 산포 `sigma_ref` 를, `r2_base` 는 baseline 예측 vector `y_base` 를
인자로 받아 그것으로 분모를 만든다. 셋 다 분모의 기준을 밖에서 받을 뿐 `y_true` 에서
끌어내지 않으며, 이것이 2 에서 말한 "baseline 을 명시적으로 고른다" 를 코드로 옮긴
모습이다. 분자인 `SS_res` 는 셋이 똑같이 쓰므로 `_ss_res` 로 따로 빼서 shape 검사와
빈 배열 검사를 한곳에서 하게 했다.

`r2_base` 를 먼저 두고 `r2_oos` 가 그것을 호출하는 것은 3.3 의 마지막 문단을 코드로
옮긴 것이다. 반면 `r2_frd` 의 분모는 어떤 예측 vector 의 잔차도 아니어서 `r2_base` 로
표현되지 않으므로 따로 계산한다.

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
