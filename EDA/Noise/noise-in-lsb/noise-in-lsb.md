# Noise in LSB
Rev. 0 | Created: 2026-08-22 | Updated: 2026-08-22 19:31 UTC

신호의 표준편차를 LSB 하나의 크기로 나눈 값은 ADC 와 DAC 를 다루는 분야에서 noise in LSB 또는 LSB rms noise 라고 부른다. 잡음을 volt 로 적으면 장치마다 입력 범위와 비트 수가 달라 서로 견줄 수 없지만, 이 값은 잡음을 그 장치의 분해능으로 잰 것이므로 장치가 달라도 같은 뜻으로 읽힌다. 이 문서는 이 값의 정의, 물리적 의미, 값에 따른 해석을 정리한다.

## 1. Definition

LSB 는 ADC 가 구분할 수 있는 가장 작은 신호 단위이며, 입력 범위를 코드 개수로 나눈 값이다. 입력 범위를 $V_{FS}$, 비트 수를 $N$ 이라 하면 잡음을 LSB 로 잰 값은 다음과 같다.

$$\mathrm{Noise}_{LSB} = \frac{\sigma_{signal}}{\mathrm{LSB}} = \frac{\sigma_{signal}}{V_{FS} / 2^N}$$

분자와 분모가 모두 신호의 단위를 가지므로 이 값은 단위가 없다.

## 2. Physical Meaning

이 값은 디지털 시스템이 신호의 변동을 몇 개의 비트 단위까지 감지하는지를 나타낸다.

### 2.1. Resolution

LSB 는 그 장치가 분해할 수 있는 최소 단위이므로, 이 값은 잡음을 분해능의 배수로 읽은 것이다. 값이 1.0 이면 신호의 $1\sigma$ 변동과 최하위 비트 하나가 같은 크기이다.

### 2.2. Bit Stability

값이 $\lt 0.5$ 이면 잡음이 분해능보다 작아 변환 결과의 최하위 비트가 안정적으로 유지된다.

값이 $\ge 1.0$ 이면 입력이 가만히 있어도 잡음 때문에 최하위 비트가 계속 뒤집힌다. 이때 흔들리는 비트는 신호가 아니라 잡음을 표시하고 있는 것이다.

### 2.3. Effective Resolution

잡음에 덮이지 않고 남은 비트 수는 다음과 같이 얻는다.

$$\mathrm{Effective\ resolution} = \log_2 \frac{V_{FS}}{\sigma_{signal}} = N - \log_2 \mathrm{Noise}_{LSB}$$

곧 이 값의 $\log_2$ 가 잡음에 잃은 비트 수이다. 잡음을 peak-to-peak 으로 잴 때는 대신 noise-free resolution 을 쓴다. Gaussian 잡음에서 $6.6\sigma$ 를 넘는 일이 0.1 % 보다 드물어 peak-to-peak 잡음을 $6.6\sigma$ 로 잡으므로, 두 값은 $\log_2 6.6 \approx 2.7$ 비트만큼 벌어진다 [1](#ref-1).

Table 1. Resolution metric by noise definition

| Metric | Definition | Signal condition |
|--------|------------|------------------|
| Effective resolution | $\log_2 (V_{FS} / \sigma)$ | DC |
| Noise-free resolution | $\log_2 (V_{FS} / 6.6\sigma)$ | DC |
| ENOB | $(\mathrm{SINAD}_{dB} - 1.76) / 6.02$ | AC |

세 지표를 섞어 쓰지 않아야 한다. Effective resolution 과 noise-free resolution 은 DC 에서 잰 잡음만 보므로 왜곡을 셈에 넣지 않는다. ENOB 은 SINAD 에서 얻으며 잡음과 왜곡을 함께 본다 [2](#ref-2). 그러므로 표준편차 하나만 가지고 얻을 수 있는 것은 effective resolution 이고, ENOB 은 그것만으로는 얻을 수 없다.

다만 IEEE Std 1057 은 ENOB 을 $\log_2 [V_{FS} / (\sigma \sqrt{12})]$ 로도 정의한다 [3](#ref-3). 이 식은 잡음이 양자화 잡음뿐일 때 ENOB 이 $N$ 이 되도록 맞춘 것이며, effective resolution 보다 $\log_2 \sqrt{12} \approx 1.79$ 비트 낮게 나온다.

## 3. Reference Values

Table 2. Interpretation by noise level

| Noise in LSB | Physical state |
|--------------|----------------|
| $\lt 0.29$ | 양자화 잡음 수준에 닿아 있다. 장치의 아날로그 잡음이 매우 작은 상태이다 |
| $\approx 1.0$ | 잡음과 분해능이 같다. 최하위 비트가 1 LSB 단위로 흔들린다 |
| $\approx 4.0$ | 잡음이 $2^2$ LSB 이므로 하위 2 비트는 잡음만 담고 있다 |

첫 행의 0.29 는 양자화 잡음의 표준편차이다. 양자화 오차가 $\pm 0.5$ LSB 사이에 고르게 놓이면 그 표준편차는 $1 / \sqrt{12} \approx 0.289$ LSB 가 되며, 이것이 아날로그 잡음이 전혀 없는 ADC 가 한 번의 변환에서 가질 수 있는 가장 낮은 값이다. 이 값에 가까울수록 장치가 이상적인 상태에 가깝다. 여러 번 변환해 평균하거나 oversampling 하면 이보다 낮은 값도 나오는데, 그것은 장치가 더 조용해진 것이 아니라 표본을 늘려 얻은 결과이다.

## References

<a id="ref-1"></a>[1] Analog Devices. [ADC Input Noise: The Good, the Bad, and the Ugly. Is No Noise Good Noise?](https://www.analog.com/en/resources/analog-dialogue/articles/adc-input-noise.html). Analog Dialogue 40-02, 2006.

<a id="ref-2"></a>[2] Analog Devices. [Understanding Noise, ENOB, and Effective Resolution in Analog-to-Digital Converters](https://www.analog.com/en/resources/technical-articles/noise-enob-and-effective-resoluition-in-analog-to-digital-converter-circuits--maxim-integrated.html).

<a id="ref-3"></a>[3] Institute of Electrical and Electronics Engineers. [IEEE Std 1057-2017, IEEE Standard for Digitizing Waveform Recorders](https://ieeexplore.ieee.org/document/8291741/). 2017.

---

## Appendix A. Terminology

- **ADC (Analog-to-Digital Converter)** 는 연속인 아날로그 신호를 정해진 비트 수의 디지털 코드로 바꾸는 장치이다.
- **DAC (Digital-to-Analog Converter)** 는 디지털 코드를 아날로그 신호로 되돌리는 장치이다.
- **DAQ (Data Acquisition)** 는 센서 신호를 받아 디지털로 바꾸어 기록하는 장치나 시스템을 가리킨다.
- **ENOB (Effective Number of Bits)** 는 잡음과 왜곡을 셈에 넣었을 때 실제로 쓸 수 있는 비트 수이다.
- **LSB (Least Significant Bit)** 는 디지털 코드의 최하위 비트이며, 그 비트 하나가 나타내는 신호 크기를 뜻하기도 한다. 이 문서에서는 뒤의 뜻으로 쓴다.
- **RMS (Root Mean Square)** 는 값을 제곱해 평균한 뒤 제곱근을 취한 값이다. 평균이 0 인 잡음에서는 표준편차와 같다.
- **SINAD (Signal-to-Noise and Distortion Ratio)** 는 신호의 power 를 잡음과 왜곡의 power 합으로 나눈 비이다.
