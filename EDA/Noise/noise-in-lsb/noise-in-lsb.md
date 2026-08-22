# Noise in LSB
Rev. 3 | Created: 2026-08-22 | Updated: 2026-08-22 20:39 UTC

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

다만 IEEE Std 1057 은 ENOB 을 $\log_2 [V_{FS} / (\sigma \sqrt{12})]$ 로도 정의한다 [3](#ref-3). 이 식은 잡음이 quantization noise 뿐일 때 ENOB 이 $N$ 이 되도록 맞춘 것이며, effective resolution 보다 $\log_2 \sqrt{12} \approx 1.79$ 비트 낮게 나온다.

## 3. Reference Values

Table 2. Interpretation by noise level

| Noise in LSB | Physical state |
|--------------|----------------|
| $\lt 0.29$ | Quantization noise 가 전체 잡음을 정한다. Input-referred noise 는 그 아래에 묻혀 있다 |
| $\approx 1.0$ | 잡음과 분해능이 같다. 최하위 비트가 1 LSB 단위로 흔들린다 |
| $\approx 4.0$ | 잡음이 $2^2$ LSB 이므로 하위 2 비트는 잡음만 담고 있다 |

첫 행의 0.29 는 quantization noise 의 표준편차이다. Quantization error 가 $\pm 0.5$ LSB 사이에 고르게 놓이면 그 표준편차는 $1 / \sqrt{12} \approx 0.289$ LSB 가 되며, 이것이 input-referred noise 가 전혀 없는 ADC 가 한 번의 변환에서 가질 수 있는 가장 낮은 값이다.

잰 값이 이 값에 붙어 있다는 것이 무엇을 뜻하는지는 잡음이 더해지는 방식에서 나온다. Quantization noise 와 input-referred noise 는 서로 독립이므로 제곱해서 더해진다.

$$\sigma_{total} = \sqrt{\sigma_{quant}^2 + \sigma_{input}^2}$$

여기서 $\sigma_{quant}$ 는 quantization noise 이고 $\sigma_{input}$ 은 input-referred noise 이다. $\sigma_{quant}$ 가 0.289 LSB 로 고정되어 있으므로, $\sigma_{total}$ 이 0.29 근처에 머물려면 $\sigma_{input}$ 이 그보다 훨씬 작아야 한다. $\sigma_{input}$ 이 0.1 LSB 여도 $\sigma_{total}$ 은 0.306 LSB 여서 quantization noise 만 있을 때와 거의 구분되지 않는다. 곧 센서, 기준 전압, 증폭기, 배선에서 들어온 잡음을 모두 합쳐도 LSB 한 칸의 몇 분의 일에 그쳐, 출력에서 보이는 흔들림이 ADC 가 값을 자르며 생긴 것뿐인 상태이다.

이 상태에서는 입력이 조용하면 출력 코드가 하나에 머문다. 대신 여러 번 변환해 평균해도 분해능이 더 좋아지지 않는데, 평균이 효과를 내려면 값을 이웃 코드로 넘겨 줄 잡음이 있어야 하기 때문이다. 잡음이 1 LSB 를 넘는 쪽에서는 반대로 $M$ 번 평균해 잡음을 $\sqrt{M}$ 만큼 줄일 수 있고, 이때 얻은 0.29 보다 낮은 값은 장치가 더 조용해진 것이 아니라 표본을 늘려 얻은 결과이다.

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
- **FDC (Fault Detection and Classification)** 는 장비가 남긴 sensor 기록의 통계를 감시해 이상을 찾아내고 그 종류를 가르는 체계이다.
- **Input-referred noise** 는 ADC 앞단이 얹은 잡음을 ADC 입력 자리에서 본 값이다. 입력을 정지시킨 채 출력 코드의 분포를 재어 얻으며, code transition noise 라고도 부른다 [1](#ref-1).
- **LSB (Least Significant Bit)** 는 디지털 코드의 최하위 비트이며, 그 비트 하나가 나타내는 신호 크기를 뜻하기도 한다. 이 문서에서는 뒤의 뜻으로 쓴다.
- **Quantization noise** 는 연속인 값을 정해진 코드로 자를 때 남는 오차이며, 그 크기는 LSB 하나가 정한다. 자르기 전의 값과 자른 뒤의 값의 차이를 quantization error 라고 한다.
- **RMS (Root Mean Square)** 는 값을 제곱해 평균한 뒤 제곱근을 취한 값이다. 평균이 0 인 잡음에서는 표준편차와 같다.
- **SINAD (Signal-to-Noise and Distortion Ratio)** 는 신호의 power 를 잡음과 왜곡의 power 합으로 나눈 비이다.
- **Trace** 는 한 공정 단계가 도는 동안 시간에 따라 기록한 sensor 값의 열이다.

## Appendix B. Application to Semiconductor Equipment Signals

반도체 제조장비는 chamber 압력, gas flow, RF power, 온도 같은 sensor 값을 DAQ 를 거쳐 trace 로 남기고, FDC 는 그 trace 의 표준편차나 폭 같은 통계로 이상을 가른다. 그런데 trace 의 표준편차에는 공정 자체의 변동과 계측 사슬이 얹은 잡음이 함께 들어 있다. 둘을 가르지 않으면 FDC 가 공정이 아니라 계측 잡음을 쫓게 되므로, 그 trace 가 장비의 분해능에 견주어 어디에 놓이는지를 먼저 알아야 한다. 이 문서의 값이 그 자리를 가리킨다.

Table 3. Reading of the value on an equipment signal

| Noise in LSB | Diagnosis | Action |
|--------------|-----------|--------|
| $\lt 0.5$ | 분해능이 잡음보다 거칠다. Trace 에 보이는 계단은 공정이 아니라 값을 자른 자국이다 | 더 작은 변동을 봐야 한다면 입력 범위를 줄이거나 비트 수가 큰 ADC 로 바꾼다 |
| $0.5$ ~ $10$ | 잡음과 분해능이 맞물려 있다. 평균이 효과를 내는 구간이다 | 필요한 만큼 평균해 잡음을 줄인다 |
| $\gt 10$ | Input-referred noise 가 지배한다. 적어도 하위 서너 비트가 잡음만 담고 있다 | 접지, 차폐, sensor 자체를 본다 |

이 구획은 규격이 아니라 조치를 가르는 실무 눈금이다. 세 줄의 쓸모는 마지막 열에 있다. 값이 작으면 고칠 곳이 ADC 뒤에 있고, 값이 크면 ADC 앞에 있다. 값이 큰 신호에 비트 수가 큰 ADC 를 붙이는 것은 잡음을 더 곱게 적을 뿐 줄이지 못하므로, 이 수 하나가 디지털을 고칠 일인지 아날로그를 고칠 일인지를 먼저 가른다.

압력 신호를 예로 들면 셈은 다음과 같다.

- 입력 범위가 10 Torr 이고 16 bit 로 받으면 1 LSB 는 $10 / 2^{16}$ Torr, 곧 약 0.153 mTorr 이다.
- 입력이 정지한 구간에서 잰 표준편차가 0.5 mTorr 이면 이 값은 $0.5 / 0.153 \approx 3.3$ 이다.
- 잃은 비트는 $\log_2 3.3 \approx 1.7$ 이므로 effective resolution 은 약 14.3 bit 이다.
- 감지하려는 공정 변동이 2 mTorr 이면 표준편차의 4 배이므로 한 점으로도 구분된다.
- 감지하려는 변동이 0.2 mTorr 이면 표준편차보다 작아 한 점으로는 보이지 않는다. 이를 $3\sigma$ 로 세우려면 잡음을 $0.2 / 3$ mTorr 까지 낮춰야 하므로 $(0.5 \times 3 / 0.2)^2$ 에서 57 점을 평균해야 한다.

마지막 줄이 이 값의 물리적 의미를 그대로 보여 준다. 감지하려는 변동과 계측 잡음의 비가 몇 점을 평균해야 하는지를 정하고, 그 점의 수가 다시 trace 의 시간 분해능을 정한다. 곧 이 값은 장비가 얼마나 작은 공정 변동을 얼마나 빠르게 볼 수 있는지의 상한이다.

쓸 때 두 가지를 지켜야 한다. 첫째, 표준편차는 입력이 정지한 구간에서 재야 한다. 압력이 오르내리는 구간에서 재면 공정 변동이 섞여 들어가 계측 잡음을 실제보다 크게 읽는다. 둘째, 이 값은 단위가 없어 입력 범위가 다른 장비끼리도 견줄 수 있으나, 견주는 대상은 각 장비의 분해능 대비 계측 사슬의 상태이다. 공정 변동의 절대 크기를 견주려면 물리 단위로 되돌려야 한다.
