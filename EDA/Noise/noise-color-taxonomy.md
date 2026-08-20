# Noise Color Taxonomy
Rev. 1 | Created: 2026-08-20 | Updated: 2026-08-20 01:33 CDT

노이즈에 붙은 색 이름은 주파수 대역에 에너지가 어떻게 나뉘어 있는지를 가리킨다. 저주파에 에너지가 몰릴수록 붉은 계열로 부르고, 고주파에 몰릴수록 푸른 계열로 부른다. 이 문서는 네 가지 색을 파형, 기울기, 청각, 공간 패턴, 활용의 순서로 정리한다.

## 1. Waveform

네 색을 같은 길이와 같은 세로 눈금으로 그리면 차이가 바로 드러난다.

![Fig 1](noise-color-taxonomy_fig/fig1_waveform.png)

Fig 1. Waveform of each noise color

Red 는 값이 천천히 오르내려 선이 매끄럽고, Blue 는 이웃한 표본끼리 값이 자주 뒤집혀 선이 가장 촘촘하다. White 는 그 둘 사이에 있고, Pink 는 White 같은 잔 움직임 위에 Red 같은 느린 흔들림이 얹혀 있다. 네 신호는 모두 표준편차를 1 로 맞추었으므로, 다른 것은 세로 폭이 아니라 값이 변하는 빠르기이다.

## 2. Spectral Slope

색을 가르는 기준은 octave 마다 에너지가 몇 dB 변하는지이다. Octave 는 주파수가 두 배가 되는 구간이므로, 이 기울기 하나가 전 대역의 에너지 분포를 정한다.

Table 1. Spectral slope by noise color

| Color | Slope | Power spectral density | Energy distribution |
|-------|-------|------------------------|---------------------|
| White | 0 dB/octave | `1/f^0` | 모든 대역이 같은 에너지를 갖는다 |
| Pink | -3 dB/octave | `1/f` | 주파수에 반비례해 줄어든다 |
| Red | -6 dB/octave | `1/f^2` | 고주파가 급격히 줄어든다 |
| Blue | +3 dB/octave | `f` | 주파수에 비례해 늘어난다 |

Red 는 brown noise 라고도 부른다. 기울기가 -6, -3, 0, +3 으로 이어지므로 네 색은 하나의 축 위에 놓이고, White 의 0 이 나머지 셋을 재는 기준이 된다.

![Fig 2](noise-color-taxonomy_fig/fig2_psd.png)

Fig 2. Power spectral density of each noise color

가로와 세로가 모두 로그 눈금이므로 주파수의 거듭제곱은 직선으로 나타나고, 그 직선의 기울기가 Table 1 의 값이다. White 만 수평이고 Pink 와 Red 는 내려가며 Blue 는 올라간다. 1 절에서 Red 의 선이 매끄럽고 Blue 의 선이 촘촘했던 것이 여기서는 두 곡선이 서로 반대 방향으로 기우는 것으로 나타난다.

가장 낮은 몇 Hz 에서 Pink 와 Blue 의 곡선이 수평 쪽으로 눕는데, 이는 그 구간에서 평균할 표본이 적어 추정이 거칠어진 것이지 기울기가 달라진 것이 아니다.

## 3. Auditory Character

White 는 TV 의 무신호 화면에서 나는 찌직 소리에 가깝다. 대역마다 에너지가 같아도 사람의 귀는 높은 대역을 더 크게 듣기 때문에, 평탄한 신호인데도 고음이 강조되어 들린다.

Pink 는 빗소리나 바람 소리처럼 들린다. 사람이 느끼는 음높이가 주파수의 로그에 비례하므로, 주파수에 반비례해 줄어드는 이 기울기가 귀에는 대역마다 고르게 들린다. 네 색 중 가장 자연스럽고 편안하게 느껴지는 이유가 여기에 있다.

Red 는 묵직한 폭포 소리나 둥둥거리는 천둥 소리에 가깝다. 고주파가 가장 빨리 줄어들어 저음만 남으므로 소리가 안정적이다.

Blue 는 쌕 하는 고음역 위주의 날카로운 소리이다. Red 와 정반대로 에너지가 고주파에 몰려 있다.

## 4. Spatial Pattern

같은 기울기를 소리가 아니라 점의 배치에 적용하면 색마다 다른 무늬가 나온다.

White 는 점이 완전히 무작위로 놓이므로 뭉치는 곳과 비어 있는 곳이 함께 생긴다. 무작위가 곧 고른 배치는 아니라는 것이 여기서 드러난다.

Pink 는 저주파 성분이 살아 있어 뭉침이 남고, Red 는 저주파가 더 강해 뭉침이 넓고 굵어진다.

Blue 는 저주파 성분이 없으므로 뭉침이 생기지 않는다. 점들이 서로 밀어내는 것처럼 고르게 흩어진다.

## 5. Application

Table 2. Application by noise color

| Color | Field | Purpose |
|-------|-------|---------|
| White | 오디오 장비 측정, 집중력 향상, 소음 차단 | 평탄한 기준 신호로 응답을 재거나 주변 소리를 덮는다 |
| Pink | 음향 기기 calibration, 수면 유도, 음향 치료 | 귀에 고르게 들리는 성질을 기준으로 삼는다 |
| Red | 깊은 수면 유도, tinnitus 완화, 중저음 강조 시험 | 저음이 강하고 안정적인 소리를 만든다 |
| Blue | Computer graphics sampling, halftoning, dithering | 점이 뭉치지 않는 배치를 얻는다 |

앞의 셋은 소리로 쓰이고 Blue 만 점의 배치로 쓰인다. Blue 를 쓰는 세 분야가 모두 표본이나 점을 고르게 흩어 놓아야 하는 일이기 때문이다.

---

## Appendix A. Terminology

- **Dithering** 은 색이나 밝기의 단계를 줄일 때 생기는 띠를 없애려고 의도적으로 노이즈를 더하는 기법이다.
- **Halftoning** 은 농담을 점의 밀도로 바꾸어 두 색만으로 중간 밝기를 표현하는 기법이다.
- **Power spectral density** 는 주파수마다 신호가 갖는 에너지의 밀도이고, 기울기는 이 값이 주파수에 따라 변하는 정도를 말한다.
- **Tinnitus** 는 외부에 소리가 없는데도 귀에서 소리가 들리는 증상이다.
