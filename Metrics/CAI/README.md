# CAI (Center Alignment Index)

1:1 line 위에서 두 변수의 데이터 중심이 일치하는 정도를 0~1 scale로 평가하는 지표다.

## 1. Definition

CAI = 1 / (1 + u²), u = (μ_x - μ_y) / √(σ_x · σ_y) 로 정의한다.
두 변수의 평균이 같으면 1이고, 편향이 커질수록 0에 가까워진다.
표준편차로 정규화하므로 측정 단위와 무관하며, Lin's CCC 분해에서 bias correction
factor의 위치 성분과 직접 관련된다.
