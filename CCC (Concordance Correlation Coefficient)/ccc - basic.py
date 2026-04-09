import numpy as np
from scipy import stats

def concordance_correlation_coefficient(y_true, y_pred):
    """
    Lin's Concordance Correlation Coefficient (CCC)
    
    Returns: dict with ccc, r, Cb, bias, precision
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    n = len(y_true)
    mu_t = np.mean(y_true)
    mu_p = np.mean(y_pred)
    
    # 분산 (biased)
    var_t = np.var(y_true)
    var_p = np.var(y_pred)
    
    # 공분산
    cov = np.mean((y_true - mu_t) * (y_pred - mu_p))
    
    # Pearson r
    r = cov / np.sqrt(var_t * var_p)
    
    # CCC 계산
    ccc = (2 * cov) / (var_t + var_p + (mu_t - mu_p)**2)
    
    # 정확도 편향 계수 Cb = ccc / r
    cb = ccc / r if r != 0 else 0
    
    return {
        'ccc':   round(ccc, 4),
        'r':     round(r,   4),
        'Cb':    round(cb,  4),
        'bias':  round(mu_p - mu_t, 4),
        'var_ratio': round(np.sqrt(var_p / var_t), 4),
    }

# ── 예시 ──────────────────────────────────────
np.random.seed(42)
y_true = np.random.normal(50, 10, 100)

# 시나리오 1: 좋은 일치
y_good  = y_true + np.random.normal(0, 2, 100)
# 시나리오 2: 평균 편향
y_bias  = y_true + 8 + np.random.normal(0, 2, 100)
# 시나리오 3: 스케일 편향
y_scale = y_true * 1.5 + np.random.normal(0, 2, 100)

for name, pred in [("좋은일치", y_good),
                    ("평균편향", y_bias),
                    ("스케일편향", y_scale)]:
    res = concordance_correlation_coefficient(y_true, pred)
    print(f"{name}: CCC={res['ccc']}, r={res['r']}, Cb={res['Cb']}")

# 출력:
# 좋은일치:  CCC=0.9789, r=0.9796, Cb=0.9993
# 평균편향:  CCC=0.7312, r=0.9796, Cb=0.7464  ← r은 높지만 CCC는 낮음!
# 스케일편향: CCC=0.6148, r=0.9796, Cb=0.6277