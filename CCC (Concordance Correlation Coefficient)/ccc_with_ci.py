import numpy as np

def ccc_with_ci(y_true, y_pred, alpha=0.05):
    """
    CCC + 부트스트랩 신뢰구간
    Fisher's Z-transform 방식도 포함
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_true)
    
    mu_t, mu_p = np.mean(y_true), np.mean(y_pred)
    var_t = np.var(y_true)
    var_p = np.var(y_pred)
    cov   = np.mean((y_true - mu_t) * (y_pred - mu_p))
    
    ccc = 2 * cov / (var_t + var_p + (mu_t - mu_p)**2)
    
    # Fisher's Z-transform 신뢰구간
    z   = np.arctanh(ccc)          # atanh 변환
    se  = 1 / np.sqrt(n - 3)
    z_alpha = 1.96                  # 95% CI
    
    ci_lower = np.tanh(z - z_alpha * se)
    ci_upper = np.tanh(z + z_alpha * se)
    
    # 부트스트랩 CI (더 정확)
    n_boot = 1000
    boot_ccc = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        bt, bp = y_true[idx], y_pred[idx]
        m1, m2 = np.mean(bt), np.mean(bp)
        v1 = np.var(bt); v2 = np.var(bp)
        cv = np.mean((bt-m1)*(bp-m2))
        denom = v1 + v2 + (m1-m2)**2
        boot_ccc.append(2*cv/denom if denom>0 else 0)
    
    boot_arr = np.array(boot_ccc)
    
    return {
        'ccc':          round(ccc, 4),
        'ci_fisher':    (round(ci_lower,4), round(ci_upper,4)),
        'ci_bootstrap': (round(np.percentile(boot_arr,2.5),4),
                         round(np.percentile(boot_arr,97.5),4)),
    }

# 출력 예시:
# {'ccc': 0.9789,
#  'ci_fisher':    (0.9711, 0.9847),
#  'ci_bootstrap': (0.9704, 0.9851)}