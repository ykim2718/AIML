import numpy as np
import matplotlib.pyplot as plt

def ccc_bland_altman_report(y_true, y_pred, method_names=("Reference", "Model")):
    """
    CCC + Bland-Altman 통합 분석
    방법 비교(Method Comparison) 완전한 리포트
    """
    y_true, y_pred = np.array(y_true, dtype=float), \
                     np.array(y_pred, dtype=float)
    n = len(y_true)
    
    # ── CCC 계산 ──────────────────
    mu_t, mu_p = np.mean(y_true), np.mean(y_pred)
    var_t = np.var(y_true);  var_p = np.var(y_pred)
    cov   = np.mean((y_true-mu_t)*(y_pred-mu_p))
    r     = cov / np.sqrt(var_t * var_p)
    ccc   = 2*cov / (var_t + var_p + (mu_t-mu_p)**2)
    
    # ── Bland-Altman 계산 ─────────
    diff  = y_pred - y_true      # 차이
    mean_ = (y_pred + y_true)/2  # 평균
    md    = np.mean(diff)        # Mean Difference (Bias)
    sd    = np.std(diff, ddof=1)
    loa_u = md + 1.96*sd        # Upper Limit of Agreement
    loa_l = md - 1.96*sd        # Lower Limit of Agreement
    
    # ── 시각화 (2-panel) ──────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    
    # Panel 1: Concordance plot
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, color='#534AB7')
    mn, mx = min(y_true.min(), y_pred.min()), \
             max(y_true.max(), y_pred.max())
    lim = [mn-2, mx+2]
    ax.plot(lim, lim, '--', color='#1D9E75', lw=1.5, label='y=x (완벽한 일치)')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(method_names[0]); ax.set_ylabel(method_names[1])
    ax.set_title(f'Concordance Plot\nCCC={ccc:.3f}, r={r:.3f}')
    ax.legend(fontsize=9)
    
    # Panel 2: Bland-Altman
    ax = axes[1]
    ax.scatter(mean_, diff, alpha=0.6, s=30, color='#D85A30')
    ax.axhline(md,    color='#D85A30', lw=2,   label=f'Bias={md:.2f}')
    ax.axhline(loa_u, color='gray',    lw=1.2, ls='--', label=f'+1.96SD={loa_u:.2f}')
    ax.axhline(loa_l, color='gray',    lw=1.2, ls='--', label=f'-1.96SD={loa_l:.2f}')
    ax.axhline(0,     color='black',  lw=0.5, ls=':')
    ax.set_xlabel('Mean of two methods')
    ax.set_ylabel('Difference (B - A)')
    ax.set_title(f'Bland-Altman Plot\nLoA: [{loa_l:.2f}, {loa_u:.2f}]')
    ax.legend(fontsize=9)
    
    plt.tight_layout()
    plt.savefig('ccc_bland_altman.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return {'CCC': ccc, 'r': r, 'bias_MD': md, 'sd': sd,
            'LoA': (loa_l, loa_u)}