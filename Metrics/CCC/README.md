# 📊 CCC & Bland-Altman Analysis Guide

> A comprehensive guide to **Concordance Correlation Coefficient (CCC)** + **Bland-Altman Plot**  
> A statistical toolkit for evaluating agreement between two measurement methods

---

## Table of Contents

- [What is CCC?](#what-is-ccc)
- [CCC Components](#ccc-components)
- [Interpretation Benchmarks](#interpretation-benchmarks)
- [Bland-Altman Plot](#bland-altman-plot)
- [Combined CCC + Bland-Altman Analysis](#combined-ccc--bland-altman-analysis)
- [Data Concentration Problem & Alternative Metrics](#data-concentration-problem--alternative-metrics)
- [Applications in AI/ML](#applications-in-aiml)
- [Python Code Examples](#python-code-examples)
- [Metric Selection Guide](#metric-selection-guide)

---

## What is CCC?

**Lin's Concordance Correlation Coefficient** (Lawrence Lin, 1989) measures how well two measurement methods agree with each other.

Unlike Pearson r, which only measures the strength of a linear relationship, CCC simultaneously captures both **accuracy** and **precision**.

### Formula

$$\rho_c = \frac{2\sigma_{xy}}{\sigma_x^2 + \sigma_y^2 + (\mu_x - \mu_y)^2} = r \times C_b$$

| Symbol | Meaning |
|--------|---------|
| $\rho_c$ | Concordance Correlation Coefficient |
| $r$ | Pearson correlation coefficient (precision) |
| $C_b$ | Bias correction factor (accuracy, range 0–1) |
| $\mu_x, \mu_y$ | Means of the two methods |
| $\sigma_x^2, \sigma_y^2$ | Variances of the two methods |
| $\sigma_{xy}$ | Covariance |

### Key Difference from Pearson r

```
r = 0.98 can coexist with CCC = 0.73 — when systematic bias is present
```

- **r**: Measures only the linear relationship against *any* straight line
- **CCC**: Measures deviation from the perfect concordance line y=x

---

## CCC Components

```
CCC (ρc)
├── Pearson r        ← Precision: how tightly data follows a straight line
└── Cb (bias factor) ← Accuracy: whether that line is y=x
    ├── Location shift (μx − μy)
    └── Scale shift    (σx vs σy)
```

- `Cb = 1.0` → Perfect accuracy (regression line coincides with y=x)
- `Cb < 1.0` → Bias present (mean difference or scale difference)

---

## Interpretation Benchmarks

| CCC value | Strength of agreement (McBride 2005) |
|-----------|--------------------------------------|
| > 0.99 | Almost perfect |
| 0.95 – 0.99 | Substantial |
| 0.90 – 0.95 | Moderate |
| < 0.90 | Poor |

### Five Bias Scenarios

| Scenario | CCC | Pearson r | Root cause |
|----------|-----|-----------|------------|
| Good agreement | High | High | — |
| Mean shift bias | **Low** | High ⚠ | Method B consistently higher or lower |
| Scale bias | **Low** | High ⚠ | Method B has different variance |
| Wide scatter | Low | Low | Insufficient measurement precision |
| Proportional bias | Medium | High ⚠ | Error grows with magnitude |

> ⚠ Concluding "agreement" based on r alone can lead to incorrect results.

---

## Bland-Altman Plot

The standard visualization for method comparison studies, proposed by Bland & Altman (1986).

- **X-axis**: Mean of the two measurements `(A + B) / 2`
- **Y-axis**: Difference between the two measurements `B − A`

### Key Statistics

| Statistic | Formula | Meaning |
|-----------|---------|---------|
| Mean Difference (MD) | `mean(B − A)` | Systematic bias; ideally 0 |
| Upper LoA | `MD + 1.96 × SD` | Upper bound of 95% of differences |
| Lower LoA | `MD − 1.96 × SD` | Lower bound of 95% of differences |
| LoA width | `3.92 × SD` | Narrower = better agreement |

### Detecting Proportional Bias

When points in the Bland-Altman plot show a **sloping pattern from left to right**, proportional bias is present — errors increase as the measured value grows. This pattern is difficult to detect from the concordance plot alone.

---

## Combined CCC + Bland-Altman Analysis

The two plots view the same problem from different angles.

```
Concordance plot  → "How well do the methods agree overall?" (summary number)
Bland-Altman      → "Why and how do they disagree?"         (bias diagnosis)
```

| Pattern | Concordance plot | Bland-Altman | Diagnosis |
|---------|-----------------|--------------|-----------|
| Good agreement | Points lie on y=x | MD≈0, narrow LoA | Methods interchangeable |
| Mean shift | Parallel offset from y=x | MD ≠ 0 (constant) | Calibration needed |
| Proportional bias | Different slope | Upward-sloping pattern | Range-specific correction needed |
| Wide scatter | Widely spread | Wide LoA | Measurement precision must improve |

---

## Data Concentration Problem & Alternative Metrics

### Problem: Data Clustered in a Narrow Range

When data is tightly clustered, CCC/r becomes **unstable due to near-zero variance**.

```python
# Narrow range → denominator approaches 0 → CCC unstable
ccc = 2 * cov / (var_x + var_y + (mean_x - mean_y)**2)
#                ↑ this value approaches 0
```

The opposite problem arises with **two-cluster data**: CCC/r appears artificially inflated due to the range effect — even when within-cluster agreement is poor.

### Alternative Metrics (Range-Independent)

| Metric | Formula | Characteristics |
|--------|---------|-----------------|
| **RMSD₁:₁** | `sqrt(mean((y−x)²)) / sqrt(2)` | Perpendicular distance to 1:1 line; optimal for normal errors |
| **MAD₁:₁** | `mean(\|y − x\|)` | Robust to outliers; same units as measurements |
| **TDI (90%)** | 90th percentile of `\|y − x\|` | "90% of differences fall within this value" |
| **CP** | `count(\|y−x\| ≤ δ) / n × 100` | Pass rate (%) against a pre-defined tolerance δ |

### Diagnostic Checklist

```
□ Are data points spread widely along y=x in the concordance plot?
□ Does the Bland-Altman x-axis cover the full clinically relevant range?
□ Is the data span / LoA width ratio sufficiently large?
□ Are two clusters mixed, creating an artificial range effect?
```

### Recommended Combinations

- Data spread over a wide range → **CCC + Bland-Altman**
- Data concentrated / narrow range → **RMSD₁:₁ + TDI(90%) + CP**

---

## Applications in AI/ML

### 1. Regression Model Evaluation

RMSE or R² alone cannot detect systematic bias in model predictions.

```python
from sklearn.metrics import make_scorer

def ccc_score(y_true, y_pred):
    mu_t, mu_p = np.mean(y_true), np.mean(y_pred)
    var_t, var_p = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mu_t) * (y_pred - mu_p))
    return 2 * cov / (var_t + var_p + (mu_t - mu_p)**2)

# Use CCC as the optimization objective in GridSearchCV
ccc_scorer = make_scorer(ccc_score, greater_is_better=True)
grid = GridSearchCV(model, param_grid, scoring=ccc_scorer)
```

### 2. Key Application Domains

| Domain | Use case |
|--------|----------|
| Medical imaging AI | Agreement between AI and clinician measurements (FDA-recommended) |
| Natural language processing | Automated scoring vs. human annotator agreement |
| Domain adaptation | Prediction consistency after source → target transfer |
| Ensemble models | Quantifying diversity via inter-model agreement |
| Biosignal prediction | Clinical tolerance evaluation for blood pressure, glucose, SpO₂ |

---

## Python Code Examples

### Basic CCC Implementation

```python
import numpy as np

def concordance_correlation_coefficient(y_true, y_pred):
    """
    Lin's Concordance Correlation Coefficient (CCC)
    Returns: dict with ccc, r, Cb, bias, var_ratio
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mu_t, mu_p = np.mean(y_true), np.mean(y_pred)
    var_t = np.var(y_true)
    var_p = np.var(y_pred)
    cov   = np.mean((y_true - mu_t) * (y_pred - mu_p))

    r   = cov / np.sqrt(var_t * var_p)
    ccc = 2 * cov / (var_t + var_p + (mu_t - mu_p)**2)
    cb  = ccc / r if r != 0 else 0

    return {
        'ccc':       round(ccc, 4),
        'r':         round(r,   4),
        'Cb':        round(cb,  4),
        'bias':      round(mu_p - mu_t, 4),
        'var_ratio': round(np.sqrt(var_p / var_t), 4),
    }
```

### With Confidence Intervals (Fisher's Z + Bootstrap)

```python
def ccc_with_ci(y_true, y_pred, n_boot=1000):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_true)

    mu_t, mu_p = np.mean(y_true), np.mean(y_pred)
    var_t, var_p = np.var(y_true), np.var(y_pred)
    cov = np.mean((y_true - mu_t) * (y_pred - mu_p))
    ccc = 2 * cov / (var_t + var_p + (mu_t - mu_p)**2)

    # Fisher's Z-transform 95% CI
    z      = np.arctanh(ccc)
    se     = 1 / np.sqrt(n - 3)
    ci_low = np.tanh(z - 1.96 * se)
    ci_up  = np.tanh(z + 1.96 * se)

    # Bootstrap CI
    boot = []
    for _ in range(n_boot):
        idx = np.random.choice(n, n, replace=True)
        bt, bp = y_true[idx], y_pred[idx]
        m1, m2 = np.mean(bt), np.mean(bp)
        v1, v2 = np.var(bt), np.var(bp)
        cv = np.mean((bt - m1) * (bp - m2))
        d  = v1 + v2 + (m1 - m2)**2
        boot.append(2 * cv / d if d > 0 else 0)

    boot = np.array(boot)
    return {
        'ccc':          round(ccc, 4),
        'ci_fisher':    (round(ci_low, 4), round(ci_up, 4)),
        'ci_bootstrap': (round(np.percentile(boot, 2.5), 4),
                         round(np.percentile(boot, 97.5), 4)),
    }
```

### Alternative Metrics for Concentrated Data

```python
def robust_agreement_metrics(y_true, y_pred, delta=5.0, tdi_pct=0.90):
    """
    Range-independent metrics for deviation from the 1:1 line.
    - RMSD_1_1 : perpendicular RMSD to the 1:1 line
    - MAD_1_1  : mean absolute deviation from identity
    - TDI      : Total Deviation Index
    - CP       : Coverage Probability
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    diffs     = y_pred - y_true
    abs_diffs = np.abs(diffs)

    rmsd_11 = np.sqrt(np.mean(diffs**2)) / np.sqrt(2)
    mad_11  = np.mean(abs_diffs)
    tdi     = np.quantile(abs_diffs, tdi_pct)
    cp      = np.mean(abs_diffs <= delta) * 100

    return {
        'RMSD_1_1': round(rmsd_11, 4),
        'MAD_1_1':  round(mad_11,  4),
        f'TDI_{int(tdi_pct*100)}pct': round(tdi, 4),
        f'CP_delta_{delta}': round(cp, 2),
    }
```

### Full CCC + Bland-Altman Report

```python
import matplotlib.pyplot as plt

def full_agreement_report(y_true, y_pred,
                          method_names=("Reference", "Model"),
                          delta=5.0):
    y_true, y_pred = np.array(y_true, dtype=float), \
                     np.array(y_pred, dtype=float)

    # CCC
    res_ccc = concordance_correlation_coefficient(y_true, y_pred)

    # Bland-Altman
    diff   = y_pred - y_true
    mean_  = (y_pred + y_true) / 2
    md     = np.mean(diff)
    sd     = np.std(diff, ddof=1)
    loa_u  = md + 1.96 * sd
    loa_l  = md - 1.96 * sd

    # Alternative metrics
    res_rob = robust_agreement_metrics(y_true, y_pred, delta=delta)

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Concordance plot
    ax = axes[0]
    ax.scatter(y_true, y_pred, alpha=0.6, s=30, color='#534AB7')
    lim = [min(y_true.min(), y_pred.min()) - 2,
           max(y_true.max(), y_pred.max()) + 2]
    ax.plot(lim, lim, '--', color='#1D9E75', lw=1.5, label='y=x')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(method_names[0]); ax.set_ylabel(method_names[1])
    ax.set_title(f"Concordance Plot\n"
                 f"CCC={res_ccc['ccc']}, r={res_ccc['r']}, "
                 f"Cb={res_ccc['Cb']}")
    ax.legend(fontsize=9)

    # Bland-Altman plot
    ax = axes[1]
    ax.scatter(mean_, diff, alpha=0.6, s=30, color='#D85A30')
    ax.axhline(md,    color='#D85A30', lw=2,   label=f'Bias={md:.2f}')
    ax.axhline(loa_u, color='gray',    lw=1.2, ls='--',
               label=f'+1.96SD={loa_u:.2f}')
    ax.axhline(loa_l, color='gray',    lw=1.2, ls='--',
               label=f'-1.96SD={loa_l:.2f}')
    ax.axhline(0,     color='black',   lw=0.5, ls=':')
    ax.set_xlabel('Mean of two methods')
    ax.set_ylabel('Difference (Model − Reference)')
    ax.set_title(f"Bland-Altman Plot\n"
                 f"LoA: [{loa_l:.2f}, {loa_u:.2f}]")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig('agreement_report.png', dpi=150, bbox_inches='tight')
    plt.show()

    return {**res_ccc, **res_rob,
            'bias_MD': round(md, 4),
            'LoA': (round(loa_l, 4), round(loa_u, 4))}
```

---

## Metric Selection Guide

```
Is data spread over a wide range?
├── YES → CCC + Bland-Altman  (standard combination)
│         └── Check BA plot slope if proportional bias is suspected
└── NO  → Is data concentrated in a narrow range?
          ├── YES → RMSD₁:₁ + TDI(90%) + CP
          │         (CCC/r cannot be trusted)
          └── Are two clusters mixed?
              └── YES → Analyze each cluster separately, then apply CCC
                        (overall CCC is inflated by the range effect)
```

### Recommended Reporting Checklist

```python
report = {
    # Required
    'CCC':      ...,   # overall agreement
    'r':        ...,   # precision
    'Cb':       ...,   # accuracy (bias correction factor)
    'CI_95':    ...,   # confidence interval

    # Bland-Altman
    'Bias_MD':  ...,   # mean difference
    'LoA':      ...,   # limits of agreement

    # Add for concentrated data or clinical validation
    'RMSD_1_1': ...,
    'TDI_90':   ...,
    'CP':       ...,   # tolerance limit δ must be stated explicitly
}
```

---

## References

- Lin, L.I. (1989). *A concordance correlation coefficient to evaluate reproducibility.* Biometrics, 45(1), 255–268.
- Bland, J.M. & Altman, D.G. (1986). *Statistical methods for assessing agreement between two methods of clinical measurement.* Lancet, 327(8476), 307–310.
- McBride, G.B. (2005). *A proposal for strength-of-agreement criteria for Lin's Concordance Correlation Coefficient.* NIWA Client Report.
- Barnhart, H.X. et al. (2007). *Overview on assessing agreement with continuous measurements.* Journal of Biopharmaceutical Statistics.

---

## License

MIT License

---

*This guide serves as a statistical reference for Method Comparison Studies.*
