# Centered R² vs Uncentered R²
Rev. 14 | Created: 2026-04-25 | Updated: 2026-09-05 21:47 CDT

## 1. Introduction: R² and Its Relation to RSQ

The coefficient of determination, denoted as R² (R-squared), is one of the most
widely used validation metrics in statistics and machine learning (ML) for
assessing how well predicted values $\hat{y}$ agree with observed values
$y_{true}$. In spreadsheet software and many engineering reports, R² is often
referred to as **RSQ** (the Excel function name `RSQ()`), which computes the
squared Pearson correlation coefficient between two variables. The default RSQ
in Excel and the standard R² reported by most regression libraries refer to the
same quantity: the **centered R²**, which measures the proportion of variance in
$y$ explained by the model relative to its mean baseline.

However, a less commonly discussed variant — the **uncentered R²** — replaces
the mean-based baseline with a zero baseline. This article compares the two,
derives the uncentered formula, provides a geometric interpretation, and
discusses when each is appropriate, with the Python implementations collected in
[Appendix B](#appendix-b-python-code). In particular, we examine whether
uncentered R² is more suitable for evaluating 1:1-line agreement between
$y_{true}$ and $\hat{y}$ in same-physical-quantity comparisons
[[3](#ref-3)][[4](#ref-4)].

## 2. Comparison of the Two Formulas

$$R^2_c = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

$$R^2_u = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n y_i^2}$$

- The difference lies entirely in the denominator. Centered R² asks "how much
  variance around the mean has been explained?", whereas uncentered R² asks "how
  much of the total sum of squares around zero has been explained?". Centered R²
  uses the mean prediction as its baseline, while uncentered R² uses zero
  prediction as its baseline [[3](#ref-3)].
- Centered R² asks: "How much better is the model compared to a baseline that
  simply predicts the mean?"
- Uncentered R² asks: "How close is the model's prediction to the 1:1 identity
  line (y=x)?"

## 3. Pros and Cons

### 3.1 Centered R²

- **Pros**: Scale- and unit-invariant, allowing comparison of models across
  different physical quantities. Equivalent to the squared Pearson correlation
  coefficient, providing rich statistical intuition.
- **Cons**: When the variance of $y$ is small (nearly constant), the denominator
  approaches zero and R² collapses or even turns negative. Measures only
  relative performance against a mean predictor and is insensitive to systematic
  bias.

### 3.2 Uncentered R²

- **Pros**: When comparing models, datasets, or experiments at the same
  physical-quantity level, uncentered R² provides a consistent absolute baseline
  (zero prediction), enabling coherent cross-comparison. Robust against
  low-variance data. Sensitive to deviations from the 1:1 line, properly
  penalizing systematic bias [[4](#ref-4)].
- **Cons**: Scale-dependent, so unsuitable for comparing models across different
  units or value ranges. When $\bar{y}$ is large, $R^2_u$ tends to be
  artificially close to 1. Uncentered $R^2$ values tend to be significantly
  higher (often reaching 0.99 or above) because the denominator, $\sum y_i^2$,
  is generally much larger than the centered denominator,
  $\sum (y_i - \bar{y})^2$. This can lead to an overly optimistic assessment if
  interpreted in isolation. However, for min-max normalized data, the uniformity
  of scale renders the uncentered $R^2$ a superior metric for quantifying the
  absolute agreement between observed and predicted values.

**Key question**: When comparing models on data of the same physical quantity,
is uncentered R² actually more accurate? The answer is essentially "yes." For
evaluating absolute agreement between $y_{true}$ and $\hat{y}$ in same-unit
data, uncentered R² (1) is not distorted by changes in dataset variance, (2) is
sensitive to constant bias, and (3) shares an absolute baseline (zero
prediction) across datasets, enabling consistent model-to-model comparison.

## 4. Applications

### 4.1 Centered R²

- Standard performance metric in general regression and ML modeling.
- Comparing model performance across heterogeneous units or domains in
  normalized form.
- Variance-explanation evaluation based on Pearson correlation
  [[1](#ref-1)].

### 4.2 Uncentered R²

- **Regression Through the Origin (RTO)**: Models physically constrained to pass
  through zero, such as Hooke's law and radioactive decay. The `noconstant`
  option in Stata, R, and EViews automatically returns uncentered R²
  [[2](#ref-2)]. When the intercept is fixed to zero, standard (centered) $R^2$
  can produce negative values or misleading interpretations. In such cases,
  uncentered $R^2$ is used as a robust alternative to evaluate the model's fit
  relative to the origin.
- **Diagnostic tests in econometrics**: Auxiliary regressions in the
  Breusch-Pagan and White tests use $nR^2$ as a chi-squared statistic, computed
  in uncentered form [[6](#ref-6)].
- **Evaluating deviation from the 1:1 line in $Y_{true}$ vs $Y_{pred}$ plots**:
  For ML prediction tasks involving same-physical-quantity data, uncentered R²
  measures absolute agreement. Suitable for calibration, Computational Fluid
  Dynamics (CFD) and Finite Element Method (FEM) validation, sensor calibration,
  and reproducibility studies.

## 5. Derivation of Uncentered R²

### 5.1 Sum of Squares Decomposition

For observations $y_i$, predictions $\hat{y}_i$, and residuals
$e_i = y_i - \hat{y}_i$ ($i = 1, \ldots, n$), define the uncentered sums of
squares — Total Sum of Squares (TSS), Explained Sum of Squares (ESS), and
Residual Sum of Squares (RSS) — as follows:

$$TSS_u = \sum_{i=1}^n y_i^2, \quad ESS_u = \sum_{i=1}^n \hat{y}_i^2, \quad RSS = \sum_{i=1}^n e_i^2$$

### 5.2 Ordinary Least Squares (OLS) Orthogonality Condition

From the OLS normal equations $X^T(y - X\hat{\beta}) = 0$, every column of the
design matrix $X$ is orthogonal to the residual vector $e$:

$$\sum_{i=1}^n \hat{y}_i \cdot e_i = \hat{\beta}^T X^T e = 0$$

The fitted-value vector and the residual vector are orthogonal. This holds
regardless of whether the model includes an intercept.

### 5.3 Uncentered Decomposition

Squaring $y_i = \hat{y}_i + e_i$ and summing over $i$:

$$\sum y_i^2 = \sum \hat{y}_i^2 + 2\sum \hat{y}_i e_i + \sum e_i^2$$

The cross term vanishes by orthogonality $\sum \hat{y}_i e_i = 0$, leaving:

$$\sum y_i^2 = \sum \hat{y}_i^2 + \sum e_i^2 \quad \Longleftrightarrow \quad TSS_u = ESS_u + RSS$$

### 5.4 Definition of Uncentered R²

Dividing both sides by $TSS_u$ gives $1 = ESS_u/TSS_u + RSS/TSS_u$, and
uncentered R² is defined as the explained ratio:

$$R^2_u \equiv \frac{ESS_u}{TSS_u} = \frac{\sum \hat{y}_i^2}{\sum y_i^2} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum y_i^2}$$

### 5.5 Relation to Centered R²

Since $\sum y_i^2 = \sum (y_i - \bar{y})^2 + n\bar{y}^2$, we have
$TSS_u = TSS_c + n\bar{y}^2$. For an intercept-included model:

$$R^2_u = \frac{TSS_c \cdot R^2_c + n\bar{y}^2}{TSS_c + n\bar{y}^2}$$

When $\bar{y}$ is large or $n$ is large, $R^2_u$ moves closer to 1 than $R^2_c$.
When $\bar{y} = 0$, the two coincide.

## 6. Geometric Interpretation

### 6.1 Vector Space Setup

Treating the $n$ observations as vectors
$\mathbf{y}, \hat{\mathbf{y}}, \mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} \in \mathbb{R}^n$,
every sum of squares becomes a squared norm: $\sum y_i^2 = \|\mathbf{y}\|^2$,
$\sum \hat{y}_i^2 = \|\hat{\mathbf{y}}\|^2$, $\sum e_i^2 = \|\mathbf{e}\|^2$.

### 6.2 OLS as Orthogonal Projection

Let $\mathcal{C}(X)$ denote the column space of the design matrix $X$. The OLS
estimate is $\hat{\mathbf{y}} = P_X \mathbf{y}$ with
$P_X = X(X^TX)^{-1}X^T$, the orthogonal projection of $\mathbf{y}$ onto
$\mathcal{C}(X)$. The residual $\mathbf{e}$ lies in the orthogonal complement,
automatically satisfying $\mathbf{e} \perp \hat{\mathbf{y}}$ [[5](#ref-5)].

### 6.3 Uncentered Decomposition as the Pythagorean Theorem

Since $\mathbf{y} = \hat{\mathbf{y}} + \mathbf{e}$ and
$\hat{\mathbf{y}} \perp \mathbf{e}$, a right triangle is formed:

```text
              y
             /|
            / |
           /  |  e  (perpendicular residual)
          /   |
         /____|
      origin  ŷ  (on the C(X) plane)
```

$$\|\mathbf{y}\|^2 = \|\hat{\mathbf{y}}\|^2 + \|\mathbf{e}\|^2$$

This is the geometric identity behind $TSS_u = ESS_u + RSS$ — not merely an
algebraic identity, but literally the Pythagorean theorem. The geometric meaning
of uncentered R² becomes:

$$R^2_u = \frac{\|\hat{\mathbf{y}}\|^2}{\|\mathbf{y}\|^2} = \cos^2\theta_u$$

where $\theta_u$ is the angle between $\mathbf{y}$ and $\hat{\mathbf{y}}$
measured from the origin. $R^2_u \to 1$ means the two vectors point in the same
direction; $R^2_u \to 0$ means they are orthogonal.

### 6.4 Centered R²: Subtracting the Mean Vector

Subtracting the mean vector $\bar{\mathbf{y}} = \bar{y} \cdot \mathbf{1}$ is
geometrically equivalent to projecting onto the hyperplane $\mathbf{1}^\perp$,
which removes the component along the constant vector $\mathbf{1}$. When the
model contains an intercept, $\mathbf{1} \in \mathcal{C}(X)$, so a right
triangle still forms within $\mathbf{1}^\perp$:

$$R^2_c = \frac{\|\tilde{\hat{\mathbf{y}}}\|^2}{\|\tilde{\mathbf{y}}\|^2} = \cos^2\theta_c$$

where $\theta_c$ is the angle between the two vectors after the mean has been
removed — equivalent to the squared Pearson correlation coefficient.

### 6.5 Geometric Distinction Between the Two R²

```text
              y
             /|
            / |
           /  |
          /   | e
         /    |
        /     |
       *------ŷ-----→ (C(X) plane)
   origin
     /
    *────────── 1 direction (constant vector)
    ȳ·1
```

- **Uncentered R²**: angle between $\mathbf{y}$ and $\hat{\mathbf{y}}$ measured
  from the origin.
- **Centered R²**: angle measured after shifting the reference point to
  $\bar{y}\cdot\mathbf{1}$.

The fundamental distinction is whether the reference point is the origin or the
mean vector. The same data, viewed from different reference points, yields
different angles between the two vectors.

### 6.6 Comparison Table

Table 1. Geometric distinction between uncentered and centered R²

| Aspect | Uncentered | Centered |
|---|---|---|
| Reference point | Origin 0 | Mean vector ȳ·1 |
| Right triangle | y, ŷ, e | ỹ, ŷ̃, e |
| Angle interpretation | cos²(∠yŷ) | cos²(∠ỹŷ̃) = r² |
| Required condition | e ⊥ ŷ | e ⊥ ŷ and 1 ∈ C(X) |
| Sensitivity to bias | Sensitive (origin-fixed) | Insensitive (mean-shift invariant) |

**Geometric meaning of 1:1-line validation**: Under a constant bias
$\hat{\mathbf{y}} = \mathbf{y} + c\mathbf{1}$, the uncentered angle widens but
the centered angle remains unchanged. Therefore, for evaluating absolute
agreement, uncentered R² responds more sensitively.

## References

<a id="ref-1"></a>[1] Draper, N. R., & Smith, H. (1998). [*Applied Regression Analysis*](https://doi.org/10.1002/9781118625590) (3rd ed.). Wiley.<br>
<a id="ref-2"></a>[2] Eisenhauer, J. G. (2003). [Regression through the origin](https://doi.org/10.1111/1467-9639.00136). *Teaching Statistics*, 25(3), 76–80.<br>
<a id="ref-3"></a>[3] Kvalseth, T. O. (1985). [Cautionary note about R²](https://doi.org/10.1080/00031305.1985.10479448). *The American Statistician*, 39(4), 279–285.<br>
<a id="ref-4"></a>[4] Legates, D. R., & McCabe, G. J. (1999). [Evaluating the use of "goodness-of-fit" measures in hydrologic and hydroclimatic model validation](https://doi.org/10.1029/1998WR900018). *Water Resources Research*, 35(1), 233–241.<br>
<a id="ref-5"></a>[5] Strang, G. (2009). [*Introduction to Linear Algebra*](https://wellesleycambridge.com/) (4th ed.). Wellesley-Cambridge Press. ISBN 978-0-9802327-1-4.<br>
<a id="ref-6"></a>[6] Wooldridge, J. M. (2010). [*Econometric Analysis of Cross Section and Panel Data*](https://mitpress.mit.edu/9780262232586/econometric-analysis-of-cross-section-and-panel-data/) (2nd ed.). MIT Press. ISBN 978-0-262-23258-6.

---

## Appendix A. Terminology

- **centered R²**: the standard R², whose denominator is the sum of squares about the mean of the data.
- **CFD**: Computational Fluid Dynamics, the numerical simulation of fluid flow.
- **ESS**: Explained Sum of Squares. In uncentered form, $\sum \hat{y}_i^2$.
- **FEM**: Finite Element Method, the numerical solution of field problems on a discretized domain.
- **MAE**: Mean Absolute Error, the mean of the absolute residuals.
- **MAPE**: Mean Absolute Percentage Error, the mean absolute residual expressed as a percentage of the observation. Undefined where an observation is zero.
- **OLS**: Ordinary Least Squares, the estimator that minimizes the sum of squared residuals.
- **RMSE**: Root Mean Squared Error, the square root of the mean squared residual.
- **RSQ**: the Excel function name for the squared Pearson correlation coefficient, which equals the centered R².
- **RSS**: Residual Sum of Squares, $\sum e_i^2$.
- **RTO**: Regression Through the Origin, a regression whose intercept is fixed to zero.
- **TSS**: Total Sum of Squares. In uncentered form, $\sum y_i^2$; in centered form, $\sum (y_i - \bar{y})^2$.
- **uncentered R²**: the R² whose denominator is the sum of squares about zero rather than about the mean.

## Appendix B. Python Code

### B.1 Centered R²

```python
import numpy as np


def centered_r2(y_true, y_pred):
    """
    Centered R² (standard R²)
    R²_c = 1 - SSE / SST_centered
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    sse = np.sum((y_true - y_pred) ** 2)
    sst_c = np.sum((y_true - np.mean(y_true)) ** 2)

    if sst_c == 0:
        return np.nan  # undefined when y has zero variance
    return 1.0 - sse / sst_c

# Example usage
y_true = np.array([10.1, 10.2, 10.3, 10.4, 10.5])
y_pred = np.array([10.0, 10.3, 10.2, 10.5, 10.4])

print(f"Centered R²:   {centered_r2(y_true, y_pred):.6f}")
# Confirm equivalence with scikit-learn
from sklearn.metrics import r2_score
print(f"sklearn r2:    {r2_score(y_true, y_pred):.6f}")
```

### B.2 Uncentered R²

```python
def uncentered_r2(y_true, y_pred):
    """
    Uncentered R² (origin-based R²)
    R²_u = 1 - SSE / SST_uncentered
         = ⟨ŷ, ŷ⟩ / ⟨y, y⟩  (when e ⊥ ŷ)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    sse = np.sum((y_true - y_pred) ** 2)
    sst_u = np.sum(y_true ** 2)

    if sst_u == 0:
        return np.nan
    return 1.0 - sse / sst_u

# Example usage — comparing predictions on a same-physical-quantity scale (e.g., voltage)
y_true = np.array([10.1, 10.2, 10.3, 10.4, 10.5])
y_pred = np.array([10.0, 10.3, 10.2, 10.5, 10.4])

print(f"Uncentered R²: {uncentered_r2(y_true, y_pred):.6f}")

# Constant-bias scenario: contrast between the two metrics
y_pred_biased = y_true + 0.5  # all predictions offset by +0.5
print(f"\nConstant bias +0.5 scenario:")
print(f"  Centered R²:   {centered_r2(y_true, y_pred_biased):.6f}  "
      f"(insensitive to bias)")
print(f"  Uncentered R²: {uncentered_r2(y_true, y_pred_biased):.6f}  "
      f"(sensitive to bias)")
```

### B.3 Integrated Evaluation for 1:1-Line Agreement

```python
def evaluate_1_to_1_line_agreement(y_true, y_pred):
    """Evaluate 1:1-line agreement on same-physical-quantity data."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    err = y_pred - y_true

    rmse = np.sqrt(np.mean(err ** 2))
    mae  = np.mean(np.abs(err))                       # Mean Absolute Error
    bias = np.mean(err)                               # Mean Error (systematic bias)

    # MAPE: Mean Absolute Percentage Error (%) — undefined when y_true contains 0
    mask = y_true != 0
    if mask.any():
        mape = np.mean(np.abs(err[mask] / y_true[mask])) * 100.0
    else:
        mape = np.nan

    return {
        "CenteredR2":   centered_r2(y_true, y_pred),
        "UncenteredR2": uncentered_r2(y_true, y_pred),
        "RMSE":         rmse,
        "MAE":          mae,
        "MAPE":         mape,
        "Bias":         bias,
    }


# Example usage
y_true = np.array([10.1, 10.2, 10.3, 10.4, 10.5])
y_pred = np.array([10.0, 10.3, 10.2, 10.5, 10.4])

results = evaluate_1_to_1_line_agreement(y_true, y_pred)
for k, v in results.items():
    print(f"{k:>15s}: {v:.6f}")
```
