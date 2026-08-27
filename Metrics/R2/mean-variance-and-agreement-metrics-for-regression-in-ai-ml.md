# Mean, Variance, and Agreement Metrics for Regression in AI/ML
Rev. 37 | Created: 2026-04-08 | Updated: 2026-08-20 11:17 CDT
> A taxonomy of regression evaluation metrics split into variance-based,
> mean-based, and agreement-based families, read against the $y=x$ line and
> against the low variance effect that makes several of them collapse.

## 1. Executive Summary

In advanced engineering domains such as semiconductor manufacturing, virtual metrology, and
multi-sensor time-series analysis, the validation of a predictive model requires more than a
single performance score. Two questions have to be separated: how well the model tracks a trend,
which is precision, and how close the prediction is to the physical truth, which is accuracy.

This document establishes a taxonomy for evaluation metrics, categorized into variance-based,
mean-based, and agreement-based indices. The hierarchy provides a systematic framework for
interpreting model performance relative to the ideal $y=x$ line, with a specific focus on the
risks associated with the low variance effect.

## 2. Metric Hierarchy

The three families divide first by what the metric is built from, and then by whether the value
carries the unit of the measurement.

```
Regression metrics
├── Mean-based
│   ├── Scale-dependent   → MAE, MSE, RMSE, Huber
│   └── Scale-independent → MPE, MAPE, SMAPE, CV(RMSE)
├── Variance-based
│   └── Scale-independent → R², Adj. R²
└── Mean+Variance-based (Hybrid)
    └── Scale-independent → CCC, KGE
```

### 2.1 Variance Index

These metrics assess the linearity and the strength of the relationship between observed and
predicted values. They focus on whether the model captures the shape of the data, regardless of
absolute magnitude.

#### Pearson Correlation Coefficient

$$r = \frac{\sum (y_i - \mu_y)(\hat{y}_i - \mu_{\hat{y}})}{\sqrt{\sum (y_i - \mu_y)^2 \sum (\hat{y}_i - \mu_{\hat{y}})^2}}$$

where $y_i$ is the observed ground truth value, $\hat{y}_i$ is the predicted value, and $\mu_y$
and $\mu_{\hat{y}}$ are the means of the observed and the predicted values.

- Relation to the 1:1 line: $r$ measures how tightly the data clusters around any straight line.
  A perfect $r=1$ does not guarantee that the data is on the $y=x$ line, since it could be on
  $y = 2x + 10$.
- Low variance effect: if the data has very low variance, for instance a sensor outputting a
  nearly constant value, the denominator approaches zero. This makes $r$ extremely sensitive to
  tiny amounts of noise, often resulting in a low or undefined correlation despite the
  prediction being physically close to the truth.
- Application: initial feature selection, and identifying sensors with similar behavioral
  patterns.

#### Coefficient Of Determination

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \mu_y)^2}$$

where $SS_{res}$ is the residual sum of squares, that is the unexplained variance, and
$SS_{tot}$ is the total sum of squares, that is the total variance in the data.

- Relation to the 1:1 line: it represents the proportion of variance explained by the model.
  While it penalizes distance from the 1:1 line more than $r$ does, it is still misleading when
  the model is systematically biased.
- Low variance effect: $R^2$ is deceptive when the target variance is low. Because the
  denominator $SS_{tot}$ is small, even a tiny prediction error produces a negative or near-zero
  $R^2$, suggesting a bad model even when the absolute error is within engineering tolerance.
- Application: standard benchmark for regression model explanatory power in manufacturing yield
  analysis.

#### Explained Variance Score

$$ExpVar = 1 - \frac{Var(y - \hat{y})}{Var(y)}$$

where $Var(y - \hat{y})$ is the variance of the residuals and $Var(y)$ is the variance of the
ground truth.

- Relation to the 1:1 line: similar to $R^2$, but it ignores the mean of the residuals. It
  focuses purely on whether the fluctuations in the prediction match the fluctuations in the
  truth.
- Low variance effect: like $R^2$, this metric collapses when $Var(y)$ is small. It fails to
  provide a meaningful score for stable processes where the goal is to hold a constant setpoint.
- Application: signal processing where the relative change matters more than the absolute
  baseline.

### 2.2 Mean Index

These metrics measure the physical distance between the predicted vector and the ground truth.
They are essential for understanding the actual cost of an error.

#### Mean Absolute Error

$$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

where $n$ is the number of samples.

- Relation to the 1:1 line: the average vertical distance to the line.
- Limitation: it does not highlight large, infrequent errors, since it treats all deviations
  linearly. It is unaffected by the low variance effect, which makes it more reliable for stable
  processes.
- Application: situations where the error cost is strictly proportional to the error magnitude.

#### Mean Squared Error And Root Mean Squared Error

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2, \quad RMSE = \sqrt{MSE}$$

- Relation to the 1:1 line: the average of the squared distances to the line. RMSE represents
  the typical distance in the original units.
- Limitation: heavily influenced by outliers. While robust to low variance in the target, these
  metrics do not say whether the model is capturing the trend of the data.
- Application: standard loss function for training, and critical for thickness prediction where
  large deviations lead to wafer scrap.

#### Mean Percentage Error And Mean Absolute Percentage Error

$$MPE = \frac{100\%}{n} \sum_{i=1}^{n} \frac{y_i - \hat{y}_i}{y_i}, \quad MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

- Relation to the 1:1 line: these metrics evaluate the relative deviation from the line. MPE
  measures the average percentage bias, that is whether the model consistently overestimates or
  underestimates, while MAPE represents the average magnitude of percentage error relative to
  the identity line.
- Limitation: the most significant weakness is the division by zero, or by a near-zero value; if
  the target is zero or very small, the metrics explode. MAPE is also asymmetric, penalizing
  overestimation more heavily than underestimation in certain contexts. Its weakness is the
  variable denominator $y_i$: when the data scale stays consistent across all points, the
  denominator acts as a constant and MAPE becomes stable, scaling linearly with absolute error
  much as MSE or RMSE do. Unlike the agreement indices of
  section 2.3, these metrics do not distinguish a scale shift from a location shift.
- Application: communicating model performance to non-technical stakeholders in business terms,
  widely used in financial forecasting and yield management.

#### Coefficient Of Variation Of RMSE

$$CV(RMSE) = \frac{RMSE}{\mu_y}$$

- Relation to the 1:1 line: it normalizes the error by the mean.
- Low variance effect: if the mean $\mu_y$ is near zero, this metric explodes. It is nonetheless
  more stable than $R^2$ for low-variance datasets whose mean is not near zero.
- Application: comparing model performance across sensor types with different scales.

### 2.3 Agreement Index

These evaluate fidelity, the requirement that the model follow the trend and match the absolute
values at the same time.

#### Lin's Concordance Correlation Coefficient

$$\rho_c = \frac{2 \rho \sigma_y \sigma_{\hat{y}}}{\sigma_y^2 + \sigma_{\hat{y}}^2 + (\mu_y - \mu_{\hat{y}})^2}$$

where $\rho$ is the Pearson correlation coefficient and $\sigma_y$ and $\sigma_{\hat{y}}$ are
the standard deviations of the observed and the predicted values. The coefficient is abbreviated
CCC below.

- Relation to the 1:1 line: it directly measures how far the data deviates from the 45-degree
  line, combining $r$ as precision with a bias penalty as accuracy.
- Low variance effect: since $\rho$ is a component, CCC also decreases when the variance of the
  data is extremely low, which masks a model that performs well in absolute distance.
- Application: validating new metrology sensors against gold-standard lab measurements.

#### Kling-Gupta Efficiency

$$KGE = 1 - \sqrt{(r-1)^2 + (\alpha-1)^2 + (\beta-1)^2}$$

where $r$ is the Pearson correlation, $\alpha = \sigma_{\hat{y}}/\sigma_y$ is the variability
ratio, and $\beta = \mu_{\hat{y}}/\mu_y$ is the bias ratio.

- Relation to the 1:1 line: a holistic agreement metric. It reaches 1.0 only if $r$, $\alpha$,
  and $\beta$ are all 1.
- Low variance effect: extremely sensitive to the variability ratio $\alpha$. If the ground
  truth has nearly zero variance, $\alpha$ becomes undefined or unstable and KGE fails.
- Application: complex industrial process control and high-fidelity time-series simulation.

## 3. Comparative Summary

Table 1. The three metric families against the 1:1 line and the low variance effect

| Category | Primary focus | Best use case | Relation to $y=x$ | Low variance effect |
|----------|---------------|---------------|-------------------|---------------------|
| Variance-based | Trend and pattern | Feature selection | High score if linear, even if biased. | High risk. Scores collapse or turn noisy even when the error is small. |
| Mean-based | Absolute error | Model training as a loss | Zero only if exactly on the line. | Robust. Stable and interpretable regardless of variance. |
| Agreement-based | Fidelity and calibration | System validation | One only if exactly on the line. | Moderate risk. Sensitivity inherited from the correlation component. |

## 4. Recommendation For Engineering Teams

When deploying a model for semiconductor or sensor infrastructure, never rely on the variance
indices alone. In high-precision manufacturing, sensors often operate within a tight, stable
range, and in that regime Pearson and $R^2$ suggest the model is failing when it may in fact be
predicting within sub-micron accuracy.

- Use RMSE or MAE as the primary source of truth in low-variance environments.
- Use the agreement indices, CCC and KGE, for system-wide validation only when the data range is
  sufficient.
- Check the low variance effect before interpreting a drop in $R^2$. It is often a mathematical
  artifact rather than a loss of predictive power.

---

## Appendix A. Terminology

- Adjusted R²: 조정 결정계수. A coefficient of determination corrected for the number of
  predictors, so that adding a predictor does not raise the score on its own.
- Huber: 후버 손실. A loss that is squared for small residuals and linear for large ones, which
  limits the pull of an outlier without discarding it.
- Scale-dependent: 척도 종속. Carrying the unit of the measurement, so the value cannot be
  compared across quantities of different magnitude.
- Scale-independent: 척도 독립. Normalized so that the value is comparable across quantities of
  different magnitude.
- SMAPE: 대칭 평균 절대 백분율 오차. A percentage error that divides by the average of the
  observed and the predicted value, which removes the asymmetry of MAPE.
- Virtual metrology: 가상 계측. Predicting a measurement from process and sensor data instead of
  measuring it directly.
