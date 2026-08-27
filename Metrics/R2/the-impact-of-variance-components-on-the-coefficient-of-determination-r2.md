# The Impact of Variance Components on the Coefficient of Determination ($R^2$)
Rev. 8 | Created: 2026-04-08 | Updated: 2026-08-20 10:18 CDT
> A note on why $R^2$ moves with the variance of the residuals and with the variance of the
> predictor, and on reading it as a ratio rather than as an absolute measure of accuracy.

## 1. Executive Summary

The coefficient of determination, denoted as $R^2$, is one of the most widely used metrics for
assessing the goodness-of-fit in linear regression models. However, its interpretation is often
fraught with misunderstanding, particularly regarding how it fluctuates not just with the
correctness of a model, but with the underlying distribution of the data. This document explores
the mathematical and conceptual reasons why changes in variance — specifically residual variance
($\sigma^2_{\epsilon}$) and predictor variance ($\sigma^2_{x}$) — exert a profound influence on
$R^2$. By analyzing the ratio of variances, it demonstrates that $R^2$ is a relative measure of
power rather than an absolute measure of model accuracy.

## 2. Mathematical Definition Of $R^2$

To understand why variance dictates the behavior of $R^2$, it must first be defined through the
lens of analysis of variance (ANOVA). In a standard linear model
$Y = \beta_0 + \beta_1 X + \epsilon$, the total variation in the dependent variable $Y$ is
partitioned into two distinct components.

- Explained variation `SS_reg` is the variation accounted for by the relationship between $X$
  and $Y$.
- Unexplained variation `SS_res` is the variation resulting from the residuals, that is the
  noise $\epsilon$.

The fundamental identity is

$$SS_{tot} = SS_{reg} + SS_{res}$$

From this, $R^2$ is defined as the proportion of the total variance in $Y$ that is explained by
$X$.

$$R^2 = \frac{SS_{reg}}{SS_{tot}} = 1 - \frac{SS_{res}}{SS_{tot}}$$

The two sums of squares are the residual sum of squares
$SS_{res} = \sum (y_i - \hat{y}_i)^2$ and the total sum of squares
$SS_{tot} = \sum (y_i - \bar{y})^2$.

## 3. The Impact Of Increased Error Variance

The first scenario involves an increase in the variance of the residuals
($\sigma^2_{\epsilon}$), assuming the true relationship $\beta_1$ and the range of $X$ remain
constant.

### 3.1 The Mathematical Mechanism

As the noise in the data increases, each observed value $y_i$ deviates further from the
regression line $\hat{y}_i$. This directly inflates the $SS_{res}$ term. In the formula
$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$, as the numerator of the fraction increases, the entire
fraction $\frac{SS_{res}}{SS_{tot}}$ grows larger. Consequently, when this larger value is
subtracted from 1, the resulting $R^2$ decreases.

### 3.2 Conceptual Interpretation

In the context of information theory and machine learning, the relationship between $X$ and $Y$
is the signal and the residuals are the noise. When the error variance increases, the noise
overwhelms the signal. Even if the underlying model is correct, that is even if the true
$\beta_1$ has been identified, the predictive power is diluted.

> Increased noise or residual variance diminishes the explanatory power of the model, leading to
> a lower $R^2$.

This illustrates that a low $R^2$ does not necessarily mean the model is wrong. It may simply
mean the environment is inherently noisy, making the dependent variable difficult to predict
with high precision.

## 4. The Impact Of Increased Predictor Variance

A more counterintuitive phenomenon occurs when the variance of the independent variable $X$
changes. If the range of $X$ values is expanded, thereby increasing $\sigma^2_{x}$, the $R^2$
typically increases, even if the error variance $\sigma^2_{\epsilon}$ remains exactly the same.

### 4.1 The Expansion Of The Denominator

In a simple linear regression, the explained variance is expressed as

$$SS_{reg} = \beta_1^2 \cdot \sum (x_i - \bar{x})^2$$

When the variance of $X$ increases, $\sum (x_i - \bar{x})^2$ increases. This causes $SS_{reg}$
to grow. Since $SS_{tot} = SS_{reg} + SS_{res}$, and $SS_{res}$ is assumed constant, the
denominator $SS_{tot}$ grows primarily because the explained part is growing.

In the fraction $\frac{SS_{res}}{SS_{tot}}$, the denominator is getting larger while the
numerator stays the same. This makes the fraction smaller, and subtracting a smaller number from
1 results in a higher $R^2$.

### 4.2 The Strength Of The Trend

When $X$ is measured over a wider range, the overall trend, that is the slope, becomes more
dominant relative to the local fluctuations. The model captures a larger portion of the total
spread of $Y$ because that spread is now driven more by the change in $X$ than by the random
error.

> A wider range or higher variance in the independent variable often inflates the $R^2$, as the
> model captures a larger portion of the overall trend.

## 5. Summary Of Variance Effects On $R^2$

The following table summarizes the relationship between the variance components and the
resulting coefficient of determination.

Table 1. Variance components and their effect on $R^2$

| Scenario | Effect on $R^2$ | Statistical reason |
|----------|-----------------|--------------------|
| Higher residual variance ($\sigma^2_{\epsilon}$) | Decreases | The unexplained portion $SS_{res}$ of the data becomes a larger fraction of the total. |
| Higher predictor variance ($\sigma^2_{x}$) | Increases | The explained portion $SS_{reg}$ grows, making the noise relatively less significant. |
| Lower total variance ($SS_{tot}$) | Decreases | When the total spread of $Y$ is small, even minor errors lead to a low $R^2$. |

## 6. Practical Implications For Machine Learning Models

In machine learning, relying solely on $R^2$ is misleading because of these variance
dependencies.

- Model comparison is not straightforward. The $R^2$ of a model trained on a narrow dataset
  cannot easily be compared with one trained on a diverse, wide-ranging dataset, because the
  latter will likely have a higher $R^2$ simply due to the variance in $X$.
- High variance in $X$ sometimes masks poor model performance in specific sub-regions of the
  data, which is a route to overfitting that the single number does not reveal.
- Feature selection is essentially an attempt to increase the explained variance $SS_{reg}$ so
  that the relative weight of the residuals falls.

## 7. Conclusion

The reason changes in variance affect $R^2$ is that $R^2$ is a ratio. An absolute error measure
such as the mean squared error or the mean absolute error reports the size of the error itself,
and moving the spread of the data leaves it alone. $R^2$ reports the error relative to the total
spread, so it moves whenever either side of that ratio moves, which is what every scenario in
Table 1 sets out.

Understanding this dynamic prevents the common pitfall of dismissing a model with a low $R^2$ in
a high-noise environment, or over-trusting a model with a high $R^2$ derived from an artificially
wide range of independent variables.

## 8. Variation With Sample Distributions Along The 1-To-1 Line

The chart below sweeps the sigma score, that is the coefficient of variation `std / mean`, over
the range 0.1 to 4.0 for samples placed along the 1-to-1 line, and records the $R^2$ that each
sample yields. It shows the same dependence from the data side rather than from the algebra: the
value moves with the spread of the sample even though the underlying relationship never changes.

Fig 1. $R^2$ against the sigma score for samples placed along the 1-to-1 line

![Fig 1](the-impact-of-variance-components-on-the-coefficient-of-determination-r2_fig/sigma_r2.png)

The figure is produced by `sigma_r2.py`, in the folder of this document.

---

## Appendix A. Terminology

- Coefficient of determination ($R^2$): 결정계수. The proportion of variance in the dependent
  variable that is predictable from the independent variable.
- Explanatory power: 설명력. The capacity of a model to represent the underlying patterns in the
  data.
- Residual variance: 잔차 분산. The variance of the differences between observed and predicted
  values.
- Mean absolute error: 평균 절대 오차. The mean of the absolute differences between observed and
  predicted values, carrying the unit of the observations.
- Mean squared error: 평균 제곱 오차. The mean of the squared differences between observed and
  predicted values.
- Signal-to-noise ratio: 신호 대 잡음비. A measure that compares the level of a desired signal to
  the level of background noise.
