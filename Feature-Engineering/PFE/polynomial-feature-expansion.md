# Polynomial Feature Expansion
Rev. 62 | Created: 2026-09-09 | Updated: 2026-09-12 02:58 CDT

Polynomial feature expansion is the operation that builds both the powers of one variable and the products of distinct variables. This document covers modelling the non-linear behaviour of numeric tabular data with those two kinds of column.

## 1. Purpose

- **Problem Statement**: Numeric tabular data has cases that are hard to express with a linear model.
- **Goal**: To build a non-linear model by adding the power terms of the original variables ($x^2$, $x^3$) and the interaction terms ($x_1 \ast x_2$).
- **Non-Goal**: Derived variables are not covered, and the learning on the expanded data set is not covered.

## 2. Summary

An expansion computes products and powers from the columns already in the table and appends them as new columns, leaving the rows as they are and growing only the columns. A table with the columns $x_1$ and $x_2$ becomes a table with $x_1$, $x_2$, $x_1^2$, $x_1 x_2$, $x_2^2$, and those three new columns are what give a linear model a curve and an interaction between variables.

What it costs is the rising column count. Once the column count nears the row count the coefficients, the $\beta$ values that multiply the columns, can no longer be pinned to one solution (section 5.1).

To lessen that curse of dimensionality, the three below are set as the defaults. A default is what is kept until the data gives a reason to do otherwise, and together the three leave the column count well under the row count and keep those $\beta$ from moving far when the sample is drawn again.

- **Degree 2** — The terms built are limited to the square of one variable and the product of two (section 5.1).
- **Standardization** — Each column is brought to mean 0 and standard deviation 1 before and after the expansion. Subtracting the mean lowers the correlation between the columns and leaves the coefficients readable; dividing by the standard deviation removes the differences in column size, so the penalty falls evenly (sections 4.2, 4.3 and 5.2).
- **Penalty** — The expanded columns carry a ridge or a lasso. Ridge divides every $\beta$ by the same factor without ever reaching zero, and lasso sets the small ones to exactly zero (section 5.2).

Standardization and the penalty are the two that get skipped. In uncentered physical units the correlation between $x$ and $x^2$ is close to 1 (section 4.2), and the columns the expansion makes are not orthogonal to one another even where the raw variables are. So the failure of an expansion arrives not as a model that fits the data badly but as coefficients whose signs flip each time the sample is drawn again. Centering lowers that correlation without taking it to zero (section 4.2), so the sign flips do not go away on centering alone. What is left falls to the penalty, which keeps the columns that resemble one another from carrying large coefficients that cancel (section 5.2).

Where the expansion should not be used is equally clear. Past a few dozen variables the column count passes the sample count, a shape that bends several times inside one variable calls for a spline rather than a higher degree, and where prediction outside the training range is needed the way a polynomial diverges beyond that range, its extrapolation behaviour, is itself the risk.

## 3. Objective

An expansion is aimed at two things, a non-linear relationship inside one variable and an interaction between variables, neither of which a linear model expresses. The first is carried by a power term, the second by an interaction term, and both are put into the columns so that the model itself stays linear.

### 3.1 Power Term

The non-linear relationship inside one variable is carried by the powers of that variable, its power terms. Adding $x^2$ and $x^3$ to a variable $x$, the model learns equation (1) and draws a curve while staying linear in its coefficients.

$$\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 \hspace{19em} (1)$$

Linear here is about the coefficients $\beta$ rather than about $x$, which is why least squares, the solve that picks the coefficients minimizing the sum of the squared residuals, carries over as it is. The coefficient $\beta_2$ carries one bend, a peak or a saturation, and $\beta_3$ carries a second one. The non-linearity sits in the columns rather than in the model, so the linear model already in use, and the inference and the penalty built on it, are kept as they are.

### 3.2 Interaction Term

The interaction between variables is carried by the product of two distinct variables, an interaction term. Adding the product $x_1 x_2$ to the variables $x_1$ and $x_2$ gives equation (2).

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1 x_2 \hspace{19em} (2)$$

The product term lets one variable change the slope of another. Differentiating equation (2) with respect to $x_1$ gives equation (3), which says as much.

$$\frac{\partial \hat{y}}{\partial x_1} = \beta_1 + \beta_3 x_2 \hspace{19em} (3)$$

Where $\beta_3$ is not zero the effect of $x_1$ differs at each level of $x_2$. On a process it reads as the effect of pressure depending on the temperature, which the first-order terms of $x_1$ and $x_2$ alone cannot write. An effect that appears only when the two variables are high together is carried by this product term and nowhere else. The long practice of writing a response surface, the surface the response traces over the process conditions, as a second-order polynomial is curvature and interaction combined, and reading the optimum off the point where that surface has zero slope, its stationary point, came from there [[1](#ref-1)].

## 4. Mechanism

### 4.1 Expansion

The columns an expansion makes are monomials, terms formed by multiplying powers of the raw variables. With $n$ variables and a highest degree of $d$, the new columns are the set of equation (4).

$$\Phi_d(\mathbf{x}) = \left\lbrace \prod_{i=1}^{n} x_i^{a_i} \ \middle|\ a_i \in \mathbb{Z}_{\ge 0}, \ 1 \le \sum_{i=1}^{n} a_i \le d \right\rbrace \hspace{19em} (4)$$

For $[X_1, X_2]$ at $d = 2$ the columns, including the intercept that is a constant column of ones, are $[1, X_1, X_2, X_1^2, X_1 X_2, X_2^2]$, which holds the square terms of section 3.1 and the product term of section 3.2 together. Written for $n$ variables the second-order model is equation (5).

$$y = \beta_0 + \sum_{i=1}^{n} \beta_i x_i + \sum_{1 \le i \le j \le n} \beta_{ij} x_i x_j + \varepsilon \hspace{19em} (5)$$

### 4.2 Standardization

Bring each column to mean 0 and standard deviation 1 before expanding. This is standardization, and subtracting the mean alone is centering. The two parts do different work. Subtracting the mean lowers the correlation between the columns and leaves the coefficients readable, which is the two paragraphs below; dividing by the standard deviation removes the differences in column size, and that part is sections 4.3 and 5.2.

The first reason to center is the drop in correlation. A correlation is the covariance of two columns over the product of their standard deviations, so where the correlation comes from is read off the covariance in the numerator. Take the samples $x_1, \dots, x_N$, their mean $\overline{x}$, and the deviations $u_i = x_i - \overline{x}$, whose own mean is 0; an overline is the sample mean. The definition of the covariance, $\mathrm{Cov}(X, Y) = E[XY] - E[X]E[Y]$ (equation (16) of [Appendix B](#appendix-b-covariance)), applied at $X = x$ and $Y = x^2$ gives equation (6): the mean of the product is $\overline{x^3}$, and the product of the means is $\overline{x}$ times $\overline{x^2}$.

$$\mathrm{cov}(x, x^2) = \overline{x^3} - \overline{x} \overline{x^2} \hspace{19em} (6)$$

Substituting $x = u + \overline{x}$ writes both means in $u$, which is equation (7). Of the expanded terms, each one multiplied by $\overline{u}$ drops out, since $\overline{u} = 0$.

$$\overline{x^3} = \overline{u^3} + 3 \overline{x} \overline{u^2} + \overline{x}^3, \qquad \overline{x^2} = \overline{u^2} + \overline{x}^2 \hspace{19em} (7)$$

Putting equation (7) into equation (6) gives equation (8).

$$\mathrm{cov}(x, x^2) = \overline{u^3} + 3 \overline{x} \overline{u^2} + \overline{x}^3 - \overline{x} (\overline{u^2} + \overline{x}^2) \hspace{19em} (8)$$

The $\overline{x}^3$ cancels and $\overline{x} \overline{u^2}$ comes off $3 \overline{x} \overline{u^2}$, so the covariance closes as equation (9).

$$\mathrm{cov}(x, x^2) = 2 \overline{x} \overline{u^2} + \overline{u^3} \hspace{19em} (9)$$

The two terms of equation (9) come from different places. The first, $2 \overline{x} \overline{u^2}$, comes only from how far the mean sits from zero; the second, $\overline{u^3}$, only from how far the distribution leans to one side, its third central moment. Subtracting the mean takes the first term to 0 and leaves the second as it was.

Weighing the two terms against each other takes dividing the covariance by the standard deviations, that is, writing the correlation; why that division is needed and where the two equations below come from are in [Appendix C](#appendix-c-correlation-of-a-variable-and-its-square). With $t = \overline{x} / \sqrt{\overline{u^2}}$, $s = \overline{u^3} / (\overline{u^2})^{3/2}$ and $k = \overline{u^4} / (\overline{u^2})^2$, equation (9) is $(\overline{u^2})^{3/2} (2t + s)$, the variance of $x$ is $\overline{u^2}$ and the variance of $x^2$ is $(\overline{u^2})^2 (k - 1 + 4t^2 + 4ts)$, so the correlation is equation (10).

$$r(x, x^2) = \frac{2t + s}{\sqrt{k - 1 + 4t^2 + 4ts}} \hspace{19em} (10)$$

Centering takes $\overline{x}$ to 0 and so forces $t = 0$, which reduces equation (10) to equation (11).

$$r(u, u^2) = \frac{s}{\sqrt{k - 1}} \hspace{19em} (11)$$

The difference between the two equations is the ground for centering. Raise $\lvert t \rvert$ in equation (10) and the numerator approaches $2t$ while the denominator approaches $2 \lvert t \rvert$, so $\lvert r \rvert$ goes to 1. Since $t$ measures how many spreads the mean sits away from zero, it is large on data in physical units, which is why the correlation between $x$ and $x^2$ there reaches 1. Equation (11) carries no $t$. The correlation after centering is set by the shape alone, $s$ and $k$, independent of where the mean sits, and it is exactly 0 for a symmetric distribution, where $s = 0$.

The second reason to center is interpretation. On centered data $\beta_1$ is the slope while the other variables sit at their means, a readable quantity. Uncentered it is the slope while the other variables are zero, and that zero is often a point the data never visits [[2](#ref-2)].

Centering lowers the correlation, though, without removing it. The collinearity an expansion manufactures is a property of the expansion rather than of the data, so a penalty is needed alongside it (section 5.2).

### 4.3 Conditioning

Conditioning is how sensitive solving the design matrix is to a small error in the input, and the number that measures it is the condition number. The design matrix is the matrix whose rows are the observations and whose columns are the terms the model uses, from which the coefficients are solved. The condition number says by what factor such an error is magnified in the solution.

What raises the condition number is the degree and the collinearity between the columns; what lowers it is standardization. Subtracting the mean cuts the overlap between the columns and dividing by the standard deviation removes the differences in their size, so both parts are needed to take the condition number lowest.

### 4.4 Heredity

Keep a product term, and the main effects composing it stay as well. The rule is called heredity, and its ground is the coordinate system rather than statistics.

Substituting the shift $x_1 = z_1 + a$, $x_2 = z_2 + b$ into a product-only model such as $y = \beta_{12} x_1 x_2$ gives equation (12).

$$\beta_{12} (z_1 + a)(z_2 + b) = \beta_{12} z_1 z_2 + \beta_{12} b z_1 + \beta_{12} a z_2 + \beta_{12} ab \hspace{19em} (12)$$

Main effects appear on their own. A product model without main effects therefore depends on where the origin was placed, and whether temperature is measured in Celsius or in kelvin changes the model. Keep the main effects and that shift is absorbed as a rearrangement of the coefficients. There is a practice of dropping a main effect on the weak form of the rule, weak heredity, under which only one of the variables forming the product need be present, but the conditions that justify it almost never hold in practice [[4](#ref-4)]. Where variable selection is automated it is likewise better to carry heredity as a Bayesian prior or as a constraint on the optimization [[5](#ref-5)] [[6](#ref-6)].

## 5. Caution

An expansion charges two prices. The column count grows fast, which invites overfitting — a model that fits the training data and misses new data — and raises the cost of the fit. The columns it makes also resemble one another, which unsettles the coefficients. The degree holds the first price down, a penalty holds the second.

### 5.1 Dimensionality And Overfitting

The column count grows as the $d$-th power of the variable count. Covering the space those columns span at one density takes exponentially more observations as their number grows, which is the curse of dimensionality, and an expansion walks into it by adding columns to data whose row count does not move. Without the intercept, the full expansion has the column count of equation (13), and `interaction_only`, which keeps only products of distinct variables, has that of equation (14).

$$p_{\mathrm{full}} = \binom{n+d}{d} - 1 \hspace{19em} (13)$$

$$p_{\mathrm{inter}} = \sum_{j=1}^{\min(d,\ n)} \binom{n}{j} \hspace{19em} (14)$$

Both counts are derived from the set of equation (4) in [Appendix C](#appendix-d-term-count-derivation).

Table 1. Column count after expansion, bias column excluded

| Variables | Degree 2, full | Degree 2, interaction only | Degree 3, full | Degree 3, interaction only |
| --- | --- | --- | --- | --- |
| 5 | 20 | 15 | 55 | 25 |
| 10 | 65 | 55 | 285 | 175 |
| 20 | 230 | 210 | 1,770 | 1,350 |
| 50 | 1,325 | 1,275 | 23,425 | 20,875 |
| 100 | 5,150 | 5,050 | 176,850 | 166,750 |

What Table 1 says is that `interaction_only` saves little. At $d = 2$ the difference is the $n$ square terms alone, so 5,150 becomes 5,050 at $n = 100$. The option is therefore not switched on to cut the column count; it is where the decision to keep curvature inside one variable out of the model is written down.

What actually sets the column count is the degree. Raising $d$ from 2 to 3 takes the columns from 230 to 1,770 at $n = 20$. As the column count approaches the row count the least-squares solution turns unstable, and past it the solution is not unique, so the ceiling on an expansion is set by the sample count rather than by the degree.

The degree is therefore chosen on the error over data kept out of the fit, the held-out error, rather than on theory, and the candidates are few. It is 2 in almost every practical case, data that needs 3 is rare, and a degree of 4 or more that appears to win is a sign that something other than an expansion should be used.

<img src="polynomial-feature-expansion_fig/fig1.png" width="1100" style="max-width: 100%;" alt="Fig 1">

Fig 1. Degree and extrapolation, conditioning, and the cost of expansion

Fig 1(a) is the first reason. Degrees 2, 5 and 9 are fitted to 60 samples; inside the training range (grey) degrees 5 and 9 are both plausible, and outside it the higher degree diverges first. The behaviour of a polynomial beyond its range is governed by its top term, so raising the degree where extrapolation is needed improves the fit inside the training range while the error outside it grows.

Fig 1(b) draws the condition numbers of section 4.3 against degree, and Fig 1(c) is the relation between the term count and the row count. On data with 5 variables, 60 rows and a true model holding one product term, the held-out RMSE, that error measured as the root of the mean squared error, falls from 1.34 at degree 1 to 0.34 at degree 2 and returns to 1.08 at degree 3. The 55 columns of degree 3 nearly reach the 60 rows. Ridge is at 0.75 in the same place, stopping close to half of that deterioration.

### 5.2 Regularization

The expanded columns always carry a penalty, a term added to the fitting criterion that charges for the size of the coefficients. The expansion raises the column count and at the same time makes columns that resemble one another, and unpenalized least squares absorbs that resemblance into two large coefficients that cancel, which is why the fit moves far on a small disturbance of the data. Ridge answers it with a penalty proportional to the sum of the squared coefficients, which is what stops the two large coefficients from cancelling [[3](#ref-3)].

Ridge is the default of the two. It divides the coefficient among the columns that resemble one another and steadies the prediction, while lasso keeps one of them and drops the rest. Lasso on expanded columns can keep a product term while deleting its main effects, breaking the heredity of section 4.4, so it is used with a hierarchical constraint rather than on its own [[6](#ref-6)].

That ridge never drives a coefficient to zero means no column can be dropped with it. It is still the default because what a penalty buys on an expansion is not a smaller column count but a steadier prediction, and the size of that is the held-out RMSE of section 5.1 falling from 1.08 to 0.75 at degree 3. Where the column count itself has to come down, that is the work of lasso or elastic net.

The penalty acts on the size of a column, so it is applied after the expanded columns are standardized, which is what puts the second standardization into the pipeline of [Appendix E](#appendix-f-implementation). The objectives of the three penalties, and how far each of them moves a coefficient, are in [Appendix D](#appendix-e-ridge-and-lasso-on-expanded-columns).

### 5.3 Failure Modes

An expansion fails in six recognizable ways. Most arrive not as a model that fits badly but as coefficients or predictions that turn unstable.

Table 2. Failure modes of a polynomial expansion

| Symptom | Cause | Countermeasure |
| --- | --- | --- |
| Held-out error worse at degree 2 than at degree 1 | Term count close to the row count | Ridge or lasso, `interaction_only`, selective expansion |
| Coefficient signs flipping across resamples | Collinearity manufactured by the expansion | Centering, a penalty, reading predictions instead of coefficients |
| Prediction diverging just outside the training range | Extrapolation behaviour of a polynomial | Spline, a range guard on the input, no extrapolation |
| A handful of rows dominating the fit | Squares amplifying leverage | Outlier handling before expansion, robust loss |
| Duplicate or all-zero columns | Dummy columns squared and crossed | `interaction_only=True`, expansion restricted to continuous columns |
| Imputed values amplified | Imputation error squared inside a product | Imputation before expansion, an indicator column for what was imputed |

The fifth row of Table 2 is written out separately because the expansion does not catch it on its own. A categorical variable is turned into numbers by giving each category a column that holds 1 where the row falls in that category and 0 otherwise, a dummy. A dummy squared is itself and becomes an exactly duplicated column, and the product of two dummies from the same categorical variable is always zero, since one row cannot fall in two categories at once. The expansion knows none of this, so columns coming from a categorical variable are either left out of the expansion or handled with `interaction_only`.

### 5.4 Diagnostics

Whether an expansion helped is confirmed in four ways.

- The held-out error curve drawn while raising the degree from 1. Watch whether the minimum stays at 2 or below.
- The condition number of the expanded design matrix and the VIF (Variance Inflation Factor) of each column. A large value after centering calls for a penalty.
- The share of samples redrawn from the data with replacement, the bootstrap, in which a coefficient keeps its sign. A product term whose sign flips is not interpreted.
- The residual plotted against the product terms. Confirm that the structure left before the expansion is gone.

## 6. Further Work

- **Sparse polynomial chaos expansion** — A way to cut a high-degree expansion down to a size that can be carried, by selecting terms sparsely over a basis of mutually orthogonal polynomials [[10](#ref-10)]. Selecting the terms by least angle regression has settled into a procedure, which makes keeping a few dozen out of several hundred candidates computationally practical. Starting needs a distributional assumption on the input variables (the basis follows that distribution) and a designed sample.
- **Hierarchical interaction selection at scale** — The lasso family that selects product terms with heredity imposed as a convex constraint [[6](#ref-6)]. Convex means the optimum found is the only one, and it solves up to several hundred variables, so the rule of section 4.4 can be enforced by the optimization rather than by a person. Starting needs a rule that narrows the candidate product terms in advance and a computational budget.
- **Learned basis** — A model that stacks one-dimensional functions learned from the data in place of a fixed monomial basis [[11](#ref-11)]. A spline-based implementation was released in 2024, which makes a direct comparison with an expansion plus ridge on the same data possible. Starting needs a held-out comparison procedure and a criterion for whether the learned basis is excessive for the sample count.

## References

<a id="ref-1"></a>
[1] Box, G. E. P. and Wilson, K. B. (1951). [On the Experimental Attainment of Optimum Conditions](https://doi.org/10.1111/j.2517-6161.1951.tb00067.x). *Journal of the Royal Statistical Society: Series B*, 13(1), 1–38.<br>
<a id="ref-2"></a>
[2] Marquardt, D. W. (1980). [Comment: You Should Standardize the Predictor Variables in Your Regression Models](https://doi.org/10.1080/01621459.1980.10477430). *Journal of the American Statistical Association*, 75(369), 87–91.<br>
<a id="ref-3"></a>
[3] Hoerl, A. E. and Kennard, R. W. (1970). [Ridge Regression: Biased Estimation for Nonorthogonal Problems](https://doi.org/10.1080/00401706.1970.10488634). *Technometrics*, 12(1), 55–67.<br>
<a id="ref-4"></a>
[4] Nelder, J. A. (1998). [The Selection of Terms in Response-Surface Models—How Strong is the Weak-Heredity Principle?](https://doi.org/10.1080/00031305.1998.10480588) *The American Statistician*, 52(4), 315–318.<br>
<a id="ref-5"></a>
[5] Chipman, H. (1996). [Bayesian variable selection with related predictors](https://doi.org/10.2307/3315687). *The Canadian Journal of Statistics*, 24(1), 17–36.<br>
<a id="ref-6"></a>
[6] Bien, J., Taylor, J. and Tibshirani, R. (2013). [A lasso for hierarchical interactions](https://doi.org/10.1214/13-AOS1096). *The Annals of Statistics*, 41(3), 1111–1141.<br>
<a id="ref-7"></a>
[7] Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M. and Duchesnay, É. (2011). [Scikit-learn: Machine Learning in Python](https://www.jmlr.org/papers/v12/pedregosa11a.html). *Journal of Machine Learning Research*, 12, 2825–2830.<br>
<a id="ref-8"></a>
[8] Rahimi, A. and Recht, B. (2007). [Random Features for Large-Scale Kernel Machines](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html). *Advances in Neural Information Processing Systems*, 20.<br>
<a id="ref-9"></a>
[9] Friedman, J. H. and Popescu, B. E. (2008). [Predictive learning via rule ensembles](https://doi.org/10.1214/07-AOAS148). *The Annals of Applied Statistics*, 2(3), 916–954.<br>
<a id="ref-10"></a>
[10] Blatman, G. and Sudret, B. (2011). [Adaptive sparse polynomial chaos expansion based on least angle regression](https://doi.org/10.1016/j.jcp.2010.12.021). *Journal of Computational Physics*, 230(6), 2345–2367.<br>
<a id="ref-11"></a>
[11] Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y. and Tegmark, M. (2024). [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756). *arXiv:2404.19756*.<br>
<a id="ref-12"></a>
[12] Tibshirani, R. (1996). [Regression Shrinkage and Selection via the Lasso](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x). *Journal of the Royal Statistical Society: Series B*, 58(1), 267–288.<br>
<a id="ref-13"></a>
[13] Zou, H. and Hastie, T. (2005). [Regularization and variable selection via the elastic net](https://doi.org/10.1111/j.1467-9868.2005.00503.x). *Journal of the Royal Statistical Society: Series B*, 67(2), 301–320.

---

## Appendix A. Terminology

- **collinearity**: The state in which two or more columns point in nearly the same direction, so that their coefficients cannot be estimated apart.
- **condition number**: The ratio of the largest singular value of a matrix to the smallest. It says how far a small error in the input is magnified in the solution.
- **curse of dimensionality**: The exponential growth, as the number of columns rises, in the number of observations needed to cover the space at one density.
- **degree**: The highest degree of a monomial the expansion admits. The degree of $X_1^2 X_2$ is 3.
- **derived variable**: A variable built from two or more columns by domain knowledge, such as a ratio or a rate.
- **design matrix**: The matrix whose rows are the observations and whose columns are the terms the model uses. The coefficients come from solving it.
- **dummy**: A column holding 1 where a row falls in one category of a categorical variable and 0 otherwise.
- **extrapolation**: Prediction over an input range the training data does not cover.
- **held-out**: Data kept out of the fit and used only to measure the error of the fitted model.
- **heredity**: The rule that a product term put into a model brings the lower-degree terms composing it with it.
- **leverage**: How strongly one observation pulls its own fitted value. It grows as the input sits further from the centre.
- **main effect**: The first-order term of a single variable, $\beta_i x_i$.
- **monomial**: A term formed by multiplying powers of the variables. $X_1^2 X_2$ is one.
- **overfitting**: The state in which a model fits the training data while missing new data.
- **penalty**: A term added to the fitting criterion that charges for the size of the coefficients, as ridge and lasso do.
- **RMSE**: The square root of the mean squared error (Root Mean Squared Error).
- **standardization**: Subtracting from each column its own mean and dividing by its standard deviation, bringing it to mean 0 and standard deviation 1.
- **VIF**: The variance inflation factor, computed from the $R^2$ of one column regressed on the rest. It is $1/(1-R^2)$.

## Appendix B. Covariance

A covariance measures how far two columns move together. Its definition on a population is equation (15), where $E[\cdot]$ is the expected value and $\mu_X$ and $\mu_Y$ are the expected values of the two variables.

$$\mathrm{Cov}(X, Y) = \sigma_{XY} = E[(X - \mu_X)(Y - \mu_Y)] \hspace{19em} (15)$$

Multiplying out under the properties of the expectation gives equation (16), the form that is easier to compute: the expected value of the product, less the product of the expected values.

$$\mathrm{Cov}(X, Y) = E[XY] - E[X]E[Y] \hspace{19em} (16)$$

Measured on a sample of $n$ observations the covariance is equation (17), where $x_i$ and $y_i$ are the $i$-th observation, $\overline{x}$ and $\overline{y}$ are the sample means, and the division by $n - 1$ drops one degree of freedom so that the population value is estimated without bias, as an unbiased estimator.

$$s_{XY} = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \overline{x})(y_i - \overline{y}) \hspace{19em} (17)$$

The sign, and the covariance of a variable with itself, say four things.

- At $\mathrm{Cov}(X, Y) \gt 0$, $Y$ rises as $X$ rises.
- At $\mathrm{Cov}(X, Y) \lt 0$, $Y$ falls as $X$ rises.
- At $\mathrm{Cov}(X, Y) = 0$ there is no linear relation between the two.
- $\mathrm{Cov}(X, X)$ is the variance $\mathrm{Var}(X)$.

Section 4.2 and the derivation in [Appendix C](#appendix-c-correlation-of-a-variable-and-its-square) use the mean divided by $1/N$. A correlation is a covariance over two standard deviations, and the same divisor appears above and below, so the correlation is the same whether $1/N$ or $1/(n-1)$ is used.

## Appendix C. Correlation Of A Variable And Its Square

Section 4.2 divides the covariance by the two standard deviations to write the correlation, and states that correlation as equation (10) and equation (11). Why the division is needed, and where the two equations come from, is below.

For two variables $X$ and $Y$ the correlation is defined as equation (18).

$$r(X, Y) = \frac{\mathrm{Cov}(X, Y)}{\mathrm{sd}(X) \cdot \mathrm{sd}(Y)} \hspace{19em} (18)$$

What this appendix solves is the case $X = x$ and $Y = x^2$. The numerator $\mathrm{Cov}(X, Y)$ is then equation (9) of section 4.2, and the two standard deviations in the denominator are $\mathrm{sd}(X) = \mathrm{sd}(x)$ and $\mathrm{sd}(Y) = \mathrm{sd}(x^2)$.

The reason to divide is units. Scaling a column $x$ by $c \gt 0$ gives $\mathrm{cov}(cx, (cx)^2) = c^3 \mathrm{cov}(x, x^2)$, so the size of a covariance moves with a change of units alone and cannot weigh the two terms against each other. Dividing by the standard deviations, $\mathrm{sd}(cx) = c \cdot \mathrm{sd}(x)$ and $\mathrm{sd}((cx)^2) = c^2 \cdot \mathrm{sd}(x^2)$ cancel that $c^3$, which is the left half of equation (19), and the Cauchy–Schwarz inequality holds the value inside $[-1, 1]$, which is the right half.

$$r(cx, (cx)^2) = r(x, x^2), \qquad \lvert r(x, x^2) \rvert \le 1 \hspace{12em} (19)$$

What is left is the two standard deviations in the denominator. $\mathrm{var}(x) = \overline{u^2}$ is the definition itself. The variance of $x^2$ is equation (16) at $X = Y = x^2$, that is $\overline{x^4} - (\overline{x^2})^2$, and substituting $x = u + \overline{x}$ into both means and reducing by $\overline{u} = 0$ gives equation (20).

$$\mathrm{var}(x^2) = \overline{x^4} - (\overline{x^2})^2 = \overline{u^4} - (\overline{u^2})^2 + 4 \overline{x}^2 \overline{u^2} + 4 \overline{x} \overline{u^3} \hspace{6em} (20)$$

The three values substituted below, $t$, $s$ and $k$, are there to gather what centering changes into one place. With the moments and $\overline{x}$ mixed together it is not visible where the location of the mean enters the correlation, but with the location held in $t$ alone and the shape in $s$ and $k$, the equation shows directly that centering takes $t$ to 0 and leaves $s$ and $k$ as they were. All three are divided by the spread and so carry no units, which makes them the same under a change of units in the data. The two are the usual quantities: $s$ is the skewness and $k$ the kurtosis.

Substituting $t = \overline{x} / \sqrt{\overline{u^2}}$, $s = \overline{u^3} / (\overline{u^2})^{3/2}$ and $k = \overline{u^4} / (\overline{u^2})^2$ writes that numerator and those two denominators as equation (21).

$$\mathrm{cov}(x, x^2) = (\overline{u^2})^{3/2} (2t + s), \quad \mathrm{sd}(x) = (\overline{u^2})^{1/2}, \quad \mathrm{sd}(x^2) = \overline{u^2} \sqrt{k - 1 + 4t^2 + 4ts} \hspace{2em} (21)$$

Put the three into equation (18) and $(\overline{u^2})^{3/2}$ cancels, so equation (10) of section 4.2 is what is left.

$$r(x, x^2) = \frac{2t + s}{\sqrt{k - 1 + 4t^2 + 4ts}} \hspace{19em} (10)$$

Both $s$ and $k$ are written in the deviations $u$ alone, which centering does not change, and centering only makes $\overline{x} = 0$, that is $t = 0$, so equation (11) is equation (10) at $t = 0$.

$$r(u, u^2) = \frac{s}{\sqrt{k - 1}} \hspace{19em} (11)$$

Raise $\lvert t \rvert$ and the denominator, $2 \lvert t \rvert \sqrt{1 + s / t + (k - 1) / (4t^2)}$, approaches $2 \lvert t \rvert$ while the numerator approaches $2t$, so $\lvert r \rvert$ goes to 1.

## Appendix D. Term Count Derivation

Set notation comes first. A set is written either by listing its elements, as in $\lbrace 2, 4, 6 \rbrace$, or by a condition, in the form $\lbrace \cdot \mid \cdot \rbrace$. In that second form a vertical bar splits the braces: left of the bar stands the shape an element takes, right of it the condition that shape has to meet. So $\lbrace n^2 \mid n \in \mathbb{Z}, \ 1 \le n \le 3 \rbrace$ reads as every $n^2$ for $n$ an integer from 1 to 3, which is the set $\lbrace 1, 4, 9 \rbrace$. A colon is used in place of the bar as often as not, and this document uses both.

Equation (4), from section 4.1, is repeated here.

$$\Phi_d(\mathbf{x}) = \left\lbrace \prod_{i=1}^{n} x_i^{a_i} \ \middle|\ a_i \in \mathbb{Z}_{\ge 0}, \ 1 \le \sum_{i=1}^{n} a_i \le d \right\rbrace \hspace{19em} (4)$$

Equation (4) is dense in notation but simple to read. On the left, $\Phi_d(\mathbf{x})$ is the collection of new columns built from one set of variable values $\mathbf{x} = (x_1, \dots, x_n)$. Left of the bar, $\prod_{i=1}^{n} x_i^{a_i}$ is each variable $x_i$ raised to $a_i$ and all of them multiplied together, which is one monomial. Each exponent $a_i$ is a non-negative integer, written $a_i \in \mathbb{Z}_{\ge 0}$, and where it is 0 that variable drops out of the product. The sum of the exponents $\sum_i a_i$ is the degree of the term, so the condition $1 \le \sum_i a_i \le d$ excludes the constant term, whose exponents sum to 0, and admits degrees up to $d$.

With two variables and $d = 2$, five pairs of exponents meet that condition. Table 3 is the five.

Table 3. Exponent pairs admitted by equation (4) at two variables and degree 2

| Exponent of $x_1$ | Exponent of $x_2$ | Degree | Term |
| --- | --- | --- | --- |
| 1 | 0 | 1 | $x_1$ |
| 0 | 1 | 1 | $x_2$ |
| 2 | 0 | 2 | $x_1^2$ |
| 1 | 1 | 2 | $x_1 x_2$ |
| 0 | 2 | 2 | $x_2^2$ |

The one pair left out is $(0, 0)$, the constant term.

Equation (4) defines the set of columns to be built without saying how large it is. That size is equation (13) and equation (14), derived below.

One monomial of degree exactly $k$ corresponds to one choice of non-negative integer exponents $(a_1, \dots, a_n)$ summing to $k$, so counting the monomials of that degree is counting those choices. That count is equation (22), whose left side carries a pair of bars $\lvert \cdot \rvert$ for the number of elements in the set they enclose.

$$\left| \lbrace (a_1, \dots, a_n) : a_i \in \mathbb{Z}_{\ge 0}, \ \sum_{i=1}^{n} a_i = k \rbrace \right| = \binom{k+n-1}{n-1} \hspace{19em} (22)$$

The count itself is stars and bars. Take the degree $k$ as $k$ identical stars, and the $n$ variables as $n$ bins separated by $n-1$ bars, so that the stars falling in a bin are the exponent $a_i$ of that variable. Counting the exponent choices is then laying $k$ stars and $n-1$ bars, $k+n-1$ symbols, in a row and choosing which $n-1$ positions carry the bars, which is $\binom{k+n-1}{n-1}$.

At $n = 2$ and $k = 2$ that is $\binom{3}{1} = 3$, and the arrangements $\ast\ast\mid$, $\ast\mid\ast$, $\mid\ast\ast$ read as the exponents $(2, 0)$, $(1, 1)$, $(0, 2)$ — the three degree-2 terms $x_1^2$, $x_1 x_2$, $x_2^2$ of Table 3.

Summing the degrees from 0 to $d$ gives equation (23). Writing it with one slack exponent $a_0 \ge 0$ such that $a_0 + \sum_i a_i = d$ collapses the sum into a single count, that of $d$ items falling into $n+1$ bins.

$$\sum_{k=0}^{d} \binom{k+n-1}{n-1} = \binom{n+d}{d} \hspace{19em} (23)$$

The set of equation (4) excludes the constant term at $k = 0$, so its size is $\binom{n+d}{d} - 1$, which is equation (13).

With `interaction_only` no variable is used twice, so a surviving term corresponds to one subset of the $n$ variables of size $j$, where $j$ runs from 1 to $\min(d, n)$. Adding those counts is equation (14). Once $d \ge n$ every subset is admitted and the sum closes as equation (24).

$$\sum_{j=1}^{n} \binom{n}{j} = 2^n - 1 \hspace{19em} (24)$$

## Appendix E. Ridge And Lasso On Expanded Columns

The penalty on the expanded columns is one of three. Written as an objective, ridge is equation (25) and lasso is equation (26) [[12](#ref-12)], where $\alpha$ sets how hard the penalty presses.

$$\hat{\boldsymbol{\beta}}_{\mathrm{ridge}} = \arg\min_{\boldsymbol{\beta}} \lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2^2 + \alpha \lVert \boldsymbol{\beta} \rVert_2^2 \hspace{15em} (25)$$

$$\hat{\boldsymbol{\beta}}_{\mathrm{lasso}} = \arg\min_{\boldsymbol{\beta}} \lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2^2 + \alpha \lVert \boldsymbol{\beta} \rVert_1 \hspace{15em} (26)$$

The shape of the penalty is the whole difference. Where the columns are standardized and orthogonal the two solutions close in equation (27): ridge divides every coefficient by the same factor and never reaches zero, while lasso sets to exactly zero every coefficient smaller in size than $\alpha / 2$ and pulls the rest toward zero by that amount.

$$\hat{\beta}_j^{\mathrm{ridge}} = \frac{\hat{\beta}_j^{\mathrm{ols}}}{1 + \alpha}, \qquad \hat{\beta}_j^{\mathrm{lasso}} = \mathrm{sign}(\hat{\beta}_j^{\mathrm{ols}}) \max \left( \lvert \hat{\beta}_j^{\mathrm{ols}} \rvert - \frac{\alpha}{2}, \ 0 \right) \hspace{9em} (27)$$

Expanded columns are far from orthogonal (section 4.2) and come in groups that resemble one another. Ridge spreads one coefficient across such a group; lasso keeps one member and zeroes the rest, and which member survives changes with the sample, so the list of terms lasso returns is itself unstable. Elastic net, equation (28) [[13](#ref-13)], mixes the two by $\rho$, which is lasso at 1 and ridge at 0. Its quadratic part keeps a group in or out together, so terms are still selected while the list moves less.

$$\hat{\boldsymbol{\beta}}_{\mathrm{enet}} = \arg\min_{\boldsymbol{\beta}} \lVert \mathbf{y} - \mathbf{X}\boldsymbol{\beta} \rVert_2^2 + \alpha \left( \rho \lVert \boldsymbol{\beta} \rVert_1 + \frac{1 - \rho}{2} \lVert \boldsymbol{\beta} \rVert_2^2 \right) \hspace{9em} (28)$$

Table 4. Penalties on expanded columns

| Penalty | Term added | A group of columns that resemble one another | Where it fits |
| --- | --- | --- | --- |
| Ridge | Sum of the squared coefficients | Coefficient shared across the group | The default on expanded columns |
| Lasso | Sum of the absolute coefficients | One kept, the rest at zero | A short term list, under a heredity constraint |
| Elastic net | Both, mixed by $\rho$ | Kept or dropped together | Selection wanted with a list that holds |

$\alpha$ is chosen on held-out error over candidates spaced by powers of ten, and it is meaningful only on standardized columns (section 5.2), which is what `RidgeCV`, `LassoCV` and `ElasticNetCV` search over. The intercept is left out of the penalty: penalizing it pulls the fitted level toward zero and moves the model off the centre of the data.

A penalty does not buy a degree of 4. What it buys is the difference between a fit that survives a column count close to the row count and one that does not, and section 5.1 gives the size of that difference.

## Appendix F. Implementation

### F.1 Options

The expansion itself is one line of `sklearn.preprocessing.PolynomialFeatures`, and only four arguments have to be settled [[7](#ref-7)].

Table 5. PolynomialFeatures arguments

| Argument | Effect | Note |
| --- | --- | --- |
| `degree` | Highest degree of the monomials | A `(min, max)` tuple for the lowest degree as well, so `(2, 2)` for second-order terms only |
| `interaction_only` | Products of distinct variables only | First-order terms kept, powers of a single variable dropped |
| `include_bias` | A constant column of ones | False where the estimator carries its own intercept |
| `order` | Memory layout of the output array | 'C' or 'F', a choice of layout rather than of content |

```python
# Python
from sklearn.preprocessing import PolynomialFeatures

# interaction_only=True keeps X1*X2 and drops X1^2, X2^2
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_expanded = poly.fit_transform(X)
term_name = poly.get_feature_names_out()
```

The names `get_feature_names_out()` returns are the only route from a coefficient back to its column. Lose them after the expansion and the coefficients remain while which product each belongs to cannot be said.

### F.2 Pipeline

An expansion is not used alone but placed between standardization and the penalized fit. The order is standardize the raw variables, expand, standardize the expanded columns again, then fit with a penalty.

The first standardization removes the conditioning problem of section 4.3, and the second makes the penalty fall evenly across the columns. The variance of a product term is close to the product of the raw variances and so differs widely from column to column; without the second standardization a ridge penalty lands almost entirely on the columns with the largest variance.

```python
# Python
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

pipeline = Pipeline([
    ('raw_scale', StandardScaler()),
    ('expand', PolynomialFeatures(include_bias=False)),
    ('term_scale', StandardScaler()),
    ('fit', Ridge()),
])
grid = {'expand__degree': [1, 2, 3], 'fit__alpha': [0.1, 1.0, 10.0, 100.0]}
search = GridSearchCV(pipeline, grid, scoring='neg_root_mean_squared_error', cv=5)
search.fit(X, y)
```

Putting the expansion inside the pipeline is not a convenience. The expansion is row-wise and leaks nothing by itself, but the standardizations on either side of it must take their means and variances from the training part of each fold, the pieces cross-validation splits the data into. Choosing the degree and the penalty together also finishes in one search only inside the pipeline.

### F.3 Cost

The cost of an expansion is linear in the column count, and that count grows by equation (13). At 100,000 rows, 100 variables and $d = 2$ the columns number 5,150, and holding them in a dense matrix, one that stores every value, takes 4.1 GB at 64 bits a value. Two routes keep the expanded columns out of memory.

The first is the kernel. The polynomial kernel of equation (29) computes the inner product of the expanded space without the expansion.

$$K(\mathbf{x}, \mathbf{z}) = (\gamma \mathbf{x}^{\top} \mathbf{z} + c)^{d} \hspace{19em} (29)$$

`KernelRidge(kernel='poly')` is that form, and since the cost falls on rows rather than on columns it suits data with many variables and few rows. What it costs is interpretation: no coefficient attaches to an individual monomial, so which product contributed cannot be read.

The second is approximation. `PolynomialCountSketch` compresses the terms a polynomial kernel uses into a fixed number of columns, a sketch, and `Nystroem` approximates the kernel matrix from a subset of the samples. Both belong to the family that approximates a kernel with a finite number of columns to keep the speed of a linear model [[8](#ref-8)], and both bound the column count at a value the user sets.

Sparse input is taken as it is. Feed in a CSR matrix, which stores only the non-zero values, and the expansion comes back in the same form, so data carrying many dummy columns does not inflate into a dense array.

### F.4 Selective Expansion

Not every pair has to be built. Hand the expansion the columns to be crossed and the column count ends at the number chosen rather than at Table 1.

```python
# Python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PolynomialFeatures

expand_column = ['temperature', 'pressure']
transformer = ColumnTransformer(
    [('expand', PolynomialFeatures(degree=2, include_bias=False), expand_column)],
    remainder='passthrough',
)
```

There are three grounds for choosing: an interaction the process already knows, a residual that shows structure over a combination of two variables, and a tree ensemble, several trees combined into one model, run first to measure interaction strength so that only the leading pairs are kept [[9](#ref-9)]. With none of the three, a penalty on the full expansion is the better move. Choosing pairs without grounds lets the analyst rather than the data decide which interactions reach the model.
