# Polynomial Feature Expansion
Rev. 4 | Created: 2026-09-09 | Updated: 2026-09-09 21:37 UTC

## 1. Purpose

- **Problem Statement**: A linear model draws its boundary only as a straight line or a hyperplane. A response that appears when two conditions are high together, or one that turns over past a peak, is not expressed however well the coefficients are estimated, and what is missing stays in the structure of the residual, where even the fact that the model is wrong does not surface.
- **Goal**: To carry a non-linear relationship and a feature interaction into a linear model by adding the products and powers of the raw variables as new columns, so that the two prices it charges, a column count that grows with the degree and coefficients that a penalty has to steady, can be controlled.
- **Non-Goal**: The learning algorithm placed on the expanded columns is not covered. Neither is the encoding of categorical variables nor the imputation of missing values.

## 2. Summary

The data is a table in which one row is one observation and one column is one variable. An expansion computes products and powers from the columns already in that table and appends them as new columns, leaving the rows as they are and growing only the columns. A table with the columns $x_1$ and $x_2$ becomes a table with $x_1$, $x_2$, $x_1^2$, $x_1 x_2$, $x_2^2$, and those three new columns are what give a linear model a curve and an interaction between variables.

What it costs is columns. Expanding 20 variables to the second degree takes the column count from 20 to 230, and once the column count approaches the row count the coefficients can no longer be estimated (section 5.1). Three defaults hold the column count, and the unsteadiness of the coefficients, inside what the data carries. First, degree 2 limits the terms built to squares and to products of two variables (section 5.1). Second, before the expansion each variable has its own mean subtracted so that its values sit near zero, which is centering (section 4.2). Third, the expanded columns carry a ridge or lasso penalty that holds the coefficients down (section 5.2).

Centering and the penalty are the two that get skipped. In uncentered physical units the correlation between $x$ and $x^2$ is close to 1 (section 4.2), and the columns the expansion makes are not orthogonal to one another even where the raw variables are. So the failure of an expansion arrives not as a model that fits the data badly but as coefficients whose signs flip each time the sample is drawn again.

Where the expansion should not be used is equally clear. Past a few dozen variables the column count passes the sample count, a shape that bends several times inside one variable calls for a spline rather than a higher degree, and where prediction outside the training range is needed the way a polynomial diverges beyond that range, its extrapolation behaviour, is itself the risk. Table 1 sets out that fork.

Table 1. Default choices and when they change

| # | Condition | Choice | Why |
| --- | --- | --- | --- |
| 1 | Fewer than a few dozen variables, curvature and pairwise effects expected | Degree 2, centered, ridge | A term count below the row count |
| 2 | Curvature judged absent, only cross effects wanted | `interaction_only=True` | Squares dropped as a modelling decision, not as a saving |
| 3 | Many variables, few rows | Polynomial kernel or a sketch | Cost on rows rather than on columns |
| 4 | Repeated bends inside one variable | Spline or GAM | Local basis instead of a higher degree |
| 5 | Prediction outside the training range | Neither expansion nor a high degree | A polynomial governed by its top term outside the range |

## 3. Objective

The expansion is aimed at two things, a non-linear relationship inside one variable and an interaction between variables. A linear model expresses neither, and the expansion puts both into the columns so that the model itself stays linear.

### 3.1 Non-linear Relationship

Adding terms such as $x^2$ and $x^3$ to a raw variable $x$ lets the model draw a curve or a curved surface while staying linear in its coefficients. Linear here is about the coefficients $\beta$ rather than about $x$, which is why least squares, the solve that picks the coefficients minimizing the sum of the squared residuals, carries over as it is. Expanding two variables to the second degree, the model learns equation (1).

$$\hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_1 x_2 + \beta_5 x_2^2 \hspace{19em} (1)$$

The square coefficients $\beta_3$ and $\beta_5$ carry the curvature inside one variable, a peak or a saturation. The non-linearity sits in the columns rather than in the model, so the linear model already in use, and the inference and the penalty built on it, are kept as they are.

### 3.2 Feature Interaction

The product term $\beta_4 x_1 x_2$ lets one variable change the slope of another. Differentiating equation (1) with respect to $x_1$ gives equation (2), which says as much.

$$\frac{\partial \hat{y}}{\partial x_1} = \beta_1 + 2 \beta_3 x_1 + \beta_4 x_2 \hspace{19em} (2)$$

Where $\beta_4$ is not zero the effect of $x_1$ differs at each level of $x_2$. On a process it reads as the effect of pressure depending on the temperature, which the first-order terms of $x_1$ and $x_2$ alone cannot write. An effect that appears only when the two variables are high together is carried by this product term and nowhere else. The long practice of writing a response surface, the surface the response traces over the process conditions, as a second-order polynomial is curvature and interaction combined, and reading the optimum off the point where that surface has zero slope, its stationary point, came from there [[1](#ref-1)].

## 4. Mechanism

### 4.1 Expansion

The columns an expansion makes are monomials, terms formed by multiplying powers of the raw variables. With $n$ variables and a highest degree of $d$, the new columns are the set of equation (3).

$$\Phi_d(\mathbf{x}) = \left\lbrace \prod_{i=1}^{n} x_i^{a_i} \ \middle|\ a_i \in \mathbb{Z}_{\ge 0}, \ 1 \le \sum_{i=1}^{n} a_i \le d \right\rbrace \hspace{19em} (3)$$

For $[X_1, X_2]$ at $d = 2$ the columns, including the intercept that is a constant column of ones, are $[1, X_1, X_2, X_1^2, X_1 X_2, X_2^2]$, which is what equation (1) is fitted on. Written for $n$ variables the second-order model is equation (4), where the two objectives of section 3 sit in the square terms and the product terms respectively.

$$y = \beta_0 + \sum_{i=1}^{n} \beta_i x_i + \sum_{1 \le i \le j \le n} \beta_{ij} x_i x_j + \varepsilon \hspace{19em} (4)$$

### 4.2 Centering And Conditioning

Subtract from each variable its own mean before expanding. This is centering, and it buys two things: a lower correlation between the columns, and a lower condition number for the design matrix, the matrix whose rows are the observations and whose columns are the terms the model uses, from which the coefficients are solved. The two paragraphs below are those two numbers.

Values in physical units usually sit far from zero, and such an $x$ and $x^2$ point in nearly the same direction. Over 60 samples on $[10, 11]$ their correlation is 0.9999, and after the mean is removed it is -0.15. The correlation after centering is proportional to the mean of the cubed deviations from the mean, the third central moment, so it is zero for a symmetric distribution and lands near zero in a sample.

Read as a condition number the difference is larger still. The condition number says by what factor a small error in the input is magnified in the solution. On the same sample the design matrix at $d = 2$ has a condition number of $1.6 \times 10^5$ in raw units and 2.8 after centering and scaling. At $d = 4$ they are $3.4 \times 10^{10}$ and 16, and at $d = 8$ they are $1.5 \times 10^{21}$ and $8.0 \times 10^{2}$ (Fig 1(b)). A 64-bit float carries about 16 significant digits, so at $d = 8$ in raw units no significant digit of the coefficients survives the solve.

The second reason to center is interpretation. On centered data $\beta_1$ is the slope while the other variables sit at their means, a readable quantity. Uncentered it is the slope while the other variables are zero, and that zero is often a point the data never visits [[2](#ref-2)].

Centering lowers the correlation, though, without removing it. The collinearity an expansion manufactures is a property of the expansion rather than of the data, so a penalty is needed alongside it (section 5.2).

### 4.3 Hierarchy

Keep a product term, and the main effects composing it stay as well. The rule is called heredity, and its ground is the coordinate system rather than statistics.

Substituting the shift $x_1 = z_1 + a$, $x_2 = z_2 + b$ into a product-only model such as $y = \beta_{12} x_1 x_2$ gives equation (5).

$$\beta_{12} (z_1 + a)(z_2 + b) = \beta_{12} z_1 z_2 + \beta_{12} b z_1 + \beta_{12} a z_2 + \beta_{12} ab \hspace{19em} (5)$$

Main effects appear on their own. A product model without main effects therefore depends on where the origin was placed, and whether temperature is measured in Celsius or in kelvin changes the model. Keep the main effects and that shift is absorbed as a rearrangement of the coefficients. There is a practice of dropping a main effect on the weak form of the rule, weak heredity, under which only one of the variables forming the product need be present, but the conditions that justify it almost never hold in practice [[4](#ref-4)]. Where variable selection is automated it is likewise better to carry heredity as a Bayesian prior or as a constraint on the optimization [[5](#ref-5)] [[6](#ref-6)].

`interaction_only=True` is the option that drops the square terms, not an option that breaks heredity. The first-order terms remain, so two variables give the columns $[X_1, X_2, X_1 X_2]$.

## 5. Caution

An expansion charges two prices. The column count grows fast, which invites overfitting — a model that fits the training data and misses new data — and raises the cost of the fit, and the columns it makes resemble one another, which unsettles the coefficients. The degree holds the first down, a penalty holds the second.

### 5.1 Dimensionality And Overfitting

The column count grows as the $d$-th power of the variable count. Without the intercept, the full expansion has the column count of equation (6), and `interaction_only`, which keeps only products of distinct variables, has that of equation (7).

$$p_{\mathrm{full}} = \binom{n+d}{d} - 1 \hspace{19em} (6)$$

$$p_{\mathrm{inter}} = \sum_{j=1}^{\min(d,\ n)} \binom{n}{j} \hspace{19em} (7)$$

Table 2. Column count after expansion, bias column excluded

| Variables | Degree 2, full | Degree 2, interaction only | Degree 3, full | Degree 3, interaction only |
| --- | --- | --- | --- | --- |
| 5 | 20 | 15 | 55 | 25 |
| 10 | 65 | 55 | 285 | 175 |
| 20 | 230 | 210 | 1,770 | 1,350 |
| 50 | 1,325 | 1,275 | 23,425 | 20,875 |
| 100 | 5,150 | 5,050 | 176,850 | 166,750 |

What Table 2 says is that `interaction_only` saves little. At $d = 2$ the difference is the $n$ square terms alone, so 5,150 becomes 5,050 at $n = 100$. The option is therefore not switched on to cut the column count; it is where the decision to keep curvature inside one variable out of the model is written down.

What actually sets the column count is the degree. Raising $d$ from 2 to 3 takes the columns from 230 to 1,770 at $n = 20$. As the column count approaches the row count the least-squares solution turns unstable, and past it the solution is not unique, so the ceiling on an expansion is set by the sample count rather than by the degree.

The degree is therefore chosen on the error over data kept out of the fit, the held-out error, rather than on theory, and the candidates are few. It is 2 in almost every practical case, data that needs 3 is rare, and a 4 that appears to win is a sign that something other than an expansion should be used.

<img src="polynomial-feature-expansion_fig/fig1.png" width="1100" style="max-width: 100%;" alt="Fig 1">

Fig 1. Degree and extrapolation, conditioning, and the cost of expansion

Fig 1(a) is the first reason. Degrees 2, 5 and 9 are fitted to 60 samples; inside the training range (grey) degrees 5 and 9 are both plausible, and outside it the higher degree diverges first. The behaviour of a polynomial beyond its range is governed by its top term, so raising the degree where extrapolation is needed improves the fit inside the training range while the error outside it grows.

Fig 1(b) draws the condition numbers of section 4.2 against degree, and Fig 1(c) is the relation between the term count and the row count. On data with 5 variables, 60 rows and a true model holding one product term, the held-out RMSE, that error measured as the root of the mean squared error, falls from 1.34 at degree 1 to 0.34 at degree 2 and returns to 1.08 at degree 3. The 55 columns of degree 3 nearly reach the 60 rows. Ridge is at 0.75 in the same place, stopping close to half of that deterioration.

### 5.2 Regularization

The expanded columns always carry a penalty, a term added to the fitting criterion that charges for the size of the coefficients. The expansion raises the column count and at the same time makes columns that resemble one another, and unpenalized least squares absorbs that resemblance into two large coefficients that cancel, which is why the fit moves far on a small disturbance of the data. Ridge answers it with a penalty proportional to the sum of the squared coefficients, which is what stops the two large coefficients from cancelling [[3](#ref-3)].

Ridge is the default of the two. It divides the coefficient among the columns that resemble one another and steadies the prediction, while lasso keeps one of them and drops the rest. Lasso on expanded columns can keep a product term while deleting its main effects, breaking the heredity of section 4.3, so it is used with a hierarchical constraint rather than on its own [[6](#ref-6)].

The penalty acts on the size of a column, so it is applied after the expanded columns are standardized, which is what puts the second standardization into the pipeline of section 6.2.

### 5.3 Failure Modes

An expansion fails in six recognizable ways. Most arrive not as a model that fits badly but as coefficients or predictions that turn unstable.

Table 3. Failure modes of a polynomial expansion

| Symptom | Cause | Countermeasure |
| --- | --- | --- |
| Held-out error worse at degree 2 than at degree 1 | Term count close to the row count | Ridge or lasso, `interaction_only`, selective expansion |
| Coefficient signs flipping across resamples | Collinearity manufactured by the expansion | Centering, a penalty, reading predictions instead of coefficients |
| Prediction diverging just outside the training range | Extrapolation behaviour of a polynomial | Spline, a range guard on the input, no extrapolation |
| A handful of rows dominating the fit | Squares amplifying leverage | Outlier handling before expansion, robust loss |
| Duplicate or all-zero columns | Dummy columns squared and crossed | `interaction_only=True`, expansion restricted to continuous columns |
| Imputed values amplified | Imputation error squared inside a product | Imputation before expansion, an indicator column for what was imputed |

The fifth row of Table 3 is written out separately because the expansion does not catch it on its own. A categorical variable is turned into numbers by giving each category a column that holds 1 where the row falls in that category and 0 otherwise, a dummy. A dummy squared is itself and becomes an exactly duplicated column, and the product of two dummies from the same categorical variable is always zero, since one row cannot fall in two categories at once. The expansion knows none of this, so columns coming from a categorical variable are either left out of the expansion or handled with `interaction_only`.

### 5.4 Diagnostics

Whether an expansion helped is confirmed in four ways.

- The held-out error curve drawn while raising the degree from 1. Watch whether the minimum stays at 2 or below.
- The condition number of the expanded design matrix and the VIF (Variance Inflation Factor) of each column. A large value after centering calls for a penalty.
- The share of samples redrawn from the data with replacement, the bootstrap, in which a coefficient keeps its sign. A product term whose sign flips is not interpreted.
- The residual plotted against the product terms. Confirm that the structure left before the expansion is gone.

## 6. Implementation

### 6.1 Options

The expansion itself is one line of `sklearn.preprocessing.PolynomialFeatures`, and only four arguments have to be settled [[7](#ref-7)].

Table 4. PolynomialFeatures arguments

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

### 6.2 Pipeline

An expansion is not used alone but placed between standardization and the penalized fit. The order is standardize the raw variables, expand, standardize the expanded columns again, then fit with a penalty.

The first standardization removes the conditioning problem of section 4.2, and the second makes the penalty fall evenly across the columns. The variance of a product term is close to the product of the raw variances and so differs widely from column to column; without the second standardization a ridge penalty lands almost entirely on the columns with the largest variance.

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

### 6.3 Cost

The cost of an expansion is linear in the column count, and that count grows by equation (6). At 100,000 rows, 100 variables and $d = 2$ the columns number 5,150, and holding them in a dense matrix, one that stores every value, takes 4.1 GB at 64 bits a value. Two routes keep the expanded columns out of memory.

The first is the kernel. The polynomial kernel of equation (8) computes the inner product of the expanded space without the expansion.

$$K(\mathbf{x}, \mathbf{z}) = (\gamma\, \mathbf{x}^{\top} \mathbf{z} + c)^{d} \hspace{19em} (8)$$

`KernelRidge(kernel='poly')` is that form, and since the cost falls on rows rather than on columns it suits data with many variables and few rows. What it costs is interpretation: no coefficient attaches to an individual monomial, so which product contributed cannot be read.

The second is approximation. `PolynomialCountSketch` compresses the terms a polynomial kernel uses into a fixed number of columns, a sketch, and `Nystroem` approximates the kernel matrix from a subset of the samples. Both belong to the family that approximates a kernel with a finite number of columns to keep the speed of a linear model [[8](#ref-8)], and both bound the column count at a value the user sets.

Sparse input is taken as it is. Feed in a CSR matrix, which stores only the non-zero values, and the expansion comes back in the same form, so data carrying many dummy columns does not inflate into a dense array.

### 6.4 Selective Expansion

Not every pair has to be built. Hand the expansion the columns to be crossed and the column count ends at the number chosen rather than at Table 2.

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

## 7. Comparison

An expansion is out of place in three situations: many variables, several bends inside one variable, and a need to extrapolate. Table 5 sets out what to move to in each.

Table 5. Alternatives to a polynomial expansion

| Method | What it buys | When to prefer | Cost |
| --- | --- | --- | --- |
| Polynomial expansion | Explicit terms, a linear model kept intact | A few dozen variables, curvature and pairwise effects | Column count, fragile extrapolation |
| Spline and P-spline | Local flexibility, a bounded basis [[11](#ref-11)] | Repeated bends inside one variable | Tensor products for interactions, growing again |
| GAM | A sum of per-variable curves, readable | Non-linear main effects, few interactions | Interaction terms declared by hand |
| Polynomial kernel | The same space without materializing it | Many variables, few rows | No coefficient attached to a term |
| Random feature or sketch | Column count fixed by the user | Many rows and many variables | Approximation error |
| Factorization machine | Pairwise coefficients factorized [[12](#ref-12)] | Sparse high-cardinality categorical data | Interaction strength only, limited reading |
| Tree ensemble | Interactions found without being named | The form of the interaction unknown | A piecewise-constant surface, no extrapolation |
| Rule ensemble | Rules alongside linear terms [[9](#ref-9)] | Interpretable interactions wanted | Rule count to be tuned |

Handing the expanded columns to PLS (Partial Least Squares) is another route. PLS projects the columns onto the directions of largest covariance with the response before regressing, so it meets the collinearity the expansion manufactures head on, and it is used on experimental data holding fewer observations than coefficients. Whichever is chosen, the order of judgement is the same. Establish first that expressive power is what is missing, then separate whether what is missing is a product term or a curvature, and choose the method after that. A treatment that compares the whole of basis expansion in one frame is available [[10](#ref-10)].

## 8. Further Work

- **Sparse polynomial chaos expansion** — A way to cut a high-degree expansion down to a size that can be carried, by selecting terms sparsely over a basis of mutually orthogonal polynomials [[13](#ref-13)]. Selecting the terms by least angle regression has settled into a procedure, which makes keeping a few dozen out of several hundred candidates computationally practical. Starting needs a distributional assumption on the input variables (the basis follows that distribution) and a designed sample.
- **Hierarchical interaction selection at scale** — The lasso family that selects product terms with heredity imposed as a convex constraint [[6](#ref-6)]. Convex means the optimum found is the only one, and it solves up to several hundred variables, so the rule of section 4.3 can be enforced by the optimization rather than by a person. Starting needs a rule that narrows the candidate product terms in advance and a computational budget.
- **Learned basis** — A model that stacks one-dimensional functions learned from the data in place of a fixed monomial basis [[14](#ref-14)]. A spline-based implementation was released in 2024, which makes a direct comparison with an expansion plus ridge on the same data possible. Starting needs a held-out comparison procedure and a criterion for whether the learned basis is excessive for the sample count.

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
[10] Hastie, T., Tibshirani, R. and Friedman, J. (2009). [The Elements of Statistical Learning: Data Mining, Inference, and Prediction](https://doi.org/10.1007/978-0-387-84858-7) (2nd ed.). Springer. ISBN 978-0-387-84857-0.<br>
<a id="ref-11"></a>
[11] Eilers, P. H. C. and Marx, B. D. (1996). [Flexible smoothing with B-splines and penalties](https://doi.org/10.1214/ss/1038425655). *Statistical Science*, 11(2), 89–121.<br>
<a id="ref-12"></a>
[12] Rendle, S. (2010). [Factorization Machines](https://doi.org/10.1109/ICDM.2010.127). *2010 IEEE International Conference on Data Mining*, 995–1000.<br>
<a id="ref-13"></a>
[13] Blatman, G. and Sudret, B. (2011). [Adaptive sparse polynomial chaos expansion based on least angle regression](https://doi.org/10.1016/j.jcp.2010.12.021). *Journal of Computational Physics*, 230(6), 2345–2367.<br>
<a id="ref-14"></a>
[14] Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T. Y. and Tegmark, M. (2024). [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756). *arXiv:2404.19756*.

---

## Appendix A. Terminology

- **collinearity**: The state in which two or more columns point in nearly the same direction, so that their coefficients cannot be estimated apart.
- **condition number**: The ratio of the largest singular value of a matrix to the smallest. It says how far a small error in the input is magnified in the solution.
- **degree**: The highest degree of a monomial the expansion admits. The degree of $X_1^2 X_2$ is 3.
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
- **VIF**: The variance inflation factor, computed from the $R^2$ of one column regressed on the rest. It is $1/(1-R^2)$.

## Appendix B. Reproduction Code

Fig 1 and every number the document quotes come from the script below.

```python
# Feature-Engineering/PFE/polynomial-feature-expansion.py
__author__ = 'yRocket'
__version__ = "0.0.1.2026.9.9"  # Semantic Versioning: Major.Minor.Patch.Date(YYYY.M.D)
import pathlib
from math import comb

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

matplotlib.use('Agg')

N_TRAIN = 60                                     # samples of the one-variable demonstration
N_ROW = 60                                       # rows of the five-variable demonstration
N_VARIABLE = 5                                   # variables of the five-variable demonstration
X_OFFSET = 10.0                                  # location offset that stands for a physical unit
NOISE_SIGMA = 0.08                               # noise of the one-variable demonstration
DEGREE_DRAWN = [2, 5, 9]                         # degrees drawn in panel (a)
DEGREE_RANGE = list(range(1, 9))                 # degrees swept in panel (b)
DEGREE_SWEEP = list(range(1, 5))                 # degrees swept in panel (c)
RIDGE_ALPHA = 1.0                                # ridge penalty on standardized expanded columns
SEED = 0
VARIABLE_COUNT = [5, 10, 20, 50, 100]            # variable counts of the term-count table
TERM_DEGREE = [2, 3]                             # degrees of the term-count table

FIGSIZE: tuple = (13.5, 4.4)
REFERENCE_WIDTH: float = 13.5                    # the width BASE_FONT_SIZE was chosen for
BASE_FONT_SIZE: float = 11.0
FONT_SIZE = BASE_FONT_SIZE * FIGSIZE[0] / REFERENCE_WIDTH
OUT_PATH = pathlib.Path(__file__).parent / 'polynomial-feature-expansion_fig' / 'fig1.png'

INK_COLOR = '#333333'
MUTED_COLOR = '#767676'
DEGREE_COLOR = [TABLEAU_COLORS['tab:blue'], TABLEAU_COLORS['tab:orange'], TABLEAU_COLORS['tab:red']]
RAW_COLOR = TABLEAU_COLORS['tab:red']
CENTERED_COLOR = TABLEAU_COLORS['tab:blue']
OLS_COLOR = TABLEAU_COLORS['tab:red']
RIDGE_COLOR = TABLEAU_COLORS['tab:blue']


def truth_1d(x: np.ndarray) -> np.ndarray:
    """Smooth one-variable response the polynomial fits approximate."""
    return np.sin(2.0 * np.pi * x) * 0.5 + 0.3 * x


def truth_2d(x: np.ndarray) -> np.ndarray:
    """Five-variable response whose only non-linear part is one interaction term."""
    return 1.0 + 2.0 * x[:, 0] - 3.0 * x[:, 1] + 0.5 * x[:, 2] + 4.0 * x[:, 0] * x[:, 1]


def design_matrix(x: np.ndarray, degree: int) -> np.ndarray:
    """Column-wise powers 1 ... degree of x, with the intercept column in front."""
    return np.column_stack([np.ones_like(x)] + [x ** power for power in range(1, degree + 1)])


def term_count_full(variable: int, degree: int) -> int:
    """Number of monomials of degree 1 ... degree in the given number of variables."""
    return comb(variable + degree, degree) - 1


def term_count_interaction(variable: int, degree: int) -> int:
    """Number of products of distinct variables, up to the given degree."""
    return sum(comb(variable, order) for order in range(1, min(degree, variable) + 1))


rng = np.random.default_rng(SEED)

x_unit = np.sort(rng.uniform(0.0, 1.0, N_TRAIN))
y_1d = truth_1d(x_unit) + rng.normal(0.0, NOISE_SIGMA, N_TRAIN)
x_grid = np.linspace(-0.4, 1.4, 400)

x_pair = rng.uniform(-1.0, 1.0, (N_ROW * 2, N_VARIABLE))
y_2d = truth_2d(x_pair) + rng.normal(0.0, 0.3, N_ROW * 2)
x_fit, x_test = x_pair[:N_ROW], x_pair[N_ROW:]
y_fit, y_test = y_2d[:N_ROW], y_2d[N_ROW:]

fig, axes = plt.subplots(1, 3, figsize=FIGSIZE)

ax = axes[0]
ax.plot(x_grid, truth_1d(x_grid), color=MUTED_COLOR, linewidth=1.2, linestyle='--', label='truth')
ax.scatter(x_unit, y_1d, s=12, color=INK_COLOR, zorder=3, label='train')
for degree, color in zip(DEGREE_DRAWN, DEGREE_COLOR):
    coefficient = np.linalg.lstsq(design_matrix(x_unit, degree), y_1d, rcond=None)[0]
    ax.plot(x_grid, design_matrix(x_grid, degree) @ coefficient, color=color, linewidth=1.4,
            label=f"degree {degree}")
ax.axvspan(0.0, 1.0, color='#000000', alpha=0.05)
ax.set_ylim(-1.6, 1.6)
ax.set_xlabel('x', fontsize=FONT_SIZE, color=INK_COLOR)
ax.set_ylabel('y', fontsize=FONT_SIZE, color=INK_COLOR)
ax.legend(fontsize=FONT_SIZE * 0.85, frameon=False, loc='lower center', ncol=2)

ax = axes[1]
condition_raw, condition_centered = [], []
for degree in DEGREE_RANGE:
    x_shifted = x_unit + X_OFFSET
    condition_raw.append(np.linalg.cond(design_matrix(x_shifted, degree)))
    x_scaled = (x_shifted - x_shifted.mean()) / x_shifted.std()
    condition_centered.append(np.linalg.cond(design_matrix(x_scaled, degree)))
ax.semilogy(DEGREE_RANGE, condition_raw, marker='o', color=RAW_COLOR, linewidth=1.4, label='raw x + 10')
ax.semilogy(DEGREE_RANGE, condition_centered, marker='s', color=CENTERED_COLOR, linewidth=1.4,
            label='centered and scaled')
ax.set_xlabel('degree', fontsize=FONT_SIZE, color=INK_COLOR)
ax.set_ylabel('condition number', fontsize=FONT_SIZE, color=INK_COLOR)
ax.set_xticks(DEGREE_RANGE)
ax.legend(fontsize=FONT_SIZE * 0.85, frameon=False, loc='upper left')

ax = axes[2]
rmse_ols, rmse_ridge = [], []
for degree in DEGREE_SWEEP:
    for model, holder in ((LinearRegression(), rmse_ols), (Ridge(alpha=RIDGE_ALPHA), rmse_ridge)):
        pipeline = make_pipeline(StandardScaler(), PolynomialFeatures(degree=degree, include_bias=False),
                                 StandardScaler(), model)
        pipeline.fit(x_fit, y_fit)
        holder.append(root_mean_squared_error(y_test, pipeline.predict(x_test)))
ax.plot(DEGREE_SWEEP, rmse_ols, marker='o', color=OLS_COLOR, linewidth=1.4, label='OLS')
ax.plot(DEGREE_SWEEP, rmse_ridge, marker='s', color=RIDGE_COLOR, linewidth=1.4,
        label=f"ridge, alpha = {RIDGE_ALPHA:g}")
ax.set_xlabel('degree', fontsize=FONT_SIZE, color=INK_COLOR)
ax.set_ylabel('held-out RMSE', fontsize=FONT_SIZE, color=INK_COLOR)
ax.set_xticks(DEGREE_SWEEP)
ax.legend(fontsize=FONT_SIZE * 0.85, frameon=False, loc='lower right')

for ax in axes:
    ax.tick_params(labelsize=FONT_SIZE * 0.9, colors=MUTED_COLOR)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#d6d6d6')

fig.subplots_adjust(left=0.06, right=0.99, top=0.96, bottom=0.22, wspace=0.28)
for ax, label in zip(axes, ('(a)', '(b)', '(c)')):
    box = ax.get_position()
    fig.text(box.x0 + box.width / 2.0, 0.045, label, ha='center', va='center', fontsize=FONT_SIZE,
             color=INK_COLOR)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=300)

x_shifted = x_unit + X_OFFSET
x_centered = x_shifted - x_shifted.mean()
print("correlation between a variable and its square")
print(f"  raw x + {X_OFFSET:g} : {np.corrcoef(x_shifted, x_shifted ** 2)[0, 1]:.4f}")
print(f"  centered     : {np.corrcoef(x_centered, x_centered ** 2)[0, 1]:.4f}")

print("\ncondition number of the design matrix")
print(f"{'degree':>7}{'raw':>14}{'centered':>14}")
for degree, raw, centered in zip(DEGREE_RANGE, condition_raw, condition_centered):
    print(f"{degree:>7}{raw:>14.3e}{centered:>14.3e}")

print("\nheld-out RMSE against degree")
print(f"{'degree':>7}{'OLS':>10}{'ridge':>10}")
for degree, ols, ridge in zip(DEGREE_SWEEP, rmse_ols, rmse_ridge):
    print(f"{degree:>7}{ols:>10.3f}{ridge:>10.3f}")

print("\nterm count without the bias column")
print(f"{'n':>5}" + ''.join(f"{'d=' + str(degree) + ' full':>14}{'d=' + str(degree) + ' inter':>15}"
                            for degree in TERM_DEGREE))
for variable in VARIABLE_COUNT:
    row = ''.join(f"{term_count_full(variable, degree):>14}{term_count_interaction(variable, degree):>15}"
                  for degree in TERM_DEGREE)
    print(f"{variable:>5}" + row)

print(f"\nsaved {OUT_PATH}")
```
