# Nonlinearity in Linear Models
Rev. 5 | Created: 2026-09-23 | Updated: 2026-09-23 09:54 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Placement](#31-placement)
- [4. Principle](#4-principle)
  - [4.1 Linear Model Mechanics and Limits](#41-linear-model-mechanics-and-limits)
  - [4.2 Nonlinear Features in a Linear Model](#42-nonlinear-features-in-a-linear-model)
- [5. Application](#5-application)
  - [5.1 Feature-Intensive Model](#51-feature-intensive-model)
  - [5.2 Algorithm-Intensive Model](#52-algorithm-intensive-model)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Python Implementation](#appendix-b-python-implementation)
- [Appendix C. Two Views of Linearity](#appendix-c-two-views-of-linearity)

## 1. Purpose

- **Problem Statement**: The linearity and the non-linearity of mathematical statistics and of a machine learning model, and the linear property and the non-linear property of the data, are not told apart, and the modeling strategy is confused as a result.
- **Goal**: On the ground of linearity as mathematical statistics defines it, compare the modeling strategies that handle feature interaction and non-linear data properties in machine learning.
- **Non-Goal**: The detail of each model is not covered.

## 2. Summary

The linearity of a model and the non-linear property of the data are two different things. Linearity is being first order in the weight $\beta$, and the non-linear property is the curvature of the relation between $x$ and $y$ together with the interaction between variables. A non-linear transform of the input therefore leaves the structure that is first order in $\beta$ in place, and least squares and the solutions of Ridge, Lasso and PLS are used as they are (section 4.2).

Who handles the non-linear property of the data splits the two modeling strategies.

- **Feature-Intensive Model**: a linear model with non-linear features (the feature engineering approach). "Keep the model simple (linear) and make the data (feature) complex." A linear model fed with columns such as $x^2$ and $x_1 x_2$ that the analyst builds.
- **Algorithm-Intensive Model**: a non-linear model with linear features (the algorithm approach). "Leave the data (feature) as it is (linear, original) and make the model complex (non-linear)." An approach where the original columns go in unchanged and a tree ensemble or a neural network learns the shape inside.

## 3. Taxonomy and its Hierarchy

Nonlinearity splits on three axes: the linearity of the model, the linearity of the feature, and the value you read out. The Feature-Intensive Model puts non-linear features into a linear model and reads the coefficient of each term, and the Algorithm-Intensive Model puts linear features into a non-linear model and reads variable importance. [Fig 1](#fig-1) is those three axes and the methods on each side.

```text
Nonlinearity in a model
|
+-- Feature-Intensive Model ........................ linear model + non-linear feature
|     +-- Power term: x^2, x^3 ..................... curvature of one variable
|     +-- Interaction term: x1 * x2 ................ joint effect of two variables
|     +-- Basis expansion: spline, RBF ............. local shape without a global degree
|
+-- Algorithm-Intensive Model ...................... non-linear model + linear feature
      +-- Tree ensemble ............................ split points cut the input space
      +-- Neural network ........................... activation function bends the response
      +-- Kernel method ............................ inner product in an implicit feature space
```

<a id="fig-1"></a>
Fig 1. The composition of the two models and the methods on each side

The hierarchy of the two models descends by the strength of the assumption. The Feature-Intensive Model writes the shape of the nonlinearity down as terms in advance, and in return reads the coefficient of each term directly. The Algorithm-Intensive Model leaves the shape unwritten, and in return cannot read from a single coefficient which range of which variable moved the prediction.

### 3.1 Placement

Table 1. Where each model sits on the three axes

| Model                     | Linearity  | Feature                          | What you read out                | Breaks when                                         |
| :-----------------------: | :--------: | :------------------------------: | :------------------------------: | :-------------------------------------------------: |
| Feature-Intensive Model   | Linear     | Non-linear (transformed columns) | Coefficient $\beta$ of each term | The shape of the nonlinearity is unknown in advance |
| Algorithm-Intensive Model | Non-linear | Linear (original columns)        | Variable importance              | The extrapolation range and a small sample          |

The Feature-Intensive Model has the analyst decide which terms to build, so there has to be ground for guessing which curve and which interaction the data holds. The Algorithm-Intensive Model fits without that ground, but a tree ensemble cannot extrapolate beyond the train data and a neural network overfits when the samples are few.

## 4. Principle

### 4.1 Linear Model Mechanics and Limits

Models such as linear regression, Ridge, Lasso and PLS (Partial Least Squares) assume the relation between the input variable $X$ and the target variable $Y$ to be a linear combination.

```math
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon \hspace{19em} (1)
```

- Advantage
  - The model is simple, so the risk of overfitting is small and the parameters converge very fast.
  - The coefficient $\beta_i$ tells directly how much each variable contributed to the change of the target.
- Limit
  - When the data holds a non-linear relation or an interaction between variables, a decision boundary shaped as a first-order plane cannot fit it, and underfitting follows.

### 4.2 Nonlinear Features in a Linear Model

In a linear model, the mathematical definition of linearity is being first order in the weight parameter $\beta$, the target of the optimization, rather than first order in the input variable $x$.

An expansion of the input space by a non-linear transform therefore keeps the structure that is first order in $\beta$, and the closed-form solution of the linear model is used as it is.

With the original input data $x_1$ and $x_2$, non-linear power terms are added and the input is mapped onto a new basis.

```math
\phi(x_1, x_2) = [1, x_1, x_2, x_1^2, x_2^2, x_1 x_2]^T \hspace{19em} (2)
```

The linear model is then fitted in this expanded space.

```math
y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 x_1^2 + \beta_4 x_2^2 + \beta_5 (x_1 x_2) \hspace{14em} (3)
```

- On the variable $x$: a non-linear model (a curve and an interaction surface are expressible)
- On the weight $\beta$: a linear model (least squares, Ridge/Lasso regularization and the other existing algorithms apply unchanged)

The model of equation (3) carries the intercept $\beta_0$, so by the definition of linear algebra it is an affine transform, and the linearity used in engineering covers that affine case as well ([Appendix C](#appendix-c-two-views-of-linearity)).

The coefficients of equation (3) are read the same way as the coefficients before the expansion. $\beta_3$ is the curvature of $x_1$ and $\beta_5$ is the contribution of the two variables moving together, and least squares fixes both.

## 5. Application

The approach splits by who handles the nonlinearity. In the Feature-Intensive Model the analyst adds the non-linear and interaction terms with `PolynomialFeatures` and the like, and then fits a linear model (Ridge, PLS and so on). The Algorithm-Intensive Model feeds the original data ($x_1$, $x_2$) unchanged and lets a tree-based ensemble (XGBoost, Random Forest) or a neural network learn the non-linear pattern inside, through its splits and its activation functions.

### 5.1 Feature-Intensive Model

`Ridge`, `Lasso` and `PLSRegression` do the fitting, and the three methods of [Fig 1](#fig-1) decide in front of them which columns go in.

- **Power term**: the $x^2$ and $x^3$ columns of one variable. `PolynomialFeatures(degree=3)` builds every term up to the degree at once.
- **Interaction term**: the product column $x_1 x_2$ of two variables. When only the product is wanted, `PolynomialFeatures(interaction_only=True)` drops the power terms.
- **Basis expansion**: a spline, which joins a polynomial per interval at the points that divide them, and RBF, which builds columns from the distance to a center. `n_knots` of `SplineTransformer` sets the interval count, and the curve of one interval changes without raising the degree.

The three methods share the conditions below.

- **Assumption**: The shape of the nonlinearity to be held can be written as terms in advance.
- **Settings**: The regularization strength `alpha`. The expanded columns differ in scale, so Ridge or Lasso bounds the coefficients.
- **Breaks when**: The true relation lies outside the terms that were written down, and underfitting remains after the expansion. Raising the degree to catch it grows the column count fast and the coefficients become unstable.
- **Where you meet it**: A place where physical ground, process variables for one, lets you guess the curvature and the interaction, and the coefficients have to be reported.

### 5.2 Algorithm-Intensive Model

The original columns go in unchanged, and the three methods of [Fig 1](#fig-1) build the nonlinearity inside the model.

- **Tree ensemble**: it cuts the input space by splits and fits a constant per range. `RandomForestRegressor`, `HistGradientBoostingRegressor`, XGBoost and LightGBM belong here.
- **Neural network**: the activation function bends the response at every layer and makes a continuous surface. `MLPRegressor` is the implementation.
- **Kernel method**: instead of growing the columns it stands in for the expanded space through the inner product between samples. `SVR(kernel='rbf')` and `KernelRidge` are the implementations.

The three methods share the conditions below.

- **Assumption**: The samples are many enough to fix the split positions and the weights. A tree ensemble fits a constant per range, so the sample count inside a range fixes the accuracy.
- **Settings**: `max_depth` and the learning rate of the tree ensemble, the layer count and the activation function of the neural network, `gamma` and `C` of the kernel method.
- **Breaks when**: For an input outside the train data a tree ensemble returns the constant of the last range and cannot extrapolate. A neural network overfits when the samples are few, and the work of a kernel method grows with the square of the sample count when they are many.
- **Where you meet it**: A place where the variables are too many to write the terms one by one, and the prediction accuracy comes before reading the coefficients.

A run comparing the accuracy of the two models on the same data is in [Appendix B](#appendix-b-python-implementation).

---

## Appendix A. Terminology

- **affine transform**: A linear transform with a constant shift added. $y = ax + b$ with $b \neq 0$ belongs here.
- **basis expansion**: A transform that turns the input variables into the values of predetermined functions and so grows the columns. Power terms, splines and RBF belong here.
- **closed-form solution**: A solution obtained from one expression without iteration. The normal equation of least squares is an example.
- **interaction**: A contribution that appears only when two variables move together. It is held by the product column $x_1 x_2$.
- **kernel method**: A method that holds a non-linear relation through the inner product between two samples instead of transforming the input itself.
- **RBF**: Radial basis function. A basis function whose value is fixed by the distance from a center.
- **spline**: A function that joins low-degree polynomials at the points that divide the intervals.
- **tree ensemble**: A model that collects the predictions of several decision trees. Random Forest and gradient boosting belong here.
- **underfitting**: A state where the model lacks the capacity to express the relation and the error is large even on the training data.

## Appendix B. Python Implementation

The two models are applied to the same data and their accuracy compared.

```python
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# --- synthetic data with one curvature term and one interaction term ---
rng = np.random.default_rng(0)
n_samples = 400
x1 = rng.uniform(-3, 3, size=n_samples)
x2 = rng.uniform(-3, 3, size=n_samples)
y = 3 + 2 * x1 - x2 + 1.5 * x1 ** 2 + 0.8 * x1 * x2 + rng.normal(0, 1.0, size=n_samples)

X = np.column_stack([x1, x2])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# the baseline: a linear model on the original columns
plain = Ridge(alpha=1.0).fit(X_train, y_train)

# the feature-intensive model: the analyst adds the power and interaction columns
feature_intensive = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False), Ridge(alpha=1.0)
).fit(X_train, y_train)

# the algorithm-intensive model: the algorithm splits the original columns on its own
algorithm_intensive = HistGradientBoostingRegressor(max_depth=3, random_state=0).fit(X_train, y_train)

for name, model in (("linear model, original columns", plain),
                    ("feature-intensive: Ridge on degree-2 columns", feature_intensive),
                    ("algorithm-intensive: HistGradientBoostingRegressor", algorithm_intensive)):
    print(f"{name:50s} R2 = {r2_score(y_test, model.predict(X_test)):.4f}")

# the coefficients the feature-intensive model reads out, one per term of equation (3)
term_names = feature_intensive[0].get_feature_names_out(["x1", "x2"])
print("feature-intensive coefficients:",
      dict(zip(term_names, np.round(feature_intensive[1].coef_, 3))))
```

The first three lines are the held-out $R^2$ of the three models, and the last line is the coefficient of each term that the Feature-Intensive Model reads out.

```text
linear model, original columns                     R2 = 0.4735
feature-intensive: Ridge on degree-2 columns       R2 = 0.9782
algorithm-intensive: HistGradientBoostingRegressor R2 = 0.9689
feature-intensive coefficients: {'x1': np.float64(1.963), 'x2': np.float64(-1.015), 'x1^2': np.float64(1.516), 'x1 x2': np.float64(0.76), 'x2^2': np.float64(0.005)}
```

Four coefficients recovered 2, -1, 1.5 and 0.8, the coefficients of the expression that made the data, and the coefficient of $x_2^2$, which that expression does not carry, stayed at 0.005.

## Appendix C. Two Views of Linearity

- Linearity in linear algebra: additivity ($f(x+y)=f(x)+f(y)$) and homogeneity ($f(cx)=cf(x)$) have to hold, and the map has to pass through the origin ($y=ax+b$ with $b \neq 0$ is an affine transform).
- Linearity in a domain or in engineering: a straight-line and proportional relation, used in a wider sense than the strict definition of linear algebra.
