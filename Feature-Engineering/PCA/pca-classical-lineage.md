# PCA Classical Lineage
Rev. 4 | Created: 2026-08-11 | Updated: 2026-08-20 01:11 CDT

> PCA is not one technique but a lineage that branched from a single root.
> This document arranges those branches along one axis — which assumption of the original PCA each of them relaxed — and records what every branch gains and what it gives up.

The original PCA carries several assumptions at once: that the data is a matrix, that it fits in memory, that no outliers are present, that a component may use a little of every variable, that the structure is linear, that samples outnumber variables, and that the target is unknown. Each branch relaxes one of them and hands back something else in exchange. Choosing a branch is therefore not a comparison of performance but an answer to **which assumption broke in your data**.

## 1. Reading The Map

Table 1. The lineage at a glance

| Branch | Assumption it relaxes | Representative methods | Cost |
|---|---|---|---|
| Computation (§3) | The full covariance can be formed | Randomized SVD, Lanczos, Frequent Directions | It accepts an approximation error |
| Probabilistic (§4) | No generative model is needed | PPCA, Factor Analysis, Bayesian PCA | It has to assume a noise structure |
| Online (§5) | All the data is present at once | Oja, GHA, CCIPCA, Incremental PCA | It depends on convergence rate and learning rate |
| Robustness (§6) | There are no outliers | L1-PCA, Robust PCA | It costs more to compute and the solution may not be unique |
| Sparsity (§7) | A component may use every variable | SCoTLASS, Sparse PCA | It gives up some orthogonality and explained variance |
| Non-linear (§8) | The structure is a linear subspace | Kernel PCA, Autoencoder | The inverse map and the interpretation get harder |
| Asymptotics (§9) | Samples outnumber variables | Shrinkage, spiked model | The estimator leans on model assumptions |
| Data structure (§10) | The data is a matrix | FPCA, Tensor PCA, Dynamic PCA | A person has to declare the structure |
| Supervised (§11) | The target variable is unknown | Supervised PCA, PLS, Contrastive PCA | Change the target and the axes change with it |
| Distribution (§12) | The data can be gathered in one place | Distributed PCA, DP-PCA | It gives up communication volume or accuracy |

The branches are not exclusive. Data that has many variables and outliers as well — sensor readings unfolded one variable per instant, for instance — normally uses §6 and §9 together. Relaxing two assumptions at once brings both costs with it, so identify the broken assumption first and apply that branch before any other.

## 2. Root

PCA rests on the fact that two different objectives arrive at the same answer. One is to find the direction that maximizes the variance of the projected data; the other is to find the low-rank subspace that approximates the original data with the smallest error. The Eckart–Young theorem says the answer to the second is the truncated SVD, and that answer coincides with the answer to the first. Add the linear autoencoder and three formulations point at the same subspace.

Table 2. Three equivalent formulations

| Formulation | Object | Solution |
|---|---|---|
| Variance maximization | The projected variance of the centered data | The leading eigenvectors of the covariance matrix |
| Best low-rank approximation | The approximation error in the Frobenius norm | The right singular vectors of the truncated SVD |
| Linear autoencoder | The reconstruction error of a linear encoder-decoder | The subspace the principal components span |

The three give the same subspace but not the same coordinates. A linear autoencoder fixes only the subspace and leaves the rotation inside it undetermined, so interpreting individual components calls for the first formulation rather than the third.

Preprocessing changes the result. Without centering the first component points along the mean; without standardization the variable with the largest unit takes the variance. Since whether pressure is written in Pa or in kPa changes the components, a correlation-matrix basis — that is, standardization — is the default for measurement data with mixed units.

## 3. Computation Branch

Several methods reach the same subspace, and the size of the data together with how many times it can be read decides which one applies.

Table 3. Ways to reach the same subspace

| Method | Idea | When it fits |
|---|---|---|
| Covariance eigendecomposition | Form the `p × p` covariance and decompose it | When `p` is small |
| Direct SVD | Decompose the data matrix itself | The most numerically stable choice, and the default |
| Power iteration, Lanczos | Draw only the leading components by iteration | When `k ≪ p` and only matrix products are available |
| Randomized SVD | Narrow the subspace first with a random projection | To get the leading `k` from a large matrix quickly |
| Frequent Directions | Keep a fixed-size sketch in a single pass | When the data cannot be read twice |

Once `p` is large, forming the `p × p` covariance is itself the bottleneck. Two hundred sensors unfolded across ten thousand instants put `p` at two million, so the covariance is never formed and the work goes straight to SVD or randomized SVD.

## 4. Probabilistic Branch

Rewriting PCA as a generative model makes missing values and the component count tractable on principle. The starting point is to see an observation as a linear map of a low-dimensional latent variable plus noise.

Table 4. Generative reformulations

| Method | Noise assumption | What it buys |
|---|---|---|
| Probabilistic PCA | Isotropic noise `σ²I` | A likelihood, missing-value handling, an EM algorithm |
| Factor Analysis | Diagonal but not isotropic noise | It admits a different noise level per variable |
| EM for PCA | — | It solves by iteration without forming the covariance |
| Bayesian PCA | A prior over the components | It prunes the component count automatically |

Where PPCA and Factor Analysis part in practice is when the noise level differs from sensor to sensor. Hold the isotropic assumption and a noisy sensor pulls the principal direction toward itself.

## 5. Online And Incremental Branch

When the data arrives as a stream, or does not fit at once, the components are updated instead of recomputed.

Table 5. Updating instead of recomputing

| Method | Update unit | Note |
|---|---|---|
| Oja, GHA | One sample | The design of the learning rate governs convergence |
| CCIPCA | One sample | It averages by sample count instead of a learning rate, so there is no rate to tune |
| Incremental PCA | A block | It stitches block SVDs together |
| Streaming SVD | A block | A forgetting factor can be placed on older data |

In streaming data the subspace itself may move over time. Whether that movement is estimation error or the signal of a process change is the central question of this branch, and in FDC it is usually the latter.

## 6. Robustness Branch

A single outlier drags a least-squares component with it. In measurement data outliers are not the exception but the standing condition.

Table 6. Resisting outliers

| Method | Idea | Cost |
|---|---|---|
| L1-PCA | Minimize absolute error instead of squared error | The optimization is not convex |
| M-estimation | Down-weight samples with large residuals | A weight function and a tuning constant are required |
| Robust PCA | Decompose the matrix into a low-rank part and a sparse part | It is solved by convex relaxation but costs to compute |

Robust PCA does not discard outliers; it puts them in the sparse part. That sparse part is the list of anomalous samples, so the reduction and the anomaly detection come out together.

## 7. Sparsity And Interpretability Branch

A principal component normally gives a non-zero loading to every variable. With hundreds of variables there is no saying what such a component means.

Table 7. Making loadings readable

| Method | Idea | What it gives up |
|---|---|---|
| Varimax rotation | Rotate the axes while holding the subspace | The ordering of variance across components |
| SCoTLASS | Place an L1 constraint on the loadings | The optimization is hard |
| Sparse PCA | Recast as a regression and use the elastic net | Orthogonality between components |
| Structured sparsity | Zero out whole groups of variables | A person supplies the grouping |

Rotation does not change the subspace, so the reconstruction error is untouched. Sparsification changes the subspace itself, so explained variance falls. This is the place where the price of interpretability is set.

## 8. Non-linear Branch

This is the branch for structure that is not a linear subspace.

Table 8. Leaving the linear subspace

| Method | Idea | Note |
|---|---|---|
| Kernel PCA | Run PCA in a feature space using inner products alone | The kernel matrix is quadratic in sample count and is approximated by Nyström |
| Autoencoder | Learn a non-linear encoder and decoder | With linear hidden layers it coincides with the PCA subspace |
| Manifold learning | Preserve neighbourhood relations | A neighbour of PCA, not a descendant |

Placing t-SNE and UMAP in this branch invites a misreading. Neither yields a projection function, so a new sample cannot be sent to the same coordinates, and neither preserves distance. They are visualization tools, not a replacement for the dimension-reduction step.

## 9. High-Dimensional Asymptotics Branch

When `p` is large relative to `n`, sample eigenvalues depart systematically from population eigenvalues. Even data that is pure noise produces large leading eigenvalues, so size alone does not make a direction signal.

Table 9. What large `p` does

| Topic | Statement |
|---|---|
| Marchenko–Pastur | The eigenvalues of pure noise spread over a specific interval |
| Spiked model | A signal has to pass a threshold before it separates from that interval |
| Eigenvalue shrinkage | Shrink the sample eigenvalues to correct the bias |
| Component count | Decide it by scree, parallel analysis, or cross-validation |

Below that threshold lies the regime where the sample eigenvector becomes unrelated to the population eigenvector. Choose the component count without checking that boundary and noise directions get adopted as principal components.

## 10. Data Structure Branch

When the data is not a matrix, unfolding it discards the information its structure was carrying.

Table 10. When the data is not a matrix

| Method | Structure it keeps | Typical data |
|---|---|---|
| Functional PCA | The smoothness of a curve | Measurement curves that run continuously in time |
| Tensor, Multilinear PCA | The product structure of several axes | wafer × sensor × time |
| 2DPCA | The rows and columns of an image | Wafer maps |
| Dynamic PCA | Lagged correlation | Process traces with autocorrelation |
| Multi-block | The relation between blocks | Equipment and metrology held together |

Unfolding a trace into one variable per instant makes neighbouring instants unrelated variables. Functional PCA preserves that adjacency, and Dynamic PCA states the lag as a variable so the autocorrelation enters the model.

## 11. Supervised And Contrastive Branch

When the target variable is known, finding the directions related to the target beats finding the directions of largest variance.

Table 11. Reduction that knows the target

| Method | Uses | Note |
|---|---|---|
| Supervised PCA | It screens variables by their correlation with `y` first | PCA itself is used unchanged |
| PLS | It maximizes the covariance of `X` and `y` | Regression and reduction happen together |
| CCA | It maximizes the correlation of two blocks | For a vector-valued target |
| LDA | Between-class variance over within-class variance | For classification only, and the component count is tied to the class count |
| Contrastive PCA | Directions peculiar to the target against a background | When a normal group is available separately |

The direction of largest variance and the direction that explains the target part company often. A large difference between tools taking the first component while the fine variation tied to yield is pushed down the list is the standard example.

## 12. Distributed And Private Branch

This is the branch for data that cannot be gathered in one place.

Table 12. Computing without gathering the data

| Method | Constraint | Idea |
|---|---|---|
| Distributed PCA | The data sits on several nodes | Per-node summaries are combined |
| Federated PCA | The raw data cannot leave its site | Only the updates are exchanged |
| Differentially Private PCA | No individual sample may be revealed | Noise is added to the covariance or the components |

Data that cannot cross between fabs, or between customers, is where this branch is needed. How much accuracy has to be surrendered is the price of honouring that constraint.

---

## Appendix A. Terminology

The terms below appear in the body without being defined there.

- **Eckart–Young** is the theorem that among matrix approximations of a given rank the truncated SVD minimizes the error.
- **Elastic net** is a regression penalty that uses L1 and L2 terms together.
- **EM** is the algorithm that fits a latent-variable model by alternating an expectation step and a maximization step.
- **FDC** is Fault Detection and Classification, the practice of finding faults from the sensor record of process equipment.
- **Frobenius norm** is the square root of the sum of the squared entries of a matrix.
- **Lanczos** is an iterative method that finds the leading eigenpairs of a symmetric matrix using matrix products alone.
- **Loading** is the coefficient of an original variable when a principal component is written as a linear combination.
- **Marchenko–Pastur** is the limiting distribution the eigenvalues of a sample covariance follow.
- **Nyström** approximates a full kernel matrix from a subset of its columns.
- **Scree** is the plot of eigenvalues in decreasing order, whose bend is taken as the component count.
- **Sketch** is a summary matrix held at a size smaller than the original data.
- **Spiked model** is a covariance model that adds a few large eigenvalues to a noise covariance.
- **SVD** decomposes a matrix into left singular vectors, singular values, and right singular vectors.
- **Varimax** is an orthogonal rotation that increases the spread of the loadings to make them easier to read.
