# PCA Algorithm
Rev. 2 | Created: 2026-08-11 | Updated: 2026-08-11 17:38 CDT

> The other documents in this folder say which variant to reach for. This one says what the plain procedure actually does.
> It fixes the notation, walks the steps in order, states what each step produces, and lists the conventions that decide whether two implementations agree.

PCA (Principal Component Analysis) is short enough to write out in full, and writing it out settles questions that a library call hides — what the sign of a component means, why the rank stops at `n − 1`, which mean gets subtracted from new data, and what is lost when components are dropped.

## 1. Notation

Table 1. Symbols used throughout

| Symbol | Shape | Meaning |
|---|---|---|
| `X` | `n × p` | The data matrix, one sample per row and one variable per column |
| `n` | — | The number of samples |
| `p` | — | The number of variables |
| `k` | — | The number of components kept |
| `μ` | `1 × p` | The column means of `X` |
| `σ` | `1 × p` | The column standard deviations of `X` |
| `X_c` | `n × p` | The centered, optionally scaled data |
| `V` | `p × r` | The loadings, one principal direction per column |
| `T` | `n × r` | The scores, the coordinates of each sample in the new basis |
| `λ_j` | — | The variance explained by component `j` |
| `r` | — | The rank of `X_c`, at most `min(n − 1, p)` |

The loadings are the new axes and the scores are the coordinates on them. The two are easy to confuse, because they come out of the same decomposition and both get called "the components" in casual use.

## 2. The Procedure

```text
1. center      X_c = X − μ                     μ from the training rows only
2. scale       X_c = X_c / σ                   optional, but the default for mixed units
3. decompose   X_c = U S Vᵀ                    thin SVD
4. fix signs   flip (u_j, v_j) by convention   the pair is only defined up to sign
5. variance    λ_j = s_j² / (n − 1)            the variance along component j
6. truncate    keep the first k columns of V
7. project     T = X_c V_k                     the scores, n × k
```

Step 1 is not optional. Skip the centering and the first component points at the mean of the data rather than at the direction of its variation, because the criterion being maximized becomes the second moment about the origin instead of the variance.

Step 2 is a modelling choice. Without scaling, the variable with the largest numerical range dominates, so a table that mixes nanometres with ohms has to be scaled. With scaling the decomposition runs on the correlation matrix instead of the covariance matrix, and the two give different components.

## 3. Two Routes To The Same Answer

The decomposition in step 3 can be reached by either of two routes, and they agree in exact arithmetic.

Table 2. Covariance route against SVD route

| Aspect | Covariance route | SVD route |
|---|---|---|
| What is formed | `C = X_cᵀ X_c / (n − 1)`, then its eigendecomposition | The SVD of `X_c` directly |
| Loadings | The eigenvectors of `C` | The columns of `V` |
| Variance | The eigenvalues of `C` | `s_j² / (n − 1)` |
| Cost | `O(np² + p³)` | `O(np · min(n, p))` |
| Conditioning | Forming `C` squares the condition number | It works on `X_c` as given |

**The SVD route is the default, and the reason is numerical.** Squaring the condition number means that variables which are nearly collinear — the normal state of measurement data — lose roughly half the available digits before the eigensolver ever starts. The covariance route stays useful when `p` is small enough that `C` is cheap and its conditioning is not in question.

## 4. What Comes Out

Table 3. The three outputs and what each is for

| Output | Shape | Reading |
|---|---|---|
| Loadings `V_k` | `p × k` | How much each original variable contributes to each component |
| Scores `T` | `n × k` | Where each sample sits in the reduced space, and the input to any downstream model |
| Explained variance `λ_j` | `k` | How much of the total variation each component accounts for |

The explained variance ratio is `λ_j / Σλ`, where the sum runs over all `r` components and not only the `k` that were kept. Dividing by the sum of the kept ones inflates every figure and hides exactly what truncation discarded.

Reconstruction follows the same path backwards. With `X̂ = T V_kᵀ`, undoing the scaling and adding `μ` back returns the approximation in the original units, and the average squared error per sample equals the sum of the discarded `λ_j`. That identity is what makes the explained variance ratio a statement about reconstruction rather than a bare number.

## 5. Choosing The Component Count

Table 4. Ways to fix `k`

| Method | Rule | Caution |
|---|---|---|
| Cumulative variance | Keep enough components to pass a stated fraction | The fraction is a choice, not a finding |
| Scree | Keep the components before the bend in the eigenvalue plot | The bend is often not sharp |
| Kaiser | On the correlation matrix, keep eigenvalues above one | Crude, and it tends to keep too many |
| Parallel analysis | Keep the components whose eigenvalues exceed those of shuffled data | It needs repeated runs but answers the right question |
| Cross-validation | Keep the count that minimizes held-out reconstruction error | The most defensible, and the most expensive |

Parallel analysis compares the eigenvalues against a null in which there is no structure, and cross-validation asks directly which count generalizes. The other three settle the question by a threshold the reader has to choose.

## 6. Applying It To New Data

A fitted PCA is three stored objects: `μ`, `σ`, and `V_k`. Transforming a new sample uses those stored values and nothing computed from the new data.

```python
# Python
T_new = ((X_new - mu) / sigma) @ V_k          # scores for new rows
X_hat = (T_new @ V_k.T) * sigma + mu          # back to the original units
```

**Recomputing `μ` or `σ` from the new rows is the standard leakage error.** It quietly moves the origin between fit and transform, so the scores of the two sets no longer live in the same coordinate system, and a validation score computed that way is not comparable to a production one.

The same rule governs cross-validation. The centering, the scaling, and the decomposition all belong inside each training fold; performing them once on the whole dataset lets the held-out rows influence the axes they are later scored against.

## 7. Conventions And Numerical Details

Table 5. Details that decide whether two implementations agree

| Detail | Statement |
|---|---|
| Sign | `(u_j, v_j)` and `(−u_j, −v_j)` describe the same component, so a convention such as forcing the largest-magnitude entry of `v_j` positive is needed for reproducibility |
| Rank | Centering removes one degree of freedom, so `r ≤ min(n − 1, p)` and any further components are numerical noise |
| Denominator | `n − 1` matches the unbiased sample variance; `n` appears in some libraries and shifts every `λ_j` by a constant factor |
| Ties | Equal eigenvalues leave the rotation inside that subspace undetermined, so individual loadings are not interpretable there |
| Missing values | The decomposition has no notion of a missing entry, so it must be imputed first or a probabilistic variant used instead |

The sign convention matters more than it appears. Two runs that differ only in sign produce loading plots that look mirrored and scores that flip about zero, which reads as a change in the data when it is nothing of the kind.

## Appendix A. Terminology

The terms below appear in the body without being defined there.

- **Condition number** measures how much a matrix amplifies relative error, so a large one means fewer reliable digits in the result.
- **Correlation matrix** is the covariance matrix of the standardized variables.
- **Covariance matrix** holds the pairwise covariances of the variables, with the variances on its diagonal.
- **Eigendecomposition** writes a symmetric matrix as its eigenvectors scaled by its eigenvalues.
- **Kaiser rule** keeps the components whose eigenvalue on the correlation matrix exceeds one.
- **Leakage** is the use of information from held-out data while fitting, which makes a validation estimate optimistic.
- **Loading** is the coefficient of an original variable in a principal component.
- **Parallel analysis** compares the observed eigenvalues against those obtained from data with the same shape but no structure.
- **Rank** is the number of linearly independent directions a matrix spans.
- **Score** is the coordinate of a sample along a principal component.
- **Scree** is the plot of eigenvalues in decreasing order, whose bend is taken as the component count.
- **SVD** decomposes a matrix into left singular vectors, singular values, and right singular vectors.
- **Thin SVD** returns only the singular vectors that correspond to non-zero singular values.
