# PCA Derivation
Rev. 6 | Created: 2026-08-11 | Updated: 2026-08-20 01:11 CDT

> The procedure can be followed without knowing why its answer is right, and most uses never need to ask.
> This document is for when the question comes up: it derives, from the requirement of keeping as much variance as possible, that the axes PCA returns must be eigenvectors of the covariance matrix.

The whole derivation is one constrained maximization solved with a Lagrange multiplier, and it is short enough to check line by line. Working through it settles why centering is mandatory, why an eigenvalue is itself a variance, why the components come out ordered and orthogonal without anyone asking for that, and why maximizing the kept variance and minimizing the reconstruction error are the same act.

## 1. Notation

Table 1. Symbols used throughout

| Symbol | Shape | Meaning |
|---|---|---|
| $x_i$ | $p \times 1$ | One centered sample, written as a column vector |
| $n$ | — | The number of samples |
| $p$ | — | The number of variables |
| $X_c$ | $n \times p$ | The centered data, one sample per row |
| $C$ | $p \times p$ | The covariance matrix, $C = X_c^\top X_c / (n - 1)$ |
| $u$ | $p \times 1$ | A candidate direction, constrained to unit length |
| $\lambda$ | — | The Lagrange multiplier, which turns out to be an eigenvalue and a variance |
| $u_j$, $\lambda_j$ | — | The j-th eigenvector and eigenvalue of $C$, ordered so that $\lambda_1 \ge \lambda_2 \ge \cdots$ |
| $V$ | $p \times k$ | The loadings matrix, the kept components stacked as columns $V = [\, u_1 \cdots u_k \,]$ |
| $k$ | — | The number of components kept |

Two remarks keep the reading smooth. First, $C$ is symmetric, because $(X_c^\top X_c)^\top = X_c^\top X_c$; the derivation leans on that fact twice, once in the gradient and once in the orthogonality argument. Second, the denominator $n - 1$ is the unbiased convention, and nothing below depends on it — replacing it with $n$ scales every eigenvalue by the same factor and leaves every eigenvector unchanged.

## 2. The Objective

The data is a cloud of $n$ points in $p$ dimensions, and the question is which single direction keeps the most of its variation.

Centering comes first, and it is not a convention but a precondition. Variance is spread about the mean, so the samples must satisfy $\sum_i x_i = 0$ before the quantity below deserves the name; skip the centering and the same formula measures the second moment about the origin instead, and the winning direction points at the mean of the cloud rather than along it.

A direction is a unit vector $u$, and the coordinate of a sample along it is the scalar $u^\top x_i$. The variance of those coordinates is the objective:

$$
\mathrm{Var}(u) \;=\; \frac{1}{n-1} \sum_{i=1}^{n} \left( u^\top x_i \right)^2
\;=\; u^\top \left( \frac{1}{n-1} \sum_{i=1}^{n} x_i x_i^\top \right) u
\;=\; u^\top C\, u
$$

The constraint $u^\top u = 1$ is what makes the problem well posed. Doubling $u$ quadruples $u^\top C u$, so without the constraint the maximum is unbounded and meaningless; the object being chosen is a direction, and fixing the length to one removes everything that is not direction. The problem is therefore:

$$
\max_{u} \; u^\top C\, u \qquad \text{subject to} \qquad u^\top u = 1
$$

![Fig 1](pca-derivation-fig1.png)

Fig 1. The problem and its answer for a two-variable cloud with $C = \bigl[\begin{smallmatrix} 4 & 2 \\ 2 & 4 \end{smallmatrix}\bigr]$. Panel (a) shows the centered cloud with the two directions the derivation finds, drawn with lengths proportional to the spread along them. Panel (b) sweeps the direction $u$ through 180 degrees and plots $u^\top C u$; the extremes sit exactly at the angles of $u_1$ and $u_2$, with heights $\lambda_1 = 6$ and $\lambda_2 = 2$.

## 3. The Lagrange Condition

A constrained maximum is found by folding the constraint into the objective with a multiplier and asking where the combined function is stationary:

$$
L(u, \lambda) \;=\; u^\top C\, u \;-\; \lambda \left( u^\top u - 1 \right)
$$

Setting the derivative with respect to $\lambda$ to zero recovers the constraint. Setting the gradient with respect to $u$ to zero gives the condition that matters:

$$
\frac{\partial L}{\partial u} \;=\; 2\,C u - 2\,\lambda u \;=\; 0
\qquad \Longrightarrow \qquad
C u \;=\; \lambda u
$$

The gradient identity $\partial (u^\top C u) / \partial u = 2\,C u$ holds because $C$ is symmetric; for a general matrix $A$ the gradient is $(A + A^\top)\, u$, and the factor of two would not collapse.

This one line converts the optimization into linear algebra. The candidates for the best direction are not arbitrary vectors to be searched; they are exactly the eigenvectors of the covariance matrix, a finite list that the eigendecomposition produces outright.

## 4. The Choice Among Stationary Points

$C u = \lambda u$ is satisfied by every eigenvector, so it marks all the stationary points and does not by itself name the maximum. One more line decides. Left-multiplying the condition by $u^\top$ and using $u^\top u = 1$:

$$
u^\top C\, u \;=\; \lambda\, u^\top u \;=\; \lambda
$$

At a stationary point the objective equals its own eigenvalue. Maximizing is therefore nothing more than picking the eigenvector with the largest eigenvalue, and the variance along that first principal direction is $\lambda_1$ itself.

Two facts fall out of this line rather than being imposed. The eigenvalue is a variance, which is why $\lambda_j$ is called the explained variance and carries the units of the data squared. And sorting the eigenvalues in decreasing order sorts the directions by how much variation each one carries, which is where the ordering of the components comes from. Panel (b) of Fig 1 is this section drawn: as $u$ rotates, $u^\top C u$ peaks at the angle of $u_1$ with height $\lambda_1$ and bottoms out at the angle of $u_2$ with height $\lambda_2$.

## 5. The Later Components

The second component answers the same question among the directions not yet used: maximize $u^\top C u$ subject to $u^\top u = 1$ and $u^\top u_1 = 0$. The second constraint gets its own multiplier:

$$
L(u, \lambda, \varphi) \;=\; u^\top C\, u \;-\; \lambda \left( u^\top u - 1 \right) \;-\; \varphi\, u^\top u_1
$$

$$
\frac{\partial L}{\partial u} \;=\; 2\,C u - 2\,\lambda u - \varphi\, u_1 \;=\; 0
$$

Left-multiplying by $u_1^\top$ kills every term but one. $u_1^\top u = 0$ by the constraint, and $u_1^\top C u = (C u_1)^\top u = \lambda_1 u_1^\top u = 0$ by the symmetry of $C$, so the equation collapses to $\varphi = 0$ — the new constraint costs nothing at the optimum — and the condition is $C u = \lambda u$ again. The second component is once more an eigenvector, the best one still admissible: the eigenvector of the second-largest eigenvalue. Repeating the argument with orthogonality to all earlier picks makes the j-th component the eigenvector of the j-th largest eigenvalue, so the single eigendecomposition delivers the entire ordered basis at once.

The orthogonality was not a stylistic preference. Eigenvectors of a symmetric matrix belonging to distinct eigenvalues are orthogonal on their own, so the constraint and the answer agree. When two eigenvalues are equal the picture degrades honestly: any rotation of the pair inside their shared plane satisfies every condition equally well, so the individual directions stop being determined even though the plane they span still is.

There is also a fixed budget being divided. The trace of $C$ is the sum of the individual variable variances and equals $\sum_j \lambda_j$, so each component claims a share of a total that no choice of basis can change. That total is the denominator of every explained variance ratio.

One more object falls out of this section. Stacking the kept components as columns gives the loadings matrix $V = [\, u_1 \cdots u_k \,]$ of shape $p \times k$, and the two facts in hand — the unit length imposed in §2 and the orthogonality just proved — compress into the single statement $V^\top V = I_k$. A matrix with that property is called column-wise orthonormal, or semi-orthogonal; the unqualified name orthogonal matrix is reserved for the square case $k = p$, where $V^\top = V^{-1}$ and the change of basis is a pure rotation that preserves every length and angle. The reversed product is a different object altogether: $V V^\top$ is the $p \times p$ projection onto the span of the kept components — the $P$ that §6 is about to use — and it cannot equal $I_p$ while $k \lt p$, because its rank is only $k$. [Appendix C](#appendix-c-the-column-wise-orthonormal-matrix) lays the two products side by side.

## 6. The Reconstruction View

PCA is often introduced by a different requirement: choose the k-dimensional subspace that minimizes the average squared distance between each sample and its projection. The two requirements meet in one identity. With $P$ the projection onto any subspace, each sample splits at a right angle:

$$
\lVert x_i \rVert^2 \;=\; \lVert P x_i \rVert^2 \;+\; \lVert x_i - P x_i \rVert^2
$$

Summing over the samples and dividing by $n - 1$, exactly as the variance does, turns this into an accounting rule: the total variance equals the variance kept in the subspace plus the mean squared reconstruction error. The left side is fixed by the data, so minimizing the error and maximizing the kept variance are the same act, and both are solved by the eigenvectors already found. Keeping the first $k$ components keeps $\lambda_1 + \cdots + \lambda_k$ and loses exactly $\lambda_{k+1} + \cdots + \lambda_p$, which is why the sum of the discarded eigenvalues is not merely correlated with the reconstruction error but equal to it.

## 7. The Bridge To The SVD

Implementations rarely form $C$, and the derivation explains what they do instead. Take the thin SVD of the centered data, $X_c = U S V^\top$, with the singular values $s_j$ on the diagonal of $S$. Then:

$$
C \;=\; \frac{X_c^\top X_c}{n-1} \;=\; V\, \frac{S^2}{n-1}\, V^\top
$$

This is already an eigendecomposition: the columns of $V$ are the eigenvectors the derivation demands, with eigenvalues $\lambda_j = s_j^2 / (n - 1)$, and the $V$ the SVD returns is exactly the column-wise orthonormal matrix that §5 built one column at a time. Decomposing $X_c$ directly therefore returns the same axes without ever building the covariance matrix, and it is the preferred route because forming $C$ squares the condition number of the problem before the eigensolver starts.

## 8. A Worked Example

Every claim above can be checked by hand on three samples of two variables.

$$
\begin{aligned}
\text{samples} \quad & (4, 2),\ (0, 0),\ (2, 4) \\
\text{mean} \quad & \mu = (2, 2) \\
\text{centered} \quad & x_1 = (2, 0),\ x_2 = (-2, -2),\ x_3 = (0, 2)
\end{aligned}
$$

Summed over the three centered samples, the squares of the first variable give 8, the squares of the second give 8, and the cross products give 4, so with $n - 1 = 2$:

$$
C = \begin{bmatrix} 4 & 2 \\ 2 & 4 \end{bmatrix}
$$

The eigenvalues solve $\det(C - \lambda I) = (4 - \lambda)^2 - 4 = 0$, so $4 - \lambda = \pm 2$ and $\lambda_1 = 6$, $\lambda_2 = 2$. Substituting back, $(C - 6I)\, u = 0$ forces equal entries, and $(C - 2I)\, u = 0$ forces opposite ones:

$$
u_1 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}
\qquad\qquad
u_2 = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ -1 \end{bmatrix}
$$

The two are orthogonal without being asked to be, as §5 promised, and this $C$ is the one drawn in Fig 1. The remaining claims each reduce to one line of arithmetic.

Table 2. The claims checked against the example

| Claim | Check | Result |
|---|---|---|
| The variance along $u_1$ equals $\lambda_1$ (§4) | The scores $u_1^\top x_i$ are $\sqrt{2}, -2\sqrt{2}, \sqrt{2}$, with variance $(2 + 8 + 2) / 2$ | $6 = \lambda_1$ |
| The total variance equals $\sum \lambda_j$ (§5) | The trace of $C$ is $4 + 4$ | $8 = 6 + 2$ |
| The discarded $\lambda$ equals the reconstruction error (§6) | Keeping only $u_1$, the residuals are the coordinates along $u_2$, namely $\sqrt{2}, 0, -\sqrt{2}$, with mean square $(2 + 0 + 2) / 2$ | $2 = \lambda_2$ |

The explained variance ratio of the first component is $6 / 8 = 0.75$, and the example is small enough that all of these numbers survive being recomputed on paper. [Appendix B](#appendix-b-the-worked-example-drawn) draws the same numbers in both coordinate systems.

---

## Appendix A. Terminology

The terms below appear in the body without being defined there.

- **Condition number** measures how much a matrix amplifies relative error, so a large one means fewer reliable digits in the result.
- **Covariance matrix** holds the pairwise covariances of the variables, with the variances on its diagonal.
- **Eigendecomposition** writes a symmetric matrix as its eigenvectors scaled by its eigenvalues.
- **Eigenvector and eigenvalue** are a vector that a matrix maps to a multiple of itself, and that multiple.
- **Gradient** is the vector of partial derivatives of a function, which vanishes where the function is flat.
- **Identity matrix** $I_k$ is the $k \times k$ matrix with ones on the diagonal and zeros elsewhere, which maps every vector to itself.
- **Lagrange multiplier** is the extra variable that folds a constraint into an objective so that a constrained optimum appears as a stationary point.
- **Orthonormal** describes a set of vectors that each have unit length and are mutually orthogonal.
- **Projection** is the component of a vector along a direction or inside a subspace.
- **Rank** is the number of linearly independent directions a matrix spans.
- **Span** is the set of all linear combinations of a set of vectors.
- **Stationary point** is a point where the gradient vanishes, which covers maxima, minima, and saddle points alike.
- **SVD** decomposes a matrix into left singular vectors, singular values, and right singular vectors.
- **Thin SVD** returns only the singular vectors that correspond to non-zero singular values.
- **Trace** is the sum of the diagonal entries of a matrix, which equals the sum of its eigenvalues.

## Appendix B. The Worked Example Drawn

![Fig 2](pca-derivation-fig2.png)

Fig 2. The worked example of §8 in pictures. Panel (a) shows the three samples, their mean $\mu = (2, 2)$, and the centering that moves each sample by $-\mu$ so the cloud sits about the origin. Panel (b) shows the centered samples with the axis $u_1$ and the direction $u_2$; the open circles are the projections onto $u_1$, whose positions along the axis are the scores $\sqrt{2}, -2\sqrt{2}, \sqrt{2}$, and the dashed segments are the residuals, whose average square $(2 + 0 + 2) / 2 = 2$ equals the discarded $\lambda_2$.

Every number in Table 2 has a visible counterpart here. The scores are where the open circles sit along $u_1$, the spread of those circles is the $\lambda_1 = 6$ the derivation maximized, and the dashed segments are what the one-component reconstruction gives up. Two details reward a second look. $x_2$ lies on the axis already, so keeping one component reconstructs it exactly. And $x_1$ and $x_3$ land on the same projected point — two different samples that the one-component summary can no longer tell apart, which is what losing $\lambda_2$ means in concrete terms.

## Appendix C. The Column-wise Orthonormal Matrix

The components of §5 are usually handled not one at a time but stacked into a single matrix, one component per column:

$$
V \;=\;
\begin{bmatrix}
\vert & \vert & & \vert \\
u_1 & u_2 & \cdots & u_k \\
\vert & \vert & & \vert
\end{bmatrix}
\;\in\; \mathbb{R}^{\,p \times k}
$$

Every entry of $V^\top V$ is a product of one column with another, $(V^\top V)_{ij} = u_i^\top u_j$, so the unit lengths of §2 fill the diagonal with ones and the orthogonality of §5 fills everything off the diagonal with zeros. No such argument exists for the rows, and the two products come out asymmetric:

$$
\underbrace{\; V^\top V = I_k \;}_{\textbf{column test:} \text{ always passes}}
\qquad\qquad
\underbrace{\; V V^\top \neq I_p \;}_{\textbf{row test:} \text{ fails whenever } k \lt p}
$$

The asymmetry is why the naming is column-wise. A rectangular $V$ that passes only the column test is called a column-wise orthonormal matrix, or a semi-orthogonal matrix; the unqualified name orthogonal matrix is reserved for the square case $k = p$, where both tests pass at once and $V^\top = V^{-1}$. The failed row test is not a defect but the point of the truncation: $V V^\top$ is the projection onto the span of the kept components — the $P$ of §6 — and a projection that changed nothing would have compressed nothing.

Table 3. The two products of the loadings matrix

| Product | Shape | Value | Reading |
|---|---|---|---|
| $V^\top V$ | $k \times k$ | $I_k$ always | The columns are unit length and mutually orthogonal |
| $V V^\top$ | $p \times p$ | $I_p$ only when $k = p$ | The projection onto the span of the kept components, with rank $k$ |

The worked example of §8 shows both cases in the smallest possible numbers. Keeping both components makes $V$ square, and both tests pass:

$$
V = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}
\qquad
V^\top V = I_2
\qquad
V V^\top = I_2
$$

Keeping only $u_1$ leaves the column test intact while the row test produces the projection that drew the open circles of Fig 2:

$$
V = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 \\ 1 \end{bmatrix}
\qquad
V^\top V = \begin{bmatrix} 1 \end{bmatrix}
\qquad
V V^\top = \frac{1}{2} \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix} \neq I_2
$$

Applying that last $V V^\top$ to $x_1 = (2, 0)$ gives $(1, 1)$, which is exactly where the dashed segment of Fig 2 lands — the matrix that fails the row test is the same map that sent $x_1$ and $x_3$ to their shared projection.
