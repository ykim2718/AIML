# Outlier Detection
Rev. 0 | Created: 2026-08-25 | Updated: 2026-08-25 16:14 CDT

> A survey of the methods that find observations departing from the pattern the rest of the data
> follows, arranged by what each one assumes, so that a method can be chosen from the shape of the
> data rather than from habit.

## 1. Scope

An outlier is an observation inconsistent with the model the rest of the sample follows. Finding
one is a statement about consistency with that model, not a verdict that the observation is wrong,
so detection and treatment stay separate: a flag opens an investigation rather than closing one.

No method here is universally correct, because each buys its answer with an assumption. A method
whose assumption the data violates still returns flags, and those flags then record the violation
rather than any real departure. Choosing therefore means matching four properties of the data
against what a method needs.

- **Dimension.** One variable, a handful, or a space too large for distances to stay meaningful.
- **Distribution.** Whether a parametric shape can be assumed, normality above all.
- **Labels.** Whether labelled examples of the departure exist. They rarely do, which is why every method below learns from unlabelled data.
- **Structure.** Whether the departure is defined against the whole sample or against a neighbourhood.

Sections 2 to 4 take the three families in turn, and section 5 puts the choice in one table.

## 2. Statistical Methods

These assume a distributional form and measure departure from it. They are cheapest to compute
and easiest to defend, and they are the right default whenever their assumption holds.

### 2.1. Z-Score

The z-score divides the deviation of an observation from the sample mean by the sample standard
deviation, and an absolute value above 3 is the conventional flag. The rule assumes normality,
under which about 0.27% of observations exceed 3 by chance alone.

Two properties limit it. The mean and the standard deviation are both computed from the sample
under test, so an outlier inflates the scale it is measured against and masks itself. That
self-reference also caps the score: in a sample of size $n$ no absolute z-score can exceed
$(n-1)/\sqrt{n}$, whatever the data are, which means a rule at 3 cannot fire below 11 observations
and a rule at 3.5 cannot fire below 15.

### 2.2. Interquartile Range

The interquartile range is the distance from the first quartile to the third. Tukey's rule flags
an observation below $Q_1 - 1.5 \cdot IQR$ or above $Q_3 + 1.5 \cdot IQR$, which is the fence
drawn by the whiskers of a box plot.

Quartiles are order statistics, so the rule needs no distributional assumption and carries a
breakdown point of 25%, where the z-score has none at all. Against a normal sample the fences sit
near $\pm 2.7\sigma$ and admit roughly 0.7% of observations, so the rule is comparable in
strictness to a z-score at 3 while surviving contamination that would defeat that score.

### 2.3. Mahalanobis Distance

For multivariate data the Mahalanobis distance measures how far an observation lies from the
centre in units that account for the covariance between variables.

$$d^2(x) = \left( x - \mu \right)^{T} \Sigma^{-1} \left( x - \mu \right)$$

The covariance term is what makes it more than a per-variable check: an observation ordinary in
every single variable can still be implausible in their combination, and only a method reading the
correlation structure will see it. Under multivariate normality, and with the centre and the
covariance known rather than estimated from the sample, $d^2$ follows a chi-square distribution
with as many degrees of freedom as there are variables, which is what supplies the cut-off.

The same self-reference returns, and more severely. Both $\mu$ and $\Sigma$ are estimated from the
contaminated sample, and a cluster of outliers inflates $\Sigma$ in exactly the direction that
hides them. A robust estimate of the pair, such as the minimum covariance determinant, is what
makes the distance usable on data that may already be contaminated.

## 3. Machine Learning Methods

These drop the distributional assumption and learn the shape of the normal region from unlabelled
data. They handle several variables at once and make no claim about an error rate, so their output
is a score to be ranked rather than a test to be passed.

### 3.1. Isolation Forest

Isolation Forest builds trees by splitting on a random variable at a random threshold, and records
how many splits an observation needs before it sits alone. An observation in a sparse region is
separated by few splits, so a short average path length across the forest is the anomaly score.

It isolates rather than profiles, which is what makes it fast: it never estimates a density or a
distance, runs in time linear in the sample size, and works on subsamples. That also makes it the
usual first choice when the data are large or have many variables.

### 3.2. One-Class SVM

One-Class SVM learns a boundary enclosing the region the training data occupy, and calls anything
outside that boundary an outlier. The kernel decides how the boundary may bend, and the parameter
$\nu$ sets an upper bound on the fraction of training data allowed to fall outside it.

The method fits a problem that is already stated as a boundary, where the question is whether a
new observation belongs to a known region. It costs more than the alternatives: the fit is
quadratic or worse in the sample size, and the answer depends on the kernel, its bandwidth, and
the scaling of the variables, none of which the data choose on their own.

### 3.3. Local Outlier Factor

Local Outlier Factor compares the density around an observation with the density around each of
its $k$ nearest neighbours. A factor near 1 means the observation is as densely surrounded as its
neighbours are, and a factor well above 1 means it sits in a sparser place than they do.

Because the comparison is local, the method finds an observation that is unremarkable against the
whole sample yet clearly apart from the group it belongs to. That is the case no global method
reaches, and it is the reason to pay for the neighbour search when the data hold clusters of
genuinely different densities.

## 4. Deep Learning Methods

These learn a representation of normal data and read the departure off the failure to reproduce
it. They are for data whose structure defeats a distance in the raw coordinates, such as images,
audio, long time series and machine traces, and they need enough clean data to train on.

### 4.1. Autoencoder

An autoencoder compresses its input to a narrow code and reconstructs the input from it. Trained
on normal data alone, it learns a representation that spends its capacity on normal structure, and
the reconstruction error then serves as the anomaly score.

The assumption is that the bottleneck is narrow enough to prevent the network from learning to
copy anything at all. Give it too much capacity and it reconstructs unseen anomalies as faithfully
as normal data, and the error separates nothing.

### 4.2. Generative Adversarial Network

An adversarial approach trains a generator to produce samples indistinguishable from normal data.
Scoring a new observation means finding the generated sample closest to it and reading the
residual: a normal observation lies on the manifold the generator learned and is matched closely,
while an anomalous one is not.

AnoGAN, the first method of this shape, combines that residual with a discriminator feature term.
Its cost is that scoring a single observation requires an iterative search in the latent space
rather than one forward pass, which is what later variants set out to remove.

## 5. Selection

**Table 1. Method by the shape of the data**

| Data | Method | Why |
|---|---|---|
| One variable, distribution unknown | Interquartile Range | It assumes no shape, and the fences carry a breakdown point of 25%. |
| One variable, approximately normal | Z-Score | The threshold carries a stated error rate, provided the sample is large enough for the ceiling of section 2.1 to sit above it. |
| A few variables, correlated | Mahalanobis Distance | It is the only entry that reads the covariance, and it needs a robust centre and scale to be trusted. |
| Many variables, or a large sample | Isolation Forest | It is linear in the sample size and assumes no distribution. |
| Clusters of differing density | Local Outlier Factor | It compares an observation against its neighbourhood rather than the whole sample. |
| A known region, new points to test | One-Class SVM | The problem is a boundary, which is what the method fits. |
| Images, audio, long time series | Autoencoder | Reconstruction error survives where a distance in raw coordinates does not. |

Two habits matter more than the choice itself. Fix the threshold before the data are seen, so that
it is not tuned to produce a preferred answer. Then read the margin rather than the verdict, since
a statistic that passes its cut-off by a hair and one that passes by a wide gap are different
findings and only the second survives a change in the choices above.

## References

<a id="ref-1"></a>
[1] Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley, Reading. [https://www.pearson.com](https://www.pearson.com). ISBN 978-0-201-07616-5.

<a id="ref-2"></a>
[2] Mahalanobis, P. C. (1936). On the Generalised Distance in Statistics. *Proceedings of the National Institute of Sciences of India*, 2(1), 49–55. [https://www.insa.nic.in](https://www.insa.nic.in).

<a id="ref-3"></a>
[3] Shiffler, R. E. (1988). [Maximum Z Scores and Outliers](https://doi.org/10.1080/00031305.1988.10475530). *The American Statistician*, 42(1), 79–80.

<a id="ref-4"></a>
[4] Rousseeuw, P. J., & Van Driessen, K. (1999). [A Fast Algorithm for the Minimum Covariance Determinant Estimator](https://doi.org/10.1080/00401706.1999.10485670). *Technometrics*, 41(3), 212–223.

<a id="ref-5"></a>
[5] Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). [LOF: Identifying Density-Based Local Outliers](https://doi.org/10.1145/335191.335388). *ACM SIGMOD Record*, 29(2), 93–104.

<a id="ref-6"></a>
[6] Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001). [Estimating the Support of a High-Dimensional Distribution](https://doi.org/10.1162/089976601750264965). *Neural Computation*, 13(7), 1443–1471.

<a id="ref-7"></a>
[7] Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). [Isolation Forest](https://doi.org/10.1109/ICDM.2008.17). *Proceedings of the Eighth IEEE International Conference on Data Mining*, 413–422.

<a id="ref-8"></a>
[8] Schlegl, T., Seeböck, P., Waldstein, S. M., Schmidt-Erfurth, U., & Langs, G. (2017). [Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://doi.org/10.1007/978-3-319-59050-9_12). *Information Processing in Medical Imaging*, Lecture Notes in Computer Science 10265, 146–157.

---

## Appendix A. Terminology

- **anomaly score** — A number ranking observations by how far they depart from the normal pattern, without a stated error rate attached to any particular value of it.
- **breakdown point** — The fraction of a sample that has to be corrupted before an estimate stops describing the rest of the data. The mean has 0% and the median has 50%.
- **contamination** — The fraction of a sample that does not come from the assumed distribution.
- **kernel** — The function that fixes the geometry a support vector method works in, and with it the shapes a learned boundary is allowed to take.
- **latent space** — The compressed coordinates a generative model maps to and from, in which a point stands for a whole reconstructed observation.
- **masking** — The effect by which an outlier inflates the centre or the scale it is measured against far enough that it, or a second outlier, no longer looks extreme.
- **minimum covariance determinant** — A robust estimate of a multivariate centre and covariance, taken from the subset of observations whose covariance matrix has the smallest determinant.
- **outlier** — An observation inconsistent with the distribution the rest of the sample follows. The label concerns consistency with a model and does not by itself establish that the observation is wrong.
- **reconstruction error** — The distance between an input and the output a model produces when it compresses and rebuilds that input.
