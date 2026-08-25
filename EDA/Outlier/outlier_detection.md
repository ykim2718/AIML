# Outlier Detection Methods
Rev. 11 | Created: 2026-08-25 | Updated: 2026-08-25 21:26 CDT

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

Sections 2 to 4 take the three families in turn and section 5 puts the choice in one table.
[Appendix B. Semiconductor Practice](#appendix-d-semiconductor-practice) then reads two standard
industrial rules against them.

## 2. Statistical Methods

These assume a distributional form and measure departure from it. They are cheapest to compute
and easiest to defend, and they are the right default whenever their assumption holds.

### 2.1. Z-Score

The z-score divides the deviation of an observation from the sample mean by the sample standard
deviation.

$$z_i = \frac{x_i - \bar{x}}{s}$$

- $z_i$ — the z-score of observation $i$.
- $x_i$ — the $i$-th observation of a sample of $n$ values.
- $\bar{x}$ (x bar) — the mean of that sample.
- $s$ — its standard deviation, formed by dividing the sum of squared deviations by $n-1$.

An absolute score above 3 is the conventional flag. The rule assumes normality, under which about
0.27% of observations exceed 3 by chance alone.

One property limits it in two ways. The mean and the standard deviation are both computed from
the sample under test, so an outlier inflates the scale it is measured against and masks itself.
That same self-reference caps the score: in a sample of size $n$ no absolute z-score can exceed
$(n-1)/\sqrt{n}$, a bound due to Shiffler (1988), which means a rule at 3 cannot fire below 11
observations and a rule at 3.5 cannot fire below 15.

### 2.2. Interquartile Range

The interquartile range is the distance from the first quartile to the third. Tukey's rule
retains an observation that falls inside the interval below and flags one that falls outside it.
The two ends are the fences drawn by the whiskers of a box plot.

$$\left[ \ Q_1 - 1.5 \cdot \mathrm{IQR}, \quad Q_3 + 1.5 \cdot \mathrm{IQR} \ \right], \qquad \mathrm{IQR} = Q_3 - Q_1$$

- $Q_1$ — the first quartile, the value a quarter of the sample falls below.
- $Q_3$ — the third quartile, the value three quarters of the sample falls below.
- $\mathrm{IQR}$ — the distance between them, which is the spread of the middle half.

Quartiles are order statistics, so the rule needs no distributional assumption and carries a
breakdown point of 25% against the 0% of the z-score. On a normal sample of standard deviation
$\sigma$ the range itself is $1.349\,\sigma$, which puts the fences at $\pm 2.7\,\sigma$ and admits
roughly 0.7% of observations. The rule is therefore comparable in strictness to a z-score at 3,
while surviving contamination that would defeat that score.
[Appendix B. Tukey's Rule](#appendix-b-tukeys-rule) works the comparison out and gives the second
fence the rule is usually drawn with.

### 2.3. Hampel Identifier

The Hampel identifier keeps the form of the z-score and replaces both of its estimates. The
median takes the place of the mean, and the median of the deviations from it, rescaled, takes the
place of the standard deviation.

$$\mathrm{MAD} = \mathrm{median}\left( \left| x_1 - \tilde{x} \right|, \ldots, \left| x_n - \tilde{x} \right| \right)$$

$$M_i = \frac{x_i - \tilde{x}}{\mathrm{MAD} / \Phi^{-1}(0.75)}$$

- $x_1, \ldots, x_n$ — the sample, and $x_i$ its $i$-th observation, as in section 2.1.
- $\tilde{x}$ (x tilde) — the median of the sample, which the deviations are taken from and which the score is centred on.
- $\mathrm{MAD}$ — the median of those absolute deviations, which is the raw robust scale before any rescaling.
- $\Phi^{-1}(0.75) = 0.674490$ — the third quartile of the standard normal distribution, which the MAD is divided by.
- $M_i$ — the modified z-score of observation $i$, read on the same scale as $z_i$ of section 2.1.

That divisor is a consistency constant, and it is there because the raw MAD is not an estimate of
$s$. On a normal sample the MAD converges to $0.674490\,\sigma$ rather than to $\sigma$, so it
understates the spread by about a third, and a score built on it would sit on a scale of its own.
Dividing by the constant, which is the same as multiplying by 1.482602, puts $M_i$ on the scale
$z_i$ is read on.
[Appendix C. Consistency Constant of the MAD](#appendix-c-consistency-constant-of-the-mad) derives
the number and shows what a short sample does to it. Without it the
threshold could not be carried between the two rules or read as a false positive rate.

The constant is a calibration rather than an assumption, and it is the only place normality
enters the method. Changing it rescales every score by the same factor and reorders nothing, so
what it fixes is where a threshold sits, not which observations are extreme.

An absolute score above 3.5 is the conventional flag, the value recommended by Iglewicz and
Hoaglin (1993). Because neither the median nor the MAD can be moved by a minority, the identifier
reaches the outliers that mask themselves in section 2.1, and it needs no iteration: every
observation is scored once, against estimates that were never contaminated.

The failure mode is a tie rather than a distribution. If more than half the sample takes one
value the MAD is 0 and no score is defined, and no estimator at the same 50% breakdown point
escapes it.

### 2.4. Generalized ESD

Testing a sample for one outlier and then repeating the test on what is left does not hold its
significance level. The generalized extreme studentized deviate procedure fixes that by declaring
an upper bound $r$ on the number of outliers first, then running $r$ stages of the same statistic.

$$R_i = \frac{\max_j \left| x_j - \bar{x}_i \right|}{s_i}, \qquad i = 1, \ldots, r$$

- $R_i$ — the extreme studentized deviate at stage $i$.
- $x_j$ — an observation of the sample, indexed by $j$ to keep it apart from the stage number.
- $\bar{x}_i$ and $s_i$ — the mean and the standard deviation of what remains of the sample once the $i-1$ observations removed at earlier stages are gone.
- $\max_j$ — a maximum over the observations still remaining. The one attaining it is removed before stage $i+1$.
- $r$ — the declared upper bound on the number of outliers, fixed before the data are read.

Each $R_i$ is compared against a critical value $\lambda_i$ derived for that stage and tabulated by
Rosner (1983). The count of outliers is the **largest** $i$ for which $R_i \gt \lambda_i$, not the
first. Reading it that way is what defeats masking: a stage can fail while a later stage, with the
masking observation already removed, succeeds. The procedure is the many-outlier method of
ISO 16269-4, and it assumes the uncontaminated part of the sample is approximately normal.

### 2.5. Mahalanobis Distance

For multivariate data the Mahalanobis distance measures how far an observation lies from the
centre in units that account for the covariance between variables.

$$d^2(x) = \left( x - \mu \right)^{T} \Sigma^{-1} \left( x - \mu \right)$$

- $x$ — one observation, written as a vector with one entry per variable.
- $\mu$ — the centre of the sample, the vector of per-variable means.
- $\Sigma$ — the covariance matrix of the variables, and $\Sigma^{-1}$ its inverse.
- $d^2(x)$ — the squared distance, which reduces to $z_i^2$ of section 2.1 when there is one variable.

The covariance term is what makes it more than a per-variable check: an observation ordinary in
every single variable can still be implausible in their combination, and only a method reading the
correlation structure will see it. Under multivariate normality, and with the centre and the
covariance known rather than estimated from the sample, $d^2$ follows a chi-square distribution
with as many degrees of freedom as there are variables, which is what supplies the cut-off.

The same self-reference returns, and more severely. Both $\mu$ and $\Sigma$ are estimated from the
contaminated sample, and a cluster of outliers inflates $\Sigma$ in exactly the direction that
hides them. A robust estimate of the pair, such as the minimum covariance determinant of
Rousseeuw and Van Driessen (1999), makes the distance usable on data that may already be
contaminated.

## 3. Machine Learning Methods

These drop the distributional assumption and learn the extent of the normal region from unlabelled
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

### 3.4. Empirical Cumulative Distribution

ECOD takes the view that an outlier is a rare event in a tail, and measures tail rarity without
fitting anything. It builds the empirical cumulative distribution of each variable separately,
reads off the left and right tail probability of every observation, and aggregates those
probabilities across variables into one score.

It is the entry with no hyperparameter to set. Every other method in this section has a
neighbourhood size, a kernel, or a contamination rate that changes the answer, and none of those
can be tuned against labels that do not exist. It is also linear in
both the sample size and the variable count, and the per-variable tail probabilities say which
variables made an observation extreme.

The cost of reading each variable on its own is that ECOD, like a per-variable check, cannot see a
departure that exists only in the combination of variables. That is the case section 2.5 covers.

## 4. Deep Learning Methods

These learn a representation of normal data and read the departure off the failure to reproduce
it. They are for data whose structure defeats a distance in the raw coordinates, such as audio,
long time series, machine traces and above all images, and they need enough clean data to train
on. Images are the case with an established recipe of its own, which is section 4.3.

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

AnoGAN, the first method of this kind, combines that residual with a discriminator feature term.
Its cost is that scoring a single observation requires an iterative search in the latent space
rather than one forward pass, which is what later variants set out to remove.

Diffusion models have since taken the same role, learning to denoise normal data and scoring an
observation by how far the denoising has to move it. They report gains over autoencoder baselines
on tabular benchmarks, at a training and inference cost higher again than the adversarial methods.

### 4.3. Industrial Image Inspection

Visual defect inspection has converged on one recipe: run a pretrained network over the image,
keep the patch features of defect-free examples, and score a new patch by its distance to that
memory. PatchCore established the recipe and reports up to 99.1% detection AUROC on the MVTec AD
benchmark, and EfficientAD reaches 95.4% detection AUROC across 32 datasets at 2.2 ms per image,
which is the latency that makes inline inspection possible.

Those figures belong to the benchmark rather than to a factory. On the harder MVTec AD 2 set,
built to carry the lighting and defect variation of real inspection, no published method exceeds
31% localization AU-PRO at a 5% false positive rate. A method that separates the benchmark
cleanly is not thereby ready for a line.

## 5. Selection

### 5.1. By the Shape of the Data

**Table 1. Method by the shape of the data**

| Data | Method | Why |
|---|---|---|
| One variable, distribution unknown | Interquartile Range | It assumes no shape, and the fences carry a breakdown point of 25%. |
| One variable, approximately normal, clean | Z-Score | The threshold carries a stated error rate, provided the sample is large enough for the ceiling of section 2.1 to sit above it. |
| One variable, contamination expected | Hampel Identifier | The median and the MAD are not moved by the outliers being looked for, so nothing masks itself. |
| One variable, several outliers, approximately normal | Generalized ESD | It states a level for the whole search rather than for one test, and reads the last passing stage rather than the first. |
| A few variables, correlated | Mahalanobis Distance | It is the only entry that reads the covariance, and it needs a robust centre and scale to be trusted. |
| Many variables, no labels to tune against | ECOD | It is the one entry with no hyperparameter at all, and it says which variables made an observation extreme. |
| Many variables, and many observations | Isolation Forest | It is linear in the sample size, works on subsamples, and assumes no distribution. |
| Clusters of differing density | Local Outlier Factor | It compares an observation against its neighbourhood rather than the whole sample. |
| A known region, new points to test | One-Class SVM | The problem is a boundary, which is what the method fits. |
| Audio, long time series, machine traces | Autoencoder | Reconstruction error survives where a distance in raw coordinates does not. |
| Images of a repeated product | Patch feature memory of section 4.3 | The pretrained features already carry what a defect looks like, and scoring is fast enough to run inline. |
| Parts within a production lot | [Part average testing](#appendix-d-semiconductor-practice) | A standard names the rule, so the limit can be audited rather than argued. |
| Equipment sensor traces | [Multivariate control chart](#appendix-d-semiconductor-practice) | Splitting the score into $T^2$ and $Q$ says which sensor to look at, not only that something moved. |

### 5.2. What the Benchmarks Report

The published comparisons do not name a winner. Across the 30 algorithms and 57 datasets of
ADBench, no unsupervised method is statistically superior to the rest. Isolation Forest and ECOD
are consistently among the better performers without dominating, and several deep methods built
for tabular data fall below them. Newer is therefore not by itself a reason to switch, and the
cheap methods of sections 2 and 3 remain the honest default on tabular data.

The exception is the case where the raw coordinates carry no usable distance. That is where the
deep methods earn their cost, and images are the clearest instance of it.

### 5.3. Two Habits

Two habits matter more than the choice itself. Fix the threshold before the data are seen, so that
it is not tuned to produce a preferred answer. Then read the margin rather than the verdict, since
a statistic that passes its cut-off by a hair and one that passes by a wide gap are different
findings and only the second survives a change in the choices above.

## References

<a id="ref-1"></a>
[1] Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley, Reading. [https://www.pearson.com](https://www.pearson.com). ISBN 978-0-201-07616-5.

<a id="ref-2"></a>
[2] Brys, G., Hubert, M., & Struyf, A. (2004). [A Robust Measure of Skewness](https://doi.org/10.1198/106186004X12632). *Journal of Computational and Graphical Statistics*, 13(4), 996–1017.

<a id="ref-3"></a>
[3] Hubert, M., & Vandervieren, E. (2008). [An Adjusted Boxplot for Skewed Distributions](https://doi.org/10.1016/j.csda.2007.11.008). *Computational Statistics and Data Analysis*, 52(12), 5186–5201.

<a id="ref-4"></a>
[4] Mahalanobis, P. C. (1936). On the Generalised Distance in Statistics. *Proceedings of the National Institute of Sciences of India*, 2(1), 49–55. [https://www.insa.nic.in](https://www.insa.nic.in).

<a id="ref-5"></a>
[5] Shiffler, R. E. (1988). [Maximum Z Scores and Outliers](https://doi.org/10.1080/00031305.1988.10475530). *The American Statistician*, 42(1), 79–80.

<a id="ref-6"></a>
[6] Rosner, B. (1983). [Percentage Points for a Generalized ESD Many-Outlier Procedure](https://doi.org/10.1080/00401706.1983.10487848). *Technometrics*, 25(2), 165–172.

<a id="ref-7"></a>
[7] Iglewicz, B., & Hoaglin, D. C. (1993). *How to Detect and Handle Outliers*. The ASQC Basic References in Quality Control: Statistical Techniques, Vol. 16. ASQC Quality Press, Milwaukee. [https://asq.org/quality-press](https://asq.org/quality-press). ISBN 978-0-87389-247-6.

<a id="ref-8"></a>
[8] ISO 16269-4:2010, *Statistical interpretation of data — Part 4: Detection and treatment of outliers*. International Organization for Standardization. [https://www.iso.org/standard/44396.html](https://www.iso.org/standard/44396.html)

<a id="ref-9"></a>
[9] Rousseeuw, P. J., & Van Driessen, K. (1999). [A Fast Algorithm for the Minimum Covariance Determinant Estimator](https://doi.org/10.1080/00401706.1999.10485670). *Technometrics*, 41(3), 212–223.

<a id="ref-10"></a>
[10] Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). [LOF: Identifying Density-Based Local Outliers](https://doi.org/10.1145/335191.335388). *ACM SIGMOD Record*, 29(2), 93–104.

<a id="ref-11"></a>
[11] Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001). [Estimating the Support of a High-Dimensional Distribution](https://doi.org/10.1162/089976601750264965). *Neural Computation*, 13(7), 1443–1471.

<a id="ref-12"></a>
[12] Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). [Isolation Forest](https://doi.org/10.1109/ICDM.2008.17). *Proceedings of the Eighth IEEE International Conference on Data Mining*, 413–422.

<a id="ref-13"></a>
[13] Schlegl, T., Seeböck, P., Waldstein, S. M., Schmidt-Erfurth, U., & Langs, G. (2017). [Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://doi.org/10.1007/978-3-319-59050-9_12). *Information Processing in Medical Imaging*, Lecture Notes in Computer Science 10265, 146–157.

<a id="ref-14"></a>
[14] Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C., & Chen, G. H. (2022). [ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions](https://doi.org/10.1109/TKDE.2022.3159580). *IEEE Transactions on Knowledge and Data Engineering*, 35(12), 12181–12193.

<a id="ref-15"></a>
[15] Han, S., Hu, X., Huang, H., Jiang, M., & Zhao, Y. (2022). [ADBench: Anomaly Detection Benchmark](https://arxiv.org/abs/2206.09426). *Advances in Neural Information Processing Systems 35, Datasets and Benchmarks Track*.

<a id="ref-16"></a>
[16] Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P. (2022). [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265). *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 14318–14328.

<a id="ref-17"></a>
[17] Batzner, K., Heckler, L., & König, R. (2024). [EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies](https://arxiv.org/abs/2303.14535). *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, 128–138.

<a id="ref-18"></a>
[18] AEC-Q001 Rev-D (2011), *Guidelines for Part Average Testing*. Automotive Electronics Council. [http://www.aecouncil.com/AECDocuments.html](http://www.aecouncil.com/AECDocuments.html)

<a id="ref-19"></a>
[19] Hsu, C.-Y., Chien, C.-F., & Lin, K.-Y. (2012). [Semiconductor Fault Detection and Classification for Yield Enhancement and Manufacturing Intelligence](https://doi.org/10.1007/s10696-012-9161-4). *Flexible Services and Manufacturing Journal*, 24(3), 358–378.

---

## Appendix A. Terminology

- **adjusted boxplot** — A box plot whose fences are moved by the skewness of the sample, so that a long tail is not read as a stream of outliers.
- **anomaly score** — A number ranking observations by how far they depart from the normal pattern, without a stated error rate attached to any particular value of it.
- **AU-PRO** — Area under the per-region overlap curve, which scores how well a method localizes a defect rather than whether it detects one.
- **AUROC** — Area under the receiver operating characteristic curve, the probability that a randomly chosen anomaly is scored above a randomly chosen normal observation.
- **box plot** — A summary drawing in which a box spans the interquartile range and lines called whiskers reach the most extreme observation still within one and a half interquartile ranges of the box, with anything past that drawn as a separate point.
- **breakdown point** — The fraction of a sample that has to be corrupted before an estimate stops describing the rest of the data. The mean has 0% and the median has 50%.
- **chi-square distribution** — The distribution of a sum of squared independent standard normal variables, carrying one degree of freedom per term. It is what turns a squared distance into a probability.
- **consistency constant** — A factor applied to a robust scale estimate so that it converges to the standard deviation under an assumed distribution. It is 0.674490 for the MAD and 1.349 for the interquartile range, which AEC-Q001 rounds to 1.35.
- **contamination** — The fraction of a sample that does not come from the assumed distribution.
- **critical value** — The value a test statistic has to exceed to be called significant. It follows from the significance level and the sample size rather than from the data under test.
- **cumulative distribution function** — The function giving, for each value, the probability of falling at or below it. The standard normal one is written $\Phi$, and its inverse turns a probability back into a number of standard deviations.
- **degrees of freedom** — The number of independent quantities a statistic is free to vary over. It fixes which chi-square distribution a squared distance is read against, one per variable here.
- **discriminator** — The network trained alongside a generator to tell generated samples from real ones. Its internal features can be reused to compare an observation against what the generator produced.
- **ECOD** — Empirical-cumulative-distribution-based outlier detection, the method of section 3.4.
- **ESD** — Extreme studentized deviate, abbreviated in the name of the generalized ESD procedure of section 2.4.
- **extreme studentized deviate** — The largest absolute deviation from the sample mean, divided by the sample standard deviation. It is the statistic each stage of the generalized ESD procedure computes.
- **false positive rate** — The fraction of normal observations that a rule flags. It is the price paid for whatever detection rate the rule reaches.
- **generator** — The network trained to produce samples a discriminator cannot tell from the training data. Once trained it stands in for the distribution the normal data came from.
- **Hotelling's T-squared** — The multivariate analogue of a squared z-score, measuring the distance of an observation from the centre inside the structure a model has fitted.
- **hyperparameter** — A setting fixed before a method runs rather than estimated from the data, such as a neighbourhood size or a kernel bandwidth. Without labels there is nothing to tune one against.
- **interquartile range** — The distance from the first quartile to the third, which is the spread of the middle half of a sample. On a normal sample it is 1.349 standard deviations.
- **IQR** — The abbreviation used throughout for the interquartile range.
- **kernel** — The function that fixes the geometry a one-class SVM works in, and with it the shapes its learned boundary is allowed to take.
- **latent space** — The compressed coordinates a generative model maps to and from, in which a point stands for a whole reconstructed observation.
- **loading** — The weight a principal component gives to one original variable, which is what lets a flag raised in component space be traced back to a sensor.
- **lot** — A batch of parts processed together and carried through manufacturing as one unit. It is the group that part average testing judges a part against.
- **manifold** — The lower-dimensional surface inside the full space that the data actually occupy. A generative model that has learned it reproduces points on it and not points off it.
- **masking** — The effect by which an outlier inflates the centre or the scale it is measured against far enough that it, or a second outlier, no longer looks extreme.
- **medcouple** — A robust measure of skewness, between minus one and one and zero for a symmetric sample, built from a median of comparisons between observations on either side of the median.
- **median absolute deviation (MAD)** — The median of the absolute deviations of the observations from the sample median, used as a scale estimate that a minority of extreme observations cannot inflate.
- **minimum covariance determinant** — A robust estimate of a multivariate centre and covariance, taken from the subset of observations whose covariance matrix has the smallest determinant.
- **MVTec AD** — A public benchmark of photographs of manufactured objects, defect-free for training and defective for testing, with the defective region marked. MVTec AD 2 is a later set built to be harder.
- **order statistic** — An observation identified by its rank in the sorted sample rather than by its value, such as the median or a quartile. Moving an extreme observation further out does not move it.
- **outlier** — An observation inconsistent with the distribution the rest of the sample follows. The label concerns consistency with a model and does not by itself establish that the observation is wrong.
- **pretrained network** — A network fitted on a large general dataset and then used without further training, for the features its intermediate layers produce rather than for its own output.
- **principal component** — A direction fitted to the data along which the variance is largest, subject to being uncorrelated with the directions already fitted. A few of them usually carry most of the variation among correlated sensors.
- **reconstruction error** — The distance between an input and the output a model produces when it compresses and rebuilds that input.
- **robust** — Describing an estimate that a minority of contaminating observations cannot move far. The breakdown point says how large that minority may be.
- **significance level** — The probability of flagging an observation when the sample is in fact clean, fixed before the data are seen. Repeating a test without accounting for the repetition raises it above the value chosen.
- **specification limit** — The boundary a measured parameter must stay inside for a part to be sold, set from the design rather than from the sample. A part can pass it and still be an outlier within its lot.
- **squared prediction error (Q statistic)** — The part of an observation that a fitted model does not explain, measured as the squared distance from the observation to its reconstruction in the model's space.
- **SVM** — Support vector machine, a classifier that separates classes by the widest margin available in the geometry a kernel fixes. The one-class variant of section 3.2 has no second class and encloses the one it has instead.

## Appendix B. Tukey's Rule

Section 2.2 states the rule in one line. This appendix records where the multiple of 1.5 comes
from, what it costs against a z-score, and where the rule stops working.

### B.1. Inner and Outer Fences

Tukey (1977) drew two pairs of fences rather than one. The inner pair is the rule of section 2.2,
and the outer pair sits at three interquartile ranges instead of one and a half.

$$Q_1 - c \cdot \mathrm{IQR} \ \le \ x_i \ \le \ Q_3 + c \cdot \mathrm{IQR}$$

- $c$ — the multiple that places the fences, 1.5 for the inner pair and 3 for the outer pair.
- $Q_1$, $Q_3$, $\mathrm{IQR}$ — as in section 2.2.

An observation past an inner fence Tukey called **outside**, and one past an outer fence **far
out**. The box plot draws its whiskers to the last observation inside the inner fences, so
everything the plot shows as a separate point is at least outside.

The two fences are meant to be read together. Outside says the observation deserves a look, and
far out says it is extreme by any reading, which is a distinction a single cut-off cannot make.

### B.2. What the Multiple Costs

The multiple of 1.5 was chosen for convenience, not derived. What it does on a normal sample is
worth stating exactly, because section 2.2 calls the rule comparable to a z-score at 3 and the two
are not identical.

**Table 2. Where each fence sits on a normal sample**

| Rule | Position | Share of a normal sample flagged |
|---|---|---|
| Inner fence, $c = 1.5$ | 2.6980 $\sigma$ | 0.6977% |
| Outer fence, $c = 3$ | 4.7214 $\sigma$ | 0.0002% |
| Classical rule at 3 | 3.0000 $\sigma$ | 0.2700% |

The inner fence is therefore the looser of the two by a factor of 2.6, and a multiple of 1.724
rather than 1.5 would put it exactly where a z-score at 3 sits. The outer fence is stricter than
either by three orders of magnitude, which is why far out is a strong statement.

Comparable is the right word for the inner fence rather than equal. Both rules flag a fraction of
a percent where a rule at two standard deviations would flag five, and the choice between them
turns on contamination rather than on the fraction.

### B.3. Skewed Samples

The fences are placed symmetrically, at the same multiple below $Q_1$ as above $Q_3$. On a skewed
sample that is the wrong shape: the long tail is a property of the distribution, and the rule
reads it as a stream of outliers while flagging nothing on the short side.

A lognormal sample makes the size of the effect plain. With 200,000 draws the standard fences flag
6.22% of the sample above the upper fence and nothing at all below the lower one, on data with no
contamination in it whatever.

The adjusted boxplot of Hubert and Vandervieren (2008) repairs this by moving each fence according
to how skewed the sample is, measured by the medcouple of Brys, Hubert and Struyf (2004).

$$\left[ \ Q_1 - 1.5 \, e^{a \cdot \mathrm{MC}} \cdot \mathrm{IQR}, \quad Q_3 + 1.5 \, e^{b \cdot \mathrm{MC}} \cdot \mathrm{IQR} \ \right]$$

- $\mathrm{MC}$ — the medcouple, a robust skewness measure between $-1$ and $1$ that is 0 for a symmetric sample.
- $a$, $b$ — $-4$ and $3$ when $\mathrm{MC} \ge 0$, and $-3$ and $4$ when it is negative, so the fence on the long side moves out and the one on the short side moves in.

On the same lognormal sample the medcouple is 0.3264, and the adjusted fences flag 1.10% above and
0.42% below in place of 6.22% and nothing. The rule still flags more than a normal sample would
give, but it no longer reports the shape of the distribution as a list of outliers.

## Appendix C. Consistency Constant of the MAD

Section 2.3 divides the MAD by 0.674490 and says the raw MAD converges to that multiple of
$\sigma$ on a normal sample. This appendix derives the number, says how well a small sample
reaches it, and shows that it belongs to the normal distribution rather than to the MAD.

### C.1. Derivation

Take $X$ from a normal distribution with mean $\mu$ and standard deviation $\sigma$. Its median is
$\mu$, so the population MAD is the number $m$ with half the probability lying within $m$ of the
centre.

$$P\left( \left| X - \mu \right| \le m \right) = \frac{1}{2}$$

The normal is symmetric about $\mu$, so the probability inside that band is what lies below
$\mu + m$ less what lies below $\mu - m$, and the two are mirror images.

$$\Phi\left( \frac{m}{\sigma} \right) - \Phi\left( -\frac{m}{\sigma} \right) = 2\,\Phi\left( \frac{m}{\sigma} \right) - 1 = \frac{1}{2}$$

- $\Phi$ (capital phi) — the cumulative distribution function of the standard normal, so $\Phi(u)$ is the probability of falling below $u$ standard deviations.
- $m$ — the population MAD, the half-width of the band holding half the probability.

Rearranging leaves $\Phi(m/\sigma) = 3/4$, and inverting it leaves the constant.

$$m = \Phi^{-1}(0.75)\,\sigma = 0.674490\,\sigma$$

That is where the third quartile comes from, and it is worth saying plainly: **half the
probability inside $\pm m$ is the same statement as three quarters of it below $+m$.** The quartile
is not chosen for the MAD, it is what the MAD turns out to be.

Doubling the same number gives the constant of section 2.2, since the interquartile range is the
band from the first quartile to the third and each sits $0.674490\,\sigma$ from the centre.

$$\mathrm{IQR} = 2 \cdot 0.674490\,\sigma = 1.348980\,\sigma$$

The two rescalings of sections 2.2 and 2.3 are therefore the same constant, used once and twice.

### C.2. Finite Samples

The derivation is a statement about the distribution, not about a sample drawn from it. The sample
MAD converges to $m$, but in a small sample it lands low: the median of a short list of deviations
sits below the population median more often than above it, and rescaling does not repair that.

**Table 3. Mean of the rescaled MAD over 60,000 normal samples, true $\sigma = 1$**

| Sample size n | Mean of MAD / 0.674490 | Bias |
|---|---|---|
| 10 | 0.9129 | −8.7% |
| 15 | 0.9459 | −5.4% |
| 30 | 0.9739 | −2.6% |
| 100 | 0.9920 | −0.8% |
| 1000 | 0.9992 | −0.1% |

The bias runs one way, so the robust scale of a short sample is too small and every modified
z-score built on it is too large. At the fifteen observations a measurement run often supplies the
scale is understated by about 5%, which moves a score of 3.3 to 3.5 without anything having
happened to the data.

### C.3. Other Distributions

The constant is a property of the normal distribution and nothing else. Repeating the derivation
under another shape gives another number, and using 0.674490 there would rescale the MAD to
something that is not the standard deviation of anything.

**Table 4. The ratio of the population MAD to the standard deviation**

| Distribution | MAD / $\sigma$ | Value |
|---|---|---|
| Normal | $\Phi^{-1}(0.75)$ | 0.674490 |
| Laplace | $\ln 2 / \sqrt{2}$ | 0.490129 |
| Uniform | $\sqrt{3} / 2$ | 0.866025 |

This is the sense in which section 2.3 calls the constant the one place normality enters the
method. Detection itself does not need it, because the constant scales every score by the same
factor and reorders nothing. What needs it is reading a threshold as a false positive rate, and
that reading is only as good as the shape assumed here.

## Appendix D. Semiconductor Practice

The methods a fab actually runs are not the newest ones. They are the ones a standard names, an
auditor can check, and a technician can act on. Two of them are worth setting beside sections 2
to 4, because both turn out to be constructions already covered there.

### D.1. Part Average Testing

Part average testing removes parts whose measured parameters are abnormal for their own lot, even
when every measurement passes its specification limit. AEC-Q001 defines it for automotive
components, and it is built on the plan of section 2.3: the robust mean is the median, and the
robust sigma is the interquartile range divided by 1.35. A part is retained when it falls inside
the interval below.

$$\tilde{x} \pm k \cdot \frac{\mathrm{IQR}}{1.35}$$

- $\tilde{x}$ — the median of the parameter across the parts being judged, which the standard calls the robust mean.
- $\mathrm{IQR}$ — their interquartile range, and $\mathrm{IQR}/1.35$ is what the standard calls the robust sigma.
- $k$ — the multiple of that sigma the limits are set at, 6 by convention.

That divisor is the $1.349\sigma$ of section 2.2, rounded. Dividing by it turns a quartile spread
back into a standard deviation, exactly as $\Phi^{-1}(0.75)$ does for the MAD. The standard picks
the quartiles rather than the MAD and picks 6 rather than 3.5, but it is the same construction:
a robust centre, a robust scale rescaled to normal units, and a multiple of that scale.

Static limits are computed once from historical data and applied to every lot. Dynamic limits are
recomputed from each lot, which is what catches a lot that is uniformly shifted yet internally
tight, and it requires a minimum sample per lot, 30 parts in the standard, before the quartiles
mean anything.

### D.2. Fault Detection and Classification

Equipment sensors report pressure, flow, power and temperature throughout a process step. Fault
detection and classification reduces each trace to summary parameters per wafer, then monitors
those parameters together rather than one at a time, because the variables move together and a
per-variable limit misses a departure that only the combination shows.

The standard construction is the multivariate control chart of section 2.5 in a reduced space.
Principal components are fitted on normal production, an observation is scored by Hotelling's
$T^2$ inside that space, and by the squared prediction error, the $Q$ statistic, for the part of it
the components do not explain. The two answer different questions: $T^2$ says the process moved
within the structure it normally has, and $Q$ says it left that structure.

Splitting the score that way is what makes the flag actionable. The loading that contributes most
to a $T^2$ or a $Q$ names the sensor to look at, which is the difference between a chart that
stops a tool and a chart that also says why.
