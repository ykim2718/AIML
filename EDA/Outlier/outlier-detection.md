# Outlier Detection Methods
Rev. 31 | Created: 2026-08-25 | Updated: 2026-09-23 11:08 CDT

- [1. Scope](#1-scope)
- [2. Taxonomy of Outliers](#2-taxonomy-of-outliers)
  - [2.1. Form](#21-form)
  - [2.2. Reference Set](#22-reference-set)
  - [2.3. Cause](#23-cause)
  - [2.4. Discordancy and Contamination](#24-discordancy-and-contamination)
  - [2.5. Position in a Regression](#25-position-in-a-regression)
  - [2.6. Labels](#26-labels)
  - [2.7. Count](#27-count)
  - [2.8. Time Series Type](#28-time-series-type)
- [3. Hierarchy of Methods](#3-hierarchy-of-methods)
  - [3.1. The Three Families](#31-the-three-families)
  - [3.2. Placement on the Axes](#32-placement-on-the-axes)
  - [3.3. Placement by the Shape of the Data](#33-placement-by-the-shape-of-the-data)
- [4. Statistical Methods](#4-statistical-methods)
  - [4.1. Z-Score](#41-z-score)
  - [4.2. Interquartile Range](#42-interquartile-range)
  - [4.3. Hampel Identifier](#43-hampel-identifier)
  - [4.4. Generalized ESD](#44-generalized-esd)
  - [4.5. Mahalanobis Distance](#45-mahalanobis-distance)
- [5. Machine Learning Methods](#5-machine-learning-methods)
  - [5.1. [Isolation Forest](#ref-19)](#51-isolation-forestref-19)
  - [5.2. [One-Class SVM](#ref-18)](#52-one-class-svmref-18)
  - [5.3. LOF (Local Outlier Factor)](#53-lof-local-outlier-factor)
  - [5.4. [ECOD](#ref-21) (Empirical Cumulative Distribution)](#54-ecodref-21-empirical-cumulative-distribution)
- [6. Deep Learning Methods](#6-deep-learning-methods)
  - [6.1. Autoencoder](#61-autoencoder)
  - [6.2. Generative Adversarial Network](#62-generative-adversarial-network)
  - [6.3. Industrial Image Inspection](#63-industrial-image-inspection)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Tukey's Rule](#appendix-b-tukeys-rule)
  - [B.1. Inner and Outer Fences](#b1-inner-and-outer-fences)
  - [B.2. What the Multiple Costs](#b2-what-the-multiple-costs)
  - [B.3. Skewed Samples](#b3-skewed-samples)
- [Appendix C. Semiconductor Practice](#appendix-c-semiconductor-practice)
  - [C.1. Part Average Testing](#c1-part-average-testing)
  - [C.2. Fault Detection and Classification](#c2-fault-detection-and-classification)
- [Appendix D. Estimator Constants and Bounds](#appendix-d-estimator-constants-and-bounds)
  - [D.1. The Ceiling on a Z-Score](#d1-the-ceiling-on-a-z-score)
  - [D.2. The MAD Consistency Constant](#d2-the-mad-consistency-constant)
  - [D.3. Critical Values of the Generalized ESD](#d3-critical-values-of-the-generalized-esd)
  - [D.4. The Chi-Square Cut-Off and a Robust Covariance](#d4-the-chi-square-cut-off-and-a-robust-covariance)
- [Appendix E. Benchmarks](#appendix-e-benchmarks)
  - [E.1. Tabular Data](#e1-tabular-data)
  - [E.2. Images](#e2-images)
- [Appendix F. What Practice Actually Runs](#appendix-f-what-practice-actually-runs)
  - [F.1. The Order Practice Meets Them](#f1-the-order-practice-meets-them)
  - [F.2. Two Habits](#f2-two-habits)

> Methods that find observations departing from the pattern the rest of the data follows,
> arranged by what each one assumes, so that a method follows from the shape of the data rather
> than from the rule practice reaches for first.

## 1. Scope

An outlier is an observation inconsistent with the model the rest of the sample follows. The flag
states inconsistency with that model, not error, so detection and treatment stay separate: a flag
opens an investigation rather than closing one.

Every method buys its answer with an assumption. Where the data violate it, the flags record the
violation rather than a departure. Two properties of the data fix which assumptions are available.

- **Dimension.** One variable, a handful, or a space too large for distances to stay meaningful.
- **Distribution.** Whether a parametric shape can be assumed, normality above all.

Section 2 sorts the kinds of outlier into the axes that name them, and section 3 sorts the methods
into the families that answer them. Sections 4 to 6 take those families in turn, each method stated
as what it assumes, what is set, what breaks it, and where it is met. The appendices carry the
derivations, the constants, the benchmark figures, the two industrial standards, and the order in
which practice meets the rules ([Appendix F](#appendix-f-what-practice-actually-runs)).

Fig 1 lays the two out.

```text
Outlier detection
|
+-- Taxonomy of outliers (section 2) - eight axes, one position on each at once
|   +-- Form ............ point | contextual | collective
|   +-- Reference set ... global | local
|   +-- Cause ........... error | foreign population | genuine rare event
|   +-- Discordancy ..... discordant | contaminant
|   +-- Regression ...... residual | leverage | influential
|   +-- Labels .......... supervised | semi-supervised | unsupervised
|   +-- Count ........... single | multiple
|   +-- Time series ..... additive | innovational | level shift | temporary change
|
+-- Hierarchy of methods (section 3) - each step gives up an assumption
    +-- Statistical (section 4) - assumes a distributional form, buys a stated error rate
    |   +-- z-score | interquartile range | Hampel | generalized ESD | Mahalanobis
    +-- Machine learning (section 5) - gives up the form, keeps the geometry
    |   +-- isolation forest | one-class SVM | local outlier factor | ECOD
    +-- Deep (section 6) - gives up the raw geometry, learns a representation
        +-- autoencoder | adversarial and diffusion | patch feature memory
```

<a id="fig-1"></a>
**Fig 1. The taxonomy of outliers and the hierarchy of methods**

## 2. Taxonomy of Outliers

The eight subsections are the eight axes of [Fig 1](#fig-1), not eight categories: an observation
has a position on every one at once. One measurement can be a point outlier, local rather than global, caused by a recording
error, discordant without being a contaminant, and high-leverage in its regression.

A method answers one axis and says nothing on the others. Section 3.2 places every method here on
these axes.

### 2.1. Form

[Chandola, Banerjee and Kumar (2009)](#ref-8) split anomalies by the form they take in the data.

- **Point.** One observation is extreme on its own. A temperature of minus 100 degrees in a record of ordinary weather.
- **Contextual.** The value is ordinary in the sample and extreme in its context. Two degrees is unremarkable in a year of readings and wrong for August.
- **Collective.** No single value is extreme, but a run of them together is. A stretch of low voltage in an electrocardiogram, where every reading is inside the normal range.

Contextual and collective anomalies need something the value does not carry, a context variable and
an ordering. A method reading only the marginal distribution finds neither.

### 2.2. Reference Set

An observation is extreme relative to the whole sample or to a neighbourhood.

- **Global.** Extreme against the sample as a whole.
- **Local.** Ordinary against the whole sample, extreme against the group it sits in.

This axis is independent of section 2.1, so pairing point with global as one label is a mistake.
[Breunig, Kriegel, Ng and Sander (2000)](#ref-17) built the local outlier factor for the local point
outlier, the case of section 5.3.

### 2.3. Cause

A flagged value arrives three ways, and what to do with it differs in each.

- **Error.** A mistake in measuring, transcribing or transmitting. The value describes the process that recorded it, not the process being studied.
- **Foreign population.** A correct measurement of something else, such as a part from another lot mixed into the batch.
- **Genuine rare event.** A correct measurement of the process under study, sitting in a tail it really has.

No statistic tells these apart. Detection produces a candidate, and the record behind it settles the
cause.

### 2.4. Discordancy and Contamination

[Barnett and Lewis (1994)](#ref-2) separate two things. A **contaminant** came from a different
distribution. A **discordant observation** looks statistically inconsistent with the rest.

Neither implies the other: a contaminant can hide inside the bulk, and a clean heavy-tailed sample
produces discordant observations at a predictable rate. Every test here tests discordancy.
Contamination is what the investigation of section 2.3 establishes.

### 2.5. Position in a Regression

Fitting a model rather than a distribution splits one axis into three, and the three come apart.

- **Residual outlier.** Far from the fitted surface in the response.
- **Leverage point.** Extreme in the predictors, which gives the observation the power to move the fit whether or not it does.
- **Influential observation.** Removing it changes the fit materially, which [Cook (1977)](#ref-4) measured with the distance that carries his name.

High leverage without influence is common, and so is influence without a large residual, where the
point has dragged the line onto itself. [Belsley, Kuh and Welsch (1980)](#ref-6) collect the
diagnostics that separate them.

### 2.6. Labels

What is known before the search starts fixes what can be done.

- **Supervised.** Labelled examples of both classes. This is a classification problem with severe class imbalance rather than an outlier problem.
- **Semi-supervised.** A training set known to be clean, with new observations to judge against it. This is novelty detection.
- **Unsupervised.** One unlabelled sample that may already contain outliers.

Sections 4 to 6 are unsupervised or semi-supervised, labelled outliers being rare. A training set
assumed clean and not clean teaches the method to treat its outliers as normal.

### 2.7. Count

How many outliers are expected changes the procedure, not just the threshold.

- **Single.** One test, one stated error rate.
- **Multiple.** Several, in unknown number, which is where masking and swamping appear. Both are defined below.

Masking is one outlier inflating the centre or the scale until a second no longer looks extreme.
Swamping is the reverse: the distortion is large enough that clean observations are flagged with it.
[Hawkins (1980)](#ref-5) treats the many-outlier problem, and section 4.4 is the procedure built for
it.

### 2.8. Time Series Type

With ordered data the form axis of section 2.1 refines into how the departure enters the series.
[Fox (1972)](#ref-3) introduced the first two, and [Chen and Liu (1993)](#ref-7) settled the
standard four.

- **Additive.** One reading is displaced and the series returns immediately.
- **Innovational.** A shock enters the process, so the displacement propagates through the readings that follow.
- **Level shift.** The series moves to a new level and stays there.
- **Temporary change.** The series moves and decays back over several readings.

All four are one collective anomaly under section 2.1, which is why that axis is too coarse for a
machine trace. A chamber that drifted permanently and one that recovered on its own differ as level
shift against temporary change.

## 3. Hierarchy of Methods

The three families of [Fig 1](#fig-1) form a hierarchy of assumptions given up. Each step down
handles data the step above cannot read, and each step down weakens what the answer may claim.

### 3.1. The Three Families

Statistical methods (section 4) assume a distributional form and measure departure from it, and that
assumption buys a stated error rate. Machine learning methods (section 5) give up the form and keep
the geometry of the data, which buys several variables at once and pays with a score that carries no
rate. Deep methods (section 6) give up the geometry of the raw coordinates and learn a
representation first, which buys data whose structure no distance reads and pays with clean training
data and cost.

The hierarchy is not a ranking. A method one step down answers a wider class of data and a narrower
class of questions, so the family is fixed by what the data are, and the axes of section 2 fix which
method inside that family is being asked for.

### 3.2. Placement on the Axes

Table 2 places every method of sections 4 to 6 on the axes of section 2, and says which of those
questions each one actually answers.

**Table 2. Where each method sits on the axes of section 2**

| # | Method | Form (2.1) | Reference set (2.2) | Labels (2.6) | Count (2.7) |
|---|---|---|---|---|---|
| 1 | Z-Score | Point | Global | Unsupervised | Single |
| 2 | Interquartile Range | Point | Global | Unsupervised | Uncontrolled |
| 3 | Hampel Identifier | Point | Global | Unsupervised | Uncontrolled, but masking cannot occur |
| 4 | Generalized ESD | Point | Global | Unsupervised | **Multiple, at a stated level** |
| 5 | Mahalanobis Distance | Point | Global | Unsupervised | Single |
| 6 | Isolation Forest | Point | Global | Unsupervised | Uncontrolled |
| 7 | One-Class SVM | Point | Global | Semi-supervised | Uncontrolled |
| 8 | Local Outlier Factor | Point | **Local** | Unsupervised | Uncontrolled |
| 9 | ECOD | Point | Global | Unsupervised | Uncontrolled |
| 10 | Autoencoder | Point or collective | Global | Semi-supervised | Uncontrolled |
| 11 | Adversarial and diffusion | Point or collective | Global | Semi-supervised | Uncontrolled |
| 12 | Patch feature memory | Collective in space | Global | Semi-supervised | Uncontrolled |

The two bold cells are the only departures. The local outlier factor alone changes the reference
set, and the generalized ESD alone controls how many outliers it looks for; single and uncontrolled
differ in what a method was built for rather than in any count it holds to.

The deep methods reach a collective anomaly by changing the data rather than the method. A window
over a series, or a patch of an image, becomes one vector, and the collective anomaly becomes a
point anomaly in it. Nothing in section 5 scores a run of observations directly.

Four axes are missing from the table, three because no method here answers them and one because
every method answers it the same way.

- **Cause (2.3).** No statistic separates an error from a rare event, which that section says outright.
- **Discordancy and contamination (2.4).** Every method here tests discordancy and none judges contamination, so the axis sorts how a result is read rather than which method produces it.
- **Position in a regression (2.5).** Leverage and influence need a fitted model, and this document fits distributions and regions instead.
- **Time series type (2.8).** A windowed method can flag a level shift, but nothing here tells a level shift from a temporary change.

A contextual anomaly is out of reach on the form axis as well: it needs a context variable, and no
method here takes one.

### 3.3. Placement by the Shape of the Data

Table 1 reads the same placement from the other side, from the shape of the data to the method whose
assumptions that shape satisfies.

**Table 1. Method by the shape of the data**

| # | Data | Method | Why |
|---|---|---|---|
| 1 | One variable, distribution unknown | Interquartile Range | It assumes no shape, and the fences carry a breakdown point of 25%. |
| 2 | One variable, approximately normal, clean | Z-Score | The threshold carries a stated error rate, provided the sample is large enough for the ceiling of section 4.1 to sit above it. |
| 3 | One variable, contamination expected | Hampel Identifier | The median and the MAD are not moved by the outliers being looked for, so nothing masks itself. |
| 4 | One variable, several outliers, approximately normal | Generalized ESD | It states a level for the whole search rather than for one test, and reads the last passing stage rather than the first. |
| 5 | A few variables, correlated | Mahalanobis Distance | It is the only entry that reads the covariance, and it needs a robust centre and scale to be trusted. |
| 6 | Many variables, no labels to tune against | ECOD | It is the one entry with no hyperparameter at all, and it says which variables made an observation extreme. |
| 7 | Many variables, and many observations | Isolation Forest | It is linear in the sample size, works on subsamples, and assumes no distribution. |
| 8 | Clusters of differing density | Local Outlier Factor | It compares an observation against its neighbourhood rather than the whole sample. |
| 9 | A known region, new points to test | One-Class SVM | The problem is a boundary, which is what the method fits. |
| 10 | Audio, long time series, machine traces | Autoencoder | Reconstruction error survives where a distance in raw coordinates does not. |
| 11 | Images of a repeated product | Patch feature memory | The pretrained features of section 6.3 already carry what a defect looks like, and scoring is fast enough to run inline. |
| 12 | Parts within a production lot | [Part average testing](#appendix-c-semiconductor-practice) | A standard names the rule, so the limit can be audited rather than argued. |
| 13 | Equipment sensor traces | [Multivariate control chart](#appendix-c-semiconductor-practice) | Splitting the score into $T^2$ and $Q$ says which sensor to look at, not only that something moved. |

## 4. Statistical Methods

The cheapest family to compute and the easiest to defend, and the right default while its
assumption holds.

### 4.1. Z-Score

The z-score divides the deviation of an observation from the sample mean by the sample standard
deviation.

```math
z_i = \frac{x_i - \bar{x}}{s}
```

- $z_i$ — the z-score of observation $i$.
- $x_i$ — the $i$-th observation of a sample of $n$ values.
- $\bar{x}$ (x bar) — the mean of that sample.
- $s$ — its standard deviation, formed by dividing the sum of squared deviations by $n-1$.

- **Assumption.** Normality, and a sample clean enough that the mean and the standard deviation are not moved by what is being looked for.
- **Setting.** An absolute score above 3, which about 0.27% of a normal sample exceeds by chance.
- **Breaks when.** Contamination is present, since the outlier inflates the scale it is measured against and masks itself, or the sample is small: the score has a ceiling of $(n-1)/\sqrt{n}$, so a rule at 3 cannot fire below 11 observations ([Appendix D](#appendix-d-estimator-constants-and-bounds)).
- **Met at.** A single parameter already known to be near normal, on data screened for gross errors.

### 4.2. Interquartile Range

Tukey's rule flags an observation outside the interval below. Its two ends are the fences the
whiskers of a box plot draw.

```math
\left[ \ Q_1 - 1.5 \cdot \mathrm{IQR}, \quad Q_3 + 1.5 \cdot \mathrm{IQR} \ \right], \qquad \mathrm{IQR} = Q_3 - Q_1
```

- $Q_1$ — the first quartile, the value a quarter of the sample falls below.
- $Q_3$ — the third quartile, the value three quarters of the sample falls below.
- $\mathrm{IQR}$ — the distance between them, which is the spread of the middle half.

- **Assumption.** No distributional shape, but symmetry: the fences are placed at the same multiple on both sides.
- **Setting.** An inner fence at 1.5 IQR and an outer fence at 3 IQR ([Appendix B](#appendix-b-tukeys-rule)).
- **Breaks when.** The sample is skewed. The long tail is a property of the distribution, and the rule reads it as a stream of outliers ([Appendix B](#appendix-b-tukeys-rule)).
- **Met at.** A first look at any single parameter, which is what the whiskers of a box plot already draw.

### 4.3. Hampel Identifier

The Hampel identifier keeps the form of the z-score and replaces both estimates: the median for the
mean, and the rescaled median of the deviations from it for the standard deviation.

```math
\mathrm{MAD} = \mathrm{median}\left( \left| x_1 - \tilde{x} \right|, \ldots, \left| x_n - \tilde{x} \right| \right)
```

```math
M_i = \frac{x_i - \tilde{x}}{\mathrm{MAD} / \Phi^{-1}(0.75)}
```

- $x_1, \ldots, x_n$ — the sample, and $x_i$ its $i$-th observation, as in section 4.1.
- $\tilde{x}$ (x tilde) — the median of the sample, which the deviations are taken from and which the score is centred on.
- $\mathrm{MAD}$ — the median of those absolute deviations, which is the raw robust scale before any rescaling.
- $\Phi^{-1}(0.75) = 0.674490$ — the third quartile of the standard normal distribution, which the MAD is divided by.
- $M_i$ — the modified z-score of observation $i$, read on the same scale as $z_i$ of section 4.1.

- **Assumption.** Nothing about the data. Normality enters only through the constant that puts the score on the z-score scale ([Appendix D](#appendix-d-estimator-constants-and-bounds)).
- **Setting.** An absolute modified z-score above 3.5, the value recommended by [Iglewicz and Hoaglin (1993)](#ref-14).
- **Breaks when.** More than half the sample takes one value, where the MAD is 0 and no score is defined.
- **Met at.** A single parameter judged within its own lot, where the outliers being looked for are in the sample.

### 4.4. Generalized ESD

Repeating a single-outlier test on what is left does not hold the significance level, since each
repetition spends it again. The generalized extreme studentized deviate procedure declares an upper
bound $r$ first, then runs $r$ stages of the same statistic, with the level stated for the whole
search.

```math
R_i = \frac{\max_j \left| x_j - \bar{x}_i \right|}{s_i}, \qquad i = 1, \ldots, r
```

- $R_i$ — the extreme studentized deviate at stage $i$.
- $x_j$ — an observation of the sample, indexed by $j$ to keep it apart from the stage number.
- $\bar{x}_i$ and $s_i$ — the mean and the standard deviation of what remains of the sample once the $i-1$ observations removed at earlier stages are gone.
- $\max_j$ — a maximum over the observations still remaining. The one attaining it is removed before stage $i+1$.
- $r$ — the declared upper bound on the number of outliers, fixed before the data are read.

- **Assumption.** The uncontaminated part of the sample is approximately normal, and an upper bound $r$ on the outlier count is fixed before the data are read.
- **Setting.** $r$, a significance level for the whole search, and the critical values $\lambda_i$ of [Rosner (1983)](#ref-13) ([Appendix D](#appendix-d-estimator-constants-and-bounds)).
- **Breaks when.** $r$ is set below the true count, or the stages are read from the first failure rather than the last passing one, which lets masking survive.
- **Met at.** A decision that has to be defended against a standard, which is the many-outlier method of [ISO 16269-4](#ref-15).

### 4.5. Mahalanobis Distance

For multivariate data the [Mahalanobis distance](#ref-11) measures how far an observation lies
from the centre in units that account for the covariance between variables.

```math
d^2(x) = \left( x - \mu \right)^{T} \Sigma^{-1} \left( x - \mu \right)
```

- $x$ — one observation, written as a vector with one entry per variable.
- $\mu$ — the centre of the sample, the vector of per-variable means.
- $\Sigma$ — the covariance matrix of the variables, and $\Sigma^{-1}$ its inverse.
- $d^2(x)$ — the squared distance, which reduces to $z_i^2$ of section 4.1 when there is one variable.

- **Assumption.** Multivariate normality, and a centre and covariance that the outliers have not moved.
- **Setting.** A chi-square cut-off with one degree of freedom per variable ([Appendix D](#appendix-d-estimator-constants-and-bounds)).
- **Breaks when.** $\mu$ and $\Sigma$ come from contaminated data, since a cluster of outliers inflates $\Sigma$ in the direction that hides it. A robust pair repairs it ([Appendix D](#appendix-d-estimator-constants-and-bounds)).
- **Met at.** A few correlated parameters read together, such as the summary parameters of one process step.

## 5. Machine Learning Methods

The normal region is learned from unlabelled data, so the output is a score to rank rather than a
test to pass.

### 5.1. [Isolation Forest](#ref-19)

Isolation Forest builds trees by splitting on a random variable at a random threshold, and records
how many splits an observation needs before it sits alone. An observation in a sparse region is
separated by few splits, so a short average path length across the forest is the anomaly score.

- **Assumption.** Outliers are few and sit in sparse regions. No distribution is assumed.
- **Setting.** The number of trees and the subsample size. The output is a rank, not an error rate.
- **Breaks when.** The departure exists only in the combination of variables, or a contamination rate is set from a guess.
- **Met at.** Wide tables with many rows, where a first ranking is wanted cheaply.

### 5.2. [One-Class SVM](#ref-18)

One-Class SVM learns a boundary enclosing the region the training data occupy, and calls anything
outside that boundary an outlier. The kernel decides how the boundary may bend, and the parameter
$\nu$ sets an upper bound on the fraction of training data allowed to fall outside it.

- **Assumption.** A training set known to be clean, which makes the problem novelty detection (section 2.6).
- **Setting.** A kernel, its bandwidth, and $\nu$, the upper bound on the training fraction outside the boundary.
- **Breaks when.** The training set was not clean, the variables are on different scales, or the sample is large: the fit is quadratic or worse in the sample size.
- **Met at.** A known-good region fixed earlier, with new material to judge against it.

### 5.3. LOF (Local Outlier Factor)

Local Outlier Factor compares the density around an observation with the density around each of
its $k$ nearest neighbours. A factor near 1 means the observation is as densely surrounded as its
neighbours are, and a factor well above 1 means it sits in a sparser place than they do.

- **Assumption.** Density is meaningful, so a distance in the raw coordinates is meaningful.
- **Setting.** The neighbourhood size $k$. A factor near 1 is ordinary and well above 1 is sparse.
- **Breaks when.** The variables are many enough that distances stop separating, or $k$ reaches across a cluster boundary.
- **Met at.** Data holding groups of different density, such as several tools or chambers pooled in one table.

### 5.4. [ECOD](#ref-21) (Empirical Cumulative Distribution)

ECOD takes the view that an outlier is a rare event in a tail, and measures tail rarity without
fitting anything. It builds the empirical cumulative distribution of each variable separately,
reads off the left and right tail probability of every observation, and aggregates those
probabilities across variables into one score.

- **Assumption.** An outlier is a tail event in at least one variable.
- **Setting.** None. It is the one method here with nothing to set.
- **Breaks when.** The departure exists only in the combination of variables, which is the case of section 4.5.
- **Met at.** Wide tables with no labels to tune against, where the flag has to name the variable responsible.

## 6. Deep Learning Methods

The departure is read off the failure to reproduce a learned representation. Audio, long time
series, machine traces and above all images are what these are for, and they need enough clean data
to train on. Images have a recipe of their own in section 6.3.

### 6.1. Autoencoder

An autoencoder compresses its input to a narrow code and reconstructs the input from it. Trained
on normal data alone, it learns a representation that spends its capacity on normal structure, and
the reconstruction error then serves as the anomaly score.

- **Assumption.** Clean training data, and a bottleneck narrow enough that the network cannot learn to copy its input.
- **Setting.** The code width, and a cut on the reconstruction error.
- **Breaks when.** Capacity is too large, so unseen anomalies are reconstructed as faithfully as normal data, or the training set was not clean (section 2.6).
- **Met at.** Equipment traces and other long ordered records that a per-point limit cannot read.

### 6.2. Generative Adversarial Network

An adversarial approach trains a generator to produce samples indistinguishable from normal data.
Scoring means finding the closest generated sample and reading the residual: a normal observation
lies on the learned manifold and is matched closely, an anomalous one is not.

[AnoGAN](#ref-20), the first of the kind, adds a discriminator feature term to that residual, and
scores one observation by an iterative search in the latent space rather than a forward pass.
Diffusion models have since taken the same role, scoring an observation by how far denoising has to
move it, at a cost higher again.

- **Assumption.** The same as section 6.1, with a generator that has learned the manifold the normal data occupy.
- **Setting.** The latent search budget for scoring, and a cut on the residual.
- **Breaks when.** The latent search does not converge, or the scoring cost per observation exceeds what the line allows.
- **Met at.** The data of section 6.1 where a generative model is already in place and its cost is accepted.

### 6.3. Industrial Image Inspection

Visual defect inspection has converged on one recipe: run a pretrained network over the image, keep
the patch features of defect-free examples, and score a new patch by its distance to that memory.
[Appendix E](#appendix-e-benchmarks) gives the reported detection rates and latencies.

- **Assumption.** Defect-free examples cover the normal appearance, and the product repeats.
- **Setting.** A pretrained backbone, the patch size, the memory size, and a distance cut.
- **Breaks when.** Lighting and defect variation exceed what the memory holds, where reported localization falls off sharply ([Appendix E](#appendix-e-benchmarks)).
- **Met at.** Inline image inspection of a repeated product, at the latency a line needs.

## References

<a id="ref-1"></a>
[1] Tukey, J. W. (1977). [*Exploratory Data Analysis*](https://www.pearson.com). Addison-Wesley, Reading. ISBN 978-0-201-07616-5.<br>
<a id="ref-2"></a>
[2] Barnett, V., & Lewis, T. (1994). [*Outliers in Statistical Data*](https://www.wiley.com/en-us/Outliers+in+Statistical+Data,+3rd+Edition-p-9780471930945), 3rd edition. Wiley, Chichester. ISBN 978-0-471-93094-5.<br>
<a id="ref-3"></a>
[3] Fox, A. J. (1972). [Outliers in Time Series](https://doi.org/10.1111/j.2517-6161.1972.tb00912.x). *Journal of the Royal Statistical Society: Series B*, 34(3), 350–363.<br>
<a id="ref-4"></a>
[4] Cook, R. D. (1977). [Detection of Influential Observation in Linear Regression](https://doi.org/10.1080/00401706.1977.10489493). *Technometrics*, 19(1), 15–18.<br>
<a id="ref-5"></a>
[5] Hawkins, D. M. (1980). [Identification of Outliers](https://doi.org/10.1007/978-94-015-3994-4). Monographs on Applied Probability and Statistics. Chapman and Hall, London. ISBN 978-94-015-3996-8.<br>
<a id="ref-6"></a>
[6] Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). [Regression Diagnostics: Identifying Influential Data and Sources of Collinearity](https://doi.org/10.1002/0471725153). Wiley, New York. ISBN 978-0-471-05856-4.<br>
<a id="ref-7"></a>
[7] Chen, C., & Liu, L.-M. (1993). [Joint Estimation of Model Parameters and Outlier Effects in Time Series](https://doi.org/10.1080/01621459.1993.10594321). *Journal of the American Statistical Association*, 88(421), 284–297.<br>
<a id="ref-8"></a>
[8] Chandola, V., Banerjee, A., & Kumar, V. (2009). [Anomaly Detection: A Survey](https://doi.org/10.1145/1541880.1541882). *ACM Computing Surveys*, 41(3), Article 15.<br>
<a id="ref-9"></a>
[9] Brys, G., Hubert, M., & Struyf, A. (2004). [A Robust Measure of Skewness](https://doi.org/10.1198/106186004X12632). *Journal of Computational and Graphical Statistics*, 13(4), 996–1017.<br>
<a id="ref-10"></a>
[10] Hubert, M., & Vandervieren, E. (2008). [An Adjusted Boxplot for Skewed Distributions](https://doi.org/10.1016/j.csda.2007.11.008). *Computational Statistics and Data Analysis*, 52(12), 5186–5201.<br>
<a id="ref-11"></a>
[11] Mahalanobis, P. C. (1936). [On the Generalised Distance in Statistics](https://www.insa.nic.in). *Proceedings of the National Institute of Sciences of India*, 2(1), 49–55.<br>
<a id="ref-12"></a>
[12] Shiffler, R. E. (1988). [Maximum Z Scores and Outliers](https://doi.org/10.1080/00031305.1988.10475530). *The American Statistician*, 42(1), 79–80.<br>
<a id="ref-13"></a>
[13] Rosner, B. (1983). [Percentage Points for a Generalized ESD Many-Outlier Procedure](https://doi.org/10.1080/00401706.1983.10487848). *Technometrics*, 25(2), 165–172.<br>
<a id="ref-14"></a>
[14] Iglewicz, B., & Hoaglin, D. C. (1993). [*How to Detect and Handle Outliers*](https://asq.org/quality-press). The ASQC Basic References in Quality Control: Statistical Techniques, Vol. 16. ASQC Quality Press, Milwaukee. ISBN 978-0-87389-247-6.<br>
<a id="ref-15"></a>
[15] ISO 16269-4:2010, [*Statistical interpretation of data — Part 4: Detection and treatment of outliers*](https://www.iso.org/standard/44396.html). International Organization for Standardization.<br>
<a id="ref-16"></a>
[16] Rousseeuw, P. J., & Van Driessen, K. (1999). [A Fast Algorithm for the Minimum Covariance Determinant Estimator](https://doi.org/10.1080/00401706.1999.10485670). *Technometrics*, 41(3), 212–223.<br>
<a id="ref-17"></a>
[17] Breunig, M. M., Kriegel, H.-P., Ng, R. T., & Sander, J. (2000). [LOF: Identifying Density-Based Local Outliers](https://doi.org/10.1145/335191.335388). *ACM SIGMOD Record*, 29(2), 93–104.<br>
<a id="ref-18"></a>
[18] Schölkopf, B., Platt, J. C., Shawe-Taylor, J., Smola, A. J., & Williamson, R. C. (2001). [Estimating the Support of a High-Dimensional Distribution](https://doi.org/10.1162/089976601750264965). *Neural Computation*, 13(7), 1443–1471.<br>
<a id="ref-19"></a>
[19] Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). [Isolation Forest](https://doi.org/10.1109/ICDM.2008.17). *Proceedings of the Eighth IEEE International Conference on Data Mining*, 413–422.<br>
<a id="ref-20"></a>
[20] Schlegl, T., Seeböck, P., Waldstein, S. M., Schmidt-Erfurth, U., & Langs, G. (2017). [Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery](https://doi.org/10.1007/978-3-319-59050-9_12). *Information Processing in Medical Imaging*, Lecture Notes in Computer Science 10265, 146–157.<br>
<a id="ref-21"></a>
[21] Li, Z., Zhao, Y., Hu, X., Botta, N., Ionescu, C., & Chen, G. H. (2022). [ECOD: Unsupervised Outlier Detection Using Empirical Cumulative Distribution Functions](https://doi.org/10.1109/TKDE.2022.3159580). *IEEE Transactions on Knowledge and Data Engineering*, 35(12), 12181–12193.<br>
<a id="ref-22"></a>
[22] Han, S., Hu, X., Huang, H., Jiang, M., & Zhao, Y. (2022). [ADBench: Anomaly Detection Benchmark](https://arxiv.org/abs/2206.09426). *Advances in Neural Information Processing Systems 35, Datasets and Benchmarks Track*.<br>
<a id="ref-23"></a>
[23] Roth, K., Pemula, L., Zepeda, J., Schölkopf, B., Brox, T., & Gehler, P. (2022). [Towards Total Recall in Industrial Anomaly Detection](https://arxiv.org/abs/2106.08265). *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, 14318–14328.<br>
<a id="ref-24"></a>
[24] Batzner, K., Heckler, L., & König, R. (2024). [EfficientAD: Accurate Visual Anomaly Detection at Millisecond-Level Latencies](https://arxiv.org/abs/2303.14535). *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision*, 128–138.<br>
<a id="ref-25"></a>
[25] AEC-Q001 Rev-D (2011), [*Guidelines for Part Average Testing*](http://www.aecouncil.com/AECDocuments.html). Automotive Electronics Council.<br>
<a id="ref-26"></a>
[26] Hsu, C.-Y., Chien, C.-F., & Lin, K.-Y. (2012). [Semiconductor Fault Detection and Classification for Yield Enhancement and Manufacturing Intelligence](https://doi.org/10.1007/s10696-012-9161-4). *Flexible Services and Manufacturing Journal*, 24(3), 358–378.

---

## Appendix A. Terminology

- **adjusted boxplot** — A box plot whose fences are moved by the skewness of the sample, so that a long tail is not read as a stream of outliers.
- **anomaly score** — A number ranking observations by how far they depart from the normal pattern, without a stated error rate attached to any particular value of it.
- **AU-PRO** — Area under the per-region overlap curve, which scores how well a method localizes a defect rather than whether it detects one.
- **AUROC** — Area under the receiver operating characteristic curve, the probability that a randomly chosen anomaly is scored above a randomly chosen normal observation.
- **box plot** — A summary drawing in which a box spans the interquartile range and lines called whiskers reach the most extreme observation still within one and a half interquartile ranges of the box, with anything past that drawn as a separate point.
- **breakdown point** — The fraction of a sample that has to be corrupted before an estimate stops describing the rest of the data. The mean has 0% and the median has 50%.
- **chi-square distribution** — The distribution of a sum of squared independent standard normal variables, carrying one degree of freedom per term. It is what turns a squared distance into a probability.
- **consistency constant** — A factor applied to a robust scale estimate so that it converges to the standard deviation under an assumed distribution. It is 0.674490 for the MAD and 1.349 for the interquartile range, which [AEC-Q001](#ref-25) rounds to 1.35.
- **contaminant** — An observation that came from a distribution other than the one the rest of the sample follows, as opposed to one that merely looks inconsistent with it.
- **contamination** — The fraction of a sample that does not come from the assumed distribution.
- **critical value** — The value a test statistic has to exceed to be called significant. It follows from the significance level and the sample size rather than from the data under test.
- **cumulative distribution function** — The function giving, for each value, the probability of falling at or below it. The standard normal one is written $\Phi$, and its inverse turns a probability back into a number of standard deviations.
- **degrees of freedom** — The number of independent quantities a statistic is free to vary over. It fixes which chi-square distribution a squared distance is read against, one per variable here.
- **discordant observation** — An observation that looks statistically inconsistent with the rest of the sample. A test of discordancy reports this and not contamination.
- **discriminator** — The network trained alongside a generator to tell generated samples from real ones. Its internal features can be reused to compare an observation against what the generator produced.
- **ECOD** — Empirical-cumulative-distribution-based outlier detection, the method of section 5.4.
- **ESD** — Extreme studentized deviate, abbreviated in the name of the generalized ESD procedure of section 4.4.
- **extreme studentized deviate** — The largest absolute deviation from the sample mean, divided by the sample standard deviation. It is the statistic each stage of the generalized ESD procedure computes.
- **false positive rate** — The fraction of normal observations that a rule flags. It is the price paid for whatever detection rate the rule reaches.
- **generator** — The network trained to produce samples a discriminator cannot tell from the training data. Once trained it stands in for the distribution the normal data came from.
- **Hotelling's T-squared** — The multivariate analogue of a squared z-score, measuring the distance of an observation from the centre inside the structure a model has fitted.
- **hyperparameter** — A setting fixed before a method runs rather than estimated from the data, such as a neighbourhood size or a kernel bandwidth. Without labels there is nothing to tune one against.
- **influential observation** — An observation whose removal changes a fitted model materially, measured by Cook's distance.
- **interquartile range** — The distance from the first quartile to the third, which is the spread of the middle half of a sample. On a normal sample it is 1.349 standard deviations.
- **IQR** — The abbreviation used throughout for the interquartile range.
- **kernel** — The function that fixes the geometry a one-class SVM works in, and with it the shapes its learned boundary is allowed to take.
- **latent space** — The compressed coordinates a generative model maps to and from, in which a point stands for a whole reconstructed observation.
- **leverage** — How extreme an observation is in the predictors of a fitted model, which sets how far it could move the fit whether or not it does.
- **loading** — The weight a principal component gives to one original variable, by which a flag raised in component space is traced back to a sensor.
- **lot** — A batch of parts processed together and carried through manufacturing as one unit. It is the group that part average testing judges a part against.
- **manifold** — The lower-dimensional surface inside the full space that the data actually occupy. A generative model that has learned it reproduces points on it and not points off it.
- **masking** — The effect by which an outlier inflates the centre or the scale it is measured against far enough that it, or a second outlier, no longer looks extreme.
- **medcouple** — A robust measure of skewness, between minus one and one and zero for a symmetric sample, built from a median of comparisons between observations on either side of the median.
- **median absolute deviation (MAD)** — The median of the absolute deviations of the observations from the sample median, used as a scale estimate that a minority of extreme observations cannot inflate.
- **minimum covariance determinant** — A robust estimate of a multivariate centre and covariance, taken from the subset of observations whose covariance matrix has the smallest determinant.
- **MVTec AD** — A public benchmark of photographs of manufactured objects, defect-free for training and defective for testing, with the defective region marked. MVTec AD 2 is a later set built to be harder.
- **novelty detection** — Judging new observations against a training set assumed to be free of outliers, as opposed to searching one sample that may already contain them.
- **order statistic** — An observation identified by its rank in the sorted sample rather than by its value, such as the median or a quartile. Moving an extreme observation further out does not move it.
- **outlier** — An observation inconsistent with the distribution the rest of the sample follows. The label concerns consistency with a model and does not by itself establish that the observation is wrong.
- **physical limit** — A bound a measured quantity cannot cross because of what it measures, such as a negative pressure or a yield above 100%. It is known before the sample is read, so a value outside it is wrong rather than merely inconsistent.
- **pretrained network** — A network fitted on a large general dataset and then used without further training, for the features its intermediate layers produce rather than for its own output.
- **principal component** — A direction fitted to the data along which the variance is largest, subject to being uncorrelated with the directions already fitted. A few of them usually carry most of the variation among correlated sensors.
- **reconstruction error** — The distance between an input and the output a model produces when it compresses and rebuilds that input.
- **robust** — Describing an estimate that a minority of contaminating observations cannot move far. The breakdown point says how large that minority may be.
- **significance level** — The probability of flagging an observation when the sample is in fact clean, fixed before the data are seen. Repeating a test without accounting for the repetition raises it above the value chosen.
- **specification limit** — The boundary a measured parameter must stay inside for a part to be sold, set from the design rather than from the sample. A part can pass it and still be an outlier within its lot.
- **squared prediction error (Q statistic)** — The part of an observation that a fitted model does not explain, measured as the squared distance from the observation to its reconstruction in the model's space.
- **SVM** — Support vector machine, a classifier that separates classes by the widest margin available in the geometry a kernel fixes. The one-class variant of section 5.2 has no second class and encloses the one it has instead.
- **swamping** — The effect by which an outlier distorts the centre or the scale far enough that clean observations are flagged alongside it.
- **winsorizing** — Replacing every observation past a chosen quantile with the value at that quantile, so that a fixed share of the sample is pulled in rather than tested. It is a treatment and not a detection rule.

## Appendix B. Tukey's Rule

Section 4.2 states the rule in one line. This appendix records where the multiple of 1.5 comes
from, what it costs against a z-score, and where the rule stops working.

### B.1. Inner and Outer Fences

[Tukey (1977)](#ref-1) drew two pairs of fences rather than one. The inner pair is the rule of section 4.2,
and the outer pair sits at three interquartile ranges instead of one and a half.

```math
Q_1 - c \cdot \mathrm{IQR} \ \le \ x_i \ \le \ Q_3 + c \cdot \mathrm{IQR}
```

- $c$ — the multiple that places the fences, 1.5 for the inner pair and 3 for the outer pair.
- $Q_1$, $Q_3$, $\mathrm{IQR}$ — as in section 4.2.

An observation past an inner fence Tukey called **outside**, and one past an outer fence **far
out**. The whiskers reach the last observation inside the inner fences, so every separate point in
the plot is at least outside.

The two are read together: outside deserves a look, far out is extreme by any reading. A single cut-
off cannot make that distinction.

### B.2. What the Multiple Costs

The multiple of 1.5 was chosen for convenience, not derived. Section 4.2 calls the rule comparable
to a z-score at 3, and the two are not identical.

**Table 4. Where each fence sits on a normal sample**

| Rule | Position | Share of a normal sample flagged |
|---|---|---|
| Inner fence, $c = 1.5$ | 2.6980 $\sigma$ | 0.6977% |
| Outer fence, $c = 3$ | 4.7214 $\sigma$ | 0.0002% |
| Classical rule at 3 | 3.0000 $\sigma$ | 0.2700% |

The inner fence is looser than the classical rule at 3 by a factor of 2.6, and a multiple of 1.724
would put it exactly there. The outer fence is stricter than either by three orders of magnitude.

Comparable rather than equal: both flag a fraction of a percent where a rule at two standard
deviations flags five, and the choice between them turns on contamination.

### B.3. Skewed Samples

The fences are symmetric, the same multiple below $Q_1$ as above $Q_3$. On a skewed sample the long
tail is a property of the distribution, and the rule reads it as a stream of outliers while flagging
nothing on the short side.

On 200,000 lognormal draws with no contamination in them, the standard fences flag 6.22% above the
upper fence and nothing at all below the lower one.

The adjusted boxplot of [Hubert and Vandervieren (2008)](#ref-10) repairs this by moving each
fence according to how skewed the sample is, measured by the medcouple of
[Brys, Hubert and Struyf (2004)](#ref-9).

```math
\left[ \ Q_1 - 1.5 e^{a \cdot \mathrm{MC}} \cdot \mathrm{IQR}, \quad Q_3 + 1.5 e^{b \cdot \mathrm{MC}} \cdot \mathrm{IQR} \ \right]
```

- $\mathrm{MC}$ — the medcouple, a robust skewness measure between $-1$ and $1$ that is 0 for a symmetric sample.
- $a$, $b$ — $-4$ and $3$ when $\mathrm{MC} \ge 0$, and $-3$ and $4$ when it is negative, so the fence on the long side moves out and the one on the short side moves in.

On that sample the medcouple is 0.3264, and the adjusted fences flag 1.10% above and 0.42% below in
place of 6.22% and nothing. Still more than a normal sample gives, but no longer the shape of the
distribution reported as a list of outliers.

## Appendix C. Semiconductor Practice

A fab runs the methods a standard names, an auditor can check, and a technician can act on. Two of
them are constructions already covered in sections 4 to 6.

### C.1. Part Average Testing

Part average testing removes parts abnormal for their own lot even when every measurement passes its
specification limit. AEC-Q001 defines it for automotive components on the plan of section 4.3: the
robust mean is the median, and the robust sigma is the interquartile range divided by 1.35. A part
is retained inside the interval below.

```math
\tilde{x} \pm k \cdot \frac{\mathrm{IQR}}{1.35}
```

- $\tilde{x}$ — the median of the parameter across the parts being judged, which the standard calls the robust mean.
- $\mathrm{IQR}$ — their interquartile range, and $\mathrm{IQR}/1.35$ is what the standard calls the robust sigma.
- $k$ — the multiple of that sigma the limits are set at, 6 by convention.

That divisor is the $1.349 \sigma$ of section 4.2, rounded, and it does for the quartile spread what
$\Phi^{-1}(0.75)$ does for the MAD. The standard picks the quartiles over the MAD and 6 over 3.5,
but the construction is the same: a robust centre, a robust scale in normal units, and a multiple of
that scale.

Static limits are computed once from historical data and applied to every lot. Dynamic limits are
recomputed from each lot, catching a lot that is uniformly shifted yet internally tight. They need
a minimum sample per lot, 30 parts in the standard, before the quartiles mean anything.

### C.2. Fault Detection and Classification

Equipment sensors report pressure, flow, power and temperature through a process step. [Fault
detection and classification](#ref-26) reduces each trace to summary parameters per wafer and
monitors them together, since a per-variable limit misses a departure that only the combination
shows.

The standard construction is the multivariate control chart of section 4.5 in a reduced space.
Principal components are fitted on normal production, and an observation is scored by Hotelling's
$T^2$ inside that space and by the squared prediction error, the $Q$ statistic, for the part the
components do not explain. $T^2$ says the process moved within the structure it normally has, and
$Q$ says it left that structure.

The split makes the flag actionable: the loading contributing most to a $T^2$ or a $Q$ names the
sensor to look at.

## Appendix D. Estimator Constants and Bounds

The settings of sections 4.1 to 4.5 rest on constants and bounds that the body states without
deriving. This appendix records where each comes from.

### D.1. The Ceiling on a Z-Score

Both estimates of the z-score come from the sample under test, so an outlier inflates the scale it
is measured against. The same self-reference caps the score: in a sample of size $n$ no absolute
z-score can exceed $(n-1)/\sqrt{n}$, a bound due to [Shiffler (1988)](#ref-12). A rule at 3
therefore cannot fire below 11 observations, and one at 3.5 cannot fire below 15.

### D.2. The MAD Consistency Constant

The raw MAD does not estimate $s$. On a normal sample it converges to $0.674490 \sigma$, about a
third short of the spread, and $0.674490$ is the third quartile of the standard normal distribution.
Dividing by it, which is multiplying by 1.482602, puts the modified z-score on the scale the z-score
is read on. Without that step no threshold carries between the two rules. The constant is a
calibration rather than an assumption: changing it rescales every score by the same factor and
reorders nothing.

### D.3. Critical Values of the Generalized ESD

Each $R_i$ is compared against a critical value $\lambda_i$ derived for that stage and tabulated by
[Rosner (1983)](#ref-13). The count of outliers is the largest $i$ for which $R_i \gt \lambda_i$,
not the first. Reading it that way is what defeats masking: a stage can fail while a later stage,
with the masking observation already removed, succeeds.

### D.4. The Chi-Square Cut-Off and a Robust Covariance

Under multivariate normality, with the centre and the covariance known rather than estimated, the
squared Mahalanobis distance follows a chi-square distribution with one degree of freedom per
variable, and the cut-off comes from there. Estimating the pair from a contaminated sample breaks
that reading, since a cluster of outliers inflates $\Sigma$ in exactly the direction that hides
them. The minimum covariance determinant of [Rousseeuw and Van Driessen (1999)](#ref-16) supplies a
robust centre and covariance and restores it.

## Appendix E. Benchmarks

Published comparisons fix how much a method can be expected to give. The figures below belong to the
benchmarks that produced them.

### E.1. Tabular Data

The published comparisons name no winner. Across the 30 algorithms and 57 datasets of
[ADBench](#ref-22), no unsupervised method is statistically superior to the rest; Isolation Forest
and ECOD are consistently among the better ones without dominating, and several deep methods built
for tabular data fall below them. Newer is not by itself a reason to switch.

The exception is raw coordinates that carry no usable distance, where the deep methods earn their
cost. Images are the clearest instance.

### E.2. Images

[PatchCore](#ref-23) established the patch feature memory of section 6.3 and reports up to 99.1%
detection AUROC on the MVTec AD benchmark. [EfficientAD](#ref-24) reaches 95.4% detection AUROC
across 32 datasets at 2.2 ms per image, the latency that makes inline inspection possible.

On MVTec AD 2, built to carry the lighting and defect variation of real inspection, no published
method exceeds 31% localization AU-PRO at a 5% false positive rate. A method that separates the
older benchmark cleanly is not thereby ready for a line.


## Appendix F. What Practice Actually Runs

A survey ranks methods by what they assume. Practice ranks them by what is already on the screen,
and the two orders are not the same.

### F.1. The Order Practice Meets Them

**Table 3. What practice actually runs, most common first**

| Rank | Rule | Why it is reached for |
|---|---|---|
| 1 | Interquartile range, the Tukey fence of section 4.2 | The box plot is usually the first drawing made, and its whiskers already are the rule. |
| 2 | Z-score cut at 3, of section 4.1 | Habit. It is the rule everyone was taught, and it is the wrong one whenever the sample is neither normal nor clean. |
| 3 | Modified z-score on the MAD, of section 4.3 | Where the work moves the moment the data are at all dirty. |
| 4 | Quantile clipping, winsorizing at the 1st and the 99th percentile | Cheap, and it needs no test at all. It fixes a share of the sample rather than a property of it. |
| 5 | 🌳A domain physical limit | It should be first. A negative pressure or a yield above 100% is settled before any statistic is computed. |

The last two entries are of a different kind from the first three. Winsorizing decides nothing: it
is a treatment applied to a fixed share of the sample whether or not that share is discordant, and
section 1 keeps treatment apart from detection. A physical limit is knowledge held before the sample
is read, and the one rule here that calls an observation wrong rather than inconsistent.

So the fifth entry belongs first. Bounds the process cannot cross are applied before anything in
sections 4 to 6 runs, since a value outside them corrupts every estimate that follows.

### F.2. Two Habits

Two habits matter more than which method is used. Fix the threshold before the data are seen, so it is not
tuned to a preferred answer. Then read the margin rather than the verdict: only a statistic that
clears its cut-off by a wide gap survives a change in the choices above.
