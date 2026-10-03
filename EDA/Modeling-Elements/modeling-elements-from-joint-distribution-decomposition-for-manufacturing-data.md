# Modeling Elements from Joint Distribution Decomposition for Manufacturing Data
Rev. 19 | Created: 2026-10-03 | Updated: 2026-10-03 11:53 CDT

- [1. Purpose](#1-purpose)
- [2. Summary](#2-summary)
- [3. Taxonomy and its Hierarchy](#3-taxonomy-and-its-hierarchy)
  - [3.1 Placement](#31-placement)
- [4. Prediction from the Joint Distribution](#4-prediction-from-the-joint-distribution)
- [5. Physical Meaning](#5-physical-meaning)
  - [5.1 P(X) Shift and Drift](#51-px-shift-and-drift)
  - [5.2 P(Y) Shift and Drift](#52-py-shift-and-drift)
  - [5.3 P(Y|X) Shift and Drift](#53-pyx-shift-and-drift)
- [References](#references)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Detection and Implementation by Cell](#appendix-b-detection-and-implementation-by-cell)
  - [B.1 P(X) Shift and Drift](#b1-px-shift-and-drift)
  - [B.2 P(Y) Shift and Drift](#b2-py-shift-and-drift)
  - [B.3 P(Y|X) Shift and Drift](#b3-pyx-shift-and-drift)
  - [B.4 Estimation Error](#b4-estimation-error)
- [Appendix C. Benchmarking](#appendix-c-benchmarking)
- [Appendix D. Prior in Bayes' Theorem](#appendix-d-prior-in-bayes-theorem)
- [Appendix E. Talk Slides](#appendix-e-talk-slides)

## 1. Purpose

- **Problem Statement**: There is no framework that ties together the elements needed to build an AI/ML model.
- **Goal**: Present a framework that, through the decomposition of the joint probability distribution $P(X,Y) = P(Y\vert{}X) \cdot P(X)$, clearly connects the elements of AI/ML on manufacturing process data (data, model, prediction), their changes (shift, drift) and the estimation error of the model.
- **Non-Goal**: The diagnostic figures of any specific project are out of scope.

## 2. Summary

> ### Good predictions come from good data and good models.

- **Taxonomy**: Factorizing the joint distribution P(X,Y) by the chain rule yields three elements, data P(X), model P(Y|X) and prediction P(Y), and each element undergoes shift, a change at once, and drift, a gradual change (section 3).
- **Error**: Prediction error splits into the estimation error of the model and the change of the X → Y relation after training, and a change in P(X) enlarges the error by moving weight toward regions where the estimation error is large (section 4).
- **Reading rule**: In a manufacturing process, which has an X → Y structure, a change in P(Y) is mostly the result of a change in P(X) or in the X → Y relation, and the three elements rest on the premises of uncertainty sources, sample selection and the i.i.d. assumption (section 3, section 5).

## 3. Taxonomy and its Hierarchy

The joint distribution factorizes into the product of the relation P(Y|X) and the data distribution P(X), the output P(Y) is obtained by integrating that product over X, and each of the three elements changes in two modes, shift and drift ([Fig 1](#fig-1)).

```text
                 P(X, Y)   joint distribution
                    |
      factorize     |   P(X, Y) = P(Y|X) * P(X)
          +---------+---------+
          |                   |
        P(X)               P(Y|X)
   data marginal       relation X -> Y
   good data           good model
          |                   |
          +---------+---------+
      marginalize   |   integrate over X
                    |
                  P(Y)
            output marginal
            good prediction

  change mode (each element): shift = changes at once, drift = changes gradually
  estimation error: P(Y|X)_model departs from the true P(Y|X), set at training
  reverse factorization: P(X, Y) = P(X|Y) * P(Y), defines P(Y) shift in a Y -> X structure
  premises: uncertainty sources, sample selection, i.i.d. and lot / chamber hierarchy
```

<a id="fig-1"></a>
Fig 1. Taxonomy and hierarchy of the joint distribution

- P(X) and P(Y|X) in the upper layer are the two factors that make up the joint distribution.
- P(Y) in the lower layer is a marginal distribution derived from the two factors, so a change in P(X) or P(Y|X) can carry over into P(Y).
- A change of the three elements splits by speed into shift, which happens at once, and drift, which happens gradually over time, and the two modes call for different detection and response methods (section 3.1).
- Estimation error is how far P(Y|X)<sub>model</sub> departs from the true P(Y|X) even when the true relation stays fixed, and it is one term of the prediction error (section 4).

In this document P(·) denotes a probability distribution: a probability mass function for a discrete variable and a probability density function for a continuous one. From the definition of the conditional distribution, P(Y|X) = P(X,Y) / P(X), the joint distribution factorizes by the chain rule of eq. (1).

```math
P(X, Y) = P(Y \mid X) \cdot P(X) \hspace{19em} (1)
```

```math
P(Y) = \int P(Y \mid X)\, P(X)\, dX \hspace{19em} (2)
```

Integrating both sides of eq. (1) over X gives eq. (2). The left side, ∫ P(X,Y) dX, is the marginal distribution P(Y) of Y with X integrated out, and the right side is the conditional distribution P(Y|X) averaged with the distribution P(X) as weight. This integration is called marginalization: eq. (1) splits the joint distribution into two factors, and eq. (2) recovers P(Y) from those two factors.

The three elements of the quote in section 2 take the three places of eqs. (1) and (2).

- **$`P(Y)`$ (Good Prediction):** good prediction; the marginal distribution obtained by integrating the joint distribution P(X,Y), the product of the two factors, over X (eq. (2)).
- **$`P(X)`$ (Good Data):** good data; a factor of eq. (1), the distribution of the measured data.
- **$`P(Y \mid X)`$ (Good Model):** good model; a factor of eq. (1), the X → Y relation that yields the metrology value given the measured data.

The joint distribution also factorizes in the direction opposite to eq. (1).

```math
P(X, Y) = P(X \mid Y) \cdot P(Y) \hspace{19em} (3)
```

P(Y) shift is defined in eq. (3) as a change in P(Y) alone while P(X|Y) stays fixed, and the literature calls it prior shift ([Appendix D](#appendix-d-prior-in-bayes-theorem)). This definition holds in a causal structure where Y causes X (Y → X), whereas a manufacturing process, in which the process runs, leaves measured data X behind and then yields the metrology value Y, has an X → Y structure. In an X → Y structure a process change alters P(X|Y) through P(Y|X) and P(X), so the premise that P(X|Y) stays fixed often fails. When it fails, a correction that relies on that premise (label shift correction) loses its basis.

The three elements rest on the three premises below.

- **Sources of uncertainty**: The metrology value is Y<sub>obs</sub> = Y + ε<sub>m</sub> and the measured data is X<sub>obs</sub> = X + ε<sub>x</sub>. The spread of P(Y<sub>obs</sub>|X<sub>obs</sub>) that the model learns contains intrinsic process variation, metrology error ε<sub>m</sub> (label noise) and sensor measurement error ε<sub>x</sub>, all three of which are aleatoric uncertainty that more data does not reduce. Beyond widening the spread, ε<sub>x</sub> pulls the slope of the estimated relation toward 0 (regression dilution) and so distorts the estimate of P(Y|X). Only the model's uncertainty from too little training data (epistemic uncertainty) shrinks with more data, and the accuracy ceiling a good model can reach is set by intrinsic process variation and the two measurement errors.
- **Sample selection**: Metrology samples only some wafers, so the distribution P(X<sub>i</sub>) of the training data X<sub>i</sub> is that of the measured wafers and can differ from P(X) over all wafers (selection bias). Inference runs on wafers that were not measured, so this difference becomes P(X) shift as it stands. When metrology goes missing depending on the value of X or Y, P(X<sub>i</sub>) or the P(Y) of the training data is skewed in the same way.
- **i.i.d. assumption and hierarchy**: Obtaining predictions for inference data from training data through eq. (4) rests on the assumption that wafers are independent and drawn from the same distribution (i.i.d.). Process data carries temporal autocorrelation and a lot and chamber hierarchy in which wafers from the same lot or chamber resemble each other; splitting training and validation while ignoring this structure puts the same lot on both sides and makes performance look better than it is.

The changes of the three elements fall into the 3×2 cells of Table 1.

Table 1. Shift and drift of the three elements

| Cell          | Example                             | Effect                                             |
| :-----------: | :---------------------------------: | :------------------------------------------------: |
| P(X) shift    | New equipment, recipe change        | The value range of X moves at once                 |
| P(X) drift    | Sensor drift                        | The measured value X<sub>obs</sub> skews gradually |
| P(Y) shift    | Target spec change                  | The metrology reference of Y changes at once       |
| P(Y) drift    | Gradual change in product mix       | Y moves gradually with the share of each product   |
| P(Y\|X) shift | PM, part replacement                | The X → Y relation changes at once                 |
| P(Y\|X) drift | Chamber aging, residue buildup Z(t) | The X → Y relation changes gradually               |

One event can affect several cells at once, and Table 1 puts each event in the cell of the element that changes first. Sensor drift is an example: it is P(X) drift and also changes the relation P(Y|X<sub>obs</sub>) on measured values. The literature and industry tools call the same cells by other names: P(X) shift is covariate shift, P(Y) shift is prior shift or label shift, P(Y|X) shift and drift are concept shift and concept drift, and model monitoring tools call P(X) drift data drift ([Appendix C](#appendix-c-benchmarking)).

### 3.1 Placement

Table 2 places the three elements on six lenses: decomposition, change, intervention, question, observation and lever.

Table 2. Six lenses on P(X), P(Y) and P(Y|X)

| Lens                    | P(X)                                                                          | P(Y)                                                                    | P(Y\|X)                                                                            |
| :---------------------: | :---------------------------------------------------------------------------: | :---------------------------------------------------------------------: | :--------------------------------------------------------------------------------: |
| Decomposition           | Marginal distribution of measured data                                        | Marginal distribution of output                                         | Conditional X → Y relation                                                         |
| Distribution change     | P(X) shift, P(X) drift                                                        | P(Y) shift, P(Y) drift                                                  | P(Y\|X) shift, P(Y\|X) drift                                                       |
| Intervention point      | Data space                                                                    | Target engineering (transform, decomposition)                           | Model and algorithm space (relation learning)                                      |
| Question                | Has inference data X<sub>o</sub> moved away from training data X<sub>i</sub>? | Has the metrology value distribution changed?                           | Has the relation between measured data and metrology value changed since training? |
| Observation (detection) | PSI·KS, domain classifier, T²·SPE control chart                               | Metrology value distribution comparison, Shewhart·EWMA control chart    | Binning CDT, residual CUSUM·Page-Hinkley, ADWIN                                    |
| Lever                   | Feature selection and augmentation, importance weighting, domain adaptation   | Target transform and decomposition, per-group scaling, prior correction | Retraining after an event, recency weighting, adaptive update                      |

The observation row lists ways to measure a change, and the lever row lists ways to fit the model while assuming a change. Shift is found by comparing two sets, the training data and the inference data, and drift is found by control charts and sequential tests that track a statistic per time window. Only the methods in the observation row pin down when a change happened. The detection, response and validation methods for the six cells and the estimation error are collected in [Appendix B](#appendix-b-detection-and-implementation-by-cell). Cases in which research and industry use this classification are collected in [Appendix C](#appendix-c-benchmarking).

## 4. Prediction from the Joint Distribution

Eq. (4) writes the relation of the three elements when a trained model is used for inference, keeping training data X<sub>i</sub> and inference data X<sub>o</sub> apart, and eq. (5) splits the error of that prediction into two terms, estimation error and relation change. X<sub>i</sub> is in-sample, the data used to train the model, and X<sub>o</sub> is out-of-sample, the new data that arrives at inference.

```math
P(Y)_{\mathrm{pred}} = \int P(Y \mid X = x;\, X_{i})_{\mathrm{model}} \cdot P(X_{o} = x)_{\mathrm{true}}\, dx \hspace{19em} (4)
```

- **$`P(Y)_{\mathrm{pred}}`$ (Overall Predicted Distribution):** the final distribution of the target variable $`Y`$ expected on out-of-sample inference data; a marginal distribution with the measured data integrated out.
- **$`P(Y \mid X = x;\, X_i)_{\mathrm{model}}`$ (Model's Conditional Prediction):** the predictive model itself. It is trained on in-sample training data $`X_i`$ and, given a measured data value $`x`$, returns the conditional distribution of $`Y`$. The subscript `model` marks it as an estimated, learned function that may differ from the true distribution. The $`X_i`$ after `;` marks the data the model was trained on, set apart from the conditioning variable after `|`. Elsewhere in this document it is shortened to P(Y|X)<sub>model</sub>.
- **$`P(X_o = x)_{\mathrm{true}}`$ (True Distribution of Out-of-Sample Data):** the true probability density that out-of-sample inference data $`X_o`$ takes the value $`x`$. It describes how $`X_o`$ is actually distributed when the model is used for inference.
- **$`\int \ldots dx`$ (Marginalization over $`x`$):** sums the predictions over every value $`x`$ the inference data can take. Each value is weighted by how likely it is at inference time, so the result is a weighted average of the predictions.

Eq. (4) carries the output marginal distribution of eq. (2) over to prediction. The true P(Y)<sub>t</sub> at inference time t is obtained by integrating the true relation at that time, P(Y|X=x)<sub>t</sub>, with the same P(X<sub>o</sub>), so subtracting P(Y)<sub>t</sub> from eq. (4) splits the prediction error into the two terms of eq. (5). t<sub>i</sub> is the training time.

```math
\begin{aligned}
P(Y)_{\mathrm{pred}} - P(Y)_{t}
&= \int \left[ P(Y \mid X = x;\, X_{i})_{\mathrm{model}} - P(Y \mid X = x)_{t_{i}} \right] P(X_{o} = x)_{\mathrm{true}}\, dx \\
&+ \int \left[ P(Y \mid X = x)_{t_{i}} - P(Y \mid X = x)_{t} \right] P(X_{o} = x)_{\mathrm{true}}\, dx
\end{aligned}
\hspace{19em} (5)
```

- **Estimation error**: the first term; the part by which the model estimates the relation differently even when the true relation stays as it was at training.
- **Relation change**: the second term; the part from a change in the true X → Y relation after training, which holds the P(Y|X) shift and drift of Table 1.
- **Weight**: both terms are weighted by the inference data distribution P(X<sub>o</sub>). P(X) shift or drift leaves the differences in brackets unchanged but moves the weight, enlarging the share of regions where the difference is large.

P(Y|X)<sub>model</sub> fails to reflect the true relation for the five reasons below; the first four belong to the first term and the last to the second.

- **Variance**: when samples are few relative to variables, the estimated relation swings widely with the training sample.
- **Bias**: when the model family cannot express the form of the true relation, error remains however many samples are added, as when a linear model fits a nonlinear relation.
- **Measurement error**: metrology error ε<sub>m</sub> widens the spread, and sensor error ε<sub>x</sub> pulls the estimated slope toward 0 (sources of uncertainty in section 3).
- **Extrapolation**: where P(X<sub>i</sub>) is sparse, no training samples constrain the model and the estimate is left undetermined. When P(X) shift or drift moves P(X<sub>o</sub>) into such a region, this share grows in eq. (5).
- **Unobserved state**: when the chamber state Z is not in the measured data X, the model learns the relation averaged over the distribution of Z at training. When the distribution of Z changes after training, the difference becomes the second term (section 5.3).

Read through eq. (5), the three elements are as follows.

- **Good prediction**: P(Y)<sub>pred</sub> is close to the true P(Y) when both terms of eq. (5) are small.
- **Good data**: X<sub>i</sub> is training data measured by equipment sensors and paired with wafer metrology values for model training, and P(X<sub>o</sub>)<sub>true</sub> is the distribution of the inference data actually measured at inference. P(X<sub>i</sub>) must cover P(X<sub>o</sub>) for the Extrapolation share to stay small.
- **Good model**: P(Y|X)<sub>model</sub> is the X → Y relation the model estimates from training data X<sub>i</sub>. The first term shrinks through the choice of estimator and validation ([B.4](#b4-estimation-error)), and the second only through detecting the relation change and retraining ([B.3](#b3-pyx-shift-and-drift)).

## 5. Physical Meaning

In a manufacturing process the three elements correspond to measured data, metrology results and the X → Y relation between measured data and metrology values. Table 3 collects that correspondence and the role of each element in eq. (4).

Table 3. Physical meaning of each term

| Term    | Role in eq. (4)                                              | Physical meaning                                                                             |
| :-----: | :----------------------------------------------------------: | :------------------------------------------------------------------------------------------: |
| P(X)    | Good data: P(X<sub>i</sub>), P(X<sub>o</sub>)<sub>true</sub> | Distribution of data measured by equipment sensors                                           |
| P(Y)    | Good prediction: P(Y)<sub>pred</sub>                         | Distribution of wafer metrology values (measurement map, spatial decomposition coefficients) |
| P(Y\|X) | Good model: P(Y\|X)<sub>model</sub>                          | X → Y relation that process physics leaves between measured data and metrology values        |

### 5.1 P(X) Shift and Drift

- Meaning: P(X) shift is a change at once of the inference data distribution P(X<sub>o</sub>) away from the training data distribution P(X<sub>i</sub>), and P(X) drift is a gradual move of P(X<sub>o</sub>) over time. The X → Y relation may stay the same.
- Interpretation: the model has learned the relation, but inference data has moved into a region that training data covers sparsely, enlarging the Extrapolation share of eq. (5).
- Scope: work in the data space covers feature selection and generation in general, along with responses to change.

### 5.2 P(Y) Shift and Drift

- Meaning: the output marginal distribution changes. It is defined in eq. (3) as a change in P(Y) alone while P(X|Y) stays fixed, and the definition holds in a Y → X causal structure.
- Reading in a manufacturing process: in an X → Y structure an observed change in P(Y) mostly appears as the result of a change in P(X) or in the X → Y relation. P(Y) itself changes in P(Y) shift, when the definition or reference of Y changes at once as with a target spec change, and in P(Y) drift, when the product mix changes gradually and moves the distribution of Y while the product is not in X.
- Target engineering: an intervention on P(Y) changes the definition and structure of the output, and covers target transforms (log, Box-Cox), spatial decomposition and multi-task target reformulation.
- Spatial decomposition: target engineering that turns a wafer measurement map into coefficients on a spatial basis. The output becomes smooth, physically meaningful coefficients, which makes P(Y|X) easier to learn.

### 5.3 P(Y|X) Shift and Drift

The X → Y relation between measured data and metrology values itself changes after training, and this is the second term of eq. (5). A change at once, as with PM or part replacement, is P(Y|X) shift, and a gradual change, as with chamber aging, is P(Y|X) drift. It is the hardest of the three to handle.

From a process physics view, a change of the X → Y relation comes from a change in an unobserved chamber state variable Z(t) (aging, residue), and eq. (6) writes that relation.

```math
P(Y \mid X, t) = \int P(Y \mid X, Z)\, P(Z \mid t)\, dZ \hspace{19em} (6)
```

Z is a latent variable. Eq. (6) holds on the premise that Z is independent of X at a given t (Z ⫫ X | t); when the premise fails, the integral must use P(Z|X,t) in place of P(Z|t). Even when the relation P(Y|X,Z) given the chamber state does not change over time, Z is not in the measured data X, so the model sees a change in P(Z|t) only as a change in P(Y|X). When P(Z|t) steps at a PM it becomes P(Y|X) shift, and when it moves gradually over time it becomes P(Y|X) drift. A relation change has two paths, response and observation.

- **Response**: drift calls for detrending, recency sample weighting and retraining on a recent window, and shift calls for retraining on the data after an event such as PM. The effect is checked in time order with temporal CV, and these methods cannot pin down when a change happened.
- **Observation**: measures the change and pins down when it happened.

Observation methods differ in how directly they look at P(Y|X).

- **Binning CDT**: splits X into bins and tests P(Y|bin) per time window to pin down when a change happened. It looks at P(Y|X) most directly.
- **Residual CUSUM, Page-Hinkley**: reports the time at which a statistic of the prediction residuals crosses a threshold.
- **I(X;Y)**: total amount of dependence (a macro indicator). Used alone it picks up changes in P(X), P(Y) and the relation together, so it comes closer to a relation change only when tracked per time window. For high-dimensional X it is hard to estimate accurately per window, so it is narrowed to I(X<sub>k</sub>;Y) over the top K variables X<sub>k</sub> by importance.

## References

<a id="ref-1"></a>
[1] Kang, S., & Kang, P. (2017). [An intelligent virtual metrology system with adaptive update for semiconductor manufacturing](https://doi.org/10.1016/j.jprocont.2017.02.002). *Journal of Process Control*, 52, 66–74.<br>
<a id="ref-2"></a>
[2] Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (Eds.). (2009). [Dataset Shift in Machine Learning](https://mitpressbookstore.mit.edu/book/9780262170055). MIT Press. ISBN 978-0-262-17005-5.<br>
<a id="ref-3"></a>
[3] Moreno-Torres, J. G., Raeder, T., Alaiz-Rodríguez, R., Chawla, N. V., & Herrera, F. (2012). [A unifying view on dataset shift in classification](https://doi.org/10.1016/j.patcog.2011.06.019). *Pattern Recognition*, 45(1), 521–530.<br>
<a id="ref-4"></a>
[4] Gama, J., Žliobaitė, I., Bifet, A., Pechenizkiy, M., & Bouchachia, A. (2014). [A survey on concept drift adaptation](https://doi.org/10.1145/2523813). *ACM Computing Surveys*, 46(4), 44.<br>
<a id="ref-5"></a>
[5] Amazon Web Services. [Data and model quality monitoring with Amazon SageMaker Model Monitor](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor.html). *Amazon SageMaker AI Developer Guide*.<br>
<a id="ref-6"></a>
[6] Google Cloud. [Introduction to Vertex AI Model Monitoring](https://docs.cloud.google.com/vertex-ai/docs/model-monitoring/overview). *Vertex AI documentation*.<br>
<a id="ref-7"></a>
[7] Evidently AI. [Concept drift in ML](https://www.evidentlyai.com/ml-in-production/concept-drift). *ML in Production guide*.

---

## Appendix A. Terminology

- **aleatoric uncertainty**: uncertainty from the spread inherent in the data. More data does not reduce it.
- **covariate**: an input variable X of the model. Variate is the general word for a single random variable and applies to both X and Y; the "co-" marks a variable that varies together (co-varies) with Y, the variable of main interest, that is, a variable observed alongside Y to explain it. In this document it is the data measured by equipment sensors, and the literature calls a change in its distribution P(X) between training and inference covariate shift.
- **CUSUM (Cumulative Sum)**: a control chart that accumulates deviations from a reference and flags the time the sum crosses a threshold as a change point.
- **drift**: a gradual change of a distribution or relation over time. It is found by tracking a statistic per time window.
- **epistemic uncertainty**: uncertainty the model carries because training data is insufficient. More data reduces it.
- **EWMA (Exponentially Weighted Moving Average)**: a control chart that catches small, steady moves with a moving average that weights recent values more.
- **I(X;Y)**: mutual information. A macro indicator of the total dependence between X and Y.
- **i.i.d. (independent and identically distributed)**: the assumption that each sample is independent and drawn from the same distribution.
- **label noise**: noise mixed into the target value Y, such as metrology error.
- **latent variable**: a variable that affects the outcome but is not observed directly, such as the aging or residue buildup of a chamber.
- **marginal distribution (주변분포)**: the distribution left for Y alone after integrating X out of the joint distribution P(X,Y). P(Y) = ∫ P(X,Y) dX, describing how Y is distributed regardless of the value of X.
- **regression dilution**: the shrinking of an estimated regression slope toward 0 when the input X carries measurement error.
- **selection bias**: a skew in which the sample distribution departs from the population distribution because of how the sample is chosen.
- **shift**: a change at once of a distribution or relation. It is found by comparing two sets, the training data and the inference data.
- **spatial decomposition**: decomposing a wafer measurement map onto a spatial basis (polynomials) and predicting its coefficients (a1, …).
- **target engineering**: transforming, decomposing or reformulating the prediction target Y into a form the model learns more easily.
- **temporal CV**: time-ordered cross-validation that trains on the past and validates on the future.

## Appendix B. Detection and Implementation by Cell

B.1 to B.3 collect, for each of the three elements of Table 1, how to detect its shift and drift (Detection) and which models and techniques respond and how they are validated (Response), and B.4 covers the estimation error of eq. (5). Shift is found by comparing two sets, and drift by tracking per time window.

### B.1 P(X) Shift and Drift

Univariate methods compare each variable's distribution separately and miss changes in the correlation between variables. Manufacturing data with hundreds of sensor variables calls for multivariate methods alongside them.

#### Shift Detection

- **PSI (Population Stability Index)**: compares, per variable, the binned proportions of training and inference data to measure how far P(X) has moved.
- **KS test (Kolmogorov–Smirnov test)**: tests, per variable, whether two data sets share a distribution by the maximum gap between their cumulative distributions.
- **KL divergence (Kullback–Leibler divergence)**: measures, in information terms, how far the inference data distribution is from the training data distribution.
- **MMD (Maximum Mean Discrepancy)**: tests multivariate distribution difference by the gap between the means of two data sets in a kernel space.
- **Domain classifier**: trains a classifier to separate training from inference data; the further the AUC rises above 0.5, the more the two distributions differ.

#### Drift Detection

- **Hotelling T²·SPE control chart (PCA-based)**: plots T² and the residual SPE of inference data in time order in a PCA model built on training data, and finds when they cross the control limits.
- **Autoencoder reconstruction error**: tracks the reconstruction error of inference data in time order in an autoencoder built on training data, and finds when it crosses a control limit.
- **Windowed PSI**: computes PSI per time window to see the trend of inference data moving away from training data.

#### Response

- **Methods**: for shift, importance weighting that weights training samples by a density ratio estimated with a domain classifier, domain adaptation that aligns training and inference input distributions, and a metrology sampling plan adjusted so that training data covers the inference range. For drift, sensor recalibration and retraining on a recent window.
- **Validation**: adversarial validation, which builds the validation set from training samples that resemble inference data.

### B.2 P(Y) Shift and Drift

#### Shift Detection

- **Metrology value distribution comparison**: compares the distribution of training metrology values with recent ones by KS test or PSI.
- **Shewhart control chart**: watches for the time the mean and spread of metrology values leave their control limits at once.

#### Drift Detection

- **EWMA control chart**: watches small, steady moves through an exponentially weighted moving average of metrology values.
- **Prediction distribution monitoring**: when metrology values arrive late, watches the move of P(Y)<sub>pred</sub> first to give early warning of a change in P(Y).

#### Response

- **Methods**: for shift, redefining the target and retraining to match a target spec change. For drift, per-group scale normalization and target engineering (log and Box-Cox transforms, spatial decomposition).
- **Validation**: evaluating error separately per spec and per group.

### B.3 P(Y|X) Shift and Drift

#### Shift Detection

- **Binning CDT (Conditional Distribution Test)**: splits X into bins and tests P(Y|bin) per bin and per time window to pin down when a change happened.
- **Event-split comparison**: compares the distribution of prediction residuals before and after a PM or part replacement.

#### Drift Detection

- **Residual CUSUM**: takes the time the cumulative sum of prediction residuals crosses a threshold as the time the relation changed.
- **Page-Hinkley**: flags a change in the mean when the gap between the cumulative deviation of prediction residuals and its minimum crosses a threshold.
- **ADWIN (Adaptive Windowing)**: splits the error window in two and, if the means differ significantly, drops the older part and flags a change.
- **Windowed performance monitoring**: computes R²·RMSE per time window to find when performance degrades. It requires metrology values.
- **Windowed I(X<sub>k</sub>;Y)**: tracks dependence changes by computing, per time window, the mutual information between the top K variables X<sub>k</sub> by importance and the metrology value. I(X;Y) over all of a high-dimensional X is hard to estimate accurately per window because the dimension is large relative to the sample size. I(X<sub>k</sub>;Y) can change from a change in P(X) alone, so it is read together with the detection results of B.1.

#### Response

- **Methods**: for shift, retraining on the data after the event and adding proxies for the chamber state Z(t) (time since PM, accumulated RF hours) as features. For drift, periodic retraining on a recent window, recency weighting, and adaptive update that measures only low-reliability wafers and updates the model at once [[1](#ref-1)].
- **Validation**: temporal CV that trains on the past and validates on the future, and a group split by lot.

### B.4 Estimation Error

B.3 covers the second term of eq. (5), from a change in the true relation P(Y|X) after training, while B.4 covers the first term, from how the model estimates that relation even when it stays fixed. The first term is set at training time by the choice of estimator, overfitting and hyperparameters.

#### Detection

- **Train–validation gap**: reveals overfitting from the difference between training and validation performance. Degradation over time is watched with the Windowed performance monitoring of B.3.
- **Out-of-range check**: checks whether inference data lies outside the range of the training data, flagging wafers where the Extrapolation share of eq. (5) grows.

#### Response

- **Methods**: regularized linear models such as PLS, ridge and lasso when samples are fewer than variables; tree ensembles such as LightGBM, XGBoost and CatBoost for nonlinear relations; Gaussian process, quantile regression or conformal prediction when uncertainty is needed. Physics knowledge enters through monotone constraints or a hybrid that learns only the residual on top of a physics equation, and hyperparameters are searched with Bayesian optimization such as Optuna.
- **Validation**: group split by lot together with temporal CV, reading R², RMSE and the coverage of prediction intervals.

## Appendix C. Benchmarking

Table 4 collects cases in which the classification of this document appears in the dataset shift literature and in industry model monitoring tools.

Table 4. Use of the shift taxonomy in research and industry

| Source                                       | Kind             | Terms used                                              | Term in this document                                        |
| :------------------------------------------: | :--------------: | :-----------------------------------------------------: | :----------------------------------------------------------: |
| Quiñonero-Candela et al. [[2](#ref-2)]       | Book             | dataset shift, covariate shift                          | P(X,Y) difference between training and inference, P(X) shift |
| Moreno-Torres et al. [[3](#ref-3)]           | Paper            | covariate shift, prior probability shift, concept shift | P(X) shift, P(Y) shift, P(Y\|X) shift                        |
| Gama et al. [[4](#ref-4)]                    | Survey           | concept drift detection, adaptation                     | Observation and response for P(Y\|X) drift in 5.3            |
| Kang & Kang [[1](#ref-1)]                    | Paper            | virtual metrology adaptive update                       | Response to P(Y\|X) drift in manufacturing data              |
| Amazon SageMaker Model Monitor [[5](#ref-5)] | Industry tool    | data quality drift, model quality drift                 | P(X) drift, loss of good prediction performance              |
| Vertex AI Model Monitoring [[6](#ref-6)]     | Industry tool    | training-serving skew, inference drift                  | P(X) shift, P(X) drift                                       |
| Evidently AI [[7](#ref-7)]                   | Open-source tool | data drift, prediction drift, concept drift             | P(X) drift, P(Y)<sub>pred</sub> drift, P(Y\|X) drift         |

- **Research**: dataset shift is defined as a change in the joint distribution P(X,Y) between training and inference [[2](#ref-2)]. Moreno-Torres et al. organized covariate shift (P(X) moves, P(Y|X) stays), prior probability shift (P(Y) moves, P(X|Y) stays) and concept shift by which term of the joint distribution changes [[3](#ref-3)], and eq. (3) and the shifts of the three elements in this document follow that work.
- **Manufacturing data**: in semiconductor virtual metrology, wafer characteristics change over time and prediction performance degrades, so an adaptive update was proposed that measures only wafers with low prediction reliability and updates the model at once with those results [[1](#ref-1)]. Gama et al. surveyed methods for detecting and adapting to concept drift [[4](#ref-4)].
- **Industry tools**: model monitoring tools watch the P(X) change that is visible without ground truth under the names data quality drift [[5](#ref-5)], training-serving skew and inference drift [[6](#ref-6)], and data drift [[7](#ref-7)]. Once ground truth arrives, they check model quality drift [[5](#ref-5)] or concept drift [[7](#ref-7)] from the gap between predictions and ground truth.
- **Framework of this document**: classifying the changes of the three elements is a standard concept in research and industry. Splitting each element into the 3×2 cells of shift and drift under neutral names, and tying them to the prediction error through eq. (5), is this document's own framing and is not a named standard framework in the sources above.

## Appendix D. Prior in Bayes' Theorem

Written in the variables of this document, Bayes' theorem is eq. (7). The unknown the model infers is the metrology value Y, and the measured data X is the evidence for that inference.

```math
P(Y \mid X) = \frac{P(X \mid Y) \cdot P(Y)}{P(X)} \hspace{19em} (7)
```

The terms mean the following.

- **Prior**: P(Y). The prior probability: the distribution held for the metrology value Y before the measured data X is seen.
- **Likelihood**: P(X|Y). The probability that the measured data X appears, assuming the metrology value Y is given.
- **Posterior**: P(Y|X). The posterior probability: the distribution of Y updated after the measured data X is observed, which is the conditional distribution the model sets out to estimate.
- **Evidence**: P(X). The marginal distribution of the measured data, a normalizing constant that makes the posterior a probability distribution.

The prior belongs to the unknown being inferred, the metrology value Y, so the prior shift of the literature is the P(Y) shift of this document. The numerator P(X|Y)·P(Y) of eq. (7) equals the right side of eq. (3), so P(Y) shift, in which P(Y) alone changes while P(X|Y) of eq. (3) stays fixed, is a change of this prior. Label shift names the same change in P(Y) after the label Y. Both names come from classification problems with a Y → X structure, in which Y produces X. Such a problem picks the cause class from the effect, as when the disease Y is identified from the symptoms X it causes. A manufacturing process runs the other way, with the process data X as the cause and the metrology value Y as the effect in an X → Y structure, so the names do not fit it as they stand. A change in P(Y) observed in a manufacturing process therefore mostly results from a change in P(X) or P(Y|X) (section 5.2).

## Appendix E. Talk Slides

Three slides present this document, and the source file is [modeling-elements-invited-talk.pptx](talk-slides/modeling-elements-invited-talk.pptx).

The first slide shows the taxonomy of section 3 ([Fig 2](#fig-2)).

<img src="talk-slides/modeling-elements-invited-talk-1.png" width="800" style="max-width: 100%;" alt="Fig 2">

<a id="fig-2"></a>
Fig 2. Talk slide 1, taxonomy from the joint distribution

The joint distribution is split into P(X), P(Y|X) and P(Y), eqs. (1) and (2) connect the three elements, and the slide also shows that each element changes in two modes, shift and drift.

The second slide shows eq. (4) of section 4 ([Fig 3](#fig-3)).

<img src="talk-slides/modeling-elements-invited-talk-2.png" width="800" style="max-width: 100%;" alt="Fig 3">

<a id="fig-3"></a>
Fig 3. Talk slide 2, prediction from the joint distribution

The three terms of eq. (4) are read as good data, good model and good prediction, and a schematic under each shows the change that breaks it: P(X) shift, P(Y|X) drift and P(Y) shift. The bottom of the slide states the two error terms of eq. (5).

The third slide condenses [Appendix B](#appendix-b-detection-and-implementation-by-cell) ([Fig 4](#fig-4)).

<img src="talk-slides/modeling-elements-invited-talk-3.png" width="800" style="max-width: 100%;" alt="Fig 4">

<a id="fig-4"></a>
Fig 4. Talk slide 3, detection, response and validation by cell

Each of the three elements' shift and drift, and the estimation error, has one panel with its detection, response and validation methods.
