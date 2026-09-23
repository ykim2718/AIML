# Within-Wafer and Wafer-to-Wafer Variance Decomposition
Rev. 2 | Created: 2026-09-22 | Updated: 2026-09-23 11:08 CDT

- [1. One-Way Variance Components](#1-one-way-variance-components)
  - [1.1 Notation](#11-notation)
  - [1.2 Decomposition Identity](#12-decomposition-identity)
  - [1.3 Random Effects Model](#13-random-effects-model)
  - [1.4 Spread of a Group Mean](#14-spread-of-a-group-mean)
  - [1.5 Limit on a Group's Sample Spread](#15-limit-on-a-groups-sample-spread)
- [2. Application to Wafer Measurements](#2-application-to-wafer-measurements)
  - [2.1 Data](#21-data)
  - [2.2 Variance Decomposition](#22-variance-decomposition)
  - [2.3 W2W Threshold](#23-w2w-threshold)
  - [2.4 WiW Excursion](#24-wiw-excursion)
- [Appendix A. Terminology](#appendix-a-terminology)
- [Appendix B. Decomposition of the Total Sum of Squares](#appendix-b-decomposition-of-the-total-sum-of-squares)
- [Appendix C. Limits of the Decomposition](#appendix-c-limits-of-the-decomposition)
  - [C.1 The Two Coefficients](#c1-the-two-coefficients)
  - [C.2 Correlated Sites Within a Wafer](#c2-correlated-sites-within-a-wafer)
  - [C.3 Wafers as a Sample of One Process](#c3-wafers-as-a-sample-of-one-process)
- [Appendix D. Derivation of the Between-Component Form](#appendix-d-derivation-of-the-between-component-form)
- [Appendix E. Derivation of the Screening Limit](#appendix-e-derivation-of-the-screening-limit)
- [Appendix F. Covariance With a Repeated Argument](#appendix-f-covariance-with-a-repeated-argument)
- [Appendix G. The Chi-Square Distribution](#appendix-g-the-chi-square-distribution)

> ANOVA (analysis of variance) divides the total spread of the observations into a few causes and shows in numbers how much each cause contributes.

When measurements are nested in layers, the eye cannot tell how much each layer contributes to the spread. ANOVA answers that question by cutting the total sum of squares into the sums of squares of the layers. It divides the spread of the values inside a layer and the spread of the means between layers by their own degrees of freedom to make mean squares, and takes their ratio as the F statistic to judge whether the difference between layers is explained by the spread inside a layer alone. Section 1 builds the decomposition in the general terms group and member, and section 2 puts wafer for group and site for member to obtain two values. One is the w2w threshold, the wafer count from which the spread between wafers separates from the measurement noise, and the other is the WiW excursion, a wafer whose site spread exceeds the limit obtained from the wafers before it.

## 1. One-Way Variance Components

For a table nested by a single factor, the total spread is cut into a between-group component and a within-group component, and those two components give the spread of a group mean and the limit on the spread of one group. This section uses the general terms group and member alone; what they stand for is fixed in section 2.

### 1.1 Notation

A table measured at several members of each group is written with the symbols below. Each entry carries where its value comes from: a design value, a measured value, an observed value or a computed value. The measured value is the single $`X_{ij}`$ the table holds, an observed value is a plain calculation on it, and a computed value passes through the variance-component model.

- $`K`$: number of groups. Design value.
- $`N`$: number of members in one group. Design value.
- $`M`$: number of observations in all, $`M = K N`$. Design value.
- $`X_{ij}`$: the value read at the $`j`$ th member of the $`i`$ th group. Measured value.
- $`\bar{X}_i`$: the mean of the $`i`$ th group. Observed value.
- $`\bar{X}`$: the grand mean of all $`M`$ values. Observed value.
- $`s_i`$: the sample standard deviation of the $`N`$ member values of the $`i`$ th group. Observed value.
- $`\sigma_{\mu_n}`$: the sample standard deviation of the first $`n`$ group means. Observed value.
- $`\hat{\sigma}_{\mu_K}`$: the spread of a group mean obtained from the variance components. It is the left side of equations (13), (15), (16) and (17). Computed value.
- $`s_i^2`$: the sample variance of the member values inside the $`i`$ th group, the within-group component. Observed value.
- $`S_{\mathrm{total}}^2`$: the sample variance of all $`M`$ values. Observed value.

### 1.2 Decomposition Identity

The total sum of squares splits, with nothing left over, into the deviations inside the groups and the deviations of the group means. This is the identity ANOVA stands on.

$$\mathrm{SST} = \mathrm{SSW} + \mathrm{SSB} \hspace{19em} (1)$$

- SST: total sum of squares. The total variation, how far every observation lies from the grand mean.
- SSW: within-group sum of squares. The variation inside the groups, how far each member value lies from its own group mean. It is the share the model leaves unexplained, so it is also written SSE (error sum of squares).
- SSB: between-group sum of squares. The variation between the groups, how far each group mean lies from the grand mean. It is the share the factor explains, so it is also written SSA (factor sum of squares).

Written out, the three sums of squares are as below; the derivation of this identity is in [Appendix B](#appendix-b-decomposition-of-the-total-sum-of-squares).

$$\sum_{i}\sum_{j} (X_{ij} - \bar{X})^2 = \sum_{i}\sum_{j} (X_{ij} - \bar{X}_i)^2 + N \sum_{i} (\bar{X}_i - \bar{X})^2 \hspace{19em} (2)$$

Each sum of squares divided by its own degrees of freedom is a mean square (MS), and that is a variance. Turning the two terms on the right into the average within-group variance and the variance of the group means gives the forms below.

$$\overline{S_{\mathrm{within}}^2} = \frac{1}{K} \sum_{i=1}^{K} s_i^2, \qquad S_{\mathrm{between}}^2 = \frac{1}{K-1} \sum_{i=1}^{K} (\bar{X}_i - \bar{X})^2 \hspace{19em} (3)$$

$$S_{\mathrm{total}}^2 = \frac{K(N-1)}{M-1} \overline{S_{\mathrm{within}}^2} + \frac{N(K-1)}{M-1} S_{\mathrm{between}}^2 \hspace{19em} (4)$$

The two coefficients approach 1 as $`K`$ and $`N`$ grow, so the form in common use drops them and keeps the approximation below. How the coefficients go to 1 is in [C.1](#c1-the-two-coefficients).

$$S_{\mathrm{total}} \approx \sqrt{\overline{S_{\mathrm{within}}^2} + S_{\mathrm{between}}^2} \hspace{19em} (5) 🌳$$

The two cases in which the size of the components sets the total standard deviation are below.

- $`S_{\mathrm{between}}^2 = 0`$: the group means are all alike, and the total standard deviation falls to the root mean square of the within-group standard deviations.
- $`S_{\mathrm{between}}^2 \gt 0`$: however small the within-group standard deviation is, group means that lie apart make the total standard deviation far larger than the standard deviation of a single group.

### 1.3 Random Effects Model

In the standard notation of the one-way random effects model, the grand mean and the share of the group are written apart. With $`\mu`$ for the grand mean, $`\alpha_i`$ for the group effect of group $`i`$ and $`e_{ij}`$ for the member error inside a group, a measured value is the sum of three terms.

$$X_{ij} = \mu + \alpha_i + e_{ij} \hspace{19em} (6)$$

$`\mu`$ is a constant that does not depend on the group, $`\mu + \alpha_i`$ is the true mean of group $`i`$, the value every member of that group would have pointed to had there been no member error, and $`\alpha_i`$ is how far that value lies from the grand mean. Conditions differ, so $`\alpha_i`$ differs from group to group, and the one-way random effects model takes $`\alpha_i`$ not as a fixed constant but as a random variable drawn afresh for each group with mean 0. That is what makes the quantity $`\mathrm{Var}(\alpha_i)`$ defined.

$`\alpha_i`$ and $`e_{ij}`$ each have mean 0, and $`e_{ij}`$ is independent both of $`\alpha_i`$ and of the error at another member of the same group. The variances of these two random variables are the two components this document separates.

$$E[\alpha_i] = 0, \quad \mathrm{Var}(\alpha_i) = \sigma_{between}^2, \qquad E[e_{ij}] = 0, \quad \mathrm{Var}(e_{ij}) = \sigma_{within}^2 \hspace{19em} (7)$$

To measure from the data the name $`\sigma_{between}^2`$ that equation (7) gives, it must be tied to an observable quantity, and since $`\alpha_i`$ is not observed the tie is found in the covariance of two member values of the same group. The grand mean $`\mu`$ is a constant and does not enter a covariance, so the only term two members $`j`$ and $`j'`$ of the same group carry alike is $`\alpha_i`$. Expanding the covariance as a bilinear form gives four terms. The second and the third are 0 because the member error is independent of the group effect, and the fourth is 0 because the errors at two different members of the same group are independent of each other. What remains is the first term, $`\mathrm{Cov}(\alpha_i, \alpha_i) = \mathrm{Var}(\alpha_i)`$. How a covariance with a repeated argument becomes a variance is in [Appendix F](#appendix-f-covariance-with-a-repeated-argument).

$$\mathrm{Cov}(X_{ij}, X_{ij'}) = \mathrm{Cov}(\alpha_i, \alpha_i) + \mathrm{Cov}(\alpha_i, e_{ij'}) + \mathrm{Cov}(e_{ij}, \alpha_i) + \mathrm{Cov}(e_{ij}, e_{ij'}) = \mathrm{Cov}(\alpha_i, \alpha_i) = \mathrm{Var}(\alpha_i) = \sigma_{between}^2 \hspace{19em} (8)$$

Equation (8) measures how alike two members of the same group are, yet that value is what measures how far the groups lie apart. $`\mathrm{Var}(\alpha_i)`$ is the amount by which $`\alpha_i`$ scatters as $`i`$ changes, that is, as the group changes, and inside one group $`\alpha_i`$ is a single fixed value. When the $`\alpha_i`$ of that group is large, both member values rise above the grand mean by the same amount, and when it is small both fall by the same amount. All that separates the two values is their own member errors $`e_{ij}`$ and $`e_{ij'}`$. So the larger the spread of $`\alpha_i`$ against the member error, the larger the share the two values move in common and the more alike they are. That is why the ICC of equation (14) is at once the correlation of two members inside a group and the share of the total variance that the between-group variance holds.

Since $`\alpha_i`$ and $`e_{ij}`$ are independent, the variance of a measured value is the sum of the two variances of equation (7).

$$S_{\mathrm{total}}^2 = \sigma_{between}^2 + \sigma_{within}^2 \hspace{19em} (9)$$

An observed group mean is the true mean of the group $`\mu + \alpha_i`$ with the average member error $`\bar{e}_i`$ laid on it. That error is the average of $`N`$ members, so its variance falls by a factor of $`N`$.

$$\bar{X}_i = \mu + \alpha_i + \bar{e}_i, \qquad \mathrm{Var}(\bar{e}_i) = \frac{\sigma_{within}^2}{N} \hspace{19em} (10)$$

### 1.4 Spread of a Group Mean

Since $`\alpha_i`$ and $`\bar{e}_i`$ are independent, the variance of the first $`n`$ group means is the sum of the two variances, where $`s_{\mu}(1..n)`$ is the standard deviation of the group effects of the first $`n`$ groups.

$$\mathrm{Var}(\bar{X}_1, \dots, \bar{X}_n) = s_{\mu}^2(1..n) + \frac{\sigma_{within}^2}{N} \hspace{19em} (11)$$

Taking the square root gives the form that accounts for the observed value.

$$\sigma_{\mu_n} = \sqrt{\frac{\sigma_{within}^2}{N} + s_{\mu}^2(1..n)} \hspace{19em} (12)$$

When the observed value $`s_{\mu}(1..n)`$ from the first $`n`$ groups equals the computed value $`\sigma_{between}`$ from the whole table, that is, when $`s_{\mu}^2(1..n) = \sigma_{between}^2`$, the right term of equation (12) may be written as $`\sigma_{between}^2`$. Taking the $`n`$ that meets this condition as $`K`$, equation (13) writes its subscript as $`\mu_K`$ rather than $`\mu_n`$.

$$\hat{\sigma}_{\mu_K} = \sqrt{\frac{\sigma_{within}^2}{N} + \sigma_{between}^2} \hspace{19em} (13) 🌳$$

Equation (5) measures the spread of one member value, and equation (13) measures the spread of one group mean. The two differ in the within component alone. Averaging $`N`$ members cancels the member error $`e_{ij}`$, which is independent from member to member, and its variance falls by a factor of $`N`$; the group effect $`\alpha_i`$, which all $`N`$ members of that group carry alike, does not fall however much is averaged.

$`\sigma_{between}^2 = S_{\mathrm{total}}^2 - \sigma_{within}^2`$ is equation (9) rewritten, so it holds for any $`n`$. Putting this identity into equation (13) turns the place of $`\sigma_{between}^2`$ into the total standard deviation. The ICC used to write that result short is the share of the total variance that the between-group variance holds.

$$\mathrm{ICC} = \frac{\sigma_{between}^2}{S_{\mathrm{total}}^2} \hspace{19em} (14)$$

Putting equation (14) in to remove $`\sigma_{between}^2`$ gives the right-hand form of equation (15).

$$\hat{\sigma}_{\mu_K} = \sqrt{S_{\mathrm{total}}^2 - \frac{N-1}{N} \sigma_{within}^2} = S_{\mathrm{total}} \sqrt{\mathrm{ICC} + \frac{1 - \mathrm{ICC}}{N}} \hspace{19em} (15)$$

Since $`\sigma_{within}^2 = S_{\mathrm{total}}^2 - \sigma_{between}^2`$, the same formula can be written with the between component in place of the within one, and that is carried out in [Appendix D](#appendix-d-derivation-of-the-between-component-form).

$$\hat{\sigma}_{\mu_K} = \sqrt{\frac{S_{\mathrm{total}}^2 + (N-1) \sigma_{between}^2}{N}} = S_{\mathrm{total}} \sqrt{\frac{1 + (N-1) \mathrm{ICC}}{N}} \hspace{19em} (16)$$

When the group effects are all 0, that is $`\sigma_{between} = 0`$ and hence ICC = 0, equations (15) and (16) leave the spread of the group means as the standard error alone. The observed group means still scatter by the member noise, so the spread is not 0.

$$\hat{\sigma}_{\mu_K} = \frac{S_{\mathrm{total}}}{\sqrt{N}} \hspace{19em} (17)$$

This is the $`\sqrt{N}`$ rule commonly expected, and it does not hold once the ICC is greater than 0.

### 1.5 Limit on a Group's Sample Spread

Whether the spread of one group departs far from the within-group spread seen so far is judged from the distribution of the sample standard deviation. To judge group $`i`$, the $`\sigma_{within}(1..i-1)`$ obtained from the groups before it is taken as the baseline, and the member standard deviation $`s_i`$ of that one group is tested against the limit below. The derivation is in [Appendix E](#appendix-e-derivation-of-the-screening-limit).

$$s_i \gt \sigma_{within}(1..i-1) \sqrt{\frac{\chi^2_{p, N-1}}{N-1}} \hspace{19em} (18)$$

$`\chi^2_{p,\,N-1}`$ is the $`p`$ quantile of the chi-square distribution with $`N-1`$ degrees of freedom, the point a value of that distribution falls below with probability $`p`$; the distribution itself is in [Appendix G](#appendix-g-the-chi-square-distribution). The subscript $`p`$ is the probability and $`N-1`$ is the degrees of freedom the $`N`$ members of one group carry.

## 2. Application to Wafer Measurements

With wafer for group and site for member, the two components of section 1 become the spread between wafers and the spread within a wafer. Process control uses this to cut the total spread into a within-wafer uniformity problem and a wafer-to-wafer reproducibility problem and so find the cause.

### 2.1 Data

The data is [example.csv](example.csv), 200 rows by 14 columns. The table was built by drawing one level of its own for each wafer and laying site noise on top of it. Over that sit a drift along run order that raises the level and the site noise together, six wafers whose level departs far from the drift, and twenty wafers whose site noise is inflated; the seed is fixed, so the same table comes out every time. One row is one wafer, and the column `wafer_id` is a serial number from `wf0001` to `wf0200` that carries the row order of the file, that is, the run order. The remaining columns `S1`~`S13` are the 13 sites on that wafer. There are no missing values and 2600 observations in all.

- All site values: mean 619.8, standard deviation 33.29, minimum 460.34, maximum 797.64.
- Wafer means: minimum 470.4, maximum 767.5, standard deviation 30.06.
- Within-wafer range: mean 43.85, maximum 212.81.
- Wafer uniformity $`s_i / \bar{X}_i`$: median 1.82%, minimum 0.94% (wf0018), maximum 8.77% (wf0185).

Drawing one violin per wafer along run order shows the position and the width of the distribution moving together from wafer to wafer. The early wafers gather near 600 and rise into the 630s later on, and a wafer that hangs alone below means the values there were far lower. With only 13 sites per wafer the shape of a violin is coarse, so the 13 site values are drawn on top of it as points. The line drawn across joins the wafer means and shows how much the position jumps from wafer to wafer.

<img src="wiw-w2w-anova_fig/site_value_violin.png" width="900" style="max-width: 100%;" alt="Fig 1">

Fig 1. Distribution of the site values on each wafer along run order, with the wafer means traced

### 2.2 Variance Decomposition

A one-way ANOVA with wafer as the factor separates the between-wafer component from the within-wafer component. The calculation runs in four steps.

- First, SS: equation (2) cuts the total sum of squares into the between-wafer and the within-wafer sums of squares.
- Second, df: the count of independent pieces of information each sum of squares holds. Between is $`K-1`$, the $`K`$ wafer means less the one grand mean, and within is $`K(N-1)`$, the $`N`$ sites of a wafer less its own mean, taken $`K`$ times.
- Third, MS and F: each sum of squares divided by its own degrees of freedom gives a mean square, and the ratio of the two mean squares is the F statistic.
- Fourth, p: under the assumption that the wafers do not differ, the F statistic follows the $`F(K-1,\ K(N-1))`$ distribution. p is the probability that this distribution yields a value larger than the F the third step produced.

The values so obtained are collected in Table 1.

Table 1. One-way ANOVA with wafer as the factor

| Source | SS | df | MS | F | p | Sigma component |
|---|---:|---:|---:|---:|---:|---:|
| Between wafer | 2,338,049 | 199 | 11,749.0 | 51.92 | ~0 | $`\sigma_{between} = \sqrt{(\mathrm{MS}_{between} - \mathrm{MS}_{within})/N} = 29.77`$ |
| Within wafer | 543,060 | 2400 | 226.3 | | | $`\sigma_{within} = \sqrt{\mathrm{MS}_{within}} = 15.04`$ |

What each column of the table means is below.

- SS: sum of squares. The between wafer row is the SSB of section 1.2 and the within wafer row the SSW; added together they make the SST, 2,881,109.
- df: degrees of freedom. The count of independent pieces of information that sum of squares holds. With 200 wafers between is 199, and the 13 sites of a wafer less its one mean, 12, taken 200 times makes within 2400.
- MS: mean square. SS divided by df, an estimate of a variance. The within value 226.3 is the spread of one site value, and the between value 11,749.0 is the spread of the wafer means with the site spread laid on it.
- F: the ratio of the two MS values. Here 11,749.0 / 226.3 = 51.92. A value that stays near 1 if the wafers do not differ.
- p: the probability of an F that large under the assumption that the wafers do not differ. Here it is close to 0, so that assumption is dropped.
- Sigma component: the standard deviation of the variance component that row carries. Within is the square root of MS within, and between is MS between less MS within, divided by the site count 13, square-rooted.

Table 2. Variance components

| Component | Sigma | Variance | Share |
|---|---:|---:|---:|
| Wafer-to-wafer | 29.77 | 886.4 | 79.7% |
| Within-wafer | 15.04 | 226.3 | 20.3% |
| Total | 33.36 | 1112.6 | 100% |

Adding the two components gives 33.36, a little larger than the observed standard deviation of section 2.1, 33.29. As section 1.2 showed, the plain sum of the two components is an approximation, and the exact relation carries coefficients smaller than 1.

The ICC (intraclass correlation) is the share of the total variance that the between-wafer variance holds, and its definition is equation (14). For this data it is 886.4 / 1112.6 = 0.797. The closer the value is to 1 the more two site values drawn from the same wafer resemble each other, and the closer it is to 0 the less knowing which wafer they came from helps in predicting the value. 0.797 says that 79.7% of the spread of one site value is set by the wafer it sits on, so to reduce the spread the wafer-level conditions come before site-level uniformity.

Since the ICC is not 0, the $`\sqrt{N}`$ rule of equation (17) does not hold for this data. Using $`S_{\mathrm{total}}/\sqrt{N}`$ as it stands gives $`33.36/\sqrt{13}`$ = 9.25, not even a third of the observed spread of the wafer means, 30.06.

The two components of Table 2 are from all 200 wafers seen at once. A single wafer carries no between-wafer variation, so fixing the left edge of a window at the first wafer and extending only its right edge one wafer at a time (expanding window), then recomputing the two components in each window, shows at which wafer count the values settle. The w2w component is as low as 7.77 at four wafers, jumps to 20.53 at ten, and for $`n \ge 100`$ stays within 27.29~30.32 before reaching 29.77 at $`n = 200`$. The WiW component rises steadily from 7.60 at $`n = 5`$ and stays within 11.38~15.14 for $`n \ge 100`$. The site noise of the later wafers is larger than that of the earlier ones. Over the first few dozen wafers both components depart far from the values of Table 2 for want of sample.

### 2.3 W2W Threshold

Fig 2 draws the two terms of equation (13), each computed from the first $`n`$ wafers. The three curves are obtained as below.

- Left term $`\sigma_{within}/\sqrt{N}`$: the per-wafer site variance $`s_i^2`$ averaged over the first $`n`$ wafers as $`\sigma_{within}(1..n) = \sqrt{\frac{1}{n} \sum_{i \le n} s_i^2}`$, divided by $`\sqrt{N}`$. It is the measurement noise that remains in a wafer mean even after averaging $`N`$ sites, a floor that does not vanish even if the wafers are all alike. It comes straight from the data, since it does not use the wafer means.
- Observed curve $`\sigma_{\mu_n}`$: the sample standard deviation of the first $`n`$ wafer means.
- Right term $`\sigma_{between}`$: equation (13) inverted, $`\sqrt{\sigma_{\mu_n}^2 - \sigma_{within}^2(1..n)/N}`$, which is the $`s_{\mu}(1..n)`$ that stands in the place of $`\sigma_{between}`$. It is the spread of the wafer effects that differ from wafer to wafer, that is, the between-wafer variation itself. Where the term under the square root is negative it is undefined and not drawn, and this data has no such $`n`$.

<img src="wiw-w2w-anova_fig/cum_stdev.png" width="900" style="max-width: 100%;" alt="Fig 2">

Fig 2. Cumulative standard deviation of the wafer means with the two terms of equation (13) and the w2w threshold, each computed from the first n wafers only

The place in Fig 2 where the right term begins to lie on the observed curve is called the w2w threshold; taken as the first $`n`$ at which the right term exceeds 98% of the observed value, it is $`n = 9`$ for this data (96% at $`n = 8`$, 98% at $`n = 9`$). Past the w2w threshold the observed curve is in effect the spread of the wafer effects itself.

Carried over to process control, the w2w threshold is the smallest sample a judgement needs. A spread measured before it cannot see the wafer-to-wafer part, so a control limit built on that value is set far too low, and only past this point does the judgement "this spread comes from wafer-level conditions, not from site uniformity" hold. Conversely, a small spread over the stretch before it must not be read as a settled process — all that can be seen there is the measurement noise.

### 2.4 WiW Excursion

When the spread of one wafer departs far from the within-wafer spread seen so far, that wafer is taken as a WiW excursion. The judgement uses equation (18) of section 1.5, and the baseline is the $`\sigma_{within}(1..i-1)`$ obtained from the wafers before it that were not judged excursions. With $`N = 13`$ and $`p = 0.999`$ for this data, $`\chi^2_{0.999,\,12} = 32.91`$, so the coefficient of equation (18) is $`\sqrt{32.91/12} = 1.656`$.

Fig 3 is that judgement. The grey points are the $`s_i`$ of one wafer, the green line is the baseline, the red line is the limit of equation (18), and the wafers past the limit are marked as red points. All three are standard deviations of site values and share a unit, so they are drawn on one axis rather than on a second axis of their own.

<img src="wiw-w2w-anova_fig/wafer_screening.png" width="900" style="max-width: 100%;" alt="Fig 3">

Fig 3. Site value spread of each wafer against the running baseline and the screening limit of equation (18)

A wafer so judged is left out of the baseline update. Taken in as it is, an excursion would raise the baseline and hide the excursions after it, so the more frequent the excursions the blunter the judgement. Against the pooled `sigma_within` of 15.04 over all 200 wafers, the baseline obtained this way is 11.54 at the last wafer, 3.50 lower, and that difference is the share the excursions load onto the pooled value.

The first 20 wafers are used only to build the baseline and are not judged. A baseline resting on a few wafers swings widely on its own and leaves the judgement to chance, and the price is that wf0010 and wf0011, the two that would have exceeded the limit set by the wafers before them, fall outside the judgement.

---

## Appendix A. Terminology

- **ANOVA**: analysis of variance. A method that divides the total sum of squares into the sums of squares of the causes and judges the significance of a cause by the ratio of the mean squares, each divided by its degrees of freedom.
- **bilinear**: the property of being linear in each of the two arguments. For a covariance it is $`\mathrm{Cov}(aX + bY, Z) = a \, \mathrm{Cov}(X, Z) + b \, \mathrm{Cov}(Y, Z)`$ in the first argument and $`\mathrm{Cov}(X, aZ + bW) = a \, \mathrm{Cov}(X, Z) + b \, \mathrm{Cov}(X, W)`$ in the second.
- **Covariance**: the expected product of the deviations of two random variables from their own means. With a repeated argument $`\mathrm{Cov}(Y, Y) = \mathrm{Var}(Y)`$, and that is carried out in [Appendix F](#appendix-f-covariance-with-a-repeated-argument).
- **group effect**: the amount $`\alpha_i`$ by which the true mean of group $`i`$ departs from the grand mean. In the one-way random effects model it is a random variable drawn afresh for each group with mean 0, and its variance is $`\sigma_{between}^2`$. The group of this document is the wafer.
- **ICC**: intraclass correlation. The share of the total variance that the between-group variance holds. It states between 0 and 1 how alike two observations drawn from the same group are, and the group of this document is the wafer. What this document uses is the ICC(1) of the one-way random effects model, whose value differs from the ICC of a two-way model.
- **one-way ANOVA**: an ANOVA whose groups are set by a single factor. It cuts the total sum of squares into a between-group and a within-group sum of squares and no further, and the factor of this document is the wafer.
- **run order**: the row order of the data file. It follows the measurement order and is used as the time axis.
- **running baseline**: the baseline used when judging one wafer. It is the within-wafer component obtained from the wafers before it that were not judged excursions.
- **sigma_between**: the standard deviation of the between-wafer variance component. It is the wafer-to-wafer value of Table 2 and, unlike the sample standard deviation of the wafer means $`S_{\mathrm{between}}`$, has the share of the within-wafer site error removed.
- **sigma_within**: the standard deviation of the within-wafer variance component. It is the square root of MS within.
- **site**: a measurement point on one wafer, matching the columns `S1`~`S13`.
- **Var**: variance. The average of the squared departures of the values from their own mean, the square of the standard deviation. For a sample of $`m`$ observations it is computed as $`\mathrm{Var}(Y) = \frac{1}{m-1} \sum_{i=1}^{m} (Y_i - \bar{Y})^2`$.
- **variogram**: the variance of the difference between the values at two points, written as a function of the distance between them. It is used to measure how alike values are with distance.
- **w2w**: wafer-to-wafer. The variation between wafers.
- **w2w threshold**: the first $`n`$ at which the right term of equation (13) exceeds 98% of the observed spread of the wafer means. Before it the difference between wafers is buried in the measurement noise and does not separate.
- **WiW**: within-wafer. The variation between the sites inside one wafer.
- **WiW excursion**: a wafer whose site standard deviation exceeds the limit set by the running baseline.

## Appendix B. Decomposition of the Total Sum of Squares

Equation (2) comes from writing the deviation from the grand mean as two pieces. One piece is how far a member value lies from its own group mean, and the other is how far that group mean lies from the grand mean.

$$X_{ij} - \bar{X} = (X_{ij} - \bar{X}_i) + (\bar{X}_i - \bar{X}) \hspace{19em} (19)$$

Squaring both sides and summing over $`i`$ and $`j`$ gives three terms. The first two are SSW and SSB, and the third is the cross term of the two pieces, in which $`\bar{X}_i - \bar{X}`$ does not change with $`j`$ and so comes out of the inner sum.

$$\sum_{i}\sum_{j} (X_{ij} - \bar{X})^2 = \sum_{i}\sum_{j} (X_{ij} - \bar{X}_i)^2 + \sum_{i}\sum_{j} (\bar{X}_i - \bar{X})^2 + 2 \sum_{i} (\bar{X}_i - \bar{X}) \sum_{j} (X_{ij} - \bar{X}_i) \hspace{19em} (20)$$

The inner sum in the third term of equation (20) adds up how far the member values of group $`i`$ lie from their own mean. Since the definition of the mean is $`\sum_{j} X_{ij} = N \bar{X}_i`$, that sum is 0 as equation (21) shows, and the whole third term of equation (20) vanishes.

$$\sum_{j=1}^{N} (X_{ij} - \bar{X}_i) = \sum_{j=1}^{N} X_{ij} - N \bar{X}_i = 0 \hspace{19em} (21)$$

Removing the third term from equation (20), rewriting the second term, which adds the same value $`N`$ times over $`j`$, as $`N \sum_i (\bar{X}_i - \bar{X})^2`$, and naming the three sums of squares that remain gives equation (22).

$$\underbrace{\sum_{i}\sum_{j} (X_{ij} - \bar{X})^2}_{\mathrm{SST}} = \underbrace{\sum_{i}\sum_{j} (X_{ij} - \bar{X}_i)^2}_{\mathrm{SSW}} + \underbrace{N \sum_{i} (\bar{X}_i - \bar{X})^2}_{\mathrm{SSB}} \hspace{19em} (22)$$

Read as the sums of squares, equation (22) is equation (2); read by the names braced under them, it is equation (1). This derivation puts no assumption on the data, so equation (1) holds for any table.

## Appendix C. Limits of the Decomposition

### C.1 The Two Coefficients

Writing the two coefficients of section 1.2 as $`a`$ and $`b`$ gives the forms below.

$$a = \frac{K(N-1)}{M-1} = \frac{KN-K}{KN-1}, \qquad b = \frac{N(K-1)}{M-1} = \frac{KN-N}{KN-1} \hspace{19em} (23)$$

Both the numerator and the denominator start from $`KN`$, so it is quicker to look at how far each falls short of 1.

$$1 - a = \frac{K-1}{KN-1}, \qquad 1 - b = \frac{N-1}{KN-1} \hspace{19em} (24)$$

Each shortfall is tied to one size alone. Dividing the numerator and the denominator of $`1-a`$ by $`K`$ and those of $`1-b`$ by $`N`$ gives the forms below.

$$1 - a = \frac{1 - 1/K}{N - 1/K}, \qquad 1 - b = \frac{1 - 1/N}{K - 1/N} \hspace{19em} (25)$$

However large $`K`$ grows, $`1-a`$ stops at $`1/N`$, and however large $`N`$ grows, $`1-b`$ stops at $`1/K`$.

$$\lim_{K \to \infty} (1 - a) = \frac{1}{N}, \qquad \lim_{N \to \infty} (1 - b) = \frac{1}{K} \hspace{19em} (26)$$

In the limit of growing one side alone, then, the coefficients stop at the values below.

$$\lim_{K \to \infty} a = 1 - \frac{1}{N}, \qquad \lim_{N \to \infty} b = 1 - \frac{1}{K} \hspace{19em} (27)$$

So what drives $`a`$ to 1 is the member count per group $`N`$, what drives $`b`$ to 1 is the group count $`K`$, and both must grow for the two coefficients to reach 1 together.

$$\lim_{N \to \infty} a = 1, \qquad \lim_{K \to \infty} b = 1, \qquad \lim_{K, N \to \infty} S_{\mathrm{total}}^2 = \overline{S_{\mathrm{within}}^2} + S_{\mathrm{between}}^2 \hspace{19em} (28)$$

At the $`K = 200`$ and $`N = 13`$ of this document, $`1 - a = 199/2599 = 0.0766`$ is all but equal to $`1/N = 0.0769`$, and $`1 - b = 12/2599 = 0.0046`$ is all but equal to $`1/K = 0.0050`$. That is, $`b`$ may already be taken as 1 while $`a`$ falls 7.7% short, and as long as only 13 sites are measured this shortfall does not shrink however many more wafers are measured. Adding $`\overline{S_{\mathrm{within}}^2} = 226.27`$ and $`S_{\mathrm{between}}^2 = 903.77`$ of this data as they are gives $`S_{\mathrm{total}} = 33.62`$, above the observed 33.29, while attaching the two coefficients brings it to the observed value.

### C.2 Correlated Sites Within a Wafer

Equation (8) takes the errors at two different sites of the same wafer as independent and so erases $`\mathrm{Cov}(e_{ij}, e_{ij'})`$ to 0. A real wafer carries components that move together along the site position, such as a radial pattern or edge roll-off, so that covariance is not 0, and the model of equation (6), which treats a site as a repetition unrelated to its place, buries that spatial structure inside $`e_{ij}`$.

Writing the correlation of two site errors as $`\rho`$, the noise left in a wafer mean is $`\mathrm{Var}(\bar{e}_i) = \sigma_{within}^2 [1 + (N-1)\rho] / N`$, of which equation (10) is the case $`\rho = 0`$. For $`\rho \gt 0`$ the real noise floor is larger than the left term $`\sigma_{within}/\sqrt{N}`$ of equation (13), and the less that is subtracted the more the right term $`s_{\mu}(1..n)`$ is inflated, so the w2w threshold of section 2.3 is caught at an earlier $`n`$ than it should be.

The same correlation bears on the limit of section 2.4. Equation (37) uses a $`\chi^2`$ with $`N-1`$ degrees of freedom because the $`N`$ sites of one wafer yield $`N-1`$ independent pieces of information, and when the sites resemble each other the effective degrees of freedom are fewer, the limit is set too narrow, and the WiW excursion judgement is more sensitive than it should be.

Measuring $`\rho`$ needs a model with the site coordinates as a factor, or a variogram, and this document does not put a value on it from this data.

### C.3 Wafers as a Sample of One Process

Equation (6) takes $`\alpha_i`$ as drawn independently for each group from one distribution with mean 0 and variance $`\sigma_{between}^2`$. The group of this document is the wafer. Only on this assumption are the 200 wafers a sample of the process, and only then does $`\sigma_{between}`$ apply beyond those 200 to the wafers still to come. Taking the same data as fixed effects makes each $`\alpha_i`$ a parameter of its own, so the conclusion stays with those 200 wafers and no single number called a wafer-to-wafer component comes out.

This data departs from that assumption. The wafer mean is 610.2 over the first 50 wafers and 630.9 over the last 50, and two stretches drawn independently from one distribution would not part this far. The $`\alpha_i`$ of the later wafers sit at a different level from the earlier ones.

So the $`\sigma_{between}`$ of this document is not the spread around one process level but a value that also holds the drift across the 200 wafers. A control limit built on it takes the drift as spread the process always carries, and applied to a new wafer it is wider than it should be.

Separating the two needs a model with run order as a factor. One may recompute $`\sigma_{between}`$ from the residuals after removing a time trend term, or split the run into stretches and take the components inside each, and this document does not carry out that separation.

## Appendix D. Derivation of the Between-Component Form

Equation (15) is written with the within component.

$$\hat{\sigma}_{\mu_K}^2 = S_{\mathrm{total}}^2 - \frac{N-1}{N} \sigma_{within}^2 \hspace{19em} (29)$$

Equation (9) gives $`S_{\mathrm{total}}^2 = \sigma_{within}^2 + \sigma_{between}^2`$, so the within component can be replaced by the other two.

$$\sigma_{within}^2 = S_{\mathrm{total}}^2 - \sigma_{between}^2 \hspace{19em} (30)$$

Substituting this and collecting the coefficient of $`S_{\mathrm{total}}^2`$ gives the form below.

$$\hat{\sigma}_{\mu_K}^2 = S_{\mathrm{total}}^2 \left(1 - \frac{N-1}{N}\right) + \frac{N-1}{N} \sigma_{between}^2 = \frac{S_{\mathrm{total}}^2 + (N-1) \sigma_{between}^2}{N} \hspace{19em} (31)$$

Putting in the ICC of equation (14) to remove $`\sigma_{between}^2`$ gives the second form, and taking its square root gives equation (16).

$$\hat{\sigma}_{\mu_K}^2 = S_{\mathrm{total}}^2 \frac{1 + (N-1) \mathrm{ICC}}{N} \hspace{19em} (32)$$

At $`N = 1`$ both forms give $`\hat{\sigma}_{\mu_K} = S_{\mathrm{total}}`$, and as $`N`$ grows $`\hat{\sigma}_{\mu_K}`$ converges to $`\sigma_{between}`$. The more members are measured, the more the within component is erased from a group mean.

## Appendix E. Derivation of the Screening Limit

Below, $`i`$ is the group number and $`j`$ the member number inside that group, keeping the notation of section 1.1. That is, $`X_{ij}`$ is the measured value at the $`j`$ th member of group $`i`$, $`\bar{X}_i`$ is the mean of that group, and $`s_i^2`$ is the sample variance of the member values inside it. The member values inside one group are taken as independent and as following the same normal distribution.

$$X_{ij} \sim \mathcal{N}(\mu + \alpha_i,\ \sigma_{within}^2), \qquad s_i^2 = \frac{1}{N-1} \sum_{j=1}^{N} (X_{ij} - \bar{X}_i)^2 \hspace{19em} (33)$$

Setting a limit needs to know how large $`s_i`$ can grow by chance alone. Even for groups from the same conditions $`s_i`$ differs every time with where the $`N`$ members are drawn, so the distribution of that wobble is what fixes the point from which a value is hard to take as chance. That distribution is the chi-square, the distribution of the sum of the squares of $`m`$ independent standard normal variables, where $`m`$ is its degrees of freedom. Knowing the distribution of $`s_i^2`$ therefore becomes a matter of counting how many standard normal squares it can be written as. Subtracting the true mean $`\mu + \alpha_i`$ of that group from a measured value and dividing by the standard deviation gives a standard normal.

$$Z_{ij} = \frac{X_{ij} - \mu - \alpha_i}{\sigma_{within}} \sim \mathcal{N}(0, 1) \hspace{19em} (34)$$

Since $`X_{ij} - \bar{X}_i = \sigma_{within}(Z_{ij} - \bar{Z}_i)`$, the sum of squares in equation (33) turns into a sum of squares of $`Z`$. Expanding each term as $`(Z_{ij} - \bar{Z}_i)^2 = Z_{ij}^2 - 2 Z_{ij} \bar{Z}_i + \bar{Z}_i^2`$ and summing from $`j = 1`$ to $`N`$ gives three pieces. The first is $`\sum_j Z_{ij}^2`$ as it stands, the second is $`-2 \bar{Z}_i \sum_j Z_{ij}`$ because $`\bar{Z}_i`$ is a constant that does not change with $`j`$ and comes out of the sum, and the third is $`N \bar{Z}_i^2`$, that constant added $`N`$ times.

$$\frac{(N-1) s_i^2}{\sigma_{within}^2} = \sum_{j=1}^{N} (Z_{ij} - \bar{Z}_i)^2 = \sum_{j=1}^{N} Z_{ij}^2 - 2 \bar{Z}_i \sum_{j=1}^{N} Z_{ij} + N \bar{Z}_i^2 \hspace{19em} (35)$$

From the definition of the mean, $`\sum_{j} Z_{ij} = N \bar{Z}_i`$, so the middle term becomes $`2 N \bar{Z}_i^2`$, and joined with the last term it leaves a single $`N \bar{Z}_i^2`$. That is, a sum of standard normal squares less the share of the mean.

$$\frac{(N-1) s_i^2}{\sigma_{within}^2} = \sum_{j=1}^{N} Z_{ij}^2 - 2 N \bar{Z}_i^2 + N \bar{Z}_i^2 = \sum_{j=1}^{N} Z_{ij}^2 - N \bar{Z}_i^2 \hspace{19em} (36)$$

The first term on the right is the sum of the squares of $`N`$ standard normals and so is $`\chi^2_N`$ by definition. $`\bar{Z}_i`$ follows a normal distribution with mean 0 and variance $`1/N`$, so $`\sqrt{N}\,\bar{Z}_i`$ is a standard normal and the second term is $`\chi^2_1`$. In a normal sample the sample mean and the sample variance are independent, so the two shares do not overlap and the degrees of freedom subtract as they are.

$$\frac{(N-1) s_i^2}{\sigma_{within}^2} \sim \chi^2_{N-1} \hspace{19em} (37)$$

Bound by the one constraint equation (21) shows, that the deviations $`X_{ij} - \bar{X}_i`$ sum to 0, only $`N-1`$ of the $`N`$ are free, so the degrees of freedom are $`N-1`$.

Taking the $`\chi^2_{p,\,N-1}`$ of section 1.5, the probability that the left side of equation (37) exceeds that point is the remainder, $`1-p`$.

$$P\left( \frac{(N-1) s_i^2}{\sigma_{within}^2} \gt \chi^2_{p, N-1} \right) = 1 - p \hspace{19em} (38)$$

Solving the bracket for $`s_i`$ and putting the running baseline in the place of the true $`\sigma_{within}`$ gives equation (18). A group past equation (18), then, is a group that produced a value which, had its spread equalled the baseline, would arise with probability $`1-p`$ only.

The baseline is not the true value but an estimate from the groups before it, so strictly the ratio of the two variances follows an F distribution. If the baseline was obtained from $`m`$ groups, its degrees of freedom are $`\nu = m(N-1)`$.

$$\frac{s_i^2}{\sigma_{within}^2(1..i-1)} \sim F(N-1,\ \nu) \hspace{19em} (39)$$

As $`\nu`$ grows, the $`p`$ quantile of $`F(N-1, \nu)`$ converges to $`\chi^2_{p,\,N-1}/(N-1)`$, so equation (18) may be used as it stands. Against the coefficient 1.656 of equation (18), the F with $`\nu = 240`$ at the 21st group, where judging begins, gives 1.696, and at the last group 1.660. Setting the limit 2.4% low early in the run is the price of using the chi-square.

## Appendix F. Covariance With a Repeated Argument

Putting the same random variable into both arguments of a covariance makes the two deviations multiplied together the same value, so it becomes a square, and its expectation is the definition of the variance itself.

$$\mathrm{Cov}(Y, Y) = E[(Y - E[Y])(Y - E[Y])] = E[(Y - E[Y])^2] = \mathrm{Var}(Y) \hspace{19em} (40)$$

The first term of equation (8) is of that shape, so $`\mathrm{Cov}(\alpha_i, \alpha_i) = \mathrm{Var}(\alpha_i)`$, and equation (7) sets that value as $`\sigma_{between}^2`$.

## Appendix G. The Chi-Square Distribution

The limit of equation (18) comes from a quantile of the chi-square distribution. The chi-square distribution is the distribution followed by the sum of the squares of $`m`$ independent standard normal variables, and that $`m`$ is called its degrees of freedom.

$$\chi^2_m = \sum_{k=1}^{m} Z_k^2, \qquad Z_k \sim \mathcal{N}(0, 1) \hspace{19em} (41)$$

Being a sum of squares it is never negative, and the distribution carries a long tail to the right. Its mean and variance are set by the degrees of freedom alone.

$$E[\chi^2_m] = m, \qquad \mathrm{Var}(\chi^2_m) = 2m \hspace{19em} (42)$$

As the degrees of freedom grow, the spread against the mean $`\sqrt{2m}/m = \sqrt{2/m}`$ shrinks, the distribution gathers around the mean and comes closer to symmetric. At the $`m = N-1 = 12`$ of this data, $`\sqrt{2/m} = 0.408`$, so the mean 12 carries a standard deviation of $`\sqrt{24} = 4.90`$, and the $`p = 0.999`$ quantile 32.91 that equation (18) uses lies in the right tail, 4.27 standard deviations from the mean.

[Appendix E](#appendix-e-derivation-of-the-screening-limit) shows that when the member values of one group are independent and follow the same normal distribution, $`(N-1) s_i^2 / \sigma_{within}^2`$ follows a chi-square distribution with $`N-1`$ degrees of freedom. That is why the limit on $`s_i`$ is set from a quantile of that distribution.
