# Outlier Detection
Rev. 4 | Created: 2026-08-17 | Updated: 2026-08-25 18:06 CDT

> This folder covers the detection of observations that are discordant with the model the rest of a sample follows.
> A survey names the methods and picks among them, and each method worked out in full keeps its own subfolder, holding the document that fixes it and the scripts that run it.

## 1. Scope

An outlier test answers one question, which is whether an observation is inconsistent with the distribution the rest of the sample follows. It does not establish that the observation is wrong, and it does not decide what should be done with it. Detection and treatment are separate steps, and the documents here keep them apart: a flag is the start of an investigation rather than the conclusion of one.

[outlier_detection.md](outlier_detection.md) surveys the field before any of that. It names the statistical, machine learning and deep learning families, records what a semiconductor line runs as part average testing and fault detection, and selects among them by what each one assumes. The documents below take the tests it summarizes and work them out.

The methods collected here work on univariate numeric data. Detecting an observation that is unremarkable in every variable but implausible in their combination is a different problem, and it is handled by the distance-based methods under [Feature-Engineering/Mahalanobis-Distance](../../Feature-Engineering/Mahalanobis-Distance/).

## 2. Methods

Table 1. Methods and the document that fixes each

| Method | Document | Question it answers |
|--------|-------------|---------------------|
| [GESD](GESD/) | [GESD/generalized-esd.md](GESD/generalized-esd.md) | How many outliers does an approximately normal sample carry, when the number is not known in advance? It is the many-outlier procedure of ISO 16269-4, and it reaches outliers that hide one another from a single-outlier test. |
| [Hampel](Hampel/) | [Hampel/hampel-identifier.md](Hampel/hampel-identifier.md) | How far does each observation sit from the bulk, when the sample is not normal? It scores every observation against the median and the MAD, which contamination cannot move, and needs no iteration. It is not part of ISO 16269-4. |
| [ZScore](../../Applied-Statistics/ZScore/) | [Applied-Statistics/ZScore/z-score-ceiling.md](../../Applied-Statistics/ZScore/z-score-ceiling.md) | What is the largest z-score a sample of a given size can produce at all? The classical score is bounded by (n-1)/sqrt(n), so a rule at a fixed cut-off is inert below a certain size and returns a clean verdict it could not have avoided. It is kept under `Applied-Statistics/` because the bound is arithmetic rather than a detection procedure. |

## 3. Order Of Use

1. Inspect the distribution before choosing a test. Every method here assumes a shape, and a sample that violates that shape produces flags recording the violation rather than any discordance.
2. Decide what is known about the number of outliers. A test built for exactly one is not applied repeatedly to find several, because the significance level does not survive the repetition.
3. Choose the method by which assumption the data can carry. The normal-theory tests state a controlled error rate but forfeit it when normality fails; the robust identifier keeps its meaning without normality but reports a score rather than a count.
4. Fix the significance level before the data are seen, so that the threshold is not chosen to produce a preferred answer.
5. Read the margin, not only the verdict. A statistic that exceeds its critical value by a hair and one that exceeds it by a wide gap are different findings, and only the second is robust to the choices made in the first three steps.
6. Investigate a flag before removing it, and report what was removed together with the change it made to the estimates.

## 4. Conventions

- Each method gets a subfolder named after it, holding one document and the scripts that reproduce every number in it.
- Samples that more than one document quotes live in `data/`, so that the documents and their scripts read the same file rather than each carrying its own copy.
- Figures live in the folder named after their document, so a document and its images move together.
- A script writes the samples behind its charts beside them as CSV, so a figure can be redrawn and the numbers a document quotes can be recomputed rather than read off the picture.
