# Ensemble Learning
Rev. 1 | Created: 2026-09-10 | Updated: 2026-09-10 21:44 UTC

## 1. Purpose

- **Problem Statement**: Several fitted models are usually combined by whatever is easiest to reach — a majority vote over labels, or an unweighted mean — which throws away the probability each model already computed, and leaves no record of which rows the combination was unsure about.
- **Goal**: Separate the combination rule from the framework that produces the members, so that a reader can pick one of each for a given problem, and can decide how the probability or confidence a member emits enters the combination.
- **Non-Goal**: The principle of each base model and the tuning of its hyperparameters, the reweighting of members on a streaming source, and the ensembles specific to deep learning such as snapshot ensembles and Monte Carlo dropout are not covered.

## 2. Summary

Feeding the meta-learner probabilities rather than labels is what wins, and it wins by more than the choice between hard and soft voting does. A probability records which rows a member was unsure about; a label does not, and no combination rule can recover that afterwards.

Averaging is the weakest way to spend a probability. Soft voting gives every member a fixed share of the answer on every row, so a member whose output is not on a probability scale moves the mean everywhere. Two repairs work: calibrate the members before averaging, or let a meta-learner fit one coefficient per member from the probabilities themselves. Averaging log-odds is not among them, because it removes the cap that the arithmetic mean puts on any one member.

A combination beats the *average* member, never automatically the *best* one. Equation (1) is exact and says the ensemble error is the mean member error minus the spread among members. Adding a weak member raises the first term, and the second term has to pay for it.

The gain has a floor set by correlation, not by count. Equation (2) shows that averaging $M$ members with pairwise correlation $\rho$ leaves $\rho\sigma^{2}$ standing however large $M$ becomes, which is why bagging, random subspaces and boosting all work by making members differ rather than by making more of them.

Appendix B measures these four statements on one held-out split.

## 3. Principle

### 3.1 Why Combining Helps

The ensemble squared error is the mean member error minus the mean spread among members, exactly and for every input. For $M$ members $f_1 \ldots f_M$ with unweighted mean $\bar f$, the ambiguity decomposition holds pointwise [[1](#ref-1)].

$$\left(\bar f - y\right)^{2} = \frac{1}{M}\sum_{m}\left(f_m - y\right)^{2} - \frac{1}{M}\sum_{m}\left(f_m - \bar f\right)^{2} \hspace{19em} (1)$$

Three consequences follow, and they are the whole reason ensembles are used.

- Second term never negative, so the ensemble never worse than the mean member.
- No claim about the best member; a strong member averaged with a weak one can lose.
- Spread among members as the only source of gain, so a member that agrees with the rest adds nothing.

For classification the same idea appears as the Condorcet argument: independent members each correct with probability $p \gt 0.5$ give a majority vote whose accuracy rises with $M$. Independence is the assumption that fails in practice, and section 3.2 is what is left when it does.

### 3.2 The Correlation Floor

Count stops paying once members are correlated. Take $M$ members with equal variance $\sigma^{2}$ and pairwise correlation $\rho$; the variance of their mean is

$$\mathrm{Var}\left(\bar f\right) = \rho\sigma^{2} + \frac{1 - \rho}{M}\sigma^{2} \hspace{19em} (2)$$

The second term vanishes as $M$ grows and the first does not. At $\rho = 0.9$ an infinite ensemble still carries 90% of a single member's variance. This is why every framework in section 6 is a device for lowering $\rho$ — by resampling rows, by hiding columns, by changing the target — and why adding a fourth copy of the same gradient boosting fit changes nothing.

Diversity is necessary but is not a quantity to maximize on its own. Ten pairwise and non-pairwise diversity statistics were compared against ensemble accuracy and none of them tracked it closely enough to be used as a selection criterion [[13](#ref-13)]. Diversity bought by weakening members is paid for out of the first term of equation (1).

### 3.3 Two Independent Axes

Two decisions are made separately and any pair of them is legal. Table 1 names them.

Table 1. The two axes of an ensemble

| Axis | Question | Choices |
|------|----------|---------|
| Framework | What is made different between members | Rows resampled, columns hidden, target changed, algorithm changed |
| Aggregation | What is done with the outputs | Vote, average, weighted average, meta-learner |

Random forest fixes both axes at once and is often read as a single method; it is a row-and-column framework with an averaging rule attached. Stacking fixes only the aggregation axis and leaves the framework free, which is why it accepts members of unrelated kinds.

## 4. Aggregation

### 4.1 Voting

Hard voting takes the majority of the labels; soft voting takes the argmax of the mean probability. Hard voting is what remains when a member emits no probability, and it is not a weaker version of soft voting — it is the version that survives a member whose probabilities are distorted, because a label cannot be overconfident.

Table 2 sets out where each one is available and what each one ignores.

Table 2. Voting rules

| Rule | Input per member | Ignores | Ties |
|------|------------------|---------|------|
| Hard voting | One label | How sure the member was | Possible at even $M$, broken by a rule fixed in advance |
| Soft voting | One probability vector | Nothing the member emitted | Not possible unless probabilities are exactly equal |
| Weighted hard voting | Label and a scalar weight | How sure the member was on this row | Broken by the weights unless two of them are equal |
| Weighted soft voting | Probability vector and a scalar weight | Nothing the member emitted | Not possible in practice |

An even number of members with hard voting forces a tie rule, and a tie rule that picks the first class is a silent bias. Three members, or an odd count generally, removes the question.

### 4.2 Averaging

For regression the counterpart of voting is the mean of the member predictions, and equation (1) governs it unchanged. The mean sits below the average member by exactly the spread among them, so it can fall below every single member at once when the members err in different directions.

### 4.3 Weights

Weights are fitted on out-of-fold predictions under a non-negativity constraint, never on the training fit and never by hand. Least squares under non-negativity is the original prescription for stacked regression and it is what keeps the coefficients interpretable and the combination stable [[7](#ref-7)].

Table 3 lists the ways weights are set, ordered by how much evidence each one uses.

Table 3. Ways to set member weights

| Method | Fitted on | Cost | Failure mode |
|--------|-----------|------|--------------|
| Equal weights | Nothing | None | A weak member drags the mean |
| Inverse validation error | Held-out error per member | One held-out split | Ignores correlation between members |
| Non-negative least squares | Out-of-fold predictions | One cross-validation pass | Overfits when the out-of-fold set is small |
| Meta-learner | Out-of-fold predictions | One cross-validation pass | Same, plus a second model to keep |

Weights fitted on predictions the member has already seen are the classic leak. A model that memorized its training rows looks perfect on them and takes the whole weight, and the failure only appears at deployment.

### 4.4 Log-Odds And Rank

Averaging log-odds is the wrong repair for a member whose probabilities are distorted. The mean of the log-odds, mapped back through the logistic function, is the geometric mean of the odds:

$$\bar z = \frac{1}{M}\sum_{m}\log\frac{p_m}{1 - p_m}, \qquad \bar p = \frac{1}{1 + e^{-\bar z}} \hspace{19em} (3)$$

The arithmetic mean of probabilities caps any one member's influence at $1/M$, because a probability is bounded to $[0, 1]$. The log-odds mean removes that cap: a member emitting $0.9999$ contributes $z \approx 9.2$ and outvotes two members sitting near zero. The distortion the repair was reached for therefore gains weight instead of losing it.

Rank averaging is the option that survives arbitrary distortion, because it uses only the order each member puts the rows in. It buys a ranking objective and gives up the probability, so it applies where the task is to rank a batch and not to answer one row at a time.

## 5. Confidence

### 5.1 Output As A Weight

Whether a member's own output can be used as a weight is decided by the family the member comes from, not by the range the number falls in. Table 4 is that decision for the common families.

Table 4. Whether a member's own output can be used as a weight

| Family | Native output | As a weight | What it needs first |
|--------|---------------|-------------|---------------------|
| Logistic regression | Probability from a fitted link | Usable as emitted where the link is right | Nothing |
| Random forest | Fraction of member votes | Not usable, compressed toward the centre | Isotonic regression |
| Gradient boosting | Logistic of an additive margin | Not usable, overconfident | Platt scaling or isotonic |
| Naive Bayes | Product of independent likelihoods | Not usable, extremely overconfident under dependent features | Isotonic regression |
| SVM | Signed distance to the boundary | Not usable, not on a probability scale | Platt scaling |
| k-nearest neighbours | Fraction of the $k$ neighbours | Not usable, granular in steps of $1/k$ | Isotonic regression |
| Neural network | Softmax of the logits | Not usable, overconfident and worse as the network grows | Temperature scaling |

One row of Table 4 is usable as emitted. The rest produce a score that rises with the true probability but does not equal it. Section 5.2 turns those scores into weights, and section 5.3 spends them.

### 5.2 Calibration

Calibration is a monotone map from the emitted score to a probability, fitted on rows the member did not train on. Platt scaling fits a one-parameter logistic and assumes the distortion is sigmoid-shaped; isotonic regression fits any non-decreasing step function and needs more rows to do it [[9](#ref-9)] [[10](#ref-10)]. Both leave the ranking of rows untouched, so accuracy at a fixed threshold can move only where the threshold crossing moves.

Fitting the map on the training rows destroys it. The map is fitted inside a split of the training set, never on the rows the result is reported on.

The score to read is the Brier score, the mean squared difference between the emitted probability and the outcome [[11](#ref-11)]. It moves with both calibration and discrimination, so it is the one number that says whether the repair helped; since the map leaves the ranking alone, a drop in the Brier score is a gain in the probability itself.

### 5.3 Four Ways To Spend Confidence

Probabilities enter the combination in four places, and the four are not equally good. Ordered by how much of the information they keep:

- **Soft voting** — a fixed share per member on every row. Cheapest, and the one that a distorted member breaks.
- **Per-row confidence weighting** — a share that varies by row, from the entropy of that member's own output on that row.
- **Meta-feature** — probabilities as columns for a second model, which learns both the weights and the correlations between members.
- **Abstention** — the ensemble probability used to decide whether to answer at all.

Per-row weighting normalizes the entropy of member $m$ on row $x$ against its maximum over $K$ classes:

$$c_m(x) = 1 - \frac{H\left(p_m(x)\right)}{\log K} \hspace{19em} (4)$$

Equation (4) rewards confidence, so on raw output it rewards the distortion of Table 4 and hands the largest weight to the worst-scaled member on every row. It may only be applied after section 5.2.

The meta-feature route is the strongest, because it is the only one of the four that sees the members jointly. Stacking with confidences rather than labels was the finding of the original study of the method's design choices [[8](#ref-8)]. The meta-learner fits one coefficient per member from the data and discounts a member the others already cover, which is a decision no fixed averaging rule can make.

Abstention converts confidence into a coverage choice. The optimum rule is to reject where the posterior is below a threshold, and the error and reject rates trade off along a curve fixed by that threshold [[12](#ref-12)]. The rows the ensemble declines are the rows it is most likely to get wrong, so accuracy on the rows it answers rises as coverage falls.

### 5.4 Disagreement

The spread among members estimates uncertainty only where the members are genuinely different. Independently trained members are a practical uncertainty estimate and are competitive with more elaborate Bayesian treatments [[14](#ref-14)], but that rests on the members having distinct blind spots.

Members that share the same predictors and the same missing variable agree confidently in the same wrong place. Their agreement then measures the blind spot they hold in common rather than the difficulty of the row, and the spread then orders the rows in a way the error does not.

The abstention signal of section 5.3 is a different quantity: the ensemble's own posterior, not the disagreement between members. Disagreement is worth measuring, and it is worth trusting only after the check in Appendix B.5 has been run on the members at hand.

## 6. Frameworks

### 6.1 Bagging

Bagging fits each member on a bootstrap resample of the rows and averages or votes the results, which lowers variance and leaves bias where it was [[2](#ref-2)]. It pays only for members that move when the sample moves, so a deep unpruned tree gains and a linear fit barely does.

Random forest adds column sampling at every split, so that members stop agreeing through the one dominant predictor [[3](#ref-3)]. That one step is what separates a random forest from bagged trees of the same depth and count.

### 6.2 Boosting

Boosting fits members in sequence, each one on what the previous ones got wrong, and adds them with weights. AdaBoost reweights the rows that were misclassified [[4](#ref-4)]; gradient boosting fits each new member to the gradient of the loss, which makes the loss function a free choice [[5](#ref-5)]. The widely used implementations descend from the second form and add second-order information, sparsity handling and training on data too large to hold in memory [[6](#ref-6)].

Sequence is the cost. Bagging members are independent and fit in parallel; boosting members are not, and the run stops early or not at all on a held-out curve. Boosting also lowers bias, which bagging does not, so it can beat a bagged ensemble of the same trees on a hard target and can overfit where bagging would not.

### 6.3 Stacking

Stacking fits a second model on the members' out-of-fold predictions [[15](#ref-15)]. Out-of-fold is the whole mechanism: a member's prediction on a row it has trained on is optimistic, and a meta-learner fitted on such rows learns to trust whichever member memorized hardest.

Four rules make the difference between stacking that helps and stacking that leaks.

- Meta-features from cross-validated predictions on the training set, never from the training fit.
- Members refitted on the full training set for scoring the held-out rows.
- Probabilities rather than labels as the meta-features [[8](#ref-8)].
- One column per binary problem, not two, since the pair sums to one and is collinear.

The meta-learner stays small. A regularized linear model on three columns has three coefficients to estimate from the out-of-fold rows; a forest on the same three columns will find structure in the folds that is not there.

### 6.4 Blending

Blending replaces the cross-validation of stacking with a single holdout: members are fitted on one part of the training set and the meta-learner on the other. It costs one fit per member instead of $K+1$ and it cannot leak, since the two parts never meet.

It pays for that twice. The members are fitted on part of the training set rather than all of it, and the meta-learner sees one holdout instead of an out-of-fold prediction for every training row.

## 7. Comparison

The choice follows from two things: what the members emit, and how many rows are left over for fitting the combination. Table 5 reads from the left column.

Table 5. Which combination to use

| Use | When | Why |
|-----|------|-----|
| Hard voting | A member emits no probability, or no held-out set exists to calibrate on | The only rule that needs nothing but labels |
| Soft voting | Members of comparable strength, probabilities already calibrated | No second model to fit or keep |
| Weighted averaging | Regression, members of unequal strength | Weights fitted once on out-of-fold rows |
| Stacking | Members of unrelated kinds, enough rows for a cross-validation pass | Learns weights and correlations together |
| Blending | One fit per member is the budget, or the folds are expensive | No leak, at the price of rows |

Appendix B measures every row of Table 5 against the others, and Appendix E does the same for the frameworks of section 6.

## 8. Further Work

- **Coverage set by guarantee rather than by threshold** — Abstention as set out in section 5.3 fixes coverage by trying thresholds and reading the result, which gives no guarantee on the rows that are answered. Conformal prediction sets the threshold from a target error rate instead, and its distribution-free coverage guarantee holds for any underlying model, so the ensemble of section 6 can be used unchanged. It needs a calibration split held out from the same distribution as deployment and exchangeable with it, which is the condition a drifting process breaks.
- **Weights that follow the process** — The non-negative least squares weights of section 4.3 are fitted once and are fixed thereafter, so a member that degrades keeps its share. Reweighting members from their running loss is now standard in streaming libraries and needs no refitting of the members themselves. It needs labels arriving with a bounded delay and a rule for how fast a weight may move, since a weight that chases noise is worse than a fixed one.

## References

<a id="ref-1"></a>[1] Krogh, A. and Vedelsby, J. (1995). [Neural Network Ensembles, Cross Validation, and Active Learning](https://proceedings.neurips.cc/paper/1994/hash/b8c37e33defde51cf91e1e03e51657da-Abstract.html). *Advances in Neural Information Processing Systems*, 7, 231-238.<br>
<a id="ref-2"></a>[2] Breiman, L. (1996). [Bagging Predictors](https://doi.org/10.1007/BF00058655). *Machine Learning*, 24(2), 123-140.<br>
<a id="ref-3"></a>[3] Breiman, L. (2001). [Random Forests](https://doi.org/10.1023/A:1010933404324). *Machine Learning*, 45(1), 5-32.<br>
<a id="ref-4"></a>[4] Freund, Y. and Schapire, R. E. (1997). [A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting](https://doi.org/10.1006/jcss.1997.1504). *Journal of Computer and System Sciences*, 55(1), 119-139.<br>
<a id="ref-5"></a>[5] Friedman, J. H. (2001). [Greedy Function Approximation: A Gradient Boosting Machine](https://doi.org/10.1214/aos/1013203451). *The Annals of Statistics*, 29(5), 1189-1232.<br>
<a id="ref-6"></a>[6] Chen, T. and Guestrin, C. (2016). [XGBoost: A Scalable Tree Boosting System](https://doi.org/10.1145/2939672.2939785). *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.<br>
<a id="ref-7"></a>[7] Breiman, L. (1996). [Stacked Regressions](https://doi.org/10.1007/BF00117832). *Machine Learning*, 24(1), 49-64.<br>
<a id="ref-8"></a>[8] Ting, K. M. and Witten, I. H. (1999). [Issues in Stacked Generalization](https://doi.org/10.1613/jair.594). *Journal of Artificial Intelligence Research*, 10, 271-289.<br>
<a id="ref-9"></a>[9] Niculescu-Mizil, A. and Caruana, R. (2005). [Predicting Good Probabilities with Supervised Learning](https://doi.org/10.1145/1102351.1102430). *Proceedings of the 22nd International Conference on Machine Learning*, 625-632.<br>
<a id="ref-10"></a>[10] Zadrozny, B. and Elkan, C. (2002). [Transforming Classifier Scores into Accurate Multiclass Probability Estimates](https://doi.org/10.1145/775047.775151). *Proceedings of the 8th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 694-699.<br>
<a id="ref-11"></a>[11] Brier, G. W. (1950). [Verification of Forecasts Expressed in Terms of Probability](https://journals.ametsoc.org/view/journals/mwre/78/1/1520-0493_1950_078_0001_vofeit_2_0_co_2.xml). *Monthly Weather Review*, 78(1), 1-3.<br>
<a id="ref-12"></a>[12] Chow, C. K. (1970). [On Optimum Recognition Error and Reject Tradeoff](https://doi.org/10.1109/TIT.1970.1054406). *IEEE Transactions on Information Theory*, 16(1), 41-46.<br>
<a id="ref-13"></a>[13] Kuncheva, L. I. and Whitaker, C. J. (2003). [Measures of Diversity in Classifier Ensembles and Their Relationship with the Ensemble Accuracy](https://doi.org/10.1023/A:1022859003006). *Machine Learning*, 51(2), 181-207.<br>
<a id="ref-14"></a>[14] Lakshminarayanan, B., Pritzel, A. and Blundell, C. (2017). [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html). *Advances in Neural Information Processing Systems*, 30, 6402-6413.<br>
<a id="ref-15"></a>[15] Wolpert, D. H. (1992). [Stacked Generalization](https://doi.org/10.1016/S0893-6080(05)80023-1). *Neural Networks*, 5(2), 241-259.

---

## Appendix A. Terminology

- **abstention**: A refusal to answer a row whose ensemble probability falls below a threshold.
- **ambiguity**: The mean squared distance between the members and their own mean, the second term of equation (1).
- **bagging**: A framework that fits each member on a bootstrap resample of the rows.
- **base model**: One fitted model inside an ensemble, also called a member.
- **blending**: Stacking whose meta-features come from one holdout split instead of cross-validation folds.
- **boosting**: A framework that fits members in sequence, each on what the previous ones got wrong.
- **Brier score**: The mean squared difference between an emitted probability and the outcome.
- **calibration**: A monotone map from an emitted score to a probability, fitted on rows the member did not train on.
- **confidence**: The number a model emits alongside its label to say how sure it is.
- **conformal prediction**: A procedure that turns any model's scores into prediction sets whose coverage rate is fixed in advance.
- **coverage**: The fraction of rows an abstaining ensemble answers.
- **ensemble**: Several fitted models used together through one combination rule.
- **hard voting**: A combination that takes the majority of the member labels.
- **isotonic regression**: A calibration map that is any non-decreasing step function.
- **leak**: The use, in fitting or in evaluation, of a value the model has already seen as training data.
- **meta-learner**: The second model in stacking, fitted on the members' out-of-fold predictions.
- **out-of-fold prediction**: A prediction for a row made by a member that did not train on it.
- **Platt scaling**: A calibration map that is a one-parameter logistic.
- **soft voting**: A combination that takes the argmax of the mean member probability.
- **stacking**: A framework that fits a meta-learner on the members' out-of-fold predictions.
- **temperature scaling**: A calibration map that divides the logits by one fitted scalar.

## Appendix B. Worked Example

Every measurement in this appendix comes from one pair of splits. Three members of unrelated kinds — logistic regression, a 200-tree random forest and Gaussian naive Bayes — are fitted on 398 rows of the breast cancer data and scored on the 171 held out. The regression parts, B.4 and B.5, use the diabetes data split 309 to 133 the same way.

Table 6. Combination rules on 171 held-out rows

| Method | Combines | Accuracy | Brier |
|--------|----------|----------|-------|
| Stacking on probabilities | Probability | 0.9649 | 0.0275 |
| Logistic regression alone | Nothing | 0.9591 | 0.0265 |
| Hard voting | Label | 0.9591 | N/A |
| Stacking on labels | Label | 0.9591 | 0.0317 |
| Random forest alone | Nothing | 0.9532 | 0.0404 |
| Confidence weighted, calibrated | Calibrated probability | 0.9532 | 0.0360 |
| Soft voting, calibrated | Calibrated probability | 0.9474 | 0.0359 |
| Soft voting, raw | Probability | 0.9415 | 0.0373 |
| Blending | Probability | 0.9298 | 0.0406 |
| Mean of log-odds | Probability | 0.9298 | 0.0572 |
| Naive Bayes alone | Nothing | 0.9240 | 0.0760 |
| Confidence weighted, raw | Probability | 0.9240 | 0.0496 |

Hard voting has no Brier score because it emits no probability, and that empty cell is the cost of the label-only route: nothing downstream can threshold it, abstain on it, or feed it to a decision rule with a cost matrix.

Table 7. Frameworks on the same 171 held-out rows

| Framework | Members | Accuracy | Brier |
|-----------|---------|----------|-------|
| Random forest | 200 trees, rows and columns sampled | 0.9532 | 0.0404 |
| Gradient boosting | 100 trees, sequential | 0.9357 | 0.0422 |
| Bagging | 200 trees, rows sampled | 0.9298 | 0.0450 |
| Single tree | 1 | 0.9064 | 0.0936 |

The code behind each row of Table 6 runs from B.1 through D.3, in that order, and the code behind Table 7 is in Appendix E.

### B.1 Data And Members

The three members, fitted once and reused by every later section. Their accuracies are the three Nothing rows of Table 6.

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=0)

models = {
    'logistic': make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
    'forest': RandomForestClassifier(n_estimators=200, random_state=0),
    'naive_bayes': GaussianNB(),
}
for model in models.values():
    model.fit(X_tr, y_tr)

for name, model in models.items():
    print(f'{name:12s} {accuracy_score(y_te, model.predict(X_te)):.4f}')
```

```text
logistic     0.9591
forest       0.9532
naive_bayes  0.9240
```

### B.2 Hard And Soft Voting

The two rules of section 4.1 applied to the same fits. Naive Bayes is the member that Table 4 marks unusable as emitted, and averaging its probabilities is what puts soft voting below hard voting here.

```python
# hard voting: the majority of the labels, nothing else
labels = np.column_stack([m.predict(X_te) for m in models.values()])
hard = (labels.mean(axis=1) > 0.5).astype(int)

# soft voting: the argmax of the mean probability
probs = np.stack([m.predict_proba(X_te) for m in models.values()])
soft = probs.mean(axis=0).argmax(axis=1)

print(f'hard voting  {accuracy_score(y_te, hard):.4f}')
print(f'soft voting  {accuracy_score(y_te, soft):.4f}')
```

```text
hard voting  0.9591
soft voting  0.9415
```

### B.3 Averaging Log-Odds

Equation (3) on the same members, showing what removing the cap on a member's influence costs.

```python
from sklearn.metrics import brier_score_loss

EPS = 1e-6
p = np.clip(np.stack([m.predict_proba(X_te)[:, 1] for m in models.values()]),
            EPS, 1 - EPS)

arithmetic = p.mean(axis=0)
geometric = 1 / (1 + np.exp(-np.log(p / (1 - p)).mean(axis=0)))

for name, q in (('arithmetic', arithmetic), ('log-odds', geometric)):
    print(f'{name:11s} {accuracy_score(y_te, (q > 0.5).astype(int)):.4f}  '
          f'brier {brier_score_loss(y_te, q):.4f}')
```

```text
arithmetic  0.9415  brier 0.0373
log-odds    0.9298  brier 0.0572
```

### B.4 Weighted Averaging With Out-Of-Fold Weights

The regression counterpart of section 4.2. Weights come from non-negative least squares on out-of-fold predictions, and the last two lines check equation (1) numerically.

```python
from scipy.optimize import nnls
from sklearn.base import clone
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neighbors import KNeighborsRegressor

X_reg, y_reg = load_diabetes(return_X_y=True)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=0)

members = {
    'ridge': make_pipeline(StandardScaler(), RidgeCV()),
    'forest': RandomForestRegressor(n_estimators=300, random_state=0),
    'knn': make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15)),
}
cv = KFold(n_splits=5, shuffle=True, random_state=0)
oof = np.column_stack([
    cross_val_predict(clone(m), Xr_tr, yr_tr, cv=cv) for m in members.values()])
for model in members.values():
    model.fit(Xr_tr, yr_tr)
pred = np.column_stack([m.predict(Xr_te) for m in members.values()])


def rmse(p):
    return np.sqrt(mean_squared_error(yr_te, p))


simple = pred.mean(axis=1)
weight, _ = nnls(oof, yr_tr)     # weights never see a row the member trained on
weight = weight / weight.sum()

for i, name in enumerate(members):
    print(f'{name:8s} rmse {rmse(pred[:, i]):.2f}')
print(f'simple   rmse {rmse(simple):.2f}')
print(f'weighted rmse {rmse(pred @ weight):.2f}  '
      f'weights {dict(zip(members, weight.round(3).tolist()))}')

# the ambiguity decomposition of equation (1), checked on the same rows
mean_member = np.mean([mean_squared_error(yr_te, pred[:, i])
                       for i in range(pred.shape[1])])
ambiguity = np.mean((pred - simple[:, None]) ** 2)
print(f'{mean_member:.1f} - {ambiguity:.1f} = {mean_member - ambiguity:.1f}, '
      f'ensemble mse {mean_squared_error(yr_te, simple):.1f}')
```

```text
ridge    rmse 55.67
forest   rmse 59.42
knn      rmse 56.32
simple   rmse 55.58
weighted rmse 55.29  weights {'ridge': 0.621, 'forest': 0.162, 'knn': 0.216}
3267.2 - 177.9 = 3089.3, ensemble mse 3089.3
```

### B.5 Member Spread As An Uncertainty Signal

The check that section 5.4 requires before disagreement is trusted, run on the members of B.4. The spread separates the two halves by a factor of nearly three and the error does not follow, so on these members the spread carries no information about the error.

```python
spread = pred.std(axis=1)
order = np.argsort(spread)
half = len(order) // 2

for label, idx in (('smallest', order[:half]), ('largest', order[half:])):
    error = np.sqrt(mean_squared_error(yr_te[idx], simple[idx]))
    print(f'{label:9s} n {len(idx)}  rmse {error:.2f}  '
          f'mean spread {spread[idx].mean():.2f}')
```

```text
smallest  n 66  rmse 55.69  mean spread 6.14
largest   n 67  rmse 55.47  mean spread 16.92
```

## Appendix C. Stacking And Blending

### C.1 Stacking On Probabilities Against Labels

The two stacking rows of Table 6. The only difference between them is what the meta-learner is fed, and the coefficients show it discounting the member that Table 4 marks unusable.

```python
from sklearn.model_selection import StratifiedKFold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

# one column per member, not two: the pair sums to one and is collinear
meta_tr = np.column_stack([
    cross_val_predict(clone(m), X_tr, y_tr, cv=cv, method='predict_proba')[:, 1]
    for m in models.values()])
meta_te = np.column_stack([m.predict_proba(X_te)[:, 1] for m in models.values()])

meta = LogisticRegression(max_iter=5000).fit(meta_tr, y_tr)
print(f'on probabilities {accuracy_score(y_te, meta.predict(meta_te)):.4f}  '
      f'brier {brier_score_loss(y_te, meta.predict_proba(meta_te)[:, 1]):.4f}')
print('coefficients', dict(zip(models, meta.coef_[0].round(3).tolist())))

# the same stack fed only the labels
label_tr = np.column_stack([
    cross_val_predict(clone(m), X_tr, y_tr, cv=cv) for m in models.values()])
label_te = np.column_stack([m.predict(X_te) for m in models.values()])
meta_label = LogisticRegression(max_iter=5000).fit(label_tr, y_tr)
print(f'on labels        {accuracy_score(y_te, meta_label.predict(label_te)):.4f}  '
      f'brier {brier_score_loss(y_te, meta_label.predict_proba(label_te)[:, 1]):.4f}')
```

```text
on probabilities 0.9649  brier 0.0275
coefficients {'logistic': 4.392, 'forest': 2.678, 'naive_bayes': 0.969}
on labels        0.9591  brier 0.0317
```

### C.2 Blending

One split instead of K folds. The members see 70% of the training rows and the meta-learner is fitted on the 120 rows they did not see, which is what section 6.4 calls paying twice.

```python
X_fit, X_bl, y_fit, y_bl = train_test_split(
    X_tr, y_tr, test_size=0.3, stratify=y_tr, random_state=1)

blend_members = [clone(m).fit(X_fit, y_fit) for m in models.values()]
blend_tr = np.column_stack([m.predict_proba(X_bl)[:, 1] for m in blend_members])
blend_te = np.column_stack([m.predict_proba(X_te)[:, 1] for m in blend_members])

blender = LogisticRegression(max_iter=5000).fit(blend_tr, y_bl)
print(f'blending {accuracy_score(y_te, blender.predict(blend_te)):.4f}  '
      f'brier {brier_score_loss(y_te, blender.predict_proba(blend_te)[:, 1]):.4f}  '
      f'meta rows {len(y_bl)}')
```

```text
blending 0.9298  brier 0.0406  meta rows 120
```

## Appendix D. Confidence

### D.1 Calibration Before Averaging

The map is fitted inside a five-fold split of the training set, so no held-out row takes part in fitting it. The extreme column counts the predictions past 0.99 or below 0.01, which is the distortion Table 4 assigns to each family.

```python
from sklearn.calibration import CalibratedClassifierCV

for name, model in models.items():
    p = model.predict_proba(X_te)[:, 1]
    print(f'{name:12s} brier {brier_score_loss(y_te, p):.4f}  '
          f'extreme {np.mean(np.abs(p - 0.5) > 0.49):.3f}')

calibrated = {name: CalibratedClassifierCV(m, method='isotonic', cv=5).fit(X_tr, y_tr)
              for name, m in models.items()}
for name, model in calibrated.items():
    print(f'{name:12s} brier {brier_score_loss(y_te, model.predict_proba(X_te)[:, 1]):.4f}'
          f'  (calibrated)')

raw = np.stack([m.predict_proba(X_te) for m in models.values()]).mean(axis=0)
cal = np.stack([m.predict_proba(X_te) for m in calibrated.values()]).mean(axis=0)
for name, q in (('raw', raw), ('calibrated', cal)):
    print(f'soft voting, {name:11s} {accuracy_score(y_te, q.argmax(axis=1)):.4f}  '
          f'brier {brier_score_loss(y_te, q[:, 1]):.4f}')
```

```text
logistic     brier 0.0265  extreme 0.719
forest       brier 0.0404  extreme 0.532
naive_bayes  brier 0.0760  extreme 0.953
logistic     brier 0.0301  (calibrated)
forest       brier 0.0419  (calibrated)
naive_bayes  brier 0.0614  (calibrated)
soft voting, raw         0.9415  brier 0.0373
soft voting, calibrated  0.9474  brier 0.0359
```

### D.2 Per-Row Confidence Weighting

Equation (4) applied twice, to the raw probabilities and to the calibrated ones. On raw output the weighting rewards the distortion and loses to plain soft voting; after calibration it wins.

```python
def confidence_weighted(probs):
    """Weight each member on each row by 1 - normalized entropy of its own output."""
    entropy = -np.sum(probs * np.log(probs + 1e-12), axis=2)
    confidence = 1 - entropy / np.log(probs.shape[2])
    weight = confidence / confidence.sum(axis=0, keepdims=True)
    return (probs * weight[:, :, None]).sum(axis=0)

raw_stack = np.stack([m.predict_proba(X_te) for m in models.values()])
cal_stack = np.stack([m.predict_proba(X_te) for m in calibrated.values()])

for name, stack in (('raw', raw_stack), ('calibrated', cal_stack)):
    q = confidence_weighted(stack)
    print(f'{name:11s} {accuracy_score(y_te, q.argmax(axis=1)):.4f}  '
          f'brier {brier_score_loss(y_te, q[:, 1]):.4f}')
```

```text
raw         0.9240  brier 0.0496
calibrated  0.9532  brier 0.0360
```

### D.3 Abstention

The error-reject tradeoff of section 5.3, measured on the calibrated ensemble.

```python
p = cal_stack.mean(axis=0)
top = p.max(axis=1)
pred_label = p.argmax(axis=1)

print('threshold  coverage  accuracy')
for threshold in (0.50, 0.90, 0.99, 0.999):
    keep = top >= threshold
    print(f'{threshold:9.3f}  {keep.mean():8.3f}  '
          f'{accuracy_score(y_te[keep], pred_label[keep]):.4f}')
```

```text
threshold  coverage  accuracy
    0.500     1.000  0.9474
    0.900     0.871  0.9866
    0.990     0.743  1.0000
    0.999     0.152  1.0000
```

## Appendix E. Frameworks

Table 7, measured on the split of B.1 so that the frameworks and the combination rules are scored on the same 171 rows. Each of these is a self-contained ensemble of trees and needs no separate aggregation step.

```python
from sklearn.ensemble import BaggingClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

frameworks = {
    'single tree': DecisionTreeClassifier(random_state=0),
    'bagging': BaggingClassifier(DecisionTreeClassifier(random_state=0),
                                 n_estimators=200, random_state=0),
    'random forest': RandomForestClassifier(n_estimators=200, random_state=0),
    'boosting': GradientBoostingClassifier(random_state=0),
}
for name, clf in frameworks.items():
    clf.fit(X_tr, y_tr)
    print(f'{name:14s} acc {accuracy_score(y_te, clf.predict(X_te)):.4f}  '
          f'brier {brier_score_loss(y_te, clf.predict_proba(X_te)[:, 1]):.4f}')
```

```text
single tree    acc 0.9064  brier 0.0936
bagging        acc 0.9298  brier 0.0450
random forest  acc 0.9532  brier 0.0404
boosting       acc 0.9357  brier 0.0422
```
