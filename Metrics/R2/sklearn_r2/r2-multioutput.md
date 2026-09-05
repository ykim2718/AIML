# The multioutput Argument of r2_score
Rev. 0 | Created: 2026-09-05 | Updated: 2026-09-05 00:15 CDT

> A note on what `sklearn.metrics.r2_score` does when the target has several columns, and on why
> the same predictions return anything from $-1.17$ to $0.99$ depending on one keyword.

## 1. Scope

`r2_score` accepts a target of shape `(n_samples,)` or `(n_samples, n_outputs)`. With one output
there is one score and nothing to decide. With several there is one score per output, and the
function has to be told what to do with them. The `multioutput` argument is that instruction.

The decision is not cosmetic. The per-output scores are computed against each output's own total
sum of squares, so an output that barely moves gets a large denominator penalty from a small error,
and averaging it beside a well behaved output can carry the reported number below zero. This
document sets out what each setting computes, and what the choice does to one worked problem.

## 2. Definitions

For output $j$ the function computes the ordinary coefficient of determination on that column
alone, with $w_i$ the sample weight and $\bar{y}_j$ the weighted mean of that column.

$$R^2_j = 1 - \frac{\sum_i w_i (y_{ij} - \hat{y}_{ij})^2}{\sum_i w_i (y_{ij} - \bar{y}_j)^2} \hspace{19em} (1)$$

Write $D_j$ for the denominator of equation (1), the weighted total sum of squares of output $j$.
Every setting other than `raw_values` reduces the vector of per-output scores to one number by a
weighted average, and the settings differ only in the weights $a_j$ they use.

$$R^2 = \frac{\sum_j a_j R^2_j}{\sum_j a_j} \hspace{19em} (2)$$

`uniform_average` takes $a_j = 1$, and `variance_weighted` takes $a_j = D_j$. The name says variance
but the weight is the total sum of squares, which is proportional to the variance only when the
sample weights are uniform.

## 3. Settings

Table 1. What each multioutput setting returns

| Setting | Aggregation weight | Returns | Note |
|---|---|---|---|
| `'raw_values'` | None, no aggregation | Array of `n_outputs` | The only setting that keeps the outputs apart |
| `'uniform_average'` | $a_j = 1$ | Scalar | The default |
| `'variance_weighted'` | $a_j = D_j$ | Scalar | Wide outputs dominate |
| Array-like of `n_outputs` | $a_j$ as given | Scalar | The caller states the weights |
| `None` | $a_j = 1$ | Scalar | Same result as `'uniform_average'` |

The default changed to `'uniform_average'` in scikit-learn 0.19 [[1](#ref-1)]. A single-output
target returns a scalar under every setting, since there is nothing to aggregate.

## 4. Worked Example

The problem below has four samples and two outputs. The first output runs in tens, the second sits
near 1 and moves by hundredths, and the predictions are wrong by about the same relative amount on
both.

```python
# Python
Y_TRUE = np.array([[10.0, 1.00], [20.0, 1.02], [30.0, 0.98], [40.0, 1.01]])
Y_PRED = np.array([[12.0, 1.00], [19.0, 1.05], [31.0, 0.95], [39.0, 1.02]])
```

The two denominators are $D_0 = 500.0$ and $D_1 = 0.000875$, a ratio of more than five hundred
thousand to one. That ratio is the whole story of the table.

Table 2. The same predictions under each setting

| Setting | Result |
|---|---|
| `'raw_values'` | `[0.9860, -1.1714]` |
| `'uniform_average'` | `-0.0927` |
| `'variance_weighted'` | `0.9860` |
| `None` | `-0.0927` |
| `[1.0, 3.0]` | `-0.6321` |

Every row is equation (2) applied to the same pair of per-output scores. The uniform average is
$(0.9860 - 1.1714)/2 = -0.0927$. The variance weighted average is
$(0.9860 \times 500 - 1.1714 \times 0.000875)/500.000875 = 0.9860$, where the narrow output
contributes almost nothing because its denominator is almost nothing. The explicit weights give
$(0.9860 \times 1 - 1.1714 \times 3)/4 = -0.6321$.

The second output scores $-1.17$ not because the prediction is far off in absolute terms, the error
there is a few hundredths, but because equation (1) divides by a total sum of squares that is itself
a few hundredths. This is the low variance effect, and `variance_weighted` hides it while
`uniform_average` lets it dominate. Neither is wrong; they answer different questions.

## 5. Constant Output

An output that never moves has $D_j = 0$, and equation (1) divides by zero. The `force_finite`
argument, added in scikit-learn 1.1 [[1](#ref-1)], decides what happens then: by default the score
is 1.0 when the prediction is perfect and 0.0 when it is not, and with `force_finite=False` the
formula runs unguarded.

Table 3. Raw scores where output 1 is constant and predicted imperfectly

| force_finite | Output 0 | Output 1 |
|---|---:|---:|
| `True` | 1.0 | 0.0 |
| `False` | 1.0 | `-inf` |

The default keeps a grid search from crashing on a constant column, at the cost of reporting 0.0
for a case that has no defined score. `variance_weighted` sidesteps the question differently: it
gives that output weight $D_j = 0$, so the aggregate ignores it entirely.

## 6. Choosing

- Report `'raw_values'` while developing. One number per output is the only form that says which
  output is failing.
- Use `'uniform_average'`, the default, when every output matters equally regardless of its scale.
  Expect it to be dragged down by any narrow output.
- Use `'variance_weighted'` when the outputs are the same physical quantity at different scales and
  a wide one deserves more say. Do not use it to make a bad number look better.
- Pass explicit weights when the outputs have known relative importance, and record the weights
  beside the score. A scalar without them cannot be reproduced.
- Check the per-output denominators before reading any aggregate. A near-zero $D_j$ means that
  output's score is about the denominator rather than about the model.

The tables above are produced by `r2_multioutput.py`, in the folder of this document.

## References

<a id="ref-1"></a>[1] scikit-learn developers. [sklearn.metrics.r2_score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html). API reference. The parameter semantics, the default change in 0.19 and the addition of `force_finite` in 1.1 are stated there and in the docstring of the installed package.

---

## Appendix A. Terminology

- **Aggregation weight**: the per-output weight equation (2) averages the per-output scores with.
- **Low variance effect**: the collapse of a variance-based score when the target barely moves, so
  that the denominator rather than the error decides the value.
- **Multioutput**: a target with more than one column, one per predicted quantity.
- **Total sum of squares**: the weighted squared deviation of one output from its own mean, written
  $D_j$ here and forming the denominator of equation (1).
