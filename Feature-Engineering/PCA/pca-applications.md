# PCA Applications
Rev. 5 | Created: 2026-08-11 | Updated: 2026-08-11 15:04 CDT

> Knowing the lineage and deciding what to run are two different jobs.
> This document first groups the directions the extensions took, then walks through the measurement data a fab produces and the assumption each kind of it breaks, and closes with a table that goes from a data condition to a method.

PCA has more than ten variants, but only two or three of them are in use at any one site. Which ones depends on the shape the data arrives in and on which of the classical assumptions that data breaks. Semiconductor data states those conditions plainly enough that naming the data type already narrows the candidates.

## 1. Four Directions Of Extension

The extensions grew along four directions, each answering a different limitation of the data or a different purpose of the analysis. The four are not exclusive, and one dataset falling under two of them is the normal case.

Table 1. Four directions and their methods

| # | Direction | Limitation or purpose | Methods |
|---|---|---|---|
| 1 | Nonlinear and kernel | The structure does not separate in a linear subspace | Kernel PCA |
| 2 | Robustness and missing data | Outliers are mixed in and values are missing | Robust PCA, Probabilistic PCA |
| 3 | Complex data structure | Unfolding into a matrix destroys the structure | Sparse PCA, Tensor PCA, Functional PCA |
| 4 | High-dimensional and large data | Memory and computing time are the limit | Randomized PCA, Incremental PCA |

**Nonlinear and kernel.** Kernel PCA takes the components in a feature space through the kernel trick, so it extracts structure that does not separate linearly. It borders on manifold learning, which preserves neighbourhood relations; what separates the two is whether a projection function is produced for new samples.

**Robustness and missing data.** Robust PCA splits the matrix into a normal low-rank part and the outliers, so the two are separated rather than one contaminating the other. Probabilistic PCA rewrites PCA as a generative model, which lets the components be estimated stably even when many values are missing.

**Complex data structure.** PCA treats the columns of a matrix as an unordered set: permute them and the solution comes back the same, with the loadings permuted along. That indifference to column order is exactly what breaks a curve or a grid. Unfold a trace into one variable per instant and the instants become unrelated columns, so nothing in the model states that instant 501 sits between instant 500 and instant 502; unfold a wafer map and neighbouring dies land far apart in the column list. The smoothness of the curve and the adjacency of the grid were carried by the ordering, and unfolding discards the ordering. Sparse PCA cuts down the variables a component uses so that the component can be read, Tensor PCA reduces a multi-way array without unfolding it, and Functional PCA treats a continuous curve as a single unit; the last has been carried further into contrastive forms that look for curve variation peculiar to one group against a background.

**High-dimensional and large data.** Randomized PCA approximates the leading components quickly through a random projection. Incremental PCA splits the data into chunks and updates the components chunk by chunk, which lifts the memory ceiling.

The four repair different things — the shape, the quality of the values, the structure, and the size. A scale problem therefore cannot be solved with a robustness method, and the reverse does not hold either.

## 2. Semiconductor Measurement Data

Classical PCA does not merely require numbers in a table. It carries a set of assumptions about what those numbers are, and every branch of the lineage exists because one of them failed somewhere.

Table 2. What classical PCA assumes about the data

| # | Assumption | What it means |
|---|---|---|
| A1 | Matrix form | One row is one sample and one column is one variable, and the columns carry no order among them |
| A2 | More samples than variables | `n` exceeds `p`, so the sample covariance reaches full rank |
| A3 | No outliers | Every sample is drawn from the same mechanism, so squared error is a fair criterion |
| A4 | Linear structure | The information lies in a linear subspace of the variables |
| A5 | One population | All rows come from the same source, so a difference between rows is a difference in the subject |
| A6 | A fixed subspace | The directions do not move while the data is being collected |
| A7 | Variance is information | A direction of large variance is a direction that matters |

Here `n` is the number of samples, that is the number of rows, and `p` is the number of variables, that is the number of columns. If one wafer is one row, `n` is the number of wafers; if one measured item is one column, `p` is the number of items. A dataset with `p` above `n` is called wide, and A2 is the assumption it breaks.

Table 3. Which assumption each data type breaks

| Data type | Shape | Breaks | How it breaks |
|---|---|---|---|
| Wafer metrology | wafer × measured item | A2, A3 | A lot holds twenty-five wafers, and a mis-measured site is not rare |
| FDC trace | wafer × sensor × time | A1, A2 | Time is an ordered axis, and unfolding it puts `p` in the hundreds of thousands |
| Wafer map | wafer × die grid | A1 | The grid has two ordered axes whose neighbours carry meaning |
| DOE result | condition × response | A7 | The conditions were placed by design, so variance reflects the plan and not the process |
| Marathon test | time × sensor | A6 | The subspace itself drifts across the run |
| Several tools together | wafer × item, tools pooled | A5 | Rows come from different machines, so the largest difference is the machine |

**A single fab produces every row of that table at the same time.** Metrology tables, sensor traces, wafer maps, and long-run logs all come off the same line, and they do not break the same assumption. The choice of method is therefore not made once for the site; it is made per dataset. A metrology table often needs nothing beyond classical PCA, while a trace needs a branch that keeps its structure — one that does not flatten the curve, the grid, or the three-way arrangement into an unordered list of columns, because the ordering is itself the information.

### 2.1 Wafer Metrology

A metrology table puts one wafer on a row and measured items such as thickness, resistance, and critical dimension on the columns. The item count sits in the tens, so computation is never the problem. Two other things are.

The first is units. Thickness is written in nanometres, resistance in ohms, and angles in degrees, so without standardization the item with the largest numbers takes the first component. Working from the correlation matrix is the default.

The second is the sample count, which is A2. Twenty-five wafers in a lot is not enough to estimate components stably. Eigenvalue shrinkage, or a conservative choice of how many components to keep, is needed, and the count should be confirmed by cross-validation rather than read off a scree plot alone.

### 2.2 FDC Trace

A trace is a three-way structure of wafer × sensor × time. Unfolding it into a wafer × (sensor·time) matrix is the most common handling, and `p` passes several hundred thousand the moment that happens.

Table 4. Three ways to handle a trace

| Approach | Idea | Trade-off |
|---|---|---|
| Parameterize, then PCA | Reduce the trace to a few shape parameters and run PCA on them | Variation the parameters do not carry is lost |
| Functional PCA | Take the components while keeping the smoothness of the curve | The curves must be registered onto a common domain first |
| Tensor decomposition | Decompose the three axes without unfolding them | Harder to read and expensive to compute |

**Parameterizing before unfolding is usually the better move.** Once every instant becomes its own variable, neighbouring instants become unrelated columns and the fact that a trace is a curve disappears from the model. A second problem remains on top of that: step length drifts slightly from wafer to wafer, so the same column stops pointing at the same moment.

Functional PCA requires that **every curve be comparable to every other at matching instants on a shared time axis**. Three things have to hold. First, values must be resamplable onto a common grid even when the sampling instants differ from curve to curve. Second, the start and the end must be anchored to the same reference; unless the moment a step begins is set to zero, the same instant refers to different process phases on different wafers. Third, whatever phase difference remains must be removed by registration — step length wanders even under the same recipe, so the time axis is stretched or compressed to line up landmarks such as a peak or a transition.

**Without that alignment a phase difference is read as an amplitude change.** A curve that merely started a little late is scored as a change in magnitude, and the first component ends up carrying the timing mismatch instead of the process variation.

Autocorrelation matters as well. If instants are left as raw variables, adjacent variables hold nearly the same value, so the first component reflects that redundancy. Dynamic PCA, which states the lag explicitly, or parameterization reduces it.

### 2.3 Wafer Map

A per-die value laid out on a grid carries its information in the neighbour relation. Flattening the grid into one row and running PCA turns adjacent dies into unrelated variables and loses the spatial pattern. A branch that keeps the row and column structure, such as 2DPCA, or a treatment that states the spatial correlation, is needed instead.

### 2.4 Multiple Tools And Chambers

When data from several tools is gathered into one table, A5 is the assumption that fails: the rows no longer come from one population. The difference between tools becomes the largest variance and takes the first component. That component is not process variation; it is the tool identifier, and leaving it in pushes the later components down.

Table 5. Removing the tool effect

| Approach | Idea | Note |
|---|---|---|
| Per-tool normalization | Center and standardize within each tool | It erases the real differences between tools as well |
| Contrastive PCA | Find the directions peculiar to the target against a normal group | It needs a background dataset to compare against |
| Multi-block | Keep the tool effect and the process effect in separate blocks | The block structure has to be declared by a person |

Whether to erase the tool difference is decided by the purpose. For yield prediction, erasing it is the better choice; for monitoring tool-to-tool matching, that component is exactly what is being looked for.

### 2.5 Drift Over Time

In long-run data the subspace itself moves slowly, which is A6 failing. Projecting onto fixed components leaves a residual that grows over time, and whether that growth is model decay or a process change is the question that splits the handling.

An incremental branch keeps the residual low by updating the components, but that same update absorbs the change it was supposed to reveal. The usual arrangement therefore fixes a baseline set of components for monitoring, keeps an updating set beside it, and watches the difference between the two.

## 3. Selection Map

The table below goes from a data condition to a method. Read it from the top, and the first row that matches is the answer.

Table 6. From condition to method

| Condition | Method | Direction | Section |
|---|---|---|---|
| The data is a curve and its smoothness is information | Functional PCA, or parameterize then PCA | Structure | §2.2 |
| The data has three or more axes | Tensor or Multilinear PCA | Structure | §2.2 |
| A spatial pattern on a grid is information | 2DPCA, or a treatment that states spatial correlation | Structure | §2.3 |
| Outliers are always present | Robust PCA or L1-PCA | Robustness | §1 |
| Many values are missing | Probabilistic PCA | Robustness | §1 |
| `p` far exceeds `n` | Eigenvalue shrinkage and a conservative component count | Scale | §2.1 |
| The data does not fit in memory | Randomized or Incremental PCA | Scale | §1 |
| The data arrives as a stream | An incremental or streaming branch | Scale | §2.5 |
| A person has to read the components | Sparse PCA or varimax rotation | Structure | §1 |
| The structure is not linear | Kernel PCA | Nonlinear | §1 |
| A prediction target is fixed | PLS or a supervised branch | — | — |
| A tool effect takes the first component | Per-tool normalization or contrastive PCA | — | §2.4 |
| None of the above | Standardize, then classical PCA | — | — |

```text
Is the column a trace?
├── yes → keep the curve structure?
│          ├── yes → functional PCA / parameterize        [2.2]
│          └── no  → unfold, then shrink eigenvalues      [2.1]
└── no  → is there a target variable?
           ├── yes → PLS / supervised branch
           └── no  → are outliers always present?
                      ├── yes → robust PCA
                      └── no  → standardize, then classical PCA
```

**The table failing to pick exactly one row is the normal case.** Data that is a trace, carries outliers, and spans several tools at once is common. Apply the matching rows from the top in order, one at a time, and after each one check whether the reconstruction error and the readability of the components improved. Adding two at once leaves no way to tell which of them worked.

## Appendix A. Terminology

The terms below appear in the body without being defined there.

- **Contrastive PCA** finds the directions whose variance is large in the target data compared with a background dataset.
- **DOE** is Design of Experiments, a study whose conditions are placed by design rather than observed as they come.
- **Dynamic PCA** appends lagged copies of the variables so that autocorrelation enters the model explicitly.
- **FDC** is Fault Detection and Classification, the practice of finding faults from the sensor record of process equipment.
- **FPCA** is Functional PCA, which treats an observation as a curve rather than a vector when taking components.
- **Incremental PCA** updates the components block by block instead of decomposing the whole matrix at once.
- **Kernel trick** computes in a feature space using inner products alone, without ever forming the coordinates of that space.
- **Manifold learning** assumes the data lies on a low-dimensional surface and finds coordinates that preserve neighbour relations.
- **Multi-block** divides the variables into blocks and models the relations between blocks separately.
- **PLS** is Partial Least Squares, which finds the directions of largest covariance between the inputs and the target.
- **Probabilistic PCA** models an observation as a linear map of a low-dimensional latent variable plus isotropic noise.
- **Randomized PCA** narrows the subspace with a random projection and then approximates the leading components.
- **Registration** stretches or compresses a misaligned time axis so that curves line up instant for instant.
- **Robust PCA** decomposes a matrix into a low-rank part and a sparse part, which separates the outliers.
- **Scree** is the plot of eigenvalues in decreasing order, whose bend is taken as the component count.
- **Sparse PCA** drives most of the loadings to zero so that a component can be read.
- **Tensor PCA** reduces an array of three or more axes without unfolding it.
- **Trace** is a series of values recorded from one subject continuously over time.
- **Varimax** is an orthogonal rotation that increases the spread of the loadings to make them easier to read.
- **2DPCA** takes components while keeping the row and column structure of an image instead of flattening it into a vector.
