# PCA Modern Lineage
Rev. 2 | Created: 2026-08-11 | Updated: 2026-08-11 14:52 CDT

> PCA was not pushed aside when representation learning arrived; it was rewritten inside it.
> This document records what remains of PCA where neural representations are handled, and what the recent branches that combine functional data, deep learning, and the target variable are aiming at.

The classical lineage divides on the question of which assumption was relaxed. The modern lineage divides on a different one: in an era when representations are learned, **where does the idea of a linear subspace still live?** It lives in three places — as the constraint that keeps training from degenerating, as the tool that reads a representation after the fact, and as the model that fuses structure or a target into the learning itself.

## 1. Reading The Map

Table 1. Where PCA survives in learned representations

| Place | Role of PCA | Section |
|---|---|---|
| A constraint during training | It keeps the representation from collapsing to a point | §2 |
| An analysis after training | It finds axes in the learned representation so it can be read | §3 |
| Fusion with structure | It ties functional structure to a neural encoder | §4 |
| Fusion with a target | It brings the target variable into the reduction | §5 |

The first two change where PCA is used. The last two fuse PCA itself with another model.

## 2. Collapse Prevention And Whitening

Self-supervised learning pulls two views of the same subject together. With only that objective, mapping every input to one vector is optimal, which is called representation collapse. The remedies work on the same object PCA works on — the covariance matrix of the representation.

Table 2. Non-contrastive objectives and the covariance

| Approach | What it constrains | Relation to PCA |
|---|---|---|
| Variance term | It holds the standard deviation of each dimension above a threshold | It stops eigenvalues from going to zero |
| Covariance term | It pushes the covariance between dimensions toward zero | Removing correlation is what PCA does |
| Whitening | It forces the covariance of the representation to the identity | Whitening is itself a post-hoc PCA transform |
| Redundancy reduction | It matches the cross-correlation of two views to the identity | It stops the same axis from being learned twice |

All three amount to one demand — that the dimensions carry different things — and that is the reason principal components are orthogonal. What differs is that PCA delivers the condition in one closed-form solution while training pushes toward it with loss terms.

**Whitening is not free, and that is where practice trips.** Forcing the covariance to the identity amplifies the small-eigenvalue directions, and those directions are mostly noise. A truncation of the components, or a floor on the eigenvalues, is therefore placed in front of it.

## 3. Reading Learned Representations

A trained representation has hundreds to thousands of dimensions, and no person chose its axes. PCA reappears where that representation has to be understood.

Table 3. Post-hoc analysis of a representation

| Tool | Question it answers | Limit |
|---|---|---|
| PCA of activations | How many dimensions the representation actually uses | It does not guarantee the axes are interpretable |
| Effective rank | How widely the eigenvalues are spread | It folds into one number, so it names no cause |
| Sparse autoencoder | It untangles concepts that are superposed on shared axes | Dictionary size and sparsity level change the answer |

PCA and a sparse autoencoder give different answers on the same representation because their assumptions differ. PCA looks for a few orthogonal axes; a sparse autoencoder looks for many axes that need not be orthogonal, of which only a few switch on per sample. Where several concepts share one axis, the latter is what pulls them apart.

## 4. Functional Data And Deep Learning Hybrids

A measurement curve or a sensor trace is a function, not a vector. Unfolding it into one variable per instant turns neighbouring instants into unrelated variables and erases the fact that the curve is smooth. Functional PCA preserves that smoothness through basis functions and a roughness penalty.

The recent branch ties that functional structure to neural models.

Table 4. Combining functional structure with learned encoders

| Form | Idea | What it targets |
|---|---|---|
| FPCA as a front end | Take coefficients with FPCA and feed those to the network | It cuts the input dimension while keeping smoothness |
| Basis layer | Put the basis expansion inside the network as a layer | Coefficients and representation are learned together |
| Functional autoencoder | The encoder and decoder take and return functions | It carries non-linear structure yet returns a curve |
| Neural ODE family | Treat the curve as the solution of a differential equation | It handles irregular sampling intervals |

**The reason to combine them is that two weaknesses cover each other.** FPCA keeps smoothness but is linear; an autoencoder carries non-linearity but does not know the input is a curve. The combination earns its cost on data where sampling intervals differ from row to row or curve lengths wander.

The cost is plain as well. The moment a network is inserted, the ordering and orthogonality of the components are gone, and familiar figures such as explained variance can no longer be quoted. Only reconstruction error remains, which weakens the basis for choosing how many components to keep.

## 5. Supervised Directions

The other direction pulls the target variable `y` into the reduction step. Unsupervised reduction returns the directions of largest variance, with no guarantee that they relate to `y`. A large difference between tools taking the first component while the fine variation tied to yield is pushed down the list is the standard example.

Table 5. Bringing the target into the reduction

| Form | Idea | Note |
|---|---|---|
| Supervised autoencoder | Minimize reconstruction loss and prediction loss together | The weighting between the two losses dominates the result |
| Deep PLS family | Stack the covariance maximization of PLS as layers | Regression and reduction happen in one model |
| Target-aware bottleneck | Keep only what the bottleneck needs to predict `y` | The information-bottleneck formulation |
| Contrastive with labels | Pull same-label samples together and push others apart | It applies only to labelled samples |

**Bringing in the target ties the axes to that target.** On the same data, changing what is predicted means relearning the axes, and handling several targets at once makes the axes a compromise between them. This is the opposite of the property that lets an unsupervised reduction be reused regardless of target, so it is a disadvantage where several models must share one set of axes.

Validation holds a trap too. Performing a `y`-aware reduction once, outside the cross-validation loop, leaks information from the validation set into the axes. The reduction has to happen inside each training fold for the performance estimate to stand.

## Appendix A. Terminology

The terms below appear in the body without being defined there.

- **Autoencoder** is a network trained to encode an input into a low-dimensional code and restore it from that code.
- **Basis function** is one of a set of reference functions used to write a function as a finite list of coefficients.
- **Effective rank** summarizes the spread of the eigenvalue distribution as one number that stands for the dimensions actually in use.
- **FPCA** is Functional PCA, which treats an observation as a curve rather than a vector when taking components.
- **Information bottleneck** is the view that seeks a representation which discards input information while retaining information about the target.
- **Neural ODE** is a network whose hidden state evolves by a differential equation, the solution of which serves as the representation.
- **Non-contrastive** describes self-supervised learning that trains without negative sample pairs.
- **PLS** is Partial Least Squares, which finds the directions of largest covariance between the inputs and the target.
- **Representation collapse** is the state in which different inputs are mapped to the same representation.
- **Self-supervised learning** trains representations from tasks built out of the data itself, without labels.
- **Sparse autoencoder** is an autoencoder penalized so that most of the hidden representation is zero.
- **Whitening** is the linear transform that forces the covariance of a representation to the identity.
