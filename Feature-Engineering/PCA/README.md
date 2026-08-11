# PCA
Rev. 3 | Created: 2026-08-11 | Updated: 2026-08-11 17:34 CDT

> This folder treats PCA as a lineage rather than as a single technique.
> The documents divide into building that lineage, holding it against field data, and placing it in the era of representation learning.

## 1. Scope

Classical PCA carries several assumptions at once, and its variants are what came of relaxing them one at a time. The documents here therefore do not rank techniques by performance; they are organized around **which assumption broke in your data and which branch that sends you to**.

## 2. Documents

Table 1. Documents in this folder

| Document | Description |
|---|---|
| [pca-algorithm.md](pca-algorithm.md) | It writes out the plain procedure — centering, decomposition, scores and loadings, the component count, and the conventions that decide whether two implementations agree. |
| [pca-classical-lineage.md](pca-classical-lineage.md) | It builds the classical lineage as ten branches — computation, probabilistic models, online estimation, robustness, sparsity, non-linearity, high-dimensional asymptotics, data structure, supervision, and distribution — with a summary table at the head serving as the map. |
| [pca-applications.md](pca-applications.md) | It groups the extensions into four directions, says which branch is actually used for each kind of semiconductor measurement data, and closes with a table that goes from a data condition to a method. |
| [pca-modern-lineage.md](pca-modern-lineage.md) | It records where PCA survives inside self-supervised representation learning, and the recent branches that combine functional data, deep learning, and the target variable. |

## 3. Code

Table 2. Scripts in this folder

| File | Description |
|---|---|
| `ccipca.py` | It implements Candid Covariance-free Incremental PCA, which updates the components one sample at a time and averages by sample count instead of a learning rate. It is an instance of the online branch. |

## 4. Order Of Use

1. If the plain procedure is what is wanted, or an implementation has to be checked against it, read the algorithm document.
2. If it is not yet clear which assumption broke, start from the summary table of the classical lineage document.
3. If the data is already understood, start from the selection table of the applications document and walk back up to only the branches it points at.
4. If learned representations are involved, or a neural model is to be combined with the reduction, read the modern lineage document.

The branches are not exclusive. One dataset falling under two or more of them is the normal case, and the way through it is to apply one at a time and check what each one changed.
