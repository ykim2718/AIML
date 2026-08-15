# Wide-Data
Rev. 1 | Created: 2026-08-15 | Updated: 2026-08-15 10:48 CDT

> This folder holds strategies for wide data, where the columns that describe one wafer far outnumber the wafers themselves.
> The documents start from the (wafer, feature, trace) sensor tensor and work toward a narrow per-wafer table.

## 1. Scope

The data this folder targets is the three-way sensor tensor of semiconductor equipment — on the order of 200 wafers by 200 sensors by a trace of $10^3$ to $10^4$ points. Flattening one wafer yields hundreds of thousands of columns against a few hundred samples, so every analysis has to begin by compressing the trace axis, the sensor axis, or both. The documents here are organized around that compression: what each method preserves, what it discards, and when it fits.

## 2. Documents

Table 1. Documents in this folder

| Document | Description |
|---|---|
| [semiconductor-sensor-trace-wide-to-narrow.md](semiconductor-sensor-trace-wide-to-narrow.md) | It surveys wide-to-narrow conversion methods in order of difficulty — summary statistics, linear projection, feature libraries, tensor decomposition, random convolution kernels, deep representations, and foundation-model embeddings — and closes with a recommended pipeline for this scale and a further-development appendix. |
| [wide-to-narrow-practice.md](wide-to-narrow-practice.md) | It works out the construction details of three methods from that landscape — weight-shared channel-independent encoding, sparse sensor selection with group lasso and sparse PCA, and PLS with a nested lot-grouped validation protocol — down to the objective functions, the code, and the pitfalls. |
