# Semiconductor Machine Signal Parameterization for ML Modeling: Quantized Signals
Rev. 4 | Created: 2026-08-01 | Updated: 2026-08-10 22:02 CDT

> A quantized signal does not vary continuously. It rests on a ladder of discrete levels and jumps between them, and the number of levels is usually not known in advance.
> This document defines the parameter row for that case, built so that the row width never depends on the level count and so that the original waveform can be rebuilt from what the row stores.

[semiconductor-machine-signal-parameterization-continuous.md](semiconductor-machine-signal-parameterization-continuous.md) reduces a continuous waveform to a fixed-width row of scalars. A quantized signal breaks the assumptions behind that reduction. Its amplitude distribution is a set of spikes rather than a density, its derivative is zero almost everywhere and enormous at a handful of instants, and its natural description is a list of levels and dwell times whose length changes from record to record. Applying the continuous parameter set to it does not fail loudly. It produces values such as RMS and spectral centroid that are computable, stable, and blind to the only structure the signal actually has.

## 1. Scope And Framing

### 1.1 When This Document Applies

A signal belongs here when its samples cluster onto a small set of distinct values and spend most of their time sitting on one of them. The cause does not matter. A converter quantizing an analog rail, a controller stepping a setpoint through a recipe, a valve moving between fixed apertures, and a state machine reporting its own mode all produce the same shape and take the same treatment.

The formal test is the residual ratio of section 4.1. Fit the ladder, subtract it, and compare the leftover to the ladder pitch. When the leftover is far smaller than the pitch, the signal is quantized and this document applies. When it is comparable to the pitch, the ladder is an artifact of the fitting and the continuous treatment is the correct one.

Table 1. Which document each signal belongs to

| Signal | Governing question | Correct document |
|--------|--------------------|------------------|
| Small signal | How much real variation exists above the noise floor | semiconductor-machine-signal-parameterization-continuous.md, section 2 |
| Large signal | How far the response departs from linear as amplitude grows | semiconductor-machine-signal-parameterization-continuous.md, section 3 |
| Quantized signal | Which levels it occupies, for how long, and in what order | This document |

A record may hold a quantized component and a continuous one at the same time, in which case the decomposition methods of section 5 of the companion document apply first and each component is then parameterized by its own rules.

### 1.2 The Two Requirements In Tension

This parameter row answers to two consumers at once, and they pull in opposite directions.

- **A fixed-width tabular row.** Tree ensembles and regression models require every record to present the same columns in the same order. The level count `n` varies per record and is unknown before extraction, so nothing whose width is a function of `n` may appear.
- **Reconstruction.** The stored parameters must be enough to rebuild the waveform, which rules out a summary that only reports averages.

The tension is real and cannot be waved away, because a genuinely fixed-width vector cannot invert an arbitrarily long signal. It is resolved by separating the two jobs into two blocks of the same row and by stating explicitly, in section 5, what fidelity each width buys. The design principle is that fidelity is a declared budget rather than a promise.

### 1.3 Output Contract

The artifact is one row of named scalars per analysis window, identical in form to the contract of the companion document. The window length, the hop, the ladder-estimation method, the transition budget `M`, and the residual threshold that admitted the record are all part of the artifact and must be recorded alongside it.

## 2. The Index Domain

### 2.1 Why The Level Count Must Not Enter The Dimension

The obvious parameterization lists the level values and the time spent on each, which costs `2n` numbers. It is rejected outright. A column count that moves with `n` cannot form a table, records with different `n` cannot be compared, and column `level_7` means a different physical thing in every row.

The resolution is to change domain before extracting anything. Estimate the ladder pitch `Δ` and the anchor `b`, then map every sample to its level index.

```text
k(t) = round( ( x(t) − b ) / Δ )        k(t) ∈ ℤ
```

`k(t)` is integer, dimensionless, and measured in units of one level. Every parameter computed on `k(t)` therefore carries the same meaning whether the record had four levels or four hundred. **The level count is demoted from a dimension of the vector to a value inside it.**

### 2.2 The Parameter Count Result

Once the signal is in the index domain it is piecewise constant, so it is fully described by where it starts and by the transitions that follow. Write `S` for the number of transitions in the window and `T` for the window length in samples.

Table 2. Components of an exact representation

| Component | Count | Note |
|-----------|-------|------|
| Ladder pitch `Δ` and anchor `b` | 2 | Restores physical units |
| Starting index `k₀` | 1 | Fixes the absolute position on the ladder |
| Jumps `d₁ … d_S`, where `d_i = k_i − k_{i−1}` | `S` | Each is a signed integer count of levels |
| Run lengths `L₀ … L_S` | `S` | One is redundant because the run lengths sum to the known `T` |

Exact reconstruction therefore costs

```text
2S + 3     numbers
```

**This is the central result. The parameter count scales with `S`, the number of transitions, and not with `n`, the number of levels.** The level count enters only as precision, since an index needs `⌈log₂ n⌉` bits to store, never as an extra parameter. A record that visits four hundred levels twice is cheaper to represent exactly than one that oscillates between two levels a thousand times.

That reframes the original question. Asking how to parameterize a signal whose level count is unknown is asking the wrong quantity to be small. The quantity to budget is the transition count, and section 5 budgets it.

## 3. Ladder Identification

Everything above assumes `Δ`, `b`, and `n̂` are already known. They are not, and recovering them is the one genuinely hard step. The methods below are ordered so that a method further down is adopted only after the one above it has been tried and shown to fail, since each step down costs more to tune and more to defend.

### 3.1 Recovering The Pitch

Table 3. Methods for recovering the pitch

| Method | Procedure | Note |
|--------|-----------|------|
| Difference histogram | Take the first difference, discard the zeros, and read the mode of the remaining magnitudes | It is one pass and it is the correct first attempt, since in a clean record most transitions are a single level |
| Lattice-span estimation | Scan a candidate pitch `δ` and evaluate the magnitude of the mean of `exp(j2πx/δ)`, which peaks when `δ` matches the true pitch | It is the rigorous version of the above, it survives noise that smears the difference histogram, and it must be guarded against the harmonic peaks at integer divisors of `Δ` |
| Amplitude-histogram peak spacing | Build a kernel density estimate of the amplitude, pick peaks by prominence, and take the median spacing between adjacent peaks | It reports the level positions as well as the pitch, so it feeds section 3.3 directly |
| Code density test | Drive the channel with a known full-scale ramp or sine, histogram the resulting codes, and derive the pitch alongside DNL and INL | It is the instrumentation industry standard, laid down in IEEE Std 1241, and it is the only method here that is traceable, but it requires a controlled excitation |

### 3.2 Recovering The Level Count

The pitch alone gives a candidate count as `n̂ = round((max k − min k)) + 1`, which is the occupied span. That number is correct only when the record actually visits every level in its span, so it is an upper bound and must be checked against a method that counts occupied levels directly.

Table 4. Methods for recovering the level count

| Method | Procedure | Note |
|--------|-----------|------|
| Occupancy threshold | Count the levels holding more than a stated fraction of the samples | It is trivial, and its threshold is exactly the hidden configuration that section 6.1 requires be recorded |
| Optimal one-dimensional clustering | Partition the amplitudes with a dynamic program that is globally optimal in one dimension, then select the cluster count by BIC | The `Ckmeans.1d.dp` formulation runs in `O(kT)` and, unlike ordinary k-means, has no initialization to get unlucky with, which makes it the recommended default |
| Gaussian mixture with BIC | Fit mixtures over a range of component counts and take the BIC minimum | It is the familiar version of the row above and it is worse here, because a one-dimensional problem does not need an EM that can converge to a local optimum |
| Hidden Markov model | Fit an HMM whose states are the levels, which uses the time ordering rather than the amplitude histogram alone | It separates two levels that overlap in amplitude but differ in what they transition to, which no histogram method can do |
| Sticky HDP-HMM | Fit a Bayesian nonparametric HMM in which the state count is part of the posterior rather than a setting | It is the principled answer when `n` must be genuinely inferred, and the variational HMM tools of single-molecule biophysics are the field-proven form of exactly this problem |

The HMM family is the honest answer to an unknown level count and it is also the expensive one. Reach for it when levels overlap in amplitude, when dwell times matter as much as level values, or when the count itself is a reported quantity. Use the clustering row above it otherwise.

### 3.3 Non-Uniform Ladders

A real ladder is rarely perfectly even, and the temptation is to abandon the pitch and store the `n̂` measured level values instead. That reintroduces the `O(n)` width that section 2.1 rejected.

Store the deviation as a curve instead of as a list. Regress the measured level value against its index and keep the low-order coefficients, which is the same construction as integral non-linearity in converter characterization.

```text
value(k) ≈ b + Δ·k + c₂·k²

b   anchor          the ladder position
Δ   pitch           the ladder spacing
c₂  curvature       the ladder bow, one number in place of n
```

One extra parameter absorbs a smoothly bowed ladder. A ladder whose deviations are not smooth is not a bowed ladder but a set of unrelated levels, and that case belongs to the HMM row of section 3.2, where the levels are states with no ordering assumption at all.

### 3.4 Locating The Transitions

The pitch tells us the ladder, not where the signal moved on it. Rounding noisy samples straight to indices produces chatter at every level boundary, which inflates `S` without limit and destroys the count result of section 2.2. Segment first, then round the segment means.

Table 5. Methods for locating the transitions

| Method | Procedure | Note |
|--------|-----------|------|
| Total variation denoising | Minimize the fit error plus a penalty on the summed absolute differences, which drives the solution to be exactly piecewise constant | It is the canonical pre-filter for this signal shape, and the direct algorithms run in near-linear time with a single regularization knob |
| PELT | Detect change points exactly under an additive cost with a pruning rule that keeps the search linear in the window length | It is the industry default for piecewise-constant segmentation and it returns the breakpoints themselves rather than a denoised trace |
| Kalafut–Visscher | Add steps one at a time and stop when an information criterion stops improving | It is the well-known parameter-free step finder, which makes it the right choice when no threshold can be justified |
| Median or Chung–Kennedy filter | Filter with a window that preserves edges instead of averaging across them | It is the cheap pre-clean, and a plain moving average must never be used here because it turns every step into a ramp |
| Hysteresis on the index | Require the index to move by more than one level, or to hold a new level for a stated minimum, before a transition is recorded | It is the crudest de-chatter and its two settings are hidden configuration that must be recorded |

The repository holds naive forms of this stage under `Models/Regression/Step-Like/`, in particular the recursive-partitioning and decision-tree segmentation scripts. They are useful for seeing the mechanism and they are not a substitute for the methods above, which have guarantees the scripts do not.

### 3.5 Selection Guide

```text
Is the ladder ...

├── even, and the record steps one level at a time    -> difference histogram        [3.1]
├── even, but the record is noisy or jumps far        -> lattice-span estimation     [3.1]
├── even, and a controlled excitation is available    -> code density, IEEE 1241     [3.1]
├── uneven but smoothly bowed                         -> add the curvature term      [3.3]
├── uneven, with levels overlapping in amplitude      -> HMM on the levels           [3.2]
└── unknown in count, and the count is a deliverable  -> sticky HDP-HMM              [3.2]

Then, in every case, segment the time axis before rounding      -> TV denoising or PELT   [3.4]
```

## 4. The Minimal Parameter Set

The row has two blocks. Block B is the descriptor summary that the model consumes, and Block A is the reconstruction payload that makes the row invertible. They share the ladder parameters, so the blocks are listed once and counted once.

### 4.1 Block B, The Descriptor Summary

Eleven columns, none of which change width with `n` or with `S`.

Table 6. Block B, the descriptor columns

| Group | Parameter | Definition | Interpretation |
|-------|-----------|------------|----------------|
| Ladder | `q_lsb` | The ladder pitch `Δ`, in physical units per level | It is the resolution of the signal and it sets the unit every other parameter is expressed in |
| Ladder | `q_anchor` | The ladder anchor `b`, the physical value at index zero | It is the absolute position of the ladder, and it is what makes two records comparable in physical units |
| Ladder | `q_inl_c2` | The quadratic coefficient of level value against index | It carries a bowed ladder in one number instead of a list of `n̂` level values |
| Occupancy | `q_nlev` | The estimated occupied level count `n̂` | It is the level count itself, present as a value rather than as a dimension |
| Occupancy | `q_occ_entropy` | The entropy of the level-occupancy distribution divided by `log n̂` | It is how evenly the record spread itself over the levels it visited, on a zero-to-one scale that does not move with `n̂` |
| Occupancy | `q_occ_mode_frac` | The fraction of samples resting on the single most-occupied level | It separates a signal parked on one level from a signal roaming across many |
| Dynamics | `q_trans_rate` | Transitions per second | It is how often the signal moved |
| Dynamics | `q_jump_rms_lsb` | The root mean square of the jump magnitudes, in levels | It is how far each move went, and a value near one means the signal steps rather than leaps |
| Dynamics | `q_dwell_cv` | The coefficient of variation of the dwell times | It separates regular clocking, where the value is near zero, from bursty dwelling, where it exceeds one |
| Dynamics | `q_net_drift_lsb` | The end index minus the start index, in levels | It is the net directional walk across the window, which a symmetric jump statistic cannot show |
| Fidelity | `q_resid_ratio` | The root mean square of the on-level residual divided by `Δ` | It is the validity gate of the entire row, since a value that is not far below one means the signal was never quantized and every parameter above it is an artifact |

### 4.2 The Core Five

When the column budget is tight, five of the eleven carry the structure and the rest refine it.

Table 7. The core five columns

| Parameter | Why it cannot be dropped |
|-----------|--------------------------|
| `q_lsb` | Without the pitch there is no index domain and nothing else has a unit |
| `q_anchor` | Without the anchor the ladder floats and records cannot be compared |
| `q_nlev` | It is the answer to how many levels, which is the question the record was opened to ask |
| `q_occ_entropy` | It is the only column that describes the distribution across levels rather than their extent |
| `q_trans_rate` | It is the only column that uses the time ordering, without which the record is a histogram |

`q_resid_ratio` is not in the core five because it is not a feature. It is an admission test, and a record failing it is routed to the continuous treatment rather than scored with a lower value.

### 4.3 Block A, The Reconstruction Payload

Block A stores the starting index and a fixed budget of `M` transitions, which is Adaptive Piecewise Constant Approximation carried out in the index domain.

```text
k₀                                    1 column
(d₁, L₁) … (d_M, L_M)                2M columns
                                    ─────────────
                                     1 + 2M columns, on top of Block B's 11
```

Total row width is therefore `12 + 2M`, and it is identical for every record in the table.

Four rules govern the budget.

- **`M` is fixed once for the whole dataset, not per record.** Setting it per record reintroduces a variable width. Choose it from the distribution of `S` across the dataset, at roughly the ninety-fifth percentile, so that most records are stored exactly and the tail is truncated.
- **Keep the `M` largest jumps by magnitude, not the first `M` in time.** Truncation should discard the least of the signal, and the largest transitions dominate the reconstruction error.
- **Records with fewer than `M` transitions are padded with null, never with zero.** A zero jump is a real and meaningful value, being a transition that did not move, so padding with zero fabricates events. This is the rule of section 6.4 of the companion document applied to this table.
- **Report the truncation.** Store `q_nlev`'s companion `S` alongside the row so that a reader can see which records were truncated, because a table that silently drops transitions reads as one that captured everything.

Block A is close to inert for a tree ensemble, which cannot make much of column `d_7`. That is expected and is not a reason to drop it. Its job is invertibility, and Block B is the block the model learns from.

### 4.4 Summary Format

```text
[Quantized Signal Data Summary]

Block B, 11 columns
1. Ladder        : LSB (mV), Anchor (mV), INL c2 (mV/level^2)
2. Occupancy     : Levels (-), Norm entropy (-), Mode fraction (%)
3. Dynamics      : Transition rate (1/s), Jump RMS (levels), Dwell CV (-), Net drift (levels)
4. Fidelity      : Residual ratio (-)

Block A, 1 + 2M columns
5. Payload       : k0 (levels), (d_i, L_i) for i = 1 .. M

Recorded alongside, not features
6. Extraction    : Window (s), Hop (s), LSB estimator, Segmenter, Budget M (-),
                   Transitions S (-), Residual admission threshold (-)
```

The extraction block is configuration rather than signal, so it does not enter the model. It must still travel with the table, because section 6.1 shows that every column above is undefined without it.

## 5. Reconstruction Tiers

Each width buys a different guarantee. Declaring which tier a table was built to is part of the artifact, because "reconstructible" without a tier is not a claim that can be checked.

Table 8. Reconstruction tiers

| Tier | Width | What is recovered | Cost |
|------|-------|-------------------|------|
| R0 | 3 | The quantizer ladder, meaning the pitch, the anchor, and the curvature that place every level, but nothing about the signal on it | Three columns |
| R1 | `3 + K` | The amplitude histogram over `K` fixed quantiles, so the signal up to an arbitrary permutation of time | Fixed and small |
| R2 | 12 | A statistically equivalent surrogate, drawable by semi-Markov sampling from the dwell and jump statistics, matching the original in distribution but not sample by sample | Block B plus `k₀` |
| R3 | `12 + 2M` | The waveform with bounded error, exact wherever `S ≤ M` and missing only the smallest transitions elsewhere | Fixed, tunable through `M` |
| R4 | `2S + 3` | The waveform sample by sample | Variable width, so it is a per-record record and not a table column set |

**R3 is the recommendation.** It is the widest tier that is still a fixed-width table, its error falls monotonically as `M` rises, and it degrades to R4 exactly when `M` reaches the largest `S` in the dataset. R4 is the information-theoretic floor for exact reconstruction and is stored as a side artifact keyed to the row when sample-exact replay is genuinely needed.

## 6. Feature-Table Rules

Sections 6.1 through 6.6 of [semiconductor-machine-signal-parameterization-continuous.md](semiconductor-machine-signal-parameterization-continuous.md) govern this table too, covering window definition, leakage, normalization across machines, and the treatment of undefined values. Only the additions specific to a quantized signal are stated here.

### 6.1 Parameters That Carry Hidden Configuration

Table 9. Parameters that carry hidden configuration

| Parameter | Hidden configuration | Consequence if unrecorded |
|-----------|----------------------|---------------------------|
| `q_lsb` | Which estimator of section 3.1 produced it | Two extractions can differ by an integer factor, since a lattice scan can lock onto a divisor of the true pitch |
| `q_nlev` | The occupancy threshold or the model-selection criterion | The count is not a property of the signal alone and is not comparable across extractions |
| `q_trans_rate` | The segmentation method and any hysteresis or minimum-dwell setting | Chatter at a level boundary inflates the rate without bound, so two settings give unrelated numbers |
| `q_resid_ratio` | The threshold at which a record is admitted as quantized | The population the table describes is undefined |
| Block A | The budget `M` and the keep-largest rule | A truncated row is indistinguishable from a complete one |

### 6.2 Naming And Units

The companion document's scheme `<channel>_<domain>_<parameter>_<unit>` carries over with `qd` as the domain tag, alongside the existing `td` and `fd`. A column is therefore `rf_fwd_qd_lsb_mV` or `chamber_pr_qd_entropy_ratio`. Parameters measured in levels take `lsb` as their unit, and this must be written out, because a jump of three is three levels and not three millivolts.

### 6.3 Redundancy Among Parameters

Several columns are algebraically linked and will appear as near-perfect collinearity, exactly as in section 6.5 of the companion document.

- `q_nlev` is bounded by the index span, so it moves with the range of the payload.
- `q_trans_rate` and the mean dwell time are reciprocal, so only one of the two is ever needed.
- `q_occ_entropy` and `q_occ_mode_frac` both measure concentration and disagree only in the tail of the occupancy distribution.

Tree ensembles tolerate this. Linear and distance-based models do not, so for those keep one column from each linked group.

### 6.4 Failure Modes

Table 10. Failure modes and their handling

| Situation | Symptom | Handling |
|-----------|---------|----------|
| The pitch estimate locks onto a divisor of the true pitch | `q_nlev` is an integer multiple of the truth and `q_jump_rms_lsb` is unusually large | Cross-check the difference histogram against the lattice scan and prefer the larger candidate pitch that still explains the data |
| The record never leaves one level | `q_trans_rate` is zero and the entropy and dwell statistics are undefined | Write the undefined columns as null, never as zero, and keep the record, because a parked channel is a real and informative state |
| The signal is not actually quantized | `q_resid_ratio` approaches or exceeds one | Route the record to the continuous treatment rather than storing a quantized row that describes an imaginary ladder |
| Chatter at a level boundary | `q_trans_rate` is enormous and `q_dwell_cv` collapses toward zero | Return to section 3.4 and segment before rounding, rather than filtering the parameter afterward |

## 7. Industry Practice By Field

The same signal shape is studied under different names in fields that arrived at their own standards, and each standard is the best available answer within its own constraints.

Table 11. Industry practice by field

| Field | Standard practice | What it is best at |
|-------|-------------------|--------------------|
| Instrumentation and converter test | Code density histogram test, DNL, INL, ENOB, under IEEE Std 1241 | Characterizing the ladder itself to a traceable standard, given a controlled excitation |
| Single-molecule biophysics | Variational and Bayesian nonparametric HMMs, and the Kalafut–Visscher step finder | Inferring the level count from the data when it is genuinely unknown, which is the closest match to the problem posed here |
| Change-point statistics | PELT, binary segmentation, and total variation denoising | Locating transitions exactly and cheaply in a long record |
| Time-series indexing and retrieval | APCA, PAA, and SAX | Fixed-width piecewise-constant representation with a tunable error bound, which is what Block A implements |
| General time-series machine learning | The `catch22` canonical feature set, and `tsfresh` for a wider sweep | A domain-agnostic descriptor block when no domain-specific one is wanted |
| Symbolic dynamics | Lempel–Ziv complexity and permutation entropy | Single-number complexity measures on the index sequence that are invariant to the level count |

Learned quantization, meaning vector quantization, product quantization, and VQ-VAE codebooks, solves a related problem and is deliberately out of scope. Those methods learn a codebook across a corpus in order to compress many signals jointly, and they need far more data than a per-record parameterization has. They also produce a code whose dimensions have no physical meaning, which forfeits the auditability that every parameter above was chosen to keep. Reach for them only when the corpus is large, the reconstruction target is perceptual rather than metrological, and no one will need to explain an individual column.

## Appendix A. Terminology

The terms below appear in the body without being defined there. They are listed in alphabetical order. Terms already defined in Appendix A of [semiconductor-machine-signal-parameterization-continuous.md](semiconductor-machine-signal-parameterization-continuous.md) are not repeated.

- **Anchor** is the physical value corresponding to level index zero, which fixes the absolute position of the ladder.
- **APCA** is Adaptive Piecewise Constant Approximation, a representation that approximates a series by a fixed number of constant segments of variable length.
- **BIC** is the Bayesian Information Criterion, a model-selection score that penalizes parameter count against likelihood.
- **catch22** is a set of twenty-two canonical time-series features selected from a much larger library for classification performance.
- **Chung–Kennedy filter** is an edge-preserving filter that chooses between forward and backward averages, designed for stepped signals.
- **Code density test** is the procedure of driving a converter with a known excitation and histogramming the output codes in order to characterize the ladder.
- **Coefficient of variation** is the standard deviation divided by the mean, a dimensionless measure of spread.
- **DNL** is Differential Non-Linearity, the deviation of an individual level spacing from the ideal pitch.
- **Dwell time** is the duration for which the signal remains on one level before transitioning.
- **EM** is Expectation-Maximization, the iterative algorithm used to fit mixture models, which converges to a local rather than a global optimum.
- **ENOB** is the Effective Number Of Bits, the resolution a converter actually achieves once noise and distortion are counted.
- **HDP-HMM** is a Hierarchical Dirichlet Process Hidden Markov Model, a Bayesian nonparametric model in which the number of states is inferred rather than set. The sticky variant adds a prior favoring self-transitions, which suppresses spurious rapid state switching.
- **HMM** is a Hidden Markov Model, a model of a sequence of observations generated by an unobserved sequence of discrete states.
- **Index domain** is the representation in which each sample is replaced by its integer level index, obtained by subtracting the anchor and dividing by the pitch.
- **INL** is Integral Non-Linearity, the accumulated deviation of the measured ladder from a straight line.
- **Jump** is the signed change in level index across a transition, measured in levels.
- **Kalafut–Visscher** is a step-detection method that adds steps greedily and stops on an information criterion, requiring no threshold.
- **Ladder** is the set of discrete levels a quantized signal rests on.
- **Lattice-span estimation** is the recovery of the spacing of a discrete-valued distribution from the peaks of its characteristic function.
- **Lempel–Ziv complexity** is a count of the distinct substrings needed to build a sequence, used as a compressibility-based complexity measure.
- **LSB** is the Least Significant Bit, used here as the unit of one level of the ladder.
- **PAA** is Piecewise Aggregate Approximation, which averages a series over fixed-length segments.
- **PELT** is Pruned Exact Linear Time, a change-point detection algorithm that finds the exact optimum under an additive cost.
- **Permutation entropy** is an entropy computed over the ordinal patterns of short subsequences.
- **Pitch** is the spacing between adjacent levels of the ladder, written `Δ` and also called the LSB.
- **Product quantization** is a compression method that splits a vector into subvectors and quantizes each against its own codebook.
- **Residual ratio** is the root mean square of the on-level residual divided by the pitch, used here as the test of whether a signal is quantized at all.
- **Run-length encoding** is the representation of a piecewise-constant sequence as a list of value and duration pairs.
- **SAX** is Symbolic Aggregate approXimation, which converts a series into a symbol string by binning PAA segments.
- **Semi-Markov process** is a state process in which the next state follows a Markov chain but the time spent in each state is drawn from its own distribution, which is the model implied by storing jump and dwell statistics separately.
- **Total variation denoising** is the recovery of a piecewise-constant estimate by penalizing the summed absolute differences of the solution.
- **tsfresh** is a library that computes a large battery of time-series features and filters them by statistical significance.
- **VQ-VAE** is a Vector Quantized Variational Autoencoder, a network that encodes inputs against a learned discrete codebook.
