# Semiconductor Machine Signal Parameterization for ML Modeling: Continuous Signals
rev. 6

> A raw machine waveform cannot enter a model as it is. It must first be reduced to a fixed-width row of numbers.
> This document defines that reduction for the small-signal regime, for the large-signal regime, and for the case where both regimes occupy the same record.

Parameterization is the step that turns a variable-length waveform into a fixed set of scalars. The choice of scalars decides what the model can learn, because anything the parameter set discards is invisible downstream no matter how strong the model is. The correct set is not universal. It depends on which regime the signal sits in, since a small signal is judged by how cleanly it resolves against noise while a large signal is judged by how badly it distorts as it approaches the limits of the machine.

## 1. Scope And Framing

### 1.1 Two Regimes

Table 1. The two regimes

| Regime | Governing question | System assumption |
|--------|--------------------|-------------------|
| Small signal | How much real variation exists above the noise floor | The response is treated as linear around an operating point |
| Large signal | How far the response departs from linear as amplitude grows | Non-linearity, saturation, and clipping must be modeled explicitly |

The split is not about absolute magnitude. It is about whether the machine is still operating inside its linear range. A one-volt swing is a small signal on a rail that saturates at fifty volts and a large signal on a rail that saturates at one and a half.

Both regimes assume the signal varies continuously. A signal that instead rests on a ladder of discrete levels and jumps between them defeats that assumption, since its amplitude distribution is a set of spikes and its derivative is zero almost everywhere. That case is a third regime and is covered by [semiconductor-machine-signal-parameterization-quantized.md](semiconductor-machine-signal-parameterization-quantized.md).

### 1.2 Output Contract

Every method in this document produces the same artifact, which is one row of named scalars per analysis window. The window length, the hop, and every threshold used are part of that artifact and must be recorded alongside it. A parameter table whose window definition was not saved cannot be reproduced and cannot be compared against a later extraction.

## 2. Small-Signal Parameters

### 2.1 The Three Core Parameters 🌳

Count, frequency, and peak-to-peak are the starting set. Each answers a different question, and the three do not substitute for one another.

Table 2. The three core parameters

| Parameter | Definition | What it captures | Typical use |
|-----------|------------|------------------|-------------|
| Count | The number of excursions crossing a fixed threshold within the window | Event occurrence, meaning how often something happened | noise burst rate, sensor event rate, acoustic emission hit count |
| Freq | The repetition rate of the signal, or the location of its dominant spectral component in Hz | Periodicity, meaning at what rate it happened | dominant component analysis, vibration analysis, AC response |
| P2P | The distance from the highest peak to the lowest trough in the window | Total excursion, meaning how far it swung | worst-case noise swing, full variation range |

These three are correct but incomplete. Count is blind to how large each event was, P2P is set entirely by two samples and therefore by any single outlier, and frequency says nothing about amplitude. Adding RMS and SNR from the sections below closes those gaps, and that five-parameter set is the recommended minimum.

### 2.2 Time-Domain Parameters

Table 3. Time-domain parameters

| Parameter | Formula or definition | Interpretation |
|-----------|-----------------------|----------------|
| RMS | The root mean square of the samples | It is the energy-equivalent amplitude and is the single most representative size measure of a small signal |
| Mean / DC offset | The arithmetic mean of the samples | It is the operating point the fluctuation is centered on |
| Standard deviation | The RMS after the mean is removed | It is the AC-only size, which equals RMS when the mean is zero |
| P2P | `max - min` | It is the full excursion and is sensitive to single outliers |
| Crest factor | `peak / RMS` | It shows how far spikes protrude above the bulk of the signal, and a value near 1.41 indicates a clean sinusoid |
| Kurtosis | The fourth standardized moment | It rises when impulsive content appears and is a common early indicator of incipient faults |
| Zero-crossing rate | The count of sign changes per second | It is a cheap proxy for dominant frequency that needs no transform |

### 2.3 Frequency-Domain Parameters 🌈

Table 4. Frequency-domain parameters

| Parameter | Definition | Interpretation |
|-----------|------------|----------------|
| Peak frequency | The frequency bin holding the most energy after an FFT | It is the dominant component of the signal |
| Bandwidth | The span `f_high - f_low` over which the signal carries meaningful energy | It shows whether energy is concentrated or spread |
| Spectral centroid | The energy-weighted mean frequency | It tracks the center of mass of the spectrum and moves smoothly, unlike peak frequency which jumps between bins |
| Band power | The integrated power inside a named band | It isolates a component of known physical origin, such as a line frequency or a rotation harmonic |
| SNR | The ratio of signal power to noise power in dB | It is the decisive parameter of small-signal work, since a parameter measured below the noise floor carries no information |
| Noise floor | The power spectral density level of the quiet region | It sets the detection limit for every other parameter |

### 2.4 Linear-Response Parameters

When the small signal is a response to a known excitation rather than a free-running observation, the regime is described by the transfer relation instead of by the waveform alone.

Table 5. Linear-response parameters

| Parameter | Definition | Interpretation |
|-----------|------------|----------------|
| Gain / slope | `ΔOutput / ΔInput` | It is the local sensitivity of the machine at the current operating point |
| Phase | The angular lag of output behind input at a given frequency | It reveals reactive behavior and control loop margin |
| Time delay | The lag expressed in time units | It is the transport or processing delay of the path |
| Coherence | The normalized cross-spectrum between input and output | It shows how much of the output is linearly explained by the input, and a low value invalidates the gain estimate |

### 2.5 Summary Format

```text
[Small Signal Data Summary]
1. Time amplitude  : P2P (mV), RMS (mV), Mean (mV), Crest factor (-), Kurtosis (-)
2. Frequency       : Peak freq (Hz), Centroid (Hz), Bandwidth (Hz), SNR (dB)
3. Event tracking  : Count (events/s), Duty cycle (%), Zero-crossing rate (1/s)
4. Linear response : Gain (-), Phase (deg), Coherence (-)
```

## 3. Large-Signal Parameters

### 3.1 Analysis Viewpoint

Large-signal work asks three questions that small-signal work never raises.

- **Non-linearity** asks how far the output waveform departs from the shape of the input as drive increases.
- **Dynamic limits** ask where the machine stops responding, meaning at what voltage, current, or power it saturates.
- **Power and efficiency** ask what fraction of the supplied energy reaches the output and what fraction becomes heat.

### 3.2 Distortion And Non-linearity

Table 6. Distortion and non-linearity parameters

| Parameter | Definition | Interpretation |
|-----------|------------|----------------|
| THD | The ratio of total harmonic power to fundamental power, in percent or dB | It is the headline measure of waveform distortion under drive |
| THD+N | THD with the noise power included | It is the honest figure when noise is not negligible, and it is always the larger of the two |
| HD2 / HD3 | The individual second and third harmonic levels | They diagnose the source, since even harmonics point to asymmetry and odd harmonics to symmetric compression |
| P1dB | The drive level at which output falls 1 dB below the extrapolated linear line | It is the conventional onset of compression |
| IP3 / TOI | The extrapolated intercept of the third-order intermodulation product with the linear response | It quantifies how badly two simultaneous tones will interfere |
| Clipping ratio | The fraction of samples resting at the saturation rail | It is a direct and cheap detector of hard limiting |

### 3.3 Power And Dynamic Limits

Table 7. Power and dynamic-limit parameters

| Parameter | Definition | Interpretation |
|-----------|------------|----------------|
| Saturation voltage or current | The rail value the output cannot exceed | It is the hard boundary of the operating envelope |
| Slew rate | `dV/dt` maximum, in V/µs | It bounds how fast the output can move, and once exceeded a square input returns as a triangle |
| Peak power | The instantaneous maximum of the product of voltage and current | It sizes the stress the device sees |
| Duty cycle | The fraction of the period the signal is active, in percent | It converts peak power into an average thermal load |
| Power dissipation | Input power minus output power | It is the heat the machine must remove |
| Efficiency | Output power divided by input power | It is the summary figure of merit for a drive stage |

### 3.4 Amplitude And Operating Point

Table 8. Amplitude and operating-point parameters

| Parameter | Role in the large-signal regime |
|-----------|---------------------------------|
| P2P | It is read as the rail-to-rail limit rather than as ordinary variation, so a P2P that stops rising as drive rises is itself the saturation evidence |
| DC bias point | It is the average operating level established under drive, and it is the point the small-signal analysis linearizes around |
| Rise and fall time | They are the large-signal transition times, which are slew-limited rather than bandwidth-limited |

### 3.5 Summary Format

```text
[Large Signal Data Summary]
1. Power & amplitude : P2P (V), RMS power (W), DC operating point (V), Peak power (W)
2. Distortion        : THD (%), THD+N (dB), HD2 / HD3 (dBc), Clipping ratio (%)
3. Non-linear limits : P1dB (dBm), V_sat (V), Slew rate (V/us), IP3 (dBm)
4. Efficiency        : Power dissipation (W), Efficiency (%)
```

## 4. Regime Comparison

Table 9. Small signal against large signal

| Aspect | Small signal | Large signal |
|--------|--------------|--------------|
| Purpose | It detects fine variation, characterizes noise, and estimates linear response | It measures distortion, locates the output ceiling, and characterizes non-linear behavior |
| Core parameters | SNR, RMS, P2P, Freq, Count | THD, P1dB, Slew rate, V_sat, Efficiency |
| System assumption | Linearity may be assumed around an operating point | Non-linearity must be modeled and cannot be assumed away |
| Limiting factor | The noise floor sets the floor of what is measurable | The saturation rail sets the ceiling of what is measurable |
| Failure mode of the parameters | Values sink below the noise floor and become meaningless | Values saturate and stop discriminating between records |

## 5. Regime Decomposition

This section covers the case named separation in the source notes. Decomposition is the more accurate word, because the goal is not to sort records into two bins but to split one record into an additive pair of a slow large-signal component and a fast small-signal residual, after which each component is parameterized by its own section above.

The canonical form is `x(t) = x_large(t) + x_small(t)`, where the large part carries the trend, the step, and the impulse envelope, and the small part carries the fine fluctuation riding on top.

### 5.1 Time-Domain Decomposition

Table 10. Time-domain decomposition methods

| Method | Procedure | Note |
|--------|-----------|------|
| Low-pass and high-pass split | The LPF output is taken as the large component and the HPF output as the small component | It is the simplest method and requires a cutoff that genuinely separates the two scales |
| Detrending | A moving average or a polynomial fit gives the large skeleton, and subtracting it leaves the pure small signal | The window length is the only tuning knob, which makes it easy to reproduce |
| Median filtering | A median filter estimates the baseline while ignoring impulses | It is preferred over a moving average when the large component contains steps |
| EMD | The signal is decomposed into intrinsic mode functions, with early IMFs holding the small signal and late IMFs holding the trend | It is adaptive and needs no cutoff, but mode mixing makes the output unstable across records unless the ensemble variant is used |

### 5.2 Time-Frequency Decomposition

Table 11. Time-frequency decomposition methods

| Method | Procedure | Note |
|--------|-----------|------|
| Wavelet transform | A multi-resolution decomposition separates coarse approximation coefficients from fine detail coefficients | It is the strongest general method here because it keeps the time location of each small-signal event |
| STFT and spectrogram | A two-dimensional threshold on the time-frequency plane divides the high-amplitude region from the low-amplitude region | It is easy to inspect visually, which makes it useful for choosing thresholds for the other methods |

Use this family when the two regimes differ in frequency content or occur at different times. Use the time-domain family when they differ mainly in slowness.

### 5.3 Statistical And Threshold Decomposition

Table 12. Statistical and threshold decomposition methods

| Method | Procedure | Note |
|--------|-----------|------|
| Derivative threshold | The first difference is taken, and intervals with a large `dX/dt` are marked large-signal | It is effectively a slew-rate detector and reacts to steps rather than to level |
| Z-score or IQR | Excursions beyond N standard deviations or beyond the interquartile fence are marked large-signal events, and the remainder is the small signal | The statistics must be computed on a robust estimator, since the large events otherwise inflate the very threshold meant to catch them |
| Rolling kurtosis or skewness | A sliding window computes higher moments to locate intervals containing impulses | It detects the arrival of impulsive content that a level threshold misses |

### 5.4 Learned Decomposition

Reach for this family only when the rule-based methods above fail, since a learned split is harder to audit and its threshold is no longer an explicit number.

Table 13. Learned decomposition methods

| Method | Procedure | Note |
|--------|-----------|------|
| K-means or DBSCAN | Windows are described by amplitude, P2P, and frequency features and then clustered into a small-signal group and a large-signal group | DBSCAN is preferable because it does not require the cluster count and it labels sparse large events as noise points |
| Isolation Forest | A model is fitted on ordinary behavior and the isolation score marks large-signal excursions | It is fast and needs no labels |
| Autoencoder | The network is trained on small-signal windows alone, and a large reconstruction error marks a large-signal event | It requires a clean training set that genuinely contains no large events |

### 5.5 Selection Guide

```text
Are the two regimes separated in ...

├── slowness (trend vs fluctuation)     -> detrending or LPF/HPF        [5.1]
├── frequency or time location          -> wavelet or STFT              [5.2]
├── amplitude or rate of change only    -> Z-score / IQR / derivative   [5.3]
└── none of the above, mixture complex  -> clustering or autoencoder    [5.4]
```

Start at the top. A method lower in the list costs more to tune and to defend, so it should be adopted only when the one above it has been tried and shown to fail.

## 6. Building The Feature Table

### 6.1 Row And Window Definition

One row corresponds to one analysis window, not to one file and not to one machine. Fix the window length and the hop before extraction and store both as columns, because a parameter such as count or RMS is meaningless without the interval it was computed over. If rows will feed a supervised model, no window may extend past the timestamp the prediction is made at, since that is leakage in its most common form.

### 6.2 Naming And Units

Use `<channel>_<domain>_<parameter>_<unit>` so that the column name alone identifies the extraction, as in `rf_fwd_td_rms_mV` or `chamber_pr_fd_snr_dB`. Units belong in the name or in an explicit schema and must never be implied, since a table mixing V and mV in one column is indistinguishable from a table recording a real thousandfold shift.

### 6.3 Parameters That Carry Hidden Configuration

Table 14. Parameters that carry hidden configuration

| Parameter | Hidden configuration | Consequence if unrecorded |
|-----------|----------------------|---------------------------|
| Count | The threshold and the hysteresis | Counts from two extractions are not comparable |
| Duty cycle | The active-state definition | The same waveform yields different percentages |
| THD | The harmonic order included and whether noise is counted | THD and THD+N differ and are silently mixed |
| SNR | The band designated as noise | The value moves by tens of dB |
| P1dB | The excitation sweep used | It is undefined without a controlled drive sweep |

Large-signal distortion parameters share one precondition. THD, P1dB, and IP3 all require a known and controlled excitation, so they are not comparable across records collected under different drive conditions. When the excitation is not controlled, keep the descriptive parameters of section 3.3 and drop the distortion parameters rather than computing values that cannot be compared.

### 6.4 Missing And Undefined Values

A parameter that is undefined for a window must be written as null, never as zero. THD with no detectable fundamental, gain with no excitation, and peak frequency in a silent window are all undefined, and encoding them as zero teaches the model that a quiet window is a perfectly linear one.

### 6.5 Redundancy Among Parameters

Several parameters are algebraically linked and will appear as near-perfect collinearity.

- Crest factor is `peak / RMS`, so it is fully determined by the other two.
- P2P is roughly `2 × peak` for a symmetric signal.
- Standard deviation equals RMS whenever the mean is zero.

Tree ensembles tolerate this. Linear and distance-based models do not, so for those keep one parameter from each linked group or apply a decorrelating transform before fitting.

### 6.6 Normalization Across Machines

Small-signal parameters carry the absolute scale of the channel, so when several machines contribute rows, per-machine offsets let the model identify the machine instead of learning the physics. Normalize within machine or within chamber, and validate by grouped split so that a machine present in training is never also scored in validation.

### 6.7 Modeling The Two Regimes Together

Table 15. Handling for each mixture of regimes

| Situation | Recommended handling |
|-----------|----------------------|
| Both regimes carry information and appear in every record | Extract both parameter blocks into one wide row and add a regime-share column such as the fraction of window energy in the large component |
| The regimes have different targets or different physics | Fit one model per regime and route rows using the decomposition of section 5 |
| Large-signal events are rare | Model the small signal continuously and treat large-signal occurrence as a separate event-detection target |

Record the regime label produced by section 5 as a column in every case. It is needed to interpret the model afterward, and a feature table that cannot say which regime a row came from cannot explain its own predictions.

## Appendix A. Terminology

The terms below appear in the body without being defined there. They are listed in alphabetical order.

- **AE** is Acoustic Emission, the elastic wave released by a material as it cracks or deforms, detected as a burst of short high-frequency hits.
- **Autoencoder** is a network trained to reconstruct its input, whose reconstruction error serves as a novelty score.
- **Bandwidth** is the width of the frequency range over which a signal or system carries meaningful energy.
- **Clipping** is the flattening of a waveform against a supply or range limit.
- **Coherence** is the normalized cross-spectrum between two signals, measuring how much of one is linearly explained by the other.
- **Compression** is the progressive loss of gain as drive increases.
- **Crest factor** is the ratio of peak amplitude to RMS.
- **DBSCAN** is Density-Based Spatial Clustering of Applications with Noise, a clustering method that finds dense regions and labels sparse points as noise.
- **DC offset** is the constant component of a signal, equal to its mean.
- **Detrending** is the removal of a slow component estimated by a fit or a moving average.
- **Duty cycle** is the fraction of a period during which a signal is in its active state.
- **EMD** is Empirical Mode Decomposition, an adaptive method that splits a signal into intrinsic mode functions.
- **Excitation** is the input deliberately applied to a system in order to observe its response.
- **FFT** is the Fast Fourier Transform, the algorithm that converts a sampled waveform into its frequency spectrum.
- **Fundamental** is the lowest and normally dominant frequency component of a periodic signal.
- **Harmonic** is a component at an integer multiple of the fundamental frequency.
- **HD2 and HD3** are the second and third harmonic levels taken individually, usually expressed in dBc.
- **Hysteresis** is the gap between the rising and falling thresholds of a detector, added to stop repeated triggering on noise.
- **IMD3** is third-order intermodulation distortion, the products generated when two tones mix in a non-linear stage.
- **IMF** is an Intrinsic Mode Function, one oscillatory component produced by EMD.
- **IP3** is the third-order intercept point, the extrapolated level at which IMD3 would equal the linear response.
- **IQR** is the Interquartile Range, the span between the first and third quartiles.
- **Isolation Forest** is an anomaly detector that scores points by how few random splits are needed to isolate them.
- **K-means** is a clustering method that partitions points into a preset number of groups by distance to group centers.
- **Kurtosis** is the fourth standardized moment, which rises in the presence of impulsive content.
- **Leakage** is the use of information in training that would not be available at prediction time.
- **LPF and HPF** are the low-pass filter and the high-pass filter, which retain components below and above a cutoff frequency.
- **Noise floor** is the background power level below which a signal cannot be distinguished.
- **P1dB** is the 1 dB compression point, the drive level at which output falls 1 dB below the extrapolated linear response.
- **P2P** is peak-to-peak, the distance from the maximum to the minimum of a waveform.
- **RMS** is the Root Mean Square, the energy-equivalent amplitude of a signal.
- **Saturation** is the condition in which output no longer increases with input.
- **Skewness** is the third standardized moment, measuring asymmetry of a distribution.
- **Slew rate** is the maximum rate of change of a signal, expressed in volts per microsecond.
- **SNR** is the Signal-to-Noise Ratio, the ratio of signal power to noise power in dB.
- **Spectral centroid** is the energy-weighted mean frequency of a spectrum.
- **Spectrogram** is a time versus frequency image of a signal.
- **STFT** is the Short-Time Fourier Transform, an FFT applied to successive short windows.
- **THD** is Total Harmonic Distortion, the ratio of total harmonic power to fundamental power.
- **THD+N** is Total Harmonic Distortion plus Noise, which counts noise power alongside the harmonics.
- **TOI** is the Third-Order Intercept, another name for IP3.
- **V_sat** is the saturation voltage, the output level beyond which the signal cannot rise.
- **Wavelet transform** is a multi-resolution decomposition that localizes components in both time and scale.
- **Zero-crossing rate** is the number of sign changes of a signal per unit time.
- **Z-score** is the number of standard deviations a value lies from the mean.
