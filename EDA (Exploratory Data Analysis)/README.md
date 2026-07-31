# EDA (Exploratory Data Analysis)
rev. 10

> This folder covers the stage that establishes the structure and the properties of the data before a model is built.
> Its documents answer what to check first when new data arrives and what to call the thing that arrives.

## 1. Scope

Exploratory data analysis starts from two questions. <br>The first is to determine what form the data takes, and the second is to decide which name that form is classified under. The former is answered by automatic profiling, and the latter by the modality taxonomy.

## 2. Documents

| Document | Description |
|----------|-------------|
| [data-profile.md](data-profile.md) | It sets out the procedure that automatically determines the shape of new data, the location of X and y, dtype, missingness, and class imbalance. |
| [data-modality-taxonomy.md](data-modality-taxonomy.md) | It covers the general classification of data modality, its extension to the semiconductor domain, and the naming case study for wafer process data. |
| [machine-signal-parameters.md](machine-signal-parameters.md) | It covers the reduction of a machine waveform into parameters for ML input, split into the small-signal regime, the large-signal regime, and the decomposition of the two. |

## 3. Order Of Use

1. When data arrives, profile it first to confirm its size, its target, and the properties of each column.
2. Map the confirmed form onto the taxonomy to fix the modality.
3. Choose the preprocessing and the model family that suit the fixed modality.
4. If the data is a waveform or a trace, follow the parameterization document to reduce it to fixed-width feature rows.

Modality is a tag set rather than an exclusive classification, so several tags attaching to one dataset at the same time is treated as normal.
