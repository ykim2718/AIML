# EDA (Exploratory Data Analysis)
rev. 16

> This folder covers the stage that establishes the structure and the properties of the data before a model is built.
> Its documents answer what to check first when new data arrives and what to call the thing that arrives.

## 1. Scope

Exploratory data analysis starts from two questions. <br>The first is to determine what form the data takes, and the second is to decide which name that form is classified under. The former is answered by the manifest that profiles every column, and the latter by the modality taxonomy.

## 2. Documents

| Document | Description |
|----------|-------------|
| [tabular-data-manifest-for-semiconductor-machine-data.md](tabular-data-manifest-for-semiconductor-machine-data.md) | It defines the JSON files that record what a stored table is, splitting the values a human writes from the values an analysis decides, and fixes the class vocabulary each column is labelled with. |
| [data-modality-taxonomy.md](data-modality-taxonomy.md) | It covers the general classification of data modality, its extension to the semiconductor domain, and the naming case study for wafer process data. |
| [machine-signal-parameterization.md](machine-signal-parameterization.md) | It covers the reduction of a machine waveform into parameters for ML input, split into the small-signal regime, the large-signal regime, and the decomposition of the two. |
| [quantized-signal-parameterization.md](quantized-signal-parameterization.md) | It covers the same reduction for a signal that rests on a ladder of discrete levels, built so that the row width does not depend on the level count and so that the waveform can be rebuilt from the row. |

## 3. Order Of Use

1. When data arrives, record its origin and its grain in the catalog, then write down in the config how its columns are renamed and cleaned.
2. Profile the columns so that the class of each one is fixed before anything else is read from the data.
3. Map the confirmed form onto the taxonomy to fix the modality.
4. Choose the preprocessing and the model family that suit the fixed modality.
5. If the data is a waveform or a trace, follow the parameterization document to reduce it to fixed-width feature rows. Where that waveform rests on a ladder of discrete levels rather than varying continuously, take the quantized-signal document instead.

Modality is a tag set rather than an exclusive classification, so several tags attaching to one dataset at the same time is treated as normal.
