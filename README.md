# AIML
rev. 2

> A working notebook of applied machine learning and statistics, kept as documents that fix the reasoning and as scripts that show the mechanics.
> The material leans toward measurement-heavy engineering data, semiconductor process and metrology data in particular.

Most of what is collected here answers a question that comes up before a model is trained rather than after. What form did the data arrive in, what does one row mean, which number should be reduced from a raw trace, and which metric actually says whether two measurements agree. The repository is organized around that order of work, so each folder holds one stage of it.

## 1. Scope

Two kinds of files live side by side.

| Kind | Role |
|------|------|
| `.md` | It fixes a decision or a taxonomy in prose, so that the same question is not re-argued later. |
| `.py` | It is a small standalone script that demonstrates one technique end to end, usually with a plot. |

The scripts are not a library. Each one runs on its own and is meant to be read as much as executed, so there is no package layout, no shared import root, and no build step.

## 2. Repository Map

| Folder | Description |
|--------|-------------|
| [EDA](EDA/) | It covers the stage before modeling — automatic data profiling, the modality taxonomy that names what arrived, and the reduction of machine waveforms into fixed-width parameter rows. |
| [Feature-Engineering](Feature-Engineering/) | It maps feature engineering across cross-sectional, sequential, and wide time-series data, with implementations of Mahalanobis distance, incremental PCA, and smoothing. |
| [Metrics](Metrics/) | It collects agreement and goodness-of-fit measures — CCC with Bland-Altman, the Center Alignment Index, correlation coefficients, and the relationship between R² and MAPE. |
| [Models](Models/) | It holds regression recipes, with an emphasis on step-like and piecewise responses that a single global fit handles badly. |
| [Applied-Statistics](Applied-Statistics/) | It covers distribution fitting, hypothesis testing, numerical work, and a cointegration example on daily price series. |
| [AI Assistant](AI%20Assistant/) | It documents how Claude Code loads rules automatically from a plugin marketplace. This folder is synced from another repository and is not edited here. |
| [.claude](.claude/) | It carries the settings that make those rules load in this repository, together with the setup and verification notes behind them. |

## 3. Documents

### 3.1 EDA

| Document | Description |
|----------|-------------|
| [EDA/README.md](EDA/README.md) | It indexes the folder and sets the order in which the documents are used. |
| [EDA/data-manifest.md](EDA/data-manifest.md) | It defines the JSON files that record what a stored dataset is, separating the values a human declares from the values an analysis decides, and fixes the class vocabulary and the integrity rules the manifest is checked against. |
| [EDA/data-profile.md](EDA/data-profile.md) | It sets out the procedure that determines the shape of new data, the location of X and y, dtype, missingness, and class imbalance, driven by `data_profiler.py`. |
| [EDA/data-modality-taxonomy.md](EDA/data-modality-taxonomy.md) | It classifies data by the form the information takes, then extends the axes to the semiconductor domain and works through wafer process data as a case study. |
| [EDA/machine-signal-parameterization.md](EDA/machine-signal-parameterization.md) | It reduces a continuous machine waveform to a parameter row, split into the small-signal regime, the large-signal regime, and the decomposition of a record that contains both. |
| [EDA/quantized-signal-parameterization.md](EDA/quantized-signal-parameterization.md) | It does the same for a signal that rests on a ladder of discrete levels, built so that the row width never depends on the level count and so that the waveform can be rebuilt from the row. |

### 3.2 Feature Engineering

| Document | Description |
|----------|-------------|
| [Feature-Engineering/fe-cs.md](Feature-Engineering/fe-cs.md) | It maps feature engineering for cross-sectional data, where rows are independent and row order carries no information. |
| [Feature-Engineering/fe-sq.md](Feature-Engineering/fe-sq.md) | It maps feature engineering for sequences, where time order is itself the information, covering univariate and multivariate series. |
| [Feature-Engineering/wts.md](Feature-Engineering/wts.md) | It consolidates techniques for wide time series where features far outnumber samples, grounded in FDC, DOE, and metrology practice. |

### 3.3 Metrics

| Document | Description |
|----------|-------------|
| [Metrics/CCC/README.md](Metrics/CCC/README.md) | It is a full guide to Lin's Concordance Correlation Coefficient and the Bland-Altman plot, including interpretation benchmarks and a metric selection guide. |
| [Metrics/CAI/README.md](Metrics/CAI/README.md) | It defines the Center Alignment Index, which scores how closely the centers of two variables sit on the 1:1 line, normalized to be unit-free. |
| [Metrics/R2/r2-vs-mape.md](Metrics/R2/r2-vs-mape.md) | It shows by derivation and by experiment that no single universal formula links R² and MAPE, and that an apparent straight line is an artifact of how the data was generated. |

### 3.4 Automation

| Document | Description |
|----------|-------------|
| [AI Assistant/Claude/automatic-rule-loading.md](AI%20Assistant/Claude/automatic-rule-loading.md) | It describes managing rules in one repository and loading them into every project through a plugin marketplace, so that skills, hooks, commands, and agents are shared between the desktop and the web interface. |
| [.claude/README.md](.claude/README.md) | It explains the two files that wire this repository to that marketplace, the catalog and the settings that enable the plugin. |
| [.claude/plugin-setup.md](.claude/plugin-setup.md) | It records the setup that lets a fresh container fetch the plugin from the remote on its first session, including the authentication failure that broke the earlier attempt and the fix for it. |

## 4. Order Of Use

1. Profile the data first to confirm its size, its target, and the properties of each column.
2. Map the confirmed form onto the modality taxonomy to fix what kind of data it is.
3. If the data is a waveform or a trace, parameterize it into fixed-width feature rows before anything else.
4. Apply the feature engineering that suits the fixed modality, cross-sectional or sequential or wide.
5. Fit a model, choosing a piecewise form when the response is step-like.
6. Score it with a metric that matches the question, since agreement, correlation, and error are three different questions.

## 5. Conventions

- The repository uses the `main` branch only. Work is committed to `main` directly, without feature branches and without pull requests.
- Every `.md` document carries a `rev. N` line directly under its H1, written in body text. The number is incremented on every edit, including a typo fix.
- Documents are written in prose rather than in bullet fragments, and tables describe an item with a full sentence.
- Some of the older documents and script comments are written in Korean. Newer documents are written in English.
- These rules are recorded in [CLAUDE.md](CLAUDE.md) so that an assistant working in the repository follows them without being told, and the shared rules that apply across projects arrive automatically through the plugin declared in [.claude/settings.json](.claude/settings.json).

## 6. Running The Code

The scripts target Python 3 and rely on the usual scientific stack.

```bash
pip install numpy scipy pandas matplotlib scikit-learn statsmodels pwlf
```

A few scripts read data from a path relative to their own folder, so run them from the directory they live in.

```bash
cd Applied-Statistics/Time-Series
python "statsmodels - coint - pair trading.py"
```

There is no repository-wide license file. The CCC guide declares the MIT License for itself.
