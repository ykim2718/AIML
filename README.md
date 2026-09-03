# AIML
Rev. 40 | Created: 2026-08-07 | Updated: 2026-09-03 12:38 CDT

> A working notebook of applied machine learning and statistics, kept as documents that fix the reasoning and as scripts that show the mechanics.
> The material leans toward measurement-heavy engineering data, semiconductor process and metrology data in particular.

Most of what is collected here answers a question that comes up before a model is trained rather than after. What form did the data arrive in, what does one row mean, which number should be reduced from a raw trace, and which metric actually says whether two measurements agree. The repository is organized around that order of work, so each folder holds one stage of it.

## 1. Scope

Two kinds of files live side by side.

Table 1. File kinds and their roles

| Kind | Role |
|------|------|
| `.md` | It fixes a decision or a taxonomy in prose, so that the same question is not re-argued later. |
| `.py` | It is a small standalone script that demonstrates one technique end to end, usually with a plot. |

The scripts are not a library. Each one runs on its own and is meant to be read as much as executed, so there is no package layout, no shared import root, and no build step.

## 2. Repository Map

Table 2. Folders and what they hold

| Folder | Description |
|--------|-------------|
| [EDA](EDA/) | It covers the stage before modeling — the manifest that records and profiles a stored table, the modality taxonomy that names what arrived, the reduction of machine waveforms into fixed-width parameter rows, the detection of outliers under `Outlier/`, one method to a subfolder, and the separation of measurement variability into its within-part and part-to-part components under `Variability/`. |
| [Feature-Engineering](Feature-Engineering/) | It maps feature engineering across cross-sectional, sequential, and wide time-series data, with a PCA lineage of its own, a wide-to-narrow survey under `Wide-Data/`, a survey of feature importance methods under `Feature-Importance/`, and implementations of Mahalanobis distance, incremental PCA, and smoothing. |
| [Metrics](Metrics/) | It collects agreement and goodness-of-fit measures — CCC with Bland-Altman, the Center Alignment Index, correlation coefficients, and the relationship between R² and MAPE. |
| [Models](Models/) | It holds the model itself and what surrounds it — regression recipes with an emphasis on step-like and piecewise responses under `Regression/`, continual learning for time series under `CLTS/`, temporal partial least squares under `Regression/TPLS/`, time-varying coefficients under `Regression/TVC/`, and the server that answers questions from a fitted model under `Inference-Server/`. |
| [Applied-Statistics](Applied-Statistics/) | It covers distribution fitting, hypothesis testing, numerical work, a cointegration example on daily price series, and the ceiling that bounds the classical z-score under `ZScore/`. |
| [AI Assistant](AI%20Assistant/) | It documents how Claude Code loads rules automatically from a plugin marketplace. This folder is synced from another repository and is not edited here. |
| [.claude](.claude/) | It carries the settings that make those rules load in this repository and the hook that installs them, with the note behind that setup kept in the repository the plugin comes from. |

## 3. Documents

### 3.1 EDA

Table 3. EDA documents

| Document | Description |
|----------|-------------|
| [EDA/README.md](EDA/README.md) | It indexes the folder and sets the order in which the documents are used. |
| [EDA/structured-data-manifest-for-semiconductor-machine-data.md](EDA/structured-data-manifest-for-semiconductor-machine-data.md) | It defines the JSON files that record what a stored table is, separating the values a human writes from the values an analysis decides, and fixes the class vocabulary and the integrity rules the manifest is checked against. |
| [EDA/data-modality-taxonomy.md](EDA/data-modality-taxonomy.md) | It classifies data by the form the information takes, then extends the axes to the semiconductor domain and works through wafer process data as a case study. |
| [EDA/semiconductor-machine-signal-parameterization-continuous.md](EDA/semiconductor-machine-signal-parameterization-continuous.md) | It is the continuous half of the parameterization pair, reducing a smoothly varying machine waveform to a parameter row across the small-signal regime, the large-signal regime, and the decomposition of a record that contains both. |
| [EDA/semiconductor-machine-signal-parameterization-quantized.md](EDA/semiconductor-machine-signal-parameterization-quantized.md) | It is the quantized half of the same pair, for a signal that rests on a ladder of discrete levels, built so that the row width never depends on the level count and so that the waveform can be rebuilt from the row. |
| [EDA/Outlier/README.md](EDA/Outlier/README.md) | It indexes the outlier folder, which keeps one detection method to a subfolder, and sets out the order in which a test is chosen, read, and acted on. |
| [EDA/Outlier/outlier_detection.md](EDA/Outlier/outlier_detection.md) | It surveys outlier detection across the statistical, machine learning and deep learning families, from the z-score and Tukey fences through Isolation Forest, ECOD and the local outlier factor to autoencoders and patch feature memories, records what a semiconductor line runs as part average testing and fault detection, and picks among them by what each one assumes. |
| [EDA/Outlier/GESD/generalized-esd.md](EDA/Outlier/GESD/generalized-esd.md) | It covers the generalized ESD procedure of ISO 16269-4, which finds an unknown number of outliers in one pass, and explains the decision rule that keeps several outliers from hiding one another. Two worked examples show a masked sample and a flag that does not survive inspection. |
| [EDA/Outlier/Hampel/hampel-identifier.md](EDA/Outlier/Hampel/hampel-identifier.md) | It covers the modified z-score built from the median and the median absolute deviation, which needs no normality and no iteration because the pair it is built on cannot be moved by a minority of outliers. |
| [EDA/Variability/WiW-W2W/wiw-w2w-anova.md](EDA/Variability/WiW-W2W/wiw-w2w-anova.md) | It splits the variance of a 261-wafer, 13-site measurement set into within-wafer, wafer-to-wafer, and lot-to-lot components by one-way and nested ANOVA, tracks the drift of the wafer means over run order, tests the cumulative standard deviation of those means against the sigma over root n law it is often assumed to follow, and selects the wafers whose within-wafer spread moves their own mean. |

### 3.2 Feature Engineering

Table 4. Feature engineering documents

| Document | Description |
|----------|-------------|
| [Feature-Engineering/fe-cs.md](Feature-Engineering/fe-cs.md) | It maps feature engineering for cross-sectional data, where rows are independent and row order carries no information. |
| [Feature-Engineering/fe-sq.md](Feature-Engineering/fe-sq.md) | It maps feature engineering for sequences, where time order is itself the information, covering univariate and multivariate series. |
| [Feature-Engineering/wts.md](Feature-Engineering/wts.md) | It consolidates techniques for wide time series where features far outnumber samples, grounded in FDC, DOE, and metrology practice. |
| [Feature-Engineering/Feature-Importance/feature-importance.md](Feature-Engineering/Feature-Importance/feature-importance.md) | It surveys the methods that put a number on a feature, arranged into seven families by where the number comes from — association in the data, structure of the fitted model, perturbation and removal, gradients, Shapley attribution, variance decomposition, and selection with a controlled error rate — sets out the seven axes on which any two of them are comparable, marks each method as standard or recent, and closes on the failure modes the families share. |
| [Feature-Engineering/Feature-Importance/feature-importance-ko.md](Feature-Engineering/Feature-Importance/feature-importance-ko.md) | It is the Korean edition of the feature importance survey, carrying the same axes, hierarchy, tables and references, with the headings and captions left in English. |
| [Feature-Engineering/PCA/README.md](Feature-Engineering/PCA/README.md) | It indexes the PCA folder, which treats PCA as a lineage of variants — the classical branches, their selection for semiconductor data, and the modern reinterpretation in representation learning. |
| [Feature-Engineering/Wide-Data/README.md](Feature-Engineering/Wide-Data/README.md) | It indexes the Wide-Data folder, which surveys how the (wafer, feature, trace) sensor tensor is compressed into a narrow per-wafer table while preserving information. |

### 3.3 Metrics

Table 5. Metrics documents

| Document | Description |
|----------|-------------|
| [Metrics/CCC/README.md](Metrics/CCC/README.md) | It is a full guide to Lin's Concordance Correlation Coefficient and the Bland-Altman plot, including interpretation benchmarks and a metric selection guide. |
| [Metrics/CAI/README.md](Metrics/CAI/README.md) | It defines the Center Alignment Index, which scores how closely the centers of two variables sit on the 1:1 line, normalized to be unit-free. |
| [Metrics/R2/r2-vs-mape.md](Metrics/R2/r2-vs-mape.md) | It shows by derivation and by experiment that no single universal formula links R² and MAPE, and that an apparent straight line is an artifact of how the data was generated. |

### 3.4 Automation

Table 6. Automation documents

| Document | Description |
|----------|-------------|
| [AI Assistant/Claude/automatic-rule-loading.md](AI%20Assistant/Claude/automatic-rule-loading.md) | It describes managing rules in one repository and loading them into every project through a plugin marketplace, so that skills, hooks, commands, and agents are shared between the desktop and the web interface. |
| [.claude/README.md](.claude/README.md) | It explains the two files that wire this repository to that marketplace, the catalog and the settings that enable the plugin. |

### 3.5 Models

Table 7. Model documents

| Document | Description |
|----------|-------------|
| [Models/CLTS/README.md](Models/CLTS/README.md) | It classifies continual learning for time series by when the model learns, how it learns, what it preserves, and which of several candidates is the one that serves, and it works through the delayed evaluation that a horizon forces on that last decision. |
| [Models/Regression/TPLS/README.md](Models/Regression/TPLS/README.md) | It organizes partial least squares for data that carries time, by the three places time enters a model — a lag structure inside one sample, a trajectory that is the sample itself, and the arrival order that decides when the model is refitted — and gives the method, the validation split, and the latent-space monitoring statistics that belong to each. |
| [Models/Regression/TVC/README.md](Models/Regression/TVC/README.md) | It covers regression whose coefficient is a function of time rather than a constant — why a coefficient moves, the parametric, spline, and random-walk forms it is given, the Kalman filter loop that estimates the random-walk form, its use against a violated proportional hazards assumption and in a time-varying VAR, and how the same estimation is laid over PLS scores. |
| [Models/Inference-Server/time-series-inference-server.md](Models/Inference-Server/time-series-inference-server.md) | It separates the model from the server around it, lists the questions a served series can be asked with the deliverable and the store each one needs, sets out the capabilities a stateless model server does not supply — context assembly, per-series state, and evaluation that arrives a horizon late — and describes the two-way interface through which a caller sets options and sends back the actuals, verdicts, and event marks a model is improved from. It then surveys the platforms that supply those capabilities and closes on fault detection and classification, where the case is put as what a production line can do once a server holds the context, the index, and the feedback. |

### 3.6 Applied Statistics

Table 8. Applied statistics documents

| Document | Description |
|----------|-------------|
| [Applied-Statistics/ZScore/z-score-ceiling.md](Applied-Statistics/ZScore/z-score-ceiling.md) | It derives the ceiling that bounds every classical z-score at (n-1)/sqrt(n), shows that the bound is attained rather than approached, and works out the sample sizes at which a rule set at a fixed cut-off can no longer flag anything. |

## 4. Order Of Use

1. Record the data in the manifest first, so that its origin, its grain, and the class of each column are fixed.
2. Map the confirmed form onto the modality taxonomy to fix what kind of data it is.
3. If the data is a waveform or a trace, parameterize it into fixed-width feature rows before anything else.
4. Apply the feature engineering that suits the fixed modality, cross-sectional or sequential or wide.
5. Fit a model, choosing a piecewise form when the response is step-like.
6. Score it with a metric that matches the question, since agreement, correlation, and error are three different questions.
7. Put the fitted model behind a server, fixing first which question is being asked of the series, what the answer to it has to contain, and what the server has to hold that the model does not.

## 5. Conventions

- The repository uses the `main` branch only. Work is committed to `main` directly, without feature branches and without pull requests.
- Markdown conventions — the version header, heading numbers, table and figure captions, terminology — are defined by the `md_rules` skill, which arrives with the plugin declared in [.claude/settings.json](.claude/settings.json), and are deliberately not restated here.
- Prose paragraphs are written in full sentences, while list items and table cells may be shortened to noun phrases, one point to an item.
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
