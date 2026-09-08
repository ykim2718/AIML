# Medallion architecture in practice: six stages from raw source files to a model-ready dataset
Rev. 20 | Created: 2026-06-23 | Updated: 2026-09-08 18:40 CDT

## 1. Overview

This report describes a pipeline that carries data from raw source files to a model-ready dataset for Artificial Intelligence and Machine Learning (AI/ML) workloads, organized by data maturity. The layering is the Databricks' Medallion architecture (Bronze → Silver → Gold), an industry standard, and the six stages of this pipeline fill its three layers. Each stage has a single responsibility and reads only the stage or stages before it, which keeps the pipeline reproducible and auditable.

Databricks defines the architecture as a data design pattern for organizing data in a lakehouse, whose aim is to improve the structure and quality of the data incrementally as it passes through the three layers [[1](#ref-1)]. The layers are named for what the data has become, not for where it is kept.

- Bronze holds the records from the source systems — an RDBMS (Relational Database Management System), IoT (Internet of Things) devices, logs, or an API (Application Programming Interface) — almost exactly as they arrive.
- Silver holds those same records cleansed, enriched, and conformed to one specification, an intermediate state between the source and the consumer.
- Gold holds the aggregated and modeled result, shaped for the consumer that reads it: a star schema for BI (Business Intelligence) reporting, a feature table for model training.

Each layer makes one guarantee about the data it holds, and that guarantee is what the layer is for.

```text
           BRONZE                         SILVER                          GOLD
  ┌───────────────────────┐      ┌───────────────────────┐      ┌───────────────────────┐
  │        keep it        │      │        make it        │      │        make it        │
  │      as it landed     │ ───> │      trustworthy      │ ───> │      model-ready      │
  └───────────────────────┘      └───────────────────────┘      └───────────────────────┘
      written once and              nulls, outliers and            features built and
      never edited, the             clocks resolved, so            reduced, and pinned
       only safety net                the data can be             to a version so that
     if a parse is wrong            queried with trust            train and serve agree
                                    and fed to a model
```

Fig 1. What each Medallion layer guarantees and what its data can be used for

Bronze guarantees that the record is what arrived, Silver that the values can be trusted, and Gold that the columns are the ones a model consumes. Section 2 places the six stages of the pipeline inside these layers, and section 3 takes each stage in turn.

## 2. Medallion Architecture Mapping

The six stages fall into the three layers of the Medallion architecture, the de facto standard for sorting data by quality and maturity. Fig 2 names the transform at each layer boundary, so that a boundary can be read as the work that crosses it, and shows Clean branching into two Silver forms that rejoin at Feature.

```text
        BRONZE                               SILVER                       GOLD
                                                  ┌─────────────┐
                                              ┌─▶ │  Structured │─┐
  ┌───────────┬───────────┐    ┌───────────┐  │   └─────────────┘ │   ┌───────────┐
  │  Original │    Raw    │──▶ │   Clean   │ ─┤                   ├──▶│  Feature  │
  └───────────┴───────────┘    └───────────┘  │   ┌─────────────┐ │   └───────────┘
        └── parse ──┘            clean        └─▶ │ Transformed │─┘     features
                                                  └─────────────┘
```

Fig 2. The six stages, the transform between them, and the layers they fall into

Table 1. Medallion layers and the stages they hold

| Layer | Stages | State | Purpose |
| --- | --- | --- | --- |
| Bronze | Original + Raw | Landed as-is; format mismatch and unstructured content included | Preserve the historical record |
| Silver | Clean + Structured + Transformed | Cleaned and conformed, then reshaped to a model-input form and re-expressed on the scale a model reads | Trusted, query-ready data, and model-ready when no new feature is needed |
| Gold | Feature | Fully engineered, highest maturity | Drop straight into a model |

Structured Data and Transformed Data are transitional. Model-agnostic work — plain reshape, standard windowing, standard scaling — stays in Silver because many models can share it, while model-specific shaping or encoding leans toward Gold. When several models reuse the same output, it is best pinned to Silver. A model that needs no engineered feature can be trained on the Silver output directly, because Structured Data already carries the input shape it reads and Transformed Data the scale.

## 3. Pipeline Stages

### 3.1 Original Data (Bronze)

The untouched files exactly as they arrive from each source. Every source and version brings its own format — CSV (Comma-Separated Values), JSON (JavaScript Object Notation), or XML (Extensible Markup Language) — its own column names, and its own header conventions. These files are stored exactly as received and are never edited in place; they are the historical record and the only safety net if a parsing bug surfaces later.

### 3.2 Raw Data (Bronze)

The same data conformed to one schema. Originals are parsed into standardized column names, units, and timestamps. The shape is now consistent, but the content is still raw: nulls, outliers, and duplicates remain. Parsing is kept idempotent so that Raw can always be regenerated from Original.

### 3.3 Clean Data (Silver)

Trustworthy data. Missing values are handled, noise and outliers are removed, and timestamps are aligned across sources. This is the first stage that can be queried with confidence. One caution: a transient spike and a genuine distribution change — dataset shift [[2](#ref-2)] — can look statistically similar, so removal rules should be set with domain review to avoid discarding real signal.

### 3.4 Structured Data (Silver)

The same values reshaped to the model's input specification. The two-dimensional (2D) form is a [samples, features] table for classical models such as XGBoost (eXtreme Gradient Boosting). The three-dimensional (3D) tensor form applies a time-series window for deep models — a Convolutional Neural Network (CNN) or Long Short-Term Memory (LSTM) — giving [samples, timesteps, features]. Group keys are carried through so the model can later be validated against unseen groups.

### 3.5 Transformed Data (Silver)

The same values re-expressed on the scale a model reads. Numeric columns are scaled, categorical columns are encoded, and a skewed column is put through a monotone transform. The arrangement of the table is untouched, which is what separates this stage from Structured Data: one changes how the values are laid out, the other changes the values themselves. Neither stage reads the other and both read Clean Data, so they can be built in either order. The parameters they fit — a scaler's mean and variance, an encoder's category list — are taken from training rows only and stored with the dataset, because refitting them at serving time is a known route to train/serve skew [[3](#ref-3)].

### 3.6 Feature Data (Gold)

The optimized dataset. Domain knowledge converts the columns it reads into the variables a model learns from — moving averages, frequency components, embeddings — alongside dimensionality reduction. When features outnumber samples ($p \gg n$), feature reduction is essential rather than optional [[4](#ref-4)]. Feature definitions are versioned to prevent train/serve skew [[3](#ref-3)].

## 4. Key Principles

The value of the pipeline is not the six labels but the discipline behind them:

- Immutability — each layer is written once and never edited in place.
- Reproducibility — each transform is deterministic, so the same input yields the same output.
- Lineage — every column can be traced back to the source record that produced it.

Together they make a bad prediction diagnosable: the column that carried it can be re-derived, and the record behind that column can be opened.

## References

<a id="ref-1"></a>
[1] Databricks. [What is Medallion Architecture?](https://www.databricks.com/blog/what-is-medallion-architecture). Databricks.<br>
<a id="ref-2"></a>
[2] Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (Eds.) (2009). [*Dataset Shift in Machine Learning*](https://doi.org/10.7551/mitpress/9780262170055.001.0001). MIT Press. ISBN 978-0-262-17005-8.<br>
<a id="ref-3"></a>
[3] Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). [Hidden Technical Debt in Machine Learning Systems](https://papers.neurips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems). *Advances in Neural Information Processing Systems*, 28.<br>
<a id="ref-4"></a>
[4] Bühlmann, P., & van de Geer, S. (2011). [*Statistics for High-Dimensional Data: Methods, Theory and Applications*](https://doi.org/10.1007/978-3-642-20192-9). Springer.

---

## Appendix A. Terminology

- **Bronze**: The Medallion layer that keeps data as it landed, with no cleaning or validation.
- **dataset shift**: A change in the joint distribution of inputs and outputs between the training stage and the serving stage.
- **Gold**: The Medallion layer that holds fully engineered, model-ready data.
- **idempotent**: A property of a transform whose repeated application gives the same result as a single application.
- **lineage**: The recorded chain of transforms that connects a column to its origin.
- **Medallion architecture**: A layering of a data lakehouse into Bronze, Silver, and Gold by data quality and maturity.
- **Silver**: The Medallion layer that holds cleaned and conformed data, reshaped and re-expressed for a model.
- **star schema**: A table layout that puts the measures in one fact table and their descriptive attributes in the dimension tables around it.
- **train/serve skew**: A mismatch between the feature values computed at training time and those computed at serving time.
