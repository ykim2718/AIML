# Five stages of a data pipeline, from raw source files to a model-ready dataset
Rev. 12 | Created: 2026-06-23 | Updated: 2026-09-08 14:22 CDT

## 1. Overview

This report describes a five-stage data pipeline that takes data from raw source files to a model-ready dataset for Artificial Intelligence and Machine Learning (AI/ML) workloads, organized by data maturity. The five stages map one-to-one onto the Databricks' Medallion architecture (Bronze → Silver → Gold) [[2](#ref-2)], an industry-standard layering popularized by Databricks. Each stage has a single responsibility and consumes only the stage before it, which keeps the pipeline reproducible and auditable.

Each layer answers a different question, and the answer it gives is a guarantee about the data it holds.

```text
           BRONZE                         SILVER                          GOLD
  ┌───────────────────────┐      ┌───────────────────────┐      ┌───────────────────────┐
  │        keep it        │      │        make it        │      │        make it        │
  │      as it landed     │ ───▶ │      trustworthy      │ ───▶ │      model-ready      │
  └───────────────────────┘      └───────────────────────┘      └───────────────────────┘
      written once and              nulls, outliers and            features built and
      never edited, the             clocks resolved, so            reduced, and pinned
       only safety net                the data can be             to a version so that
     if a parse is wrong            queried with trust            train and serve agree
```

Fig 1. What each Medallion layer is responsible for

Bronze guarantees that the record is what arrived, Silver that the values can be trusted, and Gold that the columns are the ones a model consumes. What fills the three layers is the five stages of section 2, and section 3 shows which stage falls where.

## 2. Five-Stage Pipeline

### 2.1 Original Data (Bronze)

The untouched files exactly as they arrive from each source. Every source and version brings its own format — CSV (Comma-Separated Values), JSON (JavaScript Object Notation), or XML (Extensible Markup Language) — its own column names, and its own header conventions. These files are stored exactly as received and are never edited in place; they are the historical record and the only safety net if a parsing bug surfaces later.

### 2.2 Raw Data (Bronze)

The same data conformed to one schema. Originals are parsed into standardized column names, units, and timestamps. The shape is now consistent, but the content is still raw: nulls, outliers, and duplicates remain. Parsing is kept idempotent so that Raw can always be regenerated from Original.

### 2.3 Clean Data (Silver)

Trustworthy data. Missing values are handled, noise and outliers are removed, and timestamps are aligned across sources. This is the first layer that can be queried with confidence. One caution: a transient spike and a genuine distribution change — dataset shift [[3](#ref-3)] — can look statistically similar, so removal rules should be set with domain review to avoid discarding real signal.

### 2.4 Structured Data (Silver)

The same values reshaped to the model's input specification. The two-dimensional (2D) tabular form is a [samples, features] table for classical models such as XGBoost (eXtreme Gradient Boosting). The three-dimensional (3D) tensor form applies a time-series window for deep models — a Convolutional Neural Network (CNN) or Long Short-Term Memory (LSTM) — giving [samples, timesteps, features]. Group keys are carried through so the model can later be validated against unseen groups.

### 2.5 Feature Data (Gold)

The optimized dataset. Domain knowledge converts raw inputs into the variables a model learns from — moving averages, frequency components, embeddings — alongside dimensionality reduction. When features outnumber samples ($p \gg n$), feature reduction is essential rather than optional [[1](#ref-1)]. Feature definitions are versioned to prevent train/serve skew [[4](#ref-4)].

## 3. Medallion Architecture Mapping

The five stages line up one-to-one with the Medallion architecture, the de facto standard for sorting data by quality and maturity. Fig 2 adds the transform that carries each stage to the next, so that the boundary between two layers can be read as the transform that crosses it.

```text
           BRONZE                                 SILVER                              GOLD
     (raw preservation)                   (cleaned & structured)                  (model-ready)
  ┌───────────┬───────────┐              ┌───────────┬────────────┐               ┌───────────┐
  │  Original │    Raw    │ ── clean ──▶ │   Clean   │ Structured │ ─ features ─▶ │  Feature  │
  └───────────┴───────────┘              └───────────┴────────────┘               └───────────┘
        └── parse ──┘                          └─ reshape ──┘
```

Fig 2. The five stages, the transform between each pair, and the layers they fall into

Table 1. Medallion layers and the stages they hold

| Layer | Stages | State | Purpose |
| --- | --- | --- | --- |
| Bronze | Original + Raw | Landed as-is; format mismatch and unstructured content included | Preserve the historical record |
| Silver | Clean + Structured | Cleaned, conformed, and reshaped to a model-input form | Trusted, query-ready data |
| Gold | Feature | Fully engineered, highest maturity | Drop straight into a model |

Structured Data is a transitional layer. Model-agnostic reshaping (plain reshape, standard windowing) stays in Silver because many models can share it, while model-specific shaping leans toward Gold. When several models reuse the same structured output, it is best pinned to Silver.

## 4. Key Principles

The value of the pipeline is not the five labels but the discipline behind them:

- Immutability — each layer is written once and never edited in place.
- Reproducibility — each transform is deterministic, so the same input yields the same output.
- Lineage — every column can be traced back to the source record that produced it.

Together these properties let a bad prediction be traced all the way back to the exact source record that produced it.

## References

<a id="ref-1"></a>
[1] Bühlmann, P., & van de Geer, S. (2011). [*Statistics for High-Dimensional Data: Methods, Theory and Applications*](https://doi.org/10.1007/978-3-642-20192-9). Springer.<br>
<a id="ref-2"></a>
[2] Databricks. [What is Medallion Architecture?](https://www.databricks.com/blog/what-is-medallion-architecture). Databricks.<br>
<a id="ref-3"></a>
[3] Quiñonero-Candela, J., Sugiyama, M., Schwaighofer, A., & Lawrence, N. D. (Eds.) (2009). [*Dataset Shift in Machine Learning*](https://doi.org/10.7551/mitpress/9780262170055.001.0001). MIT Press. ISBN 978-0-262-17005-8.<br>
<a id="ref-4"></a>
[4] Sculley, D., Holt, G., Golovin, D., Davydov, E., Phillips, T., Ebner, D., Chaudhary, V., Young, M., Crespo, J.-F., & Dennison, D. (2015). [Hidden Technical Debt in Machine Learning Systems](https://papers.neurips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems). *Advances in Neural Information Processing Systems*, 28.

---

## Appendix A. Terminology

- **Bronze**: The Medallion layer that keeps data as it landed, with no cleaning or validation.
- **dataset shift**: A change in the joint distribution of inputs and outputs between the training stage and the serving stage.
- **Gold**: The Medallion layer that holds fully engineered, model-ready data.
- **idempotent**: A property of a transform whose repeated application gives the same result as a single application.
- **lineage**: The recorded chain of transforms that connects a column to its origin.
- **Medallion architecture**: A layering of a data lakehouse into Bronze, Silver, and Gold by data quality and maturity.
- **Silver**: The Medallion layer that holds cleaned, conformed, and reshaped data.
- **train/serve skew**: A mismatch between the feature values computed at training time and those computed at serving time.
