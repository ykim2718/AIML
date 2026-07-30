# Data Modality Taxonomy and Semiconductor Data
rev. 5

> Data modality is the classification that splits data by "in what form the information is represented".
> This document first establishes a general taxonomy, then extends the axes to fit the semiconductor domain.

The word modality means the channel or form through which information is carried. In ordinary use it names a mode of sensing such as sight or hearing. In machine learning it names the kind of raw representation a dataset arrives in, such as text, image, waveform, or table, together with the tensor shape and the inductive bias that kind implies. A model is called unimodal when it consumes one such kind and multimodal when it consumes several. Modality therefore answers "what form is this data", which is a different question from "what does this data mean" or "where did it come from".

## 1. General Data Taxonomy

At the top level the data groups along two axes, structure and signal, and these two axes are orthogonal to each other.

### 1.1 By Structure

| Type | Description | Example |
|------|-------------|---------|
| Structured | It fits a row and column schema exactly | relational table, transaction log |
| Semi-structured | The schema is loose or self-describing | JSON, XML, YAML, log |
| Unstructured | It is a raw signal with no fixed frame | text, image, audio, video |

### 1.2 By Signal

This is the axis that machine learning usually calls modality.

| Modality | Representation |
|----------|----------------|
| Text | It handles natural language, code, and documents as a token sequence |
| Vision | It handles image and video as a pixel grid or a frame sequence |
| Audio | It handles speech, music, and ambient sound as a waveform or a spectrogram |
| Tabular | It handles numeric and categorical features with rows as samples and columns as features |
| Time-series | It handles sensor, financial, and IoT signals as values aligned on a time axis |
| Graph | It handles social, molecular, and knowledge graph data as a node and edge structure |
| Geospatial | It handles latitude and longitude, maps, and satellite imagery as coordinates or a raster |
| 3D / Geometric | It handles point cloud, mesh, and depth as spatial coordinates |

### 1.3 Composite

- **Multimodal** combines two or more kinds. VQA, which uses image and text together, is the representative case.
- **Cross-modal** translates one modality into another. Text to image and speech to text belong here.

### 1.4 Tree View

```text
Data Modality
├── By Structure
│   ├── Structured       (tables)
│   ├── Semi-structured  (JSON / XML / logs)
│   └── Unstructured     (raw signals)
│
├── By Signal
│   ├── Text
│   ├── Vision           (image / video)
│   ├── Audio
│   ├── Tabular
│   ├── Time-series
│   ├── Graph
│   ├── Geospatial
│   └── 3D / Geometric
│
└── Composite
    ├── Multimodal       (combine)
    └── Cross-modal      (translate)
```

The same data groups differently depending on which axis you view it from. An image is unstructured by the structure axis and vision by the signal axis. Therefore, when using this taxonomy you must first pin down which axis you are splitting on.

### 1.5 Scope And Limits

The structure axis applies to any data, so it covers essentially everything. The signal axis, in contrast, is an open list, so new entries must be added whenever a new domain arrives.

The following cases are distorted when they are forced into the existing eight slots.

- A reinforcement learning trajectory is a form in which state, action, and reward are interwoven.
- DNA and protein are biological sequences and are token sequences, but their vocabulary and constraints differ from natural language.
- SMILES is a chemical notation that is a string but is in fact a graph.
- Embedding and latent vector are derived representations rather than raw signals.
- An event stream has irregular timestamps, so treating it as a uniform time-series causes loss.

Real data also often occupies several slots at once. Therefore this taxonomy should be used as a tag set rather than an exclusive classification table.

## 2. Semiconductor Taxonomy

Semiconductor data is not covered by the general modality list alone. Which unit a measurement attaches to becomes the primary distinction, so one more axis for the entity hierarchy is added.

### 2.1 Axis A — Entity Hierarchy

```text
Fab -> Tool (Chamber) -> Lot -> Wafer -> Die/Site -> Device/Structure
```

| Level | Typical data |
|-------|--------------|
| Fab | line-level capacity and total WIP |
| Tool / Chamber | FDC trace, alarm, PM history |
| Lot | route, priority, queue time |
| Wafer | wafer map, metrology summary values |
| Die / Site | bin code, CD and overlay measurement points |
| Device | I–V curve, TEM cross-section |

### 2.2 Axis B — Modality

| Modality | Actual data | Tensor shape |
|----------|-------------|--------------|
| Tabular / Contextual | MES history, WIP, genealogy, WAT summary | `[row, feature]` |
| Equipment trace | temperature, pressure, RF, and flow received from FDC sensors | `[time, sensor]` |
| Spatial wafer map | sort bin map, defect map, thickness map | `[x, y]` or `[site, value]` |
| Vector field | overlay and distortion vector | `[x, y, (dx, dy)]` |
| Image | ADC defect image, SEM, optical, TEM | `[H, W, C]` |
| Layout / Geometric | GDSII polygon, mask, OPC pattern | vector polygon |
| Curve / Functional | I–V, C–V, SIMS depth profile, AFM profile | `[index, value]` |
| Event stream | equipment alarm, MES event, interlock | `(t, event)` with irregular timestamps |
| Recipe / Config | process recipe, step tree | semi-structured tree |
| Graph / Relational | route flow, tool and lot traversal graph, netlist | node and edge |
| Text | FA report, PM log, spec document | token sequence |

### 2.3 Axis C — Lifecycle Stage

This third axis is added when partitioning the problem definition.

```text
Design -> Process -> Metrology -> Inspection -> Test -> Reliability
```

| Stage | Representative data |
|-------|---------------------|
| Design | layout, netlist, OPC pattern |
| Process | recipe, FDC trace, alarm |
| Metrology | CD, thickness, overlay measurements |
| Inspection | defect map, ADC image |
| Test | WAT and PCM, sort bin map |
| Reliability | burn-in, HTOL, FA report |

### 2.4 Composite Patterns

Real analysis holds not on a single slot but on a combination.

| Pattern | Composition | Use case |
|---------|-------------|----------|
| Multi-scale time-series | intra-wafer trace and inter-wafer drift | tool drift, chamber matching |
| Spatio-temporal | wafer map and time order | defect pattern propagation tracking |
| Trace to map | FDC trace as cause, bin map as effect | virtual metrology, yield regression |
| Image with tabular | defect image with coordinates and layer context | ADC classification |
| Layout with image | GDSII and SEM | hotspot detection |

### 2.5 Tree View

```text
Semiconductor Data
├── Axis A — Entity level : Fab / Tool / Lot / Wafer / Die / Device
├── Axis B — Modality
│   ├── Tabular          (MES, WAT)
│   ├── Trace            (FDC time-series)
│   ├── Spatial map      (bin, defect, thickness)
│   ├── Vector field     (overlay)
│   ├── Image            (ADC, SEM, TEM)
│   ├── Layout           (GDSII, mask)
│   ├── Curve            (I-V, depth profile)
│   ├── Event stream     (alarm, MES log)
│   ├── Recipe tree
│   ├── Graph            (route, netlist)
│   └── Text             (FA report)
├── Axis C — Lifecycle stage
│   ├── Design           (layout, netlist, OPC)
│   ├── Process          (recipe, FDC trace, alarm)
│   ├── Metrology        (CD, thickness, overlay)
│   ├── Inspection       (defect map, ADC image)
│   ├── Test             (WAT/PCM, sort bin map)
│   └── Reliability      (burn-in, HTOL, FA report)
└── Composite
    ├── multi-scale time-series
    ├── spatio-temporal
    └── trace -> map (cause -> effect)
```

The key is to record together which entity level the data attaches to and what form it takes. Both values are written side by side, as in `wafer × spatial map` or `chamber × trace`.

## 3. Case Study — Wafer Process Data

Wafers have a processing order, and each wafer carries a process time-series. This section examines whether calling such data a 3D time-series is correct.

### 3.1 Verdict

The name 3D time-series is not used. The tensor rank is indeed 3, but the label invites misunderstanding.

### 3.2 Why Misleading

By convention 3D denotes x, y, z spatial geometry. Point cloud and CT volume are the examples. The third axis of wafer process data, however, is not space but sensor channel or wafer index. When it is called 3D on shape alone, the listener takes it as spatial volume data.

### 3.3 Two Time Axes

The key point of this data is that two time axes of different character overlap.

| Axis | Meaning | Character |
|------|---------|-----------|
| Inter-wafer | wafer and lot processing order | It is slow time and carries tool aging and drift |
| Intra-wafer | trace within one process step | It is fast time and carries the FDC sensor waveform |
| Channel | sensor kinds such as temperature, pressure, RF, gas flow | It is a variable axis, not time |

Flattened into a tensor it is rank-3 as `[wafer, time, sensor]`, but in meaning it is a bundle of multivariate time-series with wafer-order drift laid on top.

### 3.4 Correct Naming

| Scope | Recommended name |
|-------|------------------|
| within one wafer | multivariate time-series or multichannel time-series |
| accumulated across wafers | panel time-series, or tensor time-series when emphasizing the tensor |
| both time axes emphasized | hierarchical time-series or multi-scale time-series |
| die position included | spatio-temporal, at which point a true spatial axis finally appears |

```text
Wafer process data
├── intra-wafer : multivariate time-series   (time x sensor)
├── inter-wafer : panel / sequence           (drift across wafers)
└── (optional) die map : spatial x-y         -> spatio-temporal
```

### 3.5 Conclusion

For data without die spatial coordinates, the panel form of a multi-scale multivariate time-series is the most accurate name. The word 3D is reserved for when a spatial axis actually enters, as in a wafer map.

## Appendix A. Terminology

The terms below appear in the body without being defined there. They are listed in alphabetical order.

- **ADC** is Automatic Defect Classification, the automated sorting of inspection defect images into defect types.
- **AFM** is the Atomic Force Microscope, a probe instrument that measures surface topography at nanometer scale.
- **Burn-in** is a reliability screen that operates devices under elevated voltage and temperature to precipitate early failures.
- **CD** is the Critical Dimension, the measured width of a patterned feature.
- **Chamber** is the individual process module inside a tool where a wafer is actually processed.
- **Chamber matching** is the task of confirming that chambers intended to be identical behave identically.
- **CT volume** is a Computed Tomography reconstruction, a stack of slices forming a true x, y, z volume.
- **C–V** is a capacitance versus voltage measurement used to characterize dielectric and interface properties.
- **Die** is one individual chip on a wafer.
- **Drift** is the slow shift of a tool or process response over time.
- **Embedding** is a learned dense vector representation of an input.
- **FA** is Failure Analysis, the investigation that determines the physical root cause of a failed device.
- **Fab** is the fabrication plant, meaning the whole manufacturing line.
- **FDC** is Fault Detection and Classification, the system that records equipment sensor traces and flags abnormal behavior.
- **GDSII** is the standard binary format that stores mask layout as polygons.
- **Genealogy** is the record of which tools, chambers, and routes a lot or wafer passed through.
- **Hotspot** is a layout location predicted or observed to print incorrectly.
- **HTOL** is High Temperature Operating Life, an accelerated reliability stress test.
- **Interlock** is an equipment safety condition that blocks operation until it is cleared.
- **IoT** is the Internet of Things, meaning networked devices that emit continuous sensor streams.
- **I–V** is a current versus voltage measurement used to characterize device electrical behavior.
- **Latent vector** is the internal representation a model produces in a compressed space.
- **Lot** is the group of wafers, typically up to 25, that moves through the line together.
- **Mask** is the patterned plate through which a layer is exposed onto the wafer.
- **MES** is the Manufacturing Execution System, the system of record for lot movement and processing history.
- **Metrology** is the measurement of dimensions and film properties on product wafers.
- **Netlist** is the connectivity description of a circuit as devices and nets.
- **OPC** is Optical Proximity Correction, the deliberate distortion of mask patterns to compensate for optical effects.
- **Overlay** is the misalignment between a newly patterned layer and the layer beneath it.
- **Panel** is, in statistics, data observed on many units repeatedly over time.
- **PCM** is the Process Control Monitor, meaning test structures measured to monitor process health.
- **PM** is Preventive Maintenance, the scheduled servicing of a tool.
- **Queue time** is the waiting time of a lot between two process steps.
- **Raster** is a regular grid of cells holding values, as opposed to vector geometry.
- **Recipe** is the ordered set of process steps and their parameters that a tool executes.
- **RF** is Radio Frequency, the power applied to sustain a plasma.
- **Route** is the ordered sequence of process steps a product follows.
- **SEM** is the Scanning Electron Microscope, an imaging instrument used for defect review and CD measurement.
- **SIMS** is Secondary Ion Mass Spectrometry, a technique that measures elemental concentration versus depth.
- **Site** is a designated measurement location on a wafer.
- **SMILES** is the Simplified Molecular Input Line Entry System, a text notation for molecular structure.
- **Sort bin map** is the wafer map recording the electrical test bin assigned to each die.
- **Spectrogram** is a time versus frequency image of a signal.
- **TEM** is the Transmission Electron Microscope, an instrument that images thin cross-sections at atomic scale.
- **Tensor rank** is the number of axes of a tensor.
- **Token sequence** is an ordered list of discrete symbols produced by splitting an input.
- **Tool** is one piece of process equipment in the fab.
- **Trace** is the time-ordered sensor signal recorded during a process step.
- **Virtual metrology** is the prediction of a measurement value from equipment data instead of measuring it.
- **VQA** is Visual Question Answering, a task that answers a text question about an image.
- **Wafer** is the substrate disc on which dies are fabricated.
- **WAT** is the Wafer Acceptance Test, the electrical test of process monitor structures after fabrication.
- **WIP** is Work In Process, meaning the material currently in the line.
- **Yield** is the fraction of dies that pass test.
