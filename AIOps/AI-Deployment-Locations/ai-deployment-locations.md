# AI Deployment Locations For Semiconductor Equipment
Rev. 0 | Created: 2026-10-17 | Updated: 2026-10-17 08:28 CDT

## 1. Purpose

- **Problem Statement**: The names used for deploying AI on semiconductor equipment mix where the model runs with who supplies the software, so one deployment goes by two names.
- **Goal**: A reader can classify any AI on semiconductor equipment by one value of the location axis.
- **Non-Goal**: Vendor products are not compared with one another.

## 2. Summary

An AI deployment for semiconductor equipment is classified by where inference runs: edge, on-premise or cloud, and a deployment that uses two or more of them is hybrid.
Whether the software is built in-house or supplied by a vendor is a separate axis and leaves the location unchanged.

## 3. Taxonomy and its Hierarchy

Two independent axes describe a deployment, and only the location axis separates one deployment option from another.

The two axes and their values are drawn in Fig 1.

```text
AI deployment for semiconductor equipment
├── Location : Edge | On-premise | Cloud
│              Hybrid = two or more locations in one deployment
└── Provider : In-house | Vendor
               independent of Location
```

<a id="fig-1"></a>
Fig 1. Two axes of AI deployment

Any value of one axis pairs with any value of the other.
A vendor's FDC product installed on a fab server is on-premise, and the same product installed on a PC beside the tool is edge.

The locations are ordered by their distance from the tool in Fig 2.

```text
Location (distance from the tool)
├── Edge       : tool controller or a PC beside the tool
├── On-premise : servers inside the fab
└── Cloud      : servers outside the company
```

<a id="fig-2"></a>
Fig 2. Hierarchy of deployment locations

Each step down adds network hops between the tool and the model, so response latency grows, and the compute available to the model grows with it.
The data also travels further at each step: it leaves the tool at on-premise and leaves the company at cloud.
Hybrid splits roles among the three locations, typically training on one and inference on another, so it has no level of its own in Fig 2.

### 3.1 Placement

Table 1 places each location by where it runs, its latency, how far the data travels and the fab functions it serves.

Table 1. Deployment locations

| Location   | Runs on                               | Latency          | Data travel           | Fab use                                           |
| :--------: | :-----------------------------------: | :--------------: | :-------------------: | :-----------------------------------------------: |
| Edge       | Tool controller or PC beside the tool | Lowest           | Stays beside the tool | FDC fault detection, wafer-to-wafer R2R           |
| On-premise | Servers inside the fab                | Fab network hops | Stays in the fab      | Virtual metrology, APC, yield analysis over tools |
| Cloud      | Servers outside the company           | Highest          | Leaves the company    | Training on data cleared for export               |
| Hybrid     | Two or more of the above              | Set per role     | Set per role          | Training on fab servers, inference at the edge    |

The fab use column lists functions, and a function moves to another row when its latency or data rule changes.

## 4. Application

The location of a function is set by how soon its result must act on the tool and how far its data may travel.
Table 2 states what each location assumes and the condition under which it breaks.

Table 2. Assumptions and breaking conditions

| Location   | Assumes                                                       | Breaks when                                                         |
| :--------: | :-----------------------------------------------------------: | :-----------------------------------------------------------------: |
| Edge       | The model fits the compute and memory of the tool PC          | The model needs data from several tools at once                     |
| On-premise | A round trip over the fab network ends before the next action | The action must happen inside the tool's own control cycle          |
| Cloud      | Security rules allow the data to leave the company            | The data is barred from export or the result must return in a cycle |
| Hybrid     | The model moves from training to inference with its version   | The model at inference differs from the one validated               |

Each row of Table 2 applies by location, whichever provider supplies the software.
A vendor product installed at the edge carries the edge row, and an in-house model on a fab server carries the on-premise row.

---

## Appendix A. Terminology

- **APC**: Advanced process control. Adjusts process recipes from measured or predicted results across runs.
- **Edge**: The location on or beside the tool where the data is produced.
- **FDC**: Fault detection and classification. Detects abnormal tool behavior from sensor data during processing.
- **Inference**: Running a trained model on new data to produce a prediction.
- **R2R**: Run-to-run control. Corrects the recipe of the next run from the result of the previous one.
- **Virtual metrology**: Predicting a measurement from tool sensor data instead of measuring the wafer.
