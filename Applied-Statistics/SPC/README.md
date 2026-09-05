# SPC (Statistical Process Control)
Rev. 4 | Created: 2026-09-04 | Updated: 2026-09-05 08:29 CDT

> This folder covers the statistics that decide whether a running process is behaving, and whether
> its behaviour is good enough for the product.
> Its documents work outward from a single measurement on a stable process to the hundreds of sensor
> channels a tool leaves behind.

## 1. Scope

Statistical process control is the practice of separating two kinds of variation with statistics
computed from the process itself. Common cause variation is the sum of many small influences that
are always present, and it is a property of the process; special cause variation comes from an
identifiable event and can be found and removed. Confusing the two makes a process worse in either
direction, because adjusting for common cause variation adds to it, and dismissing a special cause
leaves it in place.

Every method here follows from that split. A control chart declares which of the two is present, and
a pair of charts separates the two things a distribution can do, move and widen. A capability index
asks a different question — whether the stable process meets its specification — and it is only
meaningful once the charts have settled the first one. A uniformity index is the same
dispersion statistic applied inside a single wafer rather than across a run. Multivariate methods
carry the whole idea over to the case where the measurement is not one number but a vector of
correlated sensor channels.

The material is written for semiconductor process and metrology data, so the worked examples are
film thickness, subgroup measurements across wafers, and equipment sensor traces.

## 2. Documents

Table 1. Documents in this folder

| Document | Language | Description |
|---|---|---|
| [Shewhart-Chart/shewhart-chart.md](Shewhart-Chart/shewhart-chart.md) | Korean | It covers the common and special cause split the chart is built on, why the limits sit at three sigma and what false alarm rate that buys, the chart types for variables and attributes, the average run length to a signal, the Western Electric run rules, and why being in control is not the same as being capable. |
| [S-Chart/s-chart.md](S-Chart/s-chart.md) | Korean | It covers the subgroup standard deviation as a chart statistic — why it is a biased estimator of sigma and how $c_4$ corrects it, where the $B_3$ and $B_4$ limits come from and why small subgroups have no lower limit, how the range loses efficiency as the subgroup grows, and the $A_3$ constant that ties the pair to the mean chart. |
| [Xbar-S-Chart/xbar-s-chart.md](Xbar-S-Chart/xbar-s-chart.md) | Korean | It reads the mean chart and the s chart as one pair: the constants that tie them together, why the s chart is read first, and the two paths by which the spread reaches the mean chart — inflating the scatter of the points while the limits are held, then widening the limits themselves once the new spread is accepted, which costs the mean chart an average run length of 4.5 against 33.4 on a one sigma shift. An appendix takes the case of a spread that jumps while the mean holds as a case study, calls it an excursion, and carries the defect rate that follows along with the four things to rule out before saying so. |
| [Process-Capability-Index/process-capability-index.md](Process-Capability-Index/process-capability-index.md) | English, [Korean](Process-Capability-Index/process-capability-index-ko.md) | It defines $C_p$, $k$ and $C_{pk}$, derives $C_{pk} = (1-k)C_p$, converts each index into a defect rate, and shows with three worked processes why the indices are read together: the same $C_p$ with different centring costs a factor of twenty in defect rate, and the same $C_{pk}$ can mean either a misaligned process or a wide one. It also sets out the priority graded requirement a fab attaches to each measured parameter, with a $C_{pk}$ minimum, a $k$ maximum and a disposition rule on each grade. |
| [Uniformity/wafer-uniformity-index.md](Uniformity/wafer-uniformity-index.md) | English, [Korean](Uniformity/wafer-uniformity-index-ko.md) | It gives the two standard formulas for within-wafer uniformity, the range method and the standard deviation method, what each measures physically, why neither can see the spatial signature that determines the corrective action, why the range index grows with the measurement point count, and how the index is used across deposition, etch, CMP, implant and lithography. |
| [Multivariate-SPC/multivariate-spc.md](Multivariate-SPC/multivariate-spc.md) | Korean | It covers why one chart per sensor fails on a tool with many correlated channels, the Hotelling $T^2$ chart as a Mahalanobis distance with its $F$ based limit, PCA monitoring with the score space $T^2$ and the SPE residual statistic, what each of the two catches, contribution analysis and its smearing, and the fault detection use in a fab. |

Each subfolder also holds the script that produced its figure and the csv tables the document quotes,
so every number in the text can be traced to a run rather than to a table copied from elsewhere.

## 3. Order Of Use

1. Establish the subgroup first. A subgroup must contain only common cause variation, so wafers from
   different chambers do not belong in one, and getting this wrong widens the limits until the chart
   signals nothing.
2. Chart the dispersion before the location. The limits of a mean chart are computed from an
   estimate of the spread, so those limits mean nothing until the spread itself is in control.
3. Remove the special causes the charts find, and only then compute a capability index. An index
   estimated on an unstable process does not predict the next period.
4. Read the capability pair rather than one index. $C_p$ and $k$ separate a process that is too wide
   from one that is merely off centre, and the two call for different and differently priced work.
5. Track the within-wafer uniformity index as its own statistic. It is one term of the variance
   budget, not the whole of it, so it sits beside a chart of the wafer mean rather than replacing it.
6. Move to the multivariate statistics when the thing being watched is the tool rather than the
   product. They see the fault before the wafer is measured, at the cost of a model that has to be
   built on healthy data and rebuilt after maintenance.

The order is deliberate. Steps 1 to 3 have to be done in sequence because each one supplies what the
next assumes; steps 4 to 6 are independent of each other and can be adopted in any order.
