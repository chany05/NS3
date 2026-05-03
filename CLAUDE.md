# Current Progress Notes

## Final Artifacts

- Current final CSV: `twocell_cio_threshold_fix_500.csv`
- Current final analysis directory: `twocell_results/`
- Main scenario file: `src/oran/examples/TwoCellConflictScenario.cc`
- Main analysis script: `analyze_twocell_conflicts.py`

## Current Scenario State

- Active xApps: `CCO`, `ES`, `MLB`
- Removed xApp: `MRO`
- Active ICPs:
  - `TxPower`
  - `RET` / electrical downtilt proxy
  - `CIO` proxy
- Removed from xApp-controlled ICPs:
  - `TTT`
  - `HYS`
- Active KPI map:
  - `SINR`
  - `DL Throughput`
  - `Traffic Load`
  - `RB Utilization`
- Removed/inactive from analysis map:
  - HO success/failure related constant KPMs
  - UL SINR
  - unused legacy KPMs

## Final Run Summary

Input:

```text
twocell_cio_threshold_fix_500.csv
```

Output:

```text
twocell_results/
```

Metrics from `twocell_results/metrics.txt`:

```text
threshold=0.5
level=2
warmup_step=9
action_mismatch_count=81
inactive_kpis=

[Level1-Pearson]
precision=0.4000
recall=0.3333
f1=0.3636

[Level2-GNN]
precision=0.6364
recall=0.5833
f1=0.6087
```

CSV summary:

```text
rows=248
steps=0..247
CCO actions=84
ES actions=82
MLB actions=82
```

## Current Behavior

- `CCO` controls `TxPower` upward when SINR/DL throughput is low.
- `CCO` also controls `RET` when TxPower is already at the upper bound or when restoring toward baseline.
- Current CCO RET thresholds:
  - degrade coverage: `SINR < 12 dB` or `DLThp < 1.0 Mbps`
  - restore toward baseline: `SINR > 12 dB` and `DLThp > 1.5 Mbps`
  - restore applies one ICP per CCO step, prioritizing RET before TxPower.
- `ES` controls the same `TxPower` downward when SINR/DL throughput has margin.
- This creates the intended direct conflict:
  - `CCO <-> ES via c0.TxP`
  - `CCO <-> ES via c1.TxP`
- `MLB` controls `CIO` based on RB-utilization imbalance.
- CIO no longer monotonically increases to the upper clamp.
- Current CIO range in the final CSV:
  - `c0_Param_CIO`: min `2`, max `3`, last `3`
  - `c1_Param_CIO`: min `2`, max `3`, last `2`

## Important Implementation Notes

- Fading is disabled for cleaner ICP-to-KPI relationships.
- `SINR` is currently a designed proxy, not raw PHY SINR.
- RET is applied by rotating the LTE `ThreeGppAntennaModel` gain pattern through `DowntiltAngle`.
- A3 `Hysteresis` and `TimeToTrigger` remain fixed internal handover settings, not xApp-controlled ICPs and not graph features.
- Boundary UEs are placed on the perpendicular bisector between the two cells.
- Anchor UEs are kept near each cell and should not be removed unless explicitly requested.
- Initial boundary attachment is balanced, not intentionally asymmetric.
- `rbUtil` is a throughput/capacity proxy, not the old scheduled-subframe ratio.
- `trafficLoad` is calculated as each cell's share of total DL throughput.

## CIO / MLB Fix Applied

The previous MLB behavior caused CIO to keep increasing until the upper bound. The current logic changed this:

- Removed unconditional CIO increase for underloaded cells.
- MLB now lowers CIO only for the overloaded cell.
- If imbalance disappears, CIO moves back toward the baseline.
- Baseline CIO is initialized from `initCio`.
- The ES/MLB schedule was adjusted so MLB observes the cell made overloaded by the preceding ES action.

Smoke run after this fix showed:

```text
c0_CIO_delta includes -1, 0, +1
c1_CIO_delta includes -1, +1
CIO stayed in range 2..3 instead of climbing to 15
```

## Remaining Caveats

- ES is still a proxy energy-saving xApp. It does not yet use real `Energy Efficiency` or `Power Consumption` KPMs.
- RET is not a mechanical panel rotation. It is implemented as a closest LTE-stack approximation by adding `BearingAngle` and `DowntiltAngle` attributes to `ThreeGppAntennaModel`.
- `CIO -> Load/RButil` is now more stable in lagged behavior, but zero-lag Pearson is still weak. The final validation report shows useful lag-1 signal for CIO/RButil:
  - `c0.CIO -> c0.RButil lag1=0.408`
  - `c1.CIO -> c1.RButil lag1=0.406`
- TxPower action mismatch remains expected because CCO requests `+2`, ES requests `-1`, and TxPower is clamped to `[20, 46]`.

## Next Best Step

If continuing from here, do not change the GNN method first. The scenario side should be refined first:

- Add real or proxy `Power Consumption` and `Energy Efficiency` KPMs before claiming ES is energy-based.
- If MLB needs stronger graph reconstruction, make `CIO` influence load/RB utilization more directly or evaluate the graph with lag-aware edges.
- Keep `twocell_cio_threshold_fix_500.csv` and `twocell_results/` as the current final reference unless a newer run is explicitly promoted.
