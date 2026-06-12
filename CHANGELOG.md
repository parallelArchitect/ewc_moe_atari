# Changelog

## [gb10-uma-aware] — parallelArchitect fork

### Overview

This fork adds GB10/DGX Spark unified memory awareness to the out-of-core MoE
architecture. The original codebase assumes a discrete GPU memory model —
CPU RAM and GPU VRAM as separate pools. On GB10 (Grace Blackwell Superchip),
CPU and GPU share a single physical LPDDR5X pool. This is an architectural
difference, not a flaw in the original design. The changes here adapt the
expert lifecycle management to the unified memory model.

---

### Changes to Existing Code

#### `core/tiered_store.py` — Page cache release on expert eviction

On unified LPDDR5X, a `.pt` file read from NVMe enters the kernel page cache
and stays there after the Python reference is dropped. When the next expert
loads, both the old and new expert files occupy the same physical pool
simultaneously.

Added `posix_fadvise(POSIX_FADV_DONTNEED)` after each LRU eviction to release
the evicted file's pages from page cache before the next expert loads.

#### `core/expert_manager.py` — Memory check before expert load

`cudaMemGetInfo` returns N/A on GB10 — NVML memory clock is not exposed by
the driver. Added `/proc/meminfo` MemAvailable read before each `_load_expert`
call. This is the correct memory signal on unified memory platforms.

A warning is logged if available memory drops below 2GB before a load.
The adaptive swarm router provides the decision signal for load scheduling.

---

### New Module: `gb10/`

Hardware signal readers and adaptive load scheduler for GB10 unified memory.
All hardware readings come from live kernel interfaces (sysfs, /proc).
Requires optuna>=4.9.0 for the swarm optimizer (see requirements.txt).

#### `gb10/memory.py`
Live readings from `/proc/meminfo` and `/proc/pressure/memory`.
PSI memory pressure is the ground truth stall signal on GB10 —
not GPU utilization, not nvidia-smi output.

#### `gb10/thermal.py`
Live ACPI thermal zone readings with confirmed GB10 zone labels.
Zone mapping confirmed identical across multiple OEMs and kernel versions.

CX7 ConnectX-7 NIC ASIC temperature read from mlx5 hwmon sysfs interface —
present automatically when the mlx5 driver loads. All NIC traffic routes
through the Grace CPU fabric on GB10, making CX7 thermal state relevant
to system load.

#### `gb10/fitness.py`
Expert load cost function combining four live hardware signals:

```
swap_cost = w_psi  * psi_pressure
          + w_mem  * memory_consumption
          + w_tgpu * gpu_temperature
          + w_cx7  * cx7_temperature
```

Weights are not hardcoded. The QuadSwarm finds the optimal weights
from hardware behavior at runtime.

#### `gb10/swarm_router.py`
Four-swarm optimizer using Optuna v4.9 multi-objective study.

Four specialized swarms run in parallel, each biased toward a different
hardware signal:
- PSO — minimizes swap latency (PSI pressure signal)
- Firefly — minimizes LPDDR5X consumption (MemAvailable signal)
- AFSA — minimizes memory footprint per load (memory ratio signal)
- FSO — minimizes thermal contribution (TGPU + CX7 signals)

Optuna manages the study, stores trials to SQLite, and builds a Pareto
front across all four objectives. Knowledge accumulates across training
runs — each session builds on the previous.

To visualize optimization history:
```bash
optuna-dashboard sqlite:///gb10_swarm.db
```

---

### GB10 Platform Reference

| Signal | Source |
|--------|--------|
| Memory ground truth | `/proc/meminfo` MemAvailable |
| Memory stall signal | `/proc/pressure/memory` PSI some_avg10 |
| GPU temperature | thermal_zone5 (TGPU) via acpitz sysfs |
| NIC temperature | mlx5 hwmon asic channel via sysfs |
| cudaMemGetInfo | N/A on GB10 — not usable |
| nvidia-smi clocks.mem | N/A on GB10 — not usable |

**Confirmed thermal zone mapping across multiple OEMs, kernel 6.11–6.17:**

| thermal_zone | ACPI Name | Description |
|---|---|---|
| thermal_zone0 | TSOC | SoC temperature |
| thermal_zone1 | TS0E | CPU cluster 0 efficiency cores |
| thermal_zone2 | TS0P | CPU cluster 0 performance cores |
| thermal_zone3 | TS1E | CPU cluster 1 efficiency cores |
| thermal_zone4 | TS1P | CPU cluster 1 performance cores |
| thermal_zone5 | TGPU | GPU die temperature |
| thermal_zone6 | TUNC | Unknown component |

---

### Planned — Pending GB10 Hardware Validation

- `gb10/ptx_probe.py` — PTX bandwidth measurement integrated into fitness function
- `gb10/sparkview_bridge.py` — pre-load telemetry capture and anomaly logging
- Full unified memory tier model redesign in `TieredStore`
- Multi-node swarm coordination across dual GB10 cluster

---

### GB10 Validation

See `gb10/README.md` for setup, usage, and querying results.

To share results or report issues:
https://github.com/parallelArchitect/ewc_moe_atari/issues
