# gb10/ — GB10 Unified Memory Module

GB10/DGX Spark hardware-aware workload scheduler for unified-memory platforms.

The NVIDIA Grace Blackwell Superchip (GB10) combines Grace CPU and Blackwell GPU
compute resources behind a shared LPDDR5X memory subsystem. Unlike traditional
discrete GPU systems, CPU and GPU workloads compete for the same physical memory
capacity and bandwidth.

This module provides GB10-specific telemetry collection and adaptive scheduling
logic that incorporates memory pressure (PSI), thermal state, and resource
contention signals. The scheduler uses these platform metrics to distribute work
dynamically while accounting for the characteristics of the GB10 unified-memory
architecture.

---

## Requirements

```bash
pip install -r requirements.txt   # includes optuna>=4.9.0
sudo apt install sqlite3          # optional — for querying gb10_swarm.db
```

SQLite3 is built into Python. The `sqlite3` CLI is optional for direct queries.

---

## Module Structure

```
gb10/
├── memory.py        — /proc/meminfo EMV + /proc/pressure/memory PSI
├── thermal.py       — ACPI thermal zones + mlx5 hwmon CX7 temperature
├── fitness.py       — four-signal expert load cost function
└── swarm_router.py  — QuadSwarm multi-objective optimizer (Optuna v4.9)
```

---

## Hardware Signals

All readings come directly from the Linux kernel via sysfs and /proc.

| Signal | Path | Notes |
|--------|------|-------|
| Memory available | `/proc/meminfo` MemAvailable | Ground truth on GB10 |
| Memory stall | `/proc/pressure/memory` PSI | Real stall signal |
| GPU temperature | `/sys/class/thermal/thermal_zone5/temp` | TGPU via acpitz |
| CPU perf cores | `/sys/class/thermal/thermal_zone4/temp` | TS1P — fan trigger zone |
| CX7 NIC temp | `/sys/class/hwmon/hwmon*/` mlx5 | ~45°C idle, ~59°C peak |
| NVMe temp | `/sys/class/hwmon/hwmon*/` nvme | ~38°C idle |

`cudaMemGetInfo` returns N/A on GB10. `nvidia-smi --query-gpu=clocks.mem`
returns N/A on GB10. Do not use either for memory decisions.

---

## Running the Swarm

```bash
cd ewc_moe_atari
PYTHONPATH=. python3 gb10/swarm_router.py
```

Runs 20 trials by default. Results save to `gb10_swarm.db`.
Each run adds to the existing study — knowledge accumulates across sessions.

To run more trials:

```python
from gb10.swarm_router import GB10SwarmRouter

router = GB10SwarmRouter(db_path="gb10_swarm.db", study_name="gb10_uma")
router.optimize(n_trials=200)
print(router.best_weights)
```

---

## Querying Results

```bash
# Total trials accumulated
sqlite3 gb10_swarm.db "SELECT COUNT(*) FROM trials;"

# Best weights found
sqlite3 gb10_swarm.db "
SELECT tp.param_name, tp.param_value
FROM trials t
JOIN trial_params tp ON t.trial_id = tp.trial_id
WHERE t.trial_id = (
    SELECT trial_id FROM trial_values ORDER BY value ASC LIMIT 1
)
ORDER BY tp.param_name;"

# Full objective breakdown for best trial
sqlite3 gb10_swarm.db "
SELECT t.trial_id, tp.param_name, tp.param_value, tv.value
FROM trials t
JOIN trial_params tp ON t.trial_id = tp.trial_id
JOIN trial_values tv ON t.trial_id = tv.trial_id
WHERE t.trial_id = (
    SELECT trial_id FROM trial_values ORDER BY value ASC LIMIT 1
)
ORDER BY tp.param_name;"
```

Or launch the full dashboard:

```bash
optuna-dashboard sqlite:///gb10_swarm.db
```

---

## Integration with ExpertManager

The swarm router informs `ExpertManager.retrieve_or_create()` decisions.
Use `swap_cost_snapshot()` before loading a new expert:

```python
from gb10.fitness import swap_cost_snapshot
from gb10.swarm_router import GB10SwarmRouter

router = GB10SwarmRouter()

# Before loading an expert:
weights = router.get_routing_weights()
snap = swap_cost_snapshot(**weights)

if snap["swap_cost"] > 0.8:
    # High pressure — defer load or evict first
    pass
else:
    # Safe to load
    expert = expert_manager.retrieve_or_create(embedding, code_idx)
    router.update(snap["swap_cost"], weights)
```

---

## Issues and Data Sharing

https://github.com/parallelArchitect/ewc_moe_atari/issues
