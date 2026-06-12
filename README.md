# ewc_moe_atari — GB10/DGX Spark Fork

Fork of [RespectableGlioma/ewc_moe_atari](https://github.com/RespectableGlioma/ewc_moe_atari)

This fork adapts the out-of-core Mixture of Experts continual learning architecture
for the GB10/DGX Spark unified memory platform.

---

## What This Fork Adds

The original codebase assumes a discrete GPU memory model — CPU RAM and GPU VRAM
as separate pools. On GB10 (Grace Blackwell Superchip), CPU and GPU share a single
physical LPDDR5X pool. This fork adapts the expert lifecycle management to that
architecture.

**Changes to existing code:**
- `core/tiered_store.py` — page cache release on expert eviction (`posix_fadvise`)
- `core/expert_manager.py` — memory check before expert load (`/proc/meminfo`)

**New module:**
- `gb10/` — hardware signal readers and QuadSwarm multi-objective optimizer

See [CHANGELOG.md](CHANGELOG.md) for the full technical detail.

---

## GB10 Module

```
gb10/
├── memory.py        — /proc/meminfo EMV + PSI memory pressure
├── thermal.py       — ACPI thermal zones + CX7 NIC temperature
├── fitness.py       — four-signal expert load cost function
└── swarm_router.py  — QuadSwarm multi-objective optimizer (Optuna v4.9)
```

See [gb10/README.md](gb10/README.md) for setup, usage, and querying results.

---

## Original Project

The original `ewc_moe_atari` implements continual learning across multiple Atari
games using a hierarchical out-of-core Mixture of Experts architecture. A small
meta-agent routes observations to specialized expert networks stored on disk,
allowing scaling to many games without proportional memory growth.

See [Claude.md](Claude.md) for the full architecture description.

---

## Issues and Data Sharing

https://github.com/parallelArchitect/ewc_moe_atari/issues
