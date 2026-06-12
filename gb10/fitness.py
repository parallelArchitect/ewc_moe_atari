"""
gb10/fitness.py - GB10 expert swap cost function

The fitness signal the QuadSwarm optimizes against.

On Pascal, fitness = CUDA kernel execution time (microseconds).
On GB10, fitness = expert swap cost:

    swap_cost = psi_some_avg10 * w_psi
              + (1 - mem_available_ratio) * w_mem
              + tgpu_normalized * w_tgpu
              + cx7_normalized * w_cx7

No hardcoded limits. No governors.
The swarm finds the optimal weights through measurement.
If TGPU or CX7 are None (Pascal dev, no GB10 hardware), those terms drop to 0.0.
"""

from typing import Dict, Optional
from gb10.memory import memory_snapshot
from gb10.thermal import thermal_snapshot


# Reference temperatures from azampatti ASUS GX10 field measurements.
# These are observed values, not limits. The swarm is not bounded by them.
_TGPU_REF_C  = 90.0   # observed under sustained CUTLASS GEMM load
_CX7_REF_C   = 59.0   # observed peak from mlx5 hwmon highest= field
_TS1P_REF_C  = 90.0   # observed fan trigger zone under GPU load


def swap_cost(
    psi_some_avg10: float,
    mem_available_kb: int,
    mem_total_kb: int,
    tgpu_c: Optional[float],
    cx7_c: Optional[float],
    w_psi: float  = 0.4,
    w_mem: float  = 0.4,
    w_tgpu: float = 0.1,
    w_cx7: float  = 0.1,
) -> float:
    """
    Compute expert swap cost from live GB10 signals.

    Lower cost = better time to swap this expert in.
    Higher cost = defer the swap, current state is stressed.

    Args:
        psi_some_avg10: PSI memory pressure 10s avg (0.0-100.0)
        mem_available_kb: /proc/meminfo MemAvailable
        mem_total_kb: /proc/meminfo MemTotal
        tgpu_c: GPU die temperature in Celsius (None on non-GB10)
        cx7_c: CX7 NIC ASIC temperature in Celsius (None on non-GB10)
        w_*: swarm-tunable weights (found by QuadSwarm, not hardcoded)

    Returns:
        float: swap cost in [0.0, 1.0] range
    """
    # PSI term: 0.0 (no stall) to 1.0 (100% stall)
    psi_term = min(psi_some_avg10 / 100.0, 1.0)

    # Memory term: 0.0 (all memory free) to 1.0 (no memory available)
    if mem_total_kb > 0:
        mem_term = 1.0 - (mem_available_kb / mem_total_kb)
        mem_term = max(0.0, min(mem_term, 1.0))
    else:
        mem_term = 0.0

    # Thermal terms: normalized against reference observations.
    # None = not available (Pascal dev box) = 0.0 contribution.
    tgpu_term = (tgpu_c / _TGPU_REF_C) if tgpu_c is not None else 0.0
    tgpu_term = max(0.0, min(tgpu_term, 1.0))

    cx7_term = (cx7_c / _CX7_REF_C) if cx7_c is not None else 0.0
    cx7_term = max(0.0, min(cx7_term, 1.0))

    cost = (w_psi  * psi_term +
            w_mem  * mem_term +
            w_tgpu * tgpu_term +
            w_cx7  * cx7_term)

    return cost


def swap_cost_snapshot(
    w_psi: float  = 0.4,
    w_mem: float  = 0.4,
    w_tgpu: float = 0.1,
    w_cx7: float  = 0.1,
) -> Dict:
    """
    Read all live signals and compute swap cost in one call.
    This is what the QuadSwarm calls during each trial evaluation.
    """
    mem  = memory_snapshot()
    therm = thermal_snapshot()

    cost = swap_cost(
        psi_some_avg10   = mem["psi_some_avg10"],
        mem_available_kb = mem["emv"]["mem_available_kb"],
        mem_total_kb     = mem["emv"]["mem_total_kb"],
        tgpu_c           = therm.get("TGPU"),
        cx7_c            = therm.get("CX7"),
        w_psi=w_psi, w_mem=w_mem, w_tgpu=w_tgpu, w_cx7=w_cx7,
    )

    return {
        "swap_cost":        cost,
        "psi_some_avg10":   mem["psi_some_avg10"],
        "psi_full_avg10":   mem["psi_full_avg10"],
        "mem_available_gb": mem["available_gb"],
        "mem_total_kb":     mem["emv"]["mem_total_kb"],
        "tgpu_c":           therm.get("TGPU"),
        "ts1p_c":           therm.get("TS1P"),
        "cx7_c":            therm.get("CX7"),
        "weights": {
            "w_psi": w_psi, "w_mem": w_mem,
            "w_tgpu": w_tgpu, "w_cx7": w_cx7,
        },
    }


if __name__ == "__main__":
    import json
    snap = swap_cost_snapshot()
    print(json.dumps(snap, indent=2))
