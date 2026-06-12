"""
gb10/memory.py - Live GB10 unified memory signal reader

On GB10, CPU and GPU share one physical LPDDR5X pool.
cudaMemGetInfo returns N/A — /proc/meminfo is ground truth.
PSI memory pressure is the real stall signal.

No governors. No hardcoded limits. Live measurement only.
"""

import os
from typing import Dict, Optional


def read_meminfo() -> Dict[str, int]:
    """
    Read /proc/meminfo — ground truth for GB10 unified memory.
    Returns values in kB.
    """
    meminfo = {}
    with open("/proc/meminfo") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0].rstrip(":")
                try:
                    meminfo[key] = int(parts[1])
                except ValueError:
                    pass
    return meminfo


def get_available_kb() -> int:
    """
    MemAvailable — what the kernel believes is actually usable.
    This is the correct signal for expert swap decisions on GB10.
    Not MemFree. Not MemTotal. MemAvailable.
    """
    return read_meminfo().get("MemAvailable", 0)


def get_emv() -> Dict[str, int]:
    """
    Effective Memory View — the three signals that matter on GB10.
    EMV = MemAvailable + SwapFree as the real usable pool.
    """
    m = read_meminfo()
    return {
        "mem_total_kb":     m.get("MemTotal", 0),
        "mem_available_kb": m.get("MemAvailable", 0),
        "mem_free_kb":      m.get("MemFree", 0),
        "swap_free_kb":     m.get("SwapFree", 0),
        "cached_kb":        m.get("Cached", 0),
        "buffers_kb":       m.get("Buffers", 0),
        "emv_kb":           m.get("MemAvailable", 0) + m.get("SwapFree", 0),
    }


def read_psi_memory() -> Optional[Dict[str, float]]:
    """
    Read /proc/pressure/memory — PSI is ground truth for memory stall on GB10.
    Returns avg10, avg60, avg300 (percentage of time stalled) and total (us).
    Returns None if PSI not available.
    """
    psi_path = "/proc/pressure/memory"
    if not os.path.exists(psi_path):
        return None

    result = {}
    with open(psi_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            kind = parts[0]  # "some" or "full"
            for kv in parts[1:]:
                k, v = kv.split("=")
                result[f"{kind}_{k}"] = float(v)
    return result


def get_swap_pressure() -> float:
    """
    Returns PSI some avg10 — percentage of time in the last 10 seconds
    that at least one task was stalled waiting for memory.
    0.0 = no pressure. Higher = more stall.
    This is the expert swap decision signal.
    """
    psi = read_psi_memory()
    if psi is None:
        return 0.0
    return psi.get("some_avg10", 0.0)


def get_full_stall() -> float:
    """
    PSI full avg10 — all tasks stalled. This is critical pressure.
    On GB10 this means the unified memory pool is exhausted.
    """
    psi = read_psi_memory()
    if psi is None:
        return 0.0
    return psi.get("full_avg10", 0.0)


def memory_snapshot() -> Dict:
    """
    Complete memory state snapshot for swarm fitness evaluation.
    Called before and after each expert swap trial.
    """
    emv = get_emv()
    psi = read_psi_memory() or {}
    
    return {
        "emv":              emv,
        "psi_some_avg10":   psi.get("some_avg10", 0.0),
        "psi_some_avg60":   psi.get("some_avg60", 0.0),
        "psi_full_avg10":   psi.get("full_avg10", 0.0),
        "psi_total_us":     psi.get("some_total", 0.0),
        "available_gb":     emv["mem_available_kb"] / (1024 * 1024),
    }


if __name__ == "__main__":
    import json
    snap = memory_snapshot()
    print(json.dumps(snap, indent=2))
