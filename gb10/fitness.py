"""
gb10/fitness.py - GB10 expert swap cost function

No governors. No hardcoded limits. No clamping.
The swarm sees raw hardware signals and learns the scale from data.

On GB10, fitness = expert swap cost:

    swap_cost = psi_some_avg10        * w_psi
              + mem_pressure_raw      * w_mem
              + tgpu_c                * w_tgpu
              + cx7_c                 * w_cx7

Signal scales are learned from observed maximums — not hardcoded.
If a signal exceeds the observed max, the observed max updates.
The swarm calibrates itself against the hardware it runs on.

If TGPU or CX7 are None (Pascal dev, no GB10 hardware),
those terms are excluded from the cost entirely — not zeroed,
excluded — so the swarm does not learn incorrect weight distributions
from missing signals.

Reference temperatures are not limits. They are observations.
The swarm is not bounded by them. They update when hardware exceeds them.
"""

import threading
from typing import Dict, Optional, Tuple


# Self-calibrating signal maximums.
# These are not governors. They are running observations.
# Updated live from hardware. Thread-safe.
_lock = threading.Lock()

_signal_max = {
    "psi":  100.0,   # PSI some_avg10 — 0 to 100 by definition, no cap needed
    "mem":  1.0,     # memory pressure ratio — recomputed each snapshot
    "tgpu": 90.0,    # initial observation from azampatti field data
    "cx7":  59.0,    # initial observation from azampatti field data
}

_signal_history = {
    "psi":  [],
    "tgpu": [],
    "cx7":  [],
}

_MAX_HISTORY = 1000


def _update_observed_max(key: str, value: float):
    """Update rolling observed maximum. No clamp. Hardware defines the ceiling."""
    with _lock:
        if value > _signal_max[key]:
            _signal_max[key] = value
        history = _signal_history.get(key)
        if history is not None:
            history.append(value)
            if len(history) > _MAX_HISTORY:
                history.pop(0)


def _normalize_raw(key: str, value: float) -> float:
    """
    Normalize against observed max. No clamp on output.
    If hardware exceeds prior observations, ratio exceeds 1.0.
    That is correct — it means the signal is in territory the swarm
    has not seen before. The swarm should respond to that, not suppress it.
    """
    with _lock:
        ref = _signal_max[key]
    if ref <= 0:
        return 0.0
    return value / ref


def swap_cost(
    psi_some_avg10:   float,
    mem_available_kb: int,
    mem_total_kb:     int,
    tgpu_c:           Optional[float],
    cx7_c:            Optional[float],
    w_psi:            float = 0.4,
    w_mem:            float = 0.4,
    w_tgpu:           float = 0.1,
    w_cx7:            float = 0.1,
) -> float:
    """
    Compute expert swap cost from live GB10 signals.

    No clamping. No normalization ceiling.
    Signals that exceed prior observations produce ratios > 1.0.
    The swarm sees the real magnitude.

    Active signals only — None signals excluded from cost and weight sum.
    The swarm learns correct distributions even on partial hardware.

    Args:
        psi_some_avg10:   PSI memory pressure 10s avg (0.0-100.0 by kernel definition)
        mem_available_kb: /proc/meminfo MemAvailable (EMV — ground truth on GB10)
        mem_total_kb:     /proc/meminfo MemTotal
        tgpu_c:           GPU die temperature Celsius — None on non-GB10
        cx7_c:            CX7 NIC ASIC temperature Celsius — None on non-GB10
        w_*:              swarm-tunable weights (found by QuadSwarm, not hardcoded)

    Returns:
        float: swap cost — unbounded above 1.0 under extreme hardware stress
    """
    # Update observed maximums from live readings
    _update_observed_max("psi", psi_some_avg10)

    # PSI term — raw ratio against observed max
    psi_term = _normalize_raw("psi", psi_some_avg10)

    # Memory term — ratio of consumed memory to total
    # No clamp. If mem_available_kb goes negative (kernel accounting edge case),
    # that is a real signal — memory is critically over-committed.
    if mem_total_kb > 0:
        mem_term = 1.0 - (mem_available_kb / mem_total_kb)
    else:
        mem_term = 0.0

    # Build active signal set — exclude None signals entirely
    active_terms = [
        (w_psi,  psi_term),
        (w_mem,  mem_term),
    ]

    if tgpu_c is not None:
        _update_observed_max("tgpu", tgpu_c)
        tgpu_term = _normalize_raw("tgpu", tgpu_c)
        active_terms.append((w_tgpu, tgpu_term))

    if cx7_c is not None:
        _update_observed_max("cx7", cx7_c)
        cx7_term = _normalize_raw("cx7", cx7_c)
        active_terms.append((w_cx7, cx7_term))

    # Weighted sum over active signals only
    # No normalization of weights here — the swarm decides the scale
    cost = sum(w * t for w, t in active_terms)

    return cost


def swap_cost_snapshot(
    w_psi:  float = 0.4,
    w_mem:  float = 0.4,
    w_tgpu: float = 0.1,
    w_cx7:  float = 0.1,
) -> Dict:
    """
    Read all live signals and compute swap cost in one call.
    This is what the QuadSwarm calls during each trial evaluation.

    Returns full signal snapshot including observed maximums —
    so the swarm can inspect what the hardware has taught it.
    """
    from gb10.memory import memory_snapshot
    from gb10.thermal import thermal_snapshot

    mem   = memory_snapshot()
    therm = thermal_snapshot()

    cost = swap_cost(
        psi_some_avg10   = mem["psi_some_avg10"],
        mem_available_kb = mem["emv"]["mem_available_kb"],
        mem_total_kb     = mem["emv"]["mem_total_kb"],
        tgpu_c           = therm.get("TGPU"),
        cx7_c            = therm.get("CX7"),
        w_psi=w_psi, w_mem=w_mem, w_tgpu=w_tgpu, w_cx7=w_cx7,
    )

    with _lock:
        observed_maxes = dict(_signal_max)

    return {
        "swap_cost":        cost,
        "psi_some_avg10":   mem["psi_some_avg10"],
        "psi_full_avg10":   mem["psi_full_avg10"],
        "mem_available_gb": mem["available_gb"],
        "mem_total_kb":     mem["emv"]["mem_total_kb"],
        "tgpu_c":           therm.get("TGPU"),
        "ts1p_c":           therm.get("TS1P"),
        "cx7_c":            therm.get("CX7"),
        "observed_maxes":   observed_maxes,
        "weights": {
            "w_psi": w_psi, "w_mem": w_mem,
            "w_tgpu": w_tgpu, "w_cx7": w_cx7,
        },
    }


def signal_intelligence() -> Dict:
    """
    Return what the swarm has learned about hardware signal ranges.
    This is the intelligence layer — the swarm's accumulated knowledge
    of GB10 hardware behavior across all runs.
    """
    with _lock:
        maxes = dict(_signal_max)
        histories = {k: list(v) for k, v in _signal_history.items()}

    result = {}
    for key, history in histories.items():
        if history:
            result[key] = {
                "observed_max":  maxes[key],
                "session_max":   max(history),
                "session_min":   min(history),
                "session_mean":  sum(history) / len(history),
                "sample_count":  len(history),
            }
        else:
            result[key] = {
                "observed_max": maxes[key],
                "sample_count": 0,
            }
    return result


if __name__ == "__main__":
    import json
    snap = swap_cost_snapshot()
    print(json.dumps(snap, indent=2))
    print("\nSignal intelligence:")
    print(json.dumps(signal_intelligence(), indent=2))
