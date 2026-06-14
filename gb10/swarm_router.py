"""
gb10/swarm_router.py - GB10 QuadSwarm — no governors, no bounds

Four objectives optimized simultaneously on live GB10 hardware signals:
  1. Minimize PSI memory stall pressure
  2. Minimize LPDDR5X memory consumption
  3. Minimize GPU die temperature contribution
  4. Minimize CX7 NIC thermal contribution

No clamps. No weight bounds. No floor on adaptive selection.
Weights can take any positive value — normalized to sum to 1 after
the swarm proposes them, but the search space is unbounded.

The swarm kills underperforming sub-swarms entirely.
No 0.05 floor. If a swarm is not contributing, it goes to zero.

Intelligence layer: swarms inspect signal_intelligence() from fitness.py
and self-calibrate particle initialization to the hardware's observed
signal ranges — not to hardcoded priors.

Architecture:
  GB10QuadSwarmSampler (BaseSampler)
    ├── PSO_SpeedRouter      — minimize swap latency (PSI signal)
    ├── Firefly_EnergyRouter — minimize LPDDR5X pressure (MemAvailable signal)
    ├── AFSA_MemoryRouter    — minimize memory footprint per load
    └── FSO_ThermalRouter    — minimize TGPU + CX7 contribution

Pascal validation: 8900 GFLOPS confirmed on GTX 1080 SM6.1
GB10 calibration run: pending hardware

Based on: cuda_ai_swarm QuadSwarm (Pascal GTX 1080, 13509 trials, 7.82ms)
Adapted for GB10 unified memory platform by parallelArchitect
"""

import random
import logging
import numpy as np
import optuna
from optuna.samplers import BaseSampler
from optuna.distributions import FloatDistribution
from optuna.trial import TrialState
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple

from gb10.fitness import swap_cost_snapshot, signal_intelligence
from gb10.memory import memory_snapshot
from gb10.thermal import thermal_snapshot

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class RoutingParticle:
    weights:      Dict[str, float]
    velocity:     Dict[str, float]
    cost:         float
    best_weights: Dict[str, float]
    best_cost:    float
    swarm_type:   str
    age:          int = 0


WEIGHT_PARAMS = ["w_psi", "w_mem", "w_tgpu", "w_cx7"]


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    """Normalize weights to sum to 1. No clamp on individual values."""
    total = sum(abs(v) for v in weights.values())
    if total <= 0:
        return {k: 0.25 for k in weights}
    return {k: abs(v) / total for k, v in weights.items()}


def _hardware_scale() -> Dict[str, float]:
    """
    Read what the hardware has taught us about signal ranges.
    Used to initialize particles at hardware-relevant scales,
    not at hardcoded priors.
    Returns observed maximums per signal key.
    """
    intel = signal_intelligence()
    return {
        "psi":  intel.get("psi",  {}).get("observed_max", 100.0),
        "tgpu": intel.get("tgpu", {}).get("observed_max", 90.0),
        "cx7":  intel.get("cx7",  {}).get("observed_max", 59.0),
    }


class PSO_SpeedRouter:
    """
    PSO — minimizes swap latency.
    Particles initialized at hardware-observed PSI scale.
    No velocity clamp. Particles can escape local minima.
    """
    def __init__(self, size: int = 6):
        self.particles: List[RoutingParticle] = []
        self.size = size
        self.swarm_type = "PSO_SPEED"

    def initialize(self):
        scale = _hardware_scale()
        psi_bias = min(scale["psi"] / 100.0, 2.0)  # relative PSI importance
        for _ in range(self.size):
            w = _normalize({
                "w_psi":  random.uniform(0.3, 0.6) * psi_bias,
                "w_mem":  random.uniform(0.2, 0.5),
                "w_tgpu": random.uniform(0.05, 0.3),
                "w_cx7":  random.uniform(0.05, 0.2),
            })
            self.particles.append(RoutingParticle(
                weights=w,
                velocity={k: random.uniform(-0.1, 0.1) for k in w},
                cost=float("inf"),
                best_weights=w.copy(),
                best_cost=float("inf"),
                swarm_type=self.swarm_type,
            ))

    def update(self, global_best: Optional[RoutingParticle]):
        w_inertia, c1, c2 = 0.7, 2.0, 2.0
        for p in self.particles:
            for k in p.weights:
                r1, r2 = random.random(), random.random()
                cognitive = c1 * r1 * (p.best_weights[k] - p.weights[k])
                social = (c2 * r2 * (global_best.weights[k] - p.weights[k])
                          if global_best else 0.0)
                # No velocity clamp — let particles escape
                p.velocity[k] = w_inertia * p.velocity[k] + cognitive + social
                p.weights[k] = p.weights[k] + p.velocity[k]
            p.weights = _normalize(p.weights)
            p.age += 1


class Firefly_EnergyRouter:
    """
    Firefly — minimizes LPDDR5X pressure.
    Attraction coefficient scales with observed memory pressure range.
    No position clamp. Particles attracted to lower cost, not to [0,1].
    """
    def __init__(self, size: int = 6):
        self.particles: List[RoutingParticle] = []
        self.size = size
        self.swarm_type = "FIREFLY_ENERGY"

    def initialize(self):
        for _ in range(self.size):
            w = _normalize({
                "w_psi":  random.uniform(0.1, 0.4),
                "w_mem":  random.uniform(0.3, 0.7),
                "w_tgpu": random.uniform(0.05, 0.2),
                "w_cx7":  random.uniform(0.05, 0.2),
            })
            self.particles.append(RoutingParticle(
                weights=w,
                velocity={k: 0.0 for k in w},
                cost=float("inf"),
                best_weights=w.copy(),
                best_cost=float("inf"),
                swarm_type=self.swarm_type,
            ))

    def update(self, global_best: Optional[RoutingParticle]):
        alpha, beta0, gamma = 0.3, 1.0, 0.005
        for p in self.particles:
            attracted = False
            for other in self.particles:
                if other.cost < p.cost:
                    dist = sum((p.weights[k] - other.weights[k])**2
                               for k in p.weights) ** 0.5
                    beta = beta0 * np.exp(-gamma * dist**2)
                    for k in p.weights:
                        # No clamp — full attraction + random walk
                        p.weights[k] = (p.weights[k]
                                        + beta * (other.weights[k] - p.weights[k])
                                        + alpha * (random.random() - 0.5))
                    attracted = True
            if not attracted:
                # Random walk with larger step when isolated
                for k in p.weights:
                    p.weights[k] += alpha * 2.0 * (random.random() - 0.5)
            p.weights = _normalize(p.weights)
            p.age += 1


class AFSA_MemoryRouter:
    """
    AFSA — minimizes MemAvailable consumption per expert load.
    Step size adapts to hardware memory scale.
    No position bounds.
    """
    def __init__(self, size: int = 6):
        self.particles: List[RoutingParticle] = []
        self.size = size
        self.swarm_type = "AFSA_MEMORY"

    def initialize(self):
        for _ in range(self.size):
            w = _normalize({
                "w_psi":  random.uniform(0.1, 0.3),
                "w_mem":  random.uniform(0.4, 0.8),
                "w_tgpu": random.uniform(0.05, 0.2),
                "w_cx7":  random.uniform(0.05, 0.15),
            })
            self.particles.append(RoutingParticle(
                weights=w,
                velocity={k: 0.0 for k in w},
                cost=float("inf"),
                best_weights=w.copy(),
                best_cost=float("inf"),
                swarm_type=self.swarm_type,
            ))

    def update(self, global_best: Optional[RoutingParticle]):
        # Step size is not fixed — scales with generation count
        # Larger steps early, smaller steps as swarm matures
        if self.particles:
            avg_age = sum(p.age for p in self.particles) / len(self.particles)
            step = max(0.01, 0.15 * np.exp(-avg_age / 500.0))
        else:
            step = 0.05

        for p in self.particles:
            best_neighbor = min(
                (o for o in self.particles if o is not p),
                key=lambda o: o.cost,
                default=None,
            )
            if best_neighbor and best_neighbor.cost < p.cost:
                for k in p.weights:
                    d = best_neighbor.weights[k] - p.weights[k]
                    # No clamp on step result
                    p.weights[k] += step * random.random() * np.sign(d)
            else:
                for k in p.weights:
                    p.weights[k] += step * (random.random() - 0.5)
            p.weights = _normalize(p.weights)
            p.age += 1


class FSO_ThermalRouter:
    """
    FSO — minimizes TGPU + CX7 thermal contribution.
    Volitive movement uses hardware-observed thermal scale.
    No weight clamp. School weight tracks real fitness, not [0,1].
    """
    def __init__(self, size: int = 6):
        self.particles: List[RoutingParticle] = []
        self.size = size
        self.swarm_type = "FSO_THERMAL"
        self._prev_school_cost = float("inf")

    def initialize(self):
        scale = _hardware_scale()
        # Bias thermal weights toward observed thermal ranges
        tgpu_bias = scale["tgpu"] / 90.0
        cx7_bias  = scale["cx7"]  / 59.0
        for _ in range(self.size):
            w = _normalize({
                "w_psi":  random.uniform(0.05, 0.2),
                "w_mem":  random.uniform(0.05, 0.2),
                "w_tgpu": random.uniform(0.2, 0.5) * tgpu_bias,
                "w_cx7":  random.uniform(0.15, 0.5) * cx7_bias,
            })
            p = RoutingParticle(
                weights=w,
                velocity={k: 0.0 for k in w},
                cost=float("inf"),
                best_weights=w.copy(),
                best_cost=float("inf"),
                swarm_type=self.swarm_type,
            )
            p.fish_weight = 1.0
            self.particles.append(p)

    def update(self, global_best: Optional[RoutingParticle]):
        valid = [p for p in self.particles if p.cost < float("inf")]

        if valid:
            total_fw = sum(getattr(p, "fish_weight", 1.0) for p in valid)
            if total_fw > 0:
                bary = {
                    k: sum(p.weights[k] * getattr(p, "fish_weight", 1.0)
                           for p in valid) / total_fw
                    for k in WEIGHT_PARAMS
                }
                for p in self.particles:
                    for k in p.weights:
                        # No clamp on movement
                        p.weights[k] += 0.15 * (bary[k] - p.weights[k])

        # Volitive movement — school contracting or expanding
        current_school_cost = (sum(p.cost for p in valid) / len(valid)
                               if valid else float("inf"))

        if current_school_cost < self._prev_school_cost:
            # School improving — contract toward barycenter
            bary_all = {
                k: sum(p.weights[k] for p in self.particles) / len(self.particles)
                for k in WEIGHT_PARAMS
            }
            for p in self.particles:
                for k in p.weights:
                    p.weights[k] -= 0.1 * (p.weights[k] - bary_all[k])
        else:
            # School stagnating — expand away from barycenter
            bary_all = {
                k: sum(p.weights[k] for p in self.particles) / len(self.particles)
                for k in WEIGHT_PARAMS
            }
            for p in self.particles:
                for k in p.weights:
                    p.weights[k] += 0.2 * (p.weights[k] - bary_all[k])

        self._prev_school_cost = current_school_cost
        for p in self.particles:
            p.weights = _normalize(p.weights)
            p.age += 1


class GB10QuadSwarmSampler(BaseSampler):
    """
    Optuna BaseSampler — GB10 QuadSwarm, no governors.

    Optuna search space: unbounded positive reals.
    Weights are normalized after sampling, not before.
    The swarm proposes any positive value — Optuna does not constrain it.

    No floor on adaptive weights — swarms that are not contributing
    go to zero. Dead swarms stay dead until hardware conditions change.

    Intelligence: reads signal_intelligence() to initialize particles
    at hardware-relevant scales each session.
    """

    # Elite seeds — relative weights, not bounded values
    # These represent informed priors, not governors
    ELITE_SEEDS = [
        {"w_psi": 5.0, "w_mem": 3.0, "w_tgpu": 1.0, "w_cx7": 1.0},
        {"w_psi": 3.0, "w_mem": 5.0, "w_tgpu": 1.0, "w_cx7": 1.0},
        {"w_psi": 1.0, "w_mem": 1.0, "w_tgpu": 1.0, "w_cx7": 1.0},
        {"w_psi": 4.0, "w_mem": 4.0, "w_tgpu": 1.0, "w_cx7": 1.0},
        {"w_psi": 2.0, "w_mem": 4.0, "w_tgpu": 2.0, "w_cx7": 2.0},
        {"w_psi": 1.0, "w_mem": 2.0, "w_tgpu": 4.0, "w_cx7": 3.0},
        {"w_psi": 1.0, "w_mem": 1.0, "w_tgpu": 5.0, "w_cx7": 4.0},
    ]

    def __init__(self):
        self.swarms = {
            "speed":   PSO_SpeedRouter(6),
            "energy":  Firefly_EnergyRouter(6),
            "memory":  AFSA_MemoryRouter(6),
            "thermal": FSO_ThermalRouter(6),
        }
        for s in self.swarms.values():
            s.initialize()

        self.global_best: Optional[RoutingParticle] = None
        self.generation = 0
        self.trial_count = 0
        # No floor on adaptive weights
        self.adaptive_weights = np.ones(4) / 4.0
        self._swarm_contributions = np.zeros(4)

    def sample_independent(self, study, trial, param_name, param_distribution):
        """
        Propose next weight value from QuadSwarm.
        Values are unbounded positive reals — Optuna receives raw proposals.
        Normalization happens in the objective, not here.
        """
        self.trial_count += 1

        # Elite seeding for first N trials — relative scale, not [0,1]
        if self.trial_count <= len(self.ELITE_SEEDS):
            seed = self.ELITE_SEEDS[self.trial_count - 1]
            return float(seed.get(param_name, 1.0))

        # Select swarm — zero-weight swarms are excluded entirely
        active = [(i, name, swarm)
                  for i, (name, swarm) in enumerate(self.swarms.items())
                  if self.adaptive_weights[i] > 0]

        if not active:
            # All swarms dead — reinitialize
            logger.warning("All swarms at zero — reinitializing")
            self.adaptive_weights = np.ones(4) / 4.0
            active = [(i, name, swarm) for i, (name, swarm) in enumerate(self.swarms.items())]

        active_indices = [a[0] for a in active]
        active_weights = self.adaptive_weights[active_indices]
        active_weights = active_weights / active_weights.sum()

        chosen_idx = np.random.choice(len(active), p=active_weights)
        _, _, swarm = active[chosen_idx]

        particle = swarm.particles[self.generation % len(swarm.particles)]
        # Return raw weight value — no clamp
        return float(abs(particle.weights.get(param_name, 1.0)))

    def after_trial(self, study, trial, state, values=None):
        """Update swarms after each trial. No clamping of cost values."""
        if state != TrialState.COMPLETE or not values:
            return

        # Composite cost — raw mean of objectives, no normalization
        cost = float(np.mean(values))
        weights = {k: trial.params.get(k, 1.0) for k in WEIGHT_PARAMS}
        weights = _normalize(weights)

        if self.global_best is None or cost < self.global_best.cost:
            self.global_best = RoutingParticle(
                weights=weights,
                velocity={k: 0.0 for k in weights},
                cost=cost,
                best_weights=weights.copy(),
                best_cost=cost,
                swarm_type="GLOBAL",
            )
            logger.info(f"NEW BEST: cost={cost:.6f} weights={weights}")

        # Update all swarms
        for i, (name, swarm) in enumerate(self.swarms.items()):
            prev_best = min((p.best_cost for p in swarm.particles),
                            default=float("inf"))
            for p in swarm.particles:
                if cost < p.best_cost:
                    p.best_cost = cost
                    p.best_weights = weights.copy()
                p.cost = min(p.cost, cost)
            swarm.update(self.global_best)
            new_best = min((p.best_cost for p in swarm.particles),
                           default=float("inf"))
            # Track contribution — did this swarm improve?
            if new_best < prev_best and np.isfinite(prev_best) and np.isfinite(new_best):
                self._swarm_contributions[i] += (prev_best - new_best)

        self._update_adaptive_weights(study)
        self.generation += 1

    def _update_adaptive_weights(self, study):
        """
        Adaptive weight update — no floor.
        Swarms that are not contributing go to zero.
        Softmax over contributions — swarms earn their allocation.
        """
        recent = [t for t in study.trials[-50:]
                  if t.state == TrialState.COMPLETE and t.values]
        if len(recent) < 5:
            return

        # Contribution-based allocation — no floor
        contribs = self._swarm_contributions.copy()
        if contribs.sum() > 0 and np.isfinite(contribs).all():
            # Softmax — preserves zero contributions as near-zero, not 0.05
            shifted = contribs - contribs.max()
            exp_c = np.exp(np.clip(shifted, -500, 500))
            total = exp_c.sum()
            if total > 0 and np.isfinite(total):
                self.adaptive_weights = exp_c / total
        # No minimum floor applied

    def infer_relative_search_space(self, study, trial):
        return {}

    def sample_relative(self, study, trial, search_space):
        return {}

    @property
    def intelligence(self) -> Dict:
        """What the swarm has learned — signal ranges, contributions, Pareto state."""
        return {
            "signal_intelligence":   signal_intelligence(),
            "swarm_contributions":   dict(zip(self.swarms.keys(),
                                              self._swarm_contributions.tolist())),
            "adaptive_weights":      dict(zip(self.swarms.keys(),
                                              self.adaptive_weights.tolist())),
            "global_best_cost":      self.global_best.cost if self.global_best else None,
            "global_best_weights":   self.global_best.weights if self.global_best else None,
            "generation":            self.generation,
            "trial_count":           self.trial_count,
        }


class GB10SwarmRouter:
    """
    High-level interface for GB10 QuadSwarm.
    Unbounded search. Hardware-calibrated. No governors.
    """

    def __init__(self, db_path: str = "gb10_swarm.db",
                 study_name: str = "gb10_uma"):
        self.db_path   = db_path
        self.study_name = study_name
        self.sampler   = GB10QuadSwarmSampler()

    def _objective(self, trial: optuna.Trial) -> Tuple[float, float, float, float]:
        """
        Four-objective function.
        Weights are unbounded positive reals — normalized inside objective.
        Signal values are unbounded — hardware defines the scale.
        No min() clamp on objectives. Raw hardware behavior feeds the Pareto front.
        """
        w_psi  = trial.suggest_float("w_psi",  0.0, 1e6)
        w_mem  = trial.suggest_float("w_mem",  0.0, 1e6)
        w_tgpu = trial.suggest_float("w_tgpu", 0.0, 1e6)
        w_cx7  = trial.suggest_float("w_cx7",  0.0, 1e6)

        total = w_psi + w_mem + w_tgpu + w_cx7
        if total <= 0:
            total = 1.0
        w_psi  /= total
        w_mem  /= total
        w_tgpu /= total
        w_cx7  /= total

        mem   = memory_snapshot()
        therm = thermal_snapshot()

        from gb10.fitness import _normalize_raw, _update_observed_max

        psi_raw = mem["psi_some_avg10"]
        _update_observed_max("psi", psi_raw)
        psi_val = _normalize_raw("psi", psi_raw)

        mem_val = 1.0 - min(
            mem["emv"]["mem_available_kb"] / max(mem["emv"]["mem_total_kb"], 1), 1.0
        )

        tgpu_raw = therm.get("TGPU")
        if tgpu_raw is not None:
            _update_observed_max("tgpu", tgpu_raw)
            tgpu_val = _normalize_raw("tgpu", tgpu_raw)
        else:
            tgpu_val = 0.0

        cx7_raw = therm.get("CX7")
        if cx7_raw is not None:
            _update_observed_max("cx7", cx7_raw)
            cx7_val = _normalize_raw("cx7", cx7_raw)
        else:
            cx7_val = 0.0

        # Raw objectives — no min() clamp
        obj_psi  = w_psi  * psi_val
        obj_mem  = w_mem  * mem_val
        obj_tgpu = w_tgpu * tgpu_val
        obj_cx7  = w_cx7  * cx7_val

        return obj_psi, obj_mem, obj_tgpu, obj_cx7

    def optimize(self, n_trials: int = 200,
                 show_progress: bool = True) -> optuna.Study:
        study = optuna.create_study(
            directions=["minimize", "minimize", "minimize", "minimize"],
            sampler=self.sampler,
            storage=f"sqlite:///{self.db_path}",
            study_name=self.study_name,
            load_if_exists=True,
        )
        study.optimize(
            self._objective,
            n_trials=n_trials,
            show_progress_bar=show_progress,
        )
        self.study = study
        return study

    @property
    def best_weights(self) -> Dict[str, float]:
        if self.sampler.global_best:
            return self.sampler.global_best.weights
        return {"w_psi": 0.4, "w_mem": 0.4, "w_tgpu": 0.1, "w_cx7": 0.1}

    @property
    def pareto_front(self):
        if not hasattr(self, "study"):
            return []
        return self.study.best_trials

    def dashboard(self):
        print(f"optuna-dashboard sqlite:///{self.db_path}")

    def intelligence(self) -> Dict:
        """Full swarm intelligence report — signal ranges, contributions, Pareto."""
        report = self.sampler.intelligence
        if hasattr(self, "study"):
            report["total_trials"] = len(self.study.trials)
            report["pareto_size"]  = len(self.study.best_trials)
        return report

    def stats(self) -> Dict:
        return self.intelligence()


if __name__ == "__main__":
    import json

    print("GB10 QuadSwarm — no governors, unbounded search")
    print("Four objectives: PSI | Memory | TGPU | CX7")
    print("Hardware defines the scale. Swarm learns from data.\n")

    router = GB10SwarmRouter(
        db_path="sandbox_gb10_swarm.db",
        study_name="gb10_sandbox_nogov"
    )

    study = router.optimize(n_trials=20, show_progress=True)

    print(f"\nPareto front: {len(router.pareto_front)} trials")
    print(f"Best weights: {json.dumps(router.best_weights, indent=2)}")
    print(f"\nIntelligence report:")
    print(json.dumps(router.intelligence(), indent=2))
    router.dashboard()
