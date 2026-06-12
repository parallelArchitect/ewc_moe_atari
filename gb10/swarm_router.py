"""
gb10/swarm_router.py - GB10 QuadSwarm with Optuna multi-objective optimization

Four objectives optimized simultaneously on live GB10 hardware signals:
  1. Minimize PSI memory stall pressure
  2. Minimize LPDDR5X memory consumption
  3. Minimize GPU die temperature contribution
  4. Minimize CX7 NIC thermal contribution

The QuadSwarm (PSO + Firefly + AFSA + FSO) is the Optuna sampler.
Optuna manages the study, stores trials to SQLite, builds the Pareto front.
The hardware measures. The swarm learns. The knowledge persists across runs.

No hardcoded limits. No governors. No Docker. No vLLM.
Direct hardware signals. Full GB10 silicon.

Architecture:
  GB10QuadSwarmSampler (BaseSampler)
    ├── PSO_SpeedRouter      — minimize swap latency
    ├── Firefly_EnergyRouter — minimize LPDDR5X pressure
    ├── AFSA_MemoryRouter    — minimize MemAvailable consumption
    └── FSO_ThermalRouter    — minimize TGPU + CX7 contribution

Usage:
    router = GB10SwarmRouter(db_path="gb10_swarm.db")
    router.optimize(n_trials=200)
    print(router.best_weights)
    router.dashboard()  # launch optuna-dashboard

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

from gb10.fitness import swap_cost_snapshot
from gb10.memory import memory_snapshot
from gb10.thermal import thermal_snapshot

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass
class RoutingParticle:
    weights: Dict[str, float]
    velocity: Dict[str, float]
    cost: float
    best_weights: Dict[str, float]
    best_cost: float
    swarm_type: str
    age: int = 0


WEIGHT_PARAMS = ["w_psi", "w_mem", "w_tgpu", "w_cx7"]


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(abs(v) for v in weights.values())
    if total <= 0:
        return {k: 0.25 for k in weights}
    return {k: abs(v) / total for k, v in weights.items()}


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


class PSO_SpeedRouter:
    """PSO — minimizes swap latency. Bias toward PSI + memory signals."""
    def __init__(self, size=6):
        self.particles: List[RoutingParticle] = []
        self.size = size
        self.swarm_type = "PSO_SPEED"

    def initialize(self):
        for _ in range(self.size):
            w = _normalize({"w_psi": random.uniform(0.3,0.6), "w_mem": random.uniform(0.2,0.5),
                             "w_tgpu": random.uniform(0.05,0.2), "w_cx7": random.uniform(0.05,0.2)})
            self.particles.append(RoutingParticle(
                weights=w, velocity={k: 0.0 for k in w},
                cost=float("inf"), best_weights=w.copy(), best_cost=float("inf"),
                swarm_type=self.swarm_type))

    def update(self, global_best: Optional[RoutingParticle]):
        w_inertia, c1, c2 = 0.7, 2.0, 2.0
        for p in self.particles:
            for k in p.weights:
                r1, r2 = random.random(), random.random()
                cognitive = c1 * r1 * (p.best_weights[k] - p.weights[k])
                social = c2 * r2 * (global_best.weights[k] - p.weights[k]) if global_best else 0.0
                p.velocity[k] = w_inertia * p.velocity[k] + cognitive + social
                p.weights[k] = _clamp(p.weights[k] + p.velocity[k])
            p.weights = _normalize(p.weights)
            p.age += 1


class Firefly_EnergyRouter:
    """Firefly — minimizes LPDDR5X pressure. Particles attracted to lower cost."""
    def __init__(self, size=6):
        self.particles: List[RoutingParticle] = []
        self.size = size
        self.swarm_type = "FIREFLY_ENERGY"

    def initialize(self):
        for _ in range(self.size):
            w = _normalize({"w_psi": random.uniform(0.2,0.4), "w_mem": random.uniform(0.3,0.6),
                             "w_tgpu": random.uniform(0.05,0.15), "w_cx7": random.uniform(0.05,0.15)})
            self.particles.append(RoutingParticle(
                weights=w, velocity={k: 0.0 for k in w},
                cost=float("inf"), best_weights=w.copy(), best_cost=float("inf"),
                swarm_type=self.swarm_type))

    def update(self, global_best: Optional[RoutingParticle]):
        alpha, beta0, gamma = 0.25, 1.0, 0.01
        for p in self.particles:
            for other in self.particles:
                if other.cost < p.cost:
                    dist = sum((p.weights[k] - other.weights[k])**2 for k in p.weights)**0.5
                    beta = beta0 * np.exp(-gamma * dist)
                    for k in p.weights:
                        p.weights[k] = _clamp(p.weights[k] + beta*(other.weights[k]-p.weights[k])
                                               + alpha*(random.random()-0.5))
            p.weights = _normalize(p.weights)
            p.age += 1


class AFSA_MemoryRouter:
    """AFSA — minimizes MemAvailable consumption per load."""
    def __init__(self, size=6):
        self.particles: List[RoutingParticle] = []
        self.size = size
        self.swarm_type = "AFSA_MEMORY"

    def initialize(self):
        for _ in range(self.size):
            w = _normalize({"w_psi": random.uniform(0.15,0.35), "w_mem": random.uniform(0.4,0.65),
                             "w_tgpu": random.uniform(0.05,0.2), "w_cx7": random.uniform(0.05,0.15)})
            self.particles.append(RoutingParticle(
                weights=w, velocity={k: 0.0 for k in w},
                cost=float("inf"), best_weights=w.copy(), best_cost=float("inf"),
                swarm_type=self.swarm_type))

    def update(self, global_best: Optional[RoutingParticle]):
        step = 0.05
        for p in self.particles:
            best_neighbor = min((o for o in self.particles if o is not p),
                                key=lambda o: o.cost, default=None)
            if best_neighbor and best_neighbor.cost < p.cost:
                for k in p.weights:
                    d = best_neighbor.weights[k] - p.weights[k]
                    p.weights[k] = _clamp(p.weights[k] + step * random.random() * np.sign(d))
            else:
                for k in p.weights:
                    p.weights[k] = _clamp(p.weights[k] + step * (random.random() - 0.5))
            p.weights = _normalize(p.weights)
            p.age += 1


class FSO_ThermalRouter:
    """FSO — keeps TGPU and CX7 below observed thresholds. Anti-congestion via volitive movement."""
    def __init__(self, size=6):
        self.particles: List[RoutingParticle] = []
        self.size = size
        self.swarm_type = "FSO_THERMAL"
        self._prev_school_weight = 1.0

    def initialize(self):
        for _ in range(self.size):
            w = _normalize({"w_psi": random.uniform(0.1,0.3), "w_mem": random.uniform(0.1,0.3),
                             "w_tgpu": random.uniform(0.2,0.5), "w_cx7": random.uniform(0.15,0.4)})
            p = RoutingParticle(weights=w, velocity={k: 0.0 for k in w},
                                cost=float("inf"), best_weights=w.copy(), best_cost=float("inf"),
                                swarm_type=self.swarm_type)
            p.weight = 1.0
            self.particles.append(p)

    def update(self, global_best: Optional[RoutingParticle]):
        valid = [p for p in self.particles if p.cost < float("inf")]
        if valid:
            total_w = sum(getattr(p, "weight", 1.0) for p in valid)
            if total_w > 0:
                bary = {k: sum(p.weights[k]*getattr(p,"weight",1.0) for p in valid)/total_w
                        for k in WEIGHT_PARAMS}
                for p in self.particles:
                    for k in p.weights:
                        p.weights[k] = _clamp(p.weights[k] + 0.1*(bary[k]-p.weights[k]))
        school_w = sum(getattr(p,"weight",1.0) for p in self.particles)
        if school_w < self._prev_school_weight:
            bary_all = {k: sum(p.weights[k] for p in self.particles)/len(self.particles)
                        for k in WEIGHT_PARAMS}
            for p in self.particles:
                for k in p.weights:
                    p.weights[k] = _clamp(p.weights[k] + 0.2*(p.weights[k]-bary_all[k]))
        self._prev_school_weight = school_w
        for p in self.particles:
            p.weights = _normalize(p.weights)
            p.age += 1


class GB10QuadSwarmSampler(BaseSampler):
    """
    Optuna BaseSampler — GB10 QuadSwarm for multi-objective weight optimization.

    Four swarms, four objectives, one Pareto front.
    The swarm proposes. Optuna scores and stores. Hardware measures.
    Knowledge persists to SQLite across training runs.
    """

    ELITE_SEEDS = [
        {"w_psi": 0.50, "w_mem": 0.30, "w_tgpu": 0.10, "w_cx7": 0.10},
        {"w_psi": 0.30, "w_mem": 0.50, "w_tgpu": 0.10, "w_cx7": 0.10},
        {"w_psi": 0.25, "w_mem": 0.25, "w_tgpu": 0.25, "w_cx7": 0.25},
        {"w_psi": 0.40, "w_mem": 0.40, "w_tgpu": 0.10, "w_cx7": 0.10},
        {"w_psi": 0.20, "w_mem": 0.40, "w_tgpu": 0.20, "w_cx7": 0.20},
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
        self.adaptive_weights = np.ones(4) / 4.0

    def sample_independent(self, study, trial, param_name, param_distribution):
        """Propose next weight value from QuadSwarm."""
        self.trial_count += 1

        # Elite seeding for first N trials
        if self.trial_count <= len(self.ELITE_SEEDS):
            seed = self.ELITE_SEEDS[self.trial_count - 1]
            normalized = _normalize(seed)
            return float(normalized.get(param_name, 0.25))

        # Sample from adaptive swarm selection
        swarm_names = list(self.swarms.keys())
        idx = np.random.choice(len(swarm_names), p=self.adaptive_weights)
        swarm = self.swarms[swarm_names[idx]]
        particle = swarm.particles[self.generation % len(swarm.particles)]
        return float(_clamp(particle.weights.get(param_name, 0.25)))

    def after_trial(self, study, trial, state, values=None):
        """Update swarms after each trial evaluation."""
        if state != TrialState.COMPLETE or not values:
            return

        # Composite cost from multi-objective values (v4.9: values param, not trial.values)
        cost = float(np.mean(values))
        weights = {k: trial.params.get(k, 0.25) for k in WEIGHT_PARAMS}
        weights = _normalize(weights)

        if self.global_best is None or cost < self.global_best.cost:
            self.global_best = RoutingParticle(
                weights=weights, velocity={k: 0.0 for k in weights},
                cost=cost, best_weights=weights.copy(), best_cost=cost,
                swarm_type="GLOBAL")
            logger.info(f"NEW BEST: cost={cost:.4f} weights={weights}")

        for swarm in self.swarms.values():
            for p in swarm.particles:
                if cost < p.best_cost:
                    p.best_cost = cost
                    p.best_weights = weights.copy()
                p.cost = min(p.cost, cost)
            swarm.update(self.global_best)

        self._update_adaptive_weights(study)
        self.generation += 1

    def _update_adaptive_weights(self, study):
        recent = [t for t in study.trials[-20:]
                  if t.state == TrialState.COMPLETE and t.values]
        if len(recent) < 5:
            return
        improvements = np.zeros(4)
        for i, (name, swarm) in enumerate(self.swarms.items()):
            if swarm.particles:
                best = min(p.best_cost for p in swarm.particles)
                improvements[i] = max(0.0, 1.0 - best)
        if improvements.sum() > 0:
            exp_i = np.exp(improvements)
            self.adaptive_weights = exp_i / exp_i.sum()
        self.adaptive_weights = np.maximum(self.adaptive_weights, 0.05)
        self.adaptive_weights /= self.adaptive_weights.sum()

    def infer_relative_search_space(self, study, trial):
        return {}

    def sample_relative(self, study, trial, search_space):
        return {}


class GB10SwarmRouter:
    """
    High-level interface for GB10 QuadSwarm multi-objective optimization.

    The objective reads four independent hardware signals and returns
    four separate costs — one per objective. Optuna builds the Pareto front.
    """

    def __init__(self, db_path: str = "gb10_swarm.db", study_name: str = "gb10_uma"):
        self.db_path = db_path
        self.study_name = study_name
        self.sampler = GB10QuadSwarmSampler()

    def _objective(self, trial: optuna.Trial) -> Tuple[float, float, float, float]:
        """
        Four-objective function. Each returns a normalized cost in [0, 1].
        Lower is better for all four.
        """
        w_psi  = trial.suggest_float("w_psi",  0.0, 1.0)
        w_mem  = trial.suggest_float("w_mem",  0.0, 1.0)
        w_tgpu = trial.suggest_float("w_tgpu", 0.0, 1.0)
        w_cx7  = trial.suggest_float("w_cx7",  0.0, 1.0)

        # Normalize weights before evaluation
        total = w_psi + w_mem + w_tgpu + w_cx7
        if total <= 0:
            total = 1.0
        w_psi, w_mem, w_tgpu, w_cx7 = (w_psi/total, w_mem/total,
                                         w_tgpu/total, w_cx7/total)

        # Read live hardware
        mem   = memory_snapshot()
        therm = thermal_snapshot()

        psi_val  = mem["psi_some_avg10"] / 100.0
        mem_val  = 1.0 - min(mem["emv"]["mem_available_kb"] / max(mem["emv"]["mem_total_kb"], 1), 1.0)
        tgpu_val = (therm["TGPU"] / 90.0) if therm["TGPU"] is not None else 0.0
        cx7_val  = (therm["CX7"]  / 59.0) if therm["CX7"]  is not None else 0.0

        # Four independent objectives
        obj_psi  = w_psi  * min(psi_val,  1.0)
        obj_mem  = w_mem  * min(mem_val,  1.0)
        obj_tgpu = w_tgpu * min(tgpu_val, 1.0)
        obj_cx7  = w_cx7  * min(cx7_val,  1.0)

        return obj_psi, obj_mem, obj_tgpu, obj_cx7

    def optimize(self, n_trials: int = 200, show_progress: bool = True):
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
        print(f"Launch dashboard with:")
        print(f"  optuna-dashboard sqlite:///{self.db_path}")

    def stats(self) -> Dict:
        s = {
            "best_weights":    self.best_weights,
            "sampler_gen":     self.sampler.generation,
            "trial_count":     self.sampler.trial_count,
            "adaptive_weights": dict(zip(self.sampler.swarms.keys(),
                                         self.sampler.adaptive_weights.tolist())),
        }
        if hasattr(self, "study"):
            s["total_trials"]  = len(self.study.trials)
            s["pareto_size"]   = len(self.study.best_trials)
        return s


if __name__ == "__main__":
    import json

    print("GB10 QuadSwarm — multi-objective Optuna optimization")
    print("Four objectives: PSI | Memory | TGPU | CX7")
    print("TGPU and CX7 will be None on non-GB10 hardware — expected\n")

    router = GB10SwarmRouter(
        db_path="sandbox_gb10_swarm.db",
        study_name="gb10_sandbox_test"
    )

    study = router.optimize(n_trials=20, show_progress=True)

    print(f"\nPareto front: {len(router.pareto_front)} trials")
    print(f"Best weights: {json.dumps(router.best_weights, indent=2)}")
    print(f"\nStats: {json.dumps(router.stats(), indent=2)}")
    print(f"\n{router.dashboard.__doc__ or ''}")
    router.dashboard()
