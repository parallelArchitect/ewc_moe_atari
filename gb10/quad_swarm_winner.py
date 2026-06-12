#!/usr/bin/env python3
"""
QuadSwarm Sampler - SIMPLE WORKING VERSION
Uses your proven parameter preferences
"""

import optuna
from optuna.samplers import BaseSampler
from optuna.distributions import (
    BaseDistribution,
    CategoricalDistribution,
    IntDistribution,
)
from typing import Dict, Any, Optional
import numpy as np


class QuadSwarmSampler(BaseSampler):
    """
    Simple QuadSwarm that uses your PROVEN good configurations
    """

    def __init__(self, gpu_intel, population_size: int = 20):
        self.gpu_intel = gpu_intel
        self.population_size = population_size
        self.rng = np.random.RandomState(42)

        # Performance tracking
        self.best_performance = float("inf")
        self.good_configs = []
        self.trial_count = 0

        # SEED with your PROVEN best configurations
        self.seed_elite_configs()

    def seed_elite_configs(self):
        """Seed with your working configurations that got 8.4ms"""
        elite_configs = [
            {
                "BLOCK_X": 64,
                "BLOCK_Y": 8,
                "UNROLL": 2,
                "TILE_SIZE": 32,
                "performance": 8.4,
            },
            {
                "BLOCK_X": 64,
                "BLOCK_Y": 8,
                "UNROLL": 4,
                "TILE_SIZE": 32,
                "performance": 8.5,
            },
            {
                "BLOCK_X": 64,
                "BLOCK_Y": 8,
                "UNROLL": 1,
                "TILE_SIZE": 24,
                "performance": 8.6,
            },
            {
                "BLOCK_X": 32,
                "BLOCK_Y": 8,
                "UNROLL": 2,
                "TILE_SIZE": 24,
                "performance": 9.0,
            },
            {
                "BLOCK_X": 64,
                "BLOCK_Y": 4,
                "UNROLL": 1,
                "TILE_SIZE": 28,
                "performance": 8.1,
            },
        ]

        self.good_configs = elite_configs.copy()
        print(f"🎯 QuadSwarm: Seeded with {len(elite_configs)} PROVEN configurations")
        print("🚀 Target: Beat 8.4ms baseline!")

    def sample_independent(
        self,
        study: optuna.Study,
        trial: optuna.Trial,
        param_name: str,
        param_distribution: BaseDistribution,
    ) -> Any:
        """Sample using PROVEN good configurations"""

        self.trial_count += 1

        # Get ranges from GPU intelligence
        matrix_size = study.user_attrs.get("matrix_size", 2048)
        ranges = self.gpu_intel.get_optimal_ranges(matrix_size)

        # Use proven configs for first 20 trials
        if len(self.good_configs) > 0 and self.trial_count <= 20:
            # Cycle through proven configs
            config_idx = self.trial_count % len(self.good_configs)
            proven_config = self.good_configs[config_idx]

            if param_name in proven_config:
                value = proven_config[param_name]

                # Ensure value is valid for distribution
                if isinstance(param_distribution, CategoricalDistribution):
                    if value in param_distribution.choices:
                        return value
                    else:
                        return self.rng.choice(param_distribution.choices)
                elif isinstance(param_distribution, IntDistribution):
                    return max(
                        param_distribution.low, min(param_distribution.high, int(value))
                    )

        # Fallback sampling
        if param_name == "BLOCK_X":
            return self.rng.choice([8, 16, 24, 32, 48, 64])
        elif param_name == "BLOCK_Y":
            return self.rng.choice([4, 8, 12, 16])
        elif param_name == "TILE_SIZE":
            return self.rng.choice([20, 24, 28, 32, 36, 40])
        elif param_name == "UNROLL":
            return self.rng.choice([1, 2, 4, 8])
        else:
            if isinstance(param_distribution, CategoricalDistribution):
                return self.rng.choice(param_distribution.choices)
            elif isinstance(param_distribution, IntDistribution):
                return self.rng.randint(
                    param_distribution.low, param_distribution.high + 1
                )
            else:
                return 16

    def after_trial(
        self,
        study: optuna.Study,
        trial: optuna.Trial,
        state: optuna.trial.TrialState,
        values: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Track performance and update good configs"""

        if state == optuna.trial.TrialState.COMPLETE and trial.value is not None:
            # NEW BEST DETECTION
            if trial.value < self.best_performance:
                improvement = self.best_performance - trial.value
                self.best_performance = trial.value

                print(
                    f"🔥🎯 NEW BEST: {trial.value:.4f}ms (improved by {improvement:.4f}ms)"
                )
                print(f"🚀 Params: {trial.params}")

            # Store good configurations
            if trial.value <= 10.0:  # Only store decent results
                config = trial.params.copy()
                config["performance"] = trial.value
                self.good_configs.append(config)

                # Keep only best 20
                if len(self.good_configs) > 20:
                    self.good_configs.sort(
                        key=lambda x: x.get("performance", float("inf"))
                    )
                    self.good_configs = self.good_configs[:20]

    def infer_relative_search_space(
        self, study: optuna.Study, trial: optuna.Trial
    ) -> Dict[str, BaseDistribution]:
        return {}

    def sample_relative(
        self,
        study: optuna.Study,
        trial: optuna.Trial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        return {}
