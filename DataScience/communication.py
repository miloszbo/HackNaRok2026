from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Tuple

from simulation import SimulationInstance
from environment import Environment


# -----------------------------
# Strategy Base (UPDATED)
# -----------------------------

class Strategy:
    def __init__(self, name: str):
        self.name = name

    # 🔥 now receives history
    def choose(self, stations: List[str], history: List[Dict] | None = None) -> Dict[str, str]:
        raise NotImplementedError

    def update(self, results: List[Dict], history: List[Dict] | None = None):
        pass


# -----------------------------
# Experiment Runner
# -----------------------------

class ExperimentRunner:
    def __init__(self, env: Environment, strategies: List[Strategy]):
        self.env = env
        self.strategies = strategies

        self.week = 0

        # 🔥 one simulation per strategy (isolated learning)
        self.simulations: Dict[str, SimulationInstance] = {
            s.name: SimulationInstance(env, s.name)
            for s in strategies
        }

        self.results_file = Path(env.root) / "experiments" / "summary.csv"
        self._ensure_results_file()

    def _ensure_results_file(self):
        if not self.results_file.exists():
            self.results_file.parent.mkdir(parents=True, exist_ok=True)

            with open(self.results_file, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Week",
                    "Algorithm",
                    "SuccessRate",
                    "Encounters",
                    "GlobalEncounters"
                ])

    # -----------------------------
    # MAIN STEP (FIXED)
    # -----------------------------

    def step(self):
        self.week += 1

        # 🔥 shared visit count (fair comparison)
        any_sim = next(iter(self.simulations.values()))
        global_visits = any_sim._roll_global_visits()

        for strategy in self.strategies:
            sim = self.simulations[strategy.name]

            station_ids = self.env.get_station_ids()

            # 🔥 history BEFORE decision
            history = sim.get_history() or []

            # 🔥 choose using full context
            config = strategy.choose(station_ids, history=history)

            # 🔥 run simulation
            results, history = sim.step(config, forced_visits=global_visits)

            # 🔥 update with full history
            strategy.update(results, history=history)

            success_rate, count = self._compute_metrics(results)

            self._log(strategy.name, success_rate, count, global_visits)

    # -----------------------------
    # Metrics
    # -----------------------------

    def _compute_metrics(self, results: List[Dict]) -> Tuple[float, int]:
        if not results:
            return 0.0, 0

        total = len(results)
        success = sum(1 for r in results if r["DeterTime"] != 999)

        return success / total, total

    def _log(self, algo_name: str, success_rate: float, count: int, global_visits: int):
        with open(self.results_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.week,
                algo_name,
                round(success_rate, 4),
                count,
                global_visits
            ])