from environment import Environment
from communication import ExperimentRunner

import time
from pathlib import Path

# strategies
from algorithms.static import StaticStrategy
from algorithms.random import RandomStrategy
from algorithms.rotting_thompson import CombinatorialRottingThompsonStrategy
from algorithms.combinatorial_thompson_xgb import CombinatorialThompsonXGBStrategy


# -----------------------------------
# CONFIG
# -----------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

REFRESH = 0.0
MAX_WEEKS = 20

MODE = "benchmark"  # "train" or "benchmark"


# -----------------------------------
# BUILD STRATEGIES
# -----------------------------------

def build_strategies(env):
    strategies = []

    # --- BASELINES ---
    strategies.append(StaticStrategy(env))
    strategies.append(RandomStrategy(env))

    # --- ROTTING BANDIT ---
    rotting = CombinatorialRottingThompsonStrategy(env)

    if MODE == "benchmark":
        def no_update(*args, **kwargs):
            pass
        rotting.update = no_update

    strategies.append(rotting)

    # --- XGB MODEL ---
    xgb = CombinatorialThompsonXGBStrategy(env)

    if MODE == "benchmark":
        def no_update(*args, **kwargs):
            pass
        xgb.update = no_update

    strategies.append(xgb)

    return strategies


# -----------------------------------
# MAIN
# -----------------------------------

def main():
    print("🚀 Starting run")
    print(f"MODE: {MODE}")

    # 🔥 ONE ENV = SAME SEED FOR ALL STRATEGIES
    env = Environment(project_root=PROJECT_ROOT)
    print(f"Seed: {env.seed}")

    strategies = build_strategies(env)

    print("Strategies:")
    for s in strategies:
        print(f" - {s.name}")

    runner = ExperimentRunner(env, strategies)

    for week in range(1, MAX_WEEKS + 1):
        print(f"--- Week {week} ---")
        runner.step()

        if REFRESH > 0:
            time.sleep(REFRESH)

    print("✅ Done")


# -----------------------------------

if __name__ == "__main__":
    main()