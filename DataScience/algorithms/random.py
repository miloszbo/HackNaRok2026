from __future__ import annotations

import random
from typing import Dict, List

from communication import Strategy


class RandomStrategy(Strategy):
    def __init__(self, env):
        super().__init__(name="random")
        self.rng = random.Random()
        self.sound_pool = env.sound_pool

    def choose(self, stations: List[str], history=None) -> Dict[str, str]:
        config = {}

        for station in stations:
            config[station] = self.rng.choice(self.sound_pool)

        return config

    def update(self, results, history=None):
        pass