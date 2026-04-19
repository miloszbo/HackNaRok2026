from __future__ import annotations

from typing import Dict, List

from communication import Strategy


class StaticStrategy(Strategy):
    def __init__(self, env):
        super().__init__(name="static")

        # 🔥 pick one fixed sound at start
        self.fixed_sound = env.sound_pool[0]

    def choose(self, stations: List[str], history=None) -> Dict[str, str]:
        return {
            station: self.fixed_sound
            for station in stations
        }

    def update(self, results, history=None):
        pass