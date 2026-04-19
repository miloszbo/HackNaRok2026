from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, List

from communication import Strategy


MAX_DETER_TIME = 999


class CombinatorialRottingThompsonStrategy(Strategy):
    """
    Full Combinatorial Rotting Thompson Bandit

    - per (station, sound) stats
    - global sound + category stats
    - exponential decay
    - Thompson sampling
    - exploration bonus
    - fully compatible with communication.py
    """

    def __init__(
        self,
        env,
        decay_rate: float = 0.97,
        exploration_strength: float = 0.05,
    ):
        super().__init__(name="rotting_thompson")

        self.env = env
        self.sound_pool = env.sound_pool
        self.rng = random.Random()

        self.decay_rate = decay_rate
        self.exploration_strength = exploration_strength

        self.t = 0  # global time

        # (station, sound) -> [success, failure, last_update]
        self.stats = defaultdict(lambda: [1.0, 1.0, 0])

        # global stats
        self.sound_stats = defaultdict(lambda: [1.0, 1.0, 0])
        self.category_stats = defaultdict(lambda: [1.0, 1.0, 0])

    # -----------------------------------
    # CHOOSE
    # -----------------------------------

    def choose(self, stations: List[str], history=None) -> Dict[str, str]:
        self.t += 1

        config = {}

        for station in stations:
            best_sound = None
            best_score = -1e9

            for sound in self.sound_pool:
                category = self._categorize(sound)

                # decayed stats
                s_pair, f_pair = self._get_decayed(self.stats[(station, sound)])
                s_sound, f_sound = self._get_decayed(self.sound_stats[sound])
                s_cat, f_cat = self._get_decayed(self.category_stats[category])

                # Thompson samples
                sample_pair = self.rng.betavariate(s_pair, f_pair)
                sample_sound = self.rng.betavariate(s_sound, f_sound)
                sample_cat = self.rng.betavariate(s_cat, f_cat)

                # combine
                score = (
                    0.6 * sample_pair +
                    0.25 * sample_sound +
                    0.15 * sample_cat
                )

                # exploration bonus
                count = self.stats[(station, sound)][0] + self.stats[(station, sound)][1]

                exploration = self.exploration_strength * math.sqrt(
                    math.log(self.t + 2) / (count + 1)
                )

                score += exploration

                if score > best_score:
                    best_score = score
                    best_sound = sound

            config[station] = best_sound

        return config

    # -----------------------------------
    # UPDATE
    # -----------------------------------

    def update(self, results: List[Dict], history=None):
        for r in results:
            station = r["StationID"]
            sound = r["Sound"]
            category = self._categorize(sound)

            success = 1 if r["DeterTime"] != MAX_DETER_TIME else 0

            # update all levels
            self._update_entry(self.stats, (station, sound), success)
            self._update_entry(self.sound_stats, sound, success)
            self._update_entry(self.category_stats, category, success)

    # -----------------------------------
    # DECAY LOGIC
    # -----------------------------------

    def _get_decayed(self, entry):
        s, f, last_t = entry

        dt = self.t - last_t
        decay = self.decay_rate ** dt

        s = 1 + s * decay
        f = 1 + f * decay

        # clamp
        s = max(1e-3, s)
        f = max(1e-3, f)

        return s, f

    def _update_entry(self, table, key, success):
        s, f, last_t = table[key]

        dt = self.t - last_t
        decay = self.decay_rate ** dt

        s *= decay
        f *= decay

        if success:
            s += 1
        else:
            f += 1

        table[key] = [s, f, self.t]

    # -----------------------------------
    # HELPERS
    # -----------------------------------

    def _categorize(self, sound: str) -> str:
        s = sound.lower()
        if "human" in s:
            return "human"
        if "dog" in s:
            return "dog"
        if "gun" in s:
            return "gun"
        if "wolf" in s:
            return "wolf"
        if "boom" in s:
            return "boom"
        return "other"