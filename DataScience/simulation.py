#simulation.py
from __future__ import annotations
import csv
import copy
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from environment import Environment

MAX_DETER_TIME = 999

@dataclass
class BoarGroup:
    group_id: str
    kind: str
    size: int
    stations: List[str]

    base_thresholds: Dict[str, float]
    memory: Dict[str, int] = field(default_factory=dict)

    def learning_rate(self) -> float:
        if self.kind == "solo":
            return 0.02
        return 0.04 + self.size * 0.015

    def rolls(self) -> int:
        if self.kind == "solo":
            return 1
        return min(4, 1 + self.size // 3)

class SimulationInstance:
    """
    One independent simulation run (per algorithm).
    """

    def __init__(self, env: Environment, name: str):
        self.env = env
        self.name = name

        self.root = Path(env.root) / "experiments" / name
        self.root.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.root / "encounters.csv"

        self.groups = self._init_groups()
        self.week = 0

        self._ensure_csv()

    def _init_groups(self):
        groups = []
        for g in copy.deepcopy(self.env.boar_spawn_plan):
            groups.append(
                BoarGroup(
                    group_id=g["group_id"],
                    kind=g["kind"],
                    size=g["size"],
                    stations=g["stations"],
                    base_thresholds=g["base_thresholds"],
                )
            )
        return groups

    def _ensure_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Date", "StationID", "Sound", "DeterTime"])

    def step(self, station_config: Dict[str, str], forced_visits: int | None = None):
        self.week += 1

        today = datetime.now().strftime("%Y-%m-%d")
        rows = []

        total_visits = forced_visits if forced_visits is not None else self._roll_global_visits()

        for _ in range(total_visits):
            group = self.env.rng.choice(self.groups)
            station_id = self.env.rng.choice(group.stations)

            sound = station_config.get(
                station_id,
                self.env.get_sound_for_station(station_id)
            )

            deter_time = self._encounter(group, sound)

            rows.append({
                "Date": today,
                "StationID": station_id,
                "Sound": sound,
                "DeterTime": deter_time,
            })

        self._append(rows)
        return rows, self.get_history()

    def _roll_global_visits(self) -> int:
        base = self.env.rng.choices(
            population=[4, 5, 6, 7, 8],
            weights=[1, 3, 5, 3, 1],
            k=1
        )[0]

        if self.env.rng.random() < 0.08:
            base += self.env.rng.choice([-2, -1, 1, 2])

        return max(1, base * 10)

    def _encounter(self, group: BoarGroup, sound: str):
        sound_type = self._categorize(sound)

        memory = group.memory.get(sound, 0)

        base = float(group.base_thresholds.get(sound_type, 0.55))
        raw_quality = float(self.env.get_sound_quality(sound))

        # 🔥 strong quality curve
        q = math.tanh(raw_quality / 2.0)

        # 🔥 high natural starting deterrence
        initial_effective = base + 0.85 * q
        initial_effective = max(0.05, min(0.99, initial_effective))

        # 🔥 strong exponential learning
        base_decay = 0.055
        learning_multiplier = 1.2 + 6.0 * group.learning_rate()

        decay_scale = 1.0 + 4.0 * max(0.0, q)
        penalty_scale = 1.0 + 2.5 * max(0.0, -q)

        k = base_decay * learning_multiplier * penalty_scale / decay_scale
        k = max(0.01, min(0.25, k))

        # 🔥 accelerated exponential decay
        effective = initial_effective * math.exp(-k * (memory ** 1.25))
        effective = max(0.01, min(0.99, effective))

        # 🎲 group resistance (sharper)
        roll = max(self.env.rng.random() for _ in range(group.rolls()))

        if roll < effective:
            deter_time = self._roll_deter_time(sound_type, group, q)
        else:
            deter_time = MAX_DETER_TIME

        group.memory[sound] = memory + 1
        return deter_time

    def _roll_deter_time(self, sound_type: str, group: BoarGroup, q: float):
        base_ranges = {
            "gun": (2, 8),
            "boom": (3, 11),
            "dog": (4, 14),
            "wolf": (5, 16),
            "human": (6, 18),
            "other": (7, 20),
        }

        low, high = base_ranges[sound_type]

        if group.kind == "solo":
            low += 2
            high += 3

        if group.kind != "solo":
            high += min(6, group.size)

        # 🔥 strong quality effect
        low = max(1, int(round(low - 5.0 * q)))
        high = max(low, int(round(high - 8.0 * q)))

        return self.env.rng.randint(low, high)

    def get_history(self):
        if not self.csv_path.exists():
            return []

        import csv

        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    
    def _append(self, rows):
        if self.csv_path.exists():
            with open(self.csv_path, "rb+") as f:
                f.seek(0, 2)
                if f.tell() > 0:
                    f.seek(-1, 2)
                    if f.read(1) != b"\n":
                        f.write(b"\n")

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(
                [[r["Date"], r["StationID"], r["Sound"], r["DeterTime"]] for r in rows]
            )

    def _categorize(self, s):
        s = s.lower()
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