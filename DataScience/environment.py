# environment.py
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List


class Environment:
    """
    Global world state.
    Rolled ONCE per script run.
    """

    def __init__(self, project_root=".", seed: int | None = None):
        self.root = Path(project_root)

        # 🎲 deterministic world
        import secrets
        #2484834550#
        self.seed = 2484834550#seed if seed is not None else secrets.randbits(32)
        self.rng = random.Random(self.seed)

        # 🔥 procedural stations
        self.stations = self._init_stations()

        # 🔥 full sound pool (100+)
        self.sound_pool = self._build_sound_pool()

        # 🔥 initialize ALL sound qualities
        self.sound_quality: Dict[str, float] = {
            sound: self.rng.uniform(-1, 1)
            for sound in self.sound_pool
        }

        # 🐗 pre-generated boars
        self.boar_spawn_plan = self._spawn_boars()

    # -----------------------------
    # Stations (NO CSV)
    # -----------------------------

    def _init_stations(self, n: int = 12):
        # 🔥 multiply stations by 10
        n = n * 10
        return [f"S{i:03d}" for i in range(1, n + 1)]

    def get_station_ids(self):
        return self.stations

    def get_sound_for_station(self, station_id):
        # 🔥 always random sound from full pool
        return self.rng.choice(self.sound_pool)

    # -----------------------------
    # Sound pool
    # -----------------------------

    def _build_sound_pool(self):
        return [
            # --- HUMAN ---
            "human_voice","human_scream","human_shout","human_whistle","human_clap",
            "human_run_noise","human_footsteps","human_laughter","human_argument",
            "human_group_noise","human_crowd","human_yell_close","human_yell_distant",
            "human_panic","human_warning_call",

            # --- DOG ---
            "dog_bark","dog_pack","dog_growl","dog_attack","dog_chase","dog_warning",
            "dog_snarl","dog_multiple","dog_territorial","dog_howl","dog_aggressive",
            "dog_fast_bark",

            # --- GUN ---
            "gunshot","gun_burst","gun_rifle","gun_single","gun_double",
            "gun_rapid_fire","gun_echo","gun_distant","gun_close","gun_heavy",
            "gun_pistol","gun_shock",

            # --- WOLF ---
            "wolf_howl","wolf_pack","wolf_close","wolf_distant","wolf_chase",
            "wolf_snarl","wolf_attack","wolf_warning","wolf_multiple","wolf_echo",
            "wolf_night_call",

            # --- BOOM ---
            "boom_explosion","boom_blast","boom_shockwave","boom_close","boom_distant",
            "boom_heavy","boom_noise","boom_double","boom_crack","boom_impact",
            "boom_bass","boom_metal_hit",

            # --- OTHER ---
            "forest_noise","wind_noise","rain_light","rain_heavy","storm_wind","thunder",
            "branch_snap","tree_fall","leaves_rustle","water_stream","river_flow",
            "insect_swarm","bird_flock","owl_call","crow_caw","metal_bang",
            "metal_clank","metal_scrape","wood_knock","wood_crack","rock_fall",
            "gravel_step","sand_shift","mud_step","engine_idle","engine_rev",
            "vehicle_pass","vehicle_stop","alarm_beep","alarm_loop","siren_far",
            "siren_close","electrical_buzz","static_noise","radio_noise",
            "distortion_wave","low_frequency_hum","high_pitch_ring","pressure_wave",
            "unknown_signal","random_noise","ambient_loop","strange_echo",
            "deep_resonance","click_pattern","pulse_signal","rhythmic_noise",
            "chaotic_noise","background_static","mechanical_loop","industrial_noise",
            "construction_noise","distant_activity","movement_noise",
            "unidentified_sound",
        ]

    def get_sound_quality(self, sound: str):
        return self.sound_quality[sound]

    # -----------------------------
    # Boar spawn
    # -----------------------------

    def _spawn_boars(self):
        total = int(self.rng.triangular(6, 18, 11))*10

        solo_count = self.rng.randint(0, min(3, total // 4))
        remaining = total - solo_count

        groups = []
        gid = 1

        while remaining > 0:
            size = min(self.rng.choice([3, 4, 5, 6, 7]), remaining)

            groups.append(self._make_group(gid, "family", size))
            gid += 1
            remaining -= size

        for _ in range(solo_count):
            groups.append(self._make_group(gid, "solo", 1))
            gid += 1

        return groups

    def _make_group(self, gid, kind, size):
        station_ids = self.get_station_ids()

        # 🔥 ALL stations accessible (no restriction)
        assigned = station_ids

        base = {
            "human": 0.7,
            "dog": 0.5,
            "gun": 0.6,
            "wolf": 0.6,
            "boom": 0.55,
            "other": 0.4,
        }

        if kind == "solo":
            base = {k: min(0.98, v + 0.15) for k, v in base.items()}

        return {
            "group_id": f"G{gid:03d}",
            "kind": kind,
            "size": size,
            "stations": assigned,
            "base_thresholds": base,
        }