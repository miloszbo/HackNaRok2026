from __future__ import annotations

import pickle
import random
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from xgboost import XGBRegressor

from communication import Strategy


MAX_DETER_TIME = 999


class CombinatorialThompsonXGBStrategy(Strategy):
    """
    Contextual combinatorial bandit:
    - considers ALL sounds for EVERY station
    - learns reward with XGBoost
    - injects Thompson-style uncertainty using:
        1) bootstrapped XGBoost ensemble variance
        2) Beta posteriors on station-sound, sound-wide, and category-wide success

    Input history format:
        Date, StationID, Sound, DeterTime

    Output:
        Dict[station_id] -> best sound for next week
    """

    def __init__(
        self,
        env,
        history_path: Optional[str | Path] = None,
        seed: Optional[int] = None,
        retrain_every: int = 100,
        min_history_to_train: int = 200,
        n_bootstrap_models: int = 5,
        hash_dim: int = 1024,
        time_cap: int = 30,
        benchmark_mode: bool = False,
    ):
        super().__init__(name="cts_xgb")

        self.env = env
        self.sound_pool = list(env.sound_pool)
        self.rng = random.Random(seed if seed is not None else getattr(env, "seed", None))

        self.retrain_every = retrain_every
        self.min_history_to_train = min_history_to_train
        self.n_bootstrap_models = n_bootstrap_models
        self.hash_dim = hash_dim
        self.time_cap = time_cap
        self.benchmark_mode = benchmark_mode

        self.history_rows: List[Dict] = []
        self.seen_keys = set()

        self.beta_pair = defaultdict(lambda: [1.0, 1.0])
        self.beta_sound = defaultdict(lambda: [1.0, 1.0])
        self.beta_category = defaultdict(lambda: [1.0, 1.0])

        self.live_stats = self._make_empty_stats()

        self.hasher = FeatureHasher(
            n_features=self.hash_dim,
            input_type="dict",
            alternate_sign=False,
        )
        self.models: List[XGBRegressor] = []

        self.model_dir = Path(env.root) / "models" / self.name
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "model.pkl"

        self._load_model()

        if history_path is not None:
            self.load_history(history_path)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_model(self) -> None:
        data = {
            "models": self.models,
            "history_rows": self.history_rows,
            "config": {
                "retrain_every": self.retrain_every,
                "min_history_to_train": self.min_history_to_train,
                "n_bootstrap_models": self.n_bootstrap_models,
                "hash_dim": self.hash_dim,
                "time_cap": self.time_cap,
            },
        }

        with open(self.model_path, "wb") as f:
            pickle.dump(data, f)

    def _load_model(self) -> None:
        if not self.model_path.exists():
            return

        try:
            with open(self.model_path, "rb") as f:
                data = pickle.load(f)

            self.models = data.get("models", [])
            loaded_history = data.get("history_rows", [])

            self._reset_state()

            for row in loaded_history:
                self._ingest_row(row, update_models=False)

            print(f"[{self.name}] loaded model with {len(self.history_rows)} rows")
        except Exception as e:
            print(f"[{self.name}] failed to load model: {e}")

    def _reset_state(self) -> None:
        self.history_rows = []
        self.seen_keys = set()
        self.beta_pair = defaultdict(lambda: [1.0, 1.0])
        self.beta_sound = defaultdict(lambda: [1.0, 1.0])
        self.beta_category = defaultdict(lambda: [1.0, 1.0])
        self.live_stats = self._make_empty_stats()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_history(self, history_path: str | Path) -> None:
        history_path = Path(history_path)
        if not history_path.exists():
            return

        df = pd.read_csv(history_path)
        if df.empty:
            return

        required = {"Date", "StationID", "Sound", "DeterTime"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"History file missing columns: {sorted(missing)}")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "StationID", "Sound", "DeterTime"]).copy()
        df = df.sort_values("Date")

        for row in df.to_dict("records"):
            self._ingest_row(row, update_models=False)

        if not self.benchmark_mode and len(self.history_rows) >= self.min_history_to_train:
            self._fit_models()

    def choose(self, stations: List[str], history: List[Dict] | None = None) -> Dict[str, str]:
        self._sync_external_history(history)

        config: Dict[str, str] = {}
        decision_date = self._next_decision_date()

        for station in stations:
            candidate_features = []
            candidate_sounds = []

            for sound in self.sound_pool:
                feat = self._candidate_features_from_stats(
                    station=station,
                    sound=sound,
                    decision_date=decision_date,
                    stats=self.live_stats,
                )
                candidate_features.append(feat)
                candidate_sounds.append(sound)

            mean_pred, std_pred = self._predict_distribution(candidate_features)

            best_sound = None
            best_score = -1e18

            for i, sound in enumerate(candidate_sounds):
                category = self._categorize(sound)

                a_pair, b_pair = self.beta_pair[(station, sound)]
                a_sound, b_sound = self.beta_sound[sound]
                a_cat, b_cat = self.beta_category[category]

                sample_pair = self.rng.betavariate(a_pair, b_pair)
                sample_sound = self.rng.betavariate(a_sound, b_sound)
                sample_cat = self.rng.betavariate(a_cat, b_cat)

                beta_sample = (
                    0.55 * sample_pair +
                    0.30 * sample_sound +
                    0.15 * sample_cat
                )

                mu = float(mean_pred[i])
                sigma = float(max(std_pred[i], 0.03))
                model_sample = self.rng.gauss(mu, sigma)

                pair_count = self.live_stats["pair_n"][(station, sound)]
                total_seen = max(1, len(self.history_rows))
                exploration_bonus = 0.08 * np.sqrt(np.log(total_seen + 2) / (pair_count + 1))

                score = (
                    0.70 * model_sample +
                    0.25 * beta_sample +
                    0.05 * exploration_bonus
                )

                if score > best_score:
                    best_score = score
                    best_sound = sound

            config[station] = best_sound if best_sound is not None else self.rng.choice(self.sound_pool)

        return config

    def update(self, results: List[Dict], history: List[Dict] | None = None) -> None:
        if self.benchmark_mode:
            return

        for row in results:
            self._ingest_row(row, update_models=False)

        self._sync_external_history(history)

        if len(self.history_rows) >= self.min_history_to_train:
            if len(self.history_rows) % self.retrain_every == 0:
                self._fit_models()

        if len(self.history_rows) > 0 and len(self.history_rows) % 200 == 0:
            self._save_model()

    # ------------------------------------------------------------------
    # Core training
    # ------------------------------------------------------------------

    def _fit_models(self) -> None:
        train_dicts, y = self._build_training_examples()

        if not train_dicts or len(y) < self.min_history_to_train:
            return

        X = self.hasher.transform(train_dicts)
        y = np.asarray(y, dtype=np.float32)

        models: List[XGBRegressor] = []
        n = len(y)

        for _ in range(self.n_bootstrap_models):
            rng = np.random.default_rng(self.rng.randint(0, 10**9))
            idx = rng.integers(0, n, n)

            X_boot = X[idx]
            y_boot = y[idx]

            model = XGBRegressor(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.5,
                reg_alpha=0.0,
                min_child_weight=3,
                objective="reg:squarederror",
                tree_method="hist",
                random_state=self.rng.randint(0, 10**9),
                verbosity=0,
            )
            model.fit(X_boot, y_boot)
            models.append(model)

        self.models = models
        self._save_model()

    def _build_training_examples(self):
        if not self.history_rows:
            return [], []

        rows = sorted(self.history_rows, key=lambda r: pd.to_datetime(r["Date"]))

        temp_stats = self._make_empty_stats()
        temp_beta_pair = defaultdict(lambda: [1.0, 1.0])
        temp_beta_sound = defaultdict(lambda: [1.0, 1.0])
        temp_beta_category = defaultdict(lambda: [1.0, 1.0])

        X_rows = []
        y = []

        for row in rows:
            date = pd.to_datetime(row["Date"])
            station = str(row["StationID"])
            sound = str(row["Sound"])
            deter_time = int(row["DeterTime"])
            category = self._categorize(sound)

            feat = self._candidate_features_from_stats(
                station=station,
                sound=sound,
                decision_date=date,
                stats=temp_stats,
                beta_pair=temp_beta_pair,
                beta_sound=temp_beta_sound,
                beta_category=temp_beta_category,
            )

            X_rows.append(feat)
            y.append(self._reward(deter_time))

            reward = self._reward(deter_time)
            success = 1 if deter_time != MAX_DETER_TIME else 0

            self._update_stats_container(
                stats=temp_stats,
                station=station,
                sound=sound,
                category=category,
                date=date,
                reward=reward,
            )
            self._update_beta_containers(
                beta_pair=temp_beta_pair,
                beta_sound=temp_beta_sound,
                beta_category=temp_beta_category,
                station=station,
                sound=sound,
                category=category,
                success=success,
            )

        return X_rows, y

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def _predict_distribution(self, feature_dicts: List[Dict]) -> tuple[np.ndarray, np.ndarray]:
        if not feature_dicts:
            return np.array([]), np.array([])

        X = self.hasher.transform(feature_dicts)

        if not self.models:
            mean_pred = np.full(len(feature_dicts), 0.5, dtype=np.float32)
            std_pred = np.full(len(feature_dicts), 0.15, dtype=np.float32)
            return mean_pred, std_pred

        preds = np.vstack([model.predict(X) for model in self.models])
        mean_pred = preds.mean(axis=0)
        std_pred = preds.std(axis=0)

        mean_pred = np.clip(mean_pred, 0.0, 1.0)
        std_pred = np.clip(std_pred, 0.01, 0.35)

        return mean_pred, std_pred

    # ------------------------------------------------------------------
    # Row ingestion
    # ------------------------------------------------------------------

    def _ingest_row(self, row: Dict, update_models: bool = False) -> None:
        date = pd.to_datetime(row["Date"])
        station = str(row["StationID"])
        sound = str(row["Sound"])
        deter_time = int(row["DeterTime"])
        category = self._categorize(sound)

        key = (
            str(date.date()),
            station,
            sound,
            int(deter_time),
        )
        if key in self.seen_keys:
            return
        self.seen_keys.add(key)

        normalized = {
            "Date": date,
            "StationID": station,
            "Sound": sound,
            "DeterTime": int(deter_time),
        }
        self.history_rows.append(normalized)

        reward = self._reward(deter_time)
        success = 1 if deter_time != MAX_DETER_TIME else 0

        self._update_stats_container(
            stats=self.live_stats,
            station=station,
            sound=sound,
            category=category,
            date=date,
            reward=reward,
        )

        self._update_beta_containers(
            beta_pair=self.beta_pair,
            beta_sound=self.beta_sound,
            beta_category=self.beta_category,
            station=station,
            sound=sound,
            category=category,
            success=success,
        )

        if update_models and not self.benchmark_mode and len(self.history_rows) >= self.min_history_to_train:
            self._fit_models()

    def _sync_external_history(self, history: List[Dict] | None) -> None:
        if not history:
            return

        for row in history:
            self._ingest_row(row, update_models=False)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _candidate_features_from_stats(
        self,
        station: str,
        sound: str,
        decision_date,
        stats,
        beta_pair=None,
        beta_sound=None,
        beta_category=None,
    ) -> Dict:
        category = self._categorize(sound)

        beta_pair = beta_pair if beta_pair is not None else self.beta_pair
        beta_sound = beta_sound if beta_sound is not None else self.beta_sound
        beta_category = beta_category if beta_category is not None else self.beta_category

        station_n = stats["station_n"][station]
        sound_n = stats["sound_n"][sound]
        pair_n = stats["pair_n"][(station, sound)]
        cat_n = stats["category_n"][category]

        station_avg = self._safe_avg(stats["station_reward_sum"][station], station_n)
        sound_avg = self._safe_avg(stats["sound_reward_sum"][sound], sound_n)
        pair_avg = self._safe_avg(stats["pair_reward_sum"][(station, sound)], pair_n)
        cat_avg = self._safe_avg(stats["category_reward_sum"][category], cat_n)

        station_days = self._days_since(stats["station_last_date"].get(station), decision_date, fallback=999)
        sound_days = self._days_since(stats["sound_last_date"].get(sound), decision_date, fallback=999)
        pair_days = self._days_since(stats["pair_last_date"].get((station, sound)), decision_date, fallback=999)
        cat_days = self._days_since(stats["category_last_date"].get(category), decision_date, fallback=999)

        pair_a, pair_b = beta_pair[(station, sound)]
        sound_a, sound_b = beta_sound[sound]
        cat_a, cat_b = beta_category[category]

        pair_beta_mean = pair_a / (pair_a + pair_b)
        sound_beta_mean = sound_a / (sound_a + sound_b)
        cat_beta_mean = cat_a / (cat_a + cat_b)

        d = pd.to_datetime(decision_date)

        return {
            "station": station,
            "sound": sound,
            "category": category,
            "station_sound": f"{station}__{sound}",
            "station_category": f"{station}__{category}",
            "month": float(d.month),
            "weekofyear": float(int(d.isocalendar().week)),
            "dayofyear": float(d.dayofyear),
            "weekday": float(d.weekday()),
            "station_n": float(station_n),
            "sound_n": float(sound_n),
            "pair_n": float(pair_n),
            "cat_n": float(cat_n),
            "station_avg_reward": float(station_avg),
            "sound_avg_reward": float(sound_avg),
            "pair_avg_reward": float(pair_avg),
            "cat_avg_reward": float(cat_avg),
            "station_days_since": float(station_days),
            "sound_days_since": float(sound_days),
            "pair_days_since": float(pair_days),
            "cat_days_since": float(cat_days),
            "pair_beta_mean": float(pair_beta_mean),
            "sound_beta_mean": float(sound_beta_mean),
            "cat_beta_mean": float(cat_beta_mean),
        }

    # ------------------------------------------------------------------
    # Reward definition
    # ------------------------------------------------------------------

    def _reward(self, deter_time: int) -> float:
        if deter_time == MAX_DETER_TIME:
            return 0.0

        clipped = min(max(1, int(deter_time)), self.time_cap)
        return 1.0 - ((clipped - 1) / max(1, self.time_cap - 1))

    # ------------------------------------------------------------------
    # Helpers: stats
    # ------------------------------------------------------------------

    def _make_empty_stats(self):
        return {
            "station_n": defaultdict(int),
            "sound_n": defaultdict(int),
            "pair_n": defaultdict(int),
            "category_n": defaultdict(int),
            "station_reward_sum": defaultdict(float),
            "sound_reward_sum": defaultdict(float),
            "pair_reward_sum": defaultdict(float),
            "category_reward_sum": defaultdict(float),
            "station_last_date": {},
            "sound_last_date": {},
            "pair_last_date": {},
            "category_last_date": {},
        }

    def _update_stats_container(self, stats, station, sound, category, date, reward):
        stats["station_n"][station] += 1
        stats["sound_n"][sound] += 1
        stats["pair_n"][(station, sound)] += 1
        stats["category_n"][category] += 1

        stats["station_reward_sum"][station] += reward
        stats["sound_reward_sum"][sound] += reward
        stats["pair_reward_sum"][(station, sound)] += reward
        stats["category_reward_sum"][category] += reward

        stats["station_last_date"][station] = date
        stats["sound_last_date"][sound] = date
        stats["pair_last_date"][(station, sound)] = date
        stats["category_last_date"][category] = date

    def _update_beta_containers(
        self,
        beta_pair,
        beta_sound,
        beta_category,
        station,
        sound,
        category,
        success,
    ):
        a, b = beta_pair[(station, sound)]
        beta_pair[(station, sound)] = [a + success, b + (1 - success)]

        a, b = beta_sound[sound]
        beta_sound[sound] = [a + success, b + (1 - success)]

        a, b = beta_category[category]
        beta_category[category] = [a + success, b + (1 - success)]

    # ------------------------------------------------------------------
    # Helpers: misc
    # ------------------------------------------------------------------

    def _next_decision_date(self):
        if not self.history_rows:
            return pd.Timestamp.today().normalize()
        latest = max(pd.to_datetime(r["Date"]) for r in self.history_rows)
        return latest + timedelta(days=7)

    def _safe_avg(self, s: float, n: int) -> float:
        return s / n if n > 0 else 0.5

    def _days_since(self, last_date, current_date, fallback=999) -> int:
        if last_date is None:
            return fallback
        last_date = pd.to_datetime(last_date)
        current_date = pd.to_datetime(current_date)
        return max(0, int((current_date - last_date).days))

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