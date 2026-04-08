"""
tree_agents.py
Coverage-oriented tree baseline agents for PicoRV32 ML-DV.

These agents act as contextual surrogate optimisers:
    (coverage_state, knobs) -> predicted total_reward

They share the same state summary used by the DQN agent so coverage
benchmarks remain apples-to-apples.
"""

import copy
import os
import pickle
import random
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from coverage_features import (
    ALL_COVERAGE_BINS,
    COVERAGE_GROUPS,
    KNOB_RANGES,
    KNOB_NAMES,
    _mean_state,
    _normalise_knobs,
    _random_knobs,
    compute_reward,
)

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
    print("[tree_agents] WARNING: scikit-learn not installed - tree agents will use random suggestions.")


N_SEARCH = 768
EXPLORE_FRAC = 0.25
RECENT_KNOB_WINDOW = 256
MIN_TRAIN_SAMPLES = 16
ELITE_LIMIT = 32
RARE_BIN_THRESHOLD = 2
LOCAL_MUTATION_BUDGET = 384
EDGE_MUTATION_BUDGET = 192
TARGETED_CANDIDATE_FRACTION = 0.60
TARGETED_CANDIDATE_MIN = 192
FOCUSED_LATE_STAGE_MAX_UNCOVERED = 3


REWARD_B_TARGET_BINS = {
    "mem_mix_heavy",
    "control_mix_high",
    "cross_alt_low_stall_short",
    "transition_load_then_store_rare",
    "transition_store_then_load_rare",
    "transition_store_then_store_rare",
    "cross_alt_low_stall_medium",
    "cross_trans_moderate_delay_fast",
    "cross_trans_moderate_delay_medium",
    "cross_trans_moderate_delay_slow",
}


def _bin_focus_groups(bin_name: str) -> set:
    name = str(bin_name)
    groups = set()

    if (
        name in ("stall_type_data_dominant", "mem_mix_balanced", "mem_mix_heavy")
        or name.startswith("cross_load_")
        or name.startswith("cross_store_")
    ):
        groups.add("data_heavy")

    if name in ("control_mix_high",):
        groups.add("control_high")

    if name.startswith("cross_alt_low_"):
        groups.add("alt_low")

    if name.startswith("cross_trans_moderate_"):
        groups.add("trans_moderate")

    if name.endswith("_rare") or name.endswith("_recurrent"):
        groups.add("transition_sparse")

    return groups


def _target_profiles_for_uncovered_bins(uncovered_bins: List[str]) -> List[str]:
    profiles = []
    seen = set()

    def add(profile: str) -> None:
        if profile not in seen:
            seen.add(profile)
            profiles.append(profile)

    for bin_name in uncovered_bins:
        if bin_name == "mem_mix_heavy":
            add("data_heavy")
        if bin_name == "control_mix_high":
            add("control_high")
        if bin_name == "cross_alt_low_stall_short":
            add("alt_low_short")
        elif bin_name == "cross_alt_low_stall_medium":
            add("alt_low_medium")
        elif bin_name.startswith("cross_alt_low_"):
            add("alt_low")
        if bin_name == "cross_trans_moderate_delay_fast":
            add("trans_moderate_fast")
        if bin_name == "cross_trans_moderate_delay_medium":
            add("trans_moderate_medium")
        if bin_name == "cross_trans_moderate_delay_slow":
            add("trans_moderate_slow")
        if bin_name.startswith("cross_trans_moderate_"):
            add("trans_moderate")
        if bin_name == "transition_load_then_store_rare":
            add("load_then_store_rare")
        if bin_name == "transition_store_then_load_rare":
            add("store_then_load_rare")
        if bin_name == "transition_store_then_store_rare":
            add("store_then_store_rare")
        for focus_group in _bin_focus_groups(bin_name):
            add(focus_group)

    return profiles


class _BaseTreeCoverageAgent:
    model_kind = "tree"

    def __init__(self, random_state: int = 42):
        self.random_state = int(random_state)
        self.iteration = 0
        self.best_seen: dict = {}
        self.best_reward: float = 0.0
        self.best_stall_seen: float = 0.0
        self.recent_knob_keys: List[tuple] = []
        self.bin_hit_counts: Counter = Counter()
        self.dataset: List[tuple] = []  # (state, knobs_norm, reward, sample_weight)
        self.elite_knobs: List[dict] = []
        self.model = self._build_model() if _SKLEARN_OK else None

    def _build_model(self):
        raise NotImplementedError

    @staticmethod
    def _knob_key(knobs: dict) -> tuple:
        return tuple(sorted(knobs.items()))

    def _remember_knobs(self, knobs_list: List[dict]):
        for knobs in knobs_list:
            self.recent_knob_keys.append(self._knob_key(knobs))
        if len(self.recent_knob_keys) > RECENT_KNOB_WINDOW:
            self.recent_knob_keys = self.recent_knob_keys[-RECENT_KNOB_WINDOW:]

    def _remember_elite(self, knobs: dict):
        key = self._knob_key(knobs)
        deduped = []
        seen = {key}
        deduped.append(copy.deepcopy(knobs))
        for item in self.elite_knobs:
            item_key = self._knob_key(item)
            if item_key in seen:
                continue
            seen.add(item_key)
            deduped.append(copy.deepcopy(item))
            if len(deduped) >= ELITE_LIMIT:
                break
        self.elite_knobs = deduped

    def _mutate_knobs(self, seed_knobs: dict, edge_bias: bool = False) -> dict:
        knobs = copy.deepcopy(seed_knobs)
        n_changes = random.randint(1, min(4, len(KNOB_NAMES)))
        for name in random.sample(KNOB_NAMES, n_changes):
            vals = KNOB_RANGES[name]
            cur = knobs.get(name, vals[0])
            if edge_bias and random.random() < 0.55:
                knobs[name] = random.choice([vals[0], vals[-1]])
                continue
            idx = vals.index(cur) if cur in vals else 0
            step = random.choice([-3, -2, -1, 1, 2, 3])
            idx = max(0, min(len(vals) - 1, idx + step))
            knobs[name] = vals[idx]
        return knobs

    def _edge_biased_knobs(self) -> dict:
        knobs = {}
        for name, vals in KNOB_RANGES.items():
            if random.random() < 0.7:
                knobs[name] = random.choice([vals[0], vals[-1]])
            else:
                knobs[name] = random.choice(vals)
        return knobs

    def _random_unseen_knobs(self, used_keys: set) -> dict:
        recent = set(self.recent_knob_keys)
        for _ in range(64):
            knobs = _random_knobs()
            key = self._knob_key(knobs)
            if key in used_keys or key in recent:
                continue
            return knobs
        return _random_knobs()

    def _targeted_knobs(self, profile: str) -> dict:
        knobs = _random_knobs()
        profile = str(profile)

        if profile == "alt_low_short":
            knobs["load_weight"] = random.choice([1, 2, 4, 8, 10])
            knobs["store_weight"] = random.choice([1, 2, 8, 9, 10])
            knobs["branch_weight"] = random.choice([6, 7, 8, 9, 10])
            knobs["jump_weight"] = random.choice([5, 6, 7, 8, 9, 10])
            knobs["arith_weight"] = random.choice([3, 4, 5, 6])
            knobs["mem_stride"] = random.choice([1, 2, 3, 4, 5, 6, 8])
            knobs["mem_delay_base"] = 1
            knobs["mixed_burst_bias"] = random.choice([0, 0, 0, 1])
            knobs["pointer_update_rate"] = random.choice([3, 4, 5, 6, 8, 9])
            knobs["trap_rate"] = random.choice([0, 0, 1, 2])
            knobs["trap_kind"] = random.choice([0, 1, 3])
            knobs["branch_taken_bias"] = random.choice([0, 1, 2, 3, 4, 5, 9])

        if profile == "alt_low_medium":
            knobs["load_weight"] = random.choice([1, 2, 4, 5, 7, 10])
            knobs["store_weight"] = random.choice([1, 3, 4, 7, 9, 10])
            knobs["branch_weight"] = random.choice([4, 5, 6, 8, 9, 10])
            knobs["jump_weight"] = random.choice([2, 3, 4, 5, 9])
            knobs["arith_weight"] = random.choice([3, 4, 6, 8, 10])
            knobs["mem_stride"] = random.choice([1, 2, 3, 4, 5, 6, 8])
            knobs["pointer_update_rate"] = random.choice([1, 2, 3, 5, 7, 8])
            knobs["trap_rate"] = random.choice([0, 1, 2, 3])
            knobs["trap_kind"] = random.choice([0, 1, 3])
            knobs["branch_taken_bias"] = random.choice([0, 1, 4, 5, 7])
            knobs["mixed_burst_bias"] = random.choice([0, 0, 0, 1])
            knobs["mem_delay_base"] = random.choice([2, 3, 4])

        if profile == "control_high":
            knobs["load_weight"] = random.choice([2, 3, 4, 5, 6])
            knobs["store_weight"] = random.choice([1, 2])
            knobs["branch_weight"] = random.choice([5, 6, 7, 8, 9, 10])
            knobs["jump_weight"] = random.choice([4, 5, 6, 7, 8, 9, 10])
            knobs["arith_weight"] = random.choice([3, 4, 5, 6])
            knobs["mem_stride"] = random.choice([2, 3, 4])
            knobs["pointer_update_rate"] = random.choice([1, 2, 3])
            knobs["trap_rate"] = 0
            knobs["trap_kind"] = random.choice([0, 3])
            knobs["branch_taken_bias"] = random.choice([0, 1, 2, 3, 6, 7, 8, 9, 10])
            knobs["mixed_burst_bias"] = random.choice([0, 1])
            knobs["mem_delay_base"] = random.choice([1, 2])

        if profile == "data_heavy":
            knobs["load_weight"] = random.choice([7, 8, 9, 10])
            knobs["store_weight"] = random.choice([6, 7, 8, 9, 10])
            knobs["arith_weight"] = random.choice([2, 3, 4, 5])
            knobs["branch_weight"] = random.choice([1, 2, 3, 4])
            knobs["jump_weight"] = random.choice([1, 2, 3])
            knobs["mixed_burst_bias"] = random.choice([5, 6, 7, 8, 9, 10])
            knobs["pointer_update_rate"] = random.choice([6, 7, 8, 9, 10])

        if profile == "trans_moderate_fast":
            knobs["load_weight"] = random.choice([2, 3, 4, 6, 7, 8])
            knobs["store_weight"] = random.choice([3, 4, 6, 8, 9, 10])
            knobs["branch_weight"] = random.choice([6, 7, 8, 9, 10])
            knobs["jump_weight"] = random.choice([5, 6, 7, 8, 9, 10])
            knobs["arith_weight"] = random.choice([4, 5, 6, 7, 8, 9])
            knobs["mem_stride"] = random.choice([1, 2, 3, 5, 6, 8])
            knobs["pointer_update_rate"] = random.choice([2, 3, 5, 8, 9, 10])
            knobs["trap_rate"] = random.choice([1, 2, 3])
            knobs["trap_kind"] = random.choice([0, 1, 2, 3])
            knobs["branch_taken_bias"] = random.choice([1, 2, 5, 7, 10])
            knobs["mixed_burst_bias"] = random.choice([0, 1, 3, 4, 7, 8])
            knobs["mem_delay_base"] = random.choice([1, 2])

        if profile == "trans_moderate_medium":
            knobs["load_weight"] = random.choice([1, 2, 3, 4, 5, 6, 8])
            knobs["store_weight"] = random.choice([1, 2, 6, 9, 10])
            knobs["branch_weight"] = random.choice([3, 4, 5, 9, 10])
            knobs["jump_weight"] = random.choice([6, 7, 8, 9, 10])
            knobs["arith_weight"] = random.choice([3, 7, 8, 9, 10])
            knobs["mem_stride"] = random.choice([1, 3, 5, 6, 8])
            knobs["pointer_update_rate"] = random.choice([1, 5, 7, 9, 10])
            knobs["trap_rate"] = random.choice([1, 2, 3])
            knobs["trap_kind"] = random.choice([0, 1, 3])
            knobs["branch_taken_bias"] = random.choice([0, 1, 4, 5, 7])
            knobs["mixed_burst_bias"] = random.choice([0, 1, 3, 4, 5])
            knobs["mem_delay_base"] = random.choice([3, 4, 5])

        if profile == "trans_moderate_slow":
            knobs["load_weight"] = random.choice([2, 3, 6, 8, 9])
            knobs["store_weight"] = random.choice([1, 3, 4, 8, 10])
            knobs["branch_weight"] = random.choice([2, 7, 8, 10])
            knobs["jump_weight"] = random.choice([5, 7, 8, 9, 10])
            knobs["arith_weight"] = random.choice([2, 3, 7, 8, 9])
            knobs["mem_stride"] = random.choice([3, 4, 5, 6, 8])
            knobs["pointer_update_rate"] = random.choice([2, 6, 7, 8, 9])
            knobs["trap_rate"] = random.choice([1, 2, 3])
            knobs["trap_kind"] = random.choice([0, 1, 2, 3])
            knobs["branch_taken_bias"] = random.choice([2, 3, 7, 8, 9])
            knobs["mixed_burst_bias"] = random.choice([0, 2, 4, 5, 9])
            knobs["mem_delay_base"] = random.choice([6, 7, 8])

        if profile == "load_then_store_rare":
            knobs["load_weight"] = random.choice([5, 6, 8, 9])
            knobs["store_weight"] = random.choice([1, 2])
            knobs["branch_weight"] = random.choice([4, 5, 9])
            knobs["jump_weight"] = random.choice([5, 8, 9])
            knobs["arith_weight"] = random.choice([8, 9])
            knobs["mem_stride"] = random.choice([3, 4, 8])
            knobs["pointer_update_rate"] = random.choice([1, 5, 9])
            knobs["trap_rate"] = random.choice([1, 2, 3])
            knobs["trap_kind"] = random.choice([1, 3])
            knobs["branch_taken_bias"] = random.choice([1, 3, 5, 7, 9])
            knobs["mixed_burst_bias"] = random.choice([0, 1])
            knobs["mem_delay_base"] = random.choice([3, 4, 8])

        if profile == "store_then_load_rare":
            knobs["load_weight"] = random.choice([1, 2, 4, 5, 7, 9])
            knobs["store_weight"] = random.choice([6, 9, 10])
            knobs["branch_weight"] = random.choice([4, 5, 9, 10])
            knobs["jump_weight"] = random.choice([1, 3, 5, 8, 10])
            knobs["arith_weight"] = random.choice([1, 7, 8, 10])
            knobs["mem_stride"] = random.choice([2, 5, 6, 7, 8])
            knobs["pointer_update_rate"] = random.choice([1, 5, 6, 8])
            knobs["trap_rate"] = random.choice([2, 3])
            knobs["trap_kind"] = 0
            knobs["branch_taken_bias"] = random.choice([1, 3, 7, 8])
            knobs["mixed_burst_bias"] = random.choice([0, 2, 3, 4, 5])
            knobs["mem_delay_base"] = random.choice([2, 4, 6])

        if profile == "store_then_store_rare":
            knobs["load_weight"] = random.choice([1, 2, 4, 5, 10])
            knobs["store_weight"] = random.choice([1, 3, 4, 6, 10])
            knobs["branch_weight"] = random.choice([5, 8, 9, 10])
            knobs["jump_weight"] = random.choice([1, 2, 3, 5, 10])
            knobs["arith_weight"] = random.choice([1, 2, 5, 6, 10])
            knobs["mem_stride"] = random.choice([2, 4, 6, 8])
            knobs["pointer_update_rate"] = random.choice([2, 6, 8, 9])
            knobs["trap_rate"] = random.choice([2, 3])
            knobs["trap_kind"] = random.choice([1, 3])
            knobs["branch_taken_bias"] = random.choice([4, 5, 6, 7, 10])
            knobs["mixed_burst_bias"] = random.choice([4, 5, 7, 8, 9])
            knobs["mem_delay_base"] = random.choice([3, 5, 7])

        return knobs

    def _profile_for_knobs(self, knobs: dict) -> Optional[str]:
        load = int(knobs.get("load_weight", 1))
        store = int(knobs.get("store_weight", 1))
        branch = int(knobs.get("branch_weight", 1))
        jump = int(knobs.get("jump_weight", 1))
        mix = int(knobs.get("mixed_burst_bias", 0))
        delay = int(knobs.get("mem_delay_base", 1))

        if delay == 1 and mix <= 1:
            return "alt_low_short"
        if 2 <= delay <= 4 and mix <= 1:
            return "alt_low_medium"
        if delay <= 2 and mix <= 1 and store <= 2 and (branch + jump) >= 9:
            return "control_high"
        if load >= 7 and store >= 6 and mix >= 5:
            return "data_heavy"
        if delay <= 2 and (branch + jump) >= 11:
            return "trans_moderate_fast"
        if 3 <= delay <= 5:
            return "trans_moderate_medium"
        if 6 <= delay <= 8:
            return "trans_moderate_slow"
        if load >= 5 and store <= 2:
            return "load_then_store_rare"
        if store >= 6 and load <= 5:
            return "store_then_load_rare"
        if store >= 4 and 3 <= delay <= 7 and mix >= 4:
            return "store_then_store_rare"
        return None

    def _late_stage_mode_active(self, uncovered_bins: List[str]) -> bool:
        if not uncovered_bins:
            return False
        if len(uncovered_bins) > len(REWARD_B_TARGET_BINS):
            return False
        return set(uncovered_bins).issubset(REWARD_B_TARGET_BINS)

    def _tree_residual_adjustment(self, knobs: dict, uncovered_bins: List[str]) -> float:
        if not uncovered_bins:
            return 0.0
        target_profiles = _target_profiles_for_uncovered_bins(uncovered_bins)
        if not target_profiles:
            return 0.0
        profile = self._profile_for_knobs(knobs)
        bonus = 0.0
        if profile in target_profiles:
            bonus += 0.18
        delay = int(knobs.get("mem_delay_base", 1))
        mix = int(knobs.get("mixed_burst_bias", 0))
        load = int(knobs.get("load_weight", 1))
        store = int(knobs.get("store_weight", 1))
        branch = int(knobs.get("branch_weight", 1))
        jump = int(knobs.get("jump_weight", 1))

        if "cross_alt_low_stall_short" in uncovered_bins:
            if delay == 1 and mix <= 1:
                bonus += 0.10
        if "cross_alt_low_stall_medium" in uncovered_bins:
            if 2 <= delay <= 4 and mix <= 1:
                bonus += 0.10
        if "cross_trans_moderate_delay_fast" in uncovered_bins:
            if 1 <= delay <= 2:
                bonus += 0.08
        if "cross_trans_moderate_delay_medium" in uncovered_bins:
            if 3 <= delay <= 5:
                bonus += 0.08
        if "cross_trans_moderate_delay_slow" in uncovered_bins:
            if 6 <= delay <= 8:
                bonus += 0.08
        if "transition_load_then_store_rare" in uncovered_bins:
            if load >= 5 and store <= 2 and mix <= 1:
                bonus += 0.08
        if "transition_store_then_load_rare" in uncovered_bins:
            if store >= 6 and load <= 5:
                bonus += 0.08
        if "control_mix_high" in uncovered_bins:
            if (branch + jump) >= 12 and delay <= 2 and mix <= 1:
                bonus += 0.08
        if "mem_mix_heavy" in uncovered_bins:
            if (load + store) >= 15 and mix >= 5:
                bonus += 0.08
        return bonus

    def _sample_candidate_pool(self, target_count: int, uncovered_bins: Optional[List[str]] = None) -> List[dict]:
        recent = set(self.recent_knob_keys)
        unique: Dict[tuple, dict] = {}
        seeds = []
        target_profiles = _target_profiles_for_uncovered_bins(uncovered_bins or [])
        focused_late_stage = self._late_stage_mode_active(uncovered_bins or []) and len(uncovered_bins or []) <= FOCUSED_LATE_STAGE_MAX_UNCOVERED

        if target_profiles:
            targeted_take = target_count if focused_late_stage else min(
                target_count,
                max(TARGETED_CANDIDATE_MIN, int(target_count * TARGETED_CANDIDATE_FRACTION))
            )
            weighted_profiles = target_profiles * (8 if focused_late_stage else 4)
            while len(unique) < targeted_take and weighted_profiles:
                knobs = self._targeted_knobs(random.choice(weighted_profiles))
                key = self._knob_key(knobs)
                if key in recent or key in unique:
                    continue
                unique[key] = knobs

        if self.best_seen:
            seeds.append(self.best_seen)
        seeds.extend(self.elite_knobs[: min(8, len(self.elite_knobs))])

        for seed in seeds:
            for _ in range(max(8, LOCAL_MUTATION_BUDGET // max(len(seeds), 1))):
                knobs = self._mutate_knobs(seed, edge_bias=False)
                key = self._knob_key(knobs)
                if key in recent or key in unique:
                    continue
                unique[key] = knobs
                if len(unique) >= target_count:
                    return list(unique.values())

        for seed in seeds:
            for _ in range(max(4, EDGE_MUTATION_BUDGET // max(len(seeds), 1))):
                knobs = self._mutate_knobs(seed, edge_bias=True)
                key = self._knob_key(knobs)
                if key in recent or key in unique:
                    continue
                unique[key] = knobs
                if len(unique) >= target_count:
                    return list(unique.values())

        attempts = max(target_count * 3, target_count + 32)
        for _ in range(attempts):
            knobs = self._edge_biased_knobs() if random.random() < 0.35 else _random_knobs()
            key = self._knob_key(knobs)
            if key in recent or key in unique:
                continue
            unique[key] = knobs
            if len(unique) >= target_count:
                break

        while len(unique) < target_count:
            knobs = self._edge_biased_knobs() if random.random() < 0.5 else _random_knobs()
            key = self._knob_key(knobs)
            if key in unique:
                continue
            unique[key] = knobs

        return list(unique.values())

    def _state_from_results(self, results: list) -> np.ndarray:
        return _mean_state(results, self.bin_hit_counts, state_mode="compact")

    def _feature_vector(self, state: np.ndarray, knobs: dict) -> np.ndarray:
        knobs_norm = _normalise_knobs(knobs)
        return np.concatenate([state, knobs_norm]).astype(np.float32)

    def _novelty_score(self, knobs: dict) -> float:
        refs = []
        if self.best_seen:
            refs.append(_normalise_knobs(self.best_seen))
        refs.extend(_normalise_knobs(k) for k in self.elite_knobs[:4])
        if not refs:
            return 0.0
        cand = _normalise_knobs(knobs)
        distances = [float(np.mean(np.abs(cand - ref))) for ref in refs]
        return float(min(np.mean(distances), 1.0))

    def _ingest(self, results: list):
        state = self._state_from_results(results)
        for r in results:
            reward = float(
                r["coverage"].get(
                    "total_reward",
                    r["coverage"].get("composite_reward", compute_reward(r["coverage"])),
                )
            )
            raw_stall = float(r["coverage"].get("stall_ratio", 0.0))
            knobs_norm = _normalise_knobs(r["knobs"])
            functional_bins = [
                b for b in r["coverage"].get("functional_bins", [])
                if b in ALL_COVERAGE_BINS
            ]
            new_bin_hits = sum(1 for b in functional_bins if self.bin_hit_counts.get(b, 0) == 0)
            rare_bin_hits = sum(1 for b in functional_bins if self.bin_hit_counts.get(b, 0) <= RARE_BIN_THRESHOLD)
            sample_weight = (
                1.0 +
                0.55 * min(new_bin_hits, 4) +
                0.20 * min(rare_bin_hits, 5) +
                0.25 * reward
            )
            self.dataset.append((state.copy(), knobs_norm, reward, sample_weight))
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_seen = copy.deepcopy(r["knobs"])
                self._remember_elite(r["knobs"])
            if raw_stall > self.best_stall_seen:
                self.best_stall_seen = raw_stall
            if new_bin_hits > 0 or reward >= max(0.15, self.best_reward * 0.92):
                self._remember_elite(r["knobs"])
            for b in functional_bins:
                self.bin_hit_counts[b] += 1
        self._remember_knobs([r["knobs"] for r in results])

    def _fit(self):
        if self.model is None or len(self.dataset) < MIN_TRAIN_SAMPLES:
            return
        X = np.stack([
            np.concatenate([state, knobs_norm])
            for state, knobs_norm, _, _ in self.dataset
        ]).astype(np.float32)
        y = np.asarray([reward for _, _, reward, _ in self.dataset], dtype=np.float32)
        sample_weight = np.asarray([weight for _, _, _, weight in self.dataset], dtype=np.float32)
        try:
            self.model.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            self.model.fit(X, y)

    def _score_candidates(self, state: np.ndarray, candidates: List[dict]) -> np.ndarray:
        if self.model is None or len(self.dataset) < MIN_TRAIN_SAMPLES:
            return np.zeros(len(candidates), dtype=np.float32)
        X = np.stack([self._feature_vector(state, c) for c in candidates]).astype(np.float32)
        scores = self.model.predict(X)
        return np.asarray(scores, dtype=np.float32)

    def suggest_knobs_for_iteration(self, prev_results: list, n_suggest: int = 10) -> List[dict]:
        if prev_results:
            self._ingest(prev_results)
        self._fit()
        suggestions = self._suggest(prev_results, n_suggest)
        self.iteration += 1
        self._remember_knobs(suggestions)
        n_explore = max(1, int(n_suggest * EXPLORE_FRAC))
        print(
            f"  [{self.model_kind.upper()}] iter={self.iteration}  "
            f"dataset={len(self.dataset)}  "
            f"exploit={max(0, len(suggestions) - n_explore)}  "
            f"explore={n_explore}  "
            f"best_seen_reward={self.best_reward:.4f}  "
            f"best_seen_stall={self.best_stall_seen:.4f}"
        )
        return suggestions[:n_suggest]

    def suggest_knobs_frozen(self, prev_results: Optional[list] = None, n_suggest: int = 10) -> List[dict]:
        suggestions = self._suggest(prev_results or [], n_suggest)
        return suggestions[:n_suggest]

    def _suggest(self, prev_results: list, n_suggest: int) -> List[dict]:
        state = self._state_from_results(prev_results)
        uncovered_bins = [
            b for b in ALL_COVERAGE_BINS
            if self.bin_hit_counts.get(b, 0) == 0
        ]
        n_explore = max(1, int(n_suggest * EXPLORE_FRAC))
        n_exploit = n_suggest - n_explore
        suggestions: List[dict] = []
        suggestion_keys = set()

        if self.model is not None and len(self.dataset) >= MIN_TRAIN_SAMPLES:
            candidates = self._sample_candidate_pool(N_SEARCH, uncovered_bins=uncovered_bins)
            scores = self._score_candidates(state, candidates)
            for idx in np.argsort(scores)[::-1]:
                key = self._knob_key(candidates[int(idx)])
                if key in suggestion_keys:
                    continue
                suggestions.append(candidates[int(idx)])
                suggestion_keys.add(key)
                if len(suggestions) >= n_exploit:
                    break
        else:
            n_explore = n_suggest

        for _ in range(n_explore):
            knobs = self._random_unseen_knobs(suggestion_keys)
            suggestions.append(knobs)
            suggestion_keys.add(self._knob_key(knobs))

        if self.best_seen and suggestions:
            suggestions[-1] = copy.deepcopy(self.best_seen)

        return suggestions

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "model_kind": self.model_kind,
                    "random_state": self.random_state,
                    "iteration": self.iteration,
                    "best_seen": self.best_seen,
                    "best_reward": self.best_reward,
                    "best_stall_seen": self.best_stall_seen,
                    "recent_knob_keys": self.recent_knob_keys,
                    "elite_knobs": self.elite_knobs,
                    "bin_hit_counts": dict(self.bin_hit_counts),
                    "dataset": [
                        (
                            np.asarray(state, dtype=np.float32).tolist(),
                            np.asarray(knobs_norm, dtype=np.float32).tolist(),
                            float(reward),
                            float(weight),
                        )
                        for state, knobs_norm, reward, weight in self.dataset
                    ],
                    "model": self.model,
                },
                f,
            )
        print(f"  [{self.model_kind.upper()}] Model saved: {path}")

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        self.iteration = int(ckpt.get("iteration", 0))
        self.best_seen = dict(ckpt.get("best_seen", {}))
        self.best_reward = float(ckpt.get("best_reward", 0.0))
        self.best_stall_seen = float(ckpt.get("best_stall_seen", 0.0))
        self.recent_knob_keys = [tuple(item) for item in ckpt.get("recent_knob_keys", [])]
        self.elite_knobs = [dict(item) for item in ckpt.get("elite_knobs", [])]
        self.bin_hit_counts = Counter(ckpt.get("bin_hit_counts", {}))
        self.dataset = []
        for item in ckpt.get("dataset", []):
            if len(item) >= 4:
                self.dataset.append((
                    np.asarray(item[0], dtype=np.float32),
                    np.asarray(item[1], dtype=np.float32),
                    float(item[2]),
                    float(item[3]),
                ))
            else:
                self.dataset.append((
                    np.asarray(item[0], dtype=np.float32),
                    np.asarray(item[1], dtype=np.float32),
                    float(item[2]),
                    1.0,
                ))
        self.model = ckpt.get("model", self._build_model() if _SKLEARN_OK else None)
        print(f"  [{self.model_kind.upper()}] Model loaded: {path}")


class DecisionTreeCoverageAgent(_BaseTreeCoverageAgent):
    model_kind = "dt"

    def _build_model(self):
        if not _SKLEARN_OK:
            return None
        return DecisionTreeRegressor(
            max_depth=8,
            min_samples_leaf=3,
            min_samples_split=8,
            splitter="best",
            ccp_alpha=0.0005,
            random_state=self.random_state,
        )

    def _score_candidates(self, state: np.ndarray, candidates: List[dict]) -> np.ndarray:
        if self.model is None or len(self.dataset) < MIN_TRAIN_SAMPLES:
            return np.zeros(len(candidates), dtype=np.float32)
        uncovered_bins = [
            b for b in ALL_COVERAGE_BINS
            if self.bin_hit_counts.get(b, 0) == 0
        ]
        X = np.stack([self._feature_vector(state, c) for c in candidates]).astype(np.float32)
        scores = np.asarray(self.model.predict(X), dtype=np.float32)
        leaves = self.model.apply(X)
        leaf_counts = self.model.tree_.n_node_samples[leaves].astype(np.float32)
        rarity = 1.0 - np.clip(leaf_counts / max(float(len(self.dataset)), 1.0), 0.0, 1.0)
        novelty = np.asarray([self._novelty_score(c) for c in candidates], dtype=np.float32)
        residual = np.asarray([
            self._tree_residual_adjustment(c, uncovered_bins) for c in candidates
        ], dtype=np.float32)
        return scores + 0.12 * rarity + 0.08 * novelty + residual


class RandomForestCoverageAgent(_BaseTreeCoverageAgent):
    model_kind = "rf"

    def _build_model(self):
        if not _SKLEARN_OK:
            return None
        return RandomForestRegressor(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=2,
            min_samples_split=6,
            max_features="sqrt",
            bootstrap=True,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def _rf_residual_adjustment(self, knobs: dict) -> float:
        bonus = 0.0
        penalty = 0.0

        def any_unhit(bin_names: List[str]) -> bool:
            return any(self.bin_hit_counts.get(name, 0) == 0 for name in bin_names)

        def heavily_hit(bin_name: str, threshold: int = 18) -> bool:
            return self.bin_hit_counts.get(bin_name, 0) >= threshold

        transition_hot_bins = [
            "transition_fetch_then_load_hot",
            "transition_fetch_then_store_hot",
            "transition_load_then_load_hot",
            "transition_load_then_store_hot",
            "transition_store_then_load_hot",
            "transition_store_then_store_hot",
        ]
        rare_transition_bins = [
            "transition_fetch_then_load_rare",
            "transition_fetch_then_store_rare",
            "transition_load_then_load_rare",
            "transition_load_then_store_rare",
            "transition_store_then_load_rare",
            "transition_store_then_store_rare",
        ]
        recurrent_transition_bins = [
            "transition_fetch_then_load_recurrent",
            "transition_fetch_then_store_recurrent",
            "transition_load_then_load_recurrent",
            "transition_load_then_store_recurrent",
            "transition_store_then_load_recurrent",
            "transition_store_then_store_recurrent",
        ]
        moderate_delay_cross_bins = COVERAGE_GROUPS["trans_delay_cross"][:3]

        if any(heavily_hit(name, threshold=20) for name in transition_hot_bins):
            if (knobs.get("mem_delay_base", 1) >= 7 and
                knobs.get("mixed_burst_bias", 0) >= 7 and
                knobs.get("trap_rate", 0) == 0 and
                (knobs.get("load_weight", 1) + knobs.get("store_weight", 1)) >= 12):
                penalty += 0.10

        if any_unhit(moderate_delay_cross_bins):
            delay = knobs.get("mem_delay_base", 1)
            if 3 <= delay <= 6:
                bonus += 0.10
            elif delay >= 7:
                penalty += 0.04

        if self.bin_hit_counts.get("stall_type_data_dominant", 0) == 0:
            mem_weight = knobs.get("load_weight", 1) + knobs.get("store_weight", 1)
            control_weight = knobs.get("branch_weight", 1) + knobs.get("jump_weight", 1)
            if mem_weight >= 14 and control_weight <= 9:
                bonus += 0.10
            if knobs.get("store_weight", 1) >= 7 and knobs.get("mixed_burst_bias", 0) >= 5:
                bonus += 0.04

        if any_unhit(rare_transition_bins):
            if 2 <= knobs.get("mixed_burst_bias", 0) <= 6:
                bonus += 0.05
            if knobs.get("jump_weight", 1) >= 4 and knobs.get("branch_weight", 1) >= 4:
                bonus += 0.03
            if knobs.get("mem_delay_base", 1) >= 7:
                penalty += 0.03

        if any_unhit(recurrent_transition_bins):
            if 1 <= knobs.get("mixed_burst_bias", 0) <= 5:
                bonus += 0.05
            if 3 <= knobs.get("branch_weight", 1) <= 8 and 2 <= knobs.get("jump_weight", 1) <= 7:
                bonus += 0.03

        if any_unhit(["transition_diversity_poor", "transition_diversity_moderate"]):
            if knobs.get("mixed_burst_bias", 0) <= 4:
                bonus += 0.05
            if knobs.get("mem_delay_base", 1) >= 7 and knobs.get("mixed_burst_bias", 0) >= 7:
                penalty += 0.04

        if any_unhit(["transition_entropy_low", "transition_entropy_medium"]):
            if knobs.get("mixed_burst_bias", 0) <= 5:
                bonus += 0.04
            if knobs.get("branch_taken_bias", 5) in (3, 4, 5, 6, 7):
                bonus += 0.02

        return bonus - penalty

    def _score_candidates(self, state: np.ndarray, candidates: List[dict]) -> np.ndarray:
        if self.model is None or len(self.dataset) < MIN_TRAIN_SAMPLES:
            return np.zeros(len(candidates), dtype=np.float32)
        uncovered_bins = [
            b for b in ALL_COVERAGE_BINS
            if self.bin_hit_counts.get(b, 0) == 0
        ]
        X = np.stack([self._feature_vector(state, c) for c in candidates]).astype(np.float32)
        tree_preds = np.stack([est.predict(X) for est in self.model.estimators_], axis=0).astype(np.float32)
        mean_pred = np.mean(tree_preds, axis=0)
        std_pred = np.std(tree_preds, axis=0)
        novelty = np.asarray([self._novelty_score(c) for c in candidates], dtype=np.float32)
        residual = np.asarray([self._rf_residual_adjustment(c) for c in candidates], dtype=np.float32)
        late_residual = np.asarray([
            self._tree_residual_adjustment(c, uncovered_bins) for c in candidates
        ], dtype=np.float32)
        return mean_pred + 0.35 * std_pred + 0.10 * novelty + residual + late_residual
