"""
coverage_features.py
Shared coverage-state, knob-space, and reward helpers for coverage agents.
"""

import random
from collections import Counter
from typing import Dict, Optional

import numpy as np

KNOB_RANGES: Dict[str, list] = {
    "load_weight":    list(range(1, 11)),
    "store_weight":   list(range(1, 11)),
    "branch_weight":  list(range(1, 11)),
    "jump_weight":    list(range(1, 11)),
    "arith_weight":   list(range(1, 11)),
    "mem_stride":     list(range(1,  9)),
    "pointer_update_rate": list(range(1, 11)),
    "trap_rate":      list(range(0,  4)),
    "trap_kind":      list(range(0,  4)),
    "branch_taken_bias": list(range(0, 11)),
    "mixed_burst_bias": list(range(0, 11)),
    "mem_delay_base": list(range(1,  9)),
}

KNOB_NAMES = list(KNOB_RANGES.keys())
BASE_STATE_DIM = 8
LEGACY_STATE_DIM = BASE_STATE_DIM
UNCOVERED_GROUP_NAMES = [
    "stall_depth",
    "stall_type",
    "transition_diversity",
    "transition_entropy",
    "alternation",
    "instr_window",
    "b2b_pressure",
    "delay",
    "mem_mix",
    "control_mix",
    "stall_cross",
    "alt_cross",
    "trans_delay_cross",
    "uncovered_total",
]
STATE_DIM = BASE_STATE_DIM + len(UNCOVERED_GROUP_NAMES)
KNOB_DIM = len(KNOB_NAMES)
INPUT_DIM = STATE_DIM + KNOB_DIM

COVERAGE_GROUPS = {
    "stall_depth": [
        "stall_depth_short", "stall_depth_medium", "stall_depth_long",
    ],
    "stall_type": [
        "stall_type_instr_dominant", "stall_type_balanced", "stall_type_data_dominant",
    ],
    "transition_diversity": [
        "transition_diversity_poor", "transition_diversity_moderate", "transition_diversity_rich",
    ],
    "transition_entropy": [
        "transition_entropy_low", "transition_entropy_medium", "transition_entropy_high",
    ],
    "alternation": [
        "alternation_none", "alternation_low", "alternation_medium", "alternation_high",
    ],
    "instr_window": [
        "instr_window_trivial", "instr_window_short", "instr_window_medium", "instr_window_long",
    ],
    "b2b_pressure": [
        "b2b_pressure_low", "b2b_pressure_medium", "b2b_pressure_high",
    ],
    "delay": [
        "delay_fast", "delay_medium", "delay_slow",
    ],
    "mem_mix": [
        "mem_mix_light", "mem_mix_balanced", "mem_mix_heavy",
    ],
    "control_mix": [
        "control_mix_low", "control_mix_medium", "control_mix_high",
    ],
    "stall_cross": [
        "cross_fetch_stall_short", "cross_fetch_stall_medium", "cross_fetch_stall_long",
        "cross_load_stall_short", "cross_load_stall_medium", "cross_load_stall_long",
        "cross_store_stall_short", "cross_store_stall_medium", "cross_store_stall_long",
        "cross_b2b_stall_short", "cross_b2b_stall_medium", "cross_b2b_stall_long",
    ],
    "alt_cross": [
        "cross_alt_low_stall_short", "cross_alt_low_stall_medium", "cross_alt_low_stall_long",
        "cross_alt_medium_stall_short", "cross_alt_medium_stall_medium", "cross_alt_medium_stall_long",
        "cross_alt_high_stall_short", "cross_alt_high_stall_medium", "cross_alt_high_stall_long",
    ],
    "trans_delay_cross": [
        "cross_trans_moderate_delay_fast", "cross_trans_moderate_delay_medium", "cross_trans_moderate_delay_slow",
        "cross_trans_rich_delay_fast", "cross_trans_rich_delay_medium", "cross_trans_rich_delay_slow",
    ],
}
ALL_COVERAGE_BINS = [b for bins in COVERAGE_GROUPS.values() for b in bins]


def _normalise_knobs(knobs: dict) -> np.ndarray:
    out = []
    for name in KNOB_NAMES:
        vals = KNOB_RANGES[name]
        lo, hi = vals[0], vals[-1]
        v = knobs.get(name, lo)
        out.append((v - lo) / max(hi - lo, 1))
    return np.array(out, dtype=np.float32)


def _random_knobs() -> dict:
    return {name: random.choice(vals) for name, vals in KNOB_RANGES.items()}


def compute_reward(cov: dict) -> float:
    stall_ratio = float(cov.get("stall_ratio", 0.0))
    data_stall_ratio = float(cov.get(
        "data_stall_ratio",
        (float(cov.get("load_stall_cycles", 0)) +
         float(cov.get("store_stall_cycles", 0))) /
        max(float(cov.get("total_cycles", 1)), 1.0)
    ))
    completed_accesses = max(float(cov.get("completed_accesses", 1)), 1.0)
    avg_stall_per_access = float(cov.get("total_stall_cycles", 0)) / completed_accesses
    avg_stall_per_access_norm = min(avg_stall_per_access / 8.0, 1.0)
    max_run_norm = min(float(cov.get("max_stall_run", 0)) / 9.0, 1.0)
    long_run_freq = min(float(cov.get("stall_runs_gt4", 0)) / completed_accesses, 1.0)
    transition_norm = min(float(cov.get("transition_types_hit", 0)) / 6.0, 1.0)
    transition_entropy = float(cov.get("transition_entropy", 0.0))
    alternation_ratio = float(cov.get("load_store_alternation_ratio", 0.0))
    trap_norm = min(float(cov.get("intermediate_trap_count", 0)) / 4.0, 1.0)
    near_miss_score = float(cov.get("near_miss_score", 0.0))
    active_ratio = float(cov.get("active_ratio", 0.0))
    instr_count = float(cov.get("instr_count", 0.0))
    completed_b2b_rate = float(cov.get(
        "completed_b2b_rate",
        float(cov.get("b2b_stall_count", 0)) / completed_accesses
    ))
    burst_pressure = min(
        (float(cov.get("data_burst_count", 0)) +
         float(cov.get("mixed_burst_count", 0)) +
         float(cov.get("consecutive_mixed_bursts", 0))) / completed_accesses,
        1.0
    )
    instr_count_norm = min(instr_count / 1200.0, 1.0)
    activity_score = min(active_ratio / 0.75, 1.0)
    data_mix_score = min(
        float(cov.get("mixed_burst_count", 0)) / max(completed_accesses * 0.18, 1.0),
        1.0
    )

    stall_target = 0.40
    stall_tolerance = 0.18
    stall_balance = max(0.0, 1.0 - abs(stall_ratio - stall_target) / stall_tolerance)

    diversity_score = (
        0.30 * transition_norm +
        0.24 * transition_entropy +
        0.26 * alternation_ratio +
        0.10 * near_miss_score +
        0.10 * data_mix_score
    )
    execution_score = (
        0.36 * instr_count_norm +
        0.34 * activity_score +
        0.15 * data_mix_score +
        0.10 * burst_pressure +
        0.05 * trap_norm
    )
    stress_score = (
        0.28 * stall_balance +
        0.20 * data_stall_ratio +
        0.16 * avg_stall_per_access_norm +
        0.12 * max_run_norm +
        0.08 * long_run_freq +
        0.08 * completed_b2b_rate +
        0.08 * burst_pressure
    )

    quality_gate = min(
        diversity_score / 0.42,
        execution_score / 0.40,
        1.0,
    )
    if transition_norm < 0.34 or transition_entropy < 0.30 or alternation_ratio < 0.05:
        quality_gate *= 0.20
    if instr_count < 180 or active_ratio < 0.08:
        quality_gate *= 0.35

    gated_stress = stress_score * quality_gate

    reward = (
        0.45 * diversity_score +
        0.35 * execution_score +
        0.20 * gated_stress
    )

    if float(cov.get("transition_types_hit", 0)) <= 2:
        reward *= 0.35
    if alternation_ratio < 0.05:
        reward *= 0.35
    if transition_entropy < 0.30:
        reward *= 0.40
    if instr_count < 180:
        reward *= 0.45
    if stall_ratio > 0.45:
        reward *= max(0.20, 1.0 - (stall_ratio - 0.45) * 3.0)

    return float(min(max(reward, 0.0), 1.0))


def _base_coverage_state(cov: dict) -> np.ndarray:
    return np.array([
        float(cov.get("stall_ratio", 0.0)),
        float(cov.get("data_stall_ratio", (cov.get("load_stall_cycles", 0) + cov.get("store_stall_cycles", 0))
              / max(cov.get("total_cycles", 1), 1))),
        float(cov.get("completed_b2b_rate", cov.get("b2b_stall_count", 0)
              / max(cov.get("completed_accesses", 1), 1))),
        min(float(cov.get("max_stall_run", 0)) / 9.0, 1.0),
        min(
            0.55 * float(cov.get("transition_types_hit", 0)) / 6.0 +
            0.45 * float(cov.get("transition_entropy", 0.0)),
            1.0
        ),
        min(float(cov.get("intermediate_trap_count", 0)) / 4.0, 1.0),
        min(
            0.6 * float(cov.get("near_miss_score", 0.0)) +
            0.4 * float(cov.get("load_store_alternation_ratio", 0.0)),
            1.0
        ),
        min(float(cov.get("instr_count", 0)) / 200.0, 1.0),
    ], dtype=np.float32)


def _coverage_progress_state(bin_hit_counts: Counter) -> np.ndarray:
    feats = []
    for group_name in UNCOVERED_GROUP_NAMES[:-1]:
        bins = COVERAGE_GROUPS[group_name]
        remaining = sum(1 for b in bins if bin_hit_counts.get(b, 0) == 0)
        feats.append(remaining / max(len(bins), 1))
    total_remaining = sum(1 for b in ALL_COVERAGE_BINS if bin_hit_counts.get(b, 0) == 0)
    feats.append(total_remaining / max(len(ALL_COVERAGE_BINS), 1))
    return np.array(feats, dtype=np.float32)


def _mean_state(results: list,
                bin_hit_counts: Optional[Counter] = None,
                state_mode: str = "compact") -> np.ndarray:
    if not results:
        base = np.zeros(BASE_STATE_DIM, dtype=np.float32)
    else:
        states = [_base_coverage_state(r["coverage"]) for r in results]
        base = np.mean(states, axis=0).astype(np.float32)
    if state_mode == "legacy":
        return base
    progress = _coverage_progress_state(bin_hit_counts or Counter())
    return np.concatenate([base, progress]).astype(np.float32)
