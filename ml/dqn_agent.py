# =============================================================================
# dqn_agent.py
# Deep Q-Network agent for PicoRV32 ML-DV knob optimisation.
#
# Architecture
# ─────────────
# State  : 8-dim normalised coverage vector derived from prev_results
#          [stall_ratio, data_stall_ratio, completed_b2b_rate,
#           max_run_norm, transition_norm, intermediate_trap_norm,
#           near_miss_score, instr_count_norm]
#
# Action : not a single discrete action — instead the Q-network acts as a
#          *scorer*: we sample N_CANDIDATES random knob configs, encode each
#          as a 7-dim feature vector, concatenate with the state, and pass
#          through the Q-net to get a scalar Q-value.  We then pick the
#          top-N_SUGGEST configs by predicted Q-value.  This sidesteps the
#          combinatorially large joint action space (10^4 × 8 × 4 × 8 ≈ 25M)
#          while keeping the DQN learning signal clean.
#
# Training: experience replay over (state, knobs, reward) tuples.
#           reward = total reward from the simulation result, where the
#           balanced composite score is augmented with functional-bin novelty
#           and rarity bonuses at the experiment layer.
#
# =============================================================================

import random, os, heapq, math
from typing import List, Dict, Optional
from collections import Counter

import numpy as np

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False
    print("[dqn_agent] WARNING: torch not installed — DQNAgent will use "
          "random suggestions.  Install with: pip install torch")

# ---------------------------------------------------------------------------
# Knob space  (vals = list of legal discrete values for random.choice)
# ---------------------------------------------------------------------------
KNOB_RANGES: Dict[str, list] = {
    "load_weight":    list(range(1, 11)),   # 1 … 10
    "store_weight":   list(range(1, 11)),   # 1 … 10
    "branch_weight":  list(range(1, 11)),   # 1 … 10
    "jump_weight":    list(range(1, 11)),   # 1 … 10
    "arith_weight":   list(range(1, 11)),   # 1 … 10
    "mem_stride":     list(range(1,  9)),   # 1 … 8
    "pointer_update_rate": list(range(1, 11)),  # 1 … 10
    "trap_rate":      list(range(0,  4)),   # 0 … 3
    "trap_kind":      list(range(0,  4)),   # 0 … 3
    "branch_taken_bias": list(range(0, 11)), # 0 … 10
    "mixed_burst_bias": list(range(0, 11)),  # 0 … 10
    "mem_delay_base": list(range(1,  9)),   # 1 … 8
}

KNOB_NAMES  = list(KNOB_RANGES.keys())   # stable ordering
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
STATE_DIM   = BASE_STATE_DIM + len(UNCOVERED_GROUP_NAMES)
KNOB_DIM    = len(KNOB_NAMES)
INPUT_DIM   = STATE_DIM + KNOB_DIM      # Q-net input
HIDDEN      = 64
BATCH_SIZE  = 64
REPLAY_CAP  = 8000
LR          = 3e-4
GAMMA       = 0.7   # coverage reward is mostly short-horizon
EPS_START   = 0.9
EPS_END     = 0.25
EPS_DECAY   = 0.96  # multiplied each iteration
N_CANDIDATES = 384  # random configs scored per suggestion call
RECENT_KNOB_WINDOW = 256
RARE_ARCHIVE_CAP = 96
ARCHIVE_MUTATION_FRACTION = 0.35
TARGETED_CANDIDATE_FRACTION = 0.40
TARGETED_CANDIDATE_MIN = 48
RESIDUAL_MODE_PROGRESS = 0.90
RESIDUAL_MODE_MAX_UNCOVERED = 10
RESIDUAL_PLAN_DEPTH = 4
RESIDUAL_NODE_GENERAL = "general"
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


def _bin_focus_groups(bin_name: str) -> set:
    name = str(bin_name)
    groups = set()

    if (
        name in ("stall_depth_tiny", "stall_depth_short")
        or name.endswith("_stall_short")
    ):
        groups.add("stall_low")

    if (
        name in ("stall_type_data_dominant", "mem_mix_balanced", "mem_mix_heavy")
        or name.startswith("cross_load_")
        or name.startswith("cross_store_")
    ):
        groups.add("data_heavy")

    if name in (
        "b2b_pressure_low",
        "b2b_pressure_medium",
        "instr_window_trivial",
        "instr_window_short",
        "mem_mix_light",
    ):
        groups.add("pressure_low")

    if name in ("alternation_none", "alternation_low", "alternation_medium") or name.startswith("cross_alt_low_"):
        groups.add("alt_low")

    if name in (
        "transition_diversity_poor",
        "transition_diversity_moderate",
        "transition_entropy_low",
        "transition_entropy_medium",
        "delay_medium",
    ) or name.startswith("cross_trans_moderate_"):
        groups.add("trans_moderate")

    if (
        name.endswith("_rare")
        or name.endswith("_recurrent")
        or name in ("transition_diversity_poor", "transition_diversity_moderate")
    ):
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
        if "delay_fast" in bin_name:
            add("delay_fast")
        if bin_name == "stall_depth_tiny":
            add("stall_depth_tiny")
        if bin_name == "stall_type_data_dominant":
            add("data_heavy")
        if bin_name == "mem_mix_heavy":
            add("data_heavy")
        if bin_name == "b2b_pressure_low":
            add("pressure_low")
        if bin_name == "control_mix_high":
            add("control_high")
        if bin_name == "cross_alt_low_stall_short":
            add("alt_low_short")
        elif bin_name == "cross_alt_low_stall_medium":
            add("alt_low_medium")
        elif bin_name.startswith("cross_alt_low_"):
            add("alt_low")
        if bin_name == "cross_trans_moderate_delay_medium":
            add("trans_moderate_medium")
        if bin_name == "cross_trans_moderate_delay_fast":
            add("trans_moderate_fast")
        if bin_name == "cross_trans_moderate_delay_slow":
            add("trans_moderate_slow")
        if bin_name.startswith("cross_trans_moderate_"):
            add("trans_moderate")
        if bin_name == "transition_load_then_load_rare":
            add("load_then_load_rare")
        if bin_name == "transition_load_then_store_rare":
            add("load_then_store_rare")
        if bin_name == "transition_store_then_load_rare":
            add("store_then_load_rare")
        if bin_name == "transition_store_then_store_rare":
            add("store_then_store_rare")

        for focus_group in _bin_focus_groups(bin_name):
            add(focus_group)

    return profiles


def _target_profiles_for_bin(bin_name: str) -> List[str]:
    return _target_profiles_for_uncovered_bins([bin_name])


def _residual_closure_enabled() -> bool:
    # Temporarily disabled while validating the revised coverage-bin
    # definitions without late-stage residual targeting.
    return False


def _configured_dqn_reward_mode() -> str:
    return os.environ.get("PICORV32_DQN_REWARD_MODE", "A").strip().upper() or "A"

# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _normalise_knobs(knobs: dict) -> np.ndarray:
    """Map each knob to [0, 1] based on its range."""
    out = []
    for name in KNOB_NAMES:
        vals = KNOB_RANGES[name]
        lo, hi = vals[0], vals[-1]
        v = knobs.get(name, lo)
        out.append((v - lo) / max(hi - lo, 1))
    return np.array(out, dtype=np.float32)


def compute_reward(cov: dict) -> float:
    """Constraint-gated multi-objective reward with stall as a secondary bonus."""
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
        float(cov.get("b2b_stall_count", 0)) /
        completed_accesses
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

    # Stress helps only in a moderate band and only after the baseline
    # diversity/execution conditions are satisfied.
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


def compute_reward_b(cov: dict, target_bins: Optional[List[str]] = None) -> float:
    """
    Late-stage residual reward for sparse, lower-stress closure.

    Reward B is designed for cases where the remaining uncovered bins are
    sparse residual corners such as:
      - cross_alt_low_stall_short
      - cross_alt_low_stall_medium
      - transition_load_then_store_rare
    """
    knobs = cov.get("knobs", {}) if isinstance(cov.get("knobs", {}), dict) else {}
    targets = set(target_bins or [])
    stall_ratio = float(cov.get("stall_ratio", 0.0))
    b2b_stall_rate = float(cov.get("b2b_stall_rate", 0.0))
    ctrl_frac = (
        float(cov.get("branch_instr_frac", 0.0))
        + float(cov.get("jump_instr_frac", 0.0))
    )
    transition_types = float(cov.get("transition_types_hit", 0.0))
    transition_entropy = float(cov.get("transition_entropy", 0.0))
    alternation_ratio = float(cov.get("load_store_alternation_ratio", 0.0))
    load_then_store = int(cov.get("load_then_store", 0))
    store_then_load = int(cov.get("store_then_load", 0))
    store_then_store = int(cov.get("store_then_store", 0))
    load_instr_frac = float(cov.get("load_instr_frac", 0.0))
    store_instr_frac = float(cov.get("store_instr_frac", 0.0))
    instr_count = float(cov.get("instr_count", 0.0))
    near_miss = float(cov.get("near_miss_score", 0.0))
    max_run = float(cov.get("max_stall_run", 0.0))
    delay_knob = float(knobs.get("mem_delay_base", 0.0))
    burst_knob = float(knobs.get("mixed_burst_bias", 0.0))

    def closeness(val: float, target: float, tol: float) -> float:
        if tol <= 0:
            return 1.0 if abs(val - target) < 1e-9 else 0.0
        return max(0.0, 1.0 - abs(val - target) / tol)

    mem_mix_score = min(
        1.0,
        0.45 * closeness(load_instr_frac + store_instr_frac, 0.58, 0.20)
        + 0.25 * closeness(burst_knob, 7.0, 3.0)
        + 0.20 * closeness(delay_knob, 5.0, 2.0)
        + 0.10 * closeness(max_run, 6.0, 2.5)
    )
    trans_moderate_fast_score = min(
        1.0,
        0.34 * closeness(transition_types, 4.0, 1.0)
        + 0.20 * closeness(transition_entropy, 0.68, 0.14)
        + 0.24 * closeness(delay_knob, 1.8, 0.9)
        + 0.12 * closeness(max_run, 2.6, 1.0)
        + 0.10 * closeness(alternation_ratio, 0.85, 0.35)
    )
    trans_moderate_slow_score = min(
        1.0,
        0.34 * closeness(transition_types, 4.0, 1.0)
        + 0.24 * closeness(transition_entropy, 0.72, 0.12)
        + 0.22 * closeness(delay_knob, 7.0, 1.2)
        + 0.10 * closeness(max_run, 8.0, 1.5)
        + 0.10 * closeness(b2b_stall_rate, 0.10, 0.08)
    )

    control_high_score = max(
        closeness(ctrl_frac, 0.45, 0.12),
        closeness(ctrl_frac, 0.52, 0.18),
    )
    lts_rare_score = 1.0 if load_then_store == 1 else (
        0.70 if load_then_store == 2 else
        0.35 if load_then_store == 0 and load_instr_frac > 0.02 and store_instr_frac > 0.02 else
        0.20 if load_then_store <= 4 else
        0.0
    )
    stl_rare_score = 1.0 if store_then_load == 1 else (
        0.72 if store_then_load == 2 else
        0.35 if store_then_load == 0 and store_instr_frac > 0.03 and load_instr_frac > 0.02 else
        0.20 if store_then_load <= 4 else
        0.0
    )
    store_store_rare_score = 1.0 if store_then_store == 1 else (
        0.70 if store_then_store == 2 else
        0.25 if store_then_store <= 4 else
        0.0
    )
    alt_low_short_score = min(
        1.0,
        0.38 * closeness(alternation_ratio, 0.20, 0.18)
        + 0.28 * closeness(max_run, 2.0, 0.8)
        + 0.18 * closeness(delay_knob, 1.0, 0.35)
        + 0.10 * closeness(burst_knob, 0.0, 0.6)
        + 0.06 * closeness(transition_types, 3.5, 1.5),
    )
    alt_low_medium_score = min(
        1.0,
        0.45 * closeness(alternation_ratio, 0.28, 0.18)
        + 0.30 * closeness(max_run, 4.0, 1.5)
        + 0.15 * closeness(delay_knob, 4.0, 1.0)
        + 0.10 * closeness(burst_knob, 0.0, 2.0),
    )
    trans_moderate_medium_score = min(
        1.0,
        0.34 * closeness(transition_types, 4.0, 1.0)
        + 0.22 * closeness(transition_entropy, 0.70, 0.12)
        + 0.18 * closeness(delay_knob, 4.0, 1.0)
        + 0.14 * closeness(load_then_store, 3.0, 2.0)
        + 0.12 * closeness(stall_ratio, 0.08, 0.10),
    )
    sparse_transition_score = min(
        1.0,
        0.55 * closeness(transition_types, 4.0, 2.0)
        + 0.25 * closeness(transition_entropy, 0.62, 0.22)
        + 0.20 * closeness(alternation_ratio, 0.35, 0.25),
    )
    low_stress_score = min(
        1.0,
        0.60 * closeness(stall_ratio, 0.10, 0.12)
        + 0.40 * closeness(b2b_stall_rate, 0.14, 0.12),
    )
    execution_score = min(instr_count / 120.0, 1.0)

    corner_primary = max(
        control_high_score,
        lts_rare_score,
        stl_rare_score,
        alt_low_short_score,
        alt_low_medium_score,
        trans_moderate_fast_score,
        trans_moderate_medium_score,
        trans_moderate_slow_score,
        mem_mix_score,
        store_store_rare_score,
    )
    corner_secondary = max(
        min(control_high_score, lts_rare_score),
        min(stl_rare_score, trans_moderate_fast_score),
        min(alt_low_short_score, sparse_transition_score),
        min(alt_low_medium_score, trans_moderate_medium_score),
        min(mem_mix_score, trans_moderate_slow_score),
        min(store_store_rare_score, trans_moderate_medium_score),
    )

    if targets == {"control_mix_high"}:
        reward = (
            0.54 * control_high_score
            + 0.16 * low_stress_score
            + 0.12 * sparse_transition_score
            + 0.10 * execution_score
            + 0.08 * near_miss
        )
        if control_high_score < 0.42:
            reward *= 0.20
        if ctrl_frac < 0.38:
            reward *= 0.30
        if delay_knob > 2.5:
            reward *= 0.55
        if burst_knob > 2.0:
            reward *= 0.65
        return float(min(max(reward, 0.0), 1.0))

    if targets == {"transition_load_then_store_rare"}:
        reward = (
            0.56 * lts_rare_score
            + 0.18 * sparse_transition_score
            + 0.12 * execution_score
            + 0.14 * near_miss
        )
        if lts_rare_score < 0.45:
            reward *= 0.22
        if burst_knob > 1.5:
            reward *= 0.50
        if not (3.0 <= delay_knob <= 5.0 or abs(delay_knob - 8.0) <= 0.25):
            reward *= 0.65
        if store_instr_frac > 0.18:
            reward *= 0.45
        return float(min(max(reward, 0.0), 1.0))

    if targets == {"transition_store_then_load_rare"}:
        reward = (
            0.54 * stl_rare_score
            + 0.16 * sparse_transition_score
            + 0.12 * execution_score
            + 0.10 * near_miss
            + 0.08 * closeness(delay_knob, 4.0, 2.0)
        )
        if stl_rare_score < 0.45:
            reward *= 0.22
        if load_instr_frac >= store_instr_frac:
            reward *= 0.50
        if not (2.0 <= delay_knob <= 6.0):
            reward *= 0.60
        return float(min(max(reward, 0.0), 1.0))

    if targets == {"cross_trans_moderate_delay_fast"}:
        reward = (
            0.58 * trans_moderate_fast_score
            + 0.14 * sparse_transition_score
            + 0.12 * execution_score
            + 0.08 * near_miss
            + 0.08 * low_stress_score
        )
        if trans_moderate_fast_score < 0.42:
            reward *= 0.22
        if not (1.0 <= delay_knob <= 2.0):
            reward *= 0.40
        if not (2.0 <= max_run <= 3.5):
            reward *= 0.60
        return float(min(max(reward, 0.0), 1.0))

    if targets == {"cross_trans_moderate_delay_slow"}:
        reward = (
            0.58 * trans_moderate_slow_score
            + 0.16 * sparse_transition_score
            + 0.14 * execution_score
            + 0.12 * near_miss
        )
        if trans_moderate_slow_score < 0.42:
            reward *= 0.22
        if not (6.0 <= delay_knob <= 8.0):
            reward *= 0.35
        if max_run < 7.0:
            reward *= 0.50
        return float(min(max(reward, 0.0), 1.0))

    if targets == {"mem_mix_heavy", "cross_alt_low_stall_short", "cross_trans_moderate_delay_slow"}:
        reward = (
            0.26 * mem_mix_score
            + 0.30 * alt_low_short_score
            + 0.24 * trans_moderate_slow_score
            + 0.10 * execution_score
            + 0.10 * near_miss
        )
        return float(min(max(reward, 0.0), 1.0))

    if targets == {"cross_alt_low_stall_short"}:
        reward = (
            0.64 * alt_low_short_score
            + 0.14 * sparse_transition_score
            + 0.12 * execution_score
            + 0.10 * near_miss
        )
        if alt_low_short_score < 0.45:
            reward *= 0.18
        if abs(delay_knob - 1.0) > 0.1:
            reward *= 0.45
        if burst_knob > 0.5:
            reward *= 0.35
        if not (1.5 <= max_run <= 3.0):
            reward *= 0.40
        if alternation_ratio > 0.35:
            reward *= 0.45
        return float(min(max(reward, 0.0), 1.0))

    if targets == {"cross_alt_low_stall_medium"}:
        reward = (
            0.60 * alt_low_medium_score
            + 0.18 * sparse_transition_score
            + 0.12 * execution_score
            + 0.10 * near_miss
        )
        if alt_low_medium_score < 0.40:
            reward *= 0.25
        if not (2.0 <= delay_knob <= 4.0):
            reward *= 0.70
        if burst_knob > 2.0:
            reward *= 0.55
        if not (3.0 <= max_run <= 5.0):
            reward *= 0.45
        return float(min(max(reward, 0.0), 1.0))

    reward = (
        0.42 * corner_primary
        + 0.16 * corner_secondary
        + 0.18 * sparse_transition_score
        + 0.14 * low_stress_score
        + 0.10 * execution_score
    )
    reward += 0.08 * near_miss

    if max(
        control_high_score,
        lts_rare_score,
        stl_rare_score,
        alt_low_short_score,
        alt_low_medium_score,
        trans_moderate_fast_score,
        trans_moderate_medium_score,
        trans_moderate_slow_score,
        mem_mix_score,
        store_store_rare_score,
    ) < 0.35:
        reward *= 0.30
    if load_then_store > 4 and trans_moderate_medium_score < 0.40:
        reward *= 0.35
    if store_then_load > 4 and stl_rare_score < 0.40:
        reward *= 0.35
    if store_then_store > 4 and store_store_rare_score < 0.40:
        reward *= 0.45
    if instr_count < 30:
        reward *= 0.40

    return float(min(max(reward, 0.0), 1.0))


def _selected_reward(cov: dict,
                     reward_mode: str = "A",
                     target_bins: Optional[List[str]] = None) -> float:
    if str(reward_mode).upper() == "B":
        return compute_reward_b(cov, target_bins=target_bins)
    return compute_reward(cov)


def _base_coverage_state(cov: dict) -> np.ndarray:
    return np.array([
        float(cov.get("stall_ratio",        0.0)),
        float(cov.get("data_stall_ratio",   (cov.get("load_stall_cycles", 0) +
                                             cov.get("store_stall_cycles", 0))
                                             / max(cov.get("total_cycles", 1), 1))),
        float(cov.get("completed_b2b_rate", cov.get("b2b_stall_count", 0)
                                             / max(cov.get("completed_accesses", 1), 1))),
        min(float(cov.get("max_stall_run",  0)) / 9.0, 1.0),
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
        min(float(cov.get("instr_count",    0)) / 200.0, 1.0),
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
    """Average coverage state across a batch with optional coverage-progress summary."""
    if not results:
        base = np.zeros(BASE_STATE_DIM, dtype=np.float32)
    else:
        states = [_base_coverage_state(r["coverage"]) for r in results]
        base = np.mean(states, axis=0).astype(np.float32)
    if state_mode == "legacy":
        return base
    progress = _coverage_progress_state(bin_hit_counts or Counter())
    return np.concatenate([base, progress]).astype(np.float32)


def _random_knobs() -> dict:
    return {name: random.choice(vals) for name, vals in KNOB_RANGES.items()}

# ---------------------------------------------------------------------------
# Q-network
# ---------------------------------------------------------------------------

class _QNet(nn.Module):
    def __init__(self, input_dim: int = INPUT_DIM):
        super().__init__()
        self.input_dim = int(input_dim)
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)   # → scalar per sample


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------

class _ReplayBuffer:
    def __init__(self, capacity=REPLAY_CAP):
        self.buf = []
        self.cap = capacity

    def push(self, state, knobs_norm, reward, next_state):
        if len(self.buf) >= self.cap:
            self.buf.pop(0)
        self.buf.append((state, knobs_norm, reward, next_state))

    def sample(self, n):
        return random.sample(self.buf, min(n, len(self.buf)))

    def __len__(self):
        return len(self.buf)

    def state_dict(self):
        return {
            "capacity": self.cap,
            "buf": [
                (
                    np.asarray(state, dtype=np.float32).tolist(),
                    np.asarray(knobs_norm, dtype=np.float32).tolist(),
                    float(reward),
                    np.asarray(next_state, dtype=np.float32).tolist(),
                )
                for state, knobs_norm, reward, next_state in self.buf
            ],
        }

    def load_state_dict(self, state):
        self.cap = int(state.get("capacity", REPLAY_CAP))
        self.buf = [
            (
                np.asarray(item[0], dtype=np.float32),
                np.asarray(item[1], dtype=np.float32),
                float(item[2]),
                np.asarray(item[3], dtype=np.float32) if len(item) > 3 else np.asarray(item[0], dtype=np.float32),
            )
            for item in state.get("buf", [])
        ]

# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

class DQNAgent:
    """
    DQN-based knob optimisation agent.

    Fits a Q(state, knobs) → total_reward network.  At each iteration it:
      1. Encodes the previous iteration's mean coverage as the state.
      2. Samples N_CANDIDATES random knob configurations.
      3. Scores each with the Q-net.
      4. Returns the top-n_suggest configurations (ε-greedy: some are random).
      5. Trains the Q-net on accumulated experience replay.
    """

    def __init__(self):
        self.epsilon  = EPS_START
        self.replay   = _ReplayBuffer()
        self.iteration = 0
        self.recent_knob_keys: List[tuple] = []
        self.rare_hit_archive: List[dict] = []
        self.bin_hit_counts: Counter = Counter()
        self.state_mode = "compact"
        self.state_dim = STATE_DIM
        self.input_dim = INPUT_DIM
        self.pending_state: Optional[np.ndarray] = None
        self.pending_knobs_norms: List[np.ndarray] = []
        self.pending_action_profiles: List[str] = []
        self.current_residual_node: str = RESIDUAL_NODE_GENERAL
        self.residual_graph: Dict[tuple, Dict[str, float]] = {}
        self.runtime_reward_mode: str = "A"

        if _TORCH_OK:
            self._rebuild_networks(self.input_dim)
            self.loss_fn   = nn.SmoothL1Loss()
        else:
            self.qnet = None

    # ── public interface ─────────────────────────────────────────────────────

    def suggest_knobs_for_iteration(self,
                                    prev_results: list,
                                    n_suggest: int = 10) -> List[dict]:
        """
        Given the results of the previous iteration, return n_suggest knob
        configs predicted to maximise the total reward.
        """
        # 1. Finalise the previous iteration's transitions with the newly
        # observed next-state, then train on those transitions.
        if prev_results:
            self._ingest(prev_results)
            self._train()

        # 2. Current state = mean coverage of prev iteration
        state = self._state_from_results(prev_results)

        # 3. Generate candidates and score them
        suggestions = []
        suggestion_keys = set()
        n_greedy = max(1, int(n_suggest * (1.0 - self.epsilon)))
        n_random  = n_suggest - n_greedy
        reward_mode = self._active_reward_mode()

        if self.qnet is not None and len(self.replay) >= BATCH_SIZE:
            candidates  = self._sample_candidate_pool(N_CANDIDATES * 3)
            knobs_norms = np.stack([_normalise_knobs(k) for k in candidates])
            state_rep   = np.tile(state, (len(candidates), 1))
            inp         = np.concatenate([state_rep, knobs_norms], axis=1)

            with torch.no_grad():
                t    = torch.FloatTensor(inp)
                qvals = self.qnet(t).numpy()

            top_idx = np.argsort(qvals)[::-1]
            for i in top_idx:
                key = self._knob_key(candidates[i])
                if key in suggestion_keys:
                    continue
                suggestions.append(candidates[i])
                suggestion_keys.add(key)
                if len(suggestions) >= n_greedy:
                    break
        else:
            # Not enough data yet — fall back to random for all slots
            n_random = n_suggest

        # 4. Fill remaining slots with random exploration
        for _ in range(n_random):
            knobs = self._random_unseen_knobs(suggestion_keys)
            suggestions.append(knobs)
            suggestion_keys.add(self._knob_key(knobs))

        # 5. Decay epsilon
        self.epsilon = max(EPS_END, self.epsilon * EPS_DECAY)
        self.iteration += 1
        self._remember_knobs(suggestions)
        self.pending_state = np.asarray(state, dtype=np.float32).copy()
        self.pending_knobs_norms = [
            np.asarray(_normalise_knobs(k), dtype=np.float32).copy()
            for k in suggestions[:n_suggest]
        ]
        self.pending_action_profiles = [
            self._profile_for_knobs(k)
            for k in suggestions[:n_suggest]
        ]

        return suggestions[:n_suggest]

    def suggest_knobs_frozen(self,
                             prev_results: list,
                             n_suggest: int = 10) -> List[dict]:
        """
        Score candidates with the current Q-network without ingesting or
        training on new results.
        """
        state = self._state_from_results(prev_results)

        if self.qnet is not None and len(self.replay) >= BATCH_SIZE:
            candidates = self._sample_candidate_pool(N_CANDIDATES * 3)
            knobs_norms = np.stack([_normalise_knobs(k) for k in candidates])
            state_rep = np.tile(state, (len(candidates), 1))
            inp = np.concatenate([state_rep, knobs_norms], axis=1)

            with torch.no_grad():
                qvals = self.qnet(torch.FloatTensor(inp)).numpy()

            suggestions = []
            suggestion_keys = set()
            top_idx = np.argsort(qvals)[::-1]
            for i in top_idx:
                key = self._knob_key(candidates[i])
                if key in suggestion_keys:
                    continue
                suggestions.append(candidates[i])
                suggestion_keys.add(key)
                if len(suggestions) >= n_suggest:
                    break
        else:
            suggestions = []
            suggestion_keys = set()
            for _ in range(n_suggest):
                knobs = self._random_unseen_knobs(suggestion_keys)
                suggestions.append(knobs)
                suggestion_keys.add(self._knob_key(knobs))

        return suggestions[:n_suggest]

    def save(self, path: str):
        if not _TORCH_OK or self.qnet is None:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "qnet":     self.qnet.state_dict(),
            "target":   self.target.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "epsilon":  self.epsilon,
            "iteration": self.iteration,
            "replay":   self.replay.state_dict(),
            "recent_knob_keys": self.recent_knob_keys,
            "rare_hit_archive": self.rare_hit_archive,
            "bin_hit_counts": dict(self.bin_hit_counts),
            "state_mode": self.state_mode,
            "state_dim": self.state_dim,
            "input_dim": self.input_dim,
            "pending_state": None if self.pending_state is None else np.asarray(self.pending_state, dtype=np.float32).tolist(),
            "pending_knobs_norms": [np.asarray(k, dtype=np.float32).tolist() for k in self.pending_knobs_norms],
            "pending_action_profiles": list(self.pending_action_profiles),
            "current_residual_node": self.current_residual_node,
            "residual_graph": [
                {
                    "src": src,
                    "action": action,
                    "dst": dst,
                    **stats,
                }
                for (src, action, dst), stats in self.residual_graph.items()
            ],
        }, path)
        print(f"  [DQN] Model saved → {path}")

    def load(self, path: str):
        if not _TORCH_OK or not os.path.exists(path):
            return
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            ckpt = torch.load(path, map_location="cpu")
        input_dim = int(ckpt.get("input_dim", 0) or 0)
        if not input_dim:
            first_weight = ckpt.get("qnet", {}).get("net.0.weight")
            if first_weight is not None:
                input_dim = int(first_weight.shape[1])
        if input_dim and input_dim != self.input_dim:
            self._set_state_layout_from_input_dim(input_dim)
            self._rebuild_networks(self.input_dim)
        self.qnet.load_state_dict(ckpt["qnet"])
        self.target.load_state_dict(ckpt["target"])
        if "optimiser" in ckpt:
            try:
                self.optimiser.load_state_dict(ckpt["optimiser"])
            except ValueError:
                pass
        self.epsilon  = ckpt.get("epsilon",  EPS_END)
        self.iteration = ckpt.get("iteration", 0)
        if "replay" in ckpt:
            self.replay.load_state_dict(ckpt["replay"])
        self.recent_knob_keys = [tuple(item) for item in ckpt.get("recent_knob_keys", [])]
        self.rare_hit_archive = list(ckpt.get("rare_hit_archive", []))
        self.bin_hit_counts = Counter(ckpt.get("bin_hit_counts", {}))
        self.state_mode = ckpt.get("state_mode", self.state_mode)
        self.state_dim = int(ckpt.get("state_dim", self.state_dim))
        self.input_dim = int(ckpt.get("input_dim", self.input_dim))
        pending_state = ckpt.get("pending_state")
        self.pending_state = None if pending_state is None else np.asarray(pending_state, dtype=np.float32)
        self.pending_knobs_norms = [
            np.asarray(item, dtype=np.float32) for item in ckpt.get("pending_knobs_norms", [])
        ]
        self.pending_action_profiles = [str(item) for item in ckpt.get("pending_action_profiles", [])]
        self.current_residual_node = str(ckpt.get("current_residual_node", RESIDUAL_NODE_GENERAL))
        self.residual_graph = {}
        for item in ckpt.get("residual_graph", []):
            key = (str(item.get("src", RESIDUAL_NODE_GENERAL)),
                   str(item.get("action", RESIDUAL_NODE_GENERAL)),
                   str(item.get("dst", RESIDUAL_NODE_GENERAL)))
            self.residual_graph[key] = {
                "count": float(item.get("count", 0.0)),
                "reward_sum": float(item.get("reward_sum", 0.0)),
                "novel_sum": float(item.get("novel_sum", 0.0)),
                "near_miss_sum": float(item.get("near_miss_sum", 0.0)),
            }
        print(f"  [DQN] Model loaded <- {path}  (state_mode={self.state_mode}, input_dim={self.input_dim})")

    # ── private ──────────────────────────────────────────────────────────────

    def _ingest(self, results: list):
        """Finalise the previous decision batch into replay transitions."""
        uncovered_bins = self._current_uncovered_bins()
        current_reward_mode = self._active_reward_mode()
        for r in results:
            cov = r.get("coverage", {})
            if isinstance(cov, dict):
                cov["dqn_reward_mode_active"] = current_reward_mode
                cov["dqn_reward_mode_targets"] = list(uncovered_bins)
        next_state = self._state_from_results(results)
        if self.pending_state is not None and self.pending_knobs_norms:
            pending_profiles = self.pending_action_profiles or [RESIDUAL_NODE_GENERAL] * len(self.pending_knobs_norms)
            for knobs_norm, action_profile, r in zip(self.pending_knobs_norms, pending_profiles, results):
                reward = float(
                    _selected_reward(r["coverage"], current_reward_mode, target_bins=uncovered_bins)
                    if current_reward_mode == "B"
                    else r["coverage"].get(
                        "total_reward",
                        r["coverage"].get("composite_reward", compute_reward(r["coverage"]))
                    )
                )
                self.replay.push(
                    np.asarray(self.pending_state, dtype=np.float32),
                    np.asarray(knobs_norm, dtype=np.float32),
                    reward,
                    np.asarray(next_state, dtype=np.float32),
                )
                self._update_residual_graph(self.current_residual_node, action_profile, r)

        for r in results:
            reward = float(
                _selected_reward(r["coverage"], current_reward_mode, target_bins=uncovered_bins)
                if current_reward_mode == "B"
                else r["coverage"].get(
                    "total_reward",
                    r["coverage"].get("composite_reward", compute_reward(r["coverage"]))
                )
            )
            if self._tracks_ml_progress(r):
                rare_hits = list(r["coverage"].get("remaining_bin_hits", []))
                if rare_hits:
                    self._remember_rare_hit(r["knobs"], rare_hits, reward)
                for b in r["coverage"].get("functional_bins", []):
                    if b in ALL_COVERAGE_BINS:
                        self.bin_hit_counts[b] += 1
        self._remember_knobs([r["knobs"] for r in results])
        self.current_residual_node = self._residual_node_from_results(results)
        self.pending_action_profiles = []

    @staticmethod
    def _knob_key(knobs: dict) -> tuple:
        return tuple(sorted(knobs.items()))

    def _remember_knobs(self, knobs_list: List[dict]):
        for knobs in knobs_list:
            self.recent_knob_keys.append(self._knob_key(knobs))
        if len(self.recent_knob_keys) > RECENT_KNOB_WINDOW:
            self.recent_knob_keys = self.recent_knob_keys[-RECENT_KNOB_WINDOW:]

    @staticmethod
    def _tracks_ml_progress(result: dict) -> bool:
        label = str(result.get("label", ""))
        return label.startswith("ml_iter")

    def _remember_rare_hit(self, knobs: dict, hits: List[str], reward: float):
        key = self._knob_key(knobs)
        hit_set = sorted(set(hits))
        for entry in self.rare_hit_archive:
            if tuple(entry.get("key", [])) == key:
                merged = sorted(set(entry.get("hits", [])) | set(hit_set))
                entry["hits"] = merged
                entry["reward"] = max(float(entry.get("reward", 0.0)), float(reward))
                return
        self.rare_hit_archive.append({
            "key": list(key),
            "knobs": dict(knobs),
            "hits": hit_set,
            "reward": float(reward),
        })
        self.rare_hit_archive.sort(key=lambda e: (len(e.get("hits", [])), e.get("reward", 0.0)), reverse=True)
        if len(self.rare_hit_archive) > RARE_ARCHIVE_CAP:
            self.rare_hit_archive = self.rare_hit_archive[:RARE_ARCHIVE_CAP]

    def _state_from_results(self, results: list) -> np.ndarray:
        return _mean_state(results, self.bin_hit_counts, state_mode=self.state_mode)

    def _current_uncovered_bins(self) -> List[str]:
        return [
            b for b in ALL_COVERAGE_BINS
            if self.bin_hit_counts.get(b, 0) == 0
        ]

    def _reward_b_auto_ready(self, uncovered_bins: Optional[List[str]] = None) -> bool:
        uncovered = list(uncovered_bins) if uncovered_bins is not None else self._current_uncovered_bins()
        if not uncovered:
            return False
        if len(uncovered) > len(REWARD_B_TARGET_BINS):
            return False
        return set(uncovered).issubset(REWARD_B_TARGET_BINS)

    def _active_reward_mode(self, uncovered_bins: Optional[List[str]] = None) -> str:
        configured = _configured_dqn_reward_mode()
        if configured == "B":
            mode = "B"
        elif configured == "AUTO" and self._reward_b_auto_ready(uncovered_bins):
            mode = "B"
        else:
            mode = "A"
        self.runtime_reward_mode = mode
        return mode

    def _set_state_layout_from_input_dim(self, input_dim: int) -> None:
        inferred_state_dim = int(input_dim) - KNOB_DIM
        if inferred_state_dim == LEGACY_STATE_DIM:
            self.state_mode = "legacy"
            self.state_dim = LEGACY_STATE_DIM
        elif inferred_state_dim == STATE_DIM:
            self.state_mode = "compact"
            self.state_dim = STATE_DIM
        else:
            raise RuntimeError(
                f"Unsupported DQN checkpoint input_dim={input_dim}; "
                f"expected {KNOB_DIM + LEGACY_STATE_DIM} or {KNOB_DIM + STATE_DIM}"
            )
        self.input_dim = int(input_dim)

    def _rebuild_networks(self, input_dim: int) -> None:
        if not _TORCH_OK:
            return
        self.input_dim = int(input_dim)
        self.qnet = _QNet(self.input_dim)
        self.target = _QNet(self.input_dim)
        self.target.load_state_dict(self.qnet.state_dict())
        self.optimiser = optim.Adam(self.qnet.parameters(), lr=LR, weight_decay=1e-5)

    def _mutate_knobs(self, knobs: dict, n_changes: int = 2) -> dict:
        out = dict(knobs)
        names = random.sample(KNOB_NAMES, k=min(n_changes, len(KNOB_NAMES)))
        for name in names:
            choices = [v for v in KNOB_RANGES[name] if v != out[name]]
            if choices:
                out[name] = random.choice(choices)
        return out

    def _profile_for_knobs(self, knobs: dict) -> str:
        load = int(knobs.get("load_weight", 1))
        store = int(knobs.get("store_weight", 1))
        branch = int(knobs.get("branch_weight", 1))
        jump = int(knobs.get("jump_weight", 1))
        arith = int(knobs.get("arith_weight", 1))
        ptr = int(knobs.get("pointer_update_rate", 1))
        delay = int(knobs.get("mem_delay_base", 1))
        burst = int(knobs.get("mixed_burst_bias", 0))

        if delay <= 1 and burst <= 1 and abs(load - store) >= 6:
            return "alt_low_short"
        if delay <= 2 and burst <= 1 and ptr <= 2 and store <= 2 and (branch + jump) >= 9:
            return "control_high"
        if delay <= 2 and burst <= 4:
            return "stall_depth_tiny"
        if load >= 7 and store >= 6 and branch <= 4 and jump <= 3:
            return "data_heavy"
        if load <= 4 and store <= 4 and burst <= 3 and (arith + branch) >= 10:
            return "pressure_low"
        if abs(load - store) >= 5:
            return "alt_low"
        if burst <= 4 and 3 <= load <= 6 and 3 <= store <= 6 and 2 <= branch <= 5:
            return "trans_moderate"
        if burst <= 2 and (abs(load - store) >= 3 or max(load, store) >= 8):
            return "transition_sparse"
        return RESIDUAL_NODE_GENERAL

    def _residual_node_from_bins(self, bins: List[str]) -> str:
        counts = Counter()
        for b in bins:
            for profile in _target_profiles_for_bin(b):
                counts[profile] += 1
        if not counts:
            return RESIDUAL_NODE_GENERAL
        best = counts.most_common()
        top_count = best[0][1]
        candidates = [name for name, count in best if count == top_count]
        priority = [
            "control_high",
            "alt_low_short",
            "alt_low_medium",
            "stall_depth_tiny",
            "data_heavy",
            "pressure_low",
            "trans_moderate_fast",
            "trans_moderate_medium",
            "trans_moderate_slow",
            "alt_low",
            "trans_moderate",
            "transition_sparse",
            "load_then_load_rare",
            "load_then_store_rare",
            "store_then_load_rare",
            "store_then_store_rare",
            "delay_fast",
            "stall_low",
        ]
        for p in priority:
            if p in candidates:
                return p
        return sorted(candidates)[0]

    def _residual_node_from_results(self, results: list) -> str:
        counts = Counter()
        for r in results:
            bins = [b for b in r.get("coverage", {}).get("functional_bins", []) if b in ALL_COVERAGE_BINS]
            counts[self._residual_node_from_bins(bins)] += 1
        if not counts:
            return self.current_residual_node or RESIDUAL_NODE_GENERAL
        return counts.most_common(1)[0][0]

    def _update_residual_graph(self,
                               src_node: str,
                               action_profile: str,
                               result: dict) -> None:
        bins = [b for b in result.get("coverage", {}).get("functional_bins", []) if b in ALL_COVERAGE_BINS]
        dst_node = self._residual_node_from_bins(bins)
        cov = result.get("coverage", {})
        key = (str(src_node or RESIDUAL_NODE_GENERAL),
               str(action_profile or RESIDUAL_NODE_GENERAL),
               str(dst_node or RESIDUAL_NODE_GENERAL))
        stats = self.residual_graph.setdefault(key, {
            "count": 0.0,
            "reward_sum": 0.0,
            "novel_sum": 0.0,
            "near_miss_sum": 0.0,
        })
        stats["count"] += 1.0
        stats["reward_sum"] += float(
            _selected_reward(cov, self.runtime_reward_mode, target_bins=self._current_uncovered_bins())
            if self.runtime_reward_mode == "B"
            else cov.get("total_reward", cov.get("composite_reward", compute_reward(cov)))
        )
        stats["novel_sum"] += float(len(cov.get("remaining_bin_hits", [])))
        stats["near_miss_sum"] += float(
            cov.get("targeted_near_miss_bonus", cov.get("near_miss_score", 0.0))
        )

    def _residual_mode_active(self) -> bool:
        if not _residual_closure_enabled():
            return False
        total_bins = max(len(ALL_COVERAGE_BINS), 1)
        uncovered = sum(1 for b in ALL_COVERAGE_BINS if self.bin_hit_counts.get(b, 0) == 0)
        covered_ratio = 1.0 - (float(uncovered) / float(total_bins))
        return covered_ratio >= RESIDUAL_MODE_PROGRESS or uncovered <= RESIDUAL_MODE_MAX_UNCOVERED

    def _residual_action_plan(self, uncovered_bins: List[str]) -> List[str]:
        target_nodes = _target_profiles_for_uncovered_bins(uncovered_bins)
        if not target_nodes:
            return []
        if not self._residual_mode_active():
            return target_nodes

        start = self.current_residual_node or RESIDUAL_NODE_GENERAL
        adjacency: Dict[str, List[tuple]] = {}
        for (src, action, dst), stats in self.residual_graph.items():
            count = max(float(stats.get("count", 0.0)), 1.0)
            avg_reward = float(stats.get("reward_sum", 0.0)) / count
            avg_novel = float(stats.get("novel_sum", 0.0)) / count
            avg_near = float(stats.get("near_miss_sum", 0.0)) / count
            success = min(1.0, 0.35 * avg_reward + 0.35 * min(avg_novel, 1.0) + 0.30 * min(avg_near, 1.0))
            cost = max(0.05, 1.0 - success + 0.10 / math.sqrt(count))
            adjacency.setdefault(src, []).append((cost, action, dst))

        if start not in adjacency:
            return target_nodes

        heap = [(0.0, start, [])]
        seen_cost = {}
        best_plan = None
        best_cost = None
        target_set = set(target_nodes)

        while heap:
            cost, node, plan = heapq.heappop(heap)
            if node in target_set and plan:
                best_plan = plan
                best_cost = cost
                break
            if len(plan) >= RESIDUAL_PLAN_DEPTH:
                continue
            if cost >= seen_cost.get((node, len(plan)), float("inf")):
                continue
            seen_cost[(node, len(plan))] = cost
            for edge_cost, action, dst in adjacency.get(node, []):
                next_cost = cost + edge_cost
                heapq.heappush(heap, (next_cost, dst, plan + [action]))

        if best_plan:
            ordered = []
            seen = set()
            for action in best_plan + target_nodes:
                if action not in seen:
                    seen.add(action)
                    ordered.append(action)
            return ordered
        return target_nodes

    def _targeted_knobs(self, profile: str) -> dict:
        knobs = _random_knobs()
        profile = str(profile)

        if profile == "alt_low_short":
            variant = random.choice(["load_heavy", "store_heavy", "balanced", "control_heavy"])
            if variant == "load_heavy":
                knobs["load_weight"] = random.choice([8, 9, 10])
                knobs["store_weight"] = random.choice([1, 2, 3])
                knobs["branch_weight"] = random.choice([6, 7, 8, 9, 10])
                knobs["jump_weight"] = random.choice([5, 6, 7, 8, 9])
                knobs["arith_weight"] = random.choice([1, 2, 3, 4, 5])
            elif variant == "store_heavy":
                knobs["load_weight"] = random.choice([1, 2, 3])
                knobs["store_weight"] = random.choice([8, 9, 10])
                knobs["branch_weight"] = random.choice([5, 6, 7, 8])
                knobs["jump_weight"] = random.choice([5, 6, 7, 8, 9])
                knobs["arith_weight"] = random.choice([4, 5, 6, 7])
            elif variant == "balanced":
                knobs["load_weight"] = random.choice([4, 5, 6, 7])
                knobs["store_weight"] = random.choice([4, 5, 6, 7])
                knobs["branch_weight"] = random.choice([6, 7, 8, 9])
                knobs["jump_weight"] = random.choice([5, 6, 7, 8, 9])
                knobs["arith_weight"] = random.choice([4, 5, 6])
            else:
                knobs["load_weight"] = random.choice([2, 4, 6, 9])
                knobs["store_weight"] = random.choice([1, 3, 6, 10])
                knobs["branch_weight"] = random.choice([7, 8, 9, 10])
                knobs["jump_weight"] = random.choice([7, 8, 9, 10])
                knobs["arith_weight"] = random.choice([2, 3, 4, 5])
            knobs["mem_stride"] = random.choice([1, 2, 3, 4, 5, 6, 8])
            knobs["mem_delay_base"] = 1
            knobs["mixed_burst_bias"] = random.choice([0, 0, 0, 1])
            knobs["pointer_update_rate"] = random.choice([3, 4, 5, 6, 8, 9])
            knobs["trap_rate"] = random.choice([0, 0, 1, 2])
            knobs["trap_kind"] = random.choice([0, 1, 3])
            knobs["branch_taken_bias"] = random.choice([0, 1, 2, 3, 4, 5])

        if profile == "control_high":
            # Derived from the random fresh_6 hits that reached control_mix_high:
            # low store pressure, low delay/burst, low pointer churn, and
            # branch+jump-heavy control flow.
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

        if profile == "alt_low_medium":
            variant = random.choice(["balanced", "load_heavy", "store_heavy", "control_heavy"])
            if variant == "balanced":
                knobs["load_weight"] = random.choice([3, 4, 5, 6, 7, 8])
                knobs["store_weight"] = random.choice([3, 4, 5, 6, 7, 8])
                knobs["branch_weight"] = random.choice([4, 5, 6, 7, 8, 9])
                knobs["jump_weight"] = random.choice([2, 3, 4])
                knobs["arith_weight"] = random.choice([3, 4, 6, 8, 10])
            elif variant == "load_heavy":
                knobs["load_weight"] = random.choice([4, 5, 7, 10])
                knobs["store_weight"] = random.choice([1, 2, 3])
                knobs["branch_weight"] = random.choice([6, 8, 9, 10])
                knobs["jump_weight"] = random.choice([2, 4, 5, 9])
                knobs["arith_weight"] = random.choice([4, 6, 10])
            elif variant == "store_heavy":
                knobs["load_weight"] = random.choice([1, 2, 3])
                knobs["store_weight"] = random.choice([7, 8, 9, 10])
                knobs["branch_weight"] = random.choice([1, 2, 8, 9])
                knobs["jump_weight"] = random.choice([4, 5, 9])
                knobs["arith_weight"] = random.choice([2, 4, 6])
            else:
                knobs["load_weight"] = random.choice([1, 2, 4, 5, 10])
                knobs["store_weight"] = random.choice([1, 3, 4, 9])
                knobs["branch_weight"] = random.choice([8, 9, 10])
                knobs["jump_weight"] = random.choice([2, 4, 5, 9])
                knobs["arith_weight"] = random.choice([3, 4, 6, 10])
            knobs["mem_stride"] = random.choice([1, 2, 3, 4, 5, 6, 8])
            knobs["pointer_update_rate"] = random.choice([1, 2, 3, 5, 7, 8])
            knobs["trap_rate"] = random.choice([0, 1, 2, 3])
            knobs["trap_kind"] = random.choice([0, 1, 3])
            knobs["branch_taken_bias"] = random.choice([0, 1, 4, 5, 7])
            knobs["mixed_burst_bias"] = random.choice([0, 0, 0, 1])
            knobs["mem_delay_base"] = random.choice([2, 3, 4])

        if profile in ("stall_low", "stall_depth_tiny", "delay_fast"):
            knobs["mem_delay_base"] = random.choice([1, 2])
            knobs["mixed_burst_bias"] = random.choice([0, 1, 2, 3, 4])
            knobs["trap_rate"] = random.choice([0, 1])
            knobs["branch_taken_bias"] = random.choice([3, 4, 5, 6])

        if profile == "data_heavy":
            knobs["load_weight"] = random.choice([7, 8, 9, 10])
            knobs["store_weight"] = random.choice([6, 7, 8, 9, 10])
            knobs["arith_weight"] = random.choice([2, 3, 4, 5])
            knobs["branch_weight"] = random.choice([1, 2, 3, 4])
            knobs["jump_weight"] = random.choice([1, 2, 3])
            knobs["mixed_burst_bias"] = random.choice([5, 6, 7, 8, 9, 10])
            knobs["pointer_update_rate"] = random.choice([6, 7, 8, 9, 10])

        if profile == "pressure_low":
            knobs["load_weight"] = random.choice([1, 2, 3, 4])
            knobs["store_weight"] = random.choice([1, 2, 3, 4])
            knobs["arith_weight"] = random.choice([6, 7, 8, 9, 10])
            knobs["branch_weight"] = random.choice([4, 5, 6, 7, 8, 9])
            knobs["jump_weight"] = random.choice([3, 4, 5, 6, 7])
            knobs["mixed_burst_bias"] = random.choice([0, 1, 2, 3])
            knobs["mem_delay_base"] = random.choice([1, 2, 3, 4])

        if profile == "alt_low":
            if random.random() < 0.5:
                knobs["load_weight"] = random.choice([8, 9, 10])
                knobs["store_weight"] = random.choice([1, 2, 3])
            else:
                knobs["load_weight"] = random.choice([1, 2, 3])
                knobs["store_weight"] = random.choice([8, 9, 10])
            knobs["mixed_burst_bias"] = random.choice([0, 1, 2, 3, 4])

        if profile == "trans_moderate":
            knobs["load_weight"] = random.choice([3, 4, 5, 6])
            knobs["store_weight"] = random.choice([3, 4, 5, 6])
            knobs["arith_weight"] = random.choice([4, 5, 6, 7])
            knobs["branch_weight"] = random.choice([2, 3, 4, 5])
            knobs["jump_weight"] = random.choice([1, 2, 3, 4])
            knobs["mixed_burst_bias"] = random.choice([0, 1, 2, 3, 4])

        if profile == "trans_moderate_fast":
            variant = random.choice(["balanced", "store_heavy", "control_heavy"])
            if variant == "balanced":
                knobs["load_weight"] = random.choice([4, 5, 6, 7, 8])
                knobs["store_weight"] = random.choice([6, 7, 8, 9, 10])
                knobs["branch_weight"] = random.choice([6, 7, 8, 9, 10])
                knobs["jump_weight"] = random.choice([5, 6, 7, 8, 9, 10])
                knobs["arith_weight"] = random.choice([5, 6, 7, 8, 9])
            elif variant == "store_heavy":
                knobs["load_weight"] = random.choice([1, 2, 3, 4])
                knobs["store_weight"] = random.choice([8, 9, 10])
                knobs["branch_weight"] = random.choice([6, 7, 8, 9, 10])
                knobs["jump_weight"] = random.choice([5, 6, 8, 9, 10])
                knobs["arith_weight"] = random.choice([4, 5, 6, 8, 9])
            else:
                knobs["load_weight"] = random.choice([2, 3, 6, 7])
                knobs["store_weight"] = random.choice([3, 4, 8, 10])
                knobs["branch_weight"] = random.choice([8, 9, 10])
                knobs["jump_weight"] = random.choice([8, 9, 10])
                knobs["arith_weight"] = random.choice([4, 5, 6, 7])
            knobs["mem_stride"] = random.choice([1, 2, 3, 5, 6, 8])
            knobs["pointer_update_rate"] = random.choice([2, 3, 5, 8, 9, 10])
            knobs["trap_rate"] = random.choice([1, 2, 3])
            knobs["trap_kind"] = random.choice([0, 1, 2, 3])
            knobs["branch_taken_bias"] = random.choice([1, 2, 5, 7, 10])
            knobs["mixed_burst_bias"] = random.choice([0, 1, 3, 4, 7, 8])
            knobs["mem_delay_base"] = random.choice([1, 2])

        if profile == "trans_moderate_medium":
            variant = random.choice(["balanced", "asymmetric"])
            if variant == "balanced":
                knobs["load_weight"] = random.choice([4, 5, 6, 8])
                knobs["store_weight"] = random.choice([6, 9, 10])
                knobs["branch_weight"] = random.choice([3, 4, 5, 9, 10])
                knobs["jump_weight"] = random.choice([6, 7, 8, 9, 10])
                knobs["arith_weight"] = random.choice([3, 7, 8, 9, 10])
            else:
                knobs["load_weight"] = random.choice([1, 2, 3, 8])
                knobs["store_weight"] = random.choice([1, 2, 9, 10])
                knobs["branch_weight"] = random.choice([3, 4, 5, 9])
                knobs["jump_weight"] = random.choice([6, 7, 8, 9, 10])
                knobs["arith_weight"] = random.choice([3, 8, 9, 10])
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

        if profile == "transition_sparse":
            if random.random() < 0.5:
                knobs["load_weight"] = random.choice([7, 8, 9, 10])
                knobs["store_weight"] = random.choice([1, 2, 3])
            else:
                knobs["load_weight"] = random.choice([1, 2, 3])
                knobs["store_weight"] = random.choice([7, 8, 9, 10])
            knobs["branch_weight"] = random.choice([1, 2, 3, 4])
            knobs["jump_weight"] = random.choice([0, 1, 2, 3]) if 0 in KNOB_RANGES["jump_weight"] else random.choice([1, 2, 3])
            knobs["mixed_burst_bias"] = random.choice([0, 1, 2])

        if profile == "load_then_load_rare":
            knobs["load_weight"] = random.choice([8, 9, 10])
            knobs["store_weight"] = random.choice([1, 2, 3])
            knobs["mixed_burst_bias"] = random.choice([0, 1, 2])

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

    def _sample_candidate_pool(self,
                               target_count: int,
                               avoid_recent: bool = True,
                               include_archive: bool = True) -> List[dict]:
        recent = set(self.recent_knob_keys) if avoid_recent else set()
        unique: Dict[tuple, dict] = {}
        attempts = max(target_count * 4, target_count + 32)
        uncovered_bins = [
            b for b in ALL_COVERAGE_BINS
            if self.bin_hit_counts.get(b, 0) == 0
        ]
        reward_mode = self._active_reward_mode(uncovered_bins)
        if reward_mode == "B":
            target_profiles = _target_profiles_for_uncovered_bins(uncovered_bins)
        elif _residual_closure_enabled():
            target_profiles = self._residual_action_plan(uncovered_bins)
        else:
            target_profiles = []

        focused_reward_b = reward_mode == "B" and len(uncovered_bins) <= 3
        if focused_reward_b:
            exact_profile_map = {
                "mem_mix_heavy": "data_heavy",
                "cross_alt_low_stall_short": "alt_low_short",
                "cross_alt_low_stall_medium": "alt_low_medium",
                "control_mix_high": "control_high",
                "cross_trans_moderate_delay_fast": "trans_moderate_fast",
                "cross_trans_moderate_delay_medium": "trans_moderate_medium",
                "cross_trans_moderate_delay_slow": "trans_moderate_slow",
                "transition_load_then_store_rare": "load_then_store_rare",
                "transition_store_then_load_rare": "store_then_load_rare",
                "transition_store_then_store_rare": "store_then_store_rare",
            }
            exact_profiles = [
                exact_profile_map[b]
                for b in uncovered_bins
                if b in exact_profile_map
            ]
            if exact_profiles:
                target_profiles = exact_profiles

        targeted_take = 0
        if target_profiles:
            if focused_reward_b:
                targeted_take = target_count
            else:
                targeted_take = min(
                    target_count,
                    max(TARGETED_CANDIDATE_MIN, int(target_count * TARGETED_CANDIDATE_FRACTION))
                )
            if focused_reward_b:
                weighted_profiles = target_profiles * 8
            else:
                weighted_profiles = (
                    target_profiles[:2] * 3
                    + target_profiles[2:] * 2
                    + target_profiles
                )
            for _ in range(max(targeted_take * 3, targeted_take + 16)):
                knobs = self._targeted_knobs(random.choice(weighted_profiles))
                if focused_reward_b:
                    if random.random() < 0.18:
                        knobs = self._mutate_knobs(knobs, n_changes=1)
                elif random.random() < 0.45:
                    knobs = self._mutate_knobs(knobs, n_changes=random.choice([1, 2]))
                key = self._knob_key(knobs)
                if key in recent or key in unique:
                    continue
                unique[key] = knobs
                if len(unique) >= targeted_take:
                    break
            if focused_reward_b and len(unique) < targeted_take:
                for _ in range(max(targeted_take * 8, targeted_take + 64)):
                    knobs = self._targeted_knobs(random.choice(weighted_profiles))
                    if random.random() < 0.35:
                        knobs = self._mutate_knobs(knobs, n_changes=1)
                    key = self._knob_key(knobs)
                    if key in recent or key in unique:
                        continue
                    unique[key] = knobs
                    if len(unique) >= targeted_take:
                        break

        if include_archive and not focused_reward_b:
            archive_take = min(
                len(self.rare_hit_archive),
                max(0, int(target_count * ARCHIVE_MUTATION_FRACTION))
            )
            for entry in self.rare_hit_archive[:archive_take]:
                base = dict(entry.get("knobs", {}))
                if not base:
                    continue
                for n_changes in (1, 2, 3):
                    knobs = self._mutate_knobs(base, n_changes=n_changes)
                    key = self._knob_key(knobs)
                    if key in recent or key in unique:
                        continue
                    unique[key] = knobs
                    if len(unique) >= target_count:
                        return list(unique.values())

        for _ in range(attempts):
            if focused_reward_b and target_profiles:
                knobs = self._targeted_knobs(random.choice(target_profiles))
                if random.random() < 0.25:
                    knobs = self._mutate_knobs(knobs, n_changes=1)
            else:
                knobs = _random_knobs()
            key = self._knob_key(knobs)
            if key in recent or key in unique:
                continue
            unique[key] = knobs
            if len(unique) >= target_count:
                break

        while len(unique) < target_count:
            if focused_reward_b and target_profiles:
                knobs = self._targeted_knobs(random.choice(target_profiles))
                if random.random() < 0.30:
                    knobs = self._mutate_knobs(knobs, n_changes=1)
            else:
                knobs = _random_knobs()
            key = self._knob_key(knobs)
            if key in unique:
                continue
            unique[key] = knobs

        return list(unique.values())

    def _random_unseen_knobs(self, used_keys: set) -> dict:
        recent = set(self.recent_knob_keys)
        for _ in range(64):
            knobs = _random_knobs()
            key = self._knob_key(knobs)
            if key in used_keys or key in recent:
                continue
            return knobs
        return _random_knobs()

    def _train(self, steps: int = 20):
        if not _TORCH_OK or self.qnet is None:
            return
        if len(self.replay) < BATCH_SIZE:
            return

        for _ in range(steps):
            batch = self.replay.sample(BATCH_SIZE)
            states_np = np.stack([b[0] for b in batch]).astype(np.float32)
            knobs_np = np.stack([b[1] for b in batch]).astype(np.float32)
            rewards_np = np.asarray([b[2] for b in batch], dtype=np.float32)
            next_states_np = np.stack([b[3] for b in batch]).astype(np.float32)

            states = torch.from_numpy(states_np)
            knobs_norms = torch.from_numpy(knobs_np)
            rewards = torch.from_numpy(rewards_np)
            next_states = torch.from_numpy(next_states_np)

            inp = torch.cat([states, knobs_norms], dim=1)
            q_pred = self.qnet(inp)

            # Double DQN target with sampled candidate actions.
            with torch.no_grad():
                next_knob_candidates = np.stack([
                    np.stack([
                        _normalise_knobs(k) for k in self._sample_candidate_pool(
                            N_CANDIDATES,
                            avoid_recent=False,
                            include_archive=False,
                        )
                    ]).astype(np.float32)
                    for _ in range(len(batch))
                ]).astype(np.float32)
                candidate_count = int(next_knob_candidates.shape[1])
                next_knobs = torch.from_numpy(next_knob_candidates)
                next_states_rep = next_states.unsqueeze(1).repeat(1, candidate_count, 1)
                flat_states = next_states_rep.reshape(-1, self.state_dim)
                flat_knobs = next_knobs.reshape(-1, KNOB_DIM)
                flat_inp = torch.cat([flat_states, flat_knobs], dim=1)

                q_online_all = self.qnet(flat_inp).reshape(len(batch), candidate_count)
                best_idx = torch.argmax(q_online_all, dim=1)
                q_target_all = self.target(flat_inp).reshape(len(batch), candidate_count)
                q_next = q_target_all.gather(1, best_idx.unsqueeze(1)).squeeze(1)
                q_target = rewards + GAMMA * q_next

            loss = self.loss_fn(q_pred, q_target)
            self.optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.qnet.parameters(), 1.0)
            self.optimiser.step()

        # Soft-update target network every training call
        tau = 0.01
        for tp, qp in zip(self.target.parameters(), self.qnet.parameters()):
            tp.data.copy_(tau * qp.data + (1 - tau) * tp.data)
