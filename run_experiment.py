#!/usr/bin/env python3
"""
run_experiment.py
Main orchestration for PicoRV32 ML-DV experiment.
"""

import os
import sys
import json
import csv
import random
import argparse
import subprocess
import time
import shutil
import math
from collections import Counter
from typing import List, Dict, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
WORK_DIR     = os.path.join(PROJECT_ROOT, "work")
RESULTS_DIR  = os.path.join(PROJECT_ROOT, "results")
PROGRAMS_DIR = os.path.join(PROJECT_ROOT, "programs")
ML_DIR       = os.path.join(PROJECT_ROOT, "ml")

os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Keep PyTorch on plain eager execution for this project.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_DISABLE_DYNAMO", "1")

sys.path.insert(0, ML_DIR)
sys.path.insert(0, PROGRAMS_DIR)

from coverage_agents import build_coverage_agent, coverage_agent_model_filename
from coverage_features import KNOB_RANGES, compute_reward
from dqn_agent import DQNAgent
from supervised_learning import SupervisedAgent
from gen_program import generate_program, derive_sim_seed

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
QUESTA_CMD = "vsim"
SIM_TIMEOUT = 600
ALPHA_NEW_BIN = 0.20
BETA_RARE_BIN = 0.15
GAMMA_REMAINING_BIN = 0.25
COVERAGE_SIM_LIB = "work_cov"
FAULT_SIM_LIB = "work_fault"
COVERAGE_MILESTONE_THRESHOLDS = (
    (0.70, "70"),
    (0.90, "90"),
    (0.95, "95"),
    (1.00, "100"),
)


FUNCTIONAL_BIN_CATALOG = [
    "stall_depth_short", "stall_depth_medium", "stall_depth_long",
    "stall_type_instr_dominant", "stall_type_balanced", "stall_type_data_dominant",
    "transition_diversity_poor", "transition_diversity_moderate", "transition_diversity_rich",
    "transition_entropy_low", "transition_entropy_medium", "transition_entropy_high",
    "alternation_none", "alternation_low", "alternation_medium", "alternation_high",
    "instr_window_trivial", "instr_window_short", "instr_window_medium", "instr_window_long",
    "b2b_pressure_low", "b2b_pressure_medium", "b2b_pressure_high",
    "delay_fast", "delay_medium", "delay_slow",
    "mem_mix_light", "mem_mix_balanced", "mem_mix_heavy",
    "control_mix_low", "control_mix_medium", "control_mix_high",
    "transition_fetch_then_load_rare", "transition_fetch_then_load_recurrent", "transition_fetch_then_load_hot",
    "transition_fetch_then_store_rare", "transition_fetch_then_store_recurrent", "transition_fetch_then_store_hot",
    "transition_load_then_load_rare", "transition_load_then_load_recurrent", "transition_load_then_load_hot",
    "transition_load_then_store_rare", "transition_load_then_store_recurrent", "transition_load_then_store_hot",
    "transition_store_then_load_rare", "transition_store_then_load_recurrent", "transition_store_then_load_hot",
    "transition_store_then_store_rare", "transition_store_then_store_recurrent", "transition_store_then_store_hot",
    "cross_fetch_stall_short", "cross_fetch_stall_medium", "cross_fetch_stall_long",
    "cross_load_stall_short", "cross_load_stall_medium", "cross_load_stall_long",
    "cross_store_stall_short", "cross_store_stall_medium", "cross_store_stall_long",
    "cross_b2b_stall_short", "cross_b2b_stall_medium", "cross_b2b_stall_long",
    "cross_alt_low_stall_short", "cross_alt_low_stall_medium", "cross_alt_low_stall_long",
    "cross_alt_medium_stall_short", "cross_alt_medium_stall_medium", "cross_alt_medium_stall_long",
    "cross_alt_high_stall_short", "cross_alt_high_stall_medium", "cross_alt_high_stall_long",
    "cross_trans_moderate_delay_fast", "cross_trans_moderate_delay_medium", "cross_trans_moderate_delay_slow",
    "cross_trans_rich_delay_fast", "cross_trans_rich_delay_medium", "cross_trans_rich_delay_slow",
]


def _dynamic_coverage_reward_enabled() -> bool:
    return os.environ.get("PICORV32_DYNAMIC_COVERAGE_REWARD", "").strip() == "1"


def _bin_weight(bin_name: str) -> float:
    return 2.0 if str(bin_name).startswith("cross_") else 1.0


def _bin_family(bin_name: str) -> str:
    name = str(bin_name)
    for prefix in (
        "stall_depth_",
        "stall_type_",
        "transition_diversity_",
        "transition_entropy_",
        "alternation_",
        "instr_window_",
        "b2b_pressure_",
        "delay_",
        "mem_mix_",
        "control_mix_",
    ):
        if name.startswith(prefix):
            return prefix[:-1]
    if name.startswith("transition_"):
        return "transition"
    if name.startswith("cross_fetch_") or name.startswith("cross_load_") or name.startswith("cross_store_") or name.startswith("cross_b2b_"):
        return "stall_cross"
    if name.startswith("cross_alt_"):
        return "alt_cross"
    if name.startswith("cross_trans_"):
        return "trans_delay_cross"
    return "other"


_FAMILY_TO_BINS = {}
for _bin_name in FUNCTIONAL_BIN_CATALOG:
    _FAMILY_TO_BINS.setdefault(_bin_family(_bin_name), []).append(_bin_name)


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


_FOCUS_GROUP_TO_BINS = {}
for _bin_name in FUNCTIONAL_BIN_CATALOG:
    for _focus_group in _bin_focus_groups(_bin_name):
        _FOCUS_GROUP_TO_BINS.setdefault(_focus_group, []).append(_bin_name)


def _coverage_reward_schedule(progress_ratio: float) -> Dict[str, float]:
    p = float(progress_ratio)
    if p < 0.70:
        return {
            "composite_scale": 1.00,
            "new_bin": 0.20,
            "rare_bin": 0.15,
            "remaining_bin": 0.25,
            "near_miss_bonus": 0.03,
            "underrepresented_family": 0.05,
            "group_bonus": 0.04,
            "group_composite_shift": 0.05,
            "focus_bonus": 0.02,
            "focus_composite_shift": 0.02,
            "dominant_penalty": 0.00,
        }
    if p < 0.90:
        return {
            "composite_scale": 0.78,
            "new_bin": 0.18,
            "rare_bin": 0.16,
            "remaining_bin": 0.30,
            "near_miss_bonus": 0.07,
            "underrepresented_family": 0.18,
            "group_bonus": 0.12,
            "group_composite_shift": 0.16,
            "focus_bonus": 0.10,
            "focus_composite_shift": 0.10,
            "dominant_penalty": 0.05,
        }
    return {
        "composite_scale": 0.65,
        "new_bin": 0.10,
        "rare_bin": 0.14,
        "remaining_bin": 0.42,
        "near_miss_bonus": 0.11,
        "underrepresented_family": 0.38,
        "group_bonus": 0.24,
        "group_composite_shift": 0.28,
        "focus_bonus": 0.18,
        "focus_composite_shift": 0.18,
        "dominant_penalty": 0.04,
    }


def _family_remaining_ratios(bin_hit_counts: Dict[str, int]) -> Dict[str, float]:
    ratios: Dict[str, float] = {}
    for family, family_bins in _FAMILY_TO_BINS.items():
        total = max(len(family_bins), 1)
        remaining = sum(1 for fb in family_bins if bin_hit_counts.get(fb, 0) == 0)
        ratios[family] = float(remaining) / float(total)
    return ratios


def _focus_remaining_ratios(bin_hit_counts: Dict[str, int]) -> Dict[str, float]:
    ratios: Dict[str, float] = {}
    for focus_group, focus_bins in _FOCUS_GROUP_TO_BINS.items():
        total = max(len(focus_bins), 1)
        remaining = sum(1 for fb in focus_bins if bin_hit_counts.get(fb, 0) == 0)
        ratios[focus_group] = float(remaining) / float(total)
    return ratios


def _stall_depth_bucket(max_run: float) -> str:
    if max_run <= 1:
        return "tiny"
    if max_run <= 2:
        return "short"
    if max_run <= 5:
        return "medium"
    return "long"


def _stall_type_bucket(cov: dict) -> str:
    instr = float(cov.get("instr_stall_ratio", 0.0))
    data = float(cov.get("data_stall_ratio", 0.0))
    total = instr + data
    data_share = (data / total) if total > 0.0 else 0.0
    mem_frac = float(cov.get("load_instr_frac", 0.0)) + float(cov.get("store_instr_frac", 0.0))

    # PicoRV32 fetches continuously, so true data-dominant pressure is rare.
    # Treat this bin as "data-led" rather than requiring data stalls to exceed
    # instruction stalls outright.
    if data_share >= 0.33 and data >= 0.01 and mem_frac >= 0.30:
        return "data_dominant"
    if instr > max(data * 2.5, 0.02):
        return "instr_dominant"
    return "balanced"


def _transition_diversity_bucket(cov: dict) -> str:
    hit = int(cov.get("transition_types_hit", 0))
    if hit <= 2:
        return "poor"
    if hit <= 4:
        return "moderate"
    return "rich"


def _transition_entropy_bucket(cov: dict) -> str:
    entropy = float(cov.get("transition_entropy", 0.0))
    if entropy < 0.55:
        return "low"
    if entropy < 0.75:
        return "medium"
    return "high"


def _alternation_bucket(cov: dict) -> str:
    alt = float(cov.get("load_store_alternation_ratio", 0.0))
    if alt <= 0.0:
        return "none"
    if alt <= 0.30:
        return "low"
    if alt <= 0.60:
        return "medium"
    return "high"


def _instr_window_bucket(cov: dict) -> str:
    instr = float(cov.get("instr_count", 0.0))
    if instr < 100:
        return "trivial"
    if instr < 500:
        return "short"
    if instr < 1500:
        return "medium"
    return "long"


def _b2b_pressure_bucket(cov: dict) -> str:
    b2b_rate = float(cov.get("b2b_stall_rate", 0.0))
    # In this testbench, fetch traffic keeps back-to-back stall rates well above
    # zero, so "low" needs to mean recovery-friendly rather than nearly absent.
    if b2b_rate < 0.10:
        return "low"
    if b2b_rate < 0.20:
        return "medium"
    return "high"


def _delay_bucket(knobs: dict) -> str:
    delay = int(knobs.get("mem_delay_base", 1))
    if delay <= 2:
        return "fast"
    if delay <= 5:
        return "medium"
    return "slow"


def _mem_mix_bucket(cov: dict) -> str:
    mem_frac = float(cov.get("load_instr_frac", 0.0)) + float(cov.get("store_instr_frac", 0.0))
    if mem_frac < 0.25:
        return "light"
    if mem_frac < 0.50:
        return "balanced"
    return "heavy"


def _control_mix_bucket(cov: dict) -> str:
    ctrl_frac = float(cov.get("branch_instr_frac", 0.0)) + float(cov.get("jump_instr_frac", 0.0))
    if ctrl_frac < 0.20:
        return "low"
    if ctrl_frac < 0.40:
        return "medium"
    return "high"


def _transition_count_bucket(count: int) -> Optional[str]:
    count = int(count)
    if count <= 0:
        return None
    if count == 1:
        return "rare"
    if count <= 4:
        return "recurrent"
    return "hot"


def _compute_functional_bin_hits(cov: dict, knobs: dict) -> List[str]:
    hits = set()

    stall_depth = _stall_depth_bucket(float(cov.get("max_stall_run", 0.0)))
    stall_type = _stall_type_bucket(cov)
    trans_div = _transition_diversity_bucket(cov)
    entropy = _transition_entropy_bucket(cov)
    alt = _alternation_bucket(cov)
    instr_window = _instr_window_bucket(cov)
    b2b_pressure = _b2b_pressure_bucket(cov)
    delay = _delay_bucket(knobs)
    mem_mix = _mem_mix_bucket(cov)
    ctrl_mix = _control_mix_bucket(cov)

    hits.add(f"stall_depth_{stall_depth}")
    hits.add(f"stall_type_{stall_type}")
    hits.add(f"transition_diversity_{trans_div}")
    hits.add(f"transition_entropy_{entropy}")
    hits.add(f"alternation_{alt}")
    hits.add(f"instr_window_{instr_window}")
    hits.add(f"b2b_pressure_{b2b_pressure}")
    hits.add(f"delay_{delay}")
    hits.add(f"mem_mix_{mem_mix}")
    hits.add(f"control_mix_{ctrl_mix}")

    transition_fields = {
        "transition_fetch_then_load": "fetch_then_load",
        "transition_fetch_then_store": "fetch_then_store",
        "transition_load_then_load": "load_then_load",
        "transition_load_then_store": "load_then_store",
        "transition_store_then_load": "store_then_load",
        "transition_store_then_store": "store_then_store",
    }
    for prefix, field in transition_fields.items():
        bucket = _transition_count_bucket(int(cov.get(field, 0)))
        if bucket is not None:
            hits.add(f"{prefix}_{bucket}")

    if stall_depth in ("short", "medium", "long"):
        if int(cov.get("fetch_then_load", 0)) > 0 or int(cov.get("fetch_then_store", 0)) > 0:
            hits.add(f"cross_fetch_stall_{stall_depth}")
        if float(cov.get("load_stall_cycles", 0.0)) > 0.0:
            hits.add(f"cross_load_stall_{stall_depth}")
        if float(cov.get("store_stall_cycles", 0.0)) > 0.0:
            hits.add(f"cross_store_stall_{stall_depth}")
        if b2b_pressure in ("medium", "high"):
            hits.add(f"cross_b2b_stall_{stall_depth}")
        if alt in ("low", "medium", "high"):
            hits.add(f"cross_alt_{alt}_stall_{stall_depth}")

    if trans_div in ("moderate", "rich"):
        hits.add(f"cross_trans_{trans_div}_delay_{delay}")

    return sorted(hits)


def _ordered_bucket_closeness(actual: str, target: str, order: List[str]) -> float:
    if actual not in order or target not in order:
        return 0.0
    if len(order) <= 1:
        return 1.0
    dist = abs(order.index(actual) - order.index(target))
    return max(0.0, 1.0 - float(dist) / float(len(order) - 1))


def _transition_bucket_near_miss(count: int, target_bucket: str) -> float:
    count = int(count)
    bucket = str(target_bucket)
    if bucket == "rare":
        if count == 1:
            return 1.0
        if count == 2:
            return 0.60
        if count == 0:
            return 0.18
        return max(0.0, 0.40 - 0.08 * (count - 2))
    if bucket == "recurrent":
        if 2 <= count <= 4:
            return 1.0
        if count == 1 or count == 5:
            return 0.55
        if count == 0:
            return 0.12
        return 0.20
    if bucket == "hot":
        return min(float(count) / 5.0, 1.0)
    return 0.0


def _near_miss_for_functional_bin(bin_name: str, cov: dict, knobs: dict) -> float:
    name = str(bin_name)
    stall_depth = _stall_depth_bucket(float(cov.get("max_stall_run", 0.0)))
    stall_type = _stall_type_bucket(cov)
    trans_div = _transition_diversity_bucket(cov)
    entropy = _transition_entropy_bucket(cov)
    alt = _alternation_bucket(cov)
    instr_window = _instr_window_bucket(cov)
    b2b_pressure = _b2b_pressure_bucket(cov)
    delay = _delay_bucket(knobs)
    mem_mix = _mem_mix_bucket(cov)
    ctrl_mix = _control_mix_bucket(cov)

    if name.startswith("stall_depth_"):
        return _ordered_bucket_closeness(
            stall_depth, name.replace("stall_depth_", ""), ["tiny", "short", "medium", "long"]
        )
    if name.startswith("stall_type_"):
        return _ordered_bucket_closeness(
            stall_type, name.replace("stall_type_", ""), ["instr_dominant", "balanced", "data_dominant"]
        )
    if name.startswith("transition_diversity_"):
        return _ordered_bucket_closeness(
            trans_div, name.replace("transition_diversity_", ""), ["poor", "moderate", "rich"]
        )
    if name.startswith("transition_entropy_"):
        return _ordered_bucket_closeness(
            entropy, name.replace("transition_entropy_", ""), ["low", "medium", "high"]
        )
    if name.startswith("alternation_"):
        return _ordered_bucket_closeness(
            alt, name.replace("alternation_", ""), ["none", "low", "medium", "high"]
        )
    if name.startswith("instr_window_"):
        return _ordered_bucket_closeness(
            instr_window, name.replace("instr_window_", ""), ["trivial", "short", "medium", "long"]
        )
    if name.startswith("b2b_pressure_"):
        return _ordered_bucket_closeness(
            b2b_pressure, name.replace("b2b_pressure_", ""), ["low", "medium", "high"]
        )
    if name.startswith("delay_"):
        return _ordered_bucket_closeness(delay, name.replace("delay_", ""), ["fast", "medium", "slow"])
    if name.startswith("mem_mix_"):
        return _ordered_bucket_closeness(mem_mix, name.replace("mem_mix_", ""), ["light", "balanced", "heavy"])
    if name.startswith("control_mix_"):
        return _ordered_bucket_closeness(ctrl_mix, name.replace("control_mix_", ""), ["low", "medium", "high"])

    transition_fields = {
        "transition_fetch_then_load": "fetch_then_load",
        "transition_fetch_then_store": "fetch_then_store",
        "transition_load_then_load": "load_then_load",
        "transition_load_then_store": "load_then_store",
        "transition_store_then_load": "store_then_load",
        "transition_store_then_store": "store_then_store",
    }
    for prefix, field in transition_fields.items():
        if name.startswith(prefix + "_"):
            target_bucket = name[len(prefix) + 1:]
            return _transition_bucket_near_miss(int(cov.get(field, 0)), target_bucket)

    if name.startswith("cross_alt_"):
        parts = name.split("_")
        if len(parts) >= 5:
            target_alt = parts[2]
            target_stall = parts[-1]
            return min(
                1.0,
                0.55 * _ordered_bucket_closeness(alt, target_alt, ["none", "low", "medium", "high"])
                + 0.45 * _ordered_bucket_closeness(stall_depth, target_stall, ["tiny", "short", "medium", "long"])
            )

    if name.startswith("cross_trans_"):
        parts = name.split("_")
        if len(parts) >= 5:
            target_trans = parts[2]
            target_delay = parts[-1]
            return min(
                1.0,
                0.60 * _ordered_bucket_closeness(trans_div, target_trans, ["poor", "moderate", "rich"])
                + 0.40 * _ordered_bucket_closeness(delay, target_delay, ["fast", "medium", "slow"])
            )

    return 0.0


def _targeted_remaining_near_miss_bonus(cov: dict,
                                        knobs: dict,
                                        bin_hit_counts: Dict[str, int],
                                        hit_bins: List[str]) -> float:
    unresolved = [
        b for b in FUNCTIONAL_BIN_CATALOG
        if bin_hit_counts.get(b, 0) == 0 and b not in hit_bins
    ]
    if not unresolved:
        return 0.0
    scores = [
        _near_miss_for_functional_bin(b, cov, knobs)
        for b in unresolved
    ]
    scores = sorted((float(s) for s in scores if s > 0.0), reverse=True)
    if not scores:
        return 0.0
    return float(np.mean(scores[: min(3, len(scores))]))


def _strip_fault_debug_fields(cov: dict) -> dict:
    stripped = dict(cov)
    for key in list(stripped.keys()):
        if (
            key.startswith("bug_")
            or key.startswith("oracle_")
            or key.startswith("raw_")
        ):
            stripped.pop(key, None)
    return stripped


class SimulationRunner:
    def __init__(self, questa_cmd=QUESTA_CMD, timeout=SIM_TIMEOUT, program_seed_base: int = 0,
                 include_fault_fields: bool = True):
        self.questa_cmd = questa_cmd
        self.timeout = timeout
        self.program_seed_base = int(program_seed_base)
        self.include_fault_fields = bool(include_fault_fields)

    def write_knobs(self, sim_id: int, knobs: dict, sim_seed: int) -> str:
        path = os.path.join(WORK_DIR, f"knobs_{sim_id}.json")
        payload = {"sim_id": sim_id, "program_seed": int(sim_seed) & 0xFFFFFFFF}
        payload.update(knobs)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        return path

    def read_coverage(self, sim_id: int) -> Optional[dict]:
        path = os.path.join(WORK_DIR, f"coverage_{sim_id}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            data = json.load(f)
        return data.get("coverage", {})

    def read_oracle(self, sim_id: int) -> Optional[dict]:
        path = os.path.join(WORK_DIR, f"oracle_{sim_id}.json")
        if not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def run_one(self, sim_id: int, knobs: dict) -> Optional[dict]:
        sim_seed = derive_sim_seed(self.program_seed_base, sim_id)
        try:
            generate_program(
                sim_id,
                knobs,
                output_dir=WORK_DIR,
                seed_base=self.program_seed_base,
                sim_seed=sim_seed,
            )
        except Exception as e:
            print(f"  [Sim {sim_id:04d}] Program generation failed: {e}")
            return None

        self.write_knobs(sim_id, knobs, sim_seed)

        cov_path = os.path.join(WORK_DIR, f"coverage_{sim_id}.json")
        if os.path.exists(cov_path):
            os.remove(cov_path)

        env = os.environ.copy()
        env["SIM_ID"] = str(sim_id)

        cmd = [self.questa_cmd, "-c", "-do", "scripts/simulate.tcl", "-quiet"]
        log_path = os.path.join(WORK_DIR, f"sim_{sim_id}.log")
        t0 = time.time()

        try:
            with open(log_path, "w") as log_f:
                result = subprocess.run(
                    cmd,
                    env=env,
                    cwd=PROJECT_ROOT,
                    stdout=log_f,
                    stderr=subprocess.STDOUT,
                    timeout=self.timeout,
                )
            elapsed = time.time() - t0
            if result.returncode != 0:
                print(f"  [Sim {sim_id:04d}] vsim exited with code {result.returncode} "
                      f"({elapsed:.1f}s) - see {log_path}")
        except subprocess.TimeoutExpired:
            print(f"  [Sim {sim_id:04d}] TIMEOUT after {self.timeout}s")
            return None
        except FileNotFoundError:
            print(f"  [Sim {sim_id:04d}] ERROR: '{self.questa_cmd}' not found on PATH")
            return None

        cov = self.read_coverage(sim_id)
        if cov is None:
            print(f"  [Sim {sim_id:04d}] No coverage output - simulation may have failed")
            return None

        cov["elapsed_sec"] = float(elapsed)
        cov["program_seed"] = int(sim_seed) & 0xFFFFFFFF
        cov["composite_reward"] = compute_reward(cov)
        cov["functional_bins"] = _compute_functional_bin_hits(cov, knobs)
        cov["functional_bin_distribution"] = {
            bin_name: int(bin_name in cov["functional_bins"])
            for bin_name in FUNCTIONAL_BIN_CATALOG
        }
        cov["functional_bin_count"] = len(cov["functional_bins"])
        oracle = self.read_oracle(sim_id)
        if oracle is not None and self.include_fault_fields:
            cov["oracle_enabled"] = bool(oracle.get("oracle_enabled", False))
            cov["oracle_program_seed"] = int(oracle.get("program_seed", sim_seed)) & 0xFFFFFFFF
            cov["oracle_expected_trap_count"] = oracle.get("expected_trap_count")
            cov["oracle_expected_data_region_checksum"] = oracle.get("expected_data_region_checksum")
            cov["timed_out"] = bool(cov.get("timed_out", 0))
            if cov["oracle_enabled"]:
                expected_checksum = oracle.get("expected_data_region_checksum")
                expected_traps = oracle.get("expected_trap_count")
                actual_checksum = cov.get("data_region_checksum")
                actual_traps = cov.get("trap_count")
                timeout_detected = bool(cov.get("timed_out", False))
                manifest_detected = int(cov.get("bug_manifest_count", 0) or 0) > 0
                checksum_mismatch = (
                    expected_checksum is not None and actual_checksum is not None and
                    int(actual_checksum) != int(expected_checksum)
                )
                trap_mismatch = (
                    expected_traps is not None and actual_traps is not None and
                    int(actual_traps) != int(expected_traps)
                )
                detection_reasons = []
                if manifest_detected:
                    detection_reasons.append("bug_manifest")
                if timeout_detected:
                    detection_reasons.append("timeout")
                if checksum_mismatch:
                    detection_reasons.append("checksum_mismatch")
                if trap_mismatch:
                    detection_reasons.append("trap_count_mismatch")
                cov["bug_detected"] = bool(detection_reasons)
                cov["bug_detection_reason"] = detection_reasons[0] if detection_reasons else ""
                cov["bug_detection_reasons"] = detection_reasons
            else:
                cov["bug_detected"] = False
                cov["bug_detection_reason"] = ""
                cov["bug_detection_reasons"] = []
        if not self.include_fault_fields:
            cov = _strip_fault_debug_fields(cov)
        print(f"  [Sim {sim_id:04d}] completed")
        return cov

    def run_batch(self, knobs_list: List[dict], start_sim_id: int,
                  label: str = "", stop_on_bug: bool = False) -> List[Dict]:
        print(f"\n{'='*60}")
        print(f"  Batch: {label}  ({len(knobs_list)} simulations)")
        print(f"{'='*60}")

        results = []
        for i, knobs in enumerate(knobs_list):
            sid = start_sim_id + i
            cov = self.run_one(sid, knobs)
            if cov is not None:
                results.append({
                    "sim_id": sid,
                    "label": label,
                    "knobs": knobs,
                    "coverage": cov,
                })
                if stop_on_bug and bool(cov.get("bug_detected", False)):
                    break

        if results:
            total_cycles = sum(int(r["coverage"].get("total_cycles", 0)) for r in results)
            total_elapsed = sum(float(r["coverage"].get("elapsed_sec", 0.0) or 0.0) for r in results)
            if "fault" in label.lower():
                bug_found = any(r["coverage"].get("bug_detected", False) for r in results)
                bug_status = "bug found" if bug_found else "bug not found"
                print(f"\n  Batch done : {len(results)}/{len(knobs_list)} , {bug_status}  "
                      f"cycles={total_cycles}  time={total_elapsed:.1f}s")
            else:
                ratios = [r["coverage"]["stall_ratio"] for r in results]
                rewards = [r["coverage"].get("composite_reward",
                                             compute_reward(r["coverage"]))
                           for r in results]
                print(f"\n  Batch done: {len(results)}/{len(knobs_list)} succeeded  "
                      f"reward mean={np.mean(rewards):.4f}  "
                      f"stall_ratio mean={np.mean(ratios):.4f}  "
                      f"max={np.max(ratios):.4f}  "
                      f"cycles={total_cycles}  time={total_elapsed:.1f}s")
        return results


class MLDVExperiment:
    def __init__(self,
                 agent_type="supervised",
                 n_init=20,
                 n_iters=6,
                 n_per_iter=10,
                 questa_cmd=QUESTA_CMD,
                 resume=False,
                 result_dir=RESULTS_DIR,
                 eval_mode=False,
                 ml_only: bool = False,
                 model_source_dir: Optional[str] = None,
                 reward_mode: str = "coverage",
                 knob_overrides: Optional[Dict[str, int]] = None,
                 program_seed_base: int = 0):
        self.agent_type = agent_type
        self.n_init = n_init
        self.n_iters = n_iters
        self.n_per_iter = n_per_iter
        self.resume = resume
        self.eval_mode = eval_mode
        self.ml_only = ml_only
        self.reward_mode = reward_mode
        self.results_dir = result_dir
        self.model_source_dir = model_source_dir or result_dir
        self.knob_overrides = dict(knob_overrides or {})
        self.program_seed_base = int(program_seed_base)
        self.runner = SimulationRunner(
            questa_cmd=questa_cmd,
            program_seed_base=self.program_seed_base,
            include_fault_fields=(self.reward_mode == "composite"),
        )

        self.coverage_agent = build_coverage_agent(agent_type) if agent_type in ("dqn", "rf", "dt") else None
        self.dqn = self.coverage_agent if agent_type == "dqn" else None
        self.sup = SupervisedAgent() if agent_type == "supervised" else None

        self.random_results: List[Dict] = []
        self.ml_results: List[List[Dict]] = []
        self.all_results: List[Dict] = []
        self._next_sim_id = 0
        self.initial_random_results: List[Dict] = []
        self.prior_last_ml_results: List[Dict] = []
        self.bin_hit_counts: Counter = Counter()
        self.seed_bin_hit_counts: Counter = Counter()
        self.random_arm_bin_hit_counts: Counter = Counter()
        self.ml_arm_bin_hit_counts: Counter = Counter()

        if self.eval_mode:
            self._load_model_state_only()
        elif self.resume:
            self._load_resume_state()

        self.bin_hit_counts = self._build_bin_hit_counts(self.all_results)
        self.seed_bin_hit_counts = self._build_bin_hit_counts([
            r for r in self.all_results if str(r.get("label", "")) == "random_init"
        ])
        self.random_arm_bin_hit_counts = self._build_bin_hit_counts([
            r for r in self.all_results if str(r.get("label", "")).startswith("random_iter")
        ])
        self.ml_arm_bin_hit_counts = self._build_bin_hit_counts([
            r for r in self.all_results if str(r.get("label", "")).startswith("ml_iter")
        ])

    def _new_sim_ids(self, n: int) -> range:
        r = range(self._next_sim_id, self._next_sim_id + n)
        self._next_sim_id += n
        return r

    def _load_resume_state(self):
        results_path = os.path.join(self.model_source_dir, "experiment_results.json")
        sup_path = os.path.join(self.model_source_dir, "supervised_model.pt")
        dqn_path = os.path.join(self.model_source_dir, "dqn_model.pt")

        if self.sup is not None and os.path.exists(sup_path):
            self.sup.load(sup_path)
        if self.coverage_agent is not None:
            model_path = os.path.join(self.model_source_dir, coverage_agent_model_filename(self.agent_type))
            if os.path.exists(model_path):
                self.coverage_agent.load(model_path)

        if not os.path.exists(results_path):
            return

        with open(results_path) as f:
            saved = json.load(f)
        self.all_results = list(saved.get("all_results", []))
        self.random_results = [r for r in self.all_results
                               if str(r.get("label", "")).startswith("random_")]
        self.initial_random_results = [r for r in self.all_results
                                       if str(r.get("label", "")) == "random_init"]

        ml_buckets: Dict[int, List[Dict]] = {}
        for r in self.all_results:
            label = str(r.get("label", ""))
            if label.startswith("ml_iter"):
                try:
                    idx = int(label.replace("ml_iter", ""))
                except ValueError:
                    continue
                ml_buckets.setdefault(idx, []).append(r)
        self.ml_results = [ml_buckets[idx] for idx in sorted(ml_buckets.keys())]
        if self.ml_results:
            self.prior_last_ml_results = list(self.ml_results[-1])

        if self.all_results:
            self._next_sim_id = max(int(r.get("sim_id", -1)) for r in self.all_results) + 1

        print(f"[Resume] Loaded {len(self.all_results)} prior results from {results_path}")

    def _load_model_state_only(self):
        sup_path = os.path.join(self.model_source_dir, "supervised_model.pt")
        dqn_path = os.path.join(self.model_source_dir, "dqn_model.pt")

        if self.sup is not None and os.path.exists(sup_path):
            self.sup.load(sup_path)
        if self.coverage_agent is not None:
            model_path = os.path.join(self.model_source_dir, coverage_agent_model_filename(self.agent_type))
            if os.path.exists(model_path):
                self.coverage_agent.load(model_path)

        print(f"[Eval] Loaded frozen model state from {self.model_source_dir}")

    def _build_bin_hit_counts(self, results: List[Dict]) -> Counter:
        counts: Counter = Counter()
        for r in results:
            for b in r.get("coverage", {}).get("functional_bins", []):
                if b in FUNCTIONAL_BIN_CATALOG:
                    counts[b] += 1
        return counts

    def _score_results_with_bin_bonuses(self, results: List[Dict]) -> List[Dict]:
        total_bins = len(FUNCTIONAL_BIN_CATALOG)
        for r in results:
            cov = r.get("coverage", {})
            label = str(r.get("label", ""))
            composite = float(cov.get("composite_reward", compute_reward(cov)))
            bins = [b for b in cov.get("functional_bins", []) if b in FUNCTIONAL_BIN_CATALOG]

            hit_weight = sum(_bin_weight(b) for b in bins) or 1.0
            remaining_hits = [b for b in bins if self.bin_hit_counts.get(b, 0) == 0]
            new_weight = sum(_bin_weight(b) for b in remaining_hits)
            rare_weight = sum(
                _bin_weight(b) / np.sqrt(self.bin_hit_counts.get(b, 0) + 1.0)
                for b in bins
            )
            remaining_bin_bonus = float(new_weight / hit_weight)
            covered_before = sum(1 for count in self.bin_hit_counts.values() if count > 0)
            progress_ratio = (covered_before / total_bins) if total_bins else 0.0

            family_remaining_ratios = _family_remaining_ratios(self.bin_hit_counts)
            focus_remaining_ratios = _focus_remaining_ratios(self.bin_hit_counts)
            family_scores = []
            seen_families = set()
            for b in bins:
                family = _bin_family(b)
                if family in seen_families:
                    continue
                seen_families.add(family)
                family_scores.append(family_remaining_ratios.get(family, 0.0))
            underrepresented_family_bonus = float(np.mean(family_scores)) if family_scores else 0.0
            group_target_bonus = underrepresented_family_bonus

            focus_scores = []
            seen_focus_groups = set()
            for b in bins:
                for focus_group in _bin_focus_groups(b):
                    if focus_group in seen_focus_groups:
                        continue
                    seen_focus_groups.add(focus_group)
                    focus_scores.append(focus_remaining_ratios.get(focus_group, 0.0))
            focus_target_bonus = float(np.mean(focus_scores)) if focus_scores else 0.0
            targeted_near_miss_bonus = _targeted_remaining_near_miss_bonus(
                cov,
                r.get("knobs", {}),
                self.bin_hit_counts,
                bins,
            )

            dominant_scores = []
            for b in bins:
                count = float(self.bin_hit_counts.get(b, 0))
                dominant_scores.append(count / (count + 3.0))
            dominant_region_penalty = float(np.mean(dominant_scores)) if dominant_scores else 0.0

            if self.reward_mode == "composite":
                new_bin_bonus = 0.0
                rare_bin_bonus = 0.0
                remaining_bin_bonus = 0.0
                underrepresented_family_bonus = 0.0
                group_target_bonus = 0.0
                focus_target_bonus = 0.0
                targeted_near_miss_bonus = 0.0
                dominant_region_penalty = 0.0
                effective_composite_scale = 1.0
                total_reward = composite
            else:
                new_bin_bonus = float(new_weight / hit_weight)
                rare_bin_bonus = float(rare_weight / hit_weight)
                if _dynamic_coverage_reward_enabled():
                    weights = _coverage_reward_schedule(progress_ratio)
                    effective_composite_scale = max(
                        0.20,
                        weights["composite_scale"]
                        * (
                            1.0
                            - weights["group_composite_shift"] * group_target_bonus
                            - weights["focus_composite_shift"] * focus_target_bonus
                        ),
                    )
                    total_reward = min(
                        1.0,
                        max(
                            0.0,
                            effective_composite_scale * composite
                            + weights["new_bin"] * new_bin_bonus
                            + weights["rare_bin"] * rare_bin_bonus
                            + weights["remaining_bin"] * remaining_bin_bonus
                            + weights["near_miss_bonus"] * targeted_near_miss_bonus
                            + weights["underrepresented_family"] * underrepresented_family_bonus
                            + weights["group_bonus"] * group_target_bonus
                            + weights["focus_bonus"] * focus_target_bonus
                            - weights["dominant_penalty"] * dominant_region_penalty
                        ),
                    )
                else:
                    underrepresented_family_bonus = 0.0
                    group_target_bonus = 0.0
                    focus_target_bonus = 0.0
                    targeted_near_miss_bonus = 0.0
                    dominant_region_penalty = 0.0
                    effective_composite_scale = 1.0
                    total_reward = min(
                        1.0,
                        max(
                            0.0,
                            composite
                            + ALPHA_NEW_BIN * new_bin_bonus
                            + BETA_RARE_BIN * rare_bin_bonus
                            + GAMMA_REMAINING_BIN * remaining_bin_bonus
                        ),
                    )

            cov["composite_reward"] = composite
            cov["new_bin_bonus"] = new_bin_bonus
            cov["rare_bin_bonus"] = rare_bin_bonus
            cov["remaining_bin_bonus"] = remaining_bin_bonus
            cov["underrepresented_family_bonus"] = underrepresented_family_bonus
            cov["group_target_bonus"] = group_target_bonus
            cov["focus_target_bonus"] = focus_target_bonus
            cov["targeted_near_miss_bonus"] = targeted_near_miss_bonus
            cov["effective_composite_scale"] = effective_composite_scale
            cov["dominant_region_penalty"] = dominant_region_penalty
            cov["reward_progress_ratio"] = progress_ratio
            cov["remaining_bin_hits"] = remaining_hits
            cov["total_reward"] = total_reward

            for b in bins:
                self.bin_hit_counts[b] += 1

            if self.reward_mode == "composite":
                print(
                    f"  [Sim {int(r.get('sim_id', -1)):04d}] "
                    f"Composite Reward = {composite:.4f}"
                )
            else:
                if label.startswith("random_iter"):
                    arm_counter = self.random_arm_bin_hit_counts
                    for b in bins:
                        arm_counter[b] += 1
                    arm_name = "Random-arm"
                elif label.startswith("ml_iter"):
                    arm_counter = self.ml_arm_bin_hit_counts
                    for b in bins:
                        arm_counter[b] += 1
                    arm_name = "ML-arm"
                elif label == "random_init":
                    for b in bins:
                        self.seed_bin_hit_counts[b] += 1
                    covered_bins = sum(1 for count in self.seed_bin_hit_counts.values() if count > 0)
                    uncovered_bins = total_bins - covered_bins
                    coverage_pct = (100.0 * covered_bins / total_bins) if total_bins else 0.0
                    print(
                        f"  [Sim {int(r.get('sim_id', -1)):04d}] "
                        f"Seed Coverage = {coverage_pct:.1f}%, "
                        f"Covered Bins = {covered_bins}, "
                        f"Uncovered Bins = {uncovered_bins}"
                    )
                    continue
                else:
                    arm_counter = None
                    arm_name = "Arm"

                covered_bins = sum(1 for count in arm_counter.values() if count > 0) if arm_counter is not None else 0
                uncovered_bins = total_bins - covered_bins
                coverage_pct = (100.0 * covered_bins / total_bins) if total_bins else 0.0
                print(
                    f"  [Sim {int(r.get('sim_id', -1)):04d}] "
                    f"{arm_name} Coverage = {coverage_pct:.1f}%, "
                    f"Covered Bins = {covered_bins}, "
                    f"Uncovered Bins = {uncovered_bins}"
                )

        return results

    def _apply_knob_overrides(self, knobs: dict) -> dict:
        if not self.knob_overrides:
            return dict(knobs)
        merged = dict(knobs)
        for key, value in self.knob_overrides.items():
            merged[key] = value
        return merged

    def _random_knobs(self) -> dict:
        knobs = {name: random.choice(vals) for name, vals in KNOB_RANGES.items()}
        return self._apply_knob_overrides(knobs)

    @staticmethod
    def _metric_mean(results: List[Dict], metric: str) -> float:
        vals = []
        for r in results:
            cov = r["coverage"]
            if metric in ("composite_reward", "total_reward"):
                reward_val = cov.get(metric)
                if reward_val is None and metric == "total_reward":
                    reward_val = cov.get("composite_reward")
                if reward_val is None:
                    reward_val = compute_reward(cov)
                vals.append(float(reward_val))
            else:
                vals.append(float(cov.get(metric, 0.0)))
        return float(np.mean(vals)) if vals else 0.0

    def run(self):
        resume_active = self.resume and bool(self.all_results)
        if self.eval_mode:
            planned_sims = self.n_iters * self.n_per_iter if self.ml_only else 2 * self.n_iters * self.n_per_iter
        elif resume_active:
            planned_sims = self.n_iters * self.n_per_iter
        else:
            planned_sims = self.n_init + self.n_iters * self.n_per_iter if self.ml_only else self.n_init + 2 * self.n_iters * self.n_per_iter

        print("\n" + "="*60)
        print("  PicoRV32 ML-DV Experiment")
        print(f"  Agent:          {self.agent_type}")
        print(f"  Init sims:      {self.n_init}")
        print(f"  ML iterations:  {self.n_iters}")
        print(f"  Sims per iter:  {self.n_per_iter}")
        print(f"  ML only:        {self.ml_only}")
        print(f"  Total sims:     {planned_sims}")
        print("="*60)

        if self.eval_mode:
            seed_results = []
            if self.ml_only:
                print("\n>>> Phase 1: ML-only evaluation (random baseline skipped)")
            else:
                print("\n>>> Phase 1: Random baseline (matched evaluation budget)")
                for it in range(1, self.n_iters + 1):
                    ids = self._new_sim_ids(self.n_per_iter)
                    knobs = [self._random_knobs() for _ in ids]
                    r = self._score_results_with_bin_bonuses(
                        self.runner.run_batch(knobs, ids.start, label=f"random_iter{it}")
                    )
                    self.random_results.extend(r)
                    self.all_results.extend(r)
                    if not seed_results:
                        seed_results = list(r)
                    if self._arm_reached_full_coverage("random"):
                        print("  [Coverage] Random arm reached 100.0% - moving to ML phase")
                        break
                self.initial_random_results = list(self.random_results)
            prev_results = seed_results or list(self.random_results[-self.n_per_iter:]) or self.initial_random_results or self.all_results
            start_ml_iter = 1
        elif resume_active:
            print("\n>>> Resume mode: skipping random phases")
            prev_results = self.prior_last_ml_results or self.initial_random_results or self.all_results
            start_ml_iter = len(self.ml_results) + 1
        else:
            print("\n>>> Phase 1: Random initialisation")
            ids = self._new_sim_ids(self.n_init)
            knobs = [self._random_knobs() for _ in ids]
            init_r = self._score_results_with_bin_bonuses(
                self.runner.run_batch(knobs, ids.start, label="random_init")
            )
            self.random_results.extend(init_r)
            self.initial_random_results = list(init_r)
            self.all_results.extend(init_r)

            prev_results = init_r

            if self.ml_only:
                print("\n>>> Phase 2: ML-only training (random baseline skipped)")
            else:
                print("\n>>> Phase 2: Random baseline (continued)")
                for it in range(1, self.n_iters + 1):
                    ids = self._new_sim_ids(self.n_per_iter)
                    knobs = [self._random_knobs() for _ in ids]
                    r = self._score_results_with_bin_bonuses(
                        self.runner.run_batch(knobs, ids.start, label=f"random_iter{it}")
                    )
                    self.random_results.extend(r)
                    self.all_results.extend(r)
                    if self._arm_reached_full_coverage("random"):
                        print("  [Coverage] Random arm reached 100.0% - moving to ML phase")
                        break
            start_ml_iter = 1

        print("\n>>> Phase 3: ML-guided iterations")
        for local_it in range(1, self.n_iters + 1):
            global_it = start_ml_iter + local_it - 1
            print(f"\n--- ML Iteration {global_it}/{start_ml_iter + self.n_iters - 1} ---")

            if self.eval_mode and self.sup is not None:
                ml_knobs = self.sup.suggest_knobs_frozen(
                    n_suggest=self.n_per_iter)
            elif self.eval_mode and self.coverage_agent is not None:
                ml_knobs = self.coverage_agent.suggest_knobs_frozen(
                    prev_results, n_suggest=self.n_per_iter)
            elif self.sup is not None:
                ml_knobs = self.sup.suggest_knobs_for_iteration(
                    prev_results, n_suggest=self.n_per_iter)
            elif self.coverage_agent is not None:
                ml_knobs = self.coverage_agent.suggest_knobs_for_iteration(
                    prev_results, n_suggest=self.n_per_iter)
            else:
                ml_knobs = [self._random_knobs() for _ in range(self.n_per_iter)]

            ml_knobs = [self._apply_knob_overrides(k) for k in ml_knobs]

            ids = self._new_sim_ids(self.n_per_iter)
            iter_r = self._score_results_with_bin_bonuses(
                self.runner.run_batch(ml_knobs, ids.start, label=f"ml_iter{global_it}")
            )
            self.ml_results.append(iter_r)
            self.all_results.extend(iter_r)
            prev_results = iter_r
            if self._arm_reached_full_coverage("ml"):
                print("  [Coverage] ML arm reached 100.0% - stopping ML phase early")
                break

        self._save()
        self._save_coverage_benchmark()
        self._print_summary()

    def _save(self):
        os.makedirs(self.results_dir, exist_ok=True)
        canonical_all_results = list(self.random_results)
        for batch in self.ml_results:
            canonical_all_results.extend(batch)
        if canonical_all_results and len(canonical_all_results) != len(self.all_results):
            print(
                f"[Results] Rebuilding all_results from tracked arms "
                f"({len(self.all_results)} -> {len(canonical_all_results)})"
            )
            self.all_results = canonical_all_results
        path = os.path.join(self.results_dir, "experiment_results.json")
        with open(path, "w") as f:
            json.dump({
                "config": {
                    "agent_type": self.agent_type,
                    "n_init": self.n_init,
                    "n_iters": self.n_iters,
                    "n_per_iter": self.n_per_iter,
                    "fault_bug_define": os.environ.get("PICORV32_BUG_DEFINE", ""),
                    "reward_mode": self.reward_mode,
                    "dqn_reward_mode": os.environ.get("PICORV32_DQN_REWARD_MODE", "A"),
                    "alpha_new_bin": ALPHA_NEW_BIN,
                    "beta_rare_bin": BETA_RARE_BIN,
                    "gamma_remaining_bin": GAMMA_REMAINING_BIN,
                    "ml_only": self.ml_only,
                    "knob_overrides": self.knob_overrides,
                    "program_seed_base": self.program_seed_base,
                },
                "functional_bin_catalog": FUNCTIONAL_BIN_CATALOG,
                "all_results": self.all_results,
            }, f, indent=2)
        print(f"\n[Results] Saved to {path}")
        csv_path = os.path.join(self.results_dir, "experiment_results.csv")
        _write_results_csv(csv_path, self.all_results)
        print(f"[Results] Saved CSV to {csv_path}")

        if self.sup is not None:
            self.sup.save(os.path.join(self.results_dir, "supervised_model.pt"))
        if self.coverage_agent is not None:
            self.coverage_agent.save(os.path.join(self.results_dir, coverage_agent_model_filename(self.agent_type)))

    @staticmethod
    def _coverage_curve(results: List[Dict]) -> Dict[str, List[float]]:
        seen = set()
        xs, ys, cycles, elapsed_secs = [], [], [], []
        total_cycles = 0
        total_elapsed = 0.0
        for idx, r in enumerate(results, 1):
            for b in r["coverage"].get("functional_bins", []):
                if b in FUNCTIONAL_BIN_CATALOG:
                    seen.add(b)
            total_cycles += int(r["coverage"].get("total_cycles", 0))
            total_elapsed += float(r["coverage"].get("elapsed_sec", 0.0) or 0.0)
            xs.append(idx)
            ys.append(len(seen) / len(FUNCTIONAL_BIN_CATALOG))
            cycles.append(total_cycles)
            elapsed_secs.append(total_elapsed)
        return {
            "sim_count": xs,
            "cumulative_cycles": cycles,
            "cumulative_elapsed_sec": elapsed_secs,
            "coverage_ratio": ys,
            "coverage_percent": [v * 100.0 for v in ys],
        }

    @staticmethod
    def _sims_to_threshold(curve: Dict[str, List[float]], threshold: float) -> Optional[int]:
        for sim_count, ratio in zip(curve["sim_count"], curve["coverage_ratio"]):
            if ratio >= threshold:
                return sim_count
        return None

    @staticmethod
    def _cycles_to_threshold(curve: Dict[str, List[float]], threshold: float) -> Optional[int]:
        for cycles, ratio in zip(curve.get("cumulative_cycles", []), curve["coverage_ratio"]):
            if ratio >= threshold:
                return int(cycles)
        return None

    @staticmethod
    def _seconds_to_threshold(curve: Dict[str, List[float]], threshold: float) -> Optional[float]:
        for elapsed_sec, ratio in zip(curve.get("cumulative_elapsed_sec", []), curve["coverage_ratio"]):
            if ratio >= threshold:
                return float(elapsed_sec)
        return None

    @staticmethod
    def _coverage_percent_from_counter(counter: Counter) -> float:
        total_bins = len(FUNCTIONAL_BIN_CATALOG)
        if total_bins <= 0:
            return 0.0
        covered = sum(1 for count in counter.values() if count > 0)
        return 100.0 * covered / total_bins

    def _arm_reached_full_coverage(self, arm_name: str) -> bool:
        if arm_name == "random":
            counter = self.random_arm_bin_hit_counts
        elif arm_name == "ml":
            counter = self.ml_arm_bin_hit_counts
        else:
            return False
        return self._coverage_percent_from_counter(counter) >= 100.0

    def _save_coverage_benchmark(self):
        ml_seq = [r for r in self.all_results if str(r.get("label", "")).startswith("ml_iter")]
        rand_seq = [r for r in self.all_results if str(r.get("label", "")).startswith("random_iter")]
        seed_seq = list(self.initial_random_results)

        if not ml_seq and not rand_seq:
            return

        ml_curve = self._coverage_curve(ml_seq) if ml_seq else {"sim_count": [], "coverage_ratio": [], "coverage_percent": []}
        rand_curve = self._coverage_curve(rand_seq) if rand_seq else {"sim_count": [], "coverage_ratio": [], "coverage_percent": []}
        seed_curve = self._coverage_curve(seed_seq) if seed_seq else {"sim_count": [], "coverage_ratio": [], "coverage_percent": []}

        summary = {
            "total_bins": len(FUNCTIONAL_BIN_CATALOG),
            "catalog": FUNCTIONAL_BIN_CATALOG,
            "seed": {
                "curve": seed_curve,
                "final_coverage_percent": seed_curve["coverage_percent"][-1] if seed_curve.get("coverage_percent") else 0.0,
                "seconds_to_100": self._seconds_to_threshold(seed_curve, 1.00),
            },
            "random": {
                "curve": rand_curve,
                "final_coverage_percent": rand_curve["coverage_percent"][-1] if rand_curve.get("coverage_percent") else 0.0,
                "sims_to_70": self._sims_to_threshold(rand_curve, 0.70),
                "sims_to_90": self._sims_to_threshold(rand_curve, 0.90),
                "sims_to_95": self._sims_to_threshold(rand_curve, 0.95),
                "sims_to_100": self._sims_to_threshold(rand_curve, 1.00),
                "cycles_to_70": self._cycles_to_threshold(rand_curve, 0.70),
                "cycles_to_90": self._cycles_to_threshold(rand_curve, 0.90),
                "cycles_to_95": self._cycles_to_threshold(rand_curve, 0.95),
                "cycles_to_100": self._cycles_to_threshold(rand_curve, 1.00),
                "seconds_to_100": self._seconds_to_threshold(rand_curve, 1.00),
            },
            "ml": {
                "curve": ml_curve,
                "final_coverage_percent": ml_curve["coverage_percent"][-1] if ml_curve.get("coverage_percent") else 0.0,
                "sims_to_70": self._sims_to_threshold(ml_curve, 0.70),
                "sims_to_90": self._sims_to_threshold(ml_curve, 0.90),
                "sims_to_95": self._sims_to_threshold(ml_curve, 0.95),
                "sims_to_100": self._sims_to_threshold(ml_curve, 1.00),
                "cycles_to_70": self._cycles_to_threshold(ml_curve, 0.70),
                "cycles_to_90": self._cycles_to_threshold(ml_curve, 0.90),
                "cycles_to_95": self._cycles_to_threshold(ml_curve, 0.95),
                "cycles_to_100": self._cycles_to_threshold(ml_curve, 1.00),
                "seconds_to_100": self._seconds_to_threshold(ml_curve, 1.00),
            },
        }

        out_path = os.path.join(self.results_dir, "coverage_closure.json")
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[Coverage] Saved closure summary to {out_path}")
        csv_path = os.path.join(self.results_dir, "coverage_closure.csv")
        _write_two_arm_curve_csv(csv_path, rand_curve, ml_curve)
        print(f"[Coverage] Saved closure CSV to {csv_path}")

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Coverage] matplotlib not installed - skipping closure plot")
            return

        plt.figure(figsize=(10, 6))
        if rand_curve["sim_count"]:
            plt.plot(rand_curve["sim_count"], rand_curve["coverage_percent"],
                     "b-", lw=2, label="Random")
        if ml_curve["sim_count"]:
            plt.plot(ml_curve["sim_count"], ml_curve["coverage_percent"],
                     "r-", lw=2, label="ML")
        for threshold, color in ((70, "#9ca3af"), (90, "#6b7280"), (95, "#4b5563"), (100, "#111827")):
            plt.axhline(threshold, color=color, linestyle="--", linewidth=1)
        plt.xlabel("Simulation Count")
        plt.ylabel("Cumulative Functional Coverage (%)")
        plt.title("Coverage Closure: ML vs Random")
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plot_path = os.path.join(self.results_dir, "coverage_closure_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[Coverage] Saved closure plot to {plot_path}")

    def _print_summary(self):
        print("\n" + "="*60)
        print("  EXPERIMENT SUMMARY")
        print("="*60)

        rand_init_results = self.random_results if self.eval_mode else self.initial_random_results
        if rand_init_results:
            rand_init_reward = self._metric_mean(
                rand_init_results, "total_reward")
            rand_init = self._metric_mean(
                rand_init_results, "stall_ratio")
            rand_label = "Random eval" if self.eval_mode else "Random init"
            print(f"  {rand_label} mean reward:       {rand_init_reward:.4f}")
            print(f"  {rand_label} mean stall_ratio:  {rand_init:.4f}")

        if self.ml_results and self.ml_results[-1]:
            ml_final_reward = self._metric_mean(self.ml_results[-1], "total_reward")
            ml_final = self._metric_mean(self.ml_results[-1], "stall_ratio")
            print(f"  ML final iter mean reward:     {ml_final_reward:.4f}")
            print(f"  ML final iter mean stall_ratio: {ml_final:.4f}")
            if rand_init_results:
                reward_gain = (ml_final_reward - rand_init_reward) / max(rand_init_reward, 1e-6) * 100
                gain = (ml_final - rand_init) / max(rand_init, 1e-6) * 100
                if self.eval_mode:
                    print(f"  Improvement over random eval reward: {reward_gain:+.1f}%")
                    print(f"  Improvement over random eval stall_ratio: {gain:+.1f}%")
                else:
                    print(f"  Improvement over first random init reward: {reward_gain:+.1f}%")
                    print(f"  Improvement over first random init stall_ratio: {gain:+.1f}%")
            if self.prior_last_ml_results and self.ml_results[-1] is not self.prior_last_ml_results:
                prev_reward = self._metric_mean(self.prior_last_ml_results, "total_reward")
                prev_stall = self._metric_mean(self.prior_last_ml_results, "stall_ratio")
                reward_gain = (ml_final_reward - prev_reward) / max(prev_reward, 1e-6) * 100
                stall_gain = (ml_final - prev_stall) / max(prev_stall, 1e-6) * 100
                print(f"  Improvement over previous ML reward: {reward_gain:+.1f}%")
                print(f"  Improvement over previous ML stall_ratio: {stall_gain:+.1f}%")

        ml_seq = [r for r in self.all_results if str(r.get("label", "")).startswith("ml_iter")]
        rand_seq = [r for r in self.all_results if str(r.get("label", "")).startswith("random_iter")]
        if ml_seq or rand_seq:
            rand_curve = self._coverage_curve(rand_seq) if rand_seq else {"sim_count": [], "coverage_ratio": []}
            ml_curve = self._coverage_curve(ml_seq) if ml_seq else {"sim_count": [], "coverage_ratio": []}
            print(f"  Functional bins tracked:       {len(FUNCTIONAL_BIN_CATALOG)}")
            if rand_seq:
                print(f"  Random sims to 70/90/95/100%:  "
                      f"{self._sims_to_threshold(rand_curve, 0.70)}/"
                      f"{self._sims_to_threshold(rand_curve, 0.90)}/"
                      f"{self._sims_to_threshold(rand_curve, 0.95)}/"
                      f"{self._sims_to_threshold(rand_curve, 1.00)}")
                print(f"  Random seconds to 100%:        "
                      f"{self._seconds_to_threshold(rand_curve, 1.00)}")
            if ml_seq:
                print(f"  ML sims to 70/90/95/100%:      "
                      f"{self._sims_to_threshold(ml_curve, 0.70)}/"
                      f"{self._sims_to_threshold(ml_curve, 0.90)}/"
                      f"{self._sims_to_threshold(ml_curve, 0.95)}/"
                      f"{self._sims_to_threshold(ml_curve, 1.00)}")
                print(f"  ML seconds to 100%:            "
                      f"{self._seconds_to_threshold(ml_curve, 1.00)}")

        if (
            self.reward_mode == "composite"
            and any(bool(r.get("coverage", {}).get("oracle_enabled", False)) for r in self.all_results)
        ):
            rand_bug = _first_bug_detection(rand_seq) if rand_seq else {"detected": False, "sim": None, "cycles": None}
            ml_bug = _first_bug_detection(ml_seq) if ml_seq else {"detected": False, "sim": None, "cycles": None}
            print(f"  Random first bug:             "
                  f"{rand_bug['sim']}/{rand_bug['cycles']} cycles ({rand_bug.get('reason', '') or 'none'})")
            print(f"  ML first bug:                 "
                  f"{ml_bug['sim']}/{ml_bug['cycles']} cycles ({ml_bug.get('reason', '') or 'none'})")

        print(f"\n  Total simulations run: {self._next_sim_id}")
        print(f"  Results directory:     {self.results_dir}")
        print("="*60)


def _knob_str(knobs: dict) -> str:
    return (f"L={knobs['load_weight']} S={knobs['store_weight']} "
            f"B={knobs['branch_weight']} J={knobs['jump_weight']} "
            f"A={knobs['arith_weight']} stride={knobs['mem_stride']} "
            f"ptr={knobs['pointer_update_rate']} trap={knobs['trap_rate']}/"
            f"{knobs['trap_kind']} brbias={knobs['branch_taken_bias']} "
            f"mix={knobs['mixed_burst_bias']} "
            f"delay={knobs['mem_delay_base']}")


def _compile_meta_path(sim_lib: str) -> str:
    return os.path.join(PROJECT_ROOT, sim_lib, "compile_meta.txt")


def _read_compile_meta(sim_lib: str) -> Dict[str, str]:
    meta = {}
    path = _compile_meta_path(sim_lib)
    if not os.path.exists(path):
        return meta
    with open(path) as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = line.strip().split("=", 1)
            meta[key.strip()] = value.strip()
    return meta


def ensure_sim_library(questa_cmd: str, sim_lib: str, bug_define: str = "") -> None:
    bug_define = (bug_define or "").strip()
    meta = _read_compile_meta(sim_lib)
    library_ready = os.path.exists(os.path.join(PROJECT_ROOT, sim_lib, "_info"))
    if library_ready and meta.get("bug_define", "") == bug_define and meta.get("work_lib", sim_lib) == sim_lib:
        return

    print(f"\n[Compile] Preparing simulation library '{sim_lib}' "
          f"({'clean RTL' if not bug_define else bug_define})")
    env = os.environ.copy()
    env["PICORV32_WORK_LIB"] = sim_lib
    if bug_define:
        env["PICORV32_BUG_DEFINE"] = bug_define
    else:
        env.pop("PICORV32_BUG_DEFINE", None)

    cmd = [questa_cmd, "-c", "-do", "do scripts/compile.tcl; quit -f", "-quiet"]
    result = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.returncode != 0:
        sys.stdout.write(result.stdout)
        raise RuntimeError(f"Compilation failed for simulation library '{sim_lib}'")

    meta = _read_compile_meta(sim_lib)
    if meta.get("bug_define", "") != bug_define:
        raise RuntimeError(
            f"Compiled library '{sim_lib}' metadata mismatch: expected bug_define='{bug_define}', "
            f"got '{meta.get('bug_define', '')}'"
        )


def check_environment(questa_cmd=QUESTA_CMD):
    print("\n[Environment Check]")
    ok = True
    active_bug_define = os.environ.get("PICORV32_BUG_DEFINE", "")
    active_sim_lib = os.environ.get("PICORV32_WORK_LIB", "work")

    if shutil.which(questa_cmd):
        print(f"  O QuestaSim: {shutil.which(questa_cmd)}")
    else:
        print(f"  X QuestaSim '{questa_cmd}' not on PATH")
        ok = False

    rv32 = os.path.join(PROJECT_ROOT, "picorv32.v")
    if os.path.exists(rv32):
        print(f"  O PicoRV32 RTL: {rv32}")
    else:
        print(f"  X PicoRV32 RTL not found: {rv32}")
        ok = False

    for ext in ("dll", "so"):
        lib = os.path.join(PROJECT_ROOT, "dpi", f"knobio.{ext}")
        if os.path.exists(lib):
            print(f"  O DPI library: {lib}")
            break
    else:
        print("  ! DPI library not found in dpi/ - run: do scripts/compile.tcl")

    print(f"  {'O' if os.path.exists(WORK_DIR) else '!'} work/ directory: {WORK_DIR}")
    print(f"  O Simulation library target: {active_sim_lib}")

    for pkg in ("numpy", "torch", "matplotlib"):
        try:
            __import__(pkg)
            print(f"  O Python: {pkg}")
        except ImportError:
            print(f"  ! Python: {pkg} missing  (pip install {pkg})")

    if active_bug_define:
        print(f"  O Fault injection: {active_bug_define}")
    else:
        print("  O Fault injection: none")

    print(f"\n  Overall: {'READY' if ok else 'ISSUES FOUND - fix above before running'}")
    return ok


def _next_fresh_result_dir(base_dir: str) -> str:
    max_idx = 0
    if os.path.exists(base_dir):
        for name in os.listdir(base_dir):
            full = os.path.join(base_dir, name)
            if not os.path.isdir(full):
                continue
            if not name.startswith("fresh_"):
                continue
            suffix = name[len("fresh_"):]
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))
    return os.path.join(base_dir, f"fresh_{max_idx + 1}")


def _next_eval_result_dir(base_dir: str) -> str:
    max_idx = 0
    if os.path.exists(base_dir):
        for name in os.listdir(base_dir):
            full = os.path.join(base_dir, name)
            if not os.path.isdir(full):
                continue
            if not name.startswith("eval_"):
                continue
            suffix = name[len("eval_"):]
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))
    return os.path.join(base_dir, f"eval_{max_idx + 1}")


def _next_benchmark_result_dir(base_dir: str) -> str:
    max_idx = 0
    if os.path.exists(base_dir):
        for name in os.listdir(base_dir):
            full = os.path.join(base_dir, name)
            if not os.path.isdir(full):
                continue
            if not name.startswith("benchmark_"):
                continue
            suffix = name[len("benchmark_"):]
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))
    return os.path.join(base_dir, f"benchmark_{max_idx + 1}")


def _next_fault_benchmark_result_dir(base_dir: str) -> str:
    max_idx = 0
    if os.path.exists(base_dir):
        for name in os.listdir(base_dir):
            full = os.path.join(base_dir, name)
            if not os.path.isdir(full):
                continue
            if not name.startswith("fault_benchmark_"):
                continue
            suffix = name[len("fault_benchmark_"):]
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))
    return os.path.join(base_dir, f"fault_benchmark_{max_idx + 1}")


def _next_reachability_result_dir(base_dir: str) -> str:
    max_idx = 0
    if os.path.exists(base_dir):
        for name in os.listdir(base_dir):
            full = os.path.join(base_dir, name)
            if not os.path.isdir(full):
                continue
            if not name.startswith("reachability_"):
                continue
            suffix = name[len("reachability_"):]
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))
    return os.path.join(base_dir, f"reachability_{max_idx + 1}")


def _resolve_resume_result_dir(base_dir: str, spec: str) -> str:
    if not spec:
        return base_dir

    if spec.lower() == "latest":
        candidates = []
        if os.path.exists(base_dir):
            for name in os.listdir(base_dir):
                full = os.path.join(base_dir, name)
                if os.path.isdir(full) and name.startswith("fresh_"):
                    suffix = name[len("fresh_"):]
                    if suffix.isdigit():
                        candidates.append((int(suffix), full))
        if not candidates:
            raise FileNotFoundError("No numbered fresh runs found under results/")
        return max(candidates, key=lambda item: item[0])[1]

    if os.path.isabs(spec):
        return spec

    return os.path.join(base_dir, spec)


def _coverage_curve_from_seed(seed_results: List[Dict], arm_results: List[Dict]) -> Dict[str, List[float]]:
    seen = set()
    for r in seed_results:
        for b in r.get("coverage", {}).get("functional_bins", []):
            if b in FUNCTIONAL_BIN_CATALOG:
                seen.add(b)

    xs = [0]
    ys = [len(seen) / len(FUNCTIONAL_BIN_CATALOG)]
    cycles = [0]
    elapsed_secs = [0.0]
    total_cycles = 0
    total_elapsed = 0.0
    for idx, r in enumerate(arm_results, 1):
        for b in r.get("coverage", {}).get("functional_bins", []):
            if b in FUNCTIONAL_BIN_CATALOG:
                seen.add(b)
        total_cycles += int(r.get("coverage", {}).get("total_cycles", 0))
        total_elapsed += float(r.get("coverage", {}).get("elapsed_sec", 0.0) or 0.0)
        xs.append(idx)
        ys.append(len(seen) / len(FUNCTIONAL_BIN_CATALOG))
        cycles.append(total_cycles)
        elapsed_secs.append(total_elapsed)

    return {
        "sim_count": xs,
        "cumulative_cycles": cycles,
        "cumulative_elapsed_sec": elapsed_secs,
        "coverage_ratio": ys,
        "coverage_percent": [v * 100.0 for v in ys],
    }


def _coverage_curve_restart(arm_results: List[Dict]) -> Dict[str, List[float]]:
    xs = [0]
    ys = [0.0]
    cycles = [0]
    elapsed_secs = [0.0]
    seen = set()
    total_cycles = 0
    total_elapsed = 0.0
    for idx, r in enumerate(arm_results, 1):
        for b in r.get("coverage", {}).get("functional_bins", []):
            if b in FUNCTIONAL_BIN_CATALOG:
                seen.add(b)
        total_cycles += int(r.get("coverage", {}).get("total_cycles", 0))
        total_elapsed += float(r.get("coverage", {}).get("elapsed_sec", 0.0) or 0.0)
        xs.append(idx)
        ys.append(len(seen) / len(FUNCTIONAL_BIN_CATALOG))
        cycles.append(total_cycles)
        elapsed_secs.append(total_elapsed)
    return {
        "sim_count": xs,
        "cumulative_cycles": cycles,
        "cumulative_elapsed_sec": elapsed_secs,
        "coverage_ratio": ys,
        "coverage_percent": [v * 100.0 for v in ys],
    }


def _sims_to_threshold_from_curve(curve: Dict[str, List[float]], threshold: float) -> Optional[int]:
    for sim_count, ratio in zip(curve["sim_count"], curve["coverage_ratio"]):
        if ratio >= threshold:
            return sim_count
    return None


def _cycles_to_threshold_from_curve(curve: Dict[str, List[float]], threshold: float) -> Optional[int]:
    for cycles, ratio in zip(curve.get("cumulative_cycles", []), curve["coverage_ratio"]):
        if ratio >= threshold:
            return int(cycles)
    return None


def _seconds_to_threshold_from_curve(curve: Dict[str, List[float]], threshold: float) -> Optional[float]:
    for elapsed_sec, ratio in zip(curve.get("cumulative_elapsed_sec", []), curve["coverage_ratio"]):
        if ratio >= threshold:
            return float(elapsed_sec)
    return None


def _first_bug_detection(results: List[Dict]) -> Dict[str, Optional[object]]:
    cumulative_cycles = 0
    cumulative_runtime_sec = 0.0
    for idx, r in enumerate(results, 1):
        cov = r.get("coverage", {})
        cumulative_cycles += int(cov.get("total_cycles", 0))
        cumulative_runtime_sec += float(cov.get("elapsed_sec", 0.0) or 0.0)
        if bool(cov.get("bug_detected", False)):
            return {
                "detected": True,
                "sim": idx,
                "cycles": cumulative_cycles,
                "runtime_sec": cumulative_runtime_sec,
                "sim_id": int(r.get("sim_id", -1)),
                "reason": str(cov.get("bug_detection_reason", "")),
            }
    return {
        "detected": False,
        "sim": None,
        "cycles": None,
        "runtime_sec": None,
        "sim_id": None,
        "reason": "",
    }


def _bug_detection_curve(results: List[Dict]) -> Dict[str, List[float]]:
    xs = [0]
    ys = [0.0]
    cycles = [0]
    cumulative_cycles = 0
    detected = False
    for idx, r in enumerate(results, 1):
        cov = r.get("coverage", {})
        cumulative_cycles += int(cov.get("total_cycles", 0))
        if bool(cov.get("bug_detected", False)):
            detected = True
        xs.append(idx)
        ys.append(100.0 if detected else 0.0)
        cycles.append(cumulative_cycles)
    return {
        "sim_count": xs,
        "cumulative_cycles": cycles,
        "detection_percent": ys,
    }


def _pad_curve(curve: Dict[str, List[float]], target_len: int, carry_keys: List[str]) -> Dict[str, List[float]]:
    padded = {k: list(v) for k, v in curve.items()}
    for key in carry_keys:
        values = padded.get(key, [])
        if not values:
            values = [0]
            padded[key] = values
        pad_value = values[-1]
        while len(values) < target_len:
            values.append(pad_value)
    if "sim_count" in padded:
        padded["sim_count"] = list(range(target_len))
    return padded


def _mean_std(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"mean": None, "std": None, "count": 0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "count": int(arr.size),
    }


def _threshold_stats(values: List[Optional[int]]) -> Dict[str, Optional[float]]:
    completed = [float(v) for v in values if v is not None]
    stats = _mean_std(completed)
    stats["hit_count"] = len(completed)
    stats["trial_count"] = len(values)
    return stats


def _bin_hit_distribution(results: List[Dict]) -> Counter:
    counts = Counter()
    for r in results:
        for b in r.get("coverage", {}).get("functional_bins", []):
            if b in FUNCTIONAL_BIN_CATALOG:
                counts[b] += 1
    return counts


def _write_results_csv(path: str, results: List[Dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = [
            "sim_id",
            "label",
            "program_seed",
            "oracle_program_seed",
            "total_cycles",
            "stall_ratio",
            "max_stall_run",
            "transition_types_hit",
            "transition_entropy",
            "load_store_alternation_ratio",
            "mixed_data_transition_count",
            "branch_instr_frac",
            "instr_count",
            "composite_reward",
            "total_reward",
            "functional_bin_count",
            "bug_detected",
            "bug_detection_reason",
            "functional_bins",
        ] + FUNCTIONAL_BIN_CATALOG
        writer.writerow(header)
        for r in results:
            cov = r.get("coverage", {})
            row = [
                r.get("sim_id"),
                r.get("label", ""),
                cov.get("program_seed"),
                cov.get("oracle_program_seed"),
                cov.get("total_cycles"),
                cov.get("stall_ratio"),
                cov.get("max_stall_run"),
                cov.get("transition_types_hit"),
                cov.get("transition_entropy"),
                cov.get("load_store_alternation_ratio"),
                cov.get("mixed_data_transition_count"),
                cov.get("branch_instr_frac"),
                cov.get("instr_count"),
                cov.get("composite_reward"),
                cov.get("total_reward", cov.get("composite_reward")),
                cov.get("functional_bin_count"),
                int(bool(cov.get("bug_detected", False))),
                cov.get("bug_detection_reason", ""),
                ";".join(cov.get("functional_bins", [])),
            ]
            bin_dist = cov.get("functional_bin_distribution", {})
            row.extend(int(bin_dist.get(bin_name, 0)) for bin_name in FUNCTIONAL_BIN_CATALOG)
            writer.writerow(row)


def _write_two_arm_curve_csv(path: str,
                             random_curve: Dict[str, List[float]],
                             ml_curve: Dict[str, List[float]]) -> None:
    max_len = max(len(random_curve.get("sim_count", [])), len(ml_curve.get("sim_count", [])))
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "row",
            "random_sim_count",
            "random_cumulative_cycles",
            "random_coverage_ratio",
            "random_coverage_percent",
            "ml_sim_count",
            "ml_cumulative_cycles",
            "ml_coverage_ratio",
            "ml_coverage_percent",
        ])
        for i in range(max_len):
            writer.writerow([
                i,
                random_curve.get("sim_count", [None] * max_len)[i] if i < len(random_curve.get("sim_count", [])) else None,
                random_curve.get("cumulative_cycles", [None] * max_len)[i] if i < len(random_curve.get("cumulative_cycles", [])) else None,
                random_curve.get("coverage_ratio", [None] * max_len)[i] if i < len(random_curve.get("coverage_ratio", [])) else None,
                random_curve.get("coverage_percent", [None] * max_len)[i] if i < len(random_curve.get("coverage_percent", [])) else None,
                ml_curve.get("sim_count", [None] * max_len)[i] if i < len(ml_curve.get("sim_count", [])) else None,
                ml_curve.get("cumulative_cycles", [None] * max_len)[i] if i < len(ml_curve.get("cumulative_cycles", [])) else None,
                ml_curve.get("coverage_ratio", [None] * max_len)[i] if i < len(ml_curve.get("coverage_ratio", [])) else None,
                ml_curve.get("coverage_percent", [None] * max_len)[i] if i < len(ml_curve.get("coverage_percent", [])) else None,
            ])


def _write_bin_distribution_csv(path: str,
                                random_counts: Counter,
                                ml_counts: Counter,
                                n_random_sims: int,
                                n_ml_sims: int,
                                n_trials: int = 1) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bin",
            "is_cross_bin",
            "random_hit_count",
            "random_hit_rate_per_sim",
            "random_mean_hits_per_trial",
            "ml_hit_count",
            "ml_hit_rate_per_sim",
            "ml_mean_hits_per_trial",
        ])
        for bin_name in FUNCTIONAL_BIN_CATALOG:
            random_count = int(random_counts.get(bin_name, 0))
            ml_count = int(ml_counts.get(bin_name, 0))
            writer.writerow([
                bin_name,
                "yes" if bin_name.startswith("cross_") else "no",
                random_count,
                (random_count / n_random_sims) if n_random_sims else 0.0,
                (random_count / n_trials) if n_trials else 0.0,
                ml_count,
                (ml_count / n_ml_sims) if n_ml_sims else 0.0,
                (ml_count / n_trials) if n_trials else 0.0,
            ])


REACHABILITY_TARGET_BINS = [
    "b2b_pressure_low",
    "stall_depth_tiny",
    "stall_type_data_dominant",
]


REACHABILITY_FAMILIES = [
    {
        "name": "tiny_stall_arith_heavy",
        "goal": "Push toward tiny-stall and low-b2b behaviour with minimal memory pressure.",
        "knobs": {
            "load_weight": 1,
            "store_weight": 1,
            "branch_weight": 2,
            "jump_weight": 2,
            "arith_weight": 10,
            "mem_stride": 1,
            "pointer_update_rate": 1,
            "trap_rate": 0,
            "trap_kind": 0,
            "branch_taken_bias": 5,
            "mixed_burst_bias": 0,
            "mem_delay_base": 1,
        },
        "mutate_keys": ["arith_weight", "branch_weight", "jump_weight", "mem_delay_base"],
    },
    {
        "name": "tiny_stall_control_mix",
        "goal": "Low memory traffic with more control activity, still targeting tiny-stall behaviour.",
        "knobs": {
            "load_weight": 1,
            "store_weight": 1,
            "branch_weight": 8,
            "jump_weight": 7,
            "arith_weight": 6,
            "mem_stride": 2,
            "pointer_update_rate": 1,
            "trap_rate": 0,
            "trap_kind": 0,
            "branch_taken_bias": 8,
            "mixed_burst_bias": 0,
            "mem_delay_base": 1,
        },
        "mutate_keys": ["branch_weight", "jump_weight", "arith_weight", "mem_delay_base"],
    },
    {
        "name": "data_dominant_load_heavy",
        "goal": "Try to make data stalls dominate by keeping load traffic high and control low.",
        "knobs": {
            "load_weight": 10,
            "store_weight": 6,
            "branch_weight": 1,
            "jump_weight": 1,
            "arith_weight": 1,
            "mem_stride": 1,
            "pointer_update_rate": 10,
            "trap_rate": 0,
            "trap_kind": 0,
            "branch_taken_bias": 1,
            "mixed_burst_bias": 1,
            "mem_delay_base": 8,
        },
        "mutate_keys": ["load_weight", "store_weight", "pointer_update_rate", "mem_delay_base"],
    },
    {
        "name": "data_dominant_store_heavy",
        "goal": "Bias toward store-side pressure under long delay.",
        "knobs": {
            "load_weight": 5,
            "store_weight": 10,
            "branch_weight": 1,
            "jump_weight": 1,
            "arith_weight": 1,
            "mem_stride": 1,
            "pointer_update_rate": 9,
            "trap_rate": 0,
            "trap_kind": 0,
            "branch_taken_bias": 1,
            "mixed_burst_bias": 0,
            "mem_delay_base": 8,
        },
        "mutate_keys": ["load_weight", "store_weight", "pointer_update_rate", "mem_delay_base"],
    },
    {
        "name": "data_dominant_balanced_mem",
        "goal": "Balanced load/store traffic with minimal control to probe data-dominant stall classification.",
        "knobs": {
            "load_weight": 9,
            "store_weight": 9,
            "branch_weight": 1,
            "jump_weight": 1,
            "arith_weight": 2,
            "mem_stride": 2,
            "pointer_update_rate": 8,
            "trap_rate": 0,
            "trap_kind": 0,
            "branch_taken_bias": 2,
            "mixed_burst_bias": 2,
            "mem_delay_base": 7,
        },
        "mutate_keys": ["load_weight", "store_weight", "mixed_burst_bias", "mem_delay_base"],
    },
]


def _mutate_reachability_knobs(base_knobs: Dict[str, int], mutate_keys: List[str]) -> Dict[str, int]:
    knobs = dict(base_knobs)
    n_changes = min(len(mutate_keys), random.randint(1, 3))
    for key in random.sample(mutate_keys, k=n_changes):
        values = KNOB_RANGES[key]
        current = knobs[key]
        idx = values.index(current)
        candidate_values = [current]
        if idx > 0:
            candidate_values.append(values[idx - 1])
        if idx + 1 < len(values):
            candidate_values.append(values[idx + 1])
        if len(values) > 3:
            candidate_values.append(random.choice(values))
        knobs[key] = random.choice(candidate_values)
    return knobs


def run_reachability_check(args) -> None:
    result_dir = _next_reachability_result_dir(RESULTS_DIR)
    os.makedirs(result_dir, exist_ok=True)
    print(f"[Reachability] Writing targeted check to: {result_dir}")

    runner = SimulationRunner(questa_cmd=args.questa, include_fault_fields=False)
    sim_id = 0
    all_results: List[Dict] = []
    first_hits: Dict[str, Dict] = {}
    family_rows = []

    for family in REACHABILITY_FAMILIES:
        family_name = family["name"]
        family_goal = family["goal"]
        family_hits = Counter()
        print(f"\n[Reachability] Family: {family_name}")
        print(f"  Goal: {family_goal}")

        for _ in range(args.reachability_per_family):
            knobs = _mutate_reachability_knobs(family["knobs"], family["mutate_keys"])
            cov = runner.run_one(sim_id, knobs)
            if cov is None:
                sim_id += 1
                continue

            hits = [b for b in cov.get("functional_bins", []) if b in REACHABILITY_TARGET_BINS]
            for b in hits:
                family_hits[b] += 1
                if b not in first_hits:
                    first_hits[b] = {
                        "sim_id": sim_id,
                        "family": family_name,
                        "knobs": dict(knobs),
                        "stall_ratio": cov.get("stall_ratio", 0.0),
                        "max_stall_run": cov.get("max_stall_run", 0),
                        "total_cycles": cov.get("total_cycles", 0),
                    }

            all_results.append({
                "sim_id": sim_id,
                "family": family_name,
                "goal": family_goal,
                "knobs": knobs,
                "coverage": cov,
                "target_hits": hits,
            })
            sim_id += 1

        family_rows.append({
            "family": family_name,
            "goal": family_goal,
            "sims": args.reachability_per_family,
            "b2b_pressure_low_hits": family_hits["b2b_pressure_low"],
            "stall_depth_tiny_hits": family_hits["stall_depth_tiny"],
            "stall_type_data_dominant_hits": family_hits["stall_type_data_dominant"],
        })

    summary = {
        "config": {
            "reachability_per_family": args.reachability_per_family,
            "target_bins": REACHABILITY_TARGET_BINS,
            "fault_bug_define": os.environ.get("PICORV32_BUG_DEFINE", ""),
        },
        "families": family_rows,
        "first_hits": first_hits,
        "all_results": all_results,
    }

    summary_path = os.path.join(result_dir, "reachability_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[Reachability] Saved summary to {summary_path}")

    csv_path = os.path.join(result_dir, "reachability_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "family",
            "goal",
            "sims",
            "b2b_pressure_low_hits",
            "stall_depth_tiny_hits",
            "stall_type_data_dominant_hits",
        ])
        for row in family_rows:
            writer.writerow([
                row["family"],
                row["goal"],
                row["sims"],
                row["b2b_pressure_low_hits"],
                row["stall_depth_tiny_hits"],
                row["stall_type_data_dominant_hits"],
            ])
    print(f"[Reachability] Saved family CSV to {csv_path}")

    print("\n" + "="*60)
    print("  REACHABILITY SUMMARY")
    print("="*60)
    for target_bin in REACHABILITY_TARGET_BINS:
        if target_bin in first_hits:
            hit = first_hits[target_bin]
            print(f"  {target_bin}: HIT in sim {hit['sim_id']} via {hit['family']}")
        else:
            print(f"  {target_bin}: not hit")
    print(f"  Results directory: {result_dir}")
    print("="*60)


def run_benchmark(args) -> None:
    bench_dir = _next_benchmark_result_dir(RESULTS_DIR)
    os.makedirs(bench_dir, exist_ok=True)
    print(f"[Benchmark] Writing aggregate benchmark to: {bench_dir}")

    milestone_thresholds = COVERAGE_MILESTONE_THRESHOLDS
    trial_summaries = []
    random_curves = []
    ml_curves = []
    random_restart_curves = []
    ml_restart_curves = []
    random_warm_final = []
    ml_warm_final = []
    random_restart_final = []
    ml_restart_final = []
    random_warm_seconds_to_100 = []
    ml_warm_seconds_to_100 = []
    random_restart_seconds_to_100 = []
    ml_restart_seconds_to_100 = []
    random_warm_cycle_thresholds = {label: [] for _, label in milestone_thresholds}
    ml_warm_cycle_thresholds = {label: [] for _, label in milestone_thresholds}
    random_restart_cycle_thresholds = {label: [] for _, label in milestone_thresholds}
    ml_restart_cycle_thresholds = {label: [] for _, label in milestone_thresholds}
    ml_remaining_counter = Counter()
    aggregate_random_counts = Counter()
    aggregate_ml_counts = Counter()

    for trial_idx in range(1, args.benchmark_trials + 1):
        trial_seed = (int(args.seed) + trial_idx - 1) & 0xFFFFFFFF
        random.seed(trial_seed)
        np.random.seed(trial_seed)

        trial_dir = os.path.join(bench_dir, f"trial_{trial_idx:02d}")
        os.makedirs(trial_dir, exist_ok=True)
        print(f"\n[Benchmark] Trial {trial_idx}/{args.benchmark_trials}  seed={trial_seed}")

        exp = MLDVExperiment(
            agent_type=args.agent,
            n_init=args.init,
            n_iters=args.iters,
            n_per_iter=args.per_iter,
            questa_cmd=args.questa,
            resume=False,
            result_dir=trial_dir,
            eval_mode=False,
            ml_only=args.ml_only,
            model_source_dir=trial_dir,
            reward_mode=args.reward_mode,
            program_seed_base=(trial_seed * 1000003) & 0xFFFFFFFF,
        )
        exp.run()

        seed_results = list(exp.initial_random_results)
        random_arm = [
            r for r in exp.random_results
            if str(r.get("label", "")).startswith("random_iter")
        ]
        ml_arm = [r for batch in exp.ml_results for r in batch]

        random_curve = _coverage_curve_from_seed(seed_results, random_arm)
        ml_curve = _coverage_curve_from_seed(seed_results, ml_arm)
        random_restart_curve = _coverage_curve_restart(random_arm)
        ml_restart_curve = _coverage_curve_restart(ml_arm)

        random_curves.append(random_curve)
        ml_curves.append(ml_curve)
        random_restart_curves.append(random_restart_curve)
        ml_restart_curves.append(ml_restart_curve)
        random_warm_final.append(random_curve["coverage_percent"][-1])
        ml_warm_final.append(ml_curve["coverage_percent"][-1])
        random_restart_final.append(random_restart_curve["coverage_percent"][-1])
        ml_restart_final.append(ml_restart_curve["coverage_percent"][-1])
        random_warm_seconds_to_100.append(_seconds_to_threshold_from_curve(random_curve, 1.00))
        ml_warm_seconds_to_100.append(_seconds_to_threshold_from_curve(ml_curve, 1.00))
        random_restart_seconds_to_100.append(_seconds_to_threshold_from_curve(random_restart_curve, 1.00))
        ml_restart_seconds_to_100.append(_seconds_to_threshold_from_curve(ml_restart_curve, 1.00))

        trial_random_counts = _bin_hit_distribution(random_arm)
        trial_ml_counts = _bin_hit_distribution(ml_arm)
        aggregate_random_counts.update(trial_random_counts)
        aggregate_ml_counts.update(trial_ml_counts)
        _write_bin_distribution_csv(
            os.path.join(trial_dir, "bin_distribution.csv"),
            trial_random_counts,
            trial_ml_counts,
            len(random_arm),
            len(ml_arm),
            n_trials=1,
        )

        trial_summary = {
            "trial": trial_idx,
            "seed": trial_seed,
            "result_dir": trial_dir,
            "seed_coverage_percent": random_curve["coverage_percent"][0],
            "random_final_coverage_percent": random_restart_curve["coverage_percent"][-1],
            "ml_final_coverage_percent": ml_restart_curve["coverage_percent"][-1],
            "random_warm_started_final_coverage_percent": random_curve["coverage_percent"][-1],
            "ml_warm_started_final_coverage_percent": ml_curve["coverage_percent"][-1],
            "random_seconds_to_100": _seconds_to_threshold_from_curve(random_restart_curve, 1.00),
            "ml_seconds_to_100": _seconds_to_threshold_from_curve(ml_restart_curve, 1.00),
            "random_warm_started_seconds_to_100": _seconds_to_threshold_from_curve(random_curve, 1.00),
            "ml_warm_started_seconds_to_100": _seconds_to_threshold_from_curve(ml_curve, 1.00),
        }
        for threshold, label in milestone_thresholds:
            random_cycles = _cycles_to_threshold_from_curve(random_restart_curve, threshold)
            ml_cycles = _cycles_to_threshold_from_curve(ml_restart_curve, threshold)
            random_warm_cycles = _cycles_to_threshold_from_curve(random_curve, threshold)
            ml_warm_cycles = _cycles_to_threshold_from_curve(ml_curve, threshold)
            random_restart_cycles = _cycles_to_threshold_from_curve(random_restart_curve, threshold)
            ml_restart_cycles = _cycles_to_threshold_from_curve(ml_restart_curve, threshold)
            random_warm_cycle_thresholds[label].append(random_warm_cycles)
            ml_warm_cycle_thresholds[label].append(ml_warm_cycles)
            random_restart_cycle_thresholds[label].append(random_restart_cycles)
            ml_restart_cycle_thresholds[label].append(ml_restart_cycles)
            trial_summary[f"random_cycles_to_{label}"] = random_cycles
            trial_summary[f"ml_cycles_to_{label}"] = ml_cycles
            trial_summary[f"random_warm_started_cycles_to_{label}"] = random_warm_cycles
            trial_summary[f"ml_warm_started_cycles_to_{label}"] = ml_warm_cycles
            trial_summary[f"random_restart_cycles_to_{label}"] = random_restart_cycles
            trial_summary[f"ml_restart_cycles_to_{label}"] = ml_restart_cycles

        trial_summary["random_restart_final_coverage_percent"] = random_restart_curve["coverage_percent"][-1]
        trial_summary["ml_restart_final_coverage_percent"] = ml_restart_curve["coverage_percent"][-1]

        ml_seen = set()
        for r in seed_results + ml_arm:
            for b in r.get("coverage", {}).get("functional_bins", []):
                if b in FUNCTIONAL_BIN_CATALOG:
                    ml_seen.add(b)
        for b in FUNCTIONAL_BIN_CATALOG:
            if b not in ml_seen:
                ml_remaining_counter[b] += 1

        trial_summaries.append(trial_summary)

    arm_budget = args.iters * args.per_iter
    sim_axis = list(range(arm_budget + 1))
    target_len = arm_budget + 1
    random_curves = [
        _pad_curve(c, target_len, ["coverage_ratio", "coverage_percent", "cumulative_cycles"])
        for c in random_curves
    ]
    ml_curves = [
        _pad_curve(c, target_len, ["coverage_ratio", "coverage_percent", "cumulative_cycles"])
        for c in ml_curves
    ]
    random_restart_curves = [
        _pad_curve(c, target_len, ["coverage_ratio", "coverage_percent", "cumulative_cycles"])
        for c in random_restart_curves
    ]
    ml_restart_curves = [
        _pad_curve(c, target_len, ["coverage_ratio", "coverage_percent", "cumulative_cycles"])
        for c in ml_restart_curves
    ]

    random_matrix = np.asarray([c["coverage_percent"] for c in random_curves], dtype=float)
    ml_matrix = np.asarray([c["coverage_percent"] for c in ml_curves], dtype=float)
    random_restart_matrix = np.asarray([c["coverage_percent"] for c in random_restart_curves], dtype=float)
    ml_restart_matrix = np.asarray([c["coverage_percent"] for c in ml_restart_curves], dtype=float)

    aggregate = {
        "config": {
            "agent_type": args.agent,
            "n_init": args.init,
            "n_iters": args.iters,
            "n_per_iter": args.per_iter,
            "benchmark_trials": args.benchmark_trials,
        },
        "total_bins": len(FUNCTIONAL_BIN_CATALOG),
        "catalog": FUNCTIONAL_BIN_CATALOG,
        "trial_summaries": trial_summaries,
        "random": {
            "warm_started_overall": {
                "coverage_percent_mean": np.mean(random_matrix, axis=0).tolist(),
                "coverage_percent_std": np.std(random_matrix, axis=0).tolist(),
                "final_coverage_percent": _mean_std(random_warm_final),
                "seconds_to_100": _threshold_stats(random_warm_seconds_to_100),
                "cycles_to_70": _threshold_stats(random_warm_cycle_thresholds["70"]),
                "cycles_to_90": _threshold_stats(random_warm_cycle_thresholds["90"]),
                "cycles_to_95": _threshold_stats(random_warm_cycle_thresholds["95"]),
                "cycles_to_100": _threshold_stats(random_warm_cycle_thresholds["100"]),
            },
            "restart_full_bin": {
                "coverage_percent_mean": np.mean(random_restart_matrix, axis=0).tolist(),
                "coverage_percent_std": np.std(random_restart_matrix, axis=0).tolist(),
                "final_coverage_percent": _mean_std(random_restart_final),
                "seconds_to_100": _threshold_stats(random_restart_seconds_to_100),
                "cycles_to_70": _threshold_stats(random_restart_cycle_thresholds["70"]),
                "cycles_to_90": _threshold_stats(random_restart_cycle_thresholds["90"]),
                "cycles_to_95": _threshold_stats(random_restart_cycle_thresholds["95"]),
                "cycles_to_100": _threshold_stats(random_restart_cycle_thresholds["100"]),
            },
        },
        "ml": {
            "warm_started_overall": {
                "coverage_percent_mean": np.mean(ml_matrix, axis=0).tolist(),
                "coverage_percent_std": np.std(ml_matrix, axis=0).tolist(),
                "final_coverage_percent": _mean_std(ml_warm_final),
                "seconds_to_100": _threshold_stats(ml_warm_seconds_to_100),
                "cycles_to_70": _threshold_stats(ml_warm_cycle_thresholds["70"]),
                "cycles_to_90": _threshold_stats(ml_warm_cycle_thresholds["90"]),
                "cycles_to_95": _threshold_stats(ml_warm_cycle_thresholds["95"]),
                "cycles_to_100": _threshold_stats(ml_warm_cycle_thresholds["100"]),
            },
            "restart_full_bin": {
                "coverage_percent_mean": np.mean(ml_restart_matrix, axis=0).tolist(),
                "coverage_percent_std": np.std(ml_restart_matrix, axis=0).tolist(),
                "final_coverage_percent": _mean_std(ml_restart_final),
                "seconds_to_100": _threshold_stats(ml_restart_seconds_to_100),
                "cycles_to_70": _threshold_stats(ml_restart_cycle_thresholds["70"]),
                "cycles_to_90": _threshold_stats(ml_restart_cycle_thresholds["90"]),
                "cycles_to_95": _threshold_stats(ml_restart_cycle_thresholds["95"]),
                "cycles_to_100": _threshold_stats(ml_restart_cycle_thresholds["100"]),
            },
        },
        "sim_axis": sim_axis,
        "ml_remaining_bin_frequency": dict(sorted(ml_remaining_counter.items())),
    }

    summary_path = os.path.join(bench_dir, "benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\n[Benchmark] Saved aggregate summary to {summary_path}")
    summary_csv_path = os.path.join(bench_dir, "benchmark_summary.csv")
    with open(summary_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial", "seed", "result_dir",
            "seed_coverage_percent",
            "random_final_coverage_percent",
            "ml_final_coverage_percent",
            "random_warm_started_final_coverage_percent",
            "ml_warm_started_final_coverage_percent",
            "random_restart_final_coverage_percent",
            "ml_restart_final_coverage_percent",
            "random_seconds_to_100", "ml_seconds_to_100",
            "random_warm_started_seconds_to_100", "ml_warm_started_seconds_to_100",
            "random_cycles_to_70", "random_cycles_to_90", "random_cycles_to_95", "random_cycles_to_100",
            "ml_cycles_to_70", "ml_cycles_to_90", "ml_cycles_to_95", "ml_cycles_to_100",
            "random_warm_started_cycles_to_70", "random_warm_started_cycles_to_90", "random_warm_started_cycles_to_95", "random_warm_started_cycles_to_100",
            "ml_warm_started_cycles_to_70", "ml_warm_started_cycles_to_90", "ml_warm_started_cycles_to_95", "ml_warm_started_cycles_to_100",
            "random_restart_cycles_to_70", "random_restart_cycles_to_90", "random_restart_cycles_to_95", "random_restart_cycles_to_100",
            "ml_restart_cycles_to_70", "ml_restart_cycles_to_90", "ml_restart_cycles_to_95", "ml_restart_cycles_to_100",
        ])
        for row in trial_summaries:
            writer.writerow([
                row.get("trial"), row.get("seed"), row.get("result_dir"),
                row.get("seed_coverage_percent"),
                row.get("random_final_coverage_percent"),
                row.get("ml_final_coverage_percent"),
                row.get("random_warm_started_final_coverage_percent"),
                row.get("ml_warm_started_final_coverage_percent"),
                row.get("random_restart_final_coverage_percent"),
                row.get("ml_restart_final_coverage_percent"),
                row.get("random_seconds_to_100"), row.get("ml_seconds_to_100"),
                row.get("random_warm_started_seconds_to_100"), row.get("ml_warm_started_seconds_to_100"),
                row.get("random_cycles_to_70"), row.get("random_cycles_to_90"), row.get("random_cycles_to_95"), row.get("random_cycles_to_100"),
                row.get("ml_cycles_to_70"), row.get("ml_cycles_to_90"), row.get("ml_cycles_to_95"), row.get("ml_cycles_to_100"),
                row.get("random_warm_started_cycles_to_70"), row.get("random_warm_started_cycles_to_90"), row.get("random_warm_started_cycles_to_95"), row.get("random_warm_started_cycles_to_100"),
                row.get("ml_warm_started_cycles_to_70"), row.get("ml_warm_started_cycles_to_90"), row.get("ml_warm_started_cycles_to_95"), row.get("ml_warm_started_cycles_to_100"),
                row.get("random_restart_cycles_to_70"), row.get("random_restart_cycles_to_90"), row.get("random_restart_cycles_to_95"), row.get("random_restart_cycles_to_100"),
                row.get("ml_restart_cycles_to_70"), row.get("ml_restart_cycles_to_90"), row.get("ml_restart_cycles_to_95"), row.get("ml_restart_cycles_to_100"),
            ])
    print(f"[Benchmark] Saved summary CSV to {summary_csv_path}")

    _write_bin_distribution_csv(
        os.path.join(bench_dir, "bin_distribution.csv"),
        aggregate_random_counts,
        aggregate_ml_counts,
        args.benchmark_trials * arm_budget,
        args.benchmark_trials * arm_budget,
        n_trials=args.benchmark_trials,
    )
    print(f"[Benchmark] Saved bin distribution CSV to {os.path.join(bench_dir, 'bin_distribution.csv')}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Benchmark] matplotlib not installed - skipping aggregate plot")
        return

    rand_mean = np.mean(random_restart_matrix, axis=0)
    rand_std = np.std(random_restart_matrix, axis=0)
    ml_mean = np.mean(ml_restart_matrix, axis=0)
    ml_std = np.std(ml_restart_matrix, axis=0)
    iter_axis = list(range(args.iters + 1))
    iter_indices = [min(i * args.per_iter, arm_budget) for i in iter_axis]
    rand_mean_iter = rand_mean[iter_indices]
    rand_std_iter = rand_std[iter_indices]
    ml_mean_iter = ml_mean[iter_indices]
    ml_std_iter = ml_std[iter_indices]
    if len(rand_mean_iter) > 0:
        rand_mean_iter[0] = 0.0
        rand_std_iter[0] = 0.0
    if len(ml_mean_iter) > 0:
        ml_mean_iter[0] = 0.0
        ml_std_iter[0] = 0.0

    plt.figure(figsize=(10, 6))
    plt.errorbar(
        iter_axis, rand_mean_iter, yerr=rand_std_iter,
        fmt="o-", color="blue", lw=2, capsize=4, label="Random"
    )
    plt.errorbar(
        iter_axis, ml_mean_iter, yerr=ml_std_iter,
        fmt="o-", color="red", lw=2, capsize=4, label="ML"
    )
    for threshold, color in ((50, "#9ca3af"), (75, "#6b7280"), (90, "#374151")):
        plt.axhline(threshold, color=color, linestyle="--", linewidth=1)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Percentage Coverage (%)")
    plt.xticks(iter_axis)
    plt.xlim(0, args.iters)
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plot_path = os.path.join(bench_dir, "benchmark_closure_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Benchmark] Saved aggregate plot to {plot_path}")

    print("\n" + "="*60)
    print("  BENCHMARK SUMMARY")
    print("="*60)
    print(f"  Trials:                      {args.benchmark_trials}")
    print(f"  Common random warm-start:    {args.init} sims")
    print(f"  Additional sims per arm:     {arm_budget}")
    print(f"  Random restarted coverage:   "
          f"{aggregate['random']['restart_full_bin']['final_coverage_percent']['mean']:.2f}% "
          f"+/- {aggregate['random']['restart_full_bin']['final_coverage_percent']['std']:.2f}")
    print(f"  ML restarted coverage:       "
          f"{aggregate['ml']['restart_full_bin']['final_coverage_percent']['mean']:.2f}% "
          f"+/- {aggregate['ml']['restart_full_bin']['final_coverage_percent']['std']:.2f}")
    print(f"  Random restart cycles to 70/90/95/100%:"
          f" {aggregate['random']['restart_full_bin']['cycles_to_70']['mean']}/"
          f"{aggregate['random']['restart_full_bin']['cycles_to_90']['mean']}/"
          f"{aggregate['random']['restart_full_bin']['cycles_to_95']['mean']}/"
          f"{aggregate['random']['restart_full_bin']['cycles_to_100']['mean']}")
    print(f"  ML restart cycles to 70/90/95/100%:  "
          f"{aggregate['ml']['restart_full_bin']['cycles_to_70']['mean']}/"
          f"{aggregate['ml']['restart_full_bin']['cycles_to_90']['mean']}/"
          f"{aggregate['ml']['restart_full_bin']['cycles_to_95']['mean']}/"
          f"{aggregate['ml']['restart_full_bin']['cycles_to_100']['mean']}")
    print(f"  Benchmark directory:         {bench_dir}")
    print("="*60)


def run_fault_benchmark(args) -> None:
    if args.agent not in ("dqn", "supervised"):
        raise ValueError("Fault benchmark currently supports only agent=dqn or agent=supervised")
    bench_dir = _next_fault_benchmark_result_dir(RESULTS_DIR)
    os.makedirs(bench_dir, exist_ok=True)
    active_bug_define = os.environ.get("PICORV32_BUG_DEFINE", "")
    model_source_dir = _resolve_resume_result_dir(RESULTS_DIR, args.fault_model_from)
    if not os.path.isdir(model_source_dir):
        raise FileNotFoundError(f"Fault benchmark model source not found: {model_source_dir}")

    print(f"[FaultBenchmark] Writing aggregate benchmark to: {bench_dir}")
    print(f"[FaultBenchmark] Active injected bug: {active_bug_define}")
    print(f"[FaultBenchmark] Coverage-trained model source: {model_source_dir}")
    if args.ml_only:
        print("[FaultBenchmark] Mode: ML-only quick fault tuning (random arm skipped)")
    if args.fault_stop_on_bug:
        print("[FaultBenchmark] Mode: stop scheduling new batches for an arm after its first bug is detected")

    knob_overrides = {
        "trap_rate": 0,
        "trap_kind": 0,
    }
    trial_summaries = []
    random_detection_curves = []
    ml_detection_curves = []
    random_coverage_curves = []
    ml_coverage_curves = []
    random_first_cycles = []
    random_first_sims = []
    random_first_runtime_sec = []
    ml_first_cycles = []
    ml_first_sims = []
    ml_first_runtime_sec = []
    random_reason_counts = Counter()
    ml_reason_counts = Counter()
    aggregate_random_counts = Counter()
    aggregate_ml_counts = Counter()
    random_remaining_counter = Counter()
    ml_remaining_counter = Counter()
    arm_budget = args.iters * args.per_iter

    for trial_idx in range(1, args.fault_benchmark_trials + 1):
        trial_seed = (int(args.seed) + trial_idx - 1) & 0xFFFFFFFF
        random.seed(trial_seed)
        np.random.seed(trial_seed)

        trial_dir = os.path.join(bench_dir, f"trial_{trial_idx:02d}")
        os.makedirs(trial_dir, exist_ok=True)
        print(f"\n[FaultBenchmark] Trial {trial_idx}/{args.fault_benchmark_trials}  seed={trial_seed}")

        runner = SimulationRunner(
            questa_cmd=args.questa,
            program_seed_base=(trial_seed * 1000003) & 0xFFFFFFFF,
            include_fault_fields=True,
        )
        if args.agent == "dqn":
            agent = DQNAgent()
            agent.load(os.path.join(model_source_dir, "dqn_model.pt"))
        else:
            agent = SupervisedAgent()
            agent.load(os.path.join(model_source_dir, "supervised_model.pt"))

        random_arm: List[Dict] = []
        ml_arm: List[Dict] = []
        prev_ml_results: List[Dict] = []
        next_sim_id = 0

        if not args.ml_only:
            for it in range(1, args.iters + 1):
                if args.fault_stop_on_bug and any(
                    bool(r.get("coverage", {}).get("bug_detected", False)) for r in random_arm
                ):
                    break
                ids = range(next_sim_id, next_sim_id + args.per_iter)
                next_sim_id += args.per_iter
                knobs = []
                for _ in ids:
                    knob = {name: random.choice(vals) for name, vals in KNOB_RANGES.items()}
                    for key, value in knob_overrides.items():
                        knob[key] = value
                    knobs.append(knob)
                random_arm.extend(
                    runner.run_batch(
                        knobs,
                        ids.start,
                        label=f"random_fault_iter{it}",
                        stop_on_bug=False,
                    )
                )

        for it in range(1, args.iters + 1):
            if args.fault_stop_on_bug and any(
                bool(r.get("coverage", {}).get("bug_detected", False)) for r in ml_arm
            ):
                break
            if args.agent == "dqn":
                ml_knobs = agent.suggest_knobs_frozen(prev_ml_results, n_suggest=args.per_iter)
            else:
                ml_knobs = agent.suggest_knobs_frozen(n_suggest=args.per_iter)
            ml_knobs = [dict(k) for k in ml_knobs]
            for knob in ml_knobs:
                for key, value in knob_overrides.items():
                    knob[key] = value
            ids = range(next_sim_id, next_sim_id + args.per_iter)
            next_sim_id += args.per_iter
            batch = runner.run_batch(
                ml_knobs,
                ids.start,
                label=f"ml_fault_iter{it}",
                stop_on_bug=False,
            )
            ml_arm.extend(batch)
            prev_ml_results = batch

        random_bug = _first_bug_detection(random_arm)
        ml_bug = _first_bug_detection(ml_arm)

        if random_bug["detected"]:
            random_first_cycles.append(float(random_bug["cycles"]))
            random_first_sims.append(float(random_bug["sim"]))
            random_first_runtime_sec.append(float(random_bug["runtime_sec"]))
            if random_bug["reason"]:
                random_reason_counts[random_bug["reason"]] += 1

        if ml_bug["detected"]:
            ml_first_cycles.append(float(ml_bug["cycles"]))
            ml_first_sims.append(float(ml_bug["sim"]))
            ml_first_runtime_sec.append(float(ml_bug["runtime_sec"]))
            if ml_bug["reason"]:
                ml_reason_counts[ml_bug["reason"]] += 1

        target_len = arm_budget + 1
        random_detection_curves.append(_pad_curve(
            _bug_detection_curve(random_arm), target_len, ["detection_percent", "cumulative_cycles"]
        ))
        ml_detection_curves.append(_pad_curve(
            _bug_detection_curve(ml_arm), target_len, ["detection_percent", "cumulative_cycles"]
        ))
        random_curve = _pad_curve(
            _coverage_curve_restart(random_arm), target_len, ["coverage_ratio", "coverage_percent", "cumulative_cycles"]
        )
        ml_curve = _pad_curve(
            _coverage_curve_restart(ml_arm), target_len, ["coverage_ratio", "coverage_percent", "cumulative_cycles"]
        )
        random_coverage_curves.append(random_curve)
        ml_coverage_curves.append(ml_curve)

        trial_random_counts = _bin_hit_distribution(random_arm)
        trial_ml_counts = _bin_hit_distribution(ml_arm)
        aggregate_random_counts.update(trial_random_counts)
        aggregate_ml_counts.update(trial_ml_counts)
        _write_bin_distribution_csv(
            os.path.join(trial_dir, "bin_distribution.csv"),
            trial_random_counts,
            trial_ml_counts,
            len(random_arm),
            len(ml_arm),
            n_trials=1,
        )

        random_seen = set()
        ml_seen = set()
        for r in random_arm:
            for b in r.get("coverage", {}).get("functional_bins", []):
                if b in FUNCTIONAL_BIN_CATALOG:
                    random_seen.add(b)
        for r in ml_arm:
            for b in r.get("coverage", {}).get("functional_bins", []):
                if b in FUNCTIONAL_BIN_CATALOG:
                    ml_seen.add(b)
        for b in FUNCTIONAL_BIN_CATALOG:
            if b not in random_seen:
                random_remaining_counter[b] += 1
            if b not in ml_seen:
                ml_remaining_counter[b] += 1

        with open(os.path.join(trial_dir, "experiment_results.json"), "w") as f:
            json.dump({
                "config": {
                    "agent_type": args.agent,
                    "fault_model_from": model_source_dir,
                    "n_iters": args.iters,
                    "n_per_iter": args.per_iter,
                    "fault_bug_define": active_bug_define,
                    "ml_only": bool(args.ml_only),
                    "fault_stop_on_bug": bool(args.fault_stop_on_bug),
                    "knob_overrides": knob_overrides,
                    "program_seed_base": (trial_seed * 1000003) & 0xFFFFFFFF,
                    "fault_mode": "transfer_from_clean_coverage_model",
                },
                "functional_bin_catalog": FUNCTIONAL_BIN_CATALOG,
                "all_results": random_arm + ml_arm,
            }, f, indent=2)

        with open(os.path.join(trial_dir, "coverage_closure.json"), "w") as f:
            json.dump({
                "total_bins": len(FUNCTIONAL_BIN_CATALOG),
                "catalog": FUNCTIONAL_BIN_CATALOG,
                "random": {
                    "curve": random_curve,
                    "sims_to_70": _sims_to_threshold_from_curve(random_curve, 0.70),
                    "sims_to_90": _sims_to_threshold_from_curve(random_curve, 0.90),
                    "sims_to_95": _sims_to_threshold_from_curve(random_curve, 0.95),
                    "sims_to_100": _sims_to_threshold_from_curve(random_curve, 1.00),
                    "cycles_to_70": _cycles_to_threshold_from_curve(random_curve, 0.70),
                    "cycles_to_90": _cycles_to_threshold_from_curve(random_curve, 0.90),
                    "cycles_to_95": _cycles_to_threshold_from_curve(random_curve, 0.95),
                    "cycles_to_100": _cycles_to_threshold_from_curve(random_curve, 1.00),
                },
                "ml": {
                    "curve": ml_curve,
                    "sims_to_70": _sims_to_threshold_from_curve(ml_curve, 0.70),
                    "sims_to_90": _sims_to_threshold_from_curve(ml_curve, 0.90),
                    "sims_to_95": _sims_to_threshold_from_curve(ml_curve, 0.95),
                    "sims_to_100": _sims_to_threshold_from_curve(ml_curve, 1.00),
                    "cycles_to_70": _cycles_to_threshold_from_curve(ml_curve, 0.70),
                    "cycles_to_90": _cycles_to_threshold_from_curve(ml_curve, 0.90),
                    "cycles_to_95": _cycles_to_threshold_from_curve(ml_curve, 0.95),
                    "cycles_to_100": _cycles_to_threshold_from_curve(ml_curve, 1.00),
                },
            }, f, indent=2)

        trial_summary = {
            "trial": trial_idx,
            "seed": trial_seed,
            "result_dir": trial_dir,
            "random_bug_detected": bool(random_bug["detected"]),
            "random_bug_sim": random_bug["sim"],
            "random_bug_cycles": random_bug["cycles"],
            "random_bug_runtime_sec": random_bug["runtime_sec"],
            "random_bug_reason": random_bug["reason"],
            "ml_bug_detected": bool(ml_bug["detected"]),
            "ml_bug_sim": ml_bug["sim"],
            "ml_bug_cycles": ml_bug["cycles"],
            "ml_bug_runtime_sec": ml_bug["runtime_sec"],
            "ml_bug_reason": ml_bug["reason"],
            "random_bug_hits": sum(
                1 for r in random_arm if bool(r.get("coverage", {}).get("bug_detected", False))
            ),
            "ml_bug_hits": sum(
                1 for r in ml_arm if bool(r.get("coverage", {}).get("bug_detected", False))
            ),
        }
        trial_summaries.append(trial_summary)

    sim_axis = list(range(arm_budget + 1))
    random_detection_matrix = np.asarray(
        [curve["detection_percent"] for curve in random_detection_curves], dtype=float
    )
    ml_detection_matrix = np.asarray(
        [curve["detection_percent"] for curve in ml_detection_curves], dtype=float
    )
    random_coverage_matrix = np.asarray(
        [curve["coverage_percent"] for curve in random_coverage_curves], dtype=float
    )
    ml_coverage_matrix = np.asarray(
        [curve["coverage_percent"] for curve in ml_coverage_curves], dtype=float
    )

    aggregate = {
        "config": {
            "agent_type": args.agent,
            "n_iters": args.iters,
            "n_per_iter": args.per_iter,
            "fault_benchmark_trials": args.fault_benchmark_trials,
            "fault_bug_define": active_bug_define,
            "fault_model_from": model_source_dir,
            "ml_only": bool(args.ml_only),
            "fault_stop_on_bug": bool(args.fault_stop_on_bug),
            "knob_overrides": knob_overrides,
            "fault_mode": "transfer_from_clean_coverage_model",
        },
        "total_bins": len(FUNCTIONAL_BIN_CATALOG),
        "catalog": FUNCTIONAL_BIN_CATALOG,
        "trial_summaries": trial_summaries,
        "random": {
            "first_detection_cycles": _threshold_stats(
                [ts["random_bug_cycles"] for ts in trial_summaries]
            ),
            "first_detection_sims": _threshold_stats(
                [ts["random_bug_sim"] for ts in trial_summaries]
            ),
            "first_detection_runtime_sec": _threshold_stats(
                [ts["random_bug_runtime_sec"] for ts in trial_summaries]
            ),
            "detection_percent_mean": np.mean(random_detection_matrix, axis=0).tolist(),
            "detection_percent_std": np.std(random_detection_matrix, axis=0).tolist(),
            "reason_frequency": dict(sorted(random_reason_counts.items())),
        },
        "ml": {
            "first_detection_cycles": _threshold_stats(
                [ts["ml_bug_cycles"] for ts in trial_summaries]
            ),
            "first_detection_sims": _threshold_stats(
                [ts["ml_bug_sim"] for ts in trial_summaries]
            ),
            "first_detection_runtime_sec": _threshold_stats(
                [ts["ml_bug_runtime_sec"] for ts in trial_summaries]
            ),
            "detection_percent_mean": np.mean(ml_detection_matrix, axis=0).tolist(),
            "detection_percent_std": np.std(ml_detection_matrix, axis=0).tolist(),
            "reason_frequency": dict(sorted(ml_reason_counts.items())),
        },
        "sim_axis": sim_axis,
        "random_remaining_bin_frequency": dict(sorted(random_remaining_counter.items())),
        "ml_remaining_bin_frequency": dict(sorted(ml_remaining_counter.items())),
    }

    summary_path = os.path.join(bench_dir, "fault_benchmark_summary.json")
    with open(summary_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\n[FaultBenchmark] Saved aggregate summary to {summary_path}")

    _write_bin_distribution_csv(
        os.path.join(bench_dir, "bin_distribution.csv"),
        aggregate_random_counts,
        aggregate_ml_counts,
        args.fault_benchmark_trials * arm_budget,
        args.fault_benchmark_trials * arm_budget,
        n_trials=args.fault_benchmark_trials,
    )
    print(f"[FaultBenchmark] Saved bin distribution CSV to {os.path.join(bench_dir, 'bin_distribution.csv')}")

    csv_path = os.path.join(bench_dir, "fault_detection_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "trial",
            "seed",
            "random_bug_detected",
            "random_bug_sim",
            "random_bug_cycles",
            "random_bug_runtime_sec",
            "random_bug_reason",
            "ml_bug_detected",
            "ml_bug_sim",
            "ml_bug_cycles",
            "ml_bug_runtime_sec",
            "ml_bug_reason",
            "random_bug_hits",
            "ml_bug_hits",
        ])
        for row in trial_summaries:
            writer.writerow([
                row["trial"],
                row["seed"],
                int(row["random_bug_detected"]),
                row["random_bug_sim"],
                row["random_bug_cycles"],
                row["random_bug_runtime_sec"],
                row["random_bug_reason"],
                int(row["ml_bug_detected"]),
                row["ml_bug_sim"],
                row["ml_bug_cycles"],
                row["ml_bug_runtime_sec"],
                row["ml_bug_reason"],
                row["random_bug_hits"],
                row["ml_bug_hits"],
            ])
    print(f"[FaultBenchmark] Saved trial CSV to {csv_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[FaultBenchmark] matplotlib not installed - skipping aggregate plot")
    else:
        rand_mean = np.mean(random_detection_matrix, axis=0)
        rand_std = np.std(random_detection_matrix, axis=0)
        ml_mean = np.mean(ml_detection_matrix, axis=0)
        ml_std = np.std(ml_detection_matrix, axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot(sim_axis, rand_mean, "b-", lw=2, label="Random mean")
        plt.fill_between(sim_axis, rand_mean - rand_std, rand_mean + rand_std,
                         color="blue", alpha=0.15, label="Random ±1 std")
        plt.plot(sim_axis, ml_mean, "r-", lw=2, label="ML mean")
        plt.fill_between(sim_axis, ml_mean - ml_std, ml_mean + ml_std,
                         color="red", alpha=0.15, label="ML ±1 std")
        plt.xlabel("Buggy-DUT Simulations")
        plt.ylabel("Trials With Bug Detected (%)")
        plt.title("Fault Benchmark: Detection Rate (Coverage-Trained ML vs Random)")
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plot_path = os.path.join(bench_dir, "fault_detection_plot.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[FaultBenchmark] Saved aggregate plot to {plot_path}")

    print("\n" + "="*60)
    print("  FAULT BENCHMARK SUMMARY")
    print("="*60)
    print(f"  Injected bug:                 {active_bug_define}")
    print(f"  Coverage-trained model:       {model_source_dir}")
    print(f"  Trials:                       {args.fault_benchmark_trials}")
    print(f"  Buggy-DUT sims per arm:       {arm_budget}")
    rand_cycles = aggregate['random']['first_detection_cycles']['mean']
    ml_cycles = aggregate['ml']['first_detection_cycles']['mean']
    rand_cycles = int(math.ceil(rand_cycles)) if rand_cycles is not None else None
    ml_cycles = int(math.ceil(ml_cycles)) if ml_cycles is not None else None
    random_bug_run_count = sum(ts["random_bug_hits"] for ts in trial_summaries)
    ml_bug_run_count = sum(ts["ml_bug_hits"] for ts in trial_summaries)
    print(f"  Random cycles to first bug:   {rand_cycles}")
    print(f"  ML cycles to first bug:       {ml_cycles}")
    print(f"  Random sims to first bug:     {aggregate['random']['first_detection_sims']['mean']}")
    print(f"  ML sims to first bug:         {aggregate['ml']['first_detection_sims']['mean']}")
    print(f"  Random runs with bug found:   {random_bug_run_count}")
    print(f"  ML runs with bug found:       {ml_bug_run_count}")
    print(f"  Benchmark directory:          {bench_dir}")
    print("="*60)


def run_standard_experiment(args) -> None:
    result_dir = RESULTS_DIR
    model_source_dir = RESULTS_DIR
    if args.new:
        result_dir = _next_fresh_result_dir(RESULTS_DIR)
        os.makedirs(result_dir, exist_ok=True)
        print(f"[New] Using isolated results directory: {result_dir}")
    elif args.eval_from:
        model_source_dir = _resolve_resume_result_dir(RESULTS_DIR, args.eval_from)
        if not os.path.isdir(model_source_dir):
            raise FileNotFoundError(f"Evaluation source directory not found: {model_source_dir}")
        result_dir = _next_eval_result_dir(RESULTS_DIR)
        os.makedirs(result_dir, exist_ok=True)
        print(f"[Eval] Loaded model from: {model_source_dir}")
        print(f"[Eval] Writing benchmark results to: {result_dir}")
    elif args.resume_from:
        result_dir = _resolve_resume_result_dir(RESULTS_DIR, args.resume_from)
        if not os.path.isdir(result_dir):
            raise FileNotFoundError(f"Resume directory not found: {result_dir}")
        print(f"[Resume] Using results directory: {result_dir}")
        model_source_dir = result_dir

    effective_seed = int(args.seed) & 0xFFFFFFFF
    if (args.resume_from or args.eval_from) and args.seed == 42:
        effective_seed = int(time.time() * 1000) & 0xFFFFFFFF
    random.seed(effective_seed)
    np.random.seed(effective_seed)
    print(f"[Seed] Using RNG seed: {effective_seed}")

    resume_enabled = bool(args.resume_from)
    eval_enabled = bool(args.eval_from)

    exp = MLDVExperiment(
        agent_type=args.agent,
        n_init=args.init,
        n_iters=args.iters,
        n_per_iter=args.per_iter,
        questa_cmd=args.questa,
        resume=resume_enabled,
        result_dir=result_dir,
        eval_mode=eval_enabled,
        ml_only=args.ml_only,
        model_source_dir=model_source_dir,
        reward_mode=args.reward_mode,
        program_seed_base=(effective_seed * 1000003) & 0xFFFFFFFF,
    )
    exp.run()


def main():
    parser = argparse.ArgumentParser(description="PicoRV32 ML-DV Experiment Runner")
    parser.add_argument("--agent", default="supervised",
                        choices=["supervised", "dqn", "rf", "dt"])
    parser.add_argument("--iters", type=int, default=6,
                        help="ML iterations (default 6)")
    parser.add_argument("--per-iter", type=int, default=10,
                        help="Simulations per ML iteration (default 10)")
    parser.add_argument("--init", type=int, default=20,
                        help="Initial random simulations (default 20)")
    parser.add_argument("--questa", default=QUESTA_CMD,
                        help=f"QuestaSim executable (default: {QUESTA_CMD})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reward-mode", default="coverage",
                        choices=["coverage", "composite"],
                        help="Select reward shaping: coverage uses bin bonuses, composite uses only the composite reward")
    parser.add_argument("--check", action="store_true",
                        help="Check environment and exit")
    parser.add_argument("--resume-from", default="",
                        help="Resume from a specific results directory name/path, or use 'latest'")
    parser.add_argument("--eval-from", default="",
                        help="Evaluate a frozen trained model from a specific results directory name/path, or use 'latest'")
    parser.add_argument("--new", action="store_true",
                        help="Start a clean experiment and save outputs to a new numbered results directory")
    parser.add_argument("--benchmark-trials", type=int, default=0,
                        help="Run repeated fresh matched-budget benchmark trials and aggregate the results")
    parser.add_argument("--fault-benchmark-trials", type=int, default=0,
                        help="Run repeated matched-budget bug-finding benchmark trials using the active PICORV32_BUG_DEFINE")
    parser.add_argument("--fault-model-from", default="",
                        help="Results directory name/path containing the clean coverage-trained model to use for fault benchmarking")
    parser.add_argument("--ml-only", action="store_true",
                        help="Skip random baseline/arm phases and run ML-only where applicable")
    parser.add_argument("--fault-stop-on-bug", action="store_true",
                        help="Stop a fault-benchmark arm immediately after the first detected bug")
    parser.add_argument("--reachability-check", action="store_true",
                        help="Run targeted sims to test whether suspected unreachable bins can actually be hit")
    parser.add_argument("--reachability-per-family", type=int, default=12,
                        help="Targeted simulations per reachability knob family")
    args = parser.parse_args()

    if args.resume_from and args.new:
        parser.error("--resume-from and --new cannot be used together")
    if args.eval_from and args.new:
        parser.error("--eval-from and --new cannot be used together")
    if args.eval_from and args.resume_from:
        parser.error("--eval-from cannot be combined with resume options")
    if args.benchmark_trials < 0:
        parser.error("--benchmark-trials must be >= 0")
    if args.fault_benchmark_trials < 0:
        parser.error("--fault-benchmark-trials must be >= 0")
    if args.reachability_per_family <= 0:
        parser.error("--reachability-per-family must be > 0")
    if args.benchmark_trials and (args.resume_from or args.eval_from or args.new):
        parser.error("--benchmark-trials cannot be combined with resume/eval/new options")
    if args.fault_benchmark_trials and (args.resume_from or args.eval_from or args.new or args.benchmark_trials):
        parser.error("--fault-benchmark-trials cannot be combined with resume/eval/new/benchmark options")
    if args.reachability_check and (args.resume_from or args.eval_from or args.new or args.benchmark_trials):
        parser.error("--reachability-check cannot be combined with resume/eval/new/benchmark options")
    if args.reachability_check and args.fault_benchmark_trials:
        parser.error("--reachability-check cannot be combined with --fault-benchmark-trials")
    if args.fault_benchmark_trials and not os.environ.get("PICORV32_BUG_DEFINE", ""):
        parser.error("--fault-benchmark-trials requires PICORV32_BUG_DEFINE to be set before compile/run")
    if args.fault_benchmark_trials and not args.fault_model_from:
        parser.error("--fault-benchmark-trials requires --fault-model-from to point to a clean coverage-trained model")

    if args.check:
        check_environment(args.questa)
        return

    check_environment(args.questa)

    if args.benchmark_trials:
        run_benchmark(args)
        return
    if args.fault_benchmark_trials:
        run_fault_benchmark(args)
        return
    if args.reachability_check:
        run_reachability_check(args)
        return

    try:
        run_standard_experiment(args)
    except FileNotFoundError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
