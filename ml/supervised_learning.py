# =============================================================================
# supervised_learning.py
# Supervised neural-network agent for PicoRV32 ML-DV knob optimisation.
#
# Architecture
# ─────────────
# A small MLP is trained to predict total reward from a knob config:
#   f(knobs_normalised) → total_reward_predicted
#
# At each iteration the agent:
#   1. Trains / fine-tunes the MLP on all accumulated (knobs, total_reward) data.
#   2. Uses random-restart gradient ascent through the surrogate model to find
#      high-predicted-total-reward knob configs.
#   3. Returns the top-n_suggest configs (mixed with exploration noise).
#
# This is the standard "surrogate-model" or "Bayesian-optimisation-lite"
# approach — academically clean and straightforward to explain in a dissertation.
# =============================================================================

import os, random, copy
from typing import List, Dict

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
    print("[supervised_learning] WARNING: torch not installed — SupervisedAgent will use "
          "random suggestions.  Install with: pip install torch")

# Re-use the shared coverage helpers
try:
    from coverage_features import (
        KNOB_RANGES, KNOB_NAMES, _normalise_knobs, _random_knobs, compute_reward
    )
except ImportError:
    # Fallback definitions (identical) in case import order differs
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


KNOB_DIM   = len(KNOB_NAMES)
HIDDEN     = 64
LR         = 5e-4
EPOCHS     = 60      # training epochs per iteration
BATCH_SIZE = 16
N_SEARCH   = 1024    # random candidates scored per suggestion call
EXPLORE_FRAC = 0.25  # fraction of suggestions that are random exploration

# ---------------------------------------------------------------------------
# Surrogate model (knobs → total reward)
# ---------------------------------------------------------------------------

class _Surrogate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(KNOB_DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, 1),
            nn.Sigmoid(),    # clamp output to (0, 1) — reward is in [0,1]
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ---------------------------------------------------------------------------
# Supervised Agent
# ---------------------------------------------------------------------------

class SupervisedAgent:
    """
    Surrogate-model agent.

    Maintains a dataset of (knob_config, total_reward) observations gathered
    across all simulation runs.  At each iteration:

      1. Fine-tunes the surrogate MLP on the full dataset.
      2. Scores N_SEARCH random knob configs using the trained surrogate.
      3. Returns the top configs by predicted total reward, mixed with a small
         fraction of random exploration to avoid surrogate over-exploitation.
    """

    def __init__(self):
        self.dataset: List[tuple] = []   # (knobs_norm: np.ndarray, reward: float)
        self.iteration = 0
        self.best_seen: dict = {}
        self.best_reward: float = 0.0
        self.best_stall_seen: float = 0.0

        if _TORCH_OK:
            self.model     = _Surrogate()
            self.optimiser = optim.Adam(self.model.parameters(), lr=LR)
            self.loss_fn   = nn.MSELoss()
        else:
            self.model = None

    # ── public interface ─────────────────────────────────────────────────────

    def suggest_knobs_for_iteration(self,
                                    prev_results: list,
                                    n_suggest: int = 10) -> List[dict]:
        """
        Ingest previous results, retrain surrogate, return n_suggest configs.
        """
        # 1. Add new observations to dataset
        if prev_results:
            self._ingest(prev_results)

        # 2. Train surrogate on accumulated data
        if self.model is not None and len(self.dataset) >= BATCH_SIZE:
            self._train()

        # 3. Generate suggestions
        n_explore = max(1, int(n_suggest * EXPLORE_FRAC))
        n_exploit = n_suggest - n_explore

        suggestions = []

        if self.model is not None and len(self.dataset) >= BATCH_SIZE:
            # Score a large random pool with the surrogate
            candidates  = [_random_knobs() for _ in range(N_SEARCH)]
            knobs_norms = np.stack([_normalise_knobs(k) for k in candidates])

            with torch.no_grad():
                t      = torch.FloatTensor(knobs_norms)
                scores = self.model(t).numpy()

            top_idx = np.argsort(scores)[::-1][:n_exploit]
            for i in top_idx:
                suggestions.append(candidates[i])
        else:
            # Not enough data yet — pure random
            n_explore = n_suggest

        # 4. Exploration: random configs
        for _ in range(n_explore):
            suggestions.append(_random_knobs())

        # 5. Always include best-seen config (exploitation anchor)
        if self.best_seen and len(suggestions) > 1:
            suggestions[-1] = copy.deepcopy(self.best_seen)

        self.iteration += 1
        print(f"  [SUP] iter={self.iteration}  "
              f"dataset={len(self.dataset)}  "
              f"exploit={len(suggestions)-n_explore}  "
              f"explore={n_explore}  "
              f"best_seen_reward={self.best_reward:.4f}  "
              f"best_seen_stall={self.best_stall_seen:.4f}")
        return suggestions[:n_suggest]

    def suggest_knobs_frozen(self, n_suggest: int = 10) -> List[dict]:
        """
        Return suggestions from the current surrogate without ingesting new
        observations or updating the model.
        """
        suggestions = []

        if self.model is not None and len(self.dataset) >= BATCH_SIZE:
            candidates = [_random_knobs() for _ in range(N_SEARCH)]
            knobs_norms = np.stack([_normalise_knobs(k) for k in candidates])

            with torch.no_grad():
                scores = self.model(torch.FloatTensor(knobs_norms)).numpy()

            top_idx = np.argsort(scores)[::-1][:n_suggest]
            for i in top_idx:
                suggestions.append(candidates[i])
        else:
            suggestions = [_random_knobs() for _ in range(n_suggest)]

        if self.best_seen and suggestions:
            suggestions[-1] = copy.deepcopy(self.best_seen)

        return suggestions[:n_suggest]

    def save(self, path: str):
        if not _TORCH_OK or self.model is None:
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            "model":      self.model.state_dict(),
            "optimiser":  self.optimiser.state_dict(),
            "iteration":  self.iteration,
            "best_reward": self.best_reward,
            "best_stall_seen": self.best_stall_seen,
            "best_seen":  self.best_seen,
            "dataset": [
                (np.asarray(knobs_norm, dtype=np.float32).tolist(), float(reward))
                for knobs_norm, reward in self.dataset
            ],
        }, path)
        print(f"  [SUP] Model saved: {path}")

    def load(self, path: str):
        if not _TORCH_OK or not os.path.exists(path):
            return
        ckpt = torch.load(path, map_location="cpu")
        self.model.load_state_dict(ckpt["model"])
        if "optimiser" in ckpt:
            self.optimiser.load_state_dict(ckpt["optimiser"])
        self.iteration   = ckpt.get("iteration",   0)
        self.best_reward = ckpt.get("best_reward", 0.0)
        self.best_stall_seen = ckpt.get("best_stall_seen", 0.0)
        self.best_seen = ckpt.get("best_seen", {})
        self.dataset = [
            (np.asarray(item[0], dtype=np.float32), float(item[1]))
            for item in ckpt.get("dataset", [])
        ]
        print(f"  [SUP] Model loaded: {path}")

    # ── private ──────────────────────────────────────────────────────────────

    def _ingest(self, results: list):
        for r in results:
            reward     = float(
                r["coverage"].get(
                    "total_reward",
                    r["coverage"].get("composite_reward", compute_reward(r["coverage"]))
                )
            )
            raw_stall  = float(r["coverage"].get("stall_ratio", 0.0))
            knobs_norm = _normalise_knobs(r["knobs"])
            self.dataset.append((knobs_norm, reward))
            if reward > self.best_reward:
                self.best_reward = reward
                self.best_seen   = copy.deepcopy(r["knobs"])
            if raw_stall > self.best_stall_seen:
                self.best_stall_seen = raw_stall

    def _train(self):
        if not _TORCH_OK or self.model is None:
            return

        X = torch.FloatTensor(np.array([d[0] for d in self.dataset]))
        y = torch.FloatTensor(np.array([d[1] for d in self.dataset]))

        n = len(X)
        for epoch in range(EPOCHS):
            # Mini-batch SGD
            idx   = torch.randperm(n)
            total_loss = 0.0
            for start in range(0, n, BATCH_SIZE):
                batch_idx = idx[start:start + BATCH_SIZE]
                xb, yb    = X[batch_idx], y[batch_idx]
                pred  = self.model(xb)
                loss  = self.loss_fn(pred, yb)
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()
                total_loss += loss.item()

        # Report final epoch loss quietly
        avg_loss = total_loss / max(n // BATCH_SIZE, 1)
        if self.iteration % 2 == 0:
            print(f"  [SUP] Training done — loss={avg_loss:.4f}  n={n}")
