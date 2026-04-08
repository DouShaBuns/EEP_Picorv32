#!/usr/bin/env python3
"""
run_coverage.py
Dedicated coverage-closure entry point.
"""

import argparse
import os
from copy import deepcopy

from run_experiment import (
    COVERAGE_SIM_LIB,
    QUESTA_CMD,
    check_environment,
    ensure_sim_library,
    run_standard_experiment,
)


def main():
    parser = argparse.ArgumentParser(description="PicoRV32 coverage runner")
    parser.add_argument("--agent", default="dqn", choices=["dqn", "rf", "dt"],
                        help="Coverage agent: dqn, rf (random forest), or dt (decision tree)")
    parser.add_argument("--iters", type=int, default=6,
                        help="ML iterations (default 6)")
    parser.add_argument("--per-iter", type=int, default=10,
                        help="Simulations per ML iteration (default 10)")
    parser.add_argument("--init", type=int, default=20,
                        help="Initial random simulations (default 20)")
    parser.add_argument("--questa", default=QUESTA_CMD,
                        help=f"QuestaSim executable (default: {QUESTA_CMD})")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-mode", default="00",
                        choices=["00", "01", "10"],
                        help="Training mode: 00=new, 01=resume-from, 10=eval-from")
    parser.add_argument("--model-from", default="",
                        help="Results directory name/path used by --train-mode 01 (resume) or 10 (eval)")
    parser.add_argument("--repeat-trials", type=int, default=0,
                        help="Repeat the normal fresh coverage run multiple times with incremented seeds")
    parser.add_argument("--ml-only", action="store_true",
                        help="Skip random baseline phases and run ML-only where applicable")
    parser.add_argument("--dynamic-coverage", action="store_true",
                        help="Enable adaptive/dynamic coverage reward shaping")
    parser.add_argument("--residual-closure", action="store_true",
                        help="Enable paper-inspired late residual closure mode for remaining bins")
    parser.add_argument("--dqn-reward-mode", default="A", choices=["A", "B", "AUTO", "a", "b", "auto"],
                        help="DQN learning reward mode: A=default, B=force residual reward, AUTO=switch to residual reward when only late-stage target bins remain")
    args = parser.parse_args()

    args.reward_mode = "coverage"
    args.fault_benchmark_trials = 0
    args.fault_model_from = ""
    args.fault_stop_on_bug = False
    args.new = args.train_mode == "00"
    args.resume_from = args.model_from if args.train_mode == "01" else ""
    args.eval_from = args.model_from if args.train_mode == "10" else ""

    if args.train_mode in ("01", "10") and not args.model_from:
        parser.error("--train-mode 01/10 requires --model-from")
    if args.train_mode == "00" and args.model_from:
        parser.error("--model-from is only used with --train-mode 01 or 10")
    if args.repeat_trials < 0:
        parser.error("--repeat-trials must be >= 0")
    if args.repeat_trials and args.train_mode != "00":
        parser.error("--repeat-trials cannot be combined with --train-mode 01/10")

    os.environ["PICORV32_WORK_LIB"] = COVERAGE_SIM_LIB
    os.environ.pop("PICORV32_BUG_DEFINE", None)
    os.environ["PICORV32_DYNAMIC_COVERAGE_REWARD"] = "1" if args.dynamic_coverage else "0"
    os.environ["PICORV32_RESIDUAL_CLOSURE"] = "1" if args.residual_closure else "0"
    os.environ["PICORV32_DQN_REWARD_MODE"] = str(args.dqn_reward_mode).upper()
    check_environment(args.questa)
    ensure_sim_library(args.questa, COVERAGE_SIM_LIB, bug_define="")

    if args.repeat_trials:
        for trial_idx in range(1, args.repeat_trials + 1):
            trial_args = deepcopy(args)
            trial_args.seed = (int(args.seed) + (trial_idx - 1) * 2026) & 0xFFFFFFFF
            print(f"\n[Repeat] Trial {trial_idx}/{args.repeat_trials}  seed={trial_args.seed}")
            run_standard_experiment(trial_args)
        return

    run_standard_experiment(args)


if __name__ == "__main__":
    main()
