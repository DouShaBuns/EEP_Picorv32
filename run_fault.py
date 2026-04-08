#!/usr/bin/env python3
"""
run_fault.py
Dedicated fault-debug entry point.
Defaults to the supervised agent with composite reward shaping.
"""

import argparse
import os

from run_experiment import (
    FAULT_SIM_LIB,
    QUESTA_CMD,
    check_environment,
    ensure_sim_library,
    run_fault_benchmark,
    run_standard_experiment,
)


def main():
    parser = argparse.ArgumentParser(description="PicoRV32 fault runner (supervised-focused)")
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
    parser.add_argument("--trials", type=int, default=0,
                        help="Run repeated matched-budget bug-finding benchmark trials using the active PICORV32_BUG_DEFINE")
    parser.add_argument("--load-model", default="",
                        help="Results directory name/path containing the clean coverage-trained model to use for fault benchmarking")
    parser.add_argument("--ml-only", action="store_true",
                        help="Skip random baseline/arm phases and run ML-only where applicable")
    parser.add_argument("--stop-on-bug", action="store_true",
                        help="Stop scheduling new fault batches for an arm after its first detected bug")
    args = parser.parse_args()

    args.agent = "supervised"
    args.reward_mode = "composite"
    args.benchmark_trials = 0
    args.reachability_check = False
    args.reachability_per_family = 12
    args.fault_stop_on_bug = bool(args.stop_on_bug)
    args.fault_benchmark_trials = int(args.trials)
    args.fault_model_from = args.load_model
    args.new = args.train_mode == "00"
    args.resume_from = args.model_from if args.train_mode == "01" else ""
    args.eval_from = args.model_from if args.train_mode == "10" else ""

    if args.train_mode in ("01", "10") and not args.model_from:
        parser.error("--train-mode 01/10 requires --model-from")
    if args.train_mode == "00" and args.model_from:
        parser.error("--model-from is only used with --train-mode 01 or 10")
    if args.fault_benchmark_trials < 0:
        parser.error("--fault-benchmark-trials must be >= 0")
    if args.fault_benchmark_trials and args.train_mode != "00":
        parser.error("--fault-benchmark-trials cannot be combined with --train-mode 01/10")
    if args.fault_benchmark_trials and not os.environ.get("PICORV32_BUG_DEFINE", ""):
        parser.error("--fault-benchmark-trials requires PICORV32_BUG_DEFINE to be set before compile/run")
    if args.fault_benchmark_trials and not args.fault_model_from:
        parser.error("--trials requires --load-model to point to a clean coverage-trained model")

    os.environ["PICORV32_WORK_LIB"] = FAULT_SIM_LIB
    check_environment(args.questa)
    ensure_sim_library(args.questa, FAULT_SIM_LIB, bug_define=os.environ.get("PICORV32_BUG_DEFINE", ""))

    if args.fault_benchmark_trials:
        run_fault_benchmark(args)
        return

    run_standard_experiment(args)


if __name__ == "__main__":
    main()
