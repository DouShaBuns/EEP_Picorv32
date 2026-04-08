// =============================================================================
// knob_io.h
// DPI-C bridge header for PicoRV32 ML-DV
//
// File-based communication protocol:
//   Python writes:  work/knobs_<SIM_ID>.json     (before simulation)
//   C reads:        work/knobs_<SIM_ID>.json      (in get_knobs)
//   C writes:       work/coverage_<SIM_ID>.json   (in write_coverage)
//   Python reads:   work/coverage_<SIM_ID>.json   (after simulation)
//
// SIM_ID is passed as an environment variable set by run_experiment.py
// before launching vsim, so the DPI bridge knows which files to read/write.
// =============================================================================

#ifndef KNOB_IO_H
#define KNOB_IO_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WORK_DIR  "work"
#define MAX_PATH  512

// ── get_knobs ──────────────────────────────────────────────────────────────
// Called once at simulation start from tb_top.sv
// Reads work/knobs_<SIM_ID>.json
//
// Knob JSON format:
// {
//   "sim_id": 5,
//   "load_weight":    7,
//   "store_weight":   5,
//   "branch_weight":  2,
//   "jump_weight":    4,
//   "arith_weight":   4,
//   "mem_stride":     4,
//   "pointer_update_rate": 6,
//   "trap_rate":      1,
//   "trap_kind":      3,
//   "branch_taken_bias": 7,
//   "mixed_burst_bias": 4,
//   "mem_delay_base": 4
// }
//
void get_knobs(
  int* sim_id,
  int* knob_load_weight,
  int* knob_store_weight,
  int* knob_branch_weight,
  int* knob_jump_weight,
  int* knob_arith_weight,
  int* knob_mem_stride,
  int* knob_pointer_update_rate,
  int* knob_trap_rate,
  int* knob_trap_kind,
  int* knob_branch_taken_bias,
  int* knob_mixed_burst_bias,
  int* knob_mem_delay_base
);

// ── write_coverage ─────────────────────────────────────────────────────────
// Called once at simulation end from tb_top.sv
// Writes work/coverage_<SIM_ID>.json
void write_coverage(
  int sim_id,
  // Cycle counts
  int total_cycles,
  int active_cycles,
  // Stall breakdown by access type
  int instr_stall_cycles,
  int load_stall_cycles,
  int store_stall_cycles,
  int total_stall_cycles,
  // Completed transactions
  int completed_accesses,
  int completed_instr_accesses,
  int completed_load_accesses,
  int completed_store_accesses,
  // Stall depth metrics
  int max_stall_run,
  int stall_runs_gt2,
  int stall_runs_gt4,
  int stall_runs_gt8,
  int long_instr_stall_runs,
  int long_load_stall_runs,
  int long_store_stall_runs,
  // Back-to-back stalls
  int b2b_stall_count,
  int data_burst_count,
  int mixed_burst_count,
  // Access type transitions
  int fetch_then_load,
  int fetch_then_store,
  int load_then_load,
  int load_then_store,
  int store_then_load,
  int store_then_store,
  int transition_types_hit,
  int mixed_data_transition_count,
  int consecutive_store_bursts,
  int consecutive_mixed_bursts,
  // Instruction mix
  int instr_count,
  int load_instr_count,
  int store_instr_count,
  int branch_instr_count,
  int jump_instr_count,
  int arith_instr_count,
  // Memory request counts
  int total_mem_requests,
  int instr_requests,
  int data_requests,
  // Trap coverage
  int trap_count,
  int intermediate_trap_count,
  int recovery_count,
  // Whether the testbench reached the cycle watchdog before final completion
  int timed_out,
  // Final data-memory checksum for bug-detection oracle
  int data_region_checksum,
  // Bug trigger debug
  int bug_region_hit_count,
  int bug_arm_count,
  int bug_branch_candidate_count,
  int bug_manifest_count,
  int raw_mixed_transition_count,
  int raw_shortstall_data_xfer_count,
  int raw_taken_branch_count,
  int raw_taken_branch_while_armed_count,
  int knob_load_weight,
  int knob_store_weight,
  int knob_branch_weight,
  int knob_jump_weight,
  int knob_arith_weight,
  int knob_mem_stride,
  int knob_pointer_update_rate,
  int knob_trap_rate,
  int knob_trap_kind,
  int knob_branch_taken_bias,
  int knob_mixed_burst_bias,
  int knob_mem_delay_base,
  // Near misses
  int near_trap_deep_stall,
  int near_mixed_b2b,
  int near_transition_diversity,
  int near_long_data_stall,
  int near_full_stress
);

#endif // KNOB_IO_H
