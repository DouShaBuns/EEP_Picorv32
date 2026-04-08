// =============================================================================
// coverage_pkg.sv
// Coverage type definitions for PicoRV32 ML-DV experiment
//
// PicoRV32 has a UNIFIED memory bus — instruction fetches and data accesses
// share the same interface, distinguished only by mem_instr.
// This means stalls can occur on either type, and the interesting corner cases
// are when both types of access are competing or stalling consecutively.
//
// Access types on the PicoRV32 bus:
//   Type A: Instruction fetch  (mem_instr=1, mem_wstrb=0)
//   Type B: Data load          (mem_instr=0, mem_wstrb=0)
//   Type C: Data store         (mem_instr=0, mem_wstrb!=0)
//
// Coverage metrics exported via DPI at end of simulation:
// =============================================================================
package coverage_pkg;

  // ---------------------------------------------------------------------------
  // Access type encoding - used internally in testbench
  // ---------------------------------------------------------------------------
  typedef enum logic [1:0] {
    ACC_INSTR = 2'b00,   // instruction fetch
    ACC_LOAD  = 2'b01,   // data load
    ACC_STORE = 2'b10,   // data store
    ACC_NONE  = 2'b11    // no active access
  } access_type_e;

  // ---------------------------------------------------------------------------
  // Coverage result struct
  // All fields passed back to Python ML agent via DPI write_coverage()
  // ---------------------------------------------------------------------------
  typedef struct packed {

    // --- Cycle counts ---
    int unsigned total_cycles;         // total simulation cycles after reset
    int unsigned active_cycles;        // cycles with an active bus request

    // --- Stall breakdown by access type ---
    // A "stall cycle" is any cycle where mem_valid=1 AND mem_ready=0
    int unsigned instr_stall_cycles;   // stalls during instruction fetch
    int unsigned load_stall_cycles;    // stalls during data load
    int unsigned store_stall_cycles;   // stalls during data store
    int unsigned total_stall_cycles;   // sum of above three

    // --- Completed transaction counts ---
    // Unlike total_mem_requests, these count completed accesses rather than
    // request-hold cycles. They provide a better denominator for rates.
    int unsigned completed_accesses;
    int unsigned completed_instr_accesses;
    int unsigned completed_load_accesses;
    int unsigned completed_store_accesses;

    // --- Consecutive stall tracking ---
    // "stall run" = consecutive cycles where mem_valid=1 and mem_ready=0
    // for the SAME transaction (one access held low for N cycles)
    int unsigned max_stall_run;        // longest single stall (cycles)
    int unsigned stall_runs_gt2;       // count of stall runs longer than 2 cycles
    int unsigned stall_runs_gt4;       // count of stall runs longer than 4 cycles
    int unsigned stall_runs_gt8;       // count of stall runs longer than 8 cycles

    // --- Long stall breakdown by access type ---
    // Counts only runs that are long enough to be interesting from a DV
    // perspective (strictly greater than 4 cycles).
    int unsigned long_instr_stall_runs;
    int unsigned long_load_stall_runs;
    int unsigned long_store_stall_runs;

    // --- Back-to-back and burst pressure ---
    // "B2B stall" = the cycle immediately AFTER one stalled access completes,
    // a new access begins AND also stalls. No gap between stalled accesses.
    int unsigned b2b_stall_count;
    int unsigned data_burst_count;     // consecutive data accesses with no fetch gap
    int unsigned mixed_burst_count;    // fetch↔data transitions under sustained pressure

    // --- Access type transitions ---
    // Tracks sequences like fetch→load, load→store, fetch→fetch etc.
    int unsigned fetch_then_load;
    int unsigned fetch_then_store;
    int unsigned load_then_load;
    int unsigned load_then_store;
    int unsigned store_then_load;
    int unsigned store_then_store;
    int unsigned transition_types_hit; // number of distinct transition classes seen
    int unsigned mixed_data_transition_count;
    int unsigned consecutive_store_bursts;
    int unsigned consecutive_mixed_bursts;

    // --- Instruction mix (decoded from fetched instructions) ---
    int unsigned instr_count;          // total instructions executed (fetched+ready)
    int unsigned load_instr_count;     // LW/LH/LB instructions
    int unsigned store_instr_count;    // SW/SH/SB instructions
    int unsigned branch_instr_count;   // BEQ/BNE/BLT/BGE instructions
    int unsigned jump_instr_count;     // JAL/JALR instructions
    int unsigned arith_instr_count;    // ADD/SUB/ADDI/AND/OR/XOR etc.

    // --- Memory request counts ---
    int unsigned total_mem_requests;   // total cycles where mem_valid=1
    int unsigned instr_requests;       // instruction fetch requests
    int unsigned data_requests;        // data access requests (loads + stores)

    // --- Trap and recovery tracking ---
    // trap_count includes the final terminating trap. intermediate_trap_count
    // strips that off so the ML side can reward genuine extra recovery events.
    int unsigned trap_count;               // total trap rising edges detected
    int unsigned intermediate_trap_count;  // max(trap_count - 1, 0)
    int unsigned recovery_count;           // number of soft-reset recoveries performed

    // --- Near-miss indicators ---
    // These are "close but not quite" signatures for hard-to-hit compound
    // coverage situations. They provide denser ML feedback than exact hits.
    int unsigned near_trap_deep_stall;     // trap happened with medium stall, not long stall
    int unsigned near_mixed_b2b;           // B2B stall plus mixed access sequence observed
    int unsigned near_transition_diversity;// >=3 transition classes observed
    int unsigned near_long_data_stall;     // data-side pressure close to long-stall target
    int unsigned near_full_stress;         // broad stress signature without exact hardest case

  } coverage_result_t;

endpackage
