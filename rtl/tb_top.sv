// =============================================================================
// tb_top.sv
// PicoRV32 ML-DV Testbench
//
// ── KNOB REFERENCE ───────────────────────────────────────────────────────────
//
// WHY THESE KNOBS AND NOT dep_dist:
//   PicoRV32 is NOT a pipelined CPU. It executes instructions sequentially via
//   a state machine — fetch completes fully before decode, decode before execute.
//   There are no pipeline stages and no forwarding network, so RAW (Read-After-
//   Write) hazards do NOT exist. dep_dist is meaningless here and has been
//   replaced with trap_rate, which stresses a path that actually exists in
//   PicoRV32's state machine.
//
// STATIC knobs (set by ML agent before simulation, control program generation):
//
//   load_weight  [1-10]
//     Relative frequency of load instructions (LW/LH/LB) in the program.
//     PicoRV32 effect: each load causes TWO bus transactions — one instruction
//     fetch (to get the LW opcode) and one data read (to get the value).
//     HIGH value → bus occupied more often by data reads, maximum stall density.
//     This is unique to PicoRV32's unified bus: loads cost twice as much bus
//     time as arithmetic instructions, which only cost one fetch transaction.
//
//   store_weight [1-10]
//     Relative frequency of store instructions (SW/SH/SB).
//     Same unified-bus effect as loads: each store = one fetch + one data write.
//     HIGH value → write stalls interleaved with fetch stalls.
//     Combined high load+store = maximum bus occupancy, highest stall_ratio.
//
//   branch_weight [1-10]
//     Relative frequency of taken branch instructions (BEQ/BNE/BLT/BGE).
//     PicoRV32 effect: when a branch is taken, the CPU must fetch from the
//     branch target address. If the memory model is slow (high mem_delay_base),
//     this re-fetch stalls too. HIGH value → frequent PC redirections, tests
//     the state machine transitions around branch resolution under stall.
//
//   arith_weight [1-10]
//     Relative frequency of arithmetic instructions (ADD/ADDI/AND/OR/XOR).
//     These cause ONLY ONE bus transaction (instruction fetch — no data access).
//     Acts as a "diluter" — high arith_weight reduces memory pressure.
//     LOW arith + HIGH load/store = maximum bus pressure, closest to 100% stall.
//     HIGH arith = breathing room between memory ops, tests recovery behaviour.
//
//   mem_stride [1-8]
//     Word stride between consecutive data memory accesses (in words = 4 bytes).
//     stride=1: SW/LW to 0x2000, 0x2004, 0x2008 ... (cache-friendly pattern)
//     stride=4: SW/LW to 0x2000, 0x2010, 0x2020 ... (16-byte steps)
//     stride=8: SW/LW to 0x2000, 0x2020, 0x2040 ... (32-byte steps)
//     Controls address spread of data accesses across the memory array.
//     Does not affect stall count directly but varies the address patterns
//     presented to the memory model — relevant when dynamic controller is added.
//
//   trap_rate [0-3]
//     Controls how often the program generator inserts trap-inducing instructions.
//     0 = no traps (pure load/store/branch/arith mix, clean execution)
//     1 = occasional misaligned accesses inserted at ~1.5% per slot
//         → triggers CATCH_MISALIGN trap path in PicoRV32's state machine
//     2 = misaligned accesses at ~3.0% per slot
//         → more frequent trap/recovery cycles, higher trap_count
//     3 = misaligned accesses at ~4.5% per slot
//         → maximum trap stress, tests prioritisation and recovery logic
//
//     IMPORTANT: Traps are injected INLINE within the body (not appended after).
//     After each intermediate trap the testbench soft-resets the CPU and
//     re-executes the program. trap_count tracks total trap events across all
//     resets. With trap_rate=0 this is exactly 1 (the final EBREAK).
//
// DYNAMIC knob (baseline set by ML agent, varies per-cycle in hybrid mode):
//
//   mem_delay_base [1-8]
//     Baseline memory latency in cycles before mem_ready is asserted.
//     1 = zero-wait (mem_ready same cycle as mem_valid, fastest possible)
//     4 = 3 stall cycles per access (moderate pressure)
//     8 = 7 stall cycles per access (maximum pressure)
//     Critically: affects BOTH instruction fetches AND data accesses because
//     PicoRV32 has a UNIFIED memory bus — there is no separate I-cache bus.
//     In the hybrid architecture this value is the anchor that the per-cycle
//     dynamic controller varies around based on what the CPU is currently doing.
//
// ── COVERAGE REFERENCE ───────────────────────────────────────────────────────
//
// Primary reward metric for ML agent:
//   stall_ratio = total_stall_cycles / total_cycles
//
// Secondary metrics (richer signal for ML agent, better dissertation analysis):
//   instr_stall_ratio  = instr_stall_cycles  / total_cycles
//   load_stall_ratio   = load_stall_cycles   / total_cycles
//   store_stall_ratio  = store_stall_cycles  / total_cycles
//   b2b_stall_rate     = b2b_stall_count     / total_mem_requests
//   max_stall_run      = longest consecutive stall on one transaction
//   trap_count         = number of traps hit during simulation (from trap_rate)
//
// Functional coverage bins (SV covergroup):
//   - stall length category × access type (cross)
//   - back-to-back stall occurrence
//   - access type sequence (fetch→load, load→store, etc.)
//   - trap_rate knob × stall depth (did traps occur under heavy stall?)
//   - delay setting × trap occurrence (trap during max stall = hardest case)
//
// =============================================================================

import coverage_pkg::*;

// ── DPI IMPORTS ──────────────────────────────────────────────────────────────

// get_knobs: called ONCE at start of simulation
// Reads work/knobs_<SIM_ID>.json written by Python ML agent
import "DPI-C" function void get_knobs(
  output int sim_id,
  // Static instruction-mix knobs
  output int knob_load_weight,
  output int knob_store_weight,
  output int knob_branch_weight,
  output int knob_jump_weight,
  output int knob_arith_weight,
  // Static memory access pattern knobs
  output int knob_mem_stride,
  output int knob_pointer_update_rate,
  // Static trap insertion controls
  output int knob_trap_rate,
  output int knob_trap_kind,
  output int knob_branch_taken_bias,
  output int knob_mixed_burst_bias,
  // Static baseline memory latency (dynamic layer built on top later)
  output int knob_mem_delay_base
);

// write_coverage: called ONCE at end of simulation
// Writes work/coverage_<SIM_ID>.json for Python ML agent to read
import "DPI-C" function void write_coverage(
  input int sim_id,
  // Cycle counts
  input int total_cycles,
  input int active_cycles,
  // Stall breakdown
  input int instr_stall_cycles,
  input int load_stall_cycles,
  input int store_stall_cycles,
  input int total_stall_cycles,
  // Completed transactions
  input int completed_accesses,
  input int completed_instr_accesses,
  input int completed_load_accesses,
  input int completed_store_accesses,
  // Stall depth metrics
  input int max_stall_run,
  input int stall_runs_gt2,
  input int stall_runs_gt4,
  input int stall_runs_gt8,
  input int long_instr_stall_runs,
  input int long_load_stall_runs,
  input int long_store_stall_runs,
  // Back-to-back stalls
  input int b2b_stall_count,
  input int data_burst_count,
  input int mixed_burst_count,
  // Access type transitions
  input int fetch_then_load,
  input int fetch_then_store,
  input int load_then_load,
  input int load_then_store,
  input int store_then_load,
  input int store_then_store,
  input int transition_types_hit,
  input int mixed_data_transition_count,
  input int consecutive_store_bursts,
  input int consecutive_mixed_bursts,
  // Instruction mix
  input int instr_count,
  input int load_instr_count,
  input int store_instr_count,
  input int branch_instr_count,
  input int jump_instr_count,
  input int arith_instr_count,
  // Memory request counts
  input int total_mem_requests,
  input int instr_requests,
  input int data_requests,
  // Trap coverage
  input int trap_count,
  input int intermediate_trap_count,
  input int recovery_count,
  input int timed_out,
  // Final data-memory checksum for bug-detection oracle
  input int data_region_checksum,
  // Bug trigger debug
  input int bug_region_hit_count,
  input int bug_arm_count,
  input int bug_branch_candidate_count,
  input int bug_manifest_count,
  input int raw_mixed_transition_count,
  input int raw_shortstall_data_xfer_count,
  input int raw_taken_branch_count,
  input int raw_taken_branch_while_armed_count,
  // Knob dump for per-sim debugging
  input int knob_load_weight_dbg,
  input int knob_store_weight_dbg,
  input int knob_branch_weight_dbg,
  input int knob_jump_weight_dbg,
  input int knob_arith_weight_dbg,
  input int knob_mem_stride_dbg,
  input int knob_pointer_update_rate_dbg,
  input int knob_trap_rate_dbg,
  input int knob_trap_kind_dbg,
  input int knob_branch_taken_bias_dbg,
  input int knob_mixed_burst_bias_dbg,
  input int knob_mem_delay_base_dbg,
  // Near misses
  input int near_trap_deep_stall,
  input int near_mixed_b2b,
  input int near_transition_diversity,
  input int near_long_data_stall,
  input int near_full_stress
);

// ── TOP MODULE ────────────────────────────────────────────────────────────────

module tb_top;

  // ── SIMULATION PARAMETERS ──────────────────────────────────────────────────
  parameter CLK_PERIOD = 10;       // 10ns → 100MHz
  parameter MAX_CYCLES = 100000;   // hard timeout (cycles after reset)
  parameter MEM_SIZE   = 131072;   // 128KB

  // ── CLOCK AND RESET ────────────────────────────────────────────────────────
  logic clk    = 1'b0;
  logic resetn = 1'b0;

  always #(CLK_PERIOD/2) clk = ~clk;

  // ── KNOB REGISTERS ─────────────────────────────────────────────────────────
  // Populated by get_knobs() DPI call at start of simulation
  int sim_id;
  int knob_load_weight;
  int knob_store_weight;
  int knob_branch_weight;
  int knob_jump_weight;
  int knob_arith_weight;
  int knob_mem_stride;
  int knob_pointer_update_rate;
  int knob_trap_rate;
  int knob_trap_kind;
  int knob_branch_taken_bias;
  int knob_mixed_burst_bias;
  int knob_mem_delay_base;

  // ── PICORV32 MEMORY INTERFACE SIGNALS ──────────────────────────────────────
  logic        mem_valid;   // CPU: I want a memory access
  logic        mem_instr;   // CPU: 1=fetch, 0=data
  logic        mem_ready;   // MEM: access complete
  logic [31:0] mem_addr;    // CPU: address
  logic [31:0] mem_wdata;   // CPU: write data
  logic [3:0]  mem_wstrb;   // CPU: byte write enables (0=read)
  logic [31:0] mem_rdata;   // MEM: read data
  logic        trap;        // CPU: trapped (illegal instruction / breakpoint)

  // ── COVERAGE REGISTERS ─────────────────────────────────────────────────────
  // All counters reset to 0 before simulation starts

  // Cycle counters
  int unsigned total_cycles;
  int unsigned active_cycles;

  // Stall breakdown
  int unsigned instr_stall_cycles;
  int unsigned load_stall_cycles;
  int unsigned store_stall_cycles;
  int unsigned total_stall_cycles;

  // Completed transactions
  int unsigned completed_accesses;
  int unsigned completed_instr_accesses;
  int unsigned completed_load_accesses;
  int unsigned completed_store_accesses;

  // Stall depth tracking
  int unsigned cur_stall_run;      // current consecutive stall length
  int unsigned max_stall_run;
  int unsigned stall_runs_gt2;
  int unsigned stall_runs_gt4;
  int unsigned stall_runs_gt8;
  int unsigned long_instr_stall_runs;
  int unsigned long_load_stall_runs;
  int unsigned long_store_stall_runs;

  // Back-to-back stall tracking
  int unsigned b2b_stall_count;
  int unsigned data_burst_count;
  int unsigned mixed_burst_count;
  logic        prev_was_stall;     // did the last completed access stall?

  // Access type transition tracking
  access_type_e prev_access_type;  // type of last COMPLETED access
  access_type_e prev_data_access_type; // last completed LOAD/STORE, ignoring fetches
  int unsigned  fetch_then_load;
  int unsigned  fetch_then_store;
  int unsigned  load_then_load;
  int unsigned  load_then_store;
  int unsigned  store_then_load;
  int unsigned  store_then_store;
  logic         seen_fetch_then_load;
  logic         seen_fetch_then_store;
  logic         seen_load_then_load;
  logic         seen_load_then_store;
  logic         seen_store_then_load;
  logic         seen_store_then_store;
  int unsigned  transition_types_hit;
  int unsigned  mixed_data_transition_count;
  int unsigned  consecutive_store_bursts;
  int unsigned  consecutive_mixed_bursts;
  logic         prev_data_transition_mixed;

  // Instruction mix (decoded from fetched instructions)
  int unsigned instr_count;
  int unsigned load_instr_count;
  int unsigned store_instr_count;
  int unsigned branch_instr_count;
  int unsigned jump_instr_count;
  int unsigned arith_instr_count;

  // Memory request counts
  int unsigned total_mem_requests;
  int unsigned instr_requests;
  int unsigned data_requests;

  // Trap tracking
  // Counts how many times the CPU asserted trap during simulation.
  // With trap_rate=0 this should be exactly 1 (the final EBREAK).
  // With trap_rate>0 it will be higher — each extra trap exercises a
  // different path in PicoRV32's state machine.
  int unsigned trap_count;
  int unsigned intermediate_trap_count;
  int unsigned recovery_count;
  logic        sim_timed_out = 1'b0;
  int unsigned bug_region_hit_count;
  int unsigned bug_arm_count;
  int unsigned bug_branch_candidate_count;
  int unsigned bug_manifest_count;
  int unsigned raw_mixed_transition_count;
  int unsigned raw_shortstall_data_xfer_count;
  int unsigned raw_taken_branch_count;
  int unsigned raw_taken_branch_while_armed_count;
  logic        bug_region_prev;
  logic        bug_armed_prev;
  logic        bug_branch_candidate_prev;
  logic        bug_manifest_prev;
  logic        raw_prev_data_was_load;
  logic        raw_prev_data_was_store;
  logic        raw_taken_branch_prev;
  logic        raw_taken_branch_while_armed_prev;
  logic        trap_prev;    // previous cycle trap value (edge detection)

  // Near-miss indicators for sparse hard bins
  int unsigned near_trap_deep_stall;
  int unsigned near_mixed_b2b;
int unsigned near_transition_diversity;
int unsigned near_long_data_stall;
int unsigned near_full_stress;
int unsigned data_region_checksum;

localparam int DATA_ORACLE_BASE  = 32'h0000_4000;
localparam int DATA_ORACLE_BYTES = 32'h0000_1000;

  // Trap recovery loop counter (local to initial block, declared here
  // so it is accessible in the scope of the recovery fork)
  int unsigned trap_count_seen;
  logic        cov_started = 1'b0;

  // ── ACCESS TYPE DECODE ─────────────────────────────────────────────────────
  // Combinationally decode what kind of access is currently in progress
  access_type_e current_access_type;

  always_comb begin
    if (!mem_valid)
      current_access_type = ACC_NONE;
    else if (mem_instr)
      current_access_type = ACC_INSTR;
    else if (|mem_wstrb)
      current_access_type = ACC_STORE;
    else
      current_access_type = ACC_LOAD;
  end

  // ── FUNCTIONAL COVERGROUP ──────────────────────────────────────────────────
  // Samples on every positive clock edge where an access is active

  covergroup picorv32_mem_cg @(posedge clk);
    option.per_instance = 1;

    // Is a stall happening right now?
    cp_stalling: coverpoint (mem_valid && !mem_ready) {
      bins no_stall = {1'b0};
      bins stalling = {1'b1};
    }

    // What type of access is happening?
    cp_access_type: coverpoint current_access_type {
      bins fetch = {ACC_INSTR};
      bins load  = {ACC_LOAD};
      bins store = {ACC_STORE};
      bins idle  = {ACC_NONE};
    }

    // Stall run length bucket
    // 'medium' is a reserved SV keyword — use med_depth instead
    cp_stall_depth: coverpoint max_stall_run {
      bins short_depth = {[1:2]};    // 1-2 cycle stalls
      bins med_depth   = {[3:5]};    // 3-5 cycle stalls
      bins long_depth  = {[6:8]};    // 6-8 cycle stalls
      bins zero_depth  = {0};
    }

    // mem_delay_base setting (from knob)
    // 'medium' is reserved — use med_delay instead
    cp_delay_setting: coverpoint knob_mem_delay_base {
      bins fast_delay = {[1:2]};
      bins med_delay  = {[3:5]};
      bins slow_delay = {[6:8]};
    }

    // trap_rate knob setting
    cp_trap_rate: coverpoint knob_trap_rate {
      bins trap_none     = {0};   // clean execution only
      bins trap_low      = {1};   // ~1.5% trap injection
      bins trap_med      = {2};   // ~3.0% trap injection
      bins trap_high     = {3};   // ~4.5% trap injection
    }

    // Was at least one non-final trap observed during this simulation?
    cp_trap_occurred: coverpoint (trap_count > 1) {
      bins no_trap  = {1'b0};
      bins had_trap = {1'b1};
    }

    // Cross: stall happening x access type
    // ignore_bins inline cross syntax not supported in all QuestaSim versions
    // — use simple cross without ignore for compatibility
    cx_stall_x_access: cross cp_stalling, cp_access_type;

    // Cross: delay setting x stall depth achieved
    cx_delay_x_depth: cross cp_delay_setting, cp_stall_depth;

    // Cross: trap_rate x stall depth
    cx_trap_x_depth: cross cp_trap_rate, cp_stall_depth;

    // Cross: trap occurrence x delay setting
    cx_trap_occurred_x_delay: cross cp_trap_occurred, cp_delay_setting;

  endgroup

  picorv32_mem_cg u_cg = new();

  // ── DUT: PICORV32 ──────────────────────────────────────────────────────────
  picorv32 #(
    .ENABLE_COUNTERS     (1),  // enable cycle/instruction counters
    .ENABLE_COUNTERS64   (0),
    .ENABLE_REGS_16_31   (1),  // use all 32 registers
    .ENABLE_REGS_DUALPORT(1),
    .TWO_STAGE_SHIFT     (1),
    .BARREL_SHIFTER      (0),
    .TWO_CYCLE_COMPARE   (0),
    .TWO_CYCLE_ALU       (0),
    .COMPRESSED_ISA      (0),  // RV32I only (no C extension)
    .CATCH_MISALIGN      (1),  // trap on misaligned access
    .CATCH_ILLINSN       (1),  // trap on illegal instruction
    .ENABLE_PCPI         (0),
    .ENABLE_MUL          (0),
    .ENABLE_FAST_MUL     (0),
    .ENABLE_DIV          (0),
    .ENABLE_IRQ          (0),
    .ENABLE_IRQ_QREGS    (0),
    .ENABLE_TRACE        (0),
    .REGS_INIT_ZERO      (1)   // initialise all registers to 0
  ) u_cpu (
    // ── Clock and reset ──────────────────────────────────────────────────────
    .clk        (clk),
    .resetn     (resetn),

    // ── Standard memory interface ─────────────────────────────────────────────
    // This is the only interface our testbench uses
    .mem_valid  (mem_valid),
    .mem_instr  (mem_instr),
    .mem_ready  (mem_ready),
    .mem_addr   (mem_addr),
    .mem_wdata  (mem_wdata),
    .mem_wstrb  (mem_wstrb),
    .mem_rdata  (mem_rdata),

    // ── Look-ahead memory interface ───────────────────────────────────────────
    // PicoRV32 outputs these one cycle early as a performance hint.
    // We don't use them — leave outputs unconnected (floating is fine
    // for outputs in simulation, they're driven by the CPU not by us)
    .mem_la_read  (),
    .mem_la_write (),
    .mem_la_addr  (),
    .mem_la_wdata (),
    .mem_la_wstrb (),

    // ── Co-processor interface (PCPI) ─────────────────────────────────────────
    // ENABLE_PCPI=0 so this interface is inactive.
    // Outputs left unconnected, inputs tied to safe values:
    //   pcpi_ready=0: no co-processor ready
    //   pcpi_rd=0:    no co-processor result
    //   pcpi_wait=0:  co-processor never stalls
    //   pcpi_wr=0:    co-processor never writes back
    .pcpi_valid (),
    .pcpi_insn  (),
    .pcpi_rs1   (),
    .pcpi_rs2   (),
    .pcpi_wr    (1'b0),
    .pcpi_rd    (32'h0),
    .pcpi_wait  (1'b0),
    .pcpi_ready (1'b0),

    // ── IRQ interface ─────────────────────────────────────────────────────────
    // ENABLE_IRQ=0 so no interrupts.
    // eoi (end-of-interrupt) is an output — leave unconnected
    .irq        (32'h0),
    .eoi        (),

    // ── Trace interface ───────────────────────────────────────────────────────
    // ENABLE_TRACE=0 so trace outputs are inactive — leave unconnected
    .trace_valid (),
    .trace_data  (),

    // ── Trap output ───────────────────────────────────────────────────────────
    // Goes high on EBREAK, illegal instruction, or misaligned access
    .trap       (trap)
  );

  // ── MEMORY MODEL ───────────────────────────────────────────────────────────
  memory_model #(
    .MEM_SIZE(MEM_SIZE)
  ) u_mem (
    .clk           (clk),
    .resetn        (resetn),
    .mem_valid     (mem_valid),
    .mem_instr     (mem_instr),
    .mem_ready     (mem_ready),
    .mem_addr      (mem_addr),
    .mem_wdata     (mem_wdata),
    .mem_wstrb     (mem_wstrb),
    .mem_rdata     (mem_rdata),
    .mem_delay_base(knob_mem_delay_base[3:0])
  );

  // ── COVERAGE COLLECTION ────────────────────────────────────────────────────
  // All tracking is clocked, triggered each cycle after reset

  always_ff @(posedge clk or negedge resetn) begin
    if (!resetn) begin
      if (!cov_started) begin
        total_cycles       <= 0;
        active_cycles      <= 0;
        instr_stall_cycles <= 0;
        load_stall_cycles  <= 0;
        store_stall_cycles <= 0;
        total_stall_cycles <= 0;
        completed_accesses       <= 0;
        completed_instr_accesses <= 0;
        completed_load_accesses  <= 0;
        completed_store_accesses <= 0;
        max_stall_run      <= 0;
        stall_runs_gt2     <= 0;
        stall_runs_gt4     <= 0;
        stall_runs_gt8     <= 0;
        long_instr_stall_runs <= 0;
        long_load_stall_runs  <= 0;
        long_store_stall_runs <= 0;
        b2b_stall_count    <= 0;
        data_burst_count   <= 0;
        mixed_burst_count  <= 0;
        fetch_then_load    <= 0;
        fetch_then_store   <= 0;
        load_then_load     <= 0;
        load_then_store    <= 0;
        store_then_load    <= 0;
        store_then_store   <= 0;
        seen_fetch_then_load  <= 1'b0;
        seen_fetch_then_store <= 1'b0;
        seen_load_then_load   <= 1'b0;
        seen_load_then_store  <= 1'b0;
        seen_store_then_load  <= 1'b0;
        seen_store_then_store <= 1'b0;
        transition_types_hit  <= 0;
        mixed_data_transition_count <= 0;
        consecutive_store_bursts    <= 0;
        consecutive_mixed_bursts    <= 0;
        instr_count        <= 0;
        load_instr_count   <= 0;
        store_instr_count  <= 0;
        branch_instr_count <= 0;
        jump_instr_count   <= 0;
        arith_instr_count  <= 0;
        total_mem_requests <= 0;
        instr_requests     <= 0;
        data_requests      <= 0;
        trap_count         <= 0;
        intermediate_trap_count  <= 0;
        recovery_count           <= 0;
        bug_region_hit_count     <= 0;
        bug_arm_count            <= 0;
        bug_branch_candidate_count <= 0;
        bug_manifest_count       <= 0;
        raw_mixed_transition_count <= 0;
        raw_shortstall_data_xfer_count <= 0;
        raw_taken_branch_count <= 0;
        raw_taken_branch_while_armed_count <= 0;
        near_trap_deep_stall      <= 0;
        near_mixed_b2b            <= 0;
        near_transition_diversity <= 0;
        near_long_data_stall      <= 0;
        near_full_stress          <= 0;
      end

      cur_stall_run      <= 0;
      prev_was_stall     <= 1'b0;
      prev_access_type   <= ACC_NONE;
      prev_data_access_type <= ACC_NONE;
      prev_data_transition_mixed <= 1'b0;
      trap_prev          <= 1'b0;
      bug_region_prev    <= 1'b0;
      bug_armed_prev     <= 1'b0;
      bug_branch_candidate_prev <= 1'b0;
      bug_manifest_prev  <= 1'b0;
      raw_prev_data_was_load <= 1'b0;
      raw_prev_data_was_store <= 1'b0;
      raw_taken_branch_prev <= 1'b0;
      raw_taken_branch_while_armed_prev <= 1'b0;
    end else begin
      cov_started <= 1'b1;

      total_cycles <= total_cycles + 1;

      // ── Stall tracking ──────────────────────────────────────────────────
      if (mem_valid) begin
        active_cycles      <= active_cycles + 1;
        total_mem_requests <= total_mem_requests + 1;

        if (mem_instr) instr_requests <= instr_requests + 1;
        else           data_requests  <= data_requests  + 1;

        if (!mem_ready) begin
          // Active stall this cycle
          total_stall_cycles <= total_stall_cycles + 1;
          cur_stall_run      <= cur_stall_run + 1;

          // Attribute stall to access type
          case (current_access_type)
            ACC_INSTR: instr_stall_cycles <= instr_stall_cycles + 1;
            ACC_LOAD:  load_stall_cycles  <= load_stall_cycles  + 1;
            ACC_STORE: store_stall_cycles <= store_stall_cycles + 1;
            default: ;
          endcase

          // Update running maximum
          if (cur_stall_run + 1 > max_stall_run)
            max_stall_run <= cur_stall_run + 1;

        end else begin
          // mem_ready asserted — access completes this cycle

          completed_accesses <= completed_accesses + 1;
          case (current_access_type)
            ACC_INSTR: completed_instr_accesses <= completed_instr_accesses + 1;
            ACC_LOAD:  completed_load_accesses  <= completed_load_accesses  + 1;
            ACC_STORE: completed_store_accesses <= completed_store_accesses + 1;
            default: ;
          endcase

          if ((current_access_type inside {ACC_LOAD, ACC_STORE}) && (cur_stall_run <= 2))
            raw_shortstall_data_xfer_count <= raw_shortstall_data_xfer_count + 1;

          // Check if this run was long enough to count
          if (cur_stall_run > 2) stall_runs_gt2 <= stall_runs_gt2 + 1;
          if (cur_stall_run > 4) stall_runs_gt4 <= stall_runs_gt4 + 1;
          if (cur_stall_run > 8) stall_runs_gt8 <= stall_runs_gt8 + 1;
          if (cur_stall_run > 4) begin
            case (current_access_type)
              ACC_INSTR: long_instr_stall_runs <= long_instr_stall_runs + 1;
              ACC_LOAD:  long_load_stall_runs  <= long_load_stall_runs  + 1;
              ACC_STORE: long_store_stall_runs <= long_store_stall_runs + 1;
              default: ;
            endcase
          end

          // Check for back-to-back stall:
          // This access stalled (cur_stall_run > 0) AND the previous
          // completed access also stalled
          if (cur_stall_run > 0 && prev_was_stall)
            b2b_stall_count <= b2b_stall_count + 1;

          // Record stall status for next access
          prev_was_stall <= (cur_stall_run > 0);

          // ── Access type transition tracking ─────────────────────────────
          // Fetch transitions are tracked from the immediately previous bus
          // access. Data transitions are tracked from the previous completed
          // data access so they survive the fetches that naturally occur
          // between RISC-V memory instructions on PicoRV32's unified bus.
          case ({prev_access_type, current_access_type})
            {ACC_INSTR, ACC_LOAD}: begin
              fetch_then_load <= fetch_then_load + 1;
              seen_fetch_then_load <= 1'b1;
              mixed_burst_count <= mixed_burst_count + 1;
            end
            {ACC_INSTR, ACC_STORE}: begin
              fetch_then_store <= fetch_then_store + 1;
              seen_fetch_then_store <= 1'b1;
              mixed_burst_count <= mixed_burst_count + 1;
            end
            default: ;
          endcase

          if (current_access_type inside {ACC_LOAD, ACC_STORE}) begin
            case ({prev_data_access_type, current_access_type})
              {ACC_LOAD, ACC_LOAD}: begin
                load_then_load <= load_then_load + 1;
                seen_load_then_load <= 1'b1;
                data_burst_count <= data_burst_count + 1;
                prev_data_transition_mixed <= 1'b0;
              end
              {ACC_LOAD, ACC_STORE}: begin
                load_then_store <= load_then_store + 1;
                seen_load_then_store <= 1'b1;
                data_burst_count <= data_burst_count + 1;
                mixed_burst_count <= mixed_burst_count + 1;
                mixed_data_transition_count <= mixed_data_transition_count + 1;
                if (cur_stall_run <= 2)
                  raw_mixed_transition_count <= raw_mixed_transition_count + 1;
                if (prev_data_transition_mixed)
                  consecutive_mixed_bursts <= consecutive_mixed_bursts + 1;
                prev_data_transition_mixed <= 1'b1;
              end
              {ACC_STORE, ACC_LOAD}: begin
                store_then_load <= store_then_load + 1;
                seen_store_then_load <= 1'b1;
                data_burst_count <= data_burst_count + 1;
                mixed_burst_count <= mixed_burst_count + 1;
                mixed_data_transition_count <= mixed_data_transition_count + 1;
                if (cur_stall_run <= 2)
                  raw_mixed_transition_count <= raw_mixed_transition_count + 1;
                if (prev_data_transition_mixed)
                  consecutive_mixed_bursts <= consecutive_mixed_bursts + 1;
                prev_data_transition_mixed <= 1'b1;
              end
              {ACC_STORE, ACC_STORE}: begin
                store_then_store <= store_then_store + 1;
                seen_store_then_store <= 1'b1;
                data_burst_count <= data_burst_count + 1;
                consecutive_store_bursts <= consecutive_store_bursts + 1;
                prev_data_transition_mixed <= 1'b0;
              end
              default: prev_data_transition_mixed <= 1'b0;
            endcase

            prev_data_access_type <= current_access_type;
          end

          transition_types_hit <=
              int'(seen_fetch_then_load  || ({prev_access_type, current_access_type} == {ACC_INSTR, ACC_LOAD})) +
              int'(seen_fetch_then_store || ({prev_access_type, current_access_type} == {ACC_INSTR, ACC_STORE})) +
              int'(seen_load_then_load   || ((prev_data_access_type == ACC_LOAD)  && (current_access_type == ACC_LOAD))) +
              int'(seen_load_then_store  || ((prev_data_access_type == ACC_LOAD)  && (current_access_type == ACC_STORE))) +
              int'(seen_store_then_load  || ((prev_data_access_type == ACC_STORE) && (current_access_type == ACC_LOAD))) +
              int'(seen_store_then_store || ((prev_data_access_type == ACC_STORE) && (current_access_type == ACC_STORE)));

          prev_access_type <= current_access_type;

          // Reset stall run counter for next access
          cur_stall_run <= 0;

          // ── Instruction decode (only on completed instruction fetches) ────
          if (mem_instr) begin
            instr_count <= instr_count + 1;
            // Decode RISC-V opcode from bits [6:0] of the fetched instruction
            // Note: mem_rdata is the instruction word returned by memory
            case (mem_rdata[6:0])
              7'h03: load_instr_count   <= load_instr_count   + 1; // LOAD
              7'h23: store_instr_count  <= store_instr_count  + 1; // STORE
              7'h63: begin
                branch_instr_count <= branch_instr_count + 1; // BRANCH
                raw_taken_branch_count <= raw_taken_branch_count + 1;
                if (u_cpu.ctrl_bug_armed)
                  raw_taken_branch_while_armed_count <= raw_taken_branch_while_armed_count + 1;
              end
              7'h33: arith_instr_count  <= arith_instr_count  + 1; // OP (R-type)
              7'h13: arith_instr_count  <= arith_instr_count  + 1; // OP-IMM
              7'h37: ;  // LUI  — not tracked separately
              7'h17: ;  // AUIPC — not tracked separately
              7'h6F: jump_instr_count <= jump_instr_count + 1; // JAL
              7'h67: jump_instr_count <= jump_instr_count + 1; // JALR
              default: ;
            endcase
          end

        end // mem_ready

      end else begin
        // No active access — reset stall run
        // (PicoRV32 deasserts mem_valid between transactions)
        if (cur_stall_run > 0) begin
          prev_was_stall   <= (cur_stall_run > 0);
          prev_access_type <= ACC_NONE;
          cur_stall_run    <= 0;
        end
      end

      // ── Trap edge detection ───────────────────────────────────────────────
      // PicoRV32 holds trap high once asserted (it's a level signal, not a
      // pulse). We count rising edges to know how many trap events occurred.
      // trap_prev remembers the value from the previous cycle.
      trap_prev <= trap;
      if (trap && !trap_prev) begin
        trap_count <= trap_count + 1;
        if (max_stall_run >= 3 && max_stall_run < 6)
          near_trap_deep_stall <= 1;
      end

      intermediate_trap_count <= (trap_count > 0) ? (trap_count - 1) : 0;

      if (transition_types_hit >= 3)
        near_transition_diversity <= 1;

      if (b2b_stall_count > 0 && mixed_burst_count > 0)
        near_mixed_b2b <= 1;

      if ((load_stall_cycles + store_stall_cycles) > 0 && max_stall_run >= 3 && max_stall_run < 6)
        near_long_data_stall <= 1;

      if (max_stall_run >= 3 &&
          b2b_stall_count > 0 &&
          transition_types_hit >= 2 &&
          (load_stall_cycles + store_stall_cycles) > 0)
        near_full_stress <= 1;

`ifdef BUG_CTRL_DELAYED_STORE_BRANCH_DROP
      begin
        logic bug_region_now;
        logic bug_armed_now;
        logic bug_branch_candidate_now;
        logic bug_manifest_now;

        bug_region_now = (u_cpu.ctrl_bug_seq_state == 2'd3) || u_cpu.ctrl_bug_armed;
        bug_armed_now = u_cpu.ctrl_bug_armed;
        bug_branch_candidate_now = u_cpu.ctrl_bug_branch_candidate;
        bug_manifest_now = u_cpu.ctrl_bug_manifest;

        if (bug_region_now && !bug_region_prev)
          bug_region_hit_count <= bug_region_hit_count + 1;
        if (bug_armed_now && !bug_armed_prev)
          bug_arm_count <= bug_arm_count + 1;
        if (bug_branch_candidate_now && !bug_branch_candidate_prev)
          bug_branch_candidate_count <= bug_branch_candidate_count + 1;
        if (bug_manifest_now && !bug_manifest_prev)
          bug_manifest_count <= bug_manifest_count + 1;

        bug_region_prev <= bug_region_now;
        bug_armed_prev <= bug_armed_now;
        bug_branch_candidate_prev <= bug_branch_candidate_now;
        bug_manifest_prev <= bug_manifest_now;
      end
`elsif BUG_CTRL_DELAYED_STORE_BRANCH_DROP_RARE
      begin
        logic bug_region_now;
        logic bug_armed_now;
        logic bug_branch_candidate_now;
        logic bug_manifest_now;

        bug_region_now = (u_cpu.ctrl_bug_seq_state == 2'd3) || u_cpu.ctrl_bug_armed;
        bug_armed_now = u_cpu.ctrl_bug_armed;
        bug_branch_candidate_now = u_cpu.ctrl_bug_branch_candidate;
        bug_manifest_now = u_cpu.ctrl_bug_manifest;

        if (bug_region_now && !bug_region_prev)
          bug_region_hit_count <= bug_region_hit_count + 1;
        if (bug_armed_now && !bug_armed_prev)
          bug_arm_count <= bug_arm_count + 1;
        if (bug_branch_candidate_now && !bug_branch_candidate_prev)
          bug_branch_candidate_count <= bug_branch_candidate_count + 1;
        if (bug_manifest_now && !bug_manifest_prev)
          bug_manifest_count <= bug_manifest_count + 1;

        bug_region_prev <= bug_region_now;
        bug_armed_prev <= bug_armed_now;
        bug_branch_candidate_prev <= bug_branch_candidate_now;
        bug_manifest_prev <= bug_manifest_now;
      end
`endif

    end // resetn
  end

  // ── MAIN TEST SEQUENCE ────────────────────────────────────────────────────
  string prog_file;

  initial begin

    // ── Step 1: Read knobs via DPI ──────────────────────────────────────────
    // Python ML agent has already written work/knobs_<SIM_ID>.json
    // C bridge reads it and populates our knob registers
    get_knobs(
      sim_id,
      knob_load_weight,
      knob_store_weight,
      knob_branch_weight,
      knob_jump_weight,
      knob_arith_weight,
      knob_mem_stride,
      knob_pointer_update_rate,
      knob_trap_rate,
      knob_trap_kind,
      knob_branch_taken_bias,
      knob_mixed_burst_bias,
      knob_mem_delay_base
    );

    $display("");
    $display("[TB] -----------------------------------------------");
    $display("[TB]  PicoRV32 ML-DV Simulation: SIM_ID = %0d", sim_id);
    $display("[TB] -----------------------------------------------");
    $display("[TB]  Static knobs:");
    $display("[TB]    load_weight   = %0d", knob_load_weight);
    $display("[TB]    store_weight  = %0d", knob_store_weight);
    $display("[TB]    branch_weight = %0d", knob_branch_weight);
    $display("[TB]    jump_weight   = %0d", knob_jump_weight);
    $display("[TB]    arith_weight  = %0d", knob_arith_weight);
    $display("[TB]    mem_stride    = %0d  (%0d-byte steps)",
             knob_mem_stride, knob_mem_stride * 4);
    $display("[TB]    pointer_update_rate = %0d  (%.2f probability)",
             knob_pointer_update_rate, real'(knob_pointer_update_rate) / 10.0);
    $display("[TB]    branch_taken_bias   = %0d  (%.2f probability)",
             knob_branch_taken_bias, real'(knob_branch_taken_bias) / 10.0);
    $display("[TB]    mixed_burst_bias    = %0d  (%.2f probability)",
             knob_mixed_burst_bias, real'(knob_mixed_burst_bias) / 20.0);
    $display("[TB]    trap_rate     = %0d  (%0s)",
             knob_trap_rate,
             (knob_trap_rate == 0) ? "no traps"         :
             (knob_trap_rate == 1) ? "~1.5%% trap/slot" :
             (knob_trap_rate == 2) ? "~3.0%% trap/slot" : "~4.5%% trap/slot");
    $display("[TB]    trap_kind     = %0d  (%0s)",
             knob_trap_kind,
             (knob_trap_kind == 0) ? "misaligned load" :
             (knob_trap_kind == 1) ? "misaligned store" :
             (knob_trap_kind == 2) ? "inline ebreak" : "mixed");
    $display("[TB]  Dynamic knob (base):");
    $display("[TB]    mem_delay_base = %0d  (%0d stall cycles per access)",
             knob_mem_delay_base, knob_mem_delay_base - 1);
    $display("[TB] ------------------------------------------------");

    // ── Step 2: Load generated program ─────────────────────────────────────
    // gen_program.py creates work/prog_<SIM_ID>.hex based on the knobs above
    prog_file = $sformatf("work/prog_%0d.hex", sim_id);
    $display("[TB]  Loading program: %s", prog_file);
    $readmemh(prog_file, u_mem.mem);

    // ── Step 3: Initial reset sequence ─────────────────────────────────────
    resetn = 1'b0;
    repeat(8) @(posedge clk);   // hold reset for 8 cycles
    @(posedge clk);
    resetn = 1'b1;
    $display("[TB]  Reset released, CPU starting...");

    // ── Step 4: Run until final trap, recovering from intermediate traps ───
    //
    // FIX: The original one-shot @(posedge trap) caused the simulation to
    // terminate on the FIRST trap, making trap_count always = 1 regardless
    // of trap_rate. This recovery loop instead:
    //   - Waits for each trap in sequence
    //   - On intermediate traps: soft-resets CPU, reloads program, resumes
    //   - On the final trap (EBREAK): falls through to coverage collection
    //
    // With trap_rate=0: trap_target=1, loop runs once, exits on EBREAK
    // With trap_rate=2: trap_target=3, recovers from 2 intermediate traps
    //                   then exits on the 3rd (final EBREAK)
    begin
      int trap_target;
      trap_target     = knob_trap_rate + 1;
      trap_count_seen = 0;

      fork
        begin : trap_recovery_loop
          repeat(trap_target) begin
            @(posedge trap);
            trap_count_seen = trap_count_seen + 1;

            if (trap_count_seen < trap_target) begin
              // Intermediate trap — soft-reset CPU and re-execute
              $display("[TB]  Intermediate trap #%0d at cycle %0d — recovering",
                       trap_count_seen, total_cycles);
              resetn = 1'b0;
              repeat(4) @(posedge clk);
              $readmemh(prog_file, u_mem.mem);  // reload program into memory
              @(posedge clk);
              resetn = 1'b1;
              // Wait for trap to deassert (driven low when resetn goes low)
              @(negedge trap);
              $display("[TB]  CPU resumed after trap #%0d", trap_count_seen);
            end else begin
              $display("[TB]  Final EBREAK at cycle %0d — simulation complete",
                       total_cycles);
            end
          end
        end

        begin : timeout_watchdog
          repeat(MAX_CYCLES) @(posedge clk);
          $display("[TB]  WARNING: Timeout at cycle %0d (no trap received)",
                   total_cycles);
          sim_timed_out = 1'b1;
        end
      join_any
      disable fork;
    end

    // Let counters settle after final trap
    repeat(4) @(posedge clk);

    // ── Step 5: Print coverage summary ─────────────────────────────────────
    begin
      real stall_ratio, instr_sr, load_sr, store_sr;
      stall_ratio = (total_cycles > 0) ?
                    real'(total_stall_cycles) / real'(total_cycles) : 0.0;
      instr_sr    = (total_cycles > 0) ?
                    real'(instr_stall_cycles) / real'(total_cycles) : 0.0;
      load_sr     = (total_cycles > 0) ?
                    real'(load_stall_cycles)  / real'(total_cycles) : 0.0;
      store_sr    = (total_cycles > 0) ?
                    real'(store_stall_cycles) / real'(total_cycles) : 0.0;

      data_region_checksum = 32'h811C9DC5;
      for (int checksum_idx = DATA_ORACLE_BASE;
               checksum_idx < DATA_ORACLE_BASE + DATA_ORACLE_BYTES;
               checksum_idx++) begin
        data_region_checksum =
            (data_region_checksum ^ u_mem.mem[checksum_idx[16:0]]) * 32'h01000193;
      end

      $display("[TB] -------------------------------------------------");
      $display("[TB]  Coverage Summary:");
      $display("[TB]    total_cycles       = %0d",    total_cycles);
      $display("[TB]    active_cycles      = %0d",    active_cycles);
      $display("[TB]    total_stall_cycles = %0d  (ratio=%.4f)",
               total_stall_cycles, stall_ratio);
      $display("[TB]    instr_stall_cycles = %0d  (ratio=%.4f)",
               instr_stall_cycles, instr_sr);
      $display("[TB]    load_stall_cycles  = %0d  (ratio=%.4f)",
               load_stall_cycles, load_sr);
      $display("[TB]    store_stall_cycles = %0d  (ratio=%.4f)",
               store_stall_cycles, store_sr);
      $display("[TB]    max_stall_run      = %0d cycles", max_stall_run);
      $display("[TB]    stall_runs_gt2     = %0d", stall_runs_gt2);
      $display("[TB]    stall_runs_gt4     = %0d", stall_runs_gt4);
      $display("[TB]    stall_runs_gt8     = %0d", stall_runs_gt8);
      $display("[TB]    completed_accesses = %0d  (I=%0d L=%0d S=%0d)",
               completed_accesses,
               completed_instr_accesses,
               completed_load_accesses,
               completed_store_accesses);
      $display("[TB]    b2b_stall_count    = %0d", b2b_stall_count);
      $display("[TB]    data_burst_count   = %0d", data_burst_count);
      $display("[TB]    mixed_burst_count  = %0d", mixed_burst_count);
`ifdef BUG_CTRL_DELAYED_STORE_BRANCH_DROP
      $display("[TB]    bug_region_hits    = %0d", bug_region_hit_count);
      $display("[TB]    bug_arm_count      = %0d", bug_arm_count);
      $display("[TB]    bug_branch_cands   = %0d", bug_branch_candidate_count);
      $display("[TB]    bug_manifest_count = %0d", bug_manifest_count);
      $display("[TB]    raw_mixed_xitions  = %0d", raw_mixed_transition_count);
      $display("[TB]    raw_shortstall_xfr = %0d", raw_shortstall_data_xfer_count);
      $display("[TB]    raw_taken_branch   = %0d", raw_taken_branch_count);
      $display("[TB]    raw_taken_arm      = %0d", raw_taken_branch_while_armed_count);
`elsif BUG_CTRL_DELAYED_STORE_BRANCH_DROP_RARE
      $display("[TB]    bug_region_hits    = %0d", bug_region_hit_count);
      $display("[TB]    bug_arm_count      = %0d", bug_arm_count);
      $display("[TB]    bug_branch_cands   = %0d", bug_branch_candidate_count);
      $display("[TB]    bug_manifest_count = %0d", bug_manifest_count);
      $display("[TB]    raw_mixed_xitions  = %0d", raw_mixed_transition_count);
      $display("[TB]    raw_shortstall_xfr = %0d", raw_shortstall_data_xfer_count);
      $display("[TB]    raw_taken_branch   = %0d", raw_taken_branch_count);
      $display("[TB]    raw_taken_arm      = %0d", raw_taken_branch_while_armed_count);
`endif
      $display("[TB]    instructions exec  = %0d  (L=%0d S=%0d B=%0d A=%0d)",
               instr_count,
               load_instr_count, store_instr_count,
               branch_instr_count, arith_instr_count);
      $display("[TB]    jump_instr_count   = %0d", jump_instr_count);
      $display("[TB]    fetch_then_load    = %0d", fetch_then_load);
      $display("[TB]    fetch_then_store   = %0d", fetch_then_store);
      $display("[TB]    load_then_load     = %0d", load_then_load);
      $display("[TB]    load_then_store    = %0d", load_then_store);
      $display("[TB]    store_then_load    = %0d", store_then_load);
      $display("[TB]    store_then_store   = %0d", store_then_store);
      $display("[TB]    transition_types   = %0d", transition_types_hit);
      $display("[TB]    mixed_data_trans   = %0d", mixed_data_transition_count);
      $display("[TB]    store_bursts       = %0d", consecutive_store_bursts);
      $display("[TB]    mixed_burst_runs   = %0d", consecutive_mixed_bursts);
      $display("[TB]    trap_count         = %0d  (intermediate=%0d, recoveries=%0d)",
               trap_count, intermediate_trap_count, recovery_count);
      $display("[TB]    timed_out          = %0d", sim_timed_out);
      $display("[TB]    data_checksum      = 0x%08x", data_region_checksum);
      $display("[TB]    near_miss flags    = trap_depth=%0d mixed_b2b=%0d transition=%0d data_stall=%0d full_stress=%0d",
               near_trap_deep_stall, near_mixed_b2b, near_transition_diversity,
               near_long_data_stall, near_full_stress);
      $display("[TB] -------------------------------------------------");
    end

    // ── Step 6: Write coverage via DPI ─────────────────────────────────────
    // C bridge writes work/coverage_<SIM_ID>.json for Python ML agent
    write_coverage(
      sim_id,
      // Cycle counts
      int'(total_cycles),
      int'(active_cycles),
      // Stall breakdown
      int'(instr_stall_cycles),
      int'(load_stall_cycles),
      int'(store_stall_cycles),
      int'(total_stall_cycles),
      // Completed transactions
      int'(completed_accesses),
      int'(completed_instr_accesses),
      int'(completed_load_accesses),
      int'(completed_store_accesses),
      // Stall depth
      int'(max_stall_run),
      int'(stall_runs_gt2),
      int'(stall_runs_gt4),
      int'(stall_runs_gt8),
      int'(long_instr_stall_runs),
      int'(long_load_stall_runs),
      int'(long_store_stall_runs),
      // Back-to-back
      int'(b2b_stall_count),
      int'(data_burst_count),
      int'(mixed_burst_count),
      // Transitions
      int'(fetch_then_load),
      int'(fetch_then_store),
      int'(load_then_load),
      int'(load_then_store),
      int'(store_then_load),
      int'(store_then_store),
      int'(transition_types_hit),
      int'(mixed_data_transition_count),
      int'(consecutive_store_bursts),
      int'(consecutive_mixed_bursts),
      // Instruction mix
      int'(instr_count),
      int'(load_instr_count),
      int'(store_instr_count),
      int'(branch_instr_count),
      int'(jump_instr_count),
      int'(arith_instr_count),
      // Requests
      int'(total_mem_requests),
      int'(instr_requests),
      int'(data_requests),
      // Trap coverage
      int'(trap_count),
      int'(intermediate_trap_count),
      int'(recovery_count),
      int'(sim_timed_out),
      int'(data_region_checksum),
      int'(bug_region_hit_count),
      int'(bug_arm_count),
      int'(bug_branch_candidate_count),
      int'(bug_manifest_count),
      int'(raw_mixed_transition_count),
      int'(raw_shortstall_data_xfer_count),
      int'(raw_taken_branch_count),
      int'(raw_taken_branch_while_armed_count),
      int'(knob_load_weight),
      int'(knob_store_weight),
      int'(knob_branch_weight),
      int'(knob_jump_weight),
      int'(knob_arith_weight),
      int'(knob_mem_stride),
      int'(knob_pointer_update_rate),
      int'(knob_trap_rate),
      int'(knob_trap_kind),
      int'(knob_branch_taken_bias),
      int'(knob_mixed_burst_bias),
      int'(knob_mem_delay_base),
      // Near misses
      int'(near_trap_deep_stall),
      int'(near_mixed_b2b),
      int'(near_transition_diversity),
      int'(near_long_data_stall),
      int'(near_full_stress)
    );

    $display("[TB]  Coverage written for sim_id=%0d", sim_id);
    $finish;

  end

  // ── HARD WATCHDOG ──────────────────────────────────────────────────────────
  // Failsafe in case $finish is never reached
  initial begin
    #(MAX_CYCLES * CLK_PERIOD * 3);
    $display("[TB] HARD WATCHDOG TRIGGERED — forcing $finish");
    $finish;
  end

endmodule
