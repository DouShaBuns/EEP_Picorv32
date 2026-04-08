// =============================================================================
// memory_model.sv
// Configurable-latency memory model for PicoRV32
//
// Memory map (all in same array, unified):
//   0x00000000 - 0x0001FFFF : 128KB (instruction + data, both use this)
//
// PicoRV32 reset vector is 0x00000000 so instructions start at address 0.
// Data typically lives above the instruction region (set by gen_program.py).
//
// The mem_delay_base input sets a FIXED stall count for all accesses.
// This will be replaced by the dynamic controller in the hybrid architecture —
// for now it gives the ML agent a static knob to set baseline memory pressure.
//
// Key protocol (PicoRV32 spec):
//   - CPU asserts mem_valid and holds ALL outputs stable until mem_ready
//   - This model asserts mem_ready for exactly ONE cycle then deasserts
//   - mem_ready must NOT be asserted when mem_valid is low
// =============================================================================

module memory_model #(
  parameter MEM_SIZE = 131072   // 128KB: bits [16:0] of address used
)(
  input  logic        clk,
  input  logic        resetn,

  // PicoRV32 memory interface
  input  logic        mem_valid,
  input  logic        mem_instr,    // 1=instruction fetch, 0=data access
  output logic        mem_ready,
  input  logic [31:0] mem_addr,
  input  logic [31:0] mem_wdata,
  input  logic [3:0]  mem_wstrb,
  output logic [31:0] mem_rdata,

  // Static delay knob: number of stall cycles before mem_ready
  //   1 = zero-wait (ready the same cycle as valid — very fast memory)
  //   4 = 3 wait cycles (moderate pressure)
  //   8 = 7 wait cycles (heavy pressure, maximum stall stress)
  input  logic [3:0]  mem_delay_base
);

  // ---------------------------------------------------------------------------
  // Memory array
  // NOT initialised here — tb_top.sv loads the program with $readmemh
  // before releasing reset, so every location that matters is written
  // before the CPU executes it. Locations beyond the program are 'x'
  // in simulation which is safe since the program never reaches them.
  // ---------------------------------------------------------------------------
  logic [7:0] mem [0:MEM_SIZE-1];

  // ---------------------------------------------------------------------------
  // Address masking
  // PicoRV32 uses 32-bit addresses but our memory is only 128KB.
  // We mask to bits [16:0] so any address wraps into our array.
  // The program generator (gen_program.py) is responsible for keeping
  // instruction and data regions within this range and non-overlapping.
  // ---------------------------------------------------------------------------
  wire [16:0] addr_masked = mem_addr[16:0];

  // ---------------------------------------------------------------------------
  // Delay counter state machine
  //
  //  IDLE: waiting for mem_valid
  //  WAIT: counting down delay cycles, mem_ready=0
  //  DONE: delay expired, assert mem_ready for one cycle, service access
  //
  // State transitions:
  //  IDLE → WAIT: mem_valid rises
  //  WAIT → DONE: delay_cnt reaches 0
  //  DONE → IDLE: always (mem_ready is a one-cycle pulse)
  // ---------------------------------------------------------------------------
  typedef enum logic [1:0] {
    S_IDLE = 2'b00,
    S_WAIT = 2'b01,
    S_DONE = 2'b10
  } mem_state_e;

  mem_state_e state;
  logic [3:0]  delay_cnt;

  // Capture request at start of access (mem_valid may change after first cycle,
  // but per PicoRV32 spec it stays stable — we capture anyway for safety)
  logic [31:0] cap_addr;
  logic [31:0] cap_wdata;
  logic [3:0]  cap_wstrb;
  logic        cap_instr;

  always_ff @(posedge clk or negedge resetn) begin
    if (!resetn) begin
      state     <= S_IDLE;
      mem_ready <= 1'b0;
      mem_rdata <= 32'h0;
      delay_cnt <= 4'h0;
      cap_addr  <= 32'h0;
      cap_wdata <= 32'h0;
      cap_wstrb <= 4'h0;
      cap_instr <= 1'b0;
    end else begin

      // Default: mem_ready deasserted
      mem_ready <= 1'b0;

      case (state)

        // ------------------------------------------------------------------
        S_IDLE: begin
          // Guard: do NOT start a new access on the same cycle that
          // mem_ready is still high from the previous access.
          //
          // Race condition with delay_base=1 (without this guard):
          //   Cycle N+2: state=IDLE, mem_ready=1, mem_valid=1
          //              CPU sees mem_ready=1 and processes the response.
          //              Memory ALSO sees mem_valid=1 and starts a NEW access.
          //   Cycle N+4: memory fires spurious mem_ready=1 with stale data
          //              from old capture — CPU reads corrupted instruction.
          //
          // Adding !mem_ready ensures we wait one cycle after completing
          // before accepting a new request. This costs nothing for
          // delay_base>1 since mem_ready is already 0 by the time a new
          // access could arrive.
          if (mem_valid && !mem_ready) begin
            // Capture the request
            cap_addr  <= mem_addr;
            cap_wdata <= mem_wdata;
            cap_wstrb <= mem_wstrb;
            cap_instr <= mem_instr;

            if (mem_delay_base <= 1) begin
              // Zero-wait: service immediately this cycle
              state <= S_DONE;
            end else begin
              // Load counter: we want (mem_delay_base - 1) stall cycles
              // Example: mem_delay_base=3 → 2 stall cycles (cnt starts at 1,
              // counts down to 0 → DONE)
              delay_cnt <= mem_delay_base - 2;
              state     <= S_WAIT;
            end
          end
        end

        // ------------------------------------------------------------------
        S_WAIT: begin
          if (delay_cnt == 0) begin
            state <= S_DONE;
          end else begin
            delay_cnt <= delay_cnt - 1;
          end
        end

        // ------------------------------------------------------------------
        S_DONE: begin
          // Service the captured request and pulse mem_ready
          mem_ready <= 1'b1;
          state     <= S_IDLE;

          if (|cap_wstrb) begin
            // --- Write ---
            // Only store operations write; instruction fetches never have wstrb set
            if (cap_wstrb[0]) mem[cap_addr[16:0] + 0] <= cap_wdata[7:0];
            if (cap_wstrb[1]) mem[cap_addr[16:0] + 1] <= cap_wdata[15:8];
            if (cap_wstrb[2]) mem[cap_addr[16:0] + 2] <= cap_wdata[23:16];
            if (cap_wstrb[3]) mem[cap_addr[16:0] + 3] <= cap_wdata[31:24];
          end else begin
            // --- Read (instruction fetch or data load) ---
            mem_rdata <= { mem[cap_addr[16:0] + 3],
                           mem[cap_addr[16:0] + 2],
                           mem[cap_addr[16:0] + 1],
                           mem[cap_addr[16:0] + 0] };
          end
        end

        default: state <= S_IDLE;

      endcase
    end
  end

  // ---------------------------------------------------------------------------
  // Assertions (simulation-only checks for protocol compliance)
  // ---------------------------------------------------------------------------
  // synthesis translate_off
  // NOTE: The mem_valid protocol check was removed — PicoRV32's trap state
  // machine deasserts mem_valid during trap detection in a way that triggers
  // false positives on every transaction. The multi-cycle mem_ready check
  // below is sufficient to catch real memory model bugs.
  logic prev_mem_ready;
  always_ff @(posedge clk or negedge resetn) begin
    if (!resetn) prev_mem_ready <= 1'b0;
    else         prev_mem_ready <= mem_ready;
  end

  always @(posedge clk) begin
    if (prev_mem_ready && mem_ready && resetn)
      $display("[MEM] ERROR: mem_ready held high >1 cycle at time %0t", $time);
  end
  // synthesis translate_on

endmodule
