"""
gen_program.py
RISC-V program generator for PicoRV32 ML-DV experiment

Usage:
  python programs/gen_program.py 0        <- reads work/knobs_0.json
  python programs/gen_program.py --test   <- uses built-in test knobs

Design: UNROLLED (no runtime loops)
  Why: A counted loop requires patching a BEQ offset after the body is
  emitted. The body size varies because gen_store randomly emits 1 or 5
  instructions. Wrong patch offsets cause infinite loops or early exits.
  Unrolling eliminates all of this — the program is a straight line from
  prologue to EBREAK with no backwards branches.

Memory layout:
  0x0000 - 0x3FFF : instruction memory  (program lives here, 16KB)
  0x4000 - 0x4FFF : data memory         (loads/stores target this region)
"""

import sys, os, random, json, argparse

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, THIS_DIR)
sys.path.insert(0, PROJECT_ROOT)
from riscv_encoder import *

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
INSTR_MEM_BYTES   = 0x4000
INSTR_WORDS       = INSTR_MEM_BYTES // 4   # 4096 RV32I instructions fit in 16 KB
N_COPIES          = 20      # number of body copies
N_SLOTS           = 40      # instruction slots per body copy
DATA_BASE         = 0x4000  # data memory starts here
DATA_WRAP         = 0x0F00  # wrap pointer at DATA_BASE + DATA_WRAP
DATA_REGION_BYTES = 0x1000
TOTAL_IMAGE_BYTES = DATA_BASE + DATA_REGION_BYTES
SIGNATURE_BASE    = DATA_BASE + 0x0FC0

# ---------------------------------------------------------------------------
# Register names
# ---------------------------------------------------------------------------
R0  = 0   # hardwired zero
R1  = 1   # data pointer (moves by stride)
R2  = 2   # data base (fixed, for wrap)
R3  = 3   # stride in bytes
R4  = 4   # accumulator A
R5  = 5   # accumulator B
R6  = 6   # scratch / temp


class ProgramGenerator:
    SIGNATURE_WORD_BUDGET = 6

    def __init__(self, knobs, seed=0):
        self.knobs  = knobs
        self.rng    = random.Random(seed)
        self.instrs = []
        self.pc     = 0

        # Instruction mix probabilities
        lw    = knobs['load_weight']
        sw    = knobs['store_weight']
        bw    = knobs['branch_weight']
        jw    = knobs.get('jump_weight', 1)
        aw    = knobs['arith_weight']
        total = lw + sw + bw + jw + aw
        self.p_load   = lw / total
        self.p_store  = sw / total
        self.p_branch = bw / total
        self.p_jump   = jw / total
        self.p_arith  = aw / total

        self.stride              = knobs['mem_stride'] * 4   # bytes
        self.trap_rate           = knobs.get('trap_rate', 0)
        self.pointer_update_prob = knobs.get('pointer_update_rate', 3) / 10.0
        self.branch_taken_prob   = knobs.get('branch_taken_bias', 5) / 10.0
        self.trap_kind           = knobs.get('trap_kind', 0)
        self.mixed_burst_prob    = knobs.get('mixed_burst_bias', 0) / 20.0
        self.load_pointer_update_prob = min(self.pointer_update_prob * 0.35, 0.50)
        self.oracle_enabled = (self.trap_rate == 0)
        self.model_regs = {reg: 0 for reg in range(32)}
        self.model_mem = {}

        # trap_rate=1 → ~1.5% chance per slot  (~12 traps over 20x40=800 slots)
        # trap_rate=2 → ~3.0% chance per slot  (~24 traps)
        # trap_rate=3 → ~4.5% chance per slot  (~36 traps)
        self.trap_prob = self.trap_rate * 0.015

    @staticmethod
    def _u32(val):
        return val & 0xFFFFFFFF

    @staticmethod
    def _s32(val):
        val &= 0xFFFFFFFF
        return val if val < 0x80000000 else val - 0x100000000

    def _mem_store_word(self, addr, value):
        addr = int(addr)
        value = self._u32(value)
        for i in range(4):
            self.model_mem[addr + i] = (value >> (8 * i)) & 0xFF

    def _mem_load_word(self, addr):
        addr = int(addr)
        out = 0
        for i in range(4):
            out |= (self.model_mem.get(addr + i, 0) & 0xFF) << (8 * i)
        return self._u32(out)

    def _emit_load_imm32(self, rd, imm):
        lower = imm & 0xFFF
        if lower >= 0x800:
            lower -= 0x1000
        upper = self._u32(imm - lower)
        self.e(LUI(rd, upper))
        if lower != 0:
            self.e(ADDI(rd, rd, lower))
        if self.oracle_enabled:
            self.model_regs[rd] = self._u32(imm)

    def _model_signature_epilogue(self):
        if not self.oracle_enabled:
            return
        self.model_regs[R6] = self._u32(SIGNATURE_BASE)
        self._mem_store_word(SIGNATURE_BASE + 0, self.model_regs[R1])
        self._mem_store_word(SIGNATURE_BASE + 4, self.model_regs[R4])
        self._mem_store_word(SIGNATURE_BASE + 8, self.model_regs[R5])

    def _data_region_checksum(self):
        checksum = 0x811C9DC5
        for addr in range(DATA_BASE, DATA_BASE + DATA_REGION_BYTES):
            checksum ^= self.model_mem.get(addr, 0) & 0xFF
            checksum = (checksum * 0x01000193) & 0xFFFFFFFF
        return checksum

    def build_oracle_metadata(self, sim_id):
        if not self.oracle_enabled:
            return {
                "sim_id": sim_id,
                "oracle_enabled": False,
                "reason": "trap_rate_nonzero",
                "expected_trap_count": None,
                "expected_data_region_checksum": None,
            }

        return {
            "sim_id": sim_id,
            "oracle_enabled": True,
            "reason": "trap_free_single_pass",
            "expected_trap_count": 1,
            "expected_data_region_checksum": self._data_region_checksum(),
            "data_region_base": DATA_BASE,
            "data_region_bytes": DATA_REGION_BYTES,
            "signature_base": SIGNATURE_BASE,
            "signature_words": 3,
        }

    # --- emit ---

    def e(self, instr):
        self.instrs.append(instr & 0xFFFFFFFF)
        self.pc += 4

    # --- prologue ---

    def gen_prologue(self):
        # R2 = DATA_BASE = 0x4000
        # IMPORTANT: the encoder's LUI function already extracts the upper 20 bits
        # internally via (imm >> 12). Pass the full address, not the pre-shifted value.
        # Wrong: LUI(R2, DATA_BASE >> 12)  → encoder does (4 >> 12) = 0 → R2 = 0
        # Right: LUI(R2, DATA_BASE)        → encoder does (0x4000 >> 12) = 4 → R2 = 0x4000
        self.e(LUI(R2, DATA_BASE))
        # R1 = R2 (data pointer starts at DATA_BASE)
        self.e(ADDI(R1, R2, 0))
        # R3 = stride in bytes (clamped to 12-bit ADDI range)
        self.e(ADDI(R3, R0, min(self.stride, 2044)))
        # R4, R5 = non-zero accumulators so stores write real data
        self.e(ADDI(R4, R0, 1))
        self.e(ADDI(R5, R0, 2))
        self.e(ADDI(R6, R0, 0))
        if self.oracle_enabled:
            self.model_regs[R2] = self._u32(DATA_BASE)
            self.model_regs[R1] = self.model_regs[R2]
            self.model_regs[R3] = self._u32(min(self.stride, 2044))
            self.model_regs[R4] = 1
            self.model_regs[R5] = 2
            self.model_regs[R6] = 0

    # --- instruction generators ---

    def gen_load(self):
        dest   = self.rng.choice([R4, R5])
        offset = self.rng.randint(0, 6) * 4   # word-aligned: 0,4,8,...,24
        self.e(LW(dest, R1, offset))
        if self.oracle_enabled:
            addr = self._u32(self.model_regs[R1] + offset)
            self.model_regs[dest] = self._mem_load_word(addr)
        if self.rng.random() < self.load_pointer_update_prob:
            self._advance_pointer()

    def _advance_pointer(self):
        self.e(ADD(R1, R1, R3))                    # R1 += stride
        self.e(ADDI(R6, R2, min(DATA_WRAP, 2044))) # R6 = BASE + WRAP
        self.e(BLT(R1, R6, 8))                     # if R1 < limit: skip reset
        self.e(ADDI(R1, R2, 0))                    # R1 = BASE (reset)
        if self.oracle_enabled:
            self.model_regs[R1] = self._u32(self.model_regs[R1] + self.model_regs[R3])
            self.model_regs[R6] = self._u32(self.model_regs[R2] + min(DATA_WRAP, 2044))
            if self._s32(self.model_regs[R1]) >= self._s32(self.model_regs[R6]):
                self.model_regs[R1] = self.model_regs[R2]

    def gen_store(self):
        src    = self.rng.choice([R4, R5])
        offset = self.rng.randint(0, 6) * 4
        self.e(SW(R1, src, offset))
        if self.oracle_enabled:
            addr = self._u32(self.model_regs[R1] + offset)
            self._mem_store_word(addr, self.model_regs[src])
        if self.rng.random() < self.pointer_update_prob:
            self._advance_pointer()

    def gen_branch(self):
        taken = self.rng.random() < self.branch_taken_prob
        branch_sel = self.rng.choice(["bne", "beq", "blt", "bge"])
        # Use only invariant operands so the encoded branch outcome matches the
        # software model exactly. R2 is pinned to DATA_BASE (> 0) for the whole run.
        if taken:
            if branch_sel == "beq":
                self.e(BEQ(R0, R0, 8))
            elif branch_sel == "blt":
                self.e(BLT(R0, R2, 8))
            elif branch_sel == "bge":
                self.e(BGE(R2, R0, 8))
            else:
                self.e(BNE(R2, R0, 8))
        else:
            if branch_sel == "beq":
                self.e(BEQ(R2, R0, 8))
            elif branch_sel == "blt":
                self.e(BLT(R2, R0, 8))
            elif branch_sel == "bge":
                self.e(BGE(R0, R2, 8))
            else:
                self.e(BNE(R0, R0, 8))
        # Make branch outcome observably affect the final oracle signature.
        # If a taken branch is incorrectly dropped by an injected control bug,
        # this increment will execute and perturb R4, which is recorded in the
        # final signature block.
        self.e(ADDI(R4, R4, 1))   # skipped instruction
        if self.oracle_enabled and not taken:
            self.model_regs[R4] = self._u32(self.model_regs[R4] + 1)

    def gen_jump(self):
        self.e(JAL(R0, 8))
        self.e(ADDI(R5, R5, 1))   # skipped instruction

    def gen_arith(self):
        op  = self.rng.randint(0, 4)
        rd  = self.rng.choice([R4, R5, R6])
        rs1 = self.rng.choice([R4, R5])
        rs2 = self.rng.choice([R5, R6])
        imm = self.rng.randint(1, 31)
        if   op == 0: self.e(ADD(rd, rs1, rs2))
        elif op == 1: self.e(ADDI(rd, rs1, imm))
        elif op == 2: self.e(AND(rd, rs1, rs2))
        elif op == 3: self.e(OR(rd, rs1, rs2))
        else:         self.e(XOR(rd, rs1, rs2))
        if self.oracle_enabled:
            a = self.model_regs[rs1]
            b = self.model_regs[rs2]
            if op == 0:
                self.model_regs[rd] = self._u32(a + b)
            elif op == 1:
                self.model_regs[rd] = self._u32(a + imm)
            elif op == 2:
                self.model_regs[rd] = self._u32(a & b)
            elif op == 3:
                self.model_regs[rd] = self._u32(a | b)
            else:
                self.model_regs[rd] = self._u32(a ^ b)

    def gen_misaligned_load_trap(self):
        self.e(ADDI(R6, R4, 1))   # breadcrumb — helps identify trap in coverage
        self.e(LW(R6, R2, 1))     # misaligned offset=1 → trap

    def gen_misaligned_store_trap(self):
        self.e(ADDI(R6, R5, 1))
        self.e(SW(R2, R4, 2))     # misaligned word store

    def gen_ebreak_trap(self):
        self.e(ADDI(R6, R6, 7))
        self.e(EBREAK())

    def gen_trap(self):
        trap_kind = self.trap_kind
        if trap_kind == 3:
            trap_kind = self.rng.choice([0, 1, 2])

        if trap_kind == 1:
            self.gen_misaligned_store_trap()
        elif trap_kind == 2:
            self.gen_ebreak_trap()
        else:
            self.gen_misaligned_load_trap()

    def gen_mixed_burst(self, max_ops=4):
        burst_len = 2
        if max_ops >= 4 and self.mixed_burst_prob >= 0.35 and self.rng.random() < 0.35:
            burst_len = 4
        elif max_ops >= 3 and self.mixed_burst_prob >= 0.20 and self.rng.random() < 0.55:
            burst_len = 3

        start_with_load = self.rng.random() < 0.5
        for idx in range(burst_len):
            is_load = (idx % 2 == 0) if start_with_load else (idx % 2 == 1)
            if is_load:
                self.gen_load()
            else:
                self.gen_store()
        return burst_len

    def gen_body_copy(self):
        """One unrolled copy of the body: N_SLOTS instruction slots.
        When trap_rate > 0, misaligned traps are injected inline at
        trap_prob probability per slot so they are spread throughout
        execution rather than appended after all body copies.
        This ensures multiple trap events are actually reached and
        exercises PicoRV32's trap state machine path multiple times."""
        slot = 0
        while slot < N_SLOTS:
            # Inline trap injection — must be tested BEFORE normal slot selection
            if self.trap_prob > 0 and self.rng.random() < self.trap_prob:
                self.gen_trap()
                slot += 1
                continue   # this slot used for trap; skip normal instruction

            if slot + 1 < N_SLOTS and self.mixed_burst_prob > 0 and self.rng.random() < self.mixed_burst_prob:
                remaining_slots = N_SLOTS - slot
                burst_ops = self.gen_mixed_burst(max_ops=min(4, remaining_slots))
                slot += burst_ops
                continue

            r = self.rng.random()
            if   r < self.p_load:
                self.gen_load()
            elif r < self.p_load + self.p_store:
                self.gen_store()
            elif r < self.p_load + self.p_store + self.p_branch:
                self.gen_branch()
            elif r < self.p_load + self.p_store + self.p_branch + self.p_jump:
                self.gen_jump()
            else:
                self.gen_arith()
            slot += 1

    # --- top-level ---

    def generate(self):
        self.instrs = []
        self.pc     = 0

        self.gen_prologue()

        # N_COPIES unrolled body copies — traps woven in via gen_body_copy
        for _ in range(N_COPIES):
            self.gen_body_copy()

        if len(self.instrs) >= INSTR_WORDS - self.SIGNATURE_WORD_BUDGET:
            self.instrs = self.instrs[:INSTR_WORDS - self.SIGNATURE_WORD_BUDGET]
            self.pc = 4 * len(self.instrs)

        self._emit_load_imm32(R6, SIGNATURE_BASE)
        self.e(SW(R6, R1, 0))
        self.e(SW(R6, R4, 4))
        self.e(SW(R6, R5, 8))
        self._model_signature_epilogue()

        # EBREAK — always the definitive end-of-test signal
        self.e(EBREAK())

        # NOP padding to fill the 16 KB instruction region only.
        while len(self.instrs) < INSTR_WORDS:
            self.instrs.append(NOP())

        return self.instrs


# ---------------------------------------------------------------------------
# Public API (called by run_experiment.py)
# ---------------------------------------------------------------------------

def _write_program_image(instructions, filepath):
    """Write a full memory image:
    - instruction region [0x0000, 0x3fff] contains the program bytes
    - data region        [0x4000, 0x4fff] is explicitly zero-filled
    This keeps the testbench checksum model aligned with the generated oracle.
    """
    byte_values = []
    for instr in instructions:
        word = instr & 0xFFFFFFFF
        byte_values.extend([
            (word >> 0) & 0xFF,
            (word >> 8) & 0xFF,
            (word >> 16) & 0xFF,
            (word >> 24) & 0xFF,
        ])

    if len(byte_values) > INSTR_MEM_BYTES:
        byte_values = byte_values[:INSTR_MEM_BYTES]
    else:
        byte_values.extend([0x00] * (INSTR_MEM_BYTES - len(byte_values)))

    byte_values.extend([0x00] * DATA_REGION_BYTES)

    with open(filepath, "w") as f:
        for b in byte_values[:TOTAL_IMAGE_BYTES]:
            f.write(f"{b:02x}\n")


def derive_sim_seed(seed_base, sim_id):
    mask64 = (1 << 64) - 1
    x = (int(seed_base) & mask64) ^ ((int(sim_id) + 1) * 0x9E3779B97F4A7C15)
    x &= mask64
    x ^= (x >> 30)
    x = (x * 0xBF58476D1CE4E5B9) & mask64
    x ^= (x >> 27)
    x = (x * 0x94D049BB133111EB) & mask64
    x ^= (x >> 31)
    return int(x & 0xFFFFFFFF)


def generate_program(sim_id, knobs, output_dir="work", seed_base=0, sim_seed=None):
    seed = derive_sim_seed(seed_base, sim_id) if sim_seed is None else (int(sim_seed) & 0xFFFFFFFF)
    gen    = ProgramGenerator(knobs, seed=seed)
    instrs = gen.generate()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"prog_{sim_id}.hex")
    _write_program_image(instrs, path)
    oracle_path = os.path.join(output_dir, f"oracle_{sim_id}.json")
    oracle_meta = gen.build_oracle_metadata(sim_id)
    oracle_meta["program_seed"] = seed
    with open(oracle_path, "w") as f:
        json.dump(oracle_meta, f, indent=2)

    return path


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sim_id", nargs="?", type=int, default=0,
                        help="Simulation ID — reads work/knobs_<sim_id>.json")
    parser.add_argument("--work",  default="work")
    parser.add_argument("--test",  action="store_true",
                        help="Use built-in test knobs instead of JSON")
    args = parser.parse_args()

    if args.test:
        knobs = {"load_weight":5, "store_weight":5, "branch_weight":3,
                 "jump_weight":3, "arith_weight":5, "mem_stride":4,
                 "pointer_update_rate":3, "trap_rate":0, "trap_kind":0,
                 "branch_taken_bias":5, "mixed_burst_bias":2,
                 "mem_delay_base":3}
        print("[gen_program] Using built-in test knobs")
    else:
        knob_file = os.path.join(args.work, f"knobs_{args.sim_id}.json")
        if not os.path.exists(knob_file):
            print(f"ERROR: {knob_file} not found. Use --test for defaults.")
            sys.exit(1)
        with open(knob_file) as f:
            knobs = json.load(f)
        print(f"[gen_program] Read from {knob_file}")

    path = generate_program(sim_id=args.sim_id, knobs=knobs, output_dir=args.work)
    print(f"[gen_program] Written: {path}")
