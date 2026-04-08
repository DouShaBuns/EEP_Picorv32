# =============================================================================
# riscv_encoder.py
# RISC-V RV32I instruction encoder
# Generates 32-bit machine code words from instruction parameters
#
# Supported instructions (sufficient to stress PicoRV32 memory/hazard paths):
#   R-type:  ADD, SUB, AND, OR, XOR, SLT
#   I-type:  ADDI, ANDI, ORI, SLTI, LW, JALR
#   S-type:  SW, SB, SH
#   B-type:  BEQ, BNE, BLT, BGE
#   U-type:  LUI, AUIPC
#   J-type:  JAL
#   Special: NOP (ADDI x0,x0,0), EBREAK
# =============================================================================

import struct

# ---------------------------------------------------------------------------
# RISC-V opcode constants
# ---------------------------------------------------------------------------
OP_LUI    = 0x37
OP_AUIPC  = 0x17
OP_JAL    = 0x6F
OP_JALR   = 0x67
OP_BRANCH = 0x63
OP_LOAD   = 0x03
OP_STORE  = 0x23
OP_IMM    = 0x13   # OP-IMM (ADDI, ANDI, ORI, ...)
OP_REG    = 0x33   # OP     (ADD, SUB, AND, ...)
OP_SYSTEM = 0x73

# funct3 for loads
F3_LB  = 0x0
F3_LH  = 0x1
F3_LW  = 0x2
F3_LBU = 0x4
F3_LHU = 0x5

# funct3 for stores
F3_SB  = 0x0
F3_SH  = 0x1
F3_SW  = 0x2

# funct3 for branches
F3_BEQ  = 0x0
F3_BNE  = 0x1
F3_BLT  = 0x4
F3_BGE  = 0x5
F3_BLTU = 0x6
F3_BGEU = 0x7

# funct3 for OP-IMM
F3_ADDI  = 0x0
F3_SLTI  = 0x2
F3_ANDI  = 0x7
F3_ORI   = 0x6
F3_XORI  = 0x4
F3_SLLI  = 0x1
F3_SRLI  = 0x5

# funct3/funct7 for OP (R-type)
F3_ADD  = 0x0; F7_ADD  = 0x00
F3_SUB  = 0x0; F7_SUB  = 0x20
F3_SLT  = 0x2; F7_SLT  = 0x00
F3_AND  = 0x7; F7_AND  = 0x00
F3_OR   = 0x6; F7_OR   = 0x00
F3_XOR  = 0x4; F7_XOR  = 0x00
F3_SLL  = 0x1; F7_SLL  = 0x00
F3_SRL  = 0x5; F7_SRL  = 0x00
F3_SRA  = 0x5; F7_SRA  = 0x20


def _sign_extend(val, bits):
    """Sign-extend val from bits width."""
    if val & (1 << (bits - 1)):
        val -= (1 << bits)
    return val


def _mask(val, bits):
    """Mask val to bits width (unsigned)."""
    return val & ((1 << bits) - 1)


# ---------------------------------------------------------------------------
# Instruction encoding functions
# ---------------------------------------------------------------------------

def encode_r(opcode, rd, funct3, rs1, rs2, funct7):
    """R-type: funct7[31:25] rs2[24:20] rs1[19:15] funct3[14:12] rd[11:7] opcode[6:0]"""
    return (_mask(funct7, 7) << 25 |
            _mask(rs2,    5) << 20 |
            _mask(rs1,    5) << 15 |
            _mask(funct3, 3) << 12 |
            _mask(rd,     5) <<  7 |
            _mask(opcode, 7))


def encode_i(opcode, rd, funct3, rs1, imm):
    """I-type: imm[11:0][31:20] rs1[19:15] funct3[14:12] rd[11:7] opcode[6:0]"""
    imm = _mask(imm, 12)
    return (imm << 20 |
            _mask(rs1,    5) << 15 |
            _mask(funct3, 3) << 12 |
            _mask(rd,     5) <<  7 |
            _mask(opcode, 7))


def encode_s(opcode, funct3, rs1, rs2, imm):
    """S-type: imm[11:5][31:25] rs2[24:20] rs1[19:15] funct3[14:12] imm[4:0][11:7] opcode[6:0]"""
    imm = _mask(imm, 12)
    imm_hi = (imm >> 5) & 0x7F
    imm_lo = imm & 0x1F
    return (imm_hi << 25 |
            _mask(rs2,    5) << 20 |
            _mask(rs1,    5) << 15 |
            _mask(funct3, 3) << 12 |
            imm_lo            <<  7 |
            _mask(opcode, 7))


def encode_b(opcode, funct3, rs1, rs2, imm):
    """B-type branch encoding (imm is byte offset, must be even)"""
    imm = _mask(imm, 13)
    b12   = (imm >> 12) & 1
    b11   = (imm >> 11) & 1
    b10_5 = (imm >>  5) & 0x3F
    b4_1  = (imm >>  1) & 0xF
    return (b12   << 31 |
            b10_5 << 25 |
            _mask(rs2,    5) << 20 |
            _mask(rs1,    5) << 15 |
            _mask(funct3, 3) << 12 |
            b4_1  <<  8 |
            b11   <<  7 |
            _mask(opcode, 7))


def encode_u(opcode, rd, imm):
    """U-type: imm[31:12] rd[11:7] opcode[6:0]"""
    return (_mask(imm >> 12, 20) << 12 |
            _mask(rd,         5) <<  7 |
            _mask(opcode,     7))


def encode_j(opcode, rd, imm):
    """J-type JAL encoding (imm is byte offset)"""
    imm = _mask(imm, 21)
    b20    = (imm >> 20) & 1
    b19_12 = (imm >> 12) & 0xFF
    b11    = (imm >> 11) & 1
    b10_1  = (imm >>  1) & 0x3FF
    return (b20    << 31 |
            b10_1  << 21 |
            b11    << 20 |
            b19_12 << 12 |
            _mask(rd, 5) << 7 |
            _mask(opcode, 7))


# ---------------------------------------------------------------------------
# High-level instruction constructors
# ---------------------------------------------------------------------------

def NOP():
    """ADDI x0, x0, 0"""
    return encode_i(OP_IMM, 0, F3_ADDI, 0, 0)

def EBREAK():
    """Trigger trap - used to end simulation"""
    return encode_i(OP_SYSTEM, 0, 0, 0, 1)

def LUI(rd, imm):
    return encode_u(OP_LUI, rd, imm)

def AUIPC(rd, imm):
    return encode_u(OP_AUIPC, rd, imm)

def JAL(rd, imm):
    return encode_j(OP_JAL, rd, imm)

def JALR(rd, rs1, imm):
    return encode_i(OP_JALR, rd, F3_ADDI, rs1, imm)

def BEQ(rs1, rs2, imm):
    return encode_b(OP_BRANCH, F3_BEQ, rs1, rs2, imm)

def BNE(rs1, rs2, imm):
    return encode_b(OP_BRANCH, F3_BNE, rs1, rs2, imm)

def BLT(rs1, rs2, imm):
    return encode_b(OP_BRANCH, F3_BLT, rs1, rs2, imm)

def BGE(rs1, rs2, imm):
    return encode_b(OP_BRANCH, F3_BGE, rs1, rs2, imm)

def LW(rd, rs1, imm):
    return encode_i(OP_LOAD, rd, F3_LW, rs1, imm)

def LB(rd, rs1, imm):
    return encode_i(OP_LOAD, rd, F3_LB, rs1, imm)

def LH(rd, rs1, imm):
    return encode_i(OP_LOAD, rd, F3_LH, rs1, imm)

def SW(rs1, rs2, imm):
    return encode_s(OP_STORE, F3_SW, rs1, rs2, imm)

def SB(rs1, rs2, imm):
    return encode_s(OP_STORE, F3_SB, rs1, rs2, imm)

def ADDI(rd, rs1, imm):
    return encode_i(OP_IMM, rd, F3_ADDI, rs1, imm)

def ANDI(rd, rs1, imm):
    return encode_i(OP_IMM, rd, F3_ANDI, rs1, imm)

def ORI(rd, rs1, imm):
    return encode_i(OP_IMM, rd, F3_ORI, rs1, imm)

def XORI(rd, rs1, imm):
    return encode_i(OP_IMM, rd, F3_XORI, rs1, imm)

def SLTI(rd, rs1, imm):
    return encode_i(OP_IMM, rd, F3_SLTI, rs1, imm)

def ADD(rd, rs1, rs2):
    return encode_r(OP_REG, rd, F3_ADD, rs1, rs2, F7_ADD)

def SUB(rd, rs1, rs2):
    return encode_r(OP_REG, rd, F3_SUB, rs1, rs2, F7_SUB)

def AND(rd, rs1, rs2):
    return encode_r(OP_REG, rd, F3_AND, rs1, rs2, F7_AND)

def OR(rd, rs1, rs2):
    return encode_r(OP_REG, rd, F3_OR, rs1, rs2, F7_OR)

def XOR(rd, rs1, rs2):
    return encode_r(OP_REG, rd, F3_XOR, rs1, rs2, F7_XOR)

def SLT(rd, rs1, rs2):
    return encode_r(OP_REG, rd, F3_SLT, rs1, rs2, F7_SLT)


def to_hex(instr):
    """Convert 32-bit instruction to 8-char hex string."""
    return f"{instr & 0xFFFFFFFF:08x}"


def write_hex_file(instructions, filepath):
    """
    Write list of 32-bit instructions to a hex file for $readmemh.

    CRITICAL: The memory model uses a BYTE array (logic [7:0] mem[]).
    $readmemh loads one value per line into successive array elements.
    Therefore each instruction must be written as 4 SEPARATE LINES,
    one byte per line, in LITTLE-ENDIAN order (LSB first).

    PicoRV32 is little-endian: when it reads address A it assembles:
      word = mem[A+3]<<24 | mem[A+2]<<16 | mem[A+1]<<8 | mem[A+0]

    So for instruction 0x00100093 (ADDI x1, x0, 1):
      line 0: 93   <- mem[0] = byte 0 (LSB)
      line 1: 00   <- mem[1] = byte 1
      line 2: 10   <- mem[2] = byte 2
      line 3: 00   <- mem[3] = byte 3 (MSB)
      assembled: 0x00 << 24 | 0x10 << 16 | 0x00 << 8 | 0x93 = 0x00100093 ✓

    Writing the full word on one line (e.g. "00100093") would cause
    $readmemh to load only 0x93 into mem[0] and discard the rest,
    producing garbage in memory.
    """
    with open(filepath, 'w') as f:
        for instr in instructions:
            word = instr & 0xFFFFFFFF
            f.write(f"{(word >>  0) & 0xFF:02x}\n")  # byte 0 (LSB)
            f.write(f"{(word >>  8) & 0xFF:02x}\n")  # byte 1
            f.write(f"{(word >> 16) & 0xFF:02x}\n")  # byte 2
            f.write(f"{(word >> 24) & 0xFF:02x}\n")  # byte 3 (MSB)
