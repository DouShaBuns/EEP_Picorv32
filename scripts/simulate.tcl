# =============================================================================
# simulate.tcl
# QuestaSim single simulation runner for PicoRV32 ML-DV
#
# Usage (from QuestaSim transcript):
#   set env(SIM_ID) 0
#   do scripts/simulate.tcl
# =============================================================================

if {[info exists env(SIM_ID)]} {
    set sim_id $env(SIM_ID)
} else {
    set sim_id 0
    puts "WARNING: SIM_ID not set, defaulting to 0"
    puts "Set it first with: set env(SIM_ID) <number>"
}

if {[info exists env(PICORV32_WORK_LIB)] && $env(PICORV32_WORK_LIB) ne ""} {
    set work_lib $env(PICORV32_WORK_LIB)
} else {
    set work_lib "work"
}

puts "============================================================"
puts "  Starting simulation: SIM_ID = $sim_id"
puts "  Simulation library: $work_lib"
puts "============================================================"

set knob_file "work/knobs_${sim_id}.json"
set prog_file "work/prog_${sim_id}.hex"

if {![file exists $knob_file]} {
    puts "ERROR: Knob file not found: $knob_file"
    puts "Create it manually or run the ML agent first."
    return
}

if {![file exists $prog_file]} {
    puts "ERROR: Program file not found: $prog_file"
    puts "Run: python programs/gen_program.py"
    return
}

puts "  Knob file: $knob_file  OK"
puts "  Prog file: $prog_file  OK"

if {$tcl_platform(os) eq "Windows NT"} {
    set dpi_lib "dpi/knobio"
} else {
    set dpi_lib "dpi/knobio"
}

if {[file exists "${dpi_lib}.dll"] || [file exists "${dpi_lib}.so"]} {
    puts "  Loading DPI library: $dpi_lib"
    set vsim_cmd "vsim -sv_lib $dpi_lib -lib $work_lib ${work_lib}.tb_top -suppress 3009 -quiet"
} else {
    puts "  WARNING: DPI library not found at $dpi_lib"
    puts "  Simulation will use default knobs (no DPI bridge)"
    set vsim_cmd "vsim -lib $work_lib ${work_lib}.tb_top -suppress 3009 -quiet"
}

if {[catch {eval $vsim_cmd} errmsg]} {
    puts "ERROR starting simulation: $errmsg"
    return
}

puts "  Running..."
run -all

puts ""
puts "============================================================"
puts "  Simulation $sim_id complete."
puts "  Check results: work/coverage_${sim_id}.json"
puts "============================================================"

quit -sim
