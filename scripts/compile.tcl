# =============================================================================
# compile.tcl
# QuestaSim compilation script for PicoRV32 ML-DV
#
# Usage (from QuestaSim transcript, after cd to project root):
#   do scripts/compile.tcl
# =============================================================================

puts "============================================================"
puts "  PicoRV32 ML-DV: Compilation"
puts "============================================================"

if {[info exists env(PICORV32_WORK_LIB)] && $env(PICORV32_WORK_LIB) ne ""} {
    set work_lib $env(PICORV32_WORK_LIB)
} else {
    set work_lib "work"
}

# ---------------------------------------------------------------------------
# 1. Create simulation library
# ---------------------------------------------------------------------------
puts "\n\[Step 1\] Setting up simulation library '$work_lib'..."

if {[file exists $work_lib/_info]} {
    if {$work_lib eq "work"} {
        puts "  Backing up work/ JSON and hex files..."
        set preserve_files {}
        foreach pattern [list work/*.json work/*.hex work/*.log] {
            if {[catch {glob -nocomplain -- $pattern} matches]} {
                set matches {}
            }
            set preserve_files [concat $preserve_files $matches]
        }

        file mkdir work_backup
        foreach f $preserve_files {
            set fname [file tail $f]
            catch {file copy -force $f work_backup/$fname}
        }
        puts "  Backed up [llength $preserve_files] file(s) to work_backup/"
    }

    puts "  Removing existing simulation library..."
    vdel -lib $work_lib -all
}

vlib $work_lib
vmap $work_lib $work_lib

if {$work_lib eq "work" && [file exists work_backup]} {
    if {[catch {glob -nocomplain -- work_backup/*} restore_files]} {
        set restore_files {}
    }
    foreach f $restore_files {
        set fname [file tail $f]
        catch {file copy -force $f work/$fname}
    }
    puts "  Restored [llength $restore_files] file(s) back to work/"
    catch {file delete -force work_backup}
}

puts "  Simulation library ready."

# ---------------------------------------------------------------------------
# 2. Compile DPI-C shared library
# ---------------------------------------------------------------------------
puts "\n\[Step 2\] Compiling DPI-C bridge..."

if {[info exists env(MODELSIM)]} {
    set questa_inc [file join $env(MODELSIM) .. include]
} else {
    set questa_bin [file dirname [info nameofexecutable]]
    set questa_inc [file join $questa_bin .. include]
}

puts "  QuestaSim include: $questa_inc"

if {$tcl_platform(os) eq "Windows NT"} {
    set lib_name "dpi/knobio"
    set gcc_cmd "gcc -shared -o dpi/knobio.dll dpi/knob_io.c -I\"$questa_inc\" -Idpi"
} else {
    set lib_name "dpi/knobio"
    set gcc_cmd "gcc -shared -fPIC -o dpi/knobio.so dpi/knob_io.c -I\"$questa_inc\" -Idpi"
}

puts "  Running: $gcc_cmd"
set result [catch {eval exec $gcc_cmd} output]

if {$result != 0} {
    puts "  WARNING: gcc compile failed: $output"
    puts "  Trying alternative: letting vlog handle DPI compilation..."
    set use_vlog_dpi 1
} else {
    puts "  DPI library compiled successfully."
    set use_vlog_dpi 0
}

# ---------------------------------------------------------------------------
# 3. Compile coverage package
# ---------------------------------------------------------------------------
puts "\n\[Step 3\] Compiling coverage package..."
if {[catch {vlog -sv -work $work_lib rtl/coverage_pkg.sv} errmsg]} {
    puts "ERROR: coverage_pkg.sv failed: $errmsg"
    return
}
puts "  coverage_pkg.sv OK"

# ---------------------------------------------------------------------------
# 4. Compile PicoRV32 RTL
# ---------------------------------------------------------------------------
puts "\n\[Step 4\] Compiling PicoRV32..."
if {[file exists picorv32.v]} {
    set picorv32_path "picorv32.v"
} elseif {[file exists picorv32/picorv32.v]} {
    set picorv32_path "picorv32/picorv32.v"
} else {
    puts "ERROR: picorv32.v not found in project root!"
    puts "Download it with:"
    puts "  Invoke-WebRequest -Uri https://raw.githubusercontent.com/YosysHQ/picorv32/main/picorv32.v -OutFile picorv32.v"
    return
}

set picorv32_vlog_cmd [list vlog -work $work_lib]
set active_bug_defines {}
if {[info exists env(PICORV32_BUG_DEFINE)] && $env(PICORV32_BUG_DEFINE) ne ""} {
    foreach define_name [split $env(PICORV32_BUG_DEFINE) ",; "] {
        if {$define_name eq ""} {
            continue
        }
        lappend picorv32_vlog_cmd "+define+$define_name"
        lappend active_bug_defines $define_name
    }
}
if {[llength $active_bug_defines] > 0} {
    puts "  Fault injection defines: [join $active_bug_defines {, }]"
} else {
    puts "  Fault injection defines: none"
}
set bug_define_args {}
foreach define_name $active_bug_defines {
    lappend bug_define_args "+define+$define_name"
}
lappend picorv32_vlog_cmd $picorv32_path

if {[catch {eval $picorv32_vlog_cmd} errmsg]} {
    puts "ERROR: picorv32.v failed: $errmsg"
    return
}
puts "  $picorv32_path OK"

# ---------------------------------------------------------------------------
# 5. Compile memory model
# ---------------------------------------------------------------------------
puts "\n\[Step 5\] Compiling memory model..."
set memory_model_vlog_cmd [list vlog -sv -work $work_lib]
foreach define_arg $bug_define_args {
    lappend memory_model_vlog_cmd $define_arg
}
lappend memory_model_vlog_cmd rtl/memory_model.sv
if {[catch {eval $memory_model_vlog_cmd} errmsg]} {
    puts "ERROR: memory_model.sv failed: $errmsg"
    return
}
puts "  memory_model.sv OK"

# ---------------------------------------------------------------------------
# 6. Compile testbench
# ---------------------------------------------------------------------------
puts "\n\[Step 6\] Compiling testbench..."
set tb_top_vlog_cmd [list vlog -sv -work $work_lib]
foreach define_arg $bug_define_args {
    lappend tb_top_vlog_cmd $define_arg
}
lappend tb_top_vlog_cmd rtl/tb_top.sv
if {[catch {eval $tb_top_vlog_cmd} errmsg]} {
    puts "ERROR: tb_top.sv failed: $errmsg"
    return
}
puts "  tb_top.sv OK"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
puts ""
puts "============================================================"
puts "  Compilation COMPLETE"
puts "  Simulation library: $work_lib"
set compile_meta_path [file join $work_lib compile_meta.txt]
set compile_meta_fp [open $compile_meta_path "w"]
puts $compile_meta_fp "work_lib=$work_lib"
puts $compile_meta_fp "bug_define=[join $active_bug_defines {,}]"
close $compile_meta_fp
puts ""
puts "  Next step - run one simulation manually:"
puts "    set env(SIM_ID) 0"
puts "    do scripts/simulate.tcl"
puts "============================================================"
