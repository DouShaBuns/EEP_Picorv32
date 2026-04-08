# EEP_Picorv32
This is the source code for the thesis titled "A Machine Learning Approach to Hardware Verification: From Random to Reasoned"
For the command flags, refer to the appendix section of the paper. QuestaSim is required.
If it doesnt run on your local machine, there might be pathing issues or environment variable is not being set up correctly

Instruction:
1. Clone the repo and cd ./EEP_PicoRV32

2.For coverage runs
Environment setup command:
$env:PICORV32_WORK_LIB="work_cov"
vsim -c -do "do scripts/compile.tcl; quit"

3.run PowerShell command => python run_coverage.py <all_necessary_flags>
Example: python run_coverage.py --agent dqn --init 20 --iters 21 --per-iter 10  --repeat-trials 10
Explanation: it will run the coverage closure mode with dqn agent , 20 sims warm-up, 21 iterations x 10 sims/iteration => 210 sims per trial, 10 trials

4.For fault detection runs:
Environment setujp command:
$env:PICORV32_WORK_LIB="work_fault"
$env:PICORV32_BUG_DEFINE="BUG_CTRL_DELAYED_STORE_BRANCH_DROP_RARE"
vsim -c -do "do scripts/compile.tcl; quit"

5.run PowerShell command => python run_fault.py <all_necessary_flags>
