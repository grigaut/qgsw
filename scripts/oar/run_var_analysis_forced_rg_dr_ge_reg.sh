#!/bin/bash
SRCDIR=$HOME/qgsw
cd $SRCDIR
chmod +x scripts/bash/run_va_forced_rg_dr_ge_reg.sh

# Check for --contiguous flag
contiguous=false
args=()

for arg in "$@"; do
    if [ "$arg" == "--contiguous" ]; then
        contiguous=true
    else
        args+=("$arg")
    fi
done

# Build base command with filtered arguments
cmd="./scripts/bash/run_va_forced_rg_dr_ge_reg.sh"
for arg in "${args[@]}"; do
    cmd+=" $arg"
done

# Build the four command variants
cmd1="${cmd} --indices 32 96 64 192"
cmd2="${cmd} --indices 32 96 256 384"
cmd3="${cmd} --indices 112 176 64 192"
cmd4="${cmd} --indices 112 176 256 384"

if [ "$contiguous" = true ]; then
    # Run commands sequentially in a single oarsub
    combined_cmd="$cmd1 ; $cmd2 ; $cmd3 ; $cmd4"
    oarsub -S "$combined_cmd" -n "VA-mxs-contiguous"
else
    # Run commands as separate jobs
    oarsub -S "$cmd1" -n "VA-mxs-[32 96 64 192]"
    oarsub -S "$cmd2" -n "VA-mxs-[32 96 256 384]"
    oarsub -S "$cmd3" -n "VA-mxs-[112 176 64 192]"
    oarsub -S "$cmd4" -n "VA-mxs-[112 176 256 384]"
fi

exit 0