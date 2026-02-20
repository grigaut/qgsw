#!/bin/bash
SRCDIR=$HOME/qgsw
cd $SRCDIR
chmod +x scripts/bash/run_va_enatl60_forced.sh

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
cmd="./scripts/bash/run_va_enatl60_forced.sh"
for arg in "${args[@]}"; do
    cmd+=" $arg"
done

oarsub -S "$cmd" -n "VA-eNATL60"

exit 0