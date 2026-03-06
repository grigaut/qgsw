#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="scripts/bash/run_va_surfml_perturbed.sh"
NAME="SurfML-perturbed"
source "$SRCDIR/scripts/oar/lib.sh"

cd $SRCDIR
chmod +x $SCRIPT

parse_common_flags "$@"
load_env "$SRCDIR"

# Set walltime based on --long and --contiguous flags
if [ "$long" = true ] && [ "$contiguous" = true ]; then
    walltime=60
elif [ "$contiguous" = true ]; then
    walltime=16
elif [ "$long" = true ]; then
    walltime=16
else
    walltime=4
fi
build_oar_opts "$walltime"

# Build base command with filtered arguments
cmd="./$SCRIPT"
for arg in "${args[@]}"; do
    cmd+=" $arg"
done

# Append extra python args based on flags
optim_args=""
if [ "$long" = true ]; then
    optim_args+=" -o 800"
fi

# Build the four command variants
cmd1="${cmd}${optim_args} --indices 32 96 64 192"
cmd2="${cmd}${optim_args} --indices 32 96 256 384"
cmd3="${cmd}${optim_args} --indices 112 176 64 192"
cmd4="${cmd}${optim_args} --indices 112 176 256 384"

if [ "$contiguous" = true ]; then
    combined_cmd="$cmd1 ; $cmd2 ; $cmd3 ; $cmd4"
    oarsub "${OAR_OPTS[@]}" -n "${NAME}-contiguous" "$combined_cmd"
else
    oarsub "${OAR_OPTS[@]}" -n "${NAME}-[32 96 64 192]"    "$cmd1"
    oarsub "${OAR_OPTS[@]}" -n "${NAME}-[32 96 256 384]"   "$cmd2"
    oarsub "${OAR_OPTS[@]}" -n "${NAME}-[112 176 64 192]"  "$cmd3"
    oarsub "${OAR_OPTS[@]}" -n "${NAME}-[112 176 256 384]" "$cmd4"
fi

exit 0