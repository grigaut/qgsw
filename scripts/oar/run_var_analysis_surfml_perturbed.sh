#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="scripts/bash/run_va_surfml_perturbed.sh"
NAME="SurfML-perturbed"
source "$SRCDIR/scripts/oar/lib.sh"

cd $SRCDIR
chmod +x $SCRIPT

parse_common_flags "$@"
load_env "$SRCDIR"

# Count active zones
n_zones=0
[ "$z1" = true ] && (( n_zones++ ))
[ "$z2" = true ] && (( n_zones++ ))
[ "$z3" = true ] && (( n_zones++ ))
[ "$z4" = true ] && (( n_zones++ ))

# Compute walltime
walltime=4
[ "$long_optim" = true ] && (( walltime *= 4 ))
[ "$long" = true ]       && (( walltime *= 4 ))
[ "$contiguous" = true ] && (( walltime *= n_zones ))

build_oar_opts "$walltime"

# Build base command with filtered arguments
cmd="./$SCRIPT"
for arg in "${args[@]}"; do
    cmd+=" $arg"
done

# Append extra python args based on flags
optim_args=""
if [ "$long_optim" = true ]; then
    optim_args+=" -o 800"
fi
if [ "$long" = true ]; then
    optim_args+=" -c 12"
fi

# Build the four command variants
cmd1="${cmd}${optim_args} --indices 32 96 64 192"
cmd2="${cmd}${optim_args} --indices 32 96 256 384"
cmd3="${cmd}${optim_args} --indices 112 176 64 192"
cmd4="${cmd}${optim_args} --indices 112 176 256 384"

if [ "$contiguous" = true ]; then
    combined_cmd=""
    [ "$z1" = true ] && combined_cmd+="${cmd1} ; "
    [ "$z2" = true ] && combined_cmd+="${cmd2} ; "
    [ "$z3" = true ] && combined_cmd+="${cmd3} ; "
    [ "$z4" = true ] && combined_cmd+="${cmd4} ; "
    # Strip trailing " ; "
    combined_cmd="${combined_cmd% ; }"
    oarsub "${OAR_OPTS[@]}" -n "${NAME}-contiguous" "$combined_cmd"
else
    [ "$z1" = true ] && oarsub "${OAR_OPTS[@]}" -n "${NAME}-[32 96 64 192]"    "$cmd1"
    [ "$z2" = true ] && oarsub "${OAR_OPTS[@]}" -n "${NAME}-[32 96 256 384]"   "$cmd2"
    [ "$z3" = true ] && oarsub "${OAR_OPTS[@]}" -n "${NAME}-[112 176 64 192]"  "$cmd3"
    [ "$z4" = true ] && oarsub "${OAR_OPTS[@]}" -n "${NAME}-[112 176 256 384]" "$cmd4"
fi

exit 0