#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="scripts/bash/run_va_enatl60_forced_atmp_hr.sh"
NAME="eNATL60-AtmP-Forced-HR"
source "$SRCDIR/scripts/oar/lib.sh"

cd $SRCDIR
chmod +x $SCRIPT

parse_common_flags "$@"
load_env "$SRCDIR"

# Set walltime based on --long and --contiguous flags
if [ "$long" = true ]; then
    walltime=80
else
    walltime=20
fi
build_oar_opts_hr "$walltime"

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
cmd="${cmd}${optim_args}"

oarsub "${OAR_OPTS[@]}" -n "${NAME}" "$cmd"

exit 0