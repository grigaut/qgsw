#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="scripts/bash/run_va_enatl60_forced_atmp_hr.sh"
NAME="eNATL60-AtmP-Forced-HR"
source "$SRCDIR/scripts/oar/lib.sh"

cd $SRCDIR
chmod +x $SCRIPT

parse_enatl60_flags "$@"
load_env "$SRCDIR"

# Count active seasons
n_seasons=0
[ "$summer" = true ] && (( n_seasons++ ))
[ "$autumn" = true ] && (( n_seasons++ ))
[ "$winter" = true ] && (( n_seasons++ ))
[ "$spring" = true ] && (( n_seasons++ ))

# Compute walltime
walltime=20
[ "$long_optim" = true ] && (( walltime *= 4 ))
[ "$contiguous" = true ] && (( walltime *= n_seasons ))

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

# Build the four command variants
cmd1="${cmd}${optim_args} --season=summer"
cmd2="${cmd}${optim_args} --season=autumn"
cmd3="${cmd}${optim_args} --season=winter"
cmd4="${cmd}${optim_args} --season=spring"

if [ "$contiguous" = true ]; then
    combined_cmd=""
    [ "$summer" = true ] && combined_cmd+="${cmd1} ; "
    [ "$autumn" = true ] && combined_cmd+="${cmd2} ; "
    [ "$winter" = true ] && combined_cmd+="${cmd3} ; "
    [ "$spring" = true ] && combined_cmd+="${cmd4} ; "
    # Strip trailing " ; "
    combined_cmd="${combined_cmd% ; }"
    echo "${OAR_OPTS[@]}" -n "${NAME}-contiguous" "$combined_cmd"
else
    [ "$summer" = true ] && echo "${OAR_OPTS[@]}" -n "${NAME}-Summer"    "$cmd1"
    [ "$autumn" = true ] && echo "${OAR_OPTS[@]}" -n "${NAME}-Autumn"   "$cmd2"
    [ "$winter" = true ] && echo "${OAR_OPTS[@]}" -n "${NAME}-Winter"  "$cmd3"
    [ "$spring" = true ] && echo "${OAR_OPTS[@]}" -n "${NAME}-Spring" "$cmd4"
fi

exit 0