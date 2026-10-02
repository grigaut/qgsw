#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
SCRIPT="scripts/bash/run_enatl60_diagnostics.sh"
NAME="eNATL60-Diags"
source "$SRCDIR/scripts/oar/lib.sh"

cd $SRCDIR
chmod +x $SCRIPT

load_env "$SRCDIR"
# Compute walltime
walltime=6

OAR_OPTS=(
    -q production
    -l "gpu=1,walltime=${walltime}"
    -O logs/OAR.%jobid%.stdout
    -E logs/OAR.%jobid%.stderr
)
if [ -n "$NOTIFY_EMAIL" ]; then
    OAR_OPTS+=(--notify "mail:${NOTIFY_EMAIL}")
fi

# Build base command with filtered arguments
cmd="./$SCRIPT"
for arg in "${args[@]}"; do
    cmd+=" $arg"
done


oarsub "${OAR_OPTS[@]}" -n "${NAME}" "$cmd"
# Append extra python args based on flags
exit 0