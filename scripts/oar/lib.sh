# scripts/bash/lib.sh

load_env() {
    local srcdir=$1
    if [ -f "$srcdir/.env" ]; then
        set -a
        source "$srcdir/.env"
        set +a
    fi
}

build_oar_opts() {
    local walltime=$1
    OAR_OPTS=(
        -q production
        -l "gpu=1,walltime=${walltime}"
        -O logs/OAR.%jobid%.stdout
        -E logs/OAR.%jobid%.stderr
    )
    if [ -n "$NOTIFY_EMAIL" ]; then
        OAR_OPTS+=(--notify "mail:${NOTIFY_EMAIL}")
    fi
}

build_oar_opts_hr() {
    local walltime=$1
    OAR_OPTS=(
        -p "cluster='abacus11' OR cluster='abacus1'"
        -q production
        -l "gpu=1,walltime=${walltime}"
        -O logs/OAR.%jobid%.stdout
        -E logs/OAR.%jobid%.stderr
    )
    if [ -n "$NOTIFY_EMAIL" ]; then
        OAR_OPTS+=(--notify "mail:${NOTIFY_EMAIL}")
    fi
}

parse_common_flags() {
    contiguous=false
    long=false
    long_optim=false
    any_zone=false
    z1=false
    z2=false
    z3=false
    z4=false
    args=()
    for arg in "$@"; do
        case "$arg" in
            --contiguous)  contiguous=true ;;
            --long)        long=true ;;
            --long-optim)  long_optim=true ;;
            --z1)          z1=true; any_zone=true ;;
            --z2)          z2=true; any_zone=true ;;
            --z3)          z3=true; any_zone=true ;;
            --z4)          z4=true; any_zone=true ;;
            *)             args+=("$arg") ;;
        esac
    done
    # If no zone specified, enable all
    if [ "$any_zone" = false ]; then
        z1=true; z2=true; z3=true; z4=true
    fi
}