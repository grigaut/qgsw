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
        -p "host='abacus11'"
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
    args=()
    for arg in "$@"; do
        case "$arg" in
            --contiguous) contiguous=true ;;
            --long)       long=true ;;
            *)            args+=("$arg") ;;
        esac
    done
}