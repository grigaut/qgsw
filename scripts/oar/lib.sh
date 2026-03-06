# scripts/bash/lib.sh

load_env() {
    local srcdir=$1
    if [ -f "$srcdir/.env" ]; then
        set -a
        source "$srcdir/.env"
        set +a
    fi
}
oarnodes --sql "production='YES' and state='Alive'" -J | python3 -c "
import json, sys
nodes = json.load(sys.stdin)
gpu_nodes = [info for info in nodes.values() if info.get('gpu') is not None]
sorted_nodes = sorted(gpu_nodes, key=lambda x: x.get('memnode') or 0, reverse=True)
for info in sorted_nodes[:10]:
    print(f\"{info.get('host', 'N/A')}: {info.get('memnode', 'N/A')} MB RAM, gpu={info.get('gpu')}, gpu_model={info.get('gpu_model', 'N/A')}\")
"
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
        -p "host='abacus1' OR host='abacus11'"
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