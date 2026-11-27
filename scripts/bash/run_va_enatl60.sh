#!/bin/bash

#OAR -q production
#OAR -l gpu=1,walltime=6
#OAR -O logs/OAR.%jobid%.stdout
#OAR -E logs/OAR.%jobid%.stderr
#OAR --notify mail:gaetan.rigaut@inria.fr

# To run with arguments use quotes: oarsub -S "./run_va_enatl60.sh --config=config/enatl60.toml -vv --presets=presets/eNATL60/rgsi.json"

lscpu | grep 'Model name' | cut -f 2 -d ":" | awk '{$1=$1}1'

echo JOB ID : $OAR_JOB_ID

SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)

cd $SRCDIR

date

echo Running: \".venv/bin/python3 -u scripts/python/fitting/fit_enatl60.py --run-id=$OAR_JOB_ID $@\"

LD_PRELOAD=./.venv/lib/libstdc++.so.6 .venv/bin/python3 -u scripts/python/fitting/fit_enatl60.py --run-id=$OAR_JOB_ID $@

echo Running validation script

LD_PRELOAD=./.venv/lib/libstdc++.so.6 .venv/bin/python3 -u scripts/python/validation/validate_enatl60.py --run-id=$OAR_JOB_ID

echo Building PSD videos

LD_PRELOAD=./.venv/lib/libstdc++.so.6 .venv/bin/python3 -u scripts/python/analysis/psd_errors_videos_enatl60.py --run-id=$OAR_JOB_ID

echo Building ageostrophy videos

LD_PRELOAD=./.venv/lib/libstdc++.so.6 .venv/bin/python3 -u scripts/python/analysis/ageostrophy_videos_enatl60.py --run-id=$OAR_JOB_ID

echo Building decomposition videos

LD_PRELOAD=./.venv/lib/libstdc++.so.6 .venv/bin/python3 -u scripts/python/analysis/decomposition_videos_enatl60.py --run-id=$OAR_JOB_ID

echo Removing tracking file

LD_PRELOAD=./.venv/lib/libstdc++.so.6 .venv/bin/python3 -u scripts/python/remove_tracking_file.py --run-id=$OAR_JOB_ID

date

exit 0