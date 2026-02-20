#!/bin/bash

#OAR -n run-model
#OAR -q production
#OAR -l gpu=1,walltime=24
#OAR -O logs/OAR.%jobid%.stdout
#OAR -E logs/OAR.%jobid%.stderr
#OAR --notify mail:gaetan.rigaut@inria.fr

# To run with arguments use quotes: oarsub -S "./run.sh --config=config/run.toml -vv"

lscpu | grep 'Model name' | cut -f 2 -d ":" | awk '{$1=$1}1'

echo JOB ID : $OAR_JOB_ID

SRCDIR=$HOME/qgsw

cd $SRCDIR

date

LD_PRELOAD=./.venv/lib/libstdc++.so.6 .venv/bin/python3 -u scripts/run.py $@

date

exit 0
