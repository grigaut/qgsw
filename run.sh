#!/bin/bash

#OAR -n run-model
#OAR -q production
#OAR -l gpu=1,walltime=24
###OAR --property cputype = 'Intel Xeon Silver 4214' OR cputype = 'Intel Xeon Gold 6248' OR cputype = 'Intel Xeon Silver 4114'
#OAR -O logs/OAR.%jobid%.stdout
#OAR -E logs/OAR.%jobid%.stderr

# To run with arguments use quotes: "oarsub -S "./run_model.py --config=config.toml"

lscpu | grep 'Model name' | cut -f 2 -d ":" | awk '{$1=$1}1'

echo JOB ID : $OAR_JOB_ID

SRCDIR=$HOME/qgsw

cd $SRCDIR

date

.venv/bin/python3 scripts/run.py $@

date

exit 1
