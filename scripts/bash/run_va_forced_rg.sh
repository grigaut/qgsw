#!/bin/bash

#OAR -q production
#OAR -l gpu=1,walltime=6
###OAR --property cputype = 'Intel Xeon Silver 4214' OR cputype = 'Intel Xeon Gold 6248' OR cputype = 'Intel Xeon Silver 4114'
#OAR -O logs/OAR.%jobid%.stdout
#OAR -E logs/OAR.%jobid%.stderr
#OAR --notify mail:gaetan.rigaut@inria.fr

# To run with arguments use quotes: oarsub -S "./run_va_psi2.sh --config=config/variational_analysis.toml -vv"

lscpu | grep 'Model name' | cut -f 2 -d ":" | awk '{$1=$1}1'

echo JOB ID : $OAR_JOB_ID

SRCDIR=$HOME/qgsw

cd $SRCDIR

date

.venv/bin/python3 -u scripts/variational_analysis_forced_rg.py $@

date

exit 1