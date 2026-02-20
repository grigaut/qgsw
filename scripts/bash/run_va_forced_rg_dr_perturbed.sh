#!/bin/bash

#OAR -q production
#OAR -l gpu=1,walltime=6
#OAR -O logs/OAR.%jobid%.stdout
#OAR -E logs/OAR.%jobid%.stderr
#OAR --notify mail:gaetan.rigaut@inria.fr

# To run with arguments use quotes: oarsub -S "./run_va_psi2.sh --config=config/variational_analysis.toml -vv"

lscpu | grep 'Model name' | cut -f 2 -d ":" | awk '{$1=$1}1'

echo JOB ID : $OAR_JOB_ID

SRCDIR=$HOME/qgsw

cd $SRCDIR

date

LD_PRELOAD=./.venv/lib/libstdc++.so.6 .venv/bin/python3 -u scripts/variational_analysis_forced_rg_dr_perturbed.py $@

date

exit 0