#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd $SRCDIR

# RGSI

./scripts/oar/run_var_analysis_enatl60_atmp_hr.sh --config=config/enatl60.toml -v --obs-track --gamma=0.01 "$@"

# RGSI SST

./scripts/oar/run_var_analysis_enatl60_sst_hr.sh --config=config/enatl60.toml -v --obs-track --gamma=0.01 --gamma-sst=0.01 "$@"

# RGSI SST Adv

./scripts/oar/run_var_analysis_enatl60_sst_adv_hr.sh --config=config/enatl60.toml -v --obs-track --gamma=0.01 --gamma-sst=0.01 "$@"

# Forced

./scripts/oar/run_var_analysis_enatl60_forced_atmp_hr.sh --config=config/enatl60.toml -v --obs-track --gamma=1e3 "$@"