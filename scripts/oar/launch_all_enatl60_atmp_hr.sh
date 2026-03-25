#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd $SRCDIR

# SurfML

./scripts/oar/run_var_analysis_enatl60_atmp_hr.sh --config=config/enatl60.toml -v --obs-track --gamma=0.1 "$@"

./scripts/oar/run_var_analysis_enatl60_atmp_hr.sh --config=config/enatl60.toml -v --obs-track --gamma=0.1 --no-alpha "$@"

./scripts/oar/run_var_analysis_enatl60_atmp_hr.sh --config=config/enatl60.toml -v --obs-track --no-reg "$@"

./scripts/oar/run_var_analysis_enatl60_atmp_hr.sh --config=config/enatl60.toml -v --obs-track --no-reg --no-alpha "$@"

# Forced

./scripts/oar/run_var_analysis_enatl60_forced_atmp_hr.sh --config=config/enatl60.toml -v --obs-track --gamma=1e7 "$@"

./scripts/oar/run_var_analysis_enatl60_forced_atmp_hr.sh --config=config/enatl60.toml -v --obs-track --no-reg "$@"