#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd $SRCDIR

# SurfML

./scripts/oar/run_var_analysis_enatl60_atmp.sh --config=config/variational_analysis.toml -v --obs-track --gamma=0.0001 "$@"

./scripts/oar/run_var_analysis_enatl60_atmp.sh --config=config/variational_analysis.toml -v --obs-track --gamma=0.0001 --no-alpha "$@"

./scripts/oar/run_var_analysis_enatl60_atmp.sh --config=config/variational_analysis.toml -v --obs-track --no-reg "$@"

./scripts/oar/run_var_analysis_enatl60_atmp.sh --config=config/variational_analysis.toml -v --obs-track --no-reg --no-alpha "$@"

# Forced

./scripts/oar/run_var_analysis_enatl60_forced_atmp.sh --config=config/variational_analysis.toml -v --obs-track --gamma=10 "$@"

./scripts/oar/run_var_analysis_enatl60_forced_atmp.sh --config=config/variational_analysis.toml -v --obs-track --no-reg "$@"

# Forced + SurfML

./scripts/oar/run_var_analysis_enatl60_forcedsurf_atmp.sh --config=config/variational_analysis.toml -v --obs-track --gamma=1 "$@"

./scripts/oar/run_var_analysis_enatl60_forcedsurf_atmp.sh --config=config/variational_analysis.toml -v --obs-track --no-reg "$@"