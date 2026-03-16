#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd $SRCDIR

# SurfML

./scripts/oar/run_var_analysis_enatl60_atmp.sh --config=config/variational_analysis.toml -v --obs-track --gamma=0.1 "$@"

./scripts/oar/run_var_analysis_enatl60_atmp.sh --config=config/variational_analysis.toml -v --obs-track --gamma=0.1 --no-alpha "$@"

./scripts/oar/run_var_analysis_enatl60_atmp.sh --config=config/variational_analysis.toml -v --obs-track --no-reg "$@"

./scripts/oar/run_var_analysis_enatl60_atmp.sh --config=config/variational_analysis.toml -v --obs-track --no-reg --no-alpha "$@"

# Forced

./scripts/oar/run_var_analysis_enatl60_forced_atmp.sh --config=config/variational_analysis.toml -v --obs-track --gamma=1e4 "$@"

./scripts/oar/run_var_analysis_enatl60_forced_atmp.sh --config=config/variational_analysis.toml -v --obs-track --no-reg "$@"

# Forced + SurfML

./scripts/oar/run_var_analysis_enatl60_forcedsurf_atmp.sh --config=config/variational_analysis.toml -v --obs-track --gamma=1 "$@"

./scripts/oar/run_var_analysis_enatl60_forcedsurf_atmp.sh --config=config/variational_analysis.toml -v --obs-track --no-reg "$@"