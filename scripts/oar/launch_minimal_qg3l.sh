#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd $SRCDIR

# SurfML

./scripts/oar/run_var_analysis_surfml.sh --config=config/variational_analysis.toml -v --obs-track --gamma=1e2 "$@"

./scripts/oar/run_var_analysis_surfml.sh --config=config/variational_analysis.toml -v --obs-track --no-reg "$@"

# Forced

./scripts/oar/run_var_analysis_forced_rg_dr.sh --config=config/variational_analysis.toml -v --obs-track --gamma=1e7 "$@"