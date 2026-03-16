#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd $SRCDIR

# SurfML

./scripts/oar/run_var_analysis_surfml_perturbed.sh --config=config/variational_analysis.toml -v --obs-track --gamma=1e2 "$@"

./scripts/oar/run_var_analysis_surfml_perturbed.sh --config=config/variational_analysis.toml -v --obs-track --gamma=1e2 --no-alpha "$@"

./scripts/oar/run_var_analysis_surfml_perturbed.sh --config=config/variational_analysis.toml -v --obs-track --no-reg "$@"

./scripts/oar/run_var_analysis_surfml_perturbed.sh --config=config/variational_analysis.toml -v --obs-track --no-reg --no-alpha "$@"

# Forced

./scripts/oar/run_var_analysis_forced_rg_dr_perturbed.sh --config=config/variational_analysis.toml -v --obs-track --gamma=1e7 "$@"

./scripts/oar/run_var_analysis_forced_rg_dr_perturbed.sh --config=config/variational_analysis.toml -v --obs-track --no-reg "$@"