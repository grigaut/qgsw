#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd $SRCDIR

# SurfML

./scripts/oar/run_var_analysis_surfml_perturbed.sh --config=config/variational_analysis.toml -v --obs-track --gamma=0.1 "$@"

./scripts/oar/run_var_analysis_surfml_perturbed.sh --config=config/variational_analysis.toml -v --obs-track --gamma=0.1 --no-alpha "$@"

./scripts/oar/run_var_analysis_surfml_perturbed.sh --config=config/variational_analysis.toml -v --obs-track --no-reg "$@"

./scripts/oar/run_var_analysis_surfml_perturbed.sh --config=config/variational_analysis.toml -v --obs-track --no-reg --no-alpha "$@"

# Forced

./scripts/oar/run_var_analysis_forced_rg_dr_perturbed.sh --config=config/variational_analysis.toml -v --obs-track --gamma=10000 "$@"

./scripts/oar/run_var_analysis_forced_rg_dr_perturbed.sh --config=config/variational_analysis.toml -v --obs-track --no-reg "$@"