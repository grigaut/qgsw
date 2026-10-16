#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd $SRCDIR

# RGSI

./scripts/oar/run_va_qg3l.sh --config=config/variational_analysis.toml -v --presets=presets/rgsi.json "$@"

# RGF

./scripts/oar/run_va_qg3l.sh --config=config/variational_analysis.toml -v --presets=presets/rgf.json "$@"

# RGFSI

./scripts/oar/run_va_qg3l.sh --config=config/variational_analysis.toml -v --presets=presets/rgfsi.json "$@"