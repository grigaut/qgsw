#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd $SRCDIR

# RGSI

./scripts/oar/run_va_enatl60.sh --config=config/enatl60.toml -v --presets=presets/eNATL60/rgsi.json --HR "$@"

# RGSI

./scripts/oar/run_va_enatl60.sh --config=config/enatl60.toml -v --presets=presets/eNATL60/rgsi_vd.json --HR "$@"

# RGF-VD

./scripts/oar/run_va_enatl60.sh --config=config/enatl60.toml -v --presets=presets/eNATL60/rgf_vd.json --HR "$@"

# RG0

./scripts/oar/run_va_enatl60.sh --config=config/enatl60.toml -v --presets=presets/eNATL60/rg0.json --HR "$@"