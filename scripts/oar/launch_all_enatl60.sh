#!/bin/bash
SRCDIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd $SRCDIR

# RGSI

./scripts/oar/run_va_enatl60.sh --config=config/enatl60.toml -v --presets=presets/eNATL60/rgsi.json "$@"

# RGF

./scripts/oar/run_va_enatl60.sh --config=config/enatl60.toml -v --presets=presets/eNATL60/rgf.json "$@"

# RGF-VD

./scripts/oar/run_va_enatl60.sh --config=config/enatl60.toml -v --presets=presets/eNATL60/rgf.json --basis-ext=basis-miost --basis-ext-lr=1e-13 --gamma-reg-bkg=1e18 "$@"

# RGFSI

./scripts/oar/run_va_enatl60.sh --config=config/enatl60.toml -v --presets=presets/eNATL60/rgfsi.json "$@"