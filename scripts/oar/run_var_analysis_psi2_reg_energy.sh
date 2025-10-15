#!/bin/bash


SRCDIR=$HOME/qgsw

cd $SRCDIR

chmod +x scripts/bash/run_va_psi2_reg_energy.sh

cmd="./scripts/bash/run_va_psi2_reg_energy.sh"
for arg in "$@"; do
    cmd+=" \"$arg\""
done
cmd1="${cmd} --indices 32 96 64 192"
cmd2="${cmd} --indices 32 96 256 384"
cmd3="${cmd} --indices 112 176 64 192"
cmd4="${cmd} --indices 112 176 256 384"
oarsub -S "$cmd1" -n "VA-psi2-reg-energy-[32 96 64 192]"
oarsub -S "$cmd2" -n "VA-psi2-reg-energy-[32 96 256 384]"
oarsub -S "$cmd3" -n "VA-psi2-reg-energy-[112 176 64 192]"
oarsub -S "$cmd4" -n "VA-psi2-reg-energy-[112 176 256 384]"

exit 1