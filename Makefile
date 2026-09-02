ifneq (,$(wildcard ./.env))
	include .env
    export
endif

# Virtual Environment Management
ifeq ($(OS), Windows_NT)
	BIN = ${VENV}/Scripts/
else
	BIN = ${VENV}/bin/
endif

ENV_LOCAL := environment.yml
ENV_G5000 := environment.yml
ifeq (${ENVIRONMENT}, local)
	ENVIRONMENT_FILE = ${ENV_LOCAL}
	VENV := .venv
else ifeq (${ENVIRONMENT}, grid5000)
	ENVIRONMENT_FILE = ${ENV_G5000}
	VENV := ${G5K_STORAGE}/.venv
endif
# Binaries
PYTHON:= ${BIN}/python
PIP:=${BIN}/pip
# Important Files
DEV_REQUIREMENTS:=requirements-dev.txt

# Logs
LOGS:=logs

MAMBA_CHECK := $(shell mamba -h >/dev/null 2>&1 && echo "yes" || echo "no")

ifeq ($(MAMBA_CHECK),yes)
    PKG_MANAGER := mamba
else
    PKG_MANAGER := $(CONDA_EXE)
endif


all:
	@${MAKE} install-dev
	@chmod +x scripts/oar/*.sh
	@chmod +x scripts/bash/*.sh


clean:
	@${MAKE} clean-venv
	@${MAKE} clean-logs

clean-venv:
	@${PKG_MANAGER} env remove --prefix ${VENV}

clean-logs:
	@rm logs/*


${VENV}:
	@${PKG_MANAGER} env create --file=${ENVIRONMENT_FILE} --prefix=${VENV}

venv: ${VENV}

${LOGS}:
	@mkdir -p ${LOGS}

install: ${VENV} ${LOGS}
	@${PIP} install -e .

install-dev:
	@${MAKE} install
	@${PIP} install -r ${DEV_REQUIREMENTS}
	@${BIN}/pre-commit install --hook-type pre-commit
	@${BIN}/pre-commit install --hook-type pre-push

# GRID 5000 -----------------------------------------
g5k-import-%:
	rsync -avzP ${G5K_LOGIN}@rennes.g5k:${G5K_STORAGE}/$* ${G5K_IMPORT_STORAGE}
oar-stop-all:
	oarstat -u | awk 'NR>2 {print substr($$1, 1, 7)}' | xargs -r oardel
# ---------------------------------------------------
