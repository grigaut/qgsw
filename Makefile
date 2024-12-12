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
VENV := .venv
ENV_LOCAL := environment-local.yml
ENV_G5000 := environment-g5000.yml
ifeq (${ENVIRONMENT}, local)
	ENVIRONMENT_FILE = ${ENV_LOCAL}
else ifeq (${ENVIRONMENT}, grid5000)
	ENVIRONMENT_FILE = ${ENV_G5000}
endif
# Binaries
PYTHON:= ${BIN}/python
PIP:=${BIN}/pip
# Important Files
REQUIREMENTS:=requirements.txt
DEV_REQUIREMENTS:=requirements-dev.txt

# G5K runs
# Zip file
ZIP_FILE_ROOT := qgsw
# Folders
CONFIG := ./config
SRC := src
SCRIPTS := scripts
# Files
PYPROJECT := pyproject.toml
MAKEFILE := Makefile
ALPHA_COEFFICIENTS := data/coefficients.npz

all:
	@${MAKE} install-dev

clean:
	@${CONDA_EXE} env remove --prefix ${VENV}

${VENV}:
	@${CONDA_EXE} env create --file=${ENVIRONMENT_FILE} --prefix=${VENV}

venv: ${VENV}

install: ${VENV}
	@mkdir -p logs
	@chmod +x run.sh
	@${PIP} install -e .

install-dev: ${VENV}
	@${MAKE} install
	@${PIP} install -r ${DEV_REQUIREMENTS}

# GRID 5000 -----------------------------------------
g5k-import-%:
	scp -r ${G5K_LOGIN}@rennes.g5k:${G5K_STORAGE}/$* ${G5K_IMPORT_STORAGE}/$*

# ---------------------------------------------------
