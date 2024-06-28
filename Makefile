ifneq (,$(wildcard ./.env))
	include .env
    export
endif
# Virtual Environment Management
ENVIRONMENT_FILE := environment.yml
VENV := .venv
ifeq ($(OS), Windows_NT)
	BIN = ${VENV}/Scripts/
else
	BIN = ${VENV}/bin/
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
	@${CONDA_EXE} env create --file ${ENVIRONMENT_FILE} --prefix ${VENV}

venv: ${VENV}

install: ${VENV}
	@${PIP} install -e .

install-dev: ${VENV}
	@${PIP} install -r ${DEV_REQUIREMENTS}
	@${PIP} install -e .

# GRID 5000 -----------------------------------------

# export current code to grid 5000
g5k-export:
	# Create temp directory
	$(eval tmp := $(shell mktemp --directory --tmpdir=${PWD}))
	$(eval ZIP_FILE := ${ZIP_FILE_ROOT}.zip)
	# Zip files
	find ${SRC} -iname \*.py | zip -r ${tmp}/${ZIP_FILE} -@
	find ${SCRIPTS} -iname \*.py | zip -r ${tmp}/${ZIP_FILE} -@
	find ${CONFIG} -iname \*.toml | zip -r ${tmp}/${ZIP_FILE} -@
	zip -r ${tmp}/${ZIP_FILE} ${MAKEFILE} ${PYPROJECT} ${REQUIREMENTS} ${ENVIRONMENT_FILE} ${ALPHA_COEFFICIENTS}
	# Export to g5k
	scp ${tmp}/${ZIP_FILE} ${G5K_LOGIN}@rennes.g5k:~/
	# Remove temp files
	rm -rf ${tmp}

g5k-import-%:
	scp -r ${G5K_LOGIN}@rennes.g5k:${G5K_STORAGE}/$* ${G5K_IMPORT_STORAGE}

g5k-import:
	@${MAKE} g5k-import-results

# ---------------------------------------------------
