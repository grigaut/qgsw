ifneq (,$(wildcard ./.env))
	include .env
    export
endif
# Virtual Environment Management
ENVIRONMENT_FILEPATH := environment.yml
VENV := ${PWD}/.venv
ifeq ($(OS), Windows_NT)
	BIN = ${VENV}/Scripts/
else
	BIN = ${VENV}/bin/
endif
# Binaries
PYTHON:= ${BIN}/python
PIP:=${BIN}/pip
# Important Files
REQUIREMENTS:=${PWD}/requirements.txt
DEV_REQUIREMENTS:=${PWD}/requirements-dev.txt

# G5K runs
# Zip file
ZIP_FILE := qgsw.zip
# Folders
CONFIG := ./config
SRC := src
SCRIPTS := scripts
# Files
PYPROJECT := pyproject.toml
MAKEFILE := Makefile

all:
	@${MAKE} install-dev

clean:
	@${CONDA_EXE} env remove --prefix ${VENV}

${VENV}:
	@${CONDA_EXE} env create --file ${ENVIRONMENT_FILEPATH} --prefix ${VENV}

venv: ${VENV}

install: ${VENV}
	@${PIP} install -r ${REQUIREMENTS} -f https://download.pytorch.org/whl/cu111/torch_stable.html
	@${PIP} install -e .

install-dev: ${VENV}
	@${PIP} install -r ${REQUIREMENTS} -f https://download.pytorch.org/whl/cu111/torch_stable.html
	@${PIP} install -r ${DEV_REQUIREMENTS}
	@${PIP} install -e .

# GRID 5000 -----------------------------------------

export-to-g5k:
	# Create temp directory
	$(eval tmp := $(shell mktemp --directory --tmpdir=${PWD}))
	# Zip files
	find ${SRC} -iname \*.py | zip -r ${tmp}/${ZIP_FILE} -@
	find ${SCRIPTS} -iname \*.py | zip -r ${tmp}/${ZIP_FILE} -@
	find ${CONFIG} -iname \*.toml | zip -r ${tmp}/${ZIP_FILE} -@
	zip -r ${tmp}/${ZIP_FILE} ${MAKEFILE} ${PYPROJECT}
	# Export to g5k
	scp ${tmp}/${ZIP_FILE} ${LOGIN}@rennes.g5k:~/
	# Remove temp files
	rm -rf ${tmp}

install-g5k:
	pip install .
# ---------------------------------------------------
