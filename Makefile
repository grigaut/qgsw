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
ZIP_FOLDER := ${PWD}/g5k
ZIP_FILE := ${ZIP_FOLDER}/code.zip
# Folders
CONFIG := ${PWD}/config
SRC := ${PWD}/src
SCRIPTS := ${PWD}/scripts
# Files
PYPROJECT := ${PWD}/pyproject.toml
MAKEFILE := ${PWD}/Makefile
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

${ZIP_FOLDER}:
	mkdir -p ${ZIP_FOLDER}

compress: ${ZIP_FOLDER}
	find ${SRC} -iname \*.py | zip ${ZIP_FILE} -@
	find ${SCRIPTS} -iname \*.py | zip ${ZIP_FILE} -@
	find ${CONFIG} -iname \*.toml | zip ${ZIP_FILE} -@
	zip ${ZIP_FILE} ${PYPROJECT} ${MAKEFILE}