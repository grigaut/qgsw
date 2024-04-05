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

