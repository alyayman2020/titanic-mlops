###############################################################################
# GLOBALS                                                                     #
###############################################################################

PROJECT_NAME = titanic-mlops
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

###############################################################################
# COMMANDS                                                                    #
###############################################################################

## Install Python dependencies using uv
.PHONY: requirements
requirements:
	uv sync

## Create virtual environment with uv
.PHONY: create_environment
create_environment:
	uv venv --python $(PYTHON_VERSION)
	@echo ">>> Virtual environment created. Activate with:"
	@echo ">>> Unix/macOS : source .venv/bin/activate"
	@echo ">>> Windows    : .venv\\Scripts\\activate"

## Run the full training pipeline
.PHONY: train
train:
	$(PYTHON_INTERPRETER) trainer.py

## Format code with ruff + black (ruff handles imports, black handles style)
.PHONY: format
format:
	ruff check --fix src/ trainer.py
	black src/ trainer.py

## Lint code without modifying files
.PHONY: lint
lint:
	ruff check src/ trainer.py
	black --check src/ trainer.py

## Run tests with coverage
.PHONY: test
test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

## Delete compiled Python files and caches
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov

###############################################################################
# Self Documenting Commands                                                   #
###############################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys
lines = '\n'.join([line for line in sys.stdin])
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines)
print('Available rules:\n')
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)
