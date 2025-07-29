# Makefile for Secure MPC Transformer development and deployment

# Configuration
PYTHON := python3
PIP := pip
CONDA := conda
DOCKER := docker
PROJECT_NAME := secure-mpc-transformer
VENV_NAME := mpc-transformer
SRC_DIR := src
TEST_DIR := tests
DOCS_DIR := docs

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[0;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

# Help target
.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)Secure MPC Transformer Development Commands$(NC)"
	@echo "==========================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Environment setup
.PHONY: setup
setup: ## Set up development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	$(CONDA) create -n $(VENV_NAME) python=3.10 -y
	@echo "$(GREEN)Environment created. Activate with: conda activate $(VENV_NAME)$(NC)"

.PHONY: install
install: ## Install project dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install -e ".[dev,gpu,benchmark]"
	pre-commit install
	pre-commit install --hook-type commit-msg
	detect-secrets scan --baseline .secrets.baseline
	@echo "$(GREEN)Installation complete!$(NC)"

.PHONY: install-cpu
install-cpu: ## Install CPU-only dependencies
	@echo "$(BLUE)Installing CPU-only dependencies...$(NC)"
	$(PIP) install -e ".[dev]"
	pre-commit install
	@echo "$(GREEN)CPU-only installation complete!$(NC)"

# Code quality
.PHONY: format
format: ## Format code with black and isort
	@echo "$(BLUE)Formatting code...$(NC)"
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR) --profile black
	@echo "$(GREEN)Code formatting complete!$(NC)"

.PHONY: lint
lint: ## Lint code with ruff
	@echo "$(BLUE)Linting code...$(NC)"
	ruff check $(SRC_DIR) $(TEST_DIR) --fix
	@echo "$(GREEN)Linting complete!$(NC)"

.PHONY: typecheck
typecheck: ## Type check with mypy
	@echo "$(BLUE)Type checking...$(NC)"
	mypy $(SRC_DIR) --ignore-missing-imports
	@echo "$(GREEN)Type checking complete!$(NC)"

.PHONY: security
security: ## Run security scans
	@echo "$(BLUE)Running security scans...$(NC)"
	bandit -r $(SRC_DIR) -f json -o bandit-report.json || true
	safety check --json --output safety-report.json || true
	detect-secrets scan --baseline .secrets.baseline
	@echo "$(GREEN)Security scans complete!$(NC)"
	@echo "$(YELLOW)Check bandit-report.json and safety-report.json for details$(NC)"

.PHONY: quality
quality: format lint typecheck security ## Run all code quality checks
	@echo "$(GREEN)All quality checks complete!$(NC)"

.PHONY: pre-commit
pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(NC)"
	pre-commit run --all-files
	@echo "$(GREEN)Pre-commit checks complete!$(NC)"

# Testing
.PHONY: test
test: ## Run unit tests
	@echo "$(BLUE)Running unit tests...$(NC)"
	pytest $(TEST_DIR)/unit/ -v --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Unit tests complete!$(NC)"

.PHONY: test-integration
test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	pytest $(TEST_DIR)/integration/ -v --timeout=300
	@echo "$(GREEN)Integration tests complete!$(NC)"

.PHONY: test-gpu
test-gpu: ## Run GPU tests
	@echo "$(BLUE)Running GPU tests...$(NC)"
	pytest $(TEST_DIR) --gpu -v
	@echo "$(GREEN)GPU tests complete!$(NC)"

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)Running end-to-end tests...$(NC)"
	pytest $(TEST_DIR)/e2e/ --slow -v
	@echo "$(GREEN)End-to-end tests complete!$(NC)"

.PHONY: test-all
test-all: test test-integration test-e2e ## Run all tests
	@echo "$(GREEN)All tests complete!$(NC)"

.PHONY: benchmark
benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	pytest $(TEST_DIR) --benchmark -v
	python benchmarks/run_all.py --output benchmarks/results/
	@echo "$(GREEN)Benchmarks complete!$(NC)"

# Building
.PHONY: clean
clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage .pytest_cache .mypy_cache .ruff_cache
	rm -f bandit-report.json safety-report.json
	@echo "$(GREEN)Cleanup complete!$(NC)"

.PHONY: build
build: clean ## Build Python package
	@echo "$(BLUE)Building package...$(NC)"
	$(PYTHON) -m build
	twine check dist/*
	@echo "$(GREEN)Package build complete!$(NC)"

.PHONY: build-cuda
build-cuda: ## Build CUDA kernels
	@echo "$(BLUE)Building CUDA kernels...$(NC)"
	cd kernels/cuda && make clean && make all
	@echo "$(GREEN)CUDA kernels build complete!$(NC)"

# Docker
.PHONY: docker-build-cpu
docker-build-cpu: ## Build CPU Docker image
	@echo "$(BLUE)Building CPU Docker image...$(NC)"
	$(DOCKER) build -f docker/Dockerfile.cpu -t $(PROJECT_NAME):cpu .
	@echo "$(GREEN)CPU Docker image build complete!$(NC)"

.PHONY: docker-build-gpu
docker-build-gpu: ## Build GPU Docker image
	@echo "$(BLUE)Building GPU Docker image...$(NC)"
	$(DOCKER) build -f docker/Dockerfile.gpu -t $(PROJECT_NAME):gpu .
	@echo "$(GREEN)GPU Docker image build complete!$(NC)"

.PHONY: docker-build-dev
docker-build-dev: ## Build development Docker image
	@echo "$(BLUE)Building development Docker image...$(NC)"
	$(DOCKER) build -f docker/Dockerfile.dev -t $(PROJECT_NAME):dev .
	@echo "$(GREEN)Development Docker image build complete!$(NC)"

.PHONY: docker-run-dev
docker-run-dev: ## Run development Docker container
	@echo "$(BLUE)Starting development container...$(NC)"
	$(DOCKER) run -it --rm \
		-v $(PWD):/workspace \
		-p 8080:8080 \
		--name $(PROJECT_NAME)-dev \
		$(PROJECT_NAME):dev
	@echo "$(GREEN)Development container started!$(NC)"

.PHONY: docker-compose-up
docker-compose-up: ## Start monitoring stack
	@echo "$(BLUE)Starting monitoring stack...$(NC)"
	cd monitoring && docker-compose -f docker-compose.monitoring.yml up -d
	@echo "$(GREEN)Monitoring stack started!$(NC)"
	@echo "$(YELLOW)Grafana: http://localhost:3000 (admin/admin123)$(NC)"
	@echo "$(YELLOW)Prometheus: http://localhost:9090$(NC)"

.PHONY: docker-compose-down
docker-compose-down: ## Stop monitoring stack
	@echo "$(BLUE)Stopping monitoring stack...$(NC)"
	cd monitoring && docker-compose -f docker-compose.monitoring.yml down
	@echo "$(GREEN)Monitoring stack stopped!$(NC)"

# Documentation
.PHONY: docs
docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	cd $(DOCS_DIR) && make html
	@echo "$(GREEN)Documentation build complete!$(NC)"

.PHONY: docs-serve
docs-serve: docs ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000
	@echo "$(GREEN)Documentation served at http://localhost:8000$(NC)"

# Development workflows
.PHONY: dev-setup
dev-setup: setup install pre-commit ## Complete development setup
	@echo "$(GREEN)Development setup complete!$(NC)"
	@echo "$(YELLOW)Don't forget to activate your environment: conda activate $(VENV_NAME)$(NC)"

.PHONY: dev-check
dev-check: quality test ## Run development checks
	@echo "$(GREEN)Development checks complete!$(NC)"

.PHONY: ci-check
ci-check: quality test-all ## Run CI-like checks locally
	@echo "$(GREEN)CI checks complete!$(NC)"

# Release workflow
.PHONY: version
version: ## Show current version
	@echo "$(BLUE)Current version:$(NC)"
	@$(PYTHON) -c "import tomllib; print(tomllib.load(open('pyproject.toml', 'rb'))['project']['version'])"

.PHONY: release-check
release-check: clean quality test-all build ## Run release readiness checks
	@echo "$(GREEN)Release checks complete!$(NC)"

# Monitoring and profiling
.PHONY: profile
profile: ## Run performance profiling
	@echo "$(BLUE)Running performance profiling...$(NC)"
	$(PYTHON) -m cProfile -o profile.stats benchmarks/benchmark_bert.py
	$(PYTHON) -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
	@echo "$(GREEN)Profiling complete!$(NC)"

.PHONY: monitor
monitor: ## Start monitoring dashboards
	@echo "$(BLUE)Opening monitoring dashboards...$(NC)"
	@echo "$(YELLOW)Grafana: http://localhost:3000$(NC)"
	@echo "$(YELLOW)Prometheus: http://localhost:9090$(NC)"
	@echo "$(YELLOW)Alertmanager: http://localhost:9093$(NC)"

# Utility targets
.PHONY: deps-update
deps-update: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --upgrade -e ".[dev,gpu,benchmark]"
	pre-commit autoupdate
	@echo "$(GREEN)Dependencies updated!$(NC)"

.PHONY: deps-check
deps-check: ## Check for outdated dependencies
	@echo "$(BLUE)Checking for outdated dependencies...$(NC)"
	$(PIP) list --outdated
	@echo "$(GREEN)Dependency check complete!$(NC)"

.PHONY: info
info: ## Show project information
	@echo "$(BLUE)Project Information$(NC)"
	@echo "=================="
	@echo "$(YELLOW)Project:$(NC) $(PROJECT_NAME)"
	@echo "$(YELLOW)Python:$(NC) $$($(PYTHON) --version)"
	@echo "$(YELLOW)Pip:$(NC) $$($(PIP) --version)"
	@echo "$(YELLOW)Docker:$(NC) $$($(DOCKER) --version 2>/dev/null || echo 'Not installed')"
	@echo "$(YELLOW)CUDA:$(NC) $$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 'Not available')"
	@echo "$(YELLOW)Environment:$(NC) $$(echo $$CONDA_DEFAULT_ENV 2>/dev/null || echo 'Not in conda environment')"

# Debug targets
.PHONY: debug-env
debug-env: ## Show environment information for debugging
	@echo "$(BLUE)Environment Debug Information$(NC)"
	@echo "============================"
	@env | grep -E "(PYTHON|CUDA|PATH)" | sort
	@echo ""
	@echo "$(YELLOW)Python packages:$(NC)"
	@$(PIP) list | grep -E "(torch|numpy|cryptography)"

# Maintenance
.PHONY: update-baseline
update-baseline: ## Update secrets detection baseline
	@echo "$(BLUE)Updating secrets baseline...$(NC)"
	detect-secrets scan --baseline .secrets.baseline
	@echo "$(GREEN)Secrets baseline updated!$(NC)"