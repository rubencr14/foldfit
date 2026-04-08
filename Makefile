.PHONY: install install-web test test-unit test-integration lint lint-fix typecheck backend web dev clean

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

install-web:
	cd web && npm install

# ── Run ───────────────────────────────────────────────────────────────────────

backend:
	@fuser -k 8000/tcp 2>/dev/null || true
	uvicorn foldfit.api.app:create_app --factory --reload --port 8000

web:
	@fuser -k 3000/tcp 2>/dev/null || true
	cd web && npm run dev

dev:
	@fuser -k 8000/tcp 3000/tcp 2>/dev/null || true
	@echo "Starting backend (8000) + frontend (3000)..."
	@make backend & make web

# ── Test ──────────────────────────────────────────────────────────────────────

test:
	python -m pytest tests/ -v --cov=src/foldfit --cov-report=term-missing

test-unit:
	python -m pytest tests/unit/ -v

test-integration:
	python -m pytest tests/integration/ -v

# ── Quality ───────────────────────────────────────────────────────────────────

lint:
	ruff check src/ tests/ scripts/

lint-fix:
	ruff check --fix src/ tests/ scripts/

typecheck:
	mypy src/

# ── Clean ─────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	rm -rf .coverage htmlcov/ dist/ build/ *.egg-info/
