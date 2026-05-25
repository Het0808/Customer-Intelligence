# Customer Intelligence Platform — Makefile
# All commands assume you are in the project root with dependencies installed.
# On Windows, run via Git Bash or WSL. On macOS/Linux, use directly.

.PHONY: help install lint test validate train index serve serve-docker \
        drift rag-monitor shap clean

# ── Default target ────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Customer Intelligence Platform"
	@echo "  ──────────────────────────────"
	@echo "  make install        Install all Python dependencies"
	@echo "  make lint           Run ruff linter (E, F, W rules)"
	@echo "  make test           Run 139 unit + integration tests"
	@echo "  make validate       Validate committed 500-row sample (+ bad-record check)"
	@echo "  make train          Train LR + XGBoost on sample (~15 s)"
	@echo "  make index          Build FAISS complaint index from sample (~10 s)"
	@echo "  make serve          Start FastAPI server with --reload (dev)"
	@echo "  make serve-docker   Build + start via Docker Compose"
	@echo "  make mlflow         Open MLflow UI at http://localhost:5000"
	@echo "  make drift          Run Evidently drift report + retrain check"
	@echo "  make rag-monitor    Run 10-query RAG eval + save rag_metrics.json"
	@echo "  make shap           Generate SHAP waterfall + summary plots"
	@echo "  make clean          Remove __pycache__, .pytest_cache, *.pyc"
	@echo ""

# ── Setup ─────────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Quality ───────────────────────────────────────────────────────────────────
lint:
	ruff check --select E,F,W --ignore E501,E402 src/ tests/

test:
	pytest tests/ -q --tb=short

# ── Data ──────────────────────────────────────────────────────────────────────
validate:
	@echo "--- Validating committed sample ---"
	python -m src.data_pipeline.validate --sample
	@echo "--- Confirming bad records are rejected ---"
	@if python -m src.data_pipeline.validate --sample --inject-bad 2>/dev/null; then \
	  echo "ERROR: validator silently accepted bad records" && exit 1; \
	fi
	@echo "OK: bad record correctly rejected."

# ── Training ──────────────────────────────────────────────────────────────────
train:
	python -m src.training.train --sample

train-full:
	python -m src.training.train --include-stump

# ── RAG index ─────────────────────────────────────────────────────────────────
index:
	python -m src.rag.build_index --sample

index-full:
	python -m src.rag.build_index

# ── Serving ───────────────────────────────────────────────────────────────────
serve:
	uvicorn src.serving.serve:app --host 0.0.0.0 --port 8000 --reload

serve-docker:
	docker compose up --build

mlflow:
	mlflow ui --backend-store-uri file:mlruns

# ── Monitoring & Governance ───────────────────────────────────────────────────
drift:
	python monitoring/ml_drift.py

rag-monitor:
	python monitoring/rag_monitor.py

shap:
	python docs/shap_analysis.py

# ── Combined: full pipeline from scratch ─────────────────────────────────────
all: install validate train index
	@echo ""
	@echo "  Setup complete. Run 'make serve' to start the API."
	@echo "  Run 'make test' to verify all 139 tests pass."
	@echo ""

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned up __pycache__, .pytest_cache, *.pyc"
