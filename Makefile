.PHONY: help install test benchmark run clean

help:
	@echo "FraudGraph-GNN CLI Targets:"
	@echo "  make install    - Install project dependencies"
	@echo "  make test       - Run test suite with pytest"
	@echo "  make benchmark  - Run GNN vs Tabular comparative evaluation"
	@echo "  make run        - Start FastAPI server on port 8000"
	@echo "  make clean      - Clean temporary artifacts and cache"

install:
	python -m pip install --upgrade pip
	python -m pip install -e ".[dev]"

test:
	pytest -v tests/

benchmark:
	python benchmarks/eval_gnn_vs_tabular.py

run:
	uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

clean:
	python -c "import shutil, pathlib; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('__pycache__')]; [shutil.rmtree(p, ignore_errors=True) for p in pathlib.Path('.').rglob('.pytest_cache')]"
