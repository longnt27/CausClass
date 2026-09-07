.PHONY: install test lint smoke report clean-report
SMOKE_OUT ?= output/smoke
install:
	uv sync --locked --extra dev --extra llm

test:
	uv run --no-sync pytest --cov --cov-report=term-missing --cov-report=xml

lint:
	uv run --no-sync ruff check core data utils tests app.py
	uv run --no-sync ruff format --check core/aerca_verifier.py core/smoke.py utils/reproducibility.py tests

smoke:
	uv run --no-sync python -m core.smoke --output-dir $(SMOKE_OUT)

report:
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
	python scripts/check_report_log.py report/build/main.log

clean-report:
	cd report && latexmk -C -outdir=build main.tex
