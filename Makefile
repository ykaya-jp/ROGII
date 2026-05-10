.PHONY: help install download eda baseline watch tune pseudo stack submit lint test fmt clean

SLUG := rogii-wellbore-geology-prediction
PKG  := rogii
UV   := uv

help:
	@echo "Targets:"
	@echo "  install   - uv sync (install deps incl. kagglib editable)"
	@echo "  download  - kaggle competitions download -c $(SLUG) -p data/raw + unzip"
	@echo "  eda       - launch jupyter on notebooks/00_eda.ipynb"
	@echo "  baseline  - run src/$(PKG)/train_baseline.py (LGB/XGB/CB/Linear, 5-fold CV)"
	@echo "  watch     - launch live monitor (rich-based dashboard, polls outputs/logs/exp00*_lgb.log)"
	@echo "  tune      - run src/$(PKG)/tune.py (Optuna 100 trials)"
	@echo "  pseudo    - run src/$(PKG)/pseudo.py (1-round pseudo-labeling)"
	@echo "  stack     - run src/$(PKG)/stack.py (OOF stacking)"
	@echo "  submit    - submit submissions/latest.csv (msg via M=...)"
	@echo "  lint      - ruff check + format"
	@echo "  test      - pytest tests/"
	@echo "  fmt       - ruff format"
	@echo "  clean     - remove __pycache__, mlruns, outputs/*"

install:
	$(UV) sync --extra dev
	$(UV) run pre-commit install

download:
	$(UV) run kaggle competitions download -c $(SLUG) -p data/raw
	@cd data/raw && for z in *.zip; do [ -f "$$z" ] && unzip -o -q "$$z" && rm "$$z" || true; done
	@echo "downloaded to data/raw/"
	@ls -la data/raw

eda:
	$(UV) run jupyter lab notebooks/00_eda.ipynb

baseline:
	$(UV) run python -m $(PKG).train_baseline

tune:
	$(UV) run python -m $(PKG).tune

pseudo:
	$(UV) run python -m $(PKG).pseudo

stack:
	$(UV) run python -m $(PKG).stack

submit:
	@if [ -z "$(M)" ]; then echo "usage: make submit M='your message'"; exit 1; fi
	$(UV) run python -m $(PKG).submit "$(M)"

lint:
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

fmt:
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

test:
	$(UV) run pytest tests -q

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	rm -rf mlruns mlartifacts
	rm -rf outputs/oof/* outputs/preds/* outputs/models/* outputs/logs/*
	@touch outputs/oof/.gitkeep outputs/preds/.gitkeep outputs/models/.gitkeep outputs/logs/.gitkeep

watch:
	$(UV) run python -m rogii.watch $(L)
