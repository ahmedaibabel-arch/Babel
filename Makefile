PYTHON = .venv/bin/python
VENV   = .venv

.PHONY: all setup scan extract enrich corrupt augment dataset train evaluate run status help

help:
	@echo ""
	@echo "Babel — French Corrector Training Pipeline"
	@echo ""
	@echo "  make setup      Create venv and install dependencies"
	@echo "  make scan       Run the banned-token clean-room scan"
	@echo "  make extract    Phase 1:  Extract clean sentences from books"
	@echo "  make enrich     Phase 1b: Add UD GSD + Wikipedia + Wikisource (register diversity)"
	@echo "  make corrupt    Phase 2:  Generate error examples (algorithmic)"
	@echo "  make augment    Phase 3:  Augment with Gemini model pool via SDK (12 workers)"
	@echo "  make dataset    Phase 4:  Build train/valid/test splits"
	@echo "  make train      Phase 5:  Fine-tune Gemma 3 4B with LoRA (overnight)"
	@echo "  make evaluate   Phase 6:  Evaluate on test set"
	@echo "  make run        Phase 7:  Interactive French corrector"
	@echo "  make status     Show current progress"
	@echo "  make pipeline   Run phases 1, 1b, 2, 4 in sequence"
	@echo ""

setup:
	@echo "Creating virtual environment..."
	python3 -m venv $(VENV)
	$(VENV)/bin/pip install -U pip --quiet
	$(VENV)/bin/pip install -r requirements.txt --quiet
	$(VENV)/bin/python -m spacy download fr_core_news_sm --quiet
	@echo "✓ Setup complete."

scan:
	python3 scripts/00_scan_banned_tokens.py

extract:
	$(PYTHON) scripts/01_extract_sentences.py

enrich:
	$(PYTHON) scripts/08_download_extra_corpora.py

corrupt:
	$(PYTHON) scripts/02_corrupt_sentences.py

augment:
	$(PYTHON) scripts/03_augment_agy.py

augment-ollama:
	$(PYTHON) scripts/03_augment_ollama.py

dataset:
	$(PYTHON) scripts/04_build_dataset.py

train:
	@chmod +x scripts/05_train.sh
	bash scripts/05_train.sh

evaluate:
	$(PYTHON) scripts/06_evaluate.py

run:
	$(PYTHON) scripts/07_run_babel.py

status:
	@bash scripts/status.sh

pipeline: extract enrich corrupt dataset
	@echo ""
	@echo "══════════════════════════════════════════════"
	@echo "  Data pipeline complete."
	@echo "  Run 'make augment' to add Gemini examples (recommended)."
	@echo "  Run 'make train' to start fine-tuning (overnight)."
	@echo "══════════════════════════════════════════════"
