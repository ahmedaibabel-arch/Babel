# Babel

Babel is a French correction system built to flag and correct grammar, spelling, typography, and agreement issues in short text.

It is designed as a local, MLX-based workflow for Apple Silicon:

- `scripts/01_extract_sentences.py` builds the clean sentence pool
- `scripts/02_corrupt_sentences.py` creates supervised error examples
- `scripts/03_augment_agy.py` and `scripts/03_augment_ollama.py` add harder cases
- `scripts/04_build_dataset.py` creates train/validation/test splits
- `scripts/05_train.sh` fine-tunes the adapter
- `scripts/06_evaluate.py` scores the model
- `scripts/07_run_babel.py` runs interactive correction

## What it returns

Babel expects a French sentence and returns strict JSON:

```json
{
  "errors": [
    {
      "original": "c'est rendu",
      "correction": "s'est rendu",
      "type": "complex_pronominal",
      "explanation": "Accord du participe passé avec un verbe pronominal."
    }
  ]
}
```

Clean text returns:

```json
{"errors": []}
```

## Repository layout

```text
.
├── Makefile
├── config.yaml
├── requirements.txt
├── rules/
├── scripts/
└── docs/
```

Generated datasets, logs, and model weights are intentionally excluded from version control. Rebuild them locally from the scripts when needed.

## Quick start

```bash
make setup
make pipeline
make train
make run
```

## Reproducibility notes

- The project uses public-domain and open corpora for training.
- No commercial third-party grammar-tool files, brand assets, or proprietary rule sets are included.
- Common French vocabulary may still appear in the training text, including words that overlap with product names.

See the docs folder for the source and workflow details.
