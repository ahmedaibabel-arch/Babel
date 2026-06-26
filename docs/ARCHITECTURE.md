# Architecture

Babel is a sentence-level French corrector built around a simple pipeline:

1. Extract clean sentences from source corpora.
2. Corrupt them deterministically to create paired supervision.
3. Add hard negatives and harder grammar examples.
4. Split the dataset into train, validation, and test sets.
5. Fine-tune an MLX LoRA adapter.
6. Run the adapter locally through `scripts/07_run_babel.py`.

## Design goals

- strict JSON output
- local execution on Apple Silicon
- reproducible dataset generation
- enough negative examples to avoid over-correction

## Output contract

The runtime returns an object with one key:

```json
{"errors": []}
```

When errors exist, each item should include:

- `original`
- `correction`
- `type`
- `explanation`

The runtime resolves positions for display after generation.

## Notes

- The project intentionally avoids shipping large generated assets in git.
- The adapter and dataset are meant to be rebuilt locally from the scripts.
