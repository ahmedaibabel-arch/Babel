# Contributing

Thanks for helping improve Babel.

## Local workflow

1. Install dependencies with `make setup`.
2. Rebuild data with `make pipeline` when you change extraction or corruption logic.
3. Train with `make train` after the dataset is ready.
4. Validate with `make evaluate`.

## Code conventions

- Keep scripts deterministic when they generate training data.
- Prefer small, reviewable changes over broad pipeline edits.
- Update the docs when the output format or workflow changes.

## Commit hygiene

- Stage only the files that belong in the change.
- Do not commit generated datasets, logs, or model weights.
- If you adjust the JSON contract, update `README.md` and `docs/ARCHITECTURE.md` together.
