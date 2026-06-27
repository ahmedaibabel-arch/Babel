# Architecture

Babel is a local, sentence-level French corrector with a deliberately simple pipeline:

1. Extract clean French sentences from public or public-domain sources.
2. Corrupt them deterministically to create labeled training pairs.
3. Augment the dataset with additional open-resource examples where allowed.
4. Split the data into train, validation, and test sets.
5. Fine-tune an MLX LoRA adapter for local inference.
6. Run the adapter through a strict JSON CLI.

## Runtime Shape

- Input: one French sentence at a time.
- Output: one JSON object with `errors`.
- Presentation: the CLI can recover spans for display after inference.

## Repository Boundaries

- No generated datasets or model weights are committed.
- No private source paths are used in docs, configs, or scripts.
- The repo keeps source attribution separate from implementation code.

## Verification

The clean-room scan should pass before any release or public sharing of generated artifacts.
