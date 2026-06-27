# Babel Model Card

## Status

Babel is a local French correction system with a training pipeline in this repository. No released production weights are committed here.

## Intended Use

- French spelling correction
- Grammar and agreement correction
- Typography and spacing correction
- Short explanatory feedback in strict JSON

## Out Of Scope

- Long-form editing
- Creative rewriting
- Sensitive content moderation
- High-stakes legal, medical, or financial judgment

## Input And Output

Input:

```text
French text
```

Output:

```json
{"errors": []}
```

or a list of corrections with `original`, `correction`, `type`, and `explanation`.

## Training Summary

- Sentence extraction from open and public-domain corpora
- Deterministic corruption for supervised examples
- Optional augmentation with additional open resources
- MLX LoRA fine-tuning for local Apple Silicon inference

## Limitations

- The system is only as strong as the training coverage for a given error pattern.
- Exact span reconstruction can be imperfect for some corrections.
- Generated output should be reviewed before publication in a production writing workflow.

## Licensing

See `THIRD_PARTY_NOTICES.md` and `DATA_SOURCES.md` for data and dependency attribution notes.
