# Data Policy

## Included sources

The project is built from a mix of:

- public-domain French book corpora
- open linguistic resources such as UD French GSD
- open reference material such as Wikipedia and Wikisource excerpts

## What is not included

- proprietary third-party grammar-tool content
- commercial dictionaries, rules, or assets
- generated model checkpoints or large raw corpora

## Important clarification

Common French vocabulary can still appear inside ordinary training sentences, including words that overlap with product names. That is normal language use and not a brand or asset inclusion.

## Reproducibility

The large generated files are intentionally not tracked in git. Regenerate them locally with the pipeline scripts when needed.

See `THIRD_PARTY_NOTICES.md` before using or redistributing any generated dataset.
