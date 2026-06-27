# Data Sources

This repository is designed to work with public and open resources only.

## Current Pipeline Inputs

| Source | Use | License / Status |
|---|---|---|
| Public-domain French book corpora | Sentence extraction for synthetic training examples | Public domain or similarly permissive, verify per title |
| Universal Dependencies French GSD | Grammar-rich clean sentence source | CC BY-SA 4.0 |
| French Wikipedia | Clean sentence source and hard negatives | CC BY-SA style attribution/share-alike terms |
| French Wikisource | Formal sentence source and hard negatives | Mixed public-domain / license-dependent works |

## Data Handling Rules

- Do not commit raw corpora, generated datasets, logs, or model weights.
- Keep source-specific attribution in the notices file.
- Verify the upstream license for every added corpus before using it in generated training data.

## Notes

The repository intentionally avoids restricted payloads, private manuscript data, and vendor-specific source material.
