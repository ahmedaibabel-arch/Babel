# Third-Party Notices

Babel's scripts can download or derive training examples from external sources. Review the upstream licenses before training, redistributing data, or publishing model artifacts.

## Runtime And Build Dependencies

- `mlx`
- `mlx-lm`
- `transformers`
- `sentencepiece`
- `protobuf`
- `spacy`
- `pyyaml`
- `tqdm`
- `orjson`
- `requests`
- `regex`
- `huggingface_hub`

These are listed in `requirements.txt`; verify their upstream license metadata before redistributing packaged builds.

## UD French GSD

- Source: https://github.com/UniversalDependencies/UD_French-GSD
- License: CC BY-SA 4.0
- Notes: attribution and share-alike obligations may apply to derived datasets and model artifacts.

## French Wikipedia

- Source: https://fr.wikipedia.org/
- License information: https://fr.wikipedia.org/wiki/Wikip%C3%A9dia:Copyright
- Notes: Wikipedia text is generally available under Creative Commons attribution/share-alike terms, with attribution and redistribution requirements.

## French Wikisource

- Source: https://fr.wikisource.org/
- License information: https://fr.wikisource.org/wiki/Wikisource:Copyright
- Notes: individual works may have different public-domain or license status. Verify source status before use.

## Generated Data

Generated datasets, logs, and model outputs are not committed to this repository. If you publish generated artifacts, include the required attributions and license notices for every upstream source used to produce them.
