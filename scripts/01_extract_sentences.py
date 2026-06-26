"""
Script 01 — Extract clean French sentences from the book corpus.

Reads .txt files from books_dir (2,000+ books).
Filters for quality: length, alpha density, no URLs, no Gutenberg headers.
Writes to data/sentences/clean_sentences.jsonl

Runtime: ~30-60 min for 2,000 books.
"""

import json
import re
import sys
from pathlib import Path
import yaml
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
cfg  = yaml.safe_load((ROOT / "config.yaml").read_text())

BOOKS_DIR = Path(cfg["books_dir"])
OUT_PATH  = ROOT / cfg["data"]["sentences_out"]
MIN_LEN   = cfg["data"]["min_sentence_len"]
MAX_LEN   = cfg["data"]["max_sentence_len"]

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Split on sentence-ending punctuation followed by whitespace and a capital
_SENT_RE = re.compile(
    r'(?<=[.!?…»])\s+(?=[A-ZÁÀÂÄÉÈÊËÎÏÔÙÛÜŒ«——])'
)

# Gutenberg boilerplate markers to strip
_BOILERPLATE = re.compile(
    r'(START OF|END OF|PROJECT GUTENBERG|GUTENBERG|gutenberg|Produced by|'
    r'Transcribed by|Transcription|CHAPTER|CHAPITRE|^\s*[\*_\-=]{3,})',
    re.IGNORECASE | re.MULTILINE
)

_URL_RE    = re.compile(r'https?://\S+|www\.\S+')
_NUM_ONLY  = re.compile(r'^[\d\s\.,;:\-\(\)\[\]]+$')
_ALPHA_RE  = re.compile(r'[a-zA-ZÀ-ÿ]')


def strip_boilerplate(text: str) -> str:
    """Remove Project Gutenberg headers/footers."""
    lines = text.splitlines()
    start, end = 0, len(lines)
    for i, line in enumerate(lines):
        if re.search(r'START OF (THIS|THE) PROJECT', line, re.I):
            start = i + 1
        if re.search(r'END OF (THIS|THE) PROJECT', line, re.I):
            end = i
            break
    return "\n".join(lines[start:end])


def normalize(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[_*#<>~^]+', ' ', text)
    return text.strip()


def is_good(sentence: str) -> bool:
    s = sentence.strip()
    if len(s) < MIN_LEN or len(s) > MAX_LEN:
        return False
    alpha = len(_ALPHA_RE.findall(s))
    if alpha < 20:
        return False
    if alpha / max(len(s), 1) < 0.55:
        return False
    if _URL_RE.search(s):
        return False
    if _NUM_ONLY.match(s):
        return False
    # Skip lines that look like chapter headings
    if re.match(r'^(chapitre|chapter|partie|part|livre|book|tome)\b', s, re.I):
        return False
    # Must end with a sentence-final character
    if s[-1] not in '.!?…»"\'':
        return False
    # Must contain at least one space (not a single word)
    if ' ' not in s:
        return False
    return True


def split_sentences(text: str) -> list[str]:
    parts = _SENT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def process_file(path: Path) -> list[dict]:
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return []

    raw  = strip_boilerplate(raw)
    text = normalize(raw)

    # Process in chunks to handle very large files
    chunk_size = 500_000
    results = []
    for i in range(0, len(text), chunk_size):
        chunk     = text[i:i + chunk_size]
        sentences = split_sentences(chunk)
        for s in sentences:
            if is_good(s):
                results.append({
                    "source": path.stem,
                    "sentence": s
                })
    return results


def main():
    files = sorted(BOOKS_DIR.glob("*.txt"))
    if not files:
        print(f"ERROR: No .txt files found in {BOOKS_DIR}")
        sys.exit(1)

    print(f"Found {len(files)} book files in {BOOKS_DIR}")
    print(f"Writing sentences to {OUT_PATH}")

    total = 0
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for path in tqdm(files, desc="Extracting"):
            rows = process_file(path)
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            total += len(rows)
            if total % 50_000 == 0 and total > 0:
                tqdm.write(f"  → {total:,} sentences so far")

    print(f"\n✓ Done. Extracted {total:,} clean sentences → {OUT_PATH}")


if __name__ == "__main__":
    main()
