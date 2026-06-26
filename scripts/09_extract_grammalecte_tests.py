"""
Script 09 — Extract Grammalecte test cases as Babel training examples.

Reads gc_test.txt from the bundled grammalecte_fr.zip.
Format: <line_num>   <sentence with {{error}}>   ->> <correction>

Produces two kinds of training examples:
  1. Error examples: sentence with error → JSON detection + correction
  2. Clean examples: correct sentences → {"errors": []}

These are the "Bled" equivalent — expert-curated French grammar examples
covering 90+ rule categories from Grammalecte's test suite.

Output: appended to data/examples/all_examples.jsonl
"""

import io
import json
import re
import zipfile
from pathlib import Path

import yaml

ROOT          = Path(__file__).parent.parent
cfg           = yaml.safe_load((ROOT / "config.yaml").read_text())
EXAMPLES_PATH = ROOT / cfg["data"]["examples_out"]
SYSTEM_PROMPT = cfg["system_prompt"].strip()
ZIP_PATH      = Path("/Users/ai/Mendel/Sources/Mendel/Resources/grammalecte_fr.zip")

EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)

# Maps Grammalecte rule category prefixes to Babel error types
RULE_TYPE_MAP = {
    "esp": "typographie_espacement",
    "tab": "typographie_tabulation",
    "nbsp": "typographie_espace_insecable",
    "apos": "typographie_apostrophe",
    "typo": "typographie_general",
    "ponct": "ponctuation",
    "liga": "typographie_ligature",
    "nf": "negation_forme",
    "conf": "homophone_confusion",
    "tu": "accord_tu",
    "maj": "majuscule",
    "num": "numerique",
    "eleu": "euphonie",
    "ocr": "typographie_ocr",
    "redon": "redondance",
    "mc": "mots_composes",
    "general": "grammaire_general",
}

_DOUBLE_BRACE = re.compile(r'\{\{(.+?)\}\}', re.DOTALL)
_LINE_NUM     = re.compile(r'^\s*\d+\s+')
_ARROW        = re.compile(r'\s*->>\s*')


def clean_line(raw: str) -> str:
    """Remove leading line number from a test file line."""
    return _LINE_NUM.sub('', raw).strip()


def build_error_example(corrupted: str, original: str, correction: str,
                        start: int, end: int, error_type: str) -> dict:
    assistant = {
        "errors": [{
            "start":       start,
            "end":         end,
            "original":    original,
            "correction":  correction,
            "type":        error_type,
            "confidence":  0.97,
            "explanation": f"Règle Grammalecte : erreur de type « {error_type} »."
        }]
    }
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Texte: {corrupted}"},
            {"role": "assistant", "content": json.dumps(assistant, ensure_ascii=False)}
        ]
    }


def build_clean_example(sentence: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Texte: {sentence}"},
            {"role": "assistant", "content": '{"errors": []}'}
        ]
    }


def parse_test_file(content: str) -> tuple[list[dict], int, int]:
    """Parse gc_test.txt and return (examples, n_error, n_clean)."""
    examples   = []
    n_error    = 0
    n_clean    = 0
    cur_rule   = "general"

    for raw_line in content.splitlines():
        line = raw_line.strip()

        # Track current rule section from comments like "# RULE: typo_*"
        if line.startswith('#'):
            m = re.search(r'[a-z]+_[a-z]+', line)
            if m:
                prefix = m.group(0).split('_')[0]
                cur_rule = RULE_TYPE_MAP.get(prefix, f"grammalecte_{prefix}")
            continue

        if not line:
            continue

        # Remove leading line number
        line = clean_line(line)
        if not line:
            continue

        # Split on ->>
        parts = _ARROW.split(line, maxsplit=1)

        if len(parts) == 2:
            # Error example
            raw_sentence, raw_correction = parts[0].strip(), parts[1].strip()

            # Take first correction option (before |)
            correction = raw_correction.split('|')[0].strip().strip('"')
            if not correction:
                continue

            # Find all {{...}} spans
            matches = list(_DOUBLE_BRACE.finditer(raw_sentence))
            if not matches:
                continue

            # Use first error span only (keeps labels clean)
            m       = matches[0]
            original = m.group(1)

            # Build the corrupted sentence (remove {{ }})
            corrupted = raw_sentence[:m.start()] + original + raw_sentence[m.end():]
            # Remove any remaining {{ }} from other spans
            corrupted = _DOUBLE_BRACE.sub(lambda x: x.group(1), corrupted)
            corrupted = corrupted.strip()

            if len(corrupted) < 15 or len(original) == 0:
                continue
            if original == correction:
                continue

            start = m.start()
            end   = start + len(original)

            # Infer error type from correction content
            error_type = cur_rule
            if re.search(r"[àâäéèêëîïôùûüœæ]", correction) and not re.search(r"[àâäéèêëîïôùûüœæ]", original):
                error_type = "accent_manquant"
            elif correction in ("'", "'"):
                error_type = "typographie_apostrophe"
            elif re.match(r'^[?!;:»«]', correction):
                error_type = "typographie_espacement"

            examples.append(build_error_example(
                corrupted, original, correction, start, end, error_type
            ))
            n_error += 1

        else:
            # Potential clean example — no error markers, no ->>
            sentence = line
            if '{{' in sentence or '}}' in sentence:
                continue
            if len(sentence) < 25:
                continue
            # Exclude lines that are clearly not sentences
            alpha = sum(1 for c in sentence if c.isalpha())
            if alpha < 15 or alpha / max(len(sentence), 1) < 0.5:
                continue
            # Skip lines with URLs, IPs, file paths
            if any(x in sentence for x in ['http', '://', '\\', '.fr/', 'C:\\', 'D:\\']):
                continue
            examples.append(build_clean_example(sentence))
            n_clean += 1

    return examples, n_error, n_clean


def main():
    if not ZIP_PATH.exists():
        print(f"ERROR: {ZIP_PATH} not found.")
        return

    print(f"Reading {ZIP_PATH}...")
    with zipfile.ZipFile(ZIP_PATH) as z:
        with z.open("grammalecte/fr/gc_test.txt") as f:
            content = f.read().decode("utf-8", errors="ignore")

    examples, n_error, n_clean = parse_test_file(content)

    print(f"Parsed:")
    print(f"  Error examples:  {n_error:,}")
    print(f"  Clean examples:  {n_clean:,}")
    print(f"  Total:           {len(examples):,}")

    # Sample preview
    print("\nSample error example:")
    for ex in examples[:3]:
        msg = ex["messages"]
        print(f"  Input:    {msg[1]['content'][:80]}")
        try:
            errors = json.loads(msg[2]['content'])['errors']
            if errors:
                e = errors[0]
                print(f"  Error:    '{e['original']}' → '{e['correction']}' [{e['type']}]")
        except Exception:
            pass
    print()

    existing = sum(1 for _ in EXAMPLES_PATH.open(encoding="utf-8")) if EXAMPLES_PATH.exists() else 0

    with EXAMPLES_PATH.open("a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"✓ Appended {len(examples):,} Grammalecte examples to {EXAMPLES_PATH}")
    print(f"  Previous total: {existing:,}")
    print(f"  New total:      {existing + len(examples):,}")
    print()
    print("Next: make dataset → make train")


if __name__ == "__main__":
    main()
