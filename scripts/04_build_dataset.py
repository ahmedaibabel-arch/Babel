"""
Script 04 — Build stratified train/valid/test splits from all_examples.jsonl.

Stratifies by error type so all types appear in every split.
Formats examples for MLX-LM LoRA fine-tuning (messages format).

Output:
  data/splits/train.jsonl  (90%)
  data/splits/valid.jsonl  (5%)
  data/splits/test.jsonl   (5%)
"""

import copy
import json
import random
from collections import defaultdict
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent
cfg  = yaml.safe_load((ROOT / "config.yaml").read_text())

SRC   = ROOT / cfg["data"]["examples_out"]
TRAIN = ROOT / cfg["data"]["train_out"]
VALID = ROOT / cfg["data"]["valid_out"]
TEST  = ROOT / cfg["data"]["test_out"]
GOLD  = ROOT / "data/gold/tome3_probe.jsonl"

TRAIN_RATIO = cfg["data"]["train_ratio"]
VALID_RATIO = cfg["data"]["valid_ratio"]
SEED        = cfg["corruption"]["seed"]

random.seed(SEED)

for p in [TRAIN, VALID, TEST]:
    p.parent.mkdir(parents=True, exist_ok=True)


def get_error_type(example: dict) -> str:
    try:
        errors = json.loads(example["messages"][-1]["content"]).get("errors", [])
        if errors:
            return errors[0].get("type", "unknown")
        return "clean"
    except Exception:
        return "unknown"


def assistant_errors(example: dict) -> list[dict] | None:
    try:
        return json.loads(example["messages"][-1]["content"]).get("errors", [])
    except Exception:
        return None


def with_errors(example: dict, errors: list[dict]) -> dict:
    updated = copy.deepcopy(example)
    updated["messages"][-1]["content"] = json.dumps({"errors": errors}, ensure_ascii=False)
    return updated


def is_valid_error(error: dict) -> bool:
    original = str(error.get("original", "")).strip()
    correction = str(error.get("correction", "")).strip()
    return bool(original and correction and original != correction)


def clean_example(example: dict) -> dict | None:
    errors = assistant_errors(example)
    if errors is None:
        return None
    if not errors:
        return example

    cleaned = [error for error in errors if is_valid_error(error)]
    if not cleaned:
        return None
    return with_errors(example, cleaned)


def load_gold_examples() -> list[dict]:
    if not GOLD.exists():
        return []
    examples = []
    with GOLD.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except Exception:
                continue
            cleaned = clean_example(example)
            if cleaned is not None:
                examples.append(cleaned)
    return examples


def main():
    print(f"Loading examples from {SRC}...")
    examples = []
    dropped_parse = 0
    dropped_noop = 0
    with SRC.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    raw = json.loads(line)
                except Exception:
                    dropped_parse += 1
                    continue
                cleaned = clean_example(raw)
                if cleaned is None:
                    dropped_noop += 1
                    continue
                examples.append(cleaned)

    print(f"Loaded {len(examples):,} examples.")
    print(f"Dropped parse failures: {dropped_parse:,}")
    print(f"Dropped no-op/malformed error examples: {dropped_noop:,}")

    gold_examples = load_gold_examples()
    if gold_examples:
        print(f"Loaded {len(gold_examples):,} Tome 3 gold examples from {GOLD}.")

    # Stratify by error type
    by_type: dict[str, list] = defaultdict(list)
    for ex in examples:
        t = get_error_type(ex)
        by_type[t].append(ex)

    print("Distribution by type:")
    for t, items in sorted(by_type.items(), key=lambda x: -len(x[1])):
        print(f"  {t:<40} {len(items):>7,}")

    train_examples, valid_examples, test_examples = [], [], []

    for t, items in by_type.items():
        random.shuffle(items)
        n     = len(items)
        n_tr  = max(1, int(n * TRAIN_RATIO))
        n_va  = max(1, int(n * VALID_RATIO))
        n_te  = max(1, n - n_tr - n_va)
        train_examples.extend(items[:n_tr])
        valid_examples.extend(items[n_tr:n_tr + n_va])
        test_examples.extend(items[n_tr + n_va:n_tr + n_va + n_te])

    random.shuffle(train_examples)
    random.shuffle(valid_examples)
    random.shuffle(test_examples)

    # Keep Tome 3 regression probes out of training. They are hard gates for
    # checkpoint selection, so put a copy in validation and test.
    if gold_examples:
        valid_examples.extend(copy.deepcopy(gold_examples))
        test_examples.extend(copy.deepcopy(gold_examples))
        random.shuffle(valid_examples)
        random.shuffle(test_examples)

    def write(path: Path, data: list):
        with path.open("w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    write(TRAIN, train_examples)
    write(VALID, valid_examples)
    write(TEST,  test_examples)

    print(f"\n✓ Done.")
    print(f"  Train: {len(train_examples):>8,}  → {TRAIN}")
    print(f"  Valid: {len(valid_examples):>8,}  → {VALID}")
    print(f"  Test:  {len(test_examples):>8,}  → {TEST}")
    total = len(train_examples) + len(valid_examples) + len(test_examples)
    print(f"  Total: {total:>8,}")


if __name__ == "__main__":
    main()
