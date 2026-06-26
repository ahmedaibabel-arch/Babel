"""
Script 07 — Run Babel interactively as a French corrector.

Usage:
    python scripts/07_run_babel.py
    python scripts/07_run_babel.py --text "Elle c'est rendu compte trop tard."
    python scripts/07_run_babel.py --adapter outputs/babel-v2 --text "Elle c'est rendu compte trop tard."

Loads the fine-tuned Gemma 3 4B + LoRA adapter.
Accepts French text and returns JSON corrections + human-readable summary.
"""

import argparse
import json
import sys
import time
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent
cfg  = yaml.safe_load((ROOT / "config.yaml").read_text())

MODEL_NAME   = cfg["model"]["name"]
DEFAULT_ADAPTER_PATH = ROOT / "outputs/babel-v2"
SYSTEM_PROMPT = cfg["system_prompt"].strip()

SEVERITY_COLORS = {
    "homophone":     "\033[91m",   # red
    "accent":        "\033[93m",   # yellow
    "typographie":   "\033[94m",   # blue
    "accord":        "\033[95m",   # purple
    "conjugaison":   "\033[92m",   # green
    "clean":         "\033[92m",   # green
}
RESET = "\033[0m"


def resolve_adapter_path(value: str | None) -> Path:
    adapter_path = Path(value) if value else DEFAULT_ADAPTER_PATH
    if not adapter_path.is_absolute():
        adapter_path = ROOT / adapter_path
    return adapter_path


def log_progress(message: str):
    print(f"babel-cli {message}", file=sys.stderr, flush=True)


def load_model(adapter_path: Path, quiet: bool = False):
    started = time.monotonic()
    log_progress(f"model-load-start model={MODEL_NAME} adapter={adapter_path}")
    try:
        from mlx_lm import load, generate
        if not quiet:
            print(f"Loading Babel ({MODEL_NAME} + LoRA adapter: {adapter_path})...", flush=True)
        model, tokenizer = load(MODEL_NAME, adapter_path=str(adapter_path))
        if not quiet:
            print("Ready.\n")
        log_progress(f"model-load-end duration_ms={int((time.monotonic() - started) * 1000)}")
        return model, tokenizer, generate
    except FileNotFoundError:
        print(f"ERROR: Adapter not found at {adapter_path}", file=sys.stderr)
        print("Run 'make train' first.")
        sys.exit(1)
    except ImportError:
        print("ERROR: mlx_lm not installed. Run: make setup", file=sys.stderr)
        sys.exit(1)


def correct(model, tokenizer, generate_fn, text: str, max_tokens: int = 512) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Texte: {text}"}
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    raw = generate_fn(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    start = raw.find('{')
    if start == -1:
        return {"errors": [], "raw": raw}
    try:
        result = json.loads(raw[start:])
        result["errors"] = resolve_positions(text, result.get("errors", []))
        return result
    except Exception:
        return {"errors": [], "raw": raw[start:]}


def resolve_positions(text: str, errors: list[dict]) -> list[dict]:
    resolved = []
    search_from = 0
    for error in errors:
        original = str(error.get("original", "")).strip()
        correction = str(error.get("correction", "")).strip()
        if not original or not correction or original == correction:
            continue

        pos = text.find(original, search_from)
        if pos == -1:
            pos = text.find(original)
        if pos == -1:
            continue

        updated = dict(error)
        updated["original"] = original
        updated["correction"] = correction
        updated["start"] = pos
        updated["end"] = pos + len(original)
        resolved.append(updated)
        search_from = pos + len(original)
    return resolved


def print_results(text: str, result: dict):
    errors = result.get("errors", [])
    print(f"\nInput:  {text}")
    print(f"Errors: {len(errors)}")
    print()

    if not errors:
        print(f"  {SEVERITY_COLORS['clean']}✓ Aucune erreur détectée.{RESET}")
    else:
        for i, e in enumerate(errors, 1):
            color   = next((v for k, v in SEVERITY_COLORS.items() if e.get("type","").startswith(k)), "")
            conf    = int(e.get("confidence", 0) * 100)
            print(f"  {color}[{i}] {e.get('type','?')}{RESET}  conf={conf}%")
            print(f"       Original:   « {e.get('original','')} »")
            print(f"       Correction: « {e.get('correction','')} »")
            print(f"       {e.get('explanation','')}")
            print()

    # Show corrected text
    if errors:
        corrected = text
        for e in sorted(errors, key=lambda x: x.get("start", 0), reverse=True):
            s = e.get("start", 0)
            en = e.get("end", s)
            c  = e.get("correction", "")
            corrected = corrected[:s] + c + corrected[en:]
        print(f"Corrected: {corrected}")

    print()
    return result


def interactive(model, tokenizer, generate_fn):
    print("Babel — Correcteur français")
    print("Entrez du texte français. 'q' pour quitter.")
    print("─" * 50)
    while True:
        try:
            text = input("\nTexte> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAu revoir.")
            break
        if text.lower() in ("q", "quit", "exit"):
            print("Au revoir.")
            break
        if not text:
            continue
        result = correct(model, tokenizer, generate_fn, text)
        print_results(text, result)


def run_jsonl_batch(model, tokenizer, generate_fn, max_tokens: int):
    results = []
    batch_started = time.monotonic()
    log_progress(f"batch-start max_tokens={max_tokens}")
    count = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = str(item.get("text", ""))
        item_id = item.get("id")
        count += 1
        item_started = time.monotonic()
        log_progress(f"item-start index={count} id={item_id} chars={len(text)}")
        result = correct(model, tokenizer, generate_fn, text, max_tokens=max_tokens) if text.strip() else {"errors": []}
        elapsed_ms = int((time.monotonic() - item_started) * 1000)
        raw_chars = len(str(result.get("raw", "")))
        log_progress(
            f"item-end index={count} id={item_id} chars={len(text)} "
            f"errors={len(result.get('errors', []))} raw_chars={raw_chars} duration_ms={elapsed_ms}"
        )
        results.append({"id": item_id, "result": result})
    log_progress(f"batch-end items={count} duration_ms={int((time.monotonic() - batch_started) * 1000)}")
    print(json.dumps({"items": results}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Babel — French corrector")
    parser.add_argument("--text", help="Text to correct (interactive mode if omitted)")
    parser.add_argument(
        "--adapter",
        default=str(DEFAULT_ADAPTER_PATH.relative_to(ROOT)),
        help="LoRA adapter directory. Relative paths are resolved from the Babel project root.",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON only")
    parser.add_argument("--jsonl-batch", action="store_true", help="Read JSONL {id,text} records from stdin and output one JSON batch")
    parser.add_argument("--max-tokens", type=int, default=512, help="Maximum generation tokens per correction request")
    args = parser.parse_args()

    adapter_path = resolve_adapter_path(args.adapter)
    if not adapter_path.exists():
        print(f"ERROR: Adapter not found at {adapter_path}", file=sys.stderr)
        sys.exit(1)

    model, tokenizer, generate_fn = load_model(adapter_path, quiet=args.json)

    if args.jsonl_batch:
        run_jsonl_batch(model, tokenizer, generate_fn, max_tokens=args.max_tokens)
    elif args.text:
        result = correct(model, tokenizer, generate_fn, args.text, max_tokens=args.max_tokens)
        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print_results(args.text, result)
    else:
        interactive(model, tokenizer, generate_fn)


if __name__ == "__main__":
    main()
