"""
Script 06 — Evaluate trained Babel on the test split.

Metrics per error type:
  - Precision: % of model flags that are correct
  - Recall: % of real errors that are caught
  - F1

Also measures:
  - False positive rate on clean examples
  - JSON parse success rate

Writes report to logs/eval_TIMESTAMP.json
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent
cfg  = yaml.safe_load((ROOT / "config.yaml").read_text())

TEST_PATH    = ROOT / cfg["data"]["test_out"]
ADAPTER_PATH = ROOT / cfg["training"]["adapter_path"] if "adapter_path" in cfg.get("training", {}) else ROOT / "outputs/babel-v1"
MODEL_NAME   = cfg["model"]["name"]
MAX_SEQ_LEN  = cfg["model"]["max_seq_length"]


def load_model():
    try:
        from mlx_lm import load, generate
        print(f"Loading model: {MODEL_NAME}")
        print(f"Adapter:       {ADAPTER_PATH}")
        model, tokenizer = load(MODEL_NAME, adapter_path=str(ADAPTER_PATH))
        return model, tokenizer, generate
    except ImportError:
        print("ERROR: mlx_lm not installed. Run: make setup")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading model: {e}")
        sys.exit(1)


def predict(model, tokenizer, generate_fn, user_text: str, max_tokens: int = 300) -> str:
    messages = [
        {"role": "system", "content": cfg["system_prompt"].strip()},
        {"role": "user",   "content": user_text}
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        response = generate_fn(
            model, tokenizer, prompt=prompt,
            max_tokens=max_tokens, verbose=False
        )
        # Strip everything before the first {
        start = response.find('{')
        if start == -1:
            return '{"errors": []}'
        return response[start:]
    except Exception as e:
        return '{"errors": []}'


def parse_errors(raw: str) -> list[dict]:
    try:
        parsed = json.loads(raw)
        return parsed.get("errors", [])
    except Exception:
        return None  # parse failure


def get_ground_truth(example: dict) -> tuple[list[dict], bool]:
    try:
        assistant_msg = example["messages"][-1]["content"]
        parsed        = json.loads(assistant_msg)
        errors        = parsed.get("errors", [])
        return errors, True
    except Exception:
        return [], False


def normalize_error(error: dict) -> tuple[str, str]:
    return (
        str(error.get("original", "")).strip(),
        str(error.get("correction", "")).strip(),
    )


def error_pairs(errors: list[dict]) -> set[tuple[str, str]]:
    return {
        pair for pair in (normalize_error(error) for error in errors)
        if pair[0] and pair[1] and pair[0] != pair[1]
    }


def main():
    model, tokenizer, generate_fn = load_model()

    print(f"Loading test set from {TEST_PATH}...")
    test_examples = []
    with TEST_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    test_examples.append(json.loads(line))
                except Exception:
                    continue

    print(f"Evaluating {len(test_examples)} test examples...\n")

    # Metrics containers
    by_type: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0})
    overall = {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "parse_ok": 0, "parse_fail": 0}

    t0 = time.perf_counter()

    for i, ex in enumerate(test_examples):
        user_msg    = ex["messages"][1]["content"]
        gt_errors, gt_ok = get_ground_truth(ex)
        if not gt_ok:
            continue

        gt_pairs    = error_pairs(gt_errors)
        is_clean    = len(gt_pairs) == 0

        raw     = predict(model, tokenizer, generate_fn, user_msg)
        pred    = parse_errors(raw)

        if pred is None:
            overall["parse_fail"] += 1
            pred = []
        else:
            overall["parse_ok"] += 1

        pred_pairs = error_pairs(pred)

        if is_clean:
            if not pred_pairs:
                overall["tn"] += 1
                by_type["clean"]["tn"] += 1
            else:
                overall["fp"] += 1
                for error in pred:
                    by_type[error.get("type", "unknown")]["fp"] += 1
        else:
            gt_by_pair = {normalize_error(error): error.get("type", "unknown") for error in gt_errors}
            pred_by_pair = {normalize_error(error): error.get("type", "unknown") for error in pred}

            # Count exact original -> correction matches, not just broad categories.
            for gt_pair, gt_type in gt_by_pair.items():
                if gt_pair in pred_pairs:
                    overall["tp"] += 1
                    by_type[gt_type]["tp"] += 1
                else:
                    overall["fn"] += 1
                    by_type[gt_type]["fn"] += 1

            for pred_pair, pred_type in pred_by_pair.items():
                if pred_pair not in gt_pairs:
                    overall["fp"] += 1
                    by_type[pred_type]["fp"] += 1

        if (i + 1) % 100 == 0:
            elapsed = time.perf_counter() - t0
            tp, fp, fn = overall["tp"], overall["fp"], overall["fn"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            print(f"  [{i+1:>5}/{len(test_examples)}]  P={p:.3f}  R={r:.3f}  "
                  f"parse_ok={overall['parse_ok']}  {elapsed:.0f}s")

    elapsed = time.perf_counter() - t0

    # Compute overall metrics
    tp, fp, fn, tn = overall["tp"], overall["fp"], overall["fn"], overall["tn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    clean_total = tn + fp
    fp_rate   = fp / max(clean_total, 1)

    print("\n" + "═" * 60)
    print("  BABEL EVALUATION RESULTS")
    print("═" * 60)
    print(f"  Precision:        {precision:.4f}  ({tp} TP / {tp + fp} flagged)")
    print(f"  Recall:           {recall:.4f}  ({tp} TP / {tp + fn} real errors)")
    print(f"  F1:               {f1:.4f}")
    print(f"  FP rate (clean):  {fp_rate:.4f}  ({fp} FP / {clean_total} clean)")
    print(f"  JSON parse OK:    {overall['parse_ok']} / {overall['parse_ok'] + overall['parse_fail']}")
    print(f"  Elapsed:          {elapsed:.0f}s")

    print("\n  Per-type breakdown:")
    print(f"  {'Type':<40} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("  " + "-" * 75)
    for t, m in sorted(by_type.items(), key=lambda x: x[0]):
        tp_t = m["tp"]; fp_t = m["fp"]; fn_t = m["fn"]
        p_t = tp_t / (tp_t + fp_t) if (tp_t + fp_t) > 0 else 0
        r_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
        f_t = 2 * p_t * r_t / (p_t + r_t) if (p_t + r_t) > 0 else 0
        print(f"  {t:<40} {p_t:>6.3f} {r_t:>6.3f} {f_t:>6.3f} {tp_t:>5} {fp_t:>5} {fn_t:>5}")

    print("═" * 60)

    # Write report
    report = {
        "overall": {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "fp_rate_on_clean": round(fp_rate, 4),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "parse_ok":   overall["parse_ok"],
            "parse_fail": overall["parse_fail"],
            "elapsed_s":  round(elapsed, 1),
        },
        "by_type": {
            t: {
                "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
                "precision": round(m["tp"] / max(m["tp"] + m["fp"], 1), 4),
                "recall":    round(m["tp"] / max(m["tp"] + m["fn"], 1), 4),
            }
            for t, m in by_type.items()
        }
    }

    ts        = time.strftime("%Y%m%d_%H%M%S")
    log_path  = ROOT / "logs" / f"eval_{ts}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n  Report saved: {log_path}")


if __name__ == "__main__":
    main()
