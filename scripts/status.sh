#!/usr/bin/env bash
# Status report — run at any point to see where you are.
# Usage: make status  OR  bash scripts/status.sh

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "═══════════════════════════════════════════"
echo "  Babel — Status Report  $(date)"
echo "═══════════════════════════════════════════"
echo ""

echo "── Data Files ──────────────────────────────"
for f in \
    "data/sentences/clean_sentences.jsonl" \
    "data/examples/all_examples.jsonl" \
    "data/splits/train.jsonl" \
    "data/splits/valid.jsonl" \
    "data/splits/test.jsonl"
do
    path="$ROOT/$f"
    if [ -f "$path" ]; then
        lines=$(wc -l < "$path" | tr -d ' ')
        size=$(du -sh "$path" | cut -f1)
        printf "  %-45s %8s lines  %s\n" "$f" "$lines" "$size"
    else
        printf "  %-45s  (not created yet)\n" "$f"
    fi
done
echo ""

echo "── Model / Adapter ─────────────────────────"
adapter="$ROOT/outputs/babel-v1"
if [ -d "$adapter" ]; then
    size=$(du -sh "$adapter" | cut -f1)
    files=$(ls "$adapter" | wc -l | tr -d ' ')
    echo "  Adapter: $adapter  ($size, $files files)"
else
    echo "  Adapter: not trained yet"
fi
echo ""

echo "── Training Log ────────────────────────────"
latest_log=$(ls -t "$ROOT/logs"/training_*.log 2>/dev/null | head -1)
if [ -n "$latest_log" ]; then
    echo "  Log: $latest_log"
    echo ""
    echo "  Last 20 lines:"
    tail -20 "$latest_log" | sed 's/^/    /'
else
    echo "  No training log found."
fi
echo ""

echo "── Eval Reports ────────────────────────────"
latest_eval=$(ls -t "$ROOT/logs"/eval_*.json 2>/dev/null | head -1)
if [ -n "$latest_eval" ]; then
    echo "  Latest: $latest_eval"
    python3 -c "
import json, sys
try:
    d = json.load(open('$latest_eval'))
    o = d['overall']
    print(f\"  Precision: {o['precision']:.4f}\")
    print(f\"  Recall:    {o['recall']:.4f}\")
    print(f\"  F1:        {o['f1']:.4f}\")
    print(f\"  FP rate:   {o['fp_rate_on_clean']:.4f}\")
except Exception as e:
    print(f'  (could not parse: {e})')
" 2>/dev/null
else
    echo "  No eval report found."
fi
echo ""

echo "── Disk Usage ──────────────────────────────"
printf "  %-30s  %s\n" "Babel project" "$(du -sh "$ROOT" 2>/dev/null | cut -f1)"
hf_cache="$HOME/.cache/huggingface"
[ -d "$hf_cache" ] && printf "  %-30s  %s\n" "HuggingFace cache" "$(du -sh "$hf_cache" 2>/dev/null | cut -f1)"
echo ""
echo "═══════════════════════════════════════════"
