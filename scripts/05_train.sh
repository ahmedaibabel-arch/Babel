#!/usr/bin/env bash
# Script 05 — Train Babel with MLX LoRA fine-tuning.
# Requires: mlx and mlx-lm installed in .venv
# Runtime: ~10-20 hours on Apple Silicon M3/M4

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv"
LOG="$ROOT/logs/training_$(date +%Y%m%d_%H%M%S).log"
ADAPTER="$ROOT/outputs/babel-v2"

echo "═══════════════════════════════════════════"
echo "  Babel — MLX LoRA Training"
echo "  $(date)"
echo "  Log: $LOG"
echo "═══════════════════════════════════════════"

# Check venv
if [ ! -f "$VENV/bin/python" ]; then
    echo "ERROR: .venv not found. Run: make setup"
    exit 1
fi

source "$VENV/bin/activate"

# Check data splits
for split in train valid test; do
    path="$ROOT/data/splits/$split.jsonl"
    if [ ! -f "$path" ]; then
        echo "ERROR: $path not found. Run: make dataset"
        exit 1
    fi
    count=$(wc -l < "$path")
    echo "  $split: $count examples"
done

mkdir -p "$ROOT/logs" "$ROOT/outputs"

# Check model — will download automatically if not cached
MODEL="mlx-community/gemma-3-4b-it-4bit"
echo ""
echo "Model: $MODEL"
echo "Adapter output: $ADAPTER"
echo ""
echo "Starting training... (press Ctrl+C to pause — checkpoints are saved every 500 iters)"
echo ""

# Batch size 2 (down from 4) — prevents Metal OOM on 16 GB unified memory.
# Previous run crashed at iter 1390 with kIOGPUCommandBufferCallbackErrorOutOfMemory.
python -m mlx_lm lora \
    --model "$MODEL" \
    --train \
    --data "$ROOT/data/splits" \
    --iters 10000 \
    --batch-size 2 \
    --grad-checkpoint \
    --num-layers 16 \
    --learning-rate 2e-4 \
    --val-batches 25 \
    --save-every 500 \
    --max-seq-length 512 \
    --adapter-path "$ADAPTER" \
    2>&1 | tee "$LOG"

echo ""
echo "═══════════════════════════════════════════"
echo "  Training complete."
echo "  Adapter: $ADAPTER"
echo "  Log:     $LOG"
echo "═══════════════════════════════════════════"
