#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/medium.toml}"
DATASET="${DATASET:-HuggingFaceFW/fineweb-edu}"
HF_CONFIG="${HF_CONFIG:-sample-10BT}"
VOCAB_SIZE="${VOCAB_SIZE:-16384}"
TRAIN_DOCS="${TRAIN_DOCS:-500000}"
VAL_DOCS="${VAL_DOCS:-5000}"
DATA_DIR="${DATA_DIR:-/content/kestrel-data}"

if [ -z "${RUN_DIR:-}" ]; then
  if [ -d /content/drive/MyDrive ]; then
    RUN_DIR="/content/drive/MyDrive/kestrel/medium-1"
  else
    RUN_DIR="runs/medium-1"
  fi
fi

export TOKENIZERS_PARALLELISM=false

python -m pip install -q --no-deps -e .
python -c "import datasets" 2>/dev/null || python -m pip install -q datasets

if [ ! -f "$DATA_DIR/raw/train.jsonl" ] || [ ! -f "$DATA_DIR/raw/validation.jsonl" ]; then
  python scripts/fetch_hf.py \
    --dataset "$DATASET" \
    --config "$HF_CONFIG" \
    --train-docs "$TRAIN_DOCS" \
    --val-docs "$VAL_DOCS" \
    --output "$DATA_DIR/raw"
fi

if [ ! -f "$DATA_DIR/tokenizer/tokenizer.json" ]; then
  python -m kestrel tokenizer \
    --train "$DATA_DIR/raw/train.jsonl" \
    --output "$DATA_DIR/tokenizer" \
    --vocab-size "$VOCAB_SIZE"
fi

if [ ! -f "$DATA_DIR/encoded/manifest.json" ]; then
  python -m kestrel prepare \
    --train "$DATA_DIR/raw/train.jsonl" \
    --validation "$DATA_DIR/raw/validation.jsonl" \
    --tokenizer "$DATA_DIR/tokenizer" \
    --output "$DATA_DIR/encoded"
fi

RESUME_FLAG=()
if [ -f "$RUN_DIR/latest.pt" ]; then
  RESUME_FLAG=(--resume)
fi

python -m kestrel train \
  --config "$CONFIG" \
  --data "$DATA_DIR/encoded" \
  --output "$RUN_DIR" \
  --device auto \
  "${RESUME_FLAG[@]}"
