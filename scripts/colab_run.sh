#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-configs/medium.toml}"
DATASET="${DATASET:-HuggingFaceFW/fineweb-edu}"
HF_CONFIG="${HF_CONFIG:-sample-10BT}"
VOCAB_SIZE="${VOCAB_SIZE:-16384}"
TRAIN_DOCS="${TRAIN_DOCS:-500000}"
VAL_DOCS="${VAL_DOCS:-5000}"
TOKENIZER_DOCS="${TOKENIZER_DOCS:-100000}"
PREP_ONLY="${PREP_ONLY:-0}"

if [ -z "${DATA_DIR:-}" ]; then
  if [ -d /content/drive/MyDrive ]; then
    DATA_DIR="/content/drive/MyDrive/kestrel/data"
  else
    DATA_DIR="/content/kestrel-data"
  fi
fi

# Raw JSONL can be shared between tokenizers (for example BPE and bytes).
RAW_DIR="${RAW_DIR:-$DATA_DIR/raw}"

if [ -z "${RUN_DIR:-}" ]; then
  if [ -d /content/drive/MyDrive ]; then
    RUN_DIR="/content/drive/MyDrive/kestrel/medium-1"
  else
    RUN_DIR="runs/medium-1"
  fi
fi

export TOKENIZERS_PARALLELISM=true

python -m pip install -q --no-deps -e .
python -c "import datasets" 2>/dev/null || python -m pip install -q datasets

if [ ! -f "$DATA_DIR/encoded/manifest.json" ]; then
  if [ ! -f "$RAW_DIR/train.jsonl" ] || [ ! -f "$RAW_DIR/validation.jsonl" ]; then
    python scripts/fetch_hf.py \
      --dataset "$DATASET" \
      --config "$HF_CONFIG" \
      --train-docs "$TRAIN_DOCS" \
      --val-docs "$VAL_DOCS" \
      --output "$RAW_DIR"
  fi

  if [ ! -f "$DATA_DIR/tokenizer/tokenizer.json" ]; then
    head -n "$TOKENIZER_DOCS" "$RAW_DIR/train.jsonl" > /tmp/kestrel_tok_train.jsonl
    python -m kestrel tokenizer \
      --train /tmp/kestrel_tok_train.jsonl \
      --output "$DATA_DIR/tokenizer" \
      --vocab-size "$VOCAB_SIZE"
    rm -f /tmp/kestrel_tok_train.jsonl
  fi

  python -m kestrel prepare \
    --train "$RAW_DIR/train.jsonl" \
    --validation "$RAW_DIR/validation.jsonl" \
    --tokenizer "$DATA_DIR/tokenizer" \
    --output "$DATA_DIR/encoded"
fi

if [ "$PREP_ONLY" = "1" ]; then
  exit 0
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
