#!/usr/bin/env bash
# Run the memory pilots (configs/pilot_memory_*.toml) one after another on a Colab
# GPU, with the colab CLI. They compare decay toward 0 (control, as in
# titans-loop-1) with decay toward W_init (anchor, anchor_bias) on TinyStories.
#
#   scripts/colab_pilot.sh L4      # or A100
#   WANDB_PROJECT=kestrel scripts/colab_pilot.sh L4
#
# The run uses the committed HEAD, not the working tree. Data, checkpoints and a
# console log go to Google Drive, so a run that Colab reclaims resumes from
# latest.pt on the next start. The first start fetches TinyStories and prepares
# it in DATA_DIR on Drive, which takes a while. On a restart, a finished pilot
# stops with "No steps remain" and the next pilot starts.
set -euo pipefail

GPU="${1:-}"
case "$GPU" in
  L4|A100) ;;
  *) echo "usage: scripts/colab_pilot.sh L4|A100" >&2; exit 2 ;;
esac
COLAB="${COLAB:-colab}"
SESSION="${SESSION:-kestrel-pilot-$(echo "$GPU" | tr '[:upper:]' '[:lower:]')}"
PILOTS="${PILOTS:-control anchor anchor_bias}"
DRIVE=/content/drive/MyDrive/kestrel
DATA_DIR="${DATA_DIR:-$DRIVE/data-tinystories}"
LOG=$DRIVE/pilot-memory.console.log

cd "$(git rev-parse --show-toplevel)"
if ! git diff --quiet HEAD --; then
  echo "Commit your changes first: the run uses HEAD." >&2
  exit 1
fi

if ! "$COLAB" sessions </dev/null 2>/dev/null | grep -qw "$SESSION"; then
  "$COLAB" new -s "$SESSION" --gpu "$GPU" </dev/null
fi
# An unknown or unavailable accelerator can give a different GPU. Check it.
name=$("$COLAB" ssh -s "$SESSION" -- 'nvidia-smi --query-gpu=name --format=csv,noheader' </dev/null)
if [[ "$name" != *"$GPU"* ]]; then
  echo "Session $SESSION has '$name', not $GPU. Stop it with: $COLAB stop -s $SESSION" >&2
  exit 1
fi
if ! "$COLAB" ssh -s "$SESSION" -- 'test -d /content/drive/MyDrive' </dev/null; then
  echo "Mount Google Drive (it needs you in the browser), then run this script again:" >&2
  echo "  $COLAB drivemount -s $SESSION" >&2
  exit 1
fi

archive="$(mktemp -d)/kestrel.tar.gz"
trap 'rm -rf "$(dirname "$archive")"' EXIT
git archive --format=tar.gz -o "$archive" HEAD
"$COLAB" upload -s "$SESSION" "$archive" /content/kestrel.tar.gz </dev/null

echo "Pilots ($PILOTS) on $name, commit $(git rev-parse --short HEAD)."
echo "Runs: $DRIVE/pilot-memory-<pilot>-1. Console log: $LOG. Stopping this command does not stop the runs."
echo "Check with: $COLAB status -s $SESSION. Stop the VM with: $COLAB stop -s $SESSION"
# The cell keeps the kernel busy, so Colab does not see the session as idle.
"$COLAB" exec -s "$SESSION" --timeout 0 <<EOF
!rm -rf /content/kestrel && mkdir -p /content/kestrel && tar -xzf /content/kestrel.tar.gz -C /content/kestrel
!cd /content/kestrel && for p in $PILOTS; do CONFIG=configs/pilot_memory_\$p.toml DATASET=roneneldan/TinyStories HF_CONFIG=default VOCAB_SIZE=4096 TRAIN_DOCS=500000 DATA_DIR=$DATA_DIR RUN_DIR=$DRIVE/pilot-memory-\$p-1 WANDB_PROJECT=${WANDB_PROJECT:-} bash scripts/colab_run.sh || echo "pilot \$p stopped"; done 2>&1 | tee -a $LOG
EOF
