#!/usr/bin/env bash
# Start (or resume) the byte-level H-Net training run on a Colab GPU, with the colab CLI.
#
#   scripts/colab_hnet.sh L4      # or A100
#   CONFIG=configs/hnet_titans_anchor.toml RUN_NAME=hnet-titans-anchor-1 STOP_AFTER=1500 scripts/colab_hnet.sh L4
#
# The run uses the committed HEAD, not the working tree. Data, checkpoints and a
# console log go to Google Drive, so a run that Colab reclaims resumes from
# latest.pt on the next start, on either GPU (both use the same config).
# The first start prepares byte-level data on Drive, which takes a while.
set -euo pipefail

GPU="${1:-}"
case "$GPU" in
  L4|A100) ;;
  *) echo "usage: scripts/colab_hnet.sh L4|A100" >&2; exit 2 ;;
esac
COLAB="${COLAB:-colab}"
SESSION="${SESSION:-kestrel-hnet-$(echo "$GPU" | tr '[:upper:]' '[:lower:]')}"
CONFIG="${CONFIG:-configs/hnet_titans.toml}"
RUN_NAME="${RUN_NAME:-hnet-titans-1}"
DRIVE=/content/drive/MyDrive/kestrel

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

echo "Run $RUN_NAME on $name, commit $(git rev-parse --short HEAD)."
echo "Console log: $DRIVE/$RUN_NAME.console.log. Stopping this command does not stop the run."
echo "Check with: $COLAB status -s $SESSION. Stop the VM with: $COLAB stop -s $SESSION"
# The cell keeps the kernel busy, so Colab does not see the session as idle.
"$COLAB" exec -s "$SESSION" --timeout 0 <<EOF
!rm -rf /content/kestrel && mkdir -p /content/kestrel && tar -xzf /content/kestrel.tar.gz -C /content/kestrel
!cd /content/kestrel && CONFIG=$CONFIG VOCAB_SIZE=263 DATA_DIR=$DRIVE/data-bytes RAW_DIR=$DRIVE/data/raw RUN_DIR=$DRIVE/$RUN_NAME WANDB_PROJECT=${WANDB_PROJECT:-} STOP_AFTER=${STOP_AFTER:-} bash scripts/colab_run.sh 2>&1 | tee -a $DRIVE/$RUN_NAME.console.log
EOF
