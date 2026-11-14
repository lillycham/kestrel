"""Stream a Hugging Face dataset split into Kestrel train/validation JSONL."""

import argparse
import json
from pathlib import Path

from kestrel.data import SPECIAL


def main():
    parser = argparse.ArgumentParser(description="Stream HF text dataset to Kestrel JSONL")
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--config", default="sample-10BT")
    parser.add_argument("--split", default="train")
    parser.add_argument("--field", default="text")
    parser.add_argument("--output", default="data/raw")
    parser.add_argument("--train-docs", type=int, default=500_000)
    parser.add_argument("--val-docs", type=int, default=5_000)
    parser.add_argument("--min-chars", type=int, default=64)
    args = parser.parse_args()

    from datasets import load_dataset

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    train_path = out / "train.jsonl"
    val_path = out / "validation.jsonl"

    stream = load_dataset(args.dataset, args.config, split=args.split, streaming=True)
    written_val = 0
    written_train = 0
    skipped = 0

    with val_path.open("w", encoding="utf-8") as val_f, train_path.open("w", encoding="utf-8") as train_f:
        for row in stream:
            text = row.get(args.field)
            if not isinstance(text, str) or len(text) < args.min_chars:
                skipped += 1
                continue
            if any(tok in text for tok in SPECIAL):
                skipped += 1
                continue
            line = json.dumps({"text": text}, ensure_ascii=False) + "\n"
            if written_val < args.val_docs:
                val_f.write(line)
                written_val += 1
            elif written_train < args.train_docs:
                train_f.write(line)
                written_train += 1
            else:
                break

    print(json.dumps({
        "dataset": args.dataset,
        "config": args.config,
        "train_docs": written_train,
        "val_docs": written_val,
        "skipped": skipped,
        "output": str(out),
    }))


if __name__ == "__main__":
    main()
