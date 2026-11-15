"""Measure arithmetic pass rates per difficulty bucket, for problems seen and not seen in training.

The prompts use the step-by-step templates of scripts/build_anneal_mix.py. Each
problem gets a group of samples, as GRPO would. The useful number for RL is the
fraction of "mixed" groups: groups with some right and some wrong answers.
Groups that are all right or all wrong give GRPO no signal.

    python scripts/arith_sweep.py --run runs/titans-loop-anneal-1 --seen-from data/anneal-raw/train.jsonl
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kestrel.arith import BUCKETS, bucket, draw_new, render, seen_problems  # noqa: E402
from kestrel.grpo import rollout  # noqa: E402
from kestrel.train import _load_for_inference  # noqa: E402


def pick(name, split, count, seen, rng):
    if split == "seen":
        pool = sorted(p for p in seen if p[0] == BUCKETS[name] and bucket(p) == name)
        return rng.sample(pool, min(count, len(pool)))
    chosen = set()
    while len(chosen) < count:
        chosen.add(draw_new(name, rng, seen | chosen))
    return sorted(chosen)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--seen-from", help="Training JSONL; problems in it form the 'seen' split")
    parser.add_argument("--problems", type=int, default=24, help="Problems per bucket and split")
    parser.add_argument("--group", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=80)
    parser.add_argument("--max-rows", type=int, default=256, help="Rows per generate batch")
    parser.add_argument("--buckets", nargs="*", default=list(BUCKETS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", help="Write per-sample results as JSONL")
    args = parser.parse_args()

    started = time.perf_counter()
    seen = seen_problems(args.seen_from)
    if args.seen_from:
        print(f"{len(seen)} distinct template problems in {args.seen_from} ({time.perf_counter() - started:.0f}s)", flush=True)
    splits = ["seen", "unseen"] if seen else ["unseen"]

    model, tokenizer, device, bad_words_ids = _load_for_inference(args.run, args.device, args.seed)
    rng = random.Random(args.seed)
    out = open(args.output, "w", encoding="utf-8") if args.output else None
    rows = []
    tokens_made = 0
    decode_time = 0.0
    for name in args.buckets:
        for split in splits:
            problems = pick(name, split, args.problems, seen, rng)
            t0 = time.perf_counter()
            groups = rollout(model, tokenizer, problems, args.group, args.temperature,
                             args.max_new_tokens, args.max_rows, bad_words_ids)
            decode_time += time.perf_counter() - t0
            correct = []
            for g in groups:
                tokens_made += int(g["mask"].sum())
                correct.append(int(g["rewards"].sum()))
                if out:
                    line, answer = render(g["problem"])
                    for text, reward in zip(g["texts"], g["rewards"]):
                        out.write(json.dumps({"bucket": name, "split": split, "prompt": line, "answer": answer,
                                              "correct": bool(reward), "completion": text}) + "\n")
            n, gs = max(len(correct), 1), args.group
            row = {
                "bucket": name, "split": split, "problems": len(correct),
                "pass@1": sum(correct) / (n * gs),
                "mixed": sum(0 < c < gs for c in correct) / n,
                "all_wrong": sum(c == 0 for c in correct) / n,
                "all_right": sum(c == gs for c in correct) / n,
            }
            rows.append(row)
            print(f"{name:16} {split:6} n={row['problems']:3}  pass@1={row['pass@1']:.2f}  mixed={row['mixed']:.2f}  "
                  f"all_wrong={row['all_wrong']:.2f}  all_right={row['all_right']:.2f}", flush=True)
    if out:
        out.close()
    print(json.dumps({"rows": rows, "tokens_per_second": tokens_made / max(decode_time, 1e-9),
                      "group": args.group, "temperature": args.temperature}))


if __name__ == "__main__":
    main()
