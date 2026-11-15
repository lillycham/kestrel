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
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kestrel.train import _load_for_inference  # noqa: E402

FIRST_LINE = re.compile(
    r"Compute (\d+) ([+-]) (\d+) step by step\."
    r"|Multiply (\d+) \* (\d+) step by step\."
    r"|Evaluate \((\d+) \+ (\d+)\) \* (\d+)\."
    r"|Evaluate (\d+) \* (\d+) \+ (\d+)\."
)
ANSWER = re.compile(r"Answer: (-?\d+)")


def parse(line):
    """Map a template first line to (kind, numbers), or None."""
    m = FIRST_LINE.fullmatch(line)
    if not m:
        return None
    g = m.groups()
    if g[0]:
        return ("add" if g[1] == "+" else "sub"), (int(g[0]), int(g[2]))
    if g[3]:
        return "mul", (int(g[3]), int(g[4]))
    if g[5]:
        return "addmul", (int(g[5]), int(g[6]), int(g[7]))
    return "muladd", (int(g[8]), int(g[9]), int(g[10]))


def render(kind, nums):
    """Return (first line, answer) for a problem."""
    if kind == "add":
        a, b = nums
        return f"Compute {a} + {b} step by step.", a + b
    if kind == "sub":
        a, b = nums
        return f"Compute {a} - {b} step by step.", a - b
    if kind == "mul":
        a, b = nums
        return f"Multiply {a} * {b} step by step.", a * b
    if kind == "addmul":
        a, b, c = nums
        return f"Evaluate ({a} + {b}) * {c}.", (a + b) * c
    a, c, b = nums
    return f"Evaluate {a} * {c} + {b}.", a * c + b


def bucket(kind, nums):
    """Difficulty bucket of a problem, within the ranges of the anneal generator."""
    if kind == "add":
        a, b = nums
        size = "2d" if a < 100 and b < 100 else "3d"
        carry = "carry" if a % 10 + b % 10 >= 10 else "nocarry"
        return f"add_{size}_{carry}"
    if kind == "sub":
        a, b = nums
        return "sub_borrow" if a % 10 < b % 10 else "sub_noborrow"
    return kind


def draw(kind, rng):
    """Draw a problem with the same ranges as gen_arithmetic_doc in build_anneal_mix.py."""
    if kind == "add":
        return rng.randint(10, 999), rng.randint(10, 999)
    if kind == "sub":
        a = rng.randint(20, 999)
        return a, rng.randint(10, a)
    if kind == "mul":
        return rng.randint(11, 99), rng.randint(2, 19)
    a, b, c = rng.randint(2, 40), rng.randint(2, 40), rng.randint(2, 15)
    return (a, b, c) if kind == "addmul" else (a, c, b)


BUCKETS = {
    "add_2d_nocarry": "add", "add_2d_carry": "add", "add_3d_nocarry": "add", "add_3d_carry": "add",
    "sub_noborrow": "sub", "sub_borrow": "sub", "mul": "mul", "addmul": "addmul", "muladd": "muladd",
}


def seen_problems(path):
    seen = set()
    if path is None:
        return seen
    with open(path, encoding="utf-8") as source:
        for line in source:
            if "step by step" not in line and "Evaluate" not in line:
                continue
            first = json.loads(line)["text"].split("\n", 1)[0]
            parsed = parse(first)
            if parsed:
                seen.add(parsed)
    return seen


def pick(name, split, count, seen, rng):
    kind = BUCKETS[name]
    if split == "seen":
        pool = sorted(p for p in seen if p[0] == kind and bucket(*p) == name)
        return rng.sample(pool, min(count, len(pool)))
    chosen = set()
    for _ in range(200_000):
        if len(chosen) == count:
            break
        nums = draw(kind, rng)
        if name == "add_2d_nocarry" or name == "add_2d_carry":
            nums = (nums[0] % 90 + 10, nums[1] % 90 + 10)  # Bias draws towards the rare 2-digit buckets.
        problem = (kind, nums)
        if bucket(*problem) == name and problem not in seen:
            chosen.add(problem)
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
            correct = []
            for kind, nums in problems:
                line, answer = render(kind, nums)
                # Documents follow the EOS of the document before them in the training stream.
                ids = tokenizer(tokenizer.eos_token + line + "\n", return_tensors="pt").input_ids.to(device)
                t0 = time.perf_counter()
                result = model.generate(
                    ids, max_new_tokens=args.max_new_tokens, do_sample=args.temperature > 0,
                    temperature=args.temperature, num_return_sequences=args.group,
                    eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                    bad_words_ids=bad_words_ids,
                )
                decode_time += time.perf_counter() - t0
                completions = result[:, ids.shape[1]:]
                tokens_made += int((completions != tokenizer.pad_token_id).sum())
                hits = 0
                for row in completions:
                    text = tokenizer.decode(row, skip_special_tokens=True)
                    m = ANSWER.search(text)
                    ok = m is not None and int(m.group(1)) == answer
                    hits += ok
                    if out:
                        out.write(json.dumps({"bucket": name, "split": split, "prompt": line,
                                              "answer": answer, "correct": ok, "completion": text}) + "\n")
                correct.append(hits)
            g = args.group
            n = max(len(correct), 1)
            row = {
                "bucket": name, "split": split, "problems": len(correct),
                "pass@1": sum(correct) / (n * g),
                "mixed": sum(0 < c < g for c in correct) / n,
                "all_wrong": sum(c == 0 for c in correct) / n,
                "all_right": sum(c == g for c in correct) / n,
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
