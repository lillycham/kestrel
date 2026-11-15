"""Arithmetic problems in the step-by-step templates of scripts/build_anneal_mix.py.

A problem is a pair (kind, numbers). Buckets split each kind by difficulty.
"""

import json
import re

FIRST_LINE = re.compile(
    r"Compute (\d+) ([+-]) (\d+) step by step\."
    r"|Multiply (\d+) \* (\d+) step by step\."
    r"|Evaluate \((\d+) \+ (\d+)\) \* (\d+)\."
    r"|Evaluate (\d+) \* (\d+) \+ (\d+)\."
)
ANSWER = re.compile(r"Answer: (-?\d+)")

BUCKETS = {
    "add_2d_nocarry": "add", "add_2d_carry": "add", "add_3d_nocarry": "add", "add_3d_carry": "add",
    "sub_noborrow": "sub", "sub_borrow": "sub", "mul": "mul", "addmul": "addmul", "muladd": "muladd",
}


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


def render(problem):
    """Return (first line, answer) for a problem."""
    kind, nums = problem
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


def bucket(problem):
    """Difficulty bucket of a problem, within the ranges of the anneal generator."""
    kind, nums = problem
    if kind == "add":
        a, b = nums
        size = "2d" if a < 100 and b < 100 else "3d"
        carry = "carry" if a % 10 + b % 10 >= 10 else "nocarry"
        return f"add_{size}_{carry}"
    if kind == "sub":
        a, b = nums
        return "sub_borrow" if a % 10 < b % 10 else "sub_noborrow"
    return kind


def draw(name, rng):
    """Draw a problem in bucket `name`, with the ranges of gen_arithmetic_doc in build_anneal_mix.py."""
    kind = BUCKETS[name]
    while True:
        if kind == "add":
            hi = 99 if name.startswith("add_2d") else 999
            nums = rng.randint(10, hi), rng.randint(10, hi)
        elif kind == "sub":
            a = rng.randint(20, 999)
            nums = a, rng.randint(10, a)
        elif kind == "mul":
            nums = rng.randint(11, 99), rng.randint(2, 19)
        else:
            a, b, c = rng.randint(2, 40), rng.randint(2, 40), rng.randint(2, 15)
            nums = (a, b, c) if kind == "addmul" else (a, c, b)
        problem = (kind, nums)
        if bucket(problem) == name:
            return problem


def draw_new(name, rng, exclude, tries=200_000):
    """Draw a problem in bucket `name` that is not in `exclude`."""
    for _ in range(tries):
        problem = draw(name, rng)
        if problem not in exclude:
            return problem
    raise ValueError(f"No new problem found in bucket {name}")


def prompt_text(problem, eos_token):
    # Documents follow the EOS of the document before them in the training stream.
    return eos_token + render(problem)[0] + "\n"


def is_correct(completion, problem):
    m = ANSWER.search(completion)
    return m is not None and int(m.group(1)) == render(problem)[1]


def seen_problems(path):
    """Template problems that occur in a training JSONL file."""
    seen = set()
    if path is None:
        return seen
    with open(path, encoding="utf-8") as source:
        for line in source:
            if "step by step" not in line and "Evaluate" not in line:
                continue
            parsed = parse(json.loads(line)["text"].split("\n", 1)[0])
            if parsed:
                seen.add(parsed)
    return seen
