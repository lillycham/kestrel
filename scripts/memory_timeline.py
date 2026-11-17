"""Summarise stats-only memory probes over training steps: when does each memory head die?

Reads <dir>/step_*.json (from memory_probes.py --stats-only) and prints, per step,
the loss and a forget-gate code per block call: one character per head,
'.' for alpha <= 0.05 (keeps W_init), '1' for alpha >= 0.95 (wipes), '~' between.
A call is marked '†' when its memory supplies under 1% of the gated update after chunk 0.

    python scripts/memory_timeline.py runs/titans-loop-1/memory_probes
"""

import json
import sys
from pathlib import Path


def code(alpha):
    return "." if alpha <= 0.05 else "1" if alpha >= 0.95 else "~"


def main():
    root = Path(sys.argv[1])
    files = sorted(root.glob("step_*.json"), key=lambda p: int(p.stem.split("_")[1]))
    if not files:
        sys.exit(f"no step_*.json in {root}")
    first = json.loads(files[0].read_text())
    labels = first["labels"]
    short = [l.replace("loop", "L").replace(".core", "c").replace("prelude", "pre").replace("coda", "coda") for l in labels]
    print("step    loss   " + " ".join(s.ljust(6) for s in short))
    for f in files:
        r = json.loads(f.read_text())
        loss = sum(r["loss"]["base"]) / len(r["loss"]["base"])
        cells = []
        for label in labels:
            c = r["calls"][label]
            alphas = [sum(h) / len(h) for h in c["alpha_by_head_chunk"]]
            late = c["mem_share_by_chunk"][1:]
            dead = sum(late) / len(late) < 0.01
            cells.append(("".join(code(a) for a in alphas) + ("†" if dead else "")).ljust(6))
        print(f"{f.stem.split('_')[1]:>6}  {loss:.3f}  " + " ".join(cells))


if __name__ == "__main__":
    main()
