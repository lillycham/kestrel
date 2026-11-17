"""Test whether the Titans memory carries a name past the attention window.

Each row is one story: a needle sentence that names its main character, G
tokens of the same story that never repeat the name, then a cue that the
main character's name comes next:

    <bos> Once upon a time, there was a girl named Max. <G tokens about "she"> One day,

The filler is sampled from --filler-run, continuing the needle, with every
candidate name and the end-of-story token banned. So it stays one story
about one unnamed girl, and every run sees the same text. (Filler from
another story would be a confound: after a new "Once upon a time", the
right answer is the new character, not the needle.)

The score is the rank of the true name among the candidate names (single
tokens). Inside the attention window, attention can copy the name. Past
it, only the memory (or a chain of attention hops through the loops) can
carry it.

Conditions, for each gap G:
  base      the model as trained
  nowrite   every block call retrieves from W_init only (no test-time writes)
  swap      every block call writes its memory from the next row's story: the
            memory has its usual statistics, but another name. If swap falls
            as far as nowrite, the memory carried the name; if it stays near
            base, the nowrite loss comes from an unseen W_init state.
  noneedle  the needle names another candidate (same filler); this measures
            the prior of the model for the target name, the floor for recall
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from memory_probes import Probe, call_labels, load  # noqa: E402

NAMES = ("Tom Ben Mia Max Sue Tim Anna Sam Lucy Jack Sara Bob Amy Joe Mark Jill Spot Molly Daisy Leo "
         "Rex Kitty Billy Jane Mike Ellie Bella Buddy Timmy Sally Lila Jen Jim Lily").split()
NEEDLE = "Once upon a time, there was a girl named {}."
QUERY = " One day,"  # The cue TinyStories teaches: the next token is the main character.


def name_ids(tok):
    ids = {}
    for n in NAMES:
        t = tok.encode(" " + n, add_special_tokens=False)
        if len(t) == 1:
            ids[n] = t[0]
    return ids


@torch.no_grad()
def sample_filler(run, targets, length, seed, device):
    """Continue each target's needle for length tokens, never naming a candidate or ending the story."""
    model, tok = load(run, device)
    ids = name_ids(tok)
    # Bare names too: a sentence can start with "Max" without a leading space.
    banned = set(ids.values()) | {tok.eos_token_id, tok.bos_token_id}
    for n in ids:
        t = tok.encode(n, add_special_tokens=False)
        if len(t) == 1:
            banned.add(t[0])
    prompts = torch.tensor([[tok.bos_token_id] + tok.encode(NEEDLE.format(n), add_special_tokens=False) for n in targets],
                           device=device)
    torch.manual_seed(seed)
    out = model.generate(prompts, max_new_tokens=length, temperature=0.8, use_cache=True,
                         bad_words_ids=[[t] for t in sorted(banned)])
    return out[:, prompts.shape[1]:].cpu()


@torch.no_grad()
def score(model, tok, filler, targets, shown, gap, ids, device):
    """Rank of each target name among all candidates, and its log-odds against a uniform guess."""
    query = tok.encode(QUERY, add_special_tokens=False)
    rows = [[tok.bos_token_id] + tok.encode(NEEDLE.format(s), add_special_tokens=False) + filler[i, :gap].tolist() + query
            for i, s in enumerate(shown)]
    logits = model(torch.tensor(rows, device=device)).logits[:, -1].float().log_softmax(-1).cpu()
    names = list(ids)
    scores = logits[:, [ids[n] for n in names]]
    t = scores[torch.arange(len(targets)), [names.index(n) for n in targets]]
    ranks = (scores > t[:, None]).sum(1).numpy()
    margins = (t - torch.logsumexp(scores, 1) + np.log(len(names))).numpy()
    return {"top1": float((ranks == 0).mean()), "mean_rank": float(ranks.mean()),
            "nats_over_chance": float(margins.mean()), "chance": 1 / len(names)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--gaps", default="0,16,48,96,160,256,384,480")
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--filler-run", type=Path, help="run that samples the filler (default: the first run)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    gaps = [int(g) for g in args.gaps.split(",")]

    _, tok = load(args.runs[0], "cpu")
    ids = name_ids(tok)
    names = list(ids)
    rng = np.random.default_rng(args.seed)
    targets = [names[i] for i in rng.integers(len(names), size=args.rows)]
    # Another name for each row, for the noneedle floor.
    others = [names[(names.index(t) + 1 + rng.integers(len(names) - 1)) % len(names)] for t in targets]
    filler = sample_filler(args.filler_run or args.runs[0], targets, max(gaps), args.seed, args.device)

    results = {"names": names, "filler_example": tok.decode(filler[0].tolist())}
    for run in args.runs:
        model, tok = load(run, args.device)
        limit = model.config.max_position_embeddings
        probe = Probe(model)
        all_calls = set(range(len(call_labels(model))))
        res = results[run.name] = {}
        for gap in gaps:
            length = 1 + len(tok.encode(NEEDLE.format("Max"), add_special_tokens=False)) + gap + len(tok.encode(QUERY, add_special_tokens=False))
            if length > limit:
                print(f"skip gap {gap}: {length} tokens > {limit}")
                continue
            row = res[gap] = {}
            for cond in ("base", "nowrite", "swap", "noneedle"):
                probe.nowrite = all_calls if cond == "nowrite" else set()
                probe.swap = all_calls if cond == "swap" else set()
                row[cond] = score(model, tok, filler, targets, others if cond == "noneedle" else targets, gap, ids, args.device)
            probe.nowrite, probe.swap = set(), set()
            print(f"{run.name:28s} gap {gap:4d}  " + "  ".join(
                f"{c} top1 {r['top1']:.3f} nats {r['nats_over_chance']:+.2f}" for c, r in row.items())
                + f"  (chance {row['base']['chance']:.3f})", flush=True)
    if args.out:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
