"""Probe the Titans-MAG main network of an H-Net run.

The main network runs on chunks of bytes, so its statistics are per chunk.
Rows of a batch have different chunk counts and are padded at the end, so
every statistic here uses only the real chunks of each row. As in
memory_probes.py, it records per block call the MAG gate, the memory share of
the gated update after the first memory chunk, and the forget, momentum and
inner-lr gates. It aggregates the MAG gate by chunk text (which byte strings
go to memory and which to attention) and by chunk length, and it runs the
nowrite and swap ablations, with the byte loss in buckets of byte position.

    python scripts/hnet_memory_probes.py runs/hnet-titans-1 --text data/anneal-raw/validation.jsonl
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from memory_probes import Probe, call_labels, load


def sequences(tok, path, seq_len, num_seqs):
    """Byte-token windows of seq_len + 1 from the documents of a JSONL file, joined by EOS."""
    stream, need = [], (seq_len + 1) * num_seqs
    with open(path) as f:
        for line in f:
            stream += tok(json.loads(line)["text"], add_special_tokens=False).input_ids + [tok.eos_token_id]
            if len(stream) >= need:
                break
    if len(stream) < need:
        raise ValueError(f"{path} has only {len(stream)} byte tokens; need {need}")
    return torch.tensor(stream[:need]).view(num_seqs, seq_len + 1)


@torch.no_grad()
def byte_losses(model, probe, data, batch_size, device, **setting):
    probe.nowrite, probe.swap = setting.get("nowrite", set()), setting.get("swap", set())
    out = []
    for rows in data.split(batch_size):
        rows = rows.to(device)
        logits = model(rows[:, :-1]).logits.float()
        out.append(F.cross_entropy(logits.transpose(1, 2), rows[:, 1:], reduction="none").cpu())
    probe.nowrite, probe.swap = set(), set()
    return torch.cat(out)


def buckets(losses, size):
    n = losses.shape[1] // size
    return losses[:, : n * size].view(losses.shape[0], n, size).mean(dim=(0, 2)).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--text", type=Path, default=Path("data/anneal-raw/validation.jsonl"))
    ap.add_argument("--checkpoint", default="latest.pt")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--num-seqs", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--bucket", type=int, default=512, help="bytes per loss bucket")
    ap.add_argument("--min-count", type=int, default=8, help="occurrences a chunk text needs for the gate lists")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    t0 = time.time()

    model, tok = load(args.run, args.device, args.checkpoint)
    lm = model.main.model
    probe = Probe(lm)
    labels = call_labels(lm)
    mem_chunk = lm.config.memory_chunk_size
    calls = set(range(len(labels)))
    data = sequences(tok, args.text, args.seq_len, args.num_seqs)
    decode = lambda ids: tok.decode(ids, skip_special_tokens=False)

    # 1. Recorded pass, one row at a time, so that each row's chunks are known.
    probe.record, probe.rank_batch = True, 0
    gate_by_text = defaultdict(lambda: defaultdict(list))  # call -> chunk text -> gates
    gate_by_len = defaultdict(lambda: defaultdict(list))   # call -> chunk length -> gates
    per_call = defaultdict(lambda: defaultdict(list))
    chunk_counts, base = [], []
    with torch.no_grad():
        for row in data:
            probe.stats.clear()
            x = row[None, :-1].to(args.device)
            out = model(x)
            base.append(F.cross_entropy(out.logits.float().transpose(1, 2), row[None, 1:].to(args.device),
                                        reduction="none").cpu())
            starts = out.boundaries[0].nonzero().flatten().cpu().tolist()
            count = len(starts)
            chunk_counts.append(count)
            ends = starts[1:] + [x.shape[1]]
            texts = [decode(row[s:e].tolist()) for s, e in zip(starts, ends)]
            full = count // mem_chunk  # memory chunks with no padding
            for c, label in enumerate(labels):
                s = probe.stats[c]
                gate = s["gate"][0][0, :count]
                for g, text in zip(gate.tolist(), texts):
                    gate_by_text[label][text].append(g)
                    gate_by_len[label][min(len(text), 16)].append(g)
                per_call[label]["gate"].append(gate.mean().item())
                if count > mem_chunk:
                    per_call[label]["share"].append(s["mem_share"][0][0, mem_chunk:count].mean().item())
                for name in ("alpha", "eta", "theta"):
                    per_call[label][name].append(s[name][0][0, :, :max(full, 1)].mean(-1))  # (H,)
    base = torch.cat(base)
    probe.record = False
    print(f"recorded pass {time.time() - t0:.0f}s, loss {base.mean():.4f}, "
          f"{sum(chunk_counts) / len(chunk_counts):.0f} chunks per row", flush=True)

    result = {"run": str(args.run), "checkpoint": args.checkpoint, "labels": labels, "bucket": args.bucket,
              "chunks_per_row": sum(chunk_counts) / len(chunk_counts),
              "bytes_per_chunk": args.seq_len * len(chunk_counts) / sum(chunk_counts),
              "loss": {"base": buckets(base, args.bucket)}, "calls": {}}

    # 2. Ablations. Swap writes each memory from the next row of the batch.
    for name, setting in (("nowrite", {"nowrite": calls}), ("swap", {"swap": calls})):
        result["loss"][name] = buckets(byte_losses(model, probe, data, args.batch_size, args.device, **setting), args.bucket)
        print(f"{name} {time.time() - t0:.0f}s", flush=True)
    result["loss"]["swap_one"] = {
        label: buckets(byte_losses(model, probe, data, args.batch_size, args.device, swap={c}), args.bucket)
        for c, label in enumerate(labels)}
    print(f"per-call swap {time.time() - t0:.0f}s", flush=True)

    # 3. Per-call summaries.
    for label in labels:
        pc = per_call[label]
        common = [(t, sum(g) / len(g), len(g)) for t, g in gate_by_text[label].items() if len(g) >= args.min_count]
        common.sort(key=lambda item: item[1])
        result["calls"][label] = {
            "gate_mean": sum(pc["gate"]) / len(pc["gate"]),
            "mem_share_mean": sum(pc["share"]) / len(pc["share"]) if pc["share"] else None,
            **{f"{n}_by_head": torch.stack(pc[n]).mean(0).tolist() for n in ("alpha", "eta", "theta")},
            "gate_by_chunk_len": {n: sum(g) / len(g) for n, g in sorted(gate_by_len[label].items())},
            "low_gate_chunks": [(t, round(g, 4), n) for t, g, n in common[:20]],
            "top_gate_chunks": [(t, round(g, 4), n) for t, g, n in common[::-1][:20]],
        }

    out = args.out or args.run / f"memory_probes_{Path(args.checkpoint).stem}.json"
    out.write_text(json.dumps(result))
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
