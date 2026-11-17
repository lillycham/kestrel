"""Probe the neural memory of a Titans-MAG run.

For each block call (prelude, each loop of each core layer, coda) it records:
- the MAG gate: how much of each token's update comes from memory, not attention;
- the write gates alpha (forget), eta (momentum) and theta (inner lr), per head and chunk;
- the written content R_t = W_t - prod(1 - alpha) * W_init, its size and effective rank;
- the spectrum of the learned initial memory W_init.

It also runs ablations and reports the loss by chunk: "nowrite" retrieves every
chunk from W_init (the memory never learns in context), "nomem" removes the
memory branch from the MAG sum. Chunk 0 never sees a write, so its loss must
not change under "nowrite": that is the sanity check.

    python scripts/memory_probes.py runs/titans-loop-anneal-1 --data data/anneal-prepared
"""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from kestrel.core import make_model
from kestrel.data import load_tokenizer
from kestrel.memory import TitansMAGBlock, memory_retrieve, memory_update


def load(run, device, checkpoint="latest.pt"):
    tok = load_tokenizer(run / "tokenizer")
    saved = torch.load(run / checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model = make_model(saved["config"]["model"], {
        "bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id, "pad_token_id": tok.pad_token_id,
    })
    model.load_state_dict(saved["model"])
    return model.to(device).eval(), tok


def call_labels(model):
    labels = [f"prelude{i}" for i in range(len(getattr(model, "prelude", [])))]
    labels += [f"loop{l}.core{i}" for l in range(model.num_loops) for i in range(len(model.layers))]
    labels += [f"coda{i}" for i in range(len(getattr(model, "coda", [])))]
    return labels


def effective_rank(m):
    """exp(entropy) of the normalised singular values, for a batch of matrices."""
    s = torch.linalg.svdvals(m.cpu().double())
    p = s / s.sum(-1, keepdim=True).clamp_min(1e-30)
    return torch.exp(-(p * p.clamp_min(1e-30).log()).sum(-1))


class Probe:
    """Patches every block of the model. One block object runs once per loop, so
    a counter that resets on each model call gives the block-call index."""

    def __init__(self, model):
        self.model = model
        self.call = -1
        self.record = False
        self.rank_batch = 0           # sequences per batch whose write rank is measured
        self.nowrite = set()          # block calls that never write
        self.nomem = set()            # block calls without the memory branch
        self.constgate = set()        # block calls whose MAG gate is its per-dimension mean
        self.gate_means = {}          # block call -> (D,) mean gate, from the recorded pass
        self.stats = defaultdict(lambda: defaultdict(list))
        model.register_forward_pre_hook(self._reset)
        for block in {id(b): b for b in self._blocks()}.values():
            block.register_forward_pre_hook(self._next)
            block._combine = self._combine_for(block)
            block.memory.prepare = self._prepare_for(block.memory)
            block.memory.recur = self._recur_for(block.memory)

    def _blocks(self):
        m = self.model
        return list(getattr(m, "prelude", [])) + list(m.layers) + list(getattr(m, "coda", []))

    def _reset(self, module, args):
        self.call = -1

    def _next(self, module, args):
        self.call += 1

    def _combine_for(self, block):
        def combine(x, h, attn_out, memory_out):
            mem_out = block.mem_norm(memory_out)
            gate = torch.sigmoid(block.mag_gate(h))
            if self.call in self.constgate:
                # The gate of every token becomes the mean gate of its dimension.
                gate = self.gate_means[self.call].to(gate.dtype).expand_as(gate)
            if self.call in self.nomem:
                mem_out = torch.zeros_like(mem_out)
            if self.record:
                s = self.stats[self.call]
                g = gate.float()
                s["gate"].append(g.mean(-1).cpu())                          # (B, L)
                s["gate_dim_sum"].append(g.sum(dim=(0, 1)).cpu()[None])     # (1, D)
                s["gate_dims_open"].append((g > 0.5).float().mean(-1).cpu())  # (B, L)
                mem_part = (g * mem_out.float()).norm(dim=-1)
                attn_part = ((1 - g) * attn_out.float()).norm(dim=-1)
                s["mem_share"].append((mem_part / (mem_part + attn_part)).cpu())  # (B, L)
            x = x + (1.0 - gate) * attn_out + gate * mem_out
            return x + block.mlp(block.post_attention_layernorm(x))
        return combine

    def _prepare_for(self, memory):
        original = memory.prepare

        def prepare(x):
            out = original(x)
            if self.record:
                bsz = x.shape[0]
                s = self.stats[self.call]
                for name, g in zip(("alpha", "eta", "theta"), out[3:6]):
                    # (n, B * H, 1, 1) -> (B, H, n)
                    s[name].append(g[..., 0, 0].view(-1, bsz, memory.heads).permute(1, 2, 0).cpu())
            return out
        return prepare

    def _recur_for(self, memory):
        def recur(q, k, v, alpha, eta, theta, state):
            w1_0, w2_0 = state[0], state[1]
            outputs = [memory_retrieve(q[0], w1_0, w2_0)]
            decay = torch.ones_like(alpha[0])
            nowrite = self.call in self.nowrite
            rows = self.rank_batch * memory.heads
            norms, ranks = [], []
            for idx in range(q.shape[0] - 1):
                if nowrite:
                    outputs.append(memory_retrieve(q[idx + 1], w1_0, w2_0))
                    continue
                state = memory_update(state, k[idx], v[idx], alpha[idx], eta[idx], theta[idx], memory.chunk_size)
                if len(state) == 4:  # Decay toward 0. With decay_to_init, W - W_init is the written part.
                    decay = decay * (1.0 - alpha[idx])
                if self.record:
                    r1 = state[0] - decay * w1_0
                    r2 = state[1] - decay * w2_0
                    norms.append(torch.stack([r1.flatten(1).norm(dim=-1) / w1_0.flatten(1).norm(dim=-1),
                                              r2.flatten(1).norm(dim=-1) / w2_0.flatten(1).norm(dim=-1)]).cpu())
                    if rows:
                        ranks.append(torch.stack([effective_rank(r1[:rows]), effective_rank(r2[:rows])]).float())
                outputs.append(memory_retrieve(q[idx + 1], state[0], state[1]))
            if self.record and norms:
                bsz = q.shape[1] // memory.heads
                s = self.stats[self.call]
                # (2, n - 1, B, H): relative size of the written content of W1 and W2.
                s["write_norm"].append(torch.stack(norms, 1).view(2, -1, bsz, memory.heads))
                if ranks:
                    s["write_rank"].append(torch.stack(ranks, 1).view(2, -1, self.rank_batch, memory.heads))
            return torch.stack(outputs)
        return recur


def batches(data, seq_len, num_seqs, batch_size, device):
    tokens = np.memmap(data / "validation.bin", dtype="<u4", mode="r")
    stride = (len(tokens) - seq_len - 1) // num_seqs
    for start in range(0, num_seqs, batch_size):
        rows = [tokens[i * stride: i * stride + seq_len + 1] for i in range(start, min(start + batch_size, num_seqs))]
        x = torch.tensor(np.stack(rows).astype(np.int64), device=device)
        yield x[:, :-1], x[:, 1:]


@torch.no_grad()
def token_losses(model, probe, args, **setting):
    probe.nowrite, probe.nomem = setting.get("nowrite", set()), setting.get("nomem", set())
    probe.constgate = setting.get("constgate", set())
    out = []
    for inputs, targets in batches(args.data, args.seq_len, args.num_seqs, args.batch_size, args.device):
        logits = model(inputs).logits.float()
        out.append(F.cross_entropy(logits.transpose(1, 2), targets, reduction="none").cpu())
    probe.nowrite, probe.nomem, probe.constgate = set(), set(), set()
    return torch.cat(out)  # (N, L)


def by_chunk(losses, chunk):
    n = losses.shape[1] // chunk
    return losses[:, : n * chunk].view(losses.shape[0], n, chunk).mean(dim=(0, 2)).tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--data", type=Path, default=Path("data/anneal-prepared"))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--num-seqs", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--rank-seqs", type=int, default=2, help="sequences per batch whose write rank is measured")
    ap.add_argument("--per-call", action="store_true", help="also ablate the writes of one block call at a time")
    ap.add_argument("--checkpoint", default="latest.pt", help="checkpoint file in the run directory")
    ap.add_argument("--skip-nomem", action="store_true", help="skip the nomem ablation (it hangs MPS on the SFT run)")
    ap.add_argument("--stats-only", action="store_true", help="only the recorded pass, no ablations (for a checkpoint timeline)")
    ap.add_argument("--const-gate", action="store_true",
                    help="also replace the MAG gate by its per-dimension mean: in dead-memory calls, in all, and per call")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    t0 = time.time()

    model, tok = load(args.run, args.device, args.checkpoint)
    probe = Probe(model)
    labels = call_labels(model)
    chunk = model.config.memory_chunk_size
    calls = set(range(len(labels)))

    # 1. Recorded base pass.
    probe.record, probe.rank_batch = True, args.rank_seqs
    base = token_losses(model, probe, args)
    probe.record = False
    inputs = torch.cat([x.cpu() for x, _ in batches(args.data, args.seq_len, args.num_seqs, args.batch_size, "cpu")])
    print(f"base pass {time.time() - t0:.0f}s, loss {base.mean():.4f}", flush=True)

    result = {"run": str(args.run), "labels": labels, "chunk_size": chunk, "num_seqs": args.num_seqs,
              "loss": {"base": by_chunk(base, chunk)}, "calls": {}}

    # 2. Global ablations.
    ablations = [] if args.stats_only else [("nowrite", {"nowrite": calls})] + ([] if args.skip_nomem else [("nomem", {"nomem": calls})])
    for name, setting in ablations:
        result["loss"][name] = by_chunk(token_losses(model, probe, args, **setting), chunk)
        print(f"{name} {time.time() - t0:.0f}s", flush=True)
    if args.const_gate:
        n_tokens = args.num_seqs * args.seq_len
        probe.gate_means = {c: (torch.cat(probe.stats[c]["gate_dim_sum"]).sum(0) / n_tokens).to(args.device)
                            for c in calls}
        # A call is dead when the memory supplies under 1% of the gated update after chunk 0.
        dead = {c for c in calls
                if torch.cat(probe.stats[c]["mem_share"])[:, chunk:].mean().item() < 0.01}
        result["dead_calls"] = [labels[c] for c in sorted(dead)]
        result["loss"]["constgate_dead"] = by_chunk(token_losses(model, probe, args, constgate=dead), chunk)
        result["loss"]["constgate_all"] = by_chunk(token_losses(model, probe, args, constgate=calls), chunk)
        result["loss"]["constgate_one"] = {
            label: by_chunk(token_losses(model, probe, args, constgate={c}), chunk) for c, label in enumerate(labels)}
        print(f"const-gate ablations {time.time() - t0:.0f}s, dead calls {result['dead_calls']}", flush=True)
    if args.per_call:
        result["loss"]["nowrite_one"] = {}
        for c, label in enumerate(labels):
            result["loss"]["nowrite_one"][label] = by_chunk(token_losses(model, probe, args, nowrite={c}), chunk)
        print(f"per-call ablations {time.time() - t0:.0f}s", flush=True)

    # 3. Per-call statistics.
    flat_ids = inputs.flatten()
    vocab_counts = torch.bincount(flat_ids, minlength=model.config.vocab_size)
    for c, label in enumerate(labels):
        # write_* batches are (2, n - 1, B, H); every other stat has the batch first.
        s = {k: torch.cat(v, dim=2 if k.startswith("write") else 0) for k, v in probe.stats[c].items()}
        result.setdefault("gate_dim_mean", {})[label] = (s["gate_dim_sum"].sum(0) / s["gate"].numel()).tolist()
        gate = s["gate"]  # (N, L)
        per_tok = torch.bincount(flat_ids, weights=gate.flatten().double(), minlength=len(vocab_counts))
        mean_gate = per_tok / vocab_counts.clamp_min(1)
        common = vocab_counts >= 20
        order = torch.argsort(torch.where(common, mean_gate, torch.nan).nan_to_num(-1), descending=True)
        low = torch.argsort(torch.where(common, mean_gate, torch.nan).nan_to_num(2))
        name = lambda i: tok.convert_tokens_to_string([tok.convert_ids_to_tokens(int(i))])
        mem = s["mem_share"]
        result["calls"][label] = {
            "gate_mean": gate.mean().item(),
            "gate_by_chunk": by_chunk(gate, chunk),
            "gate_dims_open": s["gate_dims_open"].mean().item(),
            "gate_hist": torch.histc(gate.flatten(), bins=20, min=0, max=1).tolist(),
            "mem_share_mean": mem.mean().item(),
            "mem_share_by_chunk": by_chunk(mem, chunk),
            "top_gate_tokens": [(name(i), round(mean_gate[i].item(), 4), int(vocab_counts[i])) for i in order[:15]],
            "low_gate_tokens": [(name(i), round(mean_gate[i].item(), 4), int(vocab_counts[i])) for i in low[:15]],
            **{f"{g}_by_head_chunk": s[g].mean(0).tolist() for g in ("alpha", "eta", "theta")},
            # (2, n-1, B, H) -> (2, H, n-1): W1 and W2.
            "write_norm": s["write_norm"].permute(0, 3, 1, 2).mean(-1).tolist(),
            "write_rank": s["write_rank"].permute(0, 3, 1, 2).mean(-1).tolist() if "write_rank" in s else None,
        }

    # 4. Spectrum of the learned initial memory, per block and head.
    result["init_spectrum"] = {}
    for block, label in zip(probe._blocks(), [l for l in labels if not l.startswith(("loop1", "loop2"))]):
        mem = block.memory
        w1 = mem.w1_init.detach().float().view(mem.heads, mem.mem_head_dim, mem.head_dim)
        w2 = mem.w2_init.detach().float().view(mem.heads, mem.head_dim, mem.mem_head_dim)
        result["init_spectrum"][label.replace("loop0.", "")] = {
            "w1": torch.linalg.svdvals(w1.cpu()).tolist(), "w2": torch.linalg.svdvals(w2.cpu()).tolist(),
            "w1_rank": effective_rank(w1).tolist(), "w2_rank": effective_rank(w2).tolist(),
        }

    out = args.out or args.run / "memory_probes.json"
    out.write_text(json.dumps(result))
    print(f"wrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
