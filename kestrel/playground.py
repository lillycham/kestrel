"""Kestrel runs in the suzunaan playground: write, chat, compare and look inside.

    python -m kestrel playground                 # runs/, then open http://localhost:8000
    python -m kestrel playground --root runs --device cpu --port 8080

Every run folder with latest.pt and tokenizer/ is a model. Its step_*.pt snapshots are
listed after the runs. Plain Qwen3 runs and Titans-MAG runs have every view. A looped
Titans model counts each run of a block as its own layer, in the order they run.

An H-Net run shows its byte layers only: the encoder, then the decoder. The logit lens
reads the encoder layers through the final norm too, although the model itself never
does. Generation runs the whole prefix again at each byte, as there is no decode cache.
"""
from contextlib import contextmanager
from pathlib import Path

import torch

from suzunaan import CHAT, TEXT, Catalog, Entry, Prompt
from suzunaan.hooks import Counter, hooks, qk_attention, zero_heads

from .core import make_model
from .data import load_tokenizer
from .hnet import HNetForCausalLM
from .memory import TitansMAGForCausalLM


def byte_decoder():
    """GPT-2's map from byte-level BPE characters back to bytes."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class KestrelBackend:
    capabilities = frozenset({"attention", "lens", "neurons", "ablate"})

    def __init__(self, checkpoint, tokenizer_dir, device):
        self.device = device
        self.tok = tokenizer = load_tokenizer(tokenizer_dir)
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        model = make_model(saved["config"]["model"], {
            "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        })
        model.load_state_dict(saved["model"])
        del saved
        self.model = model.to(device).eval()
        c = model.config
        self.titans = isinstance(model, TitansMAGForCausalLM)
        self.hnet = isinstance(model, HNetForCausalLM)
        # Titans and the H-Net byte layers call SlidingWindowAttention(h, cos, sin).
        self.windowed = self.titans or self.hnet
        inner = model if self.windowed else model.model
        self.embed, self.norm, self.head = inner.embed_tokens, inner.norm, model.lm_head
        # The blocks in the order they run (a looped core runs num_loops times), and
        # each distinct block once, for hooks.
        if self.titans:
            prelude, coda = list(getattr(model, "prelude", [])), list(getattr(model, "coda", []))
            self.runs = prelude + list(model.layers) * model.num_loops + coda
            self.blocks = prelude + list(model.layers) + coda
        elif self.hnet:
            # Only the byte layers: the main network runs on chunks, so its attention
            # and residual stream have no row per token for the views.
            self.runs = self.blocks = list(model.encoder) + list(model.decoder)
        else:
            self.runs = self.blocks = list(inner.layers)
        self.layers, self.heads = len(self.runs), c.num_attention_heads
        self.head_dim, self.context, self.vocab = c.head_dim, c.max_position_embeddings, len(tokenizer)
        self.params = sum(p.numel() for p in model.parameters())
        if self.hnet:
            # The H-Net has no decode cache yet: each step runs the whole prefix again.
            self.decoder = None
            main = "Titans-MAG" if c.main_network == "titans" else "Qwen3"
            self.shape = (f"H-Net ({main} main, width {c.main_hidden_size}, ~{c.target_chunk_size:g} bytes/chunk), "
                          f"{len(model.encoder)} + {len(model.decoder)} byte layers x {self.heads} heads "
                          f"({c.num_key_value_heads} KV), width {c.hidden_size}, context {self.context} bytes")
        else:
            kind = "Titans-MAG" if self.titans else "Qwen3"
            loops = f", {model.num_loops} loops" if self.titans and model.num_loops > 1 else ""
            self.shape = (f"{kind}, {self.layers} layers{loops} x {self.heads} heads ({c.num_key_value_heads} KV), "
                          f"width {c.hidden_size}, context {self.context}")
        self.unused = torch.arange(len(tokenizer), c.vocab_size, device=device)

        self.special = set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values())
        decode = byte_decoder()
        self.pieces, self.bytes = [], []
        for i, t in enumerate(tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))):
            if i in self.special:
                self.pieces.append(t)
                self.bytes.append(b"")
            else:
                b = bytes(decode[ch] for ch in t) if all(ch in decode for ch in t) else t.encode()
                self.pieces.append(b.decode("utf-8", errors="replace"))
                self.bytes.append(b)
        self.eos = tokenizer.eos_token_id
        self.end = tokenizer.convert_tokens_to_ids("<|end|>")

    # ---- prompts ----
    def encode(self, text):
        return self.tok(text, add_special_tokens=False).input_ids

    def prompt(self, mode, request):
        if mode == "text":
            ids = ([self.eos] if request["after_eot"] else []) + self.encode(request["text"])
            return Prompt(ids, frozenset({self.eos}))
        if mode != "chat":
            raise ValueError(f"Unknown mode {mode!r}")
        system = request.get("system", "").strip()
        head = [{"role": "system", "content": system}] if system else []
        turns = [{"role": m["role"], "content": m["content"]} for m in request["messages"]]
        if not turns or turns[-1]["role"] != "user":
            raise ValueError("A chat request ends with a user message")
        # Drop the oldest exchange until the reply has room, as `kestrel chat` does.
        room = min(request.get("max_tokens", 256), 32)
        while True:
            ids = self.encode(self.tok.apply_chat_template(head + turns, tokenize=False, add_generation_prompt=True))
            if len(ids) + room <= self.context or len(turns) <= 1:
                break
            del turns[:2]
        if len(ids) >= self.context:
            raise ValueError(f"The message is {len(ids)} tokens; this model sees {self.context}")
        return Prompt(ids, frozenset({self.eos, self.end}))

    def piece(self, i):
        return self.pieces[i] if i < len(self.pieces) else f"<unused {i}>"

    def token_bytes(self, i):
        return self.bytes[i] if i < len(self.bytes) else b""

    def is_special(self, i):
        return i in self.special

    # ---- the model ----
    def masked(self, logits):
        logits[..., self.unused] = float("-inf")
        return logits

    def logits(self, ids):
        x = torch.tensor([ids], device=self.device)
        return self.masked(self.model(x, use_cache=False).logits[0, -1].float())

    def decoder(self):
        return KestrelDecoder(self)

    def unembed(self, h):
        return self.masked(self.head(self.norm(h)).float())

    def trace(self, ids, want, visit):
        counters = {k: Counter() for k in ("resid", "attn", "mlp")}

        def attn_hook(module, args, kwargs, out):
            if self.windowed:
                h, cos, sin = args
                window = module.window
            else:
                h = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
                cos, sin = kwargs["position_embeddings"]
                window = None
            visit("attn", counters["attn"].next(), qk_attention(module, h, cos, sin, window))

        with hooks() as add:
            for b in self.blocks:
                if "attn" in want:
                    add(b.self_attn.register_forward_hook(attn_hook, with_kwargs=True))
                if "mlp" in want:
                    add(b.mlp.down_proj.register_forward_pre_hook(
                        lambda m, args: visit("mlp", counters["mlp"].next(), args[0][0])))
                if "resid" in want:
                    add(b.register_forward_hook(lambda m, args, out: visit(
                        "resid", counters["resid"].next() + 1, (out[0] if isinstance(out, tuple) else out)[0])))
            if "resid" in want:
                add(self.embed.register_forward_hook(lambda m, args, out: visit("resid", 0, out[0])))
            self.model(torch.tensor([ids], device=self.device), use_cache=False)

    @contextmanager
    def ablated(self, heads):
        counter = Counter()

        def reset(module, args, kwargs):
            counter.n = 0

        with hooks() as add:
            add(self.model.register_forward_pre_hook(reset, with_kwargs=True))
            with zero_heads([b.self_attn.o_proj for b in self.blocks], heads, self.head_dim, counter):
                yield


class KestrelDecoder:
    """Cached generation for one sample: each call runs only the ids that the cache has
    not seen. Titans-MAG keeps its own cache dict (the sliding-window keys and values
    and the memory state of each block run); Qwen3 keeps HF past_key_values. Within the
    context both give the logits of KestrelBackend.logits."""

    def __init__(self, backend):
        self.b, self.fed = backend, 0
        self.cache = {} if backend.titans else None

    def logits(self, ids):
        b = self.b
        x = torch.tensor([ids[self.fed:]], device=b.device)
        if b.titans:
            out = b.model(x, cache=self.cache, logits_to_keep=1)
        else:
            out = b.model(x, past_key_values=self.cache, use_cache=True, logits_to_keep=1)
            self.cache = out.past_key_values
        self.fed = len(ids)
        return b.masked(out.logits[0, -1].float())


def catalog(root="runs"):
    """Every run under root (newest first), then every run's snapshots."""
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"No runs folder at {root.resolve()}: give one with --root")
    runs = sorted((d for d in root.iterdir() if (d / "latest.pt").exists() and (d / "tokenizer").is_dir()),
                  key=lambda d: (d / "latest.pt").stat().st_mtime, reverse=True)
    load = lambda pt, d: lambda device: KestrelBackend(pt, d / "tokenizer", device)
    # A fine-tuned run (by name) opens on the chat tab, the rest on text.
    modes = lambda d: ("chat", "text") if "sft" in d.name else ("text", "chat")
    entries = [Entry(d.name, load(d / "latest.pt", d), modes(d)) for d in runs]
    entries += [Entry(f"{d.name}/{pt.stem}", load(pt, d), modes(d), group="snapshots")
                for d in sorted(runs) for pt in sorted(d.glob("step_*.pt"))]
    return Catalog("kestrel", entries, [TEXT, CHAT])
