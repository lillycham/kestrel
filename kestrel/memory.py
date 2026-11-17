"""Titans-MAG (Memory as Gate) causal language model with chunked closed-form neural memory."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding, apply_rotary_pos_emb


def _silu_and_grad(x):
    sig = torch.sigmoid(x)
    silu = x * sig
    grad = sig * (1.0 + x * (1.0 - sig))
    return silu, grad


def memory_retrieve(qc, w1, w2):
    return torch.bmm(F.silu(torch.bmm(qc, w1.transpose(1, 2))), w2.transpose(1, 2))


def memory_update(state, kc, vc, a_i, e_i, t_i, chunk_size):
    """One momentum step on the closed-form gradient of 0.5 / C * ||M(kc) - vc||^2.

    The state is (w1, w2, s1, s2), and the forget gate decays W toward 0. W = 0 is
    then a trap: silu(0) = 0 and the MLP has no biases, so both gradients vanish
    and a wiped memory never writes again. A state of (w1, w2, s1, s2, i1, i2)
    decays toward the initial weights (i1, i2) instead, so a full forget
    (a_i = 1) resets the memory to W_init and it can still learn.
    """
    w1, w2, s1, s2 = state[:4]
    hk = torch.bmm(kc, w1.transpose(1, 2))  # (B * H, C, m_h)
    ak, dsilu = _silu_and_grad(hk)
    pred = torch.bmm(ak, w2.transpose(1, 2))
    err = (pred - vc) * (1.0 / chunk_size)  # (B * H, C, d_h)

    gw2 = torch.bmm(err.transpose(1, 2), ak)  # (B * H, d_h, m_h)
    dak = torch.bmm(err, w2) * dsilu          # (B * H, C, m_h)
    gw1 = torch.bmm(dak.transpose(1, 2), kc)  # (B * H, m_h, d_h)

    s1 = e_i * s1 - t_i * gw1
    s2 = e_i * s2 - t_i * gw2
    if len(state) == 4:
        return (1.0 - a_i) * w1 + s1, (1.0 - a_i) * w2 + s2, s1, s2
    # i + (1 - a)(w - i), written so that autograd saves only w and i, which exist
    # anyway. The form with (w - i) saves a new (B * H, m_h, d_h) tensor per chunk.
    i1, i2 = state[4:]
    return (1.0 - a_i) * w1 + a_i * i1 + s1, (1.0 - a_i) * w2 + a_i * i2 + s2, s1, s2, i1, i2


def memory_step(state, kc, vc, a_i, e_i, t_i, q_next, chunk_size):
    """Update on one chunk, then retrieve the next chunk from the new state.

    One call per chunk (instead of an update call and a retrieve call) halves
    the compiled-function calls and autograd nodes of the recurrence.
    """
    state = memory_update(state, kc, vc, a_i, e_i, t_i, chunk_size)
    return state, memory_retrieve(q_next, state[0], state[1])


class NeuralMemory(nn.Module):
    """Two-layer fast-weight MLP memory updated per chunk via closed-form MSE gradients."""

    def __init__(self, hidden_size, memory_hidden_size, chunk_size, heads=1, forget_bias=-2.0, min_forget=0.0,
                 decay_to_init=False):
        super().__init__()
        self.decay_to_init = decay_to_init
        self.hidden_size = hidden_size
        self.memory_hidden_size = memory_hidden_size
        self.chunk_size = chunk_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.mem_head_dim = memory_hidden_size // heads
        self.forget_bias = forget_bias
        self.min_forget = min_forget
        self._compiled_retrieve = None
        self._compiled_update = None
        self._compiled_step = None
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Per-head, per-chunk scalar gates: alpha (forget), eta (momentum), theta (inner lr).
        self.gate_proj = nn.Linear(hidden_size, 3 * heads, bias=False)
        init_shape_w1 = (memory_hidden_size, hidden_size) if heads == 1 else (heads, self.mem_head_dim, self.head_dim)
        init_shape_w2 = (hidden_size, memory_hidden_size) if heads == 1 else (heads, self.head_dim, self.mem_head_dim)
        self.w1_init = nn.Parameter(torch.empty(*init_shape_w1))
        self.w2_init = nn.Parameter(torch.empty(*init_shape_w2))
        nn.init.normal_(self.w1_init, std=self.head_dim ** -0.5)
        nn.init.normal_(self.w2_init, std=self.mem_head_dim ** -0.5)

    def _gates(self, chunk_ctx):
        """Per-head forget (alpha), momentum (eta) and inner-lr (theta) gates, shaped (B * H, n, 1, 1)."""
        bsz, num_chunks, _ = chunk_ctx.shape
        gates = self.gate_proj(chunk_ctx).view(bsz, num_chunks, self.heads, 3).permute(0, 2, 1, 3).contiguous()
        gates = gates.view(bsz * self.heads, num_chunks, 3).float()
        alpha = (self.min_forget + (1.0 - self.min_forget) * torch.sigmoid(gates[..., 0] + self.forget_bias))[..., None, None]
        eta = torch.sigmoid(gates[..., 1])[..., None, None]
        theta = 0.05 * torch.sigmoid(gates[..., 2])[..., None, None]
        return alpha, eta, theta

    def _initial_state(self, bsz):
        w1_base = self.w1_init.unsqueeze(0) if self.heads == 1 else self.w1_init
        w2_base = self.w2_init.unsqueeze(0) if self.heads == 1 else self.w2_init
        w1 = w1_base.float().unsqueeze(0).expand(bsz, -1, -1, -1).reshape(bsz * self.heads, self.mem_head_dim, self.head_dim).clone()
        w2 = w2_base.float().unsqueeze(0).expand(bsz, -1, -1, -1).reshape(bsz * self.heads, self.head_dim, self.mem_head_dim).clone()
        anchors = (w1, w2) if self.decay_to_init else ()  # Nothing writes in place, so W may share them.
        return (w1, w2, torch.zeros_like(w1), torch.zeros_like(w2)) + anchors

    def _retrieve(self, qc, w1, w2):
        if self._compiled_retrieve is None:
            return memory_retrieve(qc, w1, w2)
        # Chunk slices otherwise carry a stride that changes with sequence length.
        return self._compiled_retrieve(qc.contiguous(), w1, w2)

    def _update(self, state, kc, vc, a_i, e_i, t_i):
        if self._compiled_update is None:
            return memory_update(state, kc, vc, a_i, e_i, t_i, self.chunk_size)
        return self._compiled_update(state, kc.contiguous(), vc.contiguous(), a_i.contiguous(),
                                     e_i.contiguous(), t_i.contiguous(), self.chunk_size)

    def _step(self, state, kc, vc, a_i, e_i, t_i, q_next):
        if self._compiled_step is None:
            return memory_step(state, kc, vc, a_i, e_i, t_i, q_next, self.chunk_size)
        return self._compiled_step(state, kc, vc, a_i, e_i, t_i, q_next, self.chunk_size)

    def forward(self, x, cache=None):
        if cache is not None:
            return self._forward_cached(x, cache)
        return self.finish(self.recur(*self.prepare(x)), x)

    def prepare(self, x):
        """Project a whole sequence for the chunk recurrence. Free of Python branches on its length.

        Returns chunk-major tensors, so chunk idx of q, k, v and of each gate
        is a contiguous slice whose strides do not depend on the chunk count:
        q, k, v are (n, B * H, C, d_h) and alpha, eta, theta are (n, B * H, 1, 1).
        """
        bsz, seq_len, dim = x.shape
        c = self.chunk_size
        num_chunks = (seq_len + c - 1) // c
        x = F.pad(x, (0, 0, 0, num_chunks * c - seq_len))

        def project(proj):
            h = F.normalize(proj(x).view(bsz, num_chunks, c, self.heads, self.head_dim), dim=-1)
            h = h.permute(1, 0, 3, 2, 4).contiguous()
            return h.view(num_chunks, bsz * self.heads, c, self.head_dim).float()

        # Average chunk hidden state drives scalar forget/momentum/lr gates per head.
        gates = self._gates(x.view(bsz, num_chunks, c, dim).mean(dim=2))
        alpha, eta, theta = (g.transpose(0, 1).contiguous() for g in gates)
        return project(self.q_proj), project(self.k_proj), project(self.v_proj), alpha, eta, theta, self._initial_state(bsz)

    def recur(self, q, k, v, alpha, eta, theta, state):
        """The sequential chunk loop. Returns retrievals as (n, B * H, C, d_h)."""
        # Retrieve each chunk from the state before its own update (causal).
        outputs = [self._retrieve(q[0], state[0], state[1])]
        for idx in range(q.shape[0] - 1):
            state, retrieved = self._step(state, k[idx], v[idx], alpha[idx], eta[idx], theta[idx], q[idx + 1])
            outputs.append(retrieved)
        return torch.stack(outputs)

    def finish(self, retrieved, x):
        """Map retrievals back to (B, L, D) in the dtype of x, and project them."""
        bsz, seq_len, dim = x.shape
        num_chunks = retrieved.shape[0]
        out = retrieved.view(num_chunks, bsz, self.heads, self.chunk_size, self.head_dim).permute(1, 0, 3, 2, 4)
        out = out.reshape(bsz, num_chunks * self.chunk_size, dim)[:, :seq_len].to(x.dtype)
        return self.out_proj(out)

    def _forward_cached(self, x, cache):
        """Streaming forward for decoding. Gives the same outputs as forward() on the whole sequence.

        The cache holds the memory state after the last complete chunk, plus the
        hidden states, keys and values of the current incomplete chunk. The memory
        updates only when a chunk is complete, as in the parallel path.
        """
        bsz, seq_len, dim = x.shape
        c = self.chunk_size
        if "state" not in cache:
            cache["state"] = self._initial_state(bsz)
            cache["x"] = x.new_zeros(bsz, 0, dim)
            cache["k"] = x.new_zeros(bsz * self.heads, 0, self.head_dim, dtype=torch.float32)
            cache["v"] = x.new_zeros(bsz * self.heads, 0, self.head_dim, dtype=torch.float32)

        def project(proj):
            h = F.normalize(proj(x).view(bsz, seq_len, self.heads, self.head_dim), dim=-1)
            return h.transpose(1, 2).reshape(bsz * self.heads, seq_len, self.head_dim).float()

        q, k, v = project(self.q_proj), project(self.k_proj), project(self.v_proj)
        outputs = []
        start = 0
        while start < seq_len:
            n = min(c - cache["x"].shape[1], seq_len - start)
            span = slice(start, start + n)
            w1, w2 = cache["state"][:2]
            outputs.append(self._retrieve(q[:, span], w1, w2))
            cache["x"] = torch.cat([cache["x"], x[:, span]], dim=1)
            cache["k"] = torch.cat([cache["k"], k[:, span]], dim=1)
            cache["v"] = torch.cat([cache["v"], v[:, span]], dim=1)
            if cache["x"].shape[1] == c:
                alpha, eta, theta = self._gates(cache["x"].mean(dim=1, keepdim=True))
                cache["state"] = self._update(cache["state"], cache["k"], cache["v"], alpha[:, 0], eta[:, 0], theta[:, 0])
                cache["x"], cache["k"], cache["v"] = cache["x"][:, :0], cache["k"][:, :0], cache["v"][:, :0]
            start += n

        out = torch.cat(outputs, dim=1).view(bsz, self.heads, seq_len, self.head_dim)
        out = out.transpose(1, 2).reshape(bsz, seq_len, dim).to(x.dtype)
        return self.out_proj(out)


class SlidingWindowAttention(nn.Module):
    """GQA with QK-norm, RoPE, and a local sliding causal window."""

    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        self.window = config.attention_window
        self.flash_attention = None

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin, cache=None):
        bsz, seq_len, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if self.flash_attention is not None and cache is None:
            # Native GQA avoids both KV head expansion and overlapping KV blocks.
            if not x.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("flash2 attention requires CUDA and FP16 or BF16 autocast")
            attn = self.flash_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                        causal=True, window_size=(self.window - 1, 0), dropout_p=0.0)
            return self.o_proj(attn.reshape(bsz, seq_len, -1))
        if cache is not None:
            # Keys and values are stored after QK-norm and RoPE. RoPE is relative,
            # so the cache only needs the last window - 1 positions.
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            keep = self.window - 1
            cache["k"] = k[:, :, max(0, k.shape[2] - keep):]
            cache["v"] = v[:, :, max(0, v.shape[2] - keep):]
        key_len = k.shape[2]
        past = key_len - seq_len
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        if past == 0 and self.window >= seq_len:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        elif seq_len == 1 and key_len <= self.window:
            attn = F.scaled_dot_product_attention(q, k, v)
        elif past == 0:
            attn = local_window_attention(q, k, v, self.window)
        else:
            attn = dense_window_attention(q, k, v, self.window, past)
        return self.o_proj(attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1))


def dense_window_attention(q, k, v, window, past=0):
    """Sliding-window causal attention with a full (queries, keys) mask. q is (B, H, L, D); k and v hold past + L keys."""
    pos = torch.arange(k.shape[2], device=q.device)
    diff = pos[past:, None] - pos[None, :]
    mask = (diff >= 0) & (diff < window)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def local_window_attention(q, k, v, window):
    """Sliding-window causal attention in blocks of `window` queries. Same result as dense_window_attention.

    Query t sees keys t - window + 1 .. t, which all lie in its own block or the
    block before. So each block attends to 2 * window keys, and the cost is
    O(L * window) rather than O(L^2). Inputs are (B, H, L, D) with no past keys.
    """
    bsz, heads, seq_len, dim = q.shape
    blocks = -(-seq_len // window)
    pad = blocks * window - seq_len
    # Keep this branch. A dynamic-shape graph splits on it (two variants), but
    # without it F.pad would get zero padding on these transposed inputs. Then
    # it returns a strided clone, while traced graphs assume a contiguous result.
    if pad:
        q, k, v = (F.pad(t, (0, 0, 0, pad)) for t in (q, k, v))

    def split(t):
        return t.reshape(bsz, heads, blocks, window, dim)

    def with_previous(t):
        t = split(t)
        return torch.cat([F.pad(t, (0, 0, 0, 0, 1, 0))[:, :, :-1], t], dim=3)  # (B, H, n, 2W, D)

    qb, kb, vb = split(q), with_previous(k), with_previous(v)
    # Query a of a block against key c of [previous block, this block]: distance a + W - c.
    a = torch.arange(window, device=q.device)[:, None]
    c = torch.arange(2 * window, device=q.device)[None, :]
    diff = a + window - c
    band = (diff >= 0) & (diff < window)
    mask = band.expand(blocks, window, 2 * window).clone()
    mask[0, :, :window] = False  # The first block has no previous block.
    flat = lambda t: t.reshape(bsz, heads * blocks, *t.shape[3:])
    attn = F.scaled_dot_product_attention(flat(qb), flat(kb), flat(vb), attn_mask=mask.repeat(heads, 1, 1))
    return attn.reshape(bsz, heads, blocks * window, dim)[:, :, :seq_len]


class TitansMAGBlock(nn.Module):
    """Parallel sliding-window attention + neural memory combined via a learned gate."""

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = SlidingWindowAttention(config)
        self.attn_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps) if getattr(config, "norm_attn_branch", False) else nn.Identity()
        self.memory = NeuralMemory(
            config.hidden_size, config.memory_hidden_size, config.memory_chunk_size,
            heads=getattr(config, "memory_heads", 1),
            forget_bias=getattr(config, "memory_forget_bias", -2.0),
            min_forget=getattr(config, "memory_min_forget", 0.0),
            decay_to_init=getattr(config, "memory_decay_to_init", False),
        )
        self.mem_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mag_gate = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)
        self._compiled_before = None
        self._compiled_after = None
        # A list while TitansMAGForCausalLM.memory_health runs: each call appends its statistics.
        self.health = None

    def forward(self, x, cos, sin, cache=None):
        if cache is not None:
            h = self.input_layernorm(x)
            attn_out = self.attn_norm(self.self_attn(h, cos, sin, cache.setdefault("attn", {})))
            return self._combine(x, h, attn_out, self.memory(h, cache.setdefault("mem", {})))
        if self._compiled_before is None:
            h, attn_out, prepared = self._before_memory(x, cos, sin)
            retrieved = self.memory.recur(*prepared)
            if self.health is not None:
                self.health.append(self._memory_health(h, attn_out, prepared[3], retrieved))
            return self._after_memory(x, h, attn_out, retrieved)
        # Only the length dimensions are dynamic. Batch and width stay static,
        # so no guard can tie the batch size to an equal length.
        for tensor in (x, cos, sin):
            torch._dynamo.maybe_mark_dynamic(tensor, 1)
        h, attn_out, prepared = self._compiled_before(x, cos, sin)
        retrieved = self.memory.recur(*prepared)
        torch._dynamo.maybe_mark_dynamic(h, 1)
        torch._dynamo.maybe_mark_dynamic(attn_out, 1)
        torch._dynamo.maybe_mark_dynamic(retrieved, 0)
        return self._compiled_after(x, h, attn_out, retrieved)

    def _before_memory(self, x, cos, sin):
        h = self.input_layernorm(x)
        return h, self.attn_norm(self.self_attn(h, cos, sin)), self.memory.prepare(h)

    def _after_memory(self, x, h, attn_out, retrieved):
        return self._combine(x, h, attn_out, self.memory.finish(retrieved, h))

    def _combine(self, x, h, attn_out, memory_out):
        mem_out = self.mem_norm(memory_out)
        gate = torch.sigmoid(self.mag_gate(h))
        x = x + (1.0 - gate) * attn_out + gate * mem_out
        return x + self.mlp(self.post_attention_layernorm(x))

    def _memory_health(self, h, attn_out, alpha, retrieved):
        """How much this block call uses its memory. alpha is (n, B * H, 1, 1).

        share is the memory's part of the gated update after the first chunk,
        which is the only chunk that reads W_init before any write. A memory
        whose forget gate wiped it to W = 0 cannot write again (both update
        gradients vanish there), and its share falls to about 0.
        """
        c = self.memory.chunk_size
        gate = torch.sigmoid(self.mag_gate(h)).float()
        mem = (gate * self.mem_norm(self.memory.finish(retrieved, h)).float()).norm(dim=-1)[:, c:]
        attn = ((1.0 - gate) * attn_out.float()).norm(dim=-1)[:, c:]
        share = (mem / (mem + attn).clamp_min(1e-12)).mean() if mem.numel() else torch.full((), float("nan"))
        alpha = alpha[..., 0, 0].view(alpha.shape[0], -1, self.memory.heads).mean(dim=(0, 1))
        return {"gate": gate.mean(), "share": share, "alpha": alpha}


class TitansMAGForCausalLM(nn.Module):
    """Titans-MAG causal LM with optional prelude/coda and looped recurrent core."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False
        self.num_loops = getattr(config, "num_loops", 1)
        prelude_layers = getattr(config, "prelude_layers", 0)
        coda_layers = getattr(config, "coda_layers", 0)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=getattr(config, "pad_token_id", None))
        if prelude_layers:
            self.prelude = nn.ModuleList([TitansMAGBlock(config) for _ in range(prelude_layers)])
        self.layers = nn.ModuleList([TitansMAGBlock(config) for _ in range(config.num_hidden_layers)])
        if self.num_loops > 1:
            self.loop_x0_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.loop_inject = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
            # 3D step embedding stays in AdamW (no_decay) rather than Muon.
            self.loop_embed = nn.Parameter(torch.empty(self.num_loops, 1, config.hidden_size))
        if coda_layers:
            self.coda = nn.ModuleList([TitansMAGBlock(config) for _ in range(coda_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if self.num_loops > 1:
            nn.init.normal_(self.loop_embed, mean=0.0, std=0.02)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, cache=None, logits_to_keep=0, inputs_embeds=None, hidden_only=False, **kwargs):
        """Run the model. With a cache dict, continue from the tokens that it holds.

        The cache is a plain dict that this method fills on the first call. Each
        block call (prelude, every loop of the core, coda) gets its own entry.
        With inputs_embeds and hidden_only (the H-Net main network), it maps
        vectors to final-normed vectors and never uses the embedding or head.
        """
        x = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        offset = 0 if cache is None else cache.get("pos", 0)
        seq_len = x.shape[1]
        pos_ids = torch.arange(offset, offset + seq_len, device=x.device).unsqueeze(0)
        cos, sin = self.rotary_emb(x, pos_ids)
        layer_caches = None if cache is None else iter(cache.setdefault("layers", [{} for _ in range(self._block_calls())]))

        def run(layer, x):
            if layer_caches is None and self.gradient_checkpointing and torch.is_grad_enabled():
                return checkpoint(layer, x, cos, sin, use_reentrant=False)
            return layer(x, cos, sin, None if layer_caches is None else next(layer_caches))

        if hasattr(self, "prelude"):
            for layer in self.prelude:
                x = run(layer, x)
        if self.num_loops > 1:
            x0_normed = self.loop_x0_norm(x)
            injection = self.loop_inject(x0_normed)
            for loop_idx in range(self.num_loops):
                x = x + injection + self.loop_embed[loop_idx]
                for layer in self.layers:
                    x = run(layer, x)
        else:
            for layer in self.layers:
                x = run(layer, x)
        if hasattr(self, "coda"):
            for layer in self.coda:
                x = run(layer, x)
        if cache is not None:
            cache["pos"] = offset + seq_len
        if hidden_only:
            return self.norm(x)
        if logits_to_keep:
            x = x[:, -logits_to_keep:]
        logits = self.lm_head(self.norm(x))
        return SimpleNamespace(logits=logits)

    def compile_memory(self, **compile_options):
        # Share the same callables across all layers. Neither function captures a
        # layer instance or the variable number of chunks.
        retrieve = torch.compile(memory_retrieve, **compile_options)
        update = torch.compile(memory_update, **compile_options)
        step = torch.compile(memory_step, **compile_options)
        for module in self.modules():
            if isinstance(module, NeuralMemory):
                module._compiled_retrieve = retrieve
                module._compiled_update = update
                module._compiled_step = step

    def compile_blocks(self, **compile_options):
        """Compile every block for any sequence length, except its sequential memory loop.

        Each block compiles as two regions: before the memory recurrence (norm,
        attention, memory projections) and after it (memory output, gate, MLP).
        The sequence dimension is dynamic, so a new length reuses the graphs.
        All blocks share the code of these regions, so they share the graphs.
        The loop between the regions stays in Python and calls the
        fixed-shape memory kernels of compile_memory.
        """
        self.compile_memory(**compile_options)
        for module in self.modules():
            if isinstance(module, TitansMAGBlock):
                module._compiled_before = torch.compile(module._before_memory, **compile_options)
                module._compiled_after = torch.compile(module._after_memory, **compile_options)

    def _call_labels(self):
        prelude = [f"prelude{i}" for i in range(len(getattr(self, "prelude", [])))]
        if self.num_loops > 1:
            core = [f"loop{l}.core{i}" for l in range(self.num_loops) for i in range(len(self.layers))]
        else:
            core = [f"core{i}" for i in range(len(self.layers))]
        return prelude + core + [f"coda{i}" for i in range(len(getattr(self, "coda", [])))]

    @contextmanager
    def recording_memory(self):
        """Within this context the blocks run uncompiled and append their memory statistics to the yielded list.

        A caller that runs this model inside a larger one (the H-Net main
        network) runs the larger model here, then passes the list to
        summarize_memory.
        """
        blocks = [m for m in self.modules() if isinstance(m, TitansMAGBlock)]
        compiled = [b._compiled_before for b in blocks]
        calls = []
        try:
            for b in blocks:
                b._compiled_before, b.health = None, calls
            yield calls
        finally:
            for b, c in zip(blocks, compiled):
                b._compiled_before, b.health = c, None

    @torch.no_grad()
    def memory_health(self, input_ids, dead_share=0.01, wipe_alpha=0.95):
        """Run input_ids through the uncompiled blocks and summarise how each block call uses its memory."""
        with self.recording_memory() as calls:
            self(input_ids)
        return self.summarize_memory(calls, dead_share, wipe_alpha)

    def summarize_memory(self, calls, dead_share=0.01, wipe_alpha=0.95):
        """A flat dict for the training log from the statistics of one forward pass.

        Per call: the memory share of the gated update after the first chunk,
        the mean MAG gate and the mean forget gate. Overall: the number of dead
        calls (share below dead_share) and the fraction of heads whose forget
        gate wipes the memory.
        """
        out = {}
        for label, s in zip(self._call_labels(), calls, strict=True):
            out[f"memory/share/{label}"] = s["share"].item()
            out[f"memory/gate/{label}"] = s["gate"].item()
            out[f"memory/alpha/{label}"] = s["alpha"].mean().item()
        shares = torch.stack([s["share"] for s in calls])
        alphas = torch.stack([s["alpha"] for s in calls])
        out["memory/dead_calls"] = int((shares < dead_share).sum().item())
        out["memory/wiping_heads"] = (alphas > wipe_alpha).float().mean().item()
        return out

    def set_gradient_checkpointing(self, enabled):
        """Recompute each block in the backward pass instead of storing its activations."""
        self.gradient_checkpointing = enabled

    def _block_calls(self):
        prelude = len(self.prelude) if hasattr(self, "prelude") else 0
        coda = len(self.coda) if hasattr(self, "coda") else 0
        return prelude + self.num_loops * len(self.layers) + coda

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, do_sample=True, temperature=1.0,
                 top_k=0, top_p=1.0, repetition_penalty=1.0, eos_token_id=None, bad_words_ids=None,
                 use_cache=True, num_return_sequences=1, return_logprobs=False, pad_token_id=None, **kwargs):
        """Sample new tokens.

        With use_cache, the prompt runs once and each step then feeds only the
        newest token. This gives the same logits as the uncached path while the
        sequence fits in max_position_embeddings. After that, the cached path
        keeps streaming (attention is local and RoPE is relative), but the
        uncached path drops the oldest tokens and recomputes the memory.

        num_return_sequences samples a group of continuations of one prompt
        (input_ids must have batch size 1). The prompt runs once and its cache
        is copied for each row. All rows stay at the same position, so their
        memory chunks complete on the same step. A row stops at a stop token;
        after that it gets pad_token_id (or the first stop token) until all
        rows stop.

        With return_logprobs, the result is (tokens, logprobs). logprobs has one
        value per new token: the log-probability of that token under the raw
        model distribution (temperature 1, no top-k, top-p or penalty). It is
        0 for padding after a row stops.
        """
        stop_list = list(eos_token_id) if isinstance(eos_token_id, (list, tuple, set)) else ([eos_token_id] if eos_token_id is not None else [])
        bad_ids = [item[0] for item in bad_words_ids] if bad_words_ids else []
        group = num_return_sequences
        if group > 1 and input_ids.shape[0] != 1:
            raise ValueError("num_return_sequences > 1 needs a batch of one prompt")
        fill = pad_token_id if pad_token_id is not None else (stop_list[0] if stop_list else 0)
        stop_ids = torch.tensor(stop_list, device=input_ids.device, dtype=input_ids.dtype)
        max_ctx = self.config.max_position_embeddings
        cache = {} if use_cache else None
        tokens = input_ids
        if cache is not None:
            logits = self(tokens[:, -max_ctx:], cache=cache, logits_to_keep=1).logits[:, -1, :]
            if group > 1:
                cache = _repeat_batch(cache, group)
                logits = logits.repeat(group, 1)
        tokens = tokens.repeat(group, 1)
        finished = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        logprobs = []
        for step in range(max_new_tokens):
            if cache is None:
                logits = self(tokens[:, -max_ctx:]).logits[:, -1, :]
            elif step > 0:
                logits = self(tokens[:, -1:], cache=cache, logits_to_keep=1).logits[:, -1, :]
            raw = logits.float().log_softmax(dim=-1) if return_logprobs else None
            if repetition_penalty != 1.0:
                score = torch.gather(logits, 1, tokens)
                score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                logits = logits.scatter(1, tokens, score)
            if bad_ids:
                logits[:, bad_ids] = float("-inf")
            if not do_sample:
                next_tok = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits = logits.masked_fill(logits < v[:, [-1]], float("-inf"))
                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    probs = F.softmax(sorted_logits, dim=-1)
                    remove = torch.cumsum(probs, dim=-1) - probs > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    logits = logits.scatter(1, sorted_idx, sorted_logits)
                next_tok = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            next_tok = next_tok.masked_fill(finished[:, None], fill)
            if return_logprobs:
                logprobs.append(raw.gather(1, next_tok).squeeze(1).masked_fill(finished, 0.0))
            tokens = torch.cat([tokens, next_tok], dim=-1)
            finished = finished | torch.isin(next_tok[:, 0], stop_ids)
            if bool(finished.all()):
                break
        if return_logprobs:
            empty = tokens.new_zeros(tokens.shape[0], 0, dtype=torch.float32)
            return tokens, torch.stack(logprobs, dim=1) if logprobs else empty
        return tokens


def _repeat_batch(cache, n):
    """Copy a batch-1 decode cache n times along the batch dimension.

    Memory tensors use a (batch * heads) leading dimension in batch-major order,
    so tiling a batch-1 cache with repeat gives the right layout for both kinds.
    """
    if isinstance(cache, torch.Tensor):
        return cache.repeat(n, *([1] * (cache.ndim - 1)))
    if isinstance(cache, dict):
        return {key: _repeat_batch(value, n) for key, value in cache.items()}
    if isinstance(cache, (list, tuple)):
        return type(cache)(_repeat_batch(value, n) for value in cache)
    return cache
