"""Titans-MAG (Memory as Gate) causal language model with chunked closed-form neural memory."""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding, apply_rotary_pos_emb


def _silu_and_grad(x):
    sig = torch.sigmoid(x)
    silu = x * sig
    grad = sig * (1.0 + x * (1.0 - sig))
    return silu, grad


class NeuralMemory(nn.Module):
    """Two-layer fast-weight MLP memory updated per chunk via closed-form MSE gradients."""

    def __init__(self, hidden_size, memory_hidden_size, chunk_size, heads=1, forget_bias=-2.0, min_forget=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_hidden_size = memory_hidden_size
        self.chunk_size = chunk_size
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.mem_head_dim = memory_hidden_size // heads
        self.forget_bias = forget_bias
        self.min_forget = min_forget
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

    def _project_qkv(self, proj, x, bsz, num_chunks):
        h = proj(x).view(bsz, num_chunks, self.chunk_size, self.heads, self.head_dim)
        h = F.normalize(h, dim=-1).permute(0, 3, 1, 2, 4).contiguous()
        return h.view(bsz * self.heads, num_chunks, self.chunk_size, self.head_dim).float()

    def forward(self, x):
        bsz, seq_len, dim = x.shape
        c = self.chunk_size
        pad = (c - seq_len % c) % c
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        padded_len = x.shape[1]
        num_chunks = padded_len // c

        q = self._project_qkv(self.q_proj, x, bsz, num_chunks)
        k = self._project_qkv(self.k_proj, x, bsz, num_chunks)
        v = self._project_qkv(self.v_proj, x, bsz, num_chunks)
        # Average chunk hidden state drives scalar forget/momentum/lr gates per head.
        chunk_ctx = x.view(bsz, num_chunks, c, dim).mean(dim=2)
        gates = self.gate_proj(chunk_ctx).view(bsz, num_chunks, self.heads, 3).permute(0, 2, 1, 3).contiguous()
        gates = gates.view(bsz * self.heads, num_chunks, 3).float()
        alpha = (self.min_forget + (1.0 - self.min_forget) * torch.sigmoid(gates[..., 0] + self.forget_bias))[..., None, None]
        eta = torch.sigmoid(gates[..., 1])[..., None, None]
        theta = 0.05 * torch.sigmoid(gates[..., 2])[..., None, None]

        w1_base = self.w1_init.unsqueeze(0) if self.heads == 1 else self.w1_init
        w2_base = self.w2_init.unsqueeze(0) if self.heads == 1 else self.w2_init
        w1 = w1_base.float().unsqueeze(0).expand(bsz, -1, -1, -1).reshape(bsz * self.heads, self.mem_head_dim, self.head_dim).clone()
        w2 = w2_base.float().unsqueeze(0).expand(bsz, -1, -1, -1).reshape(bsz * self.heads, self.head_dim, self.mem_head_dim).clone()
        s1 = torch.zeros_like(w1)
        s2 = torch.zeros_like(w2)

        outputs = []
        inv_c = 1.0 / c
        for idx in range(num_chunks):
            qc = q[:, idx]  # (B * H, C, d_h)
            kc = k[:, idx]
            vc = v[:, idx]

            # Retrieve from current memory state (before updating on chunk idx -> causal).
            hq = torch.bmm(qc, w1.transpose(1, 2))
            outputs.append(torch.bmm(F.silu(hq), w2.transpose(1, 2)))

            # Closed-form gradient of 0.5 / C * ||M(kc) - vc||^2 wrt w1, w2.
            hk = torch.bmm(kc, w1.transpose(1, 2))  # (B * H, C, m_h)
            ak, dsilu = _silu_and_grad(hk)
            pred = torch.bmm(ak, w2.transpose(1, 2))
            err = (pred - vc) * inv_c  # (B * H, C, d_h)

            gw2 = torch.bmm(err.transpose(1, 2), ak)  # (B * H, d_h, m_h)
            dak = torch.bmm(err, w2) * dsilu          # (B * H, C, m_h)
            gw1 = torch.bmm(dak.transpose(1, 2), kc)  # (B * H, m_h, d_h)

            a_i = alpha[:, idx]
            e_i = eta[:, idx]
            t_i = theta[:, idx]
            s1 = e_i * s1 - t_i * gw1
            s2 = e_i * s2 - t_i * gw2
            w1 = (1.0 - a_i) * w1 + s1
            w2 = (1.0 - a_i) * w2 + s2

        out = torch.cat(outputs, dim=1).view(bsz, self.heads, padded_len, self.head_dim)
        out = out.permute(0, 2, 1, 3).contiguous().view(bsz, padded_len, dim)[:, :seq_len].to(x.dtype)
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

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x, cos, sin):
        bsz, seq_len, _ = x.shape
        q = self.q_norm(self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim)).transpose(1, 2)
        k = self.k_norm(self.k_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)

        if self.window >= seq_len:
            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            pos = torch.arange(seq_len, device=x.device)
            diff = pos[:, None] - pos[None, :]
            mask = (diff >= 0) & (diff < self.window)
            attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o_proj(attn.transpose(1, 2).contiguous().view(bsz, seq_len, -1))


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
        )
        self.mem_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mag_gate = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def forward(self, x, cos, sin):
        h = self.input_layernorm(x)
        attn_out = self.attn_norm(self.self_attn(h, cos, sin))
        mem_out = self.mem_norm(self.memory(h))
        gate = torch.sigmoid(self.mag_gate(h))
        x = x + (1.0 - gate) * attn_out + gate * mem_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TitansMAGForCausalLM(nn.Module):
    """Titans-MAG causal LM with optional prelude/coda and looped recurrent core."""

    def __init__(self, config):
        super().__init__()
        self.config = config
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

    def forward(self, input_ids, **kwargs):
        x = self.embed_tokens(input_ids)
        pos_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(x, pos_ids)
        if hasattr(self, "prelude"):
            for layer in self.prelude:
                x = layer(x, cos, sin)
        if self.num_loops > 1:
            x0_normed = self.loop_x0_norm(x)
            for loop_idx in range(self.num_loops):
                x = x + self.loop_inject(x0_normed) + self.loop_embed[loop_idx]
                for layer in self.layers:
                    x = layer(x, cos, sin)
        else:
            for layer in self.layers:
                x = layer(x, cos, sin)
        if hasattr(self, "coda"):
            for layer in self.coda:
                x = layer(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        return SimpleNamespace(logits=logits)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, do_sample=True, temperature=1.0,
                 top_k=0, top_p=1.0, repetition_penalty=1.0, eos_token_id=None, bad_words_ids=None, **kwargs):
        stop_ids = set(eos_token_id) if isinstance(eos_token_id, (list, tuple, set)) else ({eos_token_id} if eos_token_id is not None else set())
        bad_ids = [item[0] for item in bad_words_ids] if bad_words_ids else []
        tokens = input_ids
        for _ in range(max_new_tokens):
            ctx = tokens[:, -self.config.max_position_embeddings:]
            logits = self(ctx).logits[:, -1, :]
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
            tokens = torch.cat([tokens, next_tok], dim=-1)
            if next_tok.item() in stop_ids:
                break
        return tokens
