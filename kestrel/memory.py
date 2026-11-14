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

    def __init__(self, hidden_size, memory_hidden_size, chunk_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_hidden_size = memory_hidden_size
        self.chunk_size = chunk_size
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        # Per-chunk scalar gates: alpha (forget), eta (momentum), theta (inner lr).
        self.gate_proj = nn.Linear(hidden_size, 3, bias=False)
        self.w1_init = nn.Parameter(torch.empty(memory_hidden_size, hidden_size))
        self.w2_init = nn.Parameter(torch.empty(hidden_size, memory_hidden_size))
        nn.init.normal_(self.w1_init, std=hidden_size ** -0.5)
        nn.init.normal_(self.w2_init, std=memory_hidden_size ** -0.5)

    def forward(self, x):
        bsz, seq_len, dim = x.shape
        c = self.chunk_size
        pad = (c - seq_len % c) % c
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        padded_len = x.shape[1]
        num_chunks = padded_len // c

        q = F.normalize(self.q_proj(x), dim=-1).view(bsz, num_chunks, c, dim).float()
        k = F.normalize(self.k_proj(x), dim=-1).view(bsz, num_chunks, c, dim).float()
        v = F.normalize(self.v_proj(x), dim=-1).view(bsz, num_chunks, c, dim).float()
        # Average chunk hidden state drives scalar forget/momentum/lr gates.
        chunk_ctx = x.view(bsz, num_chunks, c, dim).mean(dim=2)
        gates = self.gate_proj(chunk_ctx).float()
        alpha = torch.sigmoid(gates[..., 0] - 2.0)[..., None, None]  # conservative forgetting
        eta = torch.sigmoid(gates[..., 1])[..., None, None]
        theta = 0.05 * torch.sigmoid(gates[..., 2])[..., None, None]

        w1 = self.w1_init.float().unsqueeze(0).expand(bsz, -1, -1).clone()
        w2 = self.w2_init.float().unsqueeze(0).expand(bsz, -1, -1).clone()
        s1 = torch.zeros_like(w1)
        s2 = torch.zeros_like(w2)

        outputs = []
        inv_c = 1.0 / c
        for idx in range(num_chunks):
            qc = q[:, idx]  # (B, C, D)
            kc = k[:, idx]
            vc = v[:, idx]

            # Retrieve from current memory state (before updating on chunk idx -> causal).
            hq = torch.bmm(qc, w1.transpose(1, 2))
            outputs.append(torch.bmm(F.silu(hq), w2.transpose(1, 2)))

            # Closed-form gradient of 0.5 / C * ||M(kc) - vc||^2 wrt w1, w2.
            hk = torch.bmm(kc, w1.transpose(1, 2))  # (B, C, M)
            ak, dsilu = _silu_and_grad(hk)
            pred = torch.bmm(ak, w2.transpose(1, 2))
            err = (pred - vc) * inv_c  # (B, C, D)

            gw2 = torch.bmm(err.transpose(1, 2), ak)  # (B, D, M)
            dak = torch.bmm(err, w2) * dsilu          # (B, C, M)
            gw1 = torch.bmm(dak.transpose(1, 2), kc)  # (B, M, D)

            a_i = alpha[:, idx]
            e_i = eta[:, idx]
            t_i = theta[:, idx]
            s1 = e_i * s1 - t_i * gw1
            s2 = e_i * s2 - t_i * gw2
            w1 = (1.0 - a_i) * w1 + s1
            w2 = (1.0 - a_i) * w2 + s2

        out = torch.cat(outputs, dim=1)[:, :seq_len].to(x.dtype)
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
        self.memory = NeuralMemory(config.hidden_size, config.memory_hidden_size, config.memory_chunk_size)
        self.mem_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mag_gate = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def forward(self, x, cos, sin):
        h = self.input_layernorm(x)
        attn_out = self.self_attn(h, cos, sin)
        mem_out = self.mem_norm(self.memory(h))
        gate = torch.sigmoid(self.mag_gate(h))
        x = x + (1.0 - gate) * attn_out + gate * mem_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class TitansMAGForCausalLM(nn.Module):
    """Titans-MAG causal LM compatible with Kestrel's trainer, optimizers, and sampler."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=getattr(config, "pad_token_id", None))
        self.layers = nn.ModuleList([TitansMAGBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, **kwargs):
        x = self.embed_tokens(input_ids)
        pos_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        cos, sin = self.rotary_emb(x, pos_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        return SimpleNamespace(logits=logits)

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=64, do_sample=True, temperature=1.0,
                 top_k=0, top_p=1.0, eos_token_id=None, bad_words_ids=None, **kwargs):
        stop_ids = set(eos_token_id) if isinstance(eos_token_id, (list, tuple, set)) else ({eos_token_id} if eos_token_id is not None else set())
        bad_ids = [item[0] for item in bad_words_ids] if bad_words_ids else []
        tokens = input_ids
        for _ in range(max_new_tokens):
            ctx = tokens[:, -self.config.max_position_embeddings:]
            logits = self(ctx).logits[:, -1, :]
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
