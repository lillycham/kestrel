"""One-stage H-Net: a byte-level causal LM with dynamic chunking.

After Hwang, Wang and Gu, "Dynamic Chunking for End-to-End Hierarchical
Sequence Modeling" (2025). The shape is

    bytes -> embed -> encoder -> router -> downsample -> main network
          -> smoothing -> upsample (+ residual) -> decoder -> byte logits

The encoder and decoder are sliding-window attention blocks at byte width (the
paper uses Mamba-2, which needs CUDA kernels). The main network runs on the
chunk sequence and is either a plain Qwen3 decoder or Looped Titans-MAG.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding

from .memory import SlidingWindowAttention, TitansMAGForCausalLM


class SWABlock(nn.Module):
    """Pre-norm transformer block with sliding-window attention."""

    def __init__(self, config):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = SlidingWindowAttention(config)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = Qwen3MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        return x + self.mlp(self.post_attention_layernorm(x))


def boundary_probs(q, k):
    """p_t = (1 - cos(q_t, k_{t-1})) / 2, with p_0 = 1. Inputs (B, L, d); output fp32 (B, L)."""
    q = F.normalize(q.float(), dim=-1)
    k = F.normalize(k.float(), dim=-1)
    p = ((1.0 - (q[:, 1:] * k[:, :-1]).sum(-1)) / 2).clamp(0.0, 1.0)
    return torch.cat([p.new_ones(p.shape[0], 1), p], dim=1)


def ratio_loss(p, b, target):
    """The H-Net ratio loss. It is 1 when the boundary rate F and mean probability G both equal 1/target."""
    f = b.float().mean()
    g = p.mean()
    return target / (target - 1) * ((target - 1) * f * g + (1 - f) * (1 - g))


def downsample(h, b):
    """Gather the boundary positions of each row to the front, in order.

    Returns chunks (B, M, d), the positions (B, M) and a validity mask (B, M),
    where M is the most chunks in any row. Rows with fewer chunks are padded at
    the end with non-boundary positions; causal attention keeps the padding from
    reaching the real chunks.
    """
    counts = b.sum(dim=1)
    most = int(counts.max())
    order = torch.argsort((~b).to(torch.int8), dim=1, stable=True)[:, :most]
    valid = torch.arange(most, device=b.device)[None, :] < counts[:, None]
    chunks = torch.gather(h, 1, order[..., None].expand(-1, -1, h.shape[-1]))
    return chunks, order, valid


def smooth(z, p_chunk):
    """EMA over the chunk sequence: zbar_j = P_j z_j + (1 - P_j) zbar_{j-1}.

    Closed form zbar_j = sum_{i <= j} P_i prod_{i < m <= j} (1 - P_m) z_i, as an
    (M, M) lower-triangular weight matrix. This is cheap at our chunk counts.
    P_0 = 1 for every row, so the clamp on log(1 - P) never changes the result.
    """
    p_chunk = p_chunk.float()
    log_keep = torch.log((1.0 - p_chunk).clamp_min(1e-6)).cumsum(dim=1)  # (B, M)
    diff = log_keep[:, :, None] - log_keep[:, None, :]  # (B, j, i)
    most = p_chunk.shape[1]
    causal = torch.ones(most, most, dtype=torch.bool, device=z.device).tril()
    weights = torch.exp(diff.masked_fill(~causal, float("-inf"))) * p_chunk[:, None, :]
    return torch.bmm(weights, z.float()).to(z.dtype)


def upsample(zbar, b, p):
    """Give each byte the smoothed vector of its chunk, scaled by a straight-through confidence.

    The confidence is p at a boundary and 1 - p elsewhere. Its forward value is
    1; its gradient reaches the router.
    """
    index = b.long().cumsum(dim=1) - 1
    up = torch.gather(zbar, 1, index[..., None].expand(-1, -1, zbar.shape[-1]))
    confidence = torch.where(b, p, 1.0 - p)
    ste = confidence + (1.0 - confidence).detach()
    return up * ste[..., None].to(up.dtype)


class QwenMain(nn.Module):
    """A Qwen3 decoder body (no embedding or head) for the chunk sequence."""

    def __init__(self, lm):
        super().__init__()
        self.model = lm.model
        self.model.embed_tokens = None

    def forward(self, x):
        return self.model(inputs_embeds=x, use_cache=False).last_hidden_state


class TitansMain(nn.Module):
    """A Looped Titans-MAG body (no embedding or head) for the chunk sequence."""

    def __init__(self, lm):
        super().__init__()
        del lm.lm_head, lm.embed_tokens
        self.model = lm

    def forward(self, x):
        return self.model(inputs_embeds=x, hidden_only=True)


class HNetForCausalLM(nn.Module):
    def __init__(self, config, main):
        """config is the byte-level Qwen3Config; main is a QwenMain or TitansMain of width config.main_hidden_size."""
        super().__init__()
        self.config = config
        d, wide = config.hidden_size, config.main_hidden_size
        self.target_chunk_size = config.target_chunk_size
        self.ratio_loss_weight = config.ratio_loss_weight
        self.embed_tokens = nn.Embedding(config.vocab_size, d)
        self.encoder = nn.ModuleList([SWABlock(config) for _ in range(config.encoder_layers)])
        self.encoder_norm = Qwen3RMSNorm(d, eps=config.rms_norm_eps)
        self.router_q = nn.Linear(d, d, bias=False)
        self.router_k = nn.Linear(d, d, bias=False)
        self.up_proj = nn.Linear(d, wide, bias=False) if wide != d else nn.Identity()
        self.down_proj = nn.Linear(wide, d, bias=False) if wide != d else nn.Identity()
        self.residual_proj = nn.Linear(d, d, bias=False)
        self.decoder = nn.ModuleList([SWABlock(config) for _ in range(config.decoder_layers)])
        self.norm = Qwen3RMSNorm(d, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.lm_head = nn.Linear(d, config.vocab_size, bias=False)
        self.apply(TitansMAGForCausalLM._init_weights)
        # The paper starts the router at identity, so p first measures how much
        # consecutive encoder states differ, and the residual at zero, so the
        # decoder must first learn to use the main network.
        with torch.no_grad():
            self.router_q.weight.copy_(torch.eye(d))
            self.router_k.weight.copy_(torch.eye(d))
            self.residual_proj.weight.zero_()
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.main = main
        self.last_stats = {}

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids, logits_to_keep=0, **kwargs):
        x = self.embed_tokens(input_ids)
        pos = torch.arange(input_ids.shape[1], device=input_ids.device)[None, :]
        cos, sin = self.rotary_emb(x, pos)
        for layer in self.encoder:
            x = layer(x, cos, sin)
        h = self.encoder_norm(x)

        p = boundary_probs(self.router_q(h), self.router_k(h))
        b = p >= 0.5
        chunks, positions, valid = downsample(h, b)
        z = self.down_proj(self.main(self.up_proj(chunks)))
        zbar = smooth(z, torch.gather(p, 1, positions))
        x = upsample(zbar, b, p) + self.residual_proj(h)

        for layer in self.decoder:
            x = layer(x, cos, sin)
        if logits_to_keep:
            x = x[:, -logits_to_keep:]
        logits = self.lm_head(self.norm(x))

        aux = ratio_loss(p, b, self.target_chunk_size)
        self.last_stats = {
            "bytes_per_chunk": b.numel() / b.sum().clamp_min(1),
            "ratio_loss": aux.detach(),
            "chunks_per_row": valid.sum(dim=1).float().mean(),
        }
        return SimpleNamespace(logits=logits, aux_loss=self.ratio_loss_weight * aux, boundaries=b)

    @torch.no_grad()
    def generate(self, input_ids, **kwargs):
        """Sample bytes. Each step runs the whole window again; there is no decode cache yet."""
        kwargs["use_cache"] = False
        return TitansMAGForCausalLM.generate(self, input_ids, **kwargs)
