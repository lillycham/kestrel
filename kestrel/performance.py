"""Optional training kernels; checkpoint parameter and optimizer layouts stay unchanged."""

from collections import defaultdict
import math

import torch


def batched_orthogonalize(grad, coefficients, steps, eps):
    """Muon Newton–Schulz with independent normalization of each matrix."""
    x = grad.bfloat16()
    transpose = x.shape[-2] > x.shape[-1]
    if transpose:
        x = x.transpose(-1, -2)
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    a, b, c = coefficients
    for _ in range(steps):
        gram = x @ x.transpose(-1, -2)
        polynomial = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, polynomial, x, beta=a)
    return x.transpose(-1, -2) if transpose else x


class BatchedMuon(torch.optim.Muon):
    """Opt-in Muon batching, with the standard Muon state_dict format.

    Groups stay separate to preserve stage learning rates. Batches are limited
    to 16 matrices to bound temporary storage. No optimizer-wide compilation:
    the Python learning-rate schedule must not cause a new graph each step.
    """

    def __init__(self, *args, compile_kernel=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.orthogonalize = torch.compile(batched_orthogonalize) if compile_kernel else batched_orthogonalize

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            buckets = defaultdict(list)
            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse or p.is_complex():
                    raise RuntimeError("BatchedMuon needs dense, real gradients")
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buckets[(p.shape, p.dtype, p.device)].append(p)
            for params in buckets.values():
                for start in range(0, len(params), 16):
                    subset = params[start:start + 16]
                    grads = torch.stack([p.grad for p in subset])
                    buffers = torch.stack([self.state[p]["momentum_buffer"] for p in subset])
                    buffers.lerp_(grads, 1 - group["momentum"])
                    updates = grads.lerp(buffers, group["momentum"]) if group["nesterov"] else buffers
                    updates = self.orthogonalize(updates, group["ns_coefficients"], group["ns_steps"], group["eps"])
                    rows, cols = subset[0].shape
                    adjustment = group["adjust_lr_fn"]
                    if adjustment in (None, "original"):
                        scale = math.sqrt(max(1, rows / cols))
                    elif adjustment == "match_rms_adamw":
                        scale = 0.2 * math.sqrt(max(rows, cols))
                    else:
                        raise ValueError(f"Unknown Muon adjustment: {adjustment}")
                    for p, buffer, update in zip(subset, buffers, updates, strict=True):
                        self.state[p]["momentum_buffer"].copy_(buffer)
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                        p.add_(update, alpha=-group["lr"] * scale)
        return loss


def configure_training(model, settings):
    """Set runtime-only options before compilation; no parameter keys change."""
    from .memory import SlidingWindowAttention

    backend = settings.get("attention_backend", "sdpa")
    layers = [m for m in model.modules() if isinstance(m, SlidingWindowAttention)]
    if backend == "flash2":
        if next(model.parameters()).device.type != "cuda":
            raise ValueError("flash2 attention requires a CUDA device")
        try:
            from flash_attn import flash_attn_func
        except ImportError as exc:
            raise RuntimeError("attention_backend='flash2' requires the optional flash-attn package") from exc
        if not layers:
            raise ValueError("attention_backend='flash2' needs H-Net or Titans local attention")
        for layer in layers:
            layer.flash_attention = flash_attn_func
    else:
        for layer in layers:
            layer.flash_attention = None
    if hasattr(model, "smoothing"):
        model.smoothing = settings.get("hnet_smoothing", "blocked")
