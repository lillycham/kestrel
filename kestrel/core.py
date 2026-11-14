"""Library-backed decoder, optimiser split and WSD schedule."""

import math
import tomllib
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM


def read_config(path):
    with Path(path).open("rb") as source:
        config = tomllib.load(source)
    if set(config) != {"model", "train"}:
        raise ValueError("Config must contain exactly [model] and [train]")
    model, train = config["model"], config["train"]
    allowed_model = {
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "max_position_embeddings", "tie_word_embeddings", "attention_bias",
        "attention_dropout", "rms_norm_eps", "rope_theta",
    }
    if set(model) - allowed_model:
        raise ValueError(f"Unknown model options: {set(model) - allowed_model}")
    required_train = {
        "steps", "batch_size", "sequence_length", "gradient_accumulation",
        "warmup_steps", "decay_steps", "muon_lr", "adamw_lr", "weight_decay",
        "clip_norm", "eval_every", "eval_batches", "save_every", "seed",
    }
    optional_train = {"precision", "compile", "assistant_only"}
    if set(train) - (required_train | optional_train) or required_train - set(train):
        raise ValueError(f"Train options differ: {set(train) ^ required_train}")
    if "precision" in train and train["precision"] not in {"fp32", "bf16", "auto"}:
        raise ValueError("precision must be 'fp32', 'bf16' or 'auto'")
    for flag in ("compile", "assistant_only"):
        if flag in train and not isinstance(train[flag], bool):
            raise ValueError(f"{flag} must be a boolean")
    for name in ("steps", "batch_size", "sequence_length", "gradient_accumulation",
                 "eval_every", "eval_batches", "save_every"):
        if not isinstance(train[name], int) or train[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not 0 <= train["warmup_steps"] < train["steps"]:
        raise ValueError("warmup_steps must be in [0, steps)")
    if not 1 <= train["decay_steps"] <= train["steps"] - train["warmup_steps"]:
        raise ValueError("decay_steps must fit after warmup")
    if train["sequence_length"] > model["max_position_embeddings"]:
        raise ValueError("sequence_length exceeds model context")
    for name in ("muon_lr", "adamw_lr", "clip_norm"):
        if not math.isfinite(train[name]) or train[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(train["weight_decay"]) or train["weight_decay"] < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    return config


def make_model(settings, special_ids=None):
    settings = dict(settings)
    theta = settings.pop("rope_theta", 10000.0)
    width = settings["hidden_size"]
    heads = settings["num_attention_heads"]
    kv_heads = settings["num_key_value_heads"]
    if min(width, heads, kv_heads) <= 0 or heads % kv_heads:
        raise ValueError("Attention heads must be positive and divisible by KV heads")
    if width != heads * settings["head_dim"] or settings["head_dim"] % 2:
        raise ValueError("Use an even head_dim with hidden_size = heads * head_dim")
    config = Qwen3Config(
        **settings,
        **(special_ids or {}),
        rope_parameters={"rope_type": "default", "rope_theta": theta},
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config)


def make_optimizers(model, train):
    # Tied input/output weights occur once in model.parameters(). Embeddings
    # stay with AdamW even though they are matrices.
    embedding = model.get_input_embeddings().weight
    head = model.get_output_embeddings().weight
    matrices, decay, no_decay = [], [], []
    for p in model.parameters():
        if p is embedding or p is head:
            decay.append(p)
        elif p.ndim == 2:
            matrices.append(p)
        else:
            no_decay.append(p)
    muon = torch.optim.Muon(
        matrices, lr=train["muon_lr"], weight_decay=train["weight_decay"],
        adjust_lr_fn="original",
    )
    adam = torch.optim.AdamW(
        [{"params": decay, "weight_decay": train["weight_decay"]},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=train["adamw_lr"], betas=(0.9, 0.95),
    )
    for optimizer in (muon, adam):
        for group in optimizer.param_groups:
            group["base_lr"] = group["lr"]
    return [muon, adam]


def wsd(step, steps, warmup, decay):
    """Zero-based update index. Final update has a small nonzero LR.

    The schedule reaches zero immediately after the final update. This avoids
    spending the last full forward/backward pass on an update of size zero.
    """
    if step < warmup:
        return (step + 1) / warmup
    return min(1.0, max(0.0, (steps - step) / decay))


def choose_device(requested):
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable in this process")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this process")
    return device


def resolve_precision(requested, device):
    if requested == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return "bf16"
        if device.type == "mps":
            return "bf16"
        return "fp32"
    return requested
