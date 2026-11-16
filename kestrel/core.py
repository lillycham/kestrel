"""Library-backed decoder, optimiser split and WSD schedule."""

import math
import tomllib
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from .memory import TitansMAGForCausalLM


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
        "memory_chunk_size", "memory_hidden_size", "attention_window",
        "memory_heads", "memory_forget_bias", "memory_min_forget", "norm_attn_branch",
        "num_loops", "prelude_layers", "coda_layers",
        "architecture", "encoder_layers", "decoder_layers", "target_chunk_size",
        "ratio_loss_weight", "main", "lr_multiplier",
    }
    if set(model) - allowed_model:
        raise ValueError(f"Unknown model options: {set(model) - allowed_model}")
    required_train = {
        "steps", "batch_size", "sequence_length", "gradient_accumulation",
        "warmup_steps", "decay_steps", "muon_lr", "adamw_lr", "weight_decay",
        "clip_norm", "eval_every", "eval_batches", "save_every", "seed",
    }
    optional_train = {"precision", "compile", "assistant_only", "keep_checkpoints"}
    if set(train) - (required_train | optional_train) or required_train - set(train):
        raise ValueError(f"Train options differ: {set(train) ^ required_train}")
    if "precision" in train and train["precision"] not in {"fp32", "bf16", "auto"}:
        raise ValueError("precision must be 'fp32', 'bf16' or 'auto'")
    for flag in ("compile", "assistant_only", "keep_checkpoints"):
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
    architecture = settings.pop("architecture", "decoder")
    if architecture == "hnet":
        return make_hnet(settings, special_ids)
    if architecture != "decoder":
        raise ValueError("architecture must be 'decoder' or 'hnet'")
    if (HNET_KEYS | {"lr_multiplier"}) & set(settings):
        raise ValueError(f"H-Net options need architecture = 'hnet': {(HNET_KEYS | {'lr_multiplier'}) & set(settings)}")
    theta = settings.pop("rope_theta", 10000.0)
    memory_chunk = settings.pop("memory_chunk_size", None)
    memory_hidden = settings.pop("memory_hidden_size", None)
    attention_window = settings.pop("attention_window", None)
    extra_titans = {
        key: settings.pop(key) for key in (
            "memory_heads", "memory_forget_bias", "memory_min_forget",
            "norm_attn_branch", "num_loops", "prelude_layers", "coda_layers",
        ) if key in settings
    }
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
    if memory_chunk is not None or memory_hidden is not None or attention_window is not None:
        if min(memory_chunk or 0, memory_hidden or 0, attention_window or 0) <= 0:
            raise ValueError("Titans-MAG requires positive memory_chunk_size, memory_hidden_size, and attention_window")
        mem_heads = extra_titans.get("memory_heads", 1)
        forget_bias = extra_titans.get("memory_forget_bias", -2.0)
        min_forget = extra_titans.get("memory_min_forget", 0.0)
        norm_attn = extra_titans.get("norm_attn_branch", False)
        num_loops = extra_titans.get("num_loops", 1)
        prelude_layers = extra_titans.get("prelude_layers", 0)
        coda_layers = extra_titans.get("coda_layers", 0)
        if not isinstance(mem_heads, int) or mem_heads <= 0 or width % mem_heads or memory_hidden % mem_heads:
            raise ValueError("memory_heads must be a positive integer dividing hidden_size and memory_hidden_size")
        if not math.isfinite(forget_bias) or not (0.0 <= min_forget < 1.0):
            raise ValueError("memory_forget_bias must be finite and 0 <= memory_min_forget < 1")
        if not isinstance(norm_attn, bool):
            raise ValueError("norm_attn_branch must be a boolean")
        if not isinstance(num_loops, int) or num_loops < 1 or not isinstance(prelude_layers, int) or prelude_layers < 0 or not isinstance(coda_layers, int) or coda_layers < 0:
            raise ValueError("num_loops >= 1 and prelude_layers/coda_layers >= 0 must be integers")
        config.memory_chunk_size = memory_chunk
        config.memory_hidden_size = memory_hidden
        config.attention_window = attention_window
        for key, value in extra_titans.items():
            setattr(config, key, value)
        return TitansMAGForCausalLM(config)
    if extra_titans:
        raise ValueError("Looped and memory options require memory_chunk_size, memory_hidden_size, and attention_window")
    config._attn_implementation = "sdpa"
    return Qwen3ForCausalLM(config)


HNET_KEYS = {"encoder_layers", "decoder_layers", "target_chunk_size", "ratio_loss_weight", "main"}
TITANS_KEYS = {
    "memory_chunk_size", "memory_hidden_size", "memory_heads", "memory_forget_bias",
    "memory_min_forget", "norm_attn_branch", "num_loops", "prelude_layers", "coda_layers",
}


def make_hnet(settings, special_ids=None):
    """Build a one-stage H-Net. [model] holds the byte-level shape; [model.main] the chunk-level network."""
    from .hnet import HNetForCausalLM, QwenMain, TitansMain

    if HNET_KEYS - set(settings):
        raise ValueError(f"H-Net needs options: {HNET_KEYS - set(settings)}")
    if TITANS_KEYS & set(settings) or "num_hidden_layers" in settings:
        raise ValueError("Put main-network layers and memory options under [model.main]")
    main = dict(settings.pop("main"))
    # Byte-stage LR relative to the main network (H-Net paper, Appendix C, one stage):
    # sqrt(bytes per chunk * main width / byte width). The paper calls it a heuristic.
    multiplier = settings.pop("lr_multiplier", None)
    network = main.pop("network", "transformer")
    hnet = {key: settings.pop(key) for key in sorted(HNET_KEYS - {"main"})}
    for key in ("encoder_layers", "decoder_layers"):
        if not isinstance(hnet[key], int) or hnet[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not hnet["target_chunk_size"] > 1 or not math.isfinite(hnet["target_chunk_size"]):
        raise ValueError("target_chunk_size must be finite and greater than 1")
    if not math.isfinite(hnet["ratio_loss_weight"]) or hnet["ratio_loss_weight"] < 0:
        raise ValueError("ratio_loss_weight must be finite and nonnegative")
    if network == "transformer" and TITANS_KEYS & set(main):
        raise ValueError("Titans options need main network = 'titans'")
    if network == "titans" and not {"memory_chunk_size", "memory_hidden_size", "attention_window"} <= set(main):
        raise ValueError("A Titans main network needs memory_chunk_size, memory_hidden_size and attention_window")
    if network not in {"transformer", "titans"}:
        raise ValueError("main network must be 'transformer' or 'titans'")
    if "vocab_size" in main or "max_position_embeddings" in main:
        raise ValueError("The main network takes vocab_size and max_position_embeddings from [model]")
    inherited = ("max_position_embeddings", "attention_bias", "attention_dropout", "rms_norm_eps", "rope_theta")
    main_settings = {key: settings[key] for key in inherited if key in settings}
    main_settings.update(vocab_size=1, tie_word_embeddings=True, **main)
    lm = make_model(main_settings)
    body = QwenMain(lm) if network == "transformer" else TitansMain(lm)

    theta = settings.pop("rope_theta", 10000.0)
    window = settings.pop("attention_window", None)
    if not isinstance(window, int) or window <= 0:
        raise ValueError("H-Net needs a positive attention_window for its byte layers")
    width, heads, kv_heads = settings["hidden_size"], settings["num_attention_heads"], settings["num_key_value_heads"]
    if min(width, heads, kv_heads) <= 0 or heads % kv_heads:
        raise ValueError("Attention heads must be positive and divisible by KV heads")
    if width != heads * settings["head_dim"] or settings["head_dim"] % 2:
        raise ValueError("Use an even head_dim with hidden_size = heads * head_dim")
    config = Qwen3Config(
        **settings, **(special_ids or {}),
        num_hidden_layers=hnet["encoder_layers"] + hnet["decoder_layers"],
        rope_parameters={"rope_type": "default", "rope_theta": theta},
        use_cache=False,
    )
    config.architecture = "hnet"
    config.attention_window = window
    config.main_network = network
    config.main_hidden_size = main["hidden_size"]
    if multiplier is None:
        multiplier = math.sqrt(hnet["target_chunk_size"] * main["hidden_size"] / settings["hidden_size"])
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("lr_multiplier must be finite and positive")
    config.lr_multiplier = float(multiplier)
    for key, value in hnet.items():
        setattr(config, key, value)
    return HNetForCausalLM(config, body)


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
    # An H-Net trains its byte stage (everything outside model.main) at a higher LR.
    multiplier = getattr(model.config, "lr_multiplier", 1.0) if hasattr(model, "main") else 1.0
    inner = {id(p) for p in model.main.parameters()} if multiplier != 1.0 else set()

    def groups(params, lr, **extra):
        outer = [p for p in params if multiplier != 1.0 and id(p) not in inner]
        rest = [p for p in params if multiplier == 1.0 or id(p) in inner]
        return [{"params": ps, "lr": lr * m, **extra} for ps, m in ((rest, 1.0), (outer, multiplier)) if ps]

    muon = torch.optim.Muon(
        groups(matrices, train["muon_lr"]), lr=train["muon_lr"], weight_decay=train["weight_decay"],
        adjust_lr_fn="original",
    )
    adam = torch.optim.AdamW(
        groups(decay, train["adamw_lr"], weight_decay=train["weight_decay"])
        + groups(no_decay, train["adamw_lr"], weight_decay=0.0),
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
