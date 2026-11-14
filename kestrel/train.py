"""Small trainer with optional BF16/compile, accumulation and resumable state."""

from contextlib import nullcontext
from functools import partial
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .core import choose_device, make_model, make_optimizers, read_config, resolve_precision, wsd
from .data import load_data, load_tokenizer


def mask_assistant_targets(stream, starts, tokens, assistant_id, end_id, non_assistant_ids):
    """Mask target tokens outside <|assistant|> ... <|end|> spans with -100."""
    y = tokens[:, 1:].clone()
    x = tokens[:, :-1]
    delimiters = non_assistant_ids | {assistant_id, end_id}
    for row_idx, start in enumerate(starts):
        in_assistant = False
        lookback_start = max(0, int(start) - 2048)
        if int(start) > lookback_start:
            prefix = np.asarray(stream[lookback_start:int(start)], dtype=np.int64)
            for tok in reversed(prefix):
                t = int(tok)
                if t == assistant_id:
                    in_assistant = True
                    break
                if t in non_assistant_ids or t == end_id:
                    break
        for col in range(y.shape[1]):
            prev_tok = int(x[row_idx, col])
            target_tok = int(y[row_idx, col])
            if prev_tok == assistant_id:
                in_assistant = True
            elif prev_tok in non_assistant_ids:
                in_assistant = False
            if not in_assistant:
                y[row_idx, col] = -100
            if target_tok == end_id:
                in_assistant = False
    return y


def batch(stream, settings, generator, device, assistant_tokens=None):
    length = settings["sequence_length"]
    starts = torch.randint(len(stream) - length, (settings["batch_size"],), generator=generator).numpy()
    offsets = starts[:, None] + np.arange(length + 1)
    tokens = torch.from_numpy(np.asarray(stream[offsets], dtype=np.int64))
    x = tokens[:, :-1].contiguous().to(device, non_blocking=True)
    if assistant_tokens is not None:
        assistant_id, end_id, non_assistant_ids = assistant_tokens
        y = mask_assistant_targets(stream, starts, tokens, assistant_id, end_id, non_assistant_ids)
    else:
        y = tokens[:, 1:].contiguous()
    return x, y.to(device, non_blocking=True)


def loss_on(model, x, y, autocast_ctx=nullcontext):
    # Shift explicitly so we supervise every position in each sampled block.
    with autocast_ctx():
        logits = model(input_ids=x, use_cache=False).logits
        if (y != -100).sum() == 0:
            return logits.sum() * 0.0
        return torch.nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten(), ignore_index=-100)


@torch.no_grad()
def evaluate(model, stream, settings, device, autocast_ctx=nullcontext, assistant_tokens=None):
    generator = torch.Generator().manual_seed(settings["seed"] + 1)
    model.eval()
    losses = [loss_on(model, *batch(stream, settings, generator, device, assistant_tokens), autocast_ctx).item()
              for _ in range(settings["eval_batches"])]
    model.train()
    return sum(losses) / len(losses)


def checkpoint(path, model, optimizers, config, manifest, step, generator, device):
    payload = {
        "model": model.state_dict(), "optimizers": [o.state_dict() for o in optimizers],
        "config": config, "manifest": manifest, "step": step,
        "batch_rng": generator.get_state(), "torch_rng": torch.get_rng_state(),
        "device_type": device.type,
    }
    if device.type == "cuda":
        payload["device_rng"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        payload["device_rng"] = torch.mps.get_rng_state()
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(config_path, data_path, output, requested_device="auto", resume=False, stop_after=None, init_from=None):
    if resume and init_from is not None:
        raise ValueError("Cannot combine resume and init_from")
    config = read_config(config_path)
    settings = config["train"]
    device = choose_device(requested_device)
    torch.manual_seed(settings["seed"])
    generator = torch.Generator().manual_seed(settings["seed"])
    manifest, (train_tokens, validation_tokens) = load_data(data_path, settings["sequence_length"])
    tokenizer = load_tokenizer(Path(data_path) / "tokenizer")
    if manifest["vocab_size"] > config["model"]["vocab_size"]:
        raise ValueError("Tokenizer vocabulary exceeds model vocabulary")
    assistant_tokens = None
    if settings.get("assistant_only", False):
        assistant_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
        end_id = tokenizer.convert_tokens_to_ids("<|end|>")
        non_assistant = {tokenizer.convert_tokens_to_ids(t) for t in ("<|user|>", "<|system|>", "<|eos|>")}
        assistant_tokens = (assistant_id, end_id, non_assistant)
    precision = resolve_precision(settings.get("precision", "fp32"), device)
    use_compile = settings.get("compile", False)
    autocast_ctx = partial(torch.autocast, device_type=device.type, dtype=torch.bfloat16) if precision == "bf16" else nullcontext
    raw_model = make_model(config["model"], {
        "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }).to(device)
    optimizers = make_optimizers(raw_model, settings)
    output = Path(output)
    start = 0
    if resume:
        # Only tensor/primitive payloads are accepted; never load arbitrary pickle objects.
        saved = torch.load(output / "latest.pt", map_location="cpu", weights_only=True)
        if saved["config"] != config or saved["manifest"] != manifest:
            raise ValueError("Resume requires identical config and data")
        if saved["device_type"] != device.type:
            raise ValueError("Resume requires the original device type")
        raw_model.load_state_dict(saved["model"])
        for optimizer, state in zip(optimizers, saved["optimizers"], strict=True):
            optimizer.load_state_dict(state)
        generator.set_state(saved["batch_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["device_rng"])
        elif device.type == "mps":
            torch.mps.set_rng_state(saved["device_rng"])
        start = saved["step"]
    else:
        if init_from is not None:
            saved = torch.load(init_from, map_location="cpu", weights_only=True)
            if saved["config"]["model"] != config["model"]:
                raise ValueError("init_from requires identical model config")
            raw_model.load_state_dict(saved["model"])
        output.mkdir(parents=True, exist_ok=False)
        tokenizer.save_pretrained(output / "tokenizer")
        raw_model.config.save_pretrained(output)
        experiment_meta = {
            "config": config, "manifest": manifest,
            "torch": torch.__version__, "device": str(device),
            "precision": precision, "compile": use_compile,
        }
        if init_from is not None:
            experiment_meta["init_from"] = str(init_from)
        (output / "experiment.json").write_text(json.dumps(experiment_meta, indent=2) + "\n")
    end = min(settings["steps"], stop_after if stop_after is not None else settings["steps"])
    if end <= start:
        raise ValueError("No steps remain before stop-after/total steps")
    # Compile after loading/init so checkpoints always store clean unprefixed keys.
    model = torch.compile(raw_model) if use_compile else raw_model
    print(json.dumps({"device": str(device), "precision": precision, "compile": use_compile,
                      "parameters": sum(p.numel() for p in raw_model.parameters()),
                      "start_step": start, "end_step": end}), flush=True)
    model.train()
    started = time.monotonic()
    for step in range(start, end):
        scale = wsd(step, settings["steps"], settings["warmup_steps"], settings["decay_steps"])
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * scale
        total_loss = 0.0
        for _ in range(settings["gradient_accumulation"]):
            loss = loss_on(model, *batch(train_tokens, settings, generator, device, assistant_tokens), autocast_ctx)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at step {step + 1}")
            total_loss += loss.item() / settings["gradient_accumulation"]
            (loss / settings["gradient_accumulation"]).backward()
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), settings["clip_norm"], error_if_nonfinite=True)
        for optimizer in optimizers:
            optimizer.step()
        done = step + 1
        record = {"step": done, "train_loss": total_loss, "lr_scale": scale}
        if done % settings["eval_every"] == 0 or done == end:
            record["validation_loss"] = evaluate(model, validation_tokens, settings, device, autocast_ctx, assistant_tokens)
        record["elapsed_seconds"] = time.monotonic() - started
        with (output / "metrics.jsonl").open("a") as log:
            log.write(json.dumps(record) + "\n")
        if "validation_loss" in record or done == start + 1:
            print(json.dumps(record), flush=True)
        if done % settings["save_every"] == 0 or done == end:
            checkpoint(output / "latest.pt", raw_model, optimizers, config, manifest, done, generator, device)
    return raw_model


def _load_for_inference(run, requested_device, seed):
    device = choose_device(requested_device)
    torch.manual_seed(seed)
    run = Path(run)
    saved = torch.load(run / "latest.pt", map_location="cpu", weights_only=True)
    tokenizer = load_tokenizer(run / "tokenizer")
    model = make_model(saved["config"]["model"], {
        "bos_token_id": tokenizer.bos_token_id, "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }).to(device).eval()
    model.load_state_dict(saved["model"])
    # Unused model output rows must never be sampled.
    unused = [[index] for index in range(len(tokenizer), model.config.vocab_size)]
    return model, tokenizer, device, unused or None


def _sampling_options(max_new_tokens, temperature, top_k, top_p):
    if max_new_tokens < 1 or temperature < 0 or top_k < 0 or not (0 < top_p <= 1.0):
        raise ValueError("Use positive max_new_tokens, nonnegative temperature/top_k, and 0 < top_p <= 1")
    options = {"do_sample": temperature > 0}
    if temperature > 0:
        options.update(temperature=temperature, top_k=top_k, top_p=top_p)
    return options


@torch.no_grad()
def sample(run, prompt, requested_device="cpu", max_new_tokens=64, temperature=0.8, seed=42, top_k=0, top_p=1.0, raw=False):
    options = _sampling_options(max_new_tokens, temperature, top_k, top_p)
    model, tokenizer, device, bad_words_ids = _load_for_inference(run, requested_device, seed)
    text = prompt if raw else tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)
    if inputs.input_ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("Prompt plus generation exceeds configured context")
    eos_ids = tokenizer.eos_token_id if raw else [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end|>")]
    result = model.generate(
        **inputs, max_new_tokens=max_new_tokens, use_cache=True,
        eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id, bad_words_ids=bad_words_ids, **options,
    )
    tokens = result[0] if raw else result[0, inputs.input_ids.shape[1]:]
    print(tokenizer.decode(tokens, skip_special_tokens=True))


@torch.no_grad()
def chat(run, requested_device="auto", max_new_tokens=256, temperature=0.7, seed=42, top_k=50, top_p=0.9, system=None):
    """Interactive multi-turn chat loop with automatic history trimming."""
    options = _sampling_options(max_new_tokens, temperature, top_k, top_p)
    model, tokenizer, device, bad_words_ids = _load_for_inference(run, requested_device, seed)
    eos_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end|>")]
    base_history = [{"role": "system", "content": system}] if system else []
    messages = list(base_history)
    max_ctx = model.config.max_position_embeddings
    print("Commands: /reset to clear history, /quit or Ctrl-D to exit.")
    while True:
        try:
            prompt = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not prompt:
            continue
        if prompt in {"/quit", "/exit"}:
            break
        if prompt == "/reset":
            messages = list(base_history)
            print("(history cleared)")
            continue
        messages.append({"role": "user", "content": prompt})
        min_head = 1 if system else 0
        while True:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(text, return_tensors="pt").to(device)
            available = max_ctx - inputs.input_ids.shape[1]
            if available >= min(max_new_tokens, 32) or len(messages) <= min_head + 1:
                break
            # Drop the oldest user+assistant pair while preserving any system message.
            del messages[min_head:min_head + 2]
        if available < 1:
            messages.pop()
            print("(prompt exceeds model context; try a shorter message or /reset)")
            continue
        result = model.generate(
            **inputs, max_new_tokens=min(max_new_tokens, available), use_cache=True,
            eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id, bad_words_ids=bad_words_ids, **options,
        )
        reply = tokenizer.decode(result[0, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"kestrel> {reply}\n")
        messages.append({"role": "assistant", "content": reply})
