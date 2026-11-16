"""Bounded H-Net training benchmark with controlled router-length transitions.

Run with ``python -m kestrel.benchmark --help``. Synthetic boundaries isolate
shape costs; this benchmark does not measure model quality or data loading.
"""

import argparse
from contextlib import nullcontext
from functools import partial
import json
import time

import torch

from .core import choose_device, make_model, make_optimizers, read_config, resolve_precision
from .hnet import ratio_loss
from .performance import configure_training
from .train import loss_on


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/hnet_titans.toml")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", choices=["off", "fixed"], default="fixed")
    parser.add_argument("--attention-backend", choices=["sdpa", "flash2"], default="sdpa")
    parser.add_argument("--smoothing", choices=["blocked", "dense"], default="blocked")
    parser.add_argument("--batched-muon", action="store_true")
    parser.add_argument("--chunk-counts", default="256,512,768,512,256")
    parser.add_argument("--steps-per-shape", type=int, default=3)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--max-seconds", type=float, default=600,
                        help="Stop between steps after this wall-clock budget; cannot interrupt a compiler call.")
    parser.add_argument("--trace", help="Optional Chrome profiler trace path; profiling adds overhead.")
    args = parser.parse_args()
    config = read_config(args.config)
    if config["model"].get("architecture") != "hnet":
        parser.error("Use an H-Net config")
    settings = config["train"]
    counts = [int(value) for value in args.chunk_counts.split(",")]
    length = settings["sequence_length"]
    if not counts or min(counts) < 1 or max(counts) > length:
        parser.error("Chunk counts must be between 1 and sequence_length")
    if args.steps_per_shape < 2 or args.max_seconds <= 0:
        parser.error("Use at least two steps per shape and a positive time budget")
    settings.update(compile=args.compile == "fixed", attention_backend=args.attention_backend,
                    hnet_smoothing=args.smoothing, batched_muon=args.batched_muon)
    if args.batch_size is not None:
        settings["batch_size"] = args.batch_size
    if args.accumulation is not None:
        settings["gradient_accumulation"] = args.accumulation
    if min(settings["batch_size"], settings["gradient_accumulation"]) < 1:
        parser.error("Batch size and accumulation must be positive")
    device = choose_device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(settings["seed"])
    precision = resolve_precision(settings.get("precision", "auto"), device)
    autocast = partial(torch.autocast, device_type=device.type, dtype=torch.bfloat16) if precision == "bf16" else nullcontext
    model = make_model(config["model"]).to(device).train()
    configure_training(model, settings)
    if settings.get("gradient_checkpointing", False):
        model.set_gradient_checkpointing(True)
    optimizers = make_optimizers(model, settings)
    if settings["compile"]:
        model.compile_submodules()
    encode = model._compiled_encode or model._encode_and_route
    boundary = None

    def controlled_encode(ids):
        h, p, _, cos, sin, _ = encode(ids)
        return h, p, boundary, cos, sin, ratio_loss(p, boundary, model.target_chunk_size)

    # The boundary override is outside the compiled encoder, just like real
    # routing decisions at the encoder/main interface. Never change its graph.
    model._compiled_encode = controlled_encode
    x = torch.randint(7, config["model"]["vocab_size"], (settings["batch_size"], length), device=device)
    y = torch.roll(x, -1, dims=1)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elif device.type == "mps":
            torch.mps.synchronize()
    print(json.dumps({"torch": torch.__version__, "device": str(device), "precision": precision,
                      "parameters": sum(p.numel() for p in model.parameters()), "options": vars(args),
                      "batch_size": settings["batch_size"], "accumulation": settings["gradient_accumulation"],
                      "note": "Synthetic boundaries; first visits include compilation. No evaluation or checkpoint I/O."}), flush=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = torch.profiler.profile(activities=activities, record_shapes=True) if args.trace else nullcontext()
    started = time.monotonic()
    records = []
    with profiler as prof:
        for visit, count in enumerate(counts):
            boundary = torch.zeros_like(x, dtype=torch.bool)
            boundary[:, torch.arange(count, device=device) * length // count] = True
            for repeat in range(args.steps_per_shape):
                if time.monotonic() - started >= args.max_seconds:
                    print(json.dumps({"stopped": "time budget", "completed_steps": len(records)}), flush=True)
                    break
                for optimizer in optimizers:
                    optimizer.zero_grad(set_to_none=True)
                sync()
                begin = time.monotonic()
                total = torch.zeros((), device=device)
                with torch.profiler.record_function("forward_backward"):
                    for _ in range(settings["gradient_accumulation"]):
                        loss = loss_on(model, x, y, autocast, masked=False)
                        total += loss.detach() / settings["gradient_accumulation"]
                        (loss / settings["gradient_accumulation"]).backward()
                sync()
                backward_done = time.monotonic()
                with torch.profiler.record_function("clip_and_optimizer"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip_norm"], error_if_nonfinite=True)
                    for optimizer in optimizers:
                        optimizer.step()
                sync()
                end = time.monotonic()
                elapsed = end - begin
                record = {"visit": visit, "chunks": count, "repeat": repeat,
                          "forward_backward_seconds": backward_done - begin,
                          "optimizer_seconds": end - backward_done, "step_seconds": elapsed,
                          "bytes_per_second": x.numel() * settings["gradient_accumulation"] / elapsed,
                          "train_loss": total.item()}
                if device.type == "cuda":
                    record["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                    torch.cuda.reset_peak_memory_stats(device)
                records.append(record)
                print(json.dumps(record), flush=True)
                if prof:
                    prof.step()
            else:
                continue
            break
    if args.trace:
        prof.export_chrome_trace(args.trace)
    print(json.dumps({"total_seconds": time.monotonic() - started, "completed_steps": len(records)}), flush=True)


if __name__ == "__main__":
    main()
