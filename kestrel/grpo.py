"""GRPO on step-by-step arithmetic, with a verifiable 0/1 reward.

Each step samples a group of completions per problem, scores them, and takes one
policy-gradient step. The advantage is reward minus the group mean, without
division by the group's standard deviation (as in Dr. GRPO). Groups where all
rewards are equal give zero advantage and do not enter the update. Because each
batch of rollouts gets exactly one update, the policy is on-policy and the PPO
ratio is 1, so there is no clipping. There is no KL term: watch the eval of the
buckets that the model already solves instead.
"""

from collections import defaultdict
import json
import os
import random
import time
from pathlib import Path

import torch

from .arith import BUCKETS, draw_new, is_correct, prompt_text, seen_problems
from .train import _load_for_inference

# Sharpening mix: mostly the buckets with many mixed groups, plus some solved
# buckets so that a regression shows up in the training reward too.
DEFAULT_WEIGHTS = {
    "sub_noborrow": 3.0, "sub_borrow": 3.0, "mul": 2.0, "add_2d_carry": 1.0, "muladd": 1.0,
    "add_2d_nocarry": 0.5, "add_3d_nocarry": 0.5, "add_3d_carry": 0.5, "addmul": 0.5,
}


def completion_mask(completions, eos_id):
    """1 for each new token up to and including the first EOS, 0 after it."""
    is_eos = (completions == eos_id).long()
    return (is_eos.cumsum(dim=1) - is_eos) == 0


@torch.no_grad()
def rollout(model, tokenizer, problems, group, temperature, max_new_tokens, max_rows, bad_words_ids=None):
    """Sample `group` completions per problem.

    Prompts with the same token length share one generate batch, so all rows of
    a batch stay at the same position. Returns one dict per problem, in order.
    """
    device = next(model.parameters()).device
    encoded = [tokenizer(prompt_text(p, tokenizer.eos_token)).input_ids for p in problems]
    by_length = defaultdict(list)
    for index, ids in enumerate(encoded):
        by_length[len(ids)].append(index)
    per_batch = max(1, max_rows // group)
    results = [None] * len(problems)
    model.eval()
    for indices in by_length.values():
        for start in range(0, len(indices), per_batch):
            chunk = indices[start:start + per_batch]
            prompts = torch.tensor([encoded[i] for i in chunk], device=device).repeat_interleave(group, dim=0)
            tokens, logprobs = model.generate(
                prompts, max_new_tokens=max_new_tokens, do_sample=temperature > 0, temperature=max(temperature, 1e-6),
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                bad_words_ids=bad_words_ids, return_logprobs=True,
            )
            completions = tokens[:, prompts.shape[1]:]
            mask = completion_mask(completions, tokenizer.eos_token_id)
            for row, index in enumerate(chunk):
                rows = slice(row * group, (row + 1) * group)
                texts = [tokenizer.decode(c[m], skip_special_tokens=True) for c, m in zip(completions[rows], mask[rows])]
                results[index] = {
                    "problem": problems[index],
                    "prompt": torch.tensor(encoded[index]),
                    "completions": completions[rows].cpu(),
                    "logprobs": logprobs[rows].cpu(),
                    "mask": mask[rows].cpu(),
                    "texts": texts,
                    "rewards": torch.tensor([float(is_correct(t, problems[index])) for t in texts]),
                }
    return results


def group_advantages(rewards):
    return rewards - rewards.mean()


def policy_loss(model, groups, pad_id, normaliser, micro_rows):
    """Backpropagate the GRPO loss for the groups with a nonzero advantage. Returns stats."""
    device = next(model.parameters()).device
    rows = []
    for g in groups:
        advantage = group_advantages(g["rewards"])
        if advantage.abs().sum() == 0:
            continue
        for r in range(len(advantage)):
            rows.append((g["prompt"], g["completions"][r], g["mask"][r], g["logprobs"][r], advantage[r]))
    if not rows:
        return {"loss": 0.0, "rows": 0, "logprob_gap": 0.0}
    model.train()
    total_loss, gap_sum, gap_count = 0.0, 0.0, 0
    for start in range(0, len(rows), micro_rows):
        part = rows[start:start + micro_rows]
        length = max(len(p) + len(c) for p, c, *_ in part)
        ids = torch.full((len(part), length), pad_id, dtype=torch.long)
        target_mask = torch.zeros(len(part), length - 1, dtype=torch.bool)
        old = torch.zeros(len(part), length - 1)
        adv = torch.zeros(len(part))
        for i, (prompt, completion, mask, logprobs, a) in enumerate(part):
            n = len(prompt)
            ids[i, :n] = prompt
            ids[i, n:n + len(completion)] = completion
            # Position t of the logits predicts token t + 1.
            target_mask[i, n - 1:n - 1 + len(completion)] = mask
            old[i, n - 1:n - 1 + len(completion)] = logprobs
            adv[i] = a
        # Only positions from the shortest prompt on need logits. The full-vocabulary
        # logits are the largest tensor here, so skip the prompt positions.
        first = min(len(p) for p, *_ in part) - 1
        ids, adv = ids.to(device), adv.to(device)
        target_mask, old = target_mask[:, first:].to(device), old[:, first:].to(device)
        logits = model(ids, logits_to_keep=length - first).logits[:, :-1]
        targets = ids[:, first + 1:]
        # Fused cross-entropy keeps one vocabulary-sized tensor, not log_softmax plus gather.
        logp = -torch.nn.functional.cross_entropy(
            logits.float().flatten(0, 1), targets.flatten(), reduction="none",
        ).view_as(targets)
        # Dr. GRPO: a constant normaliser avoids a bias towards short or long answers.
        loss = -(adv[:, None] * logp * target_mask).sum() / normaliser
        loss.backward()
        total_loss += loss.item()
        gap_sum += ((logp.detach() - old).abs() * target_mask).sum().item()
        gap_count += int(target_mask.sum())
    return {"loss": total_loss, "rows": len(rows), "logprob_gap": gap_sum / max(gap_count, 1)}


def evaluate(model, tokenizer, eval_set, group, temperature, max_new_tokens, max_rows, bad_words_ids):
    """Pass rate per bucket on a fixed problem set."""
    results = rollout(model, tokenizer, [p for _, p in eval_set], group, temperature, max_new_tokens, max_rows, bad_words_ids)
    hits = defaultdict(list)
    for (name, _), r in zip(eval_set, results):
        hits[name].append(r["rewards"].mean().item())
    report = {name: sum(v) / len(v) for name, v in hits.items()}
    report["mean"] = sum(report.values()) / len(report)
    return report


def make_eval_set(per_bucket, seen, seed):
    rng = random.Random(seed + 10_000)
    eval_set, chosen = [], set(seen)
    for name in BUCKETS:
        for _ in range(per_bucket):
            problem = draw_new(name, rng, chosen)
            chosen.add(problem)
            eval_set.append((name, problem))
    return eval_set


def _save(path, model, optimizer, config, settings, step, rng):
    payload = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": config,
        "grpo": settings, "step": step, "torch_rng": torch.get_rng_state(), "python_rng": rng.getstate(),
    }
    device = next(model.parameters()).device
    if device.type == "cuda":
        payload["device_rng"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps":
        payload["device_rng"] = torch.mps.get_rng_state()
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def grpo(init_from, output, settings, requested_device="auto", resume=False):
    """Run GRPO from the model in run directory `init_from` and write a run directory to `output`."""
    output = Path(output)
    source = Path(output if resume else init_from)
    model, tokenizer, device, bad_words_ids = _load_for_inference(source, requested_device, settings["seed"])
    saved = torch.load(source / "latest.pt", map_location="cpu", weights_only=True)
    config = saved["config"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["lr"], betas=(0.9, 0.99), weight_decay=0.0)
    rng = random.Random(settings["seed"])
    start = 0
    if resume:
        # Only the step count may change, so that a resume can extend a run.
        if {**saved.get("grpo", {}), "steps": None} != {**settings, "steps": None}:
            raise ValueError("Resume requires identical GRPO settings, except steps")
        if saved["step"] >= settings["steps"]:
            raise ValueError("No steps remain; raise --steps to extend the run")
        optimizer.load_state_dict(saved["optimizer"])
        # Drop metrics from steps after the checkpoint; those steps run again.
        metrics = output / "metrics.jsonl"
        kept = [line for line in metrics.read_text().splitlines() if json.loads(line)["step"] <= saved["step"]]
        metrics.write_text("".join(line + "\n" for line in kept))
        torch.set_rng_state(saved["torch_rng"])
        rng.setstate(_as_tuple(saved["python_rng"]))
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["device_rng"])
        elif device.type == "mps":
            torch.mps.set_rng_state(saved["device_rng"])
        start = saved["step"]
    else:
        output.mkdir(parents=True, exist_ok=False)
        tokenizer.save_pretrained(output / "tokenizer")
        model.config.save_pretrained(output)
        (output / "experiment.json").write_text(json.dumps({
            "grpo": settings, "init_from": str(init_from), "config": config,
            "torch": torch.__version__, "device": str(device),
        }, indent=2) + "\n")

    seen = seen_problems(settings.get("seen_from"))
    eval_set = make_eval_set(settings["eval_problems"], seen, settings["seed"])
    exclude = {p for _, p in eval_set}
    names = list(settings["weights"])
    weights = [settings["weights"][n] for n in names]
    normaliser = settings["prompts"] * settings["group"] * settings["max_new_tokens"]
    eval_args = (settings["max_new_tokens"], settings["max_rows"], bad_words_ids)

    def log(record):
        with (output / "metrics.jsonl").open("a") as sink:
            sink.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)

    if start == 0:
        log({"step": 0, "eval_greedy": evaluate(model, tokenizer, eval_set, 1, 0.0, *eval_args),
             "eval_sampled": evaluate(model, tokenizer, eval_set, settings["eval_group"], settings["temperature"], *eval_args)})
    started = time.monotonic()
    for step in range(start, settings["steps"]):
        scale = min(1.0, (step + 1) / settings["warmup"]) if settings["warmup"] else 1.0
        for g in optimizer.param_groups:
            g["lr"] = settings["lr"] * scale
        problems = [draw_new(name, rng, exclude) for name in rng.choices(names, weights, k=settings["prompts"])]
        t0 = time.monotonic()
        groups = rollout(model, tokenizer, problems, settings["group"], settings["temperature"],
                         settings["max_new_tokens"], settings["max_rows"], bad_words_ids)
        t1 = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        stats = policy_loss(model, groups, tokenizer.pad_token_id, normaliser, settings["micro_rows"])
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["clip"]).item() if stats["rows"] else 0.0
        if stats["rows"]:
            optimizer.step()
        done = step + 1
        rewards = torch.stack([g["rewards"] for g in groups])
        record = {
            "step": done, "reward": rewards.mean().item(),
            "mixed": ((rewards.sum(1) > 0) & (rewards.sum(1) < settings["group"])).float().mean().item(),
            "length": sum(g["mask"].sum().item() for g in groups) / rewards.numel(),
            **stats, "grad_norm": grad_norm, "lr": settings["lr"] * scale,
            "rollout_seconds": t1 - t0, "update_seconds": time.monotonic() - t1,
            "elapsed_seconds": time.monotonic() - started,
        }
        if done % settings["eval_every"] == 0 or done == settings["steps"]:
            record["eval_greedy"] = evaluate(model, tokenizer, eval_set, 1, 0.0, *eval_args)
            record["eval_sampled"] = evaluate(model, tokenizer, eval_set, settings["eval_group"], settings["temperature"], *eval_args)
        log(record)
        if done % settings["save_every"] == 0 or done == settings["steps"]:
            _save(output / "latest.pt", model, optimizer, config, settings, done, rng)
    return model


def _as_tuple(value):
    """random.setstate needs tuples; a checkpoint may give lists back."""
    return tuple(_as_tuple(v) for v in value) if isinstance(value, (list, tuple)) else value
