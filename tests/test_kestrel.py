import json
from pathlib import Path

import pytest
import torch

from kestrel.core import make_model, make_optimizers, read_config, resolve_precision, wsd
from kestrel.data import load_tokenizer, prepare, rows, train_tokenizer
from kestrel.memory import memory_update
from kestrel.train import chat, mask_assistant_targets, sample, train


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def small_cpu_threads():
    torch.set_num_threads(1)
    yield


def test_causality_and_architecture():
    model = make_model(read_config(ROOT / "configs/smoke.toml")["model"]).eval()
    assert model.lm_head.weight is model.model.embed_tokens.weight
    assert not any(name.endswith("bias") for name, _ in model.named_parameters())
    assert model.model.layers[0].self_attn.q_norm.weight.numel() == 8
    x = torch.tensor([[10, 11, 12, 13]])
    y = torch.tensor([[10, 11, 20, 21]])
    with torch.no_grad():
        torch.testing.assert_close(model(x).logits[:, :2], model(y).logits[:, :2])


def test_optimizer_partition():
    config = read_config(ROOT / "configs/smoke.toml")
    model = make_model(config["model"])
    muon, adam = make_optimizers(model, config["train"])
    ids = [id(p) for optimizer in (muon, adam) for group in optimizer.param_groups for p in group["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    assert id(model.lm_head.weight) in {id(p) for g in adam.param_groups for p in g["params"]}
    assert all(p.ndim == 2 for g in muon.param_groups for p in g["params"])


def test_wsd_boundaries():
    assert [wsd(i, 10, 2, 3) for i in (0, 1, 2, 6, 7, 8, 9, 10)] == pytest.approx(
        [0.5, 1, 1, 1, 1, 2 / 3, 1 / 3, 0]
    )
    assert wsd(0, 4, 0, 2) == 1


def test_bad_shape():
    settings = read_config(ROOT / "configs/smoke.toml")["model"]
    with pytest.raises(ValueError):
        make_model({**settings, "num_key_value_heads": 3})


def test_reserved_content_rejected(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text(json.dumps({"text": "hello <|assistant|>"}))
    with pytest.raises(ValueError, match="reserved"):
        list(rows(source))


def test_end_to_end_and_exact_cpu_resume(tmp_path, capsys):
    training = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    training.write_text("\n".join(json.dumps({"messages": [
        {"role": "user", "content": f"hello {i}"},
        {"role": "assistant", "content": "Hello! What shall we talk about?"},
    ]}) for i in range(12)))
    validation.write_text("\n".join(json.dumps({"messages": [
        {"role": "user", "content": f"Good morning {i}"},
        {"role": "assistant", "content": "Good morning! How are you?"},
    ]}) for i in range(4)))
    tokenizer_path = tmp_path / "tokenizer"
    train_tokenizer(training, tokenizer_path, 512)
    tokenizer = load_tokenizer(tokenizer_path)
    text = "An unseen owl: 🦉 café 日本語"
    assert tokenizer.decode(tokenizer.encode(text)) == text
    chat = tokenizer.apply_chat_template([
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
    ], tokenize=False)
    assert chat == "<|user|>\nhi<|end|>\n<|assistant|>\nhello<|end|>\n<|eos|>"
    data_path = tmp_path / "encoded"
    prepare(training, validation, tokenizer_path, data_path)
    with pytest.raises(ValueError, match="separate"):
        prepare(training, training, tokenizer_path, tmp_path / "bad-data")
    config = ROOT / "configs/smoke.toml"
    full = train(config, data_path, tmp_path / "full", "cpu")
    train(config, data_path, tmp_path / "resumed", "cpu", stop_after=2)
    resumed = train(config, data_path, tmp_path / "resumed", "cpu", resume=True)
    for name, weight in full.state_dict().items():
        torch.testing.assert_close(weight, resumed.state_dict()[name], rtol=0, atol=0)
    metrics = [json.loads(line) for line in (tmp_path / "full/metrics.jsonl").read_text().splitlines()]
    assert len(metrics) == 4
    assert metrics[-1]["validation_loss"] > 0
    sample(tmp_path / "full", "hi", max_new_tokens=2, temperature=0)
    assert capsys.readouterr().out
    sample(tmp_path / "full", "hello", max_new_tokens=2, temperature=0.8, top_k=5, top_p=0.9, repetition_penalty=1.2, raw=True)
    assert capsys.readouterr().out.startswith("hello")
    with pytest.raises(ValueError):
        sample(tmp_path / "full", "hello", top_k=-1)
    with pytest.raises(ValueError):
        sample(tmp_path / "full", "hello", top_p=1.5)
    with pytest.raises(ValueError):
        sample(tmp_path / "full", "hello", repetition_penalty=0)


def test_chat_loop(tmp_path, monkeypatch, capsys):
    training = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    training.write_text("\n".join(json.dumps({"messages": [
        {"role": "user", "content": f"hello {i}"},
        {"role": "assistant", "content": f"hi {i}"},
    ]}) for i in range(12)))
    validation.write_text("\n".join(json.dumps({"messages": [
        {"role": "user", "content": f"val {i}"},
        {"role": "assistant", "content": f"reply {i}"},
    ]}) for i in range(4)))
    tokenizer_path = tmp_path / "tokenizer"
    train_tokenizer(training, tokenizer_path, 512)
    data_path = tmp_path / "encoded"
    prepare(training, validation, tokenizer_path, data_path)
    train(ROOT / "configs/smoke.toml", data_path, tmp_path / "run", "cpu", stop_after=2)

    inputs = iter(["hello", "/reset", "bye", "/quit"])
    monkeypatch.setattr("builtins.input", lambda _: next(inputs))
    chat(tmp_path / "run", requested_device="cpu", max_new_tokens=3, temperature=0)
    out = capsys.readouterr().out
    assert out.count("kestrel>") == 2
    assert "(history cleared)" in out


def test_titans_mag_causality_and_train(tmp_path):
    cfg = read_config(ROOT / "configs/titans_smoke.toml")
    model = make_model(cfg["model"]).eval()
    assert model.lm_head.weight is model.embed_tokens.weight
    muon, adam = make_optimizers(model, cfg["train"])
    ids = [id(p) for o in (muon, adam) for g in o.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))

    # Token-level causality across multiple chunks (chunk_size=4, seq=10).
    x = torch.randint(10, 100, (1, 10))
    y = x.clone()
    y[:, 5:] = torch.randint(100, 200, (1, 5))
    with torch.no_grad():
        torch.testing.assert_close(model(x).logits[:, :5], model(y).logits[:, :5])

    training = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    training.write_text("\n".join(json.dumps({"text": f"long document {i} " * 8}) for i in range(12)))
    validation.write_text("\n".join(json.dumps({"text": f"val document {i} " * 8}) for i in range(4)))
    tokenizer_path = tmp_path / "tokenizer"
    train_tokenizer(training, tokenizer_path, 512)
    data_path = tmp_path / "encoded"
    prepare(training, validation, tokenizer_path, data_path)
    train(ROOT / "configs/titans_smoke.toml", data_path, tmp_path / "titans", "cpu")
    sample(tmp_path / "titans", "long", max_new_tokens=4, temperature=0, raw=True)
    records = [json.loads(line) for line in (tmp_path / "titans/metrics.jsonl").read_text().splitlines()]
    evaluated = [r for r in records if "validation_loss" in r]
    assert evaluated and all("memory/dead_calls" in r and "memory/share/core0" in r for r in evaluated)


def test_titans_memory_health_finds_wiped_memory():
    cfg = read_config(ROOT / "configs/titans_smoke.toml")
    torch.manual_seed(0)
    model = make_model(cfg["model"]).eval()
    x = torch.randint(10, 100, (2, 32))
    labels = model._call_labels()
    stats = model.memory_health(x)
    kinds = ("share", "gate", "alpha", "theta", "drift")
    assert {f"memory/{kind}/{label}" for kind in kinds for label in labels} <= stats.keys()
    assert stats["memory/dead_calls"] == 0
    assert stats["memory/static_calls"] == 0
    # The model must run as before once the statistics are collected.
    assert all(b.health is None for b in model.modules() if hasattr(b, "health"))

    # A forget gate stuck at 1 wipes W_init on the first write. Near W = 0 the
    # update gradients vanish, so the memory cannot recover and every call is dead.
    for module in model.modules():
        if hasattr(module, "forget_bias"):
            module.forget_bias = 50.0
    stats = model.memory_health(x)
    assert stats["memory/wiping_heads"] == 1.0
    assert stats["memory/dead_calls"] == len(labels)


def test_titans_decay_to_init_survives_a_full_forget():
    cfg = dict(read_config(ROOT / "configs/titans_smoke.toml")["model"], memory_decay_to_init=True)
    torch.manual_seed(0)
    model = make_model(cfg).eval()
    for module in model.modules():
        if hasattr(module, "forget_bias"):
            module.forget_bias = 50.0
    # A full forget resets the memory to W_init, not to 0, so no call dies.
    stats = model.memory_health(torch.randint(10, 100, (2, 32)))
    assert stats["memory/wiping_heads"] == 1.0
    assert stats["memory/dead_calls"] == 0
    assert stats["memory/static_calls"] == 0

    # With no inner lr the memory never writes and keeps W = W_init. Its share
    # stays high, so it is not dead, but its drift shows that it is static.
    for module in model.modules():
        if hasattr(module, "forget_bias"):
            gates = module._gates
            module._gates = lambda ctx, gates=gates: (*gates(ctx)[:2], gates(ctx)[2] * 0.0)
    stats = model.memory_health(torch.randint(10, 100, (2, 32)))
    labels = model._call_labels()
    assert stats["memory/dead_calls"] == 0
    assert stats["memory/static_calls"] == len(labels)
    assert all(stats[f"memory/theta/{label}"] == 0.0 for label in labels)
    for module in model.modules():
        if hasattr(module, "forget_bias"):
            del module._gates

    # With no forget at all, the two rules are the same.
    memory = next(m for m in model.modules() if hasattr(m, "forget_bias"))
    state = memory._initial_state(1)
    kc, vc = torch.randn(2, memory.heads, 4, memory.head_dim).unbind(0)
    gates = [torch.full((memory.heads, 1, 1), g) for g in (0.0, 0.5, 0.05)]
    anchored = memory_update(state, kc, vc, *gates, 4)
    plain = memory_update(state[:4], kc, vc, *gates, 4)
    for a, p in zip(anchored[:4], plain, strict=True):
        torch.testing.assert_close(a, p)

    bad = dict(cfg, memory_decay_to_init="yes")
    with pytest.raises(ValueError, match="memory_decay_to_init"):
        make_model(bad)


def test_precision_and_compile_config(tmp_path):
    base = (ROOT / "configs/smoke.toml").read_text()
    valid = tmp_path / "bf16.toml"
    valid.write_text(base + 'precision = "bf16"\ncompile = false\n')
    config = read_config(valid)
    assert config["train"]["precision"] == "bf16"
    assert config["train"]["compile"] is False
    assert resolve_precision("auto", torch.device("cpu")) == "fp32"
    assert resolve_precision("bf16", torch.device("cpu")) == "bf16"

    bad_precision = tmp_path / "bad_prec.toml"
    bad_precision.write_text(base + 'precision = "fp8"\n')
    with pytest.raises(ValueError, match="precision"):
        read_config(bad_precision)

    bad_compile = tmp_path / "bad_compile.toml"
    bad_compile.write_text(base + 'compile = "yes"\n')
    with pytest.raises(ValueError, match="compile"):
        read_config(bad_compile)


def test_bf16_train_smoke(tmp_path):
    training = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    training.write_text("\n".join(json.dumps({"text": f"hello world {i}"}) for i in range(8)))
    validation.write_text("\n".join(json.dumps({"text": f"goodbye world {i}"}) for i in range(4)))
    tokenizer_path = tmp_path / "tokenizer"
    train_tokenizer(training, tokenizer_path, 512)
    data_path = tmp_path / "encoded"
    prepare(training, validation, tokenizer_path, data_path)
    config_path = tmp_path / "bf16.toml"
    config_path.write_text((ROOT / "configs/smoke.toml").read_text() + 'precision = "bf16"\n')
    train(config_path, data_path, tmp_path / "run", "cpu", stop_after=2)
    experiment = json.loads((tmp_path / "run/experiment.json").read_text())
    assert experiment["precision"] == "bf16"
    assert experiment["compile"] is False


def test_sft_init_from_and_assistant_masking(tmp_path):
    training = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    training.write_text("\n".join(json.dumps({"messages": [
        {"role": "user", "content": f"question {i}"},
        {"role": "assistant", "content": f"answer {i}"},
    ]}) for i in range(12)))
    validation.write_text("\n".join(json.dumps({"messages": [
        {"role": "user", "content": f"val question {i}"},
        {"role": "assistant", "content": f"val answer {i}"},
    ]}) for i in range(4)))
    tokenizer_path = tmp_path / "tokenizer"
    train_tokenizer(training, tokenizer_path, 512)
    tokenizer = load_tokenizer(tokenizer_path)
    data_path = tmp_path / "encoded"
    prepare(training, validation, tokenizer_path, data_path)

    # Check assistant-only masking across a boundary starting mid-assistant-turn.
    rendered_text = tokenizer.apply_chat_template([
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello world"},
    ], tokenize=False)
    ids = tokenizer(rendered_text, add_special_tokens=False)["input_ids"]
    assistant_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
    end_id = tokenizer.convert_tokens_to_ids("<|end|>")
    non_assistant = {tokenizer.convert_tokens_to_ids(t) for t in ("<|user|>", "<|system|>", "<|eos|>")}
    a_pos = ids.index(assistant_id)
    tokens = torch.tensor([ids[a_pos + 1:]], dtype=torch.long)
    y = mask_assistant_targets(ids, [a_pos + 1], tokens, assistant_id, end_id, non_assistant)
    assert (y[0, :-2] != -100).all()      # assistant content + <|end|> supervised
    assert (y[0, -2:] == -100).all()     # \n<|eos|> after <|end|> masked

    # Base run then SFT init_from with assistant_only = true.
    base_cfg = ROOT / "configs/smoke.toml"
    train(base_cfg, data_path, tmp_path / "base", "cpu", stop_after=2)
    sft_cfg = tmp_path / "sft.toml"
    sft_cfg.write_text(base_cfg.read_text() + "assistant_only = true\n")
    train(sft_cfg, data_path, tmp_path / "sft", "cpu", stop_after=2, init_from=tmp_path / "base/latest.pt")
    sft_exp = json.loads((tmp_path / "sft/experiment.json").read_text())
    assert sft_exp["init_from"].endswith("base/latest.pt")
    assert sft_exp["config"]["train"]["assistant_only"] is True
    with pytest.raises(ValueError, match="Cannot combine"):
        train(sft_cfg, data_path, tmp_path / "sft", "cpu", resume=True, init_from=tmp_path / "base/latest.pt")


def test_titans_repetition_penalty(monkeypatch):
    from types import SimpleNamespace

    cfg = read_config(ROOT / "configs/titans_smoke.toml")["model"]
    model = make_model(cfg).eval()
    base_logits = torch.full((1, 1, cfg["vocab_size"]), -10.0)
    base_logits[0, 0, 1] = 5.0
    base_logits[0, 0, 2] = 2.0
    base_logits[0, 0, 3] = -0.5
    monkeypatch.setattr(model, "forward", lambda idx, **kw: SimpleNamespace(logits=base_logits))

    # Seen positive logit (5.0 on token 1) beats 2.0 without penalty, but 5.0 / 4.0 = 1.25 < 2.0 with penalty=4.0.
    prompt = torch.tensor([[1, 3]])
    assert model.generate(prompt, max_new_tokens=1, do_sample=False, repetition_penalty=1.0)[0, -1].item() == 1
    assert model.generate(prompt, max_new_tokens=1, do_sample=False, repetition_penalty=4.0)[0, -1].item() == 2

    # When only negative logits compete, penalizing a seen negative logit (-0.5 * 4 = -2.0) makes unseen (-1.0) win.
    neg_logits = torch.full((1, 1, cfg["vocab_size"]), -10.0)
    neg_logits[0, 0, 3] = -0.5
    neg_logits[0, 0, 4] = -1.0
    monkeypatch.setattr(model, "forward", lambda idx, **kw: SimpleNamespace(logits=neg_logits))
    assert model.generate(prompt, max_new_tokens=1, do_sample=False, repetition_penalty=4.0)[0, -1].item() == 4


def test_looped_titans_mag_and_checkpoint_archive(tmp_path):
    base_titans = (ROOT / "configs/titans_smoke.toml").read_text()
    loop_cfg_path = tmp_path / "loop.toml"
    loop_cfg_path.write_text(
        base_titans
        .replace("memory_hidden_size = 16", "memory_hidden_size = 32\nmemory_heads = 4\nmemory_forget_bias = 0.0\nmemory_min_forget = 0.05\nnorm_attn_branch = true")
        .replace("num_hidden_layers = 2", "num_hidden_layers = 1\nprelude_layers = 1\nnum_loops = 3\ncoda_layers = 1")
        .replace("save_every = 2", "save_every = 2\nkeep_checkpoints = true")
    )
    cfg = read_config(loop_cfg_path)
    model = make_model(cfg["model"]).eval()
    assert hasattr(model, "prelude") and hasattr(model, "coda") and model.loop_embed.shape == (3, 1, cfg["model"]["hidden_size"])

    muon, adam = make_optimizers(model, cfg["train"])
    ids = [id(p) for o in (muon, adam) for g in o.param_groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == len(list(model.parameters()))
    adam_ids = {id(p) for g in adam.param_groups for p in g["params"]}
    assert id(model.loop_embed) in adam_ids

    # Token-level causality across multiple 4-token chunks with 4 memory heads + 3 depth loops.
    x = torch.randint(10, 100, (1, 10))
    y = x.clone()
    y[:, 5:] = torch.randint(100, 200, (1, 5))
    with torch.no_grad():
        torch.testing.assert_close(model(x).logits[:, :5], model(y).logits[:, :5])

    training = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    training.write_text("\n".join(json.dumps({"text": f"looped document {i} " * 8}) for i in range(12)))
    validation.write_text("\n".join(json.dumps({"text": f"looped val {i} " * 8}) for i in range(4)))
    tokenizer_path = tmp_path / "tokenizer"
    train_tokenizer(training, tokenizer_path, 512)
    data_path = tmp_path / "encoded"
    prepare(training, validation, tokenizer_path, data_path)
    train(loop_cfg_path, data_path, tmp_path / "loop_run", "cpu")
    assert (tmp_path / "loop_run/latest.pt").exists()
    assert (tmp_path / "loop_run/step_000002.pt").exists()
    assert (tmp_path / "loop_run/step_000004.pt").exists()
    sample(tmp_path / "loop_run", "looped", max_new_tokens=4, temperature=0, repetition_penalty=1.2, raw=True)

    # Validation checks on bad Titans/loop options.
    bad_cfg = read_config(ROOT / "configs/smoke.toml")
    bad_cfg["model"]["num_loops"] = 2
    with pytest.raises(ValueError, match="require memory_chunk_size"):
        make_model(bad_cfg["model"])


@pytest.mark.parametrize("looped,decay_to_init", [(False, False), (True, False), (True, True)])
def test_titans_cached_decode_matches_full_forward(looped, decay_to_init):
    # The prompt and each step cross memory-chunk (4) and attention-window (8) boundaries.
    cfg = dict(read_config(ROOT / "configs/titans_smoke.toml")["model"])
    cfg["memory_decay_to_init"] = decay_to_init
    if looped:
        cfg.update(memory_hidden_size=32, memory_heads=4, memory_min_forget=0.05, norm_attn_branch=True,
                   num_hidden_layers=1, prelude_layers=1, num_loops=3, coda_layers=1)
    torch.manual_seed(0)
    model = make_model(cfg).eval()
    for module in model.modules():
        if hasattr(module, "w1_init"):
            module.gate_proj.weight.data.normal_(0.0, 1.0)  # Make the memory updates matter.
    ids = torch.randint(10, 500, (1, cfg["max_position_embeddings"]))
    with torch.no_grad():
        full = model(ids).logits
        for prompt_len in (1, 3, 4, 13):
            cache = {}
            pieces = [model(ids[:, :prompt_len], cache=cache).logits]
            pieces += [model(ids[:, t:t + 1], cache=cache).logits for t in range(prompt_len, ids.shape[1])]
            torch.testing.assert_close(torch.cat(pieces, dim=1), full, rtol=1e-4, atol=1e-4)
        prompt = ids[:, :5]
        torch.testing.assert_close(
            model.generate(prompt, max_new_tokens=20, do_sample=False, use_cache=True),
            model.generate(prompt, max_new_tokens=20, do_sample=False, use_cache=False),
        )


def test_titans_group_sampling_and_logprobs():
    cfg = dict(read_config(ROOT / "configs/titans_smoke.toml")["model"])
    cfg.update(memory_hidden_size=32, memory_heads=4, num_hidden_layers=1, prelude_layers=1, num_loops=2, coda_layers=1)
    torch.manual_seed(0)
    model = make_model(cfg).eval()
    prompt = torch.randint(10, 500, (1, 6))
    single = model.generate(prompt, max_new_tokens=20, do_sample=False)
    group, logprobs = model.generate(prompt, max_new_tokens=20, do_sample=False, num_return_sequences=3, return_logprobs=True)
    assert torch.equal(group, single.repeat(3, 1))
    assert torch.equal(group, model.generate(prompt, max_new_tokens=20, do_sample=False, num_return_sequences=3, use_cache=False))
    # Logprobs agree with a full forward over the generated sequence.
    with torch.no_grad():
        full = model(group).logits.log_softmax(-1)[:, prompt.shape[1] - 1:-1]
    torch.testing.assert_close(logprobs, full.gather(2, group[:, prompt.shape[1]:, None]).squeeze(2), rtol=1e-4, atol=1e-4)

    # Rows stop on their own stop token; finished rows get padding.
    torch.manual_seed(1)
    sampled, lp = model.generate(prompt, max_new_tokens=20, do_sample=True, temperature=1.5,
                                 num_return_sequences=8, return_logprobs=True)
    stop = sampled[0, prompt.shape[1] + 2].item()
    torch.manual_seed(1)
    stopped, lp2 = model.generate(prompt, max_new_tokens=20, do_sample=True, temperature=1.5,
                                  num_return_sequences=8, return_logprobs=True, eos_token_id=stop, pad_token_id=0)
    new = stopped[:, prompt.shape[1]:]
    for row in range(8):
        hits = (sampled[row, prompt.shape[1]:] == stop).nonzero()
        end = hits[0].item() + 1 if len(hits) else new.shape[1]
        assert torch.equal(new[row, :end], sampled[row, prompt.shape[1]:prompt.shape[1] + end])
        assert (new[row, end:] == 0).all() and (lp2[row, end:] == 0).all()
    assert new.shape[1] <= 20 and (new[0, 3:] == 0).all()


def test_arith_problems_round_trip():
    import random as pyrandom
    from kestrel.arith import BUCKETS, bucket, draw, is_correct, parse, render

    rng = pyrandom.Random(0)
    for name in BUCKETS:
        for _ in range(50):
            problem = draw(name, rng)
            assert bucket(problem) == name
            assert parse(render(problem)[0]) == problem
    problem = ("sub", (127, 86))
    assert is_correct("First subtract 80: 127 - 80 = 47.\nThen subtract 6: 47 - 6 = 41.\nAnswer: 41", problem)
    assert not is_correct("Answer: 3", problem) and not is_correct("no answer", problem)


def _arith_run(tmp_path):
    """Train the Titans smoke model for a few steps on arithmetic text and return its run directory."""
    import random as pyrandom
    from kestrel.arith import BUCKETS, draw, render

    rng = pyrandom.Random(0)
    docs = []
    for i in range(60):
        line, answer = render(draw(list(BUCKETS)[i % len(BUCKETS)], rng))
        docs.append(json.dumps({"text": f"{line}\nAnswer: {answer}"}))
    training, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    training.write_text("\n".join(docs[:48]))
    validation.write_text("\n".join(docs[48:]))
    train_tokenizer(training, tmp_path / "tokenizer", 512)
    prepare(training, validation, tmp_path / "tokenizer", tmp_path / "encoded")
    train(ROOT / "configs/titans_smoke.toml", tmp_path / "encoded", tmp_path / "base", "cpu")
    return tmp_path / "base"


def test_grpo_step_direction_and_logprob_agreement(tmp_path):
    from kestrel.grpo import completion_mask, policy_loss, rollout
    from kestrel.train import _load_for_inference

    assert completion_mask(torch.tensor([[5, 2, 0, 0], [5, 6, 7, 8]]), 2).tolist() == [[True, True, False, False], [True] * 4]
    model, tokenizer, _, bad = _load_for_inference(_arith_run(tmp_path), "cpu", 0)
    problems = [("sub", (127, 86)), ("mul", (14, 8)), ("add", (47, 38))]
    groups = rollout(model, tokenizer, problems, group=4, temperature=1.0, max_new_tokens=12, max_rows=8, bad_words_ids=bad)
    assert [g["problem"] for g in groups] == problems
    # Pretend the first sample of each group is right, so every group has a nonzero advantage.
    for g in groups:
        g["rewards"] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    before = _completion_logprob(model, groups[0], 0)
    stats = policy_loss(model, groups, tokenizer.pad_token_id, normaliser=1.0, micro_rows=5)
    assert stats["rows"] == 12 and stats["logprob_gap"] < 1e-4
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= 1e-3 * p.grad
    assert _completion_logprob(model, groups[0], 0) > before


def _completion_logprob(model, g, row):
    n = len(g["prompt"])
    ids = torch.cat([g["prompt"], g["completions"][row]])[None]
    with torch.no_grad():
        logp = model(ids).logits[0, :-1].log_softmax(-1).gather(1, ids[0, 1:, None]).squeeze(1)
    return (logp[n - 1:] * g["mask"][row]).sum().item()


def test_grpo_run_and_resume(tmp_path):
    from kestrel.grpo import DEFAULT_WEIGHTS, grpo

    base = _arith_run(tmp_path)
    settings = {"steps": 2, "prompts": 3, "group": 2, "max_new_tokens": 8, "max_rows": 8, "micro_rows": 4,
                "eval_every": 2, "eval_problems": 1, "eval_group": 2, "save_every": 1, "temperature": 1.0,
                "lr": 1e-4, "warmup": 1, "clip": 1.0, "seen_from": None, "seed": 0, "weights": DEFAULT_WEIGHTS}
    grpo(base, tmp_path / "rl", settings, "cpu")
    records = [json.loads(line) for line in (tmp_path / "rl/metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [0, 1, 2] and "eval_greedy" in records[-1]
    with pytest.raises(ValueError):
        grpo(None, tmp_path / "rl", {**settings, "lr": 1.0}, "cpu", resume=True)
    with pytest.raises(ValueError):
        grpo(None, tmp_path / "rl", settings, "cpu", resume=True)
    grpo(None, tmp_path / "rl", {**settings, "steps": 3}, "cpu", resume=True)
    records = [json.loads(line) for line in (tmp_path / "rl/metrics.jsonl").read_text().splitlines()]
    assert [r["step"] for r in records] == [0, 1, 2, 3]
    sample(tmp_path / "rl", "Compute 1 + 2 step by step.", "cpu", max_new_tokens=4, raw=True)



@pytest.mark.parametrize("config", ["smoke", "titans_smoke", "looped"])
@pytest.mark.parametrize("ablate", [(), ((0, 1), (1, 2))])
def test_playground_cached_sample_is_the_uncached_sample(tmp_path, config, ablate):
    pytest.importorskip("suzunaan")
    from suzunaan import Prompt
    from suzunaan.sample import sample as playground_sample
    from kestrel.playground import KestrelBackend

    training = tmp_path / "train.jsonl"
    training.write_text("\n".join(json.dumps({"text": f"playground document {i} " * 8}) for i in range(12)))
    train_tokenizer(training, tmp_path / "tokenizer", 300)  # Fewer than vocab_size: unused rows.
    tok = load_tokenizer(tmp_path / "tokenizer")
    cfg = dict(read_config(ROOT / f"configs/{'titans_smoke' if config == 'looped' else config}.toml")["model"])
    if config == "looped":
        cfg.update(memory_hidden_size=32, memory_heads=4, num_hidden_layers=1, prelude_layers=1, num_loops=3, coda_layers=1)
    torch.manual_seed(0)
    model = make_model(cfg, {"bos_token_id": tok.bos_token_id, "eos_token_id": tok.eos_token_id,
                             "pad_token_id": tok.pad_token_id})
    for module in model.modules():
        if hasattr(module, "w1_init"):
            module.gate_proj.weight.data.normal_(0.0, 1.0)  # Make the memory updates matter.
    torch.save({"config": {"model": cfg}, "model": model.state_dict()}, tmp_path / "latest.pt")

    backend = KestrelBackend(tmp_path / "latest.pt", tmp_path / "tokenizer", torch.device("cpu"))
    calls = []
    uncached_logits = backend.logits
    backend.logits = lambda ids: calls.append(len(ids)) or uncached_logits(ids)
    # No stop token, and past the context, where the window recomputes from scratch.
    prompt = Prompt(list(range(20, 25)))
    n = backend.context + 6
    run = lambda: [t["id"] for t in playground_sample(backend, prompt, n, seed=3, top_k=20, ablate=ablate)]
    cached = run()
    assert len(calls) == n - (backend.context - 5 + 1)    # uncached only past the context
    backend.decoder = None
    assert run() == cached
    assert all(i < len(tok) for i in cached)              # the unused rows stay masked


HNET_TITANS_MAIN = '''[model.main]
network = "titans"
hidden_size = 48
intermediate_size = 96
num_hidden_layers = 1
num_attention_heads = 4
num_key_value_heads = 2
head_dim = 12
memory_chunk_size = 4
memory_hidden_size = 16
attention_window = 8
num_loops = 2
'''


def hnet_config(tmp_path, main):
    text = (ROOT / "configs/hnet_smoke.toml").read_text()
    if main == "titans":
        start, end = text.index("[model.main]"), text.index("[train]")
        text = text[:start] + HNET_TITANS_MAIN + "\n" + text[end:]
    path = tmp_path / f"hnet_{main}.toml"
    path.write_text(text)
    return path


def test_byte_tokenizer_has_no_merges(tmp_path):
    training = tmp_path / "train.jsonl"
    training.write_text("\n".join(json.dumps({"text": f"the the the {i}"}) for i in range(20)))
    train_tokenizer(training, tmp_path / "bytes", 263)
    tokenizer = load_tokenizer(tmp_path / "bytes")
    assert len(tokenizer) == 263
    text = "the café 🦉 1270"
    ids = tokenizer.encode(text, add_special_tokens=False)
    assert len(ids) == len(text.encode("utf-8"))
    assert tokenizer.decode(ids) == text


def test_hnet_smoothing_and_ratio_loss():
    from kestrel.hnet import ratio_loss, smooth

    torch.manual_seed(0)
    z = torch.randn(2, 6, 3)
    p = torch.rand(2, 6) * 0.5 + 0.5
    p[:, 0] = 1.0
    expected, state = [], torch.zeros(2, 3)
    for j in range(6):
        state = p[:, j, None] * z[:, j] + (1 - p[:, j, None]) * state
        expected.append(state)
    torch.testing.assert_close(smooth(z, p), torch.stack(expected, dim=1))

    b = torch.tensor([[True, False, False, False] * 4])
    at_target = ratio_loss(torch.full((1, 16), 0.25), b, 4.0)
    torch.testing.assert_close(at_target, torch.tensor(1.0))
    assert ratio_loss(torch.full((1, 16), 0.9), torch.ones(1, 16, dtype=torch.bool), 4.0) > at_target


@pytest.mark.parametrize("main", ["transformer", "titans"])
def test_hnet_causality_and_train(tmp_path, main):
    config_path = hnet_config(tmp_path, main)
    model = make_model(read_config(config_path)["model"]).eval()
    assert model.lm_head.weight is model.embed_tokens.weight
    torch.manual_seed(1)
    x = torch.randint(7, 263, (3, 40))
    y = x.clone()
    y[:, 20:] = torch.randint(7, 263, (3, 20))
    with torch.no_grad():
        a, b = model(x), model(y)
    torch.testing.assert_close(a.logits[:, :20], b.logits[:, :20])
    torch.testing.assert_close(a.boundaries[:, :20], b.boundaries[:, :20])
    assert a.boundaries[:, 0].all()

    training = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    training.write_text("\n".join(json.dumps({"text": f"long document {i} " * 8}) for i in range(12)))
    validation.write_text("\n".join(json.dumps({"text": f"val document {i} " * 8}) for i in range(4)))
    train_tokenizer(training, tmp_path / "tokenizer", 263)
    prepare(training, validation, tmp_path / "tokenizer", tmp_path / "encoded")
    train(config_path, tmp_path / "encoded", tmp_path / "run", "cpu")
    records = [json.loads(line) for line in (tmp_path / "run/metrics.jsonl").read_text().splitlines()]
    assert all(r["bytes_per_chunk"] >= 1 and "ratio_loss" in r for r in records)
    evaluated = [r for r in records if "validation_loss" in r]
    assert evaluated
    # A Titans main network logs its memory health; a transformer has no memory.
    assert all(("memory/dead_calls" in r) == (main == "titans") for r in evaluated)
    sample(tmp_path / "run", "long", max_new_tokens=4, temperature=0, raw=True)


def _hnet_model(tmp_path, main):
    cfg = read_config(hnet_config(tmp_path, main))["model"]
    torch.manual_seed(0)
    model = make_model(cfg, {"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0}).eval()
    for module in model.modules():
        if hasattr(module, "w1_init"):
            module.gate_proj.weight.data.normal_(0.0, 1.0)  # Make the memory updates matter.
    return cfg, model


@pytest.mark.parametrize("main", ["transformer", "titans"])
def test_hnet_cached_decode_matches_full_forward(tmp_path, main):
    # The byte windows are 8; a Titans main network has memory chunks of 4 and a window of 8.
    cfg, model = _hnet_model(tmp_path, main)
    torch.manual_seed(1)
    ids = torch.randint(7, 263, (1, cfg["max_position_embeddings"]))
    with torch.no_grad():
        full = model(ids)
        assert 4 < full.boundaries.sum() < ids.shape[1] - 4  # chunks of more than one byte, and many chunks
        for prompt_len in (1, 3, 13):
            cache = {}
            pieces = [model(ids[:, :prompt_len], cache=cache)]
            pieces += [model(ids[:, t:t + 1], cache=cache) for t in range(prompt_len, ids.shape[1])]
            torch.testing.assert_close(torch.cat([o.boundaries for o in pieces], dim=1), full.boundaries)
            torch.testing.assert_close(torch.cat([o.logits for o in pieces], dim=1), full.logits, rtol=1e-4, atol=1e-4)
        prompt = ids[:, :5]
        torch.testing.assert_close(
            model.generate(prompt, max_new_tokens=20, do_sample=False, use_cache=True),
            model.generate(prompt, max_new_tokens=20, do_sample=False, use_cache=False),
        )
        with pytest.raises(ValueError, match="batch of one"):
            model(ids.repeat(2, 1), cache={})


@pytest.mark.parametrize("main", ["transformer", "titans"])
def test_playground_hnet_backend(tmp_path, main):
    pytest.importorskip("suzunaan")
    from suzunaan import Prompt, views
    from suzunaan.sample import sample as playground_sample
    from kestrel.playground import KestrelBackend

    training = tmp_path / "train.jsonl"
    training.write_text("\n".join(json.dumps({"text": f"playground document {i}"}) for i in range(4)))
    train_tokenizer(training, tmp_path / "tokenizer", 263)  # One fewer than vocab_size: an unused row.
    cfg, model = _hnet_model(tmp_path, main)
    torch.save({"config": {"model": cfg}, "model": model.state_dict()}, tmp_path / "latest.pt")

    backend = KestrelBackend(tmp_path / "latest.pt", tmp_path / "tokenizer", torch.device("cpu"))
    assert backend.hnet and backend.layers == cfg["encoder_layers"] + cfg["decoder_layers"]
    assert "chunks" in backend.capabilities
    assert backend.shape.startswith("H-Net")
    ids = backend.prompt("text", {"text": "playground doc", "after_eot": True}).ids
    assert len(ids) == 1 + len(b"playground doc")
    with torch.no_grad():
        torch.testing.assert_close(backend.logits(ids)[:-1], model(torch.tensor([ids])).logits[0, -1, :-1])
    assert backend.logits(ids)[-1] == float("-inf")

    # Every view runs, with one row per byte layer.
    _, rows = views.attention_from(backend, ids, len(ids) - 1)
    assert len(rows) == backend.layers and len(rows[0]) == backend.heads
    _, lens = views.logit_lens(backend, ids, 0, len(ids))
    assert len(lens) == backend.layers + 1
    views.attention_matrix(backend, ids, backend.layers - 1, 1, 0, len(ids))
    views.head_map(backend, ids)
    _, active, _, _ = views.neuron_map(backend, ids, 0, 0, len(ids))
    assert len(active) == backend.layers
    p, starts = backend.chunks(ids)
    assert p.shape == starts.shape == (len(ids),) and starts[0] and torch.equal(starts, p >= 0.5)
    with torch.no_grad():
        torch.testing.assert_close(starts, model(torch.tensor([ids])).boundaries[0])

    # Cached sampling matches uncached, past the context too, with heads off or not.
    prompt = Prompt(ids)
    n = backend.context - len(ids) + 6
    for ablate in [(), ((0, 1), (1, 2))]:
        run = lambda: [t["id"] for t in playground_sample(backend, prompt, n, seed=3, top_k=20, ablate=ablate)]
        cached = run()
        backend.decoder = None
        assert run() == cached
        del backend.decoder
        assert all(i < 263 for i in cached)


@pytest.mark.parametrize("length,window", [(5, 8), (8, 8), (10, 4), (37, 8), (300, 64)])
def test_local_window_attention_matches_dense(length, window):
    from kestrel.memory import dense_window_attention, local_window_attention

    torch.manual_seed(length)
    q, k, v = (torch.randn(2, 3, length, 8) for _ in range(3))
    torch.testing.assert_close(local_window_attention(q, k, v, window), dense_window_attention(q, k, v, window))


def test_hnet_stage_learning_rates(tmp_path):
    cfg = read_config(hnet_config(tmp_path, "transformer"))
    model = make_model(cfg["model"])
    multiplier = model.config.lr_multiplier
    assert multiplier == pytest.approx((4.0 * 48 / 32) ** 0.5)
    inner = {id(p) for p in model.main.parameters()}
    seen = []
    for optimizer, base in zip(make_optimizers(model, cfg["train"]), (cfg["train"]["muon_lr"], cfg["train"]["adamw_lr"])):
        for group in optimizer.param_groups:
            in_main = {id(p) in inner for p in group["params"]}
            assert len(in_main) == 1
            assert group["base_lr"] == pytest.approx(base if in_main.pop() else base * multiplier)
            seen += [id(p) for p in group["params"]]
    assert len(seen) == len(set(seen)) == len(list(model.parameters()))

    fixed = dict(cfg["model"], lr_multiplier=1.0)
    groups = [g for o in make_optimizers(make_model(fixed), cfg["train"]) for g in o.param_groups]
    assert len(groups) == 3
    with pytest.raises(ValueError):
        make_model(dict(read_config(ROOT / "configs/smoke.toml")["model"], lr_multiplier=2.0))


@pytest.mark.parametrize("kind", ["decoder", "titans", "hnet-transformer", "hnet-titans"])
def test_gradient_checkpointing_keeps_gradients(tmp_path, kind):
    path = {"decoder": ROOT / "configs/smoke.toml", "titans": ROOT / "configs/titans_smoke.toml"}.get(kind)
    if path is None:
        path = hnet_config(tmp_path, kind.split("-")[1])
    cfg = read_config(path)
    torch.manual_seed(3)
    model = make_model(cfg["model"]).train()
    x = torch.randint(7, 200, (2, 24))

    def grads():
        model.zero_grad(set_to_none=True)
        output = model(input_ids=x, use_cache=False)
        loss = output.logits.float().logsumexp(-1).mean() + getattr(output, "aux_loss", 0.0)
        loss.backward()
        return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}

    plain = grads()
    if hasattr(model, "set_gradient_checkpointing"):
        model.set_gradient_checkpointing(True)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    checkpointed = grads()
    assert plain.keys() == checkpointed.keys()
    for name in plain:
        torch.testing.assert_close(checkpointed[name], plain[name], msg=name)

    if kind == "hnet-titans":
        text = path.read_text().replace("seed = 7", "seed = 7\ngradient_checkpointing = true")
        path.write_text(text)
        training = tmp_path / "train.jsonl"
        validation = tmp_path / "validation.jsonl"
        training.write_text("\n".join(json.dumps({"text": f"long document {i} " * 8}) for i in range(12)))
        validation.write_text("\n".join(json.dumps({"text": f"val document {i} " * 8}) for i in range(4)))
        train_tokenizer(training, tmp_path / "tokenizer", 263)
        prepare(training, validation, tmp_path / "tokenizer", tmp_path / "encoded")
        assert train(path, tmp_path / "encoded", tmp_path / "run", "cpu").main.model.gradient_checkpointing
