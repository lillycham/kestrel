import json
from pathlib import Path

import pytest
import torch

from kestrel.core import make_model, make_optimizers, read_config, resolve_precision, wsd
from kestrel.data import load_tokenizer, prepare, rows, train_tokenizer
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


@pytest.mark.parametrize("looped", [False, True])
def test_titans_cached_decode_matches_full_forward(looped):
    # The prompt and each step cross memory-chunk (4) and attention-window (8) boundaries.
    cfg = dict(read_config(ROOT / "configs/titans_smoke.toml")["model"])
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
