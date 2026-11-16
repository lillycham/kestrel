import copy
from pathlib import Path

import pytest
import torch

from kestrel.core import make_model, make_optimizers, read_config
from kestrel.hnet import smooth, smooth_dense
from kestrel.performance import BatchedMuon, configure_training
from kestrel.train import loss_on


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    torch.set_num_threads(1)


def tiny_hnet(main="titans"):
    config = read_config(ROOT / "configs/hnet_titans.toml")
    config["model"].update(hidden_size=32, intermediate_size=64, num_attention_heads=4,
                           num_key_value_heads=2, head_dim=8, max_position_embeddings=64,
                           attention_window=8, encoder_layers=1, decoder_layers=1)
    config["model"]["main"].update(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                    num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                    memory_chunk_size=4, memory_hidden_size=32, memory_heads=2,
                                    attention_window=8, num_loops=2)
    if main == "transformer":
        config["model"]["main"] = {"network": "transformer", "hidden_size": 32, "intermediate_size": 64,
                                   "num_hidden_layers": 2, "num_attention_heads": 4,
                                   "num_key_value_heads": 2, "head_dim": 8}
    return config


@pytest.mark.parametrize("length", [1, 63, 64, 65, 257, 769])
@pytest.mark.parametrize("pattern", ["random", "extremes"])
def test_block_ema_outputs_and_gradients(length, pattern):
    torch.manual_seed(11)
    z = torch.randn(2, length, 8, requires_grad=True)
    p = torch.rand(2, length)
    if pattern == "extremes":
        p[:, ::3] = 0
        p[:, 1::3] = 1
        p[:, 2::3] = 1e-7
    p[:, 0] = 1
    p.requires_grad_()
    probe = torch.randn_like(z)
    expected = smooth_dense(z, p)
    actual = smooth(z, p)
    torch.testing.assert_close(actual, expected, atol=3e-4, rtol=3e-4)
    # The old dense cumulative-log form loses gradient precision near p == 1.
    # Use a float64 recurrence as the gradient oracle, rather than copying that error.
    state = torch.zeros(2, 8, dtype=torch.float64)
    reference = []
    for i in range(length):
        state = p[:, i, None].double() * z[:, i].double() + (1 - p[:, i, None].double()).clamp_min(1e-6) * state
        reference.append(state)
    reference_grads = torch.autograd.grad((torch.stack(reference, 1) * probe).sum(), (z, p))
    actual_grads = torch.autograd.grad((actual * probe).sum(), (z, p))
    for got, want in zip(actual_grads, reference_grads):
        torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-5)


def test_block_ema_bf16_against_recurrence():
    z = torch.randn(2, 257, 8, dtype=torch.bfloat16, requires_grad=True)
    p = torch.rand(2, 257, requires_grad=True)
    state = torch.zeros(2, 8)
    expected = []
    for i in range(z.shape[1]):
        state = p[:, i, None] * z[:, i].float() + (1 - p[:, i, None]).clamp_min(1e-6) * state
        expected.append(state)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        got = smooth(z, p)
    torch.testing.assert_close(got, torch.stack(expected, dim=1).bfloat16())
    got.float().square().sum().backward()
    assert torch.isfinite(p.grad).all() and torch.isfinite(z.grad).all()


@pytest.mark.parametrize("nesterov", [False, True])
def test_batched_muon_equivalence_and_resume(nesterov):
    torch.manual_seed(17)
    left = [torch.nn.Parameter(torch.randn(*shape)) for shape in [(8, 4), (8, 4), (4, 8), (4, 8), (4, 4)]]
    right = [torch.nn.Parameter(p.detach().clone()) for p in left]
    def groups(params):
        return [{"params": params[:3], "lr": 0.02}, {"params": params[3:], "lr": 0.006}]
    ref = torch.optim.Muon(groups(left), weight_decay=0.03, nesterov=nesterov, adjust_lr_fn="original")
    opt = BatchedMuon(groups(right), weight_decay=0.03, nesterov=nesterov, adjust_lr_fn="original")
    for step in range(4):
        for i, (a, b) in enumerate(zip(left, right)):
            grad = torch.randn_like(a) if not (step == 1 and i == 0) else None
            a.grad = grad
            b.grad = grad.clone() if grad is not None else None
        ref.step()
        opt.step()
        for a, b in zip(left, right):
            torch.testing.assert_close(a, b, atol=5e-4, rtol=5e-4)
            torch.testing.assert_close(ref.state[a]["momentum_buffer"], opt.state[b]["momentum_buffer"])
        if step == 1:
            saved = copy.deepcopy(opt.state_dict())
            opt = BatchedMuon(groups(right), weight_decay=0.03, nesterov=nesterov, adjust_lr_fn="original")
            opt.load_state_dict(saved)
    # Standard Muon can also read the saved format.
    standard = torch.optim.Muon(groups(right))
    standard.load_state_dict(opt.state_dict())
    assert standard.param_groups[1]["lr"] == 0.006


@pytest.mark.parametrize("main", ["titans", "transformer"])
@pytest.mark.parametrize("aot", [False, True])
def test_compile_reuses_graphs_across_chunk_lengths(aot, main):
    torch._dynamo.reset()
    torch.manual_seed(3)
    model = make_model(tiny_hnet(main)["model"])
    eager = copy.deepcopy(model)
    keys = set(model.state_dict())
    graphs = []
    def backend(graph, inputs):
        graphs.append(graph)
        if aot:
            from torch._dynamo.backends.registry import lookup_backend
            return lookup_backend("aot_eager")(graph, inputs)
        return graph.forward
    model.compile_submodules(backend=backend, fullgraph=True)
    assert set(model.state_dict()) == keys
    compiled_encode = model._compiled_encode
    eager_encode = eager._encode_and_route
    boundary = None
    def force(encode):
        def wrapped(x):
            h, p, _, cos, sin, aux = encode(x)
            return h, p, boundary, cos, sin, aux
        return wrapped
    model._compiled_encode = force(compiled_encode)
    eager._compiled_encode = force(eager_encode)
    x = torch.randint(7, 200, (2, 48))
    # Chunk counts above the main attention window (8). The first three warm
    # up every graph variant, in training and in evaluation: Titans graphs
    # split on whether the memory chunks (4) and attention blocks (8) need
    # padding. Later counts must reuse those graphs.
    count_after_warmup = None
    for visit, count in enumerate([12, 13, 16, 24, 9, 14, 21, 12, 40]):
        boundary = torch.zeros_like(x, dtype=torch.bool)
        boundary[:, torch.arange(count) * (48 // count)] = True
        model.zero_grad(set_to_none=True)
        eager.zero_grad(set_to_none=True)
        actual, expected = model(x), eager(x)
        torch.testing.assert_close(actual.logits, expected.logits)
        (actual.logits.square().mean() + actual.aux_loss).backward()
        (expected.logits.square().mean() + expected.aux_loss).backward()
        for a, b in zip(model.parameters(), eager.parameters()):
            torch.testing.assert_close(a.grad, b.grad)
        with torch.no_grad():
            torch.testing.assert_close(model(x).logits, eager(x).logits)
        if visit == 2:
            count_after_warmup = len(graphs)
            assert count_after_warmup <= 18
        elif visit > 2:
            assert len(graphs) == count_after_warmup
    torch._dynamo.reset()


@pytest.mark.parametrize("main", ["titans", "transformer"])
def test_compiled_main_layers_see_one_dtype_under_autocast(main):
    # Under BF16 autocast, up_proj gives BF16 but later layers see FP32. If the
    # first layer got BF16, every graph variant would exist twice.
    from torch._dynamo.eval_frame import _debug_get_cache_entry_list
    from kestrel.memory import TitansMAGBlock

    torch._dynamo.reset()
    torch.manual_seed(3)
    model = make_model(tiny_hnet(main)["model"])
    model.compile_submodules(backend="aot_eager")
    encode, boundary = model._compiled_encode, None
    def forced(x):
        h, p, _, cos, sin, aux = encode(x)
        return h, p, boundary, cos, sin, aux
    model._compiled_encode = forced
    x = torch.randint(7, 200, (2, 48))
    for count in [12, 13, 16, 24, 9]:
        boundary = torch.zeros_like(x, dtype=torch.bool)
        boundary[:, :count] = True
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = model(x)
        (output.logits.float().square().mean() + output.aux_loss).backward()
    if main == "titans":
        regions = [TitansMAGBlock._before_memory, TitansMAGBlock._after_memory]
    else:
        regions = [type(model.main.model.layers[0]).forward]
    # Titans: padded or not. Transformer: the first length, then dynamic.
    assert all(len(_debug_get_cache_entry_list(f.__code__)) == 2 for f in regions)
    torch._dynamo.reset()


def test_masked_loss_empty_and_partial():
    model = make_model(tiny_hnet()["model"])
    x = torch.randint(7, 200, (2, 16))
    y = torch.full_like(x, -100)
    empty = loss_on(model, x, y, aux=False)
    assert empty.item() == 0
    empty.backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0 for p in model.parameters())
    y[:, ::2] = x[:, ::2]
    expected = torch.nn.functional.cross_entropy(model(x).logits.flatten(0, 1), y.flatten())
    torch.testing.assert_close(loss_on(model, x, y, aux=False), expected)


def test_runtime_options_and_optimizer_partition():
    cfg = tiny_hnet()
    cfg["train"]["batched_muon"] = True
    cfg["train"]["compile"] = False
    model = make_model(cfg["model"])
    configure_training(model, {"hnet_smoothing": "dense"})
    assert model.smoothing == "dense"
    with pytest.raises(ValueError, match="CUDA"):
        configure_training(model, {"attention_backend": "flash2"})
    optimizers = make_optimizers(model, cfg["train"])
    assert isinstance(optimizers[0], BatchedMuon)
    params = [id(p) for o in optimizers for g in o.param_groups for p in g["params"]]
    assert len(params) == len(set(params)) == len(list(model.parameters()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_flash2_local_attention_outputs_and_gradients():
    pytest.importorskip("flash_attn")
    torch.manual_seed(13)
    model = make_model(tiny_hnet()["model"]).cuda()
    flash = copy.deepcopy(model)
    configure_training(flash, {"attention_backend": "flash2"})
    # Compare attention directly: tiny roundoff changes can flip a hard router
    # decision in an end-to-end H-Net comparison.
    left_attn = model.encoder[0].self_attn
    right_attn = flash.encoder[0].self_attn
    x = torch.randn(2, 31, 32, device="cuda", requires_grad=True)
    other = x.detach().clone().requires_grad_()
    cos, sin = model.rotary_emb(x, torch.arange(31, device="cuda")[None, :])
    with torch.autocast("cuda", dtype=torch.bfloat16):
        left, right = left_attn(x, cos, sin), right_attn(other, cos, sin)
    torch.testing.assert_close(left, right, atol=0.002, rtol=0.02)
    left.float().square().mean().backward()
    right.float().square().mean().backward()
    torch.testing.assert_close(x.grad, other.grad, atol=1e-5, rtol=0.05)
    for a, b in zip(left_attn.parameters(), right_attn.parameters()):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-5, rtol=0.05)
