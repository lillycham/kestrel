# Kestrel

Experiments on the smallest useful conversational language model.

The decoder uses the Transformers Qwen3 implementation with random initial weights.
It uses SwiGLU, RoPE, RMSNorm, per-head QK-norm, GQA, tied embeddings and no projection biases.
Kestrel does not download pretrained weights or reuse an earlier model implementation.

## Environment

```sh
nix develop
python -m kestrel inspect configs/small.toml
python -m pytest -q
```

The flake pins the toolchain. It supplies PyTorch, Transformers, Tokenizers, NumPy and pytest.
The `.python-env/bin/python` link gives editors a stable interpreter path.
Run commands from the repository root. No editable install is necessary.

`configs/small.toml` defines an initial model shape with an 8,192-token vocabulary.
This shape is a starting point, not a result from a model-shape experiment.
`configs/smoke.toml` defines a tiny model for functional checks.

## Data

Supply separate UTF-8 JSONL files for train and validation.
Split source conversations before tokenization. Remove duplicates across splits at the source level.
Each line must use one of these formats:

```json
{"text": "A plain text document."}
{"messages": [{"role": "user", "content": "Hello!"}, {"role": "assistant", "content": "Hello! How are you?"}]}
```

Supported roles are `system`, `user` and `assistant`.
Train the tokenizer on the train split only:

```sh
python -m kestrel tokenizer --train data/train.jsonl --output data/tokenizer --vocab-size 8192
python -m kestrel prepare --train data/train.jsonl --validation data/validation.jsonl --tokenizer data/tokenizer --output data/encoded
```

The tokenizer uses byte-level BPE with a complete byte alphabet and explicit chat control tokens.
Small corpora can produce fewer tokens than the requested vocabulary size.
The model vocabulary must hold every tokenizer ID. Generation excludes unused output rows.

Preparation streams tokens to disk and records content hashes.
The train loop uses memory-mapped token files. Validation never trains the tokenizer or the model.
The file checks reject identical split files. They do not detect partial overlap or paraphrases.

The initial objective predicts all tokens, including user messages and control tokens.
The token stream packs documents with EOS separators. Attention can cross those separators within a sampled block.
Assistant-only loss and document-isolated attention are not implemented.

## Train and resume

```sh
python -m kestrel train --config configs/small.toml --data data/encoded --output runs/first --device cpu
```

Device `auto` selects CUDA, MPS or CPU, in that order.
Use an explicit device when another job uses the accelerator.
The initial trainer uses FP32. It does not enable compilation or mixed precision.
GPU support depends on the host and the PyTorch build. The initial verification covers CPU only.

Muon updates hidden matrices. AdamW updates embeddings, the output head and norm parameters.
The Muon implementation uses PyTorch's `original` shape adjustment.
Norm parameters have no weight decay. Each tied parameter occurs in one optimizer group.
The initial learning rates require experiments; they are not tuned for every model size.

The WSD schedule uses linear warmup, a constant phase and linear decay to zero after the final update.
Gradient accumulation controls the effective batch size.
Each optimizer step consumes `batch_size * sequence_length * gradient_accumulation` tokens.
Validation uses fixed samples from the separate validation stream.

To pause at an absolute step, preserve the schedule horizon:

```sh
python -m kestrel train --config configs/small.toml --data data/encoded --output runs/first --device cpu --stop-after 500
python -m kestrel train --config configs/small.toml --data data/encoded --output runs/first --device cpu --resume
```

Resume requires the same config, encoded data and device type.
Checkpoints preserve model weights, both optimizers, the step and random states.
Each save atomically replaces `latest.pt`. Metrics append to `metrics.jsonl`.
An interruption loses progress after the latest save. Configure `save_every` for the desired interval.
The initial version does not retain checkpoint history or branch a new cooldown schedule.

## Byte-level H-Net

`architecture = "hnet"` selects a one-stage H-Net with dynamic chunking (Hwang, Wang and Gu, 2025).
Sliding-window attention layers encode bytes. A router puts a chunk boundary where consecutive encoder states differ.
The main network runs on the chunks. A smoothing step and a byte decoder then predict the next byte.
`[model.main]` sets the main network: `network = "transformer"` (`configs/hnet.toml`) or `"titans"` (`configs/hnet_titans.toml`).

Prepare the data with a byte tokenizer. A vocabulary of 263 holds the 256 bytes and the 7 control tokens, and has no merges:

```sh
python -m kestrel tokenizer --train data/train.jsonl --output data/bytes-tokenizer --vocab-size 263
python -m kestrel prepare --train data/train.jsonl --validation data/validation.jsonl --tokenizer data/bytes-tokenizer --output data/bytes
```

`target_chunk_size` is the target number of bytes per chunk, and `ratio_loss_weight` scales the ratio loss.
The byte stage trains at a higher learning rate than the main network, as in the H-Net paper.
The default multiplier is `sqrt(target_chunk_size * main width / byte width)`. Set `lr_multiplier` in `[model]` to change it.
`train_loss` includes the ratio loss. `validation_loss` is cross-entropy per byte only.
The metrics log also records `bytes_per_chunk`, `ratio_loss` and `chunks_per_row`.
Generation runs the whole window again for each byte, because H-Net has no decode cache yet.

`gradient_checkpointing = true` in `[train]` recomputes each layer in the backward pass. It works for every model kind.
For `configs/hnet.toml` at 2 x 4096 bytes on MPS, it cuts memory from 3.6 GB to 1.6 GB.

### H-Net training performance

`compile = true` compiles the fixed byte encoder and decoder. The variable-length main network stays eager.
Titans shares compiled retrieval and update kernels across its memory layers. Each kernel processes one fixed-size memory chunk.
The first and later updates can need separate graphs because their momentum tensors have different gradient requirements.
A new router length does not require a new whole-main graph. Evaluation can require separate graphs too.
Compilation no longer adds padding to multiples of 256. Rows still share the maximum chunk count in each batch.

The default EMA uses 64-position blocks and a scan of block summaries. It avoids a sequence-sized square matrix.
Its products and accumulation use FP32, including under BF16 autocast. This can change rounding relative to the old dense EMA.
Use `hnet_smoothing = "dense"` under `[train]` to compare with the old formula.

Two optional `[train]` settings support separate measurements:

- `attention_backend = "flash2"` uses FlashAttention-2 for H-Net and Titans local attention. It requires CUDA, BF16/FP16, and `flash-attn`.
  It preserves the local window and GQA. The transformer main network keeps its existing SDPA implementation.
  `"sdpa"` remains the default. Cached generation uses the existing attention path.
- `batched_muon = true` groups equal-shaped matrices within each optimizer group. Each matrix retains its own normalization and momentum.
  The standard Muon checkpoint layout and stage learning rates stay unchanged. With `compile = true`, only the orthogonalization kernel compiles.
  This option defaults to false until a device benchmark shows a benefit.

These options do not change memory chunk size, loop count, or the training budget.
The trainer reports `train_step_seconds`, `padded_chunks_per_row`, and `chunk_padding_fraction`.
Step time includes any compilation for that step, but excludes evaluation and checkpoint writes.
The loss check occurs once per optimizer step, before the optimizer can update parameters. Gradient checks remain enabled.

Use a short synthetic benchmark before a full L4 run:

```sh
TORCH_LOGS=recompiles,graph_breaks nix develop -c python -m kestrel.benchmark \
  --config configs/hnet_titans.toml --device cuda --compile fixed \
  --chunk-counts 256,512,768,512,256 --steps-per-shape 3
```

Run this in an environment with a CUDA-enabled PyTorch build. The benchmark prints each step as JSON.
It forces router lengths outside the compiled encoder. It retains the configured batch size and gradient accumulation.
The first visit includes compilation costs; later steps and repeat visits show whether those costs recur.
It uses synthetic data and boundaries, so its loss does not measure model quality.
It does not load a checkpoint, save model weights, or access a dataset.

Compare with `--compile off`, then test `--attention-backend flash2`, `--batched-muon`, and `--smoothing dense` separately.
Compare `--batch-size 8 --accumulation 4` with `--batch-size 16 --accumulation 2` to keep the effective batch size fixed.
`--trace /tmp/hnet-trace.json` writes a profiler trace with forward/backward and optimizer regions; profiling adds overhead.
The default time budget is 600 seconds, checked between steps. It cannot interrupt a compiler call already in progress.

For the full training run, set the chosen options in `[train]`. The existing resume check still requires an identical config.
The current 25,000-step configs process 3.277 billion byte tokens. A 12-hour total permits less than 1.73 seconds per step before overhead.
Target 1.5 seconds or less to leave time for compilation, evaluation and checkpoint writes.

`compile = true` also compiles the main network per layer, with a dynamic chunk dimension.
Each Titans block compiles as two regions around its memory loop; the loop stays in Python.
On an L4 with `configs/hnet_titans.toml` at 8 x 4 batches, a step takes 1.79 s at 683 chunks per row and 5.49 s at 2048.
At 2048 chunks, peak memory is 20.8 GB. A batch of 16 x 2 runs out of memory at early router rates.
A full run thus takes about 13 hours on an L4. With `compile = true`, `metrics.jsonl` records `compiled_graphs`.
This value must stop after the first few hundred steps. A steady rise shows recompilation on most steps.

### Colab runs

`scripts/colab_hnet.sh L4` (or `A100`) starts or resumes the `hnet_titans.toml` run with the `colab` CLI (`google-colab-cli`).
It uses the committed HEAD. Data, checkpoints and a console log go to `MyDrive/kestrel` on Google Drive.
The script stops and asks you to run `colab drivemount` if Drive is not mounted, because that step needs a browser.
The first start makes byte-level data from the raw JSONL in `MyDrive/kestrel/data/raw`.
Both GPUs use the same config, so a run that Colab reclaims can resume on either GPU.
Stop the VM with `colab stop` after the run, because an idle VM still uses compute units.

## Sample

```sh
python -m kestrel sample --run runs/first --prompt 'Hello! How are you?' --max-new-tokens 64
```

Sampling uses CPU by default. Use `--temperature 0` for greedy output.
The command formats one user turn and generates one assistant reply.
The initial version has no interactive chat loop.
A functional smoke test does not establish conversational ability.

## Playground

```sh
python -m kestrel playground --device cpu   # then open http://localhost:8000
```

The playground uses suzunaan, a separate flake input.
It lists every run in `runs/` and the `step_*.pt` snapshots of each run.
It can write text, chat, compare two models with one seed, and show token probabilities.
It also shows attention, the logit lens, a head map and neuron activations, and it can switch off heads.
A looped Titans model counts each run of a block as a separate layer.

## Initial checks

The tests cover causal attention, tied parameters, optimizer groups and WSD boundaries.
They also cover Unicode tokenization, chat formatting, a short train run, generation and exact CPU resume.
These checks use synthetic data and a tiny model. They do not start a research run.
