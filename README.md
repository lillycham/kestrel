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
