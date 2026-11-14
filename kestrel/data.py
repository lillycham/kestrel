"""Local byte-level BPE and explicit train/validation token streams."""

import hashlib
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

SPECIAL = ["<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|' + message['role'] + '|>\\n' + message['content'] + '<|end|>\\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|assistant|>\\n' }}{% else %}{{ '<|eos|>' }}{% endif %}"
)


def rows(path):
    with Path(path).open(encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("Expected an object")
                if set(row) == {"text"} and isinstance(row["text"], str):
                    contents = [row["text"]]
                elif set(row) == {"messages"} and isinstance(row["messages"], list) and row["messages"]:
                    contents = []
                    for message in row["messages"]:
                        if (not isinstance(message, dict) or set(message) != {"role", "content"}
                                or message["role"] not in {"system", "user", "assistant"}
                                or not isinstance(message["content"], str)):
                            raise ValueError("Each message needs a supported role and string content")
                        contents.append(message["content"])
                else:
                    raise ValueError("Expected either text or nonempty messages")
                if any(token in content for content in contents for token in SPECIAL):
                    raise ValueError("Content contains a reserved control token")
                yield row
            except (ValueError, TypeError) as error:
                raise ValueError(f"{path}:{number}: {error}") from error


def texts(path):
    for row in rows(path):
        if "text" in row:
            yield row["text"]
        else:
            yield from (message["content"] for message in row["messages"])


def train_tokenizer(source, destination, vocab_size):
    if vocab_size < 256 + len(SPECIAL):
        raise ValueError("Vocabulary must fit the byte alphabet and control tokens")
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(texts(source), trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=SPECIAL,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False,
    ))
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, pad_token=SPECIAL[0], bos_token=SPECIAL[1],
        eos_token=SPECIAL[2], additional_special_tokens=SPECIAL[3:],
        chat_template=CHAT_TEMPLATE,
    )
    destination.mkdir(parents=True)
    fast.save_pretrained(destination)
    print(json.dumps({"tokenizer": str(destination), "vocab_size": len(fast)}))


def load_tokenizer(path):
    return PreTrainedTokenizerFast.from_pretrained(path, local_files_only=True)


def rendered(row, tokenizer):
    if "messages" in row:
        return tokenizer.apply_chat_template(row["messages"], tokenize=False)
    return row["text"] + tokenizer.eos_token


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def prepare(train, validation, tokenizer_path, destination):
    if Path(train).resolve() == Path(validation).resolve() or digest(train) == digest(validation):
        raise ValueError("Train and validation must be separate, different files")
    destination = Path(destination)
    if destination.exists():
        raise ValueError(f"Destination already exists: {destination}")
    tokenizer = load_tokenizer(tokenizer_path)
    destination.mkdir(parents=True)
    tokenizer.save_pretrained(destination / "tokenizer")
    metadata = {"vocab_size": len(tokenizer), "splits": {}}
    for split, path in (("train", train), ("validation", validation)):
        # Stream to disk: no corpus-sized Python token list or RAM allocation.
        raw = destination / f"{split}.bin"
        count = 0
        with raw.open("wb") as output:
            for row in rows(path):
                ids = tokenizer.encode(rendered(row, tokenizer), add_special_tokens=False)
                np.asarray(ids, dtype="<u4").tofile(output)
                count += len(ids)
        if not count:
            raise ValueError(f"Empty {split} split")
        metadata["splits"][split] = {
            "tokens": count, "source_sha256": digest(path), "sha256": digest(raw),
        }
    metadata["tokenizer_sha256"] = digest(destination / "tokenizer" / "tokenizer.json")
    (destination / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata))


def load_data(path, length):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if digest(path / "tokenizer" / "tokenizer.json") != manifest["tokenizer_sha256"]:
        raise ValueError("Tokenizer differs from data manifest")
    arrays = []
    for split in ("train", "validation"):
        raw = path / f"{split}.bin"
        entry = manifest["splits"][split]
        if digest(raw) != entry["sha256"] or raw.stat().st_size != entry["tokens"] * 4:
            raise ValueError(f"{split} differs from data manifest")
        array = np.memmap(raw, dtype="<u4", mode="r")
        if len(array) < length + 1:
            raise ValueError(f"{split} needs at least sequence_length + 1 tokens")
        arrays.append(array)
    return manifest, arrays
