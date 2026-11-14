import argparse
import json

from .core import make_model, read_config
from .data import prepare, train_tokenizer
from .train import chat, sample, train


def main():
    parser = argparse.ArgumentParser(description="Kestrel: small conversational model experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Construct a random model and count its parameters on CPU")
    inspect.add_argument("config")
    tokenizer = sub.add_parser("tokenizer", help="Train byte-level BPE on training JSONL only")
    tokenizer.add_argument("--train", required=True)
    tokenizer.add_argument("--output", required=True)
    tokenizer.add_argument("--vocab-size", type=int, default=8192)
    data = sub.add_parser("prepare", help="Encode explicit train/validation files")
    data.add_argument("--train", required=True)
    data.add_argument("--validation", required=True)
    data.add_argument("--tokenizer", required=True)
    data.add_argument("--output", required=True)
    fitting = sub.add_parser("train")
    fitting.add_argument("--config", required=True)
    fitting.add_argument("--data", required=True)
    fitting.add_argument("--output", required=True)
    fitting.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    fitting.add_argument("--resume", action="store_true")
    fitting.add_argument("--init-from", help="Initialize model weights from an existing checkpoint")
    fitting.add_argument("--stop-after", type=int, help="Absolute step to stop at without altering the WSD horizon")
    generation = sub.add_parser("sample")
    generation.add_argument("--run", required=True)
    generation.add_argument("--prompt", required=True)
    generation.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    generation.add_argument("--max-new-tokens", type=int, default=64)
    generation.add_argument("--temperature", type=float, default=0.8)
    generation.add_argument("--top-k", type=int, default=0)
    generation.add_argument("--top-p", type=float, default=1.0)
    generation.add_argument("--raw", action="store_true", help="Sample from raw prompt without chat template")
    generation.add_argument("--seed", type=int, default=42)
    interactive = sub.add_parser("chat", help="Interactive multi-turn chat loop")
    interactive.add_argument("--run", required=True)
    interactive.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    interactive.add_argument("--max-new-tokens", type=int, default=256)
    interactive.add_argument("--temperature", type=float, default=0.7)
    interactive.add_argument("--top-k", type=int, default=50)
    interactive.add_argument("--top-p", type=float, default=0.9)
    interactive.add_argument("--system")
    interactive.add_argument("--seed", type=int, default=42)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "inspect":
        model = make_model(read_config(args["config"])["model"])
        total = sum(p.numel() for p in model.parameters())
        embeddings = model.get_input_embeddings().weight.numel()
        print(json.dumps({"parameters": total, "input_embeddings": embeddings,
                          "embedding_fraction": embeddings / total}, indent=2))
    elif command == "tokenizer":
        train_tokenizer(args["train"], args["output"], args["vocab_size"])
    elif command == "prepare":
        prepare(args["train"], args["validation"], args["tokenizer"], args["output"])
    elif command == "train":
        train(args["config"], args["data"], args["output"], args["device"], args["resume"], args["stop_after"], args["init_from"])
    elif command == "sample":
        sample(args["run"], args["prompt"], args["device"], args["max_new_tokens"],
               args["temperature"], args["seed"], args["top_k"], args["top_p"], args["raw"])
    else:
        chat(args["run"], args["device"], args["max_new_tokens"], args["temperature"],
             args["seed"], args["top_k"], args["top_p"], args["system"])


if __name__ == "__main__":
    main()
