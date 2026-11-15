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
    generation.add_argument("--repetition-penalty", type=float, default=1.0)
    generation.add_argument("--raw", action="store_true", help="Sample from raw prompt without chat template")
    generation.add_argument("--seed", type=int, default=42)
    interactive = sub.add_parser("chat", help="Interactive multi-turn chat loop")
    interactive.add_argument("--run", required=True)
    interactive.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    interactive.add_argument("--max-new-tokens", type=int, default=256)
    interactive.add_argument("--temperature", type=float, default=0.7)
    interactive.add_argument("--top-k", type=int, default=50)
    interactive.add_argument("--top-p", type=float, default=0.9)
    interactive.add_argument("--repetition-penalty", type=float, default=1.1)
    interactive.add_argument("--system")
    interactive.add_argument("--seed", type=int, default=42)
    rl = sub.add_parser("grpo", help="GRPO on step-by-step arithmetic from a trained run")
    rl.add_argument("--init-from", help="Run directory to start from (not needed with --resume)")
    rl.add_argument("--output", required=True)
    rl.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    rl.add_argument("--resume", action="store_true")
    rl.add_argument("--steps", type=int, default=200)
    rl.add_argument("--prompts", type=int, default=16, help="Problems per step")
    rl.add_argument("--group", type=int, default=8, help="Samples per problem")
    rl.add_argument("--temperature", type=float, default=1.0)
    rl.add_argument("--max-new-tokens", type=int, default=80)
    rl.add_argument("--lr", type=float, default=1e-5)
    rl.add_argument("--warmup", type=int, default=10)
    rl.add_argument("--clip", type=float, default=1.0)
    rl.add_argument("--max-rows", type=int, default=128, help="Rows per generate batch")
    rl.add_argument("--micro-rows", type=int, default=16, help="Rows per backward pass")
    rl.add_argument("--eval-every", type=int, default=25)
    rl.add_argument("--eval-problems", type=int, default=16, help="Eval problems per bucket")
    rl.add_argument("--eval-group", type=int, default=4, help="Samples per eval problem for sampled pass@1")
    rl.add_argument("--save-every", type=int, default=25)
    rl.add_argument("--seen-from", help="Pretraining JSONL; keep its problems out of the eval set")
    rl.add_argument("--weights", help='Bucket weights as JSON, e.g. \'{"sub_borrow": 1}\'')
    rl.add_argument("--seed", type=int, default=0)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "grpo":
        from .grpo import DEFAULT_WEIGHTS, grpo

        positive = ("steps", "prompts", "group", "max_new_tokens", "max_rows", "micro_rows",
                    "eval_every", "eval_problems", "eval_group", "save_every")
        if any(args[k] < 1 for k in positive) or args["lr"] <= 0 or args["clip"] <= 0 or args["temperature"] <= 0:
            parser.error("GRPO sizes, lr, clip and temperature must be positive")
        if args["group"] < 2:
            parser.error("GRPO needs --group of at least 2")
        if not args["resume"] and not args["init_from"]:
            parser.error("--init-from is required unless --resume is given")
        weights = json.loads(args["weights"]) if args["weights"] else DEFAULT_WEIGHTS
        unknown = set(weights) - set(DEFAULT_WEIGHTS)
        if unknown or not weights or min(weights.values()) < 0 or sum(weights.values()) <= 0:
            parser.error(f"Weights need known buckets and nonnegative values: {sorted(unknown)}")
        settings = {k: args[k] for k in (*positive, "temperature", "lr", "warmup", "clip", "seen_from", "seed")}
        settings["weights"] = weights
        grpo(args["init_from"], args["output"], settings, args["device"], args["resume"])
        return
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
               args["temperature"], args["seed"], args["top_k"], args["top_p"], args["raw"],
               args["repetition_penalty"])
    else:
        chat(args["run"], args["device"], args["max_new_tokens"], args["temperature"],
             args["seed"], args["top_k"], args["top_p"], args["system"], args["repetition_penalty"])


if __name__ == "__main__":
    main()
