"""Evaluate v9 checkpoints on reproducible test episodes."""

import argparse
import json
from pathlib import Path

import torch

from snake_v9.checkpoints import checkpoint_settings, resolve_checkpoint
from snake_v9.evaluation import evaluate, load_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=30000)
    parser.add_argument("--initial-length", type=int, help="Evaluate on a harder starting length, e.g. 21.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        checkpoint = resolve_checkpoint(args.checkpoint)
        settings = checkpoint_settings(checkpoint)
        if args.initial_length is not None:
            settings["initial_length"] = args.initial_length
        report = evaluate(load_policy(checkpoint), settings, episodes=args.episodes, seed=args.seed)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
