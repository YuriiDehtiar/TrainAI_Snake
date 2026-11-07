"""Watch a trained v9 checkpoint without starting training."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import torch

from snake_v9.env import SnakeEnv
from snake_v9.evaluation import choose_action, load_policy
from snake_v9.checkpoints import checkpoint_settings, resolve_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--episodes", type=int, default=0, help="0 repeats until the window closes.")
    parser.add_argument("--seed", type=int, default=20000)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions rather than use argmax.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    try:
        checkpoint = resolve_checkpoint(args.checkpoint)
        settings = checkpoint_settings(checkpoint)
        module = load_policy(checkpoint)
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    env = SnakeEnv(settings, render_mode="human")
    print(f"Watching: {checkpoint}", flush=True)
    episode = 0
    try:
        while env.window_is_open and (args.episodes == 0 or episode < args.episodes):
            obs, info = env.reset(seed=args.seed + episode)
            episode += 1
            env.render()
            while env.window_is_open:
                obs, reward, terminated, truncated, info = env.step(choose_action(module, obs, stochastic=args.stochastic))
                env.render()
                if terminated or truncated:
                    print(f"Episode {episode}: {info}", flush=True)
                    if env.window_is_open:
                        plt.pause(1)
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
