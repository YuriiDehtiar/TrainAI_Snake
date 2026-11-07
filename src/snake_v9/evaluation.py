"""Deterministic, bounded evaluations scored by apples rather than survival."""

from collections import Counter
from pathlib import Path
from statistics import mean, median

import torch
from ray.rllib.core.rl_module.rl_module import RLModule

from .env import SnakeEnv


def load_policy(checkpoint):
    root = Path(checkpoint).resolve()
    if root.is_file():
        root = root.parent
    if (root / 'model.json').is_file():
        import hashlib
        import json
        from .config import make_config
        metadata = json.loads((root / 'model.json').read_text(encoding='utf-8'))
        if metadata.get('format') != 'snakeai-state-dict-v1':
            raise ValueError('Unsupported portable model format.')
        weights = root / 'policy.pt'
        if hashlib.sha256(weights.read_bytes()).hexdigest() != metadata['weight_sha256']:
            raise ValueError('Model weight checksum does not match model.json.')
        env = SnakeEnv(metadata['env_config'])
        try:
            module = make_config(env_config=metadata['env_config'], workers=0, gpu=0).get_rl_module_spec(
                spaces={'default_policy': (env.observation_space, env.action_space)},
                inference_only=False,
            ).build()
        finally:
            env.close()
        module.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
        module.cpu().eval()
        return module
    candidates = [root / "learner_group" / "learner" / "rl_module" / "default_policy",
                  root / "env_runner" / "module" / "default_policy", root]
    policy_path = next((p for p in candidates if (p / "module_state.pt").is_file()), None)
    if policy_path is None:
        raise FileNotFoundError(f"No policy checkpoint found in {root}")
    module = RLModule.from_checkpoint(str(policy_path))
    if tuple(module.observation_space.shape)[-1] != SnakeEnv.CHANNELS or module.action_space.n != 3:
        raise ValueError("This is not a v9 policy. The original v8 model remains separate.")
    module.eval()
    return module


def choose_action(module, obs, *, stochastic=False):
    device = next(module.parameters()).device
    with torch.inference_mode():
        batch = torch.as_tensor(obs[None], dtype=torch.float32, device=device)
        logits = module.forward_inference({"obs": batch})["action_dist_inputs"][0]
        if stochastic:
            return int(torch.distributions.Categorical(logits=logits).sample().item())
        return int(logits.argmax().item())


def evaluate(module, env_config, *, episodes=30, seed=10000):
    if episodes < 1:
        raise ValueError("episodes must be positive")
    env = SnakeEnv(env_config)
    rows = []
    try:
        for index in range(episodes):
            obs, info = env.reset(seed=seed + index)
            total_reward = 0.0
            while True:
                obs, reward, terminated, truncated, info = env.step(choose_action(module, obs))
                total_reward += reward
                if terminated or truncated:
                    break
            rows.append({**info, "seed": seed + index, "reward": round(total_reward, 6)})
    finally:
        env.close()
    return {
        "episodes": episodes, "seed": seed,
        "apples_mean": mean(r["apples"] for r in rows),
        "apples_median": median(r["apples"] for r in rows),
        "apples_max": max(r["apples"] for r in rows),
        "steps_mean": mean(r["steps"] for r in rows),
        "reward_mean": mean(r["reward"] for r in rows),
        "zero_apple_rate": mean(r["apples"] == 0 for r in rows),
        "early_death_rate": mean(r["steps"] <= 20 and not r["won"] for r in rows),
        "win_rate": mean(r["won"] for r in rows),
        "end_reasons": dict(Counter(r["reason"] for r in rows)),
        "rows": rows,
    }
