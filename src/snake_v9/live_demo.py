"""Continuous CPU demonstration; refresh weights only when a new game begins."""

import json
from pathlib import Path

import torch

from .env import SnakeEnv
from .evaluation import choose_action, load_policy


class LiveGame:
    def __init__(self, run_dir, *, loader=load_policy, action_fn=choose_action):
        self.run_dir = Path(run_dir)
        config = json.loads((self.run_dir / 'run.json').read_text(encoding='utf-8-sig'))
        self.env = SnakeEnv(config['env_config'])
        self.loader, self.action_fn = loader, action_fn
        self.module = None
        self.loaded_iteration = -1
        self.latest_iteration = -1
        self.games_started = self.games_finished = self.wins = 0
        self.total_reward = 0.0
        self.last_result = None
        self.observation = None
        self.active = False
        self.error = None

    def pointer(self):
        pointer = json.loads((self.run_dir / 'live/latest.json').read_text(encoding='utf-8'))
        self.latest_iteration = int(pointer['iteration'])
        return pointer

    def new_game(self):
        try:
            pointer = self.pointer()
            if self.module is None:
                self.module = self.loader(pointer['bootstrap_checkpoint'])
                self.module.to('cpu')
            if self.loaded_iteration != pointer['iteration']:
                payload = torch.load(pointer['snapshot'], map_location='cpu', weights_only=True)
                if payload['iteration'] != pointer['iteration']:
                    raise ValueError('Inference snapshot does not match its pointer.')
                self.module.load_state_dict(payload['state_dict'], strict=True)
                self.loaded_iteration = int(payload['iteration'])
            self.module.eval()
            self.error = None
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            self.error = f'{type(error).__name__}: {error}'
            if self.loaded_iteration < 0:
                return False
        self.games_started += 1
        self.seed = 500000 + self.games_started
        self.observation, _ = self.env.reset(seed=self.seed)
        self.total_reward = 0.0
        self.active = True
        return True

    def step(self):
        if not self.active:
            return None
        action = self.action_fn(self.module, self.observation)
        self.observation, reward, terminated, truncated, info = self.env.step(action)
        self.total_reward += reward
        if terminated or truncated:
            self.active = False
            self.games_finished += 1
            self.wins += int(info['won'])
            self.last_result = {**info, 'seed': self.seed, 'game': self.games_finished,
                                'reward': self.total_reward,
                                'weights_iteration': self.loaded_iteration}
            with (self.run_dir / 'demo_games.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(self.last_result) + '\n')
        return info
