"""Check that the distributed artifacts load and play reproducibly."""

import json
from pathlib import Path
import unittest

import torch

from snake_v9.checkpoints import checkpoint_settings
from snake_v9.env import SnakeEnv
from snake_v9.evaluation import choose_action, load_policy
from snake_v9.paths import ROOT


class PortableModelTests(unittest.TestCase):
    def test_distributed_policies_reproduce_their_recorded_first_game(self):
        torch.set_num_threads(2)
        for iteration in (950, 2000):
            with self.subTest(iteration=iteration):
                folder = ROOT / 'models' / f'iteration_{iteration:06d}'
                expected = json.loads((ROOT / 'results/latest' / f'evaluation_{iteration:06d}.json').read_text())['rows'][0]
                module = load_policy(folder)
                env = SnakeEnv(checkpoint_settings(folder))
                try:
                    obs, _ = env.reset(seed=expected['seed'])
                    while True:
                        obs, _, terminated, truncated, info = env.step(choose_action(module, obs))
                        if terminated or truncated:
                            break
                    for key in ('apples', 'steps', 'reason', 'won'):
                        self.assertEqual(info[key], expected[key])
                finally:
                    env.close()


if __name__ == '__main__':
    unittest.main()
