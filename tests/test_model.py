import unittest

import torch

from snake_v9.config import make_config
from snake_v9.env import SnakeEnv


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.env = SnakeEnv()
        cls.module = make_config(workers=0, gpu=0).get_rl_module_spec(
            spaces={'default_policy': (cls.env.observation_space, cls.env.action_space)},
            inference_only=False,
        ).build()

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def test_both_heads_really_have_hidden_layers(self):
        actor = [layer.out_features for layer in self.module.pi.modules() if isinstance(layer, torch.nn.Linear)]
        critic = [layer.out_features for layer in self.module.vf.modules() if isinstance(layer, torch.nn.Linear)]
        self.assertEqual(actor, [256, 128, 3])
        self.assertEqual(critic, [256, 128, 1])

    def test_cnn_reads_all_observation_channels(self):
        first_conv = next(layer for layer in self.module.encoder.modules() if isinstance(layer, torch.nn.Conv2d))
        self.assertEqual(first_conv.in_channels, 13)

    def test_policy_produces_three_finite_logits(self):
        obs, _ = self.env.reset(seed=2)
        with torch.inference_mode():
            logits = self.module.forward_inference({'obs': torch.as_tensor(obs[None])})['action_dist_inputs']
        self.assertEqual(tuple(logits.shape), (1, 3))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == '__main__':
    unittest.main()
