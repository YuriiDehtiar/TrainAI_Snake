import unittest

import torch
from ray.rllib.models.torch.torch_distributions import TorchCategorical

from snake_v9.numerics import LogSpaceCategorical, require_finite_weights


class NumericsTests(unittest.TestCase):
    def test_underflow_reproducer_has_finite_kl_and_gradient(self):
        devices = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])
        for device in devices:
            with self.subTest(device=device):
                old_logits = torch.tensor([[0., -80., -2.]], device=device)
                logits = torch.tensor([[-200., 0., -200.]], device=device, requires_grad=True)
                standard = TorchCategorical.from_logits(old_logits).kl(TorchCategorical.from_logits(logits))
                self.assertTrue(torch.isinf(standard).all())
                stable = LogSpaceCategorical.from_logits(old_logits).kl(LogSpaceCategorical.from_logits(logits))
                reference = TorchCategorical.from_logits(old_logits.double()).kl(TorchCategorical.from_logits(logits.double()))
                self.assertTrue(torch.allclose(stable.double(), reference, atol=2e-5, rtol=1e-6))
                stable.mean().backward()
                self.assertTrue(torch.isfinite(logits.grad).all())

    def test_normal_kl_and_log_prob_are_unchanged(self):
        old_logits = torch.tensor([[.2, -.6, .9], [-1., 2., .3]])
        logits = torch.tensor([[.4, -.5, .8], [-.9, 1.8, .4]], requires_grad=True)
        stable = LogSpaceCategorical.from_logits(old_logits)
        standard = TorchCategorical.from_logits(old_logits)
        self.assertIsInstance(stable, LogSpaceCategorical)
        self.assertTrue(torch.allclose(stable.kl(LogSpaceCategorical.from_logits(logits)),
                                       standard.kl(TorchCategorical.from_logits(logits))))
        self.assertTrue(torch.equal(stable.logp(torch.tensor([0, 2])), standard.logp(torch.tensor([0, 2]))))

    def test_nonfinite_weights_are_rejected(self):
        module = torch.nn.Linear(2, 1)
        require_finite_weights(module)
        with torch.no_grad():
            module.weight[0, 0] = float('nan')
        with self.assertRaisesRegex(RuntimeError, 'not published'):
            require_finite_weights(module)


if __name__ == '__main__':
    unittest.main()
