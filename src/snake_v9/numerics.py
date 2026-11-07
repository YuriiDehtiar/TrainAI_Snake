"""Categorical KL in log space, without treating softmax underflow as zero support."""

import torch
from ray.rllib.models.torch.torch_distributions import TorchCategorical


class LogSpaceCategorical(TorchCategorical):
    @classmethod
    def from_logits(cls, logits, **kwargs):
        return cls(logits=logits, **kwargs)

    def kl(self, other):
        # PyTorch 2.5's categorical KL overwrites a term with infinity when
        # q.probs underflows to zero, even though its finite q.logits still
        # represent a nonzero probability. Use those normalized log values.
        log_p, log_q = self._dist.logits, other._dist.logits
        probabilities = self._dist.probs
        difference = torch.where(probabilities > 0, log_p - log_q, 0.0)
        return (probabilities * difference).sum(dim=-1)


def require_finite_weights(module):
    checks = [torch.isfinite(value).all() for value in module.state_dict().values()
              if value.is_floating_point() or value.is_complex()]
    if checks and not torch.stack(checks).all().item():
        raise RuntimeError('Model weights contain NaN or infinity; snapshot was not published.')
