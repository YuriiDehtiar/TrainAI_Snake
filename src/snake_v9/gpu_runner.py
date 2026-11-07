"""CUDA policy sampling for the pinned RLlib 2.40 single-agent runner."""

import torch
from ray.rllib.connectors.common.numpy_to_tensor import NumpyToTensor
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner

from .performance import configure_gpu


class GPUSnakeEnvRunner(SingleAgentEnvRunner):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA sampling requires an assigned, visible GPU.")
        configure_gpu(True)
        # Sampling can have variable batch sizes around resets. Avoid autotuning
        # every one of those sizes in remote sampler processes.
        torch.backends.cudnn.benchmark = False
        self.module.to(torch.device("cuda:0"))
        converted = False
        for connector in self._env_to_module.connectors:
            if isinstance(connector, NumpyToTensor):
                connector._device = torch.device("cuda:0")
                converted = True
        if not converted:
            raise RuntimeError("The expected RLlib 2.40 tensor connector is missing.")
        print(f"CUDA sampler ready: {self.num_envs} games, device={next(self.module.parameters()).device}", flush=True)
