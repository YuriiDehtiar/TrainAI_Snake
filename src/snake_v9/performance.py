"""Optional GPU throughput settings, scoped to the current Python process."""

import os

import torch


def configure_gpu(fast=False):
    # RLlib 2.40 sets this to the invalid value '4096:8' while applying seeds.
    # Apply this after build/restore, before the first CUDA matrix multiplication.
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") == "4096:8":
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG")
    torch.backends.cuda.matmul.allow_tf32 = bool(fast)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = bool(fast)
    torch.backends.cudnn.deterministic = not fast
    torch.use_deterministic_algorithms(False)
    return {
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_convolution": torch.backends.cudnn.allow_tf32,
        "cudnn_autotune": torch.backends.cudnn.benchmark,
        "deterministic_convolution": torch.backends.cudnn.deterministic,
    }
