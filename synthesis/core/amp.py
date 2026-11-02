"""Autocast and gradient-scaler helpers.

The paper runs used ``torch.autocast('cuda', dtype=...)`` and ``torch.amp.GradScaler('cuda', init_scale=...)`` on CUDA.
Here the device type comes from the device and ``dtype=None`` (or 'float32') disables autocast, which is how the CPU
equivalence tests run both the original and the ported code in fp32.
"""
import torch

from .config import torch_dtype


def autocast(device, dtype):
    """dtype: torch dtype, a name ('float16', 'bfloat16', 'float32') or None."""
    if not isinstance(dtype, torch.dtype):
        dtype = torch_dtype(dtype)
    device_type = torch.device(device).type
    if dtype is None:
        return torch.autocast(device_type, enabled=False)
    return torch.autocast(device_type, dtype=dtype)


def grad_scaler(device, init_scale):
    """CUDA: enabled with ``init_scale`` (also under bf16 autocast, as in the paper runs). CPU: disabled (scale 1)."""
    device_type = torch.device(device).type
    return torch.amp.GradScaler(device_type, init_scale=float(init_scale), enabled=device_type == 'cuda')
