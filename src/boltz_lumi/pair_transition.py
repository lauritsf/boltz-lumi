"""Fuse the BF16 activation/product in Pairformer transitions."""

import torch
from boltz.model.layers.transition import Transition
from .kernels._build import extension


def transition_forward(module, x, chunk_size=None):
    if (
        module.training
        or torch.is_grad_enabled()
        or chunk_size is not None
        or not x.is_cuda
        or not torch.version.hip
        or not torch.is_autocast_enabled("cuda")
        or torch.get_autocast_dtype("cuda") != torch.bfloat16
        or not torch.cuda.get_device_properties(x.device).gcnArchName.startswith("gfx90a")
    ):
        return Transition.forward(module, x, chunk_size=chunk_size)
    normalized = module.norm(x)
    a = module.fc1(normalized)
    b = module.fc2(normalized)
    if a.dtype == b.dtype == torch.bfloat16 and a.is_contiguous() and b.is_contiguous():
        value = extension("pair").silu_mul(a, b)
    else:
        value = module.silu(a) * b
    del normalized, a, b
    return module.fc3(value)
