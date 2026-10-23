"""Reuse immutable atom-conditioning projections within one sampling call.

AtomAttentionEncoder changes coordinates but passes conditioning unchanged.
Token conditioning varies with the timestep and is deliberately excluded.
"""

from contextlib import contextmanager
from functools import partial
import torch


def cached_forward(module, original, cache, x):
    if (
        module.training
        or torch.is_grad_enabled()
        or torch.is_autocast_enabled("cuda")
        or not x.is_cuda
        or x.dtype != torch.float32
    ):
        return original(x)
    key = (tuple(x.shape), tuple(x.stride()), x.dtype, x.device)
    if key not in cache:
        cache[key] = original(x)
    return cache[key]


@contextmanager
def atom_conditioning_cache(score_model):
    encoder = score_model.atom_attention_encoder.atom_encoder.diffusion_transformer
    decoder = score_model.atom_attention_decoder.atom_decoder.diffusion_transformer
    targets = set()
    for layer in (*encoder.layers, *decoder.layers):
        for adaln in (layer.adaln, layer.transition.adaln):
            targets.update((adaln.s_norm, adaln.s_scale, adaln.s_bias))
        targets.update((layer.output_projection_linear, layer.transition.output_projection[0]))
    originals = []
    try:
        for module in targets:
            original = module.forward
            originals.append((module, original))
            module.forward = partial(cached_forward, module, original, {})
        yield
    finally:
        for module, original in originals:
            module.forward = original
