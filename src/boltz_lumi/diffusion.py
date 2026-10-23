"""Reuse fixed diffusion conditioning within a prediction."""

from contextlib import nullcontext
from functools import lru_cache
import torch
from boltz.model.modules.diffusionv2 import AtomDiffusion
from boltz.model.modules.transformersv2 import AdaLN, ConditionedTransitionBlock
from .kernels._build import extension
from .atom_conditioning import atom_conditioning_cache
from .pair_bias import prepare_token_bias


def optimized_sample(self, *args, **kwargs):
    if torch.is_grad_enabled() or self.training:
        return AtomDiffusion.sample(self, *args, **kwargs)
    conditioning = dict(kwargs["diffusion_conditioning"])
    conditioning["token_trans_bias"] = prepare_token_bias(conditioning["token_trans_bias"])
    kwargs["diffusion_conditioning"] = conditioning
    context = nullcontext()
    if kwargs.get("multiplicity", 1) == 1 and conditioning["c"].shape[0] == 1:
        context = atom_conditioning_cache(self.score_model)
    with context:
        return AtomDiffusion.sample(self, *args, **kwargs)


# Native epilogue operation codes.
MODULATION, GATE, GATED_SWIGLU = 0, 1, 3


@lru_cache(maxsize=None)
def supported_device(index):
    return bool(torch.version.hip) and torch.cuda.get_device_properties(
        index
    ).gcnArchName.startswith("gfx90a")


def eligible(module, *tensors):
    return (
        not module.training
        and not torch.is_grad_enabled()
        and not torch.is_autocast_enabled("cuda")
        and all(
            x.is_cuda
            and x.dtype == torch.float32
            and x.is_contiguous()
            and supported_device(x.device.index)
            for x in tensors
        )
    )


def adaln_forward(self, a, s):
    if not eligible(self, a, s):
        return AdaLN.forward(self, a, s)
    a = self.a_norm(a)
    s = self.s_norm(s)
    return extension("diffusion").epilogue(a, self.s_scale(s), self.s_bias(s), MODULATION)


def transition_forward(self, a, s):
    if not eligible(self, a, s):
        return ConditionedTransitionBlock.forward(self, a, s)
    a = self.adaln(a, s)
    projected = self.swish_gate[0](a)
    third = self.a_to_b(a)
    b = extension("diffusion").epilogue(projected, third, None, GATED_SWIGLU)
    gate = self.output_projection[0](s)
    return extension("diffusion").epilogue(gate, self.b_to_a(b), None, GATE)
