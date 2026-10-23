"""Triangle attention with fused normalization and ordered bias/softmax."""

from functools import lru_cache
import json
from pathlib import Path
import torch
from boltz.model.layers.triangular_attention.attention import TriangleAttention
from boltz.model.layers.triangular_attention.primitives import Attention
from boltz.data import const
from .kernels._build import extension
from .multiplication import measured_configs
from .kernels.triangle_softmax import ordered_softmax


UPSTREAM_CHUNK = TriangleAttention._chunk


def contiguous_mask_chunk(self, x, tri_bias, mask_bias, mask, chunk_size, use_kernels=False):
    # Small cases are dispatch-bound; the extra copy has no measured benefit.
    if x.shape[-3] > const.chunk_size_threshold:
        mask_bias = mask_bias.contiguous()
    return UPSTREAM_CHUNK(self, x, tri_bias, mask_bias, mask, chunk_size, use_kernels)


@lru_cache(maxsize=1)
def large_attention_configs():
    # Numeric keys are token counts for the measured gfx90a configurations.
    path = Path(__file__).with_name("kernels") / "gfx90a_large_attention_config.json"
    return json.loads(path.read_text())["choices"]


def ordered_triangle_forward(self, q_x, kv_x, tri_bias, mask_bias, mask, use_kernels=False):
    n = q_x.shape[-2]
    shared_bias = tri_bias.ndim == 5 or tri_bias.shape[0] == 1 or tri_bias.stride(0) == 0
    supported = (
        not use_kernels
        and not torch.is_grad_enabled()
        and not self.training
        and n >= 312
        and q_x.is_cuda
        and torch.version.hip
        and torch.cuda.get_device_properties(q_x.device).gcnArchName.startswith("gfx90a")
        and self.c_q == 128
        and self.c_hidden == 32
        and self.no_heads == 4
        and shared_bias
        and q_x.shape == kv_x.shape
        and q_x.dtype in (torch.float32, torch.bfloat16)
        and (
            not torch.is_autocast_enabled("cuda")
            or torch.get_autocast_dtype("cuda") == torch.bfloat16
        )
    )
    # Keep the measured launch choices for larger shapes; others fall back.
    choice = large_attention_configs().get(str(n)) if n > 1024 else {}
    if n > 1024:
        supported = (
            supported
            and choice is not None
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.bfloat16
            and tri_bias.dtype == torch.bfloat16
            and mask_bias.dtype == torch.float32
        )
    native = choice is not None and choice.get("implementation") == "native-block-softmax"
    if native:
        supported = (
            supported
            and n == 2511
            and torch.version.git_version == "70d99e998b4955e0049d13a98d77ae1b14db1f45"
        )
    else:
        supported = (
            supported and n <= 2048 and q_x.numel() // q_x.shape[-1] * self.no_heads * n <= 2**31
        )
    if not supported:
        return Attention.forward(
            self, q_x, kv_x, tri_bias, mask_bias, mask, use_kernels=use_kernels
        )
    q, k, v = self._prep_qkv(q_x, kv_x, apply_scale=True)
    flattened = q.ndim == 4
    if flattened:
        # chunk_layer flattens the single batch and outer triangle-row axes.
        q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
        tri_bias = tri_bias[:1].unsqueeze(1)
        mask_bias = mask_bias.unsqueeze(0)
    scores = torch.matmul(q, k.transpose(-1, -2))
    if native:
        probabilities = extension("triangle_block_softmax").softmax(scores, tri_bias, mask_bias)
    else:
        config = choice.get("config", {})
        rows = config.get("rows", 4 if mask_bias.stride(-1) != 1 else 1)
        probabilities = ordered_softmax(
            scores,
            tri_bias,
            mask_bias,
            rows=rows,
            waves_per_eu=config.get("waves_per_eu", 0),
            allow_large=n > 1024,
        )
    output = torch.matmul(probabilities, v).transpose(-2, -3)
    if flattened:
        output = output.squeeze(0)
    return self._wrap_up(output, q_x)


def normalization_forward(module, x):
    eligible = (
        not module.training
        and not torch.is_grad_enabled()
        and x.is_cuda
        and torch.version.hip
        and x.dtype == torch.float32
        and x.ndim == 4
        and x.shape[0] == 1
        and x.shape[1] == x.shape[2]
        and x.shape[-1] == 128
        and x.shape[1] in (312, 576)
        and getattr(module, "c_in", None) == (128,)
        and torch.cuda.get_device_properties(x.device).gcnArchName.startswith("gfx90a")
        and all(
            v.dtype == torch.float32 and v.is_contiguous() and v.device == x.device
            for v in (module.weight, module.bias)
        )
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
    )
    if not eligible:
        return type(module).forward(module, x)
    config = measured_configs()[str(x.shape[1])]["norm_in"]
    # Ending-node attention transposes the pair axes; LayerNorm also copies it.
    flat = x.contiguous().reshape(-1, 128)
    return (
        extension("multiplication")
        .norm(
            flat,
            module.weight,
            module.bias,
            module.eps,
            False,
            config["waves"],
            config["rows"],
            True,
        )
        .reshape(x.shape)
    )
