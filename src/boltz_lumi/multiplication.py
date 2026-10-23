"""Triangle multiplication with fused normalization, gates and input layouts."""

from functools import lru_cache
import json
from pathlib import Path
import torch
from .kernels._build import extension


@lru_cache(maxsize=1)
def measured_configs():
    # Numeric keys are token counts for the measured gfx90a configurations.
    path = Path(__file__).with_name("kernels") / "gfx90a_multiplication_config.json"
    return json.loads(path.read_text())["choices"]


def joint_supported(module):
    return all(
        layer.bias is None
        and tuple(layer.weight.shape) == (256, 128)
        and layer.weight.dtype == torch.float32
        and layer.weight.is_cuda
        and layer.weight.is_contiguous()
        and not layer.weight.is_inference()
        for layer in (module.p_in, module.g_in)
    )


def prepare(module):
    """Reuse only while parameter identity, version, dtype and device agree."""
    assert joint_supported(module)
    weights = (module.p_in.weight, module.g_in.weight)
    key = tuple((id(w), w._version, w.device, w.dtype) for w in weights)
    cached = getattr(module, "_boltz_lumi_joint_projection", None)
    if cached is None or cached[0] != key:
        with torch.no_grad():
            weight = torch.cat(weights, dim=0).to(torch.bfloat16)
        cached = (key, weight)
        module._boltz_lumi_joint_projection = cached
    return cached[1]


def project(module, normalized):
    output = torch.nn.functional.linear(normalized, prepare(module))
    return output[..., :256], output[..., 256:]


def normalized_forward(module, x, mask, use_kernels=False):
    supported = (
        not use_kernels
        and not module.training
        and not torch.is_grad_enabled()
        and x.is_cuda
        and torch.version.hip
        and x.ndim == 4
        and x.shape[0] == 1
        and x.shape[1] == x.shape[2]
        and x.shape[-1] == 128
        and x.dtype == torch.float32
        and x.is_contiguous()
        and torch.is_autocast_enabled("cuda")
        and torch.get_autocast_dtype("cuda") == torch.bfloat16
        and torch.cuda.get_device_properties(x.device).gcnArchName.startswith("gfx90a")
        and mask.dtype == torch.float32
        and mask.is_contiguous()
        and mask.device == x.device
        and mask.shape == x.shape[:-1]
        and all(
            v.dtype == torch.float32 and v.is_contiguous() and v.device == x.device
            for v in (
                module.norm_in.weight,
                module.norm_in.bias,
                module.norm_out.weight,
                module.norm_out.bias,
            )
        )
    )
    if not supported:
        return type(module).forward(module, x, mask, use_kernels=use_kernels)
    n = x.shape[1]
    config = measured_configs().get(str(n), {})
    norm_in = config.get("norm_in", {"waves": 4, "rows": 16})
    norm_out = config.get("norm_out", {"waves": 4, "rows": 16})
    gate = config.get("gate", {})
    native = extension("multiplication")
    # At 117 tokens FP32 output was faster; other sizes fuse the following cast.
    bf16_output = n != 117
    normalized = native.norm(
        x.reshape(-1, 128),
        module.norm_in.weight,
        module.norm_in.bias,
        module.norm_in.eps,
        False,
        norm_in["waves"],
        norm_in["rows"],
        bf16_output,
    ).reshape_as(x)
    if n in (117, 312) and joint_supported(module):
        projection, gates = project(module, normalized)
    else:
        projection = module.p_in(normalized)
        gates = module.g_in(normalized)
    gate_native = extension("multiplication_gate")
    outgoing = type(module).__name__ == "TriangleMultiplicationOutgoing"
    if n < 576:
        left, right = gate_native.pack(projection, gates, mask, outgoing, gate.get("channels", 16))
    else:
        from .kernels.layout_copy import configured_copy

        a, b = gate_native.split(projection, gates, mask)
        if outgoing:
            left = a.permute(0, 3, 1, 2)
            right = b.permute(0, 3, 2, 1)
        else:
            left = a.permute(0, 3, 2, 1)
            right = b.permute(0, 3, 1, 2)
        left = configured_copy(left.reshape(128, n, n))
        right = configured_copy(right.reshape(128, n, n))
        del a, b
    # These intermediates are large; release them before the matrix product.
    del projection, gates
    product = torch.bmm(left, right)
    del left, right
    normed = native.norm(
        product.reshape(128, n * n),
        module.norm_out.weight,
        module.norm_out.bias,
        module.norm_out.eps,
        True,
        norm_out["waves"],
        norm_out["rows"],
        bf16_output,
    ).reshape(1, n, n, 128)
    del product
    projection = module.p_out(normed)
    gates = module.g_out(normalized)
    return gate_native.gate(projection, gates)
