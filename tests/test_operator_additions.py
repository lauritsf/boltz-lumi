"""Check adopted operators with nonzero weights, masks and inference fallbacks."""

from contextlib import nullcontext
import json
from types import MethodType
from unittest.mock import patch
import torch
from boltz.model.layers.triangular_attention.attention import (
    TriangleAttention,
    TriangleAttentionEndingNode,
)
from boltz.model.layers.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)
from boltz_lumi.triangle import (
    normalization_forward,
    ordered_triangle_forward,
    contiguous_mask_chunk,
)
from boltz_lumi.multiplication import normalized_forward
from boltz_lumi.multiplication import prepare, project
from boltz_lumi.kernels._build import extension


def equal(actual, expected):
    if not torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    ):
        error = actual.double() - expected.double()
        norm = expected.double().norm().item()
        raise AssertionError(
            dict(
                max_absolute_error=error.abs().max().item(),
                relative_l2_error=error.norm().item() / norm if norm else None,
                failing_values=int((actual != expected).sum()),
                atol=0,
                rtol=0,
            )
        )


def randomize(module):
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            mean = 1 if "norm" in name and name.endswith("weight") else 0
            parameter.normal_(mean=mean, std=0.08)


def main():
    torch.manual_seed(732991)
    checks = 0
    pair_dim = 128
    head_dim = 32
    num_heads = 4
    chunk_size = 32
    full_mask_coverage_limit = 576
    num_masked_attention_tokens = 3
    num_masked_multiplication_tokens = 2
    attention_cases = (
        (17, False),
        (117, False),
        (312, True),
        (576, True),
        (1134, False),
        (2511, True),
    )
    multiplication_cases = ((7, False), (117, True), (312, True), (576, False))
    for cls in (TriangleAttention, TriangleAttentionEndingNode):
        module = cls(pair_dim, head_dim, num_heads).cuda().eval()
        randomize(module)
        norm = module.layer_norm
        original = norm.forward
        for n, expects_native in attention_cases:
            x = torch.randn(1, n, n, pair_dim, device="cuda") * 0.3
            # One partial-mask case suffices for each large dispatch branch.
            for masked in (False, True) if n <= full_mask_coverage_limit else (True,):
                mask = torch.ones(1, n, n, device="cuda")
                if masked:
                    mask[:, -num_masked_attention_tokens:, :] = 0
                    mask[:, :, -num_masked_attention_tokens:] = 0
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    expected = module(x, mask, chunk_size=chunk_size)
                    with (
                        patch.object(norm, "forward", MethodType(normalization_forward, norm)),
                        patch.object(
                            module.mha, "forward", MethodType(ordered_triangle_forward, module.mha)
                        ),
                        patch.object(module, "_chunk", MethodType(contiguous_mask_chunk, module))
                        if cls is TriangleAttentionEndingNode
                        else nullcontext(),
                        patch("boltz_lumi.triangle.extension", wraps=extension) as native,
                    ):
                        equal(module(x, mask, chunk_size=chunk_size), expected)
                        assert bool(native.call_count) == expects_native
                    checks += 1
        norm.train()
        x = torch.randn(1, 3, 3, pair_dim, device="cuda", requires_grad=True)
        actual = normalization_forward(norm, x)
        expected = original(x)
        equal(actual, expected)
        actual_grad = torch.autograd.grad(actual.sum(), x)[0]
        expected_grad = torch.autograd.grad(expected.sum(), x)[0]
        equal(actual_grad, expected_grad)
        checks += 1
        norm.eval()
        with torch.inference_mode():
            equal(
                normalization_forward(norm, x.detach().bfloat16()), original(x.detach().bfloat16())
            )
            checks += 1
    for cls in (TriangleMultiplicationIncoming, TriangleMultiplicationOutgoing):
        module = cls(pair_dim).cuda().eval()
        randomize(module)
        for n, expects_joint_projection in multiplication_cases:
            x = torch.randn(1, n, n, pair_dim, device="cuda") * 0.3
            for kind in ("zero", "ones", "partial", "fractional"):
                mask = torch.ones(1, n, n, device="cuda")
                if kind == "zero":
                    mask.zero_()
                if kind == "partial":
                    mask[:, -num_masked_multiplication_tokens:, :] = 0
                if kind == "fractional":
                    mask.uniform_()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    expected = module(x, mask)
                    with patch("boltz_lumi.multiplication.project", wraps=project) as joint:
                        equal(normalized_forward(module, x, mask), expected)
                        assert bool(joint.call_count) == expects_joint_projection
                    checks += 1
        old = prepare(module)
        with torch.no_grad():
            module.p_in.weight.add_(0.01)
        new = prepare(module)
        assert old.data_ptr() != new.data_ptr()
        projection_rows = module.p_in.weight.shape[0]
        equal(new[:projection_rows], module.p_in.weight.bfloat16())
        checks += 1
        x = torch.randn(1, 3, 3, pair_dim, device="cuda", requires_grad=True)
        mask = torch.ones(1, 3, 3, device="cuda")
        actual = normalized_forward(module, x, mask)
        expected = module(x, mask)
        equal(actual, expected)
        actual_grad = torch.autograd.grad(actual.sum(), x)[0]
        expected_grad = torch.autograd.grad(expected.sum(), x)[0]
        equal(actual_grad, expected_grad)
        checks += 1
    print(
        json.dumps(
            dict(
                status="passed",
                checks=checks,
                max_absolute_error=0,
                relative_l2_error=0,
                atol=0,
                rtol=0,
                failing_values=0,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
