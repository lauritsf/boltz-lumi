"""Check BF16 boundaries, tails, offsets, streams and meaningful transition weights."""

import json
import torch
from boltz.model.layers.transition import Transition
from boltz_lumi.kernels._build import extension
from boltz_lumi.pair_transition import transition_forward


def same_bits(a, b):
    if not torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)):
        delta = a.double() - b.double()
        norm = b.double().norm().item()
        raise AssertionError(
            dict(
                max_absolute_error=delta.abs().max().item(),
                relative_l2_error=delta.norm().item() / norm if norm else None,
                failing_values=int((a != b).sum()),
                atol=0,
                rtol=0,
            )
        )
    return True


def main():
    assert torch.version.hip and torch.cuda.device_count() == 1
    torch.manual_seed(81)
    native = extension("pair")
    checks = 0
    pair_dim = 128
    hidden_dim = 512
    batch_size = 2
    # Exercise SiLU over a wider input range.
    input_scale = 20
    offsets = (0, 1)
    # Cover empty and singleton inputs, tails, and larger flattened projections.
    sizes = (0, 1, 17, 257, 512, 513, 117 * hidden_dim, 312 * hidden_dim)
    with torch.inference_mode():
        for size in sizes:
            # Extra storage permits a shifted address without shortening the input.
            storage_size = size + 2
            a = torch.randn(storage_size, device="cuda", dtype=torch.bfloat16) * input_scale
            b = torch.randn_like(a)
            for offset in offsets:
                a_slice = a[offset : offset + size]
                b_slice = b[offset : offset + size]
                expected = torch.nn.functional.silu(a_slice) * b_slice
                actual = native.silu_mul(a_slice, b_slice)
                assert same_bits(actual, expected), (size, offset)
                checks += 1
        for n in [17, 117, 312, 576]:
            module = Transition(pair_dim, hidden_dim).cuda().eval()
            for parameter in module.parameters():
                parameter.normal_(mean=0, std=0.05)
            x = torch.randn(batch_size, n, pair_dim, device="cuda")
            strided_input = x[:, ::2]
            for value in (x, strided_input):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    expected = module(value)
                    actual = transition_forward(module, value)
                assert expected.abs().max() > 0
                assert same_bits(actual, expected)
                checks += 1
            expected = module(x)
            assert same_bits(transition_forward(module, x), expected)
            checks += 1
        # Also exercise a tail on a nondefault stream.
        stream_test_size = 513
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            a = torch.randn(stream_test_size, device="cuda", dtype=torch.bfloat16)
            b = torch.randn_like(a)
            expected = torch.nn.functional.silu(a) * b
            actual = native.silu_mul(a, b)
        stream.synchronize()
        assert same_bits(actual, expected)
        checks += 1
    gradient_input_dim = 32
    gradient_hidden_dim = 128
    module = Transition(gradient_input_dim, gradient_hidden_dim).cuda().eval()
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(mean=0, std=0.05)
    x = torch.randn(batch_size, 17, gradient_input_dim, device="cuda", requires_grad=True)
    actual = transition_forward(module, x)
    expected = module(x)
    assert same_bits(actual, expected)
    assert actual.requires_grad
    actual.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().max() > 0
    print(
        json.dumps(
            {
                "status": "passed",
                "checks": checks,
                "max_absolute_error": 0,
                "rms_absolute_error": 0,
                "relative_l2_error": 0,
                "bitwise_identical": True,
                "atol": 0,
                "rtol": 0,
                "failing_values": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
