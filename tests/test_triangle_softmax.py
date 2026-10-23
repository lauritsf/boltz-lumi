"""Check small/large ordered softmax, masks, strides and alignment prefixes."""

import json
import torch
from boltz.model.layers.triangular_attention.primitives import softmax_no_cast
from boltz_lumi.kernels.triangle_softmax import ordered_softmax
from boltz_lumi.kernels._build import extension


def main():
    assert torch.version.hip
    torch.manual_seed(46)
    results = []
    batch_size = 1
    num_triangle_slices = 3
    num_heads = 4
    small_softmax_limit = 1024
    block_softmax_size = 2511
    rows_per_program = 4
    score_scale = 5
    bias_scale = 2
    num_masked_positions = 3
    mask_bias = -1e9
    # Cover power-of-two boundaries and representative larger inputs.
    sizes = (117, 129, 257, 312, 513, 576, 1023, 1024, 1134, 1734, block_softmax_size)
    with torch.inference_mode():
        for dtype in (torch.bfloat16, torch.float32):
            for n in sizes:
                if n > small_softmax_limit and dtype != torch.bfloat16:
                    continue
                scores = torch.randn(
                    batch_size, num_triangle_slices, num_heads, n, n, device="cuda", dtype=dtype
                ) * score_scale
                bias = torch.randn(
                    batch_size, 1, num_heads, n, n, device="cuda", dtype=dtype
                ) * bias_scale
                # Transpose to exercise a strided mask.
                mask = torch.zeros(
                    batch_size, n, 1, 1, num_triangle_slices, device="cuda"
                ).transpose(1, 4)
                mask[:, :, :, :, -num_masked_positions:] = mask_bias
                # Mask an entire triangle slice as well as trailing positions.
                mask[:, 1] = mask_bias
                reference = scores.clone()
                reference += mask
                reference += bias
                reference = softmax_no_cast(reference, -1)
                if n == block_softmax_size:
                    actual = extension("triangle_block_softmax").softmax(scores, bias, mask)
                else:
                    actual = ordered_softmax(
                        scores,
                        bias,
                        mask,
                        rows=rows_per_program,
                        allow_large=n > small_softmax_limit,
                    )
                delta = actual.double() - reference.double()
                norm = reference.double().norm().item()
                row = dict(
                    n=n,
                    dtype=str(dtype),
                    max_absolute_error=delta.abs().max().item(),
                    relative_l2_error=delta.norm().item() / norm if norm else None,
                    failing_values=int((actual != reference).sum()),
                    atol=0,
                    rtol=0,
                )
                results.append(row)
                assert torch.equal(
                    actual.contiguous().view(torch.uint8), reference.contiguous().view(torch.uint8)
                ), row
    print(json.dumps(dict(status="passed", cases=results)), flush=True)


if __name__ == "__main__":
    main()
