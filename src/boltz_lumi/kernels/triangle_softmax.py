"""Fuse attention bias and softmax using PyTorch's wave64 reduction order.

Based on PersistentSoftmax.cuh; see third_party/pytorch-LICENSE.
"""

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import triton.language.extra.libdevice as libdevice


# Wave64 and its reduction order are part of the PyTorch numerical contract.
WAVE_SIZE = gl.constexpr(64)
REDUCTION_STEPS = gl.constexpr(6)
FP32_BYTES = gl.constexpr(4)
SMALL_SOFTMAX_LIMIT = 1024
LARGE_BF16_SOFTMAX_LIMIT = 2048
MAX_OUTPUT_ELEMENTS = 2**31


@gluon.jit
def wave_xor(value, lane, offset: gl.constexpr):
    # ds_bpermute addresses lanes in bytes.
    return gl.inline_asm_elementwise(
        "ds_bpermute_b32 $0, $1, $2;\n s_waitcnt lgkmcnt(0);",
        constraints="=v,v,v",
        args=((lane ^ offset) * FP32_BYTES, value),
        dtype=gl.float32,
        is_pure=True,
        pack=1,
    )


@gluon.jit
def ordered_bias_softmax(
    S,
    Bias,
    Mask,
    P,
    N: gl.constexpr,
    I: gl.constexpr,
    H: gl.constexpr,
    SB: gl.constexpr,
    SI: gl.constexpr,
    SH: gl.constexpr,
    SQ: gl.constexpr,
    SK: gl.constexpr,
    BB: gl.constexpr,
    BH: gl.constexpr,
    BQ: gl.constexpr,
    BK: gl.constexpr,
    MB: gl.constexpr,
    MI: gl.constexpr,
    MS: gl.constexpr,
    ROWS: gl.constexpr,
    ROW_COUNT: gl.constexpr,
    ITERATIONS: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, WAVE_SIZE], [ROWS, 1], [1, 0])
    rows = gl.program_id(0) * ROWS + gl.arange(0, ROWS, layout=gl.SliceLayout(1, layout))
    lanes = gl.arange(0, WAVE_SIZE, layout=gl.SliceLayout(0, layout))
    q = rows % N
    h = (rows // N) % H
    i = (rows // (N * H)) % I
    b = rows // (N * H * I)
    raw = ()
    for j in gl.static_range(ITERATIONS):
        k = lanes + j * WAVE_SIZE
        value = gl.load(
            S
            + b[:, None] * SB
            + i[:, None] * SI
            + h[:, None] * SH
            + q[:, None] * SQ
            + k[None, :] * SK,
            mask=(rows[:, None] < ROW_COUNT) & (k[None, :] < N),
            other=0,
        ).to(gl.float32)
        mb = gl.load(
            Mask + b[:, None] * MB + i[:, None] * MI + k[None, :] * MS,
            mask=(rows[:, None] < ROW_COUNT) & (k[None, :] < N),
            other=0,
        )
        bias = gl.load(
            Bias + b[:, None] * BB + h[:, None] * BH + q[:, None] * BQ + k[None, :] * BK,
            mask=(rows[:, None] < ROW_COUNT) & (k[None, :] < N),
            other=0,
        )
        # PyTorch rounds to the score dtype after each bias addition.
        value = (value + mb).to(S.dtype.element_ty).to(gl.float32)
        value = (value + bias).to(S.dtype.element_ty).to(gl.float32)
        raw += (gl.where(k[None, :] < N, value, float("-inf")),)
    maximum = raw[0]
    for j in gl.static_range(1, ITERATIONS):
        maximum = gl.maximum(maximum, raw[j])
    for offset in gl.static_range(REDUCTION_STEPS):
        maximum = gl.maximum(maximum, wave_xor(maximum, lanes[None, :], (WAVE_SIZE // 2) >> offset))
    exps = ()
    total = gl.full((ROWS, WAVE_SIZE), 0, gl.float32, layout)
    for j in gl.static_range(ITERATIONS):
        value = libdevice.exp(raw[j] - maximum)
        exps += (value,)
        total = total + value
    for offset in gl.static_range(REDUCTION_STEPS):
        total = total + wave_xor(total, lanes[None, :], (WAVE_SIZE // 2) >> offset)
    for j in gl.static_range(ITERATIONS):
        value = gl.div_rn(exps[j], total)
        k = lanes + j * WAVE_SIZE
        gl.store(
            P + rows[:, None] * N + k[None, :],
            value.to(P.dtype.element_ty),
            mask=(rows[:, None] < ROW_COUNT) & (k[None, :] < N),
        )


def ordered_softmax(scores, bias, mask, *, rows=4, waves_per_eu=0, allow_large=False):
    if (
        torch.is_grad_enabled()
        or scores.dtype not in (torch.float32, torch.bfloat16)
        or scores.ndim != 5
    ):
        raise ValueError("Expected inference-only FP32/BF16 triangle scores")
    # Scores are [batch, triangle_slice, head, query, key].
    batch, i, h, n, _ = scores.shape
    if n <= 0 or scores.shape[-1] != n:
        raise ValueError("Expected nonempty square score matrices")
    if n > SMALL_SOFTMAX_LIMIT and not (
        allow_large and scores.dtype == torch.bfloat16 and n <= LARGE_BF16_SOFTMAX_LIMIT
    ):
        raise ValueError("Larger shapes require explicit BF16 persistent-softmax validation")
    # Bound the number of output elements for the validated 32-bit indexing.
    row_count = batch * i * h * n
    if row_count * n > MAX_OUTPUT_ELEMENTS:
        raise ValueError("Ordered-softmax output indexing exceeds the validated 32-bit range")
    output = torch.empty(scores.shape, device=scores.device, dtype=scores.dtype)
    ordered_bias_softmax[(triton.cdiv(row_count, rows),)](
        scores,
        bias,
        mask,
        output,
        n,
        i,
        h,
        *scores.stride(),
        bias.stride(0),
        bias.stride(2),
        bias.stride(3),
        bias.stride(4),
        mask.stride(0),
        mask.stride(1),
        mask.stride(4),
        rows,
        row_count,
        max(1, triton.next_power_of_2(n) // WAVE_SIZE.value),
        num_warps=rows,
        num_stages=1,
        enable_fp_fusion=False,
        waves_per_eu=waves_per_eu,
    )
    return output
