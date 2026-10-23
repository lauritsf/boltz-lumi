# SPDX-License-Identifier: MIT
# The AITER kernel below is adapted from AMD's MIT-licensed fast_transpose.py.
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# See third_party/README.md and third_party/aiter-LICENSE.
"""Contiguous copies for triangle multiplication."""

import torch
import functools
import json
from pathlib import Path
import triton
import triton.language as tl


@triton.jit
def aiter_transpose(
    IN_ptr,
    OUT_ptr,
    M,
    N,
    stride_in_m,
    stride_in_n,
    stride_out_n,
    stride_out_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n_blocks = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_blocks
    pid_n = pid % num_n_blocks
    offs_m = tl.cast(pid_m * BLOCK_M + tl.arange(0, BLOCK_M), tl.int64)
    offs_n = tl.cast(pid_n * BLOCK_N + tl.arange(0, BLOCK_N), tl.int64)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    vals = tl.load(
        IN_ptr + offs_m[:, None] * stride_in_m + offs_n[None, :] * stride_in_n, mask=mask
    )
    tl.store(
        OUT_ptr + offs_n[None, :] * stride_out_n + offs_m[:, None] * stride_out_m, vals, mask=mask
    )


@triton.jit
def plane_copy(
    X,
    Y,
    C: tl.constexpr,
    R: tl.constexpr,
    S: tl.constexpr,
    XC: tl.constexpr,
    XR: tl.constexpr,
    XS: tl.constexpr,
    BC: tl.constexpr,
    BS: tl.constexpr,
):
    c = tl.program_id(0) * BC + tl.arange(0, BC)
    s = tl.program_id(1) * BS + tl.arange(0, BS)
    r = tl.program_id(2)
    mask = (c[:, None] < C) & (s[None, :] < S)
    value = tl.load(X + c[:, None] * XC + r * XR + s[None, :] * XS, mask=mask)
    tl.store(Y + c[:, None] * R * S + r * S + s[None, :], value, mask=mask)


def contiguous_copy(x, config):
    """Copy a strided [channel, row, column] tensor into contiguous storage."""
    if x.dtype not in (torch.float32, torch.bfloat16) or not x.is_cuda:
        raise ValueError("Expected a CUDA/ROCm FP32 or BF16 tensor")
    block_rows = config["bm"]
    block_columns = config["bn"]
    num_warps = config["warps"]
    options = {"num_warps": num_warps, "waves_per_eu": config.get("waves", 0), "num_stages": 2}
    output = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if x.ndim == 3:
        c, r, s = x.shape
        if x.stride(1) == s * x.stride(2):
            m = r * s
            n = c
            xm = x.stride(2)
            xn = x.stride(0)
            yn = r * s
            ym = 1
            plane = False
        else:
            plane = True
    else:
        raise ValueError(f"Unsupported layout: {x.shape}, {x.stride()}")
    if plane:
        grid = (triton.cdiv(c, block_rows), triton.cdiv(s, block_columns), r)
        plane_copy[grid](x, output, c, r, s, *x.stride(), block_rows, block_columns, **options)
    else:
        grid = (triton.cdiv(m, block_rows) * triton.cdiv(n, block_columns),)
        aiter_transpose[grid](x, output, m, n, xm, xn, yn, ym, block_rows, block_columns, **options)
    return output


@functools.lru_cache(maxsize=1)
def measured_configs():
    return json.loads(Path(__file__).with_name("gfx90a_layout_config.json").read_text())


def configured_copy(x):
    # Incoming multiplication swaps the operand layouts, so dispatch by strides.
    if x.ndim == 3 and x.stride(1) == x.shape[2] * x.stride(2):
        name = "matmul_left"
    else:
        name = "matmul_right"
    choice = measured_configs()["choices"][name]
    return contiguous_copy(x, choice["config"])
