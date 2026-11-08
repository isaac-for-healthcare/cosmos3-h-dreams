# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental NVFP4 packing with tensor scaling and 16-element FP8 blocks."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _partials(X, P, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X+i, i<COUNT, 0).to(tl.float32)
    tl.store(P+tl.program_id(0), tl.max(tl.abs(x), 0))


@tr.jit
def _global(P, G, COUNT: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    peak = tl.max(tl.load(P+i, i<COUNT, 0), 0)
    g = tl.where(peak==0, 1., tl.maximum(tl.div_rn(peak, 2688.), 1.e-30))
    tl.store(G, g)


@tr.jit
def _pack(X, Q, S, G, ROWS: tl.constexpr, K: tl.constexpr, BLOCKS: tl.constexpr):
    b = tl.program_id(0)*BLOCKS + tl.arange(0, BLOCKS)
    row, col = b//(K//16), b%(K//16)
    j = tl.arange(0, 16)
    x = tl.load(X+b[:, None]*16+j[None, :], row[:, None]<ROWS, 0).to(tl.float32)
    g = tl.load(G)
    peak = tl.max(tl.abs(x), 1)
    sf = tl.minimum(tl.div_rn(tl.div_rn(peak, g), 6.), 448.).to(tl.float8e4nv)
    scale = sf.to(tl.float32)
    z = tl.where(scale[:, None]==0, 0., tl.div_rn(tl.div_rn(x, g), scale[:, None]))
    even, odd = tl.split(tl.reshape(z, (BLOCKS, 8, 2)))
    packed = tl.inline_asm_elementwise(
        "{ .reg .b8 p; cvt.rn.satfinite.e2m1x2.f32 p, $2, $1; mov.b16 $0, {p, 0}; }",
        constraints="=h,f,f", args=[even, odd], dtype=tl.uint16, is_pure=True, pack=1)
    tl.store(Q+b[:, None]*8+tl.arange(0, 8)[None, :], packed.to(tl.uint8), row[:, None]<ROWS)
    # SWIZZLE_32_4_4, including initialized padding rows.
    index = (row//128)*(K//16*128) + (col//4)*512 + (row%32)*16 + ((row%128)//32)*4 + col%4
    tl.store(S+index, sf.to(tl.uint8, bitcast=True), row<tr.cdiv(ROWS, 128)*128)


@torch.library.custom_op('cosmos3_h_dreams_nvfp4::quantize', mutates_args=())
def quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert x.is_cuda and x.dtype==torch.bfloat16 and x.ndim==2 and x.is_contiguous()
    rows, k = x.shape
    assert rows>0 and k%64==0
    q = torch.empty((rows, k//2), device=x.device, dtype=torch.uint8)
    s = torch.empty((tr.cdiv(rows, 128)*128*(k//16),), device=x.device, dtype=torch.uint8)
    g = torch.empty((1,), device=x.device, dtype=torch.float32)
    n = tr.cdiv(x.numel(), 4096)
    tmp = torch.empty((n,), device=x.device, dtype=torch.float32)
    _partials[(n,)](x, tmp, x.numel(), 4096)
    _global[(1,)](tmp, g, n, tr.next_power_of_2(n))
    _pack[(tr.cdiv(tr.cdiv(rows,128)*128*(k//16),32),)](x, q, s, g, rows, k, 32)
    return q, s, g


@quantize.register_fake
def _fake(x):
    rows, k = x.shape
    return (x.new_empty((rows, k//2), dtype=torch.uint8),
            x.new_empty((((rows+127)//128)*128*(k//16),), dtype=torch.uint8),
            x.new_empty((1,), dtype=torch.float32))


def mm(a, b, sa, sb, ga, gb):
    return torch.ops.aten._scaled_mm_v2(
        a.view(torch.float4_e2m1fn_x2), b.T.view(torch.float4_e2m1fn_x2),
        [sa.view(torch.float8_e4m3fn), ga], [2, 0], [1, 0],
        [sb.view(torch.float8_e4m3fn), gb], [2, 0], [1, 0], None, torch.bfloat16)
