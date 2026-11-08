# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Manual SM120 test: packing oracle, GEMM, compile and CUDA graph replay."""
import pytest
import torch

from cosmos3_h_dreams._runtime._nvfp4 import NVFP4SwiGLU, preflight
from cosmos3_h_dreams.nvfp4_weights import PROJECTIONS

pytestmark = pytest.mark.manual


@torch.inference_mode()
def test_nvfp4_real_shapes_compile_and_graph():
    from cosmos3_h_dreams._runtime._nvfp4_kernels import quantize, mm
    preflight("cuda")
    torch.manual_seed(11)
    codes = [0., .5, 1., 1.5, 2., 3., 4., 6.]
    values = torch.tensor(codes + [-v for v in codes], device="cuda", dtype=torch.bfloat16).repeat(8)[None]
    packed, _, _ = quantize(values)
    expected_codes = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe], device="cuda", dtype=torch.uint8).repeat(8)[None]
    assert torch.equal(packed, expected_codes)
    # Decode the packed representation independently of the Triton implementation.
    def decode(q, s, g):
        rows, half_k = q.shape
        row = torch.arange(rows, device=q.device)[:, None]
        col = torch.arange(half_k // 8, device=q.device)[None, :]
        idx = (row // 128) * (half_k // 8 * 128) + col // 4 * 512 + row % 32 * 16 + row % 128 // 32 * 4 + col % 4
        scales = s[idx].view(torch.float8_e4m3fn).float()
        bits = torch.stack((q & 15, q >> 4), -1).flatten(-2)
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=q.device)
        value = lut[(bits & 7).long()] * torch.where((bits & 8) != 0, -1., 1.)
        return (value.reshape(rows, -1, 16) * scales[..., None] * g).reshape(rows, -1)
    torch.backends.cuda.matmul.allow_tf32 = False
    for rows in (1, 128, 580, 788):
        x = torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        a, sa, ga = quantize(x)
        b, sb, gb = quantize(w)
        expected = (decode(a, sa, ga) @ decode(b, sb, gb).T).bfloat16()
        actual = mm(a, b, sa, sb, ga, gb)
        relative_rmse = (actual.float() - expected.float()).square().mean().sqrt() / expected.float().square().mean().sqrt()
        assert relative_rmse < .001
        zero = quantize(torch.zeros_like(x))
        assert (zero[0] == 0).all() and (zero[1] == 0).all() and (zero[2] == 1).all()
    tensors = {}
    for name, (n, k) in zip(PROJECTIONS, ((12288, 4096), (12288, 4096), (4096, 12288))):
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k ** .5
        for suffix, value in zip((".weight", ".scale", ".global"), quantize(weight)):
            tensors[name + suffix] = value
    layer = NVFP4SwiGLU(tensors).eval()
    compiled = torch.compile(layer, fullgraph=True)
    for rows in (580, 788, 1848):
        x = torch.randn(1, rows, 4096, device="cuda", dtype=torch.bfloat16)
        eager, actual = layer(x), compiled(x)
        error = (actual.float() - eager.float()).square().mean().sqrt() / eager.float().square().mean().sqrt()
        assert error < .001, error
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            compiled(x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = compiled(x)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, actual, rtol=0, atol=0)
    assert layer(x[:, :0]).shape == (1, 0, 4096)
