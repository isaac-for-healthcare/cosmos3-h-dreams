# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run explicitly on the target GPU before the full matched rollout trial."""

import pytest
import torch

from cosmos3_h_dreams._runtime._fp8 import FP8SwiGLU, preflight
from cosmos3_h_dreams.fp8_weights import FP8, PROJECTIONS
from flashdreams.accelerated.quantization.quantizer import Granularity, quantize

pytestmark = pytest.mark.manual


@torch.inference_mode()
def test_fp8_real_shapes_compile_and_graph():
    assert torch.cuda.is_available(), "This manual check requires CUDA"
    preflight("cuda")
    torch.manual_seed(11)
    tensors = {}
    for name, (n, k) in zip(PROJECTIONS, ((12288, 4096), (12288, 4096), (4096, 12288))):
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16) / k ** .5
        tensors[name + ".weight"], tensors[name + ".scale"] = quantize(weight, FP8, Granularity.SLICE)
    layer = FP8SwiGLU(tensors).eval()
    compiled = torch.compile(layer, fullgraph=True, mode="default")
    for rows in (1, 580, 1160, 1848):
        x = torch.randn(1, rows, 4096, device="cuda", dtype=torch.bfloat16)
        eager = layer(x)
        actual = compiled(x)
        torch.cuda.synchronize()
        assert actual.shape == x.shape and torch.isfinite(actual).all().item()
        torch.testing.assert_close(actual, eager, rtol=.03, atol=.03)
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
