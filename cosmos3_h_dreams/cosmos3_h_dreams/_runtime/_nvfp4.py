# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental Blackwell NVFP4 generator MLPs; all other weights stay BF16."""

from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from cosmos3_h_dreams.nvfp4_weights import PROJECTIONS


class NVFP4Linear(nn.Module):
    """Use packed FP4 weights, swizzled FP8 block scales and one FP32 scale."""

    def __init__(self, weight, scale, global_scale):
        super().__init__()
        self.register_buffer("weight", weight, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.register_buffer("global_scale", global_scale, persistent=False)

    def _apply(self, fn, recurse=True):
        original = self.global_scale
        module = super()._apply(fn, recurse=recurse)
        if self.global_scale.dtype != original.dtype:
            self.global_scale = original.to(device=self.global_scale.device)
        return module

    def forward(self, quantized, scale, global_scale):
        from cosmos3_h_dreams._runtime._nvfp4_kernels import mm
        return mm(quantized, self.weight, scale, self.scale, global_scale, self.global_scale)


class NVFP4SwiGLU(nn.Module):
    """Share activation quantization between gate and up projections."""

    def __init__(self, tensors):
        super().__init__()
        for name in PROJECTIONS:
            self.add_module(name, NVFP4Linear(*(tensors[name + s] for s in (".weight", ".scale", ".global"))))

    def forward(self, x):
        from cosmos3_h_dreams._runtime._nvfp4_kernels import quantize
        if x.numel() == 0:
            return x.clone()
        shape = x.shape
        quantized = quantize(x.reshape(-1, shape[-1]).contiguous())
        gate, up = self.gate_proj(*quantized), self.up_proj(*quantized)
        # Match the FP8 path: explicit BF16 rounding after the FP32 product.
        hidden = (F.silu(gate.float()) * up.float()).to(x.dtype)
        return self.down_proj(*quantize(hidden)).reshape(shape)


@torch.no_grad()
def preflight(device: str) -> None:
    """Fail before checkpoint loading when the tested SM120 path is unavailable."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("NVFP4 requires CUDA")
    if torch.cuda.get_device_capability(device) != (12, 0):
        raise ValueError("This experimental NVFP4 implementation requires Blackwell SM120")
    try:
        from cosmos3_h_dreams._runtime._nvfp4_kernels import mm, quantize
        with torch.cuda.device(device):
            x = torch.ones(128, 128, device=device, dtype=torch.bfloat16)
            q, s, g = quantize(x)
            result = mm(q, q, s, s, g, g)
            if result.dtype != torch.bfloat16 or not torch.all(result == 128).item():
                raise RuntimeError("NVFP4 preflight returned incorrect values")
            torch.cuda.synchronize(device)
    except (AttributeError, ImportError, RuntimeError) as error:
        raise RuntimeError("NVFP4 is unavailable in this PyTorch/CUDA/Triton build. "
                           "See NVFP4_TRIAL.md or omit --nvfp4-weights. No precision fallback was applied.") from error


@torch.no_grad()
def install_overlay(net: nn.Module, overlay: Path, manifest: dict) -> dict:
    """Validate every replacement before modifying any generator MLP."""
    if len(net.layers) != len(manifest["shards"]):
        raise ValueError("NVFP4 layer count does not match the model")
    replacements = []
    for layer, shard in zip(net.layers, manifest["shards"]):
        tensors = load_file(str(overlay / shard["file"]), device=str(layer.gen.mlp.gate_proj.weight.device))
        if set(tensors) != {p + s for p in PROJECTIONS for s in (".weight", ".scale", ".global")}:
            raise ValueError("Unexpected NVFP4 shard tensor names")
        for name in PROJECTIONS:
            source = getattr(layer.gen.mlp, name)
            n, k = source.weight.shape
            q, s, g = (tensors[name + suffix] for suffix in (".weight", ".scale", ".global"))
            if (source.bias is not None or source.weight.dtype != torch.bfloat16
                    or n % 128 or k % 128 or [n, k] != shard["shapes"][name]
                    or q.dtype != torch.uint8 or q.shape != (n, k // 2)
                    or s.dtype != torch.uint8 or s.shape != (n * (k // 16),)
                    or g.dtype != torch.float32 or g.shape != (1,)):
                raise ValueError("NVFP4 projection contract mismatch")
            sf = s.view(torch.float8_e4m3fn).float()
            if not (torch.isfinite(g).all() & (g > 0).all() & torch.isfinite(sf).all() & (sf >= 0).all()).item():
                raise ValueError("Invalid NVFP4 scale")
        replacements.append(NVFP4SwiGLU(tensors).eval())
    for layer, replacement in zip(net.layers, replacements):
        layer.gen.mlp = replacement
    return {"recipe": manifest["recipe"], "mlps": len(replacements), "linears": 3 * len(replacements),
            "source_sha256": manifest["source_sha256"], "manifest": str((overlay / "manifest.json").resolve())}
