# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental FP8 MLPs using the existing FlashDreams quantizer and GEMM."""

from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F

from cosmos3_h_dreams.fp8_weights import FP8, PROJECTIONS
from flashdreams.accelerated.common.non_persistent_linear import NonPersistentLinear
from flashdreams.accelerated.quantization.linear import QuantizedNonPersistentLinear
from flashdreams.accelerated.quantization.quantizer import Granularity, quantize


class _LoadedFP8Linear(QuantizedNonPersistentLinear):
    """Use already quantized buffers without converting the original again."""

    def __init__(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        NonPersistentLinear.__init__(self, weight, None)
        self.dtype = FP8
        self.register_buffer("weight_scale", scale, persistent=False)


class FP8SwiGLU(nn.Module):
    """Share dynamic activation quantization between gate and up projections."""

    def __init__(self, tensors: dict[str, torch.Tensor]) -> None:
        super().__init__()
        for name in PROJECTIONS:
            self.add_module(name, _LoadedFP8Linear(tensors[name + ".weight"], tensors[name + ".scale"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0:
            return x.clone()
        quantized, scale = quantize(x, FP8, Granularity.SLICE, axis=-1)
        gate = self.gate_proj(quantized, scale, out_dtype=x.dtype)
        up = self.up_proj(quantized, scale, out_dtype=x.dtype)
        return self.down_proj(F.silu(gate) * up, Granularity.SLICE, out_dtype=x.dtype)


@torch.no_grad()
def preflight(device: str) -> None:
    """Fail before checkpoint loading if this build cannot execute rowwise FP8."""
    if not device.startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("FP8 requires CUDA")
    if torch.cuda.get_device_capability(device) < (8, 9):
        raise ValueError("FP8 requires compute capability 8.9 or newer")
    try:
        # Both projection orientations; these are kernel checks, not a benchmark.
        for k, n in ((4096, 12288), (12288, 4096)):
            x = torch.ones(16, k, device=device, dtype=FP8)
            w = torch.ones(n, k, device=device, dtype=FP8)
            y = torch._scaled_mm(x, w.T, torch.ones(16, 1, device=device),
                                 torch.ones(1, n, device=device), out_dtype=torch.bfloat16)
            if y.dtype != torch.bfloat16 or not torch.all(y == k).item():
                raise RuntimeError("FP8 preflight returned incorrect values")
        torch.cuda.synchronize(device)
    except RuntimeError as error:
        raise RuntimeError(
            "Rowwise FP8 is unavailable in this GPU/PyTorch/CUDA build. "
            "See the stack requirements in FP8_TRIAL.md or omit --fp8-weights. "
            "No precision fallback was applied."
        ) from error


@torch.no_grad()
def install_overlay(net: nn.Module, overlay: Path, manifest: dict) -> dict:
    """Validate all projections before replacing only the generation MLPs."""
    shards = manifest["shards"]
    if len(shards) != len(net.layers):
        raise ValueError("FP8 layer count does not match the loaded model")
    replacements = []
    for shard, layer in zip(shards, net.layers):
        mlp = layer.gen.mlp
        device = mlp.gate_proj.weight.device
        tensors = load_file(str(overlay / shard["file"]), device=str(device))
        expected = {p + suffix for p in PROJECTIONS for suffix in (".weight", ".scale")}
        if set(tensors) != expected:
            raise ValueError("Unexpected FP8 shard tensor names")
        for name in PROJECTIONS:
            source = getattr(mlp, name)
            weight, scale = tensors[name + ".weight"], tensors[name + ".scale"]
            if (source.bias is not None or source.weight.dtype != torch.bfloat16
                    or tuple(weight.shape) != tuple(source.weight.shape)
                    or list(weight.shape) != shard["shapes"][name]
                    or weight.dtype != FP8 or scale.dtype != torch.float32
                    or tuple(scale.shape) != (source.out_features, 1)):
                raise ValueError(f"FP8 projection contract mismatch: {shard['layer']}.{name}")
            if (not torch.isfinite(weight.float()).all().item()
                    or not torch.isfinite(scale).all().item() or not (scale > 0).all().item()):
                raise ValueError("Invalid FP8 weight or scale")
        replacements.append(FP8SwiGLU(tensors).eval())
    for layer, replacement in zip(net.layers, replacements):
        layer.gen.mlp = replacement
    return {"recipe": manifest["recipe"], "mlps": len(replacements),
            "linears": 3 * len(replacements), "source_sha256": manifest["source_sha256"],
            "manifest": str((overlay / "manifest.json").resolve())}
