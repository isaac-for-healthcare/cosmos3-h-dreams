# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a separate FP8 generator-MLP overlay; never rewrite the BF16 source.

Only local, single-file safetensors checkpoints are supported by this pilot.
The overlay still requires the exact original checkpoint for all other weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from cosmos3_h_dreams._runtime.checkpoint_remap import remap_key

RECIPE = "generator-mlp-fp8-e4m3-rowwise-v1"
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
FP8 = torch.float8_e4m3fn


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_weight(weight: torch.Tensor) -> None:
    if weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("FP8 export requires BF16 matrix weights")
    if any(size == 0 or size % 16 for size in weight.shape):
        raise ValueError("FP8 projection dimensions must be positive multiples of 16")
    if not torch.isfinite(weight).all().item():
        raise ValueError("Nonfinite source weight")


@torch.no_grad()
def export_weights(checkpoint: Path, output: Path, *, device: str = "cpu") -> dict:
    """Write one overlay shard per MLP and publish a manifest only on success."""
    from flashdreams.accelerated.quantization.quantizer import Granularity, quantize

    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_file() or checkpoint.suffix != ".safetensors":
        raise ValueError("FP8 pilot requires a local single-file .safetensors checkpoint")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    before = checkpoint.stat()
    source_sha = file_sha256(checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    layers: dict[int, dict[str, str]] = {}
    shards = []
    with safe_open(str(checkpoint), framework="pt", device=device) as source:
        for key in source.keys():
            native = remap_key(key if key.startswith("model.net.") else "model.net." + key)
            match = re.fullmatch(r"layers\.(\d+)\.gen\.mlp\.(gate_proj|up_proj|down_proj)\.weight", native or "")
            if match:
                index, projection = int(match[1]), match[2]
                if projection in layers.setdefault(index, {}):
                    raise ValueError("Duplicate generator projection")
                layers[index][projection] = key
        if not layers or sorted(layers) != list(range(len(layers))):
            raise ValueError("Missing or noncontiguous generator MLP layers")
        for index, mapping in sorted(layers.items()):
            if set(mapping) != set(PROJECTIONS):
                raise ValueError(f"Incomplete generator MLP at layer {index}")
            tensors, shapes = {}, {}
            for projection in PROJECTIONS:
                weight = source.get_tensor(mapping[projection])
                validate_weight(weight)
                quantized, scale = quantize(weight, FP8, Granularity.SLICE, axis=-1)
                tensors[projection + ".weight"] = quantized.cpu().contiguous()
                tensors[projection + ".scale"] = scale.cpu().contiguous()
                shapes[projection] = list(weight.shape)
                del weight, quantized, scale
            filename = f"layer-{index:03d}.safetensors"
            save_file(tensors, str(output / filename), metadata={"recipe": RECIPE})
            shards.append({"layer": index, "file": filename,
                           "sha256": file_sha256(output / filename), "shapes": shapes})
            del tensors
            print(f"[fp8] Exported generator MLP {index}", flush=True)
    after = checkpoint.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Source changed during export; overlay is incomplete")
    manifest = {"recipe": RECIPE, "source_sha256": source_sha,
                "source_bytes": before.st_size, "torch": str(torch.__version__),
                "quantization_device": device, "shards": shards,
                "original_weight_bytes": sum(
                    2 * shape[0] * shape[1] for s in shards for shape in s["shapes"].values()),
                "overlay_bytes": sum((output / s["file"]).stat().st_size for s in shards)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_overlay(checkpoint: Path, overlay: Path) -> dict:
    """Check source identity and every shard before altering the loaded network."""
    manifest = json.loads((overlay / "manifest.json").read_text())
    if manifest.get("recipe") != RECIPE:
        raise ValueError("Unsupported FP8 overlay recipe")
    if not checkpoint.is_file() or checkpoint.stat().st_size != manifest["source_bytes"]:
        raise ValueError("FP8 overlay requires its original single-file checkpoint")
    if file_sha256(checkpoint) != manifest["source_sha256"]:
        raise ValueError("FP8 overlay source checkpoint checksum mismatch")
    shards = manifest["shards"]
    if not shards or [s["layer"] for s in shards] != list(range(len(shards))):
        raise ValueError("Invalid FP8 overlay layer list")
    for index, shard in enumerate(shards):
        if shard["file"] != f"layer-{index:03d}.safetensors":
            raise ValueError("Invalid FP8 overlay shard filename")
        if file_sha256(overlay / shard["file"]) != shard["sha256"]:
            raise ValueError(f"FP8 overlay shard checksum mismatch: {shard['file']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(export_weights(args.checkpoint, args.output, device=args.device), indent=2))


if __name__ == "__main__":
    main()
