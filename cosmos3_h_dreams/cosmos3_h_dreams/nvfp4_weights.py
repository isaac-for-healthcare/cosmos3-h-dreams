# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Export a separate experimental NVFP4 overlay directly from original BF16."""

import argparse
import json
import re
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from cosmos3_h_dreams._runtime.checkpoint_remap import remap_key
from cosmos3_h_dreams.fp8_weights import PROJECTIONS, file_sha256

RECIPE = "generator-mlp-nvfp4-e2m1-block16-e4m3-tensor-fp32-v1"


@torch.no_grad()
def export_weights(checkpoint: Path, output: Path, *, device: str = "cuda") -> dict:
    """Keep the source immutable and publish a manifest only after all shards."""
    from cosmos3_h_dreams._runtime._nvfp4 import preflight
    from cosmos3_h_dreams._runtime._nvfp4_kernels import quantize

    preflight(device)
    checkpoint = checkpoint.resolve(strict=True)
    if not checkpoint.is_file() or checkpoint.suffix != ".safetensors":
        raise ValueError("NVFP4 requires a local single-file BF16 safetensors checkpoint")
    before = checkpoint.stat()
    source_sha = file_sha256(checkpoint)
    output.mkdir(parents=True, exist_ok=False)
    layers, shards = {}, []
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
                if (weight.dtype != torch.bfloat16 or weight.ndim != 2
                        or any(s <= 0 or s % 128 for s in weight.shape)
                        or not torch.isfinite(weight).all().item()):
                    raise ValueError("NVFP4 requires finite BF16 weights with dimensions divisible by 128")
                quantized = quantize(weight)
                for suffix, tensor in zip((".weight", ".scale", ".global"), quantized):
                    tensors[projection + suffix] = tensor.cpu().contiguous()
                shapes[projection] = list(weight.shape)
                del weight, quantized
            filename = f"layer-{index:03d}.safetensors"
            save_file(tensors, str(output / filename), metadata={"recipe": RECIPE})
            shards.append({"layer": index, "file": filename,
                           "sha256": file_sha256(output / filename), "shapes": shapes})
            print(f"[nvfp4] Exported generator MLP {index}", flush=True)
    after = checkpoint.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError("Source changed during export; overlay is incomplete")
    manifest = {"recipe": RECIPE, "source_sha256": source_sha,
                "source_bytes": before.st_size, "torch": str(torch.__version__),
                "quantization_device": device, "shards": shards,
                "original_weight_bytes": sum(2 * s[0] * s[1] for row in shards for s in row["shapes"].values()),
                "overlay_bytes": sum((output / row["file"]).stat().st_size for row in shards)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify_overlay(checkpoint: Path, overlay: Path) -> dict:
    """Check checkpoint identity and all shard checksums before model mutation."""
    manifest = json.loads((overlay / "manifest.json").read_text())
    if manifest.get("recipe") != RECIPE:
        raise ValueError("Unsupported NVFP4 overlay recipe")
    if not checkpoint.is_file() or checkpoint.stat().st_size != manifest["source_bytes"]:
        raise ValueError("NVFP4 overlay requires its original single-file checkpoint")
    if file_sha256(checkpoint) != manifest["source_sha256"]:
        raise ValueError("NVFP4 overlay source checkpoint checksum mismatch")
    shards = manifest["shards"]
    if not shards or [s["layer"] for s in shards] != list(range(len(shards))):
        raise ValueError("Invalid NVFP4 overlay layer list")
    for index, shard in enumerate(shards):
        if shard["file"] != f"layer-{index:03d}.safetensors":
            raise ValueError("Invalid NVFP4 overlay shard filename")
        if file_sha256(overlay / shard["file"]) != shard["sha256"]:
            raise ValueError(f"NVFP4 overlay shard checksum mismatch: {shard['file']}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    print(json.dumps(export_weights(args.checkpoint, args.output, device=args.device), indent=2))


if __name__ == "__main__":
    main()
