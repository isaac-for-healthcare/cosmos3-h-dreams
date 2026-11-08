# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU trust-boundary checks for the separate NVFP4 artifact and opt-in flag."""
import json
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from torch import nn

from cosmos3_h_dreams import Cosmos3HDreamsApplication
from cosmos3_h_dreams import nvfp4_weights as weights
from cosmos3_h_dreams._runtime._nvfp4 import install_overlay

pytestmark = pytest.mark.ci_cpu


def test_overlay_integrity_atomic_install_and_dtype_preservation(tmp_path):
    source = tmp_path / "source.safetensors"
    save_file({"original": torch.ones(128, dtype=torch.bfloat16)}, str(source))
    original_bytes = source.read_bytes()
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    shards, layers = [], []
    for index in range(2):
        tensors, shapes, mlp = {}, {}, nn.Module()
        for name in weights.PROJECTIONS:
            mlp.add_module(name, nn.Linear(128, 128, bias=False, dtype=torch.bfloat16))
            shapes[name] = [128, 128]
            tensors[name + ".weight"] = torch.zeros(128, 64, dtype=torch.uint8)
            tensors[name + ".scale"] = torch.zeros(1024, dtype=torch.uint8)
            tensors[name + ".global"] = torch.ones(1, dtype=torch.float32)
        filename = f"layer-{index:03d}.safetensors"
        save_file(tensors, str(overlay / filename))
        shards.append(dict(layer=index, file=filename, sha256=weights.file_sha256(overlay / filename), shapes=shapes))
        layers.append(SimpleNamespace(gen=SimpleNamespace(mlp=mlp), und=nn.Linear(2, 2)))
    manifest = dict(recipe=weights.RECIPE, source_sha256=weights.file_sha256(source), source_bytes=len(original_bytes), shards=shards)
    (overlay / "manifest.json").write_text(json.dumps(manifest))
    assert weights.verify_overlay(source, overlay) == manifest
    net = SimpleNamespace(layers=layers)
    protected = [layer.und for layer in layers]
    originals = [layer.gen.mlp for layer in layers]
    late = overlay / "layer-001.safetensors"
    tensors = load_file(str(late))
    tensors["down_proj.global"][0] = float("nan")
    save_file(tensors, str(late))
    with pytest.raises(ValueError, match="shard checksum"):
        weights.verify_overlay(source, overlay)
    with pytest.raises(ValueError, match="Invalid NVFP4 scale"):
        install_overlay(net, overlay, manifest)
    assert all(layer.gen.mlp is old for layer, old in zip(layers, originals))
    tensors["down_proj.global"][0] = 1
    save_file(tensors, str(late))
    install_overlay(net, overlay, manifest)
    net.layers[0].gen.mlp.to(dtype=torch.bfloat16)
    assert net.layers[0].gen.mlp.gate_proj.global_scale.dtype == torch.float32
    assert net.layers[0].gen.mlp.gate_proj.weight.dtype == torch.uint8
    assert all(layer.und is old for layer, old in zip(layers, protected))
    assert source.read_bytes() == original_bytes
    manifest["shards"][0]["file"] = "../source.safetensors"
    (overlay / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="filename"):
        weights.verify_overlay(source, overlay)


def test_nvfp4_is_explicit_and_excludes_fp8(tmp_path):
    frame = tmp_path / "frame.png"
    Image.new("RGB", (32, 32)).save(frame)
    args = ["--checkpoint", "/checkpoint", "--initial-frame", str(frame), "--headset"]
    app = Cosmos3HDreamsApplication()
    app.init(args)
    assert app.create_session().config.nvfp4_weights is None
    app.init([*args, "--nvfp4-weights", str(tmp_path)])
    assert app.create_session().config.nvfp4_weights == tmp_path
    with pytest.raises(ValueError, match="Choose only one"):
        app.init([*args, "--fp8-weights", str(tmp_path), "--nvfp4-weights", str(tmp_path)])
    with pytest.raises(ValueError, match="requires a CUDA"):
        app.init([*args, "--device", "cpu", "--nvfp4-weights", str(tmp_path)])
    app.init([*args, "--nvfp4-weights", str(tmp_path), "--no-nvfp4-weights"])
    assert app.create_session().config.nvfp4_weights is None
