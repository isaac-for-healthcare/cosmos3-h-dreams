# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU artifact/selection checks; GEMM emulation is not CUDA validation."""

import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file
from torch import nn

from cosmos3_h_dreams import fp8_weights as weights
from cosmos3_h_dreams._runtime import _fp8

pytestmark = pytest.mark.ci_cpu


def source_and_net(tmp_path):
    torch.manual_seed(11)
    raw, layers = {}, []
    for index in range(2):
        mlp = nn.Module()
        for name, shape in zip(weights.PROJECTIONS, ((32, 16), (32, 16), (16, 32))):
            projection = nn.Linear(shape[1], shape[0], bias=False, dtype=torch.bfloat16)
            projection.weight.data[0] = 0  # Zero rows must have a positive finite scale.
            mlp.add_module(name, projection)
            raw[f"model.net.language_model.model.layers.{index}.mlp_moe_gen.{name}.weight"] = projection.weight
        layers.append(SimpleNamespace(gen=SimpleNamespace(mlp=mlp), und=nn.Linear(16, 16)))
    raw["model.net.llm2vae.weight"] = torch.ones(16, 16, dtype=torch.bfloat16)
    path = tmp_path / "source.safetensors"
    save_file(raw, str(path))
    return path, SimpleNamespace(layers=layers)


def test_separate_overlay_roundtrip_selection_and_forward(tmp_path, monkeypatch):
    source, net = source_and_net(tmp_path)
    original = source.read_bytes()
    protected = [layer.und for layer in net.layers]
    overlay = tmp_path / "fp8"
    manifest = weights.export_weights(source, overlay)
    assert weights.verify_overlay(source, overlay) == manifest
    assert source.read_bytes() == original
    report = _fp8.install_overlay(net, overlay, manifest)
    assert report["linears"] == 6
    assert all(layer.und is old for layer, old in zip(net.layers, protected))
    tensors = load_file(str(overlay / "layer-000.safetensors"))
    calls = []

    def scaled_mm(a, b, sa, sb, *, out_dtype):
        # Independent CPU arithmetic oracle for operand order/scales/output dtype.
        calls.append(tuple(a.shape))
        return ((a.float() * sa) @ (b.float() * sb)).to(out_dtype)

    monkeypatch.setattr(torch, "_scaled_mm", scaled_mm)
    x = torch.randn(2, 3, 16, dtype=torch.bfloat16)
    q, scale = _fp8.quantize(x, weights.FP8, _fp8.Granularity.SLICE)
    decoded = q.float() * scale
    gate, up = [torch.nn.functional.linear(decoded, tensors[n + ".weight"].float() * tensors[n + ".scale"]).bfloat16()
                for n in ("gate_proj", "up_proj")]
    hidden = (torch.nn.functional.silu(gate.float()) * up.float()).bfloat16()
    hq, hs = _fp8.quantize(hidden, weights.FP8, _fp8.Granularity.SLICE)
    expected = torch.nn.functional.linear(hq.float() * hs,
        tensors["down_proj.weight"].float() * tensors["down_proj.scale"]).bfloat16()
    actual = net.layers[0].gen.mlp(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert calls == [(6, 16), (6, 16), (6, 32)]
    assert net.layers[0].gen.mlp(x[:, :0]).shape == (2, 0, 16)
    net.layers[0].gen.mlp.to(dtype=torch.float32)
    assert net.layers[0].gen.mlp.gate_proj.weight.dtype == weights.FP8
    assert net.layers[0].gen.mlp.gate_proj.weight_scale.dtype == torch.float32
    with pytest.raises(FileExistsError):
        weights.export_weights(source, overlay)
    assert source.read_bytes() == original


def test_wrong_source_or_corrupted_artifact_is_rejected(tmp_path):
    source, _ = source_and_net(tmp_path)
    overlay = tmp_path / "fp8"
    weights.export_weights(source, overlay)
    other = tmp_path / "different.safetensors"
    content = bytearray(source.read_bytes())
    content[-1] ^= 1
    other.write_bytes(content)
    with pytest.raises(ValueError, match="source checkpoint checksum"):
        weights.verify_overlay(other, overlay)
    shard = overlay / "layer-000.safetensors"
    shard.write_bytes(shard.read_bytes()[:-1])
    with pytest.raises(ValueError, match="shard checksum"):
        weights.verify_overlay(source, overlay)


def test_bad_late_projection_leaves_model_unmodified(tmp_path):
    source, net = source_and_net(tmp_path)
    overlay = tmp_path / "fp8"
    manifest = weights.export_weights(source, overlay)
    originals = [layer.gen.mlp for layer in net.layers]
    path = overlay / "layer-001.safetensors"
    tensors = load_file(str(path))
    tensors["up_proj.scale"][0] = float("nan")
    save_file(tensors, str(path))
    with pytest.raises(ValueError, match="Invalid FP8 weight or scale"):
        _fp8.install_overlay(net, overlay, manifest)
    assert all(layer.gen.mlp is old for layer, old in zip(net.layers, originals))


@pytest.mark.parametrize("bad", [torch.ones(16, 16), torch.ones(16, 15, dtype=torch.bfloat16),
                                torch.full((16, 16), float("nan"), dtype=torch.bfloat16)])
def test_invalid_source_weight_rejected(bad):
    with pytest.raises(ValueError):
        weights.validate_weight(bad)


def test_overlay_cannot_escape_directory(tmp_path):
    source, _ = source_and_net(tmp_path)
    overlay = tmp_path / "fp8"
    manifest = weights.export_weights(source, overlay)
    manifest["shards"][0]["file"] = "../source.safetensors"
    (overlay / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="filename"):
        weights.verify_overlay(source, overlay)
