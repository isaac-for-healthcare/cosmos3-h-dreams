# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks for the extracted runtime and installed-package resources."""

import json
from importlib.resources import files

import pytest
import torch
from cosmos3_h_dreams._runtime import _vae
from cosmos3_h_dreams._runtime._taehv2super import make_taehv2super_chunk_decoder
from cosmos3_h_dreams._runtime.checkpoint_load import (
    resolve_checkpoint_source,
    resolve_credential_path,
)
from cosmos3_h_dreams._runtime.fetch_checkpoint import (
    KNOWN_CHECKPOINTS,
    _cache_dir,
    ensure_checkpoint,
)
from cosmos3_h_dreams._runtime.hand_client import headset_source
from cosmos3_h_dreams._runtime.hand_client.webxr_server import page_asset
from cosmos3_h_dreams.app import Cosmos3HDreamsApplication
from cosmos3_h_dreams.config import Cosmos3HDreamsConfig
from cosmos3_h_dreams.session_config import load_session_config
from flashdreams.recipes.taehv import impl as taehv_impl
from flashdreams.recipes.wan.autoencoder.vae import (
    _WAN22_TI2V_5B_LATENT_MEAN,
    _WAN22_TI2V_5B_LATENT_STD,
)

pytestmark = pytest.mark.ci_cpu


def test_session_paths_follow_launch_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "session.yaml"
    config.write_text(
        "runtime:\n  checkpoint: model.pt\n"
        "scenes:\n  - name: first\n    initial_frame: frame.png\n"
    )
    session = load_session_config(config)
    assert session["runtime"]["checkpoint"] == str(tmp_path / "model.pt")
    assert session["scenes"][0]["initial_frame"] == tmp_path / "frame.png"


def test_checkpoint_sources_have_no_private_default(monkeypatch):
    assert not KNOWN_CHECKPOINTS
    assert ensure_checkpoint("/user/model.pt") == "/user/model.pt"
    monkeypatch.delenv("COSMOS3_H_DREAMS_TAEHV2SUPER_CHECKPOINT", raising=False)
    with pytest.raises(ValueError, match="COSMOS3_H_DREAMS_TAEHV2SUPER_CHECKPOINT"):
        make_taehv2super_chunk_decoder(device="cpu")


def test_credentials_follow_explicit_then_environment(monkeypatch):
    monkeypatch.setenv("COSMOS_CHECKPOINT_CREDENTIAL", "/user/from-env.secret")
    assert resolve_credential_path("/user/explicit.secret") == "/user/explicit.secret"
    assert resolve_credential_path() == "/user/from-env.secret"


def test_headset_resources_are_inside_the_product_package():
    package = files("cosmos3_h_dreams")
    assert b"<html" in page_asset("index.html").lower()
    assert page_asset("app.js")
    assert package.joinpath("web/headset.html").is_file()
    assert package.joinpath("web/viewer.js").is_file()


@pytest.mark.parametrize(
    "backend,denormalize", [("wan", False), ("taehv", False), ("taehv", True)]
)
def test_chunk_decoders_preserve_cache_order_and_latent_scaling(
    monkeypatch, backend, denormalize
):
    class Decoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))
            self.cache = object()
            self.indices = []

        def setup(self):
            return self

        def initialize_autoregressive_cache(self):
            return self.cache

        def prepare_cache(self):
            return self.cache

        def forward(self, latent, *, autoregressive_index, cache):
            assert cache is self.cache
            self.indices.append(autoregressive_index)
            return latent[:, :3]

        def decode(self, latent, *, cache):
            assert cache is self.cache
            self.indices.append(len(self.indices))
            return latent[:, :, :3]

    decoder = Decoder()
    if backend == "wan":
        monkeypatch.setattr(_vae, "wan22_vae_decoder_config", lambda **_: decoder)
        decode = _vae.make_wan22_chunk_decoder(device="cpu", dtype=torch.float32)
    else:
        monkeypatch.setattr(taehv_impl, "TAEHV", lambda **_: decoder)
        decode = _vae.make_taehv_chunk_decoder(
            device="cpu", dtype=torch.float32, denormalize=denormalize
        )
    latent = torch.arange(48 * 3, dtype=torch.float32).reshape(48, 3, 1, 1) / 100
    result = torch.cat([decode(latent[:, :2]), decode(latent[:, 2:])])
    expected = latent.permute(1, 0, 2, 3)
    if backend == "taehv":
        if denormalize:
            mean = torch.tensor(_WAN22_TI2V_5B_LATENT_MEAN).view(1, -1, 1, 1)
            std = torch.tensor(_WAN22_TI2V_5B_LATENT_STD).view(1, -1, 1, 1)
            expected = expected * std + mean
        expected = expected * 2 - 1
    torch.testing.assert_close(result, expected[:, :3])
    assert decoder.indices == [0, 1, 2]


def test_initial_frame_encoder_preserves_layout_and_pixel_normalization(monkeypatch):
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(1))

        def setup(self):
            return self

        def forward(self, image):
            assert image.shape == (1, 1, 3, 32, 32)
            return image[..., ::16, ::16]

    monkeypatch.setattr(_vae, "wan22_vae_encoder_config", lambda **_: Encoder())
    encode = _vae.make_wan22_encode_fn(device="cpu", dtype=torch.float32)
    image = (
        torch.arange(32 * 32 * 3, dtype=torch.int64).reshape(32, 32, 3).to(torch.uint8)
    )
    expected = image.permute(2, 0, 1).float()[:, ::16, ::16].unsqueeze(1) / 127.5 - 1
    torch.testing.assert_close(encode(image), expected)


def test_native_transport_is_rejected_before_headset_or_model_startup(tmp_path):
    with pytest.raises(ValueError, match="unknown headset transport"):
        headset_source.HeadsetSource(transport="native")
    frame = tmp_path / "frame.png"
    frame.touch()
    with pytest.raises(ValueError, match="unknown headset transport"):
        Cosmos3HDreamsConfig(
            checkpoint="model.pt",
            initial_frame=frame,
            prompt="scene",
            headset_transport="native",
        )
    with pytest.raises(SystemExit) as error:
        Cosmos3HDreamsApplication().init(
            [
                "--headset",
                "--headset-transport",
                "native",
                "--checkpoint",
                "model.pt",
                "--initial-frame",
                str(frame),
            ]
        )
    assert error.value.code == 2


def test_webxr_store_preserves_validation_freshness_and_tracking_loss(monkeypatch):
    clock = [1_000_000_000]
    monkeypatch.setattr(headset_source.time, "perf_counter_ns", lambda: clock[0])
    store = headset_source.FrameStore()
    store.consume(json.dumps({"type": "status", "stage": "session"}))
    assert store.stage == "session" and not store.ready()
    store.consume("invalid JSON")
    assert store.rejected == 1
    controller = {
        "position": [0, 0, 0],
        "orientation": [0, 0, 0, 1],
        "tracked": True,
        "axes": [0, 0],
    }
    packet = {
        "type": "xr_frame",
        "controllers": {"left": controller, "right": controller},
    }
    store.consume(json.dumps(packet))
    assert store.ready() and store.received == 1
    first = store.snapshot()
    packet["controllers"]["right"] = {**controller, "orientation": [0, 0, 0, 0]}
    store.consume(json.dumps(packet))
    assert store.rejected == 2 and store.snapshot() == first
    clock[0] += 501_000_000
    assert not store.ready()
    packet["controllers"]["right"] = controller
    store.consume(json.dumps(packet))
    assert store.ready() and store.received == 2
    packet["controllers"]["right"] = None
    store.consume(json.dumps(packet))
    assert not store.ready()
    assert store.snapshot()["right"]["tracked"] is False


def test_headset_local_input_requires_https_before_starting_server():
    source = headset_source.HeadsetSource()
    with pytest.raises(ValueError, match="HTTPS server or --headset-relay-url"):
        source._serve()
    assert source._server is None


def test_product_checkpoint_environment_and_cache(monkeypatch, tmp_path):
    monkeypatch.setenv("COSMOS3_H_DREAMS_CHECKPOINT", "/user/model.safetensors")
    assert resolve_checkpoint_source(None) == "/user/model.safetensors"
    assert resolve_checkpoint_source("explicit.pt") == "explicit.pt"
    monkeypatch.delenv("COSMOS3_H_DREAMS_CHECKPOINT")
    with pytest.raises(SystemExit, match="COSMOS3_H_DREAMS_CHECKPOINT"):
        resolve_checkpoint_source(None)
    monkeypatch.setenv("FLASHDREAMS_CACHE_DIR", str(tmp_path))
    assert _cache_dir() == tmp_path / "cosmos3_h_dreams"
