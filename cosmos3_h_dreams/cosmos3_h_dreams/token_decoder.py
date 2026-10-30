# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Cosmos3 Wan2.2 TAEHV export and browser decoder descriptor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from flashdreams.recipes.taehv.export import export_streaming_decoder
from flashdreams.recipes.taehv.impl import TAEHV

from ._runtime._vae import TAEW2_2_CHECKPOINT


class TokenDecoder(torch.nn.Module):
    """Adapt the native normalized Wan2.2 decoder to the upstream exporter."""

    def __init__(self, device: str, dtype: torch.dtype) -> None:
        super().__init__()
        self.taehv = TAEHV(
            checkpoint_path=TAEW2_2_CHECKPOINT,
            model_type="wan22",
            use_compile=False,
            use_cuda_graph=False,
        ).to(device=device, dtype=dtype).eval()

    def initialize_autoregressive_cache(self):
        """Create fresh temporal state for the reference latent."""
        return self.taehv.prepare_cache()

    def forward(self, input, autoregressive_index=0, cache=None):
        """Decode normalized latents without the LightTAE denormalization."""
        batch, views, frames, channels, height, width = input.shape
        latent = input.reshape(batch * views, frames, channels, height, width)
        rgb = self.taehv.decode(latent, cache=cache).mul(2).sub(1)
        return rgb.reshape(batch, views, *rgb.shape[1:])


def prepare_decoder(directory: Path, *, height: int, width: int, device: str) -> dict:
    """Export the matching FP32 decoder and return its wire descriptor."""
    identity = {
        "decoder_class": "taehv-wan22",
        "checkpoint": TAEW2_2_CHECKPOINT,
        "latent_shape": [1, 48, height // 16, width // 16],
        "video_dims": [width, height],
        "denormalize": False,
        "seed_required": True,
    }
    version = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:16]
    directory = directory / version
    path = directory / "fp32.onnx"
    spec_path = path.with_suffix(".spec.json")
    if not path.is_file() or not spec_path.is_file():
        decoder = TokenDecoder(device, torch.float32)
        spec = export_streaming_decoder(
            decoder, out_path=path, latent_frames=1, latent_channels=48,
            latent_height=height // 16, latent_width=width // 16,
            device=device, dtype=torch.float32, version=version,
        )
        import onnx

        onnx.checker.check_model(str(path))
        spec["identity"] = identity
        spec_path.write_text(json.dumps(spec, indent=2))
    spec = json.loads(spec_path.read_text())
    if spec.get("identity") != identity:
        raise ValueError("Decoder artifact identity does not match Cosmos3 settings")
    return {
        **spec,
        "kind": "taehv-cache-io-onnx",
        "cold_start": "zeros",
        "default_precision": "fp32",
        "precisions": {"fp32": f"/decoder/{version}/fp32.onnx"},
    }
