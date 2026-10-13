# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Resolved Cosmos3-H-Dreams replay and Quest configuration."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

from cosmos_h_dreams._runtime._compile_modes import COMPILE_DENOISE_MODES
from cosmos_h_dreams._runtime._hand_kv_session import validate_cache_options

COSMOS_H_DOMAIN_ID = 32

"""Default text condition for CMR replay and Quest sessions."""
DEFAULT_PROMPT = (
    "Endoscopic view of a dual-arm dVRK robot manipulating "
    "surgical sutures on a tabletop."
)

"""CMR checkpoint settings with the eager cache path."""
BASE: dict[str, object] = {
    "domain_id": 25,
    "prompt": DEFAULT_PROMPT,
    "device": "cuda:0",
    "fps": 10.0,
    "height": 480,
    "width": 832,
    "seed": 0,
    "chunk_size": 4,
    "denoise_steps": 4,
    "encoder": "wan",
    "decoder": "wan",
    "window_frames": 12,
    "cache_mode": "cat",
    "compile_denoise": False,
    "compile_denoise_mode": "max-autotune-no-cudagraphs",
    "cuda_graph_denoise": False,
    "cuda_graph_warmup_iters": 2,
    "overlap_cache_update": False,
    "profile": False,
    "display_fps": 30.0,
}

"""Opt-in block-cache and compiled denoising settings for CMR measurements."""
WAN_WAN_CHUNK4_BLOCK_COMPILE: dict[str, object] = {
    **BASE,
    "cache_mode": "block",
    "compile_denoise": True,
    "compile_denoise_mode": "default",
    "cuda_graph_denoise": True,
    "overlap_cache_update": True,
}

WAN_TAEHV_CHUNK4_BLOCK_COMPILE: dict[str, object] = {
    **WAN_WAN_CHUNK4_BLOCK_COMPILE,
    "decoder": "taehv",
}

WAN_WAN_CHUNK3_BLOCK_COMPILE: dict[str, object] = {
    **WAN_WAN_CHUNK4_BLOCK_COMPILE,
    "chunk_size": 3,
}

WAN_TAEHV_CHUNK3_BLOCK_COMPILE: dict[str, object] = {
    **WAN_TAEHV_CHUNK4_BLOCK_COMPILE,
    "chunk_size": 3,
}

WAN_WAN_CHUNK2_BLOCK_COMPILE: dict[str, object] = {
    **WAN_WAN_CHUNK4_BLOCK_COMPILE,
    "chunk_size": 2,
}

WAN_TAEHV_CHUNK2_BLOCK_COMPILE: dict[str, object] = {
    **WAN_TAEHV_CHUNK4_BLOCK_COMPILE,
    "chunk_size": 2,
}

WAN_TAEHV_CHUNK2_FADEMEM_COMPILE: dict[str, object] = {
    **WAN_TAEHV_CHUNK2_BLOCK_COMPILE,
    "cache_mode": "fademem",
    "fademem_merge_selection": "cosine",
}

WAN_TAEHV_CHUNK2_2STEPS_FADEMEM_COMPILE: dict[str, object] = {
    **WAN_TAEHV_CHUNK2_FADEMEM_COMPILE,
    "denoise_steps": 2,
}


"""Named CLI presets; action source and output options stay on the CLI."""
COSMOS_H_DREAMS_PRESETS: dict[str, dict[str, object]] = {
    "base": BASE,
    "wan-wan-chunk4-block-compile": WAN_WAN_CHUNK4_BLOCK_COMPILE,
    "wan-taehv-chunk4-block-compile": WAN_TAEHV_CHUNK4_BLOCK_COMPILE,
    "wan-wan-chunk3-block-compile": WAN_WAN_CHUNK3_BLOCK_COMPILE,
    "wan-taehv-chunk3-block-compile": WAN_TAEHV_CHUNK3_BLOCK_COMPILE,
    "wan-wan-chunk2-block-compile": WAN_WAN_CHUNK2_BLOCK_COMPILE,
    "wan-taehv-chunk2-block-compile": WAN_TAEHV_CHUNK2_BLOCK_COMPILE,
    "wan-taehv-chunk2-fademem-compile": WAN_TAEHV_CHUNK2_FADEMEM_COMPILE,
    "wan-taehv-chunk2-2steps-fademem-compile": WAN_TAEHV_CHUNK2_2STEPS_FADEMEM_COMPILE,
}


@dataclass(frozen=True, kw_only=True, slots=True)
class CosmosHDreamsConfig:
    """Inputs and model options for one Cosmos3-H-Dreams rollout."""

    checkpoint: str
    """Checkpoint directory, index, registry name, or remote URI."""

    initial_frame: Path
    """Image establishing latent frame zero."""

    prompt: str
    """Text description conditioning the generated rollout."""

    actions: Path | None = None
    """NumPy replay containing ``[N, 28]`` actions, if replay is selected."""

    domain_id: int = COSMOS_H_DOMAIN_ID
    """Checkpoint embodiment row selected by the domain-aware action projection."""

    normalizer: Path | None = None
    """Optional ``.npz`` containing 28D ``offset`` and ``scale`` arrays."""

    quest_transport: str = "webxr"
    """Quest controller transport: hosted ``webxr``."""

    quest_relay_url: str | None = None
    """Optional external WebXR controller relay URL."""

    quest_port: int = 8444
    """Local port used by the managed Quest controller transport."""

    quest_startup_timeout_s: float = 300.0
    """Maximum wait for the first tracked controller pair."""

    quest_manual_xr: bool = False
    """Leave entry into hosted WebXR immersive mode to the operator."""

    quest_jaw_open: float = 1.0
    """Model-space jaw target sent while the index trigger is released."""

    quest_jaw_closed: float = -2.0
    """Model-space jaw target sent while the index trigger is clicked."""

    credential_path: str | None = None
    """Optional credential used to fetch a remote checkpoint."""

    device: str = "cuda:0"
    """Torch device hosting the model and decoder."""

    fps: float = 15.0
    """Action and generated-video frame rate used for temporal mRoPE."""

    height: int = 368
    """Generated content height and VAE input height in pixels."""

    width: int = 640
    """Generated content width and VAE input width in pixels."""

    seed: int = 0
    """Base seed for deterministic rollout noise."""

    chunk_size: int = 4
    """Maximum latent frames generated by one model step."""

    denoise_steps: int = 4
    """Number of checkpoint-distilled denoising steps."""

    encoder: str = "wan"
    """Initial-frame encoder; only the Wan encoder is currently supported."""

    decoder: str = "wan"
    """Decoder implementation: ``wan``, ``taehv``, or ``taehv2super``."""

    window_frames: int = 12
    """Generated latent frames retained in the rolling attention history."""

    cache_mode: str = "cat"
    """Cache mechanism: ``cat``, ``block``, or bounded ``fademem`` memory."""

    fademem_slots: int = 12
    """Historical latent-frame entries, including frame zero, for FadeMem."""

    fademem_beta: float = 0.3
    """FadeMem power-law distance exponent in (0, 1]."""

    fademem_merge_selection: str = "age"
    """Adjacent-pair ranking: ``age`` or whole clean-latent ``cosine``."""

    overlap_cache_update: bool = False
    """Update fixed-buffer caches on a side CUDA stream."""

    compile_denoise: bool = False
    """Compile the fixed-shape denoising call with ``torch.compile``."""

    compile_denoise_mode: str = "max-autotune-no-cudagraphs"
    """Compilation mode passed to the shared denoising runtime."""

    cuda_graph_denoise: bool = False
    """Capture the compiled denoising call in a CUDA graph."""

    cuda_graph_warmup_iters: int = 2
    """Number of CUDA graph warmup calls before capture."""

    profile: bool = False
    """Record synchronized per-stage rollout measurements."""

    display_fps: float = 30.0
    """Display frame rate for the Cosmos3-H-Dreams viewer."""

    def __post_init__(self) -> None:
        if not self.initial_frame.is_file():
            raise FileNotFoundError(
                f"Cosmos3-H-Dreams input does not exist: {self.initial_frame}"
            )
        if self.actions is not None and not self.actions.is_file():
            raise FileNotFoundError(
                f"Cosmos3-H-Dreams input does not exist: {self.actions}"
            )
        if self.normalizer is not None and not self.normalizer.is_file():
            raise FileNotFoundError(
                f"Cosmos3-H-Dreams normalizer does not exist: {self.normalizer}"
            )
        if self.domain_id < 0:
            raise ValueError("domain_id must be non-negative")
        if not self.prompt.strip():
            raise ValueError("prompt must not be empty")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        if self.quest_transport != "webxr":
            raise ValueError(f"unknown Quest transport {self.quest_transport!r}")
        if not 1 <= self.quest_port <= 65535:
            raise ValueError("quest_port must be between 1 and 65535")
        if self.quest_startup_timeout_s <= 0:
            raise ValueError("quest_startup_timeout_s must be positive")
        if not (
            isfinite(self.quest_jaw_closed)
            and isfinite(self.quest_jaw_open)
            and self.quest_jaw_closed < self.quest_jaw_open
        ):
            raise ValueError("quest jaw values must be finite with closed < open")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if self.height % 16 or self.width % 16:
            raise ValueError("height and width must be divisible by 16")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 1 <= self.denoise_steps <= 4:
            raise ValueError("denoise_steps must be between 1 and 4")
        if self.window_frames <= 0:
            raise ValueError("window_frames must be positive")
        if self.encoder != "wan":
            raise ValueError(
                f"unknown encoder {self.encoder!r}; only 'wan' is supported"
            )
        if self.decoder not in {"wan", "taehv", "taehv2super"}:
            raise ValueError(f"unknown decoder {self.decoder!r}")
        if self.cache_mode not in {"cat", "block", "fademem"}:
            raise ValueError(f"unknown cache mode {self.cache_mode!r}")
        validate_cache_options(
            cache_mode=self.cache_mode,
            fademem_slots=self.fademem_slots,
            fademem_beta=self.fademem_beta,
            fademem_merge_selection=self.fademem_merge_selection,
            window_frames=self.window_frames,
            overlap_cache_update=self.overlap_cache_update,
            compile_denoise=self.compile_denoise,
        )
        if self.compile_denoise_mode not in COMPILE_DENOISE_MODES:
            raise ValueError(
                f"unknown compile denoise mode {self.compile_denoise_mode!r}"
            )
        if self.cuda_graph_denoise and not self.compile_denoise:
            raise ValueError("cuda_graph_denoise requires compile_denoise=True")
        if self.cuda_graph_warmup_iters < 0:
            raise ValueError("cuda_graph_warmup_iters must be non-negative")
        if self.display_fps <= 0:
            raise ValueError("display_fps must be positive")
