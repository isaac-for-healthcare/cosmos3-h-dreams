# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming TAEHV2 Super decoder for normalized Wan 2.2 latents."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

DecoderCache = tuple[Tensor, ...]


def _conv(in_channels: int, out_channels: int) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, padding=1)


class _SuperMemoryBlock(nn.Module):
    """Causal temporal residual block used by TAEHV2 Super."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        joined_channels = channels * 2
        self.conv = nn.Sequential(
            nn.Conv2d(
                joined_channels,
                joined_channels,
                7,
                padding=3,
                groups=joined_channels,
                bias=False,
            ),
            nn.Conv2d(joined_channels, channels * 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels * 4, channels, 1, bias=False),
        )

    def forward(self, current: Tensor, past: Tensor) -> Tensor:
        return self.conv(torch.cat([current, past], dim=1)) + current


class _TemporalGrow(nn.Module):
    """Increase temporal resolution by reshaping channels into frames."""

    def __init__(self, channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv2d(channels, channels * stride, 1, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        return x.reshape(
            -1,
            x.shape[1] // self.stride,
            x.shape[2],
            x.shape[3],
        )


class TAEHV2SuperDecoder(nn.Module):
    """TAEHV2 Super decoder with cache-explicit autoregressive streaming."""

    _dtype_reference: Tensor
    """Zero-size buffer carrying the decoder parameter dtype and device."""

    latent_channels = 48
    image_channels = 3
    patch_size = 2
    spatial_upsample_factor = 2
    temporal_upsample_factors = (1, 2, 2)
    memory_blocks_per_stage = 3
    stage_channels = (512, 256, 128, 64)

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        credential_path: str | None = None,
    ) -> None:
        super().__init__()
        self.decoder_output_channels = self.image_channels * self.patch_size**2
        self.temporal_upsample_factor = 1
        for factor in self.temporal_upsample_factors:
            self.temporal_upsample_factor *= factor
        self.frames_to_trim = self.temporal_upsample_factor - 1
        self.register_buffer("_dtype_reference", torch.empty(0))
        self.decoder = self._build_decoder()
        self._decoder_cache: DecoderCache | None = None
        self._has_decoded_prefix = False

        if checkpoint_path is not None:
            checkpoint = self._load_checkpoint(checkpoint_path, credential_path)
            decoder_state = {
                key.removeprefix("decoder."): value
                for key, value in checkpoint.items()
                if key.startswith("decoder.")
            }
            if not decoder_state:
                raise ValueError(
                    f"TAEHV2 Super checkpoint has no decoder weights: {checkpoint_path}"
                )
            self.decoder.load_state_dict(decoder_state, strict=True)

    @staticmethod
    def _load_checkpoint(
        checkpoint_path: str | Path,
        credential_path: str | None,
    ) -> dict[str, Tensor]:
        """Load and validate a TAEHV2 Super checkpoint state dictionary."""
        from flashdreams.core.checkpoint.load import load_single_checkpoint

        source = str(checkpoint_path)
        if credential_path is None:
            checkpoint = load_single_checkpoint(source, map_location="cpu")
        else:
            checkpoint = load_single_checkpoint(
                source,
                credential_path=credential_path,
                map_location="cpu",
            )
        if type(checkpoint) is not dict or not all(
            isinstance(key, str) and isinstance(value, Tensor)
            for key, value in checkpoint.items()
        ):
            raise ValueError(
                f"TAEHV2 Super checkpoint has an invalid state-dict format: {source}"
            )
        return checkpoint

    def reset(self) -> None:
        """Clear temporal state at an independent-video boundary."""
        self._decoder_cache = None
        self._has_decoded_prefix = False

    def _append_decoder_stage(
        self,
        layers: list[nn.Module],
        in_channels: int,
        out_channels: int,
        temporal_upsample_factor: int,
    ) -> None:
        """Append one checkpoint-compatible Super decoder stage."""
        layers.extend(
            _SuperMemoryBlock(in_channels) for _ in range(self.memory_blocks_per_stage)
        )
        shuffle_channels = out_channels * self.spatial_upsample_factor**2
        layers.extend(
            [
                _conv(in_channels, shuffle_channels),
                nn.ReLU(inplace=True),
                nn.PixelShuffle(self.spatial_upsample_factor),
                _TemporalGrow(out_channels, temporal_upsample_factor),
            ]
        )

    def _build_decoder(self) -> nn.Sequential:
        """Build the checkpoint-compatible TAEHV2 Super decoder topology."""
        layers: list[nn.Module] = [
            nn.Conv2d(
                self.latent_channels,
                self.stage_channels[0],
                1,
                bias=False,
            )
        ]
        transitions = zip(
            self.stage_channels[:-1],
            self.stage_channels[1:],
            self.temporal_upsample_factors,
            strict=True,
        )
        for in_channels, out_channels, temporal_factor in transitions:
            self._append_decoder_stage(
                layers,
                in_channels,
                out_channels,
                temporal_factor,
            )
        layers.append(_conv(self.stage_channels[-1], self.decoder_output_channels))
        return nn.Sequential(*layers)

    def _decode_chunk(
        self,
        latent: Tensor,
        decoder_cache: DecoderCache | None,
    ) -> tuple[Tensor, DecoderCache]:
        """Decode a chunk as a functional cache-in/cache-out operation."""
        batch, _, frames, height, width = latent.shape
        x = latent.permute(0, 2, 1, 3, 4).reshape(
            batch * frames,
            self.latent_channels,
            height,
            width,
        )
        cache_index = 0
        next_cache: list[Tensor] = []
        for layer in self.decoder:
            if isinstance(layer, _SuperMemoryBlock):
                sequence = x.reshape(
                    batch,
                    -1,
                    x.shape[1],
                    x.shape[2],
                    x.shape[3],
                )
                if decoder_cache is None:
                    cached_frame = torch.zeros_like(sequence[:, :1])
                else:
                    cached_frame = decoder_cache[cache_index]
                past = torch.cat([cached_frame, sequence], dim=1)[:, :-1]
                past = past.reshape_as(x)
                next_cache.append(sequence[:, -1:].detach())
                cache_index += 1
                x = layer(x, past)
            else:
                x = layer(x)

        x = x.reshape(
            batch,
            -1,
            self.decoder_output_channels,
            x.shape[2],
            x.shape[3],
        )
        x = F.pixel_shuffle(x.flatten(0, 1), self.patch_size)
        x = x.reshape(batch, -1, self.image_channels, x.shape[2], x.shape[3])
        return x.permute(0, 2, 1, 3, 4), tuple(next_cache)

    def forward(self, latent: Tensor, *, reset: bool = True) -> Tensor:
        """Decode a full latent video or the next streamed latent chunk."""
        if latent.ndim != 5 or latent.shape[1] != self.latent_channels:
            raise ValueError(
                f"expected latent [B,{self.latent_channels},T,H,W], "
                f"got {tuple(latent.shape)}"
            )
        if reset:
            self.reset()
        output_dtype = latent.dtype
        latent = latent.to(self._dtype_reference)
        pixels, self._decoder_cache = self._decode_chunk(
            latent,
            self._decoder_cache,
        )
        trim = self.frames_to_trim if not self._has_decoded_prefix else 0
        self._has_decoded_prefix = True
        return pixels[:, :, trim:].clamp_(0, 1).to(output_dtype)


class TAEHV2SuperChunkDecoder:
    """Stateful hand-demo adapter around one loaded TAEHV2 Super decoder."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        credential_path: str | None = None,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        decoder: TAEHV2SuperDecoder | None = None,
    ) -> None:
        if decoder is None:
            decoder = TAEHV2SuperDecoder(
                checkpoint_path,
                credential_path=credential_path,
            ).to(device=device, dtype=dtype)
            decoder.eval().requires_grad_(False)
        self.decoder = decoder
        self.device = next(decoder.parameters()).device
        self.dtype = next(decoder.parameters()).dtype

    def reset(self) -> None:
        """Clear cached temporal features without reloading decoder weights."""
        self.decoder.reset()

    @torch.inference_mode()
    def __call__(self, latent_chunk: Tensor) -> Tensor:
        """Decode normalized ``[C,T,H,W]`` latents to ``[-1,1]`` RGB frames."""
        latent = latent_chunk.unsqueeze(0).to(
            device=self.device,
            dtype=self.dtype,
        )
        pixels = self.decoder(latent, reset=False)
        return pixels[0].permute(1, 0, 2, 3).mul(2).sub(1).float()


def make_taehv2super_chunk_decoder(
    *,
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_path: str | Path | None = None,
    credential_path: str | None = None,
    device: str | torch.device = "cuda",
) -> TAEHV2SuperChunkDecoder:
    """Build the cached TAEHV2 Super decoder used by live hand sessions."""
    from cosmos3_h_dreams._runtime.checkpoint_load import resolve_credential_path

    source = str(
        checkpoint_path
        or os.environ.get("COSMOS3_H_DREAMS_TAEHV2SUPER_CHECKPOINT")
        or ""
    )
    if not source:
        raise ValueError(
            "taehv2super requires COSMOS3_H_DREAMS_TAEHV2SUPER_CHECKPOINT "
            "or an explicit decoder checkpoint path"
        )
    credential = resolve_credential_path(
        credential_path,
        checkpoint_source=source,
    )
    if source.startswith("s3://") and credential is None:
        raise SystemExit(
            "TAEHV2 Super S3 loading needs credentials. Pass --credential-path, put "
            "gcp_checkpoint.secret under credentials/, or set "
            "$COSMOS_CHECKPOINT_CREDENTIAL."
        )
    print(f"[vae] decoder=taehv2super checkpoint={source}", flush=True)
    return TAEHV2SuperChunkDecoder(
        checkpoint_path=source,
        credential_path=credential,
        device=device,
        dtype=dtype,
    )
