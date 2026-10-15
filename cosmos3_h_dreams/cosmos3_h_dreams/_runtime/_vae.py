# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Streaming Wan VAE encoding and Wan/TAEHV chunk decoding."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from flashdreams.recipes.wan.autoencoder.vae import (
    Wan22TI2V5BVAEDecoderConfig,
    Wan22TI2V5BVAEEncoderConfig,
)
from torch import Tensor


def wan22_vae_encoder_config(
    *,
    dtype: torch.dtype = torch.bfloat16,
    use_cuda_graph: bool = False,
    checkpoint_path: str | None = None,
) -> Wan22TI2V5BVAEEncoderConfig:
    """Wan 2.2 (48-channel) VAE encoder — the video tokenizer / first-frame I2V encode."""
    kwargs: dict[str, Any] = {"dtype": dtype, "use_cuda_graph": use_cuda_graph}
    if checkpoint_path is not None:
        kwargs["checkpoint_path"] = checkpoint_path
    return Wan22TI2V5BVAEEncoderConfig(**kwargs)


def wan22_vae_decoder_config(
    *,
    dtype: torch.dtype = torch.bfloat16,
    use_cuda_graph: bool = False,
    use_compile: bool = False,
    checkpoint_path: str | None = None,
) -> Wan22TI2V5BVAEDecoderConfig:
    """Full Wan 2.2 (48-channel) VAE decoder — the full latents->pixels path."""
    kwargs: dict[str, Any] = {
        "dtype": dtype,
        "use_cuda_graph": use_cuda_graph,
        "use_compile": use_compile,
    }
    if checkpoint_path is not None:
        kwargs["checkpoint_path"] = checkpoint_path
    return Wan22TI2V5BVAEDecoderConfig(**kwargs)


# madebyollin's Wan 2.2 (48-ch) tiny AE, mirrored on lightx2v/Autoencoders. This
# is the ``taew2_2`` decoder (github.com/madebyollin/taehv);
# ``lighttaew2_2.pth`` is lightx2v's further-distilled variant.
TAEW2_2_CHECKPOINT = (
    "https://huggingface.co/lightx2v/Autoencoders/resolve/main/taew2_2.pth"
)


def _infer_taehv_denormalize(checkpoint_path: str | None) -> bool:
    checkpoint = checkpoint_path or TAEW2_2_CHECKPOINT
    name = checkpoint.rsplit("/", 1)[-1].lower()
    if "lighttaew2_2" in name:
        return True
    if "taew2_2" in name:
        return False
    return True


def make_taehv_chunk_decoder(
    *,
    len_t: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_path: str | None = None,
    denormalize: bool | None = None,
    device: str | torch.device = "cuda",
) -> Callable[[Tensor], Tensor]:
    """Stateful per-chunk TAEHV decode for live camera/dream preview.

    This mirrors :func:`make_wan22_chunk_decoder`: one decoder cache is kept
    across calls so the tiny decoder sees a continuous autoregressive stream.
    Construct a fresh chunk decoder on reset or scene switch.
    """
    from flashdreams.recipes.taehv.impl import TAEHV
    from flashdreams.recipes.wan.autoencoder.vae import (
        _WAN22_TI2V_5B_LATENT_MEAN,
        _WAN22_TI2V_5B_LATENT_STD,
    )

    taehv = TAEHV(
        checkpoint_path=checkpoint_path or TAEW2_2_CHECKPOINT,
        model_type="wan22",
        use_cuda_graph=False,
        use_compile=False,
    )
    taehv = taehv.to(device=device, dtype=dtype).eval()
    dev = next(taehv.parameters()).device
    cache = taehv.prepare_cache()
    denormalize = (
        _infer_taehv_denormalize(checkpoint_path)
        if denormalize is None
        else bool(denormalize)
    )
    mean = torch.tensor(_WAN22_TI2V_5B_LATENT_MEAN, dtype=dtype, device=dev).view(
        1, 1, -1, 1, 1
    )
    std = torch.tensor(_WAN22_TI2V_5B_LATENT_STD, dtype=dtype, device=dev).view(
        1, 1, -1, 1, 1
    )

    def decode_chunk(latent_chunk: Tensor) -> Tensor:
        _c, t, _h, _w = latent_chunk.shape
        outs: list[Tensor] = []
        for start in range(0, t, len_t):
            z = (
                latent_chunk[:, start : start + len_t]
                .permute(1, 0, 2, 3)
                .unsqueeze(0)
                .to(dev, dtype)
            )
            if denormalize:
                z = z * std + mean
            px = taehv.decode(z, cache=cache).mul(2).sub(1)  # [1, T_pix, 3, H*16, W*16]
            outs.append(px[0].float())
        return torch.cat(outs, dim=0)

    return decode_chunk


def make_wan22_encode_fn(
    *,
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_path: str | None = None,
    device: str | torch.device = "cuda",
) -> Callable[[Tensor], Tensor]:
    """Build ``encode(image) -> latent[C, 1, H_lat, W_lat]`` (Wan 2.2, normalized space).

    ``image`` is an RGB tensor ``[H, W, 3]`` (uint8 0-255 or float in ``[-1, 1]``)
    or ``[3, H, W]``; a single pixel frame encodes to one latent frame. The output
    is mean/std-normalized — the same space the AR rollout emits and the decoder
    consumes — so it can seed frame 0 for image-to-video forward dynamics directly.
    """
    encoder = (
        wan22_vae_encoder_config(dtype=dtype, checkpoint_path=checkpoint_path)
        .setup()
        .to(device)
    )
    dev = next(encoder.parameters()).device

    def encode_fn(image: Tensor) -> Tensor:
        img = image
        if img.ndim == 3 and img.shape[-1] == 3:  # [H,W,3] -> [3,H,W]
            img = img.permute(2, 0, 1)
        if img.dtype == torch.uint8:
            img = img.float() / 127.5 - 1.0  # [-1, 1]
        img = img.to(device=dev, dtype=dtype)
        x = img[None, None]  # [batch=1, T=1, C=3, H, W]
        with torch.no_grad():
            z = encoder(x)  # [1, T_lat=1, z_dim, H_lat, W_lat]
        return (
            z[0].transpose(0, 1).contiguous()
        )  # [z_dim, T_lat, H_lat, W_lat] = [C, T, H, W]

    return encode_fn


def make_wan22_chunk_decoder(
    *,
    len_t: int = 1,
    dtype: torch.dtype = torch.bfloat16,
    checkpoint_path: str | None = None,
    device: str | torch.device = "cuda",
) -> Callable[[Tensor], Tensor]:
    """Stateful per-chunk Wan 2.2 decode: ``decode_chunk(latent[C, t, H, W]) -> pixels``.

    A persistent cache and running autoregressive index span calls. Frames
    within a chunk are driven ``len_t`` at a time through the causal decoder.
    """
    decoder = (
        wan22_vae_decoder_config(dtype=dtype, checkpoint_path=checkpoint_path)
        .setup()
        .to(device)
    )
    dev = next(decoder.parameters()).device
    cache = decoder.initialize_autoregressive_cache()
    state = {"ar_index": 0}

    def decode_chunk(latent_chunk: Tensor) -> Tensor:  # [C, t, H, W]
        t = int(latent_chunk.shape[1])
        outs: list[Tensor] = []
        for start in range(0, t, len_t):
            z = (
                latent_chunk[:, start : start + len_t]
                .permute(1, 0, 2, 3)
                .to(dev, dtype)
            )  # [len_t,C,H,W]
            px = decoder(
                z, autoregressive_index=state["ar_index"], cache=cache
            )  # [T_pix,3,H*16,W*16]
            outs.append(px.float())
            state["ar_index"] += 1
        return torch.cat(outs, dim=0)

    return decode_chunk
