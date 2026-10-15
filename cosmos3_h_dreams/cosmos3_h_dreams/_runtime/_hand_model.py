# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model ownership for the hand demo: checkpoint, VAE, and rollout sessions.

Holds the single GPU model. Decoding and starting a session take a lock, because
both touch the one decoder cache and a new session resets it. Generation does
not: it runs on the single live loop thread, so there is nothing to serialize it
against, and holding the lock across a whole step would block decoding. The HTTP
layer stays concurrent and only touches this through :class:`HandModel`.
"""

from __future__ import annotations

import math
import threading
import time

import numpy as np
import torch
from flashdreams.serving.realtime.media import rgb_array_to_uint8_frames
from torch import Tensor

from cosmos3_h_dreams._runtime import _agibot
from cosmos3_h_dreams._runtime._compile_modes import normalize_compile_denoise_mode
from cosmos3_h_dreams._runtime._hand_kv_session import (
    DEFAULT_WINDOW_FRAMES,
    HandKVSession,
    distilled_t_list,
    validate_cache_options,
)
from cosmos3_h_dreams._runtime._hand_scenes import Scene
from cosmos3_h_dreams._runtime._patchify import (
    crop_latent_height,
    crop_latent_width,
    pad_latent_bottom,
    pad_latent_right,
    patchify_latent,
    unpatchify_latent,
)
from cosmos3_h_dreams._runtime._spec import SPEC
from cosmos3_h_dreams._runtime._taehv2super import (
    TAEHV2SuperChunkDecoder,
    make_taehv2super_chunk_decoder,
)
from cosmos3_h_dreams._runtime._vae import (
    make_taehv_chunk_decoder,
    make_wan22_chunk_decoder,
    make_wan22_encode_fn,
)


class HandModel:
    """The AgiBot network plus its VAE, sized to the checkpoint's geometry.

    Args:
        checkpoint: Explicit checkpoint path or URI. Falls back to the
            ``COSMOS3_H_DREAMS_CHECKPOINT`` environment variable.
        device: Torch device for the network and VAE.
        seed: Canonical-application alias for ``rollout_seed``.
        chunk_size: Latent frames generated per rollout step.
        denoise_steps: Checkpoint-trained fixed-step schedule length.
        decoder: ``wan`` for quality, ``taehv`` for the tiny public decoder,
            or ``taehv2super`` for the higher-quality fast decoder.
        credential_path: Optional credentials for fetching the checkpoint.
        window_frames: Frames of generated history each session retains.
        native_gqa_attention: Let SDPA handle grouped-query attention.
        sdpa_backend: Attention kernel to pin, or ``"default"``.
        cache_mode: ``block`` uses a ring buffer; ``cat`` regrows and slices;
            ``fademem`` consolidates clean frame entries.
        cache_refresh_mode: How clean frames are written back over the chunk.
        fademem_slots: Historical frame-entry budget, including frame zero.
        fademem_beta: FadeMem power-law age exponent in ``(0, 1]``.
        fademem_merge_selection: Adjacent-pair ranking, ``age`` or latent ``cosine``.
        overlap_cache_update: Write clean K/V back on a side CUDA stream.
        compile_denoise: ``torch.compile`` the velocity call.
        compile_denoise_mode: torch.compile mode for that call.
        fixed_shape_cache_fill: Use static masked history while the cache fills.
        valid_length_cache_attention: Use one fused fixed-buffer attention path
            across cache fill and steady state.
        cuda_graph_denoise: Capture the compiled velocity call into a CUDA graph.
        cuda_graph_warmup_iters: Eager calls before graph capture.
        profile: Record per-stage rollout timings, which costs a few device
            synchronizations per step.
        model_fps: Optional frame rate used only for the model's temporal mRoPE
            conditioning. ``None`` preserves the scene's recorded frame rate.
        rollout_seed: Base seed for rollout noise. Each autoregressive chunk and
            denoising step derives its deterministic seed from this value.
    """

    def __init__(
        self,
        *,
        checkpoint: str | None = None,
        device: str = "cuda",
        seed: int | None = None,
        chunk_size: int = 4,
        denoise_steps: int = 4,
        decoder: str = "wan",
        credential_path: str | None = None,
        window_frames: int = DEFAULT_WINDOW_FRAMES,
        native_gqa_attention: bool = False,
        sdpa_backend: str = "default",
        cache_mode: str = "cat",
        cache_refresh_mode: str = "chunk",
        fademem_slots: int = 12,
        fademem_beta: float = 0.3,
        fademem_merge_selection: str = "age",
        overlap_cache_update: bool = False,
        compile_denoise: bool = False,
        compile_denoise_mode: str = "max-autotune-no-cudagraphs",
        fixed_shape_cache_fill: bool = False,
        valid_length_cache_attention: bool = False,
        cuda_graph_denoise: bool = False,
        cuda_graph_warmup_iters: int = 2,
        profile: bool = False,
        model_fps: float | None = None,
        rollout_seed: int = 0,
    ) -> None:
        from cosmos3_h_dreams._runtime.checkpoint_load import (
            load_exported_checkpoint,
            resolve_checkpoint_source,
            resolve_credential_path,
        )
        from cosmos3_h_dreams._runtime.fetch_checkpoint import ensure_checkpoint

        self.device = device
        self.chunk_size = int(chunk_size)
        self.denoise_steps = int(denoise_steps)
        self.t_list = distilled_t_list(self.denoise_steps)
        self.window_frames = int(window_frames)
        self.native_gqa_attention = bool(native_gqa_attention)
        self.sdpa_backend = str(sdpa_backend)
        self.cache_mode = str(cache_mode)
        self.cache_refresh_mode = str(cache_refresh_mode)
        self.fademem_slots = fademem_slots
        self.fademem_beta = fademem_beta
        self.fademem_merge_selection = fademem_merge_selection
        self.overlap_cache_update = bool(overlap_cache_update)
        self.compile_denoise = bool(compile_denoise)
        self.compile_denoise_mode = str(compile_denoise_mode)
        self.fixed_shape_cache_fill = bool(fixed_shape_cache_fill)
        self.valid_length_cache_attention = bool(valid_length_cache_attention)
        self.cuda_graph_denoise = bool(cuda_graph_denoise)
        self.cuda_graph_warmup_iters = int(cuda_graph_warmup_iters)
        self.profile = bool(profile)
        self.model_fps = None if model_fps is None else float(model_fps)
        if self.model_fps is not None and (
            not math.isfinite(self.model_fps) or self.model_fps <= 0.0
        ):
            raise ValueError("model_fps must be finite and greater than zero")
        if seed is not None and rollout_seed != 0 and int(seed) != int(rollout_seed):
            raise ValueError("seed and rollout_seed must match when both are provided")
        self.rollout_seed = int(rollout_seed if seed is None else seed)
        # Checked before the checkpoint loads, so a bad command line fails in
        # seconds instead of when the operator presses start.
        validate_cache_options(
            cache_mode=self.cache_mode,
            fademem_slots=self.fademem_slots,
            fademem_beta=self.fademem_beta,
            fademem_merge_selection=self.fademem_merge_selection,
            cache_refresh_mode=self.cache_refresh_mode,
            window_frames=self.window_frames,
            overlap_cache_update=self.overlap_cache_update,
            compile_denoise=self.compile_denoise,
            fixed_shape_cache_fill=self.fixed_shape_cache_fill,
            valid_length_cache_attention=self.valid_length_cache_attention,
            cuda_graph_denoise=self.cuda_graph_denoise,
            device=self.device,
        )
        normalize_compile_denoise_mode(self.compile_denoise_mode)
        if decoder not in {"wan", "taehv", "taehv2super"}:
            raise ValueError(f"unknown decoder {decoder!r}")
        self.tcf = _agibot.TEMPORAL_COMPRESSION_FACTOR
        self.grid_hw = _agibot.TOKEN_GRID
        self.latent_hw = (_agibot.LATENT_HEIGHT, _agibot.LATENT_WIDTH)
        self.width = _agibot.CONTENT_WIDTH
        self.height = _agibot.CONTENT_HEIGHT
        self.patch = SPEC.vfm.latent_patch_size

        source = ensure_checkpoint(
            resolve_checkpoint_source(checkpoint), credential_path=credential_path
        )
        self.net = load_exported_checkpoint(
            source,
            device=device,
            credential_path=resolve_credential_path(credential_path),
        ).eval()
        if not getattr(self.net.config, "action_gen", False):
            raise SystemExit(
                "checkpoint has no action pathway; the hand demo needs the AgiBot "
                "action-conditioned checkpoint"
            )

        self._encode = make_wan22_encode_fn(device=device)
        self._decoder_kind = decoder
        self._credential_path = credential_path
        self._decode_chunk = self._new_chunk_decoder()
        self._lock = threading.Lock()
        # A curated demo catalog is small, and each session owns scene-specific
        # KV/cache and CUDA-graph state. Keep one reset rollout per scene so a
        # participant can switch scenes without paying that warmup in the UX.
        self._prepared_sessions: dict[str, HandKVSession] = {}

    def _new_chunk_decoder(self):
        """Build a fresh stateful decoder; its cache must not span scenes."""
        if self._decoder_kind == "wan":
            return make_wan22_chunk_decoder(len_t=1, device=self.device)
        if self._decoder_kind == "taehv":
            return make_taehv_chunk_decoder(len_t=1, device=self.device)
        return make_taehv2super_chunk_decoder(
            device=self.device,
            credential_path=self._credential_path,
        )

    def _reset_chunk_decoder(self) -> None:
        """Reset temporal state while retaining reusable decoder weights."""
        if isinstance(self._decode_chunk, TAEHV2SuperChunkDecoder):
            self._decode_chunk.reset()
        else:
            self._decode_chunk = self._new_chunk_decoder()

    def patchify_encoded(
        self,
        latent: Tensor,
        *,
        content_height: int,
        content_width: int,
    ) -> Tensor:
        """Crop a VAE latent to the logical content region, then patchify it."""
        logical = _agibot.logical_latent_geometry(
            content_height, content_width, expected=self.latent_hw
        )
        latent = latent[..., : logical[0], : logical[1]].contiguous()
        padded = pad_latent_bottom(latent, self.grid_hw[0] * self.patch - logical[0])
        padded = pad_latent_right(padded, self.grid_hw[1] * self.patch - logical[1])
        return patchify_latent(padded, self.patch)

    def encode(self, image: Tensor) -> Tensor:
        """Encode one scene image without overlapping model preparation."""
        with self._lock:
            return self._encode(image)

    def _new_session(self, scene: Scene) -> HandKVSession:
        return HandKVSession(
            self.net,
            text_ids=scene.text_ids,
            init_latent=scene.init_latent,
            grid_hw=self.grid_hw,
            chunk_size=self.chunk_size,
            tcf=self.tcf,
            t_list=self.t_list,
            action_domain_id=_agibot.AGIBOT_DOMAIN_ID,
            fps=scene.fps if self.model_fps is None else self.model_fps,
            seed=self.rollout_seed,
            logical_latent_hw=self.latent_hw,
            window_frames=self.window_frames,
            native_gqa_attention=self.native_gqa_attention,
            sdpa_backend=self.sdpa_backend,
            cache_mode=self.cache_mode,
            fademem_slots=self.fademem_slots,
            fademem_beta=self.fademem_beta,
            fademem_merge_selection=self.fademem_merge_selection,
            cache_refresh_mode=self.cache_refresh_mode,
            overlap_cache_update=self.overlap_cache_update,
            compile_denoise=self.compile_denoise,
            compile_denoise_mode=self.compile_denoise_mode,
            fixed_shape_cache_fill=self.fixed_shape_cache_fill,
            valid_length_cache_attention=self.valid_length_cache_attention,
            cuda_graph_denoise=self.cuda_graph_denoise,
            cuda_graph_warmup_iters=self.cuda_graph_warmup_iters,
        )

    def prepare_session(self, scene: Scene, steps: int) -> None:
        """Prepare and retain a reset rollout for a future live session."""
        steps = max(0, int(steps))
        if steps == 0:
            return
        with self._lock:
            if scene.name in self._prepared_sessions:
                return
            print(
                f"[prewarm][hand:{scene.name}] running {steps} hidden chunk(s)...",
                flush=True,
            )
            started = time.perf_counter()
            session = self._new_session(scene)
            timings = session.prewarm_rollout(steps, profile=True)
            for index, timing in enumerate(timings, start=1):
                print(
                    f"[prewarm][hand:{scene.name}] step {index}/{steps}: "
                    f"model={timing.get('model_step_s', 0.0) * 1e3:.0f}ms "
                    f"denoise={timing.get('denoise_s', 0.0) * 1e3:.0f}ms "
                    f"cache={timing.get('cache_seed_prune_s', 0.0) * 1e3:.0f}ms "
                    f"graph={timing.get('cuda_graph_captured', 0.0):.0f}",
                    flush=True,
                )
            self._prepared_sessions[scene.name] = session
            print(
                f"[prewarm][hand:{scene.name}] ready in "
                f"{time.perf_counter() - started:.1f}s; rollout reset to frame 0",
                flush=True,
            )

    def new_session(self, scene: Scene) -> HandKVSession:
        """Start a rollout for ``scene`` and reset the decoder cache with it."""
        with self._lock:
            self._reset_chunk_decoder()
            session = self._prepared_sessions.pop(scene.name, None)
            if session is None:
                session = self._new_session(scene)
            # Frame 0 primes the causal decoder and is then discarded: the client
            # already holds it as the init frame.
            self._decode_latent(self._frame_zero_latent(scene))
            return session

    def release_session(self, scene: Scene, session: HandKVSession) -> None:
        """Reset and retain a stopped rollout for the next matching attempt."""
        with self._lock:
            session.reset_rollout()
            self._prepared_sessions[scene.name] = session

    def decode(self, latent: Tensor) -> list[np.ndarray]:
        """Decode one latent frame into presentation RGB frames."""
        with self._lock:
            return self._decode_latent(latent)

    def decode_tensor(self, latent: Tensor) -> Tensor:
        """Decode latent frames into a cropped ``[T,C,H,W]`` pixel tensor."""
        with self._lock:
            return self._decode_tensor_unlocked(latent)

    def decode_profiled(
        self, latent: Tensor
    ) -> tuple[list[np.ndarray], dict[str, float]]:
        """Decode with synchronized GPU-compute and host-materialization timing."""
        with self._lock:
            cropped = self._crop_decode_latent(latent)
            self._sync_device()
            started = time.perf_counter()
            pixels = self._crop_decode_pixels(self._decode_chunk(cropped))
            self._sync_device()
            decoded_at = time.perf_counter()
            frames = rgb_array_to_uint8_frames(
                pixels, layout="tchw", value_range="minus_one_one"
            )
            finished = time.perf_counter()
        return frames, {
            "decode_compute_s": decoded_at - started,
            "frame_materialize_s": finished - decoded_at,
            "decode_s": finished - started,
        }

    def _frame_zero_latent(self, scene: Scene) -> Tensor:
        return unpatchify_latent(
            scene.init_latent.to(self.device),
            channels=SPEC.vfm.vae_latent_channels,
            grid_t=1,
            grid_h=self.grid_hw[0],
            grid_w=self.grid_hw[1],
            patch_spatial=self.patch,
        )

    def _decode_latent(self, latent: Tensor) -> list[np.ndarray]:
        pixels = self._decode_tensor_unlocked(latent)
        return rgb_array_to_uint8_frames(
            pixels, layout="tchw", value_range="minus_one_one"
        )

    def _decode_tensor_unlocked(self, latent: Tensor) -> Tensor:
        """Decode while the caller owns ``self._lock``."""
        return self._crop_decode_pixels(
            self._decode_chunk(self._crop_decode_latent(latent))
        )

    def _crop_decode_latent(self, latent: Tensor) -> Tensor:
        """Crop a generated latent to the decoder's logical content canvas."""
        return crop_latent_width(
            crop_latent_height(latent, self.latent_hw[0]), self.latent_hw[1]
        )

    def _crop_decode_pixels(self, pixels: Tensor) -> Tensor:
        """Remove the partial VAE cell below the exact 640x360 content."""
        return pixels[..., : self.height, : self.width].contiguous()

    def _sync_device(self) -> None:
        """Synchronize this model's CUDA device for profiling boundaries."""
        device = torch.device(self.device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
