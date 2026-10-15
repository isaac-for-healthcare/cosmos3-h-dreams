# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stateful KV-cached AgiBot rollout for live hand puppeteering.

One :meth:`HandKVSession.step` consumes ``chunk_size * tcf`` action rows and
returns ``chunk_size`` clean latent frames::

    prefill text -> seed frame 0 (the I2V start image) -> per step:
      denoise chunk_size frames against the cache, driven by the action chunk
      -> seed the finalized clean frames' K/V -> optionally prune to the window.

Frame 0 is retained as an attention sink. Cat and block caches retain absolute
positions. FadeMem consolidates clean frame entries after removing temporal
RoPE, then prepares history and current Q/K in a shared bounded layout.

The token layout is the AgiBot one: each frame contributes a ``[action(tcf),
vision(hw)]`` supertoken, denoising happens in the logical latent rather than in
token space, and the chunk shares a single uniform timestep.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

if TYPE_CHECKING:
    from cosmos3_h_dreams._runtime._network import (
        CosmosMoTNetwork,
        _MoTLayer,
        _MoTStreamCache,
    )

from flashdreams.infra.cuda_graph import CUDAGraphWrapper

from cosmos3_h_dreams._runtime import _agibot
from cosmos3_h_dreams._runtime._attention import (
    sdpa_supports_native_gqa,
    varlen_attention_available,
)
from cosmos3_h_dreams._runtime._compile_modes import (
    normalize_compile_denoise_mode,
    torch_compile_mode_arg,
)
from cosmos3_h_dreams._runtime._fademem import CosmosFadeMem
from cosmos3_h_dreams._runtime._patchify import (
    crop_latent_height,
    crop_latent_width,
    pad_latent_bottom,
    pad_latent_right,
    patchify_latent,
    unpatchify_latent,
)
from cosmos3_h_dreams._runtime._rollout import _FRAME_SEED_STRIDE, _STEP_SEED_STRIDE
from cosmos3_h_dreams._runtime._rope import (
    action_mrope_ids,
    text_mrope_ids,
    vae_mrope_ids,
)
from cosmos3_h_dreams._runtime._sampler import fixed_step_denoise
from cosmos3_h_dreams._runtime._sdpa_backends import (
    normalize_sdpa_backend,
    sdpa_kernel_context,
)

_DISTILLED_T_LISTS: dict[int, tuple[float, ...]] = {
    # Stage-3 SF-DMD 4->2 training keeps the shift-5 midpoint, rather than
    # truncating the first two entries from the four-step schedule.
    2: (1.0, 5.0 / 6.0),
    4: (1.0, 15.0 / 16.0, 5.0 / 6.0, 5.0 / 8.0),
}
DENOISE_STEPS: tuple[int, ...] = tuple(_DISTILLED_T_LISTS)
DISTILLED_T_LIST: list[float] = list(_DISTILLED_T_LISTS[4])
"""Default four-step causal-distillation schedule for compatibility."""


def distilled_t_list(denoise_steps: int) -> list[float]:
    """Return the checkpoint-trained sigma schedule for two or four steps."""
    try:
        return list(_DISTILLED_T_LISTS[int(denoise_steps)])
    except (KeyError, TypeError, ValueError) as error:
        choices = ", ".join(str(value) for value in DENOISE_STEPS)
        raise ValueError(f"denoise_steps must be one of: {choices}") from error


DEFAULT_WINDOW_FRAMES = 24
"""Frames of generated history retained by default.

Attention cost grows with cache length, so an unbounded cache makes every step
slower than the last. Smaller windows reduce attention cost; larger windows
retain more generated history.
"""

CACHE_MODES = ("cat", "block", "fademem")
"""``cat`` reallocates a growing cache and slices it; ``block`` writes into a
fixed ring buffer; ``fademem`` consolidates clean frame entries within a budget."""

CACHE_REFRESH_MODES = ("chunk", "framewise")
"""How finalized clean frames are written back over the noisy chunk: in one
masked forward, or one frame at a time against only the history already valid."""


def validate_cache_options(
    *,
    cache_mode: str,
    cache_refresh_mode: str = "chunk",
    fademem_slots: int = 12,
    fademem_beta: float = 0.3,
    fademem_merge_selection: str = "age",
    window_frames: int = DEFAULT_WINDOW_FRAMES,
    overlap_cache_update: bool = False,
    compile_denoise: bool = False,
    fixed_shape_cache_fill: bool = False,
    valid_length_cache_attention: bool = False,
    cuda_graph_denoise: bool = False,
    device: str = "cuda",
) -> None:
    """Reject rollout option combinations that cannot work.

    Shared with :class:`~cosmos3_h_dreams._runtime._hand_model.HandModel` so the
    server rejects a bad command line at startup rather than after loading a
    checkpoint and waiting for the operator to press start.

    Raises:
        ValueError: If a mode is unknown or an option's prerequisite is missing.
    """
    if cache_mode not in CACHE_MODES:
        raise ValueError(f"cache_mode must be one of {CACHE_MODES}, got {cache_mode!r}")
    if cache_refresh_mode not in CACHE_REFRESH_MODES:
        raise ValueError(
            f"cache_refresh_mode must be one of {CACHE_REFRESH_MODES}, "
            f"got {cache_refresh_mode!r}"
        )
    if cache_mode == "fademem":
        if (
            isinstance(fademem_slots, bool)
            or not isinstance(fademem_slots, int)
            or fademem_slots < 2
        ):
            raise ValueError("fademem_slots must be an integer of at least two")
        if not 0 < fademem_beta <= 1:
            raise ValueError("fademem_beta must be finite and in (0, 1]")
        if fademem_merge_selection not in {"age", "cosine"}:
            raise ValueError("fademem_merge_selection must be age or cosine")
        if cache_refresh_mode != "chunk":
            raise ValueError("FadeMem requires chunk clean-cache refresh")
    requires_fixed_cache = {
        "cache_refresh_mode='framewise'": cache_refresh_mode == "framewise",
        "overlap_cache_update": overlap_cache_update,
        "compile_denoise": compile_denoise,
        "fixed_shape_cache_fill": fixed_shape_cache_fill,
        "valid_length_cache_attention": valid_length_cache_attention,
    }
    for name, requested in requires_fixed_cache.items():
        if requested and cache_mode not in {"block", "fademem"}:
            raise ValueError(f"{name} requires cache_mode='block' or 'fademem'")
    if cache_mode == "block" and window_frames <= 0:
        raise ValueError(
            "cache_mode='block' needs window_frames > 0; the ring buffer has a "
            "fixed size and cannot grow without bound"
        )
    if fixed_shape_cache_fill and not compile_denoise:
        raise ValueError("fixed_shape_cache_fill requires compile_denoise=True")
    if valid_length_cache_attention and not compile_denoise:
        raise ValueError("valid_length_cache_attention requires compile_denoise=True")
    if valid_length_cache_attention and fixed_shape_cache_fill:
        raise ValueError(
            "valid_length_cache_attention and fixed_shape_cache_fill are "
            "mutually exclusive"
        )
    if valid_length_cache_attention and not varlen_attention_available():
        raise ValueError(
            "valid_length_cache_attention requires PyTorch varlen FlashAttention"
        )
    device_type = torch.device(device).type
    if valid_length_cache_attention and (
        device_type != "cuda" or not torch.cuda.is_available()
    ):
        raise ValueError("valid_length_cache_attention requires a CUDA device")
    if cuda_graph_denoise and not compile_denoise:
        raise ValueError("cuda_graph_denoise requires compile_denoise=True")
    if cuda_graph_denoise and (device_type != "cuda" or not torch.cuda.is_available()):
        raise ValueError("cuda_graph_denoise requires a CUDA device")


class HandKVSession:
    """Stateful AgiBot rollout: ``step(action[n*tcf,64]) -> n clean latent frames``.

    Args:
        net: The MoT network.
        text_ids: Packed caption tokens, or ``None`` for no text conditioning.
        init_latent: ``[hw, patch_dim]`` patchified clean frame 0.
        grid_hw: Transformer token grid ``(gh, gw)``.
        chunk_size: Latent frames generated per step.
        tcf: Actions per latent frame (VAE temporal compression).
        action_domain_id: Row of the domain-aware action embedding.
        fps: Conditioning frame rate used for mRoPE temporal scaling.
        seed: Base seed; per-chunk and per-step seeds derive from it.
        t_list: Sigma schedule. Defaults to :data:`DISTILLED_T_LIST`.
        window_frames: Frames of generated history to retain, rounded up to a
            whole number of chunks. ``0`` keeps everything, which makes each step
            slower than the last and is not allowed in ``block`` cache mode.
        logical_latent_hw: Content latent size when it is smaller than the grid
            implies. Defaults to the full transformer latent.
        native_gqa_attention: Let SDPA handle grouped-query attention instead of
            expanding K/V heads. Ignored when the PyTorch build lacks support.
        sdpa_backend: Attention kernel to pin, or ``"default"`` to let PyTorch
            choose.
        cache_mode: One of :data:`CACHE_MODES`. ``block`` needs a bounded window
            and full chunks, since the ring buffer's block size is fixed.
        cache_refresh_mode: One of :data:`CACHE_REFRESH_MODES`. ``framewise``
            requires ``cache_mode="block"``.
        fademem_slots: Historical latent-frame entries, including frame zero.
            Text and the current chunk are outside this budget.
        fademem_beta: Power-law age exponent in ``(0, 1]``.
        fademem_merge_selection: Adjacent-pair ranking, ``age`` or latent ``cosine``.
        overlap_cache_update: Write the clean K/V back on a side CUDA stream so
            the frames can be decoded while the cache catches up. Requires
            ``cache_mode="block"`` or ``"fademem"``; a no-op off CUDA.
        compile_denoise: ``torch.compile`` one velocity evaluation. Requires
            a fixed-buffer cache (``block`` or ``fademem``), whose buffers give
            the stable shapes compile needs.
        compile_denoise_mode: One of
            :data:`~cosmos3_h_dreams._runtime._compile_modes.COMPILE_DENOISE_MODES`.
        fixed_shape_cache_fill: During cache fill, pad history to its eventual
            steady-state shape and mask unwritten positions. This enables a
            separate static compiled/CUDA-graph path from the first chunk while
            retaining the unmasked attention path after saturation.
        valid_length_cache_attention: Keep full-size physical K/V buffers from
            the first chunk and pass their dynamic logical prefix length to fused
            varlen FlashAttention. This uses one static compiled/CUDA-graph path
            during fill and steady state without exposing fake history.
        cuda_graph_denoise: Capture the compiled velocity call into a replayable
            CUDA graph. Requires ``compile_denoise`` and a CUDA device. A session
            preserves cache buffer addresses across resets, so captured memory is
            allocated once and stays valid for the session's lifetime; a rollout
            starts over by building a new session.
        cuda_graph_warmup_iters: Eager calls before capture, to drain Inductor's
            lazy autotuning, which is illegal during capture.

    Raises:
        ValueError: If a mode is unknown, ``block`` is asked for without a
            window, or an option that depends on ``block``, ``compile_denoise``
            or CUDA is asked for without it.
    """

    def __init__(
        self,
        net: CosmosMoTNetwork,
        *,
        text_ids: Tensor | None,
        init_latent: Tensor,
        grid_hw: tuple[int, int],
        chunk_size: int = 4,
        tcf: int = _agibot.TEMPORAL_COMPRESSION_FACTOR,
        action_domain_id: int = _agibot.AGIBOT_DOMAIN_ID,
        fps: float = _agibot.FPS,
        seed: int = 0,
        sample_type: str = "sde",
        t_list: list[float] | None = None,
        window_frames: int = DEFAULT_WINDOW_FRAMES,
        logical_latent_hw: tuple[int, int] | None = None,
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
    ) -> None:
        self.net = net
        self.spec = net.config.spec
        self.device = net.embed_tokens.weight.device
        self.text_ids = text_ids
        self.gh, self.gw = grid_hw
        self.hw = self.gh * self.gw
        self.patch = self.spec.vfm.latent_patch_size
        self.channels = self.spec.vfm.vae_latent_channels
        self.transformer_h_lat = self.gh * self.patch
        self.transformer_w_lat = self.gw * self.patch
        self.h_lat, self.w_lat = (
            int(value)
            for value in (
                logical_latent_hw or (self.transformer_h_lat, self.transformer_w_lat)
            )
        )
        if (
            self.h_lat <= 0
            or self.w_lat <= 0
            or self.h_lat > self.transformer_h_lat
            or self.w_lat > self.transformer_w_lat
            or -(-self.h_lat // self.patch) != self.gh
            or -(-self.w_lat // self.patch) != self.gw
        ):
            raise ValueError(
                "logical latent geometry must fit and map to the transformer grid: "
                f"logical={self.h_lat}x{self.w_lat}, "
                f"transformer={self.transformer_h_lat}x{self.transformer_w_lat}, "
                f"patch={self.patch}, grid={self.gh}x{self.gw}"
            )
        self.n_text = int(text_ids.shape[0]) if text_ids is not None else 0
        self.chunk_size = int(chunk_size)
        self.tcf = int(tcf)
        self.action_domain_id = int(action_domain_id)
        self.fps = float(fps)
        self.seed = int(seed)
        self.sample_type = str(sample_type)
        self.t_list = list(t_list) if t_list is not None else list(DISTILLED_T_LIST)
        self._init_latent = init_latent.to(self.device)
        self.action_dim = _agibot.MAX_ACTION_DIM
        self.stride = (
            (self.spec.vfm.base_fps / self.fps)
            if (self.spec.vfm.enable_fps_modulation and self.fps)
            else 1.0
        )

        _, offset = text_mrope_ids(self.n_text, 0.0)
        self._off = offset + self.spec.vfm.mrope_temporal_modality_margin

        # Silently ignored rather than rejected: the flag is a speed hint, and an
        # older PyTorch build should not stop the demo from running.
        self.native_gqa_attention = bool(
            native_gqa_attention and sdpa_supports_native_gqa()
        )
        self.sdpa_backend = normalize_sdpa_backend(sdpa_backend)
        self.last_timing: dict[str, float] = {}

        self.cache_mode = str(cache_mode)
        self.cache_refresh_mode = str(cache_refresh_mode)
        self.overlap_cache_update = bool(overlap_cache_update)
        self.compile_denoise = bool(compile_denoise)
        self.fixed_shape_cache_fill = bool(fixed_shape_cache_fill)
        self.valid_length_cache_attention = bool(valid_length_cache_attention)
        self.cuda_graph_denoise = bool(cuda_graph_denoise)
        validate_cache_options(
            cache_mode=self.cache_mode,
            cache_refresh_mode=self.cache_refresh_mode,
            fademem_slots=fademem_slots,
            fademem_beta=fademem_beta,
            fademem_merge_selection=fademem_merge_selection,
            window_frames=max(0, int(window_frames)),
            overlap_cache_update=self.overlap_cache_update,
            compile_denoise=self.compile_denoise,
            fixed_shape_cache_fill=self.fixed_shape_cache_fill,
            valid_length_cache_attention=self.valid_length_cache_attention,
            cuda_graph_denoise=self.cuda_graph_denoise,
            device=self.device.type,
        )
        # The cat path reallocates its cache every step, so a side stream would
        # race with that; only the fixed ring buffer can be written concurrently.
        self._cache_stream = (
            torch.cuda.Stream(device=self.device)
            if self.overlap_cache_update
            and self.device.type == "cuda"
            and torch.cuda.is_available()
            else None
        )
        self._pending_cache: list[tuple[torch.cuda.Event, tuple[Tensor, Tensor]]] = []

        self.compile_denoise_mode = normalize_compile_denoise_mode(compile_denoise_mode)
        self.cuda_graph_warmup_iters = max(0, int(cuda_graph_warmup_iters))

        # Set before the frame-0 seed below, which reads them through _flags().
        self._per_frame_tokens = self.hw + self.tcf
        # Rounded up to whole chunks so the two cache modes span the same history:
        # the ring buffer can only evict a whole block, and frames arrive a chunk
        # at a time, so a chunk is the natural eviction granularity.
        requested_window = max(0, int(window_frames))
        self._window_frames = self.chunk_size * -(-requested_window // self.chunk_size)
        self._abs_frame = 0
        self._steps = 0

        self.fademem: CosmosFadeMem | None = None
        self.cache: list[dict[str, Tensor]] | None = None
        self.stream_cache: _MoTStreamCache | None = None
        text_positions = (
            text_mrope_ids(self.n_text, 0.0)[0].to(self.device)
            if text_ids is not None and self.n_text > 0
            else None
        )
        if self.cache_mode == "block":
            self.stream_cache = net.init_stream_cache(
                chunk_grid=(self.chunk_size, self.gh, self.gw),
                window_frames=self._window_frames,
                sink_frames=1,
                tokens_per_frame=self._per_frame_tokens,
            )
            if self.fixed_shape_cache_fill or self.valid_length_cache_attention:
                for gen in self.stream_cache.gen:
                    gen._k.zero_()
                    gen._v.zero_()
            if text_ids is not None and text_positions is not None:
                net.prefill_text_stream(
                    text_ids,
                    text_positions,
                    self.stream_cache,
                    native_gqa_attention=self.native_gqa_attention,
                )
        else:
            self.cache = net.new_kv_cache()
            if text_ids is not None and text_positions is not None:
                net.prefill_text_cache(
                    text_ids,
                    text_positions,
                    self.cache,
                    native_gqa_attention=self.native_gqa_attention,
                )

        # A normal bounded cache has separate growing-history and saturated
        # execution regimes. The valid-length path instead presents one fixed
        # physical K/V shape from the first chunk and changes only a device-side
        # logical prefix length, so fill and steady state share one callable.
        self._denoise_step_fn = self._build_denoise_step_fn(dynamic=False)
        self._denoise_fill_fn = (
            self._denoise_step_fn
            if self.valid_length_cache_attention or self.cache_mode == "fademem"
            else (
                self._build_denoise_step_fn(dynamic=not self.fixed_shape_cache_fill)
                if self.compile_denoise
                else self._denoise_step_fn
            )
        )
        self._denoise_graph = (
            CUDAGraphWrapper(
                self._denoise_step_fn,
                warmup_iters=self.cuda_graph_warmup_iters,
                capture_error_mode="thread_local",
            )
            if self.cuda_graph_denoise
            else None
        )
        self._denoise_fill_graph = (
            CUDAGraphWrapper(
                self._denoise_fill_fn,
                warmup_iters=self.cuda_graph_warmup_iters,
                capture_error_mode="thread_local",
            )
            if (
                self.cuda_graph_denoise
                and (self.fixed_shape_cache_fill or self.cache_mode == "fademem")
                and not self.valid_length_cache_attention
            )
            else None
        )
        # Only valid-length attention uses one graph through fill/saturation.
        # FadeMem has fixed shapes throughout, but must hand off from a masked
        # fill graph to an unmasked steady graph when every slot becomes valid.
        self._compile_start_step = (
            0
            if self.valid_length_cache_attention
            else (
                -(-(fademem_slots - 1) // self.chunk_size)
                if self.cache_mode == "fademem" and self.compile_denoise
                else (
                    self._window_frames // self.chunk_size
                    if self.compile_denoise
                    else 0
                )
            )
        )

        # Frame 0 is seeded with a null action whose values are zeroed in the
        # cache, so the start image conditions geometry without implying motion.
        seed_frame_zero = (
            self._seed_frame_stream
            if self.cache_mode == "block"
            else self._seed_frame_cat
        )
        seed_frame_zero(
            0,
            self._init_latent,
            torch.zeros(self.tcf, self.action_dim, device=self.device),
            null_action_values=True,
        )
        if self.cache_mode == "fademem":
            self.fademem = CosmosFadeMem(
                self._cat_cache(),
                n_text=self.n_text,
                capacity=fademem_slots,
                beta=fademem_beta,
                merge_selection=fademem_merge_selection,
                seed_latent=self._init_latent,
                chunk_frames=self.chunk_size,
                positions=self._positions,
                spec=self.spec,
            )
            self.cache = None
        self._abs_frame = 1

    def _reset_rollout_state(self) -> None:
        """Reset mutable rollout state while preserving compiled CUDA graphs."""
        self._pending_cache.clear()
        self._abs_frame = 0
        self._steps = 0
        self.last_timing = {}
        if self.fademem is not None:
            self.fademem.reset()
            self._abs_frame = 1
            return
        text_positions = (
            text_mrope_ids(self.n_text, 0.0)[0].to(self.device)
            if self.text_ids is not None and self.n_text > 0
            else None
        )
        if self.cache_mode == "block":
            assert self.stream_cache is not None
            self.stream_cache.reset()
            if self.fixed_shape_cache_fill or self.valid_length_cache_attention:
                for gen in self.stream_cache.gen:
                    gen._k.zero_()
                    gen._v.zero_()
        else:
            self.cache = self.net.new_kv_cache()
            if self.text_ids is not None and text_positions is not None:
                self.net.prefill_text_cache(
                    self.text_ids,
                    text_positions,
                    self.cache,
                    native_gqa_attention=self.native_gqa_attention,
                )

        seed_frame_zero = (
            self._seed_frame_stream
            if self.cache_mode == "block"
            else self._seed_frame_cat
        )
        seed_frame_zero(
            0,
            self._init_latent,
            torch.zeros(self.tcf, self.action_dim, device=self.device),
            null_action_values=True,
        )
        self._abs_frame = 1

    def reset_rollout(self) -> None:
        """Reset to frame zero while preserving compiled denoise graphs."""
        self.wait_for_pending_cache_update(block=True)
        self._reset_rollout_state()

    def prewarm_rollout(
        self, steps: int, *, profile: bool = False
    ) -> list[dict[str, float]]:
        """Run hidden chunks, then reset while retaining compiled GPU state.

        Args:
            steps: Number of full action chunks to discard.
            profile: Record synchronized stage timings for each hidden chunk.

        Returns:
            Per-chunk timing records in execution order.
        """
        action = torch.zeros(
            self.chunk_size * self.tcf,
            self.action_dim,
            device=self.device,
            dtype=torch.float32,
        )
        timings: list[dict[str, float]] = []
        for _ in range(max(0, int(steps))):
            self.step(action, profile=profile)
            timings.append(dict(self.last_timing))
        self.reset_rollout()
        return timings

    def _positions(self, start_frame: int, n_frames: int) -> Tensor:
        """Build interleaved ``[action, vision]`` mRoPE ids for ``n_frames``."""
        vision_ids, _ = vae_mrope_ids(
            n_frames,
            self.gh,
            self.gw,
            self._off,
            reset_spatial_indices=self.spec.vfm.mrope_reset_spatial_ids,
            fps=self.fps if self.spec.vfm.enable_fps_modulation else None,
            base_fps=self.spec.vfm.base_fps,
            temporal_compression_factor=self.spec.vfm.temporal_compression_factor,
            start_frame_offset=start_frame,
        )
        action_ids = action_mrope_ids(
            n_frames, self._off, self.stride, self.tcf, start_frame=start_frame
        )
        vision_ids = vision_ids.reshape(3, n_frames, self.hw)
        action_ids = action_ids.reshape(3, n_frames, self.tcf)
        return torch.cat((action_ids, vision_ids), dim=2).reshape(3, -1).to(self.device)

    def _flags(self, n_frames: int) -> tuple[Tensor, Tensor]:
        """Return ``(is_gen, is_action)`` for the interleaved supertoken layout."""
        total = n_frames * self._per_frame_tokens
        is_gen = torch.ones(total, dtype=torch.bool, device=self.device)
        per_frame = torch.tensor(
            [True] * self.tcf + [False] * self.hw, device=self.device
        )
        return is_gen, per_frame.repeat(n_frames)

    def _build_denoise_step_fn(self, *, dynamic: bool):
        """Build one break-free velocity evaluation, compiled when requested.

        Only the velocity call is compiled, not the sampler loop: the loop carries
        ``fp32_math`` and the non-SDE sample types, which a hard-coded compiled
        loop would have to reimplement.

        The packing is written structurally rather than through the boolean-mask
        assignment :meth:`_assemble_input` uses, because Inductor lowers boolean
        assignment to ``index_put_``, which CUDA graphs cannot capture.
        """
        net = self.net
        device, dtype = self.device, net.config.dtype
        domain_id = self.action_domain_id
        native_gqa = self.native_gqa_attention
        hidden = self.spec.llm.hidden_size
        # Block mode rejects partial chunks, so the frame count is fixed for the
        # session's lifetime and every shape below is static.
        tcf, hw, frames = self.tcf, self.hw, self.chunk_size
        to_tokens, to_logical = self._to_tokens, self._to_logical
        layers = self._layer_list()

        def denoise_step(
            latent: Tensor,
            timestep: Tensor,
            cos: Tensor,
            sin: Tensor,
            action_tokens: Tensor,
            text_k: list[Tensor | None],
            text_v: list[Tensor | None],
            history_k: list[Tensor | None],
            history_v: list[Tensor | None],
            history_mask: Tensor | None,
            history_valid_length: Tensor | None,
        ) -> Tensor:
            vision = net.vae2llm(to_tokens(latent).to(device=device, dtype=dtype))
            vision = vision + net.embed_uniform_timestep(
                timestep, int(vision.shape[0])
            ).to(dtype)
            action = net.encode_action(action_tokens, domain_id=domain_id)
            # One [action(tcf), vision(hw)] supertoken per frame, built by
            # concatenation so no position is ever written by index.
            h = torch.cat(
                (action.view(frames, tcf, hidden), vision.view(frames, hw, hidden)),
                dim=1,
            ).view(1, frames * (tcf + hw), hidden)
            for index, layer in enumerate(layers):
                h = layer.gen_compute(
                    h,
                    cos,
                    sin,
                    text_k[index],
                    text_v[index],
                    history_k[index],
                    history_v[index],
                    None,
                    cached_mask=history_mask,
                    cached_valid_length=history_valid_length,
                    native_gqa_attention=native_gqa,
                )
            h = net.norm_moe_gen(h)
            flow = net.llm2vae(h)[0].view(frames, tcf + hw, -1)
            return to_logical(flow[:, tcf:].reshape(frames * hw, -1), frames)

        if not self.compile_denoise:
            return denoise_step
        return torch.compile(
            denoise_step,
            # Dynamic history benefits from one reusable symbolic graph. Keep
            # it on Inductor's default mode: max-autotune is reserved for the
            # static steady-state graph, where shapes and pointers are stable.
            mode=(
                None if dynamic else torch_compile_mode_arg(self.compile_denoise_mode)
            ),
            dynamic=dynamic,
        )

    def _clean_tokens(self, n_frames: int, vision: Tensor, action: Tensor) -> Tensor:
        """Assemble ``n_frames`` of clean, timestep-free ``[action, vision]`` tokens."""
        is_gen, is_action = self._flags(n_frames)
        zeros = torch.zeros(n_frames * self.hw, device=self.device)
        return self.net._assemble_input(
            None,
            vision,
            zeros,
            is_gen,
            # Clean frames carry no timestep embedding.
            vision_noisy_mask=zeros,
            action_tokens=action,
            is_action=is_action,
            action_domain_id=self.action_domain_id,
        )

    def _causal_within(self, n_frames: int) -> Tensor:
        """Causal across frames, bidirectional inside each frame's supertoken."""
        frame = torch.arange(n_frames, device=self.device).repeat_interleave(
            self._per_frame_tokens
        )
        return frame[None, :] <= frame[:, None]

    def _seed_frame_cat(
        self,
        frame: int,
        clean_vision: Tensor,
        action: Tensor,
        *,
        null_action_values: bool = False,
    ) -> None:
        """Append one clean ``[action, vision]`` supertoken to the growing cache."""
        _, is_action = self._flags(1)
        h = self._clean_tokens(1, clean_vision, action)
        self.net._layers_step(
            h,
            self._flags(1)[0],
            self._positions(frame, 1),
            self._cat_cache(),
            append=True,
            within_mask=self._causal_within(1),
            zero_value_mask=is_action if null_action_values else None,
            native_gqa_attention=self.native_gqa_attention,
        )

    def _seed_chunk_cat(
        self, start: int, clean_vision: Tensor, action: Tensor, n_frames: int
    ) -> None:
        """Append a chunk's clean K/V to the growing cache in ONE forward.

        A single causal-across-frames forward is equivalent to seeding the frames
        sequentially through a growing cache, since each frame still sees only
        the frames before it. Doing it per frame costs one full transformer pass
        each, which for a four-frame chunk is three more passes than the whole
        denoise loop saves.
        """
        self.net._layers_step(
            self._clean_tokens(n_frames, clean_vision, action),
            self._flags(n_frames)[0],
            self._positions(start, n_frames),
            self._cat_cache(),
            append=True,
            within_mask=self._causal_within(n_frames),
            native_gqa_attention=self.native_gqa_attention,
        )

    def _seed_frame_stream(
        self,
        frame: int,
        clean_vision: Tensor,
        action: Tensor,
        *,
        null_action_values: bool = False,
    ) -> None:
        """Write one clean supertoken into the ring buffer's pinned sink region."""
        assert self.stream_cache is not None
        _, is_action = self._flags(1)
        h = self._clean_tokens(1, clean_vision, action)
        cos, sin = self.net._mrope(self._positions(frame, 1))
        within = self._causal_within(1)
        sink = int(h.shape[1])
        # Zeroing V for the null action mirrors the cat path's zero_value_mask:
        # future frames must not read motion out of the start image, while the
        # current forward still uses the raw projection.
        keep = (~is_action).view(1, 1, -1, 1)
        for index, layer in enumerate(self._layer_list()):
            h, k, v = layer.gen_compute_with_kv(
                h,
                cos,
                sin,
                self.stream_cache.text_k[index],
                self.stream_cache.text_v[index],
                None,
                None,
                within,
                native_gqa_attention=self.native_gqa_attention,
            )
            gen = self.stream_cache.gen[index]
            gen._k[:, :, :sink].copy_(k)
            gen._v[:, :, :sink].copy_(v * keep.to(v.dtype) if null_action_values else v)
            gen._n_cached = sink
            gen._prev_chunk_idx = -1
            gen._curr_chunk_idx = None

    def _stream_layers(
        self, h: Tensor, positions: Tensor, within_mask: Tensor | None
    ) -> Tensor:
        """Run the generation half of every layer against the ring buffer."""
        assert self.stream_cache is not None
        cos, sin = self.net._mrope(positions)
        for index, layer in enumerate(self._layer_list()):
            h = layer.gen_step(
                h,
                cos,
                sin,
                self.stream_cache.text_k[index],
                self.stream_cache.text_v[index],
                self.stream_cache.gen[index],
                within_mask,
                native_gqa_attention=self.native_gqa_attention,
            )
        return h

    def _submit_refresh(
        self, start: int, clean_vision: Tensor, action: Tensor, n_frames: int
    ) -> float:
        """Run the clean K/V write-back, on a side stream when one is configured.

        The rewrite only has to finish before the *next* step reads the cache, so
        overlapping it lets decode and presentation proceed in the meantime.

        Returns:
            Seconds spent submitting, which is not the cost of the write itself.
        """
        if self._cache_stream is None:
            self._refresh_chunk(start, clean_vision, action, n_frames)
            return 0.0

        submitted = time.perf_counter()
        # Held until the event fires so the allocator cannot reuse the memory the
        # side stream is still reading from.
        clean_ref, action_ref = clean_vision.detach(), action.detach()
        ready, done = torch.cuda.Event(), torch.cuda.Event()
        torch.cuda.current_stream(self.device).record_event(ready)
        clean_ref.record_stream(self._cache_stream)
        action_ref.record_stream(self._cache_stream)
        with torch.cuda.stream(self._cache_stream):
            self._cache_stream.wait_event(ready)
            self._refresh_chunk(start, clean_ref, action_ref, n_frames)
            done.record(self._cache_stream)
        self._pending_cache.append((done, (clean_ref, action_ref)))
        self._pending_cache = [
            entry for entry in self._pending_cache if not entry[0].query()
        ]
        return time.perf_counter() - submitted

    def wait_for_pending_cache_update(self, *, block: bool = False) -> float:
        """Order the current stream after any outstanding clean-cache write.

        Args:
            block: Also wait on the host, which a caller needs before reading the
                cache from Python rather than from a queued kernel.

        Returns:
            Seconds spent waiting.
        """
        if self._cache_stream is None or not self._pending_cache:
            return 0.0
        waited = time.perf_counter()
        current = torch.cuda.current_stream(self.device)
        current.wait_event(self._pending_cache[-1][0])
        if block:
            current.synchronize()
            self._pending_cache.clear()
        else:
            self._pending_cache = [
                entry for entry in self._pending_cache if not entry[0].query()
            ]
        return time.perf_counter() - waited

    def _refresh_chunk(
        self, start: int, clean_vision: Tensor, action: Tensor, n_frames: int
    ) -> None:
        """Commit finalized clean K/V to the selected bounded memory."""
        if self.fademem is not None:
            _, current = self.fademem.memory.attention_positions(start, self.chunk_size)
            positions = self._positions(current, n_frames)
            cos, sin = self.net._mrope(positions)
            h = self._clean_tokens(n_frames, clean_vision, action)
            committed = []
            within = self._causal_within(n_frames)
            with sdpa_kernel_context(torch, self.sdpa_backend):
                for layer, (text_k, text_v), (hist_k, hist_v) in zip(
                    self._layer_list(),
                    self.fademem.text,
                    self.fademem.history(),
                    strict=True,
                ):
                    h, key, value = layer.gen_compute_with_kv(
                        h,
                        cos,
                        sin,
                        text_k,
                        text_v,
                        hist_k,
                        hist_v,
                        within,
                        native_gqa_attention=self.native_gqa_attention,
                    )
                    committed.append((key, value))
            self.fademem.commit(
                start,
                positions,
                committed,
                descriptors=clean_vision.reshape(n_frames, -1)
                if self.fademem.memory.merge_selection == "cosine"
                else None,
            )
            return
        assert self.stream_cache is not None
        if self.valid_length_cache_attention:
            self._refresh_chunk_valid_length(start, clean_vision, action, n_frames)
        elif self.cache_refresh_mode == "framewise":
            self._refresh_chunk_framewise(start, clean_vision, action, n_frames)
        else:
            self._stream_layers(
                self._clean_tokens(n_frames, clean_vision, action),
                self._positions(start, n_frames),
                self._causal_within(n_frames),
            )
        for gen in self.stream_cache.gen:
            gen.after_update(self._steps)

    def _refresh_chunk_valid_length(
        self, start: int, clean_vision: Tensor, action: Tensor, n_frames: int
    ) -> None:
        """Refresh clean K/V through the same fixed physical cache geometry.

        Each frame's queries attend to real history plus all tokens through that
        frame. Splitting only the attention call by frame exactly reproduces the
        block-causal mask while avoiding a new masked-SDPA shape and backend plan
        on every cache-fill step.
        """
        assert self.stream_cache is not None
        write_start = int(self.stream_cache.gen[0]._current_write_bounds()[0])
        (
            text_k,
            text_v,
            history_k,
            history_v,
            history_valid_length,
        ) = self.net.gather_stream_history_valid_length(self.stream_cache, write_start)
        h = self._clean_tokens(n_frames, clean_vision, action)
        cos, sin = self.net._mrope(self._positions(start, n_frames))
        current_tokens = n_frames * self._per_frame_tokens
        write_end = write_start + current_tokens
        for index, layer in enumerate(self._layer_list()):
            h, k, v = layer.gen_compute_with_kv(
                h,
                cos,
                sin,
                text_k[index],
                text_v[index],
                history_k[index],
                history_v[index],
                None,
                cached_valid_length=history_valid_length,
                causal_block_tokens=self._per_frame_tokens,
                native_gqa_attention=self.native_gqa_attention,
            )
            gen = self.stream_cache.gen[index]
            destination = gen._seq_slice(write_start, write_end)
            gen._k[destination].copy_(k)
            gen._v[destination].copy_(v)

    def _refresh_chunk_framewise(
        self, start: int, clean_vision: Tensor, action: Tensor, n_frames: int
    ) -> None:
        """Refresh one frame at a time, exposing only history that is already clean.

        The chunk refresh hides later frames behind a mask over a single forward.
        Here each frame instead attends to the real prefix, so no masking is
        needed and every frame sees finalized neighbours.
        """
        assert self.stream_cache is not None
        for offset in range(n_frames):
            h = self._clean_tokens(
                1,
                clean_vision[offset * self.hw : (offset + 1) * self.hw],
                action[offset * self.tcf : (offset + 1) * self.tcf],
            )
            cos, sin = self.net._mrope(self._positions(start + offset, 1))
            for index, layer in enumerate(self._layer_list()):
                gen = self.stream_cache.gen[index]
                frame_start = (
                    int(gen._current_write_bounds()[0])
                    + offset * self._per_frame_tokens
                )
                frame_end = frame_start + self._per_frame_tokens
                history = gen._seq_slice(0, frame_start)
                h, k, v = layer.gen_compute_with_kv(
                    h,
                    cos,
                    sin,
                    self.stream_cache.text_k[index],
                    self.stream_cache.text_v[index],
                    gen._k[history] if frame_start > 0 else None,
                    gen._v[history] if frame_start > 0 else None,
                    None,
                    native_gqa_attention=self.native_gqa_attention,
                )
                gen._k[gen._seq_slice(frame_start, frame_end)].copy_(k)
                gen._v[gen._seq_slice(frame_start, frame_end)].copy_(v)

    def _prune_cache(self) -> None:
        """Drop the oldest frames beyond the window, keeping text and frame 0."""
        if self._window_frames <= 0 or self.cache is None:
            return
        generated = self._abs_frame - 1
        if generated <= self._window_frames:
            return
        head = self.n_text + self._per_frame_tokens
        tail = self._window_frames * self._per_frame_tokens
        for layer_cache in self.cache:
            k, v = layer_cache.get("k"), layer_cache.get("v")
            if k is None or v is None:
                continue
            layer_cache["k"] = torch.cat(
                (k[:, :, :head], k[:, :, -tail:]), dim=2
            ).contiguous()
            layer_cache["v"] = torch.cat(
                (v[:, :, :head], v[:, :, -tail:]), dim=2
            ).contiguous()

    def _noise(self, seed: int, frames: int, *, dtype: torch.dtype) -> Tensor:
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        return torch.empty(
            (self.channels, frames, self.h_lat, self.w_lat),
            device=self.device,
            dtype=dtype,
        ).normal_(generator=generator)

    def _to_tokens(self, logical_latent: Tensor) -> Tensor:
        """Pad the logical latent up to the transformer grid and patchify it."""
        padded = pad_latent_bottom(logical_latent, self.transformer_h_lat - self.h_lat)
        padded = pad_latent_right(padded, self.transformer_w_lat - self.w_lat)
        return patchify_latent(padded, self.patch)

    def _to_logical(self, tokens: Tensor, n_frames: int) -> Tensor:
        """Unpatchify tokens and crop back to the logical content region."""
        latent = unpatchify_latent(
            tokens,
            channels=self.channels,
            grid_t=n_frames,
            grid_h=self.gh,
            grid_w=self.gw,
            patch_spatial=self.patch,
        )
        return crop_latent_width(crop_latent_height(latent, self.h_lat), self.w_lat)

    @property
    def accepts_partial_chunks(self) -> bool:
        """Whether :meth:`step` takes fewer than ``chunk_size`` frames.

        The ring buffer writes a fixed block, so block mode cannot. Callers that
        can only supply a short final chunk, such as recorded replay reaching the
        end of a clip, ask rather than inferring it from the cache mode.
        """
        return self.cache_mode != "block" and not (
            self.cache_mode == "fademem" and self.compile_denoise
        )

    def _layer_list(self) -> list[_MoTLayer]:
        """The MoT layers. ``nn.ModuleList`` erases its element type."""
        return cast("list[_MoTLayer]", list(self.net.layers))

    def _cat_cache(self) -> list[dict[str, Tensor]]:
        """The growing cache, which only exists in ``cat`` mode."""
        assert self.cache is not None, "cat cache is absent in block mode"
        return self.cache

    def _sync(self) -> None:
        """Wait for queued GPU work so a timing boundary means something."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @torch.no_grad()
    def step(self, action_chunk: Tensor, *, profile: bool = False) -> list[Tensor]:
        """Denoise up to ``chunk_size`` frames driven by ``action_chunk``.

        Args:
            action_chunk: ``[n*tcf, 64]`` normalized, padded actions.
            profile: Record per-stage durations in :attr:`last_timing`. Costs a
                few device synchronizations, so it is off by default.

        Returns:
            The new clean latent frames, each ``[C, 1, H_lat, W_lat]``.

        Raises:
            ValueError: If the chunk is not a whole number of latent frames or
                requests more than ``chunk_size`` frames.
        """
        self.last_timing = {}
        step_started = time.perf_counter()
        # Last step's write-back may still be in flight; this step's attention
        # reads the same buffer, so it has to be ordered behind it.
        cache_wait_s = self.wait_for_pending_cache_update()
        rows = int(action_chunk.numel() // self.action_dim)
        if action_chunk.numel() % self.action_dim or rows == 0 or rows % self.tcf:
            raise ValueError(
                f"action chunk must contain n*{self.tcf} rows of {self.action_dim}, "
                f"got {tuple(action_chunk.shape)}"
            )
        frames = rows // self.tcf
        if frames > self.chunk_size:
            raise ValueError(
                f"action chunk requests {frames} latent frames; maximum is "
                f"{self.chunk_size}"
            )
        if (
            self.cache_mode == "block"
            or (self.cache_mode == "fademem" and self.compile_denoise)
        ) and frames != self.chunk_size:
            raise ValueError(
                f"cache_mode={self.cache_mode!r} requires a full {self.chunk_size}-frame chunk, "
                f"got {frames}; the ring buffer's block size is fixed at "
                "construction. Use an eager cat or FadeMem session for partial chunks"
            )

        start = self._abs_frame
        actions = action_chunk.to(self.device, dtype=torch.float32).reshape(
            rows, self.action_dim
        )
        positions = self._positions(start, frames)
        fade_history: list[tuple[Tensor, Tensor]] = []
        if self.fademem is not None:
            positions, fade_history = self.fademem.prepare(
                start, frames, fixed_shape=self.compile_denoise
            )
            fade_cos, fade_sin = self.net._mrope(positions)
        is_gen, is_action = self._flags(frames)
        vision_tokens = frames * self.hw
        compiled_args: tuple[object, ...] | None = None
        compiled_fill_active = False
        compiled_masked_fill_active = False
        compiled_valid_length_active = False

        def velocity_fn(latent: Tensor, timestep: Tensor) -> Tensor:
            if compiled_args is not None:
                if compiled_fill_active:
                    call = self._denoise_fill_fn
                    if self._denoise_fill_graph is not None:
                        call = self._denoise_fill_graph
                else:
                    call = self._denoise_step_fn
                if self._denoise_graph is not None and not compiled_fill_active:
                    # drain() runs eagerly through the same static buffers so
                    # Inductor finishes autotuning before capture.
                    call = (
                        self._denoise_graph
                        if self._steps >= self._compile_start_step
                        else self._denoise_graph.drain
                    )
                # Always a tensor: a float would be staged as a graph constant and
                # every replay would denoise at the first sigma.
                sigma = torch.as_tensor(
                    timestep, device=self.device, dtype=torch.float32
                ).reshape(1)
                with sdpa_kernel_context(torch, self.sdpa_backend):
                    return call(latent, sigma, *compiled_args)
            h = self.net._assemble_input(
                None,
                self._to_tokens(latent),
                torch.as_tensor(timestep, device=self.device, dtype=torch.float32)
                .reshape(1)
                .expand(vision_tokens),
                is_gen,
                vision_noisy_mask=None,
                action_tokens=actions,
                is_action=is_action,
                action_domain_id=self.action_domain_id,
                uniform_vision_timestep=True,
            )
            with sdpa_kernel_context(torch, self.sdpa_backend):
                if self.fademem is not None:
                    for layer, (text_k, text_v), (hist_k, hist_v) in zip(
                        self._layer_list(), self.fademem.text, fade_history, strict=True
                    ):
                        h = layer.gen_compute(
                            h,
                            fade_cos,
                            fade_sin,
                            text_k,
                            text_v,
                            hist_k,
                            hist_v,
                            None,
                            native_gqa_attention=self.native_gqa_attention,
                        )
                elif self.cache_mode == "block":
                    # Chunkwise teacher forcing is bidirectional inside the chunk.
                    h = self._stream_layers(h, positions, None)
                else:
                    h = self.net._layers_step(
                        h,
                        is_gen,
                        positions,
                        self._cat_cache(),
                        append=False,
                        within_mask=None,
                        native_gqa_attention=self.native_gqa_attention,
                    )
            h = self.net.norm_moe_gen(h)
            return self._to_logical(self.net.llm2vae(h)[0][~is_action], frames)

        def sde_noise(step_index: int) -> Tensor:
            return self._noise(
                self.seed + start * _FRAME_SEED_STRIDE + step_index * _STEP_SEED_STRIDE,
                frames,
                dtype=torch.float32,
            )

        if profile:
            self._sync()
        denoise_started = time.perf_counter()
        if self.stream_cache is not None:
            # Reserves this step's block so every denoise pass overwrites the same
            # slot instead of appending; _refresh_chunk commits it afterwards.
            for gen in self.stream_cache.gen:
                gen.before_update(self._steps)
            if self.compile_denoise:
                cos, sin = self.net._mrope(positions)
                # Gather once per chunk, outside the compiled region. The
                # valid-length route always exposes the same full physical shape;
                # its device scalar is the only fill-state input. Legacy routes
                # retain the dynamic or masked fill behavior for comparison.
                write_start = int(self.stream_cache.gen[0]._current_write_bounds()[0])
                compiled_fill_active = self._steps < self._compile_start_step
                history_valid_length = None
                if self.valid_length_cache_attention:
                    (
                        text_k,
                        text_v,
                        history_k,
                        history_v,
                        history_valid_length,
                    ) = self.net.gather_stream_history_valid_length(
                        self.stream_cache, write_start
                    )
                    history_mask = None
                    compiled_valid_length_active = True
                elif compiled_fill_active and self.fixed_shape_cache_fill:
                    (
                        text_k,
                        text_v,
                        history_k,
                        history_v,
                        history_mask,
                    ) = self.net.gather_stream_history_masked(
                        self.stream_cache, write_start
                    )
                    compiled_masked_fill_active = True
                else:
                    text_k, text_v, history_k, history_v = (
                        self.net.gather_stream_history(self.stream_cache, write_start)
                    )
                    history_mask = None
                compiled_args = (
                    cos,
                    sin,
                    actions,
                    text_k,
                    text_v,
                    history_k,
                    history_v,
                    history_mask,
                    history_valid_length,
                )
        if self.fademem is not None and self.compile_denoise:
            compiled_fill_active = self._steps < self._compile_start_step
            compiled_masked_fill_active = compiled_fill_active
            text_k, text_v = map(list, zip(*self.fademem.text, strict=True))
            history_k, history_v = map(list, zip(*fade_history, strict=True))
            compiled_args = (
                fade_cos,
                fade_sin,
                actions,
                text_k,
                text_v,
                history_k,
                history_v,
                self.fademem.mask if compiled_masked_fill_active else None,
                self.fademem.valid_length
                if self.valid_length_cache_attention
                else None,
            )
            compiled_valid_length_active = self.valid_length_cache_attention
        clean = fixed_step_denoise(
            velocity_fn,
            self._noise(self.seed + start, frames, dtype=torch.bfloat16).float(),
            t_list=self.t_list,
            sample_type=self.sample_type,
            sde_noise_fn=sde_noise if self.sample_type == "sde" else None,
            fp32_math=True,
        )
        if profile:
            self._sync()
        denoised_at = time.perf_counter()

        clean_tokens = self._to_tokens(clean.detach())
        cache_submit_s = 0.0
        if self.cache_mode in {"block", "fademem"}:
            cache_submit_s = self._submit_refresh(start, clean_tokens, actions, frames)
        else:
            self._seed_chunk_cat(start, clean_tokens, actions, frames)
        self._abs_frame += frames
        self._steps += 1
        self._prune_cache()
        if profile:
            self._sync()
        seeded_at = time.perf_counter()

        frames_out = [clean[:, index : index + 1].detach() for index in range(frames)]
        if profile:
            self._sync()
            finished = time.perf_counter()
            # setup, denoise, cache and output partition the step exactly; the two
            # cache_update entries are components of setup and cache.
            self.last_timing = {
                "model_step_s": finished - step_started,
                "setup_s": denoise_started - step_started,
                "denoise_s": denoised_at - denoise_started,
                "cache_seed_prune_s": seeded_at - denoised_at,
                "cache_update_submit_s": cache_submit_s,
                "cache_update_wait_s": cache_wait_s,
                "output_s": finished - seeded_at,
                "cached_tokens": float(self._cached_tokens()),
                "kv_cache_bytes": float(self._kv_cache_bytes()),
                "compile_denoise_active": float(compiled_args is not None),
                "compile_denoise_fill_active": float(compiled_fill_active),
                "compile_denoise_masked_fill_active": float(
                    compiled_masked_fill_active
                ),
                "compile_denoise_valid_length_active": float(
                    compiled_valid_length_active
                ),
                "compile_denoise_static_active": float(
                    compiled_args is not None and not compiled_fill_active
                ),
                "cuda_graph_captured": float(
                    self._denoise_graph is not None
                    and getattr(self._denoise_graph, "_graph", None) is not None
                ),
                "cuda_graph_fill_captured": float(
                    self._denoise_fill_graph is not None
                    and getattr(self._denoise_fill_graph, "_graph", None) is not None
                ),
            }
        return frames_out

    def _cached_tokens(self) -> int:
        """Sequence length the next step will attend over, for diagnostics."""
        if self.fademem is not None:
            return self.n_text + self.fademem.memory.count * self._per_frame_tokens
        if self.stream_cache is not None:
            return self.n_text + int(self.stream_cache.gen[0]._n_cached)
        for layer_cache in self.cache or ():
            key = layer_cache.get("k")
            if key is not None:
                return int(key.shape[2])
        return 0

    def _kv_cache_bytes(self) -> int:
        """Allocated K/V tensor bytes, including text and the pinned frame."""
        if self.fademem is not None:
            return self.fademem.storage_bytes
        if self.stream_cache is not None:
            tensors = (
                *self.stream_cache.text_k,
                *self.stream_cache.text_v,
                *(cache._k for cache in self.stream_cache.gen),
                *(cache._v for cache in self.stream_cache.gen),
            )
        else:
            tensors = (
                tensor
                for layer in self.cache or ()
                for tensor in (layer.get("k"), layer.get("v"))
            )
        return sum(t.numel() * t.element_size() for t in tensors if t is not None)
