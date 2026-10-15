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

"""Cosmos3-H-Dreams model specialization over the shared Cosmos3 runtime."""

from typing import Any

from cosmos3_h_dreams._runtime import _agibot
from cosmos3_h_dreams._runtime._hand_kv_session import HandKVSession
from cosmos3_h_dreams._runtime._hand_model import HandModel
from cosmos3_h_dreams._runtime._hand_scenes import Scene


def _resolution_geometry(
    height: int, width: int, patch: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return latent and token-grid dimensions for one VAE-resolution pair."""
    latent_hw = (
        int(height) // _agibot.SPATIAL_COMPRESSION_FACTOR,
        int(width) // _agibot.SPATIAL_COMPRESSION_FACTOR,
    )
    return latent_hw, tuple((size + patch - 1) // patch for size in latent_hw)


class Cosmos3HDreamsModel(HandModel):
    """Select the Cosmos3-H-Dreams embodiment while reusing the shared runtime."""

    def __init__(
        self, *, action_domain_id: int, height: int, width: int, **kwargs: Any
    ) -> None:
        self.action_domain_id = int(action_domain_id)
        super().__init__(**kwargs)
        self.height = int(height)
        self.width = int(width)
        self.latent_hw, self.grid_hw = _resolution_geometry(
            self.height, self.width, self.patch
        )
        action_dim = int(getattr(self.net.config, "action_dim", -1))
        domains = int(getattr(self.net.config, "num_embodiment_domains", -1))
        if action_dim != 64:
            raise ValueError(
                f"Cosmos3-H-Dreams expects a 64D checkpoint action projection, got {action_dim}"
            )
        if not 0 <= self.action_domain_id < domains:
            raise ValueError(
                f"domain_id must be in [0, {domains}), got {self.action_domain_id}"
            )

    def _new_session(self, scene: Scene) -> HandKVSession:
        return HandKVSession(
            self.net,
            text_ids=scene.text_ids,
            init_latent=scene.init_latent,
            grid_hw=self.grid_hw,
            chunk_size=self.chunk_size,
            tcf=self.tcf,
            t_list=self.t_list,
            action_domain_id=self.action_domain_id,
            fps=scene.fps,
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
