# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native Qwen3-VL prompt tokenizer for Cosmos3-H-Dreams prompt conditioning.

Pack prompt ``text_ids`` into the sequence in three stages:

1. **Resolution suffix** (``ResolutionTextInfo`` augmentor): append
   ``"This video is of {H}x{W} resolution."`` to the caption, joined with a single
   space when the caption already ends in ``.`` (else ``". "``).
2. **Chat template** (``tokenize_caption``): wrap in ``[{"role":"user", ...}]``
   through ``apply_chat_template(tokenize=True, add_generation_prompt=True,
   add_vision_id=False)``. The interactive checkpoint has
   ``use_system_prompt=False`` (no system turn by default).
3. **Pack special tokens** (``pack_text_tokens``): append ``eos_token_id``
   (``<|im_end|>``) then ``start_of_generation`` (``<|vision_start|>``).

Uses HuggingFace ``transformers`` with the public ``Qwen/Qwen3-VL-8B-Instruct``
tokenizer and chat template.
"""

from __future__ import annotations

import functools
from typing import Any

import torch
from torch import Tensor

DEFAULT_TOKENIZER = "Qwen/Qwen3-VL-8B-Instruct"

# Resolution metadata appended to image and video prompts.
_RESOLUTION_TEMPLATE_IMAGE = "This image is of {height}x{width} resolution."
_RESOLUTION_TEMPLATE_VIDEO = "This video is of {height}x{width} resolution."
# Optional system prompts, enabled only by use_system_prompt=True.
_SYSTEM_PROMPT_IMAGE = (
    "You are a helpful assistant who will generate images from a give prompt."
)
_SYSTEM_PROMPT_VIDEO = (
    "You are a helpful assistant who will generate videos from a give prompt."
)
# Cosmos3 start-of-generation marker appended by pack_text_tokens (has_generation=True).
_START_OF_GENERATION_TOKEN = "<|vision_start|>"


@functools.lru_cache(maxsize=4)
def load_tokenizer(tokenizer_path: str = DEFAULT_TOKENIZER) -> Any:
    """Load (and cache) the Qwen3-VL tokenizer via ``transformers.AutoTokenizer``."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path)


def generation_special_token_ids(
    tokenizer_path: str = DEFAULT_TOKENIZER,
) -> tuple[int, int]:
    """Return the packed ``(eos, vision_start)`` generation-suffix IDs."""
    tokenizer = load_tokenizer(tokenizer_path)
    return (
        int(tokenizer.eos_token_id),
        int(tokenizer.convert_tokens_to_ids(_START_OF_GENERATION_TOKEN)),
    )


def _with_resolution_suffix(
    caption: str, height: int, width: int, *, is_video: bool
) -> str:
    """Append the resolution metadata string."""
    template = _RESOLUTION_TEMPLATE_VIDEO if is_video else _RESOLUTION_TEMPLATE_IMAGE
    metadata = template.format(height=height, width=width)
    separator = " " if caption.rstrip().endswith(".") else ". "
    return caption + separator + metadata


def tokenize_prompt(
    prompt: str,
    *,
    height: int,
    width: int,
    tokenizer_path: str = DEFAULT_TOKENIZER,
    is_video: bool = True,
    use_system_prompt: bool = False,
    system_prompt: str | None = None,
    append_resolution: bool = True,
    append_generation_tokens: bool = True,
) -> Tensor:
    """Tokenize a prompt to packed ``[N_text]`` IDs (see module docstring).

    Args:
        prompt: raw text prompt / caption.
        height, width: target pixel resolution for the resolution suffix
            (480p landscape is ``height=480, width=832``; equals ``gh*32``/``gw*32``).
        is_video: use the video (vs image) resolution template + system prompt.
        append_resolution: add the resolution suffix (default: on).
        append_generation_tokens: append ``eos`` + ``<|vision_start|>``.
    """
    tokenizer = load_tokenizer(tokenizer_path)
    caption = (
        _with_resolution_suffix(prompt, height, width, is_video=is_video)
        if append_resolution
        else prompt
    )

    conversations: list[dict[str, str]] = []
    if system_prompt is not None:
        conversations.append({"role": "system", "content": system_prompt})
    elif use_system_prompt:
        conversations.append(
            {
                "role": "system",
                "content": _SYSTEM_PROMPT_VIDEO if is_video else _SYSTEM_PROMPT_IMAGE,
            }
        )
    conversations.append({"role": "user", "content": caption})

    ids = list(
        tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=False,
            return_dict=False,
        )
    )
    if append_generation_tokens:
        ids.extend(generation_special_token_ids(tokenizer_path))
    return torch.tensor(ids, dtype=torch.long)
