# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for explicitly selecting PyTorch SDPA backends in probes."""

from __future__ import annotations

import inspect
from contextlib import contextmanager, nullcontext
from typing import Any, Iterator

SDPA_BACKENDS = ("default", "cudnn", "flash", "efficient", "math")

_LABELS = {
    "default": "default",
    "cudnn": "cudnn",
    "flash": "flash",
    "efficient": "efficient",
    "math": "math",
}


def normalize_sdpa_backend(value: str | None) -> str:
    """Return the canonical SDPA backend probe name."""
    raw = (value or "default").strip().lower().replace("-", "_")
    aliases = {
        "none": "default",
        "auto": "default",
        "mem_efficient": "efficient",
        "memory_efficient": "efficient",
        "efficient_attention": "efficient",
        "flash_attention": "flash",
        "cudnn_attention": "cudnn",
    }
    backend = aliases.get(raw, raw)
    if backend not in SDPA_BACKENDS:
        allowed = ", ".join(SDPA_BACKENDS)
        raise ValueError(f"SDPA backend must be one of: {allowed}; got {value!r}")
    return backend


def sdpa_backend_label(backend: str) -> str:
    """Short stable label for artifact names."""
    return _LABELS[normalize_sdpa_backend(backend)]


def collect_sdpa_environment(
    torch: Any, *, device: str | None = None
) -> dict[str, Any]:
    """Collect PyTorch/CUDA/SDPA facts for probe metadata."""
    cuda = getattr(torch, "cuda", None)
    cuda_available = bool(cuda is not None and cuda.is_available())
    info: dict[str, Any] = {
        "torch_version": getattr(torch, "__version__", None),
        "cuda_version": getattr(getattr(torch, "version", None), "cuda", None),
        "cuda_available": cuda_available,
        "cudnn_version": None,
        "device": device,
        "device_name": None,
        "supported_sdpa_backends": list_supported_sdpa_backends(torch),
    }
    backends = getattr(torch, "backends", None)
    cudnn = getattr(backends, "cudnn", None)
    if cudnn is not None:
        try:
            info["cudnn_version"] = cudnn.version()
        except Exception:  # noqa: BLE001 - best-effort metadata
            info["cudnn_version"] = None
    if cuda_available:
        try:
            current = cuda.current_device()
            info["device_name"] = cuda.get_device_name(current)
            props = cuda.get_device_properties(current)
            info["device_capability"] = list(cuda.get_device_capability(current))
            info["device_total_memory_bytes"] = int(props.total_memory)
        except Exception as exc:  # noqa: BLE001
            info["device_query_error"] = str(exc)
    return info


def list_supported_sdpa_backends(torch: Any) -> list[str]:
    """Return backend names that this PyTorch build can explicitly request."""
    supported = ["default"]
    try:
        from torch.nn.attention import SDPBackend  # type: ignore[attr-defined]

        enum_names = {name for name in dir(SDPBackend) if name.isupper()}
        if "CUDNN_ATTENTION" in enum_names:
            supported.append("cudnn")
        if "FLASH_ATTENTION" in enum_names:
            supported.append("flash")
        if (
            "EFFICIENT_ATTENTION" in enum_names
            or "MEM_EFFICIENT_ATTENTION" in enum_names
        ):
            supported.append("efficient")
        if "MATH" in enum_names:
            supported.append("math")
        return supported
    except Exception:  # noqa: BLE001 - fall back to older torch.backends.cuda API
        pass

    cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
    sdp_kernel = getattr(cuda_backend, "sdp_kernel", None)
    if sdp_kernel is None:
        return supported
    try:
        params = inspect.signature(sdp_kernel).parameters
    except Exception:  # noqa: BLE001
        params = {}
    if "enable_cudnn" in params:
        supported.append("cudnn")
    if "enable_flash" in params:
        supported.append("flash")
    if "enable_mem_efficient" in params:
        supported.append("efficient")
    if "enable_math" in params:
        supported.append("math")
    return supported


@contextmanager
def sdpa_kernel_context(torch: Any, backend: str) -> Iterator[None]:
    """Constrain scaled-dot-product attention to one backend when possible."""
    backend = normalize_sdpa_backend(backend)
    if backend == "default":
        with nullcontext():
            yield
        return

    ctx = _make_sdpa_context(torch, backend)
    with ctx:
        yield


def _make_sdpa_context(torch: Any, backend: str) -> Any:
    """Build a backend-selection context without catching body exceptions."""
    try:
        from torch.nn.attention import (  # type: ignore[attr-defined]
            SDPBackend,
            sdpa_kernel,
        )

        selected = _new_api_backend(SDPBackend, backend)
        return sdpa_kernel([selected])
    except Exception as new_api_exc:  # noqa: BLE001 - try older API below
        cuda_backend = getattr(getattr(torch, "backends", None), "cuda", None)
        sdp_kernel = getattr(cuda_backend, "sdp_kernel", None)
        if sdp_kernel is None:
            raise RuntimeError(
                f"PyTorch build cannot explicitly select SDPA backend {backend!r}"
            ) from new_api_exc
        return sdp_kernel(**_old_api_kwargs(sdp_kernel, backend))


def _new_api_backend(SDPBackend: Any, backend: str) -> Any:
    names = {
        "cudnn": ("CUDNN_ATTENTION",),
        "flash": ("FLASH_ATTENTION",),
        "efficient": ("EFFICIENT_ATTENTION", "MEM_EFFICIENT_ATTENTION"),
        "math": ("MATH",),
    }[backend]
    for name in names:
        if hasattr(SDPBackend, name):
            return getattr(SDPBackend, name)
    raise RuntimeError(f"PyTorch SDPBackend enum has no member for {backend!r}")


def _old_api_kwargs(sdp_kernel: Any, backend: str) -> dict[str, bool]:
    try:
        params = inspect.signature(sdp_kernel).parameters
    except Exception:  # noqa: BLE001
        params = {}
    requested = {
        "enable_flash": backend == "flash",
        "enable_math": backend == "math",
        "enable_mem_efficient": backend == "efficient",
        "enable_cudnn": backend == "cudnn",
    }
    kwargs = {name: value for name, value in requested.items() if name in params}
    if backend == "cudnn" and "enable_cudnn" not in kwargs:
        raise RuntimeError("old torch.backends.cuda.sdp_kernel API lacks enable_cudnn")
    return kwargs
