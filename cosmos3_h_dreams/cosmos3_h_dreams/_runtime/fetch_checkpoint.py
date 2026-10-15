# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch explicitly supplied Cosmos DCP checkpoints into the FlashDreams cache."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from cosmos3_h_dreams._runtime.checkpoint_load import resolve_credential_path
from cosmos3_h_dreams._runtime.checkpoint_remap import remap_key

# Converted checkpoints cache under the shared FlashDreams cache root
# (``$FLASHDREAMS_CACHE_DIR``, default ``~/.cache/flashdreams``),
# namespaced under ``cosmos3_h_dreams/`` so they don't collide with the raw
# ``<bucket>/<key>`` downloads the core loader caches there.
_CACHE_ENV_VAR = "FLASHDREAMS_CACHE_DIR"
_DEFAULT_CACHE_ROOT = "~/.cache/flashdreams"


def _cache_dir() -> Path:
    """Base dir for converted checkpoints (respects ``$FLASHDREAMS_CACHE_DIR``)."""
    root = os.environ.get(_CACHE_ENV_VAR) or _DEFAULT_CACHE_ROOT
    return Path(root).expanduser() / "cosmos3_h_dreams"


KNOWN_CHECKPOINTS: dict[str, str] = {}
"""Optional checkpoint aliases supplied by a deployment; no private defaults ship."""


def resolve_checkpoint_arg(value: str) -> str:
    """A registry name -> its full URI; anything else (s3:// or local path) is returned as-is."""
    return KNOWN_CHECKPOINTS.get(value, value)


def ensure_checkpoint(
    spec: str, *, credential_path: str | None = None, prefer_ema: bool = True
) -> str:
    """Resolve a checkpoint spec to a locally-loadable source, auto-fetching + converting a
    raw DCP from ``s3://`` if it isn't already cached (a no-op when cached).

    ``spec`` may be a registry name (``KNOWN_CHECKPOINTS``), an ``s3://`` raw DCP dir, an
    ``s3://`` exported ``.safetensors``, a Hugging Face index URL, or a local dir /
    ``index.json``. Returns the path/URI to hand to ``load_exported_checkpoint``. The
    credential is auto-selected from the bucket, so no explicit credential is needed as
    long as the matching ``*.secret`` is under ``credentials/``.
    """
    resolved = resolve_checkpoint_arg(str(spec))
    if not resolved.startswith("s3://"):
        return resolved  # local dir / index.json / HF URL -> load_exported_checkpoint handles it
    if resolved.rstrip("/").endswith(".safetensors"):
        return resolved  # already an exported single-file on s3
    cred = resolve_credential_path(credential_path, checkpoint_source=resolved)
    out = _default_out_dir(resolved)
    print(
        f"[fetch] ensuring checkpoint is local: {resolved}\n        -> {out}",
        flush=True,
    )
    return str(fetch_and_convert(resolved, cred, out, prefer_ema=prefer_ema))


def _select_source_prefix(raw_meta: dict[str, Any], prefer_ema: bool = True) -> str:
    """Pick which model copy to convert from a DCP that holds several.

    A training DCP holds ``net`` (student), ``net_ema`` (EMA), and
    ``net_fake_score`` (critic). Prefer ``net_ema.`` when present
    (``prefer_ema=True``, default);
    fall back to the student ``net.`` otherwise. Pass ``prefer_ema=False`` to force
    the student copy when available. EMA-only checkpoints always use EMA weights.
    """
    top = {k.split(".", 1)[0] for k in raw_meta}
    if prefer_ema and "net_ema" in top:
        return "net_ema."
    if "net" in top:
        return "net."
    if "net_ema" in top:
        return "net_ema."
    return "net."  # default; nothing will map if the layout is unexpected


def _normalize_raw_key(key: str, source_prefix: str = "net.") -> str:
    """Normalize a raw DCP key toward the exported ``model.net.*`` convention.

    The exported checkpoint keys start with ``model.net.``; the raw DCP starts with ``net.`` (student) or ``net_ema.`` (EMA-only
    inference cache). ``source_prefix`` selects which copy; it is rewritten to
    ``model.net.`` so both flow through the same remap. Also strips a
    ``torch.compile`` ``_orig_mod.`` shim.
    """
    key = key.replace("_orig_mod.", "")
    if source_prefix and key.startswith(source_prefix):
        return "model.net." + key[len(source_prefix) :]
    return key


def _storage_reader(path: str, credential_path: str | None) -> Any:
    from flashdreams.core.checkpoint.load import get_storage_reader

    kwargs: dict[str, object] = {}
    if credential_path:
        kwargs["credential_path"] = credential_path
    return get_storage_reader(path, **kwargs)  # type: ignore[arg-type]


def _install_pathlib_compat_shim() -> None:
    """Let a Python-3.13-pickled DCP ``.metadata`` unpickle under Python < 3.13.

    Python 3.13 split ``pathlib`` into a package with ``pathlib._local`` (where
    ``PosixPath`` etc. now live), so a ``.metadata`` pickled by a 3.13 env
    references ``pathlib._local.PosixPath``. Under 3.12 ``pathlib`` is a single
    module and that import fails (``'pathlib' is not a package``). Aliasing the
    submodule name to the ``pathlib`` module resolves the reference (3.12's
    ``pathlib.PosixPath`` is the same class).
    """
    import pathlib
    import sys

    if not hasattr(pathlib, "_local") and "pathlib._local" not in sys.modules:
        sys.modules["pathlib._local"] = pathlib


def _resolve_dcp_dir(
    checkpoint_path: str, credential_path: str | None
) -> tuple[str, Any]:
    """Find the DCP dir that actually holds ``.metadata``.

    Checkpoint components may live in subdirectories (``model``, ``optim``),
    so the parent directory may have no ``.metadata``. Try ``<path>`` then
    ``<path>/model`` and return whichever reads.
    """
    _install_pathlib_compat_shim()  # tolerate Python 3.13 metadata under 3.12
    base = checkpoint_path.rstrip("/")
    tried: list[str] = []
    for candidate in (base, base + "/model"):
        try:
            metadata = _storage_reader(candidate, credential_path).read_metadata()
            return candidate, metadata
        except Exception as exc:  # noqa: BLE001 - probe which layout the bucket uses
            tried.append(f"{candidate} ({type(exc).__name__})")
    raise SystemExit("could not read a DCP .metadata; tried: " + "; ".join(tried))


def read_dcp_key_metadata(
    checkpoint_path: str, credential_path: str | None
) -> tuple[str, dict[str, Any]]:
    """Return ``(resolved_dcp_dir, {raw_key: TensorStorageMetadata})`` (no tensor reads)."""
    resolved, metadata = _resolve_dcp_dir(checkpoint_path, credential_path)
    return resolved, dict(metadata.state_dict_metadata)


def _tensor_shape_dtype(entry: Any) -> tuple[tuple[int, ...], Any] | None:
    """Extract ``(shape, dtype)`` from a DCP metadata entry, or None if not a tensor."""
    size = getattr(entry, "size", None)
    props = getattr(entry, "properties", None)
    dtype = getattr(props, "dtype", None)
    if size is None or dtype is None:
        return None
    return tuple(int(s) for s in size), dtype


def analyze_keys(
    raw_meta: dict[str, Any], prefer_ema: bool = True
) -> dict[str, object]:
    """Report how many raw keys map to ``CosmosMoTNetwork`` targets via the remap."""
    source_prefix = _select_source_prefix(raw_meta, prefer_ema)
    mapped: list[str] = []
    unmapped: list[str] = []
    for key in raw_meta:
        target = remap_key(_normalize_raw_key(key, source_prefix))
        (mapped if target is not None else unmapped).append(key)
    top_prefixes = Counter(k.split(".")[0] for k in raw_meta)
    return {
        "num_keys": len(raw_meta),
        "source_prefix": source_prefix,
        "num_mapped": len(mapped),
        "num_unmapped": len(unmapped),
        "top_level_prefixes": dict(top_prefixes),
        "mapped_examples": sorted(mapped)[:8],
        "unmapped_examples": sorted(unmapped)[:20],
    }


def _interesting_key_shapes(
    raw_meta: dict[str, Any], substrings: tuple[str, ...]
) -> list[tuple[str, tuple[int, ...], Any]]:
    """``(key, shape, dtype)`` for keys containing any of ``substrings`` (e.g. action/sound).

    Used by the probe to reveal the conditioning-pathway modules and their dims
    (e.g. ``action2llm.fc.weight`` → num_embodiment_domains × (action_dim·hidden)),
    which the native tree + remap must reproduce for the camera checkpoint.
    """
    rows: list[tuple[str, tuple[int, ...], Any]] = []
    for key in sorted(raw_meta):
        low = key.lower()
        if not any(sub in low for sub in substrings):
            continue
        sd = _tensor_shape_dtype(raw_meta[key])
        if sd is not None:
            rows.append((key, sd[0], sd[1]))
    return rows


def probe(checkpoint_path: str, credential_path: str | None) -> dict[str, object]:
    resolved, raw_meta = read_dcp_key_metadata(checkpoint_path, credential_path)
    print(f"resolved DCP dir: {resolved}")
    report = analyze_keys(raw_meta)
    print(
        f"DCP probe: {report['num_keys']} keys; top-level prefixes: {report['top_level_prefixes']}"
    )
    print(
        f"  remap coverage: {report['num_mapped']} mapped / {report['num_unmapped']} unmapped"
    )
    print(f"  mapped examples:   {report['mapped_examples']}")
    print(f"  unmapped examples: {report['unmapped_examples']}")
    # Surface the conditioning-pathway modules + dims for the camera checkpoint so a
    # single probe run tells us exactly what the native action/sound tree must hold.
    cond = _interesting_key_shapes(raw_meta, ("action", "sound", "modality"))
    print(f"  conditioning-pathway keys ({len(cond)}):")
    for key, shape, dtype in cond:
        print(f"    {key}  shape={tuple(shape)} dtype={dtype}")
    report["conditioning_keys"] = [(k, list(s)) for k, s, _ in cond]
    return report


def _default_out_dir(checkpoint_path: str) -> Path:
    """Collision-free cache dir: ``<experiment>__<iter>`` under the cache root.

    Two different checkpoints can share the same ``iter_000004000`` leaf (e.g. the
    noaction and camera variants), so the bare leaf name collides. Prefix it with
    the experiment segment (the dir before ``checkpoints/``) so each checkpoint
    caches to its own dir.
    """
    parts = [p for p in checkpoint_path.rstrip("/").split("/") if p]
    if len(parts) >= 2 and parts[-1] == "model" and parts[-2].startswith("iter_"):
        parts.pop()
    leaf = parts[-1] if parts else "checkpoint"
    experiment = None
    # Nested caches can contain several 'checkpoints' segments; the experiment
    # name immediately precedes the innermost one.
    checkpoint_idxs = [i for i, p in enumerate(parts) if p == "checkpoints"]
    if checkpoint_idxs and checkpoint_idxs[-1] > 0:
        experiment = parts[checkpoint_idxs[-1] - 1]
    name = f"{experiment}__{leaf}" if experiment else leaf
    return _cache_dir() / name


def fetch_and_convert(
    checkpoint_path: str,
    credential_path: str | None,
    out_dir: Path,
    *,
    dtype: torch.dtype = torch.bfloat16,
    force: bool = False,
    prefer_ema: bool = True,
) -> Path:
    """Load the mapped weights from the DCP checkpoint, remap, and cache as safetensors.

    Skips the (slow, ~30 GB) download when ``out_dir`` already holds a converted
    checkpoint, unless ``force`` is set. "Converted" means the index exists AND
    lists the action keys when the source is a camera checkpoint — so a stale
    pre-action-remap cache is treated as a miss and re-fetched.
    """
    from safetensors.torch import save_file
    from torch.distributed.checkpoint import load as dcp_load
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    index_path = out_dir / "model.safetensors.index.json"
    if index_path.is_file() and not force:
        cached_keys = set(json.loads(index_path.read_text()).get("weight_map", {}))
        print(
            f"already cached: {len(cached_keys)} tensors -> {out_dir} (use --force to re-fetch)"
        )
        print(
            f"point the pipeline at it:  export COSMOS3_H_DREAMS_CHECKPOINT={out_dir}"
        )
        return out_dir

    resolved, raw_meta = read_dcp_key_metadata(checkpoint_path, credential_path)
    # Select the weights our remap accepts (EMA ``net_ema.*`` by default,
    # else student ``net.*``); skip the
    # ``net_fake_score`` critic. Load keyed by the raw DCP key (DCP matches by
    # name), then re-key to the exported ``model.net.*`` form so the cached file
    # loads through the standard ``load_exported_checkpoint`` remap.
    source_prefix = _select_source_prefix(raw_meta, prefer_ema)
    load_sd: dict[str, torch.Tensor] = {}
    raw_to_exported: dict[str, str] = {}
    for raw_key, entry in raw_meta.items():
        exported = _normalize_raw_key(raw_key, source_prefix)
        if remap_key(exported) is None:
            continue
        sd = _tensor_shape_dtype(entry)
        if sd is None:
            continue
        shape, src_dtype = sd
        load_sd[raw_key] = torch.empty(shape, dtype=src_dtype)
        raw_to_exported[raw_key] = exported
    if not load_sd:
        raise SystemExit(
            "no DCP keys mapped to CosmosMoTNetwork; run --probe and adjust the remap."
        )
    print(
        f"source copy: {source_prefix.rstrip('.')} "
        f"({'EMA' if source_prefix == 'net_ema.' else 'student net'})"
    )

    approx_gb = sum(t.numel() * t.element_size() for t in load_sd.values()) / 1e9
    n_action = sum(1 for k in raw_to_exported.values() if "action" in k)
    print(
        f"reading {len(load_sd)} tensors (~{approx_gb:.1f} GB, {n_action} action) from {resolved} ...\n"
        "  (this is the slow, network-bound DCP read; it is silent until done)",
        flush=True,
    )
    reader = _storage_reader(resolved, credential_path)
    dcp_load(
        load_sd,
        storage_reader=reader,
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )

    exported = {
        raw_to_exported[k]: v.to(dtype).contiguous() for k, v in load_sd.items()
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(exported, str(out_dir / "model.safetensors"))
    index = {
        "metadata": {
            "total_size": sum(v.numel() * v.element_size() for v in exported.values())
        },
        "weight_map": {k: "model.safetensors" for k in exported},
    }
    (out_dir / "model.safetensors.index.json").write_text(json.dumps(index))
    print(
        f"fetched + converted {len(exported)} tensors ({source_prefix.rstrip('.')}, no ViT) -> {out_dir}"
    )
    print(f"point the pipeline at it:  export COSMOS3_H_DREAMS_CHECKPOINT={out_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=None,
        help="a registry name (see --list) OR a DCP checkpoint dir (s3:// or local)",
    )
    parser.add_argument(
        "--list", action="store_true", help="list known checkpoint names and exit"
    )
    parser.add_argument(
        "--credential-path",
        default=None,
        help="S3/GCS credential file (default: $COSMOS_CHECKPOINT_CREDENTIAL, else the "
        "secret matching the bucket under <repo>/credentials/: the AWS bucket uses "
        "s3_checkpoint.secret, the GCS bucket gcp_checkpoint.secret)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output dir (default: $FLASHDREAMS_CACHE_DIR/cosmos3_h_dreams/"
        "<experiment>__<iter>, i.e. ~/.cache/flashdreams/cosmos3_h_dreams/...)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-download + reconvert even if the out dir is already cached",
    )
    parser.add_argument(
        "--probe", action="store_true", help="only dump raw DCP keys + remap coverage"
    )
    parser.add_argument(
        "--prefer-ema",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="convert the EMA copy (net_ema.*) when available "
        "(default); use --no-prefer-ema for the student net.* copy",
    )
    args = parser.parse_args()

    if args.list:
        print("known checkpoints (use the name as the argument):")
        for name, uri in KNOWN_CHECKPOINTS.items():
            print(f"  {name}\n      {uri}")
        return

    if not args.checkpoint:
        parser.error("give a checkpoint name (see --list) or an s3:// / local DCP path")

    checkpoint_path = resolve_checkpoint_arg(args.checkpoint)
    if args.checkpoint in KNOWN_CHECKPOINTS:
        print(f"[fetch] '{args.checkpoint}' -> {checkpoint_path}")

    credential_path = resolve_credential_path(
        args.credential_path, checkpoint_source=checkpoint_path
    )
    if credential_path:
        print(f"using credential: {credential_path}")

    if args.probe:
        probe(checkpoint_path, credential_path)
        return

    out_dir = args.out or _default_out_dir(checkpoint_path)
    fetch_and_convert(
        checkpoint_path,
        credential_path,
        out_dir,
        force=args.force,
        prefer_ema=args.prefer_ema,
    )


if __name__ == "__main__":
    main()
