"""Bounded, pickle-free persistence for frozen CPU MM-GCoT cache bundles.

Each bundle is published as one directory containing ``tensors.safetensors``
and ``metadata.json``. A directory rename makes the complete pair visible at
once. Loading accepts only known Transformers dynamic-cache classes.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from transformers import cache_utils


SCHEMA = "mmgcot_formal_cache_v2"
VERSION = 1
DEFAULT_MAX_BYTES = 8 * 1024**3
MAX_METADATA_BYTES = 4 * 1024**2
_ARMS = ("L", "R", "E")
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_LAYER_CLASSES = {
    name: getattr(cache_utils, name)
    for name in (
        "DynamicLayer",
        "DynamicSlidingWindowLayer",
        "LinearAttentionLayer",
        "LinearAttentionAndFullAttentionLayer",
    )
    if hasattr(cache_utils, name)
}


class CacheBundleError(ValueError):
    """Invalid, drifted, or unsupported frozen cache bundle."""


def _digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _check_hashes(hashes: object) -> dict[str, str]:
    if not isinstance(hashes, Mapping) or not hashes:
        raise CacheBundleError("hashes must be a nonempty mapping")
    result: dict[str, str] = {}
    for key, value in hashes.items():
        if not isinstance(key, str) or not key or not isinstance(value, str) or not _HEX.fullmatch(value):
            raise CacheBundleError("hashes must map names to lowercase SHA-256 digests")
        result[key] = value
    return result


def _check_limit(max_bytes: int) -> None:
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")


def _encode(value: object, tensors: dict[str, torch.Tensor], name: str,
            budget: list[int], max_bytes: int) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.requires_grad:
            raise CacheBundleError(f"{name} must be a detached CPU tensor")
        budget[0] += value.numel() * value.element_size()
        if budget[0] > max_bytes:
            raise CacheBundleError("tensor payload exceeds max_bytes")
        tensor_name = f"tensor_{len(tensors):08d}"
        tensors[tensor_name] = value.detach().contiguous().clone()
        return {"kind": "tensor", "name": tensor_name, "dtype": str(value.dtype),
                "shape": list(value.shape)}
    if value is None or type(value) in (bool, int, str):
        return {"kind": "primitive", "value": value}
    if type(value) is float:
        if not math.isfinite(value):
            raise CacheBundleError(f"{name} contains a nonfinite float")
        return {"kind": "primitive", "value": value}
    if isinstance(value, torch.dtype):
        return {"kind": "dtype", "value": str(value)}
    if isinstance(value, torch.device):
        if value.type != "cpu":
            raise CacheBundleError(f"{name} contains a non-CPU device")
        return {"kind": "device", "value": str(value)}
    if isinstance(value, type) and value in _LAYER_CLASSES.values():
        return {"kind": "layer_class", "name": value.__name__}
    if type(value) in (list, tuple):
        return {"kind": type(value).__name__, "items": [
            _encode(item, tensors, f"{name}.{i}", budget, max_bytes)
            for i, item in enumerate(value)]}
    if type(value) is dict and all(isinstance(key, str) for key in value):
        return {"kind": "dict", "items": {
            key: _encode(item, tensors, f"{name}.{key}", budget, max_bytes)
            for key, item in sorted(value.items())}}
    raise CacheBundleError(f"unsupported cache attribute {name}: {type(value).__name__}")


def _encode_object(obj: object, tensors: dict[str, torch.Tensor], prefix: str,
                   budget: list[int], max_bytes: int) -> dict[str, Any]:
    if prefix in _ARMS:
        if type(obj) is not cache_utils.DynamicCache:
            raise CacheBundleError(f"{prefix} must be a DynamicCache")
        if getattr(obj, "offloading", False):
            raise CacheBundleError("offloaded caches cannot be frozen as CPU bundles")
        attrs = {key: value for key, value in vars(obj).items() if key != "layers"}
    elif type(obj) not in _LAYER_CLASSES.values():
        raise CacheBundleError(f"unsupported cache layer class: {type(obj).__name__}")
    else:
        attrs = vars(obj)
    return {"class": type(obj).__name__, "attrs": {
        key: _encode(value, tensors, f"{prefix}.{key}", budget, max_bytes)
        for key, value in sorted(attrs.items())}}


def _decode(node: object, tensors: dict[str, torch.Tensor], used: set[str]) -> Any:
    if not isinstance(node, dict) or not isinstance(node.get("kind"), str):
        raise CacheBundleError("invalid attribute descriptor")
    kind = node["kind"]
    if kind == "tensor":
        name = node.get("name")
        if not isinstance(name, str) or name not in tensors or name in used:
            raise CacheBundleError("missing or duplicated tensor")
        tensor = tensors[name]
        if str(tensor.dtype) != node.get("dtype") or list(tensor.shape) != node.get("shape"):
            raise CacheBundleError(f"tensor dtype or shape mismatch: {name}")
        used.add(name)
        return tensor
    if kind == "primitive":
        value = node.get("value")
        if type(value) not in (type(None), bool, int, float, str) or (
            type(value) is float and not math.isfinite(value)
        ):
            raise CacheBundleError("invalid primitive attribute")
        return value
    if kind == "dtype":
        value = node.get("value")
        if not isinstance(value, str) or not value.startswith("torch."):
            raise CacheBundleError("invalid dtype attribute")
        dtype = getattr(torch, value[6:], None)
        if not isinstance(dtype, torch.dtype):
            raise CacheBundleError("unknown dtype attribute")
        return dtype
    if kind == "device":
        if node.get("value") != "cpu":
            raise CacheBundleError("invalid device attribute")
        return torch.device("cpu")
    if kind == "layer_class":
        name = node.get("name")
        if not isinstance(name, str) or name not in _LAYER_CLASSES:
            raise CacheBundleError("unknown layer class attribute")
        return _LAYER_CLASSES[name]
    if kind in ("list", "tuple"):
        items = node.get("items")
        if not isinstance(items, list):
            raise CacheBundleError("invalid sequence attribute")
        values = [_decode(item, tensors, used) for item in items]
        return values if kind == "list" else tuple(values)
    if kind == "dict":
        items = node.get("items")
        if not isinstance(items, dict) or not all(isinstance(key, str) for key in items):
            raise CacheBundleError("invalid mapping attribute")
        return {key: _decode(item, tensors, used) for key, item in items.items()}
    raise CacheBundleError(f"unknown attribute kind: {kind}")


def _decode_object(node: object, tensors: dict[str, torch.Tensor], used: set[str],
                   *, cache: bool) -> object:
    if not isinstance(node, dict) or not isinstance(node.get("attrs"), dict):
        raise CacheBundleError("invalid cache object descriptor")
    name = node.get("class")
    classes = {"DynamicCache": cache_utils.DynamicCache} if cache else _LAYER_CLASSES
    if not isinstance(name, str) or name not in classes:
        raise CacheBundleError(f"unsupported cache class: {name}")
    attrs = node["attrs"]
    if not all(isinstance(key, str) for key in attrs):
        raise CacheBundleError("invalid cache attribute name")
    obj = classes[name].__new__(classes[name])
    vars(obj).update({key: _decode(value, tensors, used) for key, value in attrs.items()})
    return obj


def save_cache_bundle(path: str | Path, bundle: Mapping[str, Any], *,
                      max_bytes: int = DEFAULT_MAX_BYTES) -> Path:
    """Atomically create a frozen bundle directory; never overwrite an existing one.

    ``bundle`` requires L/R/E DynamicCaches, a CPU ``rope_deltas`` tensor,
    integer ``opening_id``, and provenance ``hashes``. Other fields are not
    serialized. ``max_bytes`` bounds the aggregate raw tensors and output file.
    """
    _check_limit(max_bytes)
    if not isinstance(bundle, Mapping):
        raise CacheBundleError("bundle must be a mapping")
    if type(bundle.get("opening_id")) is not int or bundle["opening_id"] < 0:
        raise CacheBundleError("opening_id must be a non-negative integer")
    hashes = _check_hashes(bundle.get("hashes"))
    tensors: dict[str, torch.Tensor] = {}
    budget = [0]
    arms = {}
    for arm in _ARMS:
        cache = bundle.get(arm)
        arms[arm] = _encode_object(cache, tensors, arm, budget, max_bytes)
        layers = getattr(cache, "layers", None)
        if not isinstance(layers, list):
            raise CacheBundleError(f"{arm}.layers must be a list")
        arms[arm]["layers"] = [
            _encode_object(layer, tensors, f"{arm}.layers.{i}", budget, max_bytes)
            for i, layer in enumerate(layers)]
    rope = _encode(bundle.get("rope_deltas"), tensors, "rope_deltas", budget, max_bytes)
    if rope["kind"] != "tensor":
        raise CacheBundleError("rope_deltas must be a CPU tensor")
    target = Path(path)
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=target.parent))
    try:
        tensor_path = staging / "tensors.safetensors"
        save_file(tensors, str(tensor_path))
        size = tensor_path.stat().st_size
        if size > max_bytes:
            raise CacheBundleError("tensor file exceeds max_bytes")
        payload = {"schema": SCHEMA, "version": VERSION, "hashes": hashes,
                   "opening_id": bundle["opening_id"], "arms": arms,
                   "rope_deltas": rope, "tensor_sha256": _digest(tensor_path),
                   "tensor_bytes": size, "tensor_names": sorted(tensors)}
        payload["metadata_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
        metadata = _canonical(payload)
        if len(metadata) > MAX_METADATA_BYTES:
            raise CacheBundleError("metadata exceeds size limit")
        with (staging / "metadata.json").open("wb") as stream:
            stream.write(metadata)
            stream.flush()
            os.fsync(stream.fileno())
        with tensor_path.open("rb") as stream:
            os.fsync(stream.fileno())
        staging_fd = os.open(staging, os.O_RDONLY)
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        if target.exists():
            raise FileExistsError(target)
        os.rename(staging, target)
        directory_fd = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return target


def load_cache_bundle(path: str | Path, *, expected_hashes: Mapping[str, str] | None = None,
                      max_bytes: int = DEFAULT_MAX_BYTES) -> dict[str, Any]:
    """Validate integrity/provenance and return independent CPU cache objects."""
    _check_limit(max_bytes)
    root = Path(path)
    meta_path, tensor_path = root / "metadata.json", root / "tensors.safetensors"
    if meta_path.stat().st_size > MAX_METADATA_BYTES:
        raise CacheBundleError("metadata exceeds size limit")
    try:
        payload = json.loads(meta_path.read_bytes())
    except (ValueError, UnicodeError) as error:
        raise CacheBundleError("invalid metadata JSON") from error
    if (not isinstance(payload, dict) or payload.get("schema") != SCHEMA
            or type(payload.get("version")) is not int or payload["version"] != VERSION):
        raise CacheBundleError("cache schema/version mismatch")
    recorded = payload.get("metadata_sha256")
    without_hash = {key: value for key, value in payload.items() if key != "metadata_sha256"}
    try:
        calculated = hashlib.sha256(_canonical(without_hash)).hexdigest()
    except (TypeError, ValueError) as error:
        raise CacheBundleError("invalid metadata values") from error
    if not isinstance(recorded, str) or recorded != calculated:
        raise CacheBundleError("metadata SHA-256 mismatch")
    hashes = _check_hashes(payload.get("hashes"))
    if expected_hashes is not None:
        expected = _check_hashes(expected_hashes)
        if any(hashes.get(key) != value for key, value in expected.items()):
            raise CacheBundleError("provenance hash mismatch")
    if type(payload.get("opening_id")) is not int or payload["opening_id"] < 0:
        raise CacheBundleError("invalid opening_id")
    size = tensor_path.stat().st_size
    if size > max_bytes or size != payload.get("tensor_bytes"):
        raise CacheBundleError("tensor file size mismatch or exceeds max_bytes")
    if _digest(tensor_path) != payload.get("tensor_sha256"):
        raise CacheBundleError("tensor SHA-256 mismatch")
    try:
        tensors = load_file(str(tensor_path), device="cpu")
    except Exception as error:
        raise CacheBundleError("invalid safetensors payload") from error
    if sorted(tensors) != payload.get("tensor_names"):
        raise CacheBundleError("tensor name mismatch")
    used: set[str] = set()
    arms = payload.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(_ARMS):
        raise CacheBundleError("invalid cache arms")
    result: dict[str, Any] = {}
    for arm in _ARMS:
        spec = arms[arm]
        cache = _decode_object(spec, tensors, used, cache=True)
        layers = spec.get("layers")
        if not isinstance(layers, list):
            raise CacheBundleError("invalid cache layers")
        cache.layers = [_decode_object(layer, tensors, used, cache=False) for layer in layers]
        if getattr(cache, "offloading", False):
            raise CacheBundleError("offloaded cache is unsupported")
        result[arm] = cache
    result["rope_deltas"] = _decode(payload.get("rope_deltas"), tensors, used)
    if not isinstance(result["rope_deltas"], torch.Tensor) or used != set(tensors):
        raise CacheBundleError("unreferenced tensors or invalid rope_deltas")
    result["opening_id"] = payload["opening_id"]
    result["hashes"] = hashes
    return result


__all__ = ["CacheBundleError", "DEFAULT_MAX_BYTES", "SCHEMA", "VERSION",
           "save_cache_bundle", "load_cache_bundle"]
