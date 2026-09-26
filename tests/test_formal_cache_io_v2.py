"""Roundtrip and rejection tests for bounded formal cache persistence."""

from __future__ import annotations

import hashlib
import json

import pytest
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer

from mmgcot_timeline_training.formal_cache_io import (
    CacheBundleError,
    load_cache_bundle,
    save_cache_bundle,
)


def _bundle():
    caches = {}
    for index, arm in enumerate(("L", "R", "E")):
        cache = DynamicCache()
        dtype = (torch.float16, torch.bfloat16, torch.float32)[index]
        key = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4).to(dtype)
        cache.update(key, -key, 0)
        if arm == "R":
            sliding = DynamicSlidingWindowLayer(sliding_window=8)
            sliding.keys = key.clone()
            sliding.values = (-key).clone()
            sliding.is_initialized = True
            sliding.dtype = dtype
            sliding.device = torch.device("cpu")
            cache.layers.append(sliding)
        else:
            cache.layers.append(DynamicLayer())
        cache.layers[0].sample_attrs = {"flag": True, "pair": (1, "two"),
                                         "offsets": [0, None, 2.5]}
        cache.extra_tensor = torch.tensor([index], dtype=torch.int32)
        caches[arm] = cache
    return {**caches, "rope_deltas": torch.tensor([[2, 3]], dtype=torch.int64),
            "opening_id": 17, "hashes": {"trajectory_sha256": "a" * 64,
                                            "image_sha256": "b" * 64}}


def _rehash_metadata(path):
    metadata = path / "metadata.json"
    value = json.loads(metadata.read_text())
    value.pop("metadata_sha256", None)
    value["metadata_sha256"] = hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode()).hexdigest()
    metadata.write_text(json.dumps(value), encoding="utf-8")


def test_synthetic_dynamic_cache_roundtrip_and_immutable_original(tmp_path):
    source = _bundle()
    original_key = source["L"].layers[0].keys.clone()
    path = tmp_path / "cache"
    assert save_cache_bundle(path, source) == path
    restored = load_cache_bundle(path, expected_hashes=source["hashes"])
    assert set(restored) == {"L", "R", "E", "rope_deltas", "opening_id", "hashes"}
    assert restored["opening_id"] == 17
    assert restored["hashes"] == source["hashes"]
    assert restored["rope_deltas"].dtype == torch.int64
    assert torch.equal(restored["rope_deltas"], source["rope_deltas"])
    for arm in ("L", "R", "E"):
        left, right = source[arm], restored[arm]
        assert type(right) is type(left) is DynamicCache
        assert right.layer_class_to_replicate is left.layer_class_to_replicate
        assert right.offloading == left.offloading
        assert right.extra_tensor.dtype == torch.int32
        assert torch.equal(right.extra_tensor, left.extra_tensor)
        assert right.extra_tensor.data_ptr() != left.extra_tensor.data_ptr()
        for before, after in zip(left.layers, right.layers, strict=True):
            assert type(after) is type(before)
            assert vars(after).keys() == vars(before).keys()
            for name, value in vars(before).items():
                loaded = getattr(after, name)
                if isinstance(value, torch.Tensor):
                    assert loaded.dtype == value.dtype
                    assert loaded.device.type == "cpu"
                    assert torch.equal(loaded, value)
                    assert loaded.data_ptr() != value.data_ptr()
                else:
                    assert loaded == value
    restored["L"].layers[0].keys.add_(100)
    restored["L"].layers[0].sample_attrs["offsets"].append(99)
    assert torch.equal(source["L"].layers[0].keys, original_key)
    assert source["L"].layers[0].sample_attrs["offsets"] == [0, None, 2.5]
    assert not source["L"].layers[0].keys.requires_grad
    with pytest.raises(FileExistsError):
        save_cache_bundle(path, source)


def test_corruption_schema_provenance_and_class_rejected(tmp_path):
    path = tmp_path / "cache"
    source = _bundle()
    save_cache_bundle(path, source)
    with pytest.raises(CacheBundleError, match="provenance"):
        load_cache_bundle(path, expected_hashes={"trajectory_sha256": "c" * 64})

    tensor_path = path / "tensors.safetensors"
    original_tensor = tensor_path.read_bytes()
    tensor_path.write_bytes(original_tensor[:-1] + bytes([original_tensor[-1] ^ 1]))
    with pytest.raises(CacheBundleError, match="tensor SHA-256"):
        load_cache_bundle(path)
    tensor_path.write_bytes(original_tensor)

    metadata_path = path / "metadata.json"
    original_meta = metadata_path.read_text()
    metadata_path.write_text(original_meta.replace("mmgcot_formal_cache_v2", "wrong_schema"))
    with pytest.raises(CacheBundleError, match="schema/version"):
        load_cache_bundle(path)
    metadata_path.write_text(original_meta)

    value = json.loads(original_meta)
    value["version"] = 2
    metadata_path.write_text(json.dumps(value))
    with pytest.raises(CacheBundleError, match="schema/version"):
        load_cache_bundle(path)
    metadata_path.write_text(original_meta)

    metadata_path.write_text("{broken")
    with pytest.raises(CacheBundleError, match="invalid metadata JSON"):
        load_cache_bundle(path)
    metadata_path.write_text(original_meta)

    value = json.loads(original_meta)
    value["arms"]["R"]["layers"][0]["class"] = "ArbitraryObject"
    metadata_path.write_text(json.dumps(value))
    with pytest.raises(CacheBundleError, match="metadata SHA-256"):
        load_cache_bundle(path)
    _rehash_metadata(path)
    with pytest.raises(CacheBundleError, match="unsupported cache class"):
        load_cache_bundle(path)


def test_bounds_and_cpu_frozen_validation_leave_no_output(tmp_path):
    source = _bundle()
    path = tmp_path / "too_small"
    with pytest.raises(CacheBundleError, match="max_bytes"):
        save_cache_bundle(path, source, max_bytes=1)
    assert not path.exists()
    valid = tmp_path / "valid"
    save_cache_bundle(valid, source)
    with pytest.raises(CacheBundleError, match="max_bytes"):
        load_cache_bundle(valid, max_bytes=1)
    source["L"].layers[0].keys.requires_grad_(True)
    with pytest.raises(CacheBundleError, match="detached CPU"):
        save_cache_bundle(tmp_path / "grad", source)
    assert not (tmp_path / "grad").exists()
