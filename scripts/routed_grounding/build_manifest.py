#!/usr/bin/env python3
# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build a small, deterministic RefCOCOg UMD grounding manifest.

The REFER package contains the referring expressions and their target
annotation ids, while COCO is the source of image and instance metadata.  The
two APIs are deliberately kept separate here: a REFER annotation is accepted
only when its target can be loaded and validated through COCO.

The script does not copy images.  Each record stores the path to the existing
image under ``--coco-images``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import math
import random
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

DEFAULT_SEED = 3407
DATASET_NAME = "refcocog"
SPLIT_BY = "umd"
PathLike = str | Path

# These names are intentionally module globals.  Besides making the selected
# APIs obvious, this gives tests (and downstream callers with a compatible
# implementation) a small, dependency-free injection point.
REFER: Optional[type] = None
COCO: Optional[type] = None


class ManifestError(RuntimeError):
    """Raised when the input data cannot produce a valid manifest."""


def _sort_atom(value: Any) -> tuple[int, Any]:
    """Return a total-order key for ids from APIs that mix int and str ids."""

    if value is None:
        return (0, "")
    if isinstance(value, bool):
        return (1, int(value))
    if isinstance(value, int | float) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            return (2, number)
    return (3, str(value))


def _id_key(value: Any) -> str:
    """Return a stable comparison key for an API id."""

    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _same_id(left: Any, right: Any) -> bool:
    return left == right or _id_key(left) == _id_key(right)


def _jsonable(value: Any) -> Any:
    """Convert common COCO values to values accepted by ``json.dump``."""

    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]

    # Numpy scalar/array values occasionally appear in custom COCO wrappers;
    # avoid importing numpy just for this optional compatibility path.
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return _jsonable(item_method())
        except (TypeError, ValueError):
            pass
    tolist_method = getattr(value, "tolist", None)
    if callable(tolist_method):
        try:
            return _jsonable(tolist_method())
        except (TypeError, ValueError):
            pass
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple) or isinstance(value, set):
        return list(value)
    if isinstance(value, Mapping):
        return [value]
    if isinstance(value, Iterable) and not isinstance(value, str | bytes):
        return list(value)
    return [value]


def _load_refer_class() -> type:
    """Import the REFER API without requiring it at module import time."""

    if REFER is not None:
        return REFER

    errors: list[BaseException] = []
    # The released REFER repository is commonly checked out as either
    # ``<root>/refer_api/refer.py`` or ``<root>/refer.py``.  A pre-installed
    # ``refer`` module is tried first so callers can provide a modern port.
    for candidate in ("refer", "refer_api.refer"):
        try:
            module = importlib.import_module(candidate)
            loaded = getattr(module, "REFER", None)
            if loaded is not None:
                return loaded
        except (ImportError, ModuleNotFoundError, SyntaxError, OSError) as error:
            errors.append(error)

    raise ImportError(
        "Could not import the REFER API. Install/import a module named "
        "'refer' or pass a compatible class through build_manifest(..., "
        "refer_cls=...)."
    ) from (errors[-1] if errors else None)


def _import_refer_class(refer_root: Path) -> type:
    """Load REFER from a normal installation or from ``refer_root``."""

    if REFER is not None:
        return REFER

    search_dirs = [
        refer_root,
        refer_root / "refer_api",
        refer_root / "refer_api" / "refer_api",
    ]
    errors: list[BaseException] = []
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        directory_string = str(directory)
        inserted = directory_string not in sys.path
        if inserted:
            sys.path.insert(0, directory_string)
        try:
            # A previous failed import can leave a partially initialized
            # module in sys.modules, so remove only modules that failed here.
            for module_name in ("refer", "refer_api.refer"):
                try:
                    module = importlib.import_module(module_name)
                    loaded = getattr(module, "REFER", None)
                    if loaded is not None:
                        return loaded
                except (ImportError, ModuleNotFoundError, SyntaxError, OSError) as error:
                    errors.append(error)
        finally:
            if inserted:
                try:
                    sys.path.remove(directory_string)
                except ValueError:
                    pass

    # Try the regular installation next, preserving its error in the message.
    try:
        return _load_refer_class()
    except ImportError as error:
        errors.append(error)

    # The official repository release is Python 2 source.  Keep it untouched
    # for provenance and use the project's Python 3 API-compatible reader for
    # the unchanged official refs(umd).p and instances.json files.
    try:
        try:
            from scripts.routed_grounding.refer_umd_api import REFER as Python3Refer
        except ImportError:
            from refer_umd_api import REFER as Python3Refer

        return Python3Refer
    except ImportError as error:
        errors.append(error)
    raise ImportError(
        f"Could not import REFER from {refer_root}. Expected a compatible 'refer' module or refer_api/refer.py."
    ) from (errors[-1] if errors else None)


def _load_coco_class() -> type:
    """Import the pycocotools COCO API lazily."""

    if COCO is not None:
        return COCO
    errors: list[BaseException] = []
    for module_name in ("pycocotools.coco", "coco"):
        try:
            module = importlib.import_module(module_name)
            loaded = getattr(module, "COCO", None)
            if loaded is not None:
                return loaded
        except (ImportError, ModuleNotFoundError, OSError) as error:
            errors.append(error)
    raise ImportError(
        "Could not import the COCO API. Install pycocotools or pass a "
        "compatible class through build_manifest(..., coco_cls=...)."
    ) from (errors[-1] if errors else None)


def _find_coco_annotation_file(refer_root: Path, coco_images: Path) -> Path:
    """Find the COCO-format annotation file implied by the two CLI paths."""

    candidates: list[Path] = []

    def add(path: Path) -> None:
        if path not in candidates:
            candidates.append(path)

    # RefCOCO's copy is a COCO-format instances file containing exactly the
    # image/annotation universe used by REFER and is the most portable choice.
    add(refer_root / DATASET_NAME / "instances.json")
    add(refer_root / "instances.json")

    # Standard MSCOCO layouts put annotations next to the image tree or at a
    # common ancestor of it.  Include both train and val names, while still
    # preferring the RefCOCO copy above when present.
    roots = [refer_root, coco_images, coco_images.parent]
    roots.extend(coco_images.parents)
    for root in roots:
        add(root / "annotations" / "instances_train2014.json")
        add(root / "annotations" / "instances_val2014.json")
        add(root / "instances_train2014.json")
        add(root / "instances_val2014.json")

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    # A few prepared datasets use ``coco_annotations.json``.  Restrict the
    # fallback search to likely names so a REFER pickle/json is not selected
    # accidentally as a COCO file.
    likely: list[Path] = []
    for root in (refer_root, coco_images, *coco_images.parents):
        if not root.is_dir():
            continue
        for pattern in ("*coco*.json", "*instances*.json"):
            likely.extend(root.glob(pattern))
            if root != refer_root:
                likely.extend(root.glob(f"annotations/{pattern}"))
    for candidate in sorted(set(likely), key=lambda path: str(path)):
        if candidate.is_file():
            return candidate

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError("Could not find a COCO instances annotation file. Searched:\n" + searched)


def _instantiate_refer(refer_cls: type, refer_root: Path) -> Any:
    try:
        return refer_cls(str(refer_root), dataset=DATASET_NAME, splitBy=SPLIT_BY)
    except TypeError as first_error:
        # Some lightweight test doubles use positional-only arguments.
        try:
            return refer_cls(str(refer_root), DATASET_NAME, SPLIT_BY)
        except TypeError:
            raise first_error from None


def _instantiate_coco(coco_cls: type, annotation_file: Path) -> Any:
    try:
        return coco_cls(str(annotation_file))
    except TypeError as first_error:
        try:
            return coco_cls(annotation_file=str(annotation_file))
        except TypeError:
            raise first_error from None


def _call_load_many(api: Any, method_name: str, ids: Sequence[Any]) -> list[Any]:
    """Call a COCO/REFER bulk loader across common API variants."""

    method = getattr(api, method_name)
    id_list = list(ids)
    try:
        result = method(id_list)
    except TypeError:
        if len(id_list) != 1:
            raise
        result = method(id_list[0])
    return _as_list(result)


def _get_ref_ids(refer: Any, split: str) -> list[Any]:
    method = refer.getRefIds
    try:
        result = method(split=split)
    except TypeError:
        # This mirrors the positional signature in older REFER ports.
        result = method([], [], [], split)
    return _as_list(result)


def _load_refs(refer: Any, ref_ids: Sequence[Any]) -> list[dict[str, Any]]:
    if not ref_ids:
        return []
    refs = _call_load_many(refer, "loadRefs", ref_ids)
    normalised: list[dict[str, Any]] = []
    for requested_id, ref in zip(ref_ids, refs, strict=False):
        if isinstance(ref, Mapping):
            item = dict(ref)
            item.setdefault("ref_id", requested_id)
            normalised.append(item)
    # A non-conforming stub may return one ref at a time even for a list.
    if len(normalised) != len(ref_ids):
        normalised = []
        for ref_id in ref_ids:
            loaded = _call_load_many(refer, "loadRefs", [ref_id])
            if loaded and isinstance(loaded[0], Mapping):
                item = dict(loaded[0])
                item.setdefault("ref_id", ref_id)
                normalised.append(item)
    return normalised


def _get_ann_ids(coco: Any, image_id: Any) -> list[Any]:
    method = coco.getAnnIds
    try:
        result = method(imgIds=[image_id])
    except TypeError:
        try:
            result = method([image_id])
        except TypeError:
            result = method(image_id)
    values = _as_list(result)
    # A permissive test double may return annotation dictionaries directly.
    return [value.get("id") if isinstance(value, Mapping) else value for value in values]


def _load_image(coco: Any, image_id: Any) -> Optional[dict[str, Any]]:
    images = _call_load_many(coco, "loadImgs", [image_id])
    for image in images:
        if isinstance(image, Mapping):
            return dict(image)
    return None


def _load_annotations(coco: Any, ann_ids: Sequence[Any]) -> list[dict[str, Any]]:
    if not ann_ids:
        return []
    anns = _call_load_many(coco, "loadAnns", ann_ids)
    return [dict(ann) for ann in anns if isinstance(ann, Mapping)]


def _has_valid_bbox(value: Any) -> bool:
    if not isinstance(value, list | tuple) or len(value) != 4:
        return False
    try:
        numbers = [float(item) for item in value]
    except (TypeError, ValueError):
        return False
    return (
        all(math.isfinite(item) for item in numbers)
        and numbers[0] >= 0
        and numbers[1] >= 0
        and numbers[2] > 0
        and numbers[3] > 0
    )


def _has_valid_segmentation(value: Any) -> bool:
    if isinstance(value, Mapping):
        # COCO RLEs have counts and size; requiring both prevents an arbitrary
        # empty dict from being treated as a valid mask.
        return bool(value.get("counts")) and bool(value.get("size"))
    if isinstance(value, list | tuple):
        if not value:
            return False
        # Both a flat polygon and a list of polygons are accepted.  Empty
        # nested polygons are rejected.
        return any(item is not None and (not isinstance(item, list | tuple) or len(item) > 0) for item in value)
    return False


def _category_name(coco: Any, ann: Mapping[str, Any], cache: dict[str, Optional[str]]) -> Optional[str]:
    category_id = ann.get("category_id")
    if category_id is None:
        return None
    key = _id_key(category_id)
    if key in cache:
        return cache[key]

    name: Optional[str] = None
    # Always prefer the COCO API's category table.  The annotation-level
    # values are a compatibility fallback for minimal API doubles.
    try:
        cats = _call_load_many(coco, "loadCats", [category_id])
    except (AttributeError, KeyError, TypeError):
        cats = []
    for category in cats:
        if isinstance(category, Mapping):
            candidate = category.get("name", category.get("category"))
            if candidate is not None:
                name = str(candidate)
                break
        elif category is not None:
            name = str(category)
            break
    if name is None:
        candidate = ann.get("category", ann.get("category_name"))
        if candidate is not None:
            name = str(candidate)
    cache[key] = name
    return name


def _valid_instance(
    coco: Any,
    ann: Mapping[str, Any],
    image_id: Any,
    image_width: int,
    image_height: int,
    category_cache: dict[str, Optional[str]],
) -> Optional[dict[str, Any]]:
    ann_id = ann.get("id", ann.get("ann_id"))
    if ann_id is None or not _same_id(ann.get("image_id"), image_id):
        return None
    if not _has_valid_bbox(ann.get("bbox")):
        return None
    segmentation = ann.get("segmentation")
    if not _has_valid_segmentation(segmentation):
        return None
    category_id = ann.get("category_id")
    category = _category_name(coco, ann, category_cache)
    if category_id is None or category is None:
        return None
    return {
        "ann_id": _jsonable(ann_id),
        "bbox": _jsonable(copy.deepcopy(ann["bbox"])),
        "segmentation": _jsonable(copy.deepcopy(segmentation)),
        "category": category,
        "category_id": _jsonable(category_id),
        "image_width": image_width,
        "image_height": image_height,
    }


def _image_path(
    coco_images: Path,
    image: Mapping[str, Any],
    *,
    require_existing: bool = True,
) -> Optional[Path]:
    file_name = image.get("file_name")
    if not isinstance(file_name, str | Path) or not str(file_name):
        return None
    path = Path(file_name)
    if not path.is_absolute():
        path = coco_images / path
    path = path.resolve()
    return path if path.is_file() or not require_existing else None


def _sentence_items(ref: Mapping[str, Any]) -> list[tuple[Any, str]]:
    sentences = ref.get("sentences")
    if sentences is None:
        sentences = ref.get("sentence")
    if sentences is None and ref.get("expression") is not None:
        sentences = [ref.get("expression")]
    if isinstance(sentences, str | bytes | Mapping):
        sentences = [sentences]

    result: list[tuple[Any, str]] = []
    for index, sentence in enumerate(_as_list(sentences)):
        if isinstance(sentence, Mapping):
            expression = sentence.get("sent", sentence.get("expression", sentence.get("text")))
            sentence_id = sentence.get("sent_id", sentence.get("id", index))
        else:
            expression = sentence
            sentence_id = index
        if isinstance(expression, bytes):
            expression = expression.decode("utf-8")
        if expression is None:
            continue
        expression_text = str(expression).strip()
        if expression_text:
            result.append((sentence_id, expression_text))
    return result


def _candidate_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        _sort_atom(candidate["image_id"]),
        _sort_atom(candidate["ref_id"]),
        _sort_atom(candidate["sent_id"]),
        str(candidate["expression"]),
    )


def _collect_candidates(
    refer: Any,
    coco: Any,
    coco_images: Path,
    split: str,
    *,
    require_images: bool = True,
) -> list[dict[str, Any]]:
    """Collect validated expression candidates for one non-test split."""

    ref_ids = _get_ref_ids(refer, split)
    refs = _load_refs(refer, ref_ids)
    category_cache: dict[str, Optional[str]] = {}
    candidates: list[dict[str, Any]] = []

    for ref in refs:
        image_id = ref.get("image_id")
        target_ann_id = ref.get("ann_id", ref.get("target_ann_id"))
        ref_id = ref.get("ref_id")
        if image_id is None or target_ann_id is None:
            continue

        image = _load_image(coco, image_id)
        if image is None:
            continue
        image_file = _image_path(coco_images, image, require_existing=require_images)
        if image_file is None:
            continue
        try:
            image_width = int(image["width"])
            image_height = int(image["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if image_width <= 0 or image_height <= 0:
            continue

        # Loading the target separately ensures that a target is accepted only
        # when it is present in COCO, even if a non-standard getAnnIds filter
        # omits it.
        target_annotations = _load_annotations(coco, [target_ann_id])
        target_annotation = next(
            (ann for ann in target_annotations if _same_id(ann.get("id", ann.get("ann_id")), target_ann_id)),
            None,
        )
        if target_annotation is None:
            continue

        ann_ids = _get_ann_ids(coco, image_id)
        all_annotations = _load_annotations(coco, ann_ids)
        if not any(_same_id(ann.get("id", ann.get("ann_id")), target_ann_id) for ann in all_annotations):
            all_annotations.append(target_annotation)

        valid_instances: list[dict[str, Any]] = []
        for annotation in all_annotations:
            instance = _valid_instance(
                coco,
                annotation,
                image_id,
                image_width,
                image_height,
                category_cache,
            )
            if instance is not None:
                valid_instances.append(instance)
        valid_instances.sort(key=lambda item: _sort_atom(item["ann_id"]))

        target_instance = next(
            (instance for instance in valid_instances if _same_id(instance["ann_id"], target_ann_id)),
            None,
        )
        # The pilot specifically studies relational grounding, so require at
        # least two valid instances in the image and a valid target instance.
        if target_instance is None or len(valid_instances) < 2:
            continue

        for sent_id, expression in _sentence_items(ref):
            record = {
                "split": split,
                "image_id": _jsonable(image_id),
                "image_path": str(image_file),
                "file_name": str(image.get("file_name")),
                "image_width": image_width,
                "image_height": image_height,
                "ref_id": _jsonable(ref_id),
                "sent_id": _jsonable(sent_id),
                "expression": expression,
                "target_ann_id": _jsonable(target_instance["ann_id"]),
                "target_bbox": copy.deepcopy(target_instance["bbox"]),
                "target_mask": copy.deepcopy(target_instance["segmentation"]),
                "instances": copy.deepcopy(valid_instances),
            }
            candidates.append(
                {
                    "image_id": image_id,
                    "ref_id": ref_id,
                    "sent_id": sent_id,
                    "expression": expression,
                    "record": record,
                }
            )

    candidates.sort(key=_candidate_key)
    return candidates


def _select_candidates(
    candidates: Sequence[dict[str, Any]],
    limit: int,
    seed: int,
    excluded_image_ids: Optional[set[str]] = None,
) -> list[dict[str, Any]]:
    """Select one sentence per image first, then deterministically fill gaps."""

    if limit <= 0:
        return []
    excluded = excluded_image_ids or set()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in candidates:
        image_key = _id_key(candidate["image_id"])
        if image_key in excluded:
            continue
        grouped.setdefault(image_key, []).append(candidate)
    for values in grouped.values():
        values.sort(key=_candidate_key)

    # Group order is the only randomized choice.  Candidate order inside an
    # image remains stable, making the one-expression-per-image rule explicit.
    groups = sorted(grouped.items(), key=lambda item: _sort_atom(item[1][0]["image_id"]))
    random.Random(seed).shuffle(groups)

    selected: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for _, values in groups:
        if len(selected) < limit:
            selected.append(values[0])
            remaining.extend(values[1:])
        else:
            remaining.extend(values)

    # If the number of images is smaller than the requested capacity, fill it
    # with additional expressions from those images in stable order.
    remaining.sort(key=_candidate_key)
    if len(selected) < limit:
        selected.extend(remaining[: limit - len(selected)])
    selected.sort(key=_candidate_key)
    return selected


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_refer_source_files(refer_root: Path) -> list[Path]:
    candidates: list[Path] = []
    if not refer_root.is_dir():
        return candidates
    for pattern in ("refs(umd).p", "refs(umd).json", "refs(umd).pickle"):
        candidates.extend(refer_root.rglob(pattern))
    return sorted({path.resolve() for path in candidates if path.is_file()}, key=str)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(_jsonable(record), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(_jsonable(value), handle, ensure_ascii=False, sort_keys=True, indent=2)
        handle.write("\n")


def build_manifest(
    refer_root: PathLike,
    coco_images: PathLike,
    output_dir: PathLike,
    seed: int = DEFAULT_SEED,
    train_size: int = 1000,
    val_size: int = 200,
    *,
    refer_cls: Optional[type] = None,
    coco_cls: Optional[type] = None,
    require_images: bool = True,
) -> dict[str, Any]:
    """Build and write the train/val RefCOCOg UMD pilot manifest.

    ``refer_cls`` and ``coco_cls`` are optional dependency-injection hooks for
    tests.  Production callers should leave them unset so the REFER and COCO
    APIs are imported from the supplied dataset environment.
    """

    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(train_size, int) or train_size < 0:
        raise ValueError("train_size must be a non-negative integer")
    if not isinstance(val_size, int) or val_size < 0:
        raise ValueError("val_size must be a non-negative integer")

    refer_root_path = Path(refer_root).expanduser().resolve()
    coco_images_path = Path(coco_images).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    if not refer_root_path.is_dir():
        raise NotADirectoryError(f"refer-root is not a directory: {refer_root_path}")
    if not coco_images_path.is_dir():
        raise NotADirectoryError(f"coco-images is not a directory: {coco_images_path}")

    annotation_file = _find_coco_annotation_file(refer_root_path, coco_images_path)
    refer_factory = refer_cls or _import_refer_class(refer_root_path)
    coco_factory = coco_cls or _load_coco_class()
    refer = _instantiate_refer(refer_factory, refer_root_path)
    coco = _instantiate_coco(coco_factory, annotation_file)

    # Deliberately request only the two pilot splits.  In particular, no
    # getRefIds(split="test") or test ref loading occurs in this process.
    train_candidates = _collect_candidates(refer, coco, coco_images_path, "train", require_images=require_images)
    val_candidates = _collect_candidates(refer, coco, coco_images_path, "val", require_images=require_images)
    train_selected = _select_candidates(train_candidates, train_size, seed)
    train_image_keys = {_id_key(candidate["image_id"]) for candidate in train_selected}
    val_selected = _select_candidates(val_candidates, val_size, seed + 1, train_image_keys)

    train_records = [candidate["record"] for candidate in train_selected]
    val_records = [candidate["record"] for candidate in val_selected]
    all_records = train_records + val_records

    output_path.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path / "train.jsonl", train_records)
    _write_jsonl(output_path / "val.jsonl", val_records)
    _write_jsonl(output_path / "pilot_manifest.jsonl", all_records)

    source_files = _find_refer_source_files(refer_root_path)
    if annotation_file.resolve() not in source_files:
        source_files.append(annotation_file.resolve())
    source_files = sorted(set(source_files), key=str)
    source_entries = [{"path": str(path), "sha256": _sha256(path)} for path in source_files]
    source_hashes = {entry["path"]: entry["sha256"] for entry in source_entries}
    train_images = {_id_key(record["image_id"]) for record in train_records}
    val_images = {_id_key(record["image_id"]) for record in val_records}
    metadata: dict[str, Any] = {
        "dataset": DATASET_NAME,
        "split_by": SPLIT_BY,
        "seed": seed,
        "requested_sizes": {"train": train_size, "val": val_size},
        "actual_sizes": {"train": len(train_records), "val": len(val_records), "total": len(all_records)},
        "image_counts": {"train": len(train_images), "val": len(val_images), "total": len(train_images | val_images)},
        "split_image_overlap": sorted(train_images & val_images),
        "refer_root": str(refer_root_path),
        "coco_images": str(coco_images_path),
        "coco_annotation_file": str(annotation_file.resolve()),
        "images_required_at_build_time": require_images,
        "source_files": source_entries,
        "source_file_hashes": source_hashes,
    }
    metadata_path = output_path / "metadata"
    metadata_path.mkdir(parents=True, exist_ok=True)
    _write_json(metadata_path / "manifest_metadata.json", metadata)
    _write_json(metadata_path / "source_hashes.json", {"algorithm": "sha256", "files": source_entries})
    return metadata


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refer-root", required=True, type=Path, help="RefCOCO/REFER data root")
    parser.add_argument("--coco-images", required=True, type=Path, help="Directory containing COCO image files")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for JSONL and metadata outputs")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"deterministic selection seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument("--train-size", type=_nonnegative_int, default=1000, help="maximum number of train expressions")
    parser.add_argument("--val-size", type=_nonnegative_int, default=200, help="maximum number of val expressions")
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Bootstrap only: freeze records before downloading their exact COCO images.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    metadata = build_manifest(
        args.refer_root,
        args.coco_images,
        args.output_dir,
        seed=args.seed,
        train_size=args.train_size,
        val_size=args.val_size,
        require_images=not args.allow_missing_images,
    )
    print(
        f"Wrote RefCOCOg UMD pilot manifest: train={metadata['actual_sizes']['train']} "
        f"val={metadata['actual_sizes']['val']} output={args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
