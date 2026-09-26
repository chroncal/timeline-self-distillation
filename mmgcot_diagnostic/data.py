"""CPU-only MM-GCoT data preparation and inference-safe row adaptation.

The module deliberately keeps annotation fields out of :func:`model_input`.
It downloads the pinned public JSON files, selects a small deterministic
image-disjoint sample, validates image files and normalized boxes, and writes
JSONL artifacts with a provenance manifest.  It has no torch dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


SEED = 20260920
HF_DATASET = "AQUA6/MM-GCoT"
HF_REVISION = "b895c70224c7a1fbf65177908f8413eb7f2c617a"
HF_API_URL = f"https://huggingface.co/api/datasets/{HF_DATASET}"
HF_TREE_URL = f"https://huggingface.co/api/datasets/{HF_DATASET}/tree/{HF_REVISION}"
HF_RESOLVE_BASE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/{HF_REVISION}"
VG_BASE_URL = "https://cs.stanford.edu/people/rak248"

RAW_FILES: tuple[str, ...] = (
    "Train/train_dataset.json",
    "Test/CoP_dataset_attributes_test.json",
    "Test/CoP_dataset_judge_test.json",
    "Test/CoP_dataset_things_test.json",
)

TASK_TYPES: tuple[str, ...] = ("attribute", "object")
PILOT_TARGET = 30
FORMAL_TARGET = 200
PER_TASK_PILOT = 15
PER_TASK_FORMAL = 100
FORMAL_GEOMETRY_BLIND_TARGET = 50

DEFAULT_ROOT = Path("/mnt/sda/sujingyang/research/datasets/mmgcot_20260920")
DEFAULT_LOCAL_IMAGE_DIRS: tuple[Path, ...] = (
    Path("/mnt/sda/sujingyang/research/datasets/finecops_ref_20260909/images"),
    Path("/share2/sujingyang/datasets/gqa/error_span_selected_20260909"),
)

_BOX_RE = re.compile(
    r"\[\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*,\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))\s*\]"
)
_FINAL_ANSWER_RE = re.compile(r"\bFinal\s+Answer\s*:\s*(.*)\s*$", re.IGNORECASE | re.DOTALL)
_TRAIN_SUFFIX_RE = re.compile(
    r"\s*Answer the question (?:using a single word or phrase|step by step, ultimately using a single word or phrase as the answer)\.?\s*$",
    re.IGNORECASE,
)
_ATTRIBUTE_RE = re.compile(
    r"\bwhat\s+(?:is|are)\s+(?:the\s+)?"
    r"(?:color|colour|material|shape|size|condition|state|pattern|activity|texture|"
    r"appearance|style)\b",
    re.IGNORECASE,
)
_ATTRIBUTE_SHORT_RE = re.compile(
    r"\bwhat\s+(?:color|colour|material|shape|size|condition|state|pattern|activity|texture|appearance|style)\b",
    re.IGNORECASE,
)
_OBJECT_QUESTION_RE = re.compile(
    r"^\s*(?:what\s+(?:is|are)|which\s+(?:object|thing|item|animal|person|people)|"
    r"what\s+(?:object|thing|item|animal|person|people))\b",
    re.IGNORECASE,
)
_NON_OBJECT_QUESTION_RE = re.compile(
    r"\b(?:position|direction|relationship|relation|text|word|letter|number|count|"
    r"amount|many|much|age|old|time|date|role|activity|condition|state|color|"
    r"colour|material|shape|size|pattern|texture|appearance|style)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ImageRecord:
    """One materialized image and its provenance."""

    image_id: str
    image_path: str
    source_kind: str
    source_path: str | None
    source_url: str | None
    source_image_ref: str
    sha256: str
    size_bytes: int
    width: int
    height: int


@dataclass(frozen=True)
class PreparationResult:
    """Paths and counts returned by :func:`prepare_dataset`."""

    root: Path
    manifest_path: Path
    pilot_path: Path
    formal_path: Path
    exclusions_path: Path
    contact_index_path: Path
    semantic_review_path: Path
    counts: dict[str, Any]
    blocked: bool


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 of a file, following symlinks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_json_bytes(value))


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            handle.write("\n")


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def parse_four_numbers(value: Any, *, label: str = "box") -> list[float]:
    """Parse a JSON/list/string four-vector without silently repairing it."""

    parsed = value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} is not a JSON four-vector") from exc
    if not isinstance(parsed, (list, tuple)) or len(parsed) != 4:
        raise ValueError(f"{label} must contain exactly four numbers")
    return [_finite_number(item, label=f"{label}[{index}]") for index, item in enumerate(parsed)]


def xywh_to_xyxy(value: Any, *, tolerance: float = 1e-9) -> list[float]:
    """Validate normalized ``xywh`` and return normalized ``xyxy``.

    The source files use normalized ``[x, y, width, height]``.  A box that
    would leave the image is rejected rather than clipped, because clipping
    would change the frozen ground truth.
    """

    x, y, width, height = parse_four_numbers(value, label="xywh")
    if any(item < -tolerance or item > 1.0 + tolerance for item in (x, y, width, height)):
        raise ValueError("normalized xywh values must lie in [0, 1]")
    if width <= tolerance or height <= tolerance:
        raise ValueError("normalized xywh width and height must be positive")
    right = x + width
    bottom = y + height
    if x < -tolerance or y < -tolerance or right > 1.0 + tolerance or bottom > 1.0 + tolerance:
        raise ValueError("normalized xywh box leaves the image")
    result = [x, y, right, bottom]
    result = [0.0 if abs(item) <= tolerance else item for item in result]
    result = [1.0 if abs(item - 1.0) <= tolerance else item for item in result]
    return [round(item, 12) for item in result]


def is_valid_xyxy(value: Any, *, tolerance: float = 1e-9) -> bool:
    """Return whether a normalized ``xyxy`` box is finite and non-empty."""

    try:
        x1, y1, x2, y2 = parse_four_numbers(value, label="xyxy")
    except ValueError:
        return False
    return (
        -tolerance <= x1 <= 1.0 + tolerance
        and -tolerance <= y1 <= 1.0 + tolerance
        and -tolerance <= x2 <= 1.0 + tolerance
        and -tolerance <= y2 <= 1.0 + tolerance
        and x1 + tolerance < x2
        and y1 + tolerance < y2
    )


def extract_cot_boxes(cot: str) -> list[list[float]]:
    """Extract source normalized ``xywh`` boxes in their textual order."""

    if not isinstance(cot, str):
        raise ValueError("cot must be a string")
    return [[float(value) for value in match.groups()] for match in _BOX_RE.finditer(cot)]


def last_cot_bbox(cot: str) -> list[float]:
    """Return the last CoT box after converting it from normalized xywh."""

    boxes = extract_cot_boxes(cot)
    if not boxes:
        raise ValueError("cot contains no coordinate box")
    return xywh_to_xyxy(boxes[-1])


def coordinate_bbox(coordinate: Any) -> list[float]:
    """Convert the dataset's normalized coordinate string from xywh to xyxy."""

    return xywh_to_xyxy(coordinate)


def cot_coordinate_consistent(cot: str, coordinate: Any, *, tolerance: float = 1e-6) -> bool:
    """Check that the final CoT xywh and source coordinate describe one box."""

    boxes = extract_cot_boxes(cot)
    source = parse_four_numbers(coordinate, label="coordinate")
    if not boxes:
        return False
    return all(abs(left - right) <= tolerance for left, right in zip(boxes[-1], source, strict=True))


def extract_final_answer(text: str) -> str:
    """Extract the train CoT final answer, preserving source wording."""

    match = _FINAL_ANSWER_RE.search(text.strip())
    if not match or not match.group(1).strip():
        raise ValueError("CoT has no non-empty Final Answer")
    return match.group(1).strip()


def reference_cot_without_answer(text: str) -> str:
    """Return the reasoning part while keeping all source coordinate steps."""

    match = _FINAL_ANSWER_RE.search(text.strip())
    if match:
        value = text[: match.start()].rstrip()
    else:
        value = text.strip()
    if not value:
        raise ValueError("reference CoT is empty")
    return value


def extract_final_entity(cot: str) -> str:
    """Extract the final textual target before the last ``is at [box]``."""

    matches = list(_BOX_RE.finditer(cot))
    if not matches:
        raise ValueError("cot contains no target box")
    prefix = cot[: matches[-1].start()]
    line = prefix.splitlines()[-1].strip()
    line = re.sub(r"^Step\s+\d+\s*:\s*", "", line, flags=re.IGNORECASE).strip()
    line = re.sub(r"\s+(?:is\s+)?at\s*$", "", line, flags=re.IGNORECASE).strip()
    return line or "UNRESOLVED"


def clean_train_question(value: str) -> str:
    """Remove the image token and the fixed training instruction suffix."""

    question = value.replace("<image>", "").strip()
    question = _TRAIN_SUFFIX_RE.sub("", question).strip()
    return question


def infer_train_task_type(question: str) -> str | None:
    """Assign only high-confidence attribute/object labels to train rows.

    The official train JSON does not expose the test file's task label.  This
    conservative lexical partition is used solely to obtain the frozen 15/15
    pilot allocation; ambiguous rows are excluded and logged for review.
    """

    normalized = question.strip()
    if not normalized or re.match(r"^(?:is|are|does|do)\b", normalized, re.IGNORECASE):
        return None
    if _ATTRIBUTE_RE.search(normalized) or _ATTRIBUTE_SHORT_RE.search(normalized):
        return "attribute"
    if _OBJECT_QUESTION_RE.search(normalized) and not _NON_OBJECT_QUESTION_RE.search(normalized):
        return "object"
    return None


def model_input(row: Mapping[str, Any]) -> dict[str, str]:
    """Return the only two fields the inference runner may consume."""

    return {"image_path": str(row["image_path"]), "question": str(row["question"])}


def _image_id(image_ref: str) -> str:
    path = Path(image_ref)
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError(f"unsupported image reference: {image_ref}")
    return path.stem


def _stable_rank(row: Mapping[str, Any], *, seed: int, namespace: str) -> str:
    payload = json.dumps(
        [
            seed,
            namespace,
            str(row.get("image_id", "")),
            str(row.get("source_id", "")),
            int(row.get("source_row_index", -1)),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_order(rows: Iterable[Mapping[str, Any]], *, seed: int = SEED, namespace: str = "") -> list[dict[str, Any]]:
    """Return a stable seeded order independent of Python hash randomization."""

    return sorted(
        (dict(row) for row in rows),
        key=lambda row: (
            _stable_rank(row, seed=seed, namespace=namespace),
            str(row.get("source_id", "")),
            int(row.get("source_row_index", -1)),
        ),
    )


def _source_relative(root: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _raw_row_list(value: Any, *, path: Path) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise ValueError(f"{path} must contain a JSON list")
    if not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path} contains a non-object row")
    return [dict(row) for row in value]


def _make_test_candidate(
    raw: Mapping[str, Any],
    *,
    task_type: str,
    source_path: Path,
    root: Path,
    source_row_index: int,
    source_occurrence: int,
) -> dict[str, Any]:
    source_id = str(raw["id"])
    image_ref = str(raw["img"])
    cot = str(raw["cot"])
    coordinate = raw["coordinate"]
    if not cot_coordinate_consistent(cot, coordinate):
        raise ValueError("final CoT box does not match coordinate")
    bbox = coordinate_bbox(coordinate)
    if not is_valid_xyxy(bbox):
        raise ValueError("converted coordinate is not a legal normalized xyxy")
    if not str(raw.get("question", "")).strip() or not str(raw.get("answer", "")).strip():
        raise ValueError("question or answer is empty")
    return {
        "split": "formal",
        "task_type": task_type,
        "source_id": source_id,
        "source_file": str(source_path.resolve()),
        "source_row_index": source_row_index,
        "source_occurrence": source_occurrence,
        "image_ref": image_ref,
        "image_id": _image_id(image_ref),
        "question": str(raw["question"]).strip(),
        "ground_truth_bbox": bbox,
        "reference_cot": cot.strip(),
        "reference_answer": str(raw["answer"]).strip(),
        "reference_entity": extract_final_entity(cot),
        "source_coordinate_xywh": parse_four_numbers(coordinate, label="coordinate"),
        "source_kind": "hf_test",
        "_raw_source": raw,
    }


def _make_train_candidate(
    raw: Mapping[str, Any], *, source_path: Path, source_row_index: int, source_occurrence: int
) -> dict[str, Any]:
    source_id = str(raw["id"])
    conversations = raw.get("conversations")
    if not isinstance(conversations, list) or len(conversations) < 2:
        raise ValueError("train row has no two-turn conversation")
    human = next((item for item in conversations if item.get("from") == "human"), None)
    assistant = next((item for item in conversations if item.get("from") == "gpt"), None)
    if not isinstance(human, dict) or not isinstance(assistant, dict):
        raise ValueError("train row has no human/gpt turns")
    question = clean_train_question(str(human.get("value", "")))
    cot = str(assistant.get("value", "")).strip()
    if not source_id.endswith("_cot"):
        raise ValueError("train row is not the CoT variant")
    answer = extract_final_answer(cot)
    reference_cot = reference_cot_without_answer(cot)
    bbox = last_cot_bbox(reference_cot)
    if not is_valid_xyxy(bbox):
        raise ValueError("converted final CoT box is not a legal normalized xyxy")
    task_type = infer_train_task_type(question)
    if task_type is None:
        raise ValueError("train question task type is ambiguous")
    image_ref = str(raw["image"])
    return {
        "split": "pilot",
        "task_type": task_type,
        "source_id": source_id,
        "source_file": str(source_path.resolve()),
        "source_row_index": source_row_index,
        "source_occurrence": source_occurrence,
        "image_ref": image_ref,
        "image_id": _image_id(image_ref),
        "question": question,
        "ground_truth_bbox": bbox,
        "reference_cot": reference_cot,
        "reference_answer": answer,
        "reference_entity": extract_final_entity(reference_cot),
        "source_coordinate_xywh": extract_cot_boxes(reference_cot)[-1],
        "source_kind": "hf_train_cot",
        "_raw_source": raw,
    }


def _exclusion(
    candidate: Mapping[str, Any] | None,
    *,
    reason: str,
    detail: str = "",
    split: str | None = None,
    task_type: str | None = None,
    source_id: str | None = None,
    source_file: str | None = None,
    image_id: str | None = None,
) -> dict[str, Any]:
    value = candidate or {}
    return {
        "split": split if split is not None else value.get("split"),
        "task_type": task_type if task_type is not None else value.get("task_type"),
        "source_id": source_id if source_id is not None else value.get("source_id"),
        "source_file": source_file if source_file is not None else value.get("source_file"),
        "image_id": image_id if image_id is not None else value.get("image_id"),
        "reason": reason,
        "detail": detail,
    }


def _default_image_dirs(extra: Sequence[Path] = ()) -> list[Path]:
    result: list[Path] = []
    for path in [*extra, *DEFAULT_LOCAL_IMAGE_DIRS]:
        path = Path(path)
        if path.is_dir() and path not in result:
            result.append(path)
    return result


def discover_local_images(image_dirs: Sequence[Path]) -> dict[str, Path]:
    """Index numeric image files from explicitly scoped local VG-like dirs."""

    index: dict[str, Path] = {}
    for directory in image_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir(), key=lambda item: item.name):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            if not path.is_file():
                continue
            image_id = path.stem
            if image_id.isdigit() and image_id not in index:
                index[image_id] = path
    return index


def image_dimensions(path: str | Path) -> tuple[int, int]:
    """Open an image with Pillow and return width/height; no torch import."""

    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - environment issue
        raise RuntimeError("Pillow is required to validate VG images") from exc
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            if width <= 0 or height <= 0:
                raise ValueError("image has non-positive dimensions")
            return int(width), int(height)
    except Exception as exc:  # Pillow raises several format-specific errors.
        raise ValueError(f"unreadable image: {path}: {exc}") from exc


def _vg_url(image_ref: str, *, folder_override: str | None = None) -> str:
    path = Path(image_ref)
    folder = folder_override or (path.parent.name if path.parent.name in {"VG_100K", "VG_100K_2"} else "VG_100K")
    if folder not in {"VG_100K", "VG_100K_2"}:
        raise ValueError(f"unsupported VG folder in image ref: {image_ref}")
    return f"{VG_BASE_URL}/{folder}/{path.name}"


def _download_binary(path: Path, url: str) -> None:
    """Download once to a new file; never overwrite a destination."""

    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite existing image: {path}")
    partial = path.with_name(f".{path.name}.part")
    if partial.exists():
        raise FileExistsError(f"refusing to reuse stale partial download: {partial}")
    request = Request(url, headers={"User-Agent": "MM-GCoT-data-preparer/20260920"})
    try:
        with urlopen(request, timeout=90) as response, partial.open("xb") as handle:
            while chunk := response.read(1024 * 1024):
                handle.write(chunk)
    except (HTTPError, URLError, TimeoutError, OSError):
        if partial.exists():
            partial.unlink()
        raise
    partial.replace(path)


def materialize_image(
    *,
    image_id: str,
    image_ref: str,
    image_dir: Path,
    local_index: Mapping[str, Path],
) -> ImageRecord:
    """Use a local exact-ID image or download one official VG original."""

    image_dir.mkdir(parents=True, exist_ok=True)
    destination = image_dir / f"{image_id}.jpg"
    local_path = local_index.get(image_id)
    source_kind = "local_existing"
    source_path: str | None = None
    source_url: str | None = None

    if not destination.exists() and not destination.is_symlink():
        if local_path is not None:
            try:
                width, height = image_dimensions(local_path)
            except ValueError:
                local_path = None
            else:
                destination.symlink_to(local_path.absolute())
                source_path = str(local_path.absolute())
        if local_path is None:
            attempted: list[str] = []
            url_candidates = [_vg_url(image_ref)]
            alternate = "VG_100K_2" if "VG_100K_2" not in url_candidates[0] else "VG_100K"
            url_candidates.append(_vg_url(image_ref, folder_override=alternate))
            for url in url_candidates:
                attempted.append(url)
                try:
                    _download_binary(destination, url)
                except (HTTPError, URLError, TimeoutError, OSError):
                    if destination.exists() or destination.is_symlink():
                        destination.unlink()
                    continue
                source_kind = "official_vg_download"
                source_url = url
                break
            else:
                raise RuntimeError(f"unable to download {image_id}; attempted={attempted}")
    elif destination.is_symlink() or destination.is_file():
        source_kind = "existing_materialized"

    try:
        width, height = image_dimensions(destination)
    except ValueError:
        raise
    if source_path is None and local_path is not None and destination.is_symlink():
        source_path = str(local_path.absolute())
    if source_kind == "existing_materialized" and source_path is None:
        source_path = str(destination.resolve())
    return ImageRecord(
        image_id=image_id,
        image_path=str(destination.absolute()),
        source_kind=source_kind,
        source_path=source_path,
        source_url=source_url or _vg_url(image_ref),
        source_image_ref=image_ref,
        sha256=file_sha256(destination),
        size_bytes=destination.stat().st_size,
        width=width,
        height=height,
    )


def _http_json(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "MM-GCoT-data-preparer/20260920"})
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def _hf_file_metadata() -> dict[str, dict[str, Any]]:
    api = _http_json(HF_API_URL + f"?revision={HF_REVISION}")
    if str(api.get("sha")) != HF_REVISION:
        raise RuntimeError(f"HF API revision mismatch: expected {HF_REVISION}, got {api.get('sha')}")
    tree = _http_json(HF_TREE_URL + "?recursive=true")
    entries = {str(item.get("path")): dict(item) for item in tree if item.get("type") == "file"}
    missing = [path for path in RAW_FILES if path not in entries]
    if missing:
        raise RuntimeError(f"HF pinned revision is missing expected files: {missing}")
    for path in RAW_FILES:
        entry = entries[path]
        if not entry.get("oid") or not entry.get("size"):
            raise RuntimeError(f"HF API did not provide oid/size for {path}")
    return {path: entries[path] for path in RAW_FILES}


def _download_raw_file(path: Path, *, url: str, expected_size: int) -> None:
    if path.exists():
        if path.stat().st_size != expected_size:
            raise RuntimeError(
                f"existing raw file has wrong size and will not be overwritten: {path} "
                f"{path.stat().st_size}!={expected_size}"
            )
        _read_json(path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.part")
    if partial.exists():
        raise FileExistsError(f"stale raw partial exists; remove it explicitly: {partial}")
    _download_binary(path, url)
    if path.stat().st_size != expected_size:
        raise RuntimeError(f"downloaded raw file has wrong size: {path}")
    _read_json(path)


def ensure_raw_jsons(root: Path, *, raw_source_root: Path | None = None) -> list[dict[str, Any]]:
    """Ensure all official raw JSONs exist at the pinned HF revision."""

    entries = _hf_file_metadata()
    records: list[dict[str, Any]] = []
    for relative in RAW_FILES:
        entry = entries[relative]
        path = root / "raw" / relative
        url = f"{HF_RESOLVE_BASE}/{relative}"
        _download_raw_file(path, url=url, expected_size=int(entry["size"]))
        record = {
                "path": str(path.absolute()),
                "relative_path": _source_relative(root, path),
                "source_url": url,
                "hf_dataset": HF_DATASET,
                "hf_revision": HF_REVISION,
                "hf_dataset_api_sha": HF_REVISION,
                "hf_file_oid": str(entry["oid"]),
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        if raw_source_root is not None:
            source_path = raw_source_root / "raw" / relative
            if not source_path.is_file():
                raise FileNotFoundError(f"raw source file is missing: {source_path}")
            source_sha = file_sha256(source_path)
            if source_sha != record["sha256"]:
                raise RuntimeError(f"raw source hash mismatch for {relative}: {source_sha} != {record['sha256']}")
            record["copied_from"] = str(source_path.absolute())
            record["copied_from_sha256"] = source_sha
        records.append(record)
    return records


def _pair_train_variants(
    rows: Sequence[Mapping[str, Any]], *, source_path: Path, exclusions: list[dict[str, Any]]
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], int, int]], dict[str, Any]]:
    """Pair adjacent answer/grounded-CoT variants before any task filtering."""

    normal_rows = sum(not str(row.get("id", "")).endswith("_cot") for row in rows)
    cot_rows = sum(str(row.get("id", "")).endswith("_cot") for row in rows)
    stats: dict[str, Any] = {
        "total_rows": len(rows),
        "answer_only_rows": normal_rows,
        "grounded_cot_rows": cot_rows,
        "adjacent_pairs_checked": 0,
        "valid_variant_pairs": 0,
        "variant_mismatches": 0,
        "selected_variant": "_cot",
        "pairing_rule": "adjacent normal id followed by id+'_cot'; image/question/answer must match",
    }
    pairs: list[tuple[dict[str, Any], dict[str, Any], int, int]] = []
    if len(rows) % 2:
        exclusions.append(
            _exclusion(
                None,
                split="pilot",
                source_file=str(source_path.absolute()),
                reason="unpaired_train_variant",
                detail="train JSON row count is odd",
            )
        )
    for index in range(0, len(rows) - 1, 2):
        normal = dict(rows[index])
        cot = dict(rows[index + 1])
        stats["adjacent_pairs_checked"] += 1
        normal_id = str(normal.get("id", ""))
        cot_id = str(cot.get("id", ""))
        normal_question = clean_train_question(str(normal.get("conversations", [{}])[0].get("value", "")))
        cot_question = clean_train_question(str(cot.get("conversations", [{}])[0].get("value", "")))
        normal_answer = str(normal.get("conversations", [{}, {}])[-1].get("value", "")).strip()
        cot_answer_text = str(cot.get("conversations", [{}, {}])[-1].get("value", ""))
        try:
            cot_answer = extract_final_answer(cot_answer_text)
        except ValueError:
            cot_answer = ""
        valid = (
            bool(normal_id)
            and not normal_id.endswith("_cot")
            and cot_id == normal_id + "_cot"
            and str(normal.get("image", "")) == str(cot.get("image", ""))
            and normal_question == cot_question
            and normal_answer == cot_answer
        )
        if not valid:
            stats["variant_mismatches"] += 1
            exclusions.append(
                _exclusion(
                    None,
                    split="pilot",
                    source_file=str(source_path.absolute()),
                    source_id=normal_id or cot_id,
                    image_id=Path(str(cot.get("image", normal.get("image", "")))).stem or None,
                    reason="train_variant_mismatch",
                    detail=(
                        f"normal_id={normal_id!r}; cot_id={cot_id!r}; "
                        f"same_image={normal.get('image') == cot.get('image')}; "
                        f"same_question={normal_question == cot_question}; "
                        f"same_answer={normal_answer == cot_answer}"
                    ),
                )
            )
            continue
        stats["valid_variant_pairs"] += 1
        pairs.append((normal, cot, index, index + 1))
    return pairs, stats


def _raw_candidates(
    root: Path,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], list[dict[str, Any]], dict[str, Any]]:
    """Parse structural candidates and retain every rejection reason."""

    exclusions: list[dict[str, Any]] = []
    by_split_task: dict[str, dict[str, list[dict[str, Any]]]] = {
        "pilot": {task: [] for task in TASK_TYPES},
        "formal": {task: [] for task in TASK_TYPES},
    }
    train_path = root / "raw" / "Train/train_dataset.json"
    train_rows = _raw_row_list(_read_json(train_path), path=train_path)
    train_pairs, train_variant_stats = _pair_train_variants(
        train_rows, source_path=train_path, exclusions=exclusions
    )
    for _, raw, _, cot_row_index in train_pairs:
        try:
            candidate = _make_train_candidate(
                raw,
                source_path=train_path,
                source_row_index=cot_row_index,
                source_occurrence=cot_row_index + 1,
            )
        except (KeyError, TypeError, ValueError) as exc:
            exclusions.append(
                _exclusion(
                    None,
                    split="pilot",
                    source_id=str(raw.get("id")),
                    source_file=str(train_path.absolute()),
                    image_id=Path(str(raw.get("image", ""))).stem or None,
                    reason="invalid_train_row",
                    detail=str(exc),
                )
            )
            continue
        by_split_task["pilot"][candidate["task_type"]].append(candidate)

    test_specs = (
        ("CoP_dataset_attributes_test.json", "attribute"),
        ("CoP_dataset_things_test.json", "object"),
        ("CoP_dataset_judge_test.json", None),
    )
    for name, task_type in test_specs:
        path = root / "raw" / "Test" / name
        for row_index, raw in enumerate(_raw_row_list(_read_json(path), path=path)):
            if task_type is None:
                exclusions.append(
                    _exclusion(
                        None,
                        split="formal",
                        source_id=str(raw.get("id")),
                        source_file=str(path.absolute()),
                        image_id=Path(str(raw.get("img", ""))).stem or None,
                        reason="unsupported_task_type",
                        detail="judge JSON is preserved as an official raw source but the frozen protocol has attribute/object tasks only",
                    )
                )
                continue
            try:
                candidate = _make_test_candidate(
                    raw,
                    task_type=task_type,
                    source_path=path,
                    root=root,
                    source_row_index=row_index,
                    source_occurrence=row_index + 1,
                )
            except (KeyError, TypeError, ValueError) as exc:
                exclusions.append(
                    _exclusion(
                        None,
                        split="formal",
                        task_type=task_type,
                        source_id=str(raw.get("id")),
                        source_file=str(path.absolute()),
                        image_id=Path(str(raw.get("img", ""))).stem or None,
                        reason="invalid_test_row",
                        detail=str(exc),
                    )
                )
                continue
            by_split_task["formal"][task_type].append(candidate)
    train_variant_stats["valid_grounded_cot_candidates"] = sum(
        len(rows) for rows in by_split_task["pilot"].values()
    )
    train_variant_stats["candidate_image_ids"] = len(
        {
            str(candidate["image_id"])
            for rows in by_split_task["pilot"].values()
            for candidate in rows
        }
    )
    train_variant_stats["candidate_rows_removed_by_image_dedup"] = (
        train_variant_stats["valid_grounded_cot_candidates"]
        - train_variant_stats["candidate_image_ids"]
    )
    return by_split_task, exclusions, train_variant_stats


def _public_row(candidate: Mapping[str, Any], image: ImageRecord, *, split: str) -> dict[str, Any]:
    task_type = str(candidate["task_type"])
    source_id = str(candidate["source_id"])
    source_occurrence = int(candidate["source_occurrence"])
    return {
        "sample_id": f"{split}:{task_type}:{source_id}:{source_occurrence}",
        "image_id": str(candidate["image_id"]),
        "task_type": task_type,
        "image_path": str(Path(image.image_path).absolute()),
        "question": str(candidate["question"]),
        "ground_truth_bbox": [float(item) for item in candidate["ground_truth_bbox"]],
        "reference_cot": str(candidate["reference_cot"]),
        "reference_answer": str(candidate["reference_answer"]),
        "source_id": source_id,
        "source_file": str(candidate["source_file"]),
        "source_row_index": int(candidate["source_row_index"]),
        "source_occurrence": source_occurrence,
        "image_sha256": image.sha256,
    }


def _select_materialized(
    *,
    candidates_by_task: Mapping[str, Sequence[Mapping[str, Any]]],
    task_targets: Mapping[str, int],
    split: str,
    seed: int,
    used_image_ids: set[str],
    image_dir: Path,
    local_index: Mapping[str, Path],
    exclusions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, ImageRecord]]:
    """Select unique images while retrying candidates after image failures."""

    selected: list[dict[str, Any]] = []
    selected_ids = set(used_image_ids)
    image_records: dict[str, ImageRecord] = {}
    ranked: dict[str, list[dict[str, Any]]] = {
        task: stable_order(rows, seed=seed, namespace=f"{split}:{task}")
        for task, rows in candidates_by_task.items()
    }
    # The first pass protects the per-type quotas.  Test is already processed
    # before pilot by the caller, so this also implements test priority.
    for task in TASK_TYPES:
        target = int(task_targets.get(task, 0))
        count = 0
        for candidate in ranked.get(task, []):
            if count >= target:
                break
            image_id = str(candidate["image_id"])
            if image_id in selected_ids:
                exclusions.append(_exclusion(candidate, reason="duplicate_image_across_selected_split"))
                continue
            try:
                image = image_records.get(image_id)
                if image is None:
                    image = materialize_image(
                        image_id=image_id,
                        image_ref=str(candidate["image_ref"]),
                        image_dir=image_dir,
                        local_index=local_index,
                    )
                    image_records[image_id] = image
            except (HTTPError, URLError, TimeoutError, OSError, RuntimeError, ValueError) as exc:
                exclusions.append(_exclusion(candidate, reason="image_unreadable_or_download_failed", detail=str(exc)))
                continue
            selected_ids.add(image_id)
            selected.append(_public_row(candidate, image, split=split))
            count += 1

    # If one type is short, fill the requested total with the other type while
    # retaining the actual task label.  This is recorded rather than hidden.
    total_target = sum(int(value) for value in task_targets.values())
    if len(selected) < total_target:
        for task in TASK_TYPES:
            for candidate in ranked.get(task, []):
                if len(selected) >= total_target:
                    break
                image_id = str(candidate["image_id"])
                if image_id in selected_ids:
                    continue
                try:
                    image = image_records.get(image_id)
                    if image is None:
                        image = materialize_image(
                            image_id=image_id,
                            image_ref=str(candidate["image_ref"]),
                            image_dir=image_dir,
                            local_index=local_index,
                        )
                        image_records[image_id] = image
                except (HTTPError, URLError, TimeoutError, OSError, RuntimeError, ValueError) as exc:
                    exclusions.append(_exclusion(candidate, reason="image_unreadable_or_download_failed", detail=str(exc)))
                    continue
                selected_ids.add(image_id)
                selected.append(_public_row(candidate, image, split=split))
                exclusions.append(_exclusion(candidate, reason="quota_fallback_other_task_type"))
    counts = dict(Counter(row["task_type"] for row in selected))
    return selected, {task: counts.get(task, 0) for task in TASK_TYPES}, image_records


def _semantic_text_status(row: Mapping[str, Any]) -> tuple[str, str]:
    """Perform text-only screening; never claim visual/semantic human review."""

    question = str(row["question"]).strip().lower()
    entity = str(row.get("reference_entity", "")).strip().lower()
    if not entity or entity == "unresolved":
        return "manual_review_required", "reference CoT target entity could not be extracted"
    if row["task_type"] == "attribute" and not (
        _ATTRIBUTE_RE.search(question) or _ATTRIBUTE_SHORT_RE.search(question)
    ):
        return "manual_review_required", "attribute label is not lexically explicit in the question"
    if row["task_type"] == "object" and not (
        _OBJECT_QUESTION_RE.search(question) and not _NON_OBJECT_QUESTION_RE.search(question)
    ):
        return "manual_review_required", "object target is not lexically explicit in the question"
    return "text_screen_pass_visual_review_pending", "text-only check passed; image-to-entity correspondence remains unreviewed"


def _contact_row(row: Mapping[str, Any], *, geometry_blind_review: bool) -> dict[str, Any]:
    status, reason = _semantic_text_status(row)
    return {
        "sample_id": row["sample_id"],
        "split": row["sample_id"].split(":", 1)[0],
        "task_type": row["task_type"],
        "image_id": row["image_id"],
        "image_path": row["image_path"],
        "question": row["question"],
        "reference_cot_last_entity": extract_final_entity(str(row["reference_cot"])),
        "reference_answer": row["reference_answer"],
        "text_screen_status": status,
        "text_screen_reason": reason,
        "geometry_blind_review": geometry_blind_review,
    }


def _file_record(root: Path, path: Path, *, source: str | None = None) -> dict[str, Any]:
    return {
        "path": str(path.absolute()),
        "relative_path": _source_relative(root, path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
        "source": source,
    }


def _assert_no_generated_outputs(root: Path) -> None:
    generated = (
        "dataset_manifest.json",
        "pilot.jsonl",
        "formal.jsonl",
        "exclusions.jsonl",
        "contact_sheet_index.jsonl",
        "semantic_review.jsonl",
    )
    existing = [str(root / name) for name in generated if (root / name).exists()]
    if existing:
        raise FileExistsError("refusing to overwrite generated outputs: " + ", ".join(existing))


def prepare_dataset(
    root: str | Path = DEFAULT_ROOT,
    *,
    seed: int = SEED,
    local_image_dirs: Sequence[str | Path] = (),
    raw_source_root: str | Path | None = None,
) -> PreparationResult:
    """Prepare the frozen pilot/formal rows and provenance artifacts.

    The output directory may already contain the downloaded ``raw/`` tree from
    an interrupted or separately verified download.  Generated artifacts are
    fail-closed and are never overwritten.
    """

    root = Path(root).absolute()
    root.mkdir(parents=True, exist_ok=True)
    _assert_no_generated_outputs(root)
    raw_source_path = Path(raw_source_root).absolute() if raw_source_root else None
    raw_records = ensure_raw_jsons(root, raw_source_root=raw_source_path)
    candidates_by_split_task, exclusions, train_variant_stats = _raw_candidates(root)
    image_dirs = _default_image_dirs([Path(path) for path in local_image_dirs])
    local_index = discover_local_images(image_dirs)
    image_dir = root / "images"

    formal_rows, formal_counts, formal_images = _select_materialized(
        candidates_by_task=candidates_by_split_task["formal"],
        task_targets={"attribute": PER_TASK_FORMAL, "object": PER_TASK_FORMAL},
        split="formal",
        seed=seed,
        used_image_ids=set(),
        image_dir=image_dir,
        local_index=local_index,
        exclusions=exclusions,
    )
    formal_ids = {str(row["image_id"]) for row in formal_rows}
    pilot_rows, pilot_counts, pilot_images = _select_materialized(
        candidates_by_task=candidates_by_split_task["pilot"],
        task_targets={"attribute": PER_TASK_PILOT, "object": PER_TASK_PILOT},
        split="pilot",
        seed=seed,
        used_image_ids=formal_ids,
        image_dir=image_dir,
        local_index=local_index,
        exclusions=exclusions,
    )

    # Materialized image records may be shared by split selection; combine by ID.
    image_records = {**formal_images, **pilot_images}
    pilot_rows.sort(key=lambda row: row["sample_id"])
    formal_rows.sort(key=lambda row: row["sample_id"])
    all_rows = pilot_rows + formal_rows
    all_image_ids = {str(row["image_id"]) for row in all_rows}
    all_sample_ids = [str(row["sample_id"]) for row in all_rows]
    if len(all_sample_ids) != len(set(all_sample_ids)):
        raise RuntimeError("sample_id collision after source occurrence qualification")
    if len(pilot_rows) != PILOT_TARGET or len(formal_rows) != FORMAL_TARGET:
        exclusions.append(
            {
                "split": "all",
                "task_type": None,
                "source_id": None,
                "source_file": None,
                "image_id": None,
                "reason": "requested_quota_unmet",
                "detail": f"pilot={len(pilot_rows)}/{PILOT_TARGET}, formal={len(formal_rows)}/{FORMAL_TARGET}",
            }
        )

    formal_blind_rows = stable_order(formal_rows, seed=seed, namespace="formal:geometry-blind")[:FORMAL_GEOMETRY_BLIND_TARGET]
    formal_blind_ids = [str(row["sample_id"]) for row in formal_blind_rows]
    formal_blind_id_set = set(formal_blind_ids)

    contact_rows = [
        _contact_row(row, geometry_blind_review=row["sample_id"] in formal_blind_id_set)
        for row in all_rows
    ]
    semantic_rows = [
        {
            "sample_id": item["sample_id"],
            "image_id": item["image_id"],
            "task_type": item["task_type"],
            "text_screen_status": item["text_screen_status"],
            "text_screen_reason": item["text_screen_reason"],
            "visual_review_status": "pending_manual_review",
            "visual_review_required": True,
        }
        for item in contact_rows
    ]

    pilot_path = root / "pilot.jsonl"
    formal_path = root / "formal.jsonl"
    exclusions_path = root / "exclusions.jsonl"
    contact_path = root / "contact_sheet_index.jsonl"
    semantic_path = root / "semantic_review.jsonl"
    _write_jsonl(pilot_path, pilot_rows)
    _write_jsonl(formal_path, formal_rows)
    _write_jsonl(exclusions_path, exclusions)
    _write_jsonl(contact_path, contact_rows)
    _write_jsonl(semantic_path, semantic_rows)

    image_manifest = [
        {
            "image_id": record.image_id,
            "image_path": record.image_path,
            "source_kind": record.source_kind,
            "source_path": record.source_path,
            "source_url": record.source_url,
            "source_image_ref": record.source_image_ref,
            "sha256": record.sha256,
            "size_bytes": record.size_bytes,
            "width": record.width,
            "height": record.height,
        }
        for record in sorted(image_records.values(), key=lambda item: item.image_id)
    ]
    output_file_paths = [pilot_path, formal_path, exclusions_path, contact_path, semantic_path]
    file_records = raw_records + [
        _file_record(root, path, source="generated_by_mm_gcoth_data_preparer") for path in output_file_paths
    ]
    file_records.extend(
        _file_record(root, Path(record.image_path), source=record.source_url or record.source_path)
        for record in sorted(image_records.values(), key=lambda item: item.image_id)
    )
    text_status_counts = Counter(item["text_screen_status"] for item in contact_rows)
    blocked = len(pilot_rows) < PILOT_TARGET or len(formal_rows) < FORMAL_TARGET
    manifest = {
        "schema_version": "mmgcot_diagnostic_data_v1",
        "status": "pending_manual_semantic_review" if not blocked else "blocked_quota_unmet",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "dataset_root": str(root),
        "protocol": {
            "seed": seed,
            "hf_dataset": HF_DATASET,
            "hf_revision": HF_REVISION,
            "tasks": list(TASK_TYPES),
            "pilot_target_unique_images": PILOT_TARGET,
            "formal_target_unique_images": FORMAL_TARGET,
            "pilot_per_task_target": PER_TASK_PILOT,
            "formal_per_task_target": PER_TASK_FORMAL,
            "formal_geometry_blind_target_unique_images": FORMAL_GEOMETRY_BLIND_TARGET,
            "selection_order": "formal Test candidates first, then pilot Train _cot candidates excluding formal image_id",
            "selection_rank": "sha256(seed, split/task namespace, image_id, source_id, source_row_index)",
            "train_variant_selection": train_variant_stats,
            "train_task_label_source": "no official Attribute/Object label exists in Train JSON; pilot labels use the conservative explicit-question rule approved for this diagnostic and remain a manually confirmed label set",
            "train_task_label_rule": {
                "attribute": "explicit What is/are the color, material, shape, size, condition, state, pattern, activity, texture, appearance or style",
                "object": "explicit What is/are or What/Which object/thing/item/animal/person category question, excluding relation, text, number, count, age, and attribute terms",
                "other": "excluded; no ordinary noun-word fallback",
            },
            "image_qualification": ["Pillow-readable", "finite positive dimensions"],
            "bbox_qualification": ["source normalized xywh", "converted normalized xyxy legal", "final CoT target matches coordinate for Test"],
            "semantic_qualification": "text-only screen plus pending manual image/entity review; no model-effect filtering and no visual audit claim",
        },
        "source_files": raw_records,
        "raw_source_root": str(raw_source_path) if raw_source_path else None,
        "local_image_search_dirs": [str(path.absolute()) for path in image_dirs],
        "counts": {
            "candidate_rows_by_split_task": {
                split: {
                    task: len(candidates_by_split_task[split][task])
                    for task in TASK_TYPES
                }
                for split in ("pilot", "formal")
            },
            "pilot_rows": len(pilot_rows),
            "pilot_unique_images": len({row["image_id"] for row in pilot_rows}),
            "pilot_by_task": pilot_counts,
            "formal_rows": len(formal_rows),
            "formal_unique_images": len({row["image_id"] for row in formal_rows}),
            "formal_by_task": formal_counts,
            "combined_unique_images": len(all_image_ids),
            "formal_geometry_blind_unique_images": len(formal_blind_ids),
            "exclusions": len(exclusions),
            "semantic_text_status": dict(text_status_counts),
        },
        "splits": {
            "pilot": {
                "path": str(pilot_path.absolute()),
                "sample_ids": [row["sample_id"] for row in pilot_rows],
                "image_ids": sorted({row["image_id"] for row in pilot_rows}),
            },
            "formal": {
                "path": str(formal_path.absolute()),
                "sample_ids": [row["sample_id"] for row in formal_rows],
                "image_ids": sorted({row["image_id"] for row in formal_rows}),
                "geometry_blind_review_sample_ids": formal_blind_ids,
                "geometry_blind_review_image_ids": sorted({row["image_id"] for row in formal_blind_rows}),
            },
        },
        "manual_review": {
            "contact_sheet_index": str(contact_path.absolute()),
            "semantic_review_jsonl": str(semantic_path.absolute()),
            "visual_semantic_review_required": True,
            "visual_review_status": "pending_manual_review",
            "selected_rows_pending": len(all_rows),
            "text_screen_pass_but_visual_pending": text_status_counts.get("text_screen_pass_visual_review_pending", 0),
            "text_screen_manual_review_required": text_status_counts.get("manual_review_required", 0),
            "statement": "No selected image/entity correspondence has been claimed as visually audited by this preparer.",
        },
        "images": image_manifest,
        "files": file_records,
        "blocking": {
            "blocked": blocked,
            "reasons": (["requested split quota unmet"] if blocked else []),
            "gpu_model_run_started": False,
        },
    }
    manifest_path = root / "dataset_manifest.json"
    _write_json(manifest_path, manifest)
    return PreparationResult(
        root=root,
        manifest_path=manifest_path,
        pilot_path=pilot_path,
        formal_path=formal_path,
        exclusions_path=exclusions_path,
        contact_index_path=contact_path,
        semantic_review_path=semantic_path,
        counts=manifest["counts"],
        blocked=blocked,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--local-image-dir",
        action="append",
        default=[],
        type=Path,
        help="additional exact-ID local VG image directory; may be repeated",
    )
    parser.add_argument(
        "--raw-source-root",
        type=Path,
        default=None,
        help="existing MM-GCoT root whose raw files were copied/hard-linked into output",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        result = prepare_dataset(
            args.output_dir,
            seed=args.seed,
            local_image_dirs=args.local_image_dir,
            raw_source_root=args.raw_source_root,
        )
    except Exception as exc:
        print(f"MM-GCoT preparation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "root": str(result.root),
        "manifest": str(result.manifest_path),
        "pilot": str(result.pilot_path),
        "formal": str(result.formal_path),
        "exclusions": str(result.exclusions_path),
        "contact_sheet_index": str(result.contact_index_path),
        "semantic_review": str(result.semantic_review_path),
        "counts": result.counts,
        "blocked": result.blocked,
        "gpu_model_run_started": False,
    }, ensure_ascii=False, indent=2))
    return 0 if not result.blocked else 3


if __name__ == "__main__":
    raise SystemExit(main())
