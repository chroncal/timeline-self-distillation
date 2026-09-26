"""Scientific constants and CPU-only helpers; no dataset annotations in prompts."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

SEED = 20260920
MODEL = "/mnt/sda/sujingyang/models/Qwen3.5-0.8B"
BOX_OPEN = '</think><answer>{"bbox":['
NUMBER = r"(?:1000|0|[1-9][0-9]{0,2})"
BOX_REGEX = rf"{NUMBER},{NUMBER},{NUMBER},{NUMBER}\]\}}</answer>"
ENTITY_REGEX = r'[^"\\\r\n<>0-9]{1,240}"'
ENTITY_PROMPT = (
    "\nDescribe the particular image instance selected by the completed reasoning "
    "whose property or category answers the original question. Extract the final "
    "choice, not an earlier rejected candidate. Use only identifying appearance "
    "and relationships already stated in the reasoning; do not invent details. "
    "Do not include coordinate values, estimated box edges or an explanation. "
    "Return one concise identifying noun phrase, or UNRESOLVED if the reasoning "
    'did not resolve an instance.\n</think>\n<target>"'
)


def stable_seed(*parts: object) -> int:
    blob = json.dumps([SEED, *parts], ensure_ascii=False, separators=(",", ":"))
    return int.from_bytes(hashlib.sha256(blob.encode()).digest()[:8], "big") % (2**63 - 1)


def model_input(row: dict) -> dict:
    """An explicit allowlist keeps reference CoT/answer/box out of inference."""
    return {"image_path": str(row["image_path"]), "question": str(row["question"])}


def question_prompt(question: str) -> str:
    return "Answer the question using the image. Reason as needed.\nQuestion: " + question


def bbox_suffix(question: str, entity: str | None) -> str:
    target = (
        "Locate the object whose property or category answers the original question."
        if entity is None else f"The final target is: {entity}. Locate exactly this target."
    )
    return (f"\n{target} Original question: {question}\n"
            "Return only its bounding box in the original image as xmin,ymin,xmax,ymax "
            "on a 0 to 1000 normalized scale.\n" + BOX_OPEN)


def early_offset(length: int, sentence_boundaries: list[int]) -> int:
    limit = length // 4
    eligible = [b for b in sentence_boundaries if 0 < b <= limit]
    return max(eligible, default=limit)


def comma_prefix(tokenizer, ids: list[int], coordinates: int) -> list[int] | None:
    """Return original token IDs, never re-tokenize a numeric prefix."""
    for end in range(1, len(ids) + 1):
        text = tokenizer.decode(ids[:end], skip_special_tokens=False)
        if text.count(",") == coordinates and text.endswith(","):
            if re.fullmatch(r"(?:\d+,){" + str(coordinates) + "}", text):
                return ids[:end]
        if text.count(",") > coordinates:
            break
    return None


def parse_box(text: str, completed: bool = True) -> tuple[list[int] | None, bool, str]:
    match = re.fullmatch(r'(\d+),(\d+),(\d+),(\d+)\]\}</answer>', text)
    if not match:
        return None, False, "format_or_incomplete"
    box = [int(x) for x in match.groups()]
    if not completed:
        return box, False, "generation_incomplete"
    if any(x < 0 or x > 1000 for x in box):
        return box, False, "out_of_range"
    if box[0] >= box[2] or box[1] >= box[3]:
        return box, False, "nonpositive_extent"
    return box, True, ""


def iou(box: list[int] | None, gt: list[float], valid: bool) -> float:
    if not valid or box is None:
        return 0.0
    b = [x / 1000 for x in box]
    inter = max(0., min(b[2], gt[2]) - max(b[0], gt[0])) * max(0., min(b[3], gt[3]) - max(b[1], gt[1]))
    union = (b[2]-b[0])*(b[3]-b[1]) + (gt[2]-gt[0])*(gt[3]-gt[1]) - inter
    value = inter / union if union > 0 else 0.0
    if not math.isfinite(value):
        raise ValueError("nonfinite IoU")
    return value


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def frozen_protocol() -> dict:
    return dict(
        version="mmgcot_entity_timeline_v1", seed=SEED, model=MODEL,
        tasks=["attribute", "object"], pilot_images=30, formal_images=200,
        trajectories_per_image=3, reasoning_temperature=0.8, reasoning_top_p=0.95,
        reasoning_top_k=0, reasoning_max_tokens=4096, early_fraction=0.25,
        early_boundary="last complete sentence <= floor(T/4), else floor(T/4)",
        bbox_arms=["L0", "L", "E", "R"], random_draws=4, greedy_draws=1,
        bbox_temperature=1.0, bbox_top_p=1.0, bbox_max_tokens=48,
        bbox_grammar=BOX_REGEX, entity_grammar=ENTITY_REGEX, entity_max_tokens=96,
        entity_prompt=ENTITY_PROMPT, question_prompt=question_prompt("{question}"),
        conditioned_suffix=bbox_suffix("{question}", "{entity}"),
        unconditioned_suffix=bbox_suffix("{question}", None),
        B_source="L/random/draw0", B_prefix_coordinates=[1, 2], B_arms=["E", "L", "R"],
        bootstrap_replicates=10000, main_metric="image-equal random mean IoU E-L",
        invalid_box_iou=0, accuracy_threshold="IoU > 0.5", min_pixels=3136,
        max_pixels=262144, dtype="bfloat16", attention_backend="sdpa",
        frozen_weights=True, annotations_in_model_input=False,
    )
