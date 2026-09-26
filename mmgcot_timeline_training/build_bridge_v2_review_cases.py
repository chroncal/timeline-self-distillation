"""Build anonymous two-stage review cases for target bridge v2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


def _font(size: int):
    for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"):
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def build(packet_paths: list[Path], output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    for name in ("cards", "stage1", "stage2"):
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    rows = []
    for path in packet_paths:
        cohort = path.parent.name
        rows.extend((cohort, json.loads(line)) for line in path.read_text().splitlines() if line.strip())
    mapping = []
    title_font, text_font = _font(28), _font(22)
    for cohort, row in rows:
        case_id = hashlib.sha256(("semantic-review-v2:" + row["sample_id"]).encode()).hexdigest()[:12]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        scale = min(1100 / image.width, 760 / image.height, 1.0)
        display = image.resize((round(image.width * scale), round(image.height * scale)))
        draw = ImageDraw.Draw(display)
        x1, y1, x2, y2 = row["ground_truth_bbox"]
        draw.rectangle(
            [round(x1 * display.width), round(y1 * display.height),
             round(x2 * display.width), round(y2 * display.height)],
            outline=(255, 0, 0), width=max(3, round(min(display.size) / 150)),
        )
        lines = [f"Case: {case_id}", "Question: " + row["question"]]
        wrapped = [piece for line in lines for piece in (textwrap.wrap(line, width=88) or [""])]
        panel_height = 36 + len(wrapped) * 31
        canvas = Image.new("RGB", (max(1100, display.width), display.height + panel_height), "white")
        canvas.paste(display, ((canvas.width - display.width) // 2, 0))
        canvas_draw = ImageDraw.Draw(canvas)
        y = display.height + 15
        for index, line in enumerate(wrapped):
            canvas_draw.text((20, y), line, fill="black", font=title_font if index == 0 else text_font)
            y += 31
        card_path = output_dir / "cards" / f"{case_id}.png"
        canvas.save(card_path)

        stage1 = {
            "case_id": case_id,
            "question": row["question"],
            "frozen_reasoning": row["reasoning_text"],
            "reasoning_finish": row["reasoning_finish"],
            "student_target_selection_error": None,
            "allowed_labels": row["student_target_selection_error_labels"],
            "instruction": (
                "Judge only what entity the frozen student reasoning finally selected versus the red GT box. "
                "Record this decision before opening the corresponding stage2 file."
            ),
        }
        stage2 = {
            "case_id": case_id,
            "target_entity_reference": row["target_entity_reference"],
            "extractor_serialization_error": (
                row["extractor_serialization_error"]
                if "extractor_serialization_error" in row
                else "none" if row.get("bridge_parse_status") == "valid" else "grammar_mismatch"
            ),
            "extractor_content_error": None,
            "extractor_content_error_labels": row["extractor_content_error_labels"],
            "target_reference_review_label": None,
            "target_reference_review_labels": row["target_reference_review_labels"],
            "instruction": (
                "Compare target_entity_reference with the frozen reasoning already reviewed in stage1. "
                "Do not reinterpret a student selection mistake as an extractor mistake. Then label whether "
                "the extracted reference points to the red GT target."
            ),
        }
        # v4 never generates a task answer. Keep prior-schema packets readable
        # and expose the v4 intermediate stages for error attribution.
        if "task_answer" in row:
            stage2["task_answer"] = row["task_answer"]
        for field in ("target_source", "query_entity", "candidate_entity",
                      "frame_parse_status", "candidate_parse_status", "verify_parse_status"):
            if field in row:
                stage2[field] = row[field]
        (output_dir / "stage1" / f"{case_id}.json").write_text(
            json.dumps(stage1, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        (output_dir / "stage2" / f"{case_id}.json").write_text(
            json.dumps(stage2, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )
        mapping.append({
            "case_id": case_id,
            "sample_id": row["sample_id"],
            "cohort": cohort,
            "split": row["split"],
            "task_type": row["task_type"],
            "card_path": str(card_path.absolute()),
            "stage1_path": str((output_dir / "stage1" / f"{case_id}.json").absolute()),
            "stage2_path": str((output_dir / "stage2" / f"{case_id}.json").absolute()),
        })
    with (output_dir / "private_mapping.jsonl").open("x", encoding="utf-8") as handle:
        for row in mapping:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "README.txt").write_text(
        "For every case, inspect the card and stage1 JSON and record student_target_selection_error.\n"
        "Only then open stage2 and record extractor_content_error and target_reference_review_label.\n"
        "Never consult L/R/E boxes, IoU, teacher probabilities, or training results.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.packet, args.output_dir)


if __name__ == "__main__":
    main()
