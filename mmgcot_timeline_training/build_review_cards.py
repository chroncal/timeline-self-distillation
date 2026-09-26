"""Render anonymous semantic-review cards from a merged blind-review packet."""

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


def build(packet_path: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    rows = [json.loads(line) for line in packet_path.read_text().splitlines() if line.strip()]
    mapping = []
    title_font, text_font = _font(28), _font(22)
    for row in rows:
        case_id = hashlib.sha256(("semantic-review-v1:" + row["sample_id"]).encode()).hexdigest()[:12]
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        scale = min(1100 / image.width, 720 / image.height, 1.0)
        display = image.resize((round(image.width * scale), round(image.height * scale)))
        draw = ImageDraw.Draw(display)
        x1, y1, x2, y2 = row["ground_truth_bbox"]
        box = [round(x1 * display.width), round(y1 * display.height),
               round(x2 * display.width), round(y2 * display.height)]
        draw.rectangle(box, outline=(255, 0, 0), width=max(3, round(min(display.size) / 150)))
        lines = [
            f"Case: {case_id}",
            "Question: " + row["question"],
            "Target description: " + (row["target_description"] or "<EMPTY>"),
            "Choose one: same_target | different_target | description_not_unique | cannot_determine",
        ]
        wrapped = []
        for line in lines:
            wrapped.extend(textwrap.wrap(line, width=88) or [""])
        panel_height = 36 + len(wrapped) * 31
        canvas = Image.new("RGB", (max(1100, display.width), display.height + panel_height), "white")
        canvas.paste(display, ((canvas.width - display.width) // 2, 0))
        canvas_draw = ImageDraw.Draw(canvas)
        y = display.height + 15
        for index, line in enumerate(wrapped):
            canvas_draw.text((20, y), line, fill="black", font=title_font if index == 0 else text_font)
            y += 31
        card_path = output_dir / f"{case_id}.png"
        canvas.save(card_path)
        mapping.append({
            "case_id": case_id,
            "sample_id": row["sample_id"],
            "card_path": str(card_path.absolute()),
            "review_label": None,
            "allowed_review_labels": row["allowed_review_labels"],
        })
    with (output_dir / "private_mapping.jsonl").open("x", encoding="utf-8") as handle:
        for row in mapping:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    (output_dir / "README.txt").write_text(
        "Review each PNG without consulting model boxes, IoU, teacher probabilities, or training results.\n"
        "Return one allowed label per case_id. The red rectangle is the dataset GT target.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    build(args.packet, args.output_dir)


if __name__ == "__main__":
    main()
