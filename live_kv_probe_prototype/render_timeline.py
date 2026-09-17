#!/usr/bin/env python3
"""Render saved live-fork bbox probes as an interactive reasoning timeline."""

from __future__ import annotations

import argparse
import base64
import io
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECORDS = (
    ROOT
    / "outputs/research_experiments/live_kv_probe/"
    "hf_fork_formal_n5_v1/records.jsonl"
)
DEFAULT_SOURCE = (
    ROOT
    / "outputs/research_experiments/reasoning_checkpoints/"
    "pilot_seed260600564_n50_boundary_v3/trajectories.jsonl"
)
DEFAULT_MODEL = Path("/mnt/sda/sujingyang/models/Qwen3.5-0.8B")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _image_data_uri(path: Path) -> str:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.thumbnail((720, 520), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=72, optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{payload}"


def _iou(left: list[float] | None, right: list[float]) -> float | None:
    if left is None:
        return None
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return 0.0 if union <= 0 else intersection / union


def _round_box(box: list[float] | None) -> list[float] | None:
    return None if box is None else [round(float(value), 2) for value in box]


def _build_payload(args: argparse.Namespace) -> dict[str, Any]:
    source_rows = {
        str(row["sample_id"]): row
        for row in _read_jsonl(args.source)
        if row.get("ground_truth_bbox") is not None
    }
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.model), trust_remote_code=True, local_files_only=True
    )
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for row in _read_jsonl(args.records):
        condition = str(row["condition"])
        if condition == "baseline":
            continue
        sample_id = str(row["sample_id"])
        source = source_rows[sample_id]
        reasoning_ids = [int(value) for value in row["main_reasoning_token_ids"]]
        probes = sorted(row["probes"], key=lambda probe: int(probe["token_offset"]))
        previous_offset = 0
        steps = []
        ground_truth = [float(value) for value in source["ground_truth_bbox"]]
        for probe in probes:
            offset = int(probe["token_offset"])
            bbox = probe.get("bbox_xyxy")
            bbox = None if bbox is None else [float(value) for value in bbox]
            overlap = _iou(bbox, ground_truth)
            steps.append(
                {
                    "index": int(probe["checkpoint_index"]),
                    "tokenOffset": offset,
                    "tokenTotal": len(reasoning_ids),
                    "spanText": tokenizer.decode(
                        reasoning_ids[previous_offset:offset],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "prefixText": tokenizer.decode(
                        reasoning_ids[:offset],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    "bbox": _round_box(bbox),
                    "parseValid": bool(probe["parse_valid"]),
                    "iou": None if overlap is None else round(overlap, 4),
                    "hit": overlap is not None and overlap >= 0.5,
                }
            )
            previous_offset = offset
        grouped[sample_id][condition] = {"steps": steps}

    samples = []
    for sample_id in sorted(grouped, key=lambda value: int(value.split("-")[-1])):
        source = source_rows[sample_id]
        samples.append(
            {
                "id": sample_id,
                "expression": str(source["expression"]),
                "image": _image_data_uri(Path(source["image_path"])),
                "groundTruth": _round_box(
                    [float(value) for value in source["ground_truth_bbox"]]
                ),
                "conditions": grouped[sample_id],
            }
        )
    return {"samples": samples}


def _fragment(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f'''<div id="reasoning-bbox-timeline-20260916">
  <div class="viz-controls">
    <label class="form-label">样本
      <select class="form-select" id="rbt-sample"></select>
    </label>
    <label class="form-label">探针方式
      <select class="form-select" id="rbt-condition">
        <option value="natural_instruction">自然指令</option>
        <option value="forced_close">强制闭合</option>
      </select>
    </label>
    <button class="btn btn-ghost" type="button" id="rbt-prev">上一个边界</button>
    <button class="btn" type="button" id="rbt-next">下一个边界</button>
  </div>
  <label class="form-label" for="rbt-step"><span id="rbt-step-label"></span></label>
  <input class="form-range" id="rbt-step" type="range" min="0" max="0" value="0" step="1">
  <div class="rbt-status text-small" id="rbt-status" aria-live="polite"></div>
  <div class="rbt-layout">
    <div>
      <div class="rbt-expression" id="rbt-expression"></div>
      <div class="rbt-stage" role="img" aria-label="目标图像、真实框与当前 reasoning 边界的预测框">
        <img id="rbt-image" alt="">
        <div class="rbt-box rbt-gt" id="rbt-gt"><span>GT</span></div>
        <div class="rbt-box rbt-pred" id="rbt-pred"><span>probe</span></div>
      </div>
      <div class="viz-row text-small rbt-legend">
        <span><i class="rbt-swatch rbt-gt-swatch"></i>真实框</span>
        <span><i class="rbt-swatch rbt-pred-swatch"></i>探针框</span>
      </div>
    </div>
    <div class="rbt-reasoning">
      <div class="text-small text-muted">本边界刚完成的 reasoning 片段</div>
      <div id="rbt-span"></div>
      <details>
        <summary>查看截至当前边界的完整 reasoning</summary>
        <div id="rbt-prefix"></div>
      </details>
    </div>
  </div>
</div>
<style>
  #reasoning-bbox-timeline-20260916 .rbt-layout {{ display:grid; grid-template-columns:minmax(0, 1.45fr) minmax(240px, .8fr); gap:16px; align-items:start; margin-top:12px; }}
  #reasoning-bbox-timeline-20260916 .rbt-stage {{ position:relative; display:inline-block; max-width:100%; line-height:0; }}
  #reasoning-bbox-timeline-20260916 .rbt-stage img {{ display:block; max-width:100%; height:auto; }}
  #reasoning-bbox-timeline-20260916 .rbt-box {{ position:absolute; box-sizing:border-box; border:2px solid; pointer-events:none; }}
  #reasoning-bbox-timeline-20260916 .rbt-box span {{ position:absolute; left:0; top:0; padding:2px 4px; background:var(--card); color:var(--card-foreground); line-height:1.2; }}
  #reasoning-bbox-timeline-20260916 .rbt-gt {{ border-color:var(--viz-series-2); }}
  #reasoning-bbox-timeline-20260916 .rbt-pred {{ border-color:var(--viz-series-1); }}
  #reasoning-bbox-timeline-20260916 .rbt-expression {{ margin-bottom:8px; font-weight:500; }}
  #reasoning-bbox-timeline-20260916 .rbt-status {{ margin-top:6px; color:var(--foreground); }}
  #reasoning-bbox-timeline-20260916 .rbt-reasoning {{ min-width:0; }}
  #reasoning-bbox-timeline-20260916 #rbt-span,
  #reasoning-bbox-timeline-20260916 #rbt-prefix {{ white-space:pre-wrap; overflow-wrap:anywhere; margin-top:6px; }}
  #reasoning-bbox-timeline-20260916 details {{ margin-top:12px; }}
  #reasoning-bbox-timeline-20260916 .rbt-legend {{ margin-top:7px; }}
  #reasoning-bbox-timeline-20260916 .rbt-swatch {{ display:inline-block; width:12px; height:3px; margin-right:5px; vertical-align:middle; }}
  #reasoning-bbox-timeline-20260916 .rbt-gt-swatch {{ background:var(--viz-series-2); }}
  #reasoning-bbox-timeline-20260916 .rbt-pred-swatch {{ background:var(--viz-series-1); }}
  @media (max-width:600px) {{
    #reasoning-bbox-timeline-20260916 .rbt-layout {{ grid-template-columns:1fr; }}
  }}
</style>
<script>
(() => {{
  const data = {serialized};
  const root = document.getElementById('reasoning-bbox-timeline-20260916');
  const sampleSelect = root.querySelector('#rbt-sample');
  const conditionSelect = root.querySelector('#rbt-condition');
  const stepInput = root.querySelector('#rbt-step');
  const stepLabel = root.querySelector('#rbt-step-label');
  const status = root.querySelector('#rbt-status');
  const expression = root.querySelector('#rbt-expression');
  const image = root.querySelector('#rbt-image');
  const gt = root.querySelector('#rbt-gt');
  const pred = root.querySelector('#rbt-pred');
  const span = root.querySelector('#rbt-span');
  const prefix = root.querySelector('#rbt-prefix');
  const prev = root.querySelector('#rbt-prev');
  const next = root.querySelector('#rbt-next');

  data.samples.forEach((sample, index) => {{
    const option = document.createElement('option');
    option.value = String(index);
    option.textContent = `${{sample.id}} · ${{sample.expression}}`;
    sampleSelect.appendChild(option);
  }});

  function placeBox(element, box) {{
    element.style.left = `${{box[0] / 10}}%`;
    element.style.top = `${{box[1] / 10}}%`;
    element.style.width = `${{(box[2] - box[0]) / 10}}%`;
    element.style.height = `${{(box[3] - box[1]) / 10}}%`;
  }}

  function current() {{
    const sample = data.samples[Number(sampleSelect.value || 0)];
    const condition = conditionSelect.value;
    const run = sample.conditions[condition];
    const index = Math.min(Number(stepInput.value), Math.max(0, run.steps.length - 1));
    return {{sample, condition, run, index, step:run.steps[index]}};
  }}

  function resetSteps() {{
    const sample = data.samples[Number(sampleSelect.value || 0)];
    const run = sample.conditions[conditionSelect.value];
    stepInput.max = String(Math.max(0, run.steps.length - 1));
    stepInput.value = '0';
    render();
  }}

  function render() {{
    const state = current();
    const step = state.step;
    expression.textContent = `指代表达：${{state.sample.expression}}`;
    image.src = state.sample.image;
    image.alt = state.sample.expression;
    placeBox(gt, state.sample.groundTruth);
    stepLabel.textContent = `reasoning 边界 ${{state.index + 1}} / ${{state.run.steps.length}}`;
    if (!step) {{
      pred.style.display = 'none';
      status.textContent = '该条件没有探针记录';
      span.textContent = '';
      prefix.textContent = '';
      return;
    }}
    const bboxText = step.bbox ? `[${{step.bbox.join(', ')}}]` : '不可解析';
    const iouText = step.iou === null ? '—' : step.iou.toFixed(3);
    const hitText = step.hit ? '命中（IoU ≥ 0.5）' : '未命中';
    status.textContent = `token ${{step.tokenOffset}} / ${{step.tokenTotal}} · bbox ${{bboxText}} · IoU ${{iouText}} · ${{hitText}}`;
    span.textContent = step.spanText || '（该边界没有新增可见文本）';
    prefix.textContent = step.prefixText;
    if (step.parseValid && step.bbox) {{
      pred.style.display = 'block';
      placeBox(pred, step.bbox);
    }} else {{
      pred.style.display = 'none';
    }}
    prev.disabled = state.index <= 0;
    next.disabled = state.index >= state.run.steps.length - 1;
  }}

  sampleSelect.addEventListener('change', resetSteps);
  conditionSelect.addEventListener('change', resetSteps);
  stepInput.addEventListener('input', render);
  prev.addEventListener('click', () => {{ stepInput.value = String(Math.max(0, Number(stepInput.value) - 1)); render(); }});
  next.addEventListener('click', () => {{ stepInput.value = String(Math.min(Number(stepInput.max), Number(stepInput.value) + 1)); render(); }});
  sampleSelect.value = '0';
  conditionSelect.value = 'natural_instruction';
  resetSteps();
}})();
</script>
'''


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=DEFAULT_RECORDS)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = _build_payload(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(_fragment(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sample_count": len(payload["samples"]),
                "bytes": args.output.stat().st_size,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
