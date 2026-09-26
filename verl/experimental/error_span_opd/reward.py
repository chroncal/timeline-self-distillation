"""Independent bbox evaluation; never used as a training reward in this recipe."""
from __future__ import annotations

from verl.experimental.routed_grounding.router import parse_response, xyxy_iou


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    # Native-thinking prompts already contain the opening delimiter, whereas
    # verl passes only the generated response to the reward adapter.
    if not solution_str.lstrip().startswith("<think>") and solution_str.count("</think>") == 1:
        solution_str = "<think>\n" + solution_str
    parsed = parse_response(solution_str)
    valid = bool(parsed.parse_valid and parsed.bbox is not None)
    iou = float(xyxy_iou(parsed.bbox, ground_truth)) if valid else 0.0
    return {"score": float(valid and iou >= 0.5), "accuracy": float(valid and iou >= 0.5),
            "iou": iou, "invalid": float(not valid)}
