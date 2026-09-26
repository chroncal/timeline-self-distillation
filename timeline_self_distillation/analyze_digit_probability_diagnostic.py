"""CPU-only aggregation of the read-only digit attribution artifacts."""
import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from live_kv_probe_prototype.run_hf_fork import _write_json


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def classify(row):
    d = row["distributions"]
    base, teacher, student = [d[k]["alternative_probability"] for k in ("base", "teacher", row["checkpoint_model"])]
    if student <= base:
        return "alternative_not_raised_by_update"
    if teacher <= base:
        return "teacher_absolute_probability_lower_but_student_higher"
    if student > teacher:
        return "teacher_raised_update_exceeded_teacher"
    return "teacher_raised_update_between_base_and_teacher"


def run(path, historical):
    scores = read(path / "prefix_distributions.jsonl")
    lookup = {(s["sample_id"], tuple(s["prefix_ids"])): s for s in scores}
    pairs = read(path / "pair_attribution.jsonl")
    decodes = read(historical / "diagnosis_decode_n5_d16_v1/records.jsonl")
    interventions = read(path / "interventions.jsonl")
    aggregate = {}
    for model in ("reverse", "forward"):
        sub = [p for p in pairs if p["checkpoint_model"] == model]
        negative = [p for p in sub if p["iou_delta"] < 0]
        for label, selected in (("all_changed", sub), ("negative", negative)):
            classes = Counter(classify(p) for p in selected)
            aggregate[f"{model}_{label}"] = {
                "n": len(selected), "classification_counts": dict(classes),
                "classification_signed_iou_decrease_sums": {c: sum(-p["iou_delta"] for p in selected if classify(p) == c) for c in classes},
                "unique_prefixes": len({(p["sample_id"], tuple(p["prefix_ids"])) for p in selected}),
                "unique_prefix_and_alternative": len({(p["sample_id"], tuple(p["prefix_ids"]), p["alternative_token_id"]) for p in selected}),
                "teacher_raises_relative_odds": sum(p["distributions"]["teacher"]["log_odds_delta_from_base"] > 0 for p in selected),
                "student_raises_relative_odds": sum(p["distributions"][model]["log_odds_delta_from_base"] > 0 for p in selected),
                "student_exceeds_teacher_relative_odds": sum(p["distributions"][model]["alternative_vs_reference_log_odds"] > p["distributions"]["teacher"]["alternative_vs_reference_log_odds"] for p in selected),
                "base_kl_descent_raises_alternative_probability": sum(p["local_logit_gradient_directions"]["base"]["alternative_probability_velocity"] > 0 for p in selected),
                "checkpoint_kl_descent_lowers_alternative_probability": sum(p["local_logit_gradient_directions"][model]["alternative_probability_velocity"] < 0 for p in selected),
                "student_keeps_base_argmax": sum(p["distributions"][model]["top_id"] == p["distributions"]["base"]["top_id"] for p in selected),
            }
    greedy = [r for r in decodes if r["model"] == "base" and r["mode"] == "greedy"]
    entropy = {}
    for policy in ("base", "teacher", "reverse", "forward"):
        values, chosen_probs = [], []
        for record in greedy:
            for index, piece in enumerate(record["pieces"]):
                if not piece.isdigit():
                    continue
                row = lookup[(record["sample_id"], tuple(record["ids"][:index]))]
                values.append(row["distributions"][policy]["entropy"])
                chosen_probs.append(row["distributions"][policy]["probs"][row["support_ids"].index(record["ids"][index])])
        entropy[policy] = {"n": len(values), "mean_entropy_nats": mean(values), "mean_base_chosen_probability": mean(chosen_probs)}
    parity_errors = []
    old = read(historical / "diagnosis_checkpoint_n5_v1/fixed_prefix.jsonl")
    for record in old:
        for policy, old_key in (("base", "base"), ("teacher", "teacher_early_span1"), ("reverse", "reverse_checkpoint"), ("forward", "forward_checkpoint")):
            expected = record["metrics"][old_key]["chosen_token_nll"]
            for i, position in enumerate(record["numeric_target_positions"]):
                row = lookup[(record["sample_id"], tuple(record["ids"][:position]))]
                chosen_index = row["support_ids"].index(record["ids"][position])
                actual = -row["distributions"][policy]["log_probs"][chosen_index]
                parity_errors.append(abs(actual - expected[i]))
    if max(parity_errors) > 1e-5:
        raise RuntimeError(f"Historical teacher/student NLL mismatch: {max(parity_errors)}")
    cf = {}
    for model in ("reverse", "forward"):
        for donor in ("base", "teacher"):
            selected = [r for r in interventions if r["checkpoint_model"] == model and r["donor"] == donor]
            negatives = [r for r in selected if r["original_checkpoint_iou"] < r["base_iou"]]
            cf[f"{model}_{donor}"] = {
                "changed_pairs": len(selected), "base_bbox_restored_count": sum(r["restored_base_bbox_text"] for r in selected),
                "negative_pairs": len(negatives), "negative_base_bbox_restored_count": sum(r["restored_base_bbox_text"] for r in negatives),
                "negative_iou_improved": sum(r["iou"] > r["original_checkpoint_iou"] for r in negatives),
                "negative_iou_equal": sum(r["iou"] == r["original_checkpoint_iou"] for r in negatives),
                "negative_iou_worse": sum(r["iou"] < r["original_checkpoint_iou"] for r in negatives),
            }
    result = {"first_divergence": aggregate, "fixed_base_greedy_prefix_numeric_rows": entropy,
              "historical_numeric_nll_comparisons": len(parity_errors), "max_historical_nll_absolute_error": max(parity_errors),
              "single_position_interventions": cf,
              "limitations": ["Five seen images; sampled pairs are not independent training runs.",
                              "First-divergence alternative is not a globally wrong token; final IoU labels complete rollouts.",
                              "Absolute probability and relative-odds attribution differ; neither reconstructs the optimizer trajectory.",
                              "Single-position interventions use oracle divergence locations and are diagnostic, not a deployable method."]}
    _write_json(path / "analysis.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-dir", type=Path, default=Path("outputs/research_experiments/timeline_opd"))
    args = parser.parse_args()
    run(args.output_dir, args.historical_dir)
