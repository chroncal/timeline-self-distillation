"""Frozen paired A/B/C report for pure E-OPD bbox-stage ablations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from mmgcot_diagnostic.protocol import file_hash
from mmgcot_timeline_training.formal_analysis import aggregate_systems


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (20260921, 20260922, 20260923)
NAMES = {"A": "numeric-only terminal Q", "B": "coordinate-decision terminal Q",
         "C": "coordinate-decision last-two QVO"}


def _input_paths(output_root: Path, cohort: str):
    prior = ROOT / "outputs/research_experiments/mmgcot_timeline_training/pure_opd_v2"
    original = prior / "evaluation" / cohort
    new = output_root / "evaluation" / cohort
    a_paths = []
    for seed in SEEDS:
        old = original / f"e_opd_pure_seed{seed}.jsonl"
        a_paths.append(old if old.with_suffix(".complete.json").is_file() else
                       new / f"a_numeric_seed{seed}.jsonl")
    files = {
        "A": a_paths,
        "B": [new / f"terminal_q_seed{s}.jsonl" for s in SEEDS],
        "C": [new / f"last_two_qvo_seed{s}.jsonl" for s in SEEDS],
    }
    return files


def _verify(files, output_root: Path):
    frozen = json.loads((output_root / "formal/frozen_checkpoints.json").read_text())
    prior_frozen = json.loads((ROOT / "outputs/research_experiments/mmgcot_timeline_training"
                               "/pure_opd_v2/formal/frozen_checkpoints.json").read_text())
    for entry in frozen["checkpoints"].values():
        if file_hash(entry["path"]) != entry["sha256"]:
            raise RuntimeError("B/C frozen checkpoint changed")
    for arm, paths in files.items():
        for seed, path in zip(SEEDS, paths, strict=True):
            completion = path.with_suffix(".complete.json")
            meta = json.loads(completion.read_text())
            if meta["output_sha256"] != file_hash(path) or meta["seed"] != seed:
                raise RuntimeError(f"evaluation output receipt differs: {path}")
            if arm == "A":
                entry = prior_frozen["checkpoints"][f"e_opd_pure_seed{seed}"]
                if meta["checkpoint_sha256"] != entry["sha256"] or file_hash(entry["path"]) != entry["sha256"]:
                    raise RuntimeError(f"A frozen checkpoint differs: {path}")
            else:
                scope = "terminal_q" if arm == "B" else "last_two_qvo"
                entry = frozen["checkpoints"][f"{scope}_seed{seed}"]
                if meta["checkpoint_sha256"] != entry["sha256"]:
                    raise RuntimeError(f"evaluation checkpoint differs: {path}")


def _plot(result: dict, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for axis, (label, candidate, baseline) in zip(
        axes, (("B−A", "B", "A"), ("C−B", "C", "B")), strict=True
    ):
        series = []
        for seed in SEEDS:
            a = result["systems"][candidate]["per_seed"][seed]["per_image"]
            b = result["systems"][baseline]["per_seed"][seed]["per_image"]
            series.append({(r["sample_id"], r["image_id"]): r["mIoU"]
                           for r in a})
            baseline_scores = {(r["sample_id"], r["image_id"]): r["mIoU"] for r in b}
            series[-1] = {key: value - baseline_scores[key] for key, value in series[-1].items()}
        values = [sum(group[key] for group in series) / len(series) for key in series[0]]
        axis.hist(values, bins=25, color="#4477aa", alpha=0.8)
        axis.axvline(0, color="black", linewidth=1)
        axis.axvline(statistics.mean(values), color="#bb4433", linewidth=2)
        axis.set_title(f"{label}: image-paired difference")
        axis.set_xlabel("mean IoU difference")
    axes[0].set_ylabel("number of images")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _auxiliary(paths, output_root: Path, cohort: str):
    prior = ROOT / "outputs/research_experiments/mmgcot_timeline_training/pure_opd_v2"
    output = {}
    for arm in ("A", "B", "C"):
        eval_rows = [json.loads(line) for path in paths[arm] for line in path.read_text().splitlines()]
        samples = [r for r in eval_rows if r["mode"] == "sample"]
        greedy = [r for r in eval_rows if r["mode"] == "greedy"]
        run_root = prior / "formal" if arm == "A" else output_root / "formal"
        prefix = "e_opd_pure" if arm == "A" else "terminal_q" if arm == "B" else "last_two_qvo"
        durations = []
        peak_memory = []
        ending_kl = []
        widths = Counter()
        for seed in SEEDS:
            run = run_root / f"{prefix}_seed{seed}"
            for line in (run / "steps.jsonl").read_text().splitlines():
                record = json.loads(line)
                durations.append(record["seconds"])
                peak_memory.append(record["peak_memory_bytes"])
                if "ending_kl" in record:
                    ending_kl.append(record["ending_kl"])
            for line in (run / "rollouts.jsonl").read_text().splitlines():
                record = json.loads(line)
                if record.get("bbox"):
                    widths.update(len(str(int(value))) for value in record["bbox"])
        output[arm] = {
            "random_invalid_fraction": sum(not (r["valid"] and r["completed"]) for r in samples) / len(samples),
            "random_iou_le_0_05_fraction": sum(
                not (r["valid"] and r["completed"]) or float(r["iou"]) <= 0.05
                for r in samples) / len(samples),
            "greedy_mean_iou": sum(float(r["iou"]) if r["valid"] and r["completed"] else 0.0
                                   for r in greedy) / len(greedy),
            "training_step_median_seconds": statistics.median(durations),
            "training_peak_memory_bytes": max(peak_memory),
            "training_mean_ending_kl": statistics.mean(ending_kl) if ending_kl else None,
            "coordinate_widths": {str(k): v for k, v in sorted(widths.items())},
            "evaluation_cohort": cohort,
        }
    return output


def _text_report(result: dict, cohort: str, inputs: dict) -> str:
    lines = [f"# Pure E-OPD A/B/C — {cohort}", "",
             "Primary metric: four random boxes → trajectory mean → image-equal mean IoU.",
             "All arms use identical frozen trajectories and evaluation seeds; invalid outputs score zero.",
             "Test-200 is a prior diagnostic cohort and is interpreted as a conditioned retest.", "",
             "| Arm | Mean IoU | Training-seed SD | Acc@0.5 |", "|---|---:|---:|---:|"]
    for arm in ("A", "B", "C"):
        data = result["systems"][arm]
        seed_scores = [data["per_seed"][s]["mIoU"] for s in SEEDS]
        lines.append(f"| {arm}: {NAMES[arm]} | {data['mIoU']:.6f} | "
                     f"{statistics.stdev(seed_scores):.6f} | {data['Acc@0.5']:.6f} |")
    lines += ["", "| Contrast | Paired mean IoU difference | 95% image-bootstrap interval |",
              "|---|---:|---:|"]
    for pair in ("B-A", "C-B", "C-A"):
        metrics = result["paired"][pair]["mIoU"]
        lo, hi = metrics["ci95"]
        lines.append(f"| {pair} | {metrics['delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ["", "Bootstrap resamples images 10,000 times and retains all three training seeds "
              "within each sampled image. It does not substitute for the reported seed variation.", "",
              "| Arm | Greedy mean IoU | Invalid random fraction | IoU≤0.05 random fraction | "
              "Median training step (s) | Peak memory (GiB) |", 
              "|---|---:|---:|---:|---:|---:|"]
    for arm in ("A", "B", "C"):
        aux = result["auxiliary"][arm]
        lines.append(f"| {arm} | {aux['greedy_mean_iou']:.6f} | "
                     f"{aux['random_invalid_fraction']:.4f} | "
                     f"{aux['random_iou_le_0_05_fraction']:.4f} | "
                     f"{aux['training_step_median_seconds']:.1f} | "
                     f"{aux['training_peak_memory_bytes']/1024**3:.2f} |")
    lines += ["", "Coordinate digit-width counts are in `result.json` under `auxiliary`; "
              "training ending-decision KL is reported there for B/C.", "",
              "Raw inputs and their SHA256:", ""]
    for arm in ("A", "B", "C"):
        for path in inputs[arm]:
            lines.append(f"- {path}: `{file_hash(path)}`")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cohort", choices=("dev", "independent48", "test200_retest"), required=True)
    p.add_argument("--output-root", type=Path, required=True)
    args = p.parse_args()
    paths = _input_paths(args.output_root, args.cohort)
    _verify(paths, args.output_root)
    manifest = json.loads((ROOT / "configs/mmgcot_formal_v2.json").read_text())["data_manifest"]
    key = {"dev": "dev", "independent48": "independent_confirmation",
           "test200_retest": "diagnostic_selection_test200"}[args.cohort]
    selection = Path(manifest[key])
    if file_hash(selection) != manifest[f"{key}_sha256"]:
        raise RuntimeError("evaluation manifest changed")
    queue = [(r["sample_id"], r["image_id"])
             for r in (json.loads(line) for line in selection.read_text().splitlines())]
    systems = {arm: [json.loads(line) for path in files for line in path.read_text().splitlines()]
               for arm, files in paths.items()}
    result = aggregate_systems(systems, queue, mode="sample",
                               comparisons=[("B", "A"), ("C", "B"), ("C", "A")],
                               bootstrap_replicates=10_000)
    result["cohort"] = args.cohort
    result["interpretation"] = ("diagnostic_selection_conditioned_retest" if
                                 args.cohort == "test200_retest" else
                                 "independent_confirmation" if args.cohort == "independent48" else
                                 "development")
    result["selection_sha256"] = file_hash(selection)
    result["auxiliary"] = _auxiliary(paths, args.output_root, args.cohort)
    out = args.output_root / "reports" / args.cohort
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "result.json"
    if destination.exists():
        raise FileExistsError(destination)
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (out / "report.md").write_text(_text_report(result, args.cohort, paths))
    _plot(result, out / "paired_image_differences.png")
    print(f"REPORT_COMPLETE {args.cohort} n={result['n_images']}", flush=True)


if __name__ == "__main__":
    main()
