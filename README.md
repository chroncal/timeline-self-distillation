# Timeline Self-Distillation — folder backup

Private source snapshot of the timeline self-distillation module and its two supporting directories, created on 2026-09-17 (UTC).

## Scope

The following directories are copied without content changes from the working tree of `routed-grounding-repair-verl`:

- `timeline_self_distillation/`: 28 Python source/test files and one teacher-contract document.
- `live_kv_probe_prototype/`: 5 source/documentation files.
- `reasoning_checkpoints/`: 8 source/test/protocol/documentation files and 5 symbolic links.

Python bytecode and tool caches are excluded. Files outside these three directories—including experiment results, reports under `docs/`, datasets, model weights, trained checkpoints, and the `verl/` implementation—are **not included**. The repository-root README, `.gitignore`, and `SHA256SUMS` are backup metadata, not changes to the original source.

The five symbolic links in `reasoning_checkpoints/` are preserved with their original relative targets under `../outputs/research_experiments/reasoning_checkpoints/`. Their target directories are **not copied**, so those links will remain unresolved until the separate experiment outputs are restored. No links were dereferenced during this backup.

Source parent revision: `4dacbccf68f8a9d318ef4815b53af12ba4a8d165`. The parent working tree was not clean; this backup captures the actual working files, including files not tracked in that parent revision. Use this repository's commit and `SHA256SUMS` to identify the backed-up contents.

## Runtime dependencies

This is a source backup, **not a self-contained runnable distribution**. The previously omitted `live_kv_probe_prototype` and `reasoning_checkpoints` modules are now included, but scripts still import `verl` and third-party packages. Existing model/data/output paths in the source remain unchanged. Running the experiments requires a compatible original project or separately supplied dependencies and assets; dependency closure and standalone execution have not been validated by this backup operation.

## Issue #1 teacher-definition fix

`run_opd_micro.py` now defaults to a question-conditioned `zero_cot` teacher:
it forks the untouched `C0` cache and immediately appends the fixed bbox output
prefix. It consumes no generated reasoning, entity bridge, or repeated question.
The former `early_span1` and related paths remain available as explicitly named
legacy controls. See `timeline_self_distillation/TEACHER_DEFINITIONS.md` for the
exact conditioning contracts and limitations.

To verify all 42 regular snapshot files from the repository root (symbolic links are preserved separately in Git):

```bash
sha256sum -c SHA256SUMS
```
