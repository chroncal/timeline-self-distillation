# Timeline Self-Distillation — folder backup

Private source snapshot of `timeline_self_distillation/`, created on 2026-09-17 (UTC).

## Scope

The `timeline_self_distillation/` directory contains the 27 original Python source and test files, copied without content changes from the working tree of `routed-grounding-repair-verl`.

Only this directory is backed up. Python bytecode and tool caches are excluded. Files outside it—including experiment results, reports under `docs/`, datasets, model weights, and trained checkpoints—are **not included**. The repository-root README, `.gitignore`, and `SHA256SUMS` are backup metadata, not changes to the original source.

Source parent revision: `4dacbccf68f8a9d318ef4815b53af12ba4a8d165`. The parent working tree was not clean; this backup captures the actual working files, including files not tracked in that parent revision. Use this repository's commit and `SHA256SUMS` to identify the backed-up contents.

## Runtime dependencies

This is a folder backup, **not a self-contained runnable distribution**. Several scripts import sibling modules from the original project:

- `live_kv_probe_prototype`
- `reasoning_checkpoints`
- `verl`

Those directories are deliberately not copied. Existing model/data/output paths in the source remain unchanged. Running the experiments requires restoring this folder into a compatible original project and supplying its dependencies and assets.

To verify the snapshot files from the repository root:

```bash
sha256sum -c SHA256SUMS
```
