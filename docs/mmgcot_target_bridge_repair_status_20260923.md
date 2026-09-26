# MM-GCoT target bridge repair status (2026-09-23)

## Decision

Stop prompt iteration at the user's request. Keep all raw outputs. No L/R/E
geometry comparison, lambda calibration, or OPD/SFT training was started from
these bridge versions. The prerequisite target-reference interface has not
passed its previously frozen semantic confirmation gate. A parse-valid string
is not evidence that it identifies the student's selected bbox entity.

## Best previously completed comparison

The frozen v3p5 bridge, which separates `task_answer` from `target_entity`,
was faithful on 47/50 original development cases but 16/20 fresh confirmation
cases. The latter failed the predeclared >=90% fidelity gate in both task
types (8/10 each). See `mmgcot_target_bridge_v3_prompt_only_results_20260922.md`
for the two independent reviews and adjudication.

The v6p1 expanded prompt was stopped after 52/70 raw records: independent
partial review covered 49; 5 definite Object extraction errors among 23
reviewed Object cases already exceeded the maximum of 3 errors allowed in
the planned 35 Object cases. See
`mmgcot_target_bridge_v6p1_early_stop_20260923.md` for the preserved record
paths, hashes, and exact cases. This is a lower bound, not a full-70 result.

## Fixed development sentinel and prompt checks

The 12-case sentinel (`bridge_v7_sentinel12.jsonl`, SHA-256
`7da664848c9a4bf13326b42373f58538a8a842d0cb90118ce58fb3d5ccadbd10`)
contains seven earlier failures, two answer-value regressions, and three
v6p1 Object-level regressions. It is explicitly prompt-development data; its
error fraction is not an estimate of deployment performance. All attempts
replayed unchanged natural reasoning tokens from the frozen v1 sources and
kept GT out of the model input.

| Version | Change | Recorded sentinel outputs | Finding |
|---|---|---:|---|
| v7 | Independent user turn, target-only output | 6/12 | Repaired baby's shirt but changed table to fork and sandwich contents to bagel; stopped early. |
| v8 | Verify a v3p5 candidate at the same late state | 2/12 | Changed baby's target to `black`, an answer value; stopped early. |
| v9 | Short extra thinking audit before target output | 12/12 | Still output `black`, `plastic`, and `geometric pattern` as targets in known cases. |
| v10 | Concise v3p5 final role/group check | 7/12 | Repaired baby's shirt and retained passenger clothing group, but kept known bagel, carrots, bedspread, and cupcake errors; stopped at user request. |

Each recorded output above parsed successfully. The new prompts therefore
show that explicit wording can repair particular failures but does not make
the bridge reliable enough for the frozen formal protocol. These attempts
were not fully blind-reviewed as complete panels; the examples are direct
case checks against saved reasoning and previous adjudications.

## Preserved artifacts and next valid use

Raw records and protocols are under
`outputs/research_experiments/mmgcot_timeline_training/bridge_v{7p0,8p0,9p0,10p0}_sentinel12/`.
The 20-image untouched train/dev confirmation selection is frozen at
`/mnt/sda/sujingyang/research/datasets/mmgcot_timeline_training_v1/bridge_v7_fresh_confirmation20.jsonl`
(SHA-256 `38f53bd33934cad7c3bf3db5012e2b7c1a43e408702a7b3f500672f00bf7a85a`).
It has five images per split/task cell and zero sample, image-ID, or image-hash
overlap with the previous development/confirmation selections or the sealed
independent 48; it has **not** been run through any bridge prompt. The
independent 48 also remains unevaluated.

The existing v3p5 descriptors may support a clearly labeled exploratory
L/R/E diagnostic if their target identity is audited before interpreting
geometry. They cannot support a claim that a prompt-only bridge passed the
frozen interface gate, nor a formal R-OPD versus E-OPD training comparison
under the current protocol. A formal next run needs a separately frozen,
more reliable entity bridge and the untouched confirmation check first.
