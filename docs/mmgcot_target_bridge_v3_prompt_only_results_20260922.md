# MM-GCoT target bridge v3 prompt-only result

Date: 2026-09-22

## Decision

The prompt-only bridge substantially fixed the specific failure in which the
task answer was copied into the localization-entity field, but the final frozen
prompt **failed the fresh confirmation gate**.  No L/R/E geometry diagnostic,
lambda calibration, or bbox-SFT/R-OPD/E-OPD training was started.

The result supports a narrow claim: explicit counterexamples and a direct
entity-owner rule can teach Qwen3.5-0.8B not to use a bare color, material, or
shape as a bounding-box entity on the development cohort.  It does not support
the stronger claim that a single free-form prompt reliably recovers the exact
student-selected referent on unseen MM-GCoT examples.

## Frozen scientific comparison

Bridge v3 changed only the extraction prompt.  It reused the v2 regex, parser,
review labels, semantic gate, model, greedy decoder, bridge seed, and saved
student reasoning tokens.  V3 generation is rejected unless a frozen reasoning
root is supplied.  GT boxes entered only the later review cards; they were not
available to model inference or prompt construction.

The frozen gate, applied overall and separately to Attribute and Object, was:

1. parse success at least 95%;
2. faithful extraction at least 90%;
3. answer-value-only target at most 5%.

`none` and `unresolved_consistent` count as faithful extraction.  Student target
correctness versus GT is recorded separately and is not a gate input.

## Development progression

The original 50 trajectories and reasoning token IDs remained unchanged across
all prompt revisions.

| Bridge | Parse | Faithful | Answer-only | Attribute faithful | Object faithful | Gate |
|---|---:|---:|---:|---:|---:|---|
| v2 | 50/50 | 36/50 (72%) | 9/50 (18%) | 13/25 (52%) | 23/25 (92%) | Fail |
| v3p2 | 50/50 | 43/50 (86%) | 1/50 (2%) | 20/25 (80%) | 23/25 (92%) | Fail |
| v3p4 | 50/50 | 46/50 (92%) | 1/50 (2%) | 22/25 (88%) | 24/25 (96%) | Fail |
| **v3p5 frozen** | **50/50** | **47/50 (94%)** | **1/50 (2%)** | **23/25 (92%)** | **24/25 (96%)** | **Pass** |

The explicit rule repaired examples such as `blue -> blue`, `wood -> wood`,
and `rectangular -> rectangular`, producing `the fence that the bird is
perched on`, `the seat on the dark green bench`, and `the cutting board under
the pizza`.  Later prompt revisions added direct-subject examples for glasses,
frosting, and books, plus an explicit plural-group rule for freight cars.

Development prompt tuning stopped at v3p5 before evaluating the new
confirmation cohort.

## Fresh confirmation result

The confirmation cohort contains 20 previously unused train/dev examples,
five from every split/task stratum.  It has zero overlap by sample ID, image ID,
and image SHA-256 with the original 50, the used v2 confirmation 20, and the
sealed independent 48.  Its selection SHA-256 is
`e115e3ac54f9cd02f358a23a0bdb48a558c8b377fb47e1ceddb9d41ffe7e963b`.

Natural reasoning was generated once with the v2 workflow: all 20 trajectories
stopped naturally, and all v3p5 evaluations replayed those exact token IDs.
The v3p5 prompt was not changed after seeing this cohort.

| Panel | Parse | Faithful | Answer-only | Gate |
|---|---:|---:|---:|---|
| All 20 | 20/20 (100%) | 16/20 (80%) | 0/20 (0%) | Fail |
| Attribute 10 | 10/10 (100%) | 8/10 (80%) | 0/10 (0%) | Fail |
| Object 10 | 10/10 (100%) | 8/10 (80%) | 0/10 (0%) | Fail |

Two independent two-stage reviewers first judged the student's target choice
without seeing the extracted fields, then judged bridge fidelity.  A third
reviewer adjudicated four disagreements.  Final extraction errors were two
`reference_mismatches_student_selection` and two `reference_not_unique`.

The four failures were:

1. shirt selected by the student, but the bridge returned the baby wearing it;
2. a full passenger-and-clothing answer sentence rather than one localization
   referent;
3. a selected plastic lid changed to the containing box;
4. `carrots` failed to distinguish an open bag from its contents.

These are direct entity-level and uniqueness errors.  They are not caused by
GT disagreement alone: the extractor changed or underspecified the student's
selection in each case.

## Interpretation

The user's proposed prompt intervention was worthwhile and falsifiable.  It
removed the old answer-value failure on the fresh confirmation cohort
(0/20 answer-only) and improved the original-50 development result from 72% to
94% faithful extraction.  However, development success did not transfer to the
fresh cohort: exact referent fidelity was only 80% in both task types.

The remaining problem is broader than confusing an answer value with an
entity.  A single free-form generation step must simultaneously decide the
question's direct localization subject, preserve the student's selected
instance or group, retain distinguishing relations, and avoid moving between a
part and its owner/container.  Qwen3.5-0.8B did not perform this composition
reliably enough under the frozen threshold.

The next clean experiment should define a new bridge version before touching a
new confirmation cohort.  The most direct option is a two-stage prompt-only
bridge: first extract the queried entity head and relation span from the
question, then bind that span to the instance selected by the frozen reasoning.
The second stage should choose either a supported referring phrase or
`UNRESOLVED`; it should not freely rewrite both the answer and entity in one
continuation.  This changes the operational interface and therefore requires a
new development set and a newly sealed confirmation set.

## Artifacts

- frozen prompt and contract: `mmgcot_timeline_training/bridge_v3.py`
- final development packet SHA-256:
  `97c45487d21e5e97237ebed20e4ccf1bc7e2a24102b4ed0ad3ff51fe62a77c19`
- final development summary:
  `outputs/research_experiments/mmgcot_timeline_training/bridge_v3p5_original50_review_cases/summary_v3.json`
- fresh confirmation packet SHA-256:
  `0166224049737af7af1b58d8b0358ab704569d53e4bfdf47363ffc6d90fee5aa`
- fresh confirmation adjudication SHA-256:
  `0dcc438fa22062d10e5c9e5eed74e1bf31ab77fc2205692357f54f7c2c68457a`
- fresh confirmation summary SHA-256:
  `1115980bbb5a252cfd5bad11d019658c8364f7ba8959732d0d052e22b7ef4d6b`
- integrity audit: `WARN`, same-family provisional; result arithmetic, cohort
  separation, exact replay, GT separation, and quoted hashes passed audit
- sealed independent 48: not evaluated
