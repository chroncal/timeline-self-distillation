# MM-GCoT target bridge v2 frozen protocol

Date frozen: 2026-09-22, before v2 semantic review and before any OPD
calibration or formal training.

## Scientific contract

The natural student trajectory is immutable.  Bridge v2 replays the saved
reasoning token IDs and serializes two different values:

- `task_answer`: the student's final answer to the original task, such as a
  material, shape, action, attribute, or object category.
- `target_entity_reference`: a noun phrase for the particular visible image
  instance whose bbox should be predicted.

Only `target_entity_reference` may enter L/R/E bbox prompts or later teacher
conditioning.  `task_answer` is audit metadata.  The extractor must follow
the student's final selection even when it is wrong; it may not use the GT
box, reference answer, or dataset CoT to repair the student's choice.

Generation receives only the original image, original question, and exact
saved reasoning token IDs.  GT is copied to the subsequent human-review
packet only.  Each record stores the reasoning-token SHA-256 and source-record
SHA-256.  The original trajectory files remain unchanged.

Frozen prompt and grammar are defined in
`mmgcot_timeline_training/bridge_v2.py`.  The constrained continuation has
fixed `task_answer` and `target_entity` fields.  Coordinates, box edges,
confidence, explanations, and invented identifying details are prohibited.

## Separate error axes

Review is performed in two conceptual stages.  The reviewer first reads the
question and frozen reasoning to judge the student's own selected target,
then compares the extracted entity reference with that reasoning.

`student_target_selection_error`:

- `none`
- `selected_different_entity`
- `selection_not_unique`
- `selection_missing`
- `cannot_determine`

`extractor_serialization_error` is generated mechanically as `none`,
`generation_incomplete`, or `grammar_mismatch`.

`extractor_content_error`:

- `none`
- `answer_value_not_entity`
- `reference_mismatches_student_selection`
- `reference_not_supported_by_reasoning`
- `reference_not_unique`
- `unresolved_consistent`
- `cannot_determine`

The separate GT-facing label remains `same_target`, `different_target`,
`description_not_unique`, or `cannot_determine`.  It describes the student's
semantic correctness and is not used to repair, filter, or reweight a sample.

## Frozen review cohorts

Development re-review uses exactly the original 50 rows:

- selection SHA-256:
  `066e62ccaaa82b897b80e57e8542b4d948301bb6e197726bc9f499775bd322ef`
- the original prompt IDs and reasoning IDs must match the v1 source records
  exactly; no trajectory is regenerated.

External semantic confirmation uses 20 previously untouched train/dev rows,
five from each split/task stratum, selected by stable SHA-256 rank after
excluding the original 50:

- selection path:
  `/mnt/sda/sujingyang/research/datasets/mmgcot_timeline_training_v1/bridge_v2_confirmation20.jsonl`
- selection SHA-256:
  `d83ad79387857d20f86deb7ce14e824eb5ba172dd8ff48cc5b981fad9de94d16`
- no image or sample overlaps the original 50 or independent confirmation 48.

The independent 48-image confirmation cohort remains unopened by inference
until all formal checkpoints and hyperparameters are frozen.

## Semantic pass rule

Apply the same rule separately to the original 50 and untouched 20:

1. valid structured serialization in at least 95% of all rows;
2. `extractor_content_error` is `none` or `unresolved_consistent` in at least
   90% of all rows;
3. `answer_value_not_entity` in at most 5% of all rows.

The same three thresholds must also pass separately for Attribute and Object
within each cohort.  This prevents a strong Object result from masking the
specific Attribute failure that motivated bridge v2.

These thresholds assess extraction fidelity, not whether the student chose
the GT entity.  Student selection errors are reported by split and task and
remain in every downstream all-sample result.  Same-target subsets may be
reported only as explicitly labeled explanatory analyses.

Both cohorts must pass before running the small L/R/E diagnostic.  Failure on
the untouched 20 cannot be repaired by tuning on those 20; it rejects this
frozen bridge version.

## Post-bridge decision chain

If both semantic reviews pass, run a small frozen L0/L/E/R comparison with the
new entity reference and repeat the exact student-coordinate-prefix
continuation comparison for E/L/R.  Do not select examples by GT or previous
teacher performance.  The diagnostic contains 20 rows selected from the
original development-50 before any bridge-v2 geometry was generated, with
five rows in every train/dev by Attribute/Object stratum.  Its selection
SHA-256 is
`5f570e5df8aa76def01d0147a8af690ef22b3427b7bf174430bdbbe6ad9718f8`.

The predeclared screen requires `mean IoU(R)-mean IoU(L) > 0` for independent
Stage-A random boxes and for both fixed student-prefix positions, with at
least 90% complete-image coverage in each prefix panel.  E-L is reported as
the key retained-early-history control but is not used to choose the screen.
This directional small-sample screen determines whether the earlier R-guidance
observation remains operational after changing only the semantic bridge; it
is not a substitute for the later held-out training evaluation.

Only after that diagnostic remains supportive:

1. choose one SFT learning rate on dev;
2. run R and E short calibration for every common
   `lambda_opd` in `{0.1, 0.3, 1.0}`;
3. select lambda by the average of R/E dev image-equal mean IoU, taking the
   smaller lambda when the difference is below 0.002;
4. run bbox-SFT, R-OPD, and E-OPD with the same frozen lambda and formal seeds.
