# MM-GCoT target bridge v2 semantic-validation result

Date: 2026-09-22

## Decision

Target bridge v2 **failed the frozen semantic gate**.  The small L0/L/E/R
geometry diagnostic, common-lambda calibration, and three formal training arms
were not started.  The independent 48-image confirmation cohort remains
sealed.

This is a semantic-interface result, not a localization-teacher result.  No
R/E bbox, IoU, teacher distribution, checkpoint, optimizer, or parameter
update was produced in this stage.

## What changed

Bridge v2 separates:

- `task_answer`: the answer value or category concluded by the student;
- `target_entity_reference`: the visible instance whose bbox should be
  predicted.

Only the second field was eligible for later bbox conditioning.  For the
original 50 cases, v2 replayed the exact v1 prompt and reasoning token IDs;
all 50 prompt arrays, reasoning arrays, token hashes, and source-record hashes
matched.  The untouched confirmation cohort contained 20 newly sampled
train/dev cases, five from every split/task stratum, with no sample or image
overlap with the original 50 or independent 48.

Both cohorts had 100% grammar/parse completion.  GT boxes were included only
in anonymous review cards after bridge generation; they were not passed to
the extraction model.

## Blind review

Two independent reviewers first judged the student's frozen target selection
from the image, question, GT box, and reasoning.  Only after recording that
decision did they inspect the two extracted fields.  A third reviewer
adjudicated the 15/70 cases with at least one disagreement.

Raw agreement was:

| Field | Agreement |
|---|---:|
| Student target-selection error | 61/70 (87.1%) |
| Extractor content error | 64/70 (91.4%) |
| Extracted reference versus GT | 64/70 (91.4%) |

## Frozen-gate result

The gate required, both overall and separately for Attribute/Object:

1. at least 95% valid structured serialization;
2. at least 90% faithful extraction of the student's own target selection;
3. at most 5% answer-value-only entity references.

| Cohort | Parse | Faithful extraction | Answer-only entity | Gate |
|---|---:|---:|---:|---|
| Original development 50 | 50/50 (100%) | 36/50 (72%) | 9/50 (18%) | Fail |
| Untouched confirmation 20 | 20/20 (100%) | 14/20 (70%) | 4/20 (20%) | Fail |

By task:

| Cohort/task | Faithful extraction | Answer-only entity | Task gate |
|---|---:|---:|---|
| Original 50 / Attribute | 13/25 (52%) | 9/25 (36%) | Fail |
| Original 50 / Object | 23/25 (92%) | 0/25 | Pass |
| Confirmation 20 / Attribute | 6/10 (60%) | 4/10 (40%) | Fail |
| Confirmation 20 / Object | 8/10 (80%) | 0/10 | Fail |

The format constraint therefore solved syntax but did not reliably enforce
the semantic distinction.  Typical remaining Attribute failures were
`task_answer=blue, target_entity_reference=blue` and
`task_answer=stainless steel, target_entity_reference=stainless steel`.
Other failures included under-specified references such as `bedspread` or
`cupcake`, and two malformed but grammar-legal references containing only
punctuation.

## Student error versus extractor error

The two error sources were kept separate; GT was never used to rewrite the
student's target.

| Cohort | Reviewed as matching extracted GT box | Different entity | Non-unique selection |
|---|---:|---:|---:|
| Original 50 | 34/50 (68%) | 12/50 (24%) | 4/50 (8%) |
| Confirmation 20 | 10/20 (50%) | 6/20 (30%) | 4/20 (20%) |

Even when the student selection was judged correct, the extractor was
faithful in only 26/34 original cases and 7/10 confirmation cases.  Conversely,
a faithful extractor was not treated as wrong merely because it preserved a
student mistake.  Among faithfully extracted references, 26/36 original and
7/14 confirmation references pointed to the GT entity.

This separates two limitations:

1. **Extractor limitation:** v2 still substitutes an answer value for the
   entity too often, predominantly on Attribute.
2. **Premise/annotation limitation:** the frozen late reasoning was reviewed as
   uniquely selecting the extracted benchmark box in only 68% of the
   development cohort and 50% of the untouched cohort.  A later integrity
   audit found that train/dev boxes are taken from the final CoT box, which can
   refer to a different object than the apparent question target.  These rates
   therefore mix student target-selection failures with annotation-semantics
   failures and cannot be reported as pure student error.  Either source still
   prevents a clean same-entity geometric-correction interpretation.

The original-50 GT-facing same-target rate increased descriptively from
15/50 under v1 to 26/50 under v2, but this improvement is insufficient and is
not a pass under the frozen criterion.

## Consequence for the research plan

Running L/R/E now would mix geometric history effects with unresolved target
selection and extraction failures.  Running OPD would then train on a teacher
condition that frequently does not identify a unique student-selected entity.
Accordingly, geometry diagnostics and training remain blocked for this bridge
version.  No sample was dropped, replaced, repaired from GT, or reweighted.

A future v3 would require a new development protocol and a new untouched
semantic confirmation cohort.  The used 20-case confirmation cohort cannot be
reused as an unseen tuning target.  Plausible v3 directions include extracting
an entity span from the student's reasoning instead of freely generating a
noun phrase, or deriving a referring expression from the question while
keeping the student's final answer in a separate audit field.  Either changes
the operational definition and must be validated before geometry is run.

Before v3, the train/dev target annotation must also be re-audited against the
question semantics rather than assumed correct because it is the last CoT box.
The audit found a provenance metadata defect in which `source_row_index`
pointed at the preceding answer-only row although the source ID, question and
box came from the following CoT row.  The adapter code is corrected for future
freezes and covered by a regression test; frozen v2 artifacts were not edited.

The integrity audit's overall verdict is **WARN**, while the bridge acceptance
decision remains **FAIL**.  No GT leakage into bridge generation was found,
all reported numbers reproduced exactly, and both reviewers independently
reject both cohorts.  Warnings concern CoT-derived GT interpretation, a
permissive ambiguity rubric, one omitted packet field recovered mechanically,
four length-capped trajectories, and same-family LLM review provenance.

## Artifacts

- semantic summary SHA-256:
  `fa5432b6b6cde88fcba904cf745745b668961a21efbbb95f05d445cee088f7c9`
- adjudicated 70-case review SHA-256:
  `c65813c92526e9e80dafc9c46ca63f041ca581be0cc7e42004c2c005a25ceb26`
- original-50 packet SHA-256:
  `46c85dc23919958f7938116091efb16fc9046e39fb0943c2fd1ab127050e8975`
- confirmation-20 packet SHA-256:
  `62deee3118e41e83ebe72c3ce0ae0d7ba9445794591e6287ae8db5445d8afb9f`
- failure-case audit SHA-256:
  `31b5f6ffc49dfea9b3979fe631dceb40a304ef342a1f5eb7d5ff9248dc592a1f`
- integrity audit:
  `outputs/research_experiments/mmgcot_timeline_training/bridge_v2_reviews_combined/EXPERIMENT_AUDIT.md`
