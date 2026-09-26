# MM-GCoT timeline OPD training protocol v1

Date frozen: 2026-09-22

## Scientific question

Test whether a frozen same-model teacher conditioned on a shorter localization
history can improve a student's final bounding box after full reasoning.  The
student reasoning policy and target-description bridge remain frozen; only a
bbox-gated terminal query adapter is trained.

The three matched training arms are:

1. `bbox_sft`: ground-truth bbox numeric-token NLL.
2. `r_opd`: the same bbox SFT loss plus R-teacher reverse KL.
3. `e_opd`: the same bbox SFT loss plus E-teacher reverse KL.

R and E use the same frozen starting checkpoint, late target description,
student bbox token prefixes, legal grammar support, optimizer schedule,
trainable parameter whitelist, and OPD coefficient.  They differ only in the
reasoning history retained by the teacher: R retains none of the generated
reasoning; E retains the fixed 25% early prefix defined by the diagnostic.

## Evaluation cohorts

### Selection-conditioned diagnostic retest

The existing MM-GCoT Test-200 cohort was used to diagnose L/E/R and directly
informed the decision to use R as the main teacher and E as the key control.
It is therefore retained unchanged and reported only as:

> MM-GCoT Test-200 diagnostic-selection cohort training-after retest
> (selection-conditioned, non-independent).

It must not be used again to choose the teacher, OPD coefficient, learning
rate, checkpoint, sample filter, or training arm.  No sample may be removed or
replaced based on its previous or post-training result.

Frozen manifest:

- path: `/mnt/sda/sujingyang/research/datasets/mmgcot_20260920_frozen_final_v1/formal_frozen_final.jsonl`
- SHA-256: `f4975242439e3ee2e9f5c4a4ad8454a3a527db4c2cdd8ceb6168945e14eb8baf`
- size: 200 images, 100 Attribute and 100 Object
- existing diagnostic: 3 trajectories per image, 600/600 completed journals

### Independent confirmation cohort

Before any training starts, freeze a balanced 48-image cohort from official
MM-GCoT Test rows that passed the pre-existing eligibility review but were
not selected for Test-200 and never entered L/E/R diagnostic inference.
Use all 24 eligible Object rows and the first 24 of 28 eligible Attribute rows
under stable SHA-256 ordering with seed `20260921`.  The cohort must be
image-, sample-, and source-row-disjoint from train, dev, pilot-30, and
Test-200.  Run it once only after all hyperparameters and checkpoints are
frozen.

The independent confirmation result is the primary evidence for a held-out
generalization claim.  Test-200 remains useful as a matched retest of the
originally observed phenomenon.

Frozen data package:

- root: `/mnt/sda/sujingyang/research/datasets/mmgcot_timeline_training_v1`
- train SHA-256: `3f4643b2899bb5cd8b8a423b1932c1245f251811b580743b4c8d82f60d650a20`
- dev SHA-256: `215c5f1ab5c3ba14ca47a94cd16dda365f90c4bf75490fe584f0b041ce44749b`
- independent confirmation SHA-256: `96e4e8a9c6113e5587b871a085dde251d97b4184e9bda44643cdbe2f702d7847`
- blind-review selection SHA-256: `066e62ccaaa82b897b80e57e8542b4d948301bb6e197726bc9f499775bd322ef`
- package manifest SHA-256: `4670dcf888599776d2436e4006959e885b5cfd376872566add851cb76f31db34`

## Data and semantic review

Freeze image-disjoint Trainval partitions with seed `20260921`: train 310
(155/155 by task), dev 60 (30/30), and five unused reserve images, with one
question per selected image.  The earlier 320-image target is infeasible after
strict historical and confirmation exclusions leave 375 unique candidates;
images are not reused to fill the five-image gap.  Before ranking
candidates, exclude the strict union of every image previously used by a
timeline diagnostic: pilot-30, Test-200, and the seven images used by an
earlier pilot run but absent from the final pilot manifest.  Also exclude all
48 independent-confirmation image IDs from Trainval, including every Trainval
row that shares one of those images.

Before training, select 50 examples without reading any teacher prediction,
IoU, or training result: train 40 (20/20 by task) and dev 10 (5/5).  After the
frozen natural trajectory and target description are generated, provide each
reviewer only the image, original question, target description, and GT target
box.  Allowed labels are exactly:

- `same_target`
- `different_target`
- `description_not_unique`
- `cannot_determine`

Review records must not contain R/E boxes, IoU, teacher probabilities, arm
names, or training metrics.  The review characterizes semantic validity; it
does not gate, replace, or reweight training examples and does not change the
primary metric.

### Pre-training blind-review result (2026-09-22)

The frozen 50-case review was completed before calibration or formal
training.  Two independent reviewers agreed on 48/50 cases (96.0%; Cohen's
kappa 0.9267).  Blind adjudication produced:

- `same_target`: 15/50 (30%)
- `different_target`: 29/50 (58%)
- `description_not_unique`: 3/50 (6%)
- `cannot_determine`: 3/50 (6%)

The failure is task-dependent.  Attribute has 0/25 `same_target` cases
(23 `different_target`, 2 `cannot_determine`); Object has 15/25
`same_target` cases (6 `different_target`, 3 `description_not_unique`, 1
`cannot_determine`).  All three `cannot_determine` cases hit the 4096-token
reasoning limit.  Among the 47 descriptions marked usable by the automatic
generator, only 15 referred to the boxed entity.

Inspection of the blind reasons shows that the current extraction prompt
usually emits the answer value for Attribute questions (for example, a
material or shape) instead of a referring expression for the entity whose
box should be predicted.  This is a semantic interface failure: under the
current bridge, R/E would often be conditioned on a property answer rather
than a shared late-resolved target entity.

Formal training and lambda calibration therefore remain unlaunched pending a
frozen decision about the bridge definition.  Proceeding unchanged would
test answer-conditioned context reset, not the preregistered claim that late
target recognition is followed by early/reset geometric correction.  No
sample has been deleted, replaced, reweighted, or selected from this review.

Evidence:

- bridge packet SHA-256: `b61d088a802ccb0e2c471dab84ec15432f25e7c3c44fe05b986e542e934bce50`
- adjudicated review SHA-256: `80fca7095d1f2e684c7cdef82e480f55dab6be84fe333988d1b636b6fa7e4fd0`
- review summary SHA-256: `dd39abb58c1d5eb71a94113f50b98884c620a7d248ddf2e63f540a37b99ad242`
- detailed report: `docs/mmgcot_target_bridge_blind_review_20260922.md`

## Calibration and training

Select the SFT learning rate on dev from `{1e-4, 3e-4, 1e-3}`.  At that single
fixed learning rate, run both R and E short calibrations for every
`lambda_opd` in `{0.1, 0.3, 1.0}`.  For each lambda, average the R and E dev
image-equal mean IoU values.  Select the lambda with the highest average;
when averages differ by less than `0.002`, select the smaller lambda.  Freeze
that same lambda for both formal OPD arms.

Calibration changes neither the three formal training arms nor the final
comparison.  Calibration may use dev only.  The independent confirmation
cohort and Test-200 may not be evaluated until all formal checkpoints are
fixed.

Formal seeds are `20260921`, `20260922`, and `20260923`.  All arms share the
same frozen reasoning/target-description records and per-seed sample order.
Report bbox geometry, validity, failures, and paired image-level uncertainty;
KL or NLL improvement alone is not evidence of localization improvement.

## Claim boundary

An R-OPD improvement over bbox-SFT on the independent confirmation cohort
supports transfer of reset-context localization guidance.  E-OPD versus
R-OPD distinguishes retained early history from complete generated-history
removal.  Neither result by itself proves that text attention diluted visual
representations.  If semantic review is incomplete, use the narrower phrase
"target-description-conditioned localization transfer" and do not claim
same-entity boundary-drift correction.
