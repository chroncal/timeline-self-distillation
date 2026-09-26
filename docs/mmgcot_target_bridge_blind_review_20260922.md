# MM-GCoT target-bridge blind-review audit

Date: 2026-09-22

## Decision addressed

Before OPD calibration or training, this audit asks whether the model-only
late target description names the same image entity as the ground-truth box.
It does not compare R against E and contains no teacher box, IoU, arm identity,
or training result.

## Frozen material

- 50 cases selected before bridge generation: 40 train and 10 dev; balanced
  Attribute/Object within each split.
- Each card showed only the image, original question, generated target
  description, and GT box.
- Allowed labels: `same_target`, `different_target`,
  `description_not_unique`, and `cannot_determine`.
- Reviewers worked independently.  Two disagreements were adjudicated without
  seeing R/E predictions or scores.

Artifacts:

- selection SHA-256: `066e62ccaaa82b897b80e57e8542b4d948301bb6e197726bc9f499775bd322ef`
- bridge packet SHA-256: `b61d088a802ccb0e2c471dab84ec15432f25e7c3c44fe05b986e542e934bce50`
- reviewer A SHA-256: `b5aa06885835a428a31fef5013f9a6df8ec1be4f0717d058765bc0510f0f9f19`
- reviewer B SHA-256: `cdfacaf107e87e7223916e55cf9f9662b4b14d485fe43ee9ecf25c1938143d7a`
- adjudicated SHA-256: `80fca7095d1f2e684c7cdef82e480f55dab6be84fe333988d1b636b6fa7e4fd0`

## Result

The reviewers agreed on 48/50 cases (96.0%).  Cohen's kappa was 0.9267.

| Label | Count | Rate |
|---|---:|---:|
| Same target | 15 | 30% |
| Different target | 29 | 58% |
| Description not unique | 3 | 6% |
| Cannot determine | 3 | 6% |

Breakdown by task:

| Task | Same | Different | Not unique | Cannot determine |
|---|---:|---:|---:|---:|
| Attribute (25) | 0 | 23 | 0 | 2 |
| Object (25) | 15 | 6 | 3 | 1 |

Breakdown by split:

| Split | Same | Different | Not unique | Cannot determine |
|---|---:|---:|---:|---:|
| Train (40) | 11 | 23 | 3 | 3 |
| Dev (10) | 4 | 6 | 0 | 0 |

All three `cannot_determine` cases came from trajectories that reached the
4096-token reasoning cap.  Of the 47 descriptions marked `usable` by the
automatic bridge generator, 29 named a different target, three were not
unique, and 15 named the GT entity.

## Scientific interpretation

The current target-description bridge does not operationalize the proposed
late-semantic-to-early-geometry mechanism.  For Attribute questions it usually
returns the requested property value, such as a material or shape, rather than
the object that owns that property.  The teacher and student could therefore
share answer information while failing to share an entity reference.

Training with this bridge may still measure whether clearing generated history
helps answer-conditioned localization.  It cannot support the stronger claim
that the late branch first identifies an entity and the early/reset branch
then corrects that same entity's geometry.

This audit is not a basis for dropping difficult samples.  Every frozen train,
dev, confirmation, and Test-200 item remains unchanged.  The valid choices are
to freeze a corrected model-only bridge and repeat semantic validation on the
same 50 cases, proceed with a narrower scientific claim, or deliberately adopt
privileged dataset-derived entity supervision and rename the method/claim.

## Run status

No lambda calibration and no formal training arm had started when this result
was recorded.  The independent 48-image confirmation cohort remains sealed.
The prior Test-200 cohort remains unchanged and is labeled a
selection-conditioned, non-independent training-after retest.
