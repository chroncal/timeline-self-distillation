# MM-GCoT formal bbox training v2

The experiment compares `bbox_sft`, `r_opd`, and `e_opd` at a shared optimizer
schedule. The scientific endpoint is whether R/E guidance improves image-equal
mean IoU beyond the same ground-truth bbox supervision on the student's full
reasoning and identical late target-description context.

This version supersedes the v1 semantic stop rule. The bridge is fixed to the
original v3p5 prompt and parser. Parse-valid wrong or ambiguous descriptions
are retained unchanged in every arm. Empty, unresolved, or incomplete bridge
output uses the no-entity L suffix for SFT and receives zero OPD loss. GT is
never used to repair a bridge. The 50-case review remains a report-only
description of noise; it does not filter, weight, or select examples.

The student and teachers share the frozen Qwen3.5-0.8B backbone. The only
trainable parameters are the rank-8 final full-attention query adapter's
24,576 FP32 weights. The teacher disables the adapter. `L` retains all saved
natural reasoning tokens, `R` retains none, and `E` retains the diagnostic's
fixed quarter-length early prefix. All three receive the same extracted
target description and bbox suffix. The adapter is enabled before the last
opening token is consumed, so its output predicts the first coordinate digit.

The reference box is quantized by `floor(1000*x+0.5)` for GT teacher-forced
numeric token NLL. In R/E arms, the current student additionally samples one
grammar-constrained bbox on each visit. Its original token IDs are reused as
both teacher and student coordinate prefixes; the numeric positions receive
full-grammar-support `KL(student || teacher)` at temperature one. SFT uses a
separate GT cache branch. Per-example losses are divided by their numeric
position counts, then by the full effective batch of 16, including examples
without an OPD term.

All immutable scientific settings and data hashes are in
`configs/mmgcot_formal_v2.json`. Train has 310 images and dev 60; neither is
modified. The 48-image independent confirmation cohort remains sealed until
the learning rate, common lambda, and all nine final step-200 checkpoints are
fixed. The historical Test-200 is a selection-conditioned retest only.
The formal runner uses deterministic CUDA algorithms with TF32 disabled and
sets `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`. A cached one-example OPD step
measured 8.9 s with four CPU threads versus 11.7 s at the host default;
the resulting one-step adapter weights were bit-identical.

Execution sequence:

1. Generate one shared frozen trajectory and v3p5 bridge per train/dev image;
   verify manifest hashes and completion receipts.
2. Run real-model boundary, cache, gradient, and one-update acceptance checks;
   then make a fixed small L/R/E and shared-student-prefix readout.
3. Train three SFT learning-rate candidates for 60 steps on seed 20260921;
   select dev image-equal random mean IoU with the absolute 0.002 near-tie rule.
4. Train R and E at each lambda 0.1, 0.3, 1.0 for 60 steps at the shared
   learning rate. Select one lambda by average R/E dev mean IoU, again using
   the 0.002 near-tie rule and preferring the smaller value.
5. Start nine new runs (three arms, seeds 20260921–20260923), each from the
   same zero-increment adapter initialization and each fixed at step 200.
6. After freezing checkpoints, generate shared evaluation trajectories and
   evaluate the independent 48 once, then report the Test-200 retest.
7. Generate the result Markdown and paired image-difference PNG from the
   immutable final summaries, preserving the independent/retest distinction.

Primary contrast: `mIoU(R-OPD) - mIoU(bbox-SFT)`. Secondary contrasts:
`E-OPD - bbox-SFT` and `R-OPD - E-OPD`. The random four bbox draws are averaged
within trajectory, three trajectories within image, then images equally.
Invalid or unfinished boxes score zero and remain in the full-queue
denominator. Report per-seed values and paired image-bootstrap intervals;
do not interpret lower KL alone as geometric improvement.

A positive R-SFT difference on the independent cohort supports transfer of
R-conditioned coordinate guidance beyond bbox supervision under this adapter
and noisy bridge. A non-positive difference does not establish that all OPD
methods fail. E-R distinguishes retaining a fixed early prefix from clearing
generated history under the same loss and data. None of these contrasts by
itself proves an attention-dilution mechanism. The Test-200 retest is
descriptive because the earlier diagnosis informed the choice of R/E.
