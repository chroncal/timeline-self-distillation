# MM-GCoT pure-OPD ablation, frozen before launch

## Question

Does removing the ground-truth bbox SFT term let R/E teacher guidance improve
late-context bbox localization relative to both the unadapted Base-L and the
original SFT+OPD runs? This ablation was requested after the one-seed dev
readout showed Base-L > SFT+OPD > bbox-SFT. It is an added experiment; it does
not alter or replace the original nine-run protocol or its results.

## Controlled comparison

For each of R and E, initialize the same rank-8, last-layer Query-only
adapter with zero output on the original frozen Qwen3.5-0.8B checkpoint. Do
not initialize from any SFT or mixed-OPD checkpoint. Freeze natural reasoning,
target extraction, base model and teacher. Reuse the original train/dev/test
manifests, frozen trajectories, v3p5 target descriptions, L/R/E contexts,
grammar, student rollout seeds, optimizer, LR `1e-4`, OPD coefficient `0.1`,
effective batch `16`, fixed `200` steps, and seeds `20260921`, `20260922`,
`20260923`. The only planned training-objective change from each corresponding
mixed arm is setting the GT bbox SFT coefficient from `1` to `0`.

At each visit the current L-context student samples a fresh bbox. The frozen
R or E teacher scores exactly those raw coordinate-prefix token IDs; only
numeric positions contribute full-grammar-support reverse KL. The pure arms
never call the GT bbox SFT path and do not read GT coordinates during training.
Samples without a usable target description have zero OPD contribution and
remain in the effective-batch denominator. Geometrically invalid sampled
boxes are logged and still enter KL, matching the original OPD behavior.

The inherited LR and coefficient are held fixed to isolate removal of SFT;
they are not claimed optimal for pure OPD. AdamW is approximately invariant
to scaling a single loss term, but clipping and optimizer epsilon mean this
is not mathematically exact. No calibration or checkpoint selection is done
for the added arms.

## Endpoints and interpretation

The fixed step-200 checkpoints are evaluated in L context with the same
four-draw, image-equal mean IoU protocol. Compare each pure arm to its mixed
counterpart, bbox-SFT and Base-L on matched frames. Report three seed results
and paired image uncertainty; the already inspected dev set is exploratory.
The sealed independent-48 cohort and original Test-200 retest are evaluated
after all six new checkpoints are frozen, using the original prepared
trajectories. Test-200 remains an existing diagnostic-set retest.

- Pure OPD > mixed OPD and Base-L: evidence that the SFT term was blocking
  transfer in this adapter/teacher setting.
- Pure OPD > mixed OPD but < Base-L: removing SFT helps, yet teacher guidance
  alone does not surpass the original model.
- Pure OPD <= mixed OPD: SFT supplies useful signal despite its observed
  stand-alone degradation, or pure OPD is underoptimized at inherited settings.
- R/E differ: compare teacher conditions, without treating the result as proof
  of attention-based visual dilution.

Output root: `outputs/research_experiments/mmgcot_timeline_training/pure_opd_v2/`.
The original experiment's source file, caches, outputs and checkpoints remain
unchanged. The additional trainer and scheduler run in tmux; interrupted jobs
resume from the latest saved checkpoint with the same sample order and RNG.

The earlier `pure_opd_v1` startup attempt was stopped after its first steps:
torch/xgrammar were imported before GPU masking, so four nominal GPU lanes
mapped to GPU 0. No result or checkpoint from that attempt enters this run.
