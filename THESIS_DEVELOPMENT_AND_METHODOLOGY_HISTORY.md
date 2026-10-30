# Thesis Development and Methodology History

This document records how the ARTIFACT watermarking study developed from the
initial classical/CNN comparison into the formally frozen and published final
benchmark. It is intended to support thesis writing, adviser review, and
reproducibility. The numerical result files remain the authoritative source for
individual measurements; this document explains the decisions that produced the
final configuration.

## 1. Research objective

The study asks whether a CNN can replace only the bit-decision step of a
classical DWT-QIM watermark extractor and reduce raw bit error rate (BER) after
social-media-like degradation. The embedding operation, image representation,
attack condition, and selected coefficients are shared between the classical
and CNN branches. This isolates the extraction decision mechanism as the main
independent variable.

The study is not a claim that a CNN can solve every image transformation. It is
a controlled comparison of:

1. analytical binary-QIM decisions; and
2. learned bit decisions from the same degraded transform-domain evidence.

The primary outcome is raw 128-bit BER. SSIM and PSNR measure visual fidelity;
timing, memory, and CPU measurements describe practical feasibility.

## 2. Initial paper-aligned design

The planned pipeline was:

1. curate non-photorealistic digital illustrations;
2. standardize each image to 512×512 RGB;
3. convert RGB to YCbCr and process the luminance channel;
4. apply a two-level Haar DWT;
5. embed a binary payload into LH2 and HL2 using QIM;
6. apply one controlled degradation per trial;
7. compare classical extraction with a CNN bit decoder.

The planned attack suite was JPEG compression, resizing, cropping, and repeated
re-encoding. Error-correcting codes were excluded from the core comparison so
that BER would measure raw bit recovery rather than post-correction success.

The original paper draft described a creator-linked token encoded through UTF-8
and converted to 128 bits. During controlled implementation, the final
benchmark instead adopted a fixed balanced experimental payload. This distinction
must be stated accurately in the final thesis: UTF-8 is an optional application
payload encoding, not what the published confirmatory benchmark evaluated.

## 3. Data preparation

The dataset pipeline and manifests implement the following preparation steps:

- metadata-based candidate filtering;
- conversion to RGB;
- alpha compositing onto a solid background when transparency is present;
- center crop to a square;
- Lanczos resize to 512×512;
- conversion to contiguous uint8 RGB and lossless PNG storage;
- exact duplicate removal using MD5;
- near-duplicate removal using perceptual-hash distance;
- artist-disjoint train/validation/test assignment.

The final held-out split contains 500 distinct test images. The test identity
manifest published with the final results contains only opaque test IDs, image
indices, and curated-image SHA-256 values; original filenames were removed for
publication minimization.

Runtime image preprocessing is separate from attack simulation. Before
embedding or extraction, RGB is converted to YCbCr, the Y luminance channel is
selected, and the two-level Haar DWT is computed with symmetric boundary
handling. The CNN receives the selected LH2 and HL2 coefficient maps as a
128×128×2 float32 tensor. No ImageNet-style mean/std normalization is applied.

## 4. Baseline and diagnostic progression

### 4.1 Classical sanity baseline

The first fixture benchmark verified that the classical DWT-QIM round trip was
functional. Clean extraction was exact, while JPEG, severe resize, crop, and
some line-art cases produced errors. This established a working baseline but was
not treated as final thesis evidence because the fixture set was small and not
the final curated split.

### 4.2 Initial full-map CNN (Run 1)

The first CNN consumed the complete 128×128×2 LH2/HL2 map and used convolution,
pooling, flattening, and a 128-bit sigmoid output. On 32 controlled clean
examples, classical extraction was exact, but the CNN did not reach perfect
recovery after 500 epochs (mean BER approximately 0.322).

**Decision:** investigate whether the problem was weak signal, an implementation
round-trip issue, or a representation mismatch before changing the model.

### 4.3 Clean round-trip diagnosis

The diagnostic separated three points in the pipeline:

- QIM-domain extraction;
- extraction after inverse DWT but before clipping;
- extraction after reconstructed values were clipped/converted back to image
  values.

The first two were exact. Errors appeared after clipping for highly saturated
images.

**Decision:** retain those difficult images and document clipping as a separate
reconstruction limitation. Removing them would have hidden a real failure mode.

### 4.4 Signal localization diagnosis

The selected coefficients carried the payload-dependent change (approximately
Δ/2), while non-selected coefficients did not show the same signal. The
full-map CNN therefore had to discover sparse, seed-selected positions before
making bit decisions. Pooling also discarded selected locations: direct
selected-coefficient survival was only about 38.5% with 2×2 pooling and 20.1%
with 4×4 pooling.

**Decision:** preserve the classical location generator and expose the selected
coefficients to a smaller decoder. This changed the representation, not the
embedding rule.

## 5. Seed-aware decoder redesign

For each of the 128 selected coefficients, the decoder receives:

1. coefficient divided by Δ;
2. sin(π·coefficient/Δ);
3. cos(π·coefficient/Δ);
4. a subband identifier (LH2=0, HL2=1).

The resulting primary decoder is a 369-parameter pointwise Conv1D model:

`Conv1D(16, kernel=1, ReLU) → Conv1D(16, kernel=1, ReLU) → Conv1D(1, kernel=1, sigmoid)`.

Pooling, flattening, dense layers, and dropout were removed because the task is
aligned bitwise decision-making, not spatial image classification.

The coefficient-selection seed was fixed at **42**. No systematic seed search
was performed; 42 is a conventional deterministic reproducibility value, not an
optimized or cryptographic choice. The separate final payload-generation seed
is **20260901**.

### Why the final decoder has exactly 369 parameters

The parameter count follows directly from the representation and the required
bitwise alignment; it was not selected because 369 is a special number. Each of
the 128 selected coefficients is represented by four features, so the input is
`128 × 4`. The decoder applies the same pointwise feature transformation at each
payload position:

```text
Conv1D(16 filters, kernel=1, input channels=4)
Conv1D(16 filters, kernel=1, input channels=16)
Conv1D(1 filter,  kernel=1, input channels=16)
```

The exact count is:

```text
First layer:  (1 × 4 × 16) + 16 biases = 80
Second layer: (1 × 16 × 16) + 16 biases = 272
Output layer: (1 × 16 × 1) + 1 bias = 17
Total: 80 + 272 + 17 = 369 parameters
```

The `kernel=1` choice is important. It mixes the four features belonging to
one selected coefficient but does not mix neighboring coefficient positions.
Consequently, payload position 0 remains associated with output bit 0, and so
on through position 127. Pooling, flattening, and spatial kernels were excluded
because earlier diagnostics showed that they could destroy the sparse seeded
alignment. The 16-channel width is a deliberately compact baseline capacity,
not the result of a systematic architecture search. The model is therefore a
controlled learned replacement for the QIM bit decision, rather than a claim of
an optimized neural architecture.

## 6. Decision basis for the complete methodology

The following table records the justification for each major choice. This is the
chain of reasoning that should be used when defending the methodology in the
thesis.

| Choice | Basis and justification | Decision type |
|---|---|---|
| 512×512 input | Provides one fixed image geometry for the dataset and level-2 DWT; avoids variable tensor shapes. | Dataset/pipeline requirement |
| RGB→YCbCr and Y channel | Embedding in luminance preserves the chroma channels for reconstruction and follows the transform-domain design. | Paper-aligned design |
| Two-level Haar DWT | Gives an interpretable multiresolution representation with simple, reproducible implementation. | Paper-aligned baseline |
| LH2 and HL2 | Mid-frequency detail bands provide a practical imperceptibility/robustness compromise; HH was considered more vulnerable. | Paper-aligned design |
| Binary QIM | Produces an interpretable one-bit decision per selected coefficient and supports direct BER measurement. | Method requirement |
| 128 bits | Fixes the decoder output dimension and provides a sufficiently large raw bitstream for BER comparison. | Controlled design |
| Balanced payload | 64 zeros and 64 ones prevent a decoder from exploiting class imbalance and make the 50% random-guess reference meaningful. | Statistical control |
| Payload seed 20260901 | Makes the balanced payload exactly reproducible across runs and all images. | Reproducibility |
| Coefficient seed 42 | Regenerates identical embedding and extraction locations; no seed search or performance optimization was performed. | Reproducibility |
| Initial full-map CNN | Direct implementation of the original baseline hypothesis: learn decisions from the complete LH2/HL2 maps. | Baseline test |
| Selected-coefficient representation | Localization showed that the payload signal was concentrated at selected coefficients and pooling discarded many of them. | Diagnostic-driven change |
| Four coefficient features | Normalized coefficient value, sine, cosine, and subband ID expose QIM phase/scale and band identity to the bit decoder. | Representation design |
| Pointwise `kernel=1` layers | Preserve one-to-one coefficient/bit alignment and avoid mixing unrelated seeded positions. | Alignment control |
| 16 hidden channels | Supplies a small feature-mixing capacity while limiting unnecessary memorization; not architecture-search optimum. | Compact baseline |
| 369 total parameters | Follows mechanically from the selected three-layer architecture: 80 + 272 + 17. | Architectural consequence |
| No pooling/flattening/dropout | Pooling damaged sparse location access; flattening would discard the explicit positional structure; dropout was unnecessary for the controlled bitwise decoder. | Diagnostic-driven change |
| Sigmoid output | Produces one independent probability for each of the 128 binary bits. | Output requirement |
| Binary cross-entropy | Matches independent binary target bits and sigmoid probabilities. | Loss choice |
| Seed-aware decoder retained | Run 2 showed clean memorization, Run 3A showed transfer, and Run 3B demonstrated strong clean validation generalization. | Evidence-based selection |
| Attack exposure expansion | Stage 2 and Stage 2B tested whether limited JPEG/re-encoding exposure explained the robustness gap. | Hypothesis test |
| 3×3 local-context ablation | Tested whether neighboring same-subband evidence could resolve JPEG ambiguity without changing the primary representation. | Diagnostic ablation |
| Δ values 8/16/24/32 | Controlled validation-stage calibration measured the robustness/fidelity trade-off. | Calibration experiment |
| Δ=24 | Selected as the smallest delta meeting the predefined meaningful robustness-gain rule while retaining high fidelity; Δ=32 was not chosen solely for lower BER. | Frozen selection |
| No ECC/checksum | Keeps the measured BER as raw extraction performance rather than post-correction success. | Measurement control |
| JPEG, resize, crop, re-encoding | Represents the social-media-like transformations specified by the research question. | Attack design |
| Crop not retrained around after Stage 5A | Diagnostics showed shape-dependent coefficient permutation/synchronization loss, not merely a bit-classification weakness. | Failure interpretation |
| 500-image held-out set | Preserves an untouched confirmatory split for final paired evaluation. | Evaluation design |
| Paired comparisons | Both extractors receive the same attacked image, payload, locations, and condition. | Fairness control |
| Wilcoxon signed-rank test | Handles paired, bounded BER observations without requiring normal differences. | Statistical method |
| Holm adjustment | Controls family-wise error across multiple condition comparisons. | Multiple-testing control |

This classification is important: a “decision type” of baseline or
reproducibility does not mean the choice is globally optimal. It means the value
was fixed so that the experiment would be interpretable and repeatable. Only
choices explicitly marked as calibration or diagnostic selection were changed in
response to validation evidence, and the held-out test set was not used for
those decisions.

## 7. Controlled experiment sequence and decisions

### Run 2 — clean memorization

The seed-aware decoder was trained on 32 clean examples from four source images.
It reached BER 0 and 32/32 perfect payloads, with the first stable zero-BER
region around epochs 102–106.

**Conclusion:** the clean QIM bit-decision task is learnable when coefficient
alignment is explicit.

### Run 3A — zero-training transfer

The Run 2 model was evaluated without retraining on 100 validation images and
200 unseen image-payload examples. Mean BER was approximately 0.0695. Clean
transfer was strong for LH2 but substantially weaker for HL2.

**Conclusion:** the representation transfers, but the small training diversity
was insufficient for reliable cross-image generalization.

### Run 3B — clean held-out validation training

The unchanged 369-parameter decoder was trained with 500 training images and
validated on 100 separate images using 200 fixed validation examples.

- CNN validation BER: 0.000195
- Perfect payloads: 195/200
- Classical validation BER: 0.127813

**Decision:** clean generalization passed sufficiently to begin attack-aware
development.

### Stage 2 — JPEG/re-encoding exposure

Training exposure was expanded to clean, JPEG90, JPEG70, JPEG50, and one-, two-,
and three-pass Q85 re-encoding. The CNN reduced the six-attack macro BER from
approximately 0.2770 for classical extraction to 0.2401, a partial improvement.

**Decision:** expose the model to the complete attack-factorial design before
concluding that insufficient attack exposure was the bottleneck.

### Stage 2B — full-factorial exposure

All seven conditions were included for the training examples. The attack macro
BER changed only from approximately 0.2401 to 0.2385.

**Conclusion:** multiplying attack exposure produced little improvement in
moderate/severe JPEG recovery. More exposure alone was unlikely to solve the
main limitation.

### Stage 2C — non-training separability diagnostic

Single-coefficient empirical lookup and k-nearest-neighbor diagnostics were run
without CNN training. At JPEG70 and JPEG50, their BERs were close to the trained
CNN, and phase overlap increased strongly at JPEG50.

**Conclusion:** the available single-coefficient evidence was intrinsically
ambiguous at the tested delta. This was a practical separability diagnostic, not
a claim of an absolute information-theoretic limit.

### Stage 2D — local 3×3 context

The representation was changed to a same-subband 3×3 neighborhood with a shared
601-parameter decoder. Re-encoding improved, but JPEG70 and JPEG50 changed only
slightly, while clean BER worsened.

**Decision:** retain the simpler 369-parameter seed-aware decoder as primary;
keep the local model as an ablation.

### Stage 3A — QIM delta calibration

Values Δ=8, 16, 24, and 32 were tested using the same train/validation images,
payloads, coefficient seed, attacks, and decoder design. Robustness increased
with delta, while fidelity decreased:

| Δ | Six-attack CNN macro BER | Mean PSNR | Mean SSIM |
|---:|---:|---:|---:|
| 8 | 0.409030 | 56.429 dB | 0.999238 |
| 16 | 0.238236 | 56.130 dB | 0.999161 |
| 24 | 0.129232 | 55.701 dB | 0.999033 |
| 32 | 0.088359 | 55.192 dB | 0.998860 |

**Decision:** Δ=24 was selected by the predefined robustness/fidelity rule as
the smallest setting with a meaningful robustness gain while retaining very high
fidelity. Δ=32 was not selected merely because it minimized BER.

### Stage 4A and 4A.1 — resize robustness

Zero-shot resize evaluation showed meaningful transfer at Resize75 and Resize50,
but near-random performance at Resize25. The non-training alignment diagnostic
found locally surviving evidence without a useful fixed global offset.

**Conclusion:** severe Resize25 failure was primarily a spatial alignment issue,
not simply a weak bit classifier.

### Stage 4B — resize-aware local-context ablation

A controlled 3×3 resize-aware decoder was tested. It modestly improved moderate
resize but did not recover Resize25 and used a more specialized representation.

**Decision:** do not replace the primary decoder; record Stage 4B as a failed
severe-resize hypothesis/ablation.

### Stage 5A — crop robustness and synchronization diagnosis

Crop conditions produced near-random classical and CNN BER at all severities.
The crop changes image dimensions, which changes the shape-dependent seeded
coefficient permutation. The extractor consequently reads locations that no
longer correspond to the embedded positions.

**Decision:** do not apply ordinary crop-aware bit-decision training to this
failure. The dominant issue is synchronization, so crop synchronization is a
future architectural extension rather than a post hoc final-model adjustment.

## 8. Formal final freeze

Development was closed before test access. The formal freeze fixed:

- 512×512 RGB input;
- RGB→YCbCr and Y-channel processing;
- two-level Haar DWT with symmetric mode;
- LH2 and HL2 embedding subbands;
- 64 selected locations per subband;
- coefficient-selection seed 42;
- raw 128-bit payload with no ECC;
- Δ=24;
- the 369-parameter seed-aware CNN and checkpoint;
- the complete 13-condition suite;
- paired per-image comparisons;
- Wilcoxon signed-rank tests with Holm adjustment;
- rank-biserial effect sizes;
- timing, memory, and CPU measurement boundaries;
- the no-tuning/no-test-feedback policy.

### Final payload clarification

The final payload was generated before test access by concatenating 64 zero bits
and 64 one bits, then applying `numpy.random.default_rng(20260901).permutation`.
It is balanced, deterministic, and identical across all 500 test images and all
conditions. It was not derived from an image, BER, prediction, filename, or
creator identifier.

This balanced payload was chosen to prevent bit-class imbalance and make raw BER
interpretable. UTF-8 creator-text encoding remains an optional future/application
payload mode; it is not part of the published confirmatory benchmark.

## 9. Final held-out benchmark

The final methodology was frozen at commit
`d0de087ddc39706deeddfd2ff8425468ddf7ba58` and tag `final-method-freeze` before
the held-out set was accessed. The final run evaluated 500 held-out images under
13 conditions, producing 6,500 paired classical/CNN rows.

Headline results:

- Classical 12-attack macro BER: **0.270757**
- CNN 12-attack macro BER: **0.245167**
- Relative CNN reduction: **9.45%**
- Clean CNN BER: **0.000188**
- Mean PSNR: **55.797 dB**
- Mean SSIM: **0.999110**

The CNN showed its clearest gains for JPEG90, JPEG50, Resize75, Resize50, and
repeated re-encoding. JPEG70 was not a statistically significant improvement
after Holm adjustment. Resize25 remained severe for both branches. Crop remained
near random because of coefficient-location synchronization loss.

The first execution attempt was terminated by the host after approximately
1,200 seconds and produced no result files or observable benchmark results. The
benchmark was restarted with the identical frozen command and completed. This
was one completed benchmark, not two competing completed runs.

## 10. Publication and repository state

The final-results commit was sanitized before publication. The public identity
manifest contains only:

`test_id, image_index, curated_image_sha256`

Original filenames were removed. No raw test images were published. The
sanitized results commit is:

`3f937e5c6fd230d219719b0752e818a369ec2dec`

It is published on `dataset-merge` and tagged `final-heldout-results`. The
methodology-freeze commit and tag were not rewritten. The working tree was clean
after publication, and no force push occurred.

## 11. What changed and what did not

### Changed during development

- decoder representation: full map → seed-aware selected coefficients;
- decoder architecture: pooled CNN → 369-parameter pointwise Conv1D;
- training exposure: clean → JPEG/re-encoding factorial exposure;
- diagnostic context: single coefficient → optional 3×3 local context;
- embedding strength: calibrated Δ values → frozen Δ=24;
- interpretation: crop failure identified as synchronization loss;
- publication manifest: source filenames removed and opaque test IDs added.

### Intentionally not changed after evidence was collected

- the classical DWT-QIM embedding concept;
- the coefficient-selection seed (42);
- the final payload after freeze;
- the primary CNN checkpoint;
- attack definitions and condition set;
- BER/statistical calculations;
- the held-out test identities;
- results in response to final test outcomes.

## 12. Evidence map

- [README](README.md): project purpose, pipeline, defaults, and code structure.
- [Research alignment](RESEARCH_ALIGNMENT.md): scope and thesis alignment notes.
- [Experiment milestone](EXPERIMENT_MILESTONE.md): development conclusions and
  milestone results.
- [Final methodology freeze](FINAL_METHODOLOGY_FREEZE.md): authoritative frozen
  configuration and no-tuning policy.
- [Machine-readable freeze](configs/final_methodology_freeze.json): exact
  parameters, payload, attacks, and statistical policy.
- [Final benchmark summary](experiments/final_heldout_benchmark/final_summary.json):
  condition-level results and paired statistics.
- [Final benchmark README](experiments/final_heldout_benchmark/README.md): output
  directory description.
- [Experiment log](docs/experiment_log.md): factual run-record template and
  selected entries.
- [Dataset build design](docs/superpowers/specs/2026-06-22-safebooru-dataset-build-design.md):
  curation and split design.

## 13. Thesis writing guidance

Chapter IV should separate development evidence from confirmatory held-out
evidence. The development stages justify the final architecture and Δ choice;
the final benchmark answers the frozen comparison question. Claims should be
limited to the tested seed, payload, dataset, attacks, and implementation.

The strongest defensible conclusion is that the seed-aware CNN improved raw BER
in several degradation conditions while preserving very high visual fidelity,
but it did not solve severe Resize25 or crop synchronization failure. The result
supports a conditional improvement of the learned bit-decision stage, not a
universal robustness guarantee.
