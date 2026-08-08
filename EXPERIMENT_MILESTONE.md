# ARTIFACT Experimental Milestone

## 1. Current Research Question

Can a neural bit-decision decoder improve DWT-QIM watermark recovery under
social-media-like degradation while keeping embedding fixed?

## 2. Frozen Core Watermark Configuration

The controlled experiments use 512×512 RGB images converted to YCbCr and operate
on the Y luminance channel. A two-level Haar DWT is applied, with binary QIM in
the LH2 and HL2 subbands. The payload is 128 bits, coefficient seed 42 is used
for controlled experiments, and extraction is blind. There is no ECC.

The selected embedding strength is delta 24. It was frozen after Stage 3A for
subsequent development experiments. This is a validation-stage calibration
choice, not a claim that delta 24 is universally optimal.

## 3. Baseline Full-Map CNN

The original decoder accepts a 128×128×2 input and applies Conv2D(32), max
pooling, Conv2D(64), max pooling, flattening, Dense(128), and a 128-output
sigmoid layer.

In Run 1, all 32 controlled clean examples were classically recoverable. After
500 epochs, the CNN achieved mean BER 0.322021 and recovered 0/32 payloads
perfectly. The full-map formulation therefore could not memorize even this
controlled clean task.

## 4. Clean Round-Trip Finding

QIM-domain extraction and extraction after IDWT but before clipping both had
BER 0. Clipping the reconstructed Y/RGB values introduced errors for highly
saturated images. This remains a separate embedding/reconstruction limitation,
not a resolved issue. Difficult images were not removed from the actual
validation or test corpus.

## 5. Signal Localization Diagnostic

Selected coefficients retained BER 0, normalization preserved the available
information, and payload-dependent changes at selected locations were about
delta/2 = 8. Non-selected locations showed essentially no corresponding
payload-dependent change. Direct selected-coefficient survival under structural
max pooling was only about 38.5% for 2×2 pooling and 20.1% for 4×4 pooling.
The pooled full-map representation was therefore poorly matched to sparse,
seeded coefficient locations.

## 6. Seed-Aware Decoder

For each of the 128 selected coefficients in payload order, the representation
contains:

1. coefficient / delta
2. sin(pi × coefficient / delta)
3. cos(pi × coefficient / delta)
4. subband identifier

The current CNN is Conv1D(16, kernel=1, ReLU), Conv1D(16, kernel=1, ReLU), and
Conv1D(1, kernel=1, sigmoid). It has 369 parameters, with no pooling, flattening,
dense layers, or dropout. Deterministic coefficient-location regeneration
remains outside the learned decision rule and is shared conceptually with
classical extraction.

## 7. Run 2 — Memorization

Run 2 used 32 examples from four source images, fixed coefficient seed 42, and
no attacks. It reached BER 0 and 32/32 perfect payloads, with the first stable
zero-BER region around epochs 102–106. The clean QIM task is neural-learnable
when coefficients are explicitly aligned with payload bits.

## 8. Run 3A — Zero-Training Transfer

The frozen Run 2 model was evaluated without retraining on 100 validation
images and 200 unseen image-payload examples. Mean BER was 0.069531. On the
classically clean subset, CNN BER was 0.000989. LH2 BER was 0.000234, whereas
HL2 BER was 0.138828. This was partial transfer: training diversity was
insufficient, especially for HL2.

## 9. Run 3B — Clean Generalization

Run 3B used 500 training images, 1,000 training examples, 100 validation images,
and 200 fixed validation examples with the unchanged 369-parameter model.

- CNN validation BER: 0.000195
- Perfect recovery: 195/200
- Classical validation BER: 0.127813
- LH2 BER: 0.000234
- HL2 BER: 0.000156

Conclusion: **STRONG PASS.** Clean held-out generalization is considered solved
sufficiently to begin attack training.

## 10. Stage 2 — JPEG/Re-encoding Training

Conditions were Clean, JPEG90, JPEG70, JPEG50, Reencode1 (JPEG Q85 once),
Reencode2 (Q85 twice), and Reencode3 (Q85 three times).

| Condition | CNN BER |
|---|---:|
| Clean | 0.000469 |
| JPEG90 | 0.057383 |
| JPEG70 | 0.350508 |
| JPEG50 | 0.457422 |
| Reencode1 | 0.184414 |
| Reencode2 | 0.193789 |
| Reencode3 | 0.196875 |

Across the six attack conditions, classical macro BER was 0.277038 and CNN
macro BER was 0.240065, a 13.35% relative reduction. Conclusion: **PARTIAL.**

## 11. Stage 2B — Full-Factorial Exposure

Stage 2B expanded 1,000 base training pairs across all seven conditions, for
7,000 training examples.

| Condition | CNN BER |
|---|---:|
| Clean | 0.000391 |
| JPEG90 | 0.055273 |
| JPEG70 | 0.349961 |
| JPEG50 | 0.459414 |
| Reencode1 | 0.180625 |
| Reencode2 | 0.191250 |
| Reencode3 | 0.194258 |

The six-attack macro BER changed from 0.240065 in Stage 2 to 0.238464 in Stage
2B: an absolute improvement of 0.001602, or 0.67% relative. Conclusion:
**PARTIAL.** Increasing attack exposure by roughly seven times produced almost
no improvement in moderate/severe JPEG recovery. Insufficient attack exposure
is therefore unlikely to be the primary bottleneck. This does not prove formal
information-theoretic loss.

## 12. Stage 2C — Feature Separability Diagnostic

Stage 2C performed no CNN training and evaluated the delta-16 single-coefficient
representation directly. At JPEG70, Stage 2B CNN BER was 0.349961, empirical
lookup BER was 0.353086, and k-nearest-neighbor BER was 0.364766. At JPEG50,
the corresponding values were 0.459414, 0.456914, and 0.461016. Phase overlap
was 0.6674 for JPEG70 and 0.8937 for JPEG50.

At delta 16, these results supported a practical single-coefficient ambiguity
ceiling. They did not prove absolute or information-theoretic loss.

## 13. Stage 2D — Local 3×3 Context

Stage 2D changed only the representation from one selected coefficient to a
same-subband 3×3 neighborhood using a shared 601-parameter local decoder.
Repeated Q85 re-encoding improved strongly, but JPEG70 improved only from
0.349961 to 0.344609 and JPEG50 only from 0.459414 to 0.456758. Clean BER rose
to 0.005234. The result was **PARTIAL**: local context contained useful evidence
but did not resolve the primary JPEG70/JPEG50 limitation.

## 14. Stage 3A — QIM Delta Calibration

Stage 3A tested delta values 8, 16, 24, and 32 while freezing the first 500
training images, first 100 validation images, two payloads per image, payload
seeds, coefficient seed 42, Haar level-2 LH2/HL2 embedding, seven-condition
full-factorial exposure, attack definitions, and the 369-parameter seed-aware
decoder. A fresh decoder was trained separately for each delta. These remain
development/validation results; the held-out test set was not used.

| Delta | Mean PSNR | Mean SSIM | Clean | JPEG90 | JPEG70 | JPEG50 | Reencode macro | Six-attack macro |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 56.429 | 0.999238 | 0.023203 | 0.264141 | 0.480234 | 0.499297 | 0.403503 | 0.409030 |
| 16 | 56.130 | 0.999161 | 0.000352 | 0.054922 | 0.349805 | 0.459336 | 0.188451 | 0.238236 |
| 24 | 55.701 | 0.999033 | 0.000195 | 0.024297 | 0.233750 | 0.365078 | 0.050755 | 0.129232 |
| 32 | 55.192 | 0.998860 | 0.000234 | 0.012031 | 0.155938 | 0.274023 | 0.029388 | 0.088359 |

At delta 24, detailed classical/CNN BER was:

| Condition | Classical BER | CNN BER |
|---|---:|---:|
| Clean | 0.127852 | 0.000195 |
| JPEG90 | 0.151758 | 0.024297 |
| JPEG70 | 0.235078 | 0.233750 |
| JPEG50 | 0.385820 | 0.365078 |
| Reencode1 | 0.161328 | 0.040078 |
| Reencode2 | 0.170273 | 0.053906 |
| Reencode3 | 0.174297 | 0.058281 |

Robustness improved monotonically and visual fidelity decreased monotonically
as delta increased. Relative to delta 16, delta 24 reduced six-attack macro BER
by 0.109004, reduced JPEG70 BER from 0.349805 to 0.233750, reduced JPEG50 BER
from 0.459336 to 0.365078, and reduced re-encoding macro BER from 0.188451 to
0.050755. Its paired fidelity cost was 0.428 dB PSNR and 0.000128 SSIM. No
catastrophic LH2/HL2 imbalance occurred.

The predefined rule selected the smallest delta providing a meaningful
robustness gain while retaining high fidelity. Therefore **delta 24 was
selected and frozen**. Delta 32 had lower BER, but selecting it solely for
minimum BER would violate that predefined robustness/fidelity rule. Delta 24
is the selected setting for subsequent experiments, not a universal optimum.

Stage 3A also refines Stage 2C: JPEG50 was near-random for the delta-16
single-coefficient representation, but stronger embedding substantially
improved recovery. The ambiguity was therefore dependent on embedding strength,
not an absolute compression limit. JPEG50 remains difficult, but a strict
severe-JPEG ceiling is no longer supported.

# Development Complete — Pre-Final-Test Milestone

## Frozen watermark configuration currently established

- Input: 512×512 RGB, converted to YCbCr for Y-channel processing
- Transform: two-level Haar DWT with symmetric boundary handling
- Embedding subbands: LH2 and HL2
- Payload: 128 raw bits, split across 64 LH2 and 64 HL2 coefficients
- Deterministic coefficient seed: 42
- Embedding rule: binary QIM
- Delta: **24, frozen after Stage 3A**
- ECC: none in the core experiment

Delta 24 was selected by the predefined robustness/fidelity rule. Its
six-attack CNN macro BER was approximately 0.129232, compared with 0.213092
for classical extraction, while mean embedding fidelity remained approximately
55.701 dB PSNR and 0.999033 SSIM.

## Primary learned decoder

The primary decoder gathers the 128 seed-selected coefficients in payload
order. For each coefficient `c`, its input is:

1. `c / 24`
2. `sin(pi * c / 24)`
3. `cos(pi * c / 24)`
4. subband ID (`LH2 = 0`, `HL2 = 1`)

Its architecture is Conv1D(16, kernel 1, ReLU), Conv1D(16, kernel 1, ReLU),
and Conv1D(1, kernel 1, sigmoid), with 369 parameters. This is the main decoder
currently intended for the final classical-versus-neural comparison unless the
formal methodology-freeze step explicitly determines otherwise.

## Development conclusions

### Compression

Delta 24 materially improved JPEG and repeated-re-encoding robustness while
retaining very high visual fidelity. JPEG50 remains difficult, but Stage 3A
showed that the earlier delta-16 ambiguity depended on embedding strength.

### Resize

The frozen CNN transferred meaningfully to Resize75 (BER 0.094609) and
Resize50 (BER 0.113398), but Resize25 was near random (BER 0.505156). Stage
4A.1 found locally surviving but non-systematically placed evidence, with no
useful fixed offset. Stage 4B's controlled 601-parameter 3×3 decoder retained
clean recovery and modestly improved moderate resizing, but Resize25 remained
approximately 0.503203. Severe 25% resize is therefore accepted as a limitation
of the current blind framework.

### Crop

Stage 5A found near-random classical and neural BER at every crop severity.
Cropping changes the DWT dimensions and thus the shape-dependent seeded
coefficient permutation, causing extraction from unrelated locations. The
dominant failure is synchronization loss, so ordinary crop-aware bit-decision
training is not justified within the current method.

## Important methodological status

> **DEVELOPMENT IS NOW CLOSED.**

> **THE HELD-OUT TEST SET HAS NOT BEEN ACCESSED.**

- Stage 5A is the final development-stage attack experiment.
- A separate final-methodology-freeze step must occur before the test benchmark.
- No further parameter or model tuning may occur based on final test results.
- Final thesis test results are not yet known or claimed in this milestone.

## Test-Set Status

> **THE 500-IMAGE HELD-OUT TEST SET HAS NOT BEEN USED FOR MODEL SELECTION OR
> THESE DEVELOPMENT EXPERIMENTS.**

Training and model development use train/validation only. The test set remains
reserved for final frozen evaluation.

## Thesis Implications

- The original full-map CNN is retained as the documented baseline.
- The seed-aware decoder is an experimentally justified architectural refinement.
- Embedding remains classical DWT-QIM.
- The purpose remains classical versus learned bit decision under identical
  embedding and degradation conditions.
- Development experiments should later be separated from final Chapter IV test
  results.
- Failed and partial experiments will be documented as ablations rather than hidden.

# Final Methodology Frozen

The development milestone at commit `f08204a36f2872d62ab695253e018ca0dff0eb06`
and tag `milestone-development-complete` is complete. Delta 24 remains frozen,
and the Stage 3A 369-parameter seed-aware decoder is the final primary CNN; the
Stage 4B local model remains an ablation.

The fixed balanced ownership payload, complete 13-condition suite, fair paired
attack rule, primary and diagnostic metrics, computational timing boundaries,
two-sided paired Wilcoxon tests, Holm adjustment, and rank-biserial effect size
are now specified in `FINAL_METHODOLOGY_FREEZE.md` and
`configs/final_methodology_freeze.json`.

The held-out test set remains untouched. No further development, model
selection, parameter tuning, payload changes, attack changes, or statistical
changes are permitted in response to final test outcomes. Final results do not
yet exist.
