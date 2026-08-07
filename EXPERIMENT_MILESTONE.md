# ARTIFACT Experimental Milestone

## 1. Current Research Question

Can a neural bit-decision decoder improve DWT-QIM watermark recovery under
social-media-like degradation while keeping embedding fixed?

## 2. Frozen Core Watermark Configuration

The controlled experiments use 512×512 RGB images converted to YCbCr and operate
on the Y luminance channel. A two-level Haar DWT is applied, with binary QIM in
the LH2 and HL2 subbands. The payload is 128 bits, coefficient seed 42 is used
for controlled experiments, and extraction is blind. There is no ECC.

The current embedding strength is delta 16. This value remains provisional;
final calibration has not yet occurred.

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

## 12. Current Working Hypothesis

The current evidence suggests that the single-coefficient representation
reaches a practical ambiguity ceiling under moderate/severe JPEG compression.
A sufficiently disturbed selected coefficient may cross QIM regions such that
its attacked value alone no longer uniquely reveals the original embedded bit.
This remains a hypothesis requiring Stage 2C validation.

## 13. Next Planned Experiment

**Stage 2C — Feature Separability Diagnostic** will perform no CNN training. It
will measure target-conditioned QIM phase overlap, compare bit-0 and bit-1
feature distributions, evaluate empirical lookup/classical classifiers,
measure contradictory or ambiguous feature regions, quantify clean-to-attacked
coefficient displacement, and determine whether JPEG70/JPEG50 remain separable
using the current single-coefficient features.

Resize and crop training have **not** started yet.

## 14. Test-Set Status

> **THE 500-IMAGE HELD-OUT TEST SET HAS NOT BEEN USED FOR MODEL SELECTION OR
> THESE DEVELOPMENT EXPERIMENTS.**

Training and model development use train/validation only. The test set remains
reserved for final frozen evaluation.

## 15. Thesis Implications

- The original full-map CNN is retained as the documented baseline.
- The seed-aware decoder is an experimentally justified architectural refinement.
- Embedding remains classical DWT-QIM.
- The purpose remains classical versus learned bit decision under identical
  embedding and degradation conditions.
- Development experiments should later be separated from final Chapter IV test
  results.
- Failed and partial experiments will be documented as ablations rather than hidden.
