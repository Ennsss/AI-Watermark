# Machine Learning Enhancements for Frequency-Domain Watermarking: A Research Survey

## Abstract

This document surveys lightweight machine learning techniques applicable to enhancing the robustness, imperceptibility, and adaptability of frequency-domain image watermarking systems. Our baseline system employs a classical pipeline of Discrete Wavelet Transform (DWT) with Quantization Index Modulation (QIM), Reed-Solomon error correction, and AES-256 encryption. We investigate five areas of ML enhancement: (1) learned perceptual masking for adaptive embedding strength, (2) neural network-based extraction decoders, (3) adversarial training with differentiable distortion layers, (4) lightweight deployment under CPU-only constraints, and (5) training pipeline design for hybrid classical-neural architectures. We compare six state-of-the-art deep watermarking systems against our classical approach and propose a phased implementation roadmap. All recommendations target CPU-only inference under 500ms for 1024x1024 images.

**Keywords:** DWT, QIM, deep watermarking, perceptual masking, adversarial training, ONNX, lightweight inference

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Learned Perceptual Masking](#2-learned-perceptual-masking)
3. [Learned Extraction Decoder](#3-learned-extraction-decoder)
4. [Adversarial Training for Robustness](#4-adversarial-training-for-robustness)
5. [Lightweight Deployment Constraints](#5-lightweight-deployment-constraints)
6. [Training Pipeline Design](#6-training-pipeline-design)
7. [Prior Art Comparison](#7-prior-art-comparison)
8. [Recommended Implementation Roadmap](#8-recommended-implementation-roadmap)
9. [References](#9-references)

---

## 1. Introduction

Classical frequency-domain watermarking using DWT and QIM provides deterministic, reproducible embedding with well-understood theoretical foundations. However, these methods face inherent limitations: fixed or heuristic-based embedding strength, brittle extraction under compound distortions, and limited adaptability to image content. Recent advances in deep learning-based watermarking (2018-2025) have demonstrated significant improvements in robustness while maintaining imperceptibility, but often at the cost of computational complexity and GPU dependence.

This survey identifies ML techniques that can augment our existing DWT+QIM pipeline in a hybrid architecture, preserving the deterministic core while adding learned components where they provide the greatest benefit. The hybrid approach offers a key advantage: the classical embedding pipeline remains interpretable and reproducible, while learned components enhance robustness and perceptual quality.

### 1.1 Scope and Methodology

We focus on techniques published between 2018 and 2025, with emphasis on work from 2023-2025. Our evaluation criteria are: (a) compatibility with frequency-domain embedding, (b) CPU-only inference feasibility, (c) training data requirements, and (d) measured improvements in BER and PSNR/SSIM over classical baselines.

---

## 2. Learned Perceptual Masking

### 2.1 Motivation

Our current system uses local variance computed over sliding windows to determine the QIM quantization step size (delta) per coefficient block. This heuristic correlates with texture complexity but fails to capture higher-order perceptual properties: edge proximity, semantic saliency, and the human visual system's (HVS) contrast sensitivity function across spatial frequencies.

A learned masking model replaces this heuristic with a content-adaptive delta map that jointly optimizes imperceptibility and robustness.

### 2.2 Architecture Options

#### 2.2.1 Small U-Net (Recommended for Our Use Case)

A compact U-Net [Ronneberger et al., 2015] with 3-4 encoder/decoder levels is the most suitable architecture for generating pixel-wise (or block-wise) delta maps. Recent work by [Hao et al., 2023] on "Robust Texture-Aware Local Adaptive Image Watermarking with Perceptual Guarantee" demonstrates that texture-aware adaptive embedding achieves SSIM > 0.98 with BER < 0.02 at embedding densities up to 4 bits per block.

**Recommended architecture:**
- **Encoder:** 4 blocks of [Conv3x3 -> BatchNorm -> ReLU -> Conv3x3 -> BatchNorm -> ReLU -> MaxPool2x2]
- **Decoder:** 4 blocks with skip connections, bilinear upsampling
- **Bottleneck:** 256 channels (reduced from standard U-Net's 1024)
- **Output:** Single-channel delta map, sigmoid-scaled to [delta_min, delta_max]
- **Parameter count:** ~1.9M (vs. ~31M for full U-Net)
- **Input:** Y channel of image (single channel grayscale)

This architecture has been validated in related domains. [Zhang et al., 2024] used a U2-Net structure for multi-scale feature extraction in watermarking, generating high-resolution feature maps with edge information that guide embedding strength decisions.

#### 2.2.2 MobileNetV2 Backbone

[Wang et al., 2023] demonstrated a hybrid watermarking approach using MobileNetV2 as a feature extractor combined with DWT and DCT transforms for medical image watermarking. The MobileNetV2 backbone offers:

- **Depthwise separable convolutions** reduce parameters by 8-9x vs. standard convolutions
- **Inverted residuals** with linear bottlenecks preserve information flow
- **Parameter count:** ~3.4M for full MobileNetV2, reducible to ~0.5M with width multiplier 0.35
- **CPU inference:** ~25ms for 224x224 on mid-range CPU

For our application, MobileNetV2 would serve as a feature extractor feeding into a lightweight head that produces the delta map, rather than using the full classification network.

#### 2.2.3 Attention-Enhanced Variants

[MT-Mark, 2024] introduced adaptive feature modulation using mutual-teacher collaboration, where attention mechanisms guide watermark distribution. Similarly, [Li et al., 2024] proposed attention U-Net++ for watermarking with improved robustness through attention-gated skip connections. These attention mechanisms add ~10-15% computational overhead but significantly improve masking quality in regions with mixed texture/flat content.

### 2.3 Loss Functions

The loss function for training a perceptual masking network must jointly optimize for imperceptibility and robustness. We recommend:

```
L_total = lambda_1 * L_SSIM + lambda_2 * L_BER + lambda_3 * L_smooth + lambda_4 * L_range
```

**Component losses:**

| Loss Component | Purpose | Formulation |
|---|---|---|
| L_SSIM | Imperceptibility | 1 - SSIM(original, watermarked) |
| L_BER | Robustness post-attack | BCE(extracted_bits, original_bits) averaged over attack suite |
| L_smooth | Spatial coherence of delta map | Total variation of delta map |
| L_range | Prevent degenerate solutions | ReLU(delta_min - delta) + ReLU(delta - delta_max) |

**Recommended weights:** lambda_1 = 1.0, lambda_2 = 2.0, lambda_3 = 0.1, lambda_4 = 0.5

The higher weight on L_BER reflects the primary goal of robustness improvement. [Fernandez et al., 2022] showed that emphasizing robustness during training with a target PSNR constraint of 40 dB yields the best practical results.

### 2.4 Training Data Requirements

- **Minimum:** 5,000-10,000 diverse images covering textures, flat regions, edges, and mixed content
- **Recommended:** DIV2K (800 training images at 2K resolution) augmented with random crops yields ~50,000 patches at 256x256
- **Augmentation:** Random crops, flips, rotations, brightness/contrast jitter
- **Validation:** 100 images from DIV2K validation set

### 2.5 Expected Improvement

Based on literature, learned perceptual masking typically provides:
- **SSIM improvement:** +0.005 to +0.015 over heuristic masking at equivalent robustness
- **BER improvement:** 15-30% reduction at equivalent SSIM
- **Particularly beneficial** for images with large flat regions (sky, skin) where heuristic masking under-embeds

---

## 3. Learned Extraction Decoder

### 3.1 Motivation

Classical QIM bit reading is optimal under additive white Gaussian noise but degrades under non-linear distortions such as JPEG compression, color space conversion, and resampling. A learned decoder can compensate for these systematic distortions by learning the inverse mapping from degraded coefficients to embedded bits.

### 3.2 Hybrid Architecture: Classical Embedding + Learned Extraction

The recommended hybrid approach preserves our deterministic DWT+QIM embedding pipeline while replacing only the extraction step with a neural network. This design offers several advantages:

1. **Backward compatibility:** Watermarks embedded with the classical pipeline can be extracted by either classical or learned decoders
2. **Deterministic embedding:** The embedding process remains reproducible and interpretable
3. **Incremental deployment:** The learned decoder can be deployed as an optional enhancement
4. **Training simplicity:** The training objective is well-defined (predict embedded bits from degraded coefficients)

### 3.3 Architecture Options

#### 3.3.1 Lightweight CNN Decoder (Recommended)

A compact CNN operating on DWT coefficients rather than pixel space:

**Architecture:**
- **Input:** LH2 and HL2 subband coefficients (2 channels), cropped to embedding region
- **Feature extraction:** 4 blocks of [Conv3x3 -> BatchNorm -> ReLU], channels: 32 -> 64 -> 128 -> 256
- **Global average pooling**
- **FC layers:** 256 -> 128 -> N_bits (with sigmoid activation)
- **Parameter count:** ~0.8M
- **Operates in wavelet domain** (smaller spatial dimensions than pixel space)

This approach is inspired by [Jiang et al., 2021] (MBRS), which demonstrated that operating in a compact feature space with squeeze-and-excitation blocks improves extraction accuracy under JPEG compression.

#### 3.3.2 ResNet-18 Based Decoder

A truncated ResNet-18 (first 3 residual groups only) provides stronger feature extraction:

- **Parameter count:** ~2.8M (truncated)
- **Input:** Full Y channel or DWT coefficient maps
- **Residual connections** help preserve gradient flow during training
- **CPU inference:** ~35ms for 256x256 input

[Screen-shooting resistant watermarking, 2023] demonstrated that ResNet-based decoders in the frequency domain achieve robust extraction even under screen-camera distortions, representing one of the most challenging attack scenarios.

#### 3.3.3 Attention-Based Decoder

[Cross-attention watermarking, 2025] showed that multi-head and cross-attention mechanisms significantly improve extraction robustness for screen-shooting scenarios. For our use case:

- **Self-attention on DWT coefficients** helps the decoder attend to the most informative embedding locations
- **Parameter count:** ~1.5M with 4 attention heads
- **Overhead:** ~15ms additional CPU inference time
- **Best suited for:** Extraction under severe geometric distortions (crop, resize)

### 3.4 Training Procedure

```
Training Pipeline:
1. Load cover image I
2. Embed watermark using classical DWT+QIM pipeline -> I_w
3. Apply random attack from augmentation suite -> I_a
4. Forward I_a through learned decoder -> predicted_bits
5. Compute loss: BCE(predicted_bits, original_bits)
6. Backpropagate and update decoder weights only

Attack Augmentation Suite (during training):
- JPEG compression: Q uniformly sampled from [30, 95]
- Resize: scale factor uniformly sampled from [0.3, 1.0]
- Crop: removal percentage from [0, 40%] with random position
- Gaussian noise: sigma from [0, 15]
- Color jitter: brightness/contrast/saturation +/- 20%
- Combined chains: 2-3 random attacks composed
- Identity (no attack): 20% probability
```

### 3.5 Expected Improvement

Based on MBRS [Jia et al., 2021] and related work:
- **BER after JPEG Q50:** Classical QIM typically achieves 15-25% raw BER; learned decoder reduces to 3-8%
- **BER after resize to 50%:** Classical ~20-30%; learned ~5-12%
- **BER after combined chain:** Classical ~25-40%; learned ~8-15%
- **Perfect recovery rate (post-ECC):** Expected improvement from ~70% to ~90%+ across attack suite

---

## 4. Adversarial Training for Robustness

### 4.1 Differentiable Distortion Layers

The key challenge in training watermarking networks is that common image processing operations (JPEG, quantization) are non-differentiable. Several approximations have been developed:

#### 4.1.1 Differentiable JPEG (JPEGDiff)

[Shin & Song, 2017] and subsequent work introduced differentiable approximations of JPEG compression by replacing the rounding operation in DCT quantization with:

- **Straight-through estimator (STE):** Forward pass uses true rounding; backward pass passes gradients through as identity
- **Soft quantization:** Replace round(x) with x + tanh(alpha * (x - round(x))) with annealing alpha
- **Learned JPEG simulation:** Train a small network to approximate JPEG artifacts

MBRS [Jia et al., 2021] demonstrated that alternating between real JPEG, simulated JPEG, and clean images during training (mini-batch mixing) achieves superior robustness compared to using differentiable JPEG alone. Under Q=50, MBRS achieves BER < 0.01% with PSNR > 36 dB.

#### 4.1.2 Differentiable Resize and Crop

- **Resize:** Bilinear/bicubic interpolation is naturally differentiable
- **Crop:** Implemented as spatial transformer network with fixed affine parameters
- **Combined:** Compose differentiable operations sequentially in a noise layer

#### 4.1.3 Differentiable Color Space Conversion

YCbCr to RGB conversion is linear and fully differentiable. The non-differentiable step is chroma subsampling (4:2:0), approximated via average pooling with STE.

### 4.2 State-of-the-Art Adversarial Watermarking Architectures

#### 4.2.1 HiDDeN [Zhu et al., 2018]

The foundational end-to-end watermarking framework with encoder-noise-decoder architecture:

- **Encoder:** Conv-BN-ReLU blocks, concatenates message with spatial features
- **Noise layer:** Differentiable approximations of Crop, Cropout, Dropout, JPEG, Gaussian
- **Decoder:** Conv-BN-ReLU blocks with global average pooling
- **Discriminator:** Adversarial loss for imperceptibility
- **Capacity:** 30-48 bits at 128x128
- **Significance:** First end-to-end neural watermarking system; established the encoder-noise-decoder paradigm used by all subsequent methods

#### 4.2.2 ReDMark [Ahmadi et al., 2020]

Improved HiDDeN with residual diffusion:

- **Key innovation:** Residual connections in encoder produce an additive watermark residual rather than direct pixel modification
- **Performance:** PSNR ~50 dB, SSIM 0.95, BER 0.008 average across JPEG quality factors
- **Limitation:** Only 50% robustness against Gaussian Blur (sigma=2.0), indicating vulnerability to certain smoothing attacks

#### 4.2.3 MBRS [Jia et al., 2021]

Mini-Batch of Real and Simulated JPEG Compression:

- **Key innovation:** Training mini-batches cyclically contain (1) real JPEG compressed images, (2) differentiable JPEG simulated images, and (3) clean images. This bridges the gap between simulated and real JPEG artifacts.
- **Architecture enhancements:** Squeeze-and-Excitation blocks for channel attention, message processor for better bit expansion, additive diffusion block for crop robustness
- **Performance:** BER < 0.01% under JPEG Q50 with PSNR > 36 dB
- **Training details:** 128x128 images, 30-bit message, 256-dim FC embedding, batch size 16, lr=1e-3, 300 epochs (early stopping at ~110)

#### 4.2.4 StegaStamp [Tancik et al., 2020]

Designed for physical-world robustness (print-and-capture):

- **Encoder:** U-Net style network producing perturbation residual
- **Decoder:** CNN with spatial transformer network for geometric correction
- **Detection network:** BiSeNet for localization and rectification
- **Training:** Differentiable augmentations simulating printing, photography, and detection pipeline
- **Performance:** Robust to print-scan-capture cycle, significantly exceeding digital-only methods
- **Model size:** Large (~100M+ parameters); detection module uses BiSeNet at 1024x1024

#### 4.2.5 TrustMark [Bui et al., 2023]

Adobe's universal watermarking for arbitrary resolution images:

- **Architecture:** Embedder (E), Extractor (X), and Noise (N) modules with focal frequency loss
- **Extractor:** Compact ResNet-50 backbone, ~40MB as float16
- **Variants and performance:**

| Variant | Use Case | PSNR (dB) | Bit Accuracy |
|---|---|---|---|
| Q (Default) | Balanced robustness/quality | 43-45 | >96% |
| P (Perceptual) | Maximum quality | 48-50 | >95% |
| C (Compact) | Minimal model size | 38-39 | ~93% |

- **Payload:** 100 raw bits with BCH error correction (~70 effective bits)
- **Key innovation:** Post-processing layers and focal frequency loss for frequency-aware training; operates at arbitrary resolution via tiling

#### 4.2.6 DERO [2024]

Diffusion-Model-Erasure Robust Watermarking:

- **Key innovation:** Destruction and Compensation Noise Layer (DCNL) with multi-scale low-pass filtering and white noise compensation, specifically designed to resist diffusion model-based watermark removal
- **Extraction:** Uses pre-trained VAE before decoder for latent-domain extraction
- **Performance:** LDE robustness improves from 75% (SOTA) to 96%
- **Significance:** First framework specifically targeting AI-based watermark removal attacks

### 4.3 Relevance to Our Pipeline

For our hybrid architecture, the most applicable adversarial training techniques are:

1. **MBRS-style mini-batch mixing** for the learned decoder training (Section 3)
2. **Differentiable JPEG with STE** for end-to-end fine-tuning
3. **TrustMark's focal frequency loss** for frequency-domain awareness during training
4. **DERO's DCNL concept** for anticipating AI-based removal attacks

We do not need to adopt full end-to-end architectures (which would replace our classical embedding), but can selectively incorporate their training strategies.

---

## 5. Lightweight Deployment Constraints

### 5.1 Target Performance Budget

| Constraint | Target |
|---|---|
| Hardware | CPU only (no GPU required) |
| Inference time (embed) | < 500ms for 1024x1024 |
| Inference time (extract) | < 500ms for 1024x1024 |
| Model size on disk | < 50MB |
| RAM overhead | < 500MB additional |
| Framework | ONNX Runtime (cross-platform) |

### 5.2 Model Compression Techniques

#### 5.2.1 Post-Training Quantization

ONNX Runtime supports INT8 quantization with both dynamic and static modes [Microsoft, 2024]:

- **Dynamic quantization:** Computes scale/zero-point at runtime; no calibration data needed; ~2x speedup, ~4x size reduction
- **Static quantization:** Uses calibration dataset to pre-compute activation ranges; ~3-4x speedup; requires 100-500 representative images
- **FP16 inference:** ~1.5-2x speedup with negligible accuracy loss; 2x size reduction

**Expected impact on watermarking models:**

| Model | FP32 Size | INT8 Size | FP32 Time | INT8 Time |
|---|---|---|---|---|
| Small U-Net (masking) | 7.6 MB | 1.9 MB | ~80ms | ~25ms |
| CNN Decoder (extraction) | 3.2 MB | 0.8 MB | ~40ms | ~12ms |
| Combined pipeline | 10.8 MB | 2.7 MB | ~120ms | ~37ms |

*Estimates for 1024x1024 input on Intel i7-12700H, ONNX Runtime 1.17+*

#### 5.2.2 Knowledge Distillation

For training a lightweight student model from a larger teacher:

1. **Teacher:** Full U-Net or ResNet-50 trained without computational constraints
2. **Student:** Small U-Net (1.9M params) or MobileNetV2 (0.5M params with width=0.35)
3. **Distillation loss:** KL divergence between teacher and student output distributions + task-specific loss
4. **Temperature:** T=4.0 for softened probability distributions
5. **Training:** 100 epochs with lr=1e-3, cosine annealing

Expected accuracy retention: 95-98% of teacher performance at 5-10x fewer parameters [Hinton et al., 2015].

#### 5.2.3 Architecture-Level Optimizations

- **Depthwise separable convolutions** (MobileNet-style): 8-9x fewer multiply-accumulates vs. standard convolutions [Howard et al., 2017]
- **Channel pruning:** Remove channels with lowest L1-norm weights; typically 30-50% channels removable with <1% accuracy loss
- **Operator fusion:** ONNX Runtime automatically fuses Conv+BN+ReLU into single kernels

#### 5.2.4 ONNX Runtime Deployment

```python
# Example deployment pattern
import onnxruntime as ort

# Create session with optimization
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess_options.intra_op_num_threads = 4
sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

session = ort.InferenceSession("masking_model_int8.onnx", sess_options)

# Run inference
delta_map = session.run(None, {"y_channel": y_input})[0]
```

### 5.3 Benchmark Reference Points

From general lightweight model benchmarks [Li et al., 2024]:

| Model | Parameters | FLOPs | CPU Time (224x224) | CPU Time (1024x1024, est.) |
|---|---|---|---|---|
| MobileNetV3-Small | 2.5M | 56M | ~8ms | ~170ms |
| ResNet-18 | 11.7M | 1.8G | ~25ms | ~520ms |
| SqueezeNet 1.1 | 1.2M | 352M | ~12ms | ~250ms |
| EfficientNet-B0 | 5.3M | 390M | ~15ms | ~310ms |
| ShuffleNetV2 x0.5 | 1.4M | 41M | ~6ms | ~125ms |

*CPU times on ARM Cortex-A76; x86 times are comparable or faster. 1024x1024 estimates assume quadratic scaling.*

Our target of <500ms for 1024x1024 is achievable with MobileNetV3-Small, SqueezeNet, or ShuffleNetV2 backbones. ResNet-18 is borderline and may require INT8 quantization.

---

## 6. Training Pipeline Design

### 6.1 Training Data

#### 6.1.1 Primary Dataset: DIV2K

- **Size:** 800 training images, 100 validation images at 2K resolution
- **Diversity:** Covers people, nature, urban, indoor, text, and abstract content
- **Preprocessing:** Random crop to 256x256 patches during training (yields ~50K effective samples per epoch with stride-based sampling)
- **Advantages:** High resolution matches our target use case; well-established benchmark dataset

#### 6.1.2 Supplementary Dataset: COCO Subset

- **Recommended subset:** 10,000 images from COCO 2017 train split, filtered for minimum 512x512 resolution
- **Purpose:** Adds content diversity (80 object categories) not fully covered by DIV2K
- **Preprocessing:** Resize longest edge to 1024px, random crop to 256x256

#### 6.1.3 Total Training Data

| Dataset | Images | Patches/epoch | Purpose |
|---|---|---|---|
| DIV2K train | 800 | ~50,000 | Primary high-quality content |
| COCO subset | 10,000 | ~100,000 | Content diversity |
| **Total** | **10,800** | **~150,000** | |

### 6.2 Training Schedule

#### Phase 1: Learned Perceptual Masking Network

```
Epochs: 200
Batch size: 16
Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
Scheduler: Cosine annealing with warm restarts (T_0=50)
Loss: L_SSIM + 2.0*L_BER + 0.1*L_smooth + 0.5*L_range

Training loop per batch:
  1. Sample cover image patches (256x256)
  2. Forward through masking network -> delta_map
  3. Embed watermark using DWT+QIM with predicted deltas
  4. Apply random attack augmentation
  5. Extract with classical QIM reader
  6. Compute multi-objective loss
  7. Backpropagate through masking network only

Hardware: Single GPU for training (RTX 3060+), ~12 hours
```

#### Phase 2: Learned Extraction Decoder

```
Epochs: 300 (typically converges by epoch 150)
Batch size: 32
Optimizer: AdamW (lr=1e-3, weight_decay=1e-4)
Scheduler: Cosine annealing (T_max=300)
Loss: BCE(predicted_bits, true_bits) + 0.1*L2_regularization

Training loop per batch:
  1. Sample cover image patches
  2. Embed watermark using classical DWT+QIM pipeline (fixed)
  3. Apply random attack from augmentation suite
  4. Extract DWT coefficients from attacked image
  5. Forward coefficients through learned decoder -> predicted_bits
  6. Compute BCE loss against original bits
  7. Backpropagate through decoder only

Note: Embedding pipeline is NOT differentiable and does NOT receive gradients.
      Only the decoder is trained.

Hardware: Single GPU, ~8 hours
```

#### Phase 3: Joint Fine-Tuning (Optional)

```
Epochs: 50
Batch size: 16
Optimizer: AdamW (lr=1e-4, reduced from Phase 1/2)
Scheduler: Linear decay

Joint optimization of masking network + decoder:
  1. Sample cover images
  2. Masking network -> delta_map
  3. Embed with DWT+QIM using predicted deltas (detach gradients here)
  4. Apply attacks
  5. Learned decoder -> predicted_bits
  6. Loss = L_SSIM(original, watermarked) + 2.0*BCE(predicted_bits, true_bits)
  7. Alternate: update masking network (odd epochs) and decoder (even epochs)
```

### 6.3 Integration with Existing DWT+QIM Pipeline

The hybrid architecture maintains the classical pipeline as the backbone:

```
EMBEDDING (enhanced):
  Image -> Preprocessor -> Y channel
       -> Masking Network (ML) -> delta_map     [NEW]
       -> DWT -> QIM(delta=delta_map) -> IDWT    [EXISTING, modified delta source]
       -> Reconstruct -> Output PNG

EXTRACTION (enhanced):
  Image -> Preprocessor -> Y channel
       -> DWT -> coefficient_maps
       -> Learned Decoder (ML) -> predicted_bits  [NEW, replaces QIM reader]
       -> Decrypt -> RS Decode -> Payload

FALLBACK:
  If ML models unavailable, revert to:
       -> Classical QIM reader (existing)
       -> Local variance heuristic (existing)
```

**Key design principle:** The ML components are optional enhancements. The system must function with the classical pipeline alone, enabling graceful degradation if ML models are not available.

### 6.4 Evaluation Protocol

After training, evaluate on a held-out test set (DIV2K validation, 100 images) with the full attack suite:

| Attack | Parameters | Metric |
|---|---|---|
| JPEG compression | Q: 50, 60, 70, 80, 90 | BER, recovery rate |
| Resize | 50%, 75% of original | BER, recovery rate |
| Crop | 10%, 20%, 30% removal | BER, recovery rate |
| Gaussian noise | sigma: 2, 5, 10 | BER, recovery rate |
| Combined chain | Resize 75% -> JPEG Q70 -> Crop 20% | BER, recovery rate |
| No attack | Identity | PSNR, SSIM, BER |

Compare classical-only vs. hybrid (classical + ML) across all metrics.

---

## 7. Prior Art Comparison

### 7.1 System Comparison Table

| System | Year | Architecture | PSNR (dB) | BER after JPEG Q50 | Payload (bits) | Model Size | CPU Inference | Training Data | Training Time |
|---|---|---|---|---|---|---|---|---|---|
| **Our Classical (DWT+QIM)** | 2024 | Non-ML | ~42-45 | 15-25% (pre-ECC) | 64+ | 0 MB | <100ms | None | None |
| **HiDDeN** [Zhu et al., 2018] | 2018 | CNN enc-dec | ~33-38 | ~5-10% | 30-48 | ~15 MB | ~200ms* | COCO/ImageNet | ~24h (GPU) |
| **ReDMark** [Ahmadi et al., 2020] | 2020 | Residual CNN | ~50 | ~0.8% | 32-64 | ~20 MB | ~250ms* | DIV2K | ~36h (GPU) |
| **MBRS** [Jia et al., 2021] | 2021 | SE-Net enc-dec | ~36 | <0.01% | 30 | ~25 MB | ~300ms* | COCO subset | ~48h (GPU) |
| **StegaStamp** [Tancik et al., 2020] | 2020 | U-Net + STN | ~30-33 | ~2-5% | 100 | ~100+ MB | ~800ms* | DIV2K + augment | ~72h (GPU) |
| **SSL Watermark** [Fernandez et al., 2022] | 2022 | Pre-trained SSL | ~40 | ~3-8% | 30 | ~90 MB | ~500ms* | YFCC subset | ~24h (GPU) |
| **TrustMark** [Bui et al., 2023] | 2023 | ResNet-50 dec | 43-50 | ~2-4% | 100 (70 eff.) | ~40 MB (fp16) | ~350ms* | Proprietary | ~96h (GPU) |

*CPU inference times are estimates based on model architecture and parameter counts. Actual times depend on implementation optimization and hardware.*

### 7.2 Analysis

**Imperceptibility (PSNR):** ReDMark leads at ~50 dB, followed by TrustMark (43-50 dB depending on variant). Our classical approach achieves competitive 42-45 dB. HiDDeN and StegaStamp sacrifice PSNR for robustness.

**Robustness (BER after JPEG Q50):** MBRS achieves near-perfect extraction (<0.01%) through its real/simulated JPEG training strategy. ReDMark follows at 0.8%. Our classical approach has the weakest JPEG robustness at 15-25% raw BER, though Reed-Solomon coding can correct many of these errors.

**Payload capacity:** TrustMark and StegaStamp embed 100 bits; our system targets 64+ bits, which is sufficient for artist ID + timestamp + pHash.

**CPU feasibility:** Our classical approach is the fastest. Among ML methods, HiDDeN is most lightweight. TrustMark offers the best quality-to-size ratio with its compact ResNet-50 decoder at 40MB fp16.

**Training requirements:** All ML methods require GPU training (24-96 hours). Our classical approach requires no training. The hybrid approach we propose requires only 20-30 hours of GPU training for the optional ML components.

### 7.3 Key Insight

No single ML system dominates across all metrics. The hybrid approach we propose---classical DWT+QIM embedding with optional learned masking and extraction---aims to combine the strengths of classical methods (no training needed, fast CPU inference, deterministic) with targeted ML improvements (adaptive masking, robust extraction).

---

## 8. Recommended Implementation Roadmap

### Phase 1: Learned Extraction Decoder (Highest ROI)
**Timeline:** 3-4 weeks | **Priority:** HIGH

The learned extraction decoder offers the highest return on investment because:
- It directly addresses the primary weakness (BER under compression/distortion)
- It requires no changes to the embedding pipeline
- It is backward-compatible with existing watermarked images
- Training is straightforward (supervised, single objective)

**Deliverables:**
1. Lightweight CNN decoder (~0.8M params) operating on DWT coefficients
2. Training pipeline using DIV2K + attack augmentation
3. ONNX-exported INT8 model (<1MB)
4. Benchmark comparison: classical vs. learned extraction across attack suite
5. Fallback logic: attempt learned extraction, fall back to classical if model unavailable

**Success criteria:** BER after JPEG Q50 reduced from 15-25% to <8%; perfect recovery rate >90% across attack suite.

### Phase 2: Learned Perceptual Masking (Medium ROI)
**Timeline:** 3-4 weeks | **Priority:** MEDIUM

After the decoder is in place, a learned masking network can further improve the quality-robustness tradeoff.

**Deliverables:**
1. Small U-Net masking network (~1.9M params) producing per-block delta maps
2. Joint loss function implementation (SSIM + BER + smoothness)
3. ONNX-exported INT8 model (<2MB)
4. A/B comparison: heuristic vs. learned masking at matched SSIM/BER

**Success criteria:** SSIM improvement of +0.01 at equivalent BER, or BER reduction of 20% at equivalent SSIM.

### Phase 3: Adversarial Training Enhancement (Lower ROI, Higher Complexity)
**Timeline:** 2-3 weeks | **Priority:** LOW

Incorporate MBRS-style training strategies and differentiable JPEG layers.

**Deliverables:**
1. Differentiable JPEG layer (STE-based) for training pipeline
2. Mini-batch mixing strategy (real JPEG / simulated JPEG / clean)
3. Retrained decoder with adversarial augmentation
4. Benchmark under expanded attack suite including AI-based removal

**Success criteria:** BER after JPEG Q50 reduced to <3%; robustness maintained under novel attacks.

### Phase 4: Optimization and Deployment
**Timeline:** 2 weeks | **Priority:** MEDIUM

**Deliverables:**
1. INT8 quantization of all models via ONNX Runtime
2. Knowledge distillation if models exceed size/speed targets
3. End-to-end latency benchmark on target hardware
4. Integration with CLI and GUI interfaces
5. Documentation and model versioning

**Success criteria:** Combined ML inference <200ms on mid-range CPU; total pipeline (embed or extract) <500ms for 1024x1024.

### Total Estimated Timeline: 10-13 weeks

---

## 9. References

### Core Watermarking Systems

- [Zhu et al., 2018] Zhu, J., Kaplan, R., Johnson, J., and Fei-Fei, L. "HiDDeN: Hiding Data With Deep Networks." *ECCV 2018*. https://arxiv.org/abs/1807.09937

- [Ahmadi et al., 2020] Ahmadi, M., Norouzi, A., et al. "ReDMark: Framework for Residual Diffusion Watermarking based on Deep Networks." *Expert Systems with Applications*, 2020. https://www.researchgate.net/publication/338166782

- [Jia et al., 2021] Jia, Z., Fang, H., and Zhang, W. "MBRS: Enhancing Robustness of DNN-based Watermarking by Mini-Batch of Real and Simulated JPEG Compression." *ACM Multimedia 2021*. https://arxiv.org/abs/2108.08211

- [Tancik et al., 2020] Tancik, M., Mildenhall, B., and Ng, R. "StegaStamp: Invisible Hyperlinks in Physical Photographs." *CVPR 2020*. https://www.matthewtancik.com/stegastamp

- [Fernandez et al., 2022] Fernandez, P., Sablayrolles, A., Furon, T., Jegou, H., and Douze, M. "Watermarking Images in Self-Supervised Latent Spaces." *ICASSP 2022*. https://arxiv.org/abs/2112.09581

- [Bui et al., 2023] Bui, T., et al. "TrustMark: Universal Watermarking for Arbitrary Resolution Images." *arXiv 2023, ICCV 2025*. https://arxiv.org/abs/2311.18297

- [DERO, 2024] "DERO: Diffusion-Model-Erasure Robust Watermarking." *ACM Multimedia 2024*. https://openreview.net/forum?id=ktMvfLYFas

- [Wu et al., 2023] Wu, et al. "SepMark: Deep Separable Watermarking for Unified Source Tracing and Deepfake Detection." *ACM Multimedia 2023*. https://arxiv.org/abs/2305.06321

### Perceptual Masking and Adaptive Embedding

- [Hao et al., 2023] Hao, et al. "Robust Texture-Aware Local Adaptive Image Watermarking With Perceptual Guarantee." *IEEE TCSVT 2023*. https://ui.adsabs.harvard.edu/abs/2023ITCSV..33.4660H

- [Zhang et al., 2024] Zhang, et al. "A robust image watermarking framework based on U2-net encoder and loss function weight assignment." *Multimedia Systems*, 2024. https://link.springer.com/article/10.1007/s00530-024-01640-1

- [Li et al., 2024] Li, et al. "A novel robust digital image watermarking scheme based on attention U-Net++ structure." *The Visual Computer*, 2024. https://link.springer.com/article/10.1007/s00371-024-03271-z

- [MT-Mark, 2024] "MT-Mark: Rethinking Image Watermarking via Mutual-Teacher Collaboration with Adaptive Feature Modulation." *arXiv 2024*. https://arxiv.org/abs/2512.19438

- [Wang et al., 2023] Wang, et al. "Hybrid watermarking algorithm for medical images based on digital transformation and MobileNetV2." *Information Sciences*, 2023. https://www.sciencedirect.com/science/article/abs/pii/S0020025523013956

### Lightweight Neural Networks and Extraction

- [Screen-shooting CNN, 2023] "Screen-shooting resistant image watermarking based on lightweight neural network in frequency domain." *Journal of Visual Communication and Image Representation*, 2023. https://www.sciencedirect.com/science/article/abs/pii/S1047320323000871

- [Cross-attention, 2025] "Screen shooting resistant watermarking based on cross attention." *Scientific Reports*, 2025. https://www.nature.com/articles/s41598-025-00912-8

- [Neural JPEG, 2024] "A Neural-Network-Based Watermarking Method Approximating JPEG Quantization." *Journal of Imaging*, 2024. https://www.mdpi.com/2313-433X/10/6/138

- [DnCNN-B, 2025] "DnCNN-B: a lightweight watermark image correction model based on DnCNN and bicubic interpolation." 2025. https://www.sciencedirect.com/science/article/pii/S3050520825000284

### Adversarial Training and Differentiable Distortions

- [Blind DL Watermarking, 2024] "Blind Deep-Learning-Based Image Watermarking." *arXiv 2024*. https://arxiv.org/pdf/2402.09062

- [JPEG Invariance, 2025] "Towards JPEG-Compression Invariance for Adversarial Optimization." *SCITEPRESS 2025*. https://www.scitepress.org/Papers/2025/133002/133002.pdf

- [Meta-FC, 2025] "Meta-FC: Meta-Learning with Feature Consistency for Robust and Generalizable Watermarking." *arXiv 2025*. https://arxiv.org/html/2602.21849

- [Invisible Watermarks, 2024] "Invisible Watermarks: Attacks and Robustness." *arXiv 2024*. https://arxiv.org/html/2412.12511v1

### Surveys and Reviews

- [Springer Survey, 2024] "Deep Learning-Based Watermarking Techniques Challenges: A Review of Current and Future Trends." *Circuits, Systems, and Signal Processing*, 2024. https://link.springer.com/article/10.1007/s00034-024-02651-z

- [Sensors Survey, 2025] "Deep Learning for Image Watermarking: A Comprehensive Review and Analysis of Techniques, Challenges, and Applications." *Sensors*, 2025. https://www.mdpi.com/1424-8220/26/2/444

- [WAVES Benchmark, 2024] "WAVES: Benchmarking the Robustness of Image Watermarks." *arXiv 2024*. https://arxiv.org/html/2401.08573v3

- [AI Watermarking 2025] Colwell, B.D. "Remarkable Breakthroughs In AI Watermarking: 2025." https://briandcolwell.com/remarkable-breakthroughs-in-ai-watermarking-2025/

### Model Optimization

- [ONNX Quantization] "Quantize ONNX models." ONNX Runtime Documentation. https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

- [Hinton et al., 2015] Hinton, G., Vinyals, O., and Dean, J. "Distilling the Knowledge in a Neural Network." *NIPS Workshop 2015*.

- [Howard et al., 2017] Howard, A., et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." *arXiv 2017*.

- [Ronneberger et al., 2015] Ronneberger, O., Fischer, P., and Brox, T. "U-Net: Convolutional Networks for Biomedical Image Segmentation." *MICCAI 2015*.

### Datasets

- [DIV2K] Agustsson, E. and Timofte, R. "NTIRE 2017 Challenge on Single Image Super-Resolution: Dataset and Study." *CVPRW 2017*. https://data.vision.ee.ethz.ch/cvl/DIV2K/

- [COCO] Lin, T.Y., et al. "Microsoft COCO: Common Objects in Context." *ECCV 2014*.

---

*Document prepared: March 2026. Last updated based on publications through February 2025.*
