# AI Watermark System - Complete Feature List

## Overview

The AI Watermark system is a research-focused platform for embedding and extracting invisible watermarks in digital images using DWT-QIM (Discrete Wavelet Transform - Quantization Index Modulation) combined with optional CNN-assisted extraction. The system includes a complete Python backend library, command-line interface, and a full-stack web application for digital artwork provenance management.

---

## Core Watermarking Features

### 1. **Classical DWT-QIM Embedding**
- **Wavelet Transform**: Discrete Wavelet Transform (DWT) with Haar wavelet
- **Decomposition Level**: Two-level DWT decomposition
- **Target Subbands**: LH2 and HL2 (low-high and high-low at level 2)
- **Quantization**: Binary Quantization Index Modulation (QIM) for bit embedding
- **Payload**: Fixed 128-bit ownership payload
- **Deterministic Coefficient Selection**: PRNG-based coefficient location selection using configurable seeds
- **Symmetric Boundary Mode**: DWT operates in symmetric boundary extension mode
- **Delta Calibration**: Configurable quantization step (delta) with default candidates: 8, 16, 24, 32

### 2. **Classical Extraction**
- **QIM Grid Decision**: Read embedded bits using quantization grid decisions
- **Bit Error Rate (BER) Computation**: Hamming distance-based accuracy metric
- **Deterministic Extraction**: Same seed and delta required for reliable extraction
- **No Decoding Required**: Raw bit recovery without error correction

### 3. **CNN-Assisted Extraction**
- **Optional Machine Learning Branch**: Convolutional Neural Network for predicting embedded bits
- **Transform Domain Input**: LH2/HL2 coefficients stacked as 128×128×2 tensor
- **Shallow CNN Decoder**: Baseline CNN model for bit prediction
- **End-to-End Prediction**: Direct bit probability prediction from degraded coefficients
- **Non-intrusive**: CNN extraction does not modify embedding pipeline

### 4. **Payload Management**
- **Fixed Raw Payload**: Deterministic 128-bit bitstream for research consistency
- **Optional Legacy Security**:
  - AES-256 encryption
  - Reed-Solomon error correction
  - Repetition coding for redundancy
  - Custom provenance payload encoding
- **Hex String Support**: 32-character hex strings for custom 128-bit payloads

---

## Degradation & Attack Features

### 1. **Social Media Degradation Suite**
The system includes realistic attack scenarios that simulate social media compression and processing:

#### JPEG Compression
- Quality factors: 90, 70, 50
- Variable compression levels for realistic testing

#### Image Resizing
- Downsampling ratios: 75%, 50%, 25%
- Tests robustness to resolution reduction

#### Cropping Attacks
- **Mild Crop**: 5-10% of image removed
- **Moderate Crop**: 20-30% of image removed
- **Severe Crop**: 40-50% of image removed
- Tests watermark survival under partial image loss

#### Re-encoding
- Multiple passes: 1x, 2x, 3x re-encoding
- Tests cumulative degradation effects

### 2. **Optional Stress Tests**
- Gaussian noise addition
- Screenshot simulation
- Combined attack chains
- Custom degradation sequences

### 3. **Deterministic Attack Reproduction**
- Seeded random operations for reproducibility
- Configurable attack parameters
- Extensible attack framework

---

## Evaluation & Analysis Features

### 1. **Comprehensive Metrics**
- **Primary Metric - BER**: Bit Error Rate (Hamming distance / payload length)
- **Versioned Verification Policy**: Selected-record classification uses one centralized, currently provisional BER policy and stores the applied threshold on new events
- **Supporting Metrics**:
  - SSIM (Structural Similarity Index)
  - PSNR (Peak Signal-to-Noise Ratio)
  - Timing measurements
  - Execution performance

### 2. **Statistical Analysis**
- **Wilcoxon Signed-Rank Test**: Non-parametric comparison of classical vs. CNN extraction
- **Rank-Biserial Effect Size**: Measure statistical significance
- **Holm-Bonferroni Correction**: Multiple comparison correction
- **Paired Comparison Framework**: Direct classical vs. CNN evaluation

### 3. **Configuration-Based Evaluation**
- **Named Evaluation Configs**: Predefined configurations (baseline, extended, full)
- **Batch Processing**: Multiple configurations in single run
- **Output Organization**: Structured result directories and CSV exports

### 4. **Automated Reporting**
- CSV result export for analysis
- Per-image detailed metrics
- Configuration metadata in results
- Statistical summary tables

---

## Image Processing Features

### 1. **Format Support**
- PNG format (lossless)
- JPEG format (lossy)
- Automatic format detection

### 2. **Color Space Handling**
- RGB to YCbCr conversion
- Y channel extraction (luminance)
- CbCr preservation and restoration
- Automatic YCbCr to RGB conversion

### 3. **Image Size Management**
- Automatic padding to multiple of 4 (DWT requirement)
- 512×512 standard size for main experiment
- Flexible size support with padding handling
- Unpadding for output images

### 4. **Image Preprocessing**
- Alpha channel handling
- Image normalization
- Automatic resizing support
- Format conversion during processing

---

## Command-Line Interface (CLI) Features

### 1. **Embed Command**
```bash
python -m cli.main embed <image_path> [options]
```
- Embed watermark into image
- Configurable delta (quantization step)
- Optional custom payload
- Configurable coefficient seed
- Output path specification

### 2. **Extract Command**
```bash
python -m cli.main extract <image_path> [options]
```
- Extract watermark from image
- Raw bit extraction with classical QIM
- BER computation
- Parameter specification matching embedding

### 3. **Benchmark Command**
```bash
python -m cli.main benchmark <directory> [options]
```
- Classical baseline testing
- Batch processing of image directory
- Attack chain application
- CSV result export
- Performance measurement

### 4. **Evaluation Command**
```bash
python -m evaluation run [options]
```
- Structured experimental runner
- Named configuration support
- Batch evaluation of multiple configs
- Automatic result organization

---

## Web Application Features

### Product Portal
- **Dashboard** - Active-artwork and verification-outcome metrics without a redundant watermarked counter
- **My Artworks** - Responsive, searchable active registry cards with contained metadata and actions
- **Register Artwork** - Upload flow that embeds a watermark and stores the artwork record
- **Verify Image** - Verification workflow that compares a selected artwork against a suspected image
- **Verification History** - Searchable event log with payload-comparison details and downloadable reports
- **Responsive Navigation** - Sidebar layout with mobile menu support
- **Modern UI System** - Dark gradient sidebar, card-based content, gradient buttons, and polished form controls

### Backend API (FastAPI)

#### Health & Status
- **GET /health** - Service health check
- **GET /api/config** - Retrieve current configuration parameters
- **GET /api/dashboard/summary** - Dashboard counts and result statistics
- **GET /api/dashboard/recent-activity** - Recent registration and verification events

#### Artwork Registry
- **POST /api/artworks/register** - Register an artwork and embed a watermark
  - Multipart file upload
  - Title, creator name, and notes metadata
  - Saves original and watermarked files
  - Returns artwork details and a base64 preview image

- **GET /api/artworks** - List active registered artworks
- **GET /api/artworks/{artwork_id}** - Retrieve a single artwork record
- **PATCH /api/artworks/{artwork_id}/archive** - Unregister an artwork from active workflows while preserving provenance and files

#### Verification Workflow
- **POST /api/verifications** - Verify a suspected image against a selected artwork
  - Multipart file upload
  - BER calculation and result classification
  - Stores verification history

- **GET /api/verifications** - List verification records
- **GET /api/verifications/{verification_id}** - Retrieve verification details
- **GET /api/verifications/{verification_id}/report.csv** - Download a technical report with expected/extracted payloads and differing-bit count

#### Legacy Watermark Operations
- **POST /api/embed** - Embed watermark in uploaded image
  - Multipart file upload
  - Configurable delta
  - Optional custom payload
  - Returns watermarked image
  
- **POST /api/extract** - Extract watermark from image
  - Multipart file upload
  - Parameter specification
  - Returns payload, BER, confidence metrics

#### Image Handling
- Automatic 512×512 resizing
- Base64 image encoding for transmission
- File size validation (max 50MB)
- Format validation (JPEG, PNG)
- CORS enabled for development

### Frontend (React)

#### User Interface
- **Mobile-Responsive Design** - Optimized for mobile, tablet, desktop
- **Sidebar Navigation** - Persistent app navigation with active-state highlighting
- **Page-Based Routing** - Dashboard, artworks, register, verify, and history views

#### Image Management
- **Drag-and-Drop Upload** - Intuitive file selection for registration and verification
- **File Validation** - Type and size checking
- **Preview Display** - Watermarked and suspected image previews
- **Download Functionality** - Export reports and generated results

#### User Experience
- **Loading States** - Visual feedback during API calls
- **Error Alerts** - Clear error messaging
- **Success Notifications** - Operation completion feedback
- **Empty States** - Friendly guidance when records are missing
- **Responsive Buttons** - State-aware UI elements
- **Processing Feedback** - Visible workflow status during registration and verification

---

## Testing & Validation Features

### 1. **Unit Testing**
- Test coverage for all major modules
- Parametrized tests for different configurations
- Fixture-based test images
- 117+ passing tests

### 2. **Integration Testing**
- End-to-end embed/extract workflows
- Roundtrip testing (embed → extract → verify)
- Attack chain validation
- Configuration consistency checks

### 3. **Test Fixtures**
- Sample test images for development
- Deterministic test cases
- Reproducible test environments

### 4. **Performance Testing**
- Timing measurements
- Scalability validation
- Memory profiling capabilities

---

## Data Management Features

### 1. **Configuration System**
- YAML-based configuration
- Named experiment configurations
- Parameter inheritance
- Extensible configuration schema

### 2. **Result Tracking**
- Experiment logging
- Result CSV export
- Metadata preservation
- Per-image detailed metrics

### 3. **Documentation**
- Comprehensive README
- Codebase guide
- Experiment log templates
- API testing documentation
- Setup instructions for multiple OS

---

## Optional/Legacy Features

### Advanced Payload Security
- AES-256 encryption for payloads
- Reed-Solomon error correction codes
- Repetition coding for redundancy
- Custom provenance metadata encoding

### Spatial Domain Features
- Adaptive perceptual masking
- Tiled embedding for large images
- Crop synchronization
- False positive analysis

### Extended Testing
- Broad parameter sweeps
- Ablation study support
- Stress testing framework

---

## Deployment Features

### Docker Support
- Dockerfile for containerization
- Docker Compose configuration
- Multi-service orchestration (backend + frontend)

### Multiple Setup Methods
- Windows batch script setup
- Linux/macOS shell setup
- PowerShell setup script
- Manual installation instructions

### Environment Configuration
- .env-based configuration
- Environment templates
- Development vs. production modes

---

## Research & Reproducibility Features

### 1. **Deterministic Operations**
- Seeded random operations
- Reproducible coefficient selection
- Consistent attack application
- Version control integration

### 2. **Paper Alignment**
- Defaults matching published research specifications
- Documented parameter choices
- Rationale for feature selection
- Related work references

### 3. **Experiment Tracking**
- Experiment log template
- Delta calibration documentation
- CNN training history support
- Result archival

### 4. **Version Management**
- Git integration
- Dependency pinning (requirements.txt)
- Python version specification (3.10+)
- Optional dependency groups (dev, ml, stats)

---

## System Architecture Features

### 1. **Modular Design**
- Separate modules for each research component
- Clear separation between classical and CNN paths
- Extensible framework for new attacks/metrics

### 2. **Dependency Management**
- Core dependencies (wavelet, NumPy, PIL)
- Optional ML dependencies (TensorFlow)
- Optional statistical dependencies (scipy, scikit-posthoc)
- Development dependencies (pytest, pre-commit)

### 3. **Performance Optimization**
- NumPy vectorization for image operations
- Efficient DWT computation
- Batch processing support

### 4. **Error Handling**
- Comprehensive validation
- Clear error messages
- Graceful degradation
- Logging support

---

## Development Features

### 1. **Code Quality**
- Pre-commit hooks support
- Type hints (optional)
- Consistent code structure
- Documented functions

### 2. **Development Tools**
- pytest for testing
- CLI-based operations
- REPL-friendly modules
- Interactive configuration

### 3. **Documentation**
- Inline code comments
- Module docstrings
- Usage examples
- Research context explanation

---

## Summary of Key Capabilities

| Feature | Status | Type |
|---------|--------|------|
| DWT-QIM Embedding | ✅ Core | Classical |
| QIM Extraction | ✅ Core | Classical |
| CNN Extraction | ✅ Core | ML Optional |
| JPEG Compression | ✅ Core | Attack |
| Resizing Attacks | ✅ Core | Attack |
| Cropping Attacks | ✅ Core | Attack |
| Re-encoding | ✅ Core | Attack |
| BER Metrics | ✅ Core | Evaluation |
| Statistical Analysis | ✅ Core | Evaluation |
| Web API | ✅ Full Stack | Deployment |
| React Frontend | ✅ Full Stack | UI |
| CLI Interface | ✅ Core | User Interface |
| Docker Support | ✅ Deployment | DevOps |
| Extensible Framework | ✅ Core | Architecture |

