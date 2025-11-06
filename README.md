# Spectre: Weight Outlier Detection & Integrity Fingerprint Engine

**Version:** 1.0.0  
**Codename:** Spectre  
**Type:** Static Model-Weight Scan (no training data)

## Overview

Spectre is a static analyzer for detecting weight-level tampering in AI models (LLMs, Transformers, etc.). It identifies outliers, anomalies, and structural deviations in model weights without requiring training data or reference models.

### Key Features

- **Model-Agnostic**: Works with GPT, LLaMA, Mistral, Gemma, Phi, and other transformer architectures
- **Multiple Formats**: Supports `.safetensors`, `.pt`, and `.bin` checkpoints
- **14 Detectors**: Comprehensive suite of classical and advanced anomaly detection methods
- **Zero Training Data**: No access to training data required
- **High Interpretability**: Detailed outputs with visualizations and risk flags
- **CPU-Friendly**: Optimized for CPU execution (7B model in 20-25 minutes)

## Installation

### Prerequisites

- Python 3.12+
- `uv` package manager

### Setup

```bash
# Clone repository
git clone <repository-url>
cd model-stats

# Install dependencies using uv
uv sync
```

## Quick Start

### Basic Usage

You can run Spectre in several ways:

#### Option 1: Using uv (Recommended)

```bash
# Run scan with default configuration
uv run spectre/cli.py --model /path/to/model.safetensors

# Or use the entry point (after installation)
uv run spectre --model /path/to/model.safetensors

# Use custom configuration
uv run spectre/cli.py --config scan_config.yaml --model /path/to/model.safetensors

# Specify output directory
uv run spectre/cli.py --model /path/to/model.safetensors --output ./results

# Use specific ruleset
uv run spectre/cli.py --model /path/to/model.safetensors --ruleset llama_like

# Skip visualizations
uv run spectre/cli.py --model /path/to/model.safetensors --no-visuals
```

#### Option 2: Using Python directly

```bash
# After installing dependencies
python -m spectre.cli --model /path/to/model.safetensors

# Or directly
python spectre/cli.py --model /path/to/model.safetensors
```

#### Option 3: Programmatic Usage

```python
from pathlib import Path
from spectre.core.config import Config
from spectre.core.pipeline import Pipeline
from spectre.core.output import generate_all_outputs

# Create configuration
config = Config()
config.set("model.path", "/path/to/model.safetensors")
config.set("model.ruleset", "gpt_like")
config.set("output.dir", "./scan_out")

# Run pipeline
pipeline = Pipeline(config)
results = pipeline.run()

# Results are automatically saved, but you can also access them programmatically
print(f"Total tensors: {len(pipeline.feature_store.features)}")
print(f"Ensemble sigma: {pipeline.scorer.get_summary()['ensemble_sigma']}")
```

### Configuration

Create a `scan_config.yaml` file:

```yaml
model:
  path: /path/to/model.safetensors
  ruleset: gpt_like  # Options: gpt_like, llama_like, mistral_like, phi_like
  include: ["transformer.*"]
  exclude: []

svd:
  enabled: true
  rank: 96
  power_iters: 2

# ... (see scan_config.yaml for full configuration)
```

## Architecture

### Project Structure

```
spectre/
  io/
    loader.py          # Model checkpoint loading
    name_mapper.py     # Role-based parameter mapping
  detectors/
    spectral.py        # D1: Spectral signatures
    interlayer.py      # D2: Inter-layer correlation
    distribution.py    # D3: Distribution divergence
    robust.py          # D4: Robust outlier tests
    energy.py          # D5: Energy-norm anomaly
    layer_graph.py     # D6: Graph structural outlier
    rmt.py             # D7: Random Matrix Theory
    spectrogram.py     # D8: Spectrogram analysis
    gsp.py             # D9: Graph-signal processing
    tda.py             # D10: Topological Data Analysis
    sequence.py        # D11: Sequence change-point
    ot.py              # D12: Optimal Transport
    multiview.py       # D14: Multi-view adversarial
  core/
    config.py          # Configuration management
    features.py        # Feature extraction & storage
    scoring.py         # Standardization & ensemble scoring
    conformal.py       # D13: Conformal calibration
    pipeline.py        # Main orchestration
    output.py          # Output generation
  viz/
    heatmaps.py        # Similarity/z-score heatmaps
    spectra.py         # Spectral visualizations
    spectrogram.py     # Spectrogram plots
    graph.py           # Graph visualizations
  tests/
    test_detectors.py  # Unit tests
    test_attacks.py    # Synthetic attack tests
  cli.py               # Command-line interface
```

## Detector Suite

### Classical Detectors (D1-D6)

1. **D1 - Spectral Signatures**: Truncated/Randomized SVD per matrix
   - Features: singular values, stable rank, spectral decay, tail mass
   - Detects: low-rank implants, spectral spikes, heavy tails

2. **D2 - Inter-Layer Correlation**: Cosine similarity between layers
   - Features: cosine similarity, JSD distance, spectral histograms
   - Detects: broken continuity between layers, drift spikes

3. **D3 - Distribution Divergence**: Histograms, entropy, skew, kurtosis
   - Features: KLD to Gaussian/Laplace, Chebyshev tail fraction
   - Detects: unusual tail distributions, entropy anomalies

4. **D4 - Robust Outlier Tests**: Median & MAD scoring, Dixon's Q-test
   - Features: MAD z-scores, outlier fractions, IQR-based detection
   - Detects: sparse outliers, spiky malicious edits

5. **D5 - Energy-Norm Anomaly**: L1, L2, Frobenius norms
   - Features: row/column normalized energy, energy concentration
   - Detects: abnormal magnitude changes, localized energy injections

6. **D6 - Layer Graph**: Graph structural outlier detection
   - Features: LOF scores, kNN distances
   - Detects: structurally isolated layers

### Advanced Detectors (D7-D14)

7. **D7 - Random Matrix Theory**: Marchenko-Pastur distribution fit
   - Features: spike count above bulk edge, variance ratios
   - Detects: rank-1 hidden implants, unexpected noise profiles

8. **D8 - Spectrogram Analysis**: 2D STFT or wavelet spectrogram
   - Features: entropy, anisotropy, contrast
   - Detects: structured "textures" indicating manipulative patterns

9. **D9 - Graph-Signal Processing**: Neuron graph analysis
   - Features: graph Fourier transform, bandpass ratios
   - Detects: unusual neurons, dislocated attention heads

10. **D10 - Topological Data Analysis**: Persistent homology
    - Features: barcode entropy, total persistence, longest lifetimes
    - Detects: topological irregularities

11. **D11 - Sequence Change-Point**: Change-point detection
    - Features: change-point locations, matrix-profile discord
    - Detects: sudden layer shocks, local anomalies

12. **D12 - Optimal Transport**: Sliced Wasserstein distances
    - Features: Wasserstein distances between distributions
    - Detects: subtle tail shifts, mixed-distribution anomalies

13. **D13 - Conformal Calibration**: Distribution-free thresholding
    - Features: calibrated thresholds for 95% coverage
    - Ensures: controlled false-alarm rate

14. **D14 - Multi-View Adversarial**: Unsupervised subspace learning
    - Features: reconstruction error, distance to centroid
    - Detects: extremely subtle implants (default disabled)

## Outputs

### Fingerprint JSON

`fingerprint_v2.json` contains:
- Model metadata
- Summary statistics (ensemble sigma, flag, suspects)
- Per-tensor features
- Detector scores

### CSV Files

- `features.csv`: All extracted features for all tensors
- `per_layer_summary.csv`: Aggregated layer-level statistics

### Visualizations

- `similarity_heatmap.png`: Tensor similarity matrix
- `zscore_heatmap.png`: Detector z-scores heatmap
- `spectral_spectrum_<layer>.png`: Singular value spectrum
- `mp_fit_<layer>.png`: Marchenko-Pastur distribution fit
- `weight_spectrogram_<layer>.png`: Weight spectrogram
- `graph_signal_map_<role>.png`: Graph signal maps
- `sequence_changepoints_<role>.png`: Sequence change-points

## Risk Flags

- **GREEN**: ensemble < 2.0 (low risk)
- **AMBER**: ensemble 2.0-3.0 (moderate risk)
- **RED**: ensemble ≥ 3.0 (high risk)
- **HARD_RED**: any detector ≥ 4.5 (critical risk)

## Performance

- **7B model** (≈3.5k tensors): ≤ 20-25 min CPU
- Memory: Streams tensors to avoid >4GB RAM
- Optimizations: Randomized SVD, top-K suspect filtering

## Testing

Run tests:

```bash
uv run pytest spectre/tests/
```

Test suite includes:
- Unit tests for each detector
- Synthetic attack tests:
  - Low-rank injection
  - Sparse corruption
  - Column/row implant
  - Patch injection
  - Quantization error
  - Random noise

## Parameter Role Mapping

Spectre uses role-based abstraction independent of model parameter names:

| Role       | Meaning                      |
| ---------- | ---------------------------- |
| ATTN_IN    | QKV input projections        |
| ATTN_OUT   | Attention output projections  |
| MLP_FC     | Feed-forward W1 (or gate/up) |
| MLP_OUT    | Feed-forward W2 (or down)    |
| LN_GAMMA   | LayerNorm γ                  |
| LN_BETA    | LayerNorm β                  |
| EMB_TOK    | Token embedding              |
| EMB_POS    | Positional embedding         |
| FINAL_NORM | Last normalization           |
| OTHER      | Everything else              |

## Supported Rulesets

- `gpt_like`: GPT-style models
- `llama_like`: LLaMA-style models
- `mistral_like`: Mistral-style models
- `phi_like`: Phi/OLMo-style models

## License

[Specify license]

## Citation

If you use Spectre in your research, please cite:

```bibtex
@software{spectre2024,
  title={Spectre: Weight Outlier Detection \& Integrity Fingerprint Engine},
  author={LuminAI Security},
  year={2024},
  version={1.0.0}
}
```

## Contributing

[Contributing guidelines]

## Support

[Support information]

