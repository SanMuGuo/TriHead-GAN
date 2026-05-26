<div align="center">

# 🌍 TriHead-GAN

**A Generative Adversarial Network with Triple-Head Discriminator for Carbon Emission Time Series Generation**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**English** | [简体中文](README.zh-CN.md)

</div>

---

**TriHead-GAN** is a Transformer-based adversarial framework specifically designed
for **multivariate carbon emission time series generation under data scarcity**.
It pairs a **Transformer-based generator** with a **Triple-Head Discriminator**
that simultaneously supervises three complementary aspects of the joint sequence
distribution — its *marginal*, *conditional*, and *transition* components —
through three parallel CNN branches, each feeding its own task-specific head:

- **D-Head** — *distributional authenticity.* A Wasserstein critic on top of a
  dedicated 3-layer 1D-CNN with spectral normalization, providing stable WGAN-GP
  training of the marginal distribution.
- **R-Head** — *cross-variable dependency.* A separate 3-layer 1D-CNN branch fed
  only with the **non-target** features (leakage-free input), regressing the
  target variable (e.g., CO₂ concentration) at each step to enforce
  inter-variable structural consistency.
- **T-Head** — *step-wise temporal coherence.* A separate 2-layer **causal**
  1D-CNN branch that predicts adjacent-step differences, constraining the local
  transition dynamics of the sequence.

The generator processes random noise $\mathbf{z}\!\in\!\mathbb{R}^{T\times d_z}$
through linear projection + sinusoidal positional encoding, $L$ Transformer
encoder layers (for global temporal dependencies), a local temporal convolution
module with a residual connection (for fine-grained local dynamics), and
learnable-scale per-step noise injection (for temporal diversity), with a final
Tanh output bounding samples to $[-1, 1]$. Training follows the WGAN-GP framework
with linearly warmed-up auxiliary loss weights, augmented by an **anti-smoothing
loss** that matches both the **mean and standard deviation** of the per-feature
absolute first-difference distribution — preventing the generator from collapsing
local variability into an over-smoothed band.

## 🧭 Method overview

<div align="center">
  <img src="assets/architecture.png" alt="TriHead-GAN overall architecture" width="820">
  <br>
  <sub><em>The Transformer generator emits time series that the discriminator processes through three parallel CNN branches, each feeding its own head: D-Head (WGAN authenticity), R-Head (cross-variable regression with leakage-free input), and T-Head (causal temporal coherence).</em></sub>
</div>

## 📁 Repository structure

```text
.
├── configs/                 # Per-dataset YAML overrides
│   ├── etth1.yaml
│   ├── china_carbon.yaml
│   └── us_carbon.yaml
├── scripts/
│   └── run_experiment.py    # CLI entry point: train / tstr / all
├── src/
│   ├── data/                # Preprocessing, sliding windows, DataLoader factory
│   ├── models/              # Generator, Discriminator, DNN regressor, CarbonTGAN trainer
│   ├── evaluation/          # Quality metrics + visualization
│   └── utils/               # Config loading, seeding, GPU batch cache
├── requirements.txt
└── README.md
```

## ⚙️ Installation

Requires **Python ≥ 3.9** and **PyTorch ≥ 2.0**.

```bash
# (recommended) create an isolated environment
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

For GPU training, install the CUDA-enabled PyTorch build that matches your driver
(see <https://pytorch.org/get-started/locally/>).

## 📦 Data preparation

The three public datasets used in our experiments are **shipped with this repository**
under `dataset/`, so no extra download is required to reproduce the public-data results.
If you keep them elsewhere, pass the directory via `--data_dir`:

```text
dataset/
├── ETTh1.csv
├── chinaCarbon.csv
└── usCarbon.csv
```

| Dataset       | Features | Notes                                              |
|---------------|---------:|----------------------------------------------------|
| `ETTh1`       | 7        | Public Electricity Transformer Temperature dataset |
| `chinaCarbon` | 7        | Multivariate carbon-emission series                |
| `usCarbon`    | 7        | Multivariate carbon-emission series                |

> We sincerely apologize that the `changshaCarbon` dataset reported in the paper
> cannot be released here, as it was collected under an ongoing research project
> and is not yet available for public distribution. The three public datasets
> shipped above are sufficient to reproduce all public-data experiments in the
> paper.

### Data sources

The bundled CSVs are redistributed (with light preprocessing) from the following
upstream releases — please cite the original sources if you use them:

- **ETTh1** — from the
  [Time Series Library (TSLib)](https://github.com/thuml/Time-Series-Library) data pack,
  originally released by the [ETDataset](https://github.com/zhouhaoyi/ETDataset) project.
- **chinaCarbon / usCarbon** — constructed from the near-real-time CO₂ emission data
  published by [Carbon Monitor](https://carbonmonitor.org).

### CSV format conventions

Handled automatically by `src/data/preprocessing.py`:

- An optional `date` column is dropped.
- The **target column is the last column**. A column named `OT`, if present, is moved
  to the last position automatically; otherwise the existing last column is treated as
  the target. The R-Head predicts this target from the remaining features.
- Missing values are filled by linear interpolation + forward/backward fill.
- All columns are min-max scaled to `[0, 1]`.

## 🚀 Usage

The CLI exposes three subcommands via `scripts/run_experiment.py`.

### 1. Train

```bash
python scripts/run_experiment.py train --dataset ETTh1
```

This trains TriHead-GAN and writes a timestamped run directory under `outputs/`,
containing the final checkpoint and a set of generated samples.

Common overrides:

```bash
python scripts/run_experiment.py train \
    --dataset chinaCarbon \
    --data_dir dataset \
    --output_dir outputs \
    --epochs 1000 \
    --batch_size 32 \
    --lr 1e-4 \
    --seed 42 \
    --device cuda
```

| Flag                 | Description                                                       |
|----------------------|------------------------------------------------------------------|
| `--config`           | Explicit YAML override (highest priority)                        |
| `--dataset`          | `ETTh1`, `chinaCarbon`, or `usCarbon`                            |
| `--data_dir`         | Directory containing the CSV files (default `dataset`)           |
| `--output_dir`       | Output root (default `outputs`)                                  |
| `--epochs / --batch_size / --lr / --seed` | Training overrides                          |
| `--device`           | `cuda`, `cpu`, or leave unset for auto-detection                 |
| `--nondeterministic` | Enable `cudnn.benchmark` (faster, non-reproducible)             |
| `--save_metadata`    | Also dump `environment.json`, `features.txt`, `real_windows.npy`|

### 2. Downstream evaluation (TRTR / TSTR / TRTR+Aug)

Measure how useful the synthetic data is for a forecasting task:

- **TRTR** — train on real, test on real (reference)
- **TSTR** — train on synthetic, test on real (fidelity)
- **TRTR+Aug** — train on real + synthetic, test on real (augmentation gain)

```bash
python scripts/run_experiment.py tstr \
    --dataset ETTh1 \
    --predictors LSTM GRU Transformer \
    --seeds 42 123 456
```

Use `--dataset all` to sweep `ETTh1`, `chinaCarbon`, and `usCarbon`. The newest
`generated_samples.npy` is selected automatically, or pass `--generated_path` to pin
a specific file. Results are written to `outputs/tstr/tstr_results.json`.

### 3. Train + evaluate in one shot

```bash
python scripts/run_experiment.py all --dataset ETTh1
```

This trains the model and immediately runs downstream evaluation on its samples.

## 🔧 Configuration

Configuration is layered (highest priority first):

1. `--config <path>` — explicit YAML override
2. Auto-detected per-dataset config in `configs/` (matched by `data.dataset`)
3. Built-in defaults in `src/utils/config.py`

The per-dataset YAMLs only carry dataset-specific fields (window size, stride);
everything else falls back to the defaults. Any field can be overridden on the
command line.

Key defaults: `window_size=24`, `stride=12`, `epochs=1000`, `batch_size=32`,
`lr=1e-4`, `n_critic=5`, `lambda_gp=10`, generator `d_model=128 / nhead=8 /
num_layers=4`, spectral normalization enabled, `n_generate=1000`.

## 📂 Outputs

A training run produces:

```text
outputs/<dataset>_seed<seed>_<timestamp>/
├── config.yaml             # resolved configuration
├── model_final.pt          # final checkpoint (recommended for analysis)
├── model_best.pt           # min-Wasserstein-distance checkpoint*
├── model_epoch_*.pt        # periodic checkpoints
├── history.npz             # d_loss / g_loss / w_distance
└── generated_samples.npy   # (n_generate, seq_len, n_features), values in [0, 1]
```

\* `model_best.pt` is selected by minimum positive Wasserstein distance, which is a
discriminator-state metric rather than a generation-quality proxy. Prefer
`model_final.pt` for downstream analysis and reporting.

## 📊 Quality metrics (Python API)

The `src.evaluation` module provides distribution- and structure-level metrics that
complement the downstream TSTR evaluation:

```python
import numpy as np
from src.evaluation import evaluate_all, plot_tsne, plot_acf_comparison

real = np.load("outputs/<run>/real_windows.npy")      # needs --save_metadata
fake = np.load("outputs/<run>/generated_samples.npy")

scores = evaluate_all(real, fake)   # discriminative, predictive, MMD, FID, ACF diff
print(scores)

plot_tsne(real, fake, save_path="tsne.png")
plot_acf_comparison(real, fake, feature_idx=-1, save_path="acf.png")
```

All metrics follow a *lower-is-better* convention (discriminative score is
`|accuracy − 0.5|`).

## 🔁 Reproducibility

`src/utils/seed.py` seeds Python, NumPy, and PyTorch and enables deterministic cuDNN
by default. Pass `--seed` to change the seed and `--save_metadata` to record the
runtime environment alongside each run. Use `--nondeterministic` only when you
explicitly trade reproducibility for speed.

## 🙏 Acknowledgements

This project builds on and compares against the following open-source work — we thank
their authors:

- **[Time Series Library (TSLib)](https://github.com/thuml/Time-Series-Library)** —
  deep time-series models and the dataset pack used to obtain ETTh1.
- **[ETDataset](https://github.com/zhouhaoyi/ETDataset)** — original source of the
  ETTh1 dataset.
- **[Carbon Monitor](https://carbonmonitor.org)** — near-real-time CO₂ emission data
  underlying the carbon datasets.

Baseline methods compared in our experiments:

| Method | Repository | Venue |
|--------|------------|-------|
| TimeGAN | [jsyoon0823/TimeGAN](https://github.com/jsyoon0823/TimeGAN) | NeurIPS 2019 |
| RCGAN (RGAN) | [ratschlab/RGAN](https://github.com/ratschlab/RGAN) | 2017 |
| TTS-GAN | [imics-lab/tts-gan](https://github.com/imics-lab/tts-gan) | AIME 2022 |
| Diffusion-TS | [Y-debug-sys/Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS) | ICLR 2024 |
| PaD-TS | [wmd3i/PaD-TS](https://github.com/wmd3i/PaD-TS) | AAAI 2025 |
| TimeDP | [YukhoY/TimeDP](https://github.com/YukhoY/TimeDP) | AAAI 2025 |

## 📄 License

This project is released under the [MIT License](LICENSE).

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@misc{wang2026trihead,
  title  = {TriHead-GAN: A Generative Adversarial Network with Triple-Head
            Discriminator for Carbon Emission Time Series Generation},
  author = {Wang, Zesen},
  year   = {2026},
  note   = {https://github.com/SanMuGuo/TriHead-GAN}
}
```
