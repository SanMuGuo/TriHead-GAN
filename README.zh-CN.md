<div align="center">

# 🌍 TriHead-GAN

**用于碳排放时间序列生成的、带三头判别器的生成对抗网络**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[English](README.md) | **简体中文**

</div>

---

**TriHead-GAN** 是一个面向**数据稀缺的多变量碳排放时间序列生成**而专门设计的、
基于 Transformer 的对抗框架。它由一个 **基于 Transformer 的生成器** 与一个
**三头判别器（Triple-Head Discriminator）** 组成；三个判别头由三条并行的 CNN 分支分别支撑，
从联合序列分布的**边缘**、**条件**与**转移**三个互补维度同时进行监督：

- **D-Head**（分布真实性）—— 独立的 3 层、带谱归一化的 1D-CNN 分支之上的 Wasserstein critic，
  在 WGAN-GP 框架下稳定地约束边缘分布。
- **R-Head**（跨变量依赖）—— 独立的 3 层 1D-CNN 分支，**仅以去掉目标列**的非目标特征作为输入
  （无泄漏），在每个时间步回归目标变量（如 CO₂ 浓度），强制变量间的结构性一致。
- **T-Head**（时间动态）—— 独立的 2 层**因果** 1D-CNN 分支，预测相邻时间步差分，约束序列的逐步
  转移规律。

生成器把随机噪声 $\mathbf{z}\!\in\!\mathbb{R}^{T\times d_z}$ 依次经过线性投影 + 正弦位置编码、
$L$ 层 Transformer 编码层（建模全局时间依赖）、带残差的局部时序卷积模块（补充细粒度局部动态）、
逐时间步可学习幅度的噪声注入（增加时间多样性），最终通过 Tanh 输出约束到 $[-1, 1]$。训练采用
WGAN-GP 框架，所有辅助损失权重按 epoch 线性 warmup；并引入 **anti-smoothing 损失**，同时匹配
真实与生成序列在每个特征上 **绝对一阶差分** 分布的 **均值与标准差**，避免生成器将局部波动塌缩
成一条过平滑的窄带。

## 🧭 方法概览

<div align="center">
  <img src="assets/architecture.png" alt="TriHead-GAN 整体架构" width="820">
  <br>
  <sub><em>Transformer 生成器输出的时间序列经判别器中三条并行 CNN 分支分别处理，分别送入 D-Head（WGAN 真实性）、R-Head（无泄漏的跨变量回归）与 T-Head（因果时间一致性）。</em></sub>
</div>

## 📁 目录结构

```text
.
├── configs/                 # 各数据集的 YAML 覆盖配置
│   ├── etth1.yaml
│   ├── china_carbon.yaml
│   └── us_carbon.yaml
├── scripts/
│   └── run_experiment.py    # 命令行入口：train / tstr / all
├── src/
│   ├── data/                # 预处理、滑动窗口、DataLoader 工厂
│   ├── models/              # 生成器、判别器、DNN 回归器、CarbonTGAN 训练管理器
│   ├── evaluation/          # 质量指标 + 可视化
│   └── utils/               # 配置加载、随机种子、GPU 批缓存
├── requirements.txt
└── README.md
```

## ⚙️ 安装

需要 **Python ≥ 3.9** 与 **PyTorch ≥ 2.0**。

```bash
# （推荐）创建隔离环境
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

GPU 训练请安装与显卡驱动匹配的 CUDA 版 PyTorch（参见 <https://pytorch.org/get-started/locally/>）。

## 📦 数据准备

论文中使用的三份公开数据集**已随仓库一同提供**，存放在 `dataset/` 目录下，开箱即可
复现公开数据上的实验。如果你把数据放到了其它位置，可通过 `--data_dir` 指定：

```text
dataset/
├── ETTh1.csv
├── chinaCarbon.csv
└── usCarbon.csv
```

| 数据集        | 特征数 | 说明                          |
|---------------|------:|-------------------------------|
| `ETTh1`       | 7     | 公开的电力变压器温度数据集    |
| `chinaCarbon` | 7     | 中国碳排放相关多变量序列      |
| `usCarbon`    | 7     | 美国碳排放相关多变量序列      |

> 很抱歉，论文中使用的 `changshaCarbon` 数据集来自一个仍在进行中的研究项目，暂时无法
> 随本仓库一同公开。基于上述三份公开数据集，即可完整复现论文中公开数据上的所有实验。

### 数据来源

仓库中分发的 CSV 在上游数据基础上做了轻量预处理，使用时请引用原始来源：

- **ETTh1** —— 取自 [Time Series Library (TSLib)](https://github.com/thuml/Time-Series-Library) 的数据包，原始来源为 [ETDataset](https://github.com/zhouhaoyi/ETDataset) 项目。
- **chinaCarbon / usCarbon** —— 基于 [Carbon Monitor](https://carbonmonitor.org) 发布的近实时 CO₂ 排放数据构建。

### CSV 格式约定

由 `src/data/preprocessing.py` 自动处理：

- 若存在 `date` 列会被自动删除。
- **目标列固定为最后一列**。若存在名为 `OT` 的列会被自动移到最后；否则把当前最后一列当作目标。
  R-Head 会用其余特征来预测该目标列。
- 缺失值用线性插值 + 前向/后向填充补齐。
- 所有列被 min-max 归一化到 `[0, 1]`。

## 🚀 使用方法

`scripts/run_experiment.py` 提供三个子命令。

### 1. 训练

```bash
python scripts/run_experiment.py train --dataset ETTh1
```

该命令训练 TriHead-GAN，并在 `outputs/` 下生成带时间戳的运行目录，包含最终 checkpoint
和一批生成样本。

常用覆盖参数：

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

| 参数                 | 说明                                                       |
|----------------------|------------------------------------------------------------|
| `--config`           | 显式 YAML 覆盖（最高优先级）                               |
| `--dataset`          | `ETTh1`、`chinaCarbon` 或 `usCarbon`                       |
| `--data_dir`         | CSV 所在目录（默认 `dataset`）                             |
| `--output_dir`       | 输出根目录（默认 `outputs`）                               |
| `--epochs / --batch_size / --lr / --seed` | 训练超参覆盖                          |
| `--device`           | `cuda`、`cpu`，不填则自动检测                              |
| `--nondeterministic` | 启用 `cudnn.benchmark`（更快，但不可复现）                |
| `--save_metadata`    | 额外保存 `environment.json`、`features.txt`、`real_windows.npy` |

### 2. 下游评估（TRTR / TSTR / TRTR+Aug）

衡量合成数据对预测任务的有用程度：

- **TRTR** —— 真实训练、真实测试（基准）
- **TSTR** —— 合成训练、真实测试（保真度）
- **TRTR+Aug** —— 真实+合成训练、真实测试（增强收益）

```bash
python scripts/run_experiment.py tstr \
    --dataset ETTh1 \
    --predictors LSTM GRU Transformer \
    --seeds 42 123 456
```

用 `--dataset all` 可一次性遍历 `ETTh1`、`chinaCarbon`、`usCarbon`。脚本会自动选取最新的
`generated_samples.npy`，也可用 `--generated_path` 指定具体文件。结果写入
`outputs/tstr/tstr_results.json`。

### 3. 训练 + 评估一键完成

```bash
python scripts/run_experiment.py all --dataset ETTh1
```

先训练模型，随后立即在其生成样本上运行下游评估。

## 🔧 配置

配置按优先级分层（从高到低）：

1. `--config <path>` —— 显式 YAML 覆盖
2. `configs/` 中自动检测到的数据集配置（按 `data.dataset` 匹配）
3. `src/utils/config.py` 中的内置默认值

各数据集 YAML 只携带数据集相关字段（窗口大小、步长），其余回退到默认值。任何字段都可在命令行覆盖。

关键默认值：`window_size=24`、`stride=12`、`epochs=1000`、`batch_size=32`、`lr=1e-4`、
`n_critic=5`、`lambda_gp=10`，生成器 `d_model=128 / nhead=8 / num_layers=4`，默认开启谱归一化，
`n_generate=1000`。

## 📂 输出

一次训练运行会产生：

```text
outputs/<dataset>_seed<seed>_<timestamp>/
├── config.yaml             # 解析后的完整配置
├── model_final.pt          # 最终 checkpoint（推荐用于分析）
├── model_best.pt           # 最小 Wasserstein 距离的 checkpoint*
├── model_epoch_*.pt        # 周期性 checkpoint
├── history.npz             # d_loss / g_loss / w_distance
└── generated_samples.npy   # (n_generate, seq_len, n_features)，取值 ∈ [0, 1]
```

\* `model_best.pt` 按最小正 Wasserstein 距离选取，这是判别器状态指标，并非生成质量代理。
分析与汇报时建议优先使用 `model_final.pt`。

## 📊 质量指标（Python API）

`src.evaluation` 模块提供分布级与结构级指标，作为下游 TSTR 评估的补充：

```python
import numpy as np
from src.evaluation import evaluate_all, plot_tsne, plot_acf_comparison

real = np.load("outputs/<run>/real_windows.npy")      # 需训练时加 --save_metadata
fake = np.load("outputs/<run>/generated_samples.npy")

scores = evaluate_all(real, fake)   # discriminative / predictive / MMD / FID / ACF 差异
print(scores)

plot_tsne(real, fake, save_path="tsne.png")
plot_acf_comparison(real, fake, feature_idx=-1, save_path="acf.png")
```

所有指标遵循**越低越好**的约定（discriminative score 为 `|accuracy − 0.5|`）。

## 🔁 可复现性

`src/utils/seed.py` 会为 Python、NumPy、PyTorch 设置随机种子，并默认启用确定性 cuDNN。
用 `--seed` 修改种子，用 `--save_metadata` 在每次运行旁记录运行环境。仅在明确需要用可复现性
换取速度时才使用 `--nondeterministic`。

## 🙏 鸣谢

本项目的实现与对比实验参考了以下开源工作，在此向各作者致谢：

- **[Time Series Library (TSLib)](https://github.com/thuml/Time-Series-Library)** ——
  深度时间序列模型，以及用于获取 ETTh1 的数据包。
- **[ETDataset](https://github.com/zhouhaoyi/ETDataset)** —— ETTh1 数据集的原始来源。
- **[Carbon Monitor](https://carbonmonitor.org)** —— 碳排放数据集所依据的近实时 CO₂ 排放数据。

实验中对比的 baseline 方法：

| 方法 | 仓库 | 来源 |
|------|------|------|
| TimeGAN | [jsyoon0823/TimeGAN](https://github.com/jsyoon0823/TimeGAN) | NeurIPS 2019 |
| RCGAN (RGAN) | [ratschlab/RGAN](https://github.com/ratschlab/RGAN) | 2017 |
| TTS-GAN | [imics-lab/tts-gan](https://github.com/imics-lab/tts-gan) | AIME 2022 |
| Diffusion-TS | [Y-debug-sys/Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS) | ICLR 2024 |
| PaD-TS | [wmd3i/PaD-TS](https://github.com/wmd3i/PaD-TS) | AAAI 2025 |
| TimeDP | [YukhoY/TimeDP](https://github.com/YukhoY/TimeDP) | AAAI 2025 |

## 📄 许可证

本项目基于 [MIT License](LICENSE) 发布。

## 📚 引用

如果本代码对你的研究有帮助，请引用：

```bibtex
@misc{wang2026trihead,
  title  = {TriHead-GAN: A Generative Adversarial Network with Triple-Head
            Discriminator for Carbon Emission Time Series Generation},
  author = {Wang, Zesen},
  year   = {2026},
  note   = {https://github.com/SanMuGuo/TriHead-GAN}
}
```
