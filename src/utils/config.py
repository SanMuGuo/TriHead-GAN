"""Configuration utilities for dataset-specific config auto-loading."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import OmegaConf

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_CONFIG: Dict[str, Any] = {
    "seed": 42,
    "device": "auto",
    "data": {
        "dataset": "ETTh1",
        "data_dir": "dataset",
        "window_size": 24,
        "stride": 12,
        "batch_size": 32,
        "num_workers": 0,
    },
    "generator": {
        "z_dim": 64,
        "d_model": 128,
        "nhead": 8,
        "num_layers": 4,
        "dim_feedforward": 256,
        "dropout": 0.1,
    },
    "discriminator": {
        "hidden_dim": 128,
        "spectral_norm": True,
    },
    "training": {
        "epochs": 3000,
        "lr": 0.0001,
        "n_critic": 5,
        "lambda_gp": 10.0,
        "alpha": 3.0,
        "beta": 1.0,
        "gamma": 3.0,
        "delta": 1.0,
        "eta": 2.0,
        "grad_clip": 1.0,
        "ema_decay": 0.999,
        "warmup_epochs": 300,
    },
    "pretrain": {
        "epochs": 0,
        "lr": 0.001,
    },
    "eval": {
        "n_generate": 1000,
        "metrics": [
            "discriminative_score",
            "predictive_score",
            "mmd",
            "fid",
            "acf_diff",
        ],
    },
    "output": {
        "dir": "outputs",
        "save_every": 500,
        "log_every": 50,
    },
}


def find_dataset_config(dataset_name: str) -> Optional[Path]:
    """Find the dataset-specific config file by matching data.dataset field.

    Scans all YAML files in configs/ and returns the first one whose
    ``data.dataset`` matches *dataset_name*.

    Args:
        dataset_name: The dataset name to look up (e.g. "ETTh1").

    Returns:
        Path to the matching config file, or None if not found.
    """
    configs_dir = PROJECT_ROOT / "configs"
    for path in sorted(configs_dir.glob("*.yaml")):
        try:
            cfg = OmegaConf.load(str(path))
            if OmegaConf.select(cfg, "data.dataset") == dataset_name:
                return path
        except Exception:
            continue
    return None


def load_config_with_dataset(
    dataset_name: Optional[str] = None,
    override_path: Optional[str] = None,
) -> OmegaConf:
    """Load default config, auto-merge dataset-specific overrides.

    Priority (highest to lowest):
        1. Explicit override_path (--config CLI arg)
        2. Auto-detected dataset config (matched by dataset_name)
        3. Built-in public defaults

    Args:
        dataset_name: Dataset name for auto-detection.
        override_path: Explicit path to an override config file.

    Returns:
        Merged OmegaConf configuration.
    """
    default_cfg = OmegaConf.create(DEFAULT_CONFIG)

    if override_path:
        override_cfg = OmegaConf.load(override_path)
        cfg = OmegaConf.merge(default_cfg, override_cfg)
        logger.info("Loaded explicit config override: %s", override_path)
    elif dataset_name:
        ds_config_path = find_dataset_config(dataset_name)
        if ds_config_path:
            override_cfg = OmegaConf.load(str(ds_config_path))
            cfg = OmegaConf.merge(default_cfg, override_cfg)
            logger.info(
                "Auto-loaded dataset config: %s for dataset '%s'",
                ds_config_path.name, dataset_name,
            )
        else:
            cfg = default_cfg
            logger.info(
                "No dataset-specific config found for '%s', using defaults.",
                dataset_name,
            )
    else:
        cfg = default_cfg

    return cfg
