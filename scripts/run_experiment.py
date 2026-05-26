"""Public runner for Carbon-TGAN experiments.

This script keeps the GitHub-facing workflow small:

  * train: train the main Carbon-TGAN model and generate synthetic samples
  * tstr: run downstream TRTR, TSTR, and TRTR+Aug prediction experiments
  * all: train the model, then run downstream evaluation on its samples

Datasets, outputs, checkpoints, and private run metadata are intentionally
not part of the repository. Provide CSV files locally under dataset/ or pass
--data_dir.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import create_dataloader
from src.models.carbon_tgan import CarbonTGAN
from src.utils.config import load_config_with_dataset
from src.utils.seed import log_environment, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

INPUT_FRAC = 0.75
ALL_DATASETS = ["ETTh1", "chinaCarbon", "usCarbon"]
ALL_PREDICTORS = ["LSTM", "GRU", "Transformer"]


def project_path(path: str) -> Path:
    """Resolve a path relative to the public project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def resolve_device(requested: Optional[str], config_device: str = "auto") -> str:
    """Resolve the requested device into a torch device string."""
    if requested:
        return requested
    if config_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return config_device


def load_training_config(args: argparse.Namespace) -> OmegaConf:
    """Load default config and apply CLI overrides."""
    cfg = load_config_with_dataset(
        dataset_name=args.dataset,
        override_path=args.config,
    )

    if args.dataset:
        cfg.data.dataset = args.dataset
    if args.data_dir:
        cfg.data.data_dir = args.data_dir
    if args.output_dir:
        cfg.output.dir = args.output_dir
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.batch_size:
        cfg.data.batch_size = args.batch_size
    if args.lr:
        cfg.training.lr = args.lr
    if args.seed is not None:
        cfg.seed = args.seed

    return cfg


def train_main_model(args: argparse.Namespace) -> Tuple[Path, str]:
    """Train Carbon-TGAN and return the run output directory and dataset."""
    cfg = load_training_config(args)
    set_seed(int(cfg.seed))

    device = resolve_device(args.device, str(cfg.device))
    if device == "cuda" and args.nondeterministic:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        logger.warning(
            "Non-deterministic CUDA mode enabled "
            "(cudnn.benchmark=True, cudnn.deterministic=False)."
        )
    logger.info("Using device: %s", device)

    output_root = project_path(str(cfg.output.dir))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / f"{cfg.data.dataset}_seed{cfg.seed}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.save(cfg, str(output_dir / "config.yaml"))
    if args.save_metadata:
        env_info = log_environment()
        with open(str(output_dir / "environment.json"), "w", encoding="utf-8") as f:
            json.dump(env_info, f, indent=2)
    logger.info("Output directory: %s", output_dir)

    dataloader, _, feature_names, n_features = create_dataloader(
        dataset_name=str(cfg.data.dataset),
        data_dir=str(project_path(str(cfg.data.data_dir))),
        window_size=int(cfg.data.window_size),
        stride=int(cfg.data.stride),
        batch_size=int(cfg.data.batch_size),
        num_workers=int(cfg.data.num_workers),
    )
    logger.info(
        "Dataset: %s, features: %d, batches: %d",
        cfg.data.dataset, n_features, len(dataloader),
    )

    if args.save_metadata:
        with open(str(output_dir / "features.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(feature_names))
        real_windows = dataloader.dataset.data.cpu().numpy()
        np.save(str(output_dir / "real_windows.npy"), real_windows)

    model_config = {
        "z_dim": cfg.generator.z_dim,
        "n_features": n_features,
        "seq_len": cfg.data.window_size,
        "d_model": cfg.generator.d_model,
        "nhead": cfg.generator.nhead,
        "num_layers": cfg.generator.num_layers,
        "dim_feedforward": cfg.generator.dim_feedforward,
        "dropout": cfg.generator.dropout,
        "hidden_dim": cfg.discriminator.hidden_dim,
        "lr": cfg.training.lr,
        "n_critic": cfg.training.n_critic,
        "lambda_gp": cfg.training.lambda_gp,
        "alpha": cfg.training.alpha,
        "beta": cfg.training.beta,
        "gamma": cfg.training.gamma,
        "delta": cfg.training.delta,
        "epochs": cfg.training.epochs,
        "pretrain_epochs": cfg.pretrain.epochs,
        "pretrain_lr": cfg.pretrain.lr,
        "device": device,
        "grad_clip": OmegaConf.select(cfg, "training.grad_clip", default=1.0),
        "ema_decay": OmegaConf.select(cfg, "training.ema_decay", default=0.999),
        "warmup_epochs": OmegaConf.select(cfg, "training.warmup_epochs", default=1000),
        "eta": OmegaConf.select(cfg, "training.eta", default=0.0),
        "spectral_norm": OmegaConf.select(
            cfg, "discriminator.spectral_norm", default=False,
        ),
    }

    model = CarbonTGAN(model_config)
    logger.info("Pre-training regressor...")
    model.pretrain_regressor(dataloader, epochs=model_config["pretrain_epochs"])

    logger.info("Training Carbon-TGAN...")
    history = model.train(
        dataloader,
        epochs=model_config["epochs"],
        log_every=int(cfg.output.log_every),
        save_every=int(cfg.output.save_every),
        save_dir=str(output_dir),
    )

    model.save(str(output_dir / "model_final.pt"))
    np.savez(
        str(output_dir / "history.npz"),
        d_loss=history.get("d_loss", []),
        g_loss=history.get("g_loss", []),
        w_distance=history.get("w_distance", []),
    )

    fake_data = model.generate(int(cfg.eval.n_generate))
    generated_path = output_dir / "generated_samples.npy"
    np.save(str(generated_path), fake_data)
    logger.info("Generated %d samples, shape: %s", len(fake_data), fake_data.shape)
    logger.info("Training complete. Results saved to %s", output_dir)
    return output_dir, str(cfg.data.dataset)


def split_steps(window_size: int) -> Tuple[int, int]:
    """Return input length and prediction horizon for a window size."""
    input_steps = max(1, int(round(window_size * INPUT_FRAC)))
    pred_horizon = window_size - input_steps
    if pred_horizon <= 0:
        raise ValueError(
            f"window_size={window_size} too small for INPUT_FRAC={INPUT_FRAC}"
        )
    return input_steps, pred_horizon


class LSTMPredictor(nn.Module):
    """LSTM sequence predictor used for downstream evaluation."""

    def __init__(
        self, n_features: int, pred_horizon: int, hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden_dim, num_layers=2,
            batch_first=True, dropout=0.1,
        )
        self.fc = nn.Linear(hidden_dim, n_features * pred_horizon)
        self.n_features = n_features
        self.pred_horizon = pred_horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        pred = self.fc(out[:, -1, :])
        return pred.view(-1, self.pred_horizon, self.n_features)


class GRUPredictor(nn.Module):
    """GRU sequence predictor used for downstream evaluation."""

    def __init__(
        self, n_features: int, pred_horizon: int, hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            n_features, hidden_dim, num_layers=2,
            batch_first=True, dropout=0.1,
        )
        self.fc = nn.Linear(hidden_dim, n_features * pred_horizon)
        self.n_features = n_features
        self.pred_horizon = pred_horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        pred = self.fc(out[:, -1, :])
        return pred.view(-1, self.pred_horizon, self.n_features)


class TransformerPredictor(nn.Module):
    """Transformer sequence predictor used for downstream evaluation."""

    def __init__(
        self, n_features: int, pred_horizon: int, d_model: int = 64,
        nhead: int = 4, num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(n_features, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(d_model, n_features * pred_horizon)
        self.n_features = n_features
        self.pred_horizon = pred_horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.encoder(h)
        pred = self.fc(h[:, -1, :])
        return pred.view(-1, self.pred_horizon, self.n_features)


PREDICTOR_FACTORY = {
    "LSTM": LSTMPredictor,
    "GRU": GRUPredictor,
    "Transformer": TransformerPredictor,
}


def prepare_prediction_data(
    windows: np.ndarray,
    input_steps: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split windows into input prefixes and prediction targets."""
    inputs = windows[:, :input_steps, :]
    targets = windows[:, input_steps:, :]
    return inputs, targets


def latest_generated_samples(
    output_dir: Path,
    dataset_name: str,
    generation_seed: Optional[int] = None,
) -> Optional[Path]:
    """Find the newest generated_samples.npy from the main model."""
    if generation_seed is None:
        pattern = f"{dataset_name}_*/generated_samples.npy"
    else:
        pattern = f"{dataset_name}_seed{generation_seed}_*/generated_samples.npy"
    candidates = list(output_dir.glob(pattern))
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def train_predictor(
    model: nn.Module,
    train_inputs: np.ndarray,
    train_targets: np.ndarray,
    device: str,
    epochs: int,
    batch_size: int,
    lr: float,
) -> nn.Module:
    """Train a downstream predictor."""
    if len(train_inputs) == 0:
        raise ValueError("No training windows available for downstream predictor.")

    model = model.to(device)
    model.train()

    dataset = TensorDataset(
        torch.FloatTensor(train_inputs),
        torch.FloatTensor(train_targets),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6,
    )
    criterion = nn.MSELoss()

    for _ in range(epochs):
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    return model


def evaluate_predictor(
    model: nn.Module,
    test_inputs: np.ndarray,
    test_targets: np.ndarray,
    device: str,
    batch_size: int = 64,
) -> Dict[str, float]:
    """Evaluate a downstream predictor with MAE and RMSE."""
    if len(test_inputs) == 0:
        raise ValueError("No test windows available for downstream predictor.")

    model.eval()
    all_preds = []
    all_targets = []

    dataset = TensorDataset(
        torch.FloatTensor(test_inputs),
        torch.FloatTensor(test_targets),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            pred = model(x_batch)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(y_batch.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    mae = np.mean(np.abs(preds - targets))
    rmse = np.sqrt(np.mean((preds - targets) ** 2))
    return {"mae": float(mae), "rmse": float(rmse)}


def run_single_downstream(
    dataset_name: str,
    predictor_name: str,
    seed: int,
    data_dir: Path,
    generated_path: Path,
    device: str,
    downstream_epochs: int,
    downstream_batch_size: int,
    downstream_lr: float,
) -> Dict[str, object]:
    """Run one TRTR/TSTR/TRTR+Aug downstream experiment."""
    set_seed(seed)

    cfg = load_config_with_dataset(dataset_name=dataset_name)
    window_size = int(cfg.data.window_size)
    stride = int(cfg.data.stride)
    input_steps, pred_horizon = split_steps(window_size)

    dataloader, _, _, n_features = create_dataloader(
        dataset_name=dataset_name,
        data_dir=str(data_dir),
        window_size=window_size,
        stride=stride,
        batch_size=9999,
        shuffle=False,
    )

    real_windows = []
    for batch in dataloader:
        real_windows.append(batch.numpy())
    real_windows = np.concatenate(real_windows, axis=0)

    n_train = int(len(real_windows) * 0.8)
    real_train = real_windows[:n_train]
    real_test = real_windows[n_train:]
    real_train_x, real_train_y = prepare_prediction_data(real_train, input_steps)
    real_test_x, real_test_y = prepare_prediction_data(real_test, input_steps)

    synthetic = np.load(str(generated_path))
    if synthetic.ndim != 3:
        raise ValueError(
            f"Generated samples must be 3-D, got shape {synthetic.shape}."
        )
    if synthetic.shape[1] != window_size or synthetic.shape[2] != n_features:
        raise ValueError(
            "Generated sample shape mismatch: expected "
            f"(*, {window_size}, {n_features}), got {synthetic.shape}."
        )
    syn_train_x, syn_train_y = prepare_prediction_data(synthetic, input_steps)

    predictor_cls = PREDICTOR_FACTORY[predictor_name]
    results: Dict[str, object] = {
        "dataset": dataset_name,
        "method": "Carbon-TGAN",
        "predictor": predictor_name,
        "seed": seed,
        "generated_samples_path": str(generated_path),
        "n_train_real": int(n_train),
        "n_test": int(len(real_test)),
    }

    model_trtr = predictor_cls(n_features, pred_horizon)
    model_trtr = train_predictor(
        model_trtr, real_train_x, real_train_y, device,
        downstream_epochs, downstream_batch_size, downstream_lr,
    )
    trtr_metrics = evaluate_predictor(model_trtr, real_test_x, real_test_y, device)
    results["trtr_mae"] = trtr_metrics["mae"]
    results["trtr_rmse"] = trtr_metrics["rmse"]

    model_tstr = predictor_cls(n_features, pred_horizon)
    model_tstr = train_predictor(
        model_tstr, syn_train_x, syn_train_y, device,
        downstream_epochs, downstream_batch_size, downstream_lr,
    )
    tstr_metrics = evaluate_predictor(model_tstr, real_test_x, real_test_y, device)
    results["tstr_mae"] = tstr_metrics["mae"]
    results["tstr_rmse"] = tstr_metrics["rmse"]

    n_aug = min(len(syn_train_x), len(real_train_x))
    aug_x = np.concatenate([real_train_x, syn_train_x[:n_aug]], axis=0)
    aug_y = np.concatenate([real_train_y, syn_train_y[:n_aug]], axis=0)
    model_aug = predictor_cls(n_features, pred_horizon)
    model_aug = train_predictor(
        model_aug, aug_x, aug_y, device,
        downstream_epochs, downstream_batch_size, downstream_lr,
    )
    aug_metrics = evaluate_predictor(model_aug, real_test_x, real_test_y, device)
    results["aug_mae"] = aug_metrics["mae"]
    results["aug_rmse"] = aug_metrics["rmse"]

    logger.info(
        "%s/%s seed=%d | TRTR: %.4f | TSTR: %.4f | Aug: %.4f",
        dataset_name, predictor_name, seed,
        results["trtr_mae"], results["tstr_mae"], results["aug_mae"],
    )
    return results


def run_downstream(
    args: argparse.Namespace,
    generated_path_override: Optional[Path] = None,
) -> Path:
    """Run downstream evaluation and return the JSON result path."""
    device = resolve_device(args.device)
    datasets = ALL_DATASETS if args.dataset == "all" else [args.dataset]
    predictors = args.predictors or ALL_PREDICTORS
    invalid = sorted(set(predictors) - set(ALL_PREDICTORS))
    if invalid:
        raise ValueError(f"Unknown predictors: {invalid}")

    output_dir = project_path(args.output_dir)
    data_dir = project_path(args.data_dir)
    all_results: List[Dict[str, object]] = []

    for dataset_name in datasets:
        if generated_path_override is not None:
            gen_path = generated_path_override
        elif args.generated_path:
            gen_path = project_path(args.generated_path)
        else:
            gen_path = latest_generated_samples(
                output_dir, dataset_name, args.generation_seed,
            )
        if gen_path is None or not gen_path.exists():
            raise FileNotFoundError(
                "Generated samples not found. Run `train` first or pass "
                "--generated_path explicitly."
            )

        for predictor_name in predictors:
            for seed in args.seeds:
                all_results.append(
                    run_single_downstream(
                        dataset_name=dataset_name,
                        predictor_name=predictor_name,
                        seed=seed,
                        data_dir=data_dir,
                        generated_path=gen_path,
                        device=device,
                        downstream_epochs=args.downstream_epochs,
                        downstream_batch_size=args.downstream_batch_size,
                        downstream_lr=args.downstream_lr,
                    )
                )

    save_dir = project_path(args.save_dir) if args.save_dir else output_dir / "tstr"
    save_dir.mkdir(parents=True, exist_ok=True)
    results_path = save_dir / "tstr_results.json"
    with open(str(results_path), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Downstream results saved to %s", results_path)
    print_summary(all_results, predictors)
    return results_path


def print_summary(results: List[Dict[str, object]], predictors: List[str]) -> None:
    """Print a compact downstream summary table."""
    print("\n" + "=" * 80)
    print("Downstream Results Summary")
    print("=" * 80)
    datasets = sorted({str(r["dataset"]) for r in results})
    for dataset_name in datasets:
        print(f"\n--- {dataset_name} ---")
        ds_results = [r for r in results if r["dataset"] == dataset_name]
        for predictor_name in predictors:
            rows = [r for r in ds_results if r["predictor"] == predictor_name]
            if not rows:
                continue
            trtr = np.mean([float(r["trtr_mae"]) for r in rows])
            tstr = np.mean([float(r["tstr_mae"]) for r in rows])
            aug = np.mean([float(r["aug_mae"]) for r in rows])
            print(
                f"  {predictor_name:12s} | TRTR: {trtr:.4f} "
                f"| TSTR: {tstr:.4f} | Aug: {aug:.4f}"
            )


def add_train_args(parser: argparse.ArgumentParser) -> None:
    """Add training arguments to a parser."""
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--nondeterministic", action="store_true",
        help="Enable cudnn.benchmark and disable deterministic CUDA.",
    )
    parser.add_argument(
        "--save_metadata", action="store_true",
        help="Also save environment, feature names, and real-window snapshots.",
    )


def add_downstream_args(parser: argparse.ArgumentParser) -> None:
    """Add downstream evaluation arguments to a parser."""
    parser.add_argument("--dataset", type=str, default="ETTh1")
    parser.add_argument("--data_dir", type=str, default="dataset")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--generated_path", type=str, default=None)
    parser.add_argument("--generation_seed", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--predictors", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--downstream_epochs", type=int, default=200)
    parser.add_argument("--downstream_batch_size", type=int, default=32)
    parser.add_argument("--downstream_lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default=None)


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description="Run Carbon-TGAN experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train Carbon-TGAN.")
    add_train_args(train_parser)

    tstr_parser = subparsers.add_parser(
        "tstr", help="Run TRTR/TSTR/TRTR+Aug downstream evaluation.",
    )
    add_downstream_args(tstr_parser)

    all_parser = subparsers.add_parser(
        "all", help="Train Carbon-TGAN, then run downstream evaluation.",
    )
    add_train_args(all_parser)
    all_parser.add_argument("--generated_path", type=str, default=None)
    all_parser.add_argument("--generation_seed", type=int, default=None)
    all_parser.add_argument("--save_dir", type=str, default=None)
    all_parser.add_argument("--predictors", nargs="+", default=None)
    all_parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    all_parser.add_argument("--downstream_epochs", type=int, default=200)
    all_parser.add_argument("--downstream_batch_size", type=int, default=32)
    all_parser.add_argument("--downstream_lr", type=float, default=1e-3)
    return parser


def main() -> None:
    """CLI entry point."""
    args = build_parser().parse_args()
    if args.command == "train":
        train_main_model(args)
    elif args.command == "tstr":
        run_downstream(args)
    elif args.command == "all":
        train_output_dir, trained_dataset = train_main_model(args)
        generated_path = train_output_dir / "generated_samples.npy"
        args.dataset = args.dataset or trained_dataset
        args.output_dir = args.output_dir or "outputs"
        args.data_dir = args.data_dir or "dataset"
        run_downstream(args, generated_path_override=generated_path)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
