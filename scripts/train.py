"""
Training Script for Multimodal Fake News Detector
====================================================
Main entry point for training. Supports:
    - Multimodal training (text + image + cross-modal fusion)
    - Text-only ablation
    - Image-only ablation
    - Multi-dataset training

Usage:
    python scripts/train.py --config config/config.yaml
    python scripts/train.py --config config/config.yaml --mode text_only
    python scripts/train.py --dataset gossipcop --epochs 30
"""

import os
import sys
import argparse
import yaml
import torch
import random
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import MultimodalFakeNewsDetector
from src.data.dataset import get_dataloader
from src.training.trainer import Trainer
from src.utils.runtime import resolve_path, resolve_path_list


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _contains_any(directory: Path, names: list[str]) -> bool:
    return any((directory / name).exists() for name in names)


def _looks_like_generic_dataset_dir(directory: Path) -> bool:
    if not directory.exists() or not directory.is_dir():
        return False
    names = {p.name.lower() for p in directory.iterdir() if p.is_file()}
    has_single = any(n in names for n in {"dataset.csv", "data.csv", "dataset.tsv", "data.tsv"})
    has_split = (
        any(n in names for n in {"train.csv", "train.tsv"})
        and any(n in names for n in {"test.csv", "test.tsv"})
    )
    return has_single or has_split


def auto_configure_data_source(config: dict, project_root: Path) -> dict:
    """
    Auto-resolve dataset source when configured path/files are missing.
    Priority:
      1) Keep existing config if files exist.
      2) Search under ./data for train/test or dataset CSV/TSV and switch to generic.
    """
    data_cfg = config["data"]
    if data_cfg.get("data_dirs"):
        resolved_dirs = resolve_path_list(data_cfg["data_dirs"], project_root)
        config["data"]["data_dirs"] = resolved_dirs
        config["data"]["data_dir"] = resolved_dirs[0]
        print(f"[DATA] Using configured data_dirs ({len(resolved_dirs)}): {resolved_dirs}")
        return config

    dataset_name = data_cfg.get("dataset_name", "generic")
    data_dir_cfg = data_cfg.get("data_dir", "./data")
    data_dir = resolve_path(data_dir_cfg, project_root)

    # Common user mistake: passing "data\\X" while dataset is at project root "X".
    if not data_dir.exists():
        alt_from_name = (project_root / Path(data_dir_cfg).name).resolve()
        if _looks_like_generic_dataset_dir(alt_from_name):
            config["data"]["dataset_name"] = "generic"
            config["data"]["data_dir"] = str(alt_from_name)
            print(f"[DATA] Corrected data_dir to existing folder: {alt_from_name}")
            return config

    if dataset_name == "isot" and _contains_any(data_dir, ["Fake.csv", "fake.csv"]) and _contains_any(data_dir, ["True.csv", "true.csv"]):
        return config
    if dataset_name == "generic" and _contains_any(data_dir, ["dataset.csv", "data.csv", "train.csv", "train.tsv"]):
        return config

    search_root = (project_root / "data").resolve()
    project_level = project_root.resolve()
    candidates = []

    for child in project_level.iterdir():
        if not child.is_dir() or child.name.startswith("."):
            continue
        if _looks_like_generic_dataset_dir(child):
            score = 1
            name_l = child.name.lower()
            if "isot" in name_l or "fake" in name_l:
                score = 2
            candidates.append((score, child))

    if search_root.exists():
        for child in search_root.iterdir():
            if not child.is_dir():
                continue
            if _looks_like_generic_dataset_dir(child):
                score = 1
                name_l = child.name.lower()
                if "isot" in name_l or "fake" in name_l:
                    score = 2
                candidates.append((score, child))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = candidates[0][1]
        config["data"]["dataset_name"] = "generic"
        config["data"]["data_dir"] = str(chosen)
        print(f"[DATA] Auto-selected dataset folder: {chosen}")
        print("[DATA] Using adapter: generic")

    return config


def main():
    parser = argparse.ArgumentParser(description="Train Multimodal Fake News Detector")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset names for multi-dataset training")
    parser.add_argument("--data_dirs", nargs="+", default=None,
                        help="Data directories for multi-dataset training")
    parser.add_argument("--mode", type=str, default="multimodal",
                        choices=["multimodal", "text_only", "image_only"],
                        help="Training mode (for ablation studies)")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint path")
    parser.add_argument("--auto_resume", action="store_true",
                        help="Resume from logging.checkpoint_dir/latest_model.pt if it exists")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(project_root, args.config) if not os.path.isabs(args.config) else args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Apply CLI overrides
    if args.dataset:
        config["data"]["dataset_name"] = args.dataset
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
        config["data"].pop("data_dirs", None)
    if args.datasets:
        config["data"]["dataset_names"] = args.datasets
    if args.data_dirs:
        config["data"]["data_dirs"] = args.data_dirs
        config["data"]["data_dir"] = args.data_dirs[0]
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr

    config = auto_configure_data_source(config, project_root)

    # Set seed
    seed = config["training"].get("seed", 42)
    set_seed(seed)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'#' * 60}")
    print(f"  Multimodal Fake News Detection - Training")
    print(f"{'#' * 60}")
    active_dataset = config["data"].get("dataset_names", config["data"]["dataset_name"])
    active_data_dir = config["data"].get("data_dirs", config["data"]["data_dir"])
    print(f"  Mode      : {args.mode}")
    print(f"  Dataset   : {active_dataset}")
    print(f"  Data dir  : {active_data_dir}")
    print(f"  Device    : {device}")
    print(f"  Epochs    : {config['training']['num_epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  LR        : {config['training']['learning_rate']}")
    print(f"  Seed      : {seed}")
    print(f"{'#' * 60}\n")

    # --- Create DataLoaders ---
    data_cfg = config["data"]
    dataloaders = get_dataloader(
        data_dir=data_cfg.get("data_dirs", data_cfg["data_dir"]),
        dataset_name=data_cfg.get("dataset_names", data_cfg["dataset_name"]),
        tokenizer_name=config["model"]["text_encoder"]["name"],
        max_length=config["model"]["text_encoder"]["max_length"],
        image_size=config["model"]["image_encoder"]["image_size"],
        batch_size=config["training"]["batch_size"],
        val_split=config["training"].get("val_split", 0.15),
        test_split=config["training"].get("test_split", 0.15),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
        seed=seed,
    )

    # --- Create Model ---
    model = MultimodalFakeNewsDetector(config)
    params = model.get_trainable_params()
    print(f"Model Parameters:")
    print(f"  Total      : {params['total']:>12,}")
    print(f"  Trainable  : {params['trainable']:>12,}")
    print(f"  Frozen     : {params['frozen']:>12,}")
    print(f"  Trainable %: {params['trainable_pct']:>11.1f}%\n")

    # --- Create Trainer ---
    trainer = Trainer(model=model, config=config, device=device)

    # Resume from checkpoint if specified (or auto-detected latest checkpoint)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    elif args.auto_resume:
        checkpoint_dir = Path(config.get("logging", {}).get("checkpoint_dir", "./checkpoints"))
        latest_checkpoint = checkpoint_dir / "latest_model.pt"
        if latest_checkpoint.exists():
            trainer.load_checkpoint(str(latest_checkpoint))
            print(f"[CHECKPOINT] Auto-resumed from {latest_checkpoint}")

    # --- Train ---
    training_result = trainer.train(
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        mode=args.mode,
    )

    # --- Final Evaluation on Test Set ---
    print("\n\nRunning final evaluation on test set...")
    test_metrics = trainer.evaluate(
        test_loader=dataloaders["test"],
        mode=args.mode,
        generate_plots=True,
    )

    # Save final results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    import json
    results = {
        "mode": args.mode,
        "dataset": active_dataset,
        "data_dir": active_data_dir,
        "sources": dataloaders.get("sources", []),
        "training": training_result,
        "test_metrics": {k: v for k, v in test_metrics.items()
                        if not isinstance(v, (np.ndarray, list)) or k in ["f1_per_class", "confusion_matrix"]},
    }

    result_suffix = "combined" if isinstance(active_dataset, list) else str(active_dataset)
    results_path = results_dir / f"results_{args.mode}_{result_suffix}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
