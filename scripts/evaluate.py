"""
Evaluation & Ablation Study Script
=====================================
Evaluates trained models on test sets and runs comparative ablation studies
across modalities and datasets.

Usage:
    # Evaluate a single model
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt

    # Run full ablation study (multimodal vs text-only vs image-only)
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --ablation

    # Evaluate on a specific dataset
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --dataset gossipcop
"""

import os
import sys
import json
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models import MultimodalFakeNewsDetector
from src.data.dataset import get_dataloader
from src.training.trainer import Trainer
from src.training.metrics import MetricsCalculator, compare_models


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
    data_cfg = config["data"]
    dataset_name = data_cfg.get("dataset_name", "generic")
    data_dir_cfg = data_cfg.get("data_dir", "./data")
    data_dir = Path(data_dir_cfg)
    if not data_dir.is_absolute():
        data_dir = (project_root / data_dir).resolve()

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


def evaluate_model(
    checkpoint_path: str,
    config: dict,
    dataset_name: str = None,
    data_dir: str = None,
    mode: str = "multimodal",
    device: torch.device = None,
    generate_plots: bool = True,
) -> dict:
    """
    Evaluate a trained model on a test set.

    Returns:
        Metrics dictionary
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Override dataset if specified
    if dataset_name:
        config["data"]["dataset_name"] = dataset_name
    if data_dir:
        config["data"]["data_dir"] = data_dir

    # Load data
    dataloaders = get_dataloader(
        data_dir=config["data"]["data_dir"],
        dataset_name=config["data"]["dataset_name"],
        tokenizer_name=config["model"]["text_encoder"]["name"],
        max_length=config["model"]["text_encoder"]["max_length"],
        image_size=config["model"]["image_encoder"]["image_size"],
        batch_size=config["training"]["batch_size"],
        val_split=config["training"].get("val_split", 0.15),
        test_split=config["training"].get("test_split", 0.15),
        seed=config["training"].get("seed", 42),
    )

    # Load model
    model = MultimodalFakeNewsDetector(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"[EVAL] Model loaded from {checkpoint_path}")

    # Evaluate
    trainer = Trainer(model=model, config=config, device=device)
    metrics = trainer.evaluate(
        test_loader=dataloaders["test"],
        mode=mode,
        generate_plots=generate_plots,
    )

    return metrics


def run_ablation_study(
    checkpoint_path: str,
    config: dict,
    device: torch.device = None,
) -> dict:
    """
    Run full ablation study comparing multimodal vs unimodal performance.

    Tests:
        1. Multimodal (text + image + cross-modal attention)
        2. Text-only (BERT only)
        3. Image-only (ViT only)
    """
    print(f"\n{'=' * 60}")
    print(f"  ABLATION STUDY")
    print(f"{'=' * 60}\n")

    modes = ["multimodal", "text_only", "image_only"]
    all_results = {}

    for mode in modes:
        print(f"\n{'─' * 40}")
        print(f"  Evaluating: {mode.upper()}")
        print(f"{'─' * 40}")

        metrics = evaluate_model(
            checkpoint_path=checkpoint_path,
            config=config,
            mode=mode,
            device=device,
            generate_plots=True,
        )

        all_results[mode] = metrics

    # Print comparison table
    print(f"\n\n{'=' * 80}")
    print(f"  ABLATION STUDY RESULTS")
    print(f"{'=' * 80}")
    print(f"{'Mode':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'AUC-ROC':<12}")
    print(f"{'─' * 80}")

    for mode, metrics in all_results.items():
        auc = f"{metrics.get('auc_roc', 'N/A'):.4f}" if isinstance(metrics.get('auc_roc'), float) else "N/A"
        print(
            f"{mode:<20} "
            f"{metrics['accuracy']:<12.4f} "
            f"{metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f} "
            f"{metrics['f1']:<12.4f} "
            f"{auc:<12}"
        )
    print(f"{'=' * 80}\n")

    # Generate comparison plot
    compare_models(all_results, save_dir="./results")

    return all_results


def run_multi_dataset_evaluation(
    checkpoint_path: str,
    config: dict,
    datasets: list,
    device: torch.device = None,
) -> dict:
    """
    Evaluate the model across multiple benchmark datasets.
    """
    print(f"\n{'=' * 60}")
    print(f"  MULTI-DATASET EVALUATION")
    print(f"{'=' * 60}\n")

    all_results = {}

    for dataset_name in datasets:
        print(f"\n{'─' * 40}")
        print(f"  Dataset: {dataset_name}")
        print(f"{'─' * 40}")

        data_dir = os.path.join(config["data"]["data_dir"], dataset_name)
        if not os.path.exists(data_dir):
            data_dir = config["data"]["data_dir"]

        try:
            metrics = evaluate_model(
                checkpoint_path=checkpoint_path,
                config=config,
                dataset_name=dataset_name,
                data_dir=data_dir,
                mode="multimodal",
                device=device,
                generate_plots=True,
            )
            all_results[dataset_name] = metrics
        except Exception as e:
            print(f"[WARNING] Failed to evaluate on {dataset_name}: {e}")

    # Print comparison
    if all_results:
        print(f"\n\n{'=' * 80}")
        print(f"  MULTI-DATASET RESULTS (Multimodal)")
        print(f"{'=' * 80}")
        print(f"{'Dataset':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
        print(f"{'─' * 80}")

        for dataset, metrics in all_results.items():
            print(
                f"{dataset:<20} "
                f"{metrics['accuracy']:<12.4f} "
                f"{metrics['precision']:<12.4f} "
                f"{metrics['recall']:<12.4f} "
                f"{metrics['f1']:<12.4f}"
            )
        print(f"{'=' * 80}\n")

        compare_models(all_results, save_dir="./results")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multimodal Fake News Detector")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to evaluate on")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    parser.add_argument("--mode", type=str, default="multimodal",
                        choices=["multimodal", "text_only", "image_only"])
    parser.add_argument("--ablation", action="store_true", help="Run full ablation study")
    parser.add_argument("--multi_dataset", nargs="+", default=None,
                        help="Evaluate across multiple datasets")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(project_root, args.config) if not os.path.isabs(args.config) else args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    config = auto_configure_data_source(config, project_root)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ablation:
        # Full ablation study
        results = run_ablation_study(args.checkpoint, config, device)

    elif args.multi_dataset:
        # Multi-dataset evaluation
        results = run_multi_dataset_evaluation(
            args.checkpoint, config, args.multi_dataset, device
        )

    else:
        # Single evaluation
        results = evaluate_model(
            checkpoint_path=args.checkpoint,
            config=config,
            dataset_name=args.dataset,
            data_dir=args.data_dir,
            mode=args.mode,
            device=device,
        )

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    def _serialize(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return str(obj)

    results_path = results_dir / "evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=_serialize)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
