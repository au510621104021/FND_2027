"""
K-Fold Cross-Validation Training Script (Conference-Level)
=============================================================
Implements stratified k-fold cross-validation with multiple independent
runs for statistically rigorous evaluation. Produces mean ± std results
across folds and generates LaTeX-formatted tables.

This is ESSENTIAL for conference-level publications as it demonstrates
result robustness and statistical reliability.

Usage:
    # 5-fold CV with 3 independent runs
    python scripts/train_kfold.py --config config/config.yaml

    # Custom folds and runs
    python scripts/train_kfold.py --config config/config.yaml --n_folds 10 --n_runs 5

    # Ablation with k-fold
    python scripts/train_kfold.py --config config/config.yaml --ablation
"""

import os
import sys
import argparse
import yaml
import json
import copy
import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset

from src.models import MultimodalFakeNewsDetector
from src.data.dataset import MultimodalFakeNewsDataset, get_adapter, _collate_fn
from src.training.trainer import Trainer
from src.training.metrics import (
    MetricsCalculator,
    compare_models,
    aggregate_fold_results,
)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_single_fold(
    model_config: dict,
    train_dataset,
    val_dataset,
    test_indices,
    full_dataset,
    fold: int,
    mode: str,
    device: torch.device,
    results_dir: Path,
) -> dict:
    """
    Train and evaluate on a single fold.

    Returns:
        Metrics dict for this fold.
    """
    train_cfg = model_config["training"]
    data_cfg = model_config["data"]

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=data_cfg.get("pin_memory", True),
        drop_last=True,
        collate_fn=_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=data_cfg.get("pin_memory", True),
        collate_fn=_collate_fn,
    )

    # Create fresh model for each fold
    model = MultimodalFakeNewsDetector(model_config)

    # Override checkpoint/log dirs for this fold
    fold_config = copy.deepcopy(model_config)
    fold_checkpoint_dir = results_dir / f"fold_{fold}" / "checkpoints"
    fold_log_dir = results_dir / f"fold_{fold}" / "logs"
    fold_config["logging"]["checkpoint_dir"] = str(fold_checkpoint_dir)
    fold_config["logging"]["log_dir"] = str(fold_log_dir)

    # Train
    trainer = Trainer(model=model, config=fold_config, device=device)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        mode=mode,
    )

    # Evaluate on held-out test portion using validation set
    test_metrics = trainer.evaluate(
        test_loader=val_loader,
        mode=mode,
        generate_plots=False,  # Don't generate per-fold plots
    )

    return test_metrics


def run_kfold_cv(
    config: dict,
    n_folds: int = 5,
    n_runs: int = 3,
    mode: str = "multimodal",
    device: torch.device = None,
    results_dir: Path = None,
) -> dict:
    """
    Run stratified k-fold cross-validation with multiple independent runs.

    Args:
        config: Full configuration dict
        n_folds: Number of folds (default: 5)
        n_runs: Number of independent runs (default: 3)
        mode: Training mode
        device: Compute device
        results_dir: Directory to save results

    Returns:
        Aggregated results with mean ± std across runs and folds.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = results_dir or Path("results") / "kfold"
    results_dir.mkdir(parents=True, exist_ok=True)

    data_cfg = config["data"]

    print(f"\n{'#' * 60}")
    print(f"  K-Fold Cross-Validation Experiment")
    print(f"{'#' * 60}")
    print(f"  Mode       : {mode}")
    print(f"  K-Folds    : {n_folds}")
    print(f"  Runs       : {n_runs}")
    print(f"  Dataset    : {data_cfg['dataset_name']}")
    print(f"  Device     : {device}")
    print(f"{'#' * 60}\n")

    # Load the full dataset once
    adapter = get_adapter(data_cfg["dataset_name"])
    samples = adapter.load(data_cfg["data_dir"])

    if len(samples) == 0:
        raise ValueError(f"No samples found for dataset '{data_cfg['dataset_name']}' in '{data_cfg['data_dir']}'")

    labels = np.array([s["label"] for s in samples])

    all_run_results = []

    for run_idx in range(n_runs):
        run_seed = config["training"].get("seed", 42) + run_idx
        set_seed(run_seed)

        print(f"\n{'=' * 60}")
        print(f"  RUN {run_idx + 1}/{n_runs} (seed={run_seed})")
        print(f"{'=' * 60}")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=run_seed)
        fold_results = []

        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(np.zeros(len(samples)), labels)):
            print(f"\n  --- Fold {fold_idx + 1}/{n_folds} ---")
            print(f"  Train: {len(train_indices)} | Val: {len(val_indices)}")

            # Create dataset subsets
            train_samples = [samples[i] for i in train_indices]
            val_samples = [samples[i] for i in val_indices]

            train_dataset = MultimodalFakeNewsDataset(
                data_dir=data_cfg["data_dir"],
                dataset_name=data_cfg["dataset_name"],
                tokenizer_name=config["model"]["text_encoder"]["name"],
                max_length=config["model"]["text_encoder"]["max_length"],
                image_size=config["model"]["image_encoder"]["image_size"],
                train=True,
                samples=train_samples,
            )

            val_dataset = MultimodalFakeNewsDataset(
                data_dir=data_cfg["data_dir"],
                dataset_name=data_cfg["dataset_name"],
                tokenizer_name=config["model"]["text_encoder"]["name"],
                max_length=config["model"]["text_encoder"]["max_length"],
                image_size=config["model"]["image_encoder"]["image_size"],
                train=False,
                samples=val_samples,
            )

            fold_dir = results_dir / f"run_{run_idx}" / f"fold_{fold_idx}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            metrics = run_single_fold(
                model_config=config,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_indices=val_indices,
                full_dataset=None,
                fold=fold_idx,
                mode=mode,
                device=device,
                results_dir=fold_dir,
            )

            fold_results.append(metrics)

            # Print fold result
            print(f"  Fold {fold_idx + 1} Results: "
                  f"Acc={metrics['accuracy']:.4f} | "
                  f"F1={metrics['f1']:.4f} | "
                  f"MCC={metrics.get('mcc', 0):.4f}")

        all_run_results.append(fold_results)

    # Aggregate all results
    all_fold_results = [m for run in all_run_results for m in run]
    aggregated = aggregate_fold_results(all_fold_results)

    # Print final summary
    print(f"\n\n{'=' * 80}")
    print(f"  K-FOLD CROSS-VALIDATION SUMMARY ({mode})")
    print(f"  {n_folds}-Fold CV × {n_runs} Runs = {n_folds * n_runs} total evaluations")
    print(f"{'=' * 80}")

    mean_m = aggregated["mean"]
    std_m = aggregated["std"]
    for key in ["accuracy", "precision", "recall", "f1", "f1_macro", "mcc", "cohens_kappa", "auc_roc"]:
        if key in mean_m:
            print(f"  {key:20s}: {mean_m[key]:.4f} ± {std_m[key]:.4f}")
    print(f"{'=' * 80}\n")

    # Save detailed results
    results_path = results_dir / f"kfold_results_{mode}.json"
    serializable_results = {
        "mode": mode,
        "n_folds": n_folds,
        "n_runs": n_runs,
        "mean": aggregated["mean"],
        "std": aggregated["std"],
        "timestamp": datetime.now().isoformat(),
    }
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2, default=str)
    print(f"Results saved to {results_path}")

    # Generate LaTeX table
    metrics_calc = MetricsCalculator(save_dir=str(results_dir))
    metrics_calc.generate_latex_table_with_ci(
        results={mode: aggregated},
        caption=f"{n_folds}-Fold Cross-Validation Results ({mode})",
        label=f"tab:kfold_{mode}",
        filename=f"kfold_{mode}_table.tex",
    )

    return aggregated


def run_kfold_ablation(
    config: dict,
    n_folds: int = 5,
    n_runs: int = 3,
    device: torch.device = None,
) -> dict:
    """
    Run k-fold CV ablation study across all three modes (multimodal, text_only, image_only).

    Returns:
        dict mapping mode -> aggregated results
    """
    results_dir = Path("results") / "kfold_ablation"
    results_dir.mkdir(parents=True, exist_ok=True)

    modes = ["multimodal", "text_only", "image_only"]
    all_results = {}

    for mode in modes:
        print(f"\n\n{'*' * 60}")
        print(f"  ABLATION: {mode.upper()}")
        print(f"{'*' * 60}")

        aggregated = run_kfold_cv(
            config=config,
            n_folds=n_folds,
            n_runs=n_runs,
            mode=mode,
            device=device,
            results_dir=results_dir / mode,
        )
        all_results[mode] = aggregated

    # Generate combined LaTeX table
    metrics_calc = MetricsCalculator(save_dir=str(results_dir))
    metrics_calc.generate_latex_table_with_ci(
        results=all_results,
        caption=f"Ablation Study: {n_folds}-Fold Cross-Validation (mean $\\pm$ std)",
        label="tab:ablation_kfold",
        filename="ablation_kfold_table.tex",
    )

    # Generate radar chart for comparison
    mean_results = {mode: data["mean"] for mode, data in all_results.items()}
    metrics_calc.plot_metrics_radar(
        mean_results,
        filename="ablation_radar.png",
    )

    # Perform statistical significance tests between multimodal and unimodal
    if "multimodal" in all_results and "text_only" in all_results:
        mm_f1s = [r["f1"] for r in all_results["multimodal"]["per_fold"]]
        text_f1s = [r["f1"] for r in all_results["text_only"]["per_fold"]]
        t_test = MetricsCalculator.paired_t_test(mm_f1s, text_f1s)
        print(f"\n  Paired t-Test (Multimodal vs Text-Only):")
        print(f"    t-statistic: {t_test['t_statistic']:.4f}")
        print(f"    p-value    : {t_test['p_value']:.6f}")
        print(f"    Significant: {'Yes' if t_test['significant'] else 'No'} (α=0.05)")

    if "multimodal" in all_results and "image_only" in all_results:
        mm_f1s = [r["f1"] for r in all_results["multimodal"]["per_fold"]]
        img_f1s = [r["f1"] for r in all_results["image_only"]["per_fold"]]
        t_test = MetricsCalculator.paired_t_test(mm_f1s, img_f1s)
        print(f"\n  Paired t-Test (Multimodal vs Image-Only):")
        print(f"    t-statistic: {t_test['t_statistic']:.4f}")
        print(f"    p-value    : {t_test['p_value']:.6f}")
        print(f"    Significant: {'Yes' if t_test['significant'] else 'No'} (α=0.05)")

    # Save all results
    final_path = results_dir / "ablation_kfold_results.json"
    serializable = {}
    for mode, data in all_results.items():
        serializable[mode] = {"mean": data["mean"], "std": data["std"]}
    with open(final_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nAll ablation results saved to {final_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="K-Fold CV Training for Fake News Detection")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--n_folds", type=int, default=None, help="Number of folds (override config)")
    parser.add_argument("--n_runs", type=int, default=None, help="Number of independent runs (override config)")
    parser.add_argument("--mode", type=str, default="multimodal",
                        choices=["multimodal", "text_only", "image_only"])
    parser.add_argument("--ablation", action="store_true", help="Run full ablation with k-fold CV")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset name")
    parser.add_argument("--data_dir", type=str, default=None, help="Override data directory")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(project_root, args.config) if not os.path.isabs(args.config) else args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.dataset:
        config["data"]["dataset_name"] = args.dataset
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir

    cv_cfg = config.get("cross_validation", {})
    n_folds = args.n_folds or cv_cfg.get("n_folds", 5)
    n_runs = args.n_runs or cv_cfg.get("n_runs", 3)

    # Auto-resolve data directory
    from train import auto_configure_data_source
    config = auto_configure_data_source(config, project_root)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    seed = config["training"].get("seed", 42)
    set_seed(seed)

    if args.ablation:
        run_kfold_ablation(config, n_folds=n_folds, n_runs=n_runs, device=device)
    else:
        run_kfold_cv(config, n_folds=n_folds, n_runs=n_runs, mode=args.mode, device=device)


if __name__ == "__main__":
    main()
