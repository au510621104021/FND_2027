"""
Evaluation & Ablation Study Script (Conference-Level)
========================================================
Evaluates trained models on test sets and runs comparative ablation studies
across modalities and datasets, with statistical significance testing.

Features:
    - Single model evaluation with full metrics (MCC, Kappa, AUC)
    - Ablation study (multimodal vs text-only vs image-only)
    - Multi-dataset benchmark evaluation
    - Bootstrap confidence intervals
    - McNemar's test for paired model comparison
    - LaTeX table generation for papers
    - Publication-ready plots (PDF + PNG at 300 DPI)

Usage:
    # Evaluate a single model
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt

    # Run full ablation study with statistical tests
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --ablation

    # Evaluate with bootstrap confidence intervals
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --bootstrap

    # Generate LaTeX results table
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --ablation --latex
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
from src.utils.runtime import resolve_path, resolve_path_list


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
    if data_cfg.get("data_dirs"):
        resolved_dirs = resolve_path_list(data_cfg["data_dirs"], project_root)
        config["data"]["data_dirs"] = resolved_dirs
        config["data"]["data_dir"] = resolved_dirs[0]
        print(f"[DATA] Using configured data_dirs ({len(resolved_dirs)}): {resolved_dirs}")
        return config

    dataset_name = data_cfg.get("dataset_name", "generic")
    data_dir_cfg = data_cfg.get("data_dir", "./data")
    data_dir = resolve_path(data_dir_cfg, project_root)

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
    compute_bootstrap: bool = False,
) -> dict:
    """
    Evaluate a trained model on a test set with full conference-level metrics.

    Returns:
        Metrics dictionary including MCC, Cohen's Kappa, and optionally bootstrap CIs.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Override dataset if specified
    if dataset_name:
        config["data"]["dataset_name"] = dataset_name
    if data_dir:
        config["data"]["data_dir"] = data_dir
        config["data"].pop("data_dirs", None)

    # Load data
    dataloaders = get_dataloader(
        data_dir=config["data"].get("data_dirs", config["data"]["data_dir"]),
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

    # Compute bootstrap confidence intervals if requested
    if compute_bootstrap:
        stat_cfg = config.get("evaluation", {}).get("statistical_tests", {})
        n_iter = stat_cfg.get("bootstrap_n_iterations", 1000)
        ci_level = stat_cfg.get("confidence_level", 0.95)

        print(f"\n  Computing Bootstrap {ci_level*100:.0f}% CIs ({n_iter} iterations)...")
        bootstrap_cis = trainer.metrics_calc.compute_all_bootstrap_cis(
            n_iterations=n_iter,
            confidence_level=ci_level,
        )
        metrics["bootstrap_ci"] = bootstrap_cis

        # Print bootstrap results
        print(f"\n  {'Metric':20s} {'Mean':>10s} {'95% CI':>22s}")
        print(f"  {'─' * 54}")
        for metric_name, ci in bootstrap_cis.items():
            print(f"  {metric_name:20s} {ci['mean']:10.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]")

    return metrics


def run_ablation_study(
    checkpoint_path: str,
    config: dict,
    device: torch.device = None,
    compute_bootstrap: bool = False,
    generate_latex: bool = False,
) -> dict:
    """
    Run full ablation study comparing multimodal vs unimodal performance
    with statistical significance tests.

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
    all_preds = {}
    all_labels_stored = None

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
            compute_bootstrap=compute_bootstrap,
        )

        all_results[mode] = metrics

    # Print comparison table
    print(f"\n\n{'=' * 90}")
    print(f"  ABLATION STUDY RESULTS")
    print(f"{'=' * 90}")
    header = f"{'Mode':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'MCC':<12} {'AUC-ROC':<12}"
    print(header)
    print(f"{'─' * 90}")

    for mode, metrics in all_results.items():
        auc = f"{metrics.get('auc_roc', 'N/A'):.4f}" if isinstance(metrics.get('auc_roc'), float) else "N/A"
        mcc = f"{metrics.get('mcc', 0):.4f}"
        print(
            f"{mode:<20} "
            f"{metrics['accuracy']:<12.4f} "
            f"{metrics['precision']:<12.4f} "
            f"{metrics['recall']:<12.4f} "
            f"{metrics['f1']:<12.4f} "
            f"{mcc:<12} "
            f"{auc:<12}"
        )
    print(f"{'=' * 90}\n")

    # Generate comparison plots
    results_dir = Path("./results")
    results_dir.mkdir(exist_ok=True)
    compare_models(all_results, save_dir=str(results_dir))

    # Generate radar chart
    metrics_calc = MetricsCalculator(save_dir=str(results_dir))
    metrics_calc.plot_metrics_radar(all_results, filename="ablation_radar.png")

    # Generate LaTeX table
    if generate_latex:
        metrics_calc.generate_latex_table(
            all_results,
            caption="Ablation Study Results on Test Set",
            label="tab:ablation",
            filename="ablation_table.tex",
        )

    return all_results


def run_multi_dataset_evaluation(
    checkpoint_path: str,
    config: dict,
    datasets: list,
    device: torch.device = None,
    generate_latex: bool = False,
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
        print(f"\n\n{'=' * 90}")
        print(f"  MULTI-DATASET RESULTS (Multimodal)")
        print(f"{'=' * 90}")
        header = f"{'Dataset':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'MCC':<12}"
        print(header)
        print(f"{'─' * 90}")

        for dataset, metrics in all_results.items():
            mcc = f"{metrics.get('mcc', 0):.4f}"
            print(
                f"{dataset:<20} "
                f"{metrics['accuracy']:<12.4f} "
                f"{metrics['precision']:<12.4f} "
                f"{metrics['recall']:<12.4f} "
                f"{metrics['f1']:<12.4f} "
                f"{mcc:<12}"
            )
        print(f"{'=' * 90}\n")

        compare_models(all_results, save_dir="./results")

        # Generate LaTeX table
        if generate_latex:
            metrics_calc = MetricsCalculator(save_dir="./results")
            metrics_calc.generate_latex_table(
                all_results,
                caption="Cross-Dataset Evaluation Results",
                label="tab:multi_dataset",
                filename="multi_dataset_table.tex",
            )

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Multimodal Fake News Detector (Conference-Level)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Config file path")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset to evaluate on")
    parser.add_argument("--data_dir", type=str, default=None, help="Data directory")
    parser.add_argument("--data_dirs", type=str, nargs="+", default=None, help="Multiple data directories")
    parser.add_argument("--mode", type=str, default="multimodal",
                        choices=["multimodal", "text_only", "image_only"])
    parser.add_argument("--ablation", action="store_true", help="Run full ablation study")
    parser.add_argument("--multi_dataset", nargs="+", default=None,
                        help="Evaluate across multiple datasets")
    parser.add_argument("--bootstrap", action="store_true", help="Compute bootstrap CIs")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX results tables")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(project_root, args.config) if not os.path.isabs(args.config) else args.config
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
        config["data"].pop("data_dirs", None)
    if args.data_dirs:
        config["data"]["data_dirs"] = args.data_dirs
        config["data"]["data_dir"] = args.data_dirs[0]
    config = auto_configure_data_source(config, project_root)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.ablation:
        # Full ablation study
        results = run_ablation_study(
            args.checkpoint, config, device,
            compute_bootstrap=args.bootstrap,
            generate_latex=args.latex,
        )

    elif args.multi_dataset:
        # Multi-dataset evaluation
        results = run_multi_dataset_evaluation(
            args.checkpoint, config, args.multi_dataset, device,
            generate_latex=args.latex,
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
            compute_bootstrap=args.bootstrap,
        )

        # Generate LaTeX table for single evaluation
        if args.latex:
            metrics_calc = MetricsCalculator(save_dir="./results")
            metrics_calc.generate_latex_table(
                {args.mode: results},
                caption=f"Evaluation Results ({args.mode})",
                label="tab:eval_results",
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
