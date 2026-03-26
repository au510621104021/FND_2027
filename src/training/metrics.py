"""
Evaluation Metrics for Fake News Detection
============================================
Computes comprehensive classification metrics including:
    - Accuracy, Precision, Recall, F1-Score
    - AUC-ROC
    - Confusion Matrix
    - Per-class metrics
    - Confidence-based analysis
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from pathlib import Path


class MetricsCalculator:
    """
    Comprehensive metrics calculator for binary fake news classification.

    Tracks predictions across batches, computes aggregate metrics,
    and generates publication-ready visualization plots.
    """

    CLASS_NAMES = ["Real", "Fake"]

    def __init__(self, average: str = "weighted", save_dir: str = "./results"):
        self.average = average
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self):
        """Clear accumulated predictions."""
        self.all_labels = []
        self.all_preds = []
        self.all_probs = []

    def update(self, labels: np.ndarray, preds: np.ndarray, probs: np.ndarray = None):
        """
        Accumulate a batch of predictions.

        Args:
            labels: Ground truth labels [B]
            preds: Predicted labels [B]
            probs: Prediction probabilities [B, num_classes] (optional)
        """
        self.all_labels.extend(labels.tolist() if hasattr(labels, "tolist") else labels)
        self.all_preds.extend(preds.tolist() if hasattr(preds, "tolist") else preds)
        if probs is not None:
            self.all_probs.extend(
                probs.tolist() if hasattr(probs, "tolist") else probs
            )

    def compute(self) -> dict:
        """
        Compute all metrics on accumulated predictions.

        Returns:
            Dictionary with all computed metrics.
        """
        labels = np.array(self.all_labels)
        preds = np.array(self.all_preds)

        metrics = {
            "accuracy": accuracy_score(labels, preds),
            "precision": precision_score(labels, preds, average=self.average, zero_division=0),
            "recall": recall_score(labels, preds, average=self.average, zero_division=0),
            "f1": f1_score(labels, preds, average=self.average, zero_division=0),
            "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
            "f1_per_class": f1_score(labels, preds, average=None, zero_division=0).tolist(),
            "precision_per_class": precision_score(labels, preds, average=None, zero_division=0).tolist(),
            "recall_per_class": recall_score(labels, preds, average=None, zero_division=0).tolist(),
            "confusion_matrix": confusion_matrix(labels, preds).tolist(),
            "classification_report": classification_report(
                labels, preds, target_names=self.CLASS_NAMES, zero_division=0
            ),
            "num_samples": len(labels),
        }

        # AUC-ROC (requires probabilities)
        if len(self.all_probs) > 0:
            probs = np.array(self.all_probs)
            if probs.ndim == 2 and probs.shape[1] == 2:
                metrics["auc_roc"] = roc_auc_score(labels, probs[:, 1])
                metrics["avg_precision"] = average_precision_score(labels, probs[:, 1])
            elif probs.ndim == 1:
                metrics["auc_roc"] = roc_auc_score(labels, probs)
                metrics["avg_precision"] = average_precision_score(labels, probs)

        return metrics

    def print_report(self, metrics: dict = None, title: str = "Evaluation Results"):
        """Print a formatted metrics report."""
        if metrics is None:
            metrics = self.compute()

        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        print(f"  Accuracy    : {metrics['accuracy']:.4f}")
        print(f"  Precision   : {metrics['precision']:.4f}")
        print(f"  Recall      : {metrics['recall']:.4f}")
        print(f"  F1-Score    : {metrics['f1']:.4f}")
        print(f"  F1 (Macro)  : {metrics['f1_macro']:.4f}")
        if "auc_roc" in metrics:
            print(f"  AUC-ROC     : {metrics['auc_roc']:.4f}")
            print(f"  Avg Prec    : {metrics['avg_precision']:.4f}")
        print(f"  Samples     : {metrics['num_samples']}")
        print(f"{'=' * 60}")
        print(f"\n{metrics['classification_report']}")

    def plot_confusion_matrix(self, metrics: dict = None, title: str = "Confusion Matrix", filename: str = "confusion_matrix.png"):
        """Generate and save a publication-ready confusion matrix plot."""
        if metrics is None:
            metrics = self.compute()

        cm = np.array(metrics["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(8, 6))

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.CLASS_NAMES,
            yticklabels=self.CLASS_NAMES,
            ax=ax,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )

        ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold")

        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[METRICS] Confusion matrix saved to {save_path}")

    def plot_roc_curve(self, metrics: dict = None, filename: str = "roc_curve.png"):
        """Generate and save a ROC curve plot."""
        if len(self.all_probs) == 0:
            print("[WARNING] No probability scores available for ROC curve")
            return

        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        if probs.ndim == 2:
            probs = probs[:, 1]

        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="#2196F3", lw=2, label=f"ROC Curve (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
        ax.fill_between(fpr, tpr, alpha=0.1, color="#2196F3")

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[METRICS] ROC curve saved to {save_path}")

    def plot_precision_recall_curve(self, filename: str = "pr_curve.png"):
        """Generate and save a Precision-Recall curve."""
        if len(self.all_probs) == 0:
            return

        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        if probs.ndim == 2:
            probs = probs[:, 1]

        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, color="#4CAF50", lw=2, label=f"PR Curve (AP = {ap:.4f})")
        ax.fill_between(recall, precision, alpha=0.1, color="#4CAF50")

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=11)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"[METRICS] PR curve saved to {save_path}")

    def generate_all_plots(self, metrics: dict = None, prefix: str = ""):
        """Generate all available plots."""
        if metrics is None:
            metrics = self.compute()

        p = f"{prefix}_" if prefix else ""
        self.plot_confusion_matrix(metrics, filename=f"{p}confusion_matrix.png")
        self.plot_roc_curve(metrics, filename=f"{p}roc_curve.png")
        self.plot_precision_recall_curve(filename=f"{p}pr_curve.png")


def compare_models(results: dict, save_dir: str = "./results"):
    """
    Compare multiple model results side-by-side.

    Args:
        results: Dict mapping model_name → metrics_dict
        save_dir: Directory to save comparison plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(results.keys())
    metric_names = ["accuracy", "precision", "recall", "f1", "f1_macro"]

    # Filter to metrics that exist in all results
    available = [m for m in metric_names if all(m in results[n] for n in model_names)]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(available))
    width = 0.8 / len(model_names)

    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]

    for i, name in enumerate(model_names):
        values = [results[name][m] for m in available]
        bars = ax.bar(x + i * width, values, width, label=name, color=colors[i % len(colors)], alpha=0.85)
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xlabel("Metric", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ").title() for m in available])
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_path = save_dir / "model_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[METRICS] Model comparison plot saved to {save_path}")
