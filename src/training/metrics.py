"""
Evaluation Metrics for Fake News Detection (Conference-Level)
================================================================
Computes comprehensive classification metrics including:
    - Accuracy, Precision, Recall, F1-Score
    - AUC-ROC, Average Precision
    - Matthews Correlation Coefficient (MCC)
    - Cohen's Kappa
    - Confusion Matrix
    - Per-class metrics

Statistical Significance:
    - McNemar's Test (paired model comparison)
    - Bootstrap Confidence Intervals
    - Paired t-Test across folds

Publication Outputs:
    - LaTeX-formatted results tables
    - Publication-ready plots (300 DPI, serif fonts)
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
    matthews_corrcoef,
    cohen_kappa_score,
)
from scipy import stats
from pathlib import Path

# Publication-ready matplotlib configuration
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


class MetricsCalculator:
    """
    Comprehensive metrics calculator for binary fake news classification.

    Tracks predictions across batches, computes aggregate metrics,
    and generates publication-ready visualization plots.

    Includes conference-level metrics:
        - MCC (Matthews Correlation Coefficient)
        - Cohen's Kappa
        - Bootstrap Confidence Intervals
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
            Dictionary with all computed metrics including MCC and Kappa.
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
            # --- Conference-level metrics ---
            "mcc": matthews_corrcoef(labels, preds),
            "cohens_kappa": cohen_kappa_score(labels, preds),
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

    def compute_bootstrap_ci(
        self,
        metric_fn,
        n_iterations: int = 1000,
        confidence_level: float = 0.95,
        seed: int = 42,
    ) -> dict:
        """
        Compute bootstrap confidence intervals for a given metric.

        Args:
            metric_fn: Function that takes (labels, preds) -> float
            n_iterations: Number of bootstrap iterations
            confidence_level: Confidence level (e.g. 0.95 for 95% CI)
            seed: Random seed

        Returns:
            dict with 'mean', 'lower', 'upper', 'std'
        """
        rng = np.random.RandomState(seed)
        labels = np.array(self.all_labels)
        preds = np.array(self.all_preds)
        n = len(labels)

        bootstrap_scores = []
        for _ in range(n_iterations):
            indices = rng.randint(0, n, size=n)
            score = metric_fn(labels[indices], preds[indices])
            bootstrap_scores.append(score)

        bootstrap_scores = np.array(bootstrap_scores)
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_scores, 100 * alpha / 2)
        upper = np.percentile(bootstrap_scores, 100 * (1 - alpha / 2))

        return {
            "mean": np.mean(bootstrap_scores),
            "std": np.std(bootstrap_scores),
            "lower": lower,
            "upper": upper,
            "confidence_level": confidence_level,
        }

    def compute_all_bootstrap_cis(
        self,
        n_iterations: int = 1000,
        confidence_level: float = 0.95,
    ) -> dict:
        """
        Compute bootstrap CIs for all standard metrics.

        Returns:
            dict mapping metric_name -> {mean, std, lower, upper}
        """
        metric_fns = {
            "accuracy": accuracy_score,
            "precision": lambda y, p: precision_score(y, p, average=self.average, zero_division=0),
            "recall": lambda y, p: recall_score(y, p, average=self.average, zero_division=0),
            "f1": lambda y, p: f1_score(y, p, average=self.average, zero_division=0),
            "f1_macro": lambda y, p: f1_score(y, p, average="macro", zero_division=0),
            "mcc": matthews_corrcoef,
            "cohens_kappa": cohen_kappa_score,
        }

        results = {}
        for name, fn in metric_fns.items():
            results[name] = self.compute_bootstrap_ci(
                fn, n_iterations=n_iterations, confidence_level=confidence_level
            )

        return results

    @staticmethod
    def mcnemar_test(preds_a: np.ndarray, preds_b: np.ndarray, labels: np.ndarray) -> dict:
        """
        McNemar's test for comparing two classifiers on the same test set.

        Tests whether two models make the same errors on the dataset.

        Args:
            preds_a: Predictions from model A
            preds_b: Predictions from model B
            labels: Ground truth labels

        Returns:
            dict with 'chi2_statistic', 'p_value', 'significant' (at α=0.05)
        """
        correct_a = (preds_a == labels)
        correct_b = (preds_b == labels)

        # Contingency table: b01 = A wrong, B right; b10 = A right, B wrong
        b01 = np.sum(~correct_a & correct_b)
        b10 = np.sum(correct_a & ~correct_b)

        # McNemar's test with continuity correction
        if b01 + b10 == 0:
            return {"chi2_statistic": 0.0, "p_value": 1.0, "significant": False}

        chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
        p_value = 1 - stats.chi2.cdf(chi2, df=1)

        return {
            "chi2_statistic": float(chi2),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "b01": int(b01),
            "b10": int(b10),
        }

    @staticmethod
    def paired_t_test(scores_a: list, scores_b: list) -> dict:
        """
        Paired t-test for comparing two models across multiple folds/runs.

        Args:
            scores_a: List of metric scores from model A across folds
            scores_b: List of metric scores from model B across folds

        Returns:
            dict with 't_statistic', 'p_value', 'significant'
        """
        if len(scores_a) < 2 or len(scores_b) < 2:
            return {"t_statistic": 0.0, "p_value": 1.0, "significant": False}

        t_stat, p_value = stats.ttest_rel(scores_a, scores_b)

        return {
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant": p_value < 0.05,
            "mean_diff": float(np.mean(np.array(scores_a) - np.array(scores_b))),
        }

    def print_report(self, metrics: dict = None, title: str = "Evaluation Results"):
        """Print a formatted metrics report."""
        if metrics is None:
            metrics = self.compute()

        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")
        print(f"  Accuracy      : {metrics['accuracy']:.4f}")
        print(f"  Precision     : {metrics['precision']:.4f}")
        print(f"  Recall        : {metrics['recall']:.4f}")
        print(f"  F1-Score      : {metrics['f1']:.4f}")
        print(f"  F1 (Macro)    : {metrics['f1_macro']:.4f}")
        print(f"  MCC           : {metrics['mcc']:.4f}")
        print(f"  Cohen's Kappa : {metrics['cohens_kappa']:.4f}")
        if "auc_roc" in metrics:
            print(f"  AUC-ROC       : {metrics['auc_roc']:.4f}")
            print(f"  Avg Precision : {metrics['avg_precision']:.4f}")
        print(f"  Samples       : {metrics['num_samples']}")
        print(f"{'=' * 60}")
        print(f"\n{metrics['classification_report']}")

    def generate_latex_table(
        self,
        results: dict,
        caption: str = "Experimental Results",
        label: str = "tab:results",
        filename: str = "results_table.tex",
    ) -> str:
        """
        Generate a publication-ready LaTeX table from experiment results.

        Args:
            results: dict mapping method_name -> metrics_dict
            caption: Table caption
            label: LaTeX label
            filename: Output filename

        Returns:
            LaTeX table string
        """
        metric_keys = ["accuracy", "precision", "recall", "f1", "f1_macro", "mcc", "auc_roc"]
        metric_headers = ["Acc.", "Prec.", "Recall", "F1", "F1\\textsubscript{M}", "MCC", "AUC"]

        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append(r"\begin{tabular}{l" + "c" * len(metric_headers) + "}")
        lines.append(r"\toprule")
        lines.append("Method & " + " & ".join(metric_headers) + r" \\")
        lines.append(r"\midrule")

        for method_name, metrics in results.items():
            values = []
            best_vals = {}
            for key in metric_keys:
                all_vals = [m.get(key, None) for m in results.values() if m.get(key) is not None]
                if all_vals:
                    best_vals[key] = max(all_vals)

            for key in metric_keys:
                val = metrics.get(key, None)
                if val is not None:
                    # Bold the best result
                    if key in best_vals and abs(val - best_vals[key]) < 1e-6:
                        values.append(f"\\textbf{{{val:.4f}}}")
                    else:
                        values.append(f"{val:.4f}")
                else:
                    values.append("--")

            display_name = method_name.replace("_", " ").title()
            lines.append(f"{display_name} & " + " & ".join(values) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        latex_str = "\n".join(lines)

        save_path = self.save_dir / filename
        with open(save_path, "w") as f:
            f.write(latex_str)
        print(f"[LATEX] Results table saved to {save_path}")

        return latex_str

    def generate_latex_table_with_ci(
        self,
        results: dict,
        caption: str = "Results with 95\\% Confidence Intervals",
        label: str = "tab:results_ci",
        filename: str = "results_table_ci.tex",
    ) -> str:
        """
        Generate LaTeX table with mean ± std from cross-validation/multi-run results.

        Args:
            results: dict mapping method_name -> {'mean': metrics, 'std': metrics}
        """
        metric_keys = ["accuracy", "precision", "recall", "f1", "mcc", "auc_roc"]
        metric_headers = ["Acc.", "Prec.", "Recall", "F1", "MCC", "AUC"]

        lines = []
        lines.append(r"\begin{table}[htbp]")
        lines.append(r"\centering")
        lines.append(f"\\caption{{{caption}}}")
        lines.append(f"\\label{{{label}}}")
        lines.append(r"\begin{tabular}{l" + "c" * len(metric_headers) + "}")
        lines.append(r"\toprule")
        lines.append("Method & " + " & ".join(metric_headers) + r" \\")
        lines.append(r"\midrule")

        for method_name, data in results.items():
            mean_metrics = data.get("mean", {})
            std_metrics = data.get("std", {})
            values = []
            for key in metric_keys:
                mean_val = mean_metrics.get(key, None)
                std_val = std_metrics.get(key, None)
                if mean_val is not None and std_val is not None:
                    values.append(f"${mean_val:.4f} \\pm {std_val:.4f}$")
                elif mean_val is not None:
                    values.append(f"{mean_val:.4f}")
                else:
                    values.append("--")

            display_name = method_name.replace("_", " ").title()
            lines.append(f"{display_name} & " + " & ".join(values) + r" \\")

        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        lines.append(r"\end{table}")

        latex_str = "\n".join(lines)

        save_path = self.save_dir / filename
        with open(save_path, "w") as f:
            f.write(latex_str)
        print(f"[LATEX] Results table with CI saved to {save_path}")

        return latex_str

    def plot_confusion_matrix(
        self, metrics: dict = None, title: str = "Confusion Matrix",
        filename: str = "confusion_matrix.png", normalized: bool = False,
    ):
        """Generate and save a publication-ready confusion matrix plot."""
        if metrics is None:
            metrics = self.compute()

        cm = np.array(metrics["confusion_matrix"])
        if normalized:
            cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            fmt = ".2%"
        else:
            cm_plot = cm
            fmt = "d"

        fig, ax = plt.subplots(figsize=(6, 5))

        sns.heatmap(
            cm_plot,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            xticklabels=self.CLASS_NAMES,
            yticklabels=self.CLASS_NAMES,
            ax=ax,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            annot_kws={"size": 14, "weight": "bold"},
        )

        ax.set_xlabel("Predicted Label", fontsize=13, fontweight="bold")
        ax.set_ylabel("True Label", fontsize=13, fontweight="bold")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=12)

        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        # Also save as PDF for publication
        pdf_path = save_path.with_suffix(".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()
        print(f"[METRICS] Confusion matrix saved to {save_path} and {pdf_path}")

    def plot_roc_curve(self, metrics: dict = None, filename: str = "roc_curve.png"):
        """Generate and save a publication-ready ROC curve plot."""
        if len(self.all_probs) == 0:
            print("[WARNING] No probability scores available for ROC curve")
            return

        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        if probs.ndim == 2:
            probs = probs[:, 1]

        fpr, tpr, _ = roc_curve(labels, probs)
        auc = roc_auc_score(labels, probs)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, color="#1565C0", lw=2.5, label=f"Our Model (AUC = {auc:.4f})")
        ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Baseline")
        ax.fill_between(fpr, tpr, alpha=0.08, color="#1565C0")

        ax.set_xlabel("False Positive Rate", fontsize=13)
        ax.set_ylabel("True Positive Rate", fontsize=13)
        ax.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])

        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        pdf_path = save_path.with_suffix(".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()
        print(f"[METRICS] ROC curve saved to {save_path} and {pdf_path}")

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

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(recall, precision, color="#2E7D32", lw=2.5, label=f"Our Model (AP = {ap:.4f})")
        ax.fill_between(recall, precision, alpha=0.08, color="#2E7D32")

        # Add iso-F1 curves for context
        for f1_val in [0.2, 0.4, 0.6, 0.8]:
            x = np.linspace(0.01, 1, 100)
            y = f1_val * x / (2 * x - f1_val)
            valid = (y >= 0) & (y <= 1)
            ax.plot(x[valid], y[valid], '--', color='gray', alpha=0.2, lw=0.8)
            if valid.any():
                idx = np.argmax(valid) + np.sum(valid) // 2
                if idx < len(x):
                    ax.annotate(f'F1={f1_val}', xy=(x[idx], y[idx]), fontsize=7, color='gray', alpha=0.5)

        ax.set_xlabel("Recall", fontsize=13)
        ax.set_ylabel("Precision", fontsize=13)
        ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
        ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.05])

        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        pdf_path = save_path.with_suffix(".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()
        print(f"[METRICS] PR curve saved to {save_path} and {pdf_path}")

    def plot_metrics_radar(
        self,
        results: dict,
        filename: str = "radar_chart.png",
    ):
        """
        Generate a radar (spider) chart comparing models across metrics.
        Common in conference papers for visual comparison.

        Args:
            results: dict mapping model_name -> metrics_dict
        """
        metric_keys = ["accuracy", "precision", "recall", "f1", "mcc"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score", "MCC"]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        angles = np.linspace(0, 2 * np.pi, len(metric_keys), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        colors = ["#1565C0", "#2E7D32", "#E65100", "#7B1FA2", "#C62828"]

        for idx, (name, metrics) in enumerate(results.items()):
            values = [metrics.get(k, 0) for k in metric_keys]
            values += values[:1]  # Complete the circle

            color = colors[idx % len(colors)]
            ax.plot(angles, values, 'o-', linewidth=2.5, label=name.replace('_', ' ').title(), color=color)
            ax.fill(angles, values, alpha=0.08, color=color)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)

        plt.tight_layout()
        save_path = self.save_dir / filename
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        pdf_path = save_path.with_suffix(".pdf")
        plt.savefig(pdf_path, bbox_inches="tight")
        plt.close()
        print(f"[METRICS] Radar chart saved to {save_path}")

    def generate_all_plots(self, metrics: dict = None, prefix: str = ""):
        """Generate all available plots."""
        if metrics is None:
            metrics = self.compute()

        p = f"{prefix}_" if prefix else ""
        self.plot_confusion_matrix(metrics, filename=f"{p}confusion_matrix.png")
        self.plot_confusion_matrix(metrics, filename=f"{p}confusion_matrix_norm.png",
                                   title="Normalized Confusion Matrix", normalized=True)
        self.plot_roc_curve(metrics, filename=f"{p}roc_curve.png")
        self.plot_precision_recall_curve(filename=f"{p}pr_curve.png")


def compare_models(results: dict, save_dir: str = "./results"):
    """
    Compare multiple model results side-by-side with publication-quality plot.

    Args:
        results: Dict mapping model_name -> metrics_dict
        save_dir: Directory to save comparison plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    model_names = list(results.keys())
    metric_names = ["accuracy", "precision", "recall", "f1", "f1_macro", "mcc"]

    # Filter to metrics that exist in all results
    available = [m for m in metric_names if all(m in results[n] for n in model_names)]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(available))
    width = 0.8 / len(model_names)

    colors = ["#1565C0", "#2E7D32", "#E65100", "#7B1FA2", "#C62828"]

    for i, name in enumerate(model_names):
        values = [results[name][m] for m in available]
        bars = ax.bar(
            x + i * width, values, width,
            label=name.replace("_", " ").title(),
            color=colors[i % len(colors)],
            alpha=0.88,
            edgecolor='white',
            linewidth=0.5,
        )
        # Add value labels
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold",
            )

    ax.set_xlabel("Metric", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Ablation Study: Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (len(model_names) - 1) / 2)
    ax.set_xticklabels([m.replace("_", " ").title() for m in available])
    ax.legend(fontsize=10, framealpha=0.9)
    ax.set_ylim(0, 1.12)

    plt.tight_layout()
    save_path = save_dir / "model_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    pdf_path = save_path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()
    print(f"[METRICS] Model comparison plot saved to {save_path}")


def aggregate_fold_results(fold_results: list) -> dict:
    """
    Aggregate results across K folds for mean ± std reporting.

    Args:
        fold_results: List of metrics dicts, one per fold

    Returns:
        dict with 'mean' and 'std' sub-dicts for numeric metrics
    """
    if not fold_results:
        return {}

    numeric_keys = [
        "accuracy", "precision", "recall", "f1", "f1_macro",
        "mcc", "cohens_kappa", "auc_roc", "avg_precision",
    ]

    mean_metrics = {}
    std_metrics = {}

    for key in numeric_keys:
        values = [r[key] for r in fold_results if key in r]
        if values:
            mean_metrics[key] = float(np.mean(values))
            std_metrics[key] = float(np.std(values))

    return {
        "mean": mean_metrics,
        "std": std_metrics,
        "n_folds": len(fold_results),
        "per_fold": fold_results,
    }
