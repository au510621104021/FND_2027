"""
Training Pipeline for Multimodal Fake News Detector
=====================================================
Handles the full training loop including:
    - Mixed-precision training (FP16)
    - Learning rate scheduling with warmup
    - Gradient clipping
    - Early stopping
    - Checkpoint saving/loading
    - TensorBoard logging
    - Evaluation at each epoch
"""

import os
import time
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
# Modern PyTorch AMP API (compatible with PyTorch 2.x+)
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, StepLR
from torch.utils.tensorboard import SummaryWriter

from .metrics import MetricsCalculator


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""

    def __init__(self, patience: int = 5, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class Trainer:
    """
    Full training pipeline for the MultimodalFakeNewsDetector.

    Supports:
        - Multimodal, text-only, and image-only training modes
        - Mixed precision (FP16) for memory efficiency
        - Cosine / Linear / Step LR schedulers with warmup
        - TensorBoard logging
        - Best model checkpointing
    """

    def __init__(self, model, config: dict, device: torch.device = None):
        self.model = model
        self.config = config
        self.train_cfg = config["training"]
        self.log_cfg = config.get("logging", {})

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self._build_optimizer()
        self.scaler = GradScaler('cuda') if self.train_cfg.get("fp16", False) and torch.cuda.is_available() else None

        # Metrics
        self.metrics_calc = MetricsCalculator(
            average=config.get("evaluation", {}).get("average", "weighted"),
            save_dir=os.path.join(self.log_cfg.get("log_dir", "./logs"), "metrics"),
        )

        # Checkpointing
        self.checkpoint_dir = Path(self.log_cfg.get("checkpoint_dir", "./checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.writer = None
        if self.log_cfg.get("tensorboard", False):
            tb_dir = os.path.join(self.log_cfg.get("log_dir", "./logs"), "tensorboard")
            self.writer = SummaryWriter(log_dir=tb_dir)

        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.train_cfg.get("early_stopping_patience", 5)
        )

        # Training state
        self.best_val_f1 = 0.0
        self.best_epoch = 0
        self.training_history = []
        self.start_epoch = 0
        self.global_step = 0

    def _build_optimizer(self) -> AdamW:
        """Build AdamW optimizer with weight decay."""
        # Separate parameters with/without weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_groups = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.train_cfg.get("weight_decay", 0.01),
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return AdamW(
            optimizer_groups,
            lr=self.train_cfg.get("learning_rate", 2e-5),
        )

    def _build_scheduler(self, num_training_steps: int):
        """Build learning rate scheduler with warmup."""
        warmup_steps = int(num_training_steps * self.train_cfg.get("warmup_ratio", 0.1))
        scheduler_type = self.train_cfg.get("scheduler", "cosine")

        if scheduler_type == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer, T_max=num_training_steps - warmup_steps
            )
        elif scheduler_type == "step":
            main_scheduler = StepLR(self.optimizer, step_size=num_training_steps // 3, gamma=0.1)
        else:
            main_scheduler = LinearLR(
                self.optimizer, start_factor=1.0, end_factor=0.01, total_iters=num_training_steps
            )

        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=max(warmup_steps, 1)
        )

        return torch.optim.lr_scheduler.SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

    def train(
        self,
        train_loader,
        val_loader,
        mode: str = "multimodal",
    ) -> dict:
        """
        Full training loop.

        Args:
            train_loader: Training DataLoader
            val_loader: Validation DataLoader
            mode: 'multimodal' | 'text_only' | 'image_only'

        Returns:
            Training history dict
        """
        num_epochs = self.train_cfg.get("num_epochs", 20)
        remaining_epochs = max(num_epochs - self.start_epoch, 1)
        num_training_steps = len(train_loader) * remaining_epochs
        scheduler = self._build_scheduler(num_training_steps)

        scheduler_state = getattr(self, "_resume_scheduler_state", None)
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
        print(f"\n{'=' * 60}")
        print(f"  Starting Training ({mode} mode)")
        print(f"  Device: {self.device}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {self.train_cfg.get('batch_size', 16)}")
        print(f"  Learning rate: {self.train_cfg.get('learning_rate', 2e-5)}")
        params = self.model.get_trainable_params()
        print(f"  Trainable params: {params['trainable']:,} / {params['total']:,} ({params['trainable_pct']:.1f}%)")
        print(f"{'=' * 60}\n")

        for epoch in range(self.start_epoch, num_epochs):
            epoch_start = time.time()

            # --- Training Phase ---
            train_loss, train_metrics, self.global_step = self._train_epoch(
                train_loader, scheduler, mode, epoch, self.global_step
            )

            # --- Validation Phase ---
            val_loss, val_metrics = self._validate(val_loader, mode)

            epoch_time = time.time() - epoch_start

            # Log epoch results
            epoch_log = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_metrics["accuracy"],
                "val_accuracy": val_metrics["accuracy"],
                "train_f1": train_metrics["f1"],
                "val_f1": val_metrics["f1"],
                "lr": self.optimizer.param_groups[0]["lr"],
                "epoch_time": epoch_time,
            }
            if "auc_roc" in val_metrics:
                epoch_log["val_auc_roc"] = val_metrics["auc_roc"]
            self.training_history.append(epoch_log)

            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_metrics['accuracy']:.4f} | F1: {train_metrics['f1']:.4f}")
            print(f"  Val   Loss: {val_loss:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar("Loss/train", train_loss, epoch)
                self.writer.add_scalar("Loss/val", val_loss, epoch)
                self.writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
                self.writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
                self.writer.add_scalar("F1/train", train_metrics["f1"], epoch)
                self.writer.add_scalar("F1/val", val_metrics["f1"], epoch)
                self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

            # Checkpointing (always save latest, best when improved)
            is_best = val_metrics["f1"] > self.best_val_f1
            if is_best:
                self.best_val_f1 = val_metrics["f1"]
                self.best_epoch = epoch + 1
                print(f"  [BEST] New best model (F1: {self.best_val_f1:.4f})")

            if self.log_cfg.get("save_checkpoints", True):
                self._save_checkpoint(
                    epoch + 1,
                    val_metrics,
                    scheduler=scheduler,
                    global_step=self.global_step,
                    is_best=is_best,
                )

            # Early stopping check
            if self.early_stopping(val_loss):
                print(f"\n[EARLY STOPPING] No improvement for {self.early_stopping.patience} epochs.")
                print(f"  Best epoch: {self.best_epoch} (F1: {self.best_val_f1:.4f})")
                break

        # Save training history
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.training_history, f, indent=2)

        if self.writer:
            self.writer.close()

        print(f"\n{'=' * 60}")
        print(f"  Training Complete!")
        print(f"  Best Epoch: {self.best_epoch} | Best Val F1: {self.best_val_f1:.4f}")
        print(f"{'=' * 60}\n")

        return {
            "history": self.training_history,
            "best_epoch": self.best_epoch,
            "best_val_f1": self.best_val_f1,
        }

    def _train_epoch(self, loader, scheduler, mode, epoch, global_step) -> tuple:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        self.metrics_calc.reset()

        log_every = self.log_cfg.get("log_every_n_steps", 50)

        pbar = tqdm(loader, desc=f"Epoch {epoch + 1} [Train]", leave=False)
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass (with optional mixed precision)
            if self.scaler:
                with autocast('cuda'):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        token_type_ids=token_type_ids,
                        mode=mode,
                    )
                    loss = self.criterion(outputs["logits"], labels)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_cfg.get("max_grad_norm", 1.0),
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    token_type_ids=token_type_ids,
                    mode=mode,
                )
                loss = self.criterion(outputs["logits"], labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_cfg.get("max_grad_norm", 1.0),
                )
                self.optimizer.step()

            scheduler.step()
            global_step += 1

            # Track metrics
            total_loss += loss.item()
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            probs = outputs["probabilities"].detach().cpu().numpy()
            self.metrics_calc.update(labels.cpu().numpy(), preds, probs)

            # Progress bar update
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Periodic TensorBoard logging
            if self.writer and global_step % log_every == 0:
                self.writer.add_scalar("Step/loss", loss.item(), global_step)
                self.writer.add_scalar("Step/lr", self.optimizer.param_groups[0]["lr"], global_step)

        avg_loss = total_loss / len(loader)
        metrics = self.metrics_calc.compute()
        return avg_loss, metrics, global_step

    @torch.no_grad()
    def _validate(self, loader, mode: str) -> tuple:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        self.metrics_calc.reset()

        pbar = tqdm(loader, desc="[Validate]", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            token_type_ids = batch["token_type_ids"].to(self.device)
            pixel_values = batch["pixel_values"].to(self.device)
            labels = batch["label"].to(self.device)

            if self.scaler:
                with autocast('cuda'):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        pixel_values=pixel_values,
                        token_type_ids=token_type_ids,
                        mode=mode,
                    )
                    loss = self.criterion(outputs["logits"], labels)
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    token_type_ids=token_type_ids,
                    mode=mode,
                )
                loss = self.criterion(outputs["logits"], labels)

            total_loss += loss.item()
            preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
            probs = outputs["probabilities"].detach().cpu().numpy()
            self.metrics_calc.update(labels.cpu().numpy(), preds, probs)

        avg_loss = total_loss / len(loader)
        metrics = self.metrics_calc.compute()
        return avg_loss, metrics

    @torch.no_grad()
    def evaluate(self, test_loader, mode: str = "multimodal", generate_plots: bool = True) -> dict:
        """
        Full evaluation on test set with detailed metrics and plots.

        Args:
            test_loader: Test DataLoader
            mode: Inference mode
            generate_plots: Whether to generate visualization plots

        Returns:
            Complete metrics dictionary
        """
        print(f"\n{'=' * 60}")
        print(f"  Running Final Evaluation ({mode} mode)")
        print(f"{'=' * 60}")

        test_loss, test_metrics = self._validate(test_loader, mode)
        test_metrics["test_loss"] = test_loss

        self.metrics_calc.print_report(test_metrics, title=f"Test Results ({mode})")

        if generate_plots:
            self.metrics_calc.generate_all_plots(test_metrics, prefix=mode)

        return test_metrics

    def _save_checkpoint(
        self,
        epoch: int,
        metrics: dict,
        scheduler=None,
        global_step: int = 0,
        is_best: bool = False,
    ):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "global_step": global_step,
            "metrics": metrics,
            "config": self.config,
            "best_val_f1": self.best_val_f1,
            "best_epoch": self.best_epoch,
            "training_history": self.training_history,
        }

        # Save latest
        path = self.checkpoint_dir / "latest_model.pt"
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"  [CHECKPOINT] Best model saved to {best_path}")

    def load_checkpoint(self, path: str):
        """Load a model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._resume_scheduler_state = checkpoint.get("scheduler_state_dict")
        self.global_step = int(checkpoint.get("global_step", 0))
        self.start_epoch = int(checkpoint.get("epoch", 0))
        self.best_val_f1 = float(checkpoint.get("best_val_f1", 0.0))
        self.best_epoch = int(checkpoint.get("best_epoch", 0))
        self.training_history = checkpoint.get("training_history", []) or []
        print(
            f"[CHECKPOINT] Loaded model from {path} "
            f"(Epoch {checkpoint.get('epoch', '?')}, Global Step {self.global_step})"
        )
        return checkpoint.get("metrics", {})

