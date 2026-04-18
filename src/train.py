"""
Training loop for surgical workflow multi-task models.

Implements the full training pipeline including epoch loops, validation,
early stopping, model checkpointing, and learning rate scheduling.
Supports all four model variants (baseline, LSTM, MS-TCN, MS-TCN+corr).

Author: Omar Morsi (40236376)
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from src.utils import AverageMeter
from src.evaluate import compute_phase_f1, compute_tool_map


class Trainer:
    """Training manager for multi-task surgical workflow models.

    Handles the complete training lifecycle: forward/backward passes,
    validation, early stopping, checkpointing, and metric logging.

    Args:
        model (nn.Module): The MultiTaskModel to train.
        loss_fn (nn.Module): The MultiTaskLoss criterion.
        train_loader (DataLoader): Training data loader.
        val_loader (DataLoader): Validation data loader.
        config (dict): Training configuration from YAML.
        save_dir (str): Directory to save checkpoints and logs.
        device (str): Device to train on. Default: "cuda".

    Attributes:
        best_val_f1 (float): Best validation F1 achieved so far.
        history (dict): Training history with loss and metric curves.

    Example:
        >>> trainer = Trainer(model, loss_fn, train_loader, val_loader,
        ...                   config, save_dir="checkpoints", device="cuda")
        >>> history = trainer.train()
    """

    def __init__(self, model, loss_fn, train_loader, val_loader, config,
                 save_dir="checkpoints", device="cuda"):
        self.model = model.to(device)
        self.loss_fn = loss_fn.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = save_dir
        self.device = device

        os.makedirs(save_dir, exist_ok=True)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config["training"]["learning_rate"],
            weight_decay=config["training"]["weight_decay"],
        )

        # Scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer, T_0=config["training"]["scheduler_T0"]
        )

        # Early stopping state
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.patience = config["training"]["early_stopping_patience"]

        # History
        self.history = {
            "train_loss": [], "val_loss": [],
            "train_phase_loss": [], "val_phase_loss": [],
            "train_tool_loss": [], "val_tool_loss": [],
            "train_corr_loss": [], "val_corr_loss": [],
            "val_phase_f1": [], "val_tool_map": [],
        }

    def train_one_epoch(self):
        """Run one training epoch over the full training set.

        Returns:
            dict: Average loss values for the epoch with keys
                "total", "phase", "tool", "corr".
        """
        self.model.train()
        meters = {k: AverageMeter() for k in ["total", "phase", "tool", "corr"]}
        grad_clip = self.config["training"]["grad_clip_max_norm"]

        for features, phases, tools, mask in self.train_loader:
            features = features.to(self.device)
            phases = phases.to(self.device)
            tools = tools.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()

            phase_logits, tool_logits = self.model(features, mask)
            loss, loss_dict = self.loss_fn(
                phase_logits, tool_logits, phases, tools, mask
            )

            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            self.optimizer.step()

            batch_n = features.shape[0]
            for key in meters:
                meters[key].update(loss_dict.get(key, 0.0), n=batch_n)

        return {k: m.avg for k, m in meters.items()}

    @torch.no_grad()
    def validate(self):
        """Run validation and compute metrics.

        Returns:
            tuple: (loss_dict, phase_f1, tool_map) where:
                - loss_dict: Average losses for the epoch.
                - phase_f1: Macro-averaged F1 for phase recognition.
                - tool_map: Mean Average Precision for tool detection.
        """
        self.model.eval()
        meters = {k: AverageMeter() for k in ["total", "phase", "tool", "corr"]}

        all_phase_preds = []
        all_phase_targets = []
        all_tool_preds = []
        all_tool_targets = []

        for features, phases, tools, mask in self.val_loader:
            features = features.to(self.device)
            phases = phases.to(self.device)
            tools = tools.to(self.device)
            mask = mask.to(self.device)

            phase_logits, tool_logits = self.model(features, mask)
            _, loss_dict = self.loss_fn(
                phase_logits, tool_logits, phases, tools, mask
            )

            batch_n = features.shape[0]
            for key in meters:
                meters[key].update(loss_dict.get(key, 0.0), n=batch_n)

            # Collect predictions for metrics (only valid frames)
            for b in range(features.shape[0]):
                valid = mask[b]
                all_phase_preds.append(phase_logits[b][valid].argmax(dim=-1).cpu())
                all_phase_targets.append(phases[b][valid].cpu())
                all_tool_preds.append(torch.sigmoid(tool_logits[b][valid]).cpu())
                all_tool_targets.append(tools[b][valid].cpu())

        all_phase_preds = torch.cat(all_phase_preds)
        all_phase_targets = torch.cat(all_phase_targets)
        all_tool_preds = torch.cat(all_tool_preds)
        all_tool_targets = torch.cat(all_tool_targets)

        phase_f1 = compute_phase_f1(
            all_phase_preds, all_phase_targets,
            self.config["data"]["num_phases"]
        )
        tool_map = compute_tool_map(all_tool_preds, all_tool_targets)

        loss_dict = {k: m.avg for k, m in meters.items()}
        return loss_dict, phase_f1, tool_map

    def train(self):
        """Run the full training loop with early stopping.

        Trains for up to num_epochs, validates after each epoch, applies
        learning rate scheduling, and stops early if validation F1 does
        not improve for `patience` consecutive epochs.

        Returns:
            dict: Full training history with loss and metric curves.

        Example:
            >>> history = trainer.train()
            >>> print(f"Best F1: {max(history['val_phase_f1']):.4f}")
        """
        num_epochs = self.config["training"]["num_epochs"]

        for epoch in range(1, num_epochs + 1):
            start = time.time()

            # Train
            train_losses = self.train_one_epoch()
            self.scheduler.step()

            # Validate
            val_losses, val_f1, val_map = self.validate()

            elapsed = time.time() - start

            # Log
            self.history["train_loss"].append(train_losses["total"])
            self.history["val_loss"].append(val_losses["total"])
            self.history["train_phase_loss"].append(train_losses["phase"])
            self.history["val_phase_loss"].append(val_losses["phase"])
            self.history["train_tool_loss"].append(train_losses["tool"])
            self.history["val_tool_loss"].append(val_losses["tool"])
            self.history["train_corr_loss"].append(train_losses["corr"])
            self.history["val_corr_loss"].append(val_losses["corr"])
            self.history["val_phase_f1"].append(val_f1)
            self.history["val_tool_map"].append(val_map)

            print(
                f"Epoch {epoch:3d}/{num_epochs} | "
                f"Train Loss: {train_losses['total']:.4f} | "
                f"Val Loss: {val_losses['total']:.4f} | "
                f"Val F1: {val_f1:.4f} | Val mAP: {val_map:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            # Checkpointing and early stopping
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.patience_counter = 0
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_val_f1": self.best_val_f1,
                        "config": self.config,
                    },
                    os.path.join(self.save_dir, "best_model.pt"),
                )
                print(f"  -> New best model saved (F1={val_f1:.4f})")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.patience:
                    print(f"  Early stopping at epoch {epoch} "
                          f"(no improvement for {self.patience} epochs)")
                    break

        return self.history
