from __future__ import annotations

import math
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support, roc_auc_score, roc_curve

from omr.dataset import build_dataloaders
from omr.export import export_from_config
from omr.model import build_model, count_parameters
from omr.utils import ensure_dir, load_yaml, save_json, set_seed, write_csv


class EarlyStopper:
    def __init__(self, patience: int = 5, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best: float | None = None
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False
        improved = value > self.best if self.mode == "max" else value < self.best
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


@torch.no_grad()
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, Any]:
    y_pred = (y_prob >= threshold).astype(np.int32)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "threshold": float(threshold),
        "report": classification_report(y_true, y_pred, output_dict=True, zero_division=0),
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_prob)
        metrics["roc_curve"] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
        }
    else:
        metrics["roc_auc"] = float("nan")
        metrics["roc_curve"] = {"fpr": [], "tpr": [], "thresholds": []}
    return metrics


def _forward_context(device: str, amp_enabled: bool):
    if amp_enabled and device.startswith("cuda"):
        return torch.cuda.amp.autocast()
    return nullcontext()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
    scaler: torch.cuda.amp.GradScaler | None,
    amp_enabled: bool,
) -> float:
    model.train()
    losses: list[float] = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with _forward_context(device, amp_enabled):
            logits = model(x)
            loss = criterion(logits, y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else math.nan


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
) -> tuple[float, dict[str, Any]]:
    model.eval()
    losses: list[float] = []
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        p = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs.append(p)
        labels.append(y.detach().cpu().numpy().reshape(-1))
        losses.append(float(loss.detach().cpu().item()))
    y_prob = np.concatenate(probs) if probs else np.array([])
    y_true = np.concatenate(labels) if labels else np.array([])
    metrics = compute_metrics(y_true, y_prob, threshold=0.5) if len(y_true) else {}
    return float(np.mean(losses)) if losses else math.nan, metrics


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    epoch: int,
    cfg: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    ensure_dir(Path(path).parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
            "config": cfg,
            "metrics": metrics,
        },
        path,
    )


def train_from_config(config_path: str | Path, device: str | None = None) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    set_seed(int(cfg.get("seed", 42)))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = ensure_dir(cfg["train"].get("save_dir", "outputs/checkpoints"))

    train_loader, val_loader, ds_stats = build_dataloaders(cfg)
    model = build_model(cfg["model"]).to(device)

    train_pos = max(1, ds_stats["train_pos"])
    train_neg = max(1, ds_stats["train_size"] - ds_stats["train_pos"])
    pos_weight_cfg = cfg["train"].get("pos_weight")
    if pos_weight_cfg is None:
        pos_weight = torch.tensor([train_neg / train_pos], dtype=torch.float32, device=device)
    else:
        pos_weight = torch.tensor([float(pos_weight_cfg)], dtype=torch.float32, device=device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"].get("lr", 1e-3)),
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(cfg["train"].get("epochs", 20)),
    )
    amp_enabled = bool(cfg["train"].get("amp", True)) and device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if amp_enabled else None
    early = EarlyStopper(patience=int(cfg["train"].get("early_patience", 5)), mode="max")

    history: list[dict[str, Any]] = []
    best_state: dict[str, Any] | None = None
    best_score = -1e9
    epochs = int(cfg["train"].get("epochs", 20))

    print(f"Training on {device} | {ds_stats['train_size']} train / {ds_stats['val_size']} val samples")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, amp_enabled)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        score = val_metrics.get("roc_auc")
        if score is None or math.isnan(score):
            score = val_metrics.get("f1", 0.0)
        scheduler.step()
        print(
            f"Epoch {epoch:>3}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"acc={val_metrics.get('accuracy', 0):.4f} | "
            f"f1={val_metrics.get('f1', 0):.4f} | "
            f"roc_auc={val_metrics.get('roc_auc', float('nan')):.4f}"
        )

        row = {
            "epoch": epoch,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
            "val_precision": float(val_metrics.get("precision", 0.0)),
            "val_recall": float(val_metrics.get("recall", 0.0)),
            "val_f1": float(val_metrics.get("f1", 0.0)),
            "val_roc_auc": float(val_metrics.get("roc_auc", float("nan"))),
        }
        history.append(row)

        last_path = save_dir / "last.pt"
        save_checkpoint(last_path, model, optimizer, scheduler, epoch, cfg, row)

        if score > best_score:
            best_score = float(score)
            best_state = deepcopy(model.state_dict())
            best_path = save_dir / "best.pt"
            save_checkpoint(best_path, model, optimizer, scheduler, epoch, cfg, row)
            save_json(val_metrics, save_dir / "best_metrics.json")

        if early.step(float(score)):
            break

    write_csv(history, save_dir / "train_log.csv")

    if best_state is not None:
        model.load_state_dict(best_state)

    final_val_loss, final_metrics = validate(model, val_loader, criterion, device)
    save_json(final_metrics, save_dir / "final_metrics.json")
    summary = {
        "device": device,
        "params": count_parameters(model),
        "train_size": ds_stats["train_size"],
        "val_size": ds_stats["val_size"],
        "train_pos": ds_stats["train_pos"],
        "val_pos": ds_stats["val_pos"],
        "final_val_loss": float(final_val_loss),
        "final_metrics": final_metrics,
    }
    save_json(summary, save_dir / "summary.json")

    if bool(cfg.get("export", {}).get("after_train", True)):
        export_from_config(config_path, save_dir / "best.pt", device=device)

    return summary


__all__ = ["train_from_config", "compute_metrics"]
