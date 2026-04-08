"""Training loop for LoRA fine-tuning with AMP, gradient accumulation, EMA, and checkpointing."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from foldfit.domain.interfaces import PeftPort
from foldfit.domain.value_objects import TrainingConfig
from foldfit.infrastructure.training.scheduler import build_scheduler

logger = logging.getLogger(__name__)


class EMA:
    """Exponential Moving Average of model parameters.

    Maintains a shadow copy of parameters that is updated as:
        shadow = decay * shadow + (1 - decay) * param
    """

    def __init__(self, parameters: list[nn.Parameter], decay: float) -> None:
        self.decay = decay
        self.shadow: list[torch.Tensor] = [p.data.clone() for p in parameters]
        self.params = parameters

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow parameters with current model parameters."""
        for shadow, param in zip(self.shadow, self.params, strict=True):
            shadow.mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply(self) -> list[torch.Tensor]:
        """Apply EMA weights to model, returning original weights for restore."""
        originals = []
        for shadow, param in zip(self.shadow, self.params, strict=True):
            originals.append(param.data.clone())
            param.data.copy_(shadow)
        return originals

    def restore(self, originals: list[torch.Tensor]) -> None:
        """Restore original weights after EMA evaluation."""
        for original, param in zip(originals, self.params, strict=True):
            param.data.copy_(original)


class Trainer:
    """Handles the training loop for fine-tuning with LoRA.

    Supports: AMP, gradient accumulation, gradient clipping, EMA,
    early stopping, and checkpoint save/load.

    Args:
        config: Training configuration with hyperparameters.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def fit(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        peft: PeftPort,
        head: nn.Module | None,
        train_loader: DataLoader,  # type: ignore[type-arg]
        val_loader: DataLoader | None = None,  # type: ignore[type-arg]
        checkpoint_dir: str | Path | None = None,
        model_forward_fn: Any = None,
    ) -> list[dict[str, float]]:
        """Run the training loop.

        Args:
            model: The full model (trunk + head).
            loss_fn: Loss function that takes (preds, batch) -> dict with 'loss' key.
            peft: PEFT port with trainable_parameters().
            head: Optional head module (its params get lr_head).
            train_loader: Training data loader.
            val_loader: Optional validation data loader.
            checkpoint_dir: Directory for saving checkpoints.
            model_forward_fn: Optional custom forward function(model, batch) -> (preds, loss_dict).

        Returns:
            List of per-epoch metrics dicts.
        """
        cfg = self.config
        device = next(iter(peft.trainable_parameters()), torch.tensor(0)).device

        # Build param groups with separate LRs
        param_groups = self._build_param_groups(peft, head)
        optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.weight_decay)

        # Scheduler
        total_steps = len(train_loader) * cfg.epochs // cfg.accumulation_steps
        scheduler = build_scheduler(
            optimizer, cfg.scheduler, cfg.warmup_steps, total_steps, cfg.min_lr
        )

        # AMP
        scaler = GradScaler(enabled=cfg.amp)

        # EMA
        all_trainable = list(peft.trainable_parameters())
        if head is not None:
            all_trainable += list(head.parameters())
        ema = EMA(all_trainable, cfg.ema_decay) if cfg.ema_decay > 0 else None

        # Training state
        history: list[dict[str, float]] = []
        best_val_loss = float("inf")
        patience_counter = 0
        global_step = 0

        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        for epoch in range(cfg.epochs):
            # Train
            train_loss, global_step = self._train_epoch(
                model=model,
                loss_fn=loss_fn,
                loader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                all_params=all_trainable,
                global_step=global_step,
                device=device,
                ema=ema,
                model_forward_fn=model_forward_fn,
            )

            metrics: dict[str, float] = {"epoch": epoch, "train_loss": train_loss}

            # Validate
            if val_loader is not None:
                val_results = self._val_epoch(
                    model, loss_fn, val_loader, device, ema, model_forward_fn
                )
                val_loss = val_results.get("loss", 0.0)
                metrics["val_loss"] = val_loss
                # Include auxiliary metrics (ca_rmsd, gdt_ts, plddt, etc.)
                for k, v in val_results.items():
                    if k != "loss":
                        metrics[f"val_{k}"] = v

                improved = val_loss < best_val_loss
                if improved:
                    best_val_loss = val_loss
                    patience_counter = 0
                    if checkpoint_dir:
                        self._save_checkpoint(
                            Path(checkpoint_dir) / "best",
                            peft, head, optimizer, scheduler, scaler, epoch, best_val_loss,
                        )
                else:
                    patience_counter += 1

                if (
                    cfg.early_stopping_patience > 0
                    and patience_counter >= cfg.early_stopping_patience
                ):
                    logger.info(f"Early stopping at epoch {epoch}")
                    history.append(metrics)
                    break

            history.append(metrics)
            parts = [f"Epoch {epoch}: train_loss={train_loss:.6f}"]
            if val_loader:
                parts.append(f"val_loss={metrics.get('val_loss', 0):.6f}")
                if "val_ca_rmsd" in metrics:
                    parts.append(f"RMSD={metrics['val_ca_rmsd']:.3f}Å")
                if "val_gdt_ts" in metrics:
                    parts.append(f"GDT-TS={metrics['val_gdt_ts']:.3f}")
                if "val_plddt" in metrics:
                    parts.append(f"pLDDT={metrics['val_plddt']:.1f}")
            logger.info(" | ".join(parts))

        return history

    def _build_param_groups(
        self, peft: PeftPort, head: nn.Module | None
    ) -> list[dict[str, Any]]:
        cfg = self.config
        groups: list[dict[str, Any]] = []

        lr_lora = cfg.lr_lora if cfg.lr_lora is not None else cfg.learning_rate
        groups.append({"params": peft.trainable_parameters(), "lr": lr_lora})

        if head is not None:
            head_params = list(head.parameters())
            if head_params:
                lr_head = cfg.lr_head if cfg.lr_head is not None else cfg.learning_rate * 10
                groups.append({"params": head_params, "lr": lr_head})

        return groups

    def _train_epoch(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        loader: DataLoader,  # type: ignore[type-arg]
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: GradScaler,
        all_params: list[nn.Parameter],
        global_step: int,
        device: torch.device | str,
        ema: EMA | None,
        model_forward_fn: Any = None,
    ) -> tuple[float, int]:
        cfg = self.config
        model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(loader):
            batch = _to_device(batch, device)

            # Skip empty batches (all samples failed featurization)
            if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                first = batch[0]
                if isinstance(first, dict) and len(first) == 0:
                    continue

            with autocast("cuda", enabled=cfg.amp):
                if model_forward_fn is not None:
                    loss_dict = model_forward_fn(model, batch)
                else:
                    preds = model(batch)
                    loss_dict = loss_fn(preds, batch)

                loss = loss_dict["loss"] / cfg.accumulation_steps

            scaler.scale(loss).backward()

            if (batch_idx + 1) % cfg.accumulation_steps == 0:
                scaler.unscale_(optimizer)
                if cfg.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(all_params, cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

                if ema is not None:
                    ema.update()

            total_loss += loss_dict["loss"].item()
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss, global_step

    @torch.no_grad()
    def _val_epoch(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        loader: DataLoader,  # type: ignore[type-arg]
        device: torch.device | str,
        ema: EMA | None,
        model_forward_fn: Any = None,
    ) -> dict[str, float]:
        """Run validation and return loss + all metrics."""
        model.eval()
        originals = ema.apply() if ema else None

        totals: dict[str, float] = {}
        num_batches = 0

        for batch in loader:
            batch = _to_device(batch, device)

            # Skip empty batches
            if isinstance(batch, (tuple, list)) and len(batch) >= 1:
                first = batch[0]
                if isinstance(first, dict) and len(first) == 0:
                    continue

            if model_forward_fn is not None:
                loss_dict = model_forward_fn(model, batch)
            else:
                preds = model(batch)
                loss_dict = loss_fn(preds, batch)

            for k, v in loss_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else float(v)
                totals[k] = totals.get(k, 0.0) + val
            num_batches += 1

        if ema and originals is not None:
            ema.restore(originals)

        n = max(num_batches, 1)
        return {k: v / n for k, v in totals.items()}

    def _save_checkpoint(
        self,
        path: Path,
        peft: PeftPort,
        head: nn.Module | None,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        scaler: GradScaler,
        epoch: int,
        best_val_loss: float,
    ) -> None:
        path.mkdir(parents=True, exist_ok=True)
        peft.save(path / "peft")
        if head is not None:
            torch.save(head.state_dict(), path / "head.pt")
        torch.save(
            {
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            },
            path / "training_state.pt",
        )
        logger.info(f"Saved checkpoint to {path}")


def _to_device(
    data: Any, device: torch.device | str
) -> Any:
    """Recursively move tensors to device."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, dict):
        return {k: _to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(_to_device(v, device) for v in data)
    return data
