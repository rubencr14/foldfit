"""PyTorch Lightning module for LoRA-based fine-tuning of OpenFold3."""

import logging
from dataclasses import asdict
from pathlib import Path

import pytorch_lightning as pl
import torch

from openfold3.core.loss.loss_module import OpenFold3Loss
from openfold3.core.utils.checkpoint_loading_utils import (
    get_state_dict_from_checkpoint,
)
from openfold3.projects.of3_all_atom.model import OpenFold3

from finetuning.config.finetune_config import FinetuneConfig, TrainingConfig
from finetuning.lora.applicator import LoRAApplicator
from finetuning.lora.checkpoint import LoRACheckpointManager
from finetuning.lora.config import LoRAConfig
from finetuning.runner.lora_ema import LoRAExponentialMovingAverage

logger = logging.getLogger(__name__)


class LoRAFineTuningRunner(pl.LightningModule):
    """LightningModule for LoRA fine-tuning of OpenFold3.

    Composes with OpenFold3 model and OpenFold3Loss rather than inheriting
    from OpenFold3AllAtom. This separation is intentional: LoRA training
    has different parameter management semantics (only LoRA params are
    trainable, different optimizer groups, lightweight EMA).

    Args:
        model_config: ml_collections ConfigDict for OpenFold3 architecture.
        lora_config: LoRA adapter configuration.
        training_config: Training hyperparameters.
        pretrained_checkpoint_path: Path to pretrained OpenFold3 weights.
        log_dir: Optional directory for logging output.
    """

    def __init__(
        self,
        model_config,
        lora_config: LoRAConfig,
        training_config: TrainingConfig,
        pretrained_checkpoint_path: Path | None = None,
        log_dir: Path | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_config"])

        self.model_config = model_config
        self.lora_config = lora_config
        self.training_config = training_config
        self.pretrained_checkpoint_path = pretrained_checkpoint_path
        self.log_dir = log_dir

        # Build model and loss
        self.model = OpenFold3(model_config)
        self.loss = OpenFold3Loss(config=model_config.architecture.loss_module)

        # These are initialized in setup()
        self.lora_applicator = LoRAApplicator(lora_config)
        self.checkpoint_manager = LoRACheckpointManager()
        self.lora_ema: LoRAExponentialMovingAverage | None = None
        self._cached_lora_weights: dict | None = None

    def setup(self, stage: str) -> None:
        """Initialize LoRA adapters and freeze base model parameters."""
        if stage == "fit":
            self._load_pretrained_weights()
            self._apply_lora()
            self._log_parameter_counts()
            self.lora_ema = LoRAExponentialMovingAverage(
                model=self.model,
                decay=self.training_config.ema_decay,
            )

    def _load_pretrained_weights(self) -> None:
        """Load pretrained OpenFold3 weights from checkpoint."""
        if self.pretrained_checkpoint_path is None:
            logger.warning("No pretrained checkpoint provided, using random weights")
            return

        path = Path(self.pretrained_checkpoint_path)
        logger.info(f"Loading pretrained weights from {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        state_dict, _ = get_state_dict_from_checkpoint(checkpoint)

        # Strip "model." prefix if present (from Lightning checkpoints)
        cleaned = {}
        for k, v in state_dict.items():
            key = k.removeprefix("model.")
            cleaned[key] = v

        self.model.load_state_dict(cleaned, strict=False)
        logger.info("Pretrained weights loaded successfully")

    def _apply_lora(self) -> None:
        """Apply LoRA adapters and freeze base parameters."""
        adapted_count = self.lora_applicator.apply(self.model)
        if adapted_count == 0:
            raise RuntimeError(
                "No layers matched the LoRA target configuration. "
                "Check target_modules and target_blocks in the LoRA config."
            )
        self.lora_applicator.freeze_base_parameters(self.model)

    def _log_parameter_counts(self) -> None:
        """Log parameter count summary."""
        counts = self.lora_applicator.count_parameters(self.model)
        total = counts["total"]
        trainable = counts["trainable"]
        lora = counts["lora"]
        logger.info(
            f"Parameters - Total: {total:,} | "
            f"Trainable (LoRA): {trainable:,} | "
            f"LoRA: {lora:,} | "
            f"Ratio: {trainable / total:.4%}"
        )

    def forward(self, batch: dict) -> tuple[dict, dict]:
        """Run the model forward pass."""
        return self.model(batch)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Execute a single training step."""
        batch, outputs = self.model(batch)
        loss, loss_breakdown = self.loss(batch, outputs, _return_breakdown=True)

        for name, value in loss_breakdown.items():
            self.log(
                f"train/{name}",
                value,
                on_step=True,
                on_epoch=True,
                logger=True,
                sync_dist=False,
                prog_bar=(name == "loss"),
            )

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        """Update LoRA EMA after each training step."""
        if self.lora_ema is not None:
            self.lora_ema.update(self.model)

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        """Execute a single validation step using EMA weights."""
        batch, outputs = self.model(batch)
        _, loss_breakdown = self.loss(batch, outputs, _return_breakdown=True)

        for name, value in loss_breakdown.items():
            self.log(
                f"val/{name}",
                value,
                on_step=False,
                on_epoch=True,
                logger=True,
                sync_dist=False,
            )

    def on_validation_epoch_start(self) -> None:
        """Swap in EMA weights for validation."""
        if self.lora_ema is not None:
            self.lora_ema.apply_shadow(self.model)

    def on_validation_epoch_end(self) -> None:
        """Restore original LoRA weights after validation."""
        if self.lora_ema is not None:
            self.lora_ema.restore(self.model)

    def configure_optimizers(self) -> dict:
        """Configure optimizer and learning rate scheduler for LoRA params."""
        lora_params = list(self.lora_applicator.get_lora_parameters(self.model))
        if not lora_params:
            raise RuntimeError("No LoRA parameters found. Was LoRA applied?")

        optimizer = torch.optim.AdamW(
            lora_params,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            betas=(0.9, 0.95),
        )

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = self.training_config.warmup_steps

        if self.training_config.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(total_steps - warmup_steps, 1)
            )
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, total_iters=warmup_steps
            )

        if warmup_steps > 0 and self.training_config.scheduler == "cosine":
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-6, total_iters=warmup_steps
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[warmup_steps],
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Include LoRA state and config in checkpoint."""
        checkpoint["lora_state_dict"] = (
            self.checkpoint_manager.extract_lora_state_dict(self.model)
        )
        checkpoint["lora_config"] = asdict(self.lora_config)
        if self.lora_ema is not None:
            checkpoint["lora_ema"] = self.lora_ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """Restore LoRA state from checkpoint."""
        lora_state = checkpoint.get("lora_state_dict")
        if lora_state is not None:
            # Will be applied after setup() when model has LoRA layers
            self._cached_lora_weights = lora_state

        ema_state = checkpoint.get("lora_ema")
        if ema_state is not None and self.lora_ema is not None:
            self.lora_ema.load_state_dict(ema_state)
