#!/usr/bin/env python
"""Run LoRA fine-tuning using OpenFold3's native data pipeline.

Constructs an OF3 DataModule with WeightedPDBDataset pointing to our
preprocessed antibody data, then applies LoRA to the model.

Usage:
    PYTHONPATH="$PWD:$PWD/openfold-3" python -m finetuning.scripts.train_lora \
        --dataset-cache ./data/antibody_training/dataset_cache.json \
        --structure-dir ./data/antibody_training/preprocessed/structure_files \
        --reference-mol-dir ./data/antibody_training/preprocessed/reference_mols \
        --alignment-dir ./data/antibody_training/alignments \
        --checkpoint ~/.openfold3/of3-p2-155k.pt \
        --output-dir ./output/lora_antibody
"""

import gc
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Optional

import pytorch_lightning as pl
import torch
import torch.utils.data
import typer

from openfold3.core.data.framework.single_datasets.abstract_single import (
    DATASET_REGISTRY,
)
from openfold3.core.loss.loss_module import OpenFold3Loss
from openfold3.projects.of3_all_atom.config.dataset_configs import (
    DatasetConfigRegistry,
    TrainingDatasetPaths,
)
from openfold3.projects.of3_all_atom.config.model_config import model_config
from openfold3.projects.of3_all_atom.model import OpenFold3

from finetuning.lora.applicator import LoRAApplicator
from finetuning.lora.checkpoint import LoRACheckpointManager
from finetuning.lora.config import LoRAConfig
from finetuning.runner.lora_ema import LoRAExponentialMovingAverage

logger = logging.getLogger(__name__)

SKIP_BATCH_DIM = {"loss_weights", "ref_space_uid_to_perm"}


def build_of3_dataset(
    dataset_cache: Path,
    structure_dir: Path,
    reference_mol_dir: Path,
    alignment_dir: Path | None,
    token_budget: int,
):
    """Build a WeightedPDBDataset using OF3's native config system."""
    WeightedPDBConfig = DatasetConfigRegistry.get("WeightedPDBDataset")

    paths = TrainingDatasetPaths(
        dataset_cache_file=dataset_cache,
        target_structures_directory=structure_dir,
        target_structure_file_format="npz",
        reference_molecule_directory=reference_mol_dir,
        alignments_directory=alignment_dir,
    )

    config = WeightedPDBConfig(
        name="antibody-lora",
        debug_mode=True,
        dataset_paths=paths,
        crop={
            "token_crop": {
                "enabled": True,
                "token_budget": token_budget,
                "crop_weights": {
                    "contiguous": 0.3,
                    "spatial": 0.4,
                    "spatial_interface": 0.3,
                },
            },
            "chain_crop": {"enabled": True},
        },
    )

    dataset = DATASET_REGISTRY["WeightedPDBDataset"](config)
    logger.info(f"OF3 Dataset: {len(dataset)} samples")
    return dataset


def collate_fn(batch_list: list[dict]) -> dict:
    """Collate single sample into batched form by adding batch dim.

    Skips loss_weights (scalars) and ref_space_uid_to_perm (int-keyed dict).
    """
    batch = batch_list[0]

    def add_batch_dim(d: dict, skip_children: bool = False) -> dict:
        out = {}
        for k, v in d.items():
            if k in SKIP_BATCH_DIM or skip_children:
                out[k] = v
            elif isinstance(v, torch.Tensor) and v.ndim > 0:
                out[k] = v.unsqueeze(0)
            elif isinstance(v, dict) and all(isinstance(kk, str) for kk in v.keys()):
                out[k] = add_batch_dim(v)
            else:
                out[k] = v
        return out

    return add_batch_dim(batch)


class LoRATrainingModule(pl.LightningModule):
    """Lightning module for LoRA training with OpenFold3."""

    def __init__(self, model, model_cfg, lora_config, lr=5e-5, warmup=50):
        super().__init__()
        self.model = model
        self.lora_config = lora_config
        self.lr = lr
        self.warmup = warmup
        self.loss_fn = OpenFold3Loss(config=model_cfg.architecture.loss_module)
        self.lora_ema = LoRAExponentialMovingAverage(model, decay=0.999)
        self._applicator = LoRAApplicator(lora_config)

    def training_step(self, batch, batch_idx):
        pdb_id = batch.get("pdb_id", "?")
        try:
            batch, outputs = self.model(batch)
            loss, breakdown = self.loss_fn(batch, outputs, _return_breakdown=True)
            for name, value in breakdown.items():
                if isinstance(value, torch.Tensor) and value.numel() == 1:
                    self.log(f"train/{name}", value, prog_bar=(name == "loss"))
            if not torch.isfinite(loss):
                logger.warning(f"Non-finite loss for {pdb_id}")
                return None
            return loss
        except torch.OutOfMemoryError:
            logger.warning(f"OOM for {pdb_id}")
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            logger.warning(f"Train error {pdb_id}: {e}")
            return None

    def on_train_batch_end(self, outputs, batch, batch_idx):
        device = next(self.model.parameters()).device
        if self.lora_ema.shadow and next(iter(self.lora_ema.shadow.values())).device != device:
            self.lora_ema.to(device)
        self.lora_ema.update(self.model)

    def on_validation_epoch_start(self):
        self.lora_ema.apply_shadow(self.model)

    def on_validation_epoch_end(self):
        self.lora_ema.restore(self.model)

    def validation_step(self, batch, batch_idx):
        try:
            batch, outputs = self.model(batch)
            _, bd = self.loss_fn(batch, outputs, _return_breakdown=True)
            for n, v in bd.items():
                if isinstance(v, torch.Tensor) and v.numel() == 1:
                    self.log(f"val/{n}", v, sync_dist=True)
        except Exception as e:
            logger.warning(f"Val error: {e}")

    def configure_optimizers(self):
        params = list(self._applicator.get_lora_parameters(self.model))
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01, betas=(0.9, 0.95))
        total = max(self.trainer.estimated_stepping_batches - self.warmup, 1)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
        if self.warmup > 0:
            w = torch.optim.lr_scheduler.LinearLR(opt, start_factor=1e-6, total_iters=self.warmup)
            sched = torch.optim.lr_scheduler.SequentialLR(opt, [w, sched], milestones=[self.warmup])
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["lora_state_dict"] = LoRACheckpointManager.extract_lora_state_dict(self.model)
        checkpoint["lora_config"] = asdict(self.lora_config)


app = typer.Typer()


@app.command()
def main(
    dataset_cache: Annotated[Path, typer.Option(help="Path to dataset_cache.json")],
    structure_dir: Annotated[Path, typer.Option(help="Path to preprocessed structure_files/")],
    reference_mol_dir: Annotated[Path, typer.Option(help="Path to reference_mols/")],
    checkpoint: Annotated[Path, typer.Option(help="Path to pretrained OF3 checkpoint")],
    alignment_dir: Annotated[Optional[Path], typer.Option(help="Path to alignments/")] = None,
    output_dir: Annotated[Path, typer.Option(help="Output directory")] = Path("./output/lora_antibody"),
    max_epochs: Annotated[int, typer.Option(help="Number of training epochs")] = 5,
    lr: Annotated[float, typer.Option(help="Learning rate")] = 5e-5,
    lora_rank: Annotated[int, typer.Option(help="LoRA rank")] = 8,
    lora_alpha: Annotated[float, typer.Option(help="LoRA alpha")] = 16.0,
    token_budget: Annotated[int, typer.Option(help="Max tokens per crop")] = 128,
    devices: Annotated[int, typer.Option(help="Number of GPUs")] = 1,
):
    """Run LoRA fine-tuning on real antibody structures."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(42)
    torch.set_float32_matmul_precision("medium")

    # 1. Build model + LoRA
    logger.info("Building model...")
    model = OpenFold3(model_config)
    logger.info(f"Loading weights from {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cleaned = {k.removeprefix("model."): v for k, v in ckpt.items()}
    model.load_state_dict(cleaned, strict=False)
    del ckpt
    gc.collect()

    lora_config = LoRAConfig(
        rank=lora_rank, alpha=lora_alpha, dropout=0.05,
        target_modules=["linear_q", "linear_k", "linear_v", "linear_o"],
        target_blocks=["pairformer_stack"],
    )
    applicator = LoRAApplicator(lora_config)
    n = applicator.apply(model)
    applicator.freeze_base_parameters(model)
    counts = applicator.count_parameters(model)
    logger.info(f"LoRA: {n} layers, {counts['lora']:,} params ({counts['lora']/counts['total']:.2%})")

    # 2. Build OF3 dataset
    dataset = build_of3_dataset(
        dataset_cache=dataset_cache,
        structure_dir=structure_dir,
        reference_mol_dir=reference_mol_dir,
        alignment_dir=alignment_dir,
        token_budget=token_budget,
    )

    n_val = max(1, len(dataset) // 5)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 3. Training
    module = LoRATrainingModule(model, model_config, lora_config, lr=lr)

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir / "checkpoints", save_last=True, every_n_train_steps=5)
    trainer = pl.Trainer(
        max_epochs=max_epochs, devices=devices, precision="bf16-mixed",
        gradient_clip_val=1.0, callbacks=[ckpt_cb],
        default_root_dir=str(output_dir), log_every_n_steps=1,
        enable_progress_bar=True,
    )

    logger.info("Starting training...")
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    final_path = output_dir / "lora_final.pt"
    LoRACheckpointManager.save_lora_weights(model, final_path, lora_config)
    logger.info(f"Done! LoRA weights: {final_path}")


if __name__ == "__main__":
    app()
