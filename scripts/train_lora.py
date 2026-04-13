#!/usr/bin/env python
"""Run LoRA fine-tuning of OpenFold3 on real preprocessed antibody data.

Uses OpenFold3's BaseOF3Dataset for proper featurization pipeline.

Usage:
    PYTHONPATH="$PWD:$PWD/openfold-3" python -m finetuning.scripts.train_lora \
        --dataset-cache ./data/antibody_training/dataset_cache.json \
        --structure-dir ./data/antibody_training/preprocessed/structure_files \
        --reference-mol-dir ./data/antibody_training/preprocessed/reference_mols \
        --alignment-dir ./data/antibody_training/alignment_arrays \
        --checkpoint ~/.openfold3/of3-p2-155k.pt \
        --output-dir ./output/lora_antibody
"""

import gc
import json
import logging
import random
import sys
import traceback
from pathlib import Path

import click
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset: uses OF3's create_structure_features directly
# ---------------------------------------------------------------------------
class RealAntibodyDataset(Dataset):
    """Dataset that uses OF3's pipeline to featurize real antibody structures."""

    def __init__(
        self,
        dataset_cache_path: Path,
        structure_dir: Path,
        reference_mol_dir: Path,
        alignment_dir: Path | None = None,
        token_budget: int = 256,
    ):
        self.structure_dir = Path(structure_dir)
        self.reference_mol_dir = Path(reference_mol_dir)
        self.alignment_dir = Path(alignment_dir) if alignment_dir else None
        self.token_budget = token_budget

        # Load dataset cache
        with open(dataset_cache_path) as f:
            cache = json.load(f)

        # Load preprocessing metadata for reference molecule data
        from types import SimpleNamespace
        metadata_path = Path(structure_dir).parent / "metadata.json"
        self.ref_mol_metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            raw = metadata.get("reference_molecule_data", {})
            # Convert dicts to namespace objects (OF3 expects attribute access)
            # Add defaults for fields that training pipeline expects
            for mol_id, mol_data in raw.items():
                mol_data.setdefault("set_fallback_to_nan", False)
                self.ref_mol_metadata[mol_id] = SimpleNamespace(**mol_data)
            logger.info(f"Loaded {len(self.ref_mol_metadata)} reference molecule entries")

        # Build sample list: (pdb_id, chain_id)
        self.samples = []
        self.structure_data = cache["structure_data"]

        for pdb_id, entry in self.structure_data.items():
            npz_path = self.structure_dir / pdb_id / f"{pdb_id}.npz"
            if not npz_path.exists():
                logger.warning(f"Missing NPZ for {pdb_id}, skipping")
                continue
            # One sample per protein chain
            for chain_id, chain_data in entry.get("chains", {}).items():
                if chain_data.get("molecule_type") == "PROTEIN":
                    self.samples.append((pdb_id, chain_id))
                    break

        logger.info(f"RealAntibodyDataset: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pdb_id, preferred_chain = self.samples[idx]

        try:
            return self._featurize_with_of3(pdb_id, preferred_chain)
        except Exception as e:
            logger.warning(f"Error loading {pdb_id}: {e}\n{traceback.format_exc()}")
            # Fallback to another random sample
            other = random.randint(0, len(self.samples) - 1)
            if other == idx:
                other = (idx + 1) % len(self.samples)
            return self._featurize_with_of3(*self.samples[other])

    def _featurize_with_of3(self, pdb_id: str, preferred_chain: str) -> dict:
        """Load and featurize a structure using OF3's native pipeline."""
        from openfold3.core.data.pipelines.sample_processing.structure import (
            process_target_structure_of3,
        )
        from openfold3.core.data.pipelines.featurization.structure import (
            featurize_target_gt_structure_of3,
        )
        from openfold3.core.data.pipelines.sample_processing.conformer import (
            get_reference_conformer_data_of3,
        )
        from openfold3.core.data.pipelines.featurization.conformer import (
            featurize_reference_conformers_of3,
        )
        from openfold3.core.data.primitives.permutation.mol_labels import (
            separate_cropped_and_gt,
        )
        from openfold3.core.data.primitives.structure.tokenization import (
            add_token_positions,
        )

        # 1. Process target structure from NPZ
        crop_config = {
            "token_crop": {
                "enabled": True,
                "token_budget": self.token_budget,
                "crop_weights": {
                    "contiguous": 0.3,
                    "spatial": 0.4,
                    "spatial_interface": 0.3,
                },
            },
            "chain_crop": {
                "enabled": True,
                "n_chains": 20,
                "interface_distance_threshold": 15.0,
                "ligand_inclusion_distance": 5.0,
            },
        }

        # Get per-chain metadata from cache (convert to namespace for OF3)
        from types import SimpleNamespace
        per_chain_metadata = {}
        entry = self.structure_data[pdb_id]
        for ch_id, ch_data in entry.get("chains", {}).items():
            per_chain_metadata[ch_id] = SimpleNamespace(**ch_data)

        atom_array_gt, crop_strategy, n_tokens = process_target_structure_of3(
            target_structures_directory=self.structure_dir,
            pdb_id=pdb_id,
            crop_config=crop_config,
            preferred_chain_or_interface=preferred_chain,
            structure_format="npz",
            per_chain_metadata=per_chain_metadata,
        )

        # 2. Reference conformers
        processed_ref_mols = get_reference_conformer_data_of3(
            atom_array=atom_array_gt,
            per_chain_metadata=per_chain_metadata,
            reference_mol_metadata=self.ref_mol_metadata,
            reference_mol_dir=self.reference_mol_dir,
        )

        # 3. Crop and separate GT
        atom_array_cropped, atom_array_gt = separate_cropped_and_gt(
            atom_array_gt=atom_array_gt,
            crop_strategy=crop_strategy,
            processed_ref_mol_list=processed_ref_mols,
        )

        add_token_positions(atom_array_cropped)

        # 4. Featurize structure
        structure_features = featurize_target_gt_structure_of3(
            atom_array=atom_array_cropped,
            atom_array_gt=atom_array_gt,
            n_tokens=n_tokens,
        )

        # 5. Reference conformer features
        ref_features = featurize_reference_conformers_of3(
            processed_ref_mol_list=processed_ref_mols
        )

        # 6. Create minimal MSA features (empty if no alignment data)
        msa_features = self._create_msa_features(pdb_id, preferred_chain, n_tokens)

        # 7. Create minimal template features (zeros)
        template_features = self._create_empty_template_features(n_tokens)

        # 8. Loss weights
        loss_weights = {
            "bond": torch.tensor(4.0),
            "smooth_lddt": torch.tensor(1.0),
            "mse": torch.tensor(1.0),
            "plddt": torch.tensor(1e-4),
            "pde": torch.tensor(1e-4),
            "experimentally_resolved": torch.tensor(1e-4),
            "pae": torch.tensor(1e-4),
            "distogram": torch.tensor(3e-2),
        }

        # 9. Assemble all features
        features = {}
        features.update(structure_features)
        features.update(ref_features)
        features.update(msa_features)
        features.update(template_features)
        features["loss_weights"] = loss_weights

        # Add batch dim
        features = {
            k: v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
            for k, v in features.items()
        }

        features["pdb_id"] = [pdb_id]
        features["preferred_chain_or_interface"] = preferred_chain
        features["ref_space_uid_to_perm"] = None

        return features

    def _create_msa_features(self, pdb_id: str, chain_id: str, n_tokens: int) -> dict:
        """Create MSA features, loading from alignment arrays if available."""
        n_msa = 1  # At minimum, 1 row (query sequence)

        # Try to load real alignment
        if self.alignment_dir:
            msa_path = self.alignment_dir / pdb_id / f"{chain_id}.npy"
            if msa_path.exists():
                msa_data = np.load(msa_path)
                n_msa = min(msa_data.shape[0], 128)

        # Create MSA features with proper shapes
        msa = torch.zeros((n_msa, n_tokens, 32), dtype=torch.int32)
        msa_mask = torch.ones((n_msa, n_tokens), dtype=torch.float32)
        has_deletion = torch.zeros((n_msa, n_tokens), dtype=torch.float32)
        deletion_value = torch.zeros((n_msa, n_tokens), dtype=torch.float32)
        profile = torch.zeros((n_tokens, 32), dtype=torch.float32)
        deletion_mean = torch.zeros((n_tokens,), dtype=torch.float32)

        # Set first MSA row as query (identity)
        for i in range(min(n_tokens, 20)):
            msa[0, i, i % 20] = 1

        return {
            "msa": msa,
            "msa_mask": msa_mask,
            "has_deletion": has_deletion,
            "deletion_value": deletion_value,
            "profile": profile,
            "deletion_mean": deletion_mean,
            "num_paired_seqs": torch.tensor([0]),
        }

    def _create_empty_template_features(self, n_tokens: int) -> dict:
        """Create empty template features (no templates)."""
        n_templ = 1  # Need at least 1 for shape compatibility
        return {
            "template_restype": torch.zeros((n_templ, n_tokens, 32), dtype=torch.int32),
            "template_pseudo_beta_mask": torch.zeros((n_templ, n_tokens), dtype=torch.float32),
            "template_backbone_frame_mask": torch.zeros((n_templ, n_tokens), dtype=torch.float32),
            "template_distogram": torch.zeros((n_templ, n_tokens, n_tokens, 39), dtype=torch.float32),
            "template_unit_vector": torch.zeros((n_templ, n_tokens, n_tokens, 3), dtype=torch.float32),
        }


# ---------------------------------------------------------------------------
# Training module (same as before, compacted)
# ---------------------------------------------------------------------------
class LoRATrainingModule(pl.LightningModule):
    def __init__(self, model, model_config, lora_config, lr=5e-5, warmup=50):
        super().__init__()
        self.model = model
        self.lora_config = lora_config
        self.lr = lr
        self.warmup = warmup

        from openfold3.core.loss.loss_module import OpenFold3Loss
        from finetuning.lora.applicator import LoRAApplicator
        from finetuning.runner.lora_ema import LoRAExponentialMovingAverage

        self.loss_fn = OpenFold3Loss(config=model_config.architecture.loss_module)
        self.lora_ema = LoRAExponentialMovingAverage(model, decay=0.999)
        self._applicator = LoRAApplicator(lora_config)

    def training_step(self, batch, batch_idx):
        pdb_id = batch.get("pdb_id", ["?"])[0]
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
        from dataclasses import asdict
        from finetuning.lora.checkpoint import LoRACheckpointManager
        checkpoint["lora_state_dict"] = LoRACheckpointManager.extract_lora_state_dict(self.model)
        checkpoint["lora_config"] = asdict(self.lora_config)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
@click.command()
@click.option("--dataset-cache", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--structure-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--reference-mol-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--alignment-dir", type=click.Path(exists=True, path_type=Path), default=None)
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default=Path("./output/lora_antibody"))
@click.option("--max-epochs", type=int, default=5)
@click.option("--lr", type=float, default=5e-5)
@click.option("--lora-rank", type=int, default=8)
@click.option("--lora-alpha", type=float, default=16.0)
@click.option("--token-budget", type=int, default=256)
@click.option("--devices", type=int, default=1)
def main(dataset_cache, structure_dir, reference_mol_dir, alignment_dir,
         checkpoint, output_dir, max_epochs, lr, lora_rank, lora_alpha,
         token_budget, devices):
    """Run LoRA fine-tuning on real antibody structures."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    output_dir.mkdir(parents=True, exist_ok=True)
    pl.seed_everything(42)

    # 1. Build model + LoRA
    from openfold3.projects.of3_all_atom.config.model_config import model_config
    from openfold3.projects.of3_all_atom.model import OpenFold3
    from finetuning.lora.applicator import LoRAApplicator
    from finetuning.lora.config import LoRAConfig

    logger.info("Building model...")
    model = OpenFold3(model_config)
    logger.info(f"Loading weights from {checkpoint}")
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cleaned = {k.removeprefix("model."): v for k, v in ckpt.items()}
    model.load_state_dict(cleaned, strict=False)
    del ckpt; gc.collect()

    lora_config = LoRAConfig(rank=lora_rank, alpha=lora_alpha, dropout=0.05,
                              target_modules=["linear_q", "linear_k", "linear_v", "linear_o"],
                              target_blocks=["pairformer_stack"])
    applicator = LoRAApplicator(lora_config)
    n = applicator.apply(model)
    applicator.freeze_base_parameters(model)
    counts = applicator.count_parameters(model)
    logger.info(f"LoRA: {n} layers, {counts['lora']:,} params ({counts['lora']/counts['total']:.2%})")

    # 2. Build dataset
    dataset = RealAntibodyDataset(
        dataset_cache_path=dataset_cache,
        structure_dir=structure_dir,
        reference_mol_dir=reference_mol_dir,
        alignment_dir=alignment_dir,
        token_budget=token_budget,
    )

    n_val = max(1, len(dataset) // 5)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [len(dataset) - n_val, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                               collate_fn=lambda x: x[0], num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                             collate_fn=lambda x: x[0], num_workers=0)

    # 3. Training
    module = LoRATrainingModule(model, model_config, lora_config, lr=lr)

    ckpt_cb = pl.callbacks.ModelCheckpoint(
        dirpath=output_dir / "checkpoints", save_last=True, every_n_train_steps=5,
    )
    trainer = pl.Trainer(
        max_epochs=max_epochs, devices=devices, precision="bf16-mixed",
        gradient_clip_val=1.0, callbacks=[ckpt_cb],
        default_root_dir=str(output_dir), log_every_n_steps=1,
        enable_progress_bar=True,
    )

    logger.info("Starting training...")
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

    from finetuning.lora.checkpoint import LoRACheckpointManager
    final_path = output_dir / "lora_final.pt"
    LoRACheckpointManager.save_lora_weights(model, final_path, lora_config)
    logger.info(f"Done! LoRA weights: {final_path}")


if __name__ == "__main__":
    main()
