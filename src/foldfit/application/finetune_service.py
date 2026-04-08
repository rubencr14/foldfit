"""Fine-tuning service: orchestrates the full LoRA fine-tuning pipeline."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader

from foldfit.domain.entities import FinetuneJob
from foldfit.domain.interfaces import CheckpointPort, DatasetPort, ModelPort, MsaPort, PeftPort
from foldfit.domain.value_objects import FoldfitConfig
from foldfit.infrastructure.data.structure_dataset import StructureDataset, collate_structure_batch
from foldfit.infrastructure.openfold.featurizer import OpenFoldFeaturizer
from foldfit.infrastructure.openfold.loss import OpenFoldLoss
from foldfit.infrastructure.training.trainer import Trainer

logger = logging.getLogger(__name__)


class FinetuneService:
    """Orchestrates the full fine-tuning pipeline.

    Coordinates model loading, LoRA injection, dataset preparation,
    training, and checkpoint saving.

    Args:
        model: Model port for loading and running the structure model.
        peft: PEFT port for LoRA injection.
        dataset: Dataset port for fetching structure files.
        msa: MSA port for alignment computation.
        checkpoint: Checkpoint port for saving artifacts.
    """

    def __init__(
        self,
        model: ModelPort,
        peft: PeftPort,
        dataset: DatasetPort,
        msa: MsaPort,
        checkpoint: CheckpointPort,
    ) -> None:
        self._model = model
        self._peft = peft
        self._dataset = dataset
        self._msa = msa
        self._checkpoint = checkpoint

    def run(self, config: FoldfitConfig) -> FinetuneJob:
        """Execute the fine-tuning pipeline.

        Steps:
            1. Fetch structure files via dataset port.
            2. Load model and freeze trunk.
            3. Inject LoRA adapters.
            4. Build dataloaders with train/val split.
            5. Run training loop.
            6. Save checkpoint.

        Args:
            config: Full configuration object.

        Returns:
            FinetuneJob with status and metrics.
        """
        job = FinetuneJob(config=config, status="running")

        try:
            # 1. Fetch data
            pdb_paths = self._dataset.fetch_structures(config.data)
            if not pdb_paths:
                raise ValueError("No structure files found")
            logger.info(f"Found {len(pdb_paths)} structures")

            # 2. Load model
            self._model.load(
                weights_path=config.model.weights_path,
                device=config.model.device,
            )
            self._model.freeze_trunk()
            self._model.train_mode(True)

            # 3. Inject LoRA
            target_module = self._model.get_peft_target_module()
            lora_config = config.lora
            self._peft.apply(target_module, lora_config)

            # 4. Apply gradient checkpointing (essential for fitting in GPU memory)
            if config.training.gradient_checkpointing:
                from foldfit.infrastructure.training.checkpointing import apply_gradient_checkpointing

                nn_model_raw = self._model.get_model()
                n_ckpt = apply_gradient_checkpointing(nn_model_raw)
                logger.info(f"Gradient checkpointing applied to {n_ckpt} blocks")

            # 5. Build dataloaders
            train_loader, val_loader = self._build_dataloaders(config, pdb_paths)

            # 6. Train
            trainer = Trainer(config.training)
            loss_fn = OpenFoldLoss()
            nn_model = self._model.get_model()

            # OpenFold needs eval() mode even during training
            # and the DataLoader yields (features_dict, labels) tuples
            model_adapter = self._model

            def openfold_forward(model: Any, batch: Any) -> dict:
                features, _labels = batch
                model_adapter.train_mode(True)  # forces eval()
                output = model_adapter.forward(features)
                return loss_fn(output.extra.get("_raw_outputs", {}), features)

            history = trainer.fit(
                model=nn_model,
                loss_fn=loss_fn,
                peft=self._peft,
                head=None,
                train_loader=train_loader,
                val_loader=val_loader,
                checkpoint_dir=config.output.checkpoint_dir,
                model_forward_fn=openfold_forward,
            )

            # 6. Save final checkpoint
            self._checkpoint.save(
                path=Path(config.output.checkpoint_dir) / "final",
                peft=self._peft,
                head=None,
                training_state={},
                metadata={
                    "config": config.model_dump(),
                    "history": history,
                },
            )

            job.status = "completed"
            # Ensure metrics are JSON-serializable (no torch tensors)
            raw_metrics = history[-1] if history else {}
            job.metrics = {
                k: float(v) if hasattr(v, "item") else v
                for k, v in raw_metrics.items()
            }

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"Fine-tuning failed:\n{tb}")
            job.status = "failed"
            job.metrics = {"error": str(e) or tb.split("\n")[-2]}

        finally:
            # Always free GPU memory after training
            import gc
            import torch
            self._model = None  # type: ignore[assignment]
            self._peft = None  # type: ignore[assignment]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info(f"GPU freed: {torch.cuda.memory_allocated()/1024**3:.1f} GB remaining")

        return job

    def _build_dataloaders(
        self,
        config: FoldfitConfig,
        pdb_paths: list[Path],
    ) -> tuple[DataLoader, DataLoader | None]:  # type: ignore[type-arg]
        """Split data and build train/val dataloaders."""
        random.seed(config.data.split_seed)
        shuffled = list(pdb_paths)
        random.shuffle(shuffled)

        n = len(shuffled)
        n_val = int(n * config.data.val_frac)
        n_test = int(n * config.data.test_frac)
        n_train = n - n_val - n_test

        train_paths = shuffled[:n_train]
        val_paths = shuffled[n_train : n_train + n_val]

        # Build featurizer
        featurizer = OpenFoldFeaturizer(max_seq_len=config.data.max_seq_len)

        train_ds = StructureDataset(
            pdb_paths=train_paths,
            featurizer=featurizer,
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=collate_structure_batch,
        )

        val_loader = None
        if val_paths:
            val_ds = StructureDataset(
                pdb_paths=val_paths,
                featurizer=featurizer,
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_structure_batch,
            )

        return train_loader, val_loader
