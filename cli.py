"""CLI entry point for LoRA fine-tuning of OpenFold3 on antibodies."""

import logging
from pathlib import Path
from typing import Annotated, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from openfold3.core.utils.checkpoint_loading_utils import get_state_dict_from_checkpoint
from openfold3.projects.of3_all_atom.config.model_config import model_config
from openfold3.projects.of3_all_atom.model import OpenFold3

import typer

from finetuning.config.finetune_config import FinetuneConfig
from finetuning.data.antibody_dataset import AntibodyPDBDataset
from finetuning.data.antibody_fetcher import RCSBAntibodyFetcher
from finetuning.lora.applicator import LoRAApplicator
from finetuning.lora.checkpoint import LoRACheckpointManager
from finetuning.lora.config import LoRAConfig
from finetuning.runner.lora_runner import LoRAFineTuningRunner

logger = logging.getLogger(__name__)

app = typer.Typer(help="OpenFold3 LoRA Fine-Tuning CLI for antibody structures.")


@app.callback()
def callback(verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Enable debug logging")] = False):
    """Global options."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@app.command()
def train(
    config: Annotated[Path, typer.Option(help="Path to fine-tuning YAML config file")],
    pretrained_checkpoint: Annotated[Optional[Path], typer.Option(help="Override pretrained checkpoint path")] = None,
    devices: Annotated[int, typer.Option(help="Number of GPUs")] = 1,
):
    """Run LoRA fine-tuning on antibody structures."""
    ft_config = FinetuneConfig.from_yaml(config)

    if pretrained_checkpoint is not None:
        ft_config.pretrained_checkpoint = pretrained_checkpoint

    pl.seed_everything(ft_config.seed)

    runner = LoRAFineTuningRunner(
        model_config=model_config,
        lora_config=ft_config.lora,
        training_config=ft_config.training,
        pretrained_checkpoint_path=ft_config.pretrained_checkpoint,
        log_dir=ft_config.output_dir,
    )

    dataset = AntibodyPDBDataset(
        pdb_ids=ft_config.dataset.pdb_ids,
        cache_dir=ft_config.dataset.cache_dir,
        max_resolution=ft_config.dataset.max_resolution,
        require_heavy=ft_config.dataset.require_heavy_chain,
        require_light=ft_config.dataset.require_light_chain,
        cdr_scheme=ft_config.dataset.cdr_scheme,
        search_max_results=ft_config.dataset.search_max_results,
    )
    train_ds, val_ds = dataset.get_train_val_split(
        train_ratio=ft_config.dataset.train_split, seed=ft_config.seed,
    )

    train_loader = DataLoader(train_ds, batch_size=ft_config.training.batch_size,
                               shuffle=True, num_workers=ft_config.training.num_workers)
    val_loader = DataLoader(val_ds, batch_size=ft_config.training.batch_size,
                             shuffle=False, num_workers=ft_config.training.num_workers)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ft_config.output_dir / "checkpoints",
        save_top_k=3, monitor="val/loss", mode="min", save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=ft_config.training.max_epochs,
        devices=devices,
        precision=ft_config.training.precision,
        gradient_clip_val=ft_config.training.gradient_clip_val,
        callbacks=[checkpoint_callback],
        default_root_dir=str(ft_config.output_dir),
        log_every_n_steps=10,
    )

    trainer.fit(runner, train_dataloaders=train_loader, val_dataloaders=val_loader)
    logger.info(f"Training complete. Output saved to {ft_config.output_dir}")


@app.command()
def merge(
    base_checkpoint: Annotated[Path, typer.Option(help="Path to pretrained OF3 checkpoint")],
    lora_checkpoint: Annotated[Path, typer.Option(help="Path to LoRA-only checkpoint")],
    output: Annotated[Path, typer.Option(help="Path to save merged checkpoint")],
    rank: Annotated[int, typer.Option(help="LoRA rank used during training")] = 8,
    alpha: Annotated[float, typer.Option(help="LoRA alpha used during training")] = 16.0,
):
    """Merge LoRA weights into the base model checkpoint."""
    lora_ckpt = torch.load(lora_checkpoint, map_location="cpu", weights_only=True)
    lora_config_dict = lora_ckpt.get("lora_config", {})
    lora_config = LoRAConfig(
        rank=lora_config_dict.get("rank", rank),
        alpha=lora_config_dict.get("alpha", alpha),
        target_modules=lora_config_dict.get("target_modules", ["linear_q", "linear_k", "linear_v", "linear_o"]),
        target_blocks=lora_config_dict.get("target_blocks", ["pairformer_stack"]),
    )

    model = OpenFold3(model_config)
    base_ckpt = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    state_dict, _ = get_state_dict_from_checkpoint(base_ckpt)
    cleaned = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)

    LoRAApplicator(lora_config).apply(model)
    LoRACheckpointManager.load_lora_weights(model, lora_checkpoint)
    LoRACheckpointManager.merge_and_save(model, output)

    typer.echo(f"Merged checkpoint saved to {output}")


@app.command("fetch-data")
def fetch_data(
    output_dir: Annotated[Path, typer.Option(help="Directory to save downloaded structures")],
    max_resolution: Annotated[float, typer.Option(help="Max resolution cutoff in Angstroms")] = 3.0,
    max_results: Annotated[int, typer.Option(help="Max number of structures to fetch")] = 1000,
):
    """Fetch antibody structures from RCSB PDB."""
    fetcher = RCSBAntibodyFetcher(cache_dir=output_dir, max_resolution=max_resolution)
    pdb_ids = fetcher.search(max_results=max_results)
    typer.echo(f"Found {len(pdb_ids)} antibody structures")

    fetcher.download(pdb_ids)
    typer.echo(f"Downloaded structures to {output_dir}")

    ids_file = output_dir / "pdb_ids.txt"
    with open(ids_file, "w") as f:
        f.write("# Antibody PDB IDs fetched from RCSB\n")
        for pdb_id in pdb_ids:
            f.write(f"{pdb_id}\n")
    typer.echo(f"PDB ID list saved to {ids_file}")


if __name__ == "__main__":
    app()
