"""CLI entry point for LoRA fine-tuning of OpenFold3 on antibodies."""

import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """OpenFold3 LoRA Fine-Tuning CLI for antibody structures."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@cli.command()
@click.option(
    "--config", "-c",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to fine-tuning YAML config file.",
)
@click.option(
    "--pretrained-checkpoint",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Path to pretrained OpenFold3 checkpoint (overrides config).",
)
@click.option(
    "--devices",
    type=int,
    default=1,
    help="Number of GPUs to use.",
)
def train(config: Path, pretrained_checkpoint: Path | None, devices: int) -> None:
    """Run LoRA fine-tuning on antibody structures."""
    import pytorch_lightning as pl

    from openfold3.projects.of3_all_atom.config.model_config import model_config

    from finetuning.config.finetune_config import FinetuneConfig
    from finetuning.runner.lora_runner import LoRAFineTuningRunner

    ft_config = FinetuneConfig.from_yaml(config)

    if pretrained_checkpoint is not None:
        ft_config.pretrained_checkpoint = pretrained_checkpoint

    pl.seed_everything(ft_config.seed)

    # Build the runner
    runner = LoRAFineTuningRunner(
        model_config=model_config,
        lora_config=ft_config.lora,
        training_config=ft_config.training,
        pretrained_checkpoint_path=ft_config.pretrained_checkpoint,
        log_dir=ft_config.output_dir,
    )

    # Setup data
    from finetuning.data.antibody_dataset import AntibodyPDBDataset
    from torch.utils.data import DataLoader

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
        train_ratio=ft_config.dataset.train_split,
        seed=ft_config.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=ft_config.training.batch_size,
        shuffle=True,
        num_workers=ft_config.training.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=ft_config.training.batch_size,
        shuffle=False,
        num_workers=ft_config.training.num_workers,
    )

    # Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=ft_config.output_dir / "checkpoints",
        save_top_k=3,
        monitor="val/loss",
        mode="min",
        save_last=True,
    )

    # Trainer
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


@cli.command()
@click.option(
    "--base-checkpoint",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to pretrained OpenFold3 checkpoint.",
)
@click.option(
    "--lora-checkpoint",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to LoRA-only checkpoint.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to save the merged checkpoint.",
)
@click.option("--rank", type=int, default=8, help="LoRA rank used during training.")
@click.option("--alpha", type=float, default=16.0, help="LoRA alpha used during training.")
def merge(
    base_checkpoint: Path,
    lora_checkpoint: Path,
    output: Path,
    rank: int,
    alpha: float,
) -> None:
    """Merge LoRA weights into the base model checkpoint."""
    import torch

    from openfold3.core.utils.checkpoint_loading_utils import (
        get_state_dict_from_checkpoint,
    )
    from openfold3.projects.of3_all_atom.config.model_config import model_config
    from openfold3.projects.of3_all_atom.model import OpenFold3

    from finetuning.lora.applicator import LoRAApplicator
    from finetuning.lora.checkpoint import LoRACheckpointManager
    from finetuning.lora.config import LoRAConfig

    # Load LoRA checkpoint to get config
    lora_ckpt = torch.load(lora_checkpoint, map_location="cpu", weights_only=True)
    lora_config_dict = lora_ckpt.get("lora_config", {})
    lora_config = LoRAConfig(
        rank=lora_config_dict.get("rank", rank),
        alpha=lora_config_dict.get("alpha", alpha),
        target_modules=lora_config_dict.get(
            "target_modules", ["linear_q", "linear_k", "linear_v", "linear_o"]
        ),
        target_blocks=lora_config_dict.get(
            "target_blocks", ["pairformer_stack", "diffusion_module"]
        ),
    )

    # Build and load base model
    model = OpenFold3(model_config)
    base_ckpt = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    state_dict, _ = get_state_dict_from_checkpoint(base_ckpt)
    cleaned = {k.removeprefix("model."): v for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)

    # Apply LoRA, load weights, merge
    applicator = LoRAApplicator(lora_config)
    applicator.apply(model)
    LoRACheckpointManager.load_lora_weights(model, lora_checkpoint)
    LoRACheckpointManager.merge_and_save(model, output)

    click.echo(f"Merged checkpoint saved to {output}")


@cli.command("fetch-data")
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory to save downloaded structures.",
)
@click.option(
    "--max-resolution",
    type=float,
    default=3.0,
    help="Maximum resolution cutoff in Angstroms.",
)
@click.option(
    "--max-results",
    type=int,
    default=1000,
    help="Maximum number of structures to fetch.",
)
def fetch_data(output_dir: Path, max_resolution: float, max_results: int) -> None:
    """Fetch antibody structures from RCSB PDB."""
    from finetuning.data.antibody_fetcher import RCSBAntibodyFetcher

    fetcher = RCSBAntibodyFetcher(
        cache_dir=output_dir, max_resolution=max_resolution
    )
    pdb_ids = fetcher.search(max_results=max_results)
    click.echo(f"Found {len(pdb_ids)} antibody structures")

    fetcher.download(pdb_ids)
    click.echo(f"Downloaded structures to {output_dir}")

    # Save PDB IDs list
    ids_file = output_dir / "pdb_ids.txt"
    with open(ids_file, "w") as f:
        f.write("# Antibody PDB IDs fetched from RCSB\n")
        for pdb_id in pdb_ids:
            f.write(f"{pdb_id}\n")
    click.echo(f"PDB ID list saved to {ids_file}")


if __name__ == "__main__":
    cli()
