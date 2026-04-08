"""CLI entrypoint for foldfit operations."""

from typing import Optional

import typer

from foldfit.application.finetune_service import FinetuneService
from foldfit.application.inference_service import InferenceService
from foldfit.application.msa_service import MsaService
from foldfit.config import load_config
from foldfit.domain.value_objects import ModelConfig, MsaConfig
from foldfit.infrastructure.checkpoint_store import FileCheckpointStore
from foldfit.infrastructure.data.msa_provider import MsaProvider
from foldfit.infrastructure.data.sabdab_repository import SabdabRepository
from foldfit.infrastructure.openfold.adapter import OpenFoldAdapter
from foldfit.infrastructure.peft.injector import LoraInjector

app = typer.Typer(name="foldfit", help="Fine-tune OpenFold on antibody data using LoRA/QLoRA.")


@app.command()
def finetune(
    config: str = typer.Option("config.yaml", help="Path to config YAML file"),
) -> None:
    """Run fine-tuning with the given configuration."""
    cfg = load_config(config)

    service = FinetuneService(
        model=OpenFoldAdapter(),
        peft=LoraInjector(),
        dataset=SabdabRepository(),
        msa=MsaProvider(cfg.msa),
        checkpoint=FileCheckpointStore(),
    )

    typer.echo(f"Starting fine-tuning with config: {config}")
    job = service.run(cfg)
    typer.echo(f"Job {job.id}: {job.status}")
    if job.metrics:
        typer.echo(f"Metrics: {job.metrics}")


@app.command()
def predict(
    sequence: str = typer.Argument(help="Amino acid sequence"),
    adapter_path: Optional[str] = typer.Option(None, help="Path to LoRA adapter checkpoint"),
    weights_path: Optional[str] = typer.Option(None, help="Path to base model weights"),
    device: str = typer.Option("cuda", help="Device"),
) -> None:
    """Run structure prediction on a sequence."""
    model_config = ModelConfig(weights_path=weights_path, device=device)
    service = InferenceService(
        model=OpenFoldAdapter(),
        peft=LoraInjector(),
        checkpoint=FileCheckpointStore(),
    )
    service.load(model_config=model_config, adapter_path=adapter_path)
    typer.echo(f"Model loaded. Predicting structure for sequence of length {len(sequence)}...")


@app.command()
def msa(
    sequence: str = typer.Argument(help="Amino acid sequence"),
    backend: str = typer.Option("single", help="MSA backend: single, precomputed, colabfold"),
    msa_dir: Optional[str] = typer.Option(None, help="Directory for precomputed MSAs"),
) -> None:
    """Compute MSA for a sequence."""
    config = MsaConfig(backend=backend, msa_dir=msa_dir)
    provider = MsaProvider(config)
    service = MsaService(provider)

    result = service.compute(sequence, "query")
    typer.echo(f"MSA: {result['msa'].shape[0]} sequences, length {result['msa'].shape[1]}")


if __name__ == "__main__":
    app()
