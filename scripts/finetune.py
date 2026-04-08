"""Foldfit CLI: fine-tune OpenFold on antibody structures.

Commands:
    download      Download antibody structures from SAbDab/RCSB
    finetune      Run LoRA fine-tuning
    predict       Predict structure from sequence
    evaluate      Evaluate model on a set of PDB structures
    msa           Compute MSA for a single sequence
    generate-msa  Batch-generate MSAs for a directory of PDBs
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="foldfit",
    help="Fine-tune OpenFold on antibody structures using LoRA.",
    no_args_is_help=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("foldfit.cli")


# ── download ──────────────────────────────────────────────────────────────

@app.command()
def download(
    output_dir: Path = typer.Option("./data/sabdab", "-o", "--output-dir", help="Output directory"),
    max_structures: int = typer.Option(200, "-n", "--max", help="Maximum structures to download"),
    resolution: float = typer.Option(3.0, "-r", "--resolution", help="Max resolution in angstroms"),
    antibody_type: str = typer.Option(
        "all", "-t", "--type",
        help="Type filter: all / antibody / nanobody / Fab / scFv / immunoglobulin",
    ),
    organism: Optional[str] = typer.Option(None, help="Organism filter (e.g. 'Homo sapiens')"),
    method: Optional[str] = typer.Option(None, help="Method filter (e.g. 'X-RAY DIFFRACTION')"),
    skip_existing: bool = typer.Option(True, help="Skip PDBs already downloaded"),
) -> None:
    """Download antibody structures from SAbDab/RCSB PDB.

    Queries RCSB for antibody structures (with SAbDab fallback),
    filters by resolution/type/organism, and downloads PDB files.

    Examples:
        # Download 200 antibodies at <=3A resolution
        foldfit download -n 200

        # Only human nanobodies at high resolution
        foldfit download -n 100 -r 2.0 -t nanobody --organism "Homo sapiens"

        # Only X-ray structures
        foldfit download --method "X-RAY DIFFRACTION"
    """
    from foldfit.infrastructure.data.sabdab_repository import SabdabRepository

    output_dir.mkdir(parents=True, exist_ok=True)
    repo = SabdabRepository(cache_dir=output_dir)

    # Check existing
    existing = list(output_dir.glob("*.pdb"))
    if skip_existing and existing:
        typer.echo(f"Found {len(existing)} existing PDBs in {output_dir}")

    # Query PDB IDs
    typer.echo(f"Querying RCSB for antibody structures (resolution<={resolution}A, type={antibody_type})...")
    pdb_ids = repo.query_antibody_pdb_ids(
        resolution_max=resolution,
        max_results=max_structures,
        antibody_type=antibody_type,
        organism=organism or "",
        method=method or "",
    )

    if not pdb_ids:
        typer.echo("No structures found matching the filters.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Found {len(pdb_ids)} PDB IDs. Downloading...")

    success = 0
    skipped = 0
    failed = 0

    for i, pdb_id in enumerate(pdb_ids):
        dest = output_dir / f"{pdb_id}.pdb"

        if skip_existing and dest.exists():
            skipped += 1
            continue

        ok = repo.download_pdb(pdb_id, dest)
        if ok:
            success += 1
            if (success % 20 == 0) or (i == len(pdb_ids) - 1):
                typer.echo(f"  [{i+1}/{len(pdb_ids)}] Downloaded {success} so far...")
        else:
            failed += 1

    total = success + skipped
    typer.echo(
        f"Done: {total} structures in {output_dir} "
        f"({success} new, {skipped} existing, {failed} failed)"
    )


# ── finetune ──────────────────────────────────────────────────────────────

@app.command()
def finetune(
    config: str = typer.Option("config.yaml", help="Path to config YAML file"),
) -> None:
    """Run LoRA fine-tuning with the given configuration."""
    from foldfit.application.finetune_service import FinetuneService
    from foldfit.config import load_config
    from foldfit.infrastructure.checkpoint_store import FileCheckpointStore
    from foldfit.infrastructure.data.msa_provider import MsaProvider
    from foldfit.infrastructure.data.sabdab_repository import SabdabRepository
    from foldfit.infrastructure.openfold.adapter import OpenFoldAdapter
    from foldfit.infrastructure.peft.injector import LoraInjector

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
        for k, v in job.metrics.items():
            typer.echo(f"  {k}: {v}")


# ── predict ───────────────────────────────────────────────────────────────

@app.command()
def predict(
    sequence: str = typer.Argument(help="Amino acid sequence"),
    adapter_path: Optional[str] = typer.Option(None, help="Path to LoRA adapter checkpoint"),
    weights_path: Optional[str] = typer.Option(None, help="Path to base model weights"),
    device: str = typer.Option("cuda", help="Device (cuda / cpu)"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output PDB file path"),
    max_seq_len: int = typer.Option(256, help="Maximum sequence length"),
) -> None:
    """Predict structure from an amino acid sequence."""
    from foldfit.application.inference_service import InferenceService
    from foldfit.domain.value_objects import ModelConfig
    from foldfit.infrastructure.checkpoint_store import FileCheckpointStore
    from foldfit.infrastructure.openfold.adapter import OpenFoldAdapter
    from foldfit.infrastructure.peft.injector import LoraInjector

    model_config = ModelConfig(weights_path=weights_path, device=device)
    service = InferenceService(
        model=OpenFoldAdapter(),
        peft=LoraInjector(),
        checkpoint=FileCheckpointStore(),
    )

    typer.echo(f"Loading model (device={device})...")
    service.load(model_config=model_config, adapter_path=adapter_path)

    typer.echo(f"Predicting structure for sequence of length {len(sequence)}...")
    result = service.predict_from_sequence(sequence, device=device, max_seq_len=max_seq_len)

    if result.get("mean_plddt") is not None:
        typer.echo(f"Mean pLDDT: {result['mean_plddt']:.1f}")

    pdb_string = result.get("pdb_string")
    if pdb_string is None:
        typer.echo("No structure predicted.", err=True)
        raise typer.Exit(1)

    if output:
        Path(output).write_text(pdb_string)
        typer.echo(f"Structure saved to {output}")
    else:
        typer.echo(pdb_string)


# ── evaluate ──────────────────────────────────────────────────────────────

@app.command()
def evaluate(
    pdb_dir: str = typer.Argument(help="Directory with ground-truth PDB files"),
    adapter_path: Optional[str] = typer.Option(None, help="Path to LoRA adapter checkpoint"),
    weights_path: Optional[str] = typer.Option(None, help="Path to base model weights"),
    device: str = typer.Option("cuda", help="Device"),
    max_seq_len: int = typer.Option(256, help="Maximum sequence length"),
    max_structures: int = typer.Option(0, help="Max structures to evaluate (0 = all)"),
) -> None:
    """Evaluate model on a set of PDB structures (CA-RMSD, GDT-TS, pLDDT)."""
    import torch

    from foldfit.application.inference_service import InferenceService
    from foldfit.domain.value_objects import ModelConfig
    from foldfit.infrastructure.checkpoint_store import FileCheckpointStore
    from foldfit.infrastructure.openfold.adapter import OpenFoldAdapter
    from foldfit.infrastructure.openfold.featurizer import OpenFoldFeaturizer
    from foldfit.infrastructure.openfold.metrics import ca_rmsd, gdt_ts, plddt_score
    from foldfit.infrastructure.peft.injector import LoraInjector

    model_config = ModelConfig(weights_path=weights_path, device=device)
    service = InferenceService(
        model=OpenFoldAdapter(),
        peft=LoraInjector(),
        checkpoint=FileCheckpointStore(),
    )

    typer.echo("Loading model...")
    service.load(model_config=model_config, adapter_path=adapter_path)

    featurizer = OpenFoldFeaturizer(max_seq_len=max_seq_len)
    pdb_paths = sorted(
        list(Path(pdb_dir).glob("*.pdb"))
        + list(Path(pdb_dir).glob("*.cif"))
    )
    if max_structures > 0:
        pdb_paths = pdb_paths[:max_structures]

    typer.echo(f"Evaluating on {len(pdb_paths)} structures...")

    all_rmsd, all_gdt, all_plddt = [], [], []

    for i, pdb_path in enumerate(pdb_paths):
        try:
            features = featurizer.from_pdb(pdb_path)
            if not features:
                continue

            batch = {
                k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                for k, v in features.items()
            }

            output = service.predict(batch)
            raw = output.extra.get("_raw_outputs", {})

            # Strip recycling dim from batch for metrics
            clean = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[-1] == 1:
                    clean[k] = v.squeeze(-1)
                else:
                    clean[k] = v

            pred_pos = raw.get("final_atom_positions")
            gt_pos = clean.get("all_atom_positions")
            gt_mask = clean.get("all_atom_mask")

            if pred_pos is not None and gt_pos is not None and gt_mask is not None:
                rmsd_val = ca_rmsd(pred_pos, gt_pos, gt_mask).item()
                gdt_val = gdt_ts(pred_pos, gt_pos, gt_mask).item()
                all_rmsd.append(rmsd_val)
                all_gdt.append(gdt_val)

            plddt_logits = raw.get("plddt_logits")
            if plddt_logits is not None:
                plddt_val = plddt_score(plddt_logits).item()
                all_plddt.append(plddt_val)

            typer.echo(
                f"  [{i+1}/{len(pdb_paths)}] {pdb_path.stem}: "
                f"RMSD={rmsd_val:.2f}A GDT-TS={gdt_val:.3f}"
                + (f" pLDDT={plddt_val:.1f}" if plddt_logits is not None else "")
            )

        except Exception as e:
            typer.echo(f"  [{i+1}/{len(pdb_paths)}] {pdb_path.stem}: FAILED ({e})", err=True)

    if all_rmsd:
        import statistics
        typer.echo("\n--- Summary ---")
        typer.echo(f"  Structures: {len(all_rmsd)}")
        typer.echo(f"  CA-RMSD:    {statistics.mean(all_rmsd):.2f} +/- {statistics.stdev(all_rmsd) if len(all_rmsd) > 1 else 0:.2f} A")
        typer.echo(f"  GDT-TS:     {statistics.mean(all_gdt):.3f} +/- {statistics.stdev(all_gdt) if len(all_gdt) > 1 else 0:.3f}")
        if all_plddt:
            typer.echo(f"  pLDDT:      {statistics.mean(all_plddt):.1f} +/- {statistics.stdev(all_plddt) if len(all_plddt) > 1 else 0:.1f}")
    else:
        typer.echo("No structures evaluated successfully.", err=True)


# ── msa ───────────────────────────────────────────────────────────────────

@app.command()
def msa(
    sequence: str = typer.Argument(help="Amino acid sequence"),
    backend: str = typer.Option("single", help="Backend: single / precomputed / colabfold / local"),
    msa_dir: Optional[str] = typer.Option(None, help="Directory for precomputed MSAs"),
    colabfold_server: str = typer.Option("https://api.colabfold.com", help="ColabFold server URL"),
    tool: str = typer.Option("mmseqs2", help="Local tool: mmseqs2 / hhblits / jackhmmer"),
    database: Optional[list[str]] = typer.Option(None, help="Database path(s) for local backend"),
    tool_binary: Optional[str] = typer.Option(None, help="Path to tool binary"),
    n_cpu: int = typer.Option(4, help="CPUs for local alignment"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Save MSA as .msa.pt"),
) -> None:
    """Compute MSA for a single sequence."""
    import torch

    from foldfit.domain.value_objects import MsaConfig
    from foldfit.infrastructure.data.msa_provider import MsaProvider

    config = MsaConfig(
        backend=backend,
        msa_dir=msa_dir,
        colabfold_server=colabfold_server,
        tool=tool,
        tool_binary=tool_binary,
        database_paths=database or [],
        n_cpu=n_cpu,
    )
    provider = MsaProvider(config)

    typer.echo(f"Computing MSA (backend={backend})...")
    result = provider.get(sequence, "query")

    n_seqs = result["msa"].shape[0]
    seq_len = result["msa"].shape[1]
    typer.echo(f"MSA: {n_seqs} sequences, length {seq_len}")

    if output:
        torch.save(result, output)
        typer.echo(f"Saved to {output}")


# ── generate-msa ─────────────────────────────────────────────────────────

@app.command("generate-msa")
def generate_msa(
    pdb_dir: Path = typer.Option(..., help="Directory containing PDB/mmCIF files"),
    output_dir: Path = typer.Option(..., help="Output directory for MSA files"),
    backend: str = typer.Option("colabfold", help="Backend: colabfold / local / single"),
    max_msa_depth: int = typer.Option(512, help="Maximum MSA depth"),
    delay: float = typer.Option(1.0, help="Delay between API calls (colabfold only)"),
    skip_existing: bool = typer.Option(True, help="Skip PDBs with existing MSA files"),
    colabfold_server: str = typer.Option("https://api.colabfold.com", help="ColabFold server URL"),
    tool: str = typer.Option("mmseqs2", help="Local tool: mmseqs2 / hhblits / jackhmmer"),
    database: Optional[list[str]] = typer.Option(None, help="Database path(s)"),
    tool_binary: Optional[str] = typer.Option(None, help="Path to tool binary"),
    n_cpu: int = typer.Option(4, help="CPUs for local alignment"),
) -> None:
    """Batch-generate MSAs for all PDB structures in a directory."""
    import torch

    from openfold.np import protein as protein_utils
    from openfold.np import residue_constants as rc

    from foldfit.domain.value_objects import MsaConfig
    from foldfit.infrastructure.data.msa_provider import MsaProvider

    output_dir.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(
        list(pdb_dir.glob("*.pdb"))
        + list(pdb_dir.glob("*.cif"))
        + list(pdb_dir.glob("*.mmcif"))
    )
    typer.echo(f"Found {len(pdb_files)} PDB files in {pdb_dir}")

    if not pdb_files:
        typer.echo("No PDB files found.", err=True)
        raise typer.Exit(1)

    config = MsaConfig(
        backend=backend,
        msa_dir=str(output_dir),
        max_msa_depth=max_msa_depth,
        colabfold_server=colabfold_server,
        tool=tool,
        tool_binary=tool_binary,
        database_paths=database or [],
        n_cpu=n_cpu,
    )
    provider = MsaProvider(config)

    success = 0
    failed = 0

    for i, pdb_path in enumerate(pdb_files):
        pdb_id = pdb_path.stem.upper()
        cache_path = output_dir / f"{pdb_id}.msa.pt"

        if skip_existing and cache_path.exists():
            logger.info(f"[{i+1}/{len(pdb_files)}] Skipping {pdb_id} (exists)")
            success += 1
            continue

        try:
            pdb_str = pdb_path.read_text()
            seq = ""
            for chain_id in ["H", "L", "A", "B", None]:
                try:
                    prot = protein_utils.from_pdb_string(pdb_str, chain_id=chain_id)
                    if len(prot.aatype) > 0:
                        seq = "".join(rc.restypes_with_x[a] for a in prot.aatype)
                        break
                except Exception:
                    continue

            if not seq:
                typer.echo(f"  [{i+1}/{len(pdb_files)}] {pdb_id}: no sequence, skipping", err=True)
                failed += 1
                continue

            result = provider.get(sequence=seq, pdb_id=pdb_id)
            torch.save(result, cache_path)
            n_seqs = result["msa"].shape[0]
            typer.echo(f"  [{i+1}/{len(pdb_files)}] {pdb_id}: {n_seqs} seqs, len {len(seq)}")
            success += 1

            if backend == "colabfold" and delay > 0:
                time.sleep(delay)

        except Exception as e:
            typer.echo(f"  [{i+1}/{len(pdb_files)}] {pdb_id}: FAILED ({e})", err=True)
            failed += 1

    typer.echo(f"Done: {success}/{len(pdb_files)} succeeded, {failed} failed")


if __name__ == "__main__":
    app()
