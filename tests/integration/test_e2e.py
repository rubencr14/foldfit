"""End-to-end test: download → MSA → featurize → train → evaluate → predict.

Runs the full foldfit pipeline on a small number of real antibody structures.
Requires: OpenFold installed, network access for RCSB downloads.
GPU optional (falls back to CPU, slower).

Usage:
    pytest tests/integration/test_e2e.py -v -s --timeout=600
    pytest tests/integration/test_e2e.py -v -s -k test_e2e_full_pipeline
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import torch

logger = logging.getLogger(__name__)

# Skip entire module if openfold not installed
openfold = pytest.importorskip("openfold", reason="OpenFold required for e2e tests")


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_STRUCTURES = 5  # small for CI speed
MAX_SEQ_LEN = 128  # short for memory
EPOCHS = 2


@pytest.fixture(scope="module")
def work_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Shared temp directory for the whole e2e run."""
    return tmp_path_factory.mktemp("foldfit_e2e")


@pytest.fixture(scope="module")
def pdb_dir(work_dir: Path) -> Path:
    return work_dir / "pdbs"


@pytest.fixture(scope="module")
def msa_dir(work_dir: Path) -> Path:
    return work_dir / "msa"


@pytest.fixture(scope="module")
def checkpoint_dir(work_dir: Path) -> Path:
    return work_dir / "checkpoints"


# ── Step 1: Download ─────────────────────────────────────────────────────

class TestE2EDownload:
    """Download a few antibody structures from RCSB."""

    @pytest.fixture(autouse=True, scope="class")
    def _download(self, pdb_dir: Path) -> None:
        from foldfit.infrastructure.data.sabdab_repository import SabdabRepository

        pdb_dir.mkdir(parents=True, exist_ok=True)
        repo = SabdabRepository(cache_dir=pdb_dir)

        pdb_ids = repo.query_antibody_pdb_ids(
            resolution_max=2.5,
            max_results=N_STRUCTURES,
        )
        assert len(pdb_ids) > 0, "RCSB query returned no results"

        for pdb_id in pdb_ids[:N_STRUCTURES]:
            dest = pdb_dir / f"{pdb_id}.pdb"
            if not dest.exists():
                repo.download_pdb(pdb_id, dest)

    def test_pdbs_downloaded(self, pdb_dir: Path) -> None:
        pdbs = list(pdb_dir.glob("*.pdb"))
        assert len(pdbs) >= 1, f"Expected PDB files in {pdb_dir}"
        # Verify files are not empty
        for p in pdbs:
            assert p.stat().st_size > 100, f"{p.name} looks empty"


# ── Step 2: MSA Generation ───────────────────────────────────────────────

class TestE2EMSA:
    """Generate MSAs using the single backend (fast, no external deps)."""

    @pytest.fixture(autouse=True, scope="class")
    def _generate_msa(self, pdb_dir: Path, msa_dir: Path) -> None:
        from openfold.np import protein as protein_utils
        from openfold.np import residue_constants as rc

        from foldfit.domain.value_objects import MsaConfig
        from foldfit.infrastructure.data.msa_provider import MsaProvider

        msa_dir.mkdir(parents=True, exist_ok=True)
        config = MsaConfig(backend="single", msa_dir=str(msa_dir))
        provider = MsaProvider(config)

        for pdb_path in sorted(pdb_dir.glob("*.pdb"))[:N_STRUCTURES]:
            pdb_id = pdb_path.stem.upper()
            cache_path = msa_dir / f"{pdb_id}.msa.pt"
            if cache_path.exists():
                continue

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

            if seq:
                result = provider.get(sequence=seq, pdb_id=pdb_id)
                torch.save(result, cache_path)

    def test_msa_files_created(self, msa_dir: Path) -> None:
        msa_files = list(msa_dir.glob("*.msa.pt"))
        assert len(msa_files) >= 1, f"Expected .msa.pt files in {msa_dir}"

    def test_msa_tensors_valid(self, msa_dir: Path) -> None:
        msa_file = next(msa_dir.glob("*.msa.pt"))
        data = torch.load(msa_file, weights_only=True)
        assert "msa" in data
        assert "deletion_matrix" in data
        assert data["msa"].dim() == 2  # [N, L]
        assert data["msa"].shape[0] >= 1


# ── Step 3: Featurize ────────────────────────────────────────────────────

class TestE2EFeaturize:
    """Test that PDB files can be featurized into OpenFold-compatible dicts."""

    def test_featurize_pdb(self, pdb_dir: Path) -> None:
        from foldfit.infrastructure.openfold.featurizer import OpenFoldFeaturizer

        featurizer = OpenFoldFeaturizer(max_seq_len=MAX_SEQ_LEN)
        pdb_path = next(pdb_dir.glob("*.pdb"))

        features = featurizer.from_pdb(pdb_path)
        assert features, f"Featurization failed for {pdb_path.name}"

        # Check essential keys
        for key in ["aatype", "residue_index", "all_atom_positions", "all_atom_mask",
                     "msa_feat", "target_feat", "seq_length", "seq_mask"]:
            assert key in features, f"Missing key: {key}"

        # Check shapes make sense
        L = features["aatype"].shape[0]
        assert L > 0
        assert features["all_atom_positions"].shape[0] == L  # before recycling dim

    def test_featurize_with_msa_data(self, pdb_dir: Path, msa_dir: Path) -> None:
        from foldfit.domain.value_objects import MsaConfig
        from foldfit.infrastructure.data.msa_provider import MsaProvider
        from foldfit.infrastructure.openfold.featurizer import OpenFoldFeaturizer

        featurizer = OpenFoldFeaturizer(max_seq_len=MAX_SEQ_LEN)
        config = MsaConfig(backend="precomputed", msa_dir=str(msa_dir))
        provider = MsaProvider(config)

        pdb_path = next(pdb_dir.glob("*.pdb"))
        seq, _, _ = featurizer._parse_pdb(pdb_path, "A")

        if seq:
            pdb_id = pdb_path.stem.upper()
            msa_data = provider.get(sequence=seq, pdb_id=pdb_id)
            features = featurizer.from_pdb(pdb_path, msa_data=msa_data)
            assert features
            assert "msa_feat" in features


# ── Step 4: Training ─────────────────────────────────────────────────────

class TestE2ETrain:
    """Run a short fine-tuning job on downloaded structures."""

    @pytest.fixture(autouse=True, scope="class")
    def _train(self, pdb_dir: Path, msa_dir: Path, checkpoint_dir: Path) -> None:
        from foldfit.application.finetune_service import FinetuneService
        from foldfit.domain.value_objects import (
            DataConfig,
            FoldfitConfig,
            LoraConfig,
            ModelConfig,
            MsaConfig,
            OutputConfig,
            TrainingConfig,
        )
        from foldfit.infrastructure.checkpoint_store import FileCheckpointStore
        from foldfit.infrastructure.data.msa_provider import MsaProvider
        from foldfit.infrastructure.data.sabdab_repository import SabdabRepository
        from foldfit.infrastructure.openfold.adapter import OpenFoldAdapter
        from foldfit.infrastructure.peft.injector import LoraInjector

        pdb_paths = [str(p) for p in sorted(pdb_dir.glob("*.pdb"))[:N_STRUCTURES]]

        config = FoldfitConfig(
            model=ModelConfig(weights_path=None, device=DEVICE),
            data=DataConfig(
                pdb_paths=pdb_paths,
                max_seq_len=MAX_SEQ_LEN,
                val_frac=0.2,
                test_frac=0.0,
            ),
            training=TrainingConfig(
                epochs=EPOCHS,
                learning_rate=1e-4,
                batch_size=1,
                accumulation_steps=1,
                amp=(DEVICE == "cuda"),
                gradient_checkpointing=False,
                scheduler="constant",
                early_stopping_patience=0,
            ),
            lora=LoraConfig(rank=4, alpha=8.0, target_modules=["linear_q", "linear_v"]),
            msa=MsaConfig(backend="precomputed", msa_dir=str(msa_dir)),
            output=OutputConfig(checkpoint_dir=str(checkpoint_dir)),
        )

        service = FinetuneService(
            model=OpenFoldAdapter(),
            peft=LoraInjector(),
            dataset=SabdabRepository(),
            msa=MsaProvider(config.msa),
            checkpoint=FileCheckpointStore(),
        )

        self.job = service.run(config)

    def test_training_completed(self) -> None:
        assert self.job.status == "completed", f"Training failed: {self.job.metrics}"

    def test_loss_is_finite(self) -> None:
        loss = self.job.metrics.get("train_loss")
        assert loss is not None
        assert not torch.tensor(loss).isnan()
        assert not torch.tensor(loss).isinf()

    def test_checkpoint_saved(self, checkpoint_dir: Path) -> None:
        final_dir = checkpoint_dir / "final"
        assert final_dir.exists(), f"No final checkpoint in {checkpoint_dir}"
        peft_dir = final_dir / "peft"
        assert peft_dir.exists(), "No peft adapter saved"


# ── Step 5: Evaluate ─────────────────────────────────────────────────────

class TestE2EEvaluate:
    """Evaluate the fine-tuned model on the same structures."""

    def test_evaluate_produces_metrics(
        self, pdb_dir: Path, checkpoint_dir: Path
    ) -> None:
        from foldfit.application.inference_service import InferenceService
        from foldfit.domain.value_objects import ModelConfig
        from foldfit.infrastructure.checkpoint_store import FileCheckpointStore
        from foldfit.infrastructure.openfold.adapter import OpenFoldAdapter
        from foldfit.infrastructure.openfold.featurizer import OpenFoldFeaturizer
        from foldfit.infrastructure.openfold.metrics import ca_rmsd, gdt_ts
        from foldfit.infrastructure.peft.injector import LoraInjector

        adapter_path = str(checkpoint_dir / "final")
        model_config = ModelConfig(weights_path=None, device=DEVICE)

        service = InferenceService(
            model=OpenFoldAdapter(),
            peft=LoraInjector(),
            checkpoint=FileCheckpointStore(),
        )
        service.load(model_config=model_config, adapter_path=adapter_path)

        featurizer = OpenFoldFeaturizer(max_seq_len=MAX_SEQ_LEN)
        pdb_path = next(pdb_dir.glob("*.pdb"))
        features = featurizer.from_pdb(pdb_path)
        assert features

        batch = {
            k: v.unsqueeze(0).to(DEVICE) if isinstance(v, torch.Tensor) else v
            for k, v in features.items()
        }

        output = service.predict(batch)
        raw = output.extra.get("_raw_outputs", {})

        pred_pos = raw.get("final_atom_positions")
        assert pred_pos is not None, "Model did not produce atom positions"

        # Strip recycling dim
        clean = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[-1] == 1:
                clean[k] = v.squeeze(-1)
            else:
                clean[k] = v

        gt_pos = clean["all_atom_positions"]
        gt_mask = clean["all_atom_mask"]

        rmsd = ca_rmsd(pred_pos, gt_pos, gt_mask)
        gdt = gdt_ts(pred_pos, gt_pos, gt_mask)

        assert rmsd.item() >= 0, "RMSD should be non-negative"
        assert 0 <= gdt.item() <= 1, "GDT-TS should be in [0, 1]"
        logger.info(f"Eval: RMSD={rmsd.item():.2f}A, GDT-TS={gdt.item():.3f}")


# ── Step 6: Predict ──────────────────────────────────────────────────────

class TestE2EPredict:
    """Predict structure from a raw sequence using the fine-tuned model."""

    def test_predict_from_sequence(self, checkpoint_dir: Path) -> None:
        from foldfit.application.inference_service import InferenceService
        from foldfit.domain.value_objects import ModelConfig
        from foldfit.infrastructure.checkpoint_store import FileCheckpointStore
        from foldfit.infrastructure.openfold.adapter import OpenFoldAdapter
        from foldfit.infrastructure.peft.injector import LoraInjector

        adapter_path = str(checkpoint_dir / "final")
        model_config = ModelConfig(weights_path=None, device=DEVICE)

        service = InferenceService(
            model=OpenFoldAdapter(),
            peft=LoraInjector(),
            checkpoint=FileCheckpointStore(),
        )
        service.load(model_config=model_config, adapter_path=adapter_path)

        # Short antibody VH sequence
        sequence = "EVQLVESGGGLVQPGGSLRLSCAASGFTFS"
        result = service.predict_from_sequence(
            sequence, device=DEVICE, max_seq_len=MAX_SEQ_LEN
        )

        assert result["sequence_length"] == len(sequence)
        assert result["pdb_string"] is not None, "No PDB string produced"
        assert "ATOM" in result["pdb_string"], "PDB string has no ATOM records"

        if result["mean_plddt"] is not None:
            assert 0 <= result["mean_plddt"] <= 100
            logger.info(f"Predict: pLDDT={result['mean_plddt']:.1f}")

    def test_predict_output_pdb_file(self, checkpoint_dir: Path, work_dir: Path) -> None:
        from foldfit.application.inference_service import InferenceService
        from foldfit.domain.value_objects import ModelConfig
        from foldfit.infrastructure.checkpoint_store import FileCheckpointStore
        from foldfit.infrastructure.openfold.adapter import OpenFoldAdapter
        from foldfit.infrastructure.peft.injector import LoraInjector

        adapter_path = str(checkpoint_dir / "final")
        model_config = ModelConfig(weights_path=None, device=DEVICE)

        service = InferenceService(
            model=OpenFoldAdapter(),
            peft=LoraInjector(),
            checkpoint=FileCheckpointStore(),
        )
        service.load(model_config=model_config, adapter_path=adapter_path)

        sequence = "DIQMTQSPSSLSASVGDRVTITC"
        result = service.predict_from_sequence(
            sequence, device=DEVICE, max_seq_len=MAX_SEQ_LEN
        )

        out_pdb = work_dir / "prediction.pdb"
        out_pdb.write_text(result["pdb_string"])
        assert out_pdb.exists()
        assert out_pdb.stat().st_size > 100
