"""Tests for FinetuneService with mocked ports."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from foldfit.application.finetune_service import FinetuneService
from foldfit.domain.value_objects import (
    DataConfig,
    FoldfitConfig,
    LoraConfig,
    ModelConfig,
    OutputConfig,
    TrainingConfig,
)


def _make_mock_ports() -> tuple:
    """Create mock ports for testing."""
    model_port = MagicMock()
    peft_port = MagicMock()
    dataset_port = MagicMock()
    msa_port = MagicMock()
    checkpoint_port = MagicMock()

    # Setup model port
    dummy_module = nn.Linear(16, 16)
    model_port.get_model.return_value = dummy_module
    model_port.get_peft_target_module.return_value = dummy_module
    model_port.get_default_peft_targets.return_value = ["linear_q", "linear_v"]

    # Setup peft port
    param = nn.Parameter(torch.randn(4, 16))
    peft_port.trainable_parameters.return_value = [param]

    return model_port, peft_port, dataset_port, msa_port, checkpoint_port


class TestFinetuneService:
    def test_run_with_no_structures_fails(self) -> None:
        model, peft, dataset, msa, ckpt = _make_mock_ports()
        dataset.fetch_structures.return_value = []

        service = FinetuneService(model, peft, dataset, msa, ckpt)
        config = FoldfitConfig()

        job = service.run(config)
        assert job.status == "failed"
        assert "No structure files" in job.metrics.get("error", "")

    def test_run_calls_ports_in_order(self, tmp_path: Path) -> None:
        model, peft, dataset, msa, ckpt = _make_mock_ports()

        # Return fake PDB paths
        fake_pdbs = [tmp_path / f"{i}.pdb" for i in range(10)]
        for p in fake_pdbs:
            p.write_text("FAKE PDB")
        dataset.fetch_structures.return_value = fake_pdbs

        config = FoldfitConfig(
            model=ModelConfig(device="cpu"),
            training=TrainingConfig(epochs=1, accumulation_steps=1, amp=False),
            output=OutputConfig(checkpoint_dir=str(tmp_path / "ckpt")),
        )

        service = FinetuneService(model, peft, dataset, msa, ckpt)

        # The service will fail when trying to train because the mock model
        # can't actually forward. We just verify the ports get called.
        job = service.run(config)

        # Verify essential calls were made
        dataset.fetch_structures.assert_called_once()
        model.load.assert_called_once()
        model.freeze_trunk.assert_called_once()
        peft.apply.assert_called_once()
