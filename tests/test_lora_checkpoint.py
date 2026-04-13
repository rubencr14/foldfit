"""Tests for LoRA checkpoint management."""

import tempfile
from pathlib import Path

import torch
import torch.nn as nn

from finetuning.lora.applicator import LoRAApplicator
from finetuning.lora.checkpoint import LoRACheckpointManager
from finetuning.lora.config import LoRAConfig
from finetuning.lora.layers import LoRALinear


def _build_simple_model() -> nn.Module:
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_q = nn.Linear(32, 32)
            self.linear_k = nn.Linear(32, 32)
            self.output = nn.Linear(32, 16)

        def forward(self, x):
            return self.output(self.linear_q(x) + self.linear_k(x))

    return SimpleModel()


def _apply_lora(model):
    config = LoRAConfig(
        rank=4,
        alpha=8.0,
        target_modules=["linear_q", "linear_k"],
        target_blocks=[""],  # Match everything
    )
    applicator = LoRAApplicator(config)
    applicator.apply(model)
    return config


class TestLoRACheckpointManager:
    """Tests for LoRA checkpoint manager."""

    def test_extract_lora_state_dict(self):
        """Should extract only LoRA parameters."""
        model = _build_simple_model()
        _apply_lora(model)

        state = LoRACheckpointManager.extract_lora_state_dict(model)
        assert len(state) == 4  # 2 layers * (lora_A + lora_B)
        for name in state:
            assert "lora_A" in name or "lora_B" in name

    def test_save_and_load_roundtrip(self):
        """Saving and loading should produce identical LoRA weights."""
        # Build two models with identical base weights
        model = _build_simple_model()
        model2 = _build_simple_model()
        model2.load_state_dict(model.state_dict())

        config = _apply_lora(model)

        # Set non-zero LoRA weights
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data.normal_()

        x = torch.randn(2, 32)
        with torch.no_grad():
            original_output = model(x)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "lora.pt"
            LoRACheckpointManager.save_lora_weights(model, path, config)

            # Apply LoRA to the second model and load saved weights
            _apply_lora(model2)
            loaded_config = LoRACheckpointManager.load_lora_weights(model2, path)

            with torch.no_grad():
                loaded_output = model2(x)

        torch.testing.assert_close(original_output, loaded_output, atol=1e-5, rtol=1e-5)
        assert loaded_config is not None
        assert loaded_config.rank == config.rank

    def test_save_creates_parent_directories(self):
        """Save should create parent directories if they don't exist."""
        model = _build_simple_model()
        _apply_lora(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nested" / "dir" / "lora.pt"
            LoRACheckpointManager.save_lora_weights(model, path)
            assert path.exists()

    def test_merge_lora_into_model(self):
        """After merging, LoRALinear modules should be replaced with Linear."""
        model = _build_simple_model()
        _apply_lora(model)

        # Set non-zero weights
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data.normal_()

        x = torch.randn(2, 32)
        with torch.no_grad():
            pre_merge_output = model(x)

        LoRACheckpointManager.merge_lora_into_model(model)

        # Check no LoRALinear modules remain
        for _, module in model.named_modules():
            assert not isinstance(module, LoRALinear)

        with torch.no_grad():
            post_merge_output = model(x)

        torch.testing.assert_close(
            pre_merge_output, post_merge_output, atol=1e-5, rtol=1e-5
        )

    def test_merge_and_save(self):
        """merge_and_save should produce a loadable state dict."""
        model = _build_simple_model()
        _apply_lora(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "merged.pt"
            LoRACheckpointManager.merge_and_save(model, path)
            assert path.exists()

            state_dict = torch.load(path, map_location="cpu", weights_only=True)
            # Should not contain any lora_ keys
            for key in state_dict:
                assert "lora_" not in key
