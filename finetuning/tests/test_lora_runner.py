"""Tests for LoRA fine-tuning runner.

These tests use mock models to avoid requiring the full OpenFold3
model and checkpoints.
"""

import torch
import torch.nn as nn

from finetuning.lora.applicator import LoRAApplicator
from finetuning.lora.config import LoRAConfig
from finetuning.runner.lora_ema import LoRAExponentialMovingAverage


class TestLoRAExponentialMovingAverage:
    """Tests for LoRA-specific EMA."""

    def _build_model_with_lora(self):
        """Build a simple model with LoRA for testing."""

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear_q = nn.Linear(32, 32)
                self.linear_k = nn.Linear(32, 32)
                self.output = nn.Linear(32, 16)

            def forward(self, x):
                return self.output(self.linear_q(x) + self.linear_k(x))

        model = SimpleModel()
        config = LoRAConfig(
            rank=4,
            alpha=8.0,
            target_modules=["linear_q"],
            target_blocks=[""],
        )
        applicator = LoRAApplicator(config)
        applicator.apply(model)
        applicator.freeze_base_parameters(model)
        return model

    def test_ema_tracks_only_lora_params(self):
        """EMA should only track LoRA parameters."""
        model = self._build_model_with_lora()
        ema = LoRAExponentialMovingAverage(model, decay=0.999)

        for name in ema.shadow:
            assert "lora_A" in name or "lora_B" in name

    def test_ema_update(self):
        """EMA update should modify shadow parameters."""
        model = self._build_model_with_lora()
        ema = LoRAExponentialMovingAverage(model, decay=0.5)

        # Store initial shadow
        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}

        # Modify LoRA params
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data.add_(1.0)

        ema.update(model)

        # Shadow should have changed
        for name in ema.shadow:
            assert not torch.equal(ema.shadow[name], initial_shadow[name])

    def test_apply_and_restore(self):
        """apply_shadow/restore should swap and restore params."""
        model = self._build_model_with_lora()
        ema = LoRAExponentialMovingAverage(model, decay=0.5)

        # Modify LoRA params
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.data.add_(1.0)

        # Save current params
        original_params = {}
        for name, param in model.named_parameters():
            if "lora_" in name:
                original_params[name] = param.data.clone()

        ema.update(model)
        ema.apply_shadow(model)

        # Params should now be the EMA values (different from original)
        for name, param in model.named_parameters():
            if name in original_params:
                assert not torch.equal(param.data, original_params[name])

        ema.restore(model)

        # Params should be restored
        for name, param in model.named_parameters():
            if name in original_params:
                torch.testing.assert_close(param.data, original_params[name])

    def test_state_dict_roundtrip(self):
        """State dict save/load should preserve EMA state."""
        model = self._build_model_with_lora()
        ema = LoRAExponentialMovingAverage(model, decay=0.99)

        state = ema.state_dict()
        ema2 = LoRAExponentialMovingAverage(model, decay=0.5)
        ema2.load_state_dict(state)

        assert ema2.decay == 0.99
        for name in ema.shadow:
            torch.testing.assert_close(ema.shadow[name], ema2.shadow[name])

    def test_to_device(self):
        """EMA should move shadow params to specified device."""
        model = self._build_model_with_lora()
        ema = LoRAExponentialMovingAverage(model, decay=0.999)

        ema.to(torch.device("cpu"))
        for v in ema.shadow.values():
            assert v.device == torch.device("cpu")
