"""Tests for LoRA injector."""

import tempfile

import torch
import torch.nn as nn

from foldfit.domain.value_objects import LoraConfig
from foldfit.infrastructure.peft.injector import LoraInjector
from foldfit.infrastructure.peft.lora_linear import LoRALinear


def _make_dummy_model() -> nn.Module:
    """Create a dummy model mimicking OpenFold attention structure."""

    class DummyAttention(nn.Module):
        def __init__(self, dim: int = 16) -> None:
            super().__init__()
            self.linear_q = nn.Linear(dim, dim)
            self.linear_k = nn.Linear(dim, dim)
            self.linear_v = nn.Linear(dim, dim)
            self.linear_o = nn.Linear(dim, dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear_o(self.linear_q(x) + self.linear_v(x))

    class DummyBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.msa_att = DummyAttention()
            self.ffn = nn.Linear(16, 16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.ffn(self.msa_att(x))

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([DummyBlock(), DummyBlock()])

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for block in self.blocks:
                x = block(x)
            return x

    return DummyModel()


class TestLoraInjector:
    def test_injection_replaces_target_layers(self) -> None:
        model = _make_dummy_model()
        injector = LoraInjector()
        config = LoraConfig(rank=4, target_modules=["linear_q", "linear_v"])

        injector.apply(model, config)

        # Should replace linear_q and linear_v in each of 2 blocks = 4 layers
        assert injector.replaced_count == 4

        # Verify replaced layers are LoRALinear
        for block in model.blocks:  # type: ignore[attr-defined]
            assert isinstance(block.msa_att.linear_q, LoRALinear)
            assert isinstance(block.msa_att.linear_v, LoRALinear)
            # linear_k and linear_o should NOT be replaced
            assert isinstance(block.msa_att.linear_k, nn.Linear)
            assert not isinstance(block.msa_att.linear_k, LoRALinear)

    def test_non_target_layers_untouched(self) -> None:
        model = _make_dummy_model()
        injector = LoraInjector()
        config = LoraConfig(rank=4, target_modules=["linear_q"])

        injector.apply(model, config)

        # Only linear_q should be replaced (2 blocks = 2 layers)
        assert injector.replaced_count == 2

    def test_all_base_params_frozen(self) -> None:
        model = _make_dummy_model()
        injector = LoraInjector()
        config = LoraConfig(rank=4)

        injector.apply(model, config)

        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert param.requires_grad, f"{name} should be trainable"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_trainable_parameters(self) -> None:
        model = _make_dummy_model()
        injector = LoraInjector()
        config = LoraConfig(rank=4, target_modules=["linear_q", "linear_v"])

        injector.apply(model, config)

        params = injector.trainable_parameters()
        # 4 layers * 2 params (A, B) = 8 parameters
        assert len(params) == 8
        assert all(p.requires_grad for p in params)

    def test_merge_unmerge(self) -> None:
        model = _make_dummy_model()
        injector = LoraInjector()
        config = LoraConfig(rank=4)

        injector.apply(model, config)

        # Set non-zero LoRA weights
        for p in injector.trainable_parameters():
            nn.init.normal_(p)

        x = torch.randn(2, 16)
        unmerged_out = model(x)

        injector.merge()
        merged_out = model(x)
        torch.testing.assert_close(merged_out, unmerged_out, atol=1e-4, rtol=1e-4)

        injector.unmerge()
        restored_out = model(x)
        torch.testing.assert_close(restored_out, unmerged_out, atol=1e-4, rtol=1e-4)

    def test_save_load_roundtrip(self) -> None:
        # Use seeded model for reproducibility
        torch.manual_seed(42)
        model = _make_dummy_model()
        injector = LoraInjector()
        config = LoraConfig(rank=4)

        injector.apply(model, config)

        # Set non-zero weights
        for p in injector.trainable_parameters():
            nn.init.normal_(p)

        # Save original LoRA params
        original_params = [p.data.clone() for p in injector.trainable_parameters()]

        with tempfile.TemporaryDirectory() as tmpdir:
            injector.save(tmpdir)

            # Create fresh model and injector, load weights
            torch.manual_seed(42)
            model2 = _make_dummy_model()
            injector2 = LoraInjector()
            injector2.apply(model2, config)
            injector2.load(tmpdir)

            loaded_params = [p.data for p in injector2.trainable_parameters()]
            for orig, loaded in zip(original_params, loaded_params, strict=True):
                torch.testing.assert_close(loaded, orig, atol=1e-6, rtol=1e-6)

    def test_forward_still_works_after_injection(self) -> None:
        model = _make_dummy_model()
        x = torch.randn(3, 16)

        # Output before injection
        before = model(x)

        # Inject LoRA (B=0 so output should be same)
        injector = LoraInjector()
        injector.apply(model, LoraConfig(rank=4))

        after = model(x)
        torch.testing.assert_close(after, before, atol=1e-5, rtol=1e-5)
