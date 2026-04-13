"""Tests for LoRA applicator (model instrumentation)."""

import torch
import torch.nn as nn

from finetuning.lora.applicator import LoRAApplicator
from finetuning.lora.config import LoRAConfig
from finetuning.lora.layers import LoRALinear


def _build_mock_model() -> nn.Module:
    """Build a simplified model that mirrors OpenFold3's structure."""

    class MockAttention(nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.linear_q = nn.Linear(dim, dim)
            self.linear_k = nn.Linear(dim, dim)
            self.linear_v = nn.Linear(dim, dim)
            self.linear_o = nn.Linear(dim, dim)

        def forward(self, x):
            q = self.linear_q(x)
            k = self.linear_k(x)
            v = self.linear_v(x)
            attn = q * k  # simplified attention
            return self.linear_o(attn * v)

    class MockBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = MockAttention()
            self.transition = nn.Linear(64, 64)

        def forward(self, x):
            return self.transition(self.attn(x))

    class MockPairformerStack(nn.Module):
        def __init__(self, n_blocks=3):
            super().__init__()
            self.blocks = nn.ModuleList([MockBlock() for _ in range(n_blocks)])

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return x

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_embedder = nn.Linear(32, 64)
            self.pairformer_stack = MockPairformerStack(n_blocks=3)
            self.diffusion_module = MockBlock()
            self.output_head = nn.Linear(64, 32)

        def forward(self, x):
            x = self.input_embedder(x)
            x = self.pairformer_stack(x)
            x = self.diffusion_module(x)
            return self.output_head(x)

    return MockModel()


class TestLoRAApplicator:
    """Tests for LoRA applicator."""

    def test_apply_to_target_modules(self):
        """LoRA should be applied only to matching modules."""
        model = _build_mock_model()
        config = LoRAConfig(
            rank=4,
            alpha=8.0,
            target_modules=["linear_q", "linear_v"],
            target_blocks=["pairformer_stack"],
        )
        applicator = LoRAApplicator(config)
        count = applicator.apply(model)

        # 3 blocks * 2 target modules = 6
        assert count == 6

    def test_apply_to_multiple_blocks(self):
        """LoRA should be applied across multiple target blocks."""
        model = _build_mock_model()
        config = LoRAConfig(
            rank=4,
            alpha=8.0,
            target_modules=["linear_q"],
            target_blocks=["pairformer_stack", "diffusion_module"],
        )
        applicator = LoRAApplicator(config)
        count = applicator.apply(model)

        # 3 pairformer blocks + 1 diffusion block = 4
        assert count == 4

    def test_non_target_modules_unchanged(self):
        """Modules outside target blocks should not be wrapped."""
        model = _build_mock_model()
        config = LoRAConfig(
            rank=4,
            alpha=8.0,
            target_modules=["linear_q"],
            target_blocks=["pairformer_stack"],
        )
        applicator = LoRAApplicator(config)
        applicator.apply(model)

        # Input embedder and output head should remain plain Linear
        assert isinstance(model.input_embedder, nn.Linear)
        assert not isinstance(model.input_embedder, LoRALinear)
        assert isinstance(model.output_head, nn.Linear)

    def test_adapted_modules_are_lora_linear(self):
        """Adapted modules should be LoRALinear instances."""
        model = _build_mock_model()
        config = LoRAConfig(
            rank=4,
            alpha=8.0,
            target_modules=["linear_q"],
            target_blocks=["pairformer_stack"],
        )
        applicator = LoRAApplicator(config)
        applicator.apply(model)

        for block in model.pairformer_stack.blocks:
            assert isinstance(block.attn.linear_q, LoRALinear)
            assert isinstance(block.attn.linear_k, nn.Linear)
            assert not isinstance(block.attn.linear_k, LoRALinear)

    def test_freeze_base_parameters(self):
        """After freezing, only LoRA params should require gradients."""
        model = _build_mock_model()
        config = LoRAConfig(
            rank=4,
            alpha=8.0,
            target_modules=["linear_q", "linear_v"],
            target_blocks=["pairformer_stack"],
        )
        applicator = LoRAApplicator(config)
        applicator.apply(model)
        applicator.freeze_base_parameters(model)

        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                assert param.requires_grad, f"{name} should require grad"
            else:
                assert not param.requires_grad, f"{name} should be frozen"

    def test_get_lora_parameters(self):
        """Should yield only LoRA parameters."""
        model = _build_mock_model()
        config = LoRAConfig(
            rank=4,
            alpha=8.0,
            target_modules=["linear_q"],
            target_blocks=["pairformer_stack"],
        )
        applicator = LoRAApplicator(config)
        count = applicator.apply(model)

        lora_params = list(LoRAApplicator.get_lora_parameters(model))
        # Each adapted layer has lora_A and lora_B = 2 params
        assert len(lora_params) == count * 2

    def test_count_parameters(self):
        """Parameter counts should be accurate."""
        model = _build_mock_model()
        config = LoRAConfig(
            rank=4,
            alpha=8.0,
            target_modules=["linear_q"],
            target_blocks=["pairformer_stack"],
        )
        applicator = LoRAApplicator(config)
        applicator.apply(model)
        applicator.freeze_base_parameters(model)

        counts = applicator.count_parameters(model)
        assert counts["total"] > 0
        assert counts["lora"] > 0
        assert counts["trainable"] == counts["lora"]
        assert counts["lora"] < counts["total"]

    def test_forward_after_apply(self):
        """Model should produce valid output after LoRA is applied."""
        model = _build_mock_model()
        config = LoRAConfig(
            rank=4,
            alpha=8.0,
            target_modules=["linear_q", "linear_k", "linear_v", "linear_o"],
            target_blocks=["pairformer_stack"],
        )
        applicator = LoRAApplicator(config)
        applicator.apply(model)

        x = torch.randn(2, 32)
        out = model(x)
        assert out.shape == (2, 32)

    def test_backward_after_apply(self):
        """Backward pass should work after LoRA is applied."""
        model = _build_mock_model()
        config = LoRAConfig(
            rank=4,
            alpha=8.0,
            target_modules=["linear_q"],
            target_blocks=["pairformer_stack"],
        )
        applicator = LoRAApplicator(config)
        applicator.apply(model)
        applicator.freeze_base_parameters(model)

        x = torch.randn(2, 32)
        out = model(x)
        loss = out.sum()
        loss.backward()

        # LoRA params should have gradients
        for name, param in model.named_parameters():
            if "lora_" in name:
                assert param.grad is not None, f"{name} should have gradient"
