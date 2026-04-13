"""Tests for LoRA linear layer wrapper."""

import torch
import torch.nn as nn

from finetuning.lora.layers import LoRALinear


class TestLoRALinear:
    """Tests for the LoRALinear module."""

    def test_zero_init_preserves_output(self):
        """LoRA output should be zero at initialization (lora_B is zeros)."""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=4, alpha=8.0)

        x = torch.randn(2, 64)
        with torch.no_grad():
            base_out = original(x)
            lora_out = lora(x)

        torch.testing.assert_close(base_out, lora_out)

    def test_original_frozen(self):
        """Original layer parameters should not require gradients."""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=4, alpha=8.0)

        for param in lora.original.parameters():
            assert not param.requires_grad

    def test_lora_params_require_grad(self):
        """LoRA parameters should require gradients."""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=4, alpha=8.0)

        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_lora_shapes(self):
        """LoRA matrices should have correct shapes."""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=4, alpha=8.0)

        assert lora.lora_A.shape == (64, 4)
        assert lora.lora_B.shape == (4, 32)

    def test_forward_output_shape(self):
        """Forward pass should produce the same output shape as original."""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=4, alpha=8.0)

        x = torch.randn(2, 10, 64)
        out = lora(x)
        assert out.shape == (2, 10, 32)

    def test_gradient_flows_through_lora(self):
        """Gradients should flow through LoRA parameters."""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=4, alpha=8.0)

        x = torch.randn(2, 64)
        out = lora(x)
        loss = out.sum()
        loss.backward()

        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        # Original weights should not have gradients
        assert lora.original.weight.grad is None

    def test_merge_produces_equivalent_output(self):
        """Merged layer should produce the same output as LoRA layer."""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=4, alpha=8.0)

        # Train for a bit to get non-zero LoRA weights
        lora.lora_A.data.normal_()
        lora.lora_B.data.normal_()

        x = torch.randn(2, 64)
        with torch.no_grad():
            lora_out = lora(x)
            merged = lora.merge()
            merged_out = merged(x)

        torch.testing.assert_close(lora_out, merged_out, atol=1e-5, rtol=1e-5)

    def test_scaling_factor(self):
        """Scaling should be alpha / rank."""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=4, alpha=16.0)
        assert lora.scaling == 4.0

    def test_dropout(self):
        """LoRA with dropout should work in both train and eval modes."""
        original = nn.Linear(64, 32)
        lora = LoRALinear(original, rank=4, alpha=8.0, dropout=0.1)

        x = torch.randn(2, 64)
        lora.train()
        out_train = lora(x)
        lora.eval()
        out_eval = lora(x)

        assert out_train.shape == out_eval.shape

    def test_weight_and_bias_proxy(self):
        """Weight and bias properties should proxy to the original layer."""
        original = nn.Linear(64, 32, bias=True)
        lora = LoRALinear(original, rank=4, alpha=8.0)

        assert lora.weight is original.weight
        assert lora.bias is original.bias

    def test_no_bias_proxy(self):
        """Bias property should return None when original has no bias."""
        original = nn.Linear(64, 32, bias=False)
        lora = LoRALinear(original, rank=4, alpha=8.0)
        assert lora.bias is None
