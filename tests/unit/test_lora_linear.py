"""Tests for LoRALinear module."""

import torch
import torch.nn as nn

from foldfit.infrastructure.peft.lora_linear import LoRALinear


class TestLoRALinear:
    def _make_lora(
        self, in_f: int = 8, out_f: int = 4, rank: int = 2, alpha: float = 4.0
    ) -> LoRALinear:
        original = nn.Linear(in_f, out_f)
        return LoRALinear(original, rank=rank, alpha=alpha, dropout=0.0)

    def test_output_shape(self) -> None:
        lora = self._make_lora()
        x = torch.randn(2, 8)
        y = lora(x)
        assert y.shape == (2, 4)

    def test_initial_output_equals_base(self) -> None:
        """At init, lora_B is zero so LoRA contribution is zero."""
        original = nn.Linear(8, 4)
        lora = LoRALinear(original, rank=2, alpha=4.0, dropout=0.0)
        x = torch.randn(3, 8)
        base_out = original(x)
        lora_out = lora(x)
        torch.testing.assert_close(lora_out, base_out)

    def test_lora_contributes_after_training(self) -> None:
        """After modifying lora_B, output should differ from base."""
        lora = self._make_lora()
        nn.init.normal_(lora.lora_B)
        x = torch.randn(2, 8)

        base_out = torch.nn.functional.linear(x, lora.weight, lora.bias)
        lora_out = lora(x)
        assert not torch.allclose(base_out, lora_out)

    def test_merge_unmerge_roundtrip(self) -> None:
        """Merged output should equal unmerged output numerically."""
        lora = self._make_lora()
        nn.init.normal_(lora.lora_B)
        x = torch.randn(5, 8)

        # Output before merge
        unmerged_out = lora(x)

        # Merge and get output
        lora.merge()
        merged_out = lora(x)
        torch.testing.assert_close(merged_out, unmerged_out, atol=1e-5, rtol=1e-5)

        # Unmerge and verify we're back
        lora.unmerge()
        restored_out = lora(x)
        torch.testing.assert_close(restored_out, unmerged_out, atol=1e-5, rtol=1e-5)

    def test_only_lora_params_trainable(self) -> None:
        lora = self._make_lora()
        trainable = [n for n, p in lora.named_parameters() if p.requires_grad]
        assert set(trainable) == {"lora_A", "lora_B"}

    def test_base_weight_frozen(self) -> None:
        lora = self._make_lora()
        assert not lora.weight.requires_grad
        if lora.bias is not None:
            assert not lora.bias.requires_grad

    def test_scaling_factor(self) -> None:
        lora = LoRALinear(nn.Linear(8, 4), rank=4, alpha=8.0)
        assert lora.scaling == 2.0

    def test_merge_idempotent(self) -> None:
        lora = self._make_lora()
        nn.init.normal_(lora.lora_B)
        weight_before = lora.weight.data.clone()

        lora.merge()
        weight_after_first = lora.weight.data.clone()

        lora.merge()  # Should be no-op
        weight_after_second = lora.weight.data.clone()

        torch.testing.assert_close(weight_after_first, weight_after_second)

    def test_with_bias(self) -> None:
        original = nn.Linear(8, 4, bias=True)
        lora = LoRALinear(original, rank=2)
        assert lora.bias is not None
        x = torch.randn(2, 8)
        y = lora(x)
        assert y.shape == (2, 4)

    def test_without_bias(self) -> None:
        original = nn.Linear(8, 4, bias=False)
        lora = LoRALinear(original, rank=2)
        assert lora.bias is None
        x = torch.randn(2, 8)
        y = lora(x)
        assert y.shape == (2, 4)

    def test_properties(self) -> None:
        lora = self._make_lora(in_f=16, out_f=8)
        assert lora.in_features == 16
        assert lora.out_features == 8
