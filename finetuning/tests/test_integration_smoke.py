"""Integration smoke test: load real OpenFold3, apply LoRA, verify forward/backward.

This test requires:
- OpenFold3 pretrained weights at ~/.openfold3/of3-p2-155k.pt
- GPU with sufficient VRAM

Run with: PYTHONPATH="$PWD:$PWD/openfold-3" python finetuning/tests/test_integration_smoke.py
"""

import gc
import logging
import sys
import traceback
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("integration_smoke_test")

CHECKPOINT_PATH = Path.home() / ".openfold3" / "of3-p2-155k.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_load_checkpoint():
    """Test 1: Verify the checkpoint loads correctly."""
    logger.info("=" * 60)
    logger.info("TEST 1: Loading checkpoint")
    logger.info("=" * 60)

    assert CHECKPOINT_PATH.exists(), f"Checkpoint not found at {CHECKPOINT_PATH}"
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        fmt = "PyTorch Lightning"
    elif "module" in checkpoint:
        state_dict = checkpoint["module"]
        fmt = "DeepSpeed"
    else:
        state_dict = checkpoint
        fmt = "Raw state dict"

    logger.info(f"Checkpoint format: {fmt}")
    logger.info(f"Number of parameters in checkpoint: {len(state_dict)}")
    return state_dict


def test_build_and_load_model(state_dict):
    """Test 2: Build OpenFold3 model and load pretrained weights."""
    logger.info("=" * 60)
    logger.info("TEST 2: Building model and loading weights")
    logger.info("=" * 60)

    from openfold3.projects.of3_all_atom.config.model_config import model_config
    from openfold3.projects.of3_all_atom.model import OpenFold3

    model = OpenFold3(model_config)

    cleaned = {k.removeprefix("model."): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    logger.info(f"Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params:,}")

    return model, model_config


def test_apply_lora(model):
    """Test 3: Apply LoRA adapters and verify parameter counts."""
    logger.info("=" * 60)
    logger.info("TEST 3: Applying LoRA adapters")
    logger.info("=" * 60)

    from finetuning.lora.applicator import LoRAApplicator
    from finetuning.lora.config import LoRAConfig

    config = LoRAConfig(
        rank=4,
        alpha=8.0,
        dropout=0.0,
        target_modules=["linear_q", "linear_k", "linear_v", "linear_o"],
        target_blocks=["pairformer_stack"],
    )

    applicator = LoRAApplicator(config)
    adapted_count = applicator.apply(model)
    logger.info(f"Adapted {adapted_count} layers")

    applicator.freeze_base_parameters(model)
    counts = applicator.count_parameters(model)
    logger.info(f"Total params:     {counts['total']:,}")
    logger.info(f"Trainable params: {counts['trainable']:,}")
    logger.info(f"LoRA params:      {counts['lora']:,}")
    logger.info(f"LoRA ratio:       {counts['trainable'] / counts['total']:.4%}")

    assert adapted_count > 0, "No layers were adapted!"
    assert counts["lora"] > 0
    assert counts["trainable"] == counts["lora"]

    return config


def test_forward_pass(model, model_config):
    """Test 4: Run a forward pass with synthetic data using OpenFold3's test utils."""
    logger.info("=" * 60)
    logger.info("TEST 4: Forward pass with random features")
    logger.info("=" * 60)

    from openfold3.tests.data_utils import random_of3_features

    # Small batch to fit in memory
    n_token = 32
    n_msa = 4
    n_templ = 1
    batch_size = 1

    batch = random_of3_features(
        batch_size=batch_size,
        n_token=n_token,
        n_msa=n_msa,
        n_templ=n_templ,
    )

    # Add missing fields the model expects
    batch["pdb_id"] = ["SYNTHETIC_TEST"]
    batch["preferred_chain_or_interface"] = "A"
    batch["ref_space_uid_to_perm"] = None

    n_atom = batch["atom_mask"].shape[-1]

    # Add fields needed by permutation alignment / naive alignment
    batch["mol_entity_id"] = batch["entity_id"].clone()
    batch["mol_sym_id"] = torch.ones((batch_size, n_token), dtype=torch.int32)
    batch["mol_sym_component_id"] = torch.zeros((batch_size, n_token), dtype=torch.int32)
    batch["mol_sym_token_index"] = torch.arange(n_token).unsqueeze(0).expand(batch_size, -1).int()

    # Enrich ground_truth with fields for permutation alignment
    batch["ground_truth"]["token_mask"] = batch["token_mask"].clone()
    batch["ground_truth"]["token_index"] = batch["token_index"].clone()
    batch["ground_truth"]["atom_mask"] = batch["atom_mask"].clone()
    batch["ground_truth"]["num_atoms_per_token"] = batch["num_atoms_per_token"].clone()
    batch["ground_truth"]["mol_entity_id"] = batch["mol_entity_id"].clone()
    batch["ground_truth"]["mol_sym_id"] = batch["mol_sym_id"].clone()
    batch["ground_truth"]["mol_sym_component_id"] = batch["mol_sym_component_id"].clone()
    batch["ground_truth"]["mol_sym_token_index"] = batch["mol_sym_token_index"].clone()
    batch["ground_truth"]["is_ligand"] = batch["is_ligand"].clone()
    batch["ground_truth"]["start_atom_index"] = batch["start_atom_index"].clone()
    batch["ground_truth"]["atom_to_token_index"] = batch["atom_to_token_index"].clone()

    # Move to device
    def to_device(x):
        if isinstance(x, torch.Tensor):
            return x.to(DEVICE)
        return x

    batch = _map_nested(batch, to_device)

    model = model.to(DEVICE)
    model.train()

    logger.info(f"Running forward pass on {DEVICE} with {n_token} tokens...")

    with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda"), dtype=torch.bfloat16):
        batch, outputs = model(batch)

    logger.info("Forward pass SUCCEEDED")
    logger.info(f"Output keys: {list(outputs.keys())}")

    if "atom_positions_predicted" in outputs:
        pred_shape = outputs["atom_positions_predicted"].shape
        logger.info(f"Predicted positions shape: {pred_shape}")

    return batch, outputs


def test_loss_computation(model_config, batch, outputs):
    """Test 5: Compute loss and verify it's finite."""
    logger.info("=" * 60)
    logger.info("TEST 5: Loss computation")
    logger.info("=" * 60)

    from openfold3.core.loss.loss_module import OpenFold3Loss

    loss_fn = OpenFold3Loss(config=model_config.architecture.loss_module)

    with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda"), dtype=torch.bfloat16):
        loss, breakdown = loss_fn(batch, outputs, _return_breakdown=True)

    logger.info(f"Total loss: {loss.item():.6f}")
    for name, value in breakdown.items():
        if isinstance(value, torch.Tensor):
            logger.info(f"  {name}: {value.item():.6f}")
        else:
            logger.info(f"  {name}: {value}")

    assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
    logger.info("Loss is finite - PASSED")
    return loss


def test_backward_and_gradients(model, loss):
    """Test 6: Backward pass and gradient verification."""
    logger.info("=" * 60)
    logger.info("TEST 6: Backward pass and gradient verification")
    logger.info("=" * 60)

    loss.backward()

    lora_with_grad = 0
    lora_without_grad = 0
    nan_grads = []
    inf_grads = []
    zero_grads = []
    grad_norms = []

    for name, param in model.named_parameters():
        if "lora_" in name:
            if param.grad is not None:
                lora_with_grad += 1
                norm = param.grad.norm().item()
                grad_norms.append((name, norm))
                if torch.isnan(param.grad).any():
                    nan_grads.append(name)
                if torch.isinf(param.grad).any():
                    inf_grads.append(name)
                if norm == 0:
                    zero_grads.append(name)
            else:
                lora_without_grad += 1

    logger.info(f"LoRA params with gradients: {lora_with_grad}")
    logger.info(f"LoRA params WITHOUT gradients: {lora_without_grad}")
    logger.info(f"NaN gradients: {len(nan_grads)}")
    logger.info(f"Inf gradients: {len(inf_grads)}")
    logger.info(f"Zero gradients: {len(zero_grads)}")

    if nan_grads:
        logger.error(f"NaN gradients found in: {nan_grads[:5]}")
    if inf_grads:
        logger.error(f"Inf gradients found in: {inf_grads[:5]}")

    if grad_norms:
        norms = [n for _, n in grad_norms]
        logger.info(f"Gradient norm stats:")
        logger.info(f"  Min:    {min(norms):.6e}")
        logger.info(f"  Max:    {max(norms):.6e}")
        logger.info(f"  Mean:   {sum(norms)/len(norms):.6e}")

        # Top 5
        sorted_norms = sorted(grad_norms, key=lambda x: x[1], reverse=True)
        logger.info("Top 5 largest gradient norms:")
        for name, norm in sorted_norms[:5]:
            short_name = name.split(".")[-3] + "." + name.split(".")[-2] + "." + name.split(".")[-1]
            logger.info(f"  {short_name}: {norm:.6e}")

    # Verify no base model params have gradients
    base_with_grad = sum(
        1 for n, p in model.named_parameters()
        if "lora_" not in n and p.grad is not None
    )
    logger.info(f"Base model params with gradients (should be 0): {base_with_grad}")

    assert len(nan_grads) == 0, f"NaN gradients in {len(nan_grads)} params"
    assert len(inf_grads) == 0, f"Inf gradients in {len(inf_grads)} params"
    assert lora_with_grad > 0, "No LoRA params received gradients!"

    logger.info("Gradient verification PASSED")


def test_checkpoint_roundtrip(model, lora_config):
    """Test 7: Save and load LoRA checkpoint."""
    logger.info("=" * 60)
    logger.info("TEST 7: LoRA checkpoint save/load roundtrip")
    logger.info("=" * 60)

    import tempfile
    from finetuning.lora.checkpoint import LoRACheckpointManager

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "lora_test.pt"
        LoRACheckpointManager.save_lora_weights(model, path, lora_config)
        file_size_mb = path.stat().st_size / (1024 * 1024)
        full_size_mb = CHECKPOINT_PATH.stat().st_size / (1024 * 1024)
        ratio = file_size_mb / full_size_mb * 100

        logger.info(f"LoRA checkpoint: {file_size_mb:.2f} MB")
        logger.info(f"Full model:      {full_size_mb:.2f} MB")
        logger.info(f"Ratio:           {ratio:.2f}%")

        assert ratio < 10, f"LoRA checkpoint too large ({ratio:.1f}%)"
        logger.info("Checkpoint roundtrip PASSED")


def test_optimizer_step(model):
    """Test 8: Verify optimizer step changes only LoRA params."""
    logger.info("=" * 60)
    logger.info("TEST 8: Optimizer step verification")
    logger.info("=" * 60)

    from finetuning.lora.applicator import LoRAApplicator

    lora_params = list(LoRAApplicator.get_lora_parameters(model))
    optimizer = torch.optim.AdamW(lora_params, lr=1e-3)

    # Save pre-step values
    pre_step = {}
    for name, param in model.named_parameters():
        if "lora_" in name and param.requires_grad:
            pre_step[name] = param.data.clone()

    optimizer.step()

    # Verify LoRA params changed
    changed = 0
    for name, param in model.named_parameters():
        if name in pre_step:
            if not torch.equal(param.data, pre_step[name]):
                changed += 1

    logger.info(f"LoRA params changed after optimizer step: {changed}/{len(pre_step)}")
    assert changed > 0, "No LoRA params changed after optimizer step!"
    logger.info("Optimizer step PASSED")


def _map_nested(obj, fn):
    """Apply fn to all tensors in a nested dict/list structure."""
    if isinstance(obj, dict):
        return {k: _map_nested(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_map_nested(v, fn) for v in obj]
    return fn(obj)


def main():
    logger.info("Starting integration smoke tests")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Checkpoint: {CHECKPOINT_PATH}")
    logger.info(f"PyTorch: {torch.__version__}")
    if DEVICE == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    results = {}

    # Test 1
    try:
        state_dict = test_load_checkpoint()
        results["1_load_checkpoint"] = "PASSED"
    except Exception as e:
        results["1_load_checkpoint"] = f"FAILED: {e}"
        _print_results(results)
        return 1

    # Test 2
    try:
        model, model_config = test_build_and_load_model(state_dict)
        del state_dict
        gc.collect()
        results["2_build_model"] = "PASSED"
    except Exception as e:
        results["2_build_model"] = f"FAILED: {e}"
        _print_results(results)
        return 1

    # Test 3
    try:
        lora_config = test_apply_lora(model)
        results["3_apply_lora"] = "PASSED"
    except Exception as e:
        results["3_apply_lora"] = f"FAILED: {e}"
        _print_results(results)
        return 1

    # Test 4
    try:
        batch, outputs = test_forward_pass(model, model_config)
        results["4_forward_pass"] = "PASSED"
    except Exception as e:
        results["4_forward_pass"] = f"FAILED: {e}"
        traceback.print_exc()
        _print_results(results)
        return 1

    # Test 5
    try:
        loss = test_loss_computation(model_config, batch, outputs)
        results["5_loss_computation"] = "PASSED"
    except Exception as e:
        results["5_loss_computation"] = f"FAILED: {e}"
        traceback.print_exc()
        _print_results(results)
        return 1

    # Test 6
    try:
        test_backward_and_gradients(model, loss)
        results["6_backward_gradients"] = "PASSED"
    except Exception as e:
        results["6_backward_gradients"] = f"FAILED: {e}"
        traceback.print_exc()

    # Test 7
    try:
        test_checkpoint_roundtrip(model, lora_config)
        results["7_checkpoint_roundtrip"] = "PASSED"
    except Exception as e:
        results["7_checkpoint_roundtrip"] = f"FAILED: {e}"

    # Test 8
    try:
        test_optimizer_step(model)
        results["8_optimizer_step"] = "PASSED"
    except Exception as e:
        results["8_optimizer_step"] = f"FAILED: {e}"

    _print_results(results)
    failed = sum(1 for v in results.values() if "FAILED" in v)
    return 1 if failed else 0


def _print_results(results):
    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 60)
    for test_name, result in results.items():
        status = "PASS" if "PASSED" in result else "FAIL"
        logger.info(f"  [{status}] {test_name}: {result}")
    passed = sum(1 for v in results.values() if "PASSED" in v)
    logger.info(f"\n  {passed}/{len(results)} tests passed")


if __name__ == "__main__":
    sys.exit(main())
