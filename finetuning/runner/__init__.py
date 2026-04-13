"""LoRA fine-tuning runner for OpenFold3."""

from finetuning.runner.lora_ema import LoRAExponentialMovingAverage

__all__ = ["LoRAExponentialMovingAverage"]

# Lazy import for LoRAFineTuningRunner to avoid requiring pytorch_lightning
# at import time when only using EMA or other utilities.


def __getattr__(name):
    if name == "LoRAFineTuningRunner":
        from finetuning.runner.lora_runner import LoRAFineTuningRunner

        return LoRAFineTuningRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
