"""Tests for the training loop."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from foldfit.domain.value_objects import LoraConfig, TrainingConfig
from foldfit.infrastructure.peft.injector import LoraInjector
from foldfit.infrastructure.training.trainer import Trainer


def _make_dummy_setup(
    n_samples: int = 20,
    dim: int = 8,
) -> tuple[nn.Module, LoraInjector, nn.Module, DataLoader]:  # type: ignore[type-arg]
    """Create a tiny model with LoRA, a loss fn, and a dataloader."""

    class TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear_q = nn.Linear(dim, dim)
            self.linear_v = nn.Linear(dim, dim)
            self.head = nn.Linear(dim, 1)

        def forward(self, batch: dict) -> dict:  # type: ignore[type-arg]
            x = batch["x"]
            h = self.linear_q(x) + self.linear_v(x)
            return {"preds": self.head(h)}

    class MSELoss(nn.Module):
        def forward(self, preds: dict, batch: dict) -> dict:  # type: ignore[type-arg]
            pred = preds["preds"].squeeze(-1)
            target = batch["y"]
            return {"loss": nn.functional.mse_loss(pred, target)}

    model = TinyModel()
    injector = LoraInjector()
    config = LoraConfig(rank=2, alpha=4.0, target_modules=["linear_q", "linear_v"])
    injector.apply(model, config)

    # Make head trainable
    for p in model.head.parameters():
        p.requires_grad = True

    x = torch.randn(n_samples, dim)
    y = torch.randn(n_samples)
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Wrap loader to return dicts
    class DictLoader:
        def __init__(self, loader: DataLoader) -> None:  # type: ignore[type-arg]
            self.loader = loader

        def __iter__(self):  # type: ignore[no-untyped-def]
            for x_batch, y_batch in self.loader:
                yield {"x": x_batch, "y": y_batch}

        def __len__(self) -> int:
            return len(self.loader)

    return model, injector, MSELoss(), DictLoader(loader)


class TestTrainer:
    def test_loss_decreases(self) -> None:
        model, injector, loss_fn, loader = _make_dummy_setup()
        config = TrainingConfig(
            epochs=10,
            learning_rate=1e-2,
            accumulation_steps=1,
            amp=False,
            scheduler="constant",
        )
        trainer = Trainer(config)
        history = trainer.fit(
            model=model,
            loss_fn=loss_fn,
            peft=injector,
            head=model.head,
            train_loader=loader,
        )

        assert len(history) == 10
        assert history[-1]["train_loss"] < history[0]["train_loss"]

    def test_early_stopping(self) -> None:
        model, injector, loss_fn, loader = _make_dummy_setup()

        # Freeze ALL params so loss never improves
        for p in model.parameters():
            p.requires_grad = False
        # Re-enable only lora params (but with tiny LR they won't change much)
        for p in injector.trainable_parameters():
            p.requires_grad = True

        config = TrainingConfig(
            epochs=100,
            learning_rate=1e-10,  # near-zero learning, val_loss won't improve
            early_stopping_patience=2,
            accumulation_steps=1,
            amp=False,
            scheduler="constant",
        )
        trainer = Trainer(config)
        history = trainer.fit(
            model=model,
            loss_fn=loss_fn,
            peft=injector,
            head=None,
            train_loader=loader,
            val_loader=loader,
        )

        # Should stop after patience+1 epochs (1 epoch improves from inf, then 2 no-improve)
        assert len(history) <= 10

    def test_with_validation(self) -> None:
        model, injector, loss_fn, loader = _make_dummy_setup()
        config = TrainingConfig(
            epochs=3, learning_rate=1e-2, accumulation_steps=1, amp=False
        )
        trainer = Trainer(config)
        history = trainer.fit(
            model=model,
            loss_fn=loss_fn,
            peft=injector,
            head=model.head,
            train_loader=loader,
            val_loader=loader,
        )

        assert all("val_loss" in h for h in history)

    def test_gradient_accumulation(self) -> None:
        model, injector, loss_fn, loader = _make_dummy_setup()
        config = TrainingConfig(
            epochs=3,
            learning_rate=1e-2,
            accumulation_steps=2,
            amp=False,
            scheduler="constant",
        )
        trainer = Trainer(config)
        history = trainer.fit(
            model=model,
            loss_fn=loss_fn,
            peft=injector,
            head=model.head,
            train_loader=loader,
        )

        assert len(history) == 3

    def test_custom_forward_fn(self) -> None:
        model, injector, loss_fn, loader = _make_dummy_setup()

        def custom_forward(m: nn.Module, batch: dict) -> dict:  # type: ignore[type-arg]
            preds = m(batch)
            return loss_fn(preds, batch)

        config = TrainingConfig(
            epochs=3, learning_rate=1e-2, accumulation_steps=1, amp=False
        )
        trainer = Trainer(config)
        history = trainer.fit(
            model=model,
            loss_fn=loss_fn,
            peft=injector,
            head=model.head,
            train_loader=loader,
            model_forward_fn=custom_forward,
        )

        assert len(history) == 3
