"""Tests for domain value objects."""

import pytest
from pydantic import ValidationError

from foldfit.domain.value_objects import (
    DataConfig,
    FoldfitConfig,
    LoraConfig,
    ModelConfig,
    MsaConfig,
    OutputConfig,
    QLoraConfig,
    TrainingConfig,
)


class TestLoraConfig:
    def test_defaults(self) -> None:
        cfg = LoraConfig()
        assert cfg.rank == 8
        assert cfg.alpha == 16.0
        assert cfg.dropout == 0.0
        assert cfg.target_modules == ["linear_q", "linear_v"]

    def test_scaling_factor(self) -> None:
        cfg = LoraConfig(rank=4, alpha=8.0)
        assert cfg.alpha / cfg.rank == 2.0

    def test_rank_bounds(self) -> None:
        LoraConfig(rank=2)
        LoraConfig(rank=64)
        with pytest.raises(ValidationError):
            LoraConfig(rank=1)
        with pytest.raises(ValidationError):
            LoraConfig(rank=65)

    def test_alpha_positive(self) -> None:
        with pytest.raises(ValidationError):
            LoraConfig(alpha=0)
        with pytest.raises(ValidationError):
            LoraConfig(alpha=-1.0)

    def test_dropout_bounds(self) -> None:
        LoraConfig(dropout=0.0)
        LoraConfig(dropout=0.5)
        with pytest.raises(ValidationError):
            LoraConfig(dropout=-0.1)
        with pytest.raises(ValidationError):
            LoraConfig(dropout=0.6)

    def test_frozen(self) -> None:
        cfg = LoraConfig()
        with pytest.raises(ValidationError):
            cfg.rank = 16  # type: ignore[misc]


class TestQLoraConfig:
    def test_inherits_lora(self) -> None:
        cfg = QLoraConfig(rank=4, quantization_bits=4)
        assert cfg.rank == 4
        assert cfg.quantization_bits == 4
        assert cfg.double_quantization is True
        assert cfg.quantization_type == "nf4"

    def test_quantization_bits(self) -> None:
        QLoraConfig(quantization_bits=4)
        QLoraConfig(quantization_bits=8)
        with pytest.raises(ValidationError):
            QLoraConfig(quantization_bits=3)  # type: ignore[arg-type]


class TestTrainingConfig:
    def test_defaults(self) -> None:
        cfg = TrainingConfig()
        assert cfg.epochs == 20
        assert cfg.scheduler == "cosine"
        assert cfg.amp is True
        assert cfg.accumulation_steps == 4

    def test_epoch_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TrainingConfig(epochs=0)
        with pytest.raises(ValidationError):
            TrainingConfig(epochs=501)

    def test_scheduler_values(self) -> None:
        for s in ("cosine", "linear", "constant"):
            TrainingConfig(scheduler=s)
        with pytest.raises(ValidationError):
            TrainingConfig(scheduler="invalid")  # type: ignore[arg-type]


class TestDataConfig:
    def test_defaults(self) -> None:
        cfg = DataConfig()
        assert cfg.max_seq_len == 256
        assert cfg.resolution_max == 3.0

    def test_seq_len_bounds(self) -> None:
        with pytest.raises(ValidationError):
            DataConfig(max_seq_len=16)


class TestFoldfitConfig:
    def test_aggregation(self) -> None:
        cfg = FoldfitConfig()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.data, DataConfig)
        assert isinstance(cfg.training, TrainingConfig)
        assert isinstance(cfg.lora, LoraConfig)
        assert isinstance(cfg.msa, MsaConfig)
        assert isinstance(cfg.output, OutputConfig)

    def test_from_dict(self) -> None:
        cfg = FoldfitConfig(
            lora=LoraConfig(rank=16),
            training=TrainingConfig(epochs=10),
        )
        assert cfg.lora.rank == 16
        assert cfg.training.epochs == 10
