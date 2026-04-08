"""Configuration loader: YAML file -> validated FoldfitConfig."""

from __future__ import annotations

from pathlib import Path

import yaml

from foldfit.domain.value_objects import FoldfitConfig


def load_config(path: str | Path = "config.yaml") -> FoldfitConfig:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated FoldfitConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If the config values are invalid.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    return FoldfitConfig(**raw)
