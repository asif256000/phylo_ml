from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from src.configuration.training import ConfigurationError, TrainingConfig

from .trainer import KANTrainer


def _load_payload(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    if suffix == ".json":
        return json.loads(text)
    raise ValueError("Config must be YAML or JSON")


def _select_single_config(payload: Any, source: Path) -> TrainingConfig:
    if isinstance(payload, list):
        if not payload:
            raise ValueError(f"No configurations found in {source}")
        payload = payload[0]
    if not isinstance(payload, dict):
        raise ValueError("Training configuration must be a mapping")
    config = TrainingConfig.from_mapping(payload, base_path=source.parent)
    if getattr(config.model, "kind", "kan") != "kan":
        raise ValueError("Loaded configuration is not marked as model.type='kan'")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run KAN training using a YAML/JSON configuration file")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/training.yaml"),
        help="Path to the training configuration file (default: config/training.yaml)",
    )
    args = parser.parse_args()

    try:
        payload = _load_payload(args.config)
        config = _select_single_config(payload, args.config)
    except (FileNotFoundError, ValueError, ConfigurationError) as exc:
        raise SystemExit(f"Failed to load training configuration: {exc}") from exc

    trainer = KANTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
