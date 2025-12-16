from __future__ import annotations

import argparse
from pathlib import Path

from .config import ConfigurationError, load_training_config
from .trainer import CNNTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the configurable CNN using a YAML/JSON configuration file.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/training.yaml"),
        help="Path to the training configuration file (default: config/training.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        config = load_training_config(args.config)
    except (FileNotFoundError, ConfigurationError) as exc:
        raise SystemExit(f"Failed to load training configuration: {exc}") from exc

    trainer = CNNTrainer(config)
    trainer.run()


if __name__ == "__main__":
    main()
