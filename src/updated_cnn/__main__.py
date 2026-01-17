from __future__ import annotations

import argparse
from pathlib import Path

from . import run_training
from .config import ConfigurationError, load_training_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run updated CNN training (regression-only)")
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/training.yaml"),
        help="Path to YAML/JSON training config (default: config/training.yaml)",
    )
    args = parser.parse_args()
    try:
        cfg = load_training_config(args.config)
    except (FileNotFoundError, ConfigurationError, ValueError) as exc:
        raise SystemExit(f"Failed to load training configuration: {exc}") from exc

    run_training(cfg)


if __name__ == "__main__":
    main()
