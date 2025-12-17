from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from . import build_config_from_mapping, run_training


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        return yaml.safe_load(text)
    if suffix == ".json":
        return json.loads(text)
    raise ValueError("Config must be YAML or JSON")


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

    payload = _load_config(args.config)
    cfg = build_config_from_mapping(payload)
    run_training(cfg)


if __name__ == "__main__":
    main()
