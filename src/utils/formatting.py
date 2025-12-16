from __future__ import annotations

from typing import Any

try:  # torch is optional when formatting shapes outside of training contexts.
    import torch
except ImportError:  # pragma: no cover - torch is available in CI, but guard for tooling.
    torch = None  # type: ignore


def format_tuple(value: Any) -> str:
    """Render tensor-friendly tuple or scalar values as strings for logging."""
    if value is None:
        return "None"
    if torch is not None and isinstance(value, torch.Size):
        return "(" + ", ".join(str(v) for v in value) + ")"
    if isinstance(value, (tuple, list)):
        return "(" + ", ".join(str(v) for v in value) + ")"
    return str(value)


__all__ = ["format_tuple"]
