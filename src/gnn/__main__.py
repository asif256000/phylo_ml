"""Command-line entry point: python -m src.gnn"""

from .train import run_training

if __name__ == "__main__":
    run_training()
