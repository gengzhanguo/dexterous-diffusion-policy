"""
Unified logger: TensorBoard + Rich console.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.table import Table

console = Console()


class Logger:
    """Wraps TensorBoard SummaryWriter with a rich console printer."""

    def __init__(self, log_dir: str | Path, run_name: str = "run"):
        self.log_dir = Path(log_dir) / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(self.log_dir))
        self._start = time.time()
        console.log(f"[bold green]TensorBoard log dir:[/] {self.log_dir}")

    # ------------------------------------------------------------------ #
    # Scalar logging                                                       #
    # ------------------------------------------------------------------ #

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        self.writer.add_scalar(tag, value, step)

    def log_scalars(self, scalars: dict[str, float], step: int) -> None:
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    # ------------------------------------------------------------------ #
    # Console printing                                                     #
    # ------------------------------------------------------------------ #

    def print_step(self, step: int, scalars: dict[str, Any]) -> None:
        elapsed = time.time() - self._start
        parts = [f"step={step:>6d}", f"t={elapsed:>6.0f}s"]
        parts += [f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in scalars.items()]
        console.log("  ".join(parts))

    def print_eval(self, results: dict[str, Any]) -> None:
        table = Table(title="Evaluation Results", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        for k, v in results.items():
            table.add_row(k, f"{v:.4f}" if isinstance(v, float) else str(v))
        console.print(table)

    def close(self) -> None:
        self.writer.close()
