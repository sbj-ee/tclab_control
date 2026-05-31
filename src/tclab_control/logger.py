"""Time-series logging helpers (Step 1 of the roadmap).

A tiny CSV logger for (t, T1, T2, Q1, Q2) rows so experiments produce clean,
reloadable data without each script reinventing it.
"""

from __future__ import annotations

import csv
from pathlib import Path


class RunLogger:
    """Append timestamped rows to a CSV; usable as a context manager."""

    FIELDS = ["t", "T1", "T2", "Q1", "Q2"]

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = None
        self._writer = None

    def __enter__(self) -> "RunLogger":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=self.FIELDS)
        self._writer.writeheader()
        return self

    def log(self, t: float, T1: float, T2: float, Q1: float, Q2: float) -> None:
        assert self._writer is not None, "use RunLogger as a context manager"
        self._writer.writerow(
            {"t": round(t, 3), "T1": round(T1, 3), "T2": round(T2, 3),
             "Q1": round(Q1, 2), "Q2": round(Q2, 2)}
        )

    def __exit__(self, *exc) -> None:
        if self._fh is not None:
            self._fh.close()
