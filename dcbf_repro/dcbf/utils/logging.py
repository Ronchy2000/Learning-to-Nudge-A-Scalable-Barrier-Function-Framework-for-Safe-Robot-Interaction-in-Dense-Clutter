from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Optional

from torch.utils.tensorboard import SummaryWriter


class CSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer: Optional[csv.DictWriter] = None

    def log(self, row: Dict[str, Any]) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class JSONLLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")

    def log(self, row: Dict[str, Any]) -> None:
        self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def create_tb_writer(log_dir: str | Path) -> SummaryWriter:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(log_dir))
