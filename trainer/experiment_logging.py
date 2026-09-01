"""Small dependency-free JSONL logger for reproducible RL ablations."""

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JsonlMetricLogger:
    """Append one self-contained JSON object per training/evaluation event."""

    def __init__(self, path: Optional[str], run_config: Optional[Dict[str, Any]] = None):
        self.path = path
        self.run_config = run_config or {}
        if path:
            directory = os.path.dirname(os.path.abspath(path))
            os.makedirs(directory, exist_ok=True)

    def log(self, metrics: Dict[str, Any], *, step: int, split: str = "train") -> None:
        if not self.path:
            return
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "split": split,
            "step": int(step),
            "config": self.run_config,
            "metrics": metrics,
        }
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
