# telemetry/mfe_logger.py
import json
import os
from datetime import datetime


class MFELogger:
    def __init__(self, path="data/mfe_stream.jsonl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path

    def log_tick(self, run_id: str, ticket: int, price: float, entry: float, sl: float, tp: float):
        rec = {
            "ts": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "ticket": ticket,
            "price": price,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "u": (price - entry),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
