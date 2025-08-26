import json
import os
import threading
import time
from datetime import datetime
from typing import Any


class EventLogger:
    """
    JSONL event logger (thread-safe).
    Writes one JSON object per line to `path`.
    """

    def __init__(
        self,
        path: str,
        run_id: str,
        symbol: str,
        flush_interval: float = 0.5,
        max_size_mb: int = 10,
    ):
        self.path = path
        self.run_id = run_id
        self.symbol = symbol
        self.max_size_bytes = max_size_mb * 1024 * 1024
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._lock = threading.Lock()
        self._buffer = []
        self._stop = False
        self._t = threading.Thread(target=self._flusher, daemon=True, args=(flush_interval,))
        self._t.start()

    def emit(self, event: str, **fields: Any):
        rec: dict[str, Any] = {
            "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "event": event,
            "run_id": self.run_id,
            "symbol": self.symbol,
        }
        rec.update(fields)
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            self._buffer.append(line)

    def _flusher(self, interval: float):
        while not self._stop:
            self.flush()
            time.sleep(interval)
        self.flush()

    def _rotate_if_needed(self):
        """Rotate log file if it exceeds max size."""
        try:
            if os.path.exists(self.path) and os.path.getsize(self.path) > self.max_size_bytes:
                # Create backup with timestamp
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                backup_path = f"{self.path}.{timestamp}"
                os.rename(self.path, backup_path)
                # Keep only last 5 backups
                self._cleanup_old_backups()
        except Exception:
            pass  # Don't fail if rotation fails

    def _cleanup_old_backups(self):
        """Keep only last 5 backup files."""
        try:
            dir_path = os.path.dirname(self.path)
            base_name = os.path.basename(self.path)
            files = [f for f in os.listdir(dir_path) if f.startswith(base_name + ".")]
            files.sort(reverse=True)
            for old_file in files[5:]:  # Keep only last 5
                os.remove(os.path.join(dir_path, old_file))
        except Exception:
            pass  # Don't fail if cleanup fails

    def flush(self):
        buf = None
        with self._lock:
            if not self._buffer:
                return
            buf, self._buffer = self._buffer, []

        # Rotate if needed before writing
        self._rotate_if_needed()

        with open(self.path, "a", encoding="utf-8") as f:
            f.write("\n".join(buf) + "\n")

    def close(self):
        self._stop = True
        if self._t.is_alive():
            self._t.join(timeout=2)
