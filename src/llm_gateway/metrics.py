"""Thread-safe in-memory metrics store for the LLM Gateway."""

import collections
import threading
from datetime import datetime, timezone


class MetricsStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total: int = 0
        self._hits: int = 0
        self._tokens_saved: int = 0
        self._latencies: collections.deque[float] = collections.deque(maxlen=1000)
        self._log: collections.deque[dict] = collections.deque(maxlen=50)

    def record_request(
        self,
        *,
        latency_ms: float,
        cache_hit: bool,
        model: str,
        tokens_saved: int = 0,
    ) -> None:
        with self._lock:
            self._total += 1
            if cache_hit:
                self._hits += 1
            self._tokens_saved += tokens_saved
            self._latencies.append(latency_ms)
            self._log.append(
                {
                    "ts": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                    "model": model,
                    "hit": cache_hit,
                    "latency_ms": round(latency_ms, 1),
                }
            )

    @property
    def hit_rate(self) -> float:
        with self._lock:
            if self._total == 0:
                return 0.0
            return self._hits / self._total

    def p_latency(self, pct: int) -> float:
        """Return the given percentile of recorded latencies (over last 1000 entries)."""
        import numpy as np

        with self._lock:
            if not self._latencies:
                return 0.0
            return float(np.percentile(list(self._latencies), pct))

    def recent_log(self, n: int = 20) -> list[dict]:
        with self._lock:
            entries = list(self._log)
        return entries[-n:]

    def snapshot(self) -> dict:
        with self._lock:
            total = self._total
            hits = self._hits
            tokens_saved = self._tokens_saved
            latencies = list(self._latencies)

        import numpy as np

        p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
        p95 = float(np.percentile(latencies, 95)) if latencies else 0.0
        hit_rate = (hits / total) if total > 0 else 0.0

        return {
            "total_requests": total,
            "cache_hits": hits,
            "hit_rate": round(hit_rate, 4),
            "p50_latency_ms": round(p50, 1),
            "p95_latency_ms": round(p95, 1),
            "tokens_saved": tokens_saved,
            "recent_log": self.recent_log(20),
        }


# Module-level singleton
metrics: MetricsStore = MetricsStore()
