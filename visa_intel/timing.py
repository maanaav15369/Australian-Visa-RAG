"""Per-stage wall-clock timing, in milliseconds."""
import time
from contextlib import contextmanager


class StageTimer:
    def __init__(self):
        self._ms = {}

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            self._ms[name] = self._ms.get(name, 0.0) + elapsed

    def as_dict(self):
        return {name: round(ms, 1) for name, ms in self._ms.items()}

    def total_ms(self):
        return round(sum(self._ms.values()), 1)
