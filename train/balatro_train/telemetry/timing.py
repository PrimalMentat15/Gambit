"""
Timing helpers for latency instrumentation

All durations are seconds measured with ``time.perf_counter()``. Timings are
collected into plain dicts so they can be dropped straight into an event payload.
"""

import time
from contextlib import contextmanager
from typing import Dict, Iterator, Optional


@contextmanager
def phase(store: Dict[str, float], key: str) -> Iterator[None]:
    """
    Time a block and record it under ``key``

    Args:
        store: Dictionary to write the duration into
        key: Name of the timing stage

    Example:
        timings = {}
        with phase(timings, "collect"):
            trainer.collect()
    """
    start = time.perf_counter()
    try:
        yield
    finally:
        store[key] = time.perf_counter() - start


class Stopwatch:
    """
    Manual lap timer for stages that span method boundaries

    Used where a single logical phase starts in one call and ends in another,
    so the ``phase`` context manager cannot bracket it.
    """

    def __init__(self) -> None:
        self._start: Optional[float] = None
        self.elapsed: float = 0.0

    def start(self) -> "Stopwatch":
        """Begin timing"""
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        """
        End timing and return the duration

        Returns 0.0 if the stopwatch was never started.
        """
        if self._start is None:
            self.elapsed = 0.0
        else:
            self.elapsed = time.perf_counter() - self._start
            self._start = None
        return self.elapsed

    @property
    def running(self) -> bool:
        """True while timing is in progress"""
        return self._start is not None
