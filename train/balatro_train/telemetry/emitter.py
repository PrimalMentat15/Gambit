"""
Telemetry emitter

Writes newline-delimited JSON events to a file from a background thread.

The training loop runs at ~10k env-steps/sec against the Rust sim, so the one
rule that matters is that ``emit()`` must never block or raise. It does a dict
construction and a non-blocking queue put; if the queue is full the event is
dropped and counted. A dead or slow consumer can therefore never slow down or
crash training.

Stdlib only, by design: instrumenting the trainer must not add install burden.
"""

import atexit
import itertools
import json
import queue
import threading
import time
from typing import Any, Optional

from .events import EventType, make_event

# Sentinel telling the writer thread to drain and exit
_SHUTDOWN = object()


class NullEmitter:
    """
    No-op emitter used when telemetry is disabled

    Implements the same interface as TelemetryEmitter so instrumented code needs
    no conditionals around emit calls.
    """

    enabled = False
    dropped = 0

    def emit(self, event_type: str, **data: Any) -> None:
        pass

    def flush(self) -> None:
        pass

    def close(self, timeout: float = 2.0) -> None:
        pass


class TelemetryEmitter:
    """
    Buffered, drop-tolerant NDJSON event writer

    Attributes:
        path: Destination events.jsonl path
        dropped: Number of events discarded because the queue was full
    """

    enabled = True

    def __init__(self, path: str, queue_size: int = 8192, flush_interval: float = 0.5):
        self.path = path
        self.flush_interval = flush_interval

        self._queue: "queue.Queue" = queue.Queue(maxsize=queue_size)
        self._seq = itertools.count(1)
        self.dropped = 0

        self._handle = open(path, "a", encoding="utf-8", buffering=1024 * 64)
        self._thread = threading.Thread(
            target=self._writer_loop, name="telemetry-writer", daemon=True
        )
        self._thread.start()

        atexit.register(self.close)

    def emit(self, event_type: str, **data: Any) -> None:
        """
        Queue an event for writing

        Never blocks and never raises. Drops the event if the queue is full.

        Args:
            event_type: One of EventType
            **data: Event payload fields
        """
        event = make_event(next(self._seq), time.time(), event_type, data)
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            self.dropped += 1

    def _writer_loop(self) -> None:
        """Drain the queue onto disk until shutdown"""
        last_flush = time.monotonic()

        while True:
            try:
                item = self._queue.get(timeout=self.flush_interval)
            except queue.Empty:
                item = None

            if item is _SHUTDOWN:
                self._drain_remaining()
                try:
                    self._handle.flush()
                except Exception:
                    pass
                return

            if item is not None:
                self._write(item)

            now = time.monotonic()
            if now - last_flush >= self.flush_interval:
                try:
                    self._handle.flush()
                except Exception:
                    pass
                last_flush = now

    def _drain_remaining(self) -> None:
        """Write whatever is still queued at shutdown"""
        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                return
            if item is not _SHUTDOWN:
                self._write(item)

    def _write(self, event: dict) -> None:
        """Serialize one event; a bad payload must not kill the writer thread"""
        try:
            self._handle.write(json.dumps(event, default=str) + "\n")
        except Exception:
            self.dropped += 1

    def flush(self) -> None:
        """Force buffered events to disk"""
        try:
            self._handle.flush()
        except Exception:
            pass

    def close(self, timeout: float = 2.0) -> None:
        """
        Stop the writer thread and close the file

        Safe to call more than once.
        """
        if self._handle is None:
            return

        try:
            self._queue.put_nowait(_SHUTDOWN)
        except queue.Full:
            pass

        self._thread.join(timeout=timeout)

        try:
            self._handle.close()
        except Exception:
            pass
        self._handle = None


# Module-level emitter so instrumented code can call emit() without threading an
# emitter through every constructor. Defaults to a no-op.
_active: Any = NullEmitter()


def set_emitter(emitter: Any) -> None:
    """
    Install the process-wide emitter, retiring whatever was there

    Closing the old one matters wherever more than one session runs in a single
    process (tests, sweeps): each live emitter holds a writer thread and an open
    file, and an emitter nothing points at would never be drained.
    """
    global _active
    previous = _active
    _active = emitter
    if previous is not emitter:
        previous.close()


def get_emitter() -> Any:
    """Return the process-wide emitter (NullEmitter if telemetry is off)"""
    return _active


def emit(event_type: str, **data: Any) -> None:
    """Emit through the process-wide emitter"""
    _active.emit(event_type, **data)


def close_emitter(timeout: float = 2.0) -> None:
    """Close the process-wide emitter and revert to a no-op"""
    global _active
    _active.close(timeout=timeout)
    _active = NullEmitter()
