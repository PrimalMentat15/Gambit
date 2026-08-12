"""
Event bus

Tails a run's events.jsonl and delivers batches of events to panels as Qt signals.

The trainer appends to the file and never blocks on us, so tailing is the whole
integration: the monitor can start late, restart, or die entirely without
affecting training. Reading is incremental via a byte offset, and a file that
shrinks is treated as a new run.
"""

import json
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from PySide6.QtCore import QObject, QThread, Signal


class TailReader:
    """
    Incremental newline-delimited JSON reader

    Pure Python with no Qt dependency so it can be tested directly. Reads in
    binary and only decodes complete lines, which avoids splitting a multi-byte
    character across two reads.
    """

    def __init__(self, path: str):
        self.path = path
        self._offset = 0
        self._buffer = b""
        self.malformed = 0

    def rewind(self) -> None:
        """Start reading from the beginning again"""
        self._offset = 0
        self._buffer = b""

    def read(self) -> Tuple[List[Dict[str, Any]], bool]:
        """
        Read whatever has been appended since the last call

        Returns:
            (events, restarted) where restarted is True if the file shrank or
            vanished, meaning consumers should clear their state
        """
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return [], False

        restarted = False
        if size < self._offset:
            # Truncated or replaced by a new run
            self.rewind()
            restarted = True

        if size == self._offset:
            return [], restarted

        try:
            with open(self.path, "rb") as handle:
                handle.seek(self._offset)
                chunk = handle.read(size - self._offset)
        except OSError:
            return [], restarted

        self._offset += len(chunk)
        data = self._buffer + chunk
        lines = data.split(b"\n")

        # The final element is whatever follows the last newline: either empty
        # or a line the writer has not finished yet
        self._buffer = lines.pop()

        events = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                self.malformed += 1
        return events, restarted


class TailThread(QThread):
    """Polls a TailReader off the GUI thread"""

    batch = Signal(list)
    restarted = Signal()

    def __init__(self, path: str, poll_ms: int = 100, parent=None):
        super().__init__(parent)
        self.path = path
        self.poll_ms = poll_ms
        self._stop = threading.Event()

    def run(self) -> None:
        reader = TailReader(self.path)
        while not self._stop.is_set():
            events, restarted = reader.read()
            if restarted:
                self.restarted.emit()
            if events:
                self.batch.emit(events)
            self._stop.wait(self.poll_ms / 1000.0)

    def stop(self, timeout_ms: int = 2000) -> None:
        """Signal the loop to exit and wait for it"""
        self._stop.set()
        self.wait(timeout_ms)


class EventBus(QObject):
    """
    Owns the tail thread and republishes events to the application

    Switching runs tears down the old thread and starts a new one, emitting
    ``restarted`` so panels clear stale series.
    """

    batch = Signal(list)
    restarted = Signal()
    source_changed = Signal(str)

    def __init__(self, poll_ms: int = 100, parent=None):
        super().__init__(parent)
        self.poll_ms = poll_ms
        self.path: Optional[str] = None
        self._thread: Optional[TailThread] = None

        self.total_events = 0
        self._recent: List[float] = []

    def set_source(self, path: Optional[str]) -> None:
        """
        Point the bus at a different events.jsonl

        Passing the current path is a no-op so periodic run rescans do not
        constantly restart the tail.
        """
        if path == self.path:
            return

        self.stop()
        self.path = path
        self.total_events = 0
        self._recent.clear()
        self.restarted.emit()

        if not path:
            self.source_changed.emit("")
            return

        self._thread = TailThread(path, self.poll_ms, parent=self)
        self._thread.batch.connect(self._on_batch)
        self._thread.restarted.connect(self.restarted)
        self._thread.start()
        self.source_changed.emit(path)

    def _on_batch(self, events: List[Dict[str, Any]]) -> None:
        """Track throughput, then hand the batch on"""
        now = time.monotonic()
        self.total_events += len(events)
        self._recent.extend([now] * len(events))

        cutoff = now - 5.0
        if self._recent and self._recent[0] < cutoff:
            self._recent = [t for t in self._recent if t >= cutoff]

        self.batch.emit(events)

    @property
    def events_per_sec(self) -> float:
        """Event rate over the last few seconds"""
        if len(self._recent) < 2:
            return 0.0
        span = self._recent[-1] - self._recent[0]
        if span <= 0:
            return 0.0
        return len(self._recent) / span

    def stop(self) -> None:
        """Tear down the tail thread"""
        if self._thread is not None:
            self._thread.stop()
            self._thread = None
